"""Code graph (US1 to US3): call and import resolution, callers, calls and impact, against two tiny repositories."""

import subprocess
import uuid

import pytest

from golestan_code_mcp import db, graph, ingest, registry, schema, semantic
from golestan_code_mcp.config import load

ENV = {"CCCTX_MAX_FILE_BYTES": "2048"}

PY = {
    "src/lib/__init__.py": "",
    "src/lib/helpers.py": '''import json


def url_for(x):
    return encode(x)


def encode(x):
    print(x)
    return json.dumps(x)


def dumps(x):
    return x
''',
    "src/lib/app.py": '''from . import helpers


class Flask:
    def url_for(self, x):
        return helpers.url_for(x)

    def open(self):
        return self.url_for(1)
''',
    "src/lib/other.py": '''class Other:
    def url_for(self):
        return 1

    def run(self):
        return self.url_for()


def a():
    return 0
''',
    "src/lib/chain.py": '''def a():
    return 1


def b():
    return a()


def c():
    return b() + a()


def d():
    return c()


def e():
    return d() + Helper()


def ping():
    return pong()


def pong():
    return ping()


def fact(n):
    return fact(n - 1)
''',
    "tests/test_app.py": '''import lib
from lib.app import Flask

lib.url_for(0)


def test_open():
    return Flask().open()


def obj_call(app):
    return app.url_for(2)


def class_call():
    return Flask.url_for(None, 3)
''',
}

GO = {
    "main.go": '''package main

import (
	"fmt"
	u "example.com/m/util"
)

func main() {
	fmt.Println(u.Helper())
}
''',
    "util/a.go": '''package util

import "example.com/m"

func Helper() int {
	return 1
}
''',
    "util/b.go": '''package util

func Other() int {
	return Helper()
}
''',
    "web/app.js": '''import { helper } from "./lib";

function start() {
  return helper();
}
''',
    "web/lib.js": '''export function helper() {
  return 1;
}

export function a() {
  return helper();
}
''',
}


def make(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    return root


@pytest.fixture(scope="module")
def conn():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    schema.migrate(conn)
    try:
        semantic.require_model(conn)
    except semantic.ModelError:
        conn.close()
        pytest.skip("embedding model not loaded: run `ccctx model load`")
    yield conn
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def repos(conn, tmp_path_factory):
    names = []
    for files in (PY, GO):
        name = f"t-{uuid.uuid4().hex[:8]}"
        registry.add_repo(conn, make(tmp_path_factory.mktemp("g"), files), name)
        ingest.ingest(conn, name, ENV)
        names.append(name)
    return names


def edges(conn, name):
    """(caller, caller path, callee, callee path) for every call edge of the repository; (module) for a file."""
    with conn.cursor() as cur:
        cur.execute(
            """select s.name, f.path, t.name, tf.path from ccctx_calls e
                 join ccctx_symbols s on s.id = e.caller_id join ccctx_files f on f.id = s.file_id
                 join ccctx_symbols t on t.id = e.callee_id join ccctx_files tf on tf.id = t.file_id
                 join ccctx_repos r on r.id = f.repo_id where r.name = :n
               union all
               select '(module)', f.path, t.name, tf.path from ccctx_file_calls e
                 join ccctx_files f on f.id = e.file_id
                 join ccctx_symbols t on t.id = e.callee_id join ccctx_files tf on tf.id = t.file_id
                 join ccctx_repos r on r.id = f.repo_id where r.name = :n""",
            n=name,
        )
        return sorted(cur.fetchall())


def imports(conn, name):
    with conn.cursor() as cur:
        cur.execute(
            """select f.path, t.path from ccctx_imports i
                 join ccctx_files f on f.id = i.importer_id join ccctx_files t on t.id = i.imported_id
                 join ccctx_repos r on r.id = f.repo_id where r.name = :n""",
            n=name,
        )
        return sorted(cur.fetchall())


H, APP, OTHER, CHAIN, TEST = "src/lib/helpers.py", "src/lib/app.py", "src/lib/other.py", "src/lib/chain.py", \
    "tests/test_app.py"


# --- resolution (T007) --------------------------------------------------------------------------------------


def test_python_call_resolution(conn, repos):
    assert edges(conn, repos[0]) == sorted([
        ("url_for", H, "encode", H),  # print() and the stdlib json.dumps() resolve to nothing, not to dumps
        ("Flask.url_for", APP, "url_for", H),  # rule 2: helpers is an imported module, so a bare call
        ("Flask.open", APP, "Flask.url_for", APP),  # rule 3: self prefers the enclosing class
        ("Other.run", OTHER, "Other.url_for", OTHER),
        ("b", CHAIN, "a", CHAIN),  # the same file wins over other.a
        ("c", CHAIN, "b", CHAIN), ("c", CHAIN, "a", CHAIN),
        ("d", CHAIN, "c", CHAIN), ("e", CHAIN, "d", CHAIN),  # Helper lives in another repository: no edge
        ("ping", CHAIN, "pong", CHAIN), ("pong", CHAIN, "ping", CHAIN), ("fact", CHAIN, "fact", CHAIN),
        ("(module)", TEST, "url_for", H),  # lib.url_for: rule 2, then the whole repository
        ("test_open", TEST, "Flask", APP),  # bare call to a class
        ("test_open", TEST, "Flask.open", APP),  # rule 4
        ("obj_call", TEST, "Flask.url_for", APP),  # rule 4: the imported file wins over Other.url_for
        ("class_call", TEST, "Flask.url_for", APP),  # Flask is an imported class, not a module: rule 4
    ])


def test_go_and_javascript_call_resolution(conn, repos):
    assert edges(conn, repos[1]) == sorted([
        ("main", "main.go", "Helper", "util/a.go"),  # u.Helper: the aliased package import is bare
        ("Other", "util/b.go", "Helper", "util/a.go"),
        ("start", "web/app.js", "helper", "web/lib.js"),
        ("a", "web/lib.js", "helper", "web/lib.js"),
    ])


def test_imports_resolve_within_the_repository(conn, repos):
    assert imports(conn, repos[0]) == [
        (APP, H),  # from . import helpers (relative)
        (TEST, "src/lib/__init__.py"),  # import lib, under the src/ source root
        (TEST, APP),  # from lib.app import Flask: the member falls back to its module
    ]  # import json is the standard library, not src/lib/... or any json package
    assert imports(conn, repos[1]) == [
        ("main.go", "util/a.go"), ("main.go", "util/b.go"),  # one row per file of the package directory
        ("util/a.go", "main.go"),  # the module root package
        ("web/app.js", "web/lib.js"),  # ./lib, with the .js suffix tried
    ]


# --- callers, calls, impact (T009) --------------------------------------------------------------------------


def rows(results, *fields):
    return [tuple(getattr(r, f) for f in fields) for r in results]


def test_callers_of_a_bare_name_cover_every_definition(conn, repos):
    found = graph.callers(conn, "url_for", repos=repos)
    assert {r.repo for r in found} == {repos[0]}
    assert rows(found, "name", "kind", "path", "line", "callee") == [
        ("Flask.url_for", "method", APP, 6, "url_for"),
        ("Flask.open", "method", APP, 9, "Flask.url_for"),
        ("Other.run", "method", OTHER, 6, "Other.url_for"),
        ("(module)", "file", TEST, 4, "url_for"),
        ("obj_call", "function", TEST, 12, "Flask.url_for"),
        ("class_call", "function", TEST, 16, "Flask.url_for"),
    ]


def test_qualified_name_narrows_to_the_method(conn, repos):
    assert rows(graph.callers(conn, "Flask.url_for", repos=repos), "name") == [("Flask.open",), ("obj_call",), ("class_call",)]


def test_calls(conn, repos):
    found = graph.calls(conn, "url_for", repos=repos)
    assert rows(found, "caller", "line", "name", "kind", "path", "start_line", "end_line") == [
        ("Flask.url_for", 6, "url_for", "function", H, 4, 5),
        ("url_for", 5, "encode", "function", H, 8, 10),
    ]
    assert graph.calls(conn, "encode", repos=repos) == []  # only the standard library


def test_impact_hops(conn, repos):
    assert rows(graph.impact(conn, "a", 3, repos=repos), "hops", "name") == [(1, "b"), (1, "c"), (2, "d"), (3, "e")]  # diamond: c 1
    assert rows(graph.impact(conn, "a", 2, repos=repos), "hops", "name") == [(1, "b"), (1, "c"), (2, "d")]
    one = graph.impact(conn, "a", 1, repos=repos)
    assert {r.name for r in one} == {r.name for r in graph.callers(conn, "a", repos=repos)}
    assert rows(graph.impact(conn, "ping", repos=repos), "hops", "name") == [(1, "pong"), (2, "ping")]  # a cycle, once each
    assert rows(graph.impact(conn, "fact", repos=repos), "hops", "name", "kind", "path", "start_line", "end_line") == [
        (1, "fact", "function", CHAIN, 29, 30)]


@pytest.mark.parametrize("question", ["callers", "calls", "impact"])
def test_refusals(conn, repos, question):
    ask = getattr(graph, question)
    with pytest.raises(ValueError, match="no symbol named 'nope'"):
        ask(conn, "nope")
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="the name is empty"):
            ask(conn, empty)
    with pytest.raises(ValueError, match="no symbol named 'helper'"):  # only in the other repository
        ask(conn, "helper", repos=[repos[0]])
    with pytest.raises(registry.RegistryError, match="no repository named 't-missing'"):
        ask(conn, "a", repos=["t-missing"])


@pytest.mark.parametrize("depth", [0, 6, -1])
def test_depth_out_of_range(conn, repos, depth):
    with pytest.raises(ValueError, match="depth must be between 1 and 5"):
        graph.impact(conn, "a", depth, repos=repos)


def test_repos_limit_every_question(conn, repos):
    py, go = repos
    assert rows(graph.callers(conn, "a", repos=[py]), "name") == [("b",), ("c",)]
    assert graph.callers(conn, "a", repos=[go]) == []  # the JavaScript a has no callers
    assert rows(graph.callers(conn, "a", repos=repos), "name") == [("b",), ("c",)]  # both repositories
    assert {("b",), ("c",)} <= set(rows(graph.callers(conn, "a"), "name"))  # all by default
    assert rows(graph.calls(conn, "a", repos=[go]), "caller", "name") == [("a", "helper")]
    assert graph.calls(conn, "a", repos=[py]) == []
    assert rows(graph.impact(conn, "helper", 3, repos=[go]), "hops", "name") == [(1, "start"), (1, "a")]
    assert rows(graph.callers(conn, "Helper", repos=repos), "repo", "name") == [(go, "main"), (go, "Other")]
