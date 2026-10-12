"""Module wiki (009 US1 to US3): pages, citations and staleness, through `wiki` against fixture repositories."""

import json
import logging
import posixpath
import re
import subprocess
import uuid

import pytest

from golestan_code_mcp import db, ingest, registry, schema, semantic, wiki
from golestan_code_mcp.chunker import EXTENSIONS
from golestan_code_mcp.config import load

ENV = {"CCCTX_MAX_FILE_BYTES": "20000"}


def functions(prefix: str, n: int) -> str:
    return "".join(f"def {prefix}{i}():\n    return {i}\n\n\n" for i in range(n))


CORE_PY = '''"""Core widget code. See core.py at line 3 for details."""
import json
from typing import Any

from .sub.util import sub_a


class Widget:
    """A widget, see :class:`~pkg.core.Gadget` for more. Defined in core.py."""

    def spin(self):
        """Spin it."""
        return sub_a()

    def stop(self):
        return 1


class Gadget:
    """A gadget made by :func:`make_widget`."""


def make_widget():
    """Make a ``Widget`` instance."""
    return Widget()


''' + functions("f", 5)

TESTS_PY = "from pkg.core import Widget, make_widget\nimport pkg\n\n\n" + "".join(
    f"def test_{i}():\n    return {'make_widget' if i == 2 else 'Widget'}()\n\n\n" for i in range(12))

SERVER_GO = """// Copyright header, not attached.

// Package server serves widgets.
package server

import (
	"fmt"
	"net/http"
)

// Server holds the state.
type Server struct{}

// Start starts it.
func (s *Server) Start() {
	fmt.Println(http.StatusOK)
}
""" + "".join(f"\nfunc F{i}() {{\n}}\n" for i in range(8))

FILES = {
    "src/pkg/__init__.py": '"""Widgets for the demo package.\n\nSecond paragraph.\n"""\nfrom .core import Widget, make_widget\n',
    "src/pkg/core.py": CORE_PY,
    "src/pkg/sub/util.py": functions("sub_", 10).replace("def sub_0", "def sub_a"),
    "nine/mod.py": functions("n", 9),
    "nine/notes.txt": "".join(f"note {i}\n" for i in range(130)),
    "docs/guide.md": "".join(f"guide {i}\n" for i in range(200)),
    "tests/test_pkg.py": TESTS_PY,
    "root.py": functions("r", 10),
    "x/mod.py": 'class Thing:\n    """A thing in x."""\n\n\n' + functions("x", 9),
    "src/x/mod.py": functions("y", 10),
    "server/server.go": SERVER_GO,
}


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                   capture_output=True)


def write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")


def commit(root, message="change"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


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


def make_repo(conn, root, files=FILES, build=True):
    root.mkdir()
    git(root, "init", "-q")
    write(root, files)
    commit(root, "init")
    name = f"t-{uuid.uuid4().hex[:8]}"
    registry.add_repo(conn, root, name)
    ingest.ingest(conn, name, ENV)
    if build:
        wiki.build(conn, name)
    return name


@pytest.fixture(scope="module")
def built(conn, tmp_path_factory):
    """A repository whose wiki is built once and never changed: read-only tests share it."""
    return make_repo(conn, tmp_path_factory.mktemp("w") / "repo")


@pytest.fixture
def fresh(conn, tmp_path):
    """(name, root) of a repository with a built wiki that a test may change."""
    root = tmp_path / "repo"
    return make_repo(conn, root), root


def rid(conn, name):
    return registry.repo_id(conn, name)


def lob(v):
    return v.read() if hasattr(v, "read") else v


def page_rows(conn, name):
    with conn.cursor() as cur:
        cur.execute("""select p.directory, p.name, p.language, p.symbols, p.text, p.search_text, p.state,
                              p.compiler, p.stale_reason
                         from ccctx_wiki_pages p join ccctx_repos r on r.id = p.repo_id where r.name = :n""", n=name)
        return {row[1]: tuple(lob(v) for v in row) for row in cur}


def citation_rows(conn, name):
    with conn.cursor() as cur:
        cur.execute("""select p.name, c.role, c.name, c.kind, c.commit_sha, c.content_hash, c.symbol_id, c.file_id
                         from ccctx_wiki_citations c join ccctx_wiki_pages p on p.id = c.page_id
                         join ccctx_repos r on r.id = p.repo_id where r.name = :n""", n=name)
        return sorted(cur.fetchall())


def text(conn, name, page):
    return page_rows(conn, name)[page][4]


# --- threshold and naming ------------------------------------------------------------------------------------


def test_one_page_per_directory_of_ten_named_definitions(conn, built):
    pages = page_rows(conn, built)
    assert {p[0]: p[1] for p in pages.values()} == {
        "src/pkg": "pkg", "src/pkg/sub": "pkg/sub", "tests": "tests", ".": built, "x": "x", "src/x": "src/x",
        "server": "server",
    }  # nine has 9 definitions plus line windows, docs only line windows: no page
    assert pages["pkg"][3] == 10 and pages["tests"][3] == 12 and pages["pkg"][2] == "python"
    assert pages["server"][2] == "go" and {p[7] for p in pages.values()} == {wiki.COMPILER}
    assert {p[6] for p in pages.values()} == {"fresh"}


# --- the text ------------------------------------------------------------------------------------------------


def test_structure_purpose_and_sections(conn, built):
    page = text(conn, built, "pkg")
    assert page.startswith(
        "# pkg\n\n"
        "`pkg` is a Python package of 2 files and 10 definitions: 6 functions, 2 classes and 2 methods.\n"
        "Widgets for the demo package.\nCore widget code.\n\n## Key symbols\n"
        "- `Widget` (class): A widget, see Gadget for more.\n"
        "- `Gadget` (class): A gadget made by make_widget.\n"
        "- `make_widget` (function): Make a Widget instance.\n"
    )
    assert "- `Widget.stop` (method): no documentation\n" in page
    assert "## Depends on\n- `pkg/sub`: `sub_a`\n- Outside the index: `json`, `typing`\n" in page
    assert page.endswith("## Used by\n- `tests`\n")


def test_purpose_falls_back_to_types_then_to_the_stated_sentence(conn, built):
    assert "definitions: 9 functions and 1 class.\nA thing in x.\n" in text(conn, built, "x")
    sub = text(conn, built, "pkg/sub")
    assert "`pkg/sub` is a Python directory of 1 file and 10 definitions: 10 functions.\n" in sub
    assert "\nIts code has no documentation; the sections below come from its structure.\n" in sub
    assert sub.endswith("## Used by\n- `pkg`\n")


def test_go_page(conn, built):
    page = text(conn, built, "server")
    assert "`server` is a Go package of 1 file and 10 definitions: 8 functions, 1 method and 1 type.\n" in page
    assert "Package server serves widgets.\n" in page and "Copyright" not in page
    assert "- Outside the index: `fmt`, `net/http`\n" in page


def test_key_symbols_at_most_ten_ranked_by_outside_callers(conn, built):
    keys = re.findall(r"^- `([^`]+)` \(", text(conn, built, "tests"), re.M)
    assert len(keys) == 10
    page = text(conn, built, "tests")
    assert "## Depends on\n- `pkg`: `Widget`, `make_widget`\n" in page
    assert re.findall(r"^- `([^`]+)` \(", text(conn, built, "pkg"), re.M)[:3] == ["Widget", "Gadget", "make_widget"]


def test_no_path_or_line_number_in_any_page(conn, built):
    with conn.cursor() as cur:
        paths = [p for (p,) in cur.execute("select path from ccctx_files where repo_id = :r", r=rid(conn, built))]
    names = {posixpath.basename(p) for p in paths}
    assert any(n.endswith(tuple(EXTENSIONS)) for n in names)
    for page in page_rows(conn, built).values():
        assert not any(n in page[4] for n in names | set(paths)), page[1]
        assert not re.search(r"line \d+", page[4]), page[1]


# --- citations -----------------------------------------------------------------------------------------------


def module_name(path):
    directory = posixpath.dirname(path) or "."
    return directory.removeprefix("src/")


def test_every_named_symbol_and_module_is_cited(conn, built):
    for name, page in page_rows(conn, built).items():
        cited = wiki.read(conn, name, built).citations
        symbols = {c.name for c in cited if c.role in ("key", "calls")}
        modules = {module_name(c.name) for c in cited if c.role in ("depends", "uses")}
        outside = re.findall(r"`([^`]+)`", (re.findall(r"^- Outside the index: (.*)$", page[4], re.M) or [""])[0])
        keys = re.findall(r"^- `([^`]+)` \(", page[4], re.M)
        assert set(keys) <= {c.name for c in cited if c.role == "key"}, name
        for token in re.findall(r"`([^`]+)`", page[4]):
            assert token in symbols | modules | set(outside) | {name}, (name, token)
        assert {c.role for c in cited} >= {"key", "file"}


def test_citations_record_commit_and_hash_and_read_back_current_locations(conn, built):
    with conn.cursor() as cur:
        r = rid(conn, built)
        commit = cur.execute("select indexed_commit from ccctx_repos where id = :r", r=r).fetchone()[0]
        symbols = {sid: row for sid, *row in cur.execute(
            """select s.id, s.commit_sha, s.content_hash, f.path, s.start_line, s.end_line from ccctx_symbols s
                 join ccctx_files f on f.id = s.file_id where f.repo_id = :r""", r=r)}
        files = {fid: row for fid, *row in cur.execute(
            "select id, content_hash, path from ccctx_files where repo_id = :r", r=r)}
    rows = citation_rows(conn, built)
    assert rows
    for _, role, _, kind, sha, digest, symbol_id, file_id in rows:
        if role in ("key", "calls"):
            assert (sha, digest) == tuple(symbols[symbol_id][:2]) and file_id is None
        else:
            assert (sha, digest, kind) == (commit, files[file_id][0], "file") and symbol_id is None
    for name in page_rows(conn, built):
        for c in wiki.read(conn, name, built).citations:
            assert c.repo == built and c.current_commit == c.recorded_commit
            assert c.path is not None
            if c.role in ("key", "calls"):
                found = [s for s in symbols.values() if (s[2], s[3]) == (c.path, c.start_line)]
                assert found and found[0][4] == c.end_line
            else:
                assert c.path == c.name and c.start_line is None


# --- build behavior ------------------------------------------------------------------------------------------


def test_second_build_is_idempotent_and_keeps_search_text(conn, fresh):
    name, _ = fresh
    with conn.cursor() as cur:  # a marker the build would overwrite if it rewrote search_text
        cur.execute("update ccctx_wiki_pages set search_text = 'marker' where repo_id = :r", r=rid(conn, name))
    conn.commit()
    pages, cites = page_rows(conn, name), citation_rows(conn, name)
    [counts] = wiki.build(conn, name)
    assert (counts.repo, counts.pages, counts.added, counts.updated, counts.removed) == (name, 7, 0, 0, 0)
    assert page_rows(conn, name) == pages and citation_rows(conn, name) == cites
    assert {p[5] for p in pages.values()} == {"marker"}


def test_first_build_counts_and_sets_wiki_at(conn, tmp_path):
    name = make_repo(conn, tmp_path / "repo", build=False)
    with conn.cursor() as cur:
        assert cur.execute("select wiki_at from ccctx_repos where name = :n", n=name).fetchone()[0] is None
    [counts] = wiki.build(conn, name)
    assert (counts.pages, counts.added, counts.updated, counts.removed) == (7, 7, 0, 0)
    with conn.cursor() as cur:
        assert cur.execute("select wiki_at from ccctx_repos where name = :n", n=name).fetchone()[0] is not None
    assert page_rows(conn, name)["pkg"][5] == f"page pkg in src/pkg\n{text(conn, name, 'pkg')}"


def test_unknown_and_unindexed_repositories_refused(conn, tmp_path):
    with pytest.raises(wiki.WikiError, match="^no repository named 't-missing'$"):
        wiki.build(conn, "t-missing")
    (tmp_path / "empty").mkdir()
    name = f"t-{uuid.uuid4().hex[:8]}"
    registry.add_repo(conn, tmp_path / "empty", name)
    with pytest.raises(wiki.WikiError) as e:
        wiki.build(conn, name)
    assert str(e.value) == f"repository '{name}' is not indexed yet; run `ccctx ingest {name}`"


def test_page_index_exists_after_a_build(conn, built):
    with conn.cursor() as cur:
        assert cur.execute("select count(*) from user_indexes where index_name = :i", i=wiki.INDEX).fetchone()[0]
        assert wiki.ensure_index(cur)


def test_build_logs_one_record_per_repository(conn, fresh, caplog):
    name, _ = fresh
    with caplog.at_level(logging.INFO, logger="golestan_code_mcp.wiki"):
        wiki.build(conn, name)
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "golestan_code_mcp.wiki"]
    assert len(records) == 1 and records[0]["event"] == "wiki" and records[0]["repo"] == name
    assert (records[0]["built"], records[0]["stale"], records[0]["removed"]) == (7, 0, 0)
    assert isinstance(records[0]["ms"], int) and "text" not in records[0]


# --- staleness -----------------------------------------------------------------------------------------------


def execute(conn, sql, **binds):
    with conn.cursor() as cur:
        cur.execute(sql, binds)
    conn.commit()


def symbol_of(conn, name, symbol):
    with conn.cursor() as cur:
        return cur.execute("""select s.id, s.commit_sha from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                               where f.repo_id = :r and s.name = :s""", r=rid(conn, name), s=symbol).fetchone()


def mark(conn, name):
    with conn.cursor() as cur:
        marked = wiki.mark_stale(cur, rid(conn, name))
    conn.commit()
    return marked


def test_mark_stale_names_the_changed_or_gone_symbol(conn, fresh):
    name, _ = fresh
    sid, old = symbol_of(conn, name, "Widget")
    execute(conn, "update ccctx_symbols set commit_sha = :c where id = :i", c="a" * 40, i=sid)
    gone, _ = symbol_of(conn, name, "x0")
    execute(conn, "delete from ccctx_symbols where id = :i", i=gone)
    marked = mark(conn, name)
    changed = f"Widget changed ({old[:12]} -> {'a' * 12})"
    assert marked == [("pkg", changed), ("tests", changed), ("x", "x0 is gone")]  # tests calls Widget
    states = {n: (p[6], p[8]) for n, p in page_rows(conn, name).items()}
    assert states["pkg"] == ("stale", changed) and states["x"] == ("stale", "x0 is gone")
    assert {s for n, s in states.items() if n not in ("pkg", "tests", "x")} == {("fresh", None)}
    assert mark(conn, name) == []  # only fresh pages are marked


def test_a_rewritten_symbol_is_reported_as_changed(conn, fresh):
    name, _ = fresh
    sid, old = symbol_of(conn, name, "x0")
    with conn.cursor() as cur:  # what ingestion does to an edited symbol: delete it, insert it again
        cur.execute("""insert into ccctx_symbols (file_id, name, kind, start_line, end_line, source, content_hash,
                                                  commit_sha)
                       select file_id, name, kind, start_line, end_line, source, :h, :c
                         from ccctx_symbols where id = :i""", h="b" * 64, c="b" * 40, i=sid)
        cur.execute("delete from ccctx_symbols where id = :i", i=sid)
    conn.commit()
    assert ("x", f"x0 changed ({old[:12]} -> {'b' * 12})") in mark(conn, name)


def test_refresh_recompiles_adds_and_removes(conn, fresh):
    name, _ = fresh
    sid, _ = symbol_of(conn, name, "Widget")
    execute(conn, "update ccctx_symbols set commit_sha = :c where id = :i", c="b" * 40, i=sid)
    marked = mark(conn, name)
    with conn.cursor() as cur:  # x drops to 9 definitions, nine reaches 10
        cur.execute("""delete from ccctx_symbols where id = (select s.id from ccctx_symbols s
                         join ccctx_files f on f.id = s.file_id where f.repo_id = :r and s.name = 'x1')""",
                    r=rid(conn, name))
        fid = cur.execute("select id from ccctx_files where repo_id = :r and path = 'nine/mod.py'",
                          r=rid(conn, name)).fetchone()[0]
        cur.execute("""insert into ccctx_symbols (file_id, name, kind, start_line, end_line, content_hash, commit_sha,
                         search_text, source) values (:f, 'n9', 'function', 40, 41, :h, :c, 'function n9', 'def n9()')""",
                    f=fid, h="0" * 64, c="b" * 40)
    conn.commit()
    report = wiki.refresh(conn, rid(conn, name))
    # pkg and tests, and src/x: without the clash it is renamed x
    assert (report.recompiled, report.added, report.removed, report.error) == (3, 1, 1, None)
    assert report.stale == marked
    pages = page_rows(conn, name)
    assert pages["x"][0] == "src/x" and "src/x" not in pages
    assert pages["nine"][6] == "fresh" and pages["nine"][3] == 10
    assert pages["pkg"][6:9:2] == ("fresh", marked[0][1])  # the reason is kept after the recompile
    widget = [c for c in wiki.read(conn, "pkg", name).citations if c.name == "Widget"][0]
    assert widget.recorded_commit == "b" * 40 == widget.current_commit


def test_failed_refresh_leaves_pages_stale(conn, fresh, monkeypatch):
    name, _ = fresh
    for symbol in ("Widget", "r0"):
        sid, _ = symbol_of(conn, name, symbol)
        execute(conn, "update ccctx_symbols set commit_sha = :c where id = :i", c="c" * 40, i=sid)
    assert len(mark(conn, name)) == 3  # pkg, tests and the root
    real, calls = wiki._write_page, []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("database went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(wiki, "_write_page", fail_second)
    report = wiki.refresh(conn, rid(conn, name))
    assert report.error and "database went away" in report.error and report.recompiled == 0
    states = {n: p[6] for n, p in page_rows(conn, name).items()}
    assert states["pkg"] == states[name] == "stale"


def test_repository_without_a_wiki_is_untouched(conn, tmp_path):
    name = make_repo(conn, tmp_path / "repo", build=False)
    assert mark(conn, name) == []
    report = wiki.refresh(conn, rid(conn, name))
    assert (report.stale, report.recompiled, report.added, report.removed, report.error) == ([], 0, 0, 0, None)
    assert page_rows(conn, name) == {}


# --- reading -------------------------------------------------------------------------------------------------


def test_pages_and_read_follow_the_contract(conn, built):
    listed = wiki.pages(conn, built)
    assert [p.name for p in listed] == sorted(p.name for p in listed) and {p.repo for p in listed} == {built}
    pkg = next(p for p in listed if p.name == "pkg")
    assert (pkg.state, pkg.symbols, pkg.summary, pkg.stale_reason) == ("fresh", 10, "Widgets for the demo package.", None)
    page = wiki.read(conn, "pkg", built)
    assert (page.name, page.repo, page.directory, page.language, page.state, page.compiler, page.symbols) == (
        "pkg", built, "src/pkg", "python", "fresh", wiki.COMPILER, 10)
    assert page.text == text(conn, built, "pkg") and not page.truncated
    order = ["key", "file", "depends", "uses", "calls"]
    keys = [(order.index(c.role), c.name) for c in page.citations]
    assert keys == sorted(keys)


def test_read_refuses_unknown_and_ambiguous_pages(conn, built, tmp_path):
    with pytest.raises(wiki.WikiError, match=f"^no wiki page named 'nope' in '{built}'$"):
        wiki.read(conn, "nope", built)
    with pytest.raises(wiki.WikiError, match="^no wiki page named 'nope'$"):
        wiki.read(conn, "nope")
    with pytest.raises(wiki.WikiError, match="^no repository named 't-missing'$"):
        wiki.pages(conn, "t-missing")
    unique = f"amb_{uuid.uuid4().hex[:6]}"
    names = sorted(make_repo(conn, tmp_path / f"r{i}", {f"{unique}/m.py": functions("a", 10)}) for i in range(2))
    with pytest.raises(wiki.WikiError) as e:
        wiki.read(conn, unique)
    assert str(e.value) == f"wiki page '{unique}' is in more than one repository: {names[0]}, {names[1]}; pass --repo"
    assert wiki.read(conn, unique, names[1]).repo == names[1]
