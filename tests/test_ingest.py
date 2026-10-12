"""Ingest pipeline (FR-001, FR-002, FR-009 to FR-014, FR-017, FR-018) against throwaway git repositories."""

import hashlib
import json
import logging
import os
import subprocess
import uuid
from collections import Counter

import pytest

from golestan_code_mcp import db, graph, ingest, registry, schema, semantic, wiki
from golestan_code_mcp.config import load

ENV = {"CCCTX_MAX_FILE_BYTES": "2048"}

APP_PY = '''import os


def helper(x):
    return x + 1


class Service:
    def run(self):
        return helper(1)
'''
MAIN_GO = '''package main

func main() {
}
'''
WEB_JS = '''function hello() {
  return 1;
}
'''


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True, capture_output=True)


def head(path):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A committed repository with one file for every skip reason and every chunking path."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    files = {
        ".gitignore": "*.log\nbuild/\n",
        "app.py": APP_PY,
        "main.go": MAIN_GO,
        "web.js": WEB_JS,
        "broken.py": "def oops(:\n    pass\n",
        "notes.txt": "".join(f"line {i}\n" for i in range(1, 131)),
        "empty.txt": "",
        "vendor/dep.go": MAIN_GO,
        "big.txt": "x" * 3000,
        "app.log": "ignored\n",
        "build/out.txt": "ignored\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (root / "latin1.txt").write_bytes(b"caf\xe9\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def try_symlink(root):
    try:
        os.symlink(root / "app.py", root / "link.py")
        return True
    except OSError:  # Windows without symlink privilege
        return False


# --- discovery: no database -------------------------------------------------------------------------------


def test_discover_assigns_skip_reasons(repo):
    linked = try_symlink(repo)
    files, skipped = ingest.discover(repo, ENV)
    assert set(files) == {".gitignore", "app.py", "main.go", "web.js", "broken.py", "notes.txt", "empty.txt"}
    assert skipped == Counter({"ignored": 2 + linked, "vendored": 1, "binary": 2, "too large": 1})
    assert files["app.py"] == APP_PY
    assert not any(p.startswith(".git/") for p in files)


def test_discover_honours_vendor_setting(repo):
    files, skipped = ingest.discover(repo, {**ENV, "CCCTX_VENDOR_DIRS": ""})
    assert "vendor/dep.go" in files and "vendored" not in skipped


def test_git_helper_is_read_only(repo):
    before = (repo / ".git" / "index").read_bytes()
    ingest.discover(repo, ENV)
    ingest.git(repo, "status", "--porcelain")
    ingest.git(repo, "diff", "--name-only", "-z", "--no-renames", head(repo), "--")
    ingest.git(repo, "rev-parse", "--verify", "--quiet", f"{head(repo)}^{{commit}}")
    assert (repo / ".git" / "index").read_bytes() == before


# --- database half ------------------------------------------------------------------------------------------


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


def register(conn, path):
    name = f"t-{uuid.uuid4().hex[:8]}"
    registry.add_repo(conn, path, name)
    return name


def stored(conn, name):
    with conn.cursor() as cur:
        cur.execute(
            """select f.path, f.language, f.content_hash, s.name, s.kind, s.start_line, s.end_line,
                      s.source, s.content_hash, s.commit_sha
                 from ccctx_files f join ccctx_repos r on r.id = f.repo_id
                 left join ccctx_symbols s on s.file_id = f.id
                where r.name = :n""",
            n=name,
        )
        return [tuple(v.read() if hasattr(v, "read") else v for v in row) for row in cur]


def revision(conn, name):
    with conn.cursor() as cur:
        return cur.execute("select indexed_commit, indexed_dirty from ccctx_repos where name = :n", n=name).fetchone()


def content(conn, name):
    """Stored rows without commit_sha, which unchanged files keep from the run that last read them."""
    return sorted(repr(row[:9]) for row in stored(conn, name))


def symbol_ids(conn, name, path):
    with conn.cursor() as cur:
        cur.execute(
            """select s.id from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                 join ccctx_repos r on r.id = f.repo_id where r.name = :n and f.path = :p order by s.id""",
            n=name, p=path,
        )
        return [row[0] for row in cur]


def named_ids(conn, name, path):
    with conn.cursor() as cur:
        cur.execute(
            """select s.name, s.id from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                 join ccctx_repos r on r.id = f.repo_id where r.name = :n and f.path = :p""",
            n=name, p=path,
        )
        return dict(cur.fetchall())


RELATIONS = {  # refs and edges by names and paths: symbol ids change when code is edited
    "call refs": """select f.path, s.name, c.name, c.form, c.receiver, c.line from ccctx_call_refs c
                      join ccctx_files f on f.id = c.file_id left join ccctx_symbols s on s.id = c.caller_id""",
    "import refs": """select f.path, i.module, i.rel_level, i.bound from ccctx_import_refs i
                        join ccctx_files f on f.id = i.file_id""",
    "calls": """select f.path, s.name, tf.path, t.name, c.line from ccctx_calls c
                  join ccctx_symbols s on s.id = c.caller_id join ccctx_files f on f.id = s.file_id
                  join ccctx_symbols t on t.id = c.callee_id join ccctx_files tf on tf.id = t.file_id""",
    "file calls": """select f.path, tf.path, t.name, c.line from ccctx_file_calls c join ccctx_files f on f.id = c.file_id
                       join ccctx_symbols t on t.id = c.callee_id join ccctx_files tf on tf.id = t.file_id""",
    "imports": """select f.path, t.path from ccctx_imports i join ccctx_files f on f.id = i.importer_id
                    join ccctx_files t on t.id = i.imported_id""",
}


def relations(conn, name):
    with conn.cursor() as cur:
        rid = cur.execute("select id from ccctx_repos where name = :n", n=name).fetchone()[0]
        return {table: sorted(map(repr, cur.execute(sql + " where f.repo_id = :r", r=rid)))
                for table, sql in RELATIONS.items()}


def assert_matches_full(conn, name):
    """FR-006, SC-005 of 006 and FR-012, SC-005 of 007: the incremental index, refs and edges included, are what
    a full rebuild of the same tree stores."""
    before, graph_before = content(conn, name), relations(conn, name)
    assert ingest.ingest(conn, name, ENV, full=True).mode == "full"
    assert content(conn, name) == before and relations(conn, name) == graph_before


def test_ingest_summary_and_rows(conn, repo, caplog):
    name = register(conn, repo)
    with caplog.at_level(logging.INFO, logger="golestan_code_mcp.ingest"):
        summary = ingest.ingest(conn, name, ENV)

    assert summary.commit == head(repo) and not summary.dirty
    assert summary.files == 7 and summary.scanned == 7 + sum(summary.skipped.values())
    assert (summary.no_grammar, summary.parse_errors) == (3, 1)  # .gitignore, notes.txt, empty.txt / broken.py

    rows = stored(conn, name)
    symbols = [r for r in rows if r[3] is not None]
    assert summary.symbols == len(symbols)
    assert {r[0] for r in rows} == {".gitignore", "app.py", "main.go", "web.js", "broken.py", "notes.txt", "empty.txt"}
    assert {r[9] for r in symbols} == {summary.commit}  # SC-006
    by_name = {r[3]: r for r in symbols}
    assert by_name["helper"][4:7] == ("function", 4, 5)
    assert by_name["Service.run"][4] == "method" and by_name["Service"][4] == "class"
    assert by_name["main"][4] == "function" and by_name["hello"][4] == "function"
    assert {r[3] for r in symbols if r[0] == "notes.txt"} == {"notes.txt:1-60", "notes.txt:51-110", "notes.txt:101-130"}
    assert {r[4] for r in symbols if r[0] == "broken.py"} == {"chunk"}
    for path, language, file_hash, _, _, start, end, source, symbol_hash, _ in symbols:
        assert file_hash == hashlib.sha256((repo / path).read_bytes()).hexdigest()
        assert symbol_hash == hashlib.sha256(source.encode()).hexdigest()
        assert source == "\n".join((repo / path).read_text(encoding="utf-8").splitlines()[start - 1:end])
    assert {r[0]: r[1] for r in rows}["app.py"] == "python"

    record = json.loads(next(r.getMessage() for r in caplog.records if r.name == "golestan_code_mcp.ingest"))
    assert record["event"] == "ingest" and record["commit"] == summary.commit and record["symbols"] == summary.symbols


def test_unchanged_reingest_is_current(conn, repo, caplog):
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    assert (first.mode, first.note) == ("full", None) and revision(conn, name) == (head(repo), False)  # FR-001
    rows = stored(conn, name)
    with caplog.at_level(logging.INFO, logger="golestan_code_mcp.ingest"):
        second = ingest.ingest(conn, name, ENV)
    assert second.mode == "current" and (second.files, second.symbols) == (0, 0)  # FR-002
    assert (second.added, second.changed, second.removed, second.unchanged) == (0, 0, 0, first.files)
    assert stored(conn, name) == rows
    record = json.loads(caplog.records[-1].getMessage())
    assert record["mode"] == "current" and record["unchanged"] == first.files  # FR-012


def test_failed_ingest_keeps_previous_index(conn, repo, monkeypatch):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    before = stored(conn, name)
    (repo / "extra.py").write_text("def extra():\n    pass\n")

    def boom(cur, rows):
        raise RuntimeError("database went away")

    monkeypatch.setattr(ingest, "_insert_symbols", boom)
    with pytest.raises(RuntimeError):
        ingest.ingest(conn, name, ENV)
    assert stored(conn, name) == before
    assert revision(conn, name) == (head(repo), False)  # FR-007, SC-006


def test_every_symbol_gets_search_text(conn, repo):
    name = register(conn, repo)
    summary = ingest.ingest(conn, name, ENV)
    with conn.cursor() as cur:
        cur.execute(
            """select s.kind, s.name, f.path, s.source, s.search_text
                 from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
                where r.name = :n""",
            n=name,
        )
        rows = [tuple(v.read() if hasattr(v, "read") else v for v in row) for row in cur]
    assert len(rows) == summary.symbols
    for kind, symbol, path, source, text in rows:
        assert text == f"{kind} {symbol} in {path}\n{source}"


def test_ingested_symbols_are_searchable_at_once(conn, repo):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    # no sleep, no sync call: the hybrid index syncs on commit (FR-014)
    assert semantic.search(conn, "helper", [name], mode="text")[0].name == "helper"
    assert "Service.run" in [r.name for r in semantic.search(conn, "method that runs the service", [name])]


def test_missing_model_changes_nothing(conn, repo, monkeypatch):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    before = stored(conn, name)

    def missing(conn):
        raise semantic.ModelError("embedding model ALL_MINILM_L12_V2 is not set up; run `ccctx model load`")

    def no_discovery(*args):
        raise AssertionError("discover ran before the model check")

    monkeypatch.setattr(semantic, "require_model", missing)
    monkeypatch.setattr(ingest, "discover", no_discovery)
    with pytest.raises(semantic.ModelError, match="ccctx model load"):
        ingest.ingest(conn, name, ENV)
    assert stored(conn, name) == before  # FR-008


def test_dirty_tree_flagged(conn, repo):
    name = register(conn, repo)
    (repo / "untracked.py").write_text("x = 1\n")
    assert not ingest.ingest(conn, name, ENV).dirty
    (repo / "app.py").write_text(APP_PY + "\n# edited\n")
    assert ingest.ingest(conn, name, ENV).dirty


def test_refuses_non_git_and_empty_repos(conn, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ingest.IngestError, match="not a git repository"):
        ingest.ingest(conn, register(conn, plain), ENV)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    git(fresh, "init", "-q")
    with pytest.raises(ingest.IngestError, match="no commits"):
        ingest.ingest(conn, register(conn, fresh), ENV)


def test_concurrent_ingest_refused(conn, repo):
    name = register(conn, repo)
    other = db.connect(load())
    try:
        with other.cursor() as cur:
            cur.execute("select id from ccctx_repos where name = :n for update", n=name)
            with pytest.raises(ingest.IngestError, match="another ingest"):
                ingest.ingest(conn, name, ENV)
    finally:
        other.rollback()
        other.close()


# --- incremental re-ingest ----------------------------------------------------------------------------------


def write(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def test_uncommitted_edit_at_same_head_is_indexed(conn, repo):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    write(repo / "app.py", APP_PY + "\n\ndef extra():\n    pass\n")
    s = ingest.ingest(conn, name, ENV)
    assert (s.mode, s.added, s.changed, s.removed) == ("incremental", 0, 1, 0) and s.dirty
    assert revision(conn, name) == (head(repo), True)
    assert "extra" in {r[3] for r in stored(conn, name)}
    git(repo, "checkout", "--", "app.py")  # back to the commit: only the dirty flag reveals the stale rows
    s = ingest.ingest(conn, name, ENV)
    assert (s.mode, s.changed) == ("incremental", 1) and revision(conn, name) == (head(repo), False)
    assert "extra" not in {r[3] for r in stored(conn, name)}
    assert_matches_full(conn, name)


def test_only_changed_files_are_reprocessed(conn, repo):
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    kept = symbol_ids(conn, name, "app.py")
    main_id = symbol_ids(conn, name, "main.go")
    write(repo / "main.go", MAIN_GO + "\nfunc other() {\n}\n")
    (repo / "lib").mkdir()
    git(repo, "mv", "web.js", "lib/web.js")
    git(repo, "rm", "-q", "notes.txt")
    write(repo / "new.py", "def fresh():\n    pass\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    write(repo / "extra.py", "def untracked():\n    pass\n")

    s = ingest.ingest(conn, name, ENV)
    assert s.mode == "incremental" and s.commit == head(repo) and not s.dirty
    assert (s.added, s.changed, s.removed, s.unchanged) == (3, 1, 2, 4)  # FR-008
    rewritten = {"main.go", "lib/web.js", "new.py", "extra.py"}
    # main.go's untouched `main` keeps its row; only its new `other` is inserted and embedded
    assert s.files == 4 and s.symbols == len([r for r in stored(conn, name) if r[0] in rewritten]) - 1
    assert main_id[0] in symbol_ids(conn, name, "main.go")
    paths = {r[0] for r in stored(conn, name)}
    assert rewritten <= paths and not {"web.js", "notes.txt"} & paths  # FR-005
    assert symbol_ids(conn, name, "app.py") == kept  # FR-004: unchanged rows are not rewritten
    assert {r[9] for r in stored(conn, name) if r[0] == "app.py"} == {first.commit}
    assert revision(conn, name) == (head(repo), False)
    assert_matches_full(conn, name)


def test_change_reverted_in_a_later_commit_is_unchanged(conn, repo):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    write(repo / "app.py", "x = 1\n")
    git(repo, "commit", "-q", "-am", "edit")
    git(repo, "revert", "--no-edit", "HEAD")
    s = ingest.ingest(conn, name, ENV)
    assert (s.mode, s.added, s.changed, s.removed, s.unchanged) == ("incremental", 0, 0, 0, 7)
    assert revision(conn, name) == (head(repo), False)  # the revision moves even with nothing to re-index


def test_deleted_and_newly_skipped_files_leave_the_index(conn, repo):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    (repo / "app.py").unlink()
    write(repo / "main.go", "x" * 3000)  # now too large
    (repo / "web.js").write_bytes(b"\x00binary")
    s = ingest.ingest(conn, name, ENV)
    assert (s.mode, s.added, s.changed, s.removed) == ("incremental", 0, 0, 3)
    assert not {"app.py", "main.go", "web.js"} & {r[0] for r in stored(conn, name)}  # FR-005, SC-003
    assert all(r.path != "app.py" for r in semantic.search(conn, "helper", [name], mode="text"))
    assert_matches_full(conn, name)


def test_full_rebuild_on_up_to_date_repository(conn, repo):
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    s = ingest.ingest(conn, name, ENV, full=True)  # FR-009, SC-004
    assert s.mode == "full" and s.note is None and (s.files, s.symbols) == (first.files, first.symbols)


@pytest.mark.parametrize(("recorded", "note"), [
    (None, "no recorded revision"), ("f" * 40, "is no longer in the repository"),
])
def test_unknown_revision_rebuilds_in_full(conn, repo, recorded, note):
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    with conn.cursor() as cur:
        cur.execute("update ccctx_repos set indexed_commit = :c where name = :n", c=recorded, n=name)
    conn.commit()
    s = ingest.ingest(conn, name, ENV)  # FR-010
    assert s.mode == "full" and note in s.note and s.symbols == first.symbols
    assert revision(conn, name) == (head(repo), False)


def test_edited_file_keeps_its_unchanged_symbols(conn, repo):
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    before = {r[3]: r for r in stored(conn, name) if r[0] == "app.py"}
    ids = named_ids(conn, name, "app.py")
    git(repo, "checkout", "-q", "-b", "edit")
    edited = APP_PY.replace("import os\n", "import os\nimport sys\n\n\ndef top():\n    pass\n")
    edited = edited.replace("return x + 1", "return x + 2")
    (repo / "app.py").write_text(edited, encoding="utf-8", newline="\n")
    git(repo, "commit", "-q", "-am", "edit")
    s = ingest.ingest(conn, name, ENV)
    after = {r[3]: r for r in stored(conn, name) if r[0] == "app.py"}
    assert s.symbols == 2  # `top` is new and `helper` changed; Service and Service.run only moved
    assert after["Service"][5:7] == (before["Service"][5] + 5, before["Service"][6] + 5)
    assert after["Service"][9] == head(repo)  # its lines were read at the new revision
    now = named_ids(conn, name, "app.py")
    assert (now["Service"], now["Service.run"]) == (ids["Service"], ids["Service.run"])
    assert now["helper"] != ids["helper"]
    assert_matches_full(conn, name)


# --- the code graph stays current (007 US4) ------------------------------------------------------------------

LIB_PY = '''def target():
    return 1
'''
USE_PY = '''from lib import target


def caller():
    return target()


def other():
    return target()


target()
'''


@pytest.fixture
def graph_repo(repo):
    write(repo / "lib.py", LIB_PY)
    write(repo / "use.py", USE_PY)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "graph")
    return repo


def callers_of(conn, name, symbol):
    return {c.name for c in graph.callers(conn, symbol, [name])}


def test_graph_follows_edits_and_deletions(conn, graph_repo):
    name = register(conn, graph_repo)
    ingest.ingest(conn, name, ENV)
    assert callers_of(conn, name, "target") == {"caller", "other", "(module)"}
    assert callers_of(conn, name, "helper") == {"Service.run"}

    write(graph_repo / "lib.py", LIB_PY.replace("return 1", "return 2"))  # the callee gets a new id
    git(graph_repo, "commit", "-q", "-am", "edit callee")
    s = ingest.ingest(conn, name, ENV)
    assert (s.mode, s.changed) == ("incremental", 1)
    assert callers_of(conn, name, "target") == {"caller", "other", "(module)"}  # US4 scenario 2: use.py not re-read

    write(graph_repo / "use.py", USE_PY.replace("def other():\n    return target()\n\n\n", ""))
    git(graph_repo, "commit", "-q", "-am", "drop a caller")
    ingest.ingest(conn, name, ENV)
    assert callers_of(conn, name, "target") == {"caller", "(module)"}  # US4 scenario 1
    assert any("'use.py', 'lib.py'" in row for row in relations(conn, name)["imports"])
    assert_matches_full(conn, name)


def forget_graph(conn, name):
    """What an index written before step 7 looks like: no refs, no edges, no extract version."""
    with conn.cursor() as cur:
        rid = cur.execute("select id from ccctx_repos where name = :n", n=name).fetchone()[0]
        in_repo = "(select id from ccctx_files where repo_id = :r)"
        for sql in (f"delete from ccctx_call_refs where file_id in {in_repo}",
                    f"delete from ccctx_import_refs where file_id in {in_repo}",
                    f"delete from ccctx_file_calls where file_id in {in_repo}",
                    f"delete from ccctx_imports where importer_id in {in_repo}",
                    f"delete from ccctx_calls where caller_id in (select id from ccctx_symbols where file_id in {in_repo})"):
            cur.execute(sql, r=rid)
        cur.execute("update ccctx_repos set extract_version = null where id = :r", r=rid)
    conn.commit()


def extract_version(conn, name):
    with conn.cursor() as cur:
        return cur.execute("select extract_version from ccctx_repos where name = :n", n=name).fetchone()[0]


def test_repository_indexed_before_the_graph_is_reread_without_reembedding(conn, graph_repo):
    name = register(conn, graph_repo)
    first = ingest.ingest(conn, name, ENV)
    full_graph = relations(conn, name)
    ids = {path: named_ids(conn, name, path) for path in ("app.py", "lib.py", "use.py")}
    forget_graph(conn, name)
    assert callers_of(conn, name, "target") == set()

    s = ingest.ingest(conn, name, ENV)  # FR-013: same revision, yet every file is re-read for its refs
    assert s.mode == "incremental" and (s.changed, s.files) == (first.files, first.files)
    assert s.symbols == 0  # no symbol changed, so nothing is re-embedded
    assert {path: named_ids(conn, name, path) for path in ids} == ids
    assert extract_version(conn, name) == ingest.EXTRACT_VERSION
    assert relations(conn, name) == full_graph
    assert ingest.ingest(conn, name, ENV).mode == "current"


def test_failed_reread_keeps_the_old_extract_version(conn, graph_repo, monkeypatch):
    name = register(conn, graph_repo)
    ingest.ingest(conn, name, ENV)
    forget_graph(conn, name)

    def boom(cur, repo_id):
        raise RuntimeError("database went away")

    monkeypatch.setattr(graph, "rebuild", boom)
    with pytest.raises(RuntimeError):
        ingest.ingest(conn, name, ENV)
    assert extract_version(conn, name) is None and relations(conn, name)["call refs"] == []


# --- docs and the module wiki (009) --------------------------------------------------------------------------

DOC_PY = '''"""Module doc."""


def documented():
    """Does a thing."""
    return 1
'''
DOC_GO = '''package main

// Run runs.
func Run() {
}
'''


def docs(conn, name):
    with conn.cursor() as cur:
        cur.execute("""select f.path, f.doc, s.name, s.doc, s.id from ccctx_files f
                         join ccctx_repos r on r.id = f.repo_id left join ccctx_symbols s on s.file_id = f.id
                        where r.name = :n""", n=name)
        rows = cur.fetchall()
    return {p: d for p, d, *_ in rows}, {s: (d, i) for _, _, s, d, i in rows if s}


def test_docs_are_stored_and_follow_edits(conn, repo):
    write(repo / "doc.py", DOC_PY)
    write(repo / "run.go", DOC_GO)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "docs")
    name = register(conn, repo)
    ingest.ingest(conn, name, ENV)
    files, symbols = docs(conn, name)
    assert files["doc.py"] == "Module doc." and files["run.go"] is None and files["notes.txt"] is None
    assert symbols["documented"][0] == "Does a thing." and symbols["Run"][0] == "Run runs."
    assert symbols["helper"][0] is None and symbols["notes.txt:1-60"][0] is None

    write(repo / "doc.py", DOC_PY.replace("Module doc.", "Edited module doc."))
    write(repo / "run.go", DOC_GO.replace("// Run runs.", "// Run runs fast."))
    git(repo, "commit", "-q", "-am", "edit docs")
    ingest.ingest(conn, name, ENV)
    files, after = docs(conn, name)
    assert files["doc.py"] == "Edited module doc."
    assert after["Run"] == ("Run runs fast.", symbols["Run"][1])  # kept row, new doc
    assert_matches_full(conn, name)


def test_extract_version_1_is_reread_once_without_new_symbols(conn, repo):
    write(repo / "doc.py", DOC_PY)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "docs")
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    ids = {path: named_ids(conn, name, path) for path in ("app.py", "doc.py")}
    with conn.cursor() as cur:
        rid = registry.repo_id(conn, name)
        cur.execute("update ccctx_repos set extract_version = 1 where id = :r", r=rid)
        cur.execute("update ccctx_files set doc = null where repo_id = :r", r=rid)
        cur.execute("update ccctx_symbols set doc = null where file_id in (select id from ccctx_files where repo_id = :r)",
                    r=rid)
    conn.commit()
    assert ingest.EXTRACT_VERSION > 1
    s = ingest.ingest(conn, name, ENV)
    assert s.mode == "incremental" and s.changed == first.files and s.symbols == 0
    assert {path: named_ids(conn, name, path) for path in ids} == ids
    files, symbols = docs(conn, name)
    assert files["doc.py"] == "Module doc." and symbols["documented"][0] == "Does a thing."
    assert extract_version(conn, name) == ingest.EXTRACT_VERSION and ingest.ingest(conn, name, ENV).mode == "current"


def pkg_module(prefix):
    return "".join(f"def {prefix}{i}():\n    return {i}\n\n\n" for i in range(5))


@pytest.fixture
def wiki_repo(tmp_path):
    """A package of 10 definitions in two files, and an uncited file elsewhere."""
    root = tmp_path / "wrepo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "pkg").mkdir()
    (root / "other").mkdir()
    write(root / "pkg" / "a.py", pkg_module("a"))
    write(root / "pkg" / "b.py", pkg_module("b"))
    write(root / "other" / "x.py", "def lonely():\n    return 0\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def pages_of(conn, name):
    with conn.cursor() as cur:
        cur.execute("""select p.name, p.state, p.stale_reason from ccctx_wiki_pages p
                         join ccctx_repos r on r.id = p.repo_id where r.name = :n""", n=name)
        return {n: (s, why) for n, s, why in cur}


def test_editing_a_cited_function_marks_and_recompiles_its_page(conn, wiki_repo):
    name = register(conn, wiki_repo)
    old = ingest.ingest(conn, name, ENV).commit
    wiki.build(conn, name)
    write(wiki_repo / "pkg" / "a.py", pkg_module("a").replace("return 3", "return 33"))
    git(wiki_repo, "commit", "-q", "-am", "edit a3")
    s = ingest.ingest(conn, name, ENV)
    reason = f"a0 changed ({old[:12]} -> {s.commit[:12]})"  # every symbol of the file moves to the new commit
    assert s.wiki == wiki.Refresh([("pkg", reason)], 1, 0, 0, None)
    assert pages_of(conn, name) == {"pkg": ("fresh", reason)}
    citations = {c.name: c for c in wiki.read(conn, "pkg", name).citations}
    assert citations["a3"].recorded_commit == s.commit == citations["a3"].current_commit


def test_uncited_change_and_no_change_mark_nothing(conn, wiki_repo):
    name = register(conn, wiki_repo)
    ingest.ingest(conn, name, ENV)
    wiki.build(conn, name)
    write(wiki_repo / "other" / "x.py", "def lonely():\n    return 1\n")
    git(wiki_repo, "commit", "-q", "-am", "edit uncited")
    assert ingest.ingest(conn, name, ENV).wiki == wiki.Refresh([], 0, 0, 0, None)
    assert ingest.ingest(conn, name, ENV).wiki is None  # current: nothing written
    git(wiki_repo, "revert", "--no-edit", "HEAD")
    git(wiki_repo, "revert", "--no-edit", "HEAD")
    s = ingest.ingest(conn, name, ENV)  # a new revision with the same content
    assert s.mode == "incremental" and s.wiki == wiki.Refresh([], 0, 0, 0, None)
    assert pages_of(conn, name) == {"pkg": ("fresh", None)}


def test_full_reingest_marks_and_recompiles_every_page(conn, wiki_repo):
    name = register(conn, wiki_repo)
    ingest.ingest(conn, name, ENV)
    wiki.build(conn, name)
    s = ingest.ingest(conn, name, ENV, full=True)
    assert s.wiki == wiki.Refresh([("pkg", "a0 was re-indexed")], 1, 0, 0, None)
    assert pages_of(conn, name) == {"pkg": ("fresh", "a0 was re-indexed")}
    assert all(c.path for c in wiki.read(conn, "pkg", name).citations)


def test_without_a_wiki_nothing_is_created(conn, wiki_repo):
    name = register(conn, wiki_repo)
    ingest.ingest(conn, name, ENV)
    write(wiki_repo / "pkg" / "a.py", pkg_module("a").replace("return 3", "return 33"))
    git(wiki_repo, "commit", "-q", "-am", "edit")
    assert ingest.ingest(conn, name, ENV).wiki is None
    assert pages_of(conn, name) == {}


def test_failed_recompile_keeps_the_ingestion(conn, wiki_repo, monkeypatch):
    name = register(conn, wiki_repo)
    ingest.ingest(conn, name, ENV)
    wiki.build(conn, name)
    write(wiki_repo / "pkg" / "b.py", pkg_module("b").replace("return 1", "return 11"))
    git(wiki_repo, "commit", "-q", "-am", "edit b1")

    def boom(*args, **kwargs):
        raise RuntimeError("compiler broke")

    monkeypatch.setattr(wiki, "_compile", boom)
    s = ingest.ingest(conn, name, ENV)
    assert s.wiki.stale and s.wiki.stale[0][1].startswith("b0 changed") and s.wiki.recompiled == 0
    assert "compiler broke" in s.wiki.error
    assert revision(conn, name) == (head(wiki_repo), False)  # the ingestion is committed
    assert pages_of(conn, name)["pkg"][0] == "stale"
