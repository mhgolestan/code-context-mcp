"""Ingest pipeline (FR-001, FR-002, FR-009 to FR-014, FR-017, FR-018) against throwaway git repositories."""

import hashlib
import json
import logging
import os
import subprocess
import uuid
from collections import Counter

import pytest

from golestan_code_mcp import db, ingest, registry, schema, semantic
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


def test_reingest_replaces(conn, repo):
    name = register(conn, repo)
    first = ingest.ingest(conn, name, ENV)
    rows = stored(conn, name)
    second = ingest.ingest(conn, name, ENV)
    assert (second.files, second.symbols) == (first.files, first.symbols)
    assert sorted(map(repr, stored(conn, name))) == sorted(map(repr, rows))  # SC-005: one copy, same content


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
