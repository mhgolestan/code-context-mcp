"""ccctx command line (contracts/cli.md). Database cases skip when it is unreachable."""

import subprocess
import time
import uuid
from collections import Counter

import pytest

from golestan_code_mcp import cli, db, ingest, registry
from golestan_code_mcp.config import load

FAKE_PW = "Not-A-Real-Pw-42"


@pytest.fixture(scope="module")
def live():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    yield
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()
    conn.close()


def run(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_add_then_list(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    code, out, _ = run(capsys, "repo", "add", str(tmp_path), "--name", n)
    assert code == 0
    assert out.strip() == f"registered {n} {tmp_path.resolve()}"

    code, out, _ = run(capsys, "repo", "list")
    assert code == 0
    header, *rows = out.splitlines()
    assert header.split() == ["NAME", "PATH", "FILES", "SYMBOLS"]
    mine = [r for r in rows if r.split()[0] == n]
    assert len(mine) == 1 and mine[0].split()[-2:] == ["0", "0"]
    assert len({len(line.rstrip()) for line in [header, *rows]}) == 1  # right-aligned counts end in one column


@pytest.mark.parametrize(("path", "name", "message"), [
    ("{tmp}/missing", "t-ok", "does not exist"),
    ("{tmp}/file.txt", "t-ok", "is not a directory"),
    ("{tmp}", "Bad", "invalid name"),
])
def test_refusals_without_database(capsys, tmp_path, monkeypatch, path, name, message):
    monkeypatch.setenv("DB_PORT", "1")  # proves these never reach the database
    (tmp_path / "file.txt").write_text("x")
    code, out, err = run(capsys, "repo", "add", path.format(tmp=tmp_path), "--name", name)
    assert code == 1 and out == ""
    assert err.startswith("error: ") and message in err


def test_duplicate_name_exits_1(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert run(capsys, "repo", "add", str(tmp_path / "a"), "--name", n)[0] == 0
    code, _, err = run(capsys, "repo", "add", str(tmp_path / "b"), "--name", n)
    assert code == 1 and "already registered" in err


def test_empty_registry(capsys, monkeypatch, live):
    monkeypatch.setattr(registry, "list_repos", lambda conn: [])
    code, out, _ = run(capsys, "repo", "list")
    assert code == 0 and out.strip() == "no repositories registered"


def test_format_table():
    rows = [("flask", "/r/flask", 0, 0), ("gin", "/r/gin", 12, 3456)]
    assert cli.format_table(("NAME", "PATH", "FILES", "SYMBOLS"), rows, frozenset({2, 3})).splitlines() == [
        "NAME   PATH      FILES  SYMBOLS",
        "flask  /r/flask      0        0",
        "gin    /r/gin       12     3456",
    ]


def test_migrate(live, capsys):
    code, out, _ = run(capsys, "db", "migrate")
    assert code == 0
    assert out.startswith(("applied ", "schema up to date (version "))


@pytest.mark.parametrize("argv", [
    ["repo", "list"], ["db", "migrate"], ["repo", "add", ".", "--name", "t-x"],
    ["ingest", "flask"], ["symbols", "--repo", "flask"], ["stats", "--by-language"],
])
def test_unreachable_exits_2_fast_without_secret(capsys, monkeypatch, argv):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", FAKE_PW)
    started = time.monotonic()
    code, out, err = run(capsys, *argv)
    assert time.monotonic() - started < 5
    assert code == 2 and err.startswith("error: ")
    assert FAKE_PW not in out + err


def test_missing_setting_exits_1(capsys, monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "")
    code, _, err = run(capsys, "repo", "list")
    assert code == 1 and "DB_PASSWORD" in err


def summary(skipped=(), no_grammar=0, parse_errors=0, files=176, symbols=1847):
    return ingest.Summary("flask", "a" * 40, False, Counter(dict(skipped)), files, no_grammar, parse_errors, symbols, 9.2)


def test_format_summary_matches_request():
    s = summary({"ignored": 12, "binary": 24, "too large": 2})
    assert cli.format_summary(s).splitlines() == [
        "Scanned   214 files",
        "Skipped    38 files (12 ignored, 24 binary, 2 too large)",
        "Indexed 1,847 symbols in 9.2s",
    ]


def test_format_summary_fallback_and_reason_order():
    s = summary({"too large": 1, "vendored": 3}, no_grammar=15, parse_errors=2, files=20, symbols=40)
    assert cli.format_summary(s).splitlines() == [
        "Scanned 24 files",
        "Skipped  4 files (3 vendored, 1 too large)",
        "Indexed 40 symbols in 9.2s",
        "Fallback 17 files as line windows (15 no grammar, 2 parse errors)",
    ]


def test_format_summary_nothing_skipped():
    assert cli.format_summary(summary(files=3, symbols=5)).splitlines()[1] == "Skipped 0 files"


def make_repo(root):
    root.mkdir()
    (root / "app.py").write_text("def a():\n    pass\n\n\nclass B:\n    def c(self):\n        pass\n")
    (root / "broken.py").write_text("def oops(:\n")
    (root / "notes.md").write_text("# notes\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    return root


@pytest.fixture
def ingested(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(make_repo(tmp_path / "r")), "--name", n)
    code, out, err = run(capsys, "ingest", n)
    assert code == 0, err
    return n, out


def test_ingest_prints_summary(ingested):
    _, out = ingested
    lines = out.splitlines()
    assert [line.split()[0] for line in lines] == ["Scanned", "Skipped", "Indexed", "Fallback"]
    assert lines[0].split()[1] == "3" and lines[2].endswith("s")
    assert "(1 no grammar, 1 parse errors)" in lines[3]


def test_ingest_refusals_exit_1(live, capsys, tmp_path):
    assert run(capsys, "ingest", "t-missing")[0] == 1
    n = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(tmp_path), "--name", n)
    code, _, err = run(capsys, "ingest", n)
    assert code == 1 and "not a git repository" in err


def test_symbols_lists_parsed_first(ingested, capsys):
    n, _ = ingested
    code, out, _ = run(capsys, "symbols", "--repo", n, "--limit", "3")
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["NAME", "KIND", "LANGUAGE", "PATH", "LINES"]
    assert [r.split()[:2] for r in rows] == [["a", "function"], ["B", "class"], ["B.c", "method"]]
    assert rows[0].split()[2:] == ["python", "app.py", "1-2"]
    code, out, _ = run(capsys, "symbols", "--repo", n, "--kind", "chunk")
    assert {r.split()[1] for r in out.splitlines()[1:]} == {"chunk"}


def test_stats_by_language(ingested, capsys):
    n, _ = ingested
    code, out, _ = run(capsys, "stats", "--by-language", "--repo", n)
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["REPO", "LANGUAGE", "FILES", "SYMBOLS", "CHUNKS"]
    assert [r.split() for r in rows] == [[n, "markdown", "1", "0", "1"], [n, "python", "2", "3", "1"]]
