"""ccctx command line (contracts/cli.md). Database cases skip when it is unreachable."""

import re
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import replace

import pytest

from golestan_code_mcp import cli, db, graph, ingest, registry, semantic
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
def model(live):
    with db.connect(load()) as conn:
        try:
            semantic.require_model(conn)
        except (semantic.ModelError, Exception) as e:
            pytest.skip(f"embedding model not loaded: {type(e).__name__}")


@pytest.fixture
def ingested(model, capsys, tmp_path):
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


def test_ingest_refusals_exit_1(model, capsys, tmp_path):
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


# --- model load and search (004) ----------------------------------------------------------------------------


def test_model_load_when_present(model, capsys):
    code, out, _ = run(capsys, "model", "load")
    assert code == 0 and out.strip() == "model ALL_MINILM_L12_V2 already present (384 dimensions)"


HEADER = ["RANK", "SCORE", "KEYWORD", "SEMANTIC", "REPO", "KIND", "NAME", "PATH", "LINES"]


def table(out):
    header, *rows = out.splitlines()
    assert header.split() == HEADER
    return [r.split() for r in rows]


def test_search_prints_ranked_table(ingested, capsys):
    n, _ = ingested
    code, out, _ = run(capsys, "search", "a", "--repo", n)
    rows = table(out)
    assert code == 0 and rows[0][:1] == ["1"] and rows[0][6] == "a"  # the definition of `a` comes first
    assert all(len(r[1].split(".")[1]) == 2 for r in rows)  # score with two decimals
    assert all(r[2] == "-" or r[2].isdigit() for r in rows) and all(r[3] == "-" or r[3].isdigit() for r in rows)


@pytest.mark.parametrize(("mode", "keyword_dash", "semantic_dash"), [("text", False, True), ("vector", True, False)])
def test_search_modes(ingested, capsys, mode, keyword_dash, semantic_dash):
    n, _ = ingested
    code, out, _ = run(capsys, "search", "class", "--repo", n, "--mode", mode)
    rows = table(out)
    assert code == 0 and rows
    assert all((r[2] == "-") == keyword_dash and (r[3] == "-") == semantic_dash for r in rows)


def test_search_unknown_mode_refused_before_connecting(capsys, monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    code, out, err = run(capsys, "search", "url_for", "--mode", "bogus")
    assert code == 1 and out == "" and "valid modes: vector, text, hybrid" in err


def test_search_repo_flags_narrow(ingested, capsys, monkeypatch):
    n, _ = ingested
    seen = []
    monkeypatch.setattr(semantic, "search", lambda conn, question, repos=None, mode="hybrid", env=None:
                        seen.append((repos, mode)) or [])
    run(capsys, "search", "x", "--repo", n, "--repo", f"{n},gin", "--repo", n)
    run(capsys, "search", "x")
    assert seen == [([n, "gin"], "hybrid"), (None, "hybrid")]


@pytest.mark.parametrize(("mode", "message"), [
    ("vector", "no symbols similar enough (similarity >= 0.30)"),
    ("hybrid", "no matching symbols"),
    ("text", "no matching symbols"),
])
def test_search_empty_results(ingested, capsys, monkeypatch, mode, message):
    n, _ = ingested
    monkeypatch.setattr(semantic, "search", lambda *args, **kwargs: [])
    code, out, _ = run(capsys, "search", "recipe for sourdough bread", "--repo", n, "--mode", mode)
    assert code == 0 and out.strip() == message


@pytest.mark.parametrize("question", ["", "   "])
def test_search_empty_question_refused_before_connecting(capsys, monkeypatch, question):
    monkeypatch.setenv("DB_PORT", "1")
    code, out, err = run(capsys, "search", question)
    assert code == 1 and out == "" and "question" in err


def test_search_unknown_repo(model, capsys):
    code, _, err = run(capsys, "search", "routing", "--repo", "t-missing")
    assert code == 1 and "no repository named 't-missing'" in err


def test_search_missing_model_hint(live, capsys, monkeypatch):
    monkeypatch.setattr(semantic, "INDEX", "NOT_AN_INDEX")
    code, _, err = run(capsys, "search", "routing")
    assert code == 1 and "ccctx model load" in err


def test_search_unreachable_exits_2(capsys, monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", FAKE_PW)
    started = time.monotonic()
    code, out, err = run(capsys, "search", "routing")
    assert time.monotonic() - started < 5 and code == 2 and FAKE_PW not in out + err


# --- incremental ingest (006) -------------------------------------------------------------------------------


def test_format_incremental_summary():
    s = replace(summary(no_grammar=3, files=8, symbols=84), mode="incremental",
                added=2, changed=6, removed=1, unchanged=227)
    assert cli.format_summary(s).splitlines() == [
        "Changed     9 files (2 added, 6 changed, 1 removed)",
        "Unchanged 227 files",
        "Skipped     0 files",
        "Indexed    84 symbols in 9.2s",
        "Fallback    3 files as line windows (3 no grammar, 0 parse errors)",
    ]


def test_format_up_to_date():
    s = replace(summary(files=0, symbols=0), commit="0123456789ab" + "c" * 28, mode="current", unchanged=176)
    assert cli.format_summary(s) == "flask is already up to date at 0123456789ab; run with --full to re-index anyway"


def untimed(out):
    return re.sub(r"in [0-9.]+s", "", out)


def test_ingest_again_then_full(ingested, capsys):
    n, first = ingested
    code, out, _ = run(capsys, "ingest", n)
    assert code == 0 and out.startswith(f"{n} is already up to date at ") and "--full" in out
    code, out, _ = run(capsys, "ingest", n, "--full")
    assert code == 0 and untimed(out) == untimed(first)  # same counts as the first ingest (SC-004)
    assert [line.split()[0] for line in out.splitlines()] == ["Scanned", "Skipped", "Indexed", "Fallback"]


def test_note_goes_to_stderr(live, capsys, monkeypatch):
    monkeypatch.setattr(ingest, "ingest", lambda conn, name, full=False: replace(
        summary(), note="no recorded revision for 'flask'; indexing in full"))
    code, out, err = run(capsys, "ingest", "flask")
    assert code == 0 and out.startswith("Scanned")
    assert "note: no recorded revision for 'flask'; indexing in full" in err


# --- callers, calls, impact (007) ---------------------------------------------------------------------------


@pytest.fixture
def graphed(model, capsys, tmp_path):
    root = tmp_path / "g"
    root.mkdir()
    (root / "g.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return a()\n\n\na()\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    n = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(root), "--name", n)
    assert run(capsys, "ingest", n)[0] == 0
    return n


def test_graph_tables(graphed, capsys):
    n = graphed
    code, out, _ = run(capsys, "callers", "a", "--repo", n)
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["CALLER", "KIND", "REPO", "PATH", "LINE", "CALLS"]
    assert [r.split() for r in rows] == [["b", "function", n, "g.py", "6", "a"], ["(module)", "file", n, "g.py", "9", "a"]]

    code, out, _ = run(capsys, "calls", "b", "--repo", n)
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["FROM", "LINE", "CALLEE", "KIND", "REPO", "PATH", "LINES"]
    assert [r.split() for r in rows] == [["b", "6", "a", "function", n, "g.py", "1-2"]]

    code, out, _ = run(capsys, "impact", "a", "--repo", n, "--depth", "1")
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["HOPS", "NAME", "KIND", "REPO", "PATH", "LINES"]
    assert [r.split() for r in rows] == [["1", "b", "function", n, "g.py", "5-6"]]


@pytest.mark.parametrize(("argv", "message"), [
    (["callers", "b"], "no callers of 'b'"),
    (["calls", "a"], "'a' calls no indexed symbols"),
    (["impact", "b"], "nothing calls 'b'"),
])
def test_graph_empty_results(graphed, capsys, argv, message):
    code, out, _ = run(capsys, *argv, "--repo", graphed)
    assert code == 0 and out.strip() == message


@pytest.mark.parametrize("command", ["callers", "calls", "impact"])
def test_graph_unknown_name_exits_1(graphed, capsys, command):
    code, out, err = run(capsys, command, "zzz_missing", "--repo", graphed)
    assert code == 1 and out == "" and "error: no symbol named 'zzz_missing'" in err


def test_graph_repo_flags_and_default_depth(live, capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(graph, "callers", lambda conn, name, repos=None: seen.append(("callers", repos)) or [])
    monkeypatch.setattr(graph, "calls", lambda conn, name, repos=None: seen.append(("calls", repos)) or [])
    monkeypatch.setattr(graph, "impact", lambda conn, name, depth=3, repos=None: seen.append((depth, repos)) or [])
    run(capsys, "callers", "x", "--repo", "t-a", "--repo", "t-a,t-b")
    run(capsys, "calls", "x")
    run(capsys, "impact", "x", "--repo", "t-a")
    run(capsys, "impact", "x", "--depth", "5")
    assert seen == [("callers", ["t-a", "t-b"]), ("calls", None), (3, ["t-a"]), (5, None)]


@pytest.mark.parametrize("depth", ["0", "6", "-1", "x", "2.5"])
def test_bad_depth_refused_before_connecting(capsys, monkeypatch, depth):
    monkeypatch.setenv("DB_PORT", "1")
    code, out, err = run(capsys, "impact", "url_for", "--depth", depth)
    assert code == 1 and out == "" and err.strip() == "error: depth must be between 1 and 5"


@pytest.mark.parametrize("command", ["callers", "calls", "impact"])
def test_graph_empty_name_refused_before_connecting(capsys, monkeypatch, command):
    monkeypatch.setenv("DB_PORT", "1")
    code, out, err = run(capsys, command, " ")
    assert code == 1 and out == "" and err.strip() == "error: the name is empty"
