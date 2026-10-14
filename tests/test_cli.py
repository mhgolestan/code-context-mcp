"""ccctx command line (contracts/cli.md). Database cases skip when it is unreachable."""

import asyncio
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import replace

import pytest

from golestan_code_mcp import cli, db, graph, ingest, registry, semantic, server
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


HEADER = ["RANK", "SCORE", "FEEDBACK", "KEYWORD", "SEMANTIC", "REPO", "KIND", "NAME", "PATH", "LINES"]


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


def test_search_shows_the_feedback_column(live, capsys, monkeypatch):
    hits = [semantic.Result(1, 48.0, 0.6, 1, 1, "flask", "function", "url_for", "src/flask/helpers.py", 1, 9, "python",
                            "def url_for():", relevance=40.0, feedback=0.2),
            semantic.Result(2, 30.0, 0.5, 2, 2, "flask", "function", "redirect", "src/flask/helpers.py", 10, 19,
                            "python", "def redirect():", relevance=30.0)]
    seen = []
    monkeypatch.setattr(semantic, "search", lambda *a, **k: seen.append(k.get("feedback", True)) or hits)
    code, out, _ = run(capsys, "search", "where is url_for")
    header, first, second = out.splitlines()
    assert code == 0 and header.split()[:3] == ["RANK", "SCORE", "FEEDBACK"]
    assert first.split()[:3] == ["1", "48.00", "+20%"] and second.split()[:3] == ["2", "30.00", "2"]  # blank at 0
    run(capsys, "search", "where is url_for", "--no-feedback")
    assert seen == [True, False]


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
    monkeypatch.setattr(semantic, "search", lambda conn, question, repos=None, mode="hybrid", env=None,
                        language=None, limit=10, **_: seen.append((repos, mode)) or [])
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



def test_search_language_and_limit_match_core(ingested, capsys):
    n, _ = ingested
    code, out, _ = run(capsys, "search", "a", "--repo", n, "--language", "python", "--limit", "2")
    with db.connect(load()) as conn:
        expected = semantic.search(conn, "a", [n], language="python", limit=2)
    assert code == 0 and [(r[0], r[6], r[7]) for r in table(out)] == [(str(h.rank), h.name, h.path) for h in expected]
    assert 1 <= len(expected) <= 2


def test_search_language_and_limit_reach_the_core(capsys, monkeypatch, live):
    seen = []
    monkeypatch.setattr(semantic, "search", lambda conn, question, repos=None, mode="hybrid", env=None,
                        language=None, limit=10, **_: seen.append((language, limit)) or [])
    run(capsys, "search", "x", "--language", "go", "--limit", "3")
    run(capsys, "search", "x")
    assert seen == [("go", 3), (None, 10)]


@pytest.mark.parametrize(("argv", "message"), [
    (["--limit", "0"], "limit must be between 1 and 50"),
    (["--limit", "51"], "limit must be between 1 and 50"),
    (["--limit", "ten"], "limit must be between 1 and 50"),
    (["--language", "cobol"], f"unknown language 'cobol'; valid languages: {', '.join(semantic.LANGUAGES)}"),
])
def test_search_bad_language_or_limit_refused_before_connecting(capsys, monkeypatch, argv, message):
    monkeypatch.setenv("DB_PORT", "1")
    code, out, err = run(capsys, "search", "routing", *argv)
    assert code == 1 and out == "" and err.strip() == f"error: {message}"

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



# --- ccctx mcp (008) ----------------------------------------------------------------------------------------


def test_mcp_starts_the_server_without_connecting(capsys, monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")  # the server reports an outage per call; starting never connects
    started = []
    monkeypatch.setattr(server, "main", lambda: started.append(True))
    code, out, _ = run(capsys, "mcp")
    assert code == 0 and started == [True] and out == ""


def test_mcp_child_process_serves_the_tools_from_any_directory(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # the default child environment carries no DB_* settings, so they must come from the project's .env
    params = StdioServerParameters(command=sys.executable, args=["-m", "golestan_code_mcp.cli", "mcp"],
                                   cwd=str(tmp_path))

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()  # fails if anything but protocol messages reaches stdout
            tools = [t.name for t in (await client.list_tools()).tools]
            repos = (await client.call_tool("list_repositories", {})).structuredContent
            # recall loads the memory package (numpy): on Windows a lazy load under the stdio reader used to hang
            recall = await asyncio.wait_for(client.call_tool("recall", {"limit": 1}), 60)
            return tools, repos, recall.structuredContent

    tools, repos, recall = asyncio.run(session())
    assert tools == ["search_code", "get_symbol", "find_callers", "find_callees", "impact_of", "read_wiki",
                     "list_repositories", "recall", "mark_used"]
    try:
        db.connect(load()).close()
    except Exception:
        return  # without a database the listing above is the whole check
    assert repos["database"] == "reachable", repos["reason"]
    assert recall["status"] in ("ok", "unavailable")  # answered, whatever the store's state


# --- wiki (009 contracts/cli.md) ------------------------------------------------------------------------------


WIKI_PKG = '"""Ledger accounting for the demo."""\n' + "".join(
    f'\n\ndef post{i}():\n    """Post entry {i}."""\n    return {i}\n' for i in range(10))
WIKI_USER = "from pkg.core import post0\n\n\ndef use():\n    return post0()\n"


def make_wiki_repo(root):
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "app").mkdir()
    (root / "src" / "pkg" / "__init__.py").write_text('"""The demo ledger package."""\n')
    (root / "src" / "pkg" / "core.py").write_text(WIKI_PKG)
    (root / "app" / "main.py").write_text(WIKI_USER)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    return root


@pytest.fixture
def wikirepo(model, capsys, tmp_path):
    """An ingested repository with one qualifying directory, src/pkg (the page `pkg`), and its wiki built."""
    n = f"t-{uuid.uuid4().hex[:8]}"
    root = make_wiki_repo(tmp_path / "w")
    run(capsys, "repo", "add", str(root), "--name", n)
    assert run(capsys, "ingest", n)[0] == 0
    code, out, err = run(capsys, "wiki", "build", "--repo", n)
    assert code == 0, err
    return n, root, out, err


def test_wiki_build_prints_one_line_per_repository(wikirepo, capsys):
    n, _, out, err = wikirepo
    assert re.fullmatch(rf"{n}: 1 pages \(1 added, 0 updated, 0 removed\) in \d+\.\ds\n", out)
    assert '"event": "wiki"' in err
    code, out, _ = run(capsys, "wiki", "build", "--repo", n)
    assert code == 0 and re.fullmatch(rf"{n}: 1 pages \(0 added, 0 updated, 0 removed\) in \d+\.\ds\n", out)


def test_wiki_build_notes_when_pages_are_not_searchable(wikirepo, capsys, monkeypatch):
    from golestan_code_mcp import wiki

    monkeypatch.setattr(wiki, "ensure_index", lambda cur: False)
    code, _, err = run(capsys, "wiki", "build", "--repo", wikirepo[0])
    assert code == 0 and "note: wiki pages are not searchable until you run `ccctx model load`" in err


def test_wiki_list_columns(wikirepo, capsys):
    n = wikirepo[0]
    code, out, _ = run(capsys, "wiki", "list", "--repo", n)
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["NAME", "REPO", "STATE", "SYMBOLS", "COMPILED", "LAST", "STALE"]
    assert len(rows) == 1
    assert re.fullmatch(rf"pkg\s+{n}\s+fresh\s+10\s+\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d\s+-", rows[0])


def test_wiki_list_without_pages(live, capsys, monkeypatch):
    from golestan_code_mcp import wiki

    monkeypatch.setattr(wiki, "pages", lambda conn, repo=None: [])
    code, out, _ = run(capsys, "wiki", "list")
    assert code == 0 and out.strip() == "no wiki pages"


def test_wiki_show_prints_the_stored_text_then_sources(wikirepo, capsys):
    from golestan_code_mcp import wiki

    n = wikirepo[0]
    with db.connect(load()) as conn:
        page = wiki.read(conn, "pkg", n)
    code, out, _ = run(capsys, "wiki", "show", "pkg", "--repo", n)
    assert code == 0
    header, _, rest = out.partition("\n")
    assert re.fullmatch(rf"pkg \({n}, src/pkg\) fresh, compiled \d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d by ccctx-wiki/1",
                        header)
    assert page.text.rstrip("\n") in rest
    sources = rest.split("## Sources\n", 1)[1].splitlines()
    assert len(sources) == len(page.citations)
    assert "- `post0` (function, key) src/pkg/core.py:" in rest
    assert "- `src/pkg/core.py` (file, file)" in rest


def test_wiki_citations_table(wikirepo, capsys):
    n = wikirepo[0]
    code, out, _ = run(capsys, "wiki", "citations", "pkg", "--repo", n)
    header, *rows = out.splitlines()
    assert code == 0 and header.split() == ["NAME", "KIND", "ROLE", "REPO", "PATH", "LINES", "COMPILED", "CURRENT"]
    key = next(r.split() for r in rows if r.split()[0] == "post0")
    assert key[1:5] == ["function", "key", n, "src/pkg/core.py"] and re.fullmatch(r"\d+-\d+", key[5])
    assert len(key[6]) == 12 and key[6] == key[7]
    file_row = next(r.split() for r in rows if r.split()[0] == "core.py")
    assert file_row[1:3] == ["file", "file"] and file_row[5] == "-"


def test_wiki_refusals_exit_1(wikirepo, capsys, tmp_path):
    n = wikirepo[0]
    assert run(capsys, "wiki", "show", "pkg", "--repo", "t-nope") == (1, "", "error: no repository named 't-nope'\n")
    assert run(capsys, "wiki", "show", "nope", "--repo", n)[2] == f"error: no wiki page named 'nope' in '{n}'\n"
    empty = f"t-{uuid.uuid4().hex[:8]}"
    (tmp_path / "e").mkdir()
    run(capsys, "repo", "add", str(tmp_path / "e"), "--name", empty)
    code, _, err = run(capsys, "wiki", "build", "--repo", empty)
    assert code == 1 and err == f"error: repository '{empty}' is not indexed yet; run `ccctx ingest {empty}`\n"


def test_wiki_ambiguous_page_needs_repo(wikirepo, capsys, tmp_path):
    other = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(make_wiki_repo(tmp_path / "w2")), "--name", other)
    run(capsys, "ingest", other)
    run(capsys, "wiki", "build", "--repo", other)
    code, _, err = run(capsys, "wiki", "show", "pkg")
    listed = re.fullmatch(r"error: wiki page 'pkg' is in more than one repository: (.+); pass --repo\n", err)
    names = listed.group(1).split(", ")  # earlier tests' repositories have a `pkg` page too until teardown
    assert code == 1 and {wikirepo[0], other} <= set(names) and names == sorted(names)


def test_wiki_unreachable_exits_2(capsys, monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    assert run(capsys, "wiki", "list")[0] == 2


def test_ingest_prints_the_pages_it_recompiled(wikirepo, capsys):
    n, root, _, _ = wikirepo
    core = root / "src" / "pkg" / "core.py"
    core.write_text(WIKI_PKG.replace("return 0\n", "return -1\n"))
    for args in (["add", "-A"], ["commit", "-q", "-m", "tweak"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    code, out, _ = run(capsys, "ingest", n)
    lines = out.splitlines()
    assert code == 0
    assert re.fullmatch(r"Wiki      pkg stale: \S+ (changed \([0-9a-f]{12} -> [0-9a-f]{12}\)|is gone), recompiled",
                        lines[-2])
    assert lines[-1] == "Wiki      1 stale, 1 recompiled, 0 added, 0 removed"
    assert "fresh" in run(capsys, "wiki", "list", "--repo", n)[1]


def test_ingest_summary_wiki_lines():
    from golestan_code_mcp.wiki import Refresh

    s = replace(summary(), mode="incremental", added=0, changed=1, removed=0, unchanged=3,
                wiki=Refresh([("flask/json", "dumps changed (5f2c1a9e04b7 -> 9e0d3c1b2a47)")], 1))
    assert cli.format_summary(s).splitlines()[-2:] == [
        "Wiki      flask/json stale: dumps changed (5f2c1a9e04b7 -> 9e0d3c1b2a47), recompiled",
        "Wiki      1 stale, 1 recompiled, 0 added, 0 removed",
    ]
    failed = replace(s, wiki=Refresh([("flask/json", "dumps is gone")], error="boom"))
    assert cli.format_summary(failed).splitlines()[-2] == "Wiki      flask/json stale: dumps is gone, not recompiled"
    assert cli.format_summary(replace(s, wiki=Refresh([]))).splitlines()[-1] == \
        "Wiki      0 stale, 0 recompiled, 0 added, 0 removed"
    assert "Wiki" not in cli.format_summary(replace(s, wiki=None))


def test_ingest_warns_when_the_wiki_was_not_recompiled(model, capsys, monkeypatch):
    from golestan_code_mcp.wiki import Refresh

    s = replace(summary(), mode="incremental", added=0, changed=1, removed=0, unchanged=3,
                wiki=Refresh([("pkg", "post0 is gone")], error="boom"))
    monkeypatch.setattr(ingest, "ingest", lambda conn, name, full=False: s)
    code, _, err = run(capsys, "ingest", "flask")
    assert code == 0 and "warning: wiki not recompiled: boom" in err


def test_search_shows_page_rows(live, capsys, monkeypatch):
    page = semantic.Result(1, 41.7, 0.63, 1, 1, "flask", "page", "flask/json", "src/flask/json", None, None,
                           "python", "# flask/json", "stale")
    monkeypatch.setattr(semantic, "search", lambda *a, **k: [page])
    code, out, _ = run(capsys, "search", "what does the json module do")
    row = out.splitlines()[1]
    assert code == 0 and "page (stale)" in row and row.split()[-1] == "-" and "src/flask/json" in row


# --- ccctx memory list (specs/010-agent-memory contracts/cli.md) ---------------------------------------------

from golestan_code_mcp import memory  # noqa: E402


def a_run(tool, question, repo, results=(), status="ok", reason=None, client="test-client 9.9"):
    arguments = {"query": question, "repo": repo} if question else {}
    return {"tool": tool, "arguments": arguments, "question": question, "client": client, "status": status,
            "reason": reason, "ms": 3, "result_count": len(results), "results": list(results), "repos": [repo]}


@pytest.fixture
def remembered(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    assert run(capsys, "repo", "add", str(tmp_path), "--name", n)[0] == 0
    with db.connect(load()) as conn:
        memory.store_runs(conn, [
            a_run("search_code", "where is url_for", n, [{"kind": "function", "repo": n, "name": "url_for",
                                                          "path": "src/h.py", "start_line": 188, "end_line": 246}]),
            a_run("find_callers", "url_for", n, [{"kind": "method", "repo": n, "name": "Flask.go", "path": "app.py",
                                                  "line": 88, "callee": "url_for"}], client=None),
            a_run("read_wiki", None, n, [{"kind": "page", "repo": n, "name": "flask/json", "path": "src/flask/json"}]),
            a_run("get_symbol", "nope", n, status="unavailable", reason="no symbol named 'nope'"),
        ])
    return n


def test_memory_list_newest_first(remembered, capsys):
    code, out, _ = run(capsys, "memory", "list", "--repo", remembered)
    rows = out.splitlines()
    assert code == 0 and rows[0].split() == ["RUN", "WHEN", "TOOL", "QUESTION", "CLIENT", "STATUS", "RESULTS", "TOP"]
    tools = [line.split()[3] for line in rows[1:]]
    assert tools == ["get_symbol", "read_wiki", "find_callers", "search_code"]
    assert "no symbol named 'nope'" in rows[1] and "unavailable" in rows[1]
    assert "flask/json src/flask/json" in rows[2] and " - " in rows[2]  # a page's TOP, and no question
    assert "Flask.go app.py:88" in rows[3] and rows[3].count(" - ") >= 1  # a call line's TOP, and no client
    assert "url_for src/h.py:188-246" in rows[4] and "test-client 9.9" in rows[4]


def test_memory_list_limit_and_empty(remembered, capsys, tmp_path):
    code, out, _ = run(capsys, "memory", "list", "--repo", remembered, "--limit", "1")
    assert code == 0 and len(out.splitlines()) == 2
    other = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(tmp_path / "x"), "--name", other) if (tmp_path / "x").mkdir() is None else None
    assert run(capsys, "memory", "list", "--repo", other)[1].strip() == "no runs recorded"


def test_memory_list_refusals(live, capsys):
    code, _, err = run(capsys, "memory", "list", "--repo", "t-nowhere")
    assert code == 1 and "no repository named 't-nowhere'" in err
    with pytest.raises(SystemExit) as exit_:
        cli.main(["memory", "list", "--limit", "0"])
    assert exit_.value.code == 2


def test_memory_list_shows_used_of_returned(remembered, capsys):
    from golestan_code_mcp import feedback

    with db.connect(load()) as conn, conn.cursor() as cur:
        rid = registry.repo_id(conn, remembered)
        search = cur.execute("""select max(r.id) from ccctx_memory_runs r join ccctx_memory_run_repos l
                                 on l.run_id = r.id where l.repo_id = :r and r.tool = 'search_code'""",
                             r=rid).fetchone()[0]
        feedback.write_use(cur, {"repo_id": rid, "path": "src/h.py", "name": "url_for", "kind": "function",
                                 "how": "cited", "client": "t", "session": uuid.uuid4().hex, "run_id": search})
        conn.commit()
    code, out, _ = run(capsys, "memory", "list", "--repo", remembered)
    rows = {line.split()[3]: line for line in out.splitlines()[1:]}
    assert code == 0 and " 1/1 " in rows["search_code"]
    assert not re.search(r" \d+/\d+ ", rows["find_callers"])  # the plain count otherwise


def symbol_run(n, question, symbol, path, kind):
    return a_run("search_code", question, n, [{"kind": kind, "repo": n, "name": symbol, "path": path,
                                               "start_line": 1, "end_line": 9}])


@pytest.fixture
def distilled(ingested, capsys, monkeypatch):
    """An ingested repository with three runs about two of its symbols, not yet consolidated."""
    n, _ = ingested
    monkeypatch.delenv("CCCTX_MEMORY_MODEL", raising=False)
    with db.connect(load()) as conn:
        with conn.cursor() as cur:
            rows = cur.execute("select s.name, f.path, s.kind from ccctx_symbols s join ccctx_files f on"
                               " f.id = s.file_id join ccctx_repos r on r.id = f.repo_id where r.name = :n and"
                               " s.kind in ('function', 'method') order by s.kind, s.name", n=n).fetchall()
        (first, p1, k1), (second, p2, k2) = rows  # a (function), then B.c (method)
        memory.store_runs(conn, [symbol_run(n, "where is the first", first, p1, k1),
                                 symbol_run(n, "where's the first?", first, p1, k1),
                                 symbol_run(n, "and the second", second, p2, k2)])
    yield n, first, second
    with db.connect(load()) as conn:
        memory.forget(conn, n)


def test_memory_consolidate_line_and_rerun(distilled, capsys):
    n, _, _ = distilled
    code, out, _ = run(capsys, "memory", "consolidate", "--repo", n)
    assert code == 0
    assert re.fullmatch(rf"{n}: 3 runs read, 2 created, 1 merged, 0 weakened, 0 stale, 0 re-asked in \d+\.\ds"
                        rf" \(template/1\)", out.strip())
    code, out, _ = run(capsys, "memory", "consolidate", "--repo", n)
    assert re.fullmatch(rf"{n}: 0 runs read, 0 created, 0 merged, 0 weakened, 0 stale, 0 re-asked in \d+\.\ds"
                        rf" \(template/1\)", out.strip())


def test_memory_facts_table(distilled, capsys):
    n, first, second = distilled
    run(capsys, "memory", "consolidate", "--repo", n)
    code, out, _ = run(capsys, "memory", "facts", "--repo", n)
    rows = out.splitlines()
    assert code == 0
    assert rows[0].split() == ["FACT", "REPO", "SUBJECT", "CONFIDENCE", "STATE", "DISTILLER", "RUNS", "TEXT"]
    cells = [line.split() for line in rows[1:]]
    assert [c[2] for c in cells] == [first, second]  # 0.750 before 0.500
    assert [c[3] for c in cells] == ["0.750", "0.500"] and all(len(c[0]) == 8 for c in cells)
    assert cells[0][6].count(",") == 1 and cells[0][6].split(",") == sorted(cells[0][6].split(","), key=int)
    assert all(len(line.split("template/1", 1)[1].split(None, 1)[1]) <= 125 for line in rows[1:])


def test_memory_facts_empty_and_refusals(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    run(capsys, "repo", "add", str(tmp_path), "--name", n)
    assert run(capsys, "memory", "facts", "--repo", n)[1].strip() == "no facts"
    for cmd in ("facts", "consolidate"):
        code, _, err = run(capsys, "memory", cmd, "--repo", "t-nowhere")
        assert code == 1 and "no repository named 't-nowhere'" in err


def test_ingest_summary_memory_line():
    quiet = memory.Consolidation("r")
    busy = memory.Consolidation("r", runs=3, created=1, merged=2, stale=1, reasked=2)
    assert "Memory" not in cli.format_summary(replace(summary(), memory=quiet))
    assert cli.format_summary(replace(summary(), memory=None)) == cli.format_summary(summary())
    assert cli.format_summary(replace(summary(), memory=busy)).splitlines()[-1] == (
        "Memory    3 runs read, 1 created, 2 merged, 0 weakened, 1 stale, 2 re-asked")
    assert cli.format_summary(replace(summary(), memory="memory down")).splitlines()[-1] == (
        "Memory    not consolidated: memory down")
    current = replace(summary(), mode="current", commit="a" * 40, memory=busy)
    assert cli.format_summary(current).splitlines()[1].startswith("Memory    3 runs read")


# --- ccctx feedback (specs/011-usage-feedback contracts/cli.md) -------------------------------------------------

from golestan_code_mcp import feedback  # noqa: E402


def test_feedback_add_then_list(ingested, capsys):
    n, _ = ingested
    code, out, _ = run(capsys, "feedback", "add", "a", "--repo", n, "--count", "3", "--days-ago", "30")
    assert code == 0 and out.strip() == f"recorded 3 uses of a (app.py) in {n}, dated 30 days ago"
    assert run(capsys, "feedback", "add", "B.c", "--repo", n, "--path", "app.py")[1].strip() == (
        f"recorded 1 use of B.c (app.py) in {n}, dated 0 days ago")
    code, out, _ = run(capsys, "feedback", "list", "--repo", n)
    header, first, second = out.splitlines()
    assert code == 0 and header.split() == ["KIND", "NAME", "PATH", "USES", "WEIGHT", "BOOST", "LAST", "USED", "INDEXED"]
    assert first.split()[:6] == ["function", "a", "app.py", "3", "1.50", "+6%"] and first.split()[-1] == "yes"
    assert second.split()[:6] == ["method", "B.c", "app.py", "1", "1.00", "+4%"]
    assert re.search(r" \d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", first)


def test_feedback_list_empty_and_unknown(ingested, capsys):
    n, _ = ingested
    assert run(capsys, "feedback", "list", "--repo", n)[1].strip() == f"no feedback recorded for {n}"
    code, _, err = run(capsys, "feedback", "list", "--repo", "t-nowhere")
    assert code == 1 and "no repository named 't-nowhere'" in err


def test_feedback_add_refusals(ingested, capsys, monkeypatch):
    n, _ = ingested
    code, _, err = run(capsys, "feedback", "add", "zz_nothing", "--repo", n)
    assert code == 1 and err.startswith("error: not found: no indexed symbol or page named 'zz_nothing'")
    monkeypatch.setattr(feedback, "add", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("'a' is at several paths; pass --path:\nx.py\ny.py")))
    code, _, err = run(capsys, "feedback", "add", "a", "--repo", n)
    assert code == 1 and err == "error: 'a' is at several paths; pass --path:\nx.py\ny.py\n"


@pytest.mark.parametrize("argv", [["--days-ago", "-1"], ["--count", "0"], ["--count", "100001"], ["--count", "x"]])
def test_feedback_add_bad_numbers_exit_2(capsys, argv):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["feedback", "add", "a", "--repo", "flask", *argv])
    assert exit_.value.code == 2


def test_feedback_reset(ingested, capsys, monkeypatch):
    n, _ = ingested
    run(capsys, "feedback", "add", "a", "--repo", n, "--count", "2")
    assert run(capsys, "feedback", "reset", "--repo", n)[1].strip() == f"deleted 2 uses in {n}"
    assert run(capsys, "feedback", "list", "--repo", n)[1].strip() == f"no feedback recorded for {n}"
    seen = []
    monkeypatch.setattr(feedback, "reset", lambda conn, repo: seen.append(repo) or 7)
    assert run(capsys, "feedback", "reset", "--all")[1].strip() == "deleted 7 uses in all repositories"
    assert seen == [None]


@pytest.mark.parametrize("argv", [[], ["--repo", "flask", "--all"]])
def test_feedback_reset_needs_exactly_one_scope(capsys, argv):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["feedback", "reset", *argv])
    assert exit_.value.code == 2


# --- ccctx eval (specs/011-usage-feedback contracts/cli.md) -----------------------------------------------------


def test_eval_prints_questions_recall_and_misses(live, capsys, monkeypatch, tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text("{}", encoding="utf-8")
    seen = []
    monkeypatch.setattr(feedback, "load_gold", lambda path, conn: seen.append(path) or ["q"] * 3)
    monkeypatch.setattr(feedback, "evaluate", lambda conn, g, with_feedback, limit: seen.append((with_feedback, limit))
                        or feedback.Evaluation(3, 2, ["how does gin bind a JSON request body?"]))
    code, out, _ = run(capsys, "eval", "--gold", str(gold))
    assert code == 0 and out.splitlines() == ["Questions: 3", "recall@10: 0.67 (2/3)",
                                              "missed: how does gin bind a JSON request body?"]
    code, out, _ = run(capsys, "eval", "--gold", str(gold), "--with-feedback", "--limit", "5")
    assert out.splitlines()[1] == "recall@5: 0.67 (2/3)"
    assert seen == [str(gold), (False, 10), str(gold), (True, 5)]


def test_eval_gold_errors_exit_1(live, capsys, tmp_path):
    code, _, err = run(capsys, "eval", "--gold", str(tmp_path / "none.json"))
    assert code == 1 and err.startswith("error: gold file not found: ")
    (tmp_path / "bad.json").write_text('{"version": 1, "questions": [{"question": "x", "paths": []}]}')
    code, _, err = run(capsys, "eval", "--gold", str(tmp_path / "bad.json"))
    assert code == 1 and err.strip() == "error: gold file: question 1 has no paths"


@pytest.mark.parametrize("limit", ["0", "51", "x"])
def test_eval_bad_limit_exits_2(capsys, limit):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["eval", "--gold", "gold.json", "--limit", limit])
    assert exit_.value.code == 2


# --- language coverage (012, US3): the Reparsed line ---


def test_incremental_summary_reports_reparsed_files():
    s = replace(summary(parse_errors=1, files=48, symbols=412), mode="incremental", added=0, changed=48, removed=0,
                unchanged=51, reparsed=Counter(typescript=2, csharp=46))
    assert cli.format_summary(s).splitlines() == [
        "Changed    48 files (0 added, 48 changed, 0 removed)",
        "Unchanged  51 files",
        "Skipped     0 files",
        "Indexed   412 symbols in 9.2s",
        "Reparsed   48 files whose language is now understood (46 csharp, 2 typescript)",
        "Fallback    1 files as line windows (0 no grammar, 1 parse errors)",
    ]
    tie = replace(s, reparsed=Counter(rust=1, java=1))
    assert "(1 java, 1 rust)" in cli.format_summary(tie)


def test_no_reparsed_line_without_reparsed_files():
    s = replace(summary(files=8, symbols=84), mode="incremental", added=2, changed=6, removed=0, unchanged=227)
    assert not any(line.startswith("Reparsed") for line in cli.format_summary(s).splitlines())
