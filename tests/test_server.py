"""MCP tool contract (specs/008-mcp-tools/contracts/tools.md, extended by specs/009-module-wiki/contracts/tools.md),
driven through an in-memory MCP client."""

import asyncio
import json
import logging
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from golestan_code_mcp import (
    db,
    graph,
    ingest,
    registry,
    schema,
    semantic,
    server,
    wiki,
)
from golestan_code_mcp.config import load
from golestan_code_mcp.server import mcp

SPECS = Path(__file__).resolve().parents[1] / "specs"
CONTRACT = SPECS / "008-mcp-tools" / "contracts" / "tools.md"
WIKI_CONTRACT = SPECS / "009-module-wiki" / "contracts" / "tools.md"
MEMORY_CONTRACT = SPECS / "010-agent-memory" / "contracts" / "tools.md"
FEEDBACK_CONTRACT = SPECS / "011-usage-feedback" / "contracts" / "tools.md"
LANGUAGE_CONTRACT = SPECS / "012-language-coverage" / "contracts" / "tools.md"
TOOLS = ["search_code", "get_symbol", "find_callers", "find_callees", "impact_of", "read_wiki", "list_repositories",
         "recall", "mark_used"]
FACT_SENTENCE = ("A hit of kind fact is something an earlier session worked out about the code at its path; recall "
                 "lists all of them.")


async def _session(action):
    async with create_connected_server_and_client_session(mcp) as client:
        return await action(client)


def raw_call(tool, **arguments):
    return asyncio.run(_session(lambda c: c.call_tool(tool, arguments)))


def call(tool, **arguments):
    result = raw_call(tool, **arguments)
    assert not result.isError
    return result.structuredContent


def listed():
    return {t.name: t for t in asyncio.run(_session(lambda c: c.list_tools())).tools}


def contract():
    """The tool and parameter descriptions written in the contracts: step 8's, with steps 9 and 10 on top."""
    text = CONTRACT.read_text(encoding="utf-8")
    wiki_text = WIKI_CONTRACT.read_text(encoding="utf-8")
    tools = dict(re.findall(r"^\*\*`(\w+)`\*\*: (.+)$", text, re.MULTILINE))
    tools |= dict(re.findall(r"^\*\*`(\w+)`\*\*: (.+)$", wiki_text, re.MULTILINE))
    params = {}
    table = text.split("### Parameter descriptions", 1)[1].split("Parameter order", 1)[0]
    for line in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
        if len(cells) == 6 and cells[2].startswith("`"):
            for tool in re.findall(r"`(\w+)`", cells[1]):
                params[(tool, cells[2].strip("`"))] = cells[4]
    table = wiki_text.split("## `read_wiki`", 1)[1].split("### Parameters", 1)[1].split("Parameter order", 1)[0]
    for line in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
        if len(cells) == 5 and cells[1].startswith("`"):
            params[("read_wiki", cells[1].strip("`"))] = cells[3]
    memory_text = MEMORY_CONTRACT.read_text(encoding="utf-8")
    tools |= dict(re.findall(r"^\*\*`(recall)`\*\*: (.+)$", memory_text, re.MULTILINE))
    tools["search_code"] = tools["search_code"].replace("read it with read_wiki.", f"read it with read_wiki. {FACT_SENTENCE}")
    table = memory_text.split("## `recall`", 1)[1].split("### Parameters", 1)[1].split("Unknown arguments", 1)[0]
    for line in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
        if len(cells) == 5 and cells[1].startswith("`"):
            params[("recall", cells[1].strip("`"))] = cells[3]
    feedback_text = FEEDBACK_CONTRACT.read_text(encoding="utf-8")
    tools |= dict(re.findall(r"^\*\*`(mark_used)`\*\*: (.+)$", feedback_text, re.MULTILINE))
    scores, added = (re.search(rf'^{k}: "(.+)"$', feedback_text, re.MULTILINE).group(1) for k in ("Scores", "Added"))
    old_scores = ("the scores behind its rank: the fused score, the keyword rank and the semantic rank (null when that "
                  "side did not find it).")
    assert old_scores in tools["search_code"]
    tools["search_code"] = tools["search_code"].replace(old_scores, f"{scores} {added}")
    for section, tool in (("## `mark_used`", "mark_used"), ("## `search_code`", "search_code")):
        table = feedback_text.split(section, 1)[1].split("| Parameter | Type | Description |", 1)[1].split("\n\n", 1)[0]
        for line in table.splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) == 5 and cells[1].startswith("`"):
                params[(tool, cells[1].strip("`"))] = cells[3]
    languages = LANGUAGE_CONTRACT.read_text(encoding="utf-8").split("| Parameter | Description |", 1)[1]
    params[("search_code", "language")] = re.search(r"^\| `language` \| (.+) \|$", languages, re.M).group(1)
    return tools, params


def instructions():
    """The server instructions quoted in specs/010-agent-memory/contracts/tools.md."""
    section = MEMORY_CONTRACT.read_text(encoding="utf-8").split("## Server instructions", 1)[1].split("## ", 1)[0]
    return " ".join(line[2:].strip() for line in section.splitlines() if line.startswith("> "))


def test_lists_the_nine_tools_as_the_contract_describes_them():
    tools = listed()
    assert list(tools) == TOOLS
    descriptions, params = contract()
    for name, tool in tools.items():
        assert tool.description == descriptions[name], name
        assert tool.annotations.readOnlyHint is (name != "mark_used")
        assert tool.inputSchema.get("additionalProperties") is False
        for param, spec in tool.inputSchema.get("properties", {}).items():
            assert spec["description"] == params[(name, param)], (name, param)
    shape = {n: (list(t.inputSchema.get("properties", {})), t.inputSchema.get("required", [])) for n, t in tools.items()}
    assert shape == {
        "search_code": (["query", "repo", "language", "limit", "feedback"], ["query"]),
        "get_symbol": (["name", "repo"], ["name"]),
        "find_callers": (["name", "repo"], ["name"]),
        "find_callees": (["name", "repo"], ["name"]),
        "impact_of": (["name", "depth", "repo"], ["name"]),
        "read_wiki": (["page", "repo"], []),
        "list_repositories": ([], []),
        "recall": (["repo", "topic", "limit"], []),
        "mark_used": (["repo", "path", "name"], ["repo", "path", "name"]),
    }
    marks = tools["mark_used"].annotations
    assert (marks.idempotentHint, marks.destructiveHint) == (True, False)
    assert tools["search_code"].inputSchema["properties"]["feedback"]["default"] is True
    assert tools["recall"].inputSchema["properties"]["limit"]["default"] == 20
    assert tools["search_code"].inputSchema["properties"]["limit"]["default"] == 10
    assert tools["impact_of"].inputSchema["properties"]["depth"]["default"] == 3


def test_search_code_hits_can_be_pages():
    hit = asyncio.run(_session(lambda c: c.list_tools())).tools[0].outputSchema["$defs"]["SearchHit"]["properties"]
    for line in ("start_line", "end_line"):
        assert {t.get("type") for t in hit[line]["anyOf"]} == {"integer", "null"}
    assert {t.get("type") for t in hit["state"]["anyOf"]} == {"string", "null"}


@pytest.mark.parametrize(("tool", "arguments", "argument"), [
    ("find_callers", {"name": "url_for", "repos": ["flask"]}, "repos"),
    ("search_code", {"query": "routing", "question": "routing"}, "question"),
    ("read_wiki", {"name": "flask/json"}, "name"),
])
def test_unknown_arguments_are_refused(monkeypatch, tool, arguments, argument):
    monkeypatch.setenv("DB_PORT", "1")
    result = raw_call(tool, **arguments)
    assert result.isError and argument in result.content[0].text


def test_list_repositories_matches_registry():
    try:
        conn = db.connect(load())
    except Exception:
        pytest.skip("database unreachable")
    with conn:
        schema.migrate(conn)
        expected = [vars(r) for r in registry.list_repos(conn)]
    assert call("list_repositories") == {"database": "reachable", "reason": None, "repositories": expected}


# --- search_code ------------------------------------------------------------------------------------------------

def ready():
    try:
        conn = db.connect(load())
        semantic.require_model(conn)
    except Exception:
        pytest.skip("database unreachable or hybrid index not built")
    return conn


@pytest.mark.parametrize("arguments", [
    {"query": "how are requests routed", "repo": "flask"},
    {"query": "url_for"},
    {"query": "middleware chain", "repo": "gin", "language": "go", "limit": 3},
    {"query": "url_for", "language": "python", "limit": 50},
])
def test_search_code_matches_core(arguments):
    conn = ready()
    repo = arguments.get("repo")
    try:
        with conn:
            expected = [vars(r) for r in semantic.search(conn, arguments["query"], [repo] if repo else None,
                                                         language=arguments.get("language"),
                                                         limit=arguments.get("limit", 10))]
    except Exception as e:  # repositories not registered in this database
        pytest.skip(f"sample repositories not indexed: {e}")
    result = call("search_code", **arguments)
    assert result == {"status": "ok", "reason": None, "results": expected}
    assert all(h["repo"] and h["path"] for h in result["results"])
    assert all(h["start_line"] <= h["end_line"] for h in result["results"] if h["kind"] != "page")
    assert all(h["start_line"] is None and h["state"] for h in result["results"] if h["kind"] == "page")


@pytest.mark.parametrize(("arguments", "reason"), [
    ({"query": "  "}, "the question is empty"),
    ({"query": "routing", "limit": 0}, "limit must be between 1 and 50"),
    ({"query": "routing", "limit": 51}, "limit must be between 1 and 50"),
    ({"query": "routing", "language": "cobol"},
     f"unknown language 'cobol'; valid languages: {', '.join(semantic.LANGUAGES)}"),
])
def test_search_code_input_refused_before_connecting(monkeypatch, arguments, reason):
    monkeypatch.setenv("DB_PORT", "1")
    assert call("search_code", **arguments) == {"status": "unavailable", "reason": reason, "results": []}


def test_search_code_unknown_repository_is_a_result():
    try:
        db.connect(load()).close()
    except Exception:
        pytest.skip("database unreachable")
    assert call("search_code", query="routing", repo="t-missing") == {
        "status": "unavailable", "reason": "no repository named 't-missing'", "results": []}


def test_missing_password_named(monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "")
    result = call("search_code", query="routing")
    assert result["status"] == "unavailable" and "DB_PASSWORD" in result["reason"]


# --- get_symbol -------------------------------------------------------------------------------------------------

def definitions(name, repo=None):
    try:
        with db.connect(load()) as conn:
            found, total = graph.definitions(conn, name, [repo] if repo else None)
    except Exception as e:  # database down or flask not indexed here
        pytest.skip(f"flask not indexed: {e}")
    return [vars(d) for d in found], total


@pytest.mark.parametrize("arguments", [
    {"name": "Flask.dispatch_request"},
    {"name": "url_for", "repo": "flask"},
    {"name": "get"},  # many definitions: capped at 5
])
def test_get_symbol_matches_core(arguments):
    expected, total = definitions(arguments["name"], arguments.get("repo"))
    result = call("get_symbol", **arguments)
    assert result == {"status": "ok", "reason": None, "results": expected, "total": total,
                      "truncated": total > len(expected)}
    assert len(result["results"]) <= 5 and all(h["source"] for h in result["results"])


def test_get_symbol_refusals_are_results(monkeypatch):
    empty = {"results": [], "total": 0, "truncated": False}
    try:
        db.connect(load()).close()
    except Exception:
        pytest.skip("database unreachable")
    assert call("get_symbol", name="t_no_such_symbol_anywhere") == {
        "status": "unavailable", "reason": "no symbol named 't_no_such_symbol_anywhere'", **empty}
    assert call("get_symbol", name="url_for", repo="t-missing") == {
        "status": "unavailable", "reason": "no repository named 't-missing'", **empty}
    monkeypatch.setenv("DB_PORT", "1")
    assert call("get_symbol", name="  ") == {"status": "unavailable", "reason": "the name is empty", **empty}


# --- find_callers, find_callees, impact_of ----------------------------------------------------------------------

CORE = {"find_callers": graph.callers, "find_callees": graph.calls, "impact_of": graph.impact}
GRAPH_CALLS = [
    ("find_callers", {"name": "url_for", "repo": "flask"}),
    ("find_callers", {"name": "Flask.url_for"}),
    ("find_callees", {"name": "url_for", "repo": "flask"}),
    ("impact_of", {"name": "url_for", "repo": "flask"}),
    ("impact_of", {"name": "Flask.url_for", "depth": 1, "repo": "flask"}),
]


def core(tool, arguments):
    repo = arguments.get("repo")
    extra = [arguments.get("depth", 3)] if tool == "impact_of" else []
    try:
        with db.connect(load()) as conn:
            return [vars(r) for r in CORE[tool](conn, arguments["name"], *extra, [repo] if repo else None)]
    except Exception as e:  # database down or flask not indexed here
        pytest.skip(f"flask not indexed: {e}")


@pytest.mark.parametrize(("tool", "arguments"), GRAPH_CALLS)
def test_graph_tools_match_core(tool, arguments):
    expected = core(tool, arguments)
    result = call(tool, **arguments)
    assert result == {"status": "ok", "reason": None, "results": expected[:200], "truncated": len(expected) > 200}
    assert all(h["repo"] and h["path"] for h in result["results"])


def test_graph_results_are_capped(monkeypatch):
    expected = core("find_callers", {"name": "url_for", "repo": "flask"})
    if len(expected) < 2:
        pytest.skip("flask graph not built")
    monkeypatch.setattr(server, "LIMIT", 1)
    result = call("find_callers", name="url_for", repo="flask")
    assert result["results"] == expected[:1] and result["truncated"] is True


@pytest.mark.parametrize(("tool", "arguments", "reason"), [
    ("find_callers", {"name": "  "}, "the name is empty"),
    ("find_callees", {"name": ""}, "the name is empty"),
    ("impact_of", {"name": "url_for", "depth": 0}, "depth must be between 1 and 5"),
    ("impact_of", {"name": "url_for", "depth": 6}, "depth must be between 1 and 5"),
])
def test_graph_input_refused_before_connecting(monkeypatch, tool, arguments, reason):
    monkeypatch.setenv("DB_PORT", "1")
    assert call(tool, **arguments) == {"status": "unavailable", "reason": reason, "results": [], "truncated": False}


@pytest.mark.parametrize(("tool", "arguments", "reason"), [
    ("find_callees", {"name": "url_for", "repo": "t-missing"}, "no repository named 't-missing'"),
    ("find_callers", {"name": "t_no_such_symbol_anywhere"}, "no symbol named 't_no_such_symbol_anywhere'"),
])
def test_graph_refusals_are_results(tool, arguments, reason):
    try:
        db.connect(load()).close()
    except Exception:
        pytest.skip("database unreachable")
    assert call(tool, **arguments) == {"status": "unavailable", "reason": reason, "results": [], "truncated": False}


# --- outages and logging ----------------------------------------------------------------------------------------

UNREACHABLE = [
    ("search_code", {"query": "routing"}),
    ("get_symbol", {"name": "url_for"}),
    ("find_callers", {"name": "url_for"}),
    ("find_callees", {"name": "url_for"}),
    ("impact_of", {"name": "url_for"}),
    ("read_wiki", {"page": "flask/json"}),
]


@pytest.mark.parametrize(("tool", "arguments"), UNREACHABLE)
def test_unreachable_is_a_result(monkeypatch, tool, arguments):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    started = time.monotonic()
    result = call(tool, **arguments)
    assert time.monotonic() - started < 10
    assert result["status"] == "unavailable" and result["reason"] and result.get("results", result.get("pages")) == []
    assert "Not-A-Real-Pw-42" not in result["reason"]


def test_list_repositories_unreachable_is_a_result(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    result = call("list_repositories")
    assert result["database"] == "unreachable" and result["reason"] and result["repositories"] == []
    assert "Not-A-Real-Pw-42" not in result["reason"]


def test_next_call_after_an_outage_succeeds(monkeypatch):
    try:
        db.connect(load()).close()
    except Exception:
        pytest.skip("database unreachable")
    monkeypatch.setenv("DB_PORT", "1")
    assert call("list_repositories")["database"] == "unreachable"
    monkeypatch.undo()  # the database is back; same server, no restart
    assert call("list_repositories")["database"] == "reachable"


def test_each_tool_call_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("DB_PORT", "1")
    caplog.set_level(logging.INFO, logger="golestan_code_mcp.server")
    call("list_repositories")
    call("search_code", query="secret_marker_question")
    call("get_symbol", name="secret_marker_name")
    call("find_callers", name="secret_marker_name")
    call("impact_of", name="secret_marker_name")
    call("read_wiki", page="secret_marker_name")
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "golestan_code_mcp.server"]
    assert [(r["event"], r["tool"], r["status"], r["results"]) for r in records] == [
        ("tool", "list_repositories", "unreachable", 0),
        ("tool", "search_code", "unavailable", 0),
        ("tool", "get_symbol", "unavailable", 0),
        ("tool", "find_callers", "unavailable", 0),
        ("tool", "impact_of", "unavailable", 0),
        ("tool", "read_wiki", "unavailable", 0),
    ]
    assert all(isinstance(r["ms"], int) for r in records)
    assert "secret_marker_question" not in caplog.text and "secret_marker_name" not in caplog.text  # never arguments


# --- read_wiki --------------------------------------------------------------------------------------------------

PKG = '"""Ledger accounting for the demo."""\n' + "".join(
    f'\n\ndef post{i}():\n    """Post entry {i}."""\n    return {i}\n' for i in range(10))


def wiki_repo(conn, root):
    """Register, ingest and build the wiki of a repository with one page, `pkg`. Returns its name."""
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text('"""The demo ledger package."""\n')
    (root / "src" / "pkg" / "core.py").write_text(PKG)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                       capture_output=True)
    name = f"t-{uuid.uuid4().hex[:8]}"
    registry.add_repo(conn, str(root), name)
    ingest.ingest(conn, name)
    wiki.build(conn, name)
    return name


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    conn = ready()
    with conn:
        schema.migrate(conn)
        names = [wiki_repo(conn, tmp_path_factory.mktemp("w")), wiki_repo(conn, tmp_path_factory.mktemp("w"))]
        yield names
        with conn.cursor() as cur:
            cur.executemany("delete from ccctx_repos where name = :n", [{"n": n} for n in names])
        conn.commit()


def test_read_wiki_matches_core(pages):
    with db.connect(load()) as conn:
        page = wiki.read(conn, "pkg", pages[0])
    result = call("read_wiki", page="pkg", repo=pages[0])
    got = result["page"]
    assert result["status"] == "ok" and result["pages"] == [] and result["truncated"] is False
    assert (got["name"], got["repo"], got["directory"], got["language"], got["state"], got["compiler"],
            got["stale_reason"], got["symbols"], got["text"], got["truncated"]) == (
        page.name, page.repo, page.directory, page.language, page.state, page.compiler, page.stale_reason,
        page.symbols, page.text, page.truncated)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", got["compiled_at"])
    assert [(c["role"], c["path"], c["start_line"], c["end_line"], c["commit"], c["current_commit"])
            for c in got["citations"]] == [
        (c.role, c.path, c.start_line, c.end_line, c.recorded_commit, c.current_commit) for c in page.citations]
    files = [c for c in got["citations"] if c["kind"] == "file"]
    assert files and all(c["name"] == c["path"].rsplit("/", 1)[-1] and c["start_line"] is None for c in files)


def test_read_wiki_lists_the_pages(pages):
    with db.connect(load()) as conn:
        expected = [{"name": p.name, "repo": p.repo, "state": p.state, "summary": p.summary}
                    for p in wiki.pages(conn, pages[0])]
    assert call("read_wiki", repo=pages[0]) == {
        "status": "ok", "reason": None, "page": None, "pages": expected, "truncated": False}


def test_read_wiki_list_is_capped(pages, monkeypatch):
    with db.connect(load()) as conn:
        one = wiki.pages(conn, pages[0])[0]
    monkeypatch.setattr(wiki, "pages", lambda conn, repo=None: [one] * 201)
    result = call("read_wiki")
    assert len(result["pages"]) == 200 and result["truncated"] is True


@pytest.mark.parametrize(("arguments", "reason"), [
    ({"page": "pkg", "repo": "t-missing"}, "no repository named 't-missing'"),
    ({"page": "t_no_such_page"}, "no wiki page named 't_no_such_page'"),
])
def test_read_wiki_refusals_are_results(pages, arguments, reason):
    assert call("read_wiki", **arguments) == {
        "status": "unavailable", "reason": reason, "page": None, "pages": [], "truncated": False}


def test_read_wiki_ambiguous_name_asks_for_repo(pages):
    result = call("read_wiki", page="pkg")
    assert result["status"] == "unavailable" and result["page"] is None
    assert re.fullmatch(r"wiki page 'pkg' is in more than one repository: .+; pass repo", result["reason"])
    assert all(n in result["reason"] for n in pages)


def test_read_wiki_empty_page_refused_before_connecting(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    assert call("read_wiki", page="  ") == {
        "status": "unavailable", "reason": "the page name is empty", "page": None, "pages": [], "truncated": False}


# --- recording (specs/010-agent-memory contracts/tools.md "Recording") -----------------------------------------

from mcp.types import Implementation  # noqa: E402

from golestan_code_mcp import memory  # noqa: E402

CLIENT = Implementation(name="test-client", version="9.9")


def as_client(tool, **arguments):
    async def go():
        async with create_connected_server_and_client_session(mcp, client_info=CLIENT) as client:
            return await client.call_tool(tool, arguments)
    return asyncio.run(go())


@pytest.fixture
def recorded(real_recording, monkeypatch):
    """(conn, newest run id before the test). Runs are read back after memory.flush(). Records for real.

    Starts with no results returned: other tests' flask searches must not turn this test's reads into uses."""
    monkeypatch.setattr(server, "_returned", {})
    try:
        conn = db.connect(load())
    except Exception:
        pytest.skip("database unreachable")
    schema.migrate(conn)
    with conn.cursor() as cur:
        before = cur.execute("select nvl(max(id), 0) from ccctx_memory_runs").fetchone()[0]
    yield conn, before
    memory.flush()  # the runs these tests make are artifacts: never let them reach a repository's facts
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_memory_runs where id > :b", b=before)
    conn.commit()
    conn.close()


def runs_since(conn, before):
    memory.flush()
    with conn.cursor() as cur:
        cur.execute("select tool, arguments, question, client, status, reason, result_count from ccctx_memory_runs"
                    " where id > :b order by id", b=before)
        return cur.fetchall()


MARK = f"zz_{uuid.uuid4().hex[:8]}"
CALLS = [
    ("search_code", {"query": f"where is {MARK}"}),
    ("get_symbol", {"name": MARK}),
    ("find_callers", {"name": MARK}),
    ("find_callees", {"name": MARK}),
    ("impact_of", {"name": MARK, "depth": 2}),
    ("read_wiki", {}),
    ("list_repositories", {}),
    ("recall", {}),
]


@pytest.mark.parametrize(("tool", "arguments"), CALLS)
def test_every_tool_call_is_one_run(recorded, tool, arguments):
    conn, before = recorded
    result = as_client(tool, **arguments)
    assert not result.isError
    rows = runs_since(conn, before)
    assert len(rows) == 1
    name, args, question, client, status, reason, count = rows[0]
    assert (name, client) == (tool, "test-client 9.9")
    assert arguments.items() <= args.items() and None not in args.values()  # the effective call, defaults included
    assert question == next((arguments[k] for k in ("query", "name", "page", "topic") if k in arguments), None)
    envelope = result.structuredContent
    assert status == envelope.get("status", "ok") and reason == envelope.get("reason")
    assert count == len(next((v for k, v in envelope.items() if k in ("results", "pages", "repositories", "facts")), []))


@pytest.mark.parametrize(("tool", "arguments", "reason"), [
    ("search_code", {"query": ""}, "the question is empty"),
    ("get_symbol", {"name": "url_for", "repo": "t-nowhere"}, "no repository named 't-nowhere'"),
])
def test_refused_calls_are_recorded_unavailable(recorded, tool, arguments, reason):
    conn, before = recorded
    as_client(tool, **arguments)
    [(name, _, _, _, status, why, count)] = runs_since(conn, before)
    assert (name, status, count) == (tool, "unavailable", 0) and reason in why


def test_recording_failure_leaves_the_answer_alone(recorded, monkeypatch, caplog):
    conn, before = recorded
    expected = as_client("list_repositories").structuredContent
    memory.flush()

    def broken(*_):
        raise RuntimeError("store down")

    monkeypatch.setattr(memory, "store_runs", broken)
    with caplog.at_level(logging.WARNING, logger="golestan_code_mcp.memory"):
        assert as_client("list_repositories").structuredContent == expected
        memory.flush()
    failures = [json.loads(r.getMessage()) for r in caplog.records if "memory_record_failed" in r.getMessage()]
    assert failures == [{"event": "memory_record_failed", "tool": "list_repositories", "reason": "store down"}]


def test_tool_log_record_is_unchanged(recorded, caplog):
    with caplog.at_level(logging.INFO, logger="golestan_code_mcp.server"):
        as_client("list_repositories")
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "golestan_code_mcp.server"]
    assert records and set(records[-1]) == {"event", "tool", "ms", "status", "results"}


# --- recall (specs/010-agent-memory contracts/tools.md) --------------------------------------------------------


def test_server_instructions_match_the_contract():
    assert mcp.instructions == instructions()


@pytest.mark.parametrize(("arguments", "reason"), [
    ({"limit": 0}, "limit must be between 1 and 50"),
    ({"limit": 51}, "limit must be between 1 and 50"),
    ({"topic": "  "}, "the topic is empty"),
])
def test_recall_refuses_bad_input_before_connecting(monkeypatch, arguments, reason):
    monkeypatch.setenv("DB_PORT", "1")
    out = call("recall", **arguments)
    assert out == {"status": "unavailable", "reason": reason, "facts": [], "runs": [], "total": 0, "truncated": False}


def test_recall_database_down(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    out = call("recall", repo="flask")
    assert out["status"] == "unavailable" and out["reason"] and out["facts"] == [] and out["runs"] == []


def test_recall_unknown_repository():
    ready().close()
    out = call("recall", repo="t-nowhere")
    assert out["status"] == "unavailable" and out["reason"] == "no repository named 't-nowhere'"


def test_recall_matches_core_and_shape(monkeypatch):
    conn = ready()
    calls = []

    class Spy:
        def generate(self, *a, **k):
            calls.append(a)

    monkeypatch.setattr(memory, "model", lambda env=None: (Spy(), "llm:spy"))
    with conn:
        expected = memory.recall(conn, "flask", None, 20)
    out = call("recall", repo="flask")
    assert out["status"] == "ok" and out["total"] == expected.total
    assert out["truncated"] == (expected.total > len(out["facts"]))
    assert [f["id"] for f in out["facts"]] == [f.id for f in expected.facts]
    assert [r["id"] for r in out["runs"]] == [r.id for r in expected.runs]
    assert len(out["runs"]) <= 10 and all(r["tool"] != "recall" for r in out["runs"])
    for f in out["facts"]:
        assert set(f) == {"id", "repo", "subject", "kind", "state", "stale_reason", "confidence", "distiller", "text",
                          "questions", "runs", "updated_at", "sources"}
        assert len(f["questions"]) <= 5 and len(f["runs"]) <= 20 and f["sources"][0]["role"] == "subject"
    for r in out["runs"]:
        assert set(r) == {"id", "at", "tool", "question", "client", "status", "result_count", "top"}
    assert calls == []  # recall never calls the model


def test_recording_adds_no_visible_delay(recorded, monkeypatch):
    """SC-008: the median of 20 calls with recording is within 5% (plus 10 ms of timer noise) of the median without."""
    import statistics

    def median_ms():
        times = []
        for _ in range(20):
            started = time.perf_counter()
            server.get_symbol(name="url_for", repo="flask")
            times.append(time.perf_counter() - started)
        return statistics.median(times)

    median_ms()  # warm up the connection path
    with_recording = median_ms()
    memory.flush()
    monkeypatch.setattr(memory, "record", lambda run: None)
    without = median_ms()
    assert with_recording <= without * 1.05 + 0.010


# --- usage feedback (specs/011-usage-feedback contracts/tools.md) -----------------------------------------------

from golestan_code_mcp import feedback  # noqa: E402


@pytest.fixture(scope="module")
def used_repo(tmp_path_factory):
    """A `t-` repository with one wiki page, `pkg`, and its symbols post0 to post9 in src/pkg/core.py."""
    conn = ready()
    with conn:
        schema.migrate(conn)
        name = wiki_repo(conn, tmp_path_factory.mktemp("u"))
        yield name
        with conn.cursor() as cur:
            cur.execute("delete from ccctx_repos where name = :n", n=name)
        conn.commit()


@pytest.fixture
def fresh_session(recorded, used_repo, monkeypatch):
    """A new server session over the `t-` repository: no results returned yet, its own session id, no uses."""
    conn, _ = recorded
    monkeypatch.setattr(server, "_returned", {})
    monkeypatch.setattr(server, "SESSION", uuid.uuid4().hex)
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_feedback where repo_id = :r", r=registry.repo_id(conn, used_repo))
    conn.commit()
    return used_repo


def uses(conn, repo):
    memory.flush()
    with conn.cursor() as cur:
        cur.execute("""select f.name, f.path, f.kind, f.how, f.run_id, f.client from ccctx_feedback f
                        where f.repo_id = :r order by f.id""", r=registry.repo_id(conn, repo))
        return cur.fetchall()


def newest_run(conn, tool):
    memory.flush()
    with conn.cursor() as cur:
        return cur.execute("select max(id) from ccctx_memory_runs where tool = :t", t=tool).fetchone()[0]


def search_hit(repo, name):
    out = as_client("search_code", query="post entry for the demo ledger", repo=repo, limit=20).structuredContent
    return next(h for h in out["results"] if h["name"] == name)


def test_mark_used_after_a_search_is_linked(recorded, fresh_session):
    conn, _ = recorded
    hit = search_hit(fresh_session, "post3")
    out = as_client("mark_used", repo=fresh_session, path=hit["path"], name="post3").structuredContent
    assert out == {"status": "ok", "reason": None, "repo": fresh_session, "path": hit["path"], "name": "post3",
                   "kind": "function", "linked": True}
    assert uses(conn, fresh_session) == [("post3", hit["path"], "function", "cited", newest_run(conn, "search_code"),
                                          "test-client 9.9")]


def test_mark_used_without_a_search_is_unlinked(recorded, fresh_session):
    conn, _ = recorded
    out = as_client("mark_used", repo=fresh_session, path="src/pkg/core.py", name="post1").structuredContent
    assert out["status"] == "ok" and out["linked"] is False
    assert [(n, how, run) for n, _, _, how, run, _ in uses(conn, fresh_session)] == [("post1", "cited", None)]


@pytest.mark.parametrize("arguments", [
    {"path": "src/pkg/core.py", "name": "no_such_post"},
    {"path": "src/pkg/other.py", "name": "post1"},
])
def test_mark_used_on_nothing_records_nothing(recorded, fresh_session, arguments):
    conn, _ = recorded
    out = as_client("mark_used", repo=fresh_session, **arguments).structuredContent
    assert out["status"] == "unavailable" and out["reason"].startswith("not found: ") and out["kind"] is None
    assert out["linked"] is False and uses(conn, fresh_session) == []


def test_mark_used_unknown_repository(recorded):
    out = as_client("mark_used", repo="t-nowhere", path="a.py", name="f").structuredContent
    assert out["status"] == "unavailable" and out["reason"] == "no repository named 't-nowhere'"


def test_reading_a_returned_hit_is_an_opened_use(recorded, fresh_session):
    conn, _ = recorded
    out = as_client("search_code", query="post entry for the demo ledger", repo=fresh_session, limit=20)
    returned = {h["name"] for h in out.structuredContent["results"]}
    run = newest_run(conn, "search_code")
    assert as_client("get_symbol", name="post5", repo=fresh_session).structuredContent["status"] == "ok"
    assert as_client("read_wiki", page="pkg", repo=fresh_session).structuredContent["status"] == "ok"
    expected = [("post5", "src/pkg/core.py", "function", "opened", run, "test-client 9.9")]
    if "pkg" in returned:  # the page is an opened use only when the search returned it
        expected.append(("pkg", "src/pkg", "page", "opened", run, "test-client 9.9"))
    assert uses(conn, fresh_session) == expected


def test_reading_without_a_search_is_not_a_use(recorded, fresh_session):
    conn, _ = recorded
    as_client("get_symbol", name="post5", repo=fresh_session)
    as_client("read_wiki", page="pkg", repo=fresh_session)
    assert uses(conn, fresh_session) == []


def test_opened_then_cited_is_one_cited_use(recorded, fresh_session):
    conn, _ = recorded
    hit = search_hit(fresh_session, "post7")
    as_client("get_symbol", name="post7", repo=fresh_session)
    as_client("mark_used", repo=fresh_session, path=hit["path"], name="post7")
    as_client("mark_used", repo=fresh_session, path=hit["path"], name="post7")
    assert [(n, how) for n, _, _, how, _, _ in uses(conn, fresh_session)] == [("post7", "cited")]


def test_a_use_recording_failure_leaves_the_answer_alone(recorded, fresh_session, monkeypatch, caplog):
    conn, _ = recorded
    expected = as_client("mark_used", repo=fresh_session, path="src/pkg/core.py", name="post2").structuredContent
    memory.flush()

    def broken(*_):
        raise RuntimeError("feedback down")

    monkeypatch.setattr(feedback, "write_use", broken)
    with caplog.at_level(logging.WARNING, logger="golestan_code_mcp.memory"):
        out = as_client("mark_used", repo=fresh_session, path="src/pkg/core.py", name="post4").structuredContent
        memory.flush()
    assert out == expected | {"name": "post4"}
    failures = [json.loads(r.getMessage()) for r in caplog.records if "feedback_record_failed" in r.getMessage()]
    assert failures == [{"event": "feedback_record_failed", "reason": "feedback down"}]
    assert [n for n, *_ in uses(conn, fresh_session)] == ["post2"]


def test_search_hits_carry_relevance_and_feedback(fresh_session):
    on = call("search_code", query="post entry for the demo ledger", repo=fresh_session)["results"]
    off = call("search_code", query="post entry for the demo ledger", repo=fresh_session, feedback=False)["results"]
    assert on and all(0 <= h["feedback"] <= 0.2 for h in on)
    assert all(h["score"] == pytest.approx(h["relevance"] * (1 + h["feedback"])) for h in on)
    assert all(h["feedback"] == 0 for h in off)


def test_search_with_feedback_off_matches_the_core(fresh_session):
    with db.connect(load()) as conn:
        expected = [vars(r) for r in semantic.search(conn, "post entry", [fresh_session], feedback=False)]
    assert call("search_code", query="post entry", repo=fresh_session, feedback=False)["results"] == expected


# --- language coverage (specs/012-language-coverage contracts/tools.md) ------------------------------------------


def test_an_unknown_language_lists_the_new_ones(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    reason = call("search_code", query="x", language="cobol")["reason"]
    assert reason == f"unknown language 'cobol'; valid languages: {', '.join(semantic.LANGUAGES)}"
    assert {"csharp", "typescript", "tsx", "c", "cpp", "java", "kotlin", "rust", "php", "ruby", "swift"} <= set(
        reason.split(": ", 1)[1].split(", "))


def test_search_code_finds_csharp_definitions_as_the_core_does(tmp_path):
    import shutil

    conn = ready()
    with conn:
        root = tmp_path / "repo"
        shutil.copytree(Path(__file__).parent / "fixtures" / "languages" / "csharp", root)
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
            subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                           capture_output=True)
        name = f"t-{uuid.uuid4().hex[:8]}"
        registry.add_repo(conn, str(root), name)
        try:
            ingest.ingest(conn, name)
            expected = [vars(r) for r in semantic.search(conn, "start the browser", [name], language="csharp")]
            result = call("search_code", query="start the browser", repo=name, language="csharp")
            assert result == {"status": "ok", "reason": None, "results": expected}
            assert expected and {h["language"] for h in expected} == {"csharp"}
            assert "DriverFixture.Start" in {h["name"] for h in expected}
        finally:
            with conn.cursor() as cur:
                cur.execute("delete from ccctx_repos where name = :n", n=name)
            conn.commit()
