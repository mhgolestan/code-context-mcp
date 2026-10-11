"""MCP tool contract (specs/008-mcp-tools/contracts/tools.md), driven through an in-memory MCP client."""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from golestan_code_mcp import db, graph, registry, schema, semantic, server
from golestan_code_mcp.config import load
from golestan_code_mcp.server import mcp

CONTRACT = Path(__file__).resolve().parents[1] / "specs" / "008-mcp-tools" / "contracts" / "tools.md"
TOOLS = ["search_code", "get_symbol", "find_callers", "find_callees", "impact_of", "list_repositories"]


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
    """The tool and parameter descriptions written in the contract."""
    text = CONTRACT.read_text(encoding="utf-8")
    tools = dict(re.findall(r"^\*\*`(\w+)`\*\*: (.+)$", text, re.MULTILINE))
    params = {}
    table = text.split("### Parameter descriptions", 1)[1].split("Parameter order", 1)[0]
    for line in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
        if len(cells) == 6 and cells[2].startswith("`"):
            for tool in re.findall(r"`(\w+)`", cells[1]):
                params[(tool, cells[2].strip("`"))] = cells[4]
    return tools, params


def test_lists_the_six_tools_as_the_contract_describes_them():
    tools = listed()
    assert list(tools) == TOOLS
    descriptions, params = contract()
    for name, tool in tools.items():
        assert tool.description == descriptions[name], name
        assert tool.annotations.readOnlyHint is True
        assert tool.inputSchema.get("additionalProperties") is False
        for param, spec in tool.inputSchema.get("properties", {}).items():
            assert spec["description"] == params[(name, param)], (name, param)
    shape = {n: (list(t.inputSchema.get("properties", {})), t.inputSchema.get("required", [])) for n, t in tools.items()}
    assert shape == {
        "search_code": (["query", "repo", "language", "limit"], ["query"]),
        "get_symbol": (["name", "repo"], ["name"]),
        "find_callers": (["name", "repo"], ["name"]),
        "find_callees": (["name", "repo"], ["name"]),
        "impact_of": (["name", "depth", "repo"], ["name"]),
        "list_repositories": ([], []),
    }
    assert tools["search_code"].inputSchema["properties"]["limit"]["default"] == 10
    assert tools["impact_of"].inputSchema["properties"]["depth"]["default"] == 3


@pytest.mark.parametrize(("tool", "arguments", "argument"), [
    ("find_callers", {"name": "url_for", "repos": ["flask"]}, "repos"),
    ("search_code", {"query": "routing", "question": "routing"}, "question"),
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
    assert all(h["repo"] and h["path"] and h["start_line"] <= h["end_line"] for h in result["results"])


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
]


@pytest.mark.parametrize(("tool", "arguments"), UNREACHABLE)
def test_unreachable_is_a_result(monkeypatch, tool, arguments):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    started = time.monotonic()
    result = call(tool, **arguments)
    assert time.monotonic() - started < 10
    assert result["status"] == "unavailable" and result["reason"] and result["results"] == []
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
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "golestan_code_mcp.server"]
    assert [(r["event"], r["tool"], r["status"], r["results"]) for r in records] == [
        ("tool", "list_repositories", "unreachable", 0),
        ("tool", "search_code", "unavailable", 0),
        ("tool", "get_symbol", "unavailable", 0),
        ("tool", "find_callers", "unavailable", 0),
        ("tool", "impact_of", "unavailable", 0),
    ]
    assert all(isinstance(r["ms"], int) for r in records)
    assert "secret_marker_question" not in caplog.text and "secret_marker_name" not in caplog.text  # never arguments
