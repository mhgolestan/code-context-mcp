"""health tool contract (contracts/health-tool.md), driven through an in-memory MCP client."""

import asyncio
import json
import logging
import time

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from golestan_code_mcp import __version__, db, registry, schema, semantic
from golestan_code_mcp.config import load
from golestan_code_mcp.server import mcp


async def _session(action):
    async with create_connected_server_and_client_session(mcp) as client:
        return await action(client)


def call(tool, **arguments):
    result = asyncio.run(_session(lambda c: c.call_tool(tool, arguments)))
    assert not result.isError
    return result.structuredContent


def call_health():
    return call("health")


def test_lists_three_tools():
    tools = {t.name: t for t in asyncio.run(_session(lambda c: c.list_tools())).tools}
    assert list(tools) == ["health", "list_repositories", "search"]
    assert all(t.description for t in tools.values())
    assert tools["health"].inputSchema.get("properties", {}) == {}
    assert tools["list_repositories"].inputSchema.get("properties", {}) == {}
    schema_ = tools["search"].inputSchema
    assert set(schema_["properties"]) == {"question", "repo", "repos", "mode"} and schema_["required"] == ["question"]


def test_reachable_when_database_up():
    try:
        if not db.check(load(), wait=0).ok:
            raise RuntimeError
    except Exception:
        pytest.skip("database unreachable")
    assert call_health() == {
        "server": "golestan_code_mcp",
        "version": __version__,
        "database": "reachable",
        "reason": None,
    }


def test_unreachable_is_a_result_not_an_error(monkeypatch):
    monkeypatch.setenv("DB_USER", "nobody")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    monkeypatch.setenv("DB_PORT", "1")
    started = time.monotonic()
    health = call_health()
    assert time.monotonic() - started < 5
    assert health["database"] == "unreachable" and health["reason"]
    assert "Not-A-Real-Pw-42" not in health["reason"]


def test_missing_password_named(monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "")
    health = call_health()
    assert health["database"] == "unreachable"
    assert "DB_PASSWORD" in health["reason"]


def test_list_repositories_matches_registry():
    try:
        conn = db.connect(load())
    except Exception:
        pytest.skip("database unreachable")
    with conn:
        schema.migrate(conn)
        expected = [vars(r) for r in registry.list_repos(conn)]
    assert call("list_repositories") == {"database": "reachable", "reason": None, "repositories": expected}


def test_list_repositories_unreachable_is_a_result(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    started = time.monotonic()
    result = call("list_repositories")
    assert time.monotonic() - started < 5
    assert result["database"] == "unreachable" and result["reason"]
    assert result["repositories"] == []
    assert "Not-A-Real-Pw-42" not in result["reason"]


def ready():
    try:
        conn = db.connect(load())
        semantic.require_model(conn)
    except Exception:
        pytest.skip("database unreachable or hybrid index not built")
    return conn


@pytest.mark.parametrize("arguments", [
    {"question": "how are incoming requests routed to handlers"},
    {"question": "url_for", "mode": "hybrid"},
    {"question": "url_for", "mode": "text", "repos": ["flask"]},
    {"question": "middleware chain", "repo": "gin"},  # the single repo input still works
    {"question": "middleware chain", "repo": "gin", "repos": ["gin"]},
])
def test_search_matches_core(arguments):
    conn = ready()
    names = list(dict.fromkeys((arguments.get("repos") or []) + ([arguments["repo"]] if "repo" in arguments else [])))
    try:
        with conn:
            expected = [vars(r) for r in semantic.search(conn, arguments["question"], names or None,
                                                         arguments.get("mode", "hybrid"))]
    except Exception as e:  # repositories not registered in this database
        pytest.skip(f"sample repositories not indexed: {e}")
    result = call("search", **arguments)
    assert result == {"status": "ok", "reason": None, "results": expected}  # SC-008


@pytest.mark.parametrize("arguments", [
    {"question": "  "},
    {"question": "routing", "repo": "t-missing"},
    {"question": "routing", "repos": ["t-missing"]},
    {"question": "routing", "mode": "bogus"},
])
def test_search_refusals_are_results(arguments):
    try:
        db.connect(load()).close()
    except Exception:
        pytest.skip("database unreachable")
    result = call("search", **arguments)
    assert result["status"] == "unavailable" and result["reason"] and result["results"] == []


def test_each_tool_call_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("DB_PORT", "1")
    caplog.set_level(logging.INFO, logger="golestan_code_mcp.server")
    call("health")
    call("list_repositories")
    call("search", question="secret_marker_question")
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "golestan_code_mcp.server"]
    assert [(r["event"], r["tool"], r["status"], r["results"]) for r in records] == [
        ("tool", "health", "unreachable", None),
        ("tool", "list_repositories", "unreachable", 0),
        ("tool", "search", "unavailable", 0),
    ]
    assert all(isinstance(r["ms"], int) for r in records)
    assert "secret_marker_question" not in caplog.text  # never the question or source text


def test_search_unreachable_is_a_result(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", "Not-A-Real-Pw-42")
    started = time.monotonic()
    result = call("search", question="routing")
    assert time.monotonic() - started < 5
    assert result["status"] == "unavailable" and result["results"] == []
    assert "Not-A-Real-Pw-42" not in result["reason"]
