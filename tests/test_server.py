"""health tool contract (contracts/health-tool.md), driven through an in-memory MCP client."""

import asyncio
import time

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from golestan_code_mcp import __version__, db, registry, schema
from golestan_code_mcp.config import load
from golestan_code_mcp.server import mcp


async def _session(action):
    async with create_connected_server_and_client_session(mcp) as client:
        return await action(client)


def call(tool):
    result = asyncio.run(_session(lambda c: c.call_tool(tool, {})))
    assert not result.isError
    return result.structuredContent


def call_health():
    return call("health")


def test_lists_health_and_list_repositories():
    tools = asyncio.run(_session(lambda c: c.list_tools())).tools
    assert [t.name for t in tools] == ["health", "list_repositories"]
    for tool in tools:
        assert tool.description
        assert tool.inputSchema.get("properties", {}) == {}


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
