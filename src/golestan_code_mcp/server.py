"""golestan_code_mcp stdio MCP server. Logs go to stderr; stdout carries the protocol."""

import functools
import json
import logging
import sys
import time
from typing import Literal

import oracledb
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from golestan_code_mcp import __version__, db, registry, schema, semantic
from golestan_code_mcp.config import ConfigError, load

NAME = "golestan_code_mcp"
mcp = FastMCP(NAME)
log = logging.getLogger("golestan_code_mcp.server")


def _logged(tool):
    """One structured record per call: tool, latency, status, result count. Never the arguments or results."""
    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        started, result = time.monotonic(), None
        try:
            result = tool(*args, **kwargs)
            return result
        finally:
            items = getattr(result, "results", getattr(result, "repositories", None))
            log.info(json.dumps({
                "event": "tool", "tool": tool.__name__, "ms": round((time.monotonic() - started) * 1000),
                "status": "error" if result is None else getattr(result, "status", None) or result.database,
                "results": None if items is None else len(items),
            }))
    return wrapper


class Health(BaseModel):
    server: str
    version: str
    database: Literal["reachable", "unreachable"]
    reason: str | None


@mcp.tool(description="Report the server name, its version, and whether the database is reachable.")
@_logged
def health() -> Health:
    try:
        db.connect(load()).close()
    except ConfigError as e:
        return Health(server=NAME, version=__version__, database="unreachable", reason=str(e))
    except oracledb.Error as e:
        return Health(server=NAME, version=__version__, database="unreachable", reason=db.describe(e))
    return Health(server=NAME, version=__version__, database="reachable", reason=None)


class RepoInfo(BaseModel):
    name: str
    path: str
    files: int
    symbols: int


class RepoList(BaseModel):
    database: Literal["reachable", "unreachable"]
    reason: str | None
    repositories: list[RepoInfo]


@mcp.tool(description="List the registered repositories with their paths and how many files and symbols each holds.")
@_logged
def list_repositories() -> RepoList:
    try:
        with db.connect(load()) as conn:
            schema.migrate(conn)
            repos = registry.list_repos(conn)
    except ConfigError as e:
        return RepoList(database="unreachable", reason=str(e), repositories=[])
    except oracledb.Error as e:
        return RepoList(database="unreachable", reason=db.describe(e), repositories=[])
    return RepoList(database="reachable", reason=None, repositories=[RepoInfo(**vars(r)) for r in repos])


class SearchHit(BaseModel):
    rank: int
    score: float
    similarity: float | None
    keyword_rank: int | None
    semantic_rank: int | None
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


class SearchResults(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    results: list[SearchHit]


@mcp.tool(
    description="Search indexed code by exact name or by meaning. Returns up to 10 symbols with repository, path "
    "and line range. mode: hybrid (default, keyword and semantic fused with reciprocal rank fusion; an "
    "identifier's definitions come first), text (every word must appear) or vector (meaning only, similarity "
    "0.3 or more). repos (or repo) limits the search to those repositories; all are searched by default. "
    "Each result explains its rank: fused score, keyword rank and semantic rank (null when that side missed it)."
)
@_logged
def search(question: str, repo: str | None = None, repos: list[str] | None = None,
           mode: str = "hybrid") -> SearchResults:
    names = list(dict.fromkeys((repos or []) + ([repo] if repo else []))) or None
    try:
        with db.connect(load()) as conn:
            schema.migrate(conn)
            hits = semantic.search(conn, question, names, mode)  # refuses an empty question, bad mode or repo
    except ValueError as e:  # ConfigError, RegistryError and ModelError are ValueErrors
        return SearchResults(status="unavailable", reason=str(e), results=[])
    except oracledb.Error as e:
        return SearchResults(status="unavailable", reason=db.describe(e), results=[])
    return SearchResults(status="ok", reason=None, results=[SearchHit(**vars(h)) for h in hits])


def main() -> None:
    log.addHandler(logging.StreamHandler(sys.stderr))  # stdout carries the protocol
    log.setLevel(logging.INFO)
    log.propagate = False
    mcp.run("stdio")


if __name__ == "__main__":
    main()
