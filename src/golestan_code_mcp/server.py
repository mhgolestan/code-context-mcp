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

from golestan_code_mcp import __version__, db, graph, registry, schema, semantic
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


LIMIT = 200  # graph tools return at most this many rows; truncated says when more existed


class CallerHit(BaseModel):
    repo: str
    kind: str
    name: str
    path: str
    line: int
    callee: str


class CalleeHit(BaseModel):
    caller: str
    line: int
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


class ImpactHit(BaseModel):
    hops: int
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


class CallerResults(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    results: list[CallerHit]
    truncated: bool


class CalleeResults(CallerResults):
    results: list[CalleeHit]


class ImpactResults(CallerResults):
    results: list[ImpactHit]


def _graph(answer, model, query, name: str, repo: str | None, repos: list[str] | None, *args):
    """Run one graph question with the CLI's refusals, as an envelope of at most LIMIT rows."""
    names = list(dict.fromkeys((repos or []) + ([repo] if repo else []))) or None
    try:
        graph.check_name(name)
        if query is graph.impact:
            graph.check_depth(*args)  # both refused before connecting
        with db.connect(load()) as conn:
            schema.migrate(conn)
            rows = query(conn, name, *args, names)
    except ValueError as e:  # ConfigError and RegistryError are ValueErrors
        return answer(status="unavailable", reason=str(e), results=[], truncated=False)
    except oracledb.Error as e:
        return answer(status="unavailable", reason=db.describe(e), results=[], truncated=False)
    return answer(status="ok", reason=None, results=[model(**vars(r)) for r in rows[:LIMIT]],
                  truncated=len(rows) > LIMIT)


@mcp.tool(
    description="List every call to the symbols matching name (bare like url_for, or qualified like Flask.url_for; "
    "case-sensitive), with the calling symbol, repository, path and line. Calls come from the parse tree, so "
    "comments, docstrings and strings never appear. A module-level call has kind file and name (module). "
    "repos (or repo) limits the question to those repositories. At most 200 results; truncated says when more exist."
)
@_logged
def callers(name: str, repo: str | None = None, repos: list[str] | None = None) -> CallerResults:
    return _graph(CallerResults, CallerHit, graph.callers, name, repo, repos)


@mcp.tool(
    description="List the indexed symbols called from the body of each definition matching name, with the line of "
    "each call and where the called symbol is defined. Calls into code that is not indexed are not listed. "
    "repos (or repo) limits the question to those repositories. At most 200 results; truncated says when more exist."
)
@_logged
def calls(name: str, repo: str | None = None, repos: list[str] | None = None) -> CalleeResults:
    return _graph(CalleeResults, CalleeHit, graph.calls, name, repo, repos)


@mcp.tool(
    description="Blast radius: list every symbol that reaches name through calls in at most depth hops (1 to 5, "
    "default 3), once each at its shortest distance, nearest first. repos (or repo) limits the question to those "
    "repositories. At most 200 results; truncated says when more exist."
)
@_logged
def impact(name: str, depth: int = 3, repo: str | None = None, repos: list[str] | None = None) -> ImpactResults:
    return _graph(ImpactResults, ImpactHit, graph.impact, name, repo, repos, depth)


def main() -> None:
    log.addHandler(logging.StreamHandler(sys.stderr))  # stdout carries the protocol
    log.setLevel(logging.INFO)
    log.propagate = False
    mcp.run("stdio")


if __name__ == "__main__":
    main()
