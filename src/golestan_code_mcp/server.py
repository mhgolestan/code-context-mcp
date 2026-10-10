"""golestan_code_mcp stdio MCP server. Logs go to stderr; stdout carries the protocol."""

from typing import Literal

import oracledb
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from golestan_code_mcp import __version__, db, registry, schema, semantic
from golestan_code_mcp.config import ConfigError, load

NAME = "golestan_code_mcp"
mcp = FastMCP(NAME)


class Health(BaseModel):
    server: str
    version: str
    database: Literal["reachable", "unreachable"]
    reason: str | None


@mcp.tool(description="Report the server name, its version, and whether the database is reachable.")
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
    similarity: float
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
    description="Find indexed code by meaning: the 10 symbols closest to a plain-English question, "
    "each with its similarity (0.3 or more), repository, path and line range. Optionally limit to one repository."
)
def search(question: str, repo: str | None = None) -> SearchResults:
    try:
        if not question.strip():
            raise ValueError("the question is empty")
        with db.connect(load()) as conn:
            schema.migrate(conn)
            hits = semantic.search(conn, question, repo)
    except ValueError as e:  # ConfigError, RegistryError and ModelError are ValueErrors
        return SearchResults(status="unavailable", reason=str(e), results=[])
    except oracledb.Error as e:
        return SearchResults(status="unavailable", reason=db.describe(e), results=[])
    return SearchResults(status="ok", reason=None, results=[SearchHit(**vars(h)) for h in hits])


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
