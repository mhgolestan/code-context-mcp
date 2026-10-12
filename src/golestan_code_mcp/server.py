"""golestan_code_mcp stdio MCP server. Logs go to stderr; stdout carries the protocol."""

import functools
import json
import logging
import posixpath
import sys
import time
from datetime import UTC
from typing import Annotated, Literal

import oracledb
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from golestan_code_mcp import db, graph, registry, schema, semantic, wiki
from golestan_code_mcp.config import ConfigError, load

NAME = "golestan_code_mcp"
mcp = FastMCP(NAME)
log = logging.getLogger("golestan_code_mcp.server")
READ_ONLY = ToolAnnotations(readOnlyHint=True)
LIMIT = 200  # graph tools return at most this many rows; truncated says when more existed

Name = Annotated[str, Field(description="A bare name (url_for), which matches url_for and every Class.url_for, or a "
                            "qualified name (Flask.url_for), which matches only that definition. Case-sensitive.")]
Repo = Annotated[str | None, Field(description="Look only in this registered repository, for example flask. "
                                   "list_repositories gives the valid names. Default: all repositories.")]


def _logged(tool):
    """One structured record per call: tool, latency, status, result count. Never the arguments or results."""
    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        started, result = time.monotonic(), None
        try:
            result = tool(*args, **kwargs)
            return result
        finally:
            items = next((getattr(result, a) for a in ("results", "repositories", "pages") if hasattr(result, a)),
                         None)
            if items is not None and getattr(result, "page", None) is not None:
                items = [result.page]
            log.info(json.dumps({
                "event": "tool", "tool": tool.__name__, "ms": round((time.monotonic() - started) * 1000),
                "status": "error" if result is None else getattr(result, "status", None) or result.database,
                "results": None if items is None else len(items),
            }))
    return wrapper


def _answer(envelope, check, query, empty=None):
    """Refuse bad input before connecting, then answer from the core. Every failure is an unavailable result,
    with the envelope's `empty` fields (no results by default)."""
    empty = {"results": []} if empty is None else empty
    try:
        check()
        with db.connect(load()) as conn:
            schema.migrate(conn)
            return query(conn)
    except ValueError as e:  # ConfigError, RegistryError, ModelError and WikiError are ValueErrors
        return envelope(status="unavailable", reason=str(e), **empty)
    except oracledb.Error as e:
        return envelope(status="unavailable", reason=db.describe(e), **empty)


def _repos(repo: str | None) -> list[str] | None:
    return [repo] if repo else None


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
    start_line: int | None
    end_line: int | None
    language: str
    snippet: str
    state: Literal["fresh", "stale"] | None = None


class SearchResults(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    results: list[SearchHit]


@mcp.tool(
    description="Find code by what it does or by name: \"how are requests routed?\", \"where are sessions saved?\", "
    "\"url_for\". Use this first when you do not yet know which symbol to look at. If you already know the "
    "name, use get_symbol to read it, find_callers for its call sites, or impact_of for what a change would "
    "affect. Returns up to limit hits (default 10), best first. Each hit has the repository, path, symbol "
    "name, kind, language, line range, a snippet of the symbol's opening lines, and the scores behind its "
    "rank: the fused score, the keyword rank and the semantic rank (null when that side did not find it). "
    "Keyword and semantic search are fused, and the definitions of an exact identifier come first. A hit of "
    "kind page is a wiki page about a whole module: its name is the page name, its path is the module's "
    "directory, its line range is null and its state says whether it is fresh or stale; read it with "
    "read_wiki. Next step: call get_symbol with a hit's name to read its full source, or read_wiki with a "
    "page hit's name.",
    annotations=READ_ONLY,
)
@_logged
def search_code(
    query: Annotated[str, Field(description="A question in plain English (how are requests routed) or an "
                                "identifier (url_for). Taken literally; there is no query syntax.")],
    repo: Annotated[str | None, Field(description="Search only this registered repository, for example flask. "
                                      "list_repositories gives the valid names. Default: all repositories.")] = None,
    language: Annotated[str | None, Field(description="Search only files of this language: one of "
                                          f"{', '.join(semantic.LANGUAGES)}. Default: all languages.")] = None,
    limit: Annotated[int, Field(description="How many hits to return, 1 to 50. Default 10.")] = 10,
) -> SearchResults:
    def check():
        if not query.strip():
            raise ValueError("the question is empty")
        semantic.check_limit(limit)
        semantic.check_language(language)

    def query_(conn):
        hits = semantic.search(conn, query, _repos(repo), language=language, limit=limit)
        return SearchResults(status="ok", reason=None, results=[SearchHit(**vars(h)) for h in hits])

    return _answer(SearchResults, check, query_)


class SymbolHit(BaseModel):
    name: str
    kind: str
    repo: str
    path: str
    start_line: int
    end_line: int
    source: str
    cut: bool


class SymbolResults(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    results: list[SymbolHit]
    total: int = 0
    truncated: bool = False


@mcp.tool(
    description="Read the full source of a named definition (function, method, class, type) when you know its name, "
    "for example from a search_code hit. Use this instead of search_code when the name is known, and instead of "
    "opening files: it returns the source exactly as it was indexed. A bare name returns every definition with that "
    "name, each labeled with its qualified name; pass a qualified name or a repo to narrow. Returns up to 5 matches, "
    "each with its qualified name, kind, repository, path, line range and source. A source longer than 400 lines is "
    "cut there and marked cut, and its full line range is still given. total is the number of matches and "
    "truncated is true when more than 5 matched. Unparsed files indexed as line windows have no names and are not "
    "found here; use search_code for them.",
    annotations=READ_ONLY,
)
@_logged
def get_symbol(name: Name, repo: Repo = None) -> SymbolResults:
    def query(conn):
        found, total = graph.definitions(conn, name, _repos(repo))
        return SymbolResults(status="ok", reason=None, results=[SymbolHit(**vars(d)) for d in found],
                             total=total, truncated=total > len(found))

    return _answer(SymbolResults, lambda: graph.check_name(name), query)


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
    truncated: bool = False


class CalleeResults(CallerResults):
    results: list[CalleeHit]


class ImpactResults(CallerResults):
    results: list[ImpactHit]


def _graph(envelope, hit, check, rows):
    """A graph question as an envelope of at most LIMIT rows."""
    def query(conn):
        found = rows(conn)
        return envelope(status="ok", reason=None, results=[hit(**vars(r)) for r in found[:LIMIT]],
                        truncated=len(found) > LIMIT)

    return _answer(envelope, check, query)


@mcp.tool(
    description="Who calls this? Lists every call to the definitions matching name: the calling symbol, its kind, the "
    "repository, path and line of the call, and which definition is called. Use it before changing a function's "
    "signature or behavior, to see its direct call sites. For everything a change could reach through chains of "
    "calls, use impact_of; for what this symbol itself calls, use find_callees. Calls come from the parse tree, so "
    "comments, docstrings and strings never count. A call at module level has kind file and name (module). At most "
    "200 rows; truncated is true when more exist. Next step: get_symbol on a caller to read it.",
    annotations=READ_ONLY,
)
@_logged
def find_callers(name: Name, repo: Repo = None) -> CallerResults:
    return _graph(CallerResults, CallerHit, lambda: graph.check_name(name),
                  lambda conn: graph.callers(conn, name, _repos(repo)))


@mcp.tool(
    description="What does this call? Lists the indexed symbols called from the body of each definition matching "
    "name, with the line of each call and where the called symbol is defined (repository, path, line range). Use it "
    "to follow what a function does one level down; for who calls it, use find_callers. Calls into code that is not "
    "indexed, such as the standard library or dependencies, are not listed. At most 200 rows; truncated is true "
    "when more exist. Next step: get_symbol on a callee to read it.",
    annotations=READ_ONLY,
)
@_logged
def find_callees(name: Name, repo: Repo = None) -> CalleeResults:
    return _graph(CalleeResults, CalleeHit, lambda: graph.check_name(name),
                  lambda conn: graph.calls(conn, name, _repos(repo)))


@mcp.tool(
    description="What could break if I change this? Lists every symbol that reaches the definitions matching name "
    "through calls in at most depth hops, once each at its shortest distance (hops), nearest first, with the "
    "repository, path and line range. Use it to gauge the reach of a change; for just the direct call sites and "
    "their lines, use find_callers. At most 200 rows; truncated is true when more exist. Next step: get_symbol on "
    "the nearest symbols to see how they use it.",
    annotations=READ_ONLY,
)
@_logged
def impact_of(
    name: Name,
    depth: Annotated[int, Field(description="How many call hops to follow, 1 to 5. Default 3. 1 gives only the "
                                "direct callers.")] = 3,
    repo: Repo = None,
) -> ImpactResults:
    def check():
        graph.check_name(name)
        graph.check_depth(depth)

    return _graph(ImpactResults, ImpactHit, check, lambda conn: graph.impact(conn, name, depth, _repos(repo)))


class WikiCitation(BaseModel):
    name: str
    kind: str
    role: Literal["key", "file", "depends", "uses", "calls"]
    repo: str | None
    path: str | None
    start_line: int | None
    end_line: int | None
    commit: str
    current_commit: str | None


class WikiPage(BaseModel):
    name: str
    repo: str
    directory: str
    language: str
    state: Literal["fresh", "stale"]
    compiled_at: str
    compiler: str
    stale_reason: str | None
    symbols: int
    text: str
    citations: list[WikiCitation]
    truncated: bool


class WikiPageInfo(BaseModel):
    name: str
    repo: str
    state: Literal["fresh", "stale"]
    summary: str


class WikiResults(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    page: WikiPage | None = None
    pages: list[WikiPageInfo] = []
    truncated: bool = False


def _iso(at) -> str:
    return (at.astimezone(UTC) if at.tzinfo else at).strftime("%Y-%m-%dT%H:%M:%SZ")


@mcp.tool(
    description="What does this module do? Reads the tool's wiki page about a module (a directory of indexed code): its "
    "purpose, its key symbols with one line each, the modules it depends on and the modules that use it. Use "
    "it when a question is about a module as a whole rather than one function, or after search_code returns a"
    " hit of kind page. Pass page to read one page; it returns the text, whether it is fresh or stale (and "
    "what made it stale), when and by what it was compiled, and its citations, each with the repository, path"
    " and line range of the symbol or file it names. Omit page to list the pages, optionally for one repo, "
    "with each page's name, state and first purpose line. A stale page describes code that has changed since;"
    " check its citations. Next step: get_symbol on a key symbol to read its source.",
    annotations=READ_ONLY,
)
@_logged
def read_wiki(
    page: Annotated[str | None, Field(description="The page name, for example flask/json, as returned by "
                                      "search_code or by listing. Case-sensitive. Default: list the pages "
                                      "instead.")] = None,
    repo: Annotated[str | None, Field(description="Look only in this registered repository, for example flask. "
                                      "list_repositories gives the valid names. Needed when two repositories "
                                      "have a page of the same name. Default: all repositories.")] = None,
) -> WikiResults:
    def check():
        if page is not None and not page.strip():
            raise ValueError("the page name is empty")

    def query(conn):
        if page is None:
            found = wiki.pages(conn, repo)
            return WikiResults(status="ok", reason=None, truncated=len(found) > LIMIT, pages=[
                WikiPageInfo(name=p.name, repo=p.repo, state=p.state, summary=p.summary) for p in found[:LIMIT]])
        try:
            p = wiki.read(conn, page, repo)
        except wiki.WikiError as e:  # the CLI's flag is this tool's parameter
            raise ValueError(str(e).replace("; pass --repo", "; pass repo")) from None
        citations = [WikiCitation(name=posixpath.basename(c.name) if c.kind == "file" else c.name, kind=c.kind,
                                  role=c.role, repo=c.repo, path=c.path, start_line=c.start_line,
                                  end_line=c.end_line, commit=c.recorded_commit, current_commit=c.current_commit)
                     for c in p.citations]
        return WikiResults(status="ok", reason=None, page=WikiPage(
            name=p.name, repo=p.repo, directory=p.directory, language=p.language, state=p.state,
            compiled_at=_iso(p.compiled_at), compiler=p.compiler, stale_reason=p.stale_reason, symbols=p.symbols,
            text=p.text, citations=citations, truncated=p.truncated))

    return _answer(WikiResults, check, query, empty={})


class RepoInfo(BaseModel):
    name: str
    path: str
    files: int
    symbols: int


class RepoList(BaseModel):
    database: Literal["reachable", "unreachable"]
    reason: str | None
    repositories: list[RepoInfo]


@mcp.tool(
    description="Which codebases can I ask about? Lists the registered repositories with their paths and how many "
    "files and symbols each holds. Use it to learn the valid repo values for the other tools, or after a repo was "
    "refused. A repository with 0 files is registered but not indexed yet.",
    annotations=READ_ONLY,
)
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


# FastMCP drops arguments a tool does not declare. Refuse them instead: a stale `repos` would otherwise widen a
# search to every repository without saying so.
for _tool in mcp._tool_manager.list_tools():
    _strict = type(_tool.fn_metadata.arg_model.__name__, (_tool.fn_metadata.arg_model,),
                   {"model_config": ConfigDict(extra="forbid")})
    _tool.fn_metadata.arg_model = _strict
    _tool.parameters = _strict.model_json_schema(by_alias=True)


def main() -> None:
    log.addHandler(logging.StreamHandler(sys.stderr))  # stdout carries the protocol
    log.setLevel(logging.INFO)
    log.propagate = False
    mcp.run("stdio")


if __name__ == "__main__":
    main()
