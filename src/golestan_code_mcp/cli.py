"""ccctx: register, ingest and inspect codebases (contracts/cli.md). Exit 0 ok, 1 refused input, 2 database down."""

import argparse
import logging
import posixpath
import re
import sys
from datetime import UTC

import oracledb

from golestan_code_mcp import db, graph, ingest, registry, schema, semantic, wiki
from golestan_code_mcp.config import environment, load

SKIP_ORDER = ["ignored", "vendored", "binary", "too large"]
KINDS = ["function", "method", "class", "type", "interface", "chunk"]


def format_table(header: tuple[str, ...], rows: list[tuple], numeric: frozenset[int] = frozenset()) -> str:
    """Two-space separated columns; columns in `numeric` are right-aligned."""
    table = [header] + [tuple(str(v) for v in row) for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return "\n".join(
        "  ".join(v.rjust(w) if i in numeric else v.ljust(w) for i, (v, w) in enumerate(zip(row, widths))).rstrip()
        for row in table
    )


def format_summary(s: ingest.Summary) -> str:
    if s.mode == "current":
        return f"{s.repo} is already up to date at {s.commit[:12]}; run with --full to re-index anyway"
    text = _format_incremental(s) if s.mode == "incremental" else _format_full(s)
    return text if s.wiki is None else "\n".join([text, *_format_wiki(s.wiki)])


def _format_wiki(w: wiki.Refresh) -> list[str]:
    done = "not recompiled" if w.error else "recompiled"
    return [f"{'Wiki':<9} {page} stale: {reason}, {done}" for page, reason in w.stale] + [
        (f"{'Wiki':<9} {len(w.stale)} stale, {0 if w.error else w.recompiled} recompiled, {w.added} added,"
         f" {w.removed} removed")]


def _format_full(s: ingest.Summary) -> str:
    skipped = sum(s.skipped.values())
    width = max(len(f"{n:,}") for n in (s.scanned, skipped, s.symbols))
    reasons = ", ".join(f"{s.skipped[r]} {r}" for r in SKIP_ORDER if s.skipped[r])
    lines = [
        f"Scanned {s.scanned:>{width},} files",
        f"Skipped {skipped:>{width},} files" + (f" ({reasons})" if reasons else ""),
        f"Indexed {s.symbols:>{width},} symbols in {s.seconds:.1f}s",
    ]
    if s.no_grammar or s.parse_errors:
        lines.append(
            f"Fallback {s.no_grammar + s.parse_errors:>{width - 1},} files as line windows"
            f" ({s.no_grammar} no grammar, {s.parse_errors} parse errors)"
        )
    return "\n".join(lines)


def _format_incremental(s: ingest.Summary) -> str:
    reasons = ", ".join(f"{s.skipped[r]} {r}" for r in SKIP_ORDER if s.skipped[r])
    rows = [
        ("Changed", s.added + s.changed + s.removed,
         f" files ({s.added} added, {s.changed} changed, {s.removed} removed)"),
        ("Unchanged", s.unchanged, " files"),
        ("Skipped", sum(s.skipped.values()), " files" + (f" ({reasons})" if reasons else "")),
        ("Indexed", s.symbols, f" symbols in {s.seconds:.1f}s"),
    ]
    if s.no_grammar or s.parse_errors:
        rows.append(("Fallback", s.no_grammar + s.parse_errors,
                     f" files as line windows ({s.no_grammar} no grammar, {s.parse_errors} parse errors)"))
    width = max(len(f"{n:,}") for _, n, _ in rows)
    return "\n".join(f"{label:<9} {n:>{width},}{rest}" for label, n, rest in rows)


def _repo_add(conn, args) -> str:
    repo = registry.add_repo(conn, args.path, args.name)
    return f"registered {repo.name} {repo.path}"


def _repo_list(conn, args) -> str:
    repos = registry.list_repos(conn)
    if not repos:
        return "no repositories registered"
    rows = [(r.name, r.path, r.files, r.symbols) for r in repos]
    return format_table(("NAME", "PATH", "FILES", "SYMBOLS"), rows, frozenset({2, 3}))


def _ingest(conn, args) -> str:
    summary = ingest.ingest(conn, args.name, full=args.full)
    if summary.note:
        print(f"note: {summary.note}", file=sys.stderr)
    if summary.dirty and summary.mode != "current":
        print("warning: working tree has uncommitted changes; indexed the files as they are on disk", file=sys.stderr)
    if summary.wiki and summary.wiki.error:
        print(f"warning: wiki not recompiled: {summary.wiki.error}", file=sys.stderr)
    return format_summary(summary)


def _symbols(conn, args) -> str:
    repo_id = registry.repo_id(conn, args.repo)
    with conn.cursor() as cur:
        cur.execute(
            """select s.name, s.kind, f.language, f.path, s.start_line || '-' || s.end_line
                 from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                where f.repo_id = :r and (:k is null or s.kind = :k)
                order by case when s.kind = 'chunk' then 1 else 0 end, f.path, s.start_line, s.end_line desc
                fetch first :lim rows only""",
            r=repo_id, k=args.kind, lim=args.limit,
        )
        rows = cur.fetchall()
    return format_table(("NAME", "KIND", "LANGUAGE", "PATH", "LINES"), rows) if rows else "no symbols"


def _stats(conn, args) -> str:
    repo_id = registry.repo_id(conn, args.repo) if args.repo else None
    with conn.cursor() as cur:
        cur.execute(
            """select r.name, f.language, count(distinct f.id),
                      count(case when s.kind <> 'chunk' then 1 end), count(case when s.kind = 'chunk' then 1 end)
                 from ccctx_repos r join ccctx_files f on f.repo_id = r.id
                 left join ccctx_symbols s on s.file_id = f.id
                where (:r is null or r.id = :r)
                group by r.name, f.language
                order by r.name, f.language""",
            r=repo_id,
        )
        rows = cur.fetchall()
    if not rows:
        return "nothing indexed"
    return format_table(("REPO", "LANGUAGE", "FILES", "SYMBOLS", "CHUNKS"), rows, frozenset({2, 3, 4}))


def _model_load(conn, args) -> str:
    return semantic.load_model(conn, environment())


def _repos(values: list[str] | None) -> list[str] | None:
    """--repo given any number of times, each possibly comma-separated; duplicates dropped, order kept."""
    names = [n.strip() for v in values or [] for n in v.split(",") if n.strip()]
    return list(dict.fromkeys(names)) or None


def _dash(rank: int | None):
    return "-" if rank is None else rank


def _search(conn, args) -> str:
    results = semantic.search(conn, args.question, _repos(args.repo), args.mode, environment(), args.language,
                              args.limit)
    if not results:
        if args.mode == "vector":
            return f"no symbols similar enough (similarity >= {semantic.FLOOR:.2f})"
        return "no matching symbols"
    rows = [(r.rank, f"{r.score:.2f}", _dash(r.keyword_rank), _dash(r.semantic_rank), r.repo,
             f"{r.kind} (stale)" if r.state == "stale" else r.kind, r.name, r.path,
             "-" if r.start_line is None else f"{r.start_line}-{r.end_line}") for r in results]
    header = ("RANK", "SCORE", "KEYWORD", "SEMANTIC", "REPO", "KIND", "NAME", "PATH", "LINES")
    return format_table(header, rows, frozenset({0, 1, 2, 3}))


def _callers(conn, args) -> str:
    results = graph.callers(conn, args.name, _repos(args.repo))
    if not results:
        return f"no callers of '{args.name.strip()}'"
    rows = [(r.name, r.kind, r.repo, r.path, r.line, r.callee) for r in results]
    return format_table(("CALLER", "KIND", "REPO", "PATH", "LINE", "CALLS"), rows, frozenset({4}))


def _calls(conn, args) -> str:
    results = graph.calls(conn, args.name, _repos(args.repo))
    if not results:
        return f"'{args.name.strip()}' calls no indexed symbols"
    rows = [(r.caller, r.line, r.name, r.kind, r.repo, r.path, f"{r.start_line}-{r.end_line}") for r in results]
    return format_table(("FROM", "LINE", "CALLEE", "KIND", "REPO", "PATH", "LINES"), rows, frozenset({1}))


def _impact(conn, args) -> str:
    results = graph.impact(conn, args.name, args.depth, _repos(args.repo))
    if not results:
        return f"nothing calls '{args.name.strip()}'"
    rows = [(r.hops, r.name, r.kind, r.repo, r.path, f"{r.start_line}-{r.end_line}") for r in results]
    return format_table(("HOPS", "NAME", "KIND", "REPO", "PATH", "LINES"), rows, frozenset({0}))


GRAPH = (_callers, _calls, _impact)


def _when(at) -> str:
    """A compile time in UTC, YYYY-MM-DD HH:MM:SS."""
    return (at.astimezone(UTC) if at.tzinfo else at).strftime("%Y-%m-%d %H:%M:%S")


def _wiki_build(conn, args) -> str:
    built = wiki.build(conn, args.repo)
    with conn.cursor() as cur:
        if not wiki.ensure_index(cur):
            print("note: wiki pages are not searchable until you run `ccctx model load`", file=sys.stderr)
    return "\n".join(f"{b.repo}: {b.pages} pages ({b.added} added, {b.updated} updated, {b.removed} removed)"
                     f" in {b.seconds:.1f}s" for b in built)


def _wiki_list(conn, args) -> str:
    pages = wiki.pages(conn, args.repo)
    if not pages:
        return "no wiki pages"
    rows = [(p.name, p.repo, p.state, p.symbols, _when(p.compiled_at), p.stale_reason or "-") for p in pages]
    return format_table(("NAME", "REPO", "STATE", "SYMBOLS", "COMPILED", "LAST STALE"), rows, frozenset({3}))


def _source(c: wiki.CitationView) -> str:
    """One Sources line: the citation located through its key, or `gone`."""
    if c.path is None:
        where = "gone"
    elif c.kind == "file":
        where = ""
    else:
        where = f"{c.path}:{c.start_line}-{c.end_line}"
    return f"- `{c.name}` ({c.kind}, {c.role}) {where}".rstrip()


def _wiki_show(conn, args) -> str:
    p = wiki.read(conn, args.page, args.repo)
    lines = [f"{p.name} ({p.repo}, {p.directory}) {p.state}, compiled {_when(p.compiled_at)} by {p.compiler}"]
    if p.state == "stale":
        lines.append(f"stale: {p.stale_reason}")
    return "\n".join([*lines, "", p.text.rstrip("\n"), "", "## Sources", *map(_source, p.citations)])


def _wiki_citations(conn, args) -> str:
    p = wiki.read(conn, args.page, args.repo)
    rows = [(posixpath.basename(c.name) if c.kind == "file" else c.name, c.kind, c.role, c.repo or "-", c.path or "-",
             "-" if c.start_line is None else f"{c.start_line}-{c.end_line}", c.recorded_commit[:12],
             (c.current_commit or "gone")[:12] if c.path else "gone") for c in p.citations]
    return format_table(("NAME", "KIND", "ROLE", "REPO", "PATH", "LINES", "COMPILED", "CURRENT"), rows)


def _db_migrate(conn, args) -> str:
    applied = schema.migrate(conn)
    names = {v: n for v, n, _ in schema.MIGRATIONS}
    if applied:
        return "\n".join(f"applied {v} {names[v]}" for v in applied)
    return f"schema up to date (version {schema.version(conn)})"


def _positive(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int(text)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccctx", description="Code context: manage indexed codebases.")
    groups = p.add_subparsers(dest="group", required=True)

    repo = groups.add_parser("repo", help="register and list repositories").add_subparsers(dest="cmd", required=True)
    add = repo.add_parser("add", help="register a local directory")
    add.add_argument("path")
    add.add_argument("--name", required=True)
    add.set_defaults(run=_repo_add)
    repo.add_parser("list", help="list registered repositories").set_defaults(run=_repo_list)

    ing = groups.add_parser("ingest", help="index a registered repository")
    ing.add_argument("name")
    ing.add_argument("--full", action="store_true", help="discard the index and rebuild it from the working tree")
    ing.set_defaults(run=_ingest)

    sym = groups.add_parser("symbols", help="list indexed symbols")
    sym.add_argument("--repo", required=True)
    sym.add_argument("--limit", type=_positive, default=20)
    sym.add_argument("--kind", choices=KINDS)
    sym.set_defaults(run=_symbols)

    st = groups.add_parser("stats", help="index statistics")
    st.add_argument("--by-language", action="store_true", required=True)
    st.add_argument("--repo")
    st.set_defaults(run=_stats)

    srch = groups.add_parser("search", help="find code by name or meaning (keyword + semantic, fused)")
    srch.add_argument("question")
    srch.add_argument("--mode", default="hybrid", help="vector, text or hybrid (default)")
    srch.add_argument("--repo", action="append", help="limit to these repositories; repeat or comma-separate")
    srch.add_argument("--language", help="limit to files of this language")
    srch.add_argument("--limit", default="10", help=f"hits to return, 1 to {semantic.MAX_LIMIT} (default 10)")
    srch.set_defaults(run=_search)

    for run, cmd, text in [(_callers, "callers", "list every call to a symbol"),
                           (_calls, "calls", "list the indexed symbols a symbol calls"),
                           (_impact, "impact", "list the symbols a change could affect, by call hops")]:
        g = groups.add_parser(cmd, help=text)
        g.add_argument("name")
        g.add_argument("--repo", action="append", help="limit to these repositories; repeat or comma-separate")
        if run is _impact:
            g.add_argument("--depth", default="3", help=f"call hops to follow, 1 to {graph.MAX_DEPTH} (default 3)")
        g.set_defaults(run=run)

    wk = groups.add_parser("wiki", help="module pages written from the index").add_subparsers(dest="cmd", required=True)
    for cmd, run, text in [("build", _wiki_build, "compile the pages of a repository, or of every indexed one"),
                           ("list", _wiki_list, "list the pages with their state"),
                           ("show", _wiki_show, "print a page and where each cited symbol is now"),
                           ("citations", _wiki_citations, "list a page's citations")]:
        w = wk.add_parser(cmd, help=text)
        if run in (_wiki_show, _wiki_citations):
            w.add_argument("page")
        w.add_argument("--repo")
        w.set_defaults(run=run)
    model = groups.add_parser("model", help="embedding model").add_subparsers(dest="cmd", required=True)
    model.add_parser("load", help="download once and load the embedding model").set_defaults(run=_model_load)

    groups.add_parser("mcp", help="run the MCP server over standard input and output").set_defaults(run=None)

    dbp = groups.add_parser("db", help="database administration").add_subparsers(dest="cmd", required=True)
    dbp.add_parser("migrate", help="apply missing schema migrations").set_defaults(run=_db_migrate)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # a docstring's character the console cannot show must not end the run
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parser().parse_args(argv)
    if args.group == "mcp":  # before connecting: the server reports an outage per call
        from golestan_code_mcp import server  # only this command needs the MCP library

        server.main()
        return 0
    handler = logging.StreamHandler(sys.stderr)  # the structured ingest and wiki records go to stderr
    for log in (ingest.log, wiki.log):
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    conn = None
    try:
        if args.run is _repo_add:  # refuse bad input before touching the database
            registry.validate_name(args.name)
            registry.resolve_dir(args.path)
        if args.run is _search:  # refuse bad input before touching the database
            if not args.question.strip():
                raise ValueError("the question is empty")
            if args.mode not in semantic.MODES:
                raise ValueError(f"unknown mode '{args.mode}'; valid modes: {', '.join(semantic.MODES)}")
            args.limit = semantic.check_limit(int(args.limit) if re.fullmatch(r"-?[0-9]+", args.limit) else None)
            semantic.check_language(args.language)
        if args.run in GRAPH:  # refuse bad input before touching the database
            graph.check_name(args.name)
            if args.run is _impact:  # a bad --depth exits 1, not argparse's 2
                args.depth = graph.check_depth(int(args.depth) if re.fullmatch(r"-?[0-9]+", args.depth) else None)
        conn = db.connect(load())
        with conn:
            if args.run is not _db_migrate:
                schema.migrate(conn)
            print(args.run(conn, args))
    except (ValueError, ingest.IngestError) as e:  # ConfigError, RegistryError and ModelError are ValueErrors
        print(f"error: {e}", file=sys.stderr)
        return 1
    except oracledb.Error as e:
        code = getattr(e.args[0], "full_code", "")
        if code in db.STARTING_CODES:
            reason = "database is still starting"
        else:  # a failure after connecting is not an outage
            reason = f"database {'error' if conn else 'unreachable'}: {db.describe(e)}"
        print(f"error: {reason}", file=sys.stderr)
        return 2
    finally:
        ingest.log.removeHandler(handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
