"""ccctx: register, ingest and inspect codebases (contracts/cli.md). Exit 0 ok, 1 refused input, 2 database down."""

import argparse
import logging
import sys

import oracledb

from golestan_code_mcp import db, ingest, registry, schema, semantic
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
    summary = ingest.ingest(conn, args.name)
    if summary.dirty:
        print("warning: working tree has uncommitted changes; indexed the files as they are on disk", file=sys.stderr)
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
    results = semantic.search(conn, args.question, _repos(args.repo), args.mode, environment())
    if not results:
        if args.mode == "vector":
            return f"no symbols similar enough (similarity >= {semantic.FLOOR:.2f})"
        return "no matching symbols"
    rows = [(r.rank, f"{r.score:.2f}", _dash(r.keyword_rank), _dash(r.semantic_rank), r.repo, r.kind, r.name, r.path,
             f"{r.start_line}-{r.end_line}") for r in results]
    header = ("RANK", "SCORE", "KEYWORD", "SEMANTIC", "REPO", "KIND", "NAME", "PATH", "LINES")
    return format_table(header, rows, frozenset({0, 1, 2, 3}))


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
    srch.set_defaults(run=_search)

    model = groups.add_parser("model", help="embedding model").add_subparsers(dest="cmd", required=True)
    model.add_parser("load", help="download once and load the embedding model").set_defaults(run=_model_load)

    dbp = groups.add_parser("db", help="database administration").add_subparsers(dest="cmd", required=True)
    dbp.add_parser("migrate", help="apply missing schema migrations").set_defaults(run=_db_migrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handler = logging.StreamHandler(sys.stderr)  # the structured ingest record goes to stderr
    ingest.log.addHandler(handler)
    ingest.log.setLevel(logging.INFO)
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
