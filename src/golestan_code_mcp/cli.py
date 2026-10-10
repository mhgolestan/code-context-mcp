"""ccctx: register and list codebases (contracts/cli.md). Exit 0 ok, 1 refused input, 2 database down."""

import argparse
import sys

import oracledb

from golestan_code_mcp import db, registry, schema
from golestan_code_mcp.config import ConfigError, load


def format_table(repos: list[registry.Repo]) -> str:
    rows = [("NAME", "PATH", "FILES", "SYMBOLS")] + [(r.name, r.path, str(r.files), str(r.symbols)) for r in repos]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return "\n".join(
        "  ".join([name.ljust(widths[0]), path.ljust(widths[1]), files.rjust(widths[2]), symbols.rjust(widths[3])])
        for name, path, files, symbols in rows
    )


def _repo_add(conn, args) -> str:
    repo = registry.add_repo(conn, args.path, args.name)
    return f"registered {repo.name} {repo.path}"


def _repo_list(conn, args) -> str:
    repos = registry.list_repos(conn)
    return format_table(repos) if repos else "no repositories registered"


def _db_migrate(conn, args) -> str:
    applied = schema.migrate(conn)
    names = {v: n for v, n, _ in schema.MIGRATIONS}
    if applied:
        return "\n".join(f"applied {v} {names[v]}" for v in applied)
    return f"schema up to date (version {schema.version(conn)})"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccctx", description="Code context: manage indexed codebases.")
    groups = p.add_subparsers(dest="group", required=True)

    repo = groups.add_parser("repo", help="register and list repositories").add_subparsers(dest="cmd", required=True)
    add = repo.add_parser("add", help="register a local directory")
    add.add_argument("path")
    add.add_argument("--name", required=True)
    add.set_defaults(run=_repo_add)
    repo.add_parser("list", help="list registered repositories").set_defaults(run=_repo_list)

    dbp = groups.add_parser("db", help="database administration").add_subparsers(dest="cmd", required=True)
    dbp.add_parser("migrate", help="apply missing schema migrations").set_defaults(run=_db_migrate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.run is _repo_add:  # refuse bad input before touching the database
            registry.validate_name(args.name)
            registry.resolve_dir(args.path)
        with db.connect(load()) as conn:
            if args.run is not _db_migrate:
                schema.migrate(conn)
            print(args.run(conn, args))
    except (ConfigError, registry.RegistryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except oracledb.Error as e:
        code = getattr(e.args[0], "full_code", "")
        reason = "database is still starting" if code in db.STARTING_CODES else f"database unreachable: {db.describe(e)}"
        print(f"error: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
