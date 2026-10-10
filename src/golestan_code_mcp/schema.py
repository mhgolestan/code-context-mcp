"""Ordered, re-runnable schema migrations (contracts/schema.md).

Oracle commits DDL as it runs, so a migration cannot be rolled back as a whole. Each statement
tolerates "already exists" and the version row is written only after every statement succeeded,
so a half-applied migration is finished by the next run.
"""

import oracledb

ALREADY_EXISTS = {"ORA-00955", "ORA-02260", "ORA-02261", "ORA-01408"}

TRACKING = """create table ccctx_migrations (
  version     number primary key,
  name        varchar2(128) not null,
  applied_at  timestamp with time zone default systimestamp not null
)"""

MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "core tables", [
        """create table ccctx_repos (
          id             number generated always as identity primary key,
          name           varchar2(64)   not null unique,
          path           varchar2(1024) not null unique,
          registered_at  timestamp with time zone default systimestamp not null,
          constraint ccctx_repos_name_ck check (regexp_like(name, '^[a-z0-9_-]{1,64}$'))
        )""",
        """create table ccctx_files (
          id            number generated always as identity primary key,
          repo_id       number not null references ccctx_repos (id) on delete cascade,
          path          varchar2(1024) not null,
          language      varchar2(32)   not null,
          content_hash  varchar2(64)   not null,
          constraint ccctx_files_path_uk unique (repo_id, path)
        )""",
        """create table ccctx_symbols (
          id            number generated always as identity primary key,
          file_id       number not null references ccctx_files (id) on delete cascade,
          name          varchar2(512) not null,
          kind          varchar2(16)  not null,
          start_line    number not null,
          end_line      number not null,
          source        clob not null,
          content_hash  varchar2(64) not null,
          constraint ccctx_symbols_kind_ck
            check (kind in ('function','method','class','interface','type','module','chunk')),
          constraint ccctx_symbols_lines_ck check (start_line >= 1 and end_line >= start_line)
        )""",
        "create index ccctx_symbols_file_ix on ccctx_symbols (file_id)",
    ]),
]


def _run(cur, sql: str) -> None:
    try:
        cur.execute(sql)
    except oracledb.DatabaseError as e:
        if getattr(e.args[0], "full_code", "") not in ALREADY_EXISTS:
            raise


def migrate(conn) -> list[int]:
    """Apply every missing migration in order. Returns the versions applied by this call."""
    with conn.cursor() as cur:
        _run(cur, TRACKING)
        applied = {row[0] for row in cur.execute("select version from ccctx_migrations")}
        done = []
        for version, name, statements in MIGRATIONS:
            if version in applied:
                continue
            for sql in statements:
                _run(cur, sql)
            cur.execute("insert into ccctx_migrations (version, name) values (:v, :n)", v=version, n=name)
            conn.commit()
            done.append(version)
        return done


def version(conn) -> int:
    with conn.cursor() as cur:
        return cur.execute("select nvl(max(version), 0) from ccctx_migrations").fetchone()[0]
