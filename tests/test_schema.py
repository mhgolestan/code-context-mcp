"""Schema migrations (contracts/schema.md). Skipped when the database is unreachable."""

import uuid

import oracledb
import pytest

from golestan_code_mcp import db, schema
from golestan_code_mcp.config import load

TABLES = {"CCCTX_MIGRATIONS", "CCCTX_REPOS", "CCCTX_FILES", "CCCTX_SYMBOLS"}


@pytest.fixture(scope="module")
def conn():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    schema.migrate(conn)
    yield conn
    conn.close()


def insert(cur, sql, **binds):
    new_id = cur.var(int)
    cur.execute(f"{sql} returning id into :new_id", new_id=new_id, **binds)
    return new_id.getvalue()[0]


@pytest.fixture
def repo_id(conn):
    """A throwaway repository, removed (with everything under it) after the test."""
    name = f"t-{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        rid = insert(cur, "insert into ccctx_repos (name, path) values (:n, :p)", n=name, p=f"/tmp/{name}")
    conn.commit()
    yield rid
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where id = :i", i=rid)
    conn.commit()


def add_file(cur, repo_id, path="a.py"):
    return insert(
        cur,
        "insert into ccctx_files (repo_id, path, language, content_hash) values (:r, :p, 'python', :h)",
        r=repo_id, p=path, h="0" * 64,
    )


SHA = "a" * 40


def add_symbol(cur, file_id, kind="function", start=1, end=2, sha=SHA):
    return insert(
        cur,
        "insert into ccctx_symbols (file_id, name, kind, start_line, end_line, source, content_hash, commit_sha)"
        " values (:f, 'f', :k, :s, :e, 'def f(): pass', :h, :c)",
        f=file_id, k=kind, s=start, e=end, h="0" * 64, c=sha,
    )


def counts(conn):
    with conn.cursor() as cur:
        return {t: cur.execute(f"select count(*) from {t}").fetchone()[0] for t in sorted(TABLES)}


def test_tables_exist_and_version_recorded(conn):
    with conn.cursor() as cur:
        cur.execute("select table_name from user_tables")
        assert TABLES <= {row[0] for row in cur}
        cur.execute("select version, name from ccctx_migrations order by version")
        assert cur.fetchall()[:3] == [(1, "core tables"), (2, "symbol commit"), (3, "symbol embedding")]
        cur.execute("select column_name, data_type, nullable from user_tab_columns where table_name = 'CCCTX_SYMBOLS'")
        columns = {name: (kind, nullable) for name, kind, nullable in cur}
        assert "COMMIT_SHA" in columns
        assert columns["EMBEDDING"] == ("VECTOR", "Y")  # 384 x FLOAT32 is checked on real rows in test_semantic
        cur.execute("select table_name from user_tables where table_name = 'CCCTX_EMBEDDING_MODEL'")
        assert cur.fetchone()


def test_migrate_three_times_changes_nothing(conn):
    before = counts(conn)
    for _ in range(3):
        assert schema.migrate(conn) == []
        assert counts(conn) == before


@pytest.mark.parametrize("name", ["Flask", "a b", "a/b", "x" * 65])
def test_bad_repo_name_rejected(conn, name):
    with conn.cursor() as cur, pytest.raises(oracledb.DatabaseError):
        cur.execute("insert into ccctx_repos (name, path) values (:n, :p)", n=name, p=f"/tmp/bad-{uuid.uuid4().hex}")
    conn.rollback()


def test_file_path_unique_per_repo(conn, repo_id):
    with conn.cursor() as cur:
        add_file(cur, repo_id)
        with pytest.raises(oracledb.IntegrityError):
            add_file(cur, repo_id)
    conn.rollback()


@pytest.mark.parametrize(("kind", "start", "end"), [("lambda", 1, 2), ("function", 5, 4), ("function", 0, 1)])
def test_bad_symbol_rejected(conn, repo_id, kind, start, end):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        with pytest.raises(oracledb.DatabaseError):
            add_symbol(cur, file_id, kind, start, end)
    conn.rollback()


def test_delete_cascades(conn, repo_id):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        add_symbol(cur, file_id, "class")
        cur.execute("delete from ccctx_repos where id = :i", i=repo_id)
        cur.execute("select count(*) from ccctx_files where repo_id = :i", i=repo_id)
        assert cur.fetchone()[0] == 0
        cur.execute("select count(*) from ccctx_symbols where file_id = :i", i=file_id)
        assert cur.fetchone()[0] == 0
    conn.rollback()


@pytest.mark.parametrize("sha", ["a" * 40, "0123456789abcdef" * 4])
def test_commit_sha_accepts_sha1_and_sha256(conn, repo_id, sha):
    with conn.cursor() as cur:
        add_symbol(cur, add_file(cur, repo_id), sha=sha)
    conn.rollback()


@pytest.mark.parametrize("sha", ["abc", "A" * 40, "g" * 40, "a" * 41, None])
def test_commit_sha_rejects_bad_values(conn, repo_id, sha):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        with pytest.raises(oracledb.DatabaseError):
            add_symbol(cur, file_id, sha=sha)
    conn.rollback()
