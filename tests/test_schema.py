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
        assert cur.fetchall()[:4] == [
            (1, "core tables"), (2, "symbol commit"), (3, "symbol embedding"), (4, "hybrid search text")
        ]
        cur.execute("select column_name, data_type, nullable from user_tab_columns where table_name = 'CCCTX_SYMBOLS'")
        columns = {name: (kind, nullable) for name, kind, nullable in cur}
        assert "COMMIT_SHA" in columns
        assert columns["SEARCH_TEXT"] == ("CLOB", "Y")
        assert "EMBEDDING" not in columns  # replaced by the hybrid index's own vectors
        cur.execute("select count(*) from user_indexes where index_name = 'CCCTX_SYMBOLS_EMBEDDING_IX'")
        assert cur.fetchone()[0] == 0
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


def test_backfill_fills_search_text(conn, repo_id):
    with conn.cursor() as cur:
        symbol_id = add_symbol(cur, add_file(cur, repo_id, "pkg/mod.py"), "method", 3, 9)
        cur.execute(schema.BACKFILL_SEARCH_TEXT)
        text = cur.execute("select search_text from ccctx_symbols where id = :i", i=symbol_id).fetchone()[0].read()
    conn.rollback()
    assert text == "method f in pkg/mod.py\ndef f(): pass"


def test_repo_indexed_revision_columns(conn):
    with conn.cursor() as cur:
        cur.execute("select name from ccctx_migrations where version = 5")
        assert cur.fetchone() == ("repo indexed revision",)
        cur.execute("select column_name, data_type, nullable from user_tab_columns where table_name = 'CCCTX_REPOS'")
        columns = {name: (kind, nullable) for name, kind, nullable in cur}
    assert columns["INDEXED_COMMIT"] == ("VARCHAR2", "Y")
    assert columns["INDEXED_DIRTY"] == ("BOOLEAN", "Y")
    assert columns["INDEXED_AT"][1] == "Y" and columns["INDEXED_AT"][0].startswith("TIMESTAMP")


@pytest.mark.parametrize("sha", [None, "a" * 40, "0123456789abcdef" * 4])
def test_indexed_commit_accepts_null_sha1_sha256(conn, repo_id, sha):
    with conn.cursor() as cur:
        cur.execute("update ccctx_repos set indexed_commit = :c where id = :i", c=sha, i=repo_id)
    conn.rollback()


@pytest.mark.parametrize("sha", ["abc", "A" * 40, "g" * 40, "a" * 41])
def test_indexed_commit_rejects_bad_values(conn, repo_id, sha):
    with conn.cursor() as cur, pytest.raises(oracledb.DatabaseError):
        cur.execute("update ccctx_repos set indexed_commit = :c where id = :i", c=sha, i=repo_id)
    conn.rollback()


GRAPH_TABLES = {"CCCTX_CALL_REFS", "CCCTX_IMPORT_REFS", "CCCTX_CALLS", "CCCTX_FILE_CALLS", "CCCTX_IMPORTS"}


def test_code_graph_tables_and_extract_version(conn):
    with conn.cursor() as cur:
        cur.execute("select name from ccctx_migrations where version = 6")
        assert cur.fetchone() == ("code graph",)
        cur.execute("select table_name from user_tables")
        assert GRAPH_TABLES <= {row[0] for row in cur}
        cur.execute("select data_type, nullable from user_tab_columns"
                    " where table_name = 'CCCTX_REPOS' and column_name = 'EXTRACT_VERSION'")
        assert cur.fetchone() == ("NUMBER", "Y")


def test_call_ref_form_rejects_other_values(conn, repo_id):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        for form in ("bare", "attr"):
            cur.execute("insert into ccctx_call_refs (file_id, name, form, line) values (:f, 'g', :m, 1)", f=file_id, m=form)
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_CALL_REFS_FORM_CK"):
            cur.execute("insert into ccctx_call_refs (file_id, name, form, line) values (:f, 'g', 'call', 1)", f=file_id)
    conn.rollback()


def test_import_pair_unique(conn, repo_id):
    with conn.cursor() as cur:
        a, b = add_file(cur, repo_id, "a.py"), add_file(cur, repo_id, "b.py")
        cur.execute("insert into ccctx_imports (importer_id, imported_id) values (:a, :b)", a=a, b=b)
        cur.execute("insert into ccctx_imports (importer_id, imported_id) values (:b, :a)", a=a, b=b)  # reverse is a new pair
        with pytest.raises(oracledb.IntegrityError, match="CCCTX_IMPORTS_PAIR_UK"):
            cur.execute("insert into ccctx_imports (importer_id, imported_id) values (:a, :b)", a=a, b=b)
    conn.rollback()


def test_file_delete_cascades_refs_and_edges(conn, repo_id):
    with conn.cursor() as cur:
        f1, f2 = add_file(cur, repo_id, "a.py"), add_file(cur, repo_id, "b.py")
        s1, s3 = add_symbol(cur, f1), add_symbol(cur, f2)
        cur.execute("insert into ccctx_call_refs (file_id, caller_id, name, form, receiver, line)"
                    " values (:f, :s, 'g', 'attr', 'self', 2)", f=f1, s=s1)
        cur.execute("insert into ccctx_import_refs (file_id, module, rel_level, bound) values (:f, 'b', 1, 'b')", f=f1)
        for caller, callee in ((s1, s3), (s3, s1)):
            cur.execute("insert into ccctx_calls (caller_id, callee_id, line) values (:a, :b, 1)", a=caller, b=callee)
        for file_id, callee in ((f1, s3), (f2, s1)):
            cur.execute("insert into ccctx_file_calls (file_id, callee_id, line) values (:f, :s, 1)", f=file_id, s=callee)
        for a, b in ((f1, f2), (f2, f1)):
            cur.execute("insert into ccctx_imports (importer_id, imported_id) values (:a, :b)", a=a, b=b)
        cur.execute("delete from ccctx_files where id = :i", i=f1)
        left = {
            table: cur.execute(f"select count(*) from {table} where {where}", ids).fetchone()[0]
            for table, where, ids in (
                ("ccctx_symbols", "id = :s", {"s": s1}),
                ("ccctx_call_refs", "file_id = :f", {"f": f1}),
                ("ccctx_import_refs", "file_id = :f", {"f": f1}),
                ("ccctx_calls", "caller_id = :s or callee_id = :s", {"s": s1}),
                ("ccctx_file_calls", "file_id = :f or callee_id = :s", {"f": f1, "s": s1}),
                ("ccctx_imports", "importer_id = :f or imported_id = :f", {"f": f1}),
            )
        }
        assert left == dict.fromkeys(left, 0)
        cur.execute("select count(*) from ccctx_symbols where id = :s", s=s3)
        assert cur.fetchone()[0] == 1  # the other file's symbol survives
    conn.rollback()


def test_graph_one_hop_calls(conn, repo_id):
    with conn.cursor() as cur:
        cur.execute("select count(*) from user_property_graphs where graph_name = 'CCCTX_GRAPH'")
        assert cur.fetchone()[0] == 1
        file_id = add_file(cur, repo_id)
        caller, callee = add_symbol(cur, file_id, start=1, end=2), add_symbol(cur, file_id, start=4, end=5)
        cur.execute("insert into ccctx_calls (caller_id, callee_id, line) values (:a, :b, 2)", a=caller, b=callee)
        rows = cur.execute(
            "select caller_id, line, callee_id from graph_table (ccctx_graph"
            " match (c is symbol) -[e is calls]-> (t is symbol) where t.id = :t"
            " columns (c.id as caller_id, e.line as line, t.id as callee_id))", t=callee,
        ).fetchall()
    conn.rollback()
    assert rows == [(caller, 2, callee)]
