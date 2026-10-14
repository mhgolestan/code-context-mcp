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


WIKI_TABLES = {"CCCTX_WIKI_PAGES", "CCCTX_WIKI_CITATIONS"}
TABLES |= WIKI_TABLES  # so the re-run test counts the wiki tables too


def columns_of(cur, table):
    cur.execute("select column_name, data_type, nullable, char_length from user_tab_columns where table_name = :t", t=table)
    return {name: (kind, nullable, length) for name, kind, nullable, length in cur}


def test_module_wiki_columns_and_tables(conn):
    with conn.cursor() as cur:
        cur.execute("select name from ccctx_migrations where version = 7")
        assert cur.fetchone() == ("module wiki",)
        assert columns_of(cur, "CCCTX_FILES")["DOC"] == ("VARCHAR2", "Y", 1000)
        assert columns_of(cur, "CCCTX_SYMBOLS")["DOC"] == ("VARCHAR2", "Y", 1000)
        wiki_at = columns_of(cur, "CCCTX_REPOS")["WIKI_AT"]
        assert wiki_at[0] == "TIMESTAMP(6) WITH TIME ZONE" and wiki_at[1] == "Y"
        pages = columns_of(cur, "CCCTX_WIKI_PAGES")
        assert {c: v[1] for c, v in pages.items()} == {
            "ID": "N", "REPO_ID": "N", "DIRECTORY": "N", "NAME": "N", "LANGUAGE": "N", "SYMBOLS": "N", "TEXT": "N",
            "SEARCH_TEXT": "N", "STATE": "N", "COMPILED_AT": "N", "COMPILER": "N", "STALE_REASON": "Y",
        }
        assert pages["TEXT"][0] == pages["SEARCH_TEXT"][0] == "CLOB"
        assert pages["COMPILED_AT"][0] == "TIMESTAMP(6) WITH TIME ZONE"
        citations = columns_of(cur, "CCCTX_WIKI_CITATIONS")
        assert {c: v[1] for c, v in citations.items()} == {
            "ID": "N", "PAGE_ID": "N", "SYMBOL_ID": "Y", "FILE_ID": "Y", "ROLE": "N", "NAME": "N", "KIND": "N",
            "COMMIT_SHA": "N", "CONTENT_HASH": "N",
        }
        cur.execute("select column_name from user_ind_columns where table_name = 'CCCTX_WIKI_CITATIONS'"
                    " and column_position = 1")
        assert {"PAGE_ID", "SYMBOL_ID", "FILE_ID"} <= {row[0] for row in cur}


def add_page(cur, repo_id, directory="pkg", name=None, state="fresh", symbols=10):
    return insert(
        cur,
        "insert into ccctx_wiki_pages (repo_id, directory, name, language, symbols, text, search_text, state,"
        " compiled_at, compiler) values (:r, :d, :n, 'python', :s, 'text', 'search', :st, systimestamp, 'ccctx-wiki/1')",
        r=repo_id, d=directory, n=name or directory, s=symbols, st=state,
    )


def add_citation(cur, page_id, role, symbol_id=None, file_id=None):
    return insert(
        cur,
        "insert into ccctx_wiki_citations (page_id, symbol_id, file_id, role, name, kind, commit_sha, content_hash)"
        " values (:p, :s, :f, :r, 'f', 'function', :c, :h)",
        p=page_id, s=symbol_id, f=file_id, r=role, c=SHA, h="0" * 64,
    )


def test_wiki_page_accepts_good_row(conn, repo_id):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        page_id = add_page(cur, repo_id, state="stale")
        add_citation(cur, page_id, "key", symbol_id=add_symbol(cur, file_id))
        add_citation(cur, page_id, "calls", symbol_id=add_symbol(cur, file_id, start=4, end=5))
        for role in ("file", "depends", "uses"):
            add_citation(cur, page_id, role, file_id=file_id)
    conn.rollback()


@pytest.mark.parametrize(("change", "constraint"), [
    ({"state": "old"}, "CCCTX_WIKI_PAGES_STATE_CK"),
    ({"symbols": 9}, "CCCTX_WIKI_PAGES_SYMBOLS_CK"),
])
def test_wiki_page_rejects_bad_values(conn, repo_id, change, constraint):
    with conn.cursor() as cur, pytest.raises(oracledb.DatabaseError, match=constraint):
        add_page(cur, repo_id, **change)
    conn.rollback()


@pytest.mark.parametrize(("second", "constraint"), [
    ({"directory": "pkg", "name": "other"}, "CCCTX_WIKI_PAGES_DIR_UK"),
    ({"directory": "src/pkg", "name": "pkg"}, "CCCTX_WIKI_PAGES_NAME_UK"),
])
def test_wiki_page_directory_and_name_unique_per_repo(conn, repo_id, second, constraint):
    with conn.cursor() as cur:
        add_page(cur, repo_id, "pkg", "pkg")
        with pytest.raises(oracledb.IntegrityError, match=constraint):
            add_page(cur, repo_id, **second)
    conn.rollback()


def test_wiki_citation_rejects_bad_role_and_key_with_file(conn, repo_id):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        symbol_id = add_symbol(cur, file_id)
        page_id = add_page(cur, repo_id)
        cur.execute("select count(*) from user_constraints where constraint_name = 'CCCTX_WIKI_CITATIONS_ROLE_CK'")
        assert cur.fetchone()[0] == 1
        # an unknown role fails the target check too; which check Oracle reports first is not defined
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_WIKI_CITATIONS_(ROLE|TARGET)_CK"):
            add_citation(cur, page_id, "owner", file_id=file_id)
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_WIKI_CITATIONS_TARGET_CK"):
            add_citation(cur, page_id, "key", symbol_id=symbol_id, file_id=file_id)
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_WIKI_CITATIONS_TARGET_CK"):
            add_citation(cur, page_id, "file", symbol_id=symbol_id, file_id=file_id)
    conn.rollback()


def test_wiki_citation_keys_set_null_on_delete(conn, repo_id):
    with conn.cursor() as cur:
        f1, f2 = add_file(cur, repo_id, "a.py"), add_file(cur, repo_id, "b.py")
        symbol_id = add_symbol(cur, f2)
        page_id = add_page(cur, repo_id)
        key = add_citation(cur, page_id, "key", symbol_id=symbol_id)
        file_cit = add_citation(cur, page_id, "file", file_id=f1)
        cur.execute("delete from ccctx_symbols where id = :i", i=symbol_id)
        cur.execute("delete from ccctx_files where id = :i", i=f1)
        rows = cur.execute("select id, symbol_id, file_id from ccctx_wiki_citations where page_id = :p order by id",
                           p=page_id).fetchall()
    conn.rollback()
    assert rows == sorted([(key, None, None), (file_cit, None, None)])


def test_wiki_page_and_repo_delete_cascade(conn, repo_id):
    with conn.cursor() as cur:
        file_id = add_file(cur, repo_id)
        page_id = add_page(cur, repo_id)
        add_citation(cur, page_id, "file", file_id=file_id)
        cur.execute("delete from ccctx_wiki_pages where id = :i", i=page_id)
        assert cur.execute("select count(*) from ccctx_wiki_citations where page_id = :p", p=page_id).fetchone()[0] == 0
        page_id = add_page(cur, repo_id)
        add_citation(cur, page_id, "file", file_id=file_id)
        cur.execute("delete from ccctx_repos where id = :i", i=repo_id)
        assert cur.execute("select count(*) from ccctx_wiki_pages where repo_id = :r", r=repo_id).fetchone()[0] == 0
        assert cur.execute("select count(*) from ccctx_wiki_citations where page_id = :p", p=page_id).fetchone()[0] == 0
    conn.rollback()


MEMORY_TABLES = {"CCCTX_MEMORY_RUNS", "CCCTX_MEMORY_RUN_REPOS"}
TABLES |= MEMORY_TABLES


def add_run(cur, tool="search_code", status="ok"):
    return insert(cur, "insert into ccctx_memory_runs (tool, arguments, status, ms, result_count, results)"
                       " values (:t, json('{}'), :s, 1, 0, json('[]'))", t=tool, s=status)


def test_agent_memory_tables_and_package_schema(conn):
    with conn.cursor() as cur:
        cur.execute("select name from ccctx_migrations where version = 8")
        assert cur.fetchone() == ("agent memory",)
        runs = columns_of(cur, "CCCTX_MEMORY_RUNS")
        assert {c: v[1] for c, v in runs.items()} == {
            "ID": "N", "CALLED_AT": "N", "TOOL": "N", "ARGUMENTS": "N", "QUESTION": "Y", "CLIENT": "Y", "STATUS": "N",
            "REASON": "Y", "MS": "Y", "RESULT_COUNT": "N", "RESULTS": "N", "SCOPED": "N",
        }
        assert runs["ARGUMENTS"][0] == runs["RESULTS"][0] == "JSON"
        assert runs["SCOPED"][0] == "BOOLEAN"
        links = columns_of(cur, "CCCTX_MEMORY_RUN_REPOS")
        assert {c: v[1] for c, v in links.items()} == {"RUN_ID": "N", "REPO_ID": "N", "CONSOLIDATED": "N"}
        cur.execute("select column_name from user_ind_columns where table_name = 'CCCTX_MEMORY_RUN_REPOS'"
                    " and column_position = 1")
        assert "REPO_ID" in {row[0] for row in cur}
        cur.execute("select table_name from user_tables where table_name like 'CCCTX_MEM\\_%' escape '\\'")
        assert "CCCTX_MEM_MEMORY" in {row[0] for row in cur}  # the package's tables come from this migration


def test_memory_run_defaults_and_status_check(conn, repo_id):
    with conn.cursor() as cur:
        run_id = add_run(cur)
        cur.execute("insert into ccctx_memory_run_repos (run_id, repo_id) values (:r, :p)", r=run_id, p=repo_id)
        called, scoped = cur.execute("select called_at, scoped from ccctx_memory_runs where id = :i",
                                     i=run_id).fetchone()
        assert called is not None and scoped is False
        assert cur.execute("select consolidated from ccctx_memory_run_repos where run_id = :i",
                           i=run_id).fetchone() == (False,)
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_MEMORY_RUNS_STATUS_CK"):
            add_run(cur, status="done")
        with pytest.raises(oracledb.IntegrityError):
            cur.execute("insert into ccctx_memory_run_repos (run_id, repo_id) values (:r, :p)", r=run_id, p=repo_id)
    conn.rollback()


def test_memory_run_and_repo_delete_cascade_links(conn, repo_id):
    with conn.cursor() as cur:
        kept, gone = add_run(cur), add_run(cur)
        for run_id in (kept, gone):
            cur.execute("insert into ccctx_memory_run_repos (run_id, repo_id) values (:r, :p)", r=run_id, p=repo_id)
        cur.execute("delete from ccctx_memory_runs where id = :i", i=gone)
        assert cur.execute("select count(*) from ccctx_memory_run_repos where run_id = :i", i=gone).fetchone()[0] == 0
        cur.execute("delete from ccctx_repos where id = :i", i=repo_id)
        assert cur.execute("select count(*) from ccctx_memory_run_repos where run_id = :i", i=kept).fetchone()[0] == 0
        assert cur.execute("select count(*) from ccctx_memory_runs where id = :i", i=kept).fetchone()[0] == 1
    conn.rollback()


# --- usage feedback (specs/011-usage-feedback/data-model.md) ---

TABLES |= {"CCCTX_FEEDBACK"}


def add_use(cur, repo_id, how="cited", run_id=None):
    return insert(cur, "insert into ccctx_feedback (repo_id, path, name, kind, how, client, session_id, run_id)"
                       " values (:r, 'a.py', 'f', 'function', :h, 'c', 's', :run)", r=repo_id, h=how, run=run_id)


def test_usage_feedback_table(conn):
    with conn.cursor() as cur:
        cur.execute("select name from ccctx_migrations where version = 9")
        assert cur.fetchone() == ("usage feedback",)
        uses = columns_of(cur, "CCCTX_FEEDBACK")
        assert {c: v[1] for c, v in uses.items()} == {
            "ID": "N", "REPO_ID": "N", "PATH": "N", "NAME": "N", "KIND": "N", "HOW": "N", "CLIENT": "Y",
            "SESSION_ID": "N", "RUN_ID": "Y", "USED_AT": "N",
        }
        assert uses["USED_AT"][0].startswith("TIMESTAMP")
        cur.execute("select column_name from user_ind_columns where index_name = 'CCCTX_FEEDBACK_RESULT_IX'"
                    " order by column_position")
        assert [row[0] for row in cur] == ["REPO_ID", "NAME"]


def test_use_how_check_and_default_time(conn, repo_id):
    with conn.cursor() as cur:
        use = add_use(cur, repo_id, "opened")
        assert cur.execute("select used_at from ccctx_feedback where id = :i", i=use).fetchone()[0] is not None
        with pytest.raises(oracledb.DatabaseError, match="CCCTX_FEEDBACK_HOW_CK"):
            add_use(cur, repo_id, "liked")
    conn.rollback()


def test_uses_go_with_their_repository_and_outlive_their_run(conn, repo_id):
    with conn.cursor() as cur:
        run_id = add_run(cur)
        use = add_use(cur, repo_id, run_id=run_id)
        cur.execute("delete from ccctx_memory_runs where id = :i", i=run_id)
        assert cur.execute("select run_id from ccctx_feedback where id = :i", i=use).fetchone() == (None,)
        cur.execute("delete from ccctx_repos where id = :i", i=repo_id)
        assert cur.execute("select count(*) from ccctx_feedback where id = :i", i=use).fetchone()[0] == 0
    conn.rollback()


# --- language coverage (specs/012-language-coverage/data-model.md) ---


def test_language_coverage_columns(conn):
    with conn.cursor() as cur:
        assert cur.execute("select name from ccctx_migrations where version = 10").fetchone() == ("language coverage",)
        assert columns_of(cur, "CCCTX_FILES")["NAMESPACES"][:2] == ("JSON", "Y")
        assert columns_of(cur, "CCCTX_SYMBOLS")["EXPORTED"][:2] == ("BOOLEAN", "Y")


def test_exported_backfill_follows_the_name_rule(conn, repo_id):
    with conn.cursor() as cur:
        go = insert(cur, "insert into ccctx_files (repo_id, path, language, content_hash) values (:r, 'a.go', 'go', :h)",
                    r=repo_id, h="0" * 64)
        py = add_file(cur, repo_id, "b.py")
        rows = {}
        for fid, name, kind in [(go, "Serve", "function"), (go, "Server.close", "method"), (py, "Flask._private",
                                "method"), (py, "url_for", "function"), (py, "b.py:1-60", "chunk")]:
            rows[name] = insert(cur, "insert into ccctx_symbols (file_id, name, kind, start_line, end_line, source,"
                                     " content_hash, commit_sha) values (:f, :n, :k, 1, 2, 'x', :h, :c)",
                                f=fid, n=name, k=kind, h="0" * 64, c=SHA)
        update = next(s for v, n, steps in schema.MIGRATIONS if v == 10 for s in steps
                      if isinstance(s, str) and s.lstrip().startswith("update"))
        cur.execute(update)
        got = {n: cur.execute("select exported from ccctx_symbols where id = :i", i=i).fetchone()[0]
               for n, i in rows.items()}
        assert got == {"Serve": True, "Server.close": False, "Flask._private": False, "url_for": True,
                       "b.py:1-60": None}
        cur.execute("update ccctx_symbols set exported = true where id = :i", i=rows["Flask._private"])
        cur.execute(update)  # only null rows are touched
        assert cur.execute("select exported from ccctx_symbols where id = :i",
                           i=rows["Flask._private"]).fetchone()[0] is True
    conn.rollback()


def test_schema_version_is_ten(conn):
    assert schema.version(conn) == 10
