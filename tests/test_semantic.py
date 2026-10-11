"""Hybrid search core (contracts/schema.md, contracts/cli.md): model setup, the hybrid index, modes, scoping.

Tests that need the database skip when it is down; tests that search skip when the hybrid index is not built.
Every search names the throwaway repositories, so other indexed code never leaks into the assertions.
"""

import uuid

import pytest

from golestan_code_mcp import db, registry, schema, semantic
from golestan_code_mcp.config import load

RUN = uuid.uuid4().hex[:6]
A, B = f"t-{RUN}a", f"t-{RUN}b"

SYMBOLS = {
    A: [
        ("url_for", "function", "pkg/helpers.py",
         "def url_for(endpoint, **values):\n    # build a URL for the given endpoint\n    return current_app.url_for(endpoint, **values)\n"),
        ("App.url_for", "method", "pkg/app.py",
         "def url_for(self, endpoint, **values):\n    # generate a URL to the endpoint with the given values\n    return self.url_map.build(endpoint, values)\n"),
        ("redirect_home", "function", "pkg/views.py",
         "def redirect_home():\n    # send the user back to the start page\n    return redirect(url_for('index'))\n"),
        ("route_request", "function", "pkg/routing.py",
         "def route_request(path):\n    # match the incoming URL path against registered routes and dispatch to its handler\n"
         "    return ROUTES[path]()\n"),
        ("bake_bread", "function", "pkg/kitchen.py",
         "def bake_bread():\n    # mix flour, water, yeast and salt, knead the dough and bake the loaf\n    return oven.bake(dough)\n"),
    ],
    B: [
        ("Engine.Use", "method", "gin.go",
         "func (engine *Engine) Use(middleware ...HandlerFunc) IRoutes {\n\t// attach global middleware\n"
         "\tengine.RouterGroup.Use(middleware...)\n\treturn engine\n}\n"),
        ("handleChain", "function", "context.go",
         "func handleChain(c *Context) {\n\t// run each middleware in the chain for the request, then the handler\n"
         "\tfor c.index < len(c.handlers) { c.handlers[c.index](c); c.index++ }\n}\n"),
    ],
}

LITERALS = ["a_b", "x-y", "(foo)", "c&d", "50%", '"quoted"', "{brace}", "back\\slash", "a | b", "NOT near"]


# --- no database ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_refused(question):
    with pytest.raises(ValueError, match="question"):
        semantic.search(None, question)


def test_unknown_mode_refused_before_searching():
    with pytest.raises(ValueError, match="valid modes: vector, text, hybrid"):
        semantic.search(None, "url_for", mode="bogus")


@pytest.mark.parametrize(("question", "expected"), [
    ("url_for", "{url_for}"),
    ("middleware chain", "{middleware} AND {chain}"),
    ("{x}", "{x}"),
    ("a\\b", "{ab}"),
    ("{} \\", ""),
])
def test_text_query_escapes_every_word(question, expected):
    assert semantic.text_query(question) == expected


@pytest.mark.parametrize(("question", "expected"), [
    ("requests routed to handlers", "$requests ACCUM $routed ACCUM $to ACCUM $handlers"),
    ("url_for redirect", "{url_for} ACCUM $redirect"),
    ("c&d (x)", "{c&d} ACCUM {(x)}"),
])
def test_hybrid_keyword_side_accepts_any_word(question, expected):
    assert semantic.text_query(question, every_word=False) == expected


def test_failed_download_names_the_url_and_leaves_nothing(tmp_path):
    url = "http://127.0.0.1:1/all_MiniLM_L12_v2.onnx"
    with pytest.raises(semantic.ModelError, match="127.0.0.1:1"):
        semantic.fetch_model({"CCCTX_MODEL_URL": url, "CCCTX_MODEL_CACHE": str(tmp_path)})
    assert list(tmp_path.iterdir()) == []


# --- database and hybrid index ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conn():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    schema.migrate(conn)
    try:
        semantic.require_model(conn)
    except semantic.ModelError:
        conn.close()
        pytest.skip("hybrid index not built: run `ccctx model load`")
    yield conn
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()
    conn.close()


def add_symbol(cur, file_id, name, kind, path, source):
    cur.setinputsizes(t=semantic.oracledb.DB_TYPE_CLOB, s=semantic.oracledb.DB_TYPE_CLOB)
    cur.execute(
        """insert into ccctx_symbols (file_id, name, kind, start_line, end_line, content_hash, commit_sha,
                                      search_text, source)
           values (:f, :n, :k, 1, :e, :h, :c, :t, :s)""",
        f=file_id, n=name, k=kind, e=source.count("\n"), h="0" * 64, c="a" * 40,
        t=f"{kind} {name} in {path}\n{source}", s=source,
    )


@pytest.fixture(scope="module")
def repos(conn):
    """Two throwaway repositories whose symbols the hybrid index picks up on commit."""
    with conn.cursor() as cur:
        for repo, symbols in SYMBOLS.items():
            rid = cur.var(int)
            cur.execute("insert into ccctx_repos (name, path) values (:n, :p) returning id into :i",
                        n=repo, p=f"/tmp/{repo}", i=rid)
            for name, kind, path, source in symbols:
                fid = cur.var(int)
                cur.execute("insert into ccctx_files (repo_id, path, language, content_hash)"
                            " values (:r, :p, 'python', :h) returning id into :i",
                            r=rid.getvalue()[0], p=path, h="0" * 64, i=fid)
                add_symbol(cur, fid.getvalue()[0], name, kind, path, source)
    conn.commit()
    return A, B


def names(results):
    return [r.name for r in results]


def test_hybrid_index_is_set_up(conn):
    with conn.cursor() as cur:
        assert ("ALL_MINILM_L12_V2",) in cur.execute("SELECT model_name FROM user_mining_models").fetchall()
        v = cur.execute("SELECT VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING 'hello world' as data) AS v").fetchone()[0]
        assert len(v) == 384
        assert cur.execute("select index_name from user_indexes where index_name = :i", i=semantic.INDEX).fetchone()
        assert cur.execute("select model_name, dimension from ccctx_embedding_model").fetchall() == [
            ("ALL_MINILM_L12_V2", 384)
        ]


def test_load_model_again_changes_nothing(conn):
    # an unreachable URL proves nothing is downloaded when everything is already set up
    status = semantic.load_model(conn, {"CCCTX_MODEL_URL": "http://127.0.0.1:1/x.onnx"})
    assert status == "model ALL_MINILM_L12_V2 already present (384 dimensions)"


def test_require_model_names_the_setup_command(conn, monkeypatch):
    monkeypatch.setattr(semantic, "INDEX", "NOT_AN_INDEX")
    with pytest.raises(semantic.ModelError, match="ccctx model load"):
        semantic.require_model(conn)


def test_definitions_rank_above_call_sites(conn, repos):
    results = semantic.search(conn, "url_for", [A])
    assert set(names(results)[:2]) == {"url_for", "App.url_for"}
    assert "redirect_home" in names(results)[2:]  # a call site: found, but after both definitions
    assert [r.rank for r in results] == list(range(1, len(results) + 1)) and len(results) <= semantic.LIMIT


def test_qualified_name_ranks_that_method_first(conn, repos):
    assert semantic.search(conn, "App.url_for", [A])[0].name == "App.url_for"


def test_case_does_not_matter(conn, repos):
    assert set(names(semantic.search(conn, "URL_FOR", [A]))[:2]) == {"url_for", "App.url_for"}


def test_no_boost_without_a_definition(conn, repos):
    results = semantic.search(conn, "endpoint", [A])  # appears in code, defined nowhere
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_text_mode_needs_every_word(conn, repos):
    results = semantic.search(conn, "middleware chain", [B], mode="text")
    assert names(results) == ["handleChain"]  # Engine.Use has "middleware" but not "chain"
    assert results[0].keyword_rank == 1 and results[0].semantic_rank is None


def test_vector_mode_ranks_by_meaning_above_the_floor(conn, repos):
    results = semantic.search(conn, "how are incoming requests routed to handlers", [A], mode="vector")
    assert results[0].name == "route_request"
    assert all(r.similarity >= semantic.FLOOR and r.keyword_rank is None and r.semantic_rank for r in results)


def test_hybrid_results_explain_their_rank(conn, repos):
    results = semantic.search(conn, "url_for", [A])
    top = results[0]
    assert top.score > 0 and top.keyword_rank and top.semantic_rank
    assert all(r.keyword_rank or r.semantic_rank for r in results)


def test_repository_scoping(conn, repos):
    assert {r.repo for r in semantic.search(conn, "middleware", [B])} == {B}
    both = semantic.search(conn, "url for middleware", [A, B, A])
    assert {r.repo for r in both} <= {A, B} and both


def test_unknown_repo_refused(conn):
    with pytest.raises(registry.RegistryError, match="no repository named 't-missing'"):
        semantic.search(conn, "routing", [A, "t-missing"])


@pytest.mark.parametrize("question", LITERALS)
def test_special_characters_are_literal(conn, repos, question):
    for mode in semantic.MODES:
        semantic.search(conn, question, [A], mode=mode)  # must not raise


def test_fresh_commit_is_searchable_at_once(conn, repos):
    token = f"zebra_{RUN}"
    with conn.cursor() as cur:
        fid = cur.execute(
            "select f.id from ccctx_files f join ccctx_repos r on r.id = f.repo_id where r.name = :n and rownum = 1",
            n=A).fetchone()[0]
        add_symbol(cur, fid, token, "function", "pkg/fresh.py", f"def {token}():\n    pass\n")
    conn.commit()
    assert semantic.search(conn, token, [A], mode="text")[0].name == token  # SYNC (ON COMMIT)


def test_operator_words_stay_literal():
    assert semantic.text_query("NOT near and", every_word=False) == "{NOT} ACCUM {near} ACCUM {and}"
