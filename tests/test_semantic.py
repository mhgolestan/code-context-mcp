"""Hybrid search core (contracts/schema.md, contracts/cli.md, 008 contracts/tools.md `search_code`): model setup,
the hybrid index, modes, scoping by repository and language, the result count and snippets.

Tests that need the database skip when it is down; tests that search skip when the hybrid index is not built.
Every search names the throwaway repositories, so other indexed code never leaks into the assertions, and the
teardown removes only this run's repositories, so parallel test runs keep theirs.
"""

import re
import uuid

import pytest

from golestan_code_mcp import chunker, db, registry, schema, semantic
from golestan_code_mcp.config import load

RUN = uuid.uuid4().hex[:6]
A, B, C = f"t-{RUN}a", f"t-{RUN}b", f"t-{RUN}c"
TOKEN = f"quokka_{RUN}"  # appears in every symbol of C and nowhere else
LANGUAGE_LIST = "batch, css, go, html, ini, javascript, json, markdown, proto, python, rst, shell, sql, text, toml, yaml"

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
    C: [  # 4 Go and 6 Python symbols sharing TOKEN; the language comes from the path, as the chunker assigns it
        *[(f"Quokka{i}", "function", f"q{i}.go", f"func Quokka{i}() {{\n\t// {TOKEN}\n}}\n") for i in range(4)],
        *[(f"quokka_py{i}", "function", f"q{i}.py", f"def quokka_py{i}():\n    # {TOKEN}\n    pass\n")
          for i in range(4)],
        ("tall", "function", "tall.py", "def tall():\n" + "".join(f"    x{n} = '{TOKEN}'\n" for n in range(30))),
        ("wide", "function", "wide.py", f"def wide():\n    return '{TOKEN}' + '" + "w" * 1000 + "'\n"),
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


@pytest.mark.parametrize("limit", [0, 51, -1, 2.5, "3"])
def test_limit_outside_1_to_50_refused(limit):
    with pytest.raises(ValueError, match="^limit must be between 1 and 50$"):
        semantic.check_limit(limit)
    with pytest.raises(ValueError, match="^limit must be between 1 and 50$"):
        semantic.search(None, "url_for", limit=limit)


@pytest.mark.parametrize("limit", [1, 10, 50])
def test_limit_inside_1_to_50_accepted(limit):
    semantic.check_limit(limit)  # must not raise


def test_languages_are_the_chunkers_plus_text():
    assert sorted(semantic.LANGUAGES) == sorted({*chunker.EXTENSIONS.values(), "text"})
    assert ", ".join(sorted(semantic.LANGUAGES)) == LANGUAGE_LIST


def test_unknown_language_refused_before_connecting(monkeypatch):
    monkeypatch.setenv("DB_PORT", "1")  # no connection is made: conn is None and the port is closed
    reason = "^" + re.escape(f"unknown language 'cobol'; valid languages: {LANGUAGE_LIST}") + "$"
    with pytest.raises(ValueError, match=reason):
        semantic.check_language("cobol")
    with pytest.raises(ValueError, match=reason):
        semantic.search(None, "url_for", language="cobol")
    for language in ("python", "go", "text"):
        semantic.check_language(language)  # must not raise


def test_question_then_limit_then_language_checked():
    with pytest.raises(ValueError, match="question"):
        semantic.search(None, "  ", language="cobol", limit=0)
    with pytest.raises(ValueError, match="limit"):
        semantic.search(None, "url_for", language="cobol", limit=0)


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
        cur.execute("delete from ccctx_repos where name like :p", p=f"t-{RUN}%")
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
    """Three throwaway repositories whose symbols the hybrid index picks up on commit."""
    with conn.cursor() as cur:
        for repo, symbols in SYMBOLS.items():
            rid = cur.var(int)
            cur.execute("insert into ccctx_repos (name, path) values (:n, :p) returning id into :i",
                        n=repo, p=f"/tmp/{repo}", i=rid)
            for name, kind, path, source in symbols:
                fid = cur.var(int)
                cur.execute("insert into ccctx_files (repo_id, path, language, content_hash)"
                            " values (:r, :p, :l, :h) returning id into :i",
                            r=rid.getvalue()[0], p=path, l=chunker.detect_language(path), h="0" * 64, i=fid)
                add_symbol(cur, fid.getvalue()[0], name, kind, path, source)
    conn.commit()
    return A, B, C


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


# --- language, limit and snippets (008 contracts/tools.md `search_code`) ----------------------------------


def opening(source: str) -> str:
    """The contract's snippet: the source's first 10 lines, cut to 800 characters, no trailing newline."""
    return "\n".join(source[:800].split("\n")[:10]).rstrip("\n")


def in_language(repo, language):
    return [name for name, _, path, _ in SYMBOLS[repo] if chunker.detect_language(path) == language]


@pytest.mark.parametrize("language", ["python", "go"])
def test_language_keeps_only_that_language(conn, repos, language):
    results = semantic.search(conn, TOKEN, [C], language=language, limit=50)
    assert sorted(names(results)) == sorted(in_language(C, language))
    assert {r.language for r in results} == {language}


def test_repo_and_language_still_fill_limit(conn, repos):
    # a 2-hit pool holds out-of-scope hits; widening must go on until `limit` in-scope hits are found
    results = semantic.search(conn, TOKEN, [C], language="go", limit=4, env={"CCCTX_SEARCH_CANDIDATES": "2"})
    assert sorted(names(results)) == sorted(in_language(C, "go"))


def test_limit_caps_the_hits(conn, repos):
    results = semantic.search(conn, TOKEN, [C], limit=3)
    assert len(results) == 3 and [r.rank for r in results] == [1, 2, 3]


def test_every_hit_has_its_language_and_opening_snippet(conn, repos):
    sources = {name: source for name, _, _, source in SYMBOLS[C]}
    results = semantic.search(conn, TOKEN, [C], limit=50)
    assert sorted(names(results)) == sorted(sources)
    for r in results:
        assert r.language == chunker.detect_language(r.path)
        assert r.snippet == opening(sources[r.name])
        assert len(r.snippet) <= 800 and r.snippet.count("\n") <= 9 and not r.snippet.endswith("\n")
    by_name = {r.name: r for r in results}
    assert by_name["tall"].snippet.count("\n") == 9  # 31 lines stored, 10 shown
    assert by_name["wide"].snippet == sources["wide"][:800]  # a 1000-character second line, cut at 800 characters


def test_defaults_give_the_same_hits_as_before(conn, repos):
    def key(results):
        return [(r.rank, r.name, r.score, r.keyword_rank, r.semantic_rank, r.similarity) for r in results]

    before = semantic.search(conn, "url_for", [A])
    assert key(semantic.search(conn, "url_for", [A], language=None, limit=semantic.LIMIT)) == key(before)
    assert key(semantic.search(conn, "url_for", [A], language="python")) == key(before)  # all of A is Python
