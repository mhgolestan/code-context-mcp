"""Semantic core (contracts/schema.md, contracts/cli.md): model setup, embeddings and search.

Tests that need the database skip when it is down; tests that need embeddings skip when the model is not loaded.
"""

import uuid

import pytest

from golestan_code_mcp import db, registry, schema, semantic
from golestan_code_mcp.config import load

SYMBOLS = {
    "route_request": "def route_request(path):\n    # match the URL path against the registered routes\n"
                     "    # and dispatch it to the matching handler function\n    return ROUTES[path]()\n",
    "bake_bread": "def bake_bread():\n    # mix flour, water, yeast and salt, knead the dough,\n"
                  "    # let it rise overnight and bake the loaf\n    return oven.bake(dough)\n",
    "parse_settings": "def parse_settings(text):\n    # read key = value lines into a dict\n"
                      "    return dict(line.split('=') for line in text.splitlines())\n",
}


# --- no database ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_refused(question):
    with pytest.raises(ValueError, match="question"):
        semantic.search(None, question)


def test_failed_download_names_the_url_and_leaves_nothing(tmp_path):
    url = "http://127.0.0.1:1/all_MiniLM_L12_v2.onnx"
    with pytest.raises(semantic.ModelError, match="127.0.0.1:1"):
        semantic.fetch_model({"CCCTX_MODEL_URL": url, "CCCTX_MODEL_CACHE": str(tmp_path)})
    assert list(tmp_path.iterdir()) == []


# --- database and model -----------------------------------------------------------------------------------


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
        pytest.skip("embedding model not loaded: run `ccctx model load`")
    yield conn
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def repo(conn):
    """A throwaway repository whose symbols are embedded by the database."""
    name = f"t-{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        repo_id = cur.var(int)
        cur.execute("insert into ccctx_repos (name, path) values (:n, :p) returning id into :i",
                    n=name, p=f"/tmp/{name}", i=repo_id)
        file_id = cur.var(int)
        cur.execute("insert into ccctx_files (repo_id, path, language, content_hash)"
                    " values (:r, 'app.py', 'python', :h) returning id into :i",
                    r=repo_id.getvalue()[0], h="0" * 64, i=file_id)
        for symbol, source in SYMBOLS.items():
            cur.execute(
                f"""insert into ccctx_symbols (file_id, name, kind, start_line, end_line, source, content_hash,
                                              commit_sha, embedding)
                    values (:f, :n, 'function', 1, 4, :s, :h, :c,
                            vector_embedding({semantic.MODEL} using :s as data))""",
                f=file_id.getvalue()[0], n=symbol, s=source, h="0" * 64, c="a" * 40,
            )
    conn.commit()
    return name


def test_model_is_loaded_and_recorded(conn):
    with conn.cursor() as cur:
        assert ("ALL_MINILM_L12_V2",) in cur.execute("SELECT model_name FROM user_mining_models").fetchall()
        v = cur.execute("SELECT VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING 'hello world' as data) AS v").fetchone()[0]
        assert len(v) == 384  # SC-001
        assert cur.execute("select model_name, dimension from ccctx_embedding_model").fetchall() == [
            ("ALL_MINILM_L12_V2", 384)
        ]
        assert cur.execute("select index_name from user_indexes where index_name = :i",
                           i=semantic.INDEX).fetchone()


def test_load_model_again_changes_nothing(conn):
    # an unreachable URL proves nothing is downloaded when the model is already set up
    status = semantic.load_model(conn, {"CCCTX_MODEL_URL": "http://127.0.0.1:1/x.onnx"})
    assert status == "model ALL_MINILM_L12_V2 already present (384 dimensions)"


def test_require_model_names_the_setup_command(conn, monkeypatch):
    monkeypatch.setattr(semantic, "MODEL", "NOT_A_MODEL")
    with pytest.raises(semantic.ModelError, match="ccctx model load"):
        semantic.require_model(conn)


def test_embeddings_are_384_float32(conn, repo):
    with conn.cursor() as cur:
        cur.execute(
            """select distinct vector_dimension_count(s.embedding), vector_dimension_format(s.embedding)
                 from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
                where r.name = :n""",
            n=repo,
        )
        assert cur.fetchall() == [(384, "FLOAT32")]


def test_search_ranks_by_meaning(conn, repo):
    results = semantic.search(conn, "how are incoming requests routed to handlers", repo)
    assert results and results[0].name == "route_request"
    assert results[0].repo == repo and results[0].path == "app.py" and (results[0].start_line, results[0].end_line) == (1, 4)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True) and all(s >= semantic.FLOOR for s in sims)  # SC-005
    assert [r.rank for r in results] == list(range(1, len(results) + 1)) and len(results) <= semantic.LIMIT
    assert semantic.search(conn, "recipe for sourdough bread", repo)[0].name == "bake_bread"


def test_floor_drops_weak_matches(conn, repo):
    for r in semantic.search(conn, "orbital mechanics of the outer planets", repo):
        assert r.similarity >= semantic.FLOOR


def test_unknown_repo_refused(conn):
    with pytest.raises(registry.RegistryError, match="no repository named"):
        semantic.search(conn, "routing", "t-missing")
