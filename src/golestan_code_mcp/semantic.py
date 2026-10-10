"""Semantic search: the in-database embedding model, the vector index and the nearest-neighbour query.

The database computes every embedding with VECTOR_EMBEDDING. Only the published model file is downloaded,
once; no source code or question text ever leaves the database to be embedded.
"""

import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import oracledb

from golestan_code_mcp import db
from golestan_code_mcp.config import model_cache, model_url
from golestan_code_mcp.registry import repo_id

MODEL = "ALL_MINILM_L12_V2"
DIMENSION = 384
LIMIT = 10
FLOOR = 0.3
INDEX = "CCCTX_SYMBOLS_EMBEDDING_IX"
MODEL_FILE = "all_MiniLM_L12_v2.onnx"
METADATA = '{"function":"embedding","embeddingOutput":"embedding","input":{"input":["DATA"]}}'


class ModelError(ValueError):
    """The embedding model is missing or could not be set up. The message is safe to show the user."""


@dataclass(frozen=True)
class Result:
    rank: int
    similarity: float
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


def _state(cur) -> tuple[bool, int | None, bool]:
    """(model loaded, recorded dimension, index exists)."""
    loaded = cur.execute("select count(*) from user_mining_models where model_name = :m", m=MODEL).fetchone()[0]
    row = cur.execute("select dimension from ccctx_embedding_model where model_name = :m", m=MODEL).fetchone()
    index = cur.execute("select count(*) from user_indexes where index_name = :i", i=INDEX).fetchone()[0]
    return bool(loaded), row[0] if row else None, bool(index)


def require_model(conn) -> None:
    with conn.cursor() as cur:
        loaded, dimension, index = _state(cur)
    if dimension not in (None, DIMENSION):
        raise ModelError(f"model {MODEL} has {dimension} dimensions, expected {DIMENSION}")
    if not (loaded and dimension and index):
        raise ModelError(f"embedding model {MODEL} is not set up; run `ccctx model load`")


def fetch_model(env: Mapping[str, str]) -> Path:
    """The model file in the local cache, downloaded once. A partial download never counts as cached."""
    dest = model_cache(env) / MODEL_FILE
    if dest.is_file() and dest.stat().st_size:
        return dest
    url = model_url(env)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
    except (OSError, ValueError) as e:
        part.unlink(missing_ok=True)
        raise ModelError(f"could not download the model from {url}: {e}") from None
    part.replace(dest)
    return dest


def load_model(conn, env: Mapping[str, str]) -> str:
    """Load, record and index the model, skipping whatever is already done. Returns a status line."""
    with conn.cursor() as cur:
        loaded, dimension, index = _state(cur)
        if loaded and dimension == DIMENSION and index:
            return f"model {MODEL} already present ({DIMENSION} dimensions)"
        if not loaded:
            data = fetch_model(env).read_bytes()
            cur.setinputsizes(b=oracledb.DB_TYPE_BLOB)
            cur.execute("begin dbms_vector.load_onnx_model(:m, :b, json(:j)); end;", m=MODEL, b=data, j=METADATA)
        actual = cur.execute(
            f"select vector_dimension_count(vector_embedding({MODEL} using 'hello world' as data)) from dual"
        ).fetchone()[0]
        if actual != DIMENSION:
            raise ModelError(f"model {MODEL} has {actual} dimensions, expected {DIMENSION}")
        if dimension is None:
            cur.execute("insert into ccctx_embedding_model (model_name, dimension) values (:m, :d)",
                        m=MODEL, d=DIMENSION)
            conn.commit()
        if not index:
            try:
                cur.execute(
                    f"create vector index {INDEX} on ccctx_symbols (embedding) ORGANIZATION INMEMORY NEIGHBOR GRAPH"
                    " DISTANCE COSINE WITH TARGET ACCURACY 95"
                )
            except oracledb.DatabaseError as e:
                if getattr(e.args[0], "full_code", "") != "ORA-00955":
                    raise ModelError(f"could not build the vector index: {db.describe(e)}") from None
    return f"model {MODEL} loaded ({DIMENSION} dimensions)"


def search(conn, question: str, repo: str | None = None) -> list[Result]:
    """The closest symbols to the question by cosine distance, at most LIMIT, none below FLOOR."""
    question = (question or "").strip()
    if not question:
        raise ValueError("the question is empty")
    rid = repo_id(conn, repo) if repo else None
    require_model(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""with q as (select vector_embedding({MODEL} using :question as data) v from dual),
                top as (
                  select r.name repo, s.kind, s.name, f.path, s.start_line, s.end_line,
                         1 - vector_distance(s.embedding, q.v, cosine) similarity
                    from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                    join ccctx_repos r on r.id = f.repo_id cross join q
                   where s.embedding is not null and (:r is null or r.id = :r)
                   order by vector_distance(s.embedding, q.v, cosine)
                   fetch approx first {LIMIT} rows only with target accuracy 95)
                select repo, kind, name, path, start_line, end_line, similarity
                  from top where similarity >= :floor order by similarity desc""",
            question=question, r=rid, floor=FLOOR,
        )
        rows = cur.fetchall()
    return [Result(i, float(sim), repo_, kind, name, path, start, end)
            for i, (repo_, kind, name, path, start, end, sim) in enumerate(rows, 1)]
