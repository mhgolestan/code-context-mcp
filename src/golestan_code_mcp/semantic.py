"""Code search: the in-database embedding model, the hybrid vector index and the fused query.

One hybrid vector index over ccctx_symbols.search_text serves both keyword and semantic matching. The database
computes every vector with the ONNX model; only the published model file is downloaded, once, and no source
code or question text ever leaves the database to be embedded.
"""

import json
import re
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import oracledb

from golestan_code_mcp import db
from golestan_code_mcp.config import (
    environment,
    model_cache,
    model_url,
    search_candidates,
    text_weight,
    vector_weight,
)
from golestan_code_mcp.registry import repo_id

MODEL = "ALL_MINILM_L12_V2"
DIMENSION = 384
LIMIT = 10
FLOOR = 0.3
INDEX = "CCCTX_SYMBOLS_HYBRID_IX"
LEXER = "CCCTX_CODE_LEXER"
MODES = ("vector", "text", "hybrid")
MODEL_FILE = "all_MiniLM_L12_v2.onnx"
METADATA = '{"function":"embedding","embeddingOutput":"embedding","input":{"input":["DATA"]}}'
IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*")
MAX_CANDIDATES = 4000  # widest pool a repository filter may ask for
# Oracle Text operator words: never stemmed ($NOT is a syntax error), only ever searched literally in braces
RESERVED = {
    "ABOUT", "ACCUM", "AND", "BT", "BTG", "BTI", "BTP", "EQUIV", "FUZZY", "HASPATH", "INPATH", "MDATA", "MINUS",
    "NEAR", "NOT", "NT", "NTG", "NTI", "NTP", "OR", "PT", "RT", "SQE", "SYN", "TR", "TRSYN", "TT", "WITHIN",
}


class ModelError(ValueError):
    """The embedding model or the hybrid index is missing or could not be set up. Safe to show the user."""


@dataclass(frozen=True)
class Result:
    rank: int
    score: float  # fused score in hybrid mode, the one side's score otherwise (0-100)
    similarity: float | None  # the semantic side's similarity (0-1), None if it did not find this symbol
    keyword_rank: int | None
    semantic_rank: int | None
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


def _state(cur) -> tuple[bool, int | None, bool]:
    """(model loaded, recorded dimension, hybrid index exists)."""
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


def _ensure_lexer(cur) -> None:
    """A text lexer that keeps `_` inside words, so `url_for` is one token and not `url` plus the stopword `for`."""
    if cur.execute("select count(*) from ctx_user_preferences where pre_name = :p", p=LEXER).fetchone()[0]:
        return
    cur.execute(f"""begin
                      ctx_ddl.create_preference('{LEXER}', 'BASIC_LEXER');
                      ctx_ddl.set_attribute('{LEXER}', 'PRINTJOINS', '_');
                    end;""")


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
            _ensure_lexer(cur)
            try:  # SYNC (ON COMMIT): a committed ingest is searchable at once, never half-visible
                cur.execute(
                    f"create hybrid vector index {INDEX} on ccctx_symbols (search_text)"
                    f" parameters ('MODEL {MODEL} VECTOR_IDXTYPE HNSW SYNC (ON COMMIT) LEXER {LEXER}')"
                )
            except oracledb.DatabaseError as e:
                if getattr(e.args[0], "full_code", "") != "ORA-00955":
                    raise ModelError(f"could not build the hybrid index: {db.describe(e)}") from None
    return f"model {MODEL} loaded ({DIMENSION} dimensions)"


def text_query(question: str, every_word: bool = True) -> str:
    """The keyword query, built so the question's characters are never operators. Empty if no word is left.

    every_word (text mode): each word literally, in braces, all required (AND).
    otherwise (hybrid's keyword side): any word, ranked by how many match (ACCUM). A plain alphabetic word
    is stemmed ($routed also finds route), anything else stays literal, so identifiers such as url_for match exactly.
    """
    words = [w for w in (w.replace("{", "").replace("}", "").replace("\\", "") for w in question.split()) if w]
    if every_word:
        return " AND ".join(f"{{{w}}}" for w in words)
    return " ACCUM ".join(f"${w}" if _stemmable(w) else f"{{{w}}}" for w in words)


def _stemmable(word: str) -> bool:
    return word.isascii() and word.isalpha() and word.upper() not in RESERVED


def _hybrid(cur, request: dict) -> list[dict]:
    lob = cur.execute("select dbms_hybrid_vector.search(json(:j)) from dual", j=json.dumps(request)).fetchone()[0]
    return json.loads(lob.read() if hasattr(lob, "read") else lob) if lob else []


def _symbols(cur, rowids: list[str]) -> dict[str, tuple]:
    """rowid -> (repo id, repo, kind, name, path, start, end) for the given symbol rowids."""
    found = {}
    for i in range(0, len(rowids), 500):  # stay well inside Oracle's 1000-item IN list
        chunk = rowids[i:i + 500]
        binds = ", ".join(f"chartorowid(:{n + 1})" for n in range(len(chunk)))
        cur.execute(
            f"""select rowidtochar(s.rowid), r.id, r.name, s.kind, s.name, f.path, s.start_line, s.end_line
                  from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
                 where s.rowid in ({binds})""",
            chunk,
        )
        found.update({row[0]: row[1:] for row in cur})
    return found


def _definitions(cur, name: str) -> dict[str, tuple]:
    """Symbols whose name, or the last part of their dotted name, equals the identifier (any case)."""
    cur.execute(
        """select rowidtochar(s.rowid), r.id, r.name, s.kind, s.name, f.path, s.start_line, s.end_line
             from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
            where s.kind <> 'chunk'
              and (lower(s.name) = :q
                   or (length(s.name) > length(:q) and lower(substr(s.name, -length(:q) - 1)) = '.' || :q))
            order by f.path, s.start_line""",
        q=name.lower(),
    )
    return {row[0]: row[1:] for row in cur}


def search(conn, question: str, repos: list[str] | None = None, mode: str = "hybrid",
           env: Mapping[str, str] | None = None) -> list[Result]:
    """At most LIMIT symbols for the question. Identifier definitions first, then fused order.

    hybrid: keyword and semantic sides fused with RRF (UNION). vector: semantic side only, none below FLOOR.
    text: keyword side only, every word must appear. `repos` narrows to those repositories; None searches all.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("the question is empty")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; valid modes: {', '.join(MODES)}")
    env = environment() if env is None else env
    wanted = {repo_id(conn, name) for name in repos} if repos else None  # unknown names refused before searching
    require_model(conn)

    request: dict = {"hybrid_index_name": INDEX, "return": {"topN": search_candidates(env)}}
    if mode != "text":
        request["vector"] = {"search_text": question, "score_weight": vector_weight(env)}
    keywords = text_query(question, every_word=mode == "text")
    if mode != "vector" and keywords:
        request["text"] = {"contains": keywords, "score_weight": text_weight(env)}
    if "vector" in request and "text" in request:
        request |= {"search_scorer": "rrf", "search_fusion": "UNION"}

    with conn.cursor() as cur:
        hits, symbols = _candidates(cur, request, wanted)
        definitions = _definitions(cur, question) if mode != "vector" and IDENTIFIER.fullmatch(question) else {}

    # One entry per symbol (its best chunk), in fused order, inside the chosen repositories. Fallback windows
    # of one file overlap and look alike (a lockfile has dozens), so only a file's best window is kept.
    fused, seen = [], set()
    for h in hits:
        rowid = h["rowid"]
        if rowid in seen or rowid not in symbols or (wanted is not None and symbols[rowid][0] not in wanted):
            continue
        seen.add(rowid)
        rid, _, kind, _, path, _, _ = symbols[rowid]
        if kind == "chunk":
            if (rid, path) in seen:
                continue
            seen.add((rid, path))
        # vector_score is the index's cosine similarity mapped to 0-100: (1 + cosine) / 2 * 100
        similarity = h["vector_score"] / 50 - 1 if h.get("vector_score") else None
        keyword_hit = bool(h.get("text_score"))
        if not keyword_hit and (similarity is None or similarity < FLOOR):
            continue  # a weak meaning-only match is worse than none; keyword matches always count
        fused.append({"rowid": rowid, "score": float(h["score"]), "similarity": similarity,
                      "text": h["text_rank"] if h.get("text_score") else None,
                      "vector": h["vector_rank"] if h.get("vector_score") else None})
    for side in ("text", "vector"):  # ranks among the symbols searched, not across the whole index
        for n, f in enumerate(sorted((f for f in fused if f[side]), key=lambda f: f[side]), 1):
            f[side] = n

    by_rowid = {f["rowid"]: f for f in fused}
    first = [by_rowid.get(r) or {"rowid": r, "score": 0.0, "similarity": None, "text": None, "vector": None}
             for r, d in definitions.items() if wanted is None or d[0] in wanted]
    first.sort(key=lambda f: (f["rowid"] not in by_rowid, -f["score"]))  # found by the search first, then path
    ordered = (first + [f for f in fused if f["rowid"] not in definitions])[:LIMIT]

    return [
        Result(rank, f["score"], f["similarity"], f["text"], f["vector"], *(symbols.get(f["rowid"]) or definitions[f["rowid"]])[1:])
        for rank, f in enumerate(ordered, 1)
    ]


def _candidates(cur, request: dict, wanted: set[int] | None) -> tuple[list[dict], dict[str, tuple]]:
    """Hits and their symbols. With a repository filter, widen the pool until LIMIT of them are in scope."""
    if "vector" not in request and "text" not in request:
        return [], {}
    while True:
        hits = _hybrid(cur, request)
        symbols = _symbols(cur, list(dict.fromkeys(h["rowid"] for h in hits)))
        topn = request["return"]["topN"]
        in_scope = {r for r, s in symbols.items() if wanted is None or s[0] in wanted}
        # simplified: stops at MAX_CANDIDATES; a tiny repository beside huge ones may still return fewer
        if wanted is None or len(in_scope) >= LIMIT or len(hits) < topn or topn >= MAX_CANDIDATES:
            return hits, symbols
        request["return"]["topN"] = min(topn * 4, MAX_CANDIDATES)
