"""Code search: the in-database embedding model, the hybrid vector index and the fused query.

One hybrid vector index over ccctx_symbols.search_text serves both keyword and semantic matching. The database
computes every vector with the ONNX model; only the published model file is downloaded, once, and no source
code or question text ever leaves the database to be embedded. Wiki pages have their own hybrid index (wiki.INDEX),
searched with the same request and placed among the symbols by semantic similarity.
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
from golestan_code_mcp.chunker import EXTENSIONS
from golestan_code_mcp.config import (
    environment,
    feedback_cap,
    feedback_half_life,
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
MAX_LIMIT = 50
LANGUAGES = tuple(sorted(set(EXTENSIONS.values()) | {"text"}))  # every language the chunker assigns
SNIPPET_LINES = 10
SNIPPET_CHARS = 800
FLOOR = 0.3
INDEX = "CCCTX_SYMBOLS_HYBRID_IX"
LEXER = "CCCTX_CODE_LEXER"
MODES = ("vector", "text", "hybrid")
MODEL_FILE = "all_MiniLM_L12_v2.onnx"
METADATA = '{"function":"embedding","embeddingOutput":"embedding","input":{"input":["DATA"]}}'
IDENTIFIER = re.compile(r"[A-Za-z_][\w.]*")
MAX_CANDIDATES = 4000  # widest pool a repository filter may ask for
PAGE_CANDIDATES = 200  # page index chunks to read: every page of a few repositories
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
    path: str  # a page hit's directory
    start_line: int | None  # None for a page hit: a page has no lines in the code
    end_line: int | None
    language: str
    snippet: str  # the opening of the stored source: at most SNIPPET_LINES lines and SNIPPET_CHARS characters
    state: str | None = None  # a page hit's fresh or stale; None for a symbol
    relevance: float = 0.0  # the score before feedback
    feedback: float = 0.0  # the usage feedback boost: score = relevance * (1 + feedback); 0 for facts (011)


def check_limit(limit) -> int:
    """The limit, if it is an integer from 1 to MAX_LIMIT. Needs no database."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return limit


def check_language(language: str | None) -> str | None:
    """The language, if it is None or one the chunker assigns. Needs no database."""
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"unknown language '{language}'; valid languages: {', '.join(LANGUAGES)}")
    return language


def _snippet(text: str | None) -> str:
    return "\n".join((text or "").split("\n")[:SNIPPET_LINES]).rstrip("\r\n")


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
    from golestan_code_mcp import wiki  # wiki imports this module

    with conn.cursor() as cur:
        loaded, dimension, index = _state(cur)
        if loaded and dimension == DIMENSION and index:
            wiki.ensure_index(cur)  # an index built before the wiki existed gets its page index here
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
        wiki.ensure_index(cur)
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
    """rowid -> (repo id, repo, kind, name, path, start, end, language, snippet) for the given symbol rowids."""
    found = {}
    for i in range(0, len(rowids), 500):  # stay well inside Oracle's 1000-item IN list
        chunk = rowids[i:i + 500]
        binds = ", ".join(f"chartorowid(:{n + 1})" for n in range(len(chunk)))
        cur.execute(
            f"""select rowidtochar(s.rowid), r.id, r.name, s.kind, s.name, f.path, s.start_line, s.end_line,
                       f.language, dbms_lob.substr(s.source, {SNIPPET_CHARS}, 1)
                  from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
                 where s.rowid in ({binds})""",
            chunk,
        )
        found.update({row[0]: (*row[1:-1], _snippet(row[-1])) for row in cur})
    return found


def _page_rows(cur, rowids: list[str]) -> dict[str, tuple]:
    """rowid -> the _symbols tuple shape for wiki pages: kind 'page', the directory as path, no lines, plus state."""
    found = {}
    for i in range(0, len(rowids), 500):
        chunk = rowids[i:i + 500]
        binds = ", ".join(f"chartorowid(:{n + 1})" for n in range(len(chunk)))
        cur.execute(
            f"""select rowidtochar(p.rowid), r.id, r.name, 'page', p.name, p.directory, null, null, p.language,
                       dbms_lob.substr(p.text, {SNIPPET_CHARS}, 1), p.state
                  from ccctx_wiki_pages p join ccctx_repos r on r.id = p.repo_id
                 where p.rowid in ({binds})""",
            chunk,
        )
        found.update({row[0]: (*row[1:-2], _snippet(row[-2]), row[-1]) for row in cur})
    return found


def _definitions(cur, name: str) -> dict[str, tuple]:
    """Symbols whose name, or the last part of their dotted name, equals the identifier (any case)."""
    cur.execute(
        f"""select rowidtochar(s.rowid), r.id, r.name, s.kind, s.name, f.path, s.start_line, s.end_line,
                       f.language, dbms_lob.substr(s.source, {SNIPPET_CHARS}, 1)
             from ccctx_symbols s join ccctx_files f on f.id = s.file_id join ccctx_repos r on r.id = f.repo_id
            where s.kind <> 'chunk'
              and (lower(s.name) = :q
                   or (length(s.name) > length(:q) and lower(substr(s.name, -length(:q) - 1)) = '.' || :q))
            order by f.path, s.start_line""",
        q=name.lower(),
    )
    return {row[0]: (*row[1:-1], _snippet(row[-1])) for row in cur}


def search(conn, question: str, repos: list[str] | None = None, mode: str = "hybrid",
           env: Mapping[str, str] | None = None, language: str | None = None, limit: int = LIMIT,
           feedback: bool = True) -> list[Result]:
    """At most `limit` symbols for the question. Identifier definitions first, then fused order.

    feedback: results earlier sessions used are raised by at most the cap, among the candidates found (011 R5).

    hybrid: keyword and semantic sides fused with RRF (UNION). vector: semantic side only, none below FLOOR.
    text: keyword side only, every word must appear. `repos` narrows to those repositories; None searches all.
    `language` narrows to files of that language. A hit is in scope when its repository and language both match.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("the question is empty")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; valid modes: {', '.join(MODES)}")
    check_limit(limit)
    check_language(language)
    env = environment() if env is None else env
    wanted = {repo_id(conn, name) for name in repos} if repos else None  # unknown names refused before searching

    def in_scope(s: tuple) -> bool:
        return (wanted is None or s[0] in wanted) and (language is None or s[7] == language)

    scoped = wanted is not None or language is not None
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
        hits, symbols = _candidates(cur, request, in_scope if scoped else None, limit)
        definitions = _definitions(cur, question) if mode != "vector" and IDENTIFIER.fullmatch(question) else {}
        page_hits, pages = _page_candidates(cur, request)

    fused = _fuse(hits, symbols, in_scope)
    found = _fuse(page_hits, pages, in_scope)
    boosts = _boosts(conn, symbols | definitions | pages, env) if feedback else {}
    if boosts:
        _boost(fused + found, boosts)
        fused.sort(key=lambda f: -f["score"])  # stable: ties keep the fused order
    by_rowid = {f["rowid"]: f for f in fused}
    first = [by_rowid.get(r) or {"rowid": r, "score": 0.0, "relevance": 0.0, "feedback": 0.0, "similarity": None,
                                 "text": None, "vector": None}
             for r, d in definitions.items() if in_scope(d)]
    first.sort(key=lambda f: (f["rowid"] not in by_rowid, -f["score"]))  # found by the search first, then path
    named = [f for f in found if names_module(question, pages[f["rowid"]][3])]  # "the json module": its page first
    rest = _place([f for f in fused if f["rowid"] not in definitions], [f for f in found if f not in named])
    rest = sorted(named, key=lambda f: -(f["similarity"] or -1)) + rest
    rows = symbols | definitions | pages
    if mode != "text":  # fact hits are placed by similarity, like pages, and never above a definition (010 R14)
        from golestan_code_mcp import memory  # memory imports this module

        facts = memory.search_facts(conn, question, repos, language)
        rest = _place(rest, [{"rowid": f"fact:{n}", "score": 0.0, "relevance": 0.0, "feedback": 0.0,
                              "similarity": f["similarity"], "text": None, "vector": n} for n, f in enumerate(facts, 1)])
        rows |= {f"fact:{n}": (None, f["repo"], "fact", f["name"], f["path"], f["start_line"], f["end_line"],
                               f["language"], f["snippet"], "fresh") for n, f in enumerate(facts, 1)}
    return [
        Result(rank, f["score"], f["similarity"], f["text"], f["vector"], *rows[f["rowid"]][1:],
               relevance=f["relevance"], feedback=f["feedback"])
        for rank, f in enumerate((first + rest)[:limit], 1)
    ]


def _boosts(conn, rows: dict[str, tuple], env: Mapping[str, str]) -> dict[str, float]:
    """rowid -> boost for the candidates with uses, by identity (repo id, path, name). Empty with a cap of 0."""
    cap = feedback_cap(env)
    if not cap or not rows:
        return {}
    from golestan_code_mcp import feedback as usage  # feedback is this function's caller's parameter

    with conn.cursor() as cur:
        weights = usage.weights(cur, {r[0] for r in rows.values()}, feedback_half_life(env))
    return {rowid: usage.boost(weights[key][1], cap) for rowid, r in rows.items()
            if (key := (r[0], r[4], r[3])) in weights}


def _boost(entries: list[dict], boosts: dict[str, float]) -> None:
    """Raise each entry's score by its boost, and a page's similarity for placing it (at most 1)."""
    for f in entries:
        b = boosts.get(f["rowid"], 0.0)
        if b:
            f["feedback"], f["score"] = b, f["relevance"] * (1 + b)
            if f["similarity"] is not None:
                f["placed"] = min(1.0, f["similarity"] * (1 + b))


def _fuse(hits: list[dict], rows: dict[str, tuple], in_scope) -> list[dict]:
    """One entry per row (its best chunk), in fused order, inside the scope, ranked among the rows kept.

    Fallback windows of one file overlap and look alike (a lockfile has dozens), so only a file's best window is kept.
    """
    fused, seen = [], set()
    for h in hits:
        rowid = h["rowid"]
        if rowid in seen or rowid not in rows or not in_scope(rows[rowid]):
            continue
        seen.add(rowid)
        rid, _, kind, _, path, *_ = rows[rowid]
        if kind == "chunk":
            if (rid, path) in seen:
                continue
            seen.add((rid, path))
        # vector_score is the index's cosine similarity mapped to 0-100: (1 + cosine) / 2 * 100
        similarity = h["vector_score"] / 50 - 1 if h.get("vector_score") else None
        keyword_hit = bool(h.get("text_score"))
        if not keyword_hit and (similarity is None or similarity < FLOOR):
            continue  # a weak meaning-only match is worse than none; keyword matches always count
        fused.append({"rowid": rowid, "score": float(h["score"]), "relevance": float(h["score"]), "feedback": 0.0,
                      "similarity": similarity,
                      "text": h["text_rank"] if h.get("text_score") else None,
                      "vector": h["vector_rank"] if h.get("vector_score") else None})
    for side in ("text", "vector"):  # ranks among the rows searched, not across the whole index
        for n, f in enumerate(sorted((f for f in fused if f[side]), key=lambda f: f[side]), 1):
            f[side] = n
    return fused


MODULE_WORDS = r"(?:module|package|directory)"


def names_module(question: str, page: str) -> bool:
    """Whether the question names the page's module as one: `the json module`, `package binding`.

    Similarity alone ranked a docs chunk about JSON above the flask/json page (research R8's fallback). A bare
    mention is not enough: every question about Flask would put the root `flask` page first.
    """
    word = re.escape(page.rpartition("/")[2])
    return bool(re.search(rf"\b{word}\s+{MODULE_WORDS}\b|\b{MODULE_WORDS}\s+{word}\b", question, re.IGNORECASE))


def _place(hits: list[dict], pages: list[dict]) -> list[dict]:
    """Each page above the first hit whose semantic similarity is lower; a page without one goes last.

    The two indexes' fused scores do not compare, but their similarities come from one model (research R8).
    """
    placed = list(hits)
    for page in pages:
        similarity = page.get("placed", page["similarity"])  # a used page's similarity, boosted (011 R5)
        if similarity is None:
            placed.append(page)
            continue
        at = next((i for i, h in enumerate(placed)
                   if h["similarity"] is None or h["similarity"] < similarity), len(placed))
        placed.insert(at, page)
    return placed


def _page_candidates(cur, request: dict) -> tuple[list[dict], dict[str, tuple]]:
    """Page hits for the same request, with their rows. Empty without a page index or anything to ask."""
    from golestan_code_mcp import wiki  # wiki imports this module

    if "vector" not in request and "text" not in request:
        return [], {}
    if not cur.execute("select count(*) from user_indexes where index_name = :i", i=wiki.INDEX).fetchone()[0]:
        return [], {}
    hits = _hybrid(cur, request | {"hybrid_index_name": wiki.INDEX, "return": {"topN": PAGE_CANDIDATES}})
    return hits, _page_rows(cur, list(dict.fromkeys(h["rowid"] for h in hits)))


def _candidates(cur, request: dict, in_scope, limit: int) -> tuple[list[dict], dict[str, tuple]]:
    """Hits and their symbols. With a repository or language filter, widen the pool until `limit` are in scope."""
    if "vector" not in request and "text" not in request:
        return [], {}
    while True:
        hits = _hybrid(cur, request)
        symbols = _symbols(cur, list(dict.fromkeys(h["rowid"] for h in hits)))
        topn = request["return"]["topN"]
        found = sum(1 for s in symbols.values() if in_scope is None or in_scope(s))
        # simplified: stops at MAX_CANDIDATES; a tiny repository beside huge ones may still return fewer
        if in_scope is None or found >= limit or len(hits) < topn or topn >= MAX_CANDIDATES:
            return hits, symbols
        request["return"]["topN"] = min(topn * 4, MAX_CANDIDATES)
