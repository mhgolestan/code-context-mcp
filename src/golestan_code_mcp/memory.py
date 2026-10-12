"""Agent memory (specs/010-agent-memory): runs of every MCP tool call, facts distilled from them, and recall.

Runs live in ccctx_memory_runs. Facts are `fact` records in Oracle's AI Agent Memory package (research R1), scoped
by repository name. The package is imported only where facts are touched, never on the recording path.
"""

import atexit
import hashlib
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import oracledb

from golestan_code_mcp import __version__, db, graph, registry, semantic, wiki
from golestan_code_mcp.config import environment, load, memory_model

AGENT = "ccctx"
STORE_ID = "CCCTX_MEM"  # the package's tables are named CCCTX_MEM_*
CONSOLIDATOR = f"memory/{__version__}"
TEMPLATE = "template/1"
MAX_INPUT_TOKENS = 128  # the ONNX model's input window; the package cannot infer it

log = logging.getLogger("golestan_code_mcp.memory")

# What a run keeps of each result: its identity and location, never its text (FR-002, research R3).
KEEP = ("kind", "repo", "name", "path", "start_line", "end_line", "line", "score", "similarity", "callee", "caller",
        "hops", "state")
QUESTION = ("query", "name", "page", "topic")


class MemoryStoreError(ValueError):
    """The memory store is missing or unusable. Safe to show the user."""


# --- runs: the episodic log ---


def _rows(data: dict) -> list[dict]:
    if data.get("page"):
        p = data["page"]
        return [{"kind": "page", "name": p["name"], "repo": p["repo"], "path": p["directory"]}]
    if "pages" in data and "results" not in data:
        return [{"kind": "page", "name": p["name"], "repo": p["repo"], "state": p["state"]} for p in data["pages"]]
    if "repositories" in data:
        return [{"kind": "repository", "name": r["name"]} for r in data["repositories"]]
    if "facts" in data:
        return [{"kind": "fact", "repo": f["repo"], "name": f["subject"]} for f in data["facts"]]
    return [{k: row[k] for k in KEEP if row.get(k) is not None} for row in data.get("results", [])]


def run_of(tool: str, arguments: dict, result, client: str | None, ms: int) -> dict:
    """One run from a tool call: what was asked, by whom, and the identities and locations of what came back."""
    data = None if result is None else result.model_dump()
    if data is None:
        status, reason, rows = "error", None, []
    else:
        status = data.get("status") or ("ok" if data.get("database") == "reachable" else "unavailable")
        reason, rows = data.get("reason"), _rows(data)
    question = next((arguments[k] for k in QUESTION if arguments.get(k) is not None), None)
    repos = [arguments["repo"]] if arguments.get("repo") else sorted({r["repo"] for r in rows if r.get("repo")})
    return {"tool": tool, "arguments": arguments, "question": None if question is None else str(question)[:1000],
            "client": client, "status": status, "reason": reason and reason[:1000], "ms": ms,
            "result_count": len(rows), "results": rows, "repos": repos}


def store_runs(conn, runs: list[dict]) -> None:
    """Insert runs and their repository links, in order, then commit. Unknown repositories are skipped."""
    with conn.cursor() as cur:
        ids = dict(cur.execute("select name, id from ccctx_repos").fetchall())  # a handful of rows
        for run in runs:
            run_id = cur.var(int)
            links = [ids[n] for n in run["repos"] if n in ids]
            cur.execute(
                "insert into ccctx_memory_runs (tool, arguments, question, client, status, reason, ms, result_count,"
                " results, scoped) values (:t, json(:a), :q, :c, :s, :r, :ms, :n, json(:res), :sc)"
                " returning id into :i",
                t=run["tool"], a=json.dumps(run["arguments"], default=str), q=run["question"], c=run["client"],
                s=run["status"], r=run["reason"], ms=run["ms"], n=run["result_count"],
                res=json.dumps(run["results"], default=str), sc=bool(links), i=run_id)
            rid = run_id.getvalue()[0]
            cur.executemany("insert into ccctx_memory_run_repos (run_id, repo_id) values (:r, :p)",
                            [{"r": rid, "p": p} for p in links])
    conn.commit()


# One writer thread with one connection keeps runs in call order and off the answering path (research R2).
_runs: queue.Queue = queue.Queue()
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()
_STOP = object()


def record(run: dict) -> None:
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = threading.Thread(target=_write_runs, name="ccctx-memory-writer", daemon=True)
            _writer.start()
            atexit.register(_drain)
    _runs.put(run)


def _write_runs() -> None:
    conn = None
    while True:
        run = _runs.get()
        try:
            if run is _STOP:
                return
            if conn is None:
                conn = db.connect(load())
            store_runs(conn, [run])
        except Exception as e:  # never reaches the client (FR-003); the run is dropped
            reason = db.describe(e) if isinstance(e, oracledb.Error) else str(e)
            log.warning(json.dumps({"event": "memory_record_failed", "tool": run["tool"], "reason": reason[:300]}))
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - a broken connection may fail to close too
                    pass
                conn = None
        finally:
            _runs.task_done()


def flush() -> None:
    """Wait until every recorded run has been written or dropped."""
    _runs.join()


def _drain() -> None:
    if _writer is not None and _writer.is_alive():
        _runs.put(_STOP)
        _writer.join(5)


@dataclass(frozen=True)
class RunView:
    id: int
    called_at: datetime
    tool: str
    question: str | None
    client: str | None
    status: str
    reason: str | None
    result_count: int
    top: dict | None


def runs(conn, repo: str | None = None, limit: int = 20, skip_tool: str | None = None) -> list[RunView]:
    """The newest runs first, optionally only those belonging to a repository."""
    rid = registry.repo_id(conn, repo) if repo else None
    with conn.cursor() as cur:
        cur.execute(
            "select r.id, r.called_at, r.tool, r.question, r.client, r.status, r.reason, r.result_count,"
            " json_query(r.results, '$[0]' returning varchar2(4000) null on error)"
            " from ccctx_memory_runs r"
            " where (:rid is null or exists (select 1 from ccctx_memory_run_repos l"
            "                                 where l.run_id = r.id and l.repo_id = :rid))"
            "   and (:skip is null or r.tool <> :skip)"
            " order by r.id desc fetch first :lim rows only", rid=rid, skip=skip_tool, lim=limit)
        return [RunView(*row[:8], json.loads(row[8]) if row[8] else None) for row in cur]


def _store(conn, policy, embedder, store_id: str):
    from oracleagentmemory.core import OracleDBMemoryStore

    return OracleDBMemoryStore(embedder=embedder, pool=conn, vector_dim=semantic.DIMENSION, schema_policy=policy,
                               memory_store_id=store_id)


def create_package_schema(conn, store_id: str = STORE_ID) -> None:
    """Migration 8's step: the package creates (or repairs) its tables. Needs no embedding model."""
    from oracleagentmemory.core import SchemaPolicy

    _store(conn, SchemaPolicy.CREATE_IF_NECESSARY, None, store_id)


def client(conn, store_id: str = STORE_ID):
    """A package client over the migrated store, embedding with the in-database model. Never creates tables."""
    semantic.require_model(conn)
    from oracleagentmemory.core import MemoryExtractionConfig, OracleAgentMemory, SchemaPolicy
    from oracleagentmemory.core.embedders import OracleDBEmbedder

    embedder = OracleDBEmbedder(connection=conn, model=semantic.MODEL, embedding_dimension=semantic.DIMENSION,
                                max_input_tokens=MAX_INPUT_TOKENS)
    try:
        store = _store(conn, SchemaPolicy.REQUIRE_EXISTING, embedder, store_id)
    except RuntimeError:
        raise MemoryStoreError(f"memory store {store_id} is missing or out of date; run `ccctx db migrate`") from None
    return OracleAgentMemory(store=store, embedder=embedder,
                             memory_extraction_config=MemoryExtractionConfig(extract_memories=False))


def model(env: Mapping[str, str] | None = None):
    """(package Llm, distiller name) for the configured model, or None: then the templates write every fact."""
    configured = memory_model(environment() if env is None else env)
    if configured is None:
        return None
    from oracleagentmemory.core.llms.llm import Llm

    options = {k: v for k, v in (("api_base", configured.base), ("api_key", configured.key)) if v}
    return Llm(model=configured.model, temperature=0, **options), f"llm:{configured.model}"


# --- findings: what one run says about one subject (research R5) ---

NO_FACTS = {"list_repositories", "recall"}
NAMES = 5  # callers, callees or reached symbols named in a fact
SOURCE_LINES = 200  # of the subject's source sent to a configured model (research R13)


@dataclass
class Finding:
    subject_key: str
    subject: str
    kind: str  # the subject's symbol kind, or "page"
    path: str  # the subject's file, or a page's directory
    language: str
    aspect: str  # defined, module, callers, callees or impact
    value: dict
    tool: str
    arguments: dict
    run_id: int
    question: str | None
    refs: list[dict]  # the subject first, then related symbols


def _plain(value):
    """JSON from the database as plain Python: Oracle returns numbers as Decimal, which json cannot write."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def _json(value):
    return _plain(json.loads(value) if isinstance(value, str) else value)


def pending(cur, repo_id: int) -> list[dict]:
    """The repository's runs not yet consolidated, oldest first."""
    cur.execute("""select r.id, r.tool, r.arguments, r.question, r.status, r.results
                     from ccctx_memory_run_repos l join ccctx_memory_runs r on r.id = l.run_id
                    where l.repo_id = :r and not l.consolidated
                    order by r.id""", r=repo_id)
    return [{"id": i, "tool": t, "arguments": _json(a) or {}, "question": q, "status": s, "results": _json(res) or []}
            for i, t, a, q, s, res in cur.fetchall()]


def _symbol(cur, repo_id: int, name: str, kind: str | None = None, path: str | None = None) -> dict | None:
    """The current definition with this name (and kind), preferring the given path."""
    row = cur.execute(
        """select s.name, s.kind, f.path, f.language, s.content_hash, s.start_line, s.end_line
             from ccctx_symbols s join ccctx_files f on f.id = s.file_id
            where f.repo_id = :r and s.name = :n and (:k is null or s.kind = :k)
            order by case when f.path = :p then 0 else 1 end, f.path, s.start_line
            fetch first 1 rows only""", r=repo_id, n=name, k=kind, p=path).fetchone()
    if row is None:
        return None
    keys = ("name", "kind", "path", "language", "content_hash", "start_line", "end_line")
    return dict(zip(keys, row))


def _page(cur, repo_id: int, name: str) -> dict | None:
    row = cur.execute("select name, directory, language, text from ccctx_wiki_pages where repo_id = :r and name = :n",
                      r=repo_id, n=name).fetchone()
    if row is None:
        return None
    text = row[3].read() if hasattr(row[3], "read") else row[3]
    return {"name": row[0], "kind": "page", "path": row[1], "language": row[2], "text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(), "start_line": None, "end_line": None}


def _ref(record: dict, role: str) -> dict:
    return {"role": role, "kind": record["kind"], "name": record["name"], "path": record["path"],
            "content_hash": record["content_hash"]}


def _key(record: dict) -> str:
    if record["kind"] == "page":
        return f"page:{record['name']}"
    return f"symbol:{record['path']}:{record['name']}:{record['kind']}"


def _names(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["name"] for r in rows))[:NAMES]


def finding(cur, repo_id: int, repo: str, run: dict) -> Finding | None:
    """The one finding a run gives for a repository, about its first result's code, or None."""
    if run["status"] != "ok" or run["tool"] in NO_FACTS:
        return None
    rows = [r for r in run["results"] if r.get("repo") == repo]
    tool, args = run["tool"], run["arguments"]
    if not rows or (tool == "read_wiki" and not args.get("page")):
        return None
    top, related, value = rows[0], [], {}
    if tool == "read_wiki" or (tool == "search_code" and top["kind"] == "page"):
        subject, aspect = _page(cur, repo_id, top["name"]), "module"
        if subject:
            value = {"directory": subject["path"], "summary": wiki._summary(subject["text"])}
    elif tool in ("search_code", "get_symbol"):
        kind = None if top["kind"] == "fact" else top["kind"]  # a fact hit stands for its own subject (R14)
        subject, aspect = _symbol(cur, repo_id, top["name"], kind, top.get("path")), "defined"
        if subject:
            value = {"path": subject["path"]} | ({"definitions": len(rows)} if tool == "get_symbol" else {})
    elif tool in ("find_callers", "find_callees"):
        column = "callee" if tool == "find_callers" else "caller"
        mine = [r for r in rows if r.get(column) == top.get(column)]
        subject, aspect = _symbol(cur, repo_id, top.get(column)), tool.removeprefix("find_")
        value, related = {"count": len(mine), "names": _names(mine)}, mine
    elif tool == "impact_of":
        try:
            found, _ = graph.definitions(cur.connection, args.get("name", ""), [repo])
        except ValueError:
            found = []
        subject = _symbol(cur, repo_id, found[0].name, found[0].kind, found[0].path) if found else None
        aspect = "impact"
        value, related = {"count": len(rows), "depth": args.get("depth", 3), "names": _names(rows)}, rows
    else:
        return None
    if subject is None:
        return None
    refs = [_ref(subject, "subject")]
    for name in value.get("names", []):
        row = next(r for r in related if r["name"] == name)
        if (rec := _symbol(cur, repo_id, name, row.get("kind"), row.get("path"))) is not None:
            refs.append(_ref(rec, "related"))
    return Finding(_key(subject), subject["name"], subject["kind"], subject["path"], subject["language"], aspect,
                   value, tool, args, run["id"], run["question"], refs)


# --- facts: one per subject per repository (research R6, R7) ---


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip().rstrip("?.! ")


def confidence(agree: int, contradict: int) -> float:
    return round((1 - 0.5 ** agree) * 0.5 ** contradict, 3)


def _same(aspect: str, old: dict, new: dict) -> bool:
    """Whether a newer finding for an aspect says what the fact already holds."""
    if aspect == "defined":
        return old.get("path") == new.get("path")
    if aspect == "module" or (aspect == "impact" and old.get("depth") != new.get("depth")):
        return True
    return (old.get("count"), old.get("names")) == (new.get("count"), new.get("names"))


@dataclass
class _Fact:
    id: str | None
    meta: dict
    text: str = ""
    changed: bool = False
    weakened: bool = False


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _quoted(names: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)


def template_text(meta: dict) -> str:
    """The template distiller: what the findings say, in sentences, and nothing more."""
    s, kind, path, f = meta["subject"], meta["kind"], meta["path"], meta["findings"]
    if kind == "page":
        summary = f.get("module", {}).get("summary", "").strip()
        parts = [f"`{s}` is the wiki page for `{path}`." + (f" {summary.rstrip('.')}." if summary else "")]
    else:
        parts = [f"`{s}` is a {kind} in `{path}`."]
        if (n := f.get("defined", {}).get("definitions") or 0) > 1:
            parts.append(f"It is one of {n} definitions named `{s.rsplit('.', 1)[-1]}`.")
    if (c := f.get("callers")) and c["count"]:
        parts.append(f"It is called by {_plural(c['count'], 'symbol')}, including {_quoted(c['names'])}.")
    if (c := f.get("callees")) and c["count"]:
        parts.append(f"It calls {_plural(c['count'], 'indexed symbol')}, including {_quoted(c['names'])}.")
    if (c := f.get("impact")) and c["count"]:
        parts.append(f"A change to it reaches {_plural(c['count'], 'symbol')} within {c['depth']} call hops, nearest"
                     f" first: {_quoted(c['names'])}.")
    if meta["questions"]:
        parts.append("Asked as " + ", ".join(f'"{q}"' for q in meta["questions"][:3]) + ".")
    return " ".join(parts)


WORD = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
CODE = re.compile(r"\w*_\w+|[a-z]+[A-Z]\w*|[A-Z][a-z0-9]+[A-Z]\w*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
SPAN = re.compile(r"`([^`\n]+)`")


def _names_in(text: str) -> list[str]:
    """The code identifiers a text names: a backticked identifier, and code-shaped words (snake_case, CamelCase,
    dotted) anywhere. Inside a backticked phrase only code-shaped words count, so prose caught between unpaired
    backticks is not taken for a name."""
    found = []
    for span in SPAN.findall(text):
        whole = span.strip().removesuffix("()")
        found += [whole] if WORD.fullmatch(whole) else [w for w in WORD.findall(span) if CODE.fullmatch(w)]
    found += [w for w in WORD.findall(SPAN.sub(" ", text)) if CODE.fullmatch(w)]
    return list(dict.fromkeys(found))


def unknown_names(text: str, allowed: set[str]) -> list[str]:
    """The identifiers the text names that appear nowhere in what was allowed (research R13).

    A dotted name passes when each of its parts appears, so `pkg.widgets` matches the path pkg/widgets.py.
    """
    known = {w for a in allowed for w in WORD.findall(a)}
    parts = known | {p for k in known for p in k.split(".")}
    return [n for n in _names_in(text) if n not in known and not all(p in parts for p in n.split("."))]


def _source(cur, repo_id: int, meta: dict) -> str:
    if meta["kind"] == "page":
        page = _page(cur, repo_id, meta["subject"])
        text = page["text"] if page else ""
    else:
        row = cur.execute("""select s.source from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                              where f.repo_id = :r and s.name = :n and s.kind = :k and f.path = :p
                              fetch first 1 rows only""", r=repo_id, n=meta["subject"], k=meta["kind"],
                          p=meta["path"]).fetchone()
        text = (row[0].read() if hasattr(row[0], "read") else row[0]) if row else ""
    return "\n".join(text.split("\n")[:SOURCE_LINES])


ANSWER = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _prompt(cur, repo_id: int, meta: dict) -> str:
    """Exactly what research R13 says is sent: the subject's source, its findings, related code, the questions."""
    findings = {a: {k: v for k, v in f.items() if k not in ("tool", "arguments", "run")}
                for a, f in meta["findings"].items()}
    related = [f"- `{r['name']}` ({r['kind']}) in `{r['path']}`" for r in meta["refs"] if r["role"] == "related"]
    return "\n".join([
        "Write at most 3 sentences stating what the findings and the source below show about the subject. Name only"
        " code that appears below, in backticks. Reply as JSON: {\"text\": \"...\"}.",
        "", f"Subject: `{meta['subject']}` ({meta['kind']}) in `{meta['path']}`",
        "", "Source:", _source(cur, repo_id, meta),
        "", "Findings:", json.dumps(findings, sort_keys=True),
        "", "Related code:", *(related or ["- none"]),
        "", "Questions asked:", *(f"- {q}" for q in meta["questions"]),
    ])


def _distill(cur, repo_id: int, repo: str, meta: dict, llm) -> tuple[str, str]:
    """(text, distiller): the configured model's text when it is valid, else the template's."""
    template = template_text(meta)
    if llm is None:
        return template, TEMPLATE
    model_, name = llm
    prompt = _prompt(cur, repo_id, meta)
    try:
        text = json.loads(model_.generate(prompt, response_json_schema=ANSWER).text)["text"]
        text = text.strip()
        if not text:
            raise ValueError("empty text")
    except Exception as e:  # noqa: BLE001 - any model failure falls back; consolidation goes on
        log.warning(json.dumps({"event": "memory_model_failed", "repo": repo, "reason": type(e).__name__}))
        return template, TEMPLATE
    sent = prompt.split("\n\nSubject:", 1)[1]  # everything it was given about the code, not the instructions
    if bad := unknown_names(text, {sent}):
        log.warning(json.dumps({"event": "memory_model_rejected", "repo": repo, "names": bad}))
        return template, TEMPLATE
    return text, name


@dataclass
class Consolidation:
    repo: str
    runs: int = 0
    created: int = 0
    merged: int = 0
    weakened: int = 0
    stale: int = 0
    reasked: int = 0
    distillers: dict = field(default_factory=dict)
    seconds: float = 0.0


def _new_meta(f: Finding) -> dict:
    now = _now()
    return {"subject_key": f.subject_key, "subject": f.subject, "kind": f.kind, "path": f.path,
            "language": f.language, "findings": {}, "questions": [], "runs": [], "refs": f.refs, "agree": 0,
            "contradict": 0, "confidence": 0.0, "state": "fresh", "stale_reason": None, "distiller": TEMPLATE,
            "consolidator": CONSOLIDATOR, "created_at": now, "updated_at": now}


def _contradict(fact: _Fact) -> None:
    fact.meta["contradict"] += 1
    fact.changed = fact.weakened = True


def _merge(facts: dict[str, _Fact], f: Finding, links: list[tuple[str, str]], c: Consolidation) -> None:
    fact = facts.get(f.subject_key)
    if fact is not None and f.run_id in fact.meta["runs"]:
        return  # counted before a crash between writing facts and flagging links (research R9)
    if fact is None:
        fact = facts[f.subject_key] = _Fact(None, _new_meta(f))
        c.created += 1
    else:
        c.merged += 1
    meta = fact.meta
    old = meta["findings"].get(f.aspect)
    value = dict(f.value)
    if f.aspect == "defined" and old and value.get("definitions") is None and old.get("definitions"):
        value["definitions"] = old["definitions"]
    if old is not None and not _same(f.aspect, old, value):
        _contradict(fact)
    meta["findings"][f.aspect] = value | {"tool": f.tool, "arguments": f.arguments, "run": f.run_id}
    if f.question:
        # the same call answered by other code is a contradiction; the same words to another tool are not
        # (get_symbol and find_callers resolve a bare name to different definitions, both correctly)
        asked = [f.tool, _norm(f.question)]
        for other in facts.values():
            if other is not fact and asked in other.meta.setdefault("asked", []):
                other.meta["asked"].remove(asked)
                if not any(a[1] == asked[1] for a in other.meta["asked"]):
                    other.meta["questions"] = [q for q in other.meta["questions"] if _norm(q) != asked[1]]
                _contradict(other)
                links.append((meta["subject_key"], other.meta["subject_key"]))
        meta["asked"] = ([asked] + [a for a in meta.setdefault("asked", []) if a != asked])[:20]
        kept = [q for q in meta["questions"] if _norm(q) != asked[1]]
        meta["questions"] = ([f.question] + kept)[:20]
    meta["runs"].append(f.run_id)
    meta["agree"] += 1
    refs = {(r["role"], r["kind"], r["name"]): r for r in meta["refs"]}
    refs.update({(r["role"], r["kind"], r["name"]): r for r in f.refs})
    meta["refs"] = [r for r in refs.values() if r["role"] == "subject"] + [
        r for r in refs.values() if r["role"] == "related"]
    meta["path"], meta["language"] = f.path, f.language
    if meta["state"] == "stale":  # a new run confirms it (FR-015)
        meta["state"] = "fresh"
    fact.changed = True


def _load(mem, repo: str) -> dict[str, _Fact]:
    records = mem.list_memories(user_id=repo, agent_id=AGENT, limit=None)
    return {r.metadata["subject_key"]: _Fact(r.id, _plain(dict(r.metadata)), r.content or "")
            for r in records if r.record_type == "fact" and r.metadata and "subject_key" in r.metadata}


def _status(meta: dict):
    from oracleagentmemory.apis.records import RecordStatus

    return RecordStatus.VALID if meta["state"] == "fresh" else RecordStatus.INVALID


def _write(cur, mem, repo_id: int, repo: str, facts: dict[str, _Fact], llm, c: Consolidation) -> None:
    for fact in facts.values():
        if not fact.changed:
            continue
        meta = fact.meta
        meta["confidence"] = confidence(meta["agree"], meta["contradict"])
        meta["updated_at"], meta["consolidator"] = _now(), CONSOLIDATOR
        fact.text, meta["distiller"] = _distill(cur, repo_id, repo, meta, llm)
        c.distillers[meta["distiller"]] = c.distillers.get(meta["distiller"], 0) + 1
        if fact.id is None:
            fact.id = mem.add_memory(fact.text, memory_type="fact", user_id=repo, agent_id=AGENT, metadata=meta,
                                     ttl_days=None, status=_status(meta), autonomous_linking=False)
        else:
            mem.update_memory(fact.id, content=fact.text, metadata=meta, status=_status(meta))
        c.weakened += fact.weakened


def _lock(repo_id: int, repo: str):
    """Hold the repository row on a second connection: the package commits the main one (research R9)."""
    lock = db.connect(load())
    try:
        lock.cursor().execute("select id from ccctx_repos where id = :r for update wait 60", r=repo_id)
    except oracledb.DatabaseError as e:
        lock.close()
        if getattr(e.args[0], "full_code", "") == "ORA-30006":
            raise MemoryStoreError(f"another consolidation of {repo} is running") from None
        raise
    return lock


def _ask_again(conn, repo: str, tool: str, args: dict) -> list[dict]:
    """A finding's call run again through the core the tool uses, never the tool: no run is recorded (R8)."""
    try:
        if tool == "search_code":
            hits = semantic.search(conn, args["query"], [repo], language=args.get("language"),
                                   limit=args.get("limit", semantic.LIMIT))
            rows = [vars(h) for h in hits if h.kind != "fact"]  # a fact must not confirm itself
        elif tool == "get_symbol":
            rows = [vars(d) for d in graph.definitions(conn, args["name"], [repo])[0]]
        elif tool == "find_callers":
            rows = [vars(r) for r in graph.callers(conn, args["name"], [repo])]
        elif tool == "find_callees":
            rows = [vars(r) for r in graph.calls(conn, args["name"], [repo])]
        elif tool == "impact_of":
            rows = [vars(r) for r in graph.impact(conn, args["name"], args.get("depth", 3), [repo])]
        else:  # read_wiki
            page = wiki.read(conn, args["page"], repo)
            rows = [{"kind": "page", "name": page.name, "repo": page.repo, "path": page.directory}]
    except ValueError:  # the name or page no longer exists
        rows = []
    return [{k: r[k] for k in KEEP if r.get(k) is not None} | {"repo": repo} for r in rows]


def _stale(fact: _Fact, reason: str) -> None:
    fact.meta["state"], fact.meta["stale_reason"] = "stale", reason[:512]
    _contradict(fact)


def _rekey(facts: dict[str, _Fact], fact: _Fact, key: str) -> None:
    if key != fact.meta["subject_key"] and key not in facts:
        facts[key] = facts.pop(fact.meta["subject_key"])
        fact.meta["subject_key"] = key


def recheck(cur, mem, repo_id: int, repo: str, facts: dict[str, _Fact]) -> tuple[int, int]:
    """Check each fresh fact's subject against the current index (research R8). Returns (marked stale, re-asked).

    Same code at the same place: nothing. Same code moved: re-pointed. Changed code: every finding is asked again,
    and the fact stays fresh only if the same subject answers. No definition left: stale, a contradiction.
    """
    stale = reasked = 0
    for fact in [f for f in facts.values() if f.meta["state"] == "fresh"]:
        meta = fact.meta
        ref = meta["refs"][0]
        if ref["kind"] == "page":
            now = _page(cur, repo_id, ref["name"])
        else:
            now = _symbol(cur, repo_id, ref["name"], ref["kind"], ref["path"])
            if now and now["path"] != ref["path"] and now["content_hash"] != ref["content_hash"]:
                now = next((r for r in _definitions(cur, repo_id, ref) if r["content_hash"] == ref["content_hash"]),
                           now)  # the same code moved beats another definition of the name
        if now is None:
            _stale(fact, f"`{ref['name']}` was removed")
            stale += 1
            continue
        if now["content_hash"] == ref["content_hash"]:
            if now["path"] != ref["path"]:  # moved, unchanged: re-point only
                meta["refs"][0], meta["path"] = _ref(now, "subject"), now["path"]
                _rekey(facts, fact, _key(now))
                fact.changed = True
            continue
        reasked += 1
        answered = []
        for aspect, found in meta["findings"].items():
            run = {"id": found["run"], "tool": found["tool"], "arguments": found["arguments"], "status": "ok",
                   "question": next((found["arguments"][k] for k in QUESTION if found["arguments"].get(k)), None),
                   "results": _ask_again(cur.connection, repo, found["tool"], found["arguments"])}
            answered.append((aspect, run, finding(cur, repo_id, repo, run)))
        # only the fact's own subject confirms it: changed code somewhere else is other code (clarification Q1)
        wrong = next(((run, f) for _, run, f in answered if f is None or f.subject_key != meta["subject_key"]), None)
        if wrong:
            run, f = wrong
            other = f"now answered by `{f.subject}` in `{f.path}`" if f else "no longer answered"
            _stale(fact, f're-asked "{run["question"]}": {other}')
            stale += 1
            continue
        for aspect, _, f in answered:
            meta["findings"][aspect] = dict(f.value) | {k: meta["findings"][aspect][k]
                                                        for k in ("tool", "arguments", "run")}
            if aspect == "defined" and meta["findings"][aspect].get("definitions") is None:
                meta["findings"][aspect].pop("definitions", None)
        meta["refs"] = answered[0][2].refs + [r for _, _, f in answered[1:] for r in f.refs[1:]]
        meta["path"], meta["language"] = now["path"], now["language"]
        _rekey(facts, fact, _key(now))
        fact.changed = True
    return stale, reasked


def _definitions(cur, repo_id: int, ref: dict) -> list[dict]:
    """Every current definition with the reference's name and kind."""
    keys = ("name", "kind", "path", "language", "content_hash", "start_line", "end_line")
    cur.execute("""select s.name, s.kind, f.path, f.language, s.content_hash, s.start_line, s.end_line
                     from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                    where f.repo_id = :r and s.name = :n and s.kind = :k""", r=repo_id, n=ref["name"], k=ref["kind"])
    return [dict(zip(keys, row)) for row in cur.fetchall()]


def _consolidate(conn, mem, repo: str, llm) -> Consolidation:
    started, repo_id = time.monotonic(), registry.repo_id(conn, repo)
    c = Consolidation(repo)
    lock = _lock(repo_id, repo)
    try:
        facts = _load(mem, repo)
        with conn.cursor() as cur:
            c.stale, c.reasked = recheck(cur, mem, repo_id, repo, facts)
            runs_ = pending(cur, repo_id)
            links: list[tuple[str, str]] = []
            for run in runs_:
                if (f := finding(cur, repo_id, repo, run)) is not None:
                    _merge(facts, f, links, c)
            c.runs = len(runs_)
            _write(cur, mem, repo_id, repo, facts, llm, c)
            for new, old in dict.fromkeys(links):
                try:
                    mem.link_records(facts[new].id, "fact", facts[old].id, "fact", "contradicts")
                except (KeyError, ValueError, RuntimeError):  # already linked, in either orientation
                    pass
            cur.executemany("update ccctx_memory_run_repos set consolidated = true where run_id = :r and repo_id = :p",
                            [{"r": run["id"], "p": repo_id} for run in runs_])
            cur.execute("""delete from ccctx_memory_runs r where r.scoped
                             and not exists (select 1 from ccctx_memory_run_repos l where l.run_id = r.id)""")
        conn.commit()
    finally:
        lock.rollback()
        lock.close()
    c.seconds = round(time.monotonic() - started, 1)
    log.info(json.dumps({"event": "consolidate", "repo": repo, "runs": c.runs, "created": c.created,
                         "merged": c.merged, "weakened": c.weakened, "stale": c.stale, "reasked": c.reasked,
                         "distillers": c.distillers, "seconds": c.seconds}))
    return c


def consolidate(conn, repo: str | None = None, env: Mapping[str, str] | None = None) -> list[Consolidation]:
    """Distil the runs not yet consolidated into facts, for one repository or every one with runs or facts."""
    if repo is not None:
        registry.repo_id(conn, repo)
    mem, llm = client(conn), model(env)
    if repo is None:
        with conn.cursor() as cur:
            names = {n for (n,) in cur.execute("select distinct r.name from ccctx_memory_run_repos l"
                                               " join ccctx_repos r on r.id = l.repo_id")}
            known = {n for (n,) in cur.execute("select name from ccctx_repos")}
        names |= {r.user_id for r in mem.list_memories(agent_id=AGENT, limit=None)} & known
    return [_consolidate(conn, mem, name, llm) for name in ([repo] if repo else sorted(names))]


@dataclass(frozen=True)
class FactView:
    id: str
    repo: str
    subject: str
    kind: str
    text: str
    confidence: float
    state: str
    stale_reason: str | None
    distiller: str
    consolidator: str
    runs: list[int]
    questions: list[str]
    sources: list[dict]
    language: str
    updated_at: str


def _sources(cur, repo_id: int, repo: str, refs: list[dict]) -> list[dict]:
    out = []
    for ref in refs:
        rec = (_page(cur, repo_id, ref["name"]) if ref["kind"] == "page"
               else _symbol(cur, repo_id, ref["name"], ref["kind"], ref["path"]))
        out.append({"role": ref["role"], "name": ref["name"], "kind": ref["kind"], "repo": repo,
                    "path": rec["path"] if rec else None, "start_line": rec["start_line"] if rec else None,
                    "end_line": rec["end_line"] if rec else None})
    return out


def _records(conn, mem, repo: str | None) -> list:
    """The fact records of a repository, or of every registered one."""
    with conn.cursor() as cur:
        known = {n for (n,) in cur.execute("select name from ccctx_repos")}
    scope = {"user_id": repo} if repo else {}
    return [r for r in mem.list_memories(agent_id=AGENT, limit=None, **scope)
            if r.record_type == "fact" and r.metadata and "subject_key" in r.metadata and r.user_id in known]


def _ordered(records: list) -> list:
    """Fresh first, then by confidence, then the most recently updated."""
    records = sorted(records, key=lambda r: r.metadata["updated_at"], reverse=True)
    return sorted(records, key=lambda r: (r.metadata["state"] != "fresh", -r.metadata["confidence"]))


def _views(conn, records) -> list[FactView]:
    """The records as views, each source located where it is now. Only call it on what will be returned."""
    with conn.cursor() as cur:
        ids = dict(cur.execute("select name, id from ccctx_repos").fetchall())
        return [FactView(r.id, r.user_id, m["subject"], m["kind"], r.content or "", m["confidence"], m["state"],
                         m.get("stale_reason"), m["distiller"], m["consolidator"], m["runs"], m["questions"],
                         _sources(cur, ids[r.user_id], r.user_id, m["refs"]), m.get("language", ""), m["updated_at"])
                for r in records for m in [_plain(r.metadata)]]


def facts(conn, repo: str | None = None, limit: int = 20) -> list[FactView]:
    """Facts, fresh first, then by confidence, then the most recently updated."""
    if repo is not None:
        registry.repo_id(conn, repo)
    return _views(conn, _ordered(_records(conn, client(conn), repo))[:limit])


@dataclass(frozen=True)
class Recall:
    facts: list[FactView]
    runs: list[RunView]
    total: int


RECALL_RUNS = 10


def recall(conn, repo: str | None = None, topic: str | None = None, limit: int = 20) -> Recall:
    """What earlier sessions worked out (research R10): facts, by topic when one is given, and the newest runs."""
    if repo is not None:
        registry.repo_id(conn, repo)
    mem = client(conn)
    records = _records(conn, mem, repo)
    if topic:
        by_id = {r.id: r for r in records}
        ranks: dict[str, int] = {}
        for name in ([repo] if repo else sorted({r.user_id for r in records})):
            hits = mem.search(topic, user_id=name, agent_id=AGENT, record_types=["fact"],
                              include_invalid_results=True, max_results=limit)
            for i, hit in enumerate(h for h in hits if h.id in by_id):  # repositories interleave rank by rank
                ranks.setdefault(hit.id, i)
        ranked = sorted((by_id[i] for i in ranks), key=lambda r: ranks[r.id])
        rest = [r for r in _ordered(records) if r.id not in ranks]
        chosen = sorted(ranked + rest, key=lambda r: r.metadata["state"] != "fresh")  # stable: fresh first
    else:
        chosen = _ordered(records)
    return Recall(_views(conn, chosen[:limit]), runs(conn, repo, RECALL_RUNS, skip_tool="recall"), len(records))


def forget(conn, repo: str) -> None:
    """Remove a repository's facts. A future `repo remove` must call this (FR-018)."""
    mem = client(conn)
    mem.delete_user(repo, cascade=True)
    for r in mem.list_memories(user_id=repo, limit=None):
        mem.delete_memory(r.id)


# --- facts in search (research R14, FR-021) ---

FACT_CANDIDATES = 5
SIMILARITY_SQL = (f"select 1 - vector_distance(vector_embedding({semantic.MODEL} using :q as data),"
                  f" vector_embedding({semantic.MODEL} using :t as data), cosine) from dual")


def search_facts(conn, question: str, repos: list[str] | None = None, language: str | None = None,
                 k: int = FACT_CANDIDATES) -> list[dict]:
    """Fresh facts matching the question, each with its similarity to it, its subject's location and its text.

    Repositories with nothing consolidated are skipped without touching the memory package.
    """
    with conn.cursor() as cur:
        names = [n for (n,) in cur.execute("""select distinct r.name from ccctx_memory_run_repos l
                                                 join ccctx_repos r on r.id = l.repo_id where l.consolidated""")]
    names = sorted(n for n in names if repos is None or n in repos)
    if not names:
        return []
    mem = client(conn)
    records = []
    for name in names:
        for hit in mem.search(question, user_id=name, agent_id=AGENT, record_types=["fact"],
                              include_invalid_results=False, max_results=k):
            r, m = hit.record, hit.record.metadata or {}
            if r.record_type == "fact" and m.get("state") == "fresh" and language in (None, m.get("language")):
                records.append(r)
    hits = []
    with conn.cursor() as cur:
        for view in _views(conn, records):
            similarity = cur.execute(SIMILARITY_SQL, q=question, t=view.text).fetchone()[0]
            subject = view.sources[0]
            if similarity < semantic.FLOOR or subject["path"] is None:
                continue  # a weak match is worse than none, like a meaning-only symbol hit
            hits.append({"similarity": float(similarity), "repo": view.repo, "name": view.subject,
                         "path": subject["path"], "start_line": subject["start_line"],
                         "end_line": subject["end_line"], "language": view.language,
                         "snippet": view.text[:semantic.SNIPPET_CHARS]})
    return sorted(hits, key=lambda h: -h["similarity"])
