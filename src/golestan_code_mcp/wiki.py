"""Module wiki: one page per directory of at least THRESHOLD named definitions, compiled from the index alone.

A page says what its directory is for, names its key symbols and the modules it depends on and is used by. Every
record it names is a citation: a foreign key to the symbol or file, with the commit and content hash it had at
compile time. Locations are read through those keys and never written into the text. An ingestion marks a page
stale when a cited symbol's commit changes or a cited record is gone, then recompiles it (research R7).
"""

import json
import logging
import posixpath
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import oracledb

from golestan_code_mcp import db, graph, semantic
from golestan_code_mcp.chunker import EXTENSIONS, GRAMMARS

log = logging.getLogger("golestan_code_mcp.wiki")

COMPILER = "ccctx-wiki/1"  # bump when the page rules change
THRESHOLD = 10
INDEX = "CCCTX_WIKI_HYBRID_IX"
MAX_KEYS = 10
MAX_TYPES = 4
MAX_CALLED = 5  # named symbols per depended-on module
MAX_PURPOSE = 3
MAX_SENTENCE = 300
MAX_TEXT = 8000
MAX_CITATIONS = 200
TYPES = ("class", "interface", "type")
ROLES = ("key", "file", "depends", "uses", "calls")  # the citation order
SYMBOL_ROLES = ("key", "calls")
LANGUAGE_NAMES = {"python": "Python", "go": "Go", "javascript": "JavaScript"}
NO_DOC = "Its code has no documentation; the sections below come from its structure."

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SPHINX_ROLE = re.compile(r":(?:[\w-]+:)+`[~!]?(?:[^`<]*<)?([^`>]+)>?`")  # :class:`~a.B` and :ref:`text <a.B>`
DIRECTIVE = re.compile(r"(?<!\S)\.\. [\w:-]+::")
UNDERLINE = re.compile(r"(?<!\S)([=\-~^\"'#*+])\1{2,}(?!\S)")
_EXTENSIONS = "|".join(sorted((re.escape(e[1:]) for e in EXTENSIONS), key=len, reverse=True))
# A file name with an indexed extension (not a dotted name going on, like flask.json.dumps), or a line number.
UNSAFE = re.compile(rf"[\w-]\.(?:{_EXTENSIONS})(?!\w|\.\w)|\b(?i:line)\s+\d+")


class WikiError(ValueError):
    """Refused wiki input. The message is safe to show the user."""


@dataclass
class Refresh:
    stale: list[tuple[str, str]]  # (page, reason) of the pages that went stale
    recompiled: int = 0
    added: int = 0
    removed: int = 0
    error: str | None = None  # why the recompile failed; the pages then stay stale


@dataclass(frozen=True)
class Built:
    repo: str
    pages: int
    added: int
    updated: int  # existing pages whose text changed
    removed: int
    seconds: float


@dataclass(frozen=True)
class PageInfo:
    name: str
    repo: str
    directory: str
    language: str
    state: str
    symbols: int
    compiled_at: datetime
    stale_reason: str | None
    summary: str  # the first purpose sentence, or the structural one for a page without documentation


@dataclass(frozen=True)
class CitationView:
    name: str
    kind: str
    role: str
    repo: str | None  # read through the key; None once the record is gone
    path: str | None
    start_line: int | None  # None for a file
    end_line: int | None
    recorded_commit: str
    current_commit: str | None


@dataclass(frozen=True)
class Page:
    name: str
    repo: str
    directory: str
    language: str
    state: str
    compiled_at: datetime
    compiler: str
    stale_reason: str | None
    symbols: int
    text: str
    citations: list[CitationView]  # at most MAX_CITATIONS, in ROLES then name order
    truncated: bool


@dataclass
class _Index:
    """What a repository's pages are compiled from, read once."""
    repo: str
    commit: str
    files: dict[int, tuple]  # id -> (path, language, doc, content_hash)
    symbols: dict[int, tuple]  # id -> (file id, name, kind, doc, commit_sha, content_hash), named definitions only
    calls: list[tuple[int, int]]  # (calling file id, callee symbol id), one per call edge
    imports: list[tuple[int, int]]  # (importer file id, imported file id)
    external: dict[int, set[str]]
    folder: dict[int, str] = field(default_factory=dict)  # file id -> directory
    names: dict[str, str] = field(default_factory=dict)  # directory -> module name
    outside: Counter = field(default_factory=Counter)  # symbol id -> calls from other directories
    total: Counter = field(default_factory=Counter)  # symbol id -> all calls


@dataclass
class _Draft:
    directory: str
    name: str
    language: str
    symbols: int
    text: str
    citations: list[dict]


@dataclass(frozen=True)
class _Stored:
    id: int
    name: str
    text: str
    state: str
    reason: str | None


def _lob(value):
    return value.read() if hasattr(value, "read") else value


def ensure_index(cur) -> bool:
    """Create the page index when the model is loaded and the index is missing. Whether the index exists now.

    DDL commits, so call it before writing anything that must stay one transaction.
    """
    if cur.execute("select count(*) from user_indexes where index_name = :i", i=INDEX).fetchone()[0]:
        return True
    if not semantic._state(cur)[0]:
        return False
    semantic._ensure_lexer(cur)
    try:  # same model, lexer and sync as the symbol index, so similarities compare across both (research R8)
        cur.execute(
            f"create hybrid vector index {INDEX} on ccctx_wiki_pages (search_text) parameters ('MODEL"
            f" {semantic.MODEL} VECTOR_IDXTYPE HNSW SYNC (ON COMMIT) LEXER {semantic.LEXER}')"
        )
    except oracledb.DatabaseError as e:
        if getattr(e.args[0], "full_code", "") != "ORA-00955":
            raise semantic.ModelError(f"could not build the page index: {db.describe(e)}") from None
    return True


# --- compiling -----------------------------------------------------------------------------------------------


def _plain_name(directory: str, repo: str) -> str:
    return repo if directory == "." else directory.removeprefix("src/")


def _names(directories, repo: str) -> dict[str, str]:
    """Each directory's name: one leading src/ dropped, the root named after the repository, a clash by full path."""
    plain = {d: _plain_name(d, repo) for d in directories}
    clashes = Counter(plain.values())
    return {d: d if clashes[n] > 1 else n for d, n in plain.items()}


def _load(cur, repo_id: int, repo: str) -> _Index:
    r = {"r": repo_id}
    commit = cur.execute("select indexed_commit from ccctx_repos where id = :r", r).fetchone()[0]
    index = _Index(
        repo, commit,
        {fid: rest for fid, *rest in cur.execute(
            "select id, path, language, doc, content_hash from ccctx_files where repo_id = :r", r)},
        {sid: tuple(rest) for sid, *rest in cur.execute(
            "select s.id, s.file_id, s.name, s.kind, s.doc, s.commit_sha, s.content_hash from ccctx_symbols s"
            " join ccctx_files f on f.id = s.file_id where f.repo_id = :r and s.kind <> 'chunk'", r)},
        cur.execute(
            "select s.file_id, c.callee_id from ccctx_calls c join ccctx_symbols s on s.id = c.caller_id"
            " join ccctx_files f on f.id = s.file_id where f.repo_id = :r"
            " union all select c.file_id, c.callee_id from ccctx_file_calls c"
            " join ccctx_files f on f.id = c.file_id where f.repo_id = :r", r).fetchall(),
        cur.execute("select i.importer_id, i.imported_id from ccctx_imports i"
                    " join ccctx_files f on f.id = i.importer_id where f.repo_id = :r", r).fetchall(),
        graph.external_imports(cur, repo_id),
    )
    index.folder = {fid: posixpath.dirname(f[0]) or "." for fid, f in index.files.items()}
    for caller, callee in index.calls:
        index.total[callee] += 1
        if index.folder[caller] != index.folder[index.symbols[callee][0]]:
            index.outside[callee] += 1
    return index


def _sentences(doc: str | None) -> list[str]:
    """The doc's sentences that are safe to quote: Sphinx roles reduced to their last name, directives, RST
    underlines and backticks removed, and no sentence that names a file or a line."""
    text = SPHINX_ROLE.sub(lambda m: m.group(1).rsplit(".", 1)[-1], doc or "")
    text = UNDERLINE.sub("", text).replace("`", "")  # a backticked name in the page is always a cited one
    found = []
    # simplified: docs are stored as one collapsed paragraph, so a directive drops its whole sentence
    for sentence in SENTENCE_END.split(text):
        sentence = " ".join(sentence.split())
        if sentence and not DIRECTIVE.search(sentence) and not UNSAFE.search(sentence):
            found.append(sentence if len(sentence) <= MAX_SENTENCE else sentence[:MAX_SENTENCE - 3].rstrip() + "...")
    return found


def _plural(n: int, word: str) -> str:
    return f"{n} {word if n == 1 else word + ('es' if word.endswith('s') else 's')}"


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _ticked(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _symbol_citation(index: _Index, role: str, sid: int) -> dict:
    _, name, kind, _, commit, digest = index.symbols[sid]
    return {"s": sid, "f": None, "role": role, "name": name, "kind": kind, "sha": commit, "hash": digest}


def _file_citation(index: _Index, role: str, fid: int) -> dict:
    path, _, _, digest = index.files[fid]
    return {"s": None, "f": fid, "role": role, "name": path, "kind": "file", "sha": index.commit, "hash": digest}


def _is_test(path: str) -> bool:
    base = posixpath.basename(path)
    return base.startswith("test_") or base.endswith(("_test.go", "_test.py")) or base == "conftest.py"


def _page(index: _Index, directory: str) -> _Draft:
    name, sym = index.names[directory], index.symbols
    files = sorted((fid for fid, d in index.folder.items() if d == directory and index.files[fid][1] in GRAMMARS),
                   key=lambda fid: index.files[fid][0])
    own = [sid for sid, s in sym.items() if index.folder[s[0]] == directory]
    language = min(Counter(index.files[sym[s][0]][1] for s in own).items(), key=lambda kv: (-kv[1], kv[0]))[0]
    package = language == "go" or any(posixpath.basename(index.files[f][0]) == "__init__.py" for f in files)
    kinds = sorted(Counter(sym[s][2] for s in own).items(), key=lambda kv: (-kv[1], kv[0]))
    structure = (f"`{name}` is a {LANGUAGE_NAMES.get(language, language)} {'package' if package else 'directory'}"
                 f" of {_plural(len(files), 'file')} and {len(own)} definitions:"
                 f" {_and([_plural(n, kind) for kind, n in kinds])}.")

    def rank(s):  # code before its tests, then what other modules call, then what is exported and documented
        short = sym[s][1].rpartition(".")[2]
        exported = short[:1].isupper() if language == "go" else not short.startswith("_")
        return (_is_test(index.files[sym[s][0]][0]), -index.outside[s], not exported, not sym[s][3],
                -index.total[s], sym[s][1])

    seen: set[str] = set()  # a name defined twice (Go build tags) is listed once, at its best place
    ranked = [s for s in sorted(own, key=rank) if not (sym[s][1] in seen or seen.add(sym[s][1]))]
    types = [s for s in ranked if sym[s][2] in TYPES][:MAX_TYPES]
    keys = types + [s for s in ranked if s not in types][:MAX_KEYS - len(types)]

    base = index.repo if directory == "." else posixpath.basename(directory)
    first = sorted(files, key=lambda f: (posixpath.splitext(posixpath.basename(index.files[f][0]))[0]
                                         not in ("__init__", base), index.files[f][0]))
    purpose = [p for p in (" ".join(_sentences(index.files[f][2])) for f in first) if p][:MAX_PURPOSE]
    if not purpose:
        purpose = [p for p in (" ".join(_sentences(sym[s][3])) for s in types[:2]) if p]

    depends, users, called = defaultdict(set), defaultdict(set), defaultdict(Counter)
    for importer, imported in index.imports:
        if index.folder[importer] == directory and index.folder[imported] != directory:
            depends[index.folder[imported]].add(imported)
        elif index.folder[imported] == directory and index.folder[importer] != directory:
            users[index.folder[importer]].add(importer)
    for caller, callee in index.calls:
        target = index.folder[sym[callee][0]]
        if index.folder[caller] == directory and target in depends:
            called[target][callee] += 1

    citations = [_symbol_citation(index, "key", s) for s in keys] + [_file_citation(index, "file", f) for f in files]
    depends_lines = []
    for other in sorted(depends, key=index.names.get):
        top = sorted(called[other], key=lambda s: (-called[other][s], sym[s][1]))[:MAX_CALLED]
        depends_lines.append(f"- `{index.names[other]}`" + (f": {_ticked(sym[s][1] for s in top)}" if top else ""))
        citations += [_file_citation(index, "depends", f) for f in sorted(depends[other])]
        citations += [_symbol_citation(index, "calls", s) for s in top]
    outside = sorted(set().union(*(index.external.get(f, set()) for f in files)))
    if outside:
        depends_lines.append(f"- Outside the index: {_ticked(outside)}")
    for other in users:
        citations += [_file_citation(index, "uses", f) for f in sorted(users[other])]
    used_by = _ticked(sorted(index.names[u] for u in users))

    lines = [f"# {name}", "", structure, *(purpose or [NO_DOC]), "", "## Key symbols"]
    for s in keys:
        said = _sentences(sym[s][3])
        lines.append(f"- `{sym[s][1]}` ({sym[s][2]}): {said[0] if said else 'no documentation'}")
    lines += ["", "## Depends on", *(depends_lines or ["- none"]), "", "## Used by", f"- {used_by or 'none'}"]
    text = "\n".join(lines) + "\n"
    if len(text) > MAX_TEXT:  # simplified: cut at a line; the citations of what was cut stay, which is harmless
        text = text[:text.rfind("\n", 0, MAX_TEXT) + 1]
    return _Draft(directory, name, language, len(own), text, citations)


def _compile(cur, repo_id: int, repo: str) -> dict[str, _Draft]:
    """A draft page for every qualifying directory of the repository, by directory."""
    index = _load(cur, repo_id, repo)
    counts = Counter(index.folder[s[0]] for s in index.symbols.values())
    qualifying = sorted(d for d, n in counts.items() if n >= THRESHOLD)
    pages = _names(qualifying, repo)
    index.names = {**_names(set(index.folder.values()), repo), **pages}
    return {d: _page(index, d) for d in qualifying}


# --- writing -------------------------------------------------------------------------------------------------


def _stored(cur, repo_id: int) -> dict[str, _Stored]:
    cur.execute("select directory, id, name, text, state, stale_reason from ccctx_wiki_pages where repo_id = :r",
                r=repo_id)
    return {d: _Stored(i, n, _lob(t), s, w) for d, i, n, t, s, w in cur.fetchall()}


def _write_page(cur, repo_id: int, draft: _Draft, old: _Stored | None) -> bool:
    """Store one page as fresh, compiled now, and replace its citations. Whether its text changed.

    An unchanged text is not rewritten, so the page index does not embed it again (research R11).
    """
    search_text = f"page {draft.name} in {draft.directory}\n{draft.text}"
    fields = {"n": draft.name, "l": draft.language, "c": draft.symbols, "p": COMPILER}
    if old is None:
        page = cur.var(int)
        cur.setinputsizes(t=oracledb.DB_TYPE_CLOB, s=oracledb.DB_TYPE_CLOB)
        cur.execute("insert into ccctx_wiki_pages (repo_id, directory, name, language, symbols, state, compiled_at,"
                    " compiler, text, search_text) values (:r, :d, :n, :l, :c, 'fresh', systimestamp, :p, :t, :s)"
                    " returning id into :i", fields | {"r": repo_id, "d": draft.directory, "t": draft.text,
                                                       "s": search_text, "i": page})
        page_id = page.getvalue()[0]
    else:
        page_id = old.id
        cur.execute("update ccctx_wiki_pages set name = :n, language = :l, symbols = :c, state = 'fresh',"
                    " compiled_at = systimestamp, compiler = :p where id = :i", fields | {"i": page_id})
        if old.text != draft.text:
            cur.setinputsizes(t=oracledb.DB_TYPE_CLOB, s=oracledb.DB_TYPE_CLOB)
            cur.execute("update ccctx_wiki_pages set text = :t, search_text = :s where id = :i",
                        t=draft.text, s=search_text, i=page_id)
        cur.execute("delete from ccctx_wiki_citations where page_id = :i", i=page_id)
    seen, rows = set(), []
    for c in draft.citations:  # at most one row per (page, role, record)
        if (c["role"], c["s"], c["f"]) not in seen:
            seen.add((c["role"], c["s"], c["f"]))
            rows.append(c | {"page": page_id})
    cur.executemany("insert into ccctx_wiki_citations (page_id, symbol_id, file_id, role, name, kind, commit_sha,"
                    " content_hash) values (:page, :s, :f, :role, :name, :kind, :sha, :hash)", rows)
    return old is None or old.text != draft.text


def _apply(cur, repo_id: int, drafts: dict[str, _Draft], old: dict[str, _Stored], write) -> tuple[int, int, int]:
    """Delete the pages that no longer qualify, then write the existing pages `write` picks, then the new ones.

    Returns (added, rewritten, removed). Existing pages go first, so a page renamed by a clash frees its name.
    """
    gone = [{"i": o.id} for d, o in old.items() if d not in drafts]
    if gone:
        cur.executemany("delete from ccctx_wiki_pages where id = :i", gone)
    rewritten = 0
    for d in sorted(drafts.keys() & old.keys()):
        if write(drafts[d], old[d]):
            _write_page(cur, repo_id, drafts[d], old[d])
            rewritten += 1
    added = sorted(drafts.keys() - old.keys())
    for d in added:
        _write_page(cur, repo_id, drafts[d], None)
    return len(added), rewritten, len(gone)


def _log(repo: str, started: float, built: int, stale: int, recompiled: int, removed: int) -> None:
    log.info(json.dumps({"event": "wiki", "repo": repo, "built": built, "stale": stale, "recompiled": recompiled,
                         "removed": removed, "ms": round((time.monotonic() - started) * 1000)}))


def _repo(cur, name: str) -> int:
    row = cur.execute("select id from ccctx_repos where name = :n", n=name).fetchone()
    if row is None:
        raise WikiError(f"no repository named '{name}'")
    return row[0]


def _targets(cur, repo: str | None) -> list[tuple[int, str]]:
    """(id, name) of the repositories to build: the named one, or every indexed one, in name order."""
    if repo is not None:
        rid = _repo(cur, repo)
        if not cur.execute("select count(*) from ccctx_files where repo_id = :r and rownum = 1", r=rid).fetchone()[0]:
            raise WikiError(f"repository '{repo}' is not indexed yet; run `ccctx ingest {repo}`")
        return [(rid, repo)]
    rows = cur.execute("select r.id, r.name from ccctx_repos r"
                       " where exists (select 1 from ccctx_files f where f.repo_id = r.id)").fetchall()
    if not rows:
        raise WikiError("no indexed repositories")
    return sorted(rows, key=lambda row: row[1])


def build(conn, repo: str | None = None) -> list[Built]:
    """Compile every qualifying directory of the repository, or of every indexed one. One transaction each."""
    with conn.cursor() as cur:
        targets = _targets(cur, repo)
        ensure_index(cur)
    return [_build(conn, rid, name) for rid, name in targets]


def _build(conn, repo_id: int, repo: str) -> Built:
    started = time.monotonic()
    with conn.cursor() as cur:
        try:
            cur.execute("select id from ccctx_repos where id = :r for update", r=repo_id)  # waits for an ingest
            drafts = _compile(cur, repo_id, repo)
            old = _stored(cur, repo_id)
            updated = sum(old[d].text != drafts[d].text for d in drafts.keys() & old.keys())
            added, rewritten, removed = _apply(cur, repo_id, drafts, old, lambda draft, stored: True)
            cur.execute("update ccctx_repos set wiki_at = nvl(wiki_at, systimestamp) where id = :r", r=repo_id)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    _log(repo, started, len(drafts), 0, rewritten, removed)
    return Built(repo, len(drafts), added, updated, removed, round(time.monotonic() - started, 1))


# --- staleness -----------------------------------------------------------------------------------------------


def enabled(cur, repo_id: int) -> bool:
    """Whether the repository has a wiki: `wiki build` ran for it once."""
    row = cur.execute("select wiki_at from ccctx_repos where id = :r", r=repo_id).fetchone()
    return bool(row and row[0])


def _rewritten(cur, repo_id: int, name: str) -> str | None:
    """The commit of the symbol now carrying a cited name whose record is gone, or None if no symbol has it."""
    row = cur.execute("""select min(s.commit_sha) from ccctx_symbols s join ccctx_files f on f.id = s.file_id
                          where f.repo_id = :r and s.name = :n and s.kind <> 'chunk'""", r=repo_id, n=name).fetchone()
    return row[0]


def mark_stale(cur, repo_id: int) -> list[tuple[str, str]]:
    """Mark stale every fresh page with a symbol citation whose symbol changed commit or is gone, or a file citation
    whose file is gone. Returns (page, reason) in page-name order. Runs inside the caller's transaction."""
    if not enabled(cur, repo_id):
        return []
    rows = cur.execute(
        """select p.id, p.name, c.role, c.name, c.commit_sha, s.commit_sha, c.symbol_id, c.file_id
             from ccctx_wiki_pages p join ccctx_wiki_citations c on c.page_id = p.id
             left join ccctx_symbols s on s.id = c.symbol_id
            where p.repo_id = :r and p.state = 'fresh'
              and ((c.role in ('key', 'calls') and (c.symbol_id is null or s.commit_sha <> c.commit_sha))
                   or (c.role in ('file', 'depends', 'uses') and c.file_id is null))""", r=repo_id).fetchall()
    first: dict[int, tuple[str, str]] = {}
    for pid, page, role, cited, recorded, current, sid, fid in sorted(rows, key=lambda r: (ROLES.index(r[2]), r[3])):
        if pid not in first:
            if role in SYMBOL_ROLES and sid is None:  # an edited symbol is deleted and inserted again
                current = _rewritten(cur, repo_id, cited)
            gone = (current if role in SYMBOL_ROLES else fid) is None
            cited = cited[:400]  # stale_reason holds 512
            if gone:
                reason = f"{cited} is gone"
            elif current == recorded:  # a full re-ingestion rewrote it at the same commit
                reason = f"{cited} was re-indexed"
            else:
                reason = f"{cited} changed ({recorded[:12]} -> {current[:12]})"
            first[pid] = (page, reason)
    if first:
        cur.executemany("update ccctx_wiki_pages set state = 'stale', stale_reason = :w where id = :i",
                        [{"i": pid, "w": reason} for pid, (_, reason) in first.items()])
    return sorted(first.values())


def refresh(conn, repo_id: int) -> Refresh:
    """Recompile every stale page, add pages for directories that now qualify and remove the others, in a
    transaction of its own. Never raises: on failure it rolls back, the pages stay stale and `error` says why."""
    started, stale = time.monotonic(), []
    with conn.cursor() as cur:
        try:
            repo, wiki_at = cur.execute("select name, wiki_at from ccctx_repos where id = :r for update",
                                        r=repo_id).fetchone()
            if wiki_at is None:
                conn.rollback()
                return Refresh([])
            old = _stored(cur, repo_id)
            stale = sorted((o.name, o.reason) for o in old.values() if o.state == "stale")
            drafts = _compile(cur, repo_id, repo)
            added, recompiled, removed = _apply(cur, repo_id, drafts, old,
                                                lambda draft, page: page.state == "stale" or page.name != draft.name)
            conn.commit()
        except Exception as e:
            conn.rollback()
            return Refresh(stale, error=db.describe(e) if isinstance(e, oracledb.Error) else str(e))
    _log(repo, started, added + recompiled, len(stale), recompiled, removed)
    return Refresh(stale, recompiled, added, removed)


# --- reading -------------------------------------------------------------------------------------------------


def _summary(text: str) -> str:
    lines = text.split("\n")
    purpose = lines[3] if len(lines) > 3 and lines[3] != NO_DOC else lines[2] if len(lines) > 2 else ""
    return SENTENCE_END.split(purpose, maxsplit=1)[0]


def pages(conn, repo: str | None = None) -> list[PageInfo]:
    """Every page, or one repository's, by repository then page name."""
    with conn.cursor() as cur:
        rid = _repo(cur, repo) if repo is not None else None
        cur.execute("""select p.name, r.name, p.directory, p.language, p.state, p.symbols, p.compiled_at,
                              p.stale_reason, p.text
                         from ccctx_wiki_pages p join ccctx_repos r on r.id = p.repo_id
                        where :r is null or p.repo_id = :r""", r=rid)
        rows = [(*row[:-1], _summary(_lob(row[-1]))) for row in cur.fetchall()]
    return sorted((PageInfo(*row) for row in rows), key=lambda p: (p.repo, p.name))


def read(conn, page: str, repo: str | None = None) -> Page:
    """One page with its citations, each located through its key. A name in several repositories needs `repo`."""
    with conn.cursor() as cur:
        rid = _repo(cur, repo) if repo is not None else None
        cur.execute("""select p.id, p.name, r.name, p.directory, p.language, p.state, p.compiled_at, p.compiler,
                              p.stale_reason, p.symbols, p.text
                         from ccctx_wiki_pages p join ccctx_repos r on r.id = p.repo_id
                        where p.name = :n and (:r is null or p.repo_id = :r)""", n=page, r=rid)
        found = [(*row[:-1], _lob(row[-1])) for row in cur.fetchall()]
        if not found:
            raise WikiError(f"no wiki page named '{page}'" + (f" in '{repo}'" if repo is not None else ""))
        if len(found) > 1:
            names = ", ".join(sorted(row[2] for row in found))
            raise WikiError(f"wiki page '{page}' is in more than one repository: {names}; pass --repo")
        page_id, *fields = found[0]
        cur.execute("""select c.name, c.kind, c.role, nvl(sr.name, fr.name), nvl(sf.path, f.path), s.start_line,
                              s.end_line, c.commit_sha,
                              case when c.role in ('key', 'calls') then s.commit_sha else fr.indexed_commit end
                         from ccctx_wiki_citations c
                         left join ccctx_symbols s on s.id = c.symbol_id
                         left join ccctx_files sf on sf.id = s.file_id
                         left join ccctx_repos sr on sr.id = sf.repo_id
                         left join ccctx_files f on f.id = c.file_id
                         left join ccctx_repos fr on fr.id = f.repo_id
                        where c.page_id = :p""", p=page_id)
        citations = sorted((CitationView(*row) for row in cur.fetchall()),
                           key=lambda c: (ROLES.index(c.role), c.name))
    return Page(*fields, citations[:MAX_CITATIONS], len(citations) > MAX_CITATIONS)
