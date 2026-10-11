"""Ingest a registered repository: list files with read-only git, chunk them, replace the index in one transaction.

Files are only read. Git runs without a shell, without optional locks and with the repository's fsmonitor
and hooks disabled, so nothing configured inside the repository can run.
"""

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import oracledb

from golestan_code_mcp import chunker, graph, semantic
from golestan_code_mcp.config import environment, max_file_bytes, vendor_dirs
from golestan_code_mcp.registry import get_repo

log = logging.getLogger("golestan_code_mcp.ingest")

EXTRACT_VERSION = 1  # the call and import extraction rules; a repository stored under others is re-read
GIT_SAFE = ["-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", "--no-optional-locks"]


class IngestError(Exception):
    """Refused ingest. The message is safe to show the user."""


@dataclass
class Summary:
    repo: str
    commit: str
    dirty: bool
    skipped: Counter
    files: int
    no_grammar: int
    parse_errors: int
    symbols: int
    seconds: float
    mode: str = "full"  # full, incremental, or current (already up to date, nothing written)
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    note: str | None = None  # why a run without --full rebuilt in full

    @property
    def scanned(self) -> int:
        return self.files + sum(self.skipped.values())


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *GIT_SAFE, "-C", str(root), *args], capture_output=True)
    if done.returncode:
        reason = done.stderr.decode(errors="replace").strip().splitlines()
        raise IngestError(f"git {args[0]} failed in {root}: {reason[0] if reason else done.returncode}")
    return done.stdout.decode("utf-8", "surrogateescape")


def _paths(root: Path, *args: str) -> list[str]:
    """Paths printed NUL-separated (-z) by a git command, de-duplicated in order."""
    return list(dict.fromkeys(p for p in git(root, *args).split("\0") if p))


def _ls(root: Path, *flags: str) -> list[str]:
    return _paths(root, "ls-files", "-z", *flags)


def _read(root: Path, rel: str, limit: int, vendored: set[str]) -> tuple[str | None, str | None]:
    """(text, None) for a file worth indexing, (None, reason) for a skipped one, (None, None) for a gone one."""
    path = root / rel
    if vendored & set(PurePosixPath(rel).parts[:-1]):
        return None, "vendored"
    if path.is_symlink():
        return None, "ignored"
    if not path.is_file():  # tracked but deleted from the working tree
        return None, None
    if path.stat().st_size > limit:
        return None, "too large"
    data = path.read_bytes()
    if b"\0" in data:
        return None, "binary"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "binary"


def discover(root: Path, env: Mapping[str, str], only: list[str] | None = None) -> tuple[dict[str, str], Counter]:
    """Text of every file worth indexing, keyed by its repository-relative path, and skip counts per reason.

    `only` limits the read to those listed paths; ignored files are then not counted.
    """
    limit, vendored = max_file_bytes(env), vendor_dirs(env)
    if only is None:
        skipped = Counter(ignored=len(_ls(root, "--others", "--ignored", "--exclude-standard")))
        only = _ls(root, "--cached", "--others", "--exclude-standard")
    else:
        skipped = Counter()
    files = {}
    for rel in only:
        text, reason = _read(root, rel, limit, vendored)
        if reason:
            skipped[reason] += 1
        elif text is not None:
            files[rel] = text
    return files, +skipped


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _insert_symbols(cur, rows: list[dict]) -> None:
    """Store each symbol with its searchable text; the hybrid index vectorizes it when the ingest commits."""
    if not rows:
        return
    cur.setinputsizes(search_text=oracledb.DB_TYPE_CLOB, source=oracledb.DB_TYPE_CLOB)
    # The two LOB binds must come last, or long sources fail with ORA-24816.
    cur.executemany(
        "insert into ccctx_symbols (file_id, name, kind, start_line, end_line, content_hash, commit_sha,"
        " search_text, source) values (:file_id, :name, :kind, :start_line, :end_line, :content_hash,"
        " :commit_sha, :search_text, :source)",
        [{k: v for k, v in row.items() if k != "path"}
         | {"search_text": f"{row['kind']} {row['name']} in {row['path']}\n{row['source']}"} for row in rows],
    )


def _lock(cur, name: str, repo_id: int) -> tuple[str | None, bool | None, int | None, dict[str, str]]:
    """Lock the repository row; return its recorded revision, dirty flag, extract version and path to hash map."""
    try:
        cur.execute("select id from ccctx_repos where id = :i for update nowait", i=repo_id)
    except oracledb.DatabaseError as e:
        if getattr(e.args[0], "full_code", "") == "ORA-00054":
            raise IngestError(f"another ingest of '{name}' is running") from None
        raise
    recorded, was_dirty, version = cur.execute(
        "select indexed_commit, indexed_dirty, extract_version from ccctx_repos where id = :i", i=repo_id).fetchone()
    stored = dict(cur.execute("select path, content_hash from ccctx_files where repo_id = :i", i=repo_id))
    return recorded, was_dirty, version, stored


def _full_reason(root: Path, name: str, recorded: str | None, stored: dict) -> tuple[bool, str | None]:
    """Whether an ingest without --full must rebuild in full, and the note saying why."""
    if recorded is None:
        return True, f"no recorded revision for '{name}'; indexing in full" if stored else None
    try:
        git(root, "rev-parse", "--verify", "--quiet", f"{recorded}^{{commit}}")
    except IngestError:
        return True, f"last indexed revision {recorded[:12]} is no longer in the repository; indexing in full"
    return False, None


def _candidates(root: Path, recorded: str, was_dirty: bool) -> tuple[list[str], set[str]]:
    """Paths that may differ from the index, and every path git lists now."""
    listed = set(_ls(root, "--cached", "--others", "--exclude-standard"))
    if was_dirty:  # an uncommitted edit was indexed: git cannot tell whether it was reverted since
        return sorted(listed), listed
    changed = _paths(root, "diff", "--name-only", "-z", "--no-renames", recorded, "--")
    untracked = _ls(root, "--others", "--exclude-standard")
    return sorted(set(changed + untracked) & listed), listed


def _prepare(files: dict[str, str]) -> tuple[list[tuple], int, int]:
    prepared, no_grammar, parse_errors = [], 0, 0
    for rel, text in sorted(files.items()):
        language = chunker.detect_language(rel)
        parsed = chunker.parse(language, text)
        if parsed is None:  # line windows have no calls or imports
            if language in chunker.GRAMMARS:
                parse_errors += 1
            else:
                no_grammar += 1
            parsed = chunker.windows(PurePosixPath(rel).name, text), [], []
        prepared.append((rel, language, text, *parsed))
    return prepared, no_grammar, parse_errors


def _keep_unchanged(cur, file_id: int, commit: str, rows: list[dict]) -> list[dict]:
    """Keep a changed file's symbols whose code is identical, at their new lines; return the rows to insert.

    A kept symbol keeps its search text, so the hybrid index does not embed it again.
    """
    old: dict[tuple, list[int]] = {}
    for sid, name, kind, digest in cur.execute(
            "select id, name, kind, content_hash from ccctx_symbols where file_id = :f", f=file_id):
        old.setdefault((name, kind, digest), []).append(sid)
    moved, new = [], []
    for row in rows:
        ids = old.get((row["name"], row["kind"], row["content_hash"]))
        if ids:
            moved.append({"i": ids.pop(), "s": row["start_line"], "e": row["end_line"], "c": commit})
        else:
            new.append(row)
    stale = [{"i": sid} for ids in old.values() for sid in ids]
    if stale:
        cur.executemany("delete from ccctx_symbols where id = :i", stale)
    if moved:
        cur.executemany("update ccctx_symbols set start_line = :s, end_line = :e, commit_sha = :c where id = :i",
                        moved)
    return new


def _write(cur, repo_id: int, commit: str, dirty: bool, prepared: list[tuple], delete: list[str] | None,
           changed: set[str]) -> int:
    """Delete the given paths (all of them for None), store the prepared files and their refs, record the
    revision and rebuild the repository's edges.

    Files in `changed` are updated in place. Returns the number of symbols inserted, which the index embeds.
    """
    if delete is None:
        cur.execute("delete from ccctx_files where repo_id = :i", i=repo_id)
    elif delete:
        cur.executemany("delete from ccctx_files where repo_id = :r and path = :p",
                        [{"r": repo_id, "p": rel} for rel in delete])  # symbols go with them (on delete cascade)
    rows, fids = [], {}
    for rel, language, text, chunks, _, _ in prepared:
        file_id = cur.var(int)
        if rel in changed:
            cur.execute("update ccctx_files set language = :l, content_hash = :h where repo_id = :r and path = :p"
                        " returning id into :i", r=repo_id, p=rel, l=language, h=_sha(text), i=file_id)
        else:
            cur.execute(
                "insert into ccctx_files (repo_id, path, language, content_hash)"
                " values (:r, :p, :l, :h) returning id into :i",
                r=repo_id, p=rel, l=language, h=_sha(text), i=file_id,
            )
        fid = fids[rel] = file_id.getvalue()[0]
        file_rows = [
            dict(file_id=fid, name=c.name[:512], kind=c.kind, start_line=c.start_line, end_line=c.end_line,
                 source=c.source, content_hash=_sha(c.source), commit_sha=commit, path=rel)
            for c in chunks
        ]
        rows += _keep_unchanged(cur, fid, commit, file_rows) if rel in changed else file_rows
    _insert_symbols(cur, rows)
    _insert_refs(cur, repo_id, prepared, fids, changed)
    cur.execute("update ccctx_repos set indexed_commit = :c, indexed_dirty = :d, indexed_at = systimestamp,"
                " extract_version = :v where id = :i", c=commit, d=dirty, v=EXTRACT_VERSION, i=repo_id)
    graph.rebuild(cur, repo_id)
    return len(rows)


def _insert_refs(cur, repo_id: int, prepared: list[tuple], fids: dict[str, int], changed: set[str]) -> None:
    """Replace the call and import refs of every file read, each caller mapped to its kept or inserted symbol."""
    gone = [{"f": fids[rel]} for rel in changed if rel in fids]
    if gone:
        cur.executemany("delete from ccctx_call_refs where file_id = :f", gone)
        cur.executemany("delete from ccctx_import_refs where file_id = :f", gone)
    ids = {(fid, name): sid for sid, fid, name in cur.execute(
        "select s.id, s.file_id, s.name from ccctx_symbols s join ccctx_files f on f.id = s.file_id"
        " where f.repo_id = :r and s.kind <> 'chunk'", r=repo_id)}
    calls, imports = [], []
    for rel, _, _, _, file_calls, file_imports in prepared:
        fid = fids[rel]
        calls += [{"f": fid, "c": ids.get((fid, c.caller[:512])) if c.caller else None, "n": c.name[:512],
                   "o": c.form, "r": c.receiver and c.receiver[:512], "l": c.line} for c in file_calls]
        imports += [{"f": fid, "m": i.module[:1024], "v": i.rel_level, "b": i.bound and i.bound[:512]}
                    for i in file_imports]
    if calls:
        cur.executemany("insert into ccctx_call_refs (file_id, caller_id, name, form, receiver, line)"
                        " values (:f, :c, :n, :o, :r, :l)", calls)
    if imports:
        cur.executemany("insert into ccctx_import_refs (file_id, module, rel_level, bound) values (:f, :m, :v, :b)",
                        imports)


def ingest(conn, name: str, env: Mapping[str, str] | None = None, full: bool = False) -> Summary:
    """Index what changed since the last ingest, or everything with full=True. One transaction either way."""
    started = time.monotonic()
    env = environment() if env is None else env
    semantic.require_model(conn)
    repo_id, root = get_repo(conn, name)
    try:
        git(root, "rev-parse", "--is-inside-work-tree")
    except IngestError:
        raise IngestError(f"{root} is not a git repository") from None
    try:
        commit = git(root, "rev-parse", "--verify", "HEAD").strip()
    except IngestError:
        raise IngestError(f"{root} has no commits yet") from None
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no").strip())

    with conn.cursor() as cur:
        try:
            recorded, was_dirty, version, stored = _lock(cur, name, repo_id)
            stale = version != EXTRACT_VERSION  # refs from older rules, or none: re-read every listed file
            note = None
            if not full:
                full, note = _full_reason(root, name, recorded, stored)
            if full:
                files, skipped = discover(root, env)
                delete, removed = None, []
                mode, added, changed, unchanged = "full", len(files), 0, 0
            else:
                candidates, listed = _candidates(root, recorded, bool(was_dirty) or stale)
                readable, skipped = discover(root, env, candidates)
                # gone from git's listing (deleted, renamed away, newly ignored), or now gone or skipped on disk
                removed = sorted({rel for rel in stored if rel not in listed}
                                 | {rel for rel in candidates if rel in stored and rel not in readable})
                files = {rel: text for rel, text in readable.items() if stale or stored.get(rel) != _sha(text)}
                changed = sum(rel in stored for rel in files)
                added, unchanged = len(files) - changed, len(stored) - changed - len(removed)
                delete = removed
                current = not stale and commit == recorded and not files and not removed and bool(was_dirty) == dirty
                mode = "current" if current else "incremental"
            prepared, no_grammar, parse_errors = _prepare(files)
            if mode == "current":
                conn.rollback()  # nothing to write; releases the lock
                symbols = 0
            else:
                symbols = _write(cur, repo_id, commit, dirty, prepared, delete,
                                 set() if delete is None else stored.keys() & files.keys())
                conn.commit()
        except BaseException:
            conn.rollback()
            raise

    summary = Summary(name, commit, dirty, skipped, len(files), no_grammar, parse_errors, symbols,
                      round(time.monotonic() - started, 1), mode, added, changed, len(removed), unchanged, note)
    log.info(json.dumps({
        "event": "ingest", "repo": name, "commit": commit, "dirty": dirty, "scanned": summary.scanned,
        "skipped": dict(skipped), "files": summary.files, "no_grammar": no_grammar,
        "parse_errors": parse_errors, "symbols": symbols, "seconds": summary.seconds,
        "mode": mode, "added": added, "changed": changed, "removed": len(removed), "unchanged": unchanged,
    }))
    return summary
