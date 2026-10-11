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

from golestan_code_mcp import chunker, semantic
from golestan_code_mcp.config import environment, max_file_bytes, vendor_dirs
from golestan_code_mcp.registry import get_repo

log = logging.getLogger("golestan_code_mcp.ingest")

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

    @property
    def scanned(self) -> int:
        return self.files + sum(self.skipped.values())


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *GIT_SAFE, "-C", str(root), *args], capture_output=True)
    if done.returncode:
        reason = done.stderr.decode(errors="replace").strip().splitlines()
        raise IngestError(f"git {args[0]} failed in {root}: {reason[0] if reason else done.returncode}")
    return done.stdout.decode("utf-8", "surrogateescape")


def _ls(root: Path, *flags: str) -> list[str]:
    return list(dict.fromkeys(p for p in git(root, "ls-files", "-z", *flags).split("\0") if p))


def discover(root: Path, env: Mapping[str, str]) -> tuple[dict[str, str], Counter]:
    """Text of every file worth indexing, keyed by its repository-relative path, and skip counts per reason."""
    limit, vendored = max_file_bytes(env), vendor_dirs(env)
    skipped = Counter(ignored=len(_ls(root, "--others", "--ignored", "--exclude-standard")))
    files = {}
    for rel in _ls(root, "--cached", "--others", "--exclude-standard"):
        path = root / rel
        if vendored & set(PurePosixPath(rel).parts[:-1]):
            skipped["vendored"] += 1
        elif path.is_symlink():
            skipped["ignored"] += 1
        elif not path.is_file():  # tracked but deleted from the working tree
            continue
        elif path.stat().st_size > limit:
            skipped["too large"] += 1
        else:
            data = path.read_bytes()
            try:
                if b"\0" in data:
                    raise UnicodeDecodeError("utf-8", data, 0, 1, "NUL byte")
                files[rel] = data.decode("utf-8")
            except UnicodeDecodeError:
                skipped["binary"] += 1
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


def _store(conn, name: str, repo_id: int, commit: str, prepared: list[tuple]) -> int:
    """Replace the repository's files and symbols in one transaction. Returns the number of symbols stored."""
    with conn.cursor() as cur:
        try:
            try:
                cur.execute("select id from ccctx_repos where id = :i for update nowait", i=repo_id)
            except oracledb.DatabaseError as e:
                if getattr(e.args[0], "full_code", "") == "ORA-00054":
                    raise IngestError(f"another ingest of '{name}' is running") from None
                raise
            cur.execute("delete from ccctx_files where repo_id = :i", i=repo_id)
            rows = []
            for rel, language, text, chunks in prepared:
                file_id = cur.var(int)
                cur.execute(
                    "insert into ccctx_files (repo_id, path, language, content_hash)"
                    " values (:r, :p, :l, :h) returning id into :i",
                    r=repo_id, p=rel, l=language, h=_sha(text), i=file_id,
                )
                fid = file_id.getvalue()[0]
                rows += [
                    dict(file_id=fid, name=c.name[:512], kind=c.kind, start_line=c.start_line, end_line=c.end_line,
                         source=c.source, content_hash=_sha(c.source), commit_sha=commit, path=rel)
                    for c in chunks
                ]
            _insert_symbols(cur, rows)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return len(rows)


def ingest(conn, name: str, env: Mapping[str, str] | None = None) -> Summary:
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

    files, skipped = discover(root, env)
    prepared, no_grammar, parse_errors = [], 0, 0
    for rel, text in sorted(files.items()):
        language = chunker.detect_language(rel)
        items = chunker.chunk(language, text)
        if items is None:
            if language in chunker.GRAMMARS:
                parse_errors += 1
            else:
                no_grammar += 1
            items = chunker.windows(PurePosixPath(rel).name, text)
        prepared.append((rel, language, text, items))

    symbols = _store(conn, name, repo_id, commit, prepared)
    summary = Summary(name, commit, dirty, skipped, len(files), no_grammar, parse_errors, symbols,
                      round(time.monotonic() - started, 1))
    log.info(json.dumps({
        "event": "ingest", "repo": name, "commit": commit, "dirty": dirty, "scanned": summary.scanned,
        "skipped": dict(skipped), "files": summary.files, "no_grammar": no_grammar,
        "parse_errors": parse_errors, "symbols": symbols, "seconds": summary.seconds,
    }))
    return summary
