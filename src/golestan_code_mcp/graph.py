"""Code graph: resolve each file's stored call and import refs to edges, and answer callers, calls and impact
with GRAPH_TABLE ... MATCH queries over ccctx_graph.

Resolution is by name within one repository, guided by the call form and the file's imports. It is not type
inference, so dynamic calls and calls through variables are not edges.
"""

import posixpath
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath

from golestan_code_mcp.registry import repo_id as registered

MAX_DEPTH = 5
SELF = {"self", "cls", "this"}
BARE_KINDS = {"function", "class", "type"}  # what a bare call x(...) can reach; type is Go's conversion
JS_SUFFIXES = ("", ".js", ".mjs", ".cjs", ".jsx", "/index.js")


def _python_index(paths: dict[str, int]) -> dict[str, list[int]]:
    """Every Python file under each path suffix that starts at a source root (a directory that is no package).

    `src/flask/helpers.py` is found as `flask/helpers.py` but not as `helpers.py`, so `import json` never
    reaches `src/flask/json/__init__.py`.
    """
    packages = {str(PurePosixPath(p).parent) for p in paths if p.endswith("__init__.py")}
    index: dict[str, list[int]] = {}
    for path, fid in paths.items():
        if path.endswith(".py"):
            parts = path.split("/")
            for i in range(len(parts)):
                if i == 0 or "/".join(parts[:i]) not in packages:
                    index.setdefault("/".join(parts[i:]), []).append(fid)
    return index


def _python(module: str, level: int, importer: str, paths: dict[str, int],
            index: dict[str, list[int]]) -> tuple[list[int], bool]:
    """The files a Python import names, and whether it names a module (not a member of one)."""
    names = [n for n in module.split(".") if n]
    dirs = list(PurePosixPath(importer).parent.parts)
    if level - 1 > len(dirs):
        return [], False
    base = dirs[:len(dirs) - level + 1] if level else []
    for stem, is_module in ((names, True), (names[:-1], False)):  # the module itself, then its parent package
        if not stem and not level:
            continue
        for candidate in ([f"{'/'.join(base + stem)}.py"] if stem else []) + ["/".join(base + stem + ["__init__.py"])]:
            found = ([paths[candidate]] if candidate in paths else []) if level else index.get(candidate, [])
            if found:
                return found, is_module
    return [], False


def _go_module(modules: set[str], dirs: set[str]) -> str | None:
    """The repository's Go module path: what its own package imports have in front of a repository directory."""
    votes = Counter(m[:-len(d) - 1] for m in modules for d in dirs if d and m.endswith("/" + d))
    return votes.most_common(1)[0][0] if votes else None


def _imports(files: dict[int, tuple[str, str]], refs: list[tuple]) -> tuple[dict[int, set[int]], dict[int, set[str]]]:
    """Per importing file: the files it imports, and the names it binds to a module of this repository."""
    paths = {path: fid for fid, (path, _) in files.items()}
    index = _python_index(paths)
    go_dirs: dict[str, list[int]] = {}
    for fid, (path, language) in files.items():
        if language == "go":
            go_dirs.setdefault(posixpath.dirname(path), []).append(fid)
    go_module = _go_module({m for fid, m, _, _ in refs if files[fid][1] == "go"}, set(go_dirs))

    imported: dict[int, set[int]] = {}
    modules: dict[int, set[str]] = {}
    for fid, module, level, bound in refs:
        path, language = files[fid]
        found, is_module = [], True
        if language == "python":
            found, is_module = _python(module, level, path, paths, index)
        elif language == "go" and go_module and (module == go_module or module.startswith(go_module + "/")):
            found = go_dirs.get(module[len(go_module) + 1:], [])
        elif language == "javascript":
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), module))
            found = next(([paths[target + s]] for s in JS_SUFFIXES if target + s in paths), [])
        found = [f for f in found if f != fid]
        if found:
            imported.setdefault(fid, set()).update(found)
            if is_module and bound:
                modules.setdefault(fid, set()).add(bound)
    return imported, modules


def _resolve(call: tuple, symbols: dict[int, tuple], by_name: dict[str, list[int]], by_method: dict[str, list[int]],
             imported: dict[int, set[int]], modules: dict[int, set[str]]) -> list[int]:
    """The symbols a call reaches, by the four rules, then the same-file, imported-file, whole-repository order."""
    fid, caller_id, name, form, receiver, _ = call
    head = (receiver or "").split(".")[0]
    if form == "bare" or head in modules.get(fid, ()):  # rules 1 and 2
        matches = [s for s in by_name.get(name, []) if symbols[s][2] in BARE_KINDS]
    else:  # rules 3 and 4
        matches = by_method.get(name, [])
        if receiver in SELF and caller_id in symbols:
            caller, kind = symbols[caller_id][1:3]
            enclosing = caller if kind == "class" else caller.rpartition(".")[0]
            matches = [s for s in matches if symbols[s][1] == f"{enclosing}.{name}"] or matches
    for scope in ({fid}, imported.get(fid, set())):
        if near := [s for s in matches if symbols[s][0] in scope]:
            return near
    return matches


def rebuild(cur, repo_id: int) -> None:
    """Delete and rebuild the repository's import and call edges from its stored refs and symbols."""
    # simplified: whole repository rebuilt each ingest, rebuild only the affected names when it shows up in timings
    r = {"r": repo_id}
    files = {fid: (path, language) for fid, path, language in cur.execute(
        "select id, path, language from ccctx_files where repo_id = :r", r)}
    symbols = {sid: (fid, name, kind) for sid, fid, name, kind in cur.execute(
        "select s.id, s.file_id, s.name, s.kind from ccctx_symbols s join ccctx_files f on f.id = s.file_id"
        " where f.repo_id = :r and s.kind <> 'chunk'", r)}
    import_refs = cur.execute(
        "select i.file_id, i.module, i.rel_level, i.bound from ccctx_import_refs i"
        " join ccctx_files f on f.id = i.file_id where f.repo_id = :r", r).fetchall()
    call_refs = cur.execute(
        "select c.file_id, c.caller_id, c.name, c.form, c.receiver, c.line from ccctx_call_refs c"
        " join ccctx_files f on f.id = c.file_id where f.repo_id = :r", r).fetchall()

    by_name: dict[str, list[int]] = {}
    by_method: dict[str, list[int]] = {}
    for sid, (_, name, kind) in symbols.items():
        by_name.setdefault(name, []).append(sid)
        if kind == "method":
            by_method.setdefault(name.rpartition(".")[2], []).append(sid)
    imported, modules = _imports(files, import_refs)

    calls, file_calls = [], []
    for call in call_refs:
        for callee in _resolve(call, symbols, by_name, by_method, imported, modules):
            if call[1] is None:
                file_calls.append({"f": call[0], "c": callee, "l": call[5]})
            else:
                calls.append({"f": call[1], "c": callee, "l": call[5]})

    cur.execute("delete from ccctx_imports where importer_id in (select id from ccctx_files where repo_id = :r)", r)
    cur.execute("delete from ccctx_file_calls where file_id in (select id from ccctx_files where repo_id = :r)", r)
    cur.execute("delete from ccctx_calls where caller_id in (select s.id from ccctx_symbols s"
                " join ccctx_files f on f.id = s.file_id where f.repo_id = :r)", r)
    pairs = [{"a": a, "b": b} for a, targets in imported.items() for b in targets]
    if pairs:
        cur.executemany("insert into ccctx_imports (importer_id, imported_id) values (:a, :b)", pairs)
    if calls:
        cur.executemany("insert into ccctx_calls (caller_id, callee_id, line) values (:f, :c, :l)", calls)
    if file_calls:
        cur.executemany("insert into ccctx_file_calls (file_id, callee_id, line) values (:f, :c, :l)", file_calls)


@dataclass(frozen=True)
class Caller:
    repo: str
    kind: str  # 'file' for a module-level call
    name: str  # '(module)' for a module-level call
    path: str
    line: int
    callee: str  # the qualified name of the definition called


@dataclass(frozen=True)
class Callee:
    caller: str  # the matched definition making the call
    line: int
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Impact:
    hops: int
    repo: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


JOIN = ("join ccctx_symbols s on s.id = g.{} join ccctx_files f on f.id = s.file_id"
        " join ccctx_repos r on r.id = f.repo_id")


def check_depth(depth) -> int:
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}")
    return depth


def _targets(conn, cur, name: str, repos: list[str] | None) -> tuple[dict[int, str], set[int] | None]:
    """id -> qualified name of the definitions matching the name in the chosen repositories, and those repo ids.

    Case-sensitive. A qualified name matches exactly; a bare name matches x and *.x.
    """
    wanted = {registered(conn, n) for n in repos} if repos else None  # unknown names refused first
    cur.execute(
        """select s.id, f.repo_id, s.name from ccctx_symbols s join ccctx_files f on f.id = s.file_id
            where s.kind <> 'chunk'
              and (s.name = :n or (:bare = 1 and substr(s.name, -length(:n) - 1) = '.' || :n))""",
        n=name, bare=int("." not in name),
    )
    targets = {sid: qualified for sid, rid, qualified in cur if wanted is None or rid in wanted}
    if not targets:
        raise ValueError(f"no symbol named '{name}'")
    return targets, wanted


def check_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("the name is empty")
    return name


def _match(cur, sql: str, ids: list[int]) -> list[tuple]:
    """Run a graph query whose `{ids}` is the target id list, 500 ids at a time (Oracle allows 1000)."""
    rows = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        cur.execute(sql.replace("{ids}", ", ".join(f":{n + 1}" for n in range(len(chunk)))), chunk)
        rows += cur.fetchall()
    return rows


def callers(conn, name: str, repos: list[str] | None = None) -> list[Caller]:
    """Every call to a definition matching the name, by repository, path, then line."""
    name = check_name(name)
    with conn.cursor() as cur:
        targets, wanted = _targets(conn, cur, name, repos)
        found = _match(cur, f"""select r.id, r.name, s.kind, s.name, f.path, g.line, g.callee_id
              from graph_table (ccctx_graph
                match (c is symbol) -[e is calls]-> (t is symbol)
                where t.id in ({{ids}})
                columns (c.id as caller_id, e.line as line, t.id as callee_id)) g {JOIN.format("caller_id")}""",
                       list(targets))
        found += _match(cur, """select r.id, r.name, 'file', '(module)', f.path, g.line, g.callee_id
              from graph_table (ccctx_graph
                match (c is code_file) -[e is calls]-> (t is symbol)
                where t.id in ({ids})
                columns (c.id as file_id, e.line as line, t.id as callee_id)) g
              join ccctx_files f on f.id = g.file_id join ccctx_repos r on r.id = f.repo_id""", list(targets))
    rows = [Caller(repo, kind, caller, path, line, targets[callee])
            for rid, repo, kind, caller, path, line, callee in found if wanted is None or rid in wanted]
    return sorted(rows, key=lambda c: (c.repo, c.path, c.line, c.name, c.callee))


def calls(conn, name: str, repos: list[str] | None = None) -> list[Callee]:
    """Every indexed symbol called from a definition matching the name, by calling definition, then line."""
    name = check_name(name)
    with conn.cursor() as cur:
        targets, wanted = _targets(conn, cur, name, repos)
        found = _match(cur, f"""select g.caller_id, g.line, r.id, r.name, s.kind, s.name, f.path, s.start_line,
                     s.end_line
              from graph_table (ccctx_graph
                match (t is symbol) -[e is calls]-> (c is symbol)
                where t.id in ({{ids}})
                columns (t.id as caller_id, e.line as line, c.id as callee_id)) g {JOIN.format("callee_id")}""",
                       list(targets))
    rows = [Callee(targets[caller], line, repo, *rest)
            for caller, line, rid, repo, *rest in found if wanted is None or rid in wanted]
    return sorted(rows, key=lambda c: (c.caller, c.line, c.repo, c.path, c.name))


def impact(conn, name: str, depth: int = 3, repos: list[str] | None = None) -> list[Impact]:
    """Every symbol reaching a definition matching the name in at most `depth` call hops, once, nearest first."""
    name = check_name(name)
    depth = check_depth(depth)  # written into the SQL: a quantifier cannot be a bind variable
    # simplified: {1,N} enumerates every path, so a heavily called name costs seconds at depth 3 (gin New: 2.4 s);
    # walk one hop per level from the frontier if that binds
    with conn.cursor() as cur:
        targets, wanted = _targets(conn, cur, name, repos)
        found = _match(cur, f"""select g.hops, r.id, r.name, s.kind, s.name, f.path, s.start_line, s.end_line, s.id
              from (select id, min(hops) as hops
                      from graph_table (ccctx_graph
                        match (c is symbol) -[e is calls]->{{1,{depth}}} (t is symbol)
                        where t.id in ({{ids}})
                        columns (c.id as id, count(e.line) as hops))
                     group by id) g {JOIN.format("id")}""", list(targets))
    nearest: dict[int, tuple] = {}
    for row in found:  # a symbol may come back from several 500-id batches
        if (wanted is None or row[1] in wanted) and (row[-1] not in nearest or row[0] < nearest[row[-1]][0]):
            nearest[row[-1]] = row
    rows = [Impact(hops, repo, *rest) for hops, _, repo, *rest, _ in nearest.values()]
    return sorted(rows, key=lambda i: (i.hops, i.repo, i.path, i.start_line, i.name))
