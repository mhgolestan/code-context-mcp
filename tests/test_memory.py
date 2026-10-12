"""Agent memory (010): runs, facts, consolidation, re-ask and recall, through `memory` against the local database."""

import subprocess
import sys
import time
import uuid

import pytest

from golestan_code_mcp import db, memory, schema
from golestan_code_mcp.config import load


@pytest.fixture(scope="module")
def conn():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    schema.migrate(conn)
    yield conn
    conn.close()


# --- the package client (T006) ---


def test_client_requires_the_migrated_store(conn):
    missing = f"ZZ{uuid.uuid4().hex[:8].upper()}"
    with pytest.raises(ValueError, match="ccctx db migrate"):
        memory.client(conn, store_id=missing)
    with conn.cursor() as cur:
        assert cur.execute("select count(*) from user_tables where table_name like :p",
                           p=f"{missing}%").fetchone()[0] == 0


def test_client_writes_reads_and_searches_facts(conn):
    repo = f"t-{uuid.uuid4().hex[:8]}"
    mem = memory.client(conn)
    try:
        fid = mem.add_memory("`spin_widget` spins a widget.", memory_type="fact", user_id=repo, agent_id=memory.AGENT,
                             metadata={"subject_key": "k"}, ttl_days=None, autonomous_linking=False)
        assert [r.id for r in mem.list_memories(user_id=repo, agent_id=memory.AGENT, limit=None)] == [fid]
        hits = mem.search("how do widgets spin", user_id=repo, agent_id=memory.AGENT, record_types=["fact"],
                          max_results=5)
        assert [h.id for h in hits] == [fid]
    finally:
        mem.delete_user(repo, cascade=True)
    assert mem.list_memories(user_id=repo, limit=None) == []


def test_no_model_configured_means_no_llm():
    assert memory.model({}) is None
    assert memory.model({"CCCTX_MEMORY_MODEL": " "}) is None


def test_configured_model_gets_a_package_llm():
    llm, name = memory.model({"CCCTX_MEMORY_MODEL": "ollama/qwen3"})
    assert name == "llm:ollama/qwen3"
    assert type(llm).__module__.startswith("oracleagentmemory")


def test_importing_memory_does_not_import_the_package():
    code = "import sys, golestan_code_mcp.memory, golestan_code_mcp.server; print('oracleagentmemory' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# --- building runs (T008): no database ---

from golestan_code_mcp import server as srv  # noqa: E402

HIT = dict(rank=1, score=40.0, similarity=0.6, keyword_rank=1, semantic_rank=2, repo="flask", kind="function",
           name="url_for", path="src/flask/helpers.py", start_line=188, end_line=246, language="python",
           snippet="def url_for(...):")
FORBIDDEN = {"source", "snippet", "text"}


def fields(run):
    return {k for row in run["results"] for k in row}


def test_search_run_keeps_locations_not_snippets():
    result = srv.SearchResults(status="ok", reason=None, results=[srv.SearchHit(**HIT)])
    run = memory.run_of("search_code", {"query": "where is url_for defined", "repo": "flask"}, result, "c 1", 12)
    assert run["question"] == "where is url_for defined" and run["status"] == "ok" and run["reason"] is None
    assert run["result_count"] == 1 and run["repos"] == ["flask"] and run["client"] == "c 1" and run["ms"] == 12
    assert run["results"] == [{"kind": "function", "repo": "flask", "name": "url_for", "path": "src/flask/helpers.py",
                               "start_line": 188, "end_line": 246, "score": 40.0, "similarity": 0.6}]
    assert not fields(run) & FORBIDDEN


def test_symbol_run_drops_source():
    result = srv.SymbolResults(status="ok", reason=None, total=2, results=[srv.SymbolHit(
        name="url_for", kind="function", repo="flask", path="a.py", start_line=1, end_line=9, source="x" * 50,
        cut=False)])
    run = memory.run_of("get_symbol", {"name": "url_for"}, result, None, 3)
    assert run["question"] == "url_for" and run["repos"] == ["flask"] and not fields(run) & FORBIDDEN


@pytest.mark.parametrize(("tool", "envelope", "hit", "row", "kept"), [
    ("find_callers", srv.CallerResults, srv.CallerHit,
     dict(repo="flask", kind="method", name="Flask.redirect", path="app.py", line=88, callee="url_for"),
     {"line": 88, "callee": "url_for"}),
    ("find_callees", srv.CalleeResults, srv.CalleeHit,
     dict(caller="url_for", line=4, repo="flask", kind="function", name="_ctx", path="g.py", start_line=1,
          end_line=3), {"caller": "url_for", "start_line": 1}),
    ("impact_of", srv.ImpactResults, srv.ImpactHit,
     dict(hops=2, repo="flask", kind="function", name="f", path="g.py", start_line=1, end_line=3), {"hops": 2}),
])
def test_graph_runs_keep_their_columns(tool, envelope, hit, row, kept):
    run = memory.run_of(tool, {"name": "url_for"}, envelope(status="ok", reason=None, results=[hit(**row)]), None, 1)
    assert kept.items() <= run["results"][0].items() and run["repos"] == ["flask"]


def test_wiki_page_is_one_row_and_list_is_pages():
    page = srv.WikiPage(name="flask/json", repo="flask", directory="src/flask/json", language="python",
                        state="fresh", compiled_at="2026-10-12T00:00:00Z", compiler="w", stale_reason=None, symbols=10,
                        text="long page text", citations=[], truncated=False)
    run = memory.run_of("read_wiki", {"page": "flask/json"}, srv.WikiResults(status="ok", reason=None, page=page),
                        None, 1)
    assert run["results"] == [{"kind": "page", "name": "flask/json", "repo": "flask", "path": "src/flask/json"}]
    listing = srv.WikiResults(status="ok", reason=None, pages=[srv.WikiPageInfo(
        name="flask/json", repo="flask", state="fresh", summary="JSON helpers")])
    run = memory.run_of("read_wiki", {}, listing, None, 1)
    assert run["question"] is None and run["results"][0]["kind"] == "page" and not fields(run) & FORBIDDEN


def test_list_repositories_and_failures():
    repos = srv.RepoList(database="reachable", reason=None,
                         repositories=[srv.RepoInfo(name="flask", path="/x", files=1, symbols=2)])
    run = memory.run_of("list_repositories", {}, repos, None, 1)
    assert (run["status"], run["question"], run["repos"]) == ("ok", None, [])
    assert run["results"] == [{"kind": "repository", "name": "flask"}]
    down = srv.RepoList(database="unreachable", reason="down", repositories=[])
    assert memory.run_of("list_repositories", {}, down, None, 1)["status"] == "unavailable"
    refused = srv.SearchResults(status="unavailable", reason="the question is empty", results=[])
    run = memory.run_of("search_code", {"query": ""}, refused, None, 0)
    assert (run["status"], run["reason"], run["result_count"], run["results"]) == (
        "unavailable", "the question is empty", 0, [])
    crashed = memory.run_of("get_symbol", {"name": "x"}, None, None, 5)
    assert (crashed["status"], crashed["result_count"]) == ("error", 0)


def test_repositories_come_from_results_without_a_repo_argument():
    hits = [srv.SearchHit(**HIT), srv.SearchHit(**HIT | {"repo": "gin", "rank": 2}), srv.SearchHit(**HIT | {"rank": 3})]
    run = memory.run_of("search_code", {"query": "url_for"}, srv.SearchResults(status="ok", reason=None, results=hits),
                        None, 1)
    assert run["repos"] == ["flask", "gin"]


# --- a fixture repository and real tool calls (US2 to US4) ---

import json  # noqa: E402

from golestan_code_mcp import ingest, registry, semantic, wiki  # noqa: E402

ENV = {"CCCTX_MAX_FILE_BYTES": "20000"}
WIDGETS = '''"""Widgets and how they spin."""


def spin_widget(w):
    """Spin a widget once."""
    return turn_helper(w)


def turn_helper(w):
    return w


class Widget:
    """A widget."""

    def run(self):
        return spin_widget(self)
'''
OTHER = "from pkg.widgets import spin_widget\n\n\ndef use_widget():\n    return spin_widget(1)\n"
FILES = {"pkg/__init__.py": "", "pkg/widgets.py": WIDGETS, "pkg/other.py": OTHER}


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                   capture_output=True)


def write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")


def commit(root, message="change"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


@pytest.fixture(scope="module")
def ready(conn):
    try:
        semantic.require_model(conn)
    except semantic.ModelError:
        pytest.skip("embedding model not loaded: run `ccctx model load`")
    yield conn
    with conn.cursor() as cur:
        names = [r[0] for r in cur.execute("select name from ccctx_repos where name like 't-%'")]
    mem = memory.client(conn)
    for name in names:
        mem.delete_user(name, cascade=True)
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()


def make_repo(conn, root, files=FILES):
    root.mkdir(parents=True)
    git(root, "init", "-q")
    write(root, files)
    commit(root, "init")
    name = f"t-{uuid.uuid4().hex[:8]}"
    registry.add_repo(conn, root, name)
    ingest.ingest(conn, name, ENV)
    return name


@pytest.fixture
def repo(ready, tmp_path):
    """(name, root) of a fresh indexed repository with no memory yet."""
    root = tmp_path / "repo"
    return make_repo(ready, root), root


def ask(tool, **arguments):
    """Call a tool the way a client does; its run is recorded by the server."""
    result = getattr(srv, tool)(**arguments)
    memory.flush()
    return result


def pending_runs(conn, name):
    with conn.cursor() as cur:
        return memory.pending(cur, registry.repo_id(conn, name))


def facts_of(conn, name):
    return {f.subject: f for f in memory.facts(conn, name, limit=50)}


def synthetic(name, tool, question, results, arguments=None):
    return {"tool": tool, "arguments": arguments or {"query": question, "repo": name}, "question": question,
            "client": None, "status": "ok", "reason": None, "ms": 1, "result_count": len(results),
            "results": results, "repos": [name]}


def hit(name, symbol, path="pkg/widgets.py", kind="function"):
    return {"kind": kind, "repo": name, "name": symbol, "path": path, "start_line": 1, "end_line": 2}


# --- findings (T014) ---


def test_search_hit_finding_is_about_the_top_symbol(ready, repo):
    name, _ = repo
    ask("search_code", query="spin_widget", repo=name)
    with ready.cursor() as cur:
        [run] = memory.pending(cur, registry.repo_id(ready, name))
        f = memory.finding(cur, registry.repo_id(ready, name), name, run)
    assert f.subject_key == "symbol:pkg/widgets.py:spin_widget:function" and f.subject == "spin_widget"
    assert (f.aspect, f.tool, f.question, f.run_id) == ("defined", "search_code", "spin_widget", run["id"])
    assert f.refs[0]["role"] == "subject" and f.refs[0]["content_hash"]


@pytest.mark.parametrize(("tool", "arguments", "subject", "aspect"), [
    ("get_symbol", {"name": "spin_widget"}, "spin_widget", "defined"),
    ("find_callers", {"name": "spin_widget"}, "spin_widget", "callers"),
    ("find_callees", {"name": "spin_widget"}, "spin_widget", "callees"),
    ("impact_of", {"name": "spin_widget", "depth": 2}, "spin_widget", "impact"),
])
def test_graph_and_symbol_findings(ready, repo, tool, arguments, subject, aspect):
    name, _ = repo
    ask(tool, repo=name, **arguments)
    with ready.cursor() as cur:
        [run] = memory.pending(cur, registry.repo_id(ready, name))
        f = memory.finding(cur, registry.repo_id(ready, name), name, run)
    assert (f.subject, f.aspect, f.tool) == (subject, aspect, tool)
    if aspect in ("callers", "callees", "impact"):
        assert f.value["count"] >= 1 and f.value["names"]


def test_page_finding_from_read_wiki(ready, tmp_path):
    files = dict(FILES) | {f"pkg/m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(10)}
    name = make_repo(ready, tmp_path / "w", files)
    wiki.build(ready, name)
    ask("read_wiki", page="pkg", repo=name)
    with ready.cursor() as cur:
        [run] = memory.pending(cur, registry.repo_id(ready, name))
        f = memory.finding(cur, registry.repo_id(ready, name), name, run)
    assert (f.subject_key, f.aspect, f.kind) == ("page:pkg", "module", "page") and f.value["summary"]


@pytest.mark.parametrize("make", [
    lambda n: synthetic(n, "search_code", "nothing", []),
    lambda n: synthetic(n, "list_repositories", None, [{"kind": "repository", "name": n}], {}),
    lambda n: synthetic(n, "read_wiki", None, [{"kind": "page", "repo": n, "name": "pkg", "state": "fresh"}], {}),
    lambda n: synthetic(n, "recall", None, [{"kind": "fact", "repo": n, "name": "spin_widget"}], {"repo": n}),
    lambda n: synthetic(n, "search_code", "gone", [hit(n, "no_such_symbol")]),
    lambda n: dict(synthetic(n, "search_code", "x", [hit(n, "spin_widget")]), status="unavailable"),
])
def test_runs_that_give_no_finding(ready, repo, make):
    name, _ = repo
    with db.connect(load()) as c:
        memory.store_runs(c, [make(name)])
    with ready.cursor() as cur:
        [run] = memory.pending(cur, registry.repo_id(ready, name))
        assert memory.finding(cur, registry.repo_id(ready, name), name, run) is None


def test_a_run_over_two_repositories_gives_each_its_own_top(ready, repo, tmp_path):
    name, _ = repo
    other = make_repo(ready, tmp_path / "o")
    run = synthetic(name, "search_code", "spin", [hit(other, "turn_helper"), hit(name, "spin_widget")], {"query": "spin"})
    run["repos"] = [name, other]
    with db.connect(load()) as c:
        memory.store_runs(c, [run])
    with ready.cursor() as cur:
        for repo_name, subject in ((name, "spin_widget"), (other, "turn_helper")):
            rid = registry.repo_id(ready, repo_name)
            [r] = memory.pending(cur, rid)
            assert memory.finding(cur, rid, repo_name, r).subject == subject


# --- consolidation (T015) ---


def test_near_identical_questions_merge_into_one_fact(ready, repo):
    name, _ = repo
    ask("search_code", query="where is spin_widget defined", repo=name)
    ask("search_code", query="where's spin_widget defined?", repo=name)
    ask("get_symbol", name="spin_widget", repo=name)
    ask("find_callers", name="spin_widget", repo=name)
    [c] = memory.consolidate(ready, name)
    assert (c.runs, c.created, c.merged, c.weakened, c.stale) == (4, 1, 3, 0, 0)
    facts = facts_of(ready, name)
    assert list(facts) == ["spin_widget"]
    f = facts["spin_widget"]
    assert f.confidence == 0.938 and f.state == "fresh" and len(f.runs) == 4
    assert f.questions[:2] == ["spin_widget", "where's spin_widget defined?"]
    assert "where is spin_widget defined" in f.questions
    assert f.distiller == memory.TEMPLATE and f.consolidator == memory.CONSOLIDATOR
    assert f.text.startswith("`spin_widget` is a function in `pkg/widgets.py`.")
    assert "It is called by" in f.text and 'Asked as "spin_widget"' in f.text
    with ready.cursor() as cur:
        assert {r for (r,) in cur.execute("select id from ccctx_memory_runs where id in (select column_value from"
                                          " json_table(:j, '$[*]' columns (column_value number path '$')))",
                                          j=json.dumps(f.runs))} == set(f.runs)


def test_confidence_rises_with_agreeing_runs(ready, repo):
    name, _ = repo
    seen = []
    for question in ("spin one", "spin two", "spin three"):
        with db.connect(load()) as c:
            memory.store_runs(c, [synthetic(name, "search_code", question, [hit(name, "spin_widget")])])
        memory.consolidate(ready, name)
        seen.append(facts_of(ready, name)["spin_widget"].confidence)
    assert seen == [0.5, 0.75, 0.875]


def test_a_wording_that_moves_weakens_the_old_fact_and_links_them(ready, repo):
    name, _ = repo
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "search_code", "Who turns it?", [hit(name, "spin_widget")]),
                              synthetic(name, "search_code", "spin widgets", [hit(name, "spin_widget")])])
    memory.consolidate(ready, name)
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "search_code", "who turns it", [hit(name, "turn_helper")])])

    def links():
        with ready.cursor() as cur:
            return cur.execute("select count(*) from ccctx_mem_memory_link").fetchone()[0]

    before = links()
    [c] = memory.consolidate(ready, name)
    facts = facts_of(ready, name)
    assert (c.created, c.weakened) == (1, 1)
    assert facts["spin_widget"].confidence == 0.375 and "Who turns it?" not in facts["spin_widget"].questions
    assert facts["turn_helper"].questions == ["who turns it"]
    assert links() == before + 1  # turn_helper's fact contradicts spin_widget's


def test_an_aspect_that_changes_value_is_a_contradiction(ready, repo):
    name, _ = repo
    callers = [{"kind": "function", "repo": name, "name": "use_widget", "path": "pkg/other.py", "line": 5,
                "callee": "spin_widget"}]
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "find_callers", "spin_widget", callers, {"name": "spin_widget"})])
    memory.consolidate(ready, name)
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "find_callers", "spin_widget", callers * 2, {"name": "spin_widget"})])
    [c] = memory.consolidate(ready, name)
    assert c.weakened == 1 and facts_of(ready, name)["spin_widget"].confidence == 0.375


def test_consolidation_is_idempotent_and_skips_counted_runs(ready, repo):
    name, _ = repo
    ask("get_symbol", name="spin_widget", repo=name)
    memory.consolidate(ready, name)
    before = {f.id: (f.text, f.confidence, f.runs) for f in memory.facts(ready, name)}
    [c] = memory.consolidate(ready, name)
    assert (c.runs, c.created, c.merged, c.weakened, c.stale, c.reasked) == (0, 0, 0, 0, 0, 0)
    with ready.cursor() as cur:  # a crash after the facts were written: the links read as unconsolidated again
        cur.execute("update ccctx_memory_run_repos set consolidated = false where repo_id = :r",
                    r=registry.repo_id(ready, name))
    ready.commit()
    [c] = memory.consolidate(ready, name)
    assert (c.runs, c.created, c.merged) == (1, 0, 0)
    assert {f.id: (f.text, f.confidence, f.runs) for f in memory.facts(ready, name)} == before


def test_two_consolidations_at_once_count_each_run_once(ready, repo):
    import threading
    name, _ = repo
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "search_code", f"q{i}", [hit(name, "spin_widget")]) for i in range(6)])
    errors = []

    def work():
        try:
            with db.connect(load()) as c:
                memory.consolidate(c, name)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    [f] = memory.facts(ready, name)
    assert len(f.runs) == 6 and f.confidence == round(1 - 0.5 ** 6, 3)


def test_consolidation_logs_counts_never_text(ready, repo, caplog):
    import logging
    name, _ = repo
    ask("search_code", query="where is spin_widget secretly", repo=name)
    with caplog.at_level(logging.INFO, logger="golestan_code_mcp.memory"):
        memory.consolidate(ready, name)
    [record] = [json.loads(r.getMessage()) for r in caplog.records if '"consolidate"' in r.getMessage()]
    assert set(record) == {"event", "repo", "runs", "created", "merged", "weakened", "stale", "reasked", "distillers",
                           "seconds"}
    assert "secretly" not in caplog.text


# --- distillers (T016) ---


class FakeLlm:
    def __init__(self, text="`spin_widget` spins a `Widget` and is used by `use_widget`.", fail=False):
        self.text, self.fail, self.prompts = text, fail, []

    def generate(self, prompt, response_json_schema=None, **_):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("model down")
        from oracleagentmemory.apis.llms.llm import LlmResponse
        return LlmResponse(json.dumps({"text": self.text}))


def with_model(monkeypatch, llm):
    monkeypatch.setattr(memory, "model", lambda env=None: (llm, "llm:fake/model"))


def test_model_writes_the_text_from_exactly_the_documented_inputs(ready, repo, monkeypatch):
    name, _ = repo
    llm = FakeLlm()
    with_model(monkeypatch, llm)
    ask("search_code", query="where is spin_widget defined", repo=name, limit=7)
    ask("find_callers", name="spin_widget", repo=name)
    memory.consolidate(ready, name)
    f = facts_of(ready, name)["spin_widget"]
    assert (f.text, f.distiller) == (llm.text, "llm:fake/model")
    assert len(llm.prompts) == 1  # one call per changed fact, none from the package (linking is off)
    prompt = llm.prompts[0]
    assert "def spin_widget(w):" in prompt and "where is spin_widget defined" in prompt and "use_widget" in prompt
    assert '"limit"' not in prompt and "client" not in prompt.lower()


def test_model_failure_falls_back_to_the_template(ready, repo, monkeypatch, caplog):
    import logging
    name, _ = repo
    with_model(monkeypatch, FakeLlm(fail=True))
    ask("get_symbol", name="spin_widget", repo=name)
    with caplog.at_level(logging.WARNING, logger="golestan_code_mcp.memory"):
        [c] = memory.consolidate(ready, name)
    f = facts_of(ready, name)["spin_widget"]
    assert f.distiller == memory.TEMPLATE and f.text.startswith("`spin_widget` is a function")
    assert c.distillers == {memory.TEMPLATE: 1}
    assert sum("memory_model_failed" in r.getMessage() for r in caplog.records) == 1


def test_model_text_naming_unknown_code_is_rejected(ready, repo, monkeypatch, caplog):
    import logging
    name, _ = repo
    with_model(monkeypatch, FakeLlm("`spin_widget` calls `launch_rockets` and `secret_module.boom`."))
    ask("get_symbol", name="spin_widget", repo=name)
    with caplog.at_level(logging.WARNING, logger="golestan_code_mcp.memory"):
        memory.consolidate(ready, name)
    assert facts_of(ready, name)["spin_widget"].distiller == memory.TEMPLATE
    [rec] = [json.loads(r.getMessage()) for r in caplog.records if "memory_model_rejected" in r.getMessage()]
    assert set(rec["names"]) == {"launch_rockets", "secret_module.boom"} and "calls" not in json.dumps(rec)


def test_the_model_never_decides_subject_runs_or_confidence(ready, tmp_path, monkeypatch):
    outcomes = []
    for i, llm in enumerate((None, FakeLlm())):
        name = make_repo(ready, tmp_path / f"r{i}")
        if llm:
            with_model(monkeypatch, llm)
        for q in ("where is spin_widget defined", "spin it"):
            ask("search_code", query=q, repo=name)
        memory.consolidate(ready, name)
        outcomes.append({s: (len(f.runs), f.confidence) for s, f in facts_of(ready, name).items()})
    assert outcomes[0] == outcomes[1]


def test_unknown_names():
    allowed = {"spin_widget", "Widget", "pkg/widgets.py", "turn_helper"}
    assert memory.unknown_names("`spin_widget` lives in `pkg/widgets.py` and wraps `turn_helper`.", allowed) == []
    assert memory.unknown_names("It calls `boom_now` via os.path.join and RocketLauncher.", allowed) == [
        "boom_now", "os.path.join", "RocketLauncher"]


# --- recall (T023) ---


def seed(conn, name, subjects):
    """One consolidated fact per (symbol, question, count of agreeing runs)."""
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "search_code", f"{q} {i}", [hit(name, s)])
                              for s, q, n in subjects for i in range(n)])
    memory.consolidate(conn, name)


def test_recall_without_topic_is_fresh_then_confident(ready, repo):
    name, _ = repo
    seed(ready, name, [("spin_widget", "spin", 1), ("turn_helper", "turn", 2), ("use_widget", "use", 3)])
    r = memory.recall(ready, name)
    assert [f.subject for f in r.facts] == ["use_widget", "turn_helper", "spin_widget"] and r.total == 3
    assert [f.sources[0]["path"] for f in r.facts] == ["pkg/other.py", "pkg/widgets.py", "pkg/widgets.py"]
    assert r.runs and len(r.runs) <= 10 and all(x.tool != "recall" for x in r.runs)


def test_recall_by_topic_puts_the_related_fact_first(ready, repo):
    name, _ = repo
    seed(ready, name, [("turn_helper", "how is a widget turned", 3), ("spin_widget", "how do widgets spin", 1)])
    assert memory.recall(ready, name, "spinning widgets").facts[0].subject == "spin_widget"
    assert memory.recall(ready, name, None).facts[0].subject == "turn_helper"


def test_recall_without_a_repository_covers_every_one(ready, repo, tmp_path):
    name, _ = repo
    other = make_repo(ready, tmp_path / "o")
    seed(ready, name, [("spin_widget", "spin", 1)])
    seed(ready, other, [("turn_helper", "turn", 1)])
    found = {(f.repo, f.subject) for f in memory.recall(ready, None, limit=50).facts}
    assert {(name, "spin_widget"), (other, "turn_helper")} <= found
    assert {(name, "spin_widget"), (other, "turn_helper")} <= {
        (f.repo, f.subject) for f in memory.recall(ready, None, "spin turn", limit=50).facts}


def test_recall_size_does_not_grow_with_the_log(ready, repo):
    name, _ = repo
    seed(ready, name, [("spin_widget", "spin", 1)])
    small = memory.recall(ready, name)
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "search_code", f"q{i}", []) for i in range(2000)])
    started = time.monotonic()
    big = memory.recall(ready, name)
    assert time.monotonic() - started < 2  # SC-007
    assert (len(big.facts), len(big.runs), big.total) == (len(small.facts), 10, small.total)


def test_recall_unknown_repository(ready):
    with pytest.raises(ValueError, match="no repository named 't-nowhere'"):
        memory.recall(ready, "t-nowhere")


# --- facts in search (T026, FR-021) ---


def test_fresh_facts_are_search_hits_below_definitions(ready, repo):
    name, _ = repo
    seed(ready, name, [("spin_widget", "how do widgets spin", 2)])
    hits = semantic.search(ready, "spin_widget", [name])
    assert hits[0].kind == "function" and hits[0].name == "spin_widget"  # the definition stays first
    [fact] = [h for h in hits if h.kind == "fact"]
    assert (fact.name, fact.path, fact.language, fact.state) == ("spin_widget", "pkg/widgets.py", "python", "fresh")
    assert fact.start_line and fact.snippet.startswith("`spin_widget` is a function") and fact.keyword_rank is None
    assert len(semantic.search(ready, "how do widgets spin", [name], limit=2)) == 2  # facts count toward the limit


def test_fact_hits_follow_filters_and_skip_stale(ready, repo, tmp_path):
    name, _ = repo
    other = make_repo(ready, tmp_path / "o")
    seed(ready, name, [("spin_widget", "how do widgets spin", 1)])
    assert not [h for h in semantic.search(ready, "how do widgets spin", [other]) if h.kind == "fact"]
    assert not [h for h in semantic.search(ready, "how do widgets spin", [name], language="go") if h.kind == "fact"]
    mem = memory.client(ready)
    [rec] = mem.list_memories(user_id=name, agent_id=memory.AGENT, limit=None)
    from oracleagentmemory.apis.records import RecordStatus
    mem.update_memory(rec.id, metadata=rec.metadata | {"state": "stale"}, status=RecordStatus.INVALID)
    assert not [h for h in semantic.search(ready, "how do widgets spin", [name]) if h.kind == "fact"]


def test_search_without_facts_never_loads_the_package(ready, repo):
    name, _ = repo
    code = ("import sys; from golestan_code_mcp import db, semantic; from golestan_code_mcp.config import load; "
            f"c = db.connect(load()); semantic.search(c, 'spin', ['{name}']); print('oracleagentmemory' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out.endswith("False")


def test_a_fact_hit_run_confirms_its_fact(ready, repo):
    name, _ = repo
    seed(ready, name, [("spin_widget", "how do widgets spin", 1)])
    run = synthetic(name, "search_code", "what spins", [{"kind": "fact", "repo": name, "name": "spin_widget",
                                                         "path": "pkg/widgets.py", "start_line": 4, "end_line": 6}])
    with db.connect(load()) as c:
        memory.store_runs(c, [run])
    memory.consolidate(ready, name)
    assert facts_of(ready, name)["spin_widget"].confidence == 0.75


# --- keeping pace with the code (T028 to T031) ---


def change(conn, name, root, files, remove=()):
    for rel in remove:
        (root / rel).unlink()
    write(root, files)
    commit(root)
    return ingest.ingest(conn, name, ENV)


def run_count(conn):
    with conn.cursor() as cur:
        return cur.execute("select count(*) from ccctx_memory_runs").fetchone()[0]


def test_removed_code_makes_its_fact_stale_and_halves_it(ready, repo):
    name, root = repo
    seed(ready, name, [("turn_helper", "what turns", 1)])
    summary = change(ready, name, root, {"pkg/widgets.py": WIDGETS.replace("return turn_helper(w)", "return w")
                                         .replace("def turn_helper(w):\n    return w\n\n\n", "")})
    assert summary.memory.stale == 1
    f = facts_of(ready, name)["turn_helper"]
    assert (f.state, f.stale_reason, f.confidence) == ("stale", "`turn_helper` was removed", 0.25)


def test_moved_code_is_repointed_without_a_re_ask(ready, repo):
    name, root = repo
    seed(ready, name, [("use_widget", "who uses widgets", 1)])
    summary = change(ready, name, root, {"pkg/moved.py": OTHER}, remove=["pkg/other.py"])
    assert (summary.memory.stale, summary.memory.reasked) == (0, 0)
    f = facts_of(ready, name)["use_widget"]
    assert (f.state, f.sources[0]["path"]) == ("fresh", "pkg/moved.py") and "`pkg/moved.py`" in f.text


def test_edited_code_answered_the_same_way_stays_fresh(ready, repo):
    name, root = repo
    ask("get_symbol", name="spin_widget", repo=name)
    memory.consolidate(ready, name)
    before = run_count(ready)
    edited = WIDGETS.replace('    """Spin a widget once."""\n', '    """Spin a widget once."""\n    w = w or 0\n')
    summary = change(ready, name, root, {"pkg/widgets.py": edited})
    assert (summary.memory.stale, summary.memory.reasked) == (0, 1)
    assert run_count(ready) == before  # a re-ask is not a run
    f = facts_of(ready, name)["spin_widget"]
    assert f.state == "fresh" and f.sources[0]["end_line"] == 7 and f.confidence == 0.5


def test_edited_code_answered_by_other_code_goes_stale(ready, repo):
    name, root = repo
    ask("get_symbol", name="spin_widget", repo=name)
    memory.consolidate(ready, name)
    renamed = WIDGETS.replace("def spin_widget(w):", "def spin_widget_old(w):").replace(
        "return spin_widget(self)", "return spin_widget_old(self)")
    summary = change(ready, name, root, {"pkg/widgets.py": renamed,
                                         "pkg/other.py": OTHER.replace("from pkg.widgets import spin_widget\n", "")
                                         + "\n\ndef spin_widget(w):\n    return 2\n"})
    assert summary.memory.stale == 1 and summary.memory.reasked == 1
    f = next(f for f in memory.facts(ready, name, 50) if f.sources[0]["name"] == "spin_widget" and f.state == "stale")
    assert f.stale_reason == 're-asked "spin_widget": now answered by `spin_widget` in `pkg/other.py`'
    assert f.confidence == 0.25


def test_stale_facts_wait_for_a_run_and_consolidation_is_idempotent(ready, repo):
    name, root = repo
    seed(ready, name, [("turn_helper", "what turns", 1)])
    change(ready, name, root, {"pkg/widgets.py": WIDGETS.replace("def turn_helper(w):", "def turn_helper2(w):")})
    [c] = memory.consolidate(ready, name)
    assert (c.stale, c.reasked, c.runs) == (0, 0, 0)  # stale facts are not re-checked
    change(ready, name, root, {"pkg/widgets.py": WIDGETS})
    assert facts_of(ready, name)["turn_helper"].state == "stale"  # code is back, but no run confirmed it
    seed(ready, name, [("turn_helper", "what turns now", 1)])
    assert facts_of(ready, name)["turn_helper"].state == "fresh"


def test_every_ingestion_consolidates_even_when_current(ready, repo):
    name, _ = repo
    ask("get_symbol", name="spin_widget", repo=name)
    summary = ingest.ingest(ready, name, ENV)
    assert summary.mode == "current" and summary.memory.runs == 1 and summary.memory.created == 1
    again = ingest.ingest(ready, name, ENV)
    assert (again.memory.runs, again.memory.created, again.memory.stale, again.memory.reasked) == (0, 0, 0, 0)


def test_a_consolidation_failure_never_undoes_the_ingestion(ready, repo, monkeypatch):
    name, root = repo

    def broken(*_, **__):
        raise RuntimeError("memory down")

    monkeypatch.setattr(memory, "consolidate", broken)
    summary = change(ready, name, root, {"pkg/new.py": "def brand_new():\n    return 1\n"})
    assert summary.memory == "memory down"
    with ready.cursor() as cur:
        assert cur.execute("select count(*) from ccctx_symbols s join ccctx_files f on f.id = s.file_id where"
                           " f.repo_id = :r and s.name = 'brand_new'", r=registry.repo_id(ready, name)).fetchone()[0]


def test_forget_removes_only_that_repository(ready, repo, tmp_path):
    name, _ = repo
    other = make_repo(ready, tmp_path / "o")
    seed(ready, name, [("spin_widget", "spin", 1)])
    seed(ready, other, [("spin_widget", "spin", 1)])
    memory.forget(ready, name)
    assert memory.facts(ready, name) == [] and len(memory.facts(ready, other)) == 1


def test_runs_only_a_removed_repository_held_go_with_it(ready, repo, tmp_path):
    name, _ = repo
    other = make_repo(ready, tmp_path / "o")
    shared = synthetic(name, "search_code", "spin", [hit(name, "spin_widget")])
    shared["repos"] = [name, other]
    with db.connect(load()) as c:
        memory.store_runs(c, [shared, synthetic(name, "search_code", "only here", [hit(name, "spin_widget")])])
    memory.forget(ready, name)
    with ready.cursor() as cur:
        cur.execute("delete from ccctx_repos where name = :n", n=name)
    ready.commit()
    memory.consolidate(ready, other)
    with ready.cursor() as cur:
        assert cur.execute("select count(*) from ccctx_memory_runs where question = 'only here'").fetchone()[0] == 0
        assert cur.execute("select count(*) from ccctx_memory_runs r join ccctx_memory_run_repos l on"
                           " l.run_id = r.id join ccctx_repos p on p.id = l.repo_id where p.name = :o",
                           o=other).fetchone()[0] == 1


# --- scale (T036, SC-004) ---


def test_a_thousand_runs_consolidate_within_a_minute(ready, repo):
    name, _ = repo
    subjects = [("spin_widget", "function", "pkg/widgets.py"), ("turn_helper", "function", "pkg/widgets.py"),
                ("Widget", "class", "pkg/widgets.py"), ("Widget.run", "method", "pkg/widgets.py"),
                ("use_widget", "function", "pkg/other.py")]
    runs_ = [synthetic(name, "search_code", f"question {i}", [hit(name, s, p, k)])
             for i in range(1000) for s, k, p in [subjects[i % len(subjects)]]]
    with db.connect(load()) as c:
        memory.store_runs(c, runs_)
    started = time.monotonic()
    [c] = memory.consolidate(ready, name)
    assert time.monotonic() - started < 60 and (c.runs, c.created) == (1000, 5)
    before = {f.id: (f.text, f.confidence, len(f.runs)) for f in memory.facts(ready, name)}
    [again] = memory.consolidate(ready, name)
    assert (again.runs, again.created, again.merged) == (0, 0, 0)
    assert {f.id: (f.text, f.confidence, len(f.runs)) for f in memory.facts(ready, name)} == before


def test_the_same_words_to_another_tool_are_not_a_contradiction(ready, repo):
    name, _ = repo
    defs = [hit(name, "Widget.run", kind="method")]
    callers = [{"kind": "method", "repo": name, "name": "Widget.run", "path": "pkg/widgets.py", "line": 17,
                "callee": "spin_widget"}]
    with db.connect(load()) as c:
        memory.store_runs(c, [synthetic(name, "get_symbol", "run", defs, {"name": "run"}),
                              synthetic(name, "find_callers", "run", callers, {"name": "run"})])
    [c] = memory.consolidate(ready, name)
    assert (c.created, c.weakened) == (2, 0)
    assert {s: f.confidence for s, f in facts_of(ready, name).items()} == {"Widget.run": 0.5, "spin_widget": 0.5}


def test_names_from_the_source_sent_are_allowed_and_prose_is_not_a_name():
    sent = "def get(self, name, default=None):\n    return self.__dict__.get(name, default)\nmarkupsafe.escape"
    assert memory.unknown_names("`get` wraps `self.__dict__.get` and markupsafe.escape.", {sent}) == []
    # unpaired backticks: the prose between them is not a name, the code-shaped words still are
    assert memory.unknown_names("`get` method is defined in `app.py` as noted by `boom_now", {sent}) == ["app.py", "boom_now"]
    assert memory.unknown_names("It returns view_args via `--debug`.", {sent}) == ["view_args"]
