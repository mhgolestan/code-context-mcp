"""Usage feedback (specs/011-usage-feedback): uses, weights, decay and the bounded boost, against the local database."""

import subprocess
import uuid
from types import SimpleNamespace

import pytest

from golestan_code_mcp import db, feedback, ingest, registry, schema, semantic, wiki
from golestan_code_mcp.config import load

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
FILES = {"pkg/__init__.py": "", "pkg/widgets.py": WIDGETS} | {
    f"pkg/m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(10)}


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, check=True,
                   capture_output=True)


def make_repo(conn, root, files):
    """A throwaway `t-` repository, ingested, with its wiki pages built."""
    root.mkdir(parents=True)
    git(root, "init", "-q")
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    name = f"t-fb{uuid.uuid4().hex[:6]}"
    registry.add_repo(conn, root, name)
    ingest.ingest(conn, name, ENV)
    wiki.build(conn, name)
    return SimpleNamespace(name=name, id=registry.repo_id(conn, name), root=root)


@pytest.fixture(scope="module")
def conn():
    try:
        conn = db.connect(load())
        schema.migrate(conn)
        semantic.require_model(conn)
    except Exception as e:  # config missing, database down or no model: skip, never fail
        pytest.skip(f"database or model unavailable: {type(e).__name__}")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def fb(conn, tmp_path_factory):
    repo = make_repo(conn, tmp_path_factory.mktemp("fb") / "repo", FILES)
    yield repo
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where id = :i", i=repo.id)  # its uses go with it
    conn.commit()


@pytest.fixture
def clean(conn, fb):
    """The fixture repository with no uses, before and after the test."""
    def wipe():
        with conn.cursor() as cur:
            cur.execute("delete from ccctx_feedback where repo_id = :r", r=fb.id)
        conn.commit()
    wipe()
    yield fb
    wipe()


def use(repo, name="spin_widget", path="pkg/widgets.py", how="cited", session=None, days_ago=0, kind="function"):
    return {"repo_id": repo.id, "path": path, "name": name, "kind": kind, "how": how, "client": "test",
            "session": session or uuid.uuid4().hex, "run_id": None, "days_ago": days_ago}


def write(conn, *uses):
    with conn.cursor() as cur:
        for u in uses:
            feedback.write_use(cur, u)
    conn.commit()


def stored(conn, repo, name="spin_widget"):
    with conn.cursor() as cur:
        cur.execute("select how, session_id from ccctx_feedback where repo_id = :r and name = :n order by id",
                    r=repo.id, n=name)
        return cur.fetchall()


# --- the boost (T007) ---


def test_boost_grows_linearly_to_the_cap():
    assert feedback.boost(0, 0.2) == 0
    assert feedback.boost(1, 0.2) == pytest.approx(0.04)
    assert feedback.boost(2.5, 0.2) == pytest.approx(0.1)
    assert feedback.boost(feedback.SATURATION, 0.2) == feedback.boost(1000, 0.2) == 0.2
    assert feedback.boost(1000, 0) == 0


# --- writing uses: one per result per session ---


def test_a_cited_use_replaces_an_opened_one_in_the_same_session(conn, clean):
    s = uuid.uuid4().hex
    write(conn, use(clean, how="opened", session=s), use(clean, how="cited", session=s))
    assert stored(conn, clean) == [("cited", s)]
    write(conn, use(clean, how="cited", session=s), use(clean, how="opened", session=s))
    assert stored(conn, clean) == [("cited", s)]


def test_two_opened_uses_in_one_session_are_one(conn, clean):
    s = uuid.uuid4().hex
    write(conn, use(clean, how="opened", session=s), use(clean, how="opened", session=s))
    assert stored(conn, clean) == [("opened", s)]


def test_another_session_adds_a_use(conn, clean):
    write(conn, use(clean), use(clean))
    assert [how for how, _ in stored(conn, clean)] == ["cited", "cited"]


# --- weights and decay ---


def test_weights_halve_every_half_life(conn, clean):
    write(conn, use(clean), use(clean, "turn_helper", how="opened"), use(clean, "Widget", days_ago=30, kind="class"),
          use(clean, "Widget.run", days_ago=60, kind="method"))
    with conn.cursor() as cur:
        w = feedback.weights(cur, [clean.id], 30)
    weight = {name: v[1] for (_, _, name), v in w.items()}
    assert weight["spin_widget"] == pytest.approx(1.0, abs=0.01)
    assert weight["turn_helper"] == pytest.approx(0.5, abs=0.01)
    assert weight["Widget"] == pytest.approx(0.5, abs=0.01)
    assert weight["Widget.run"] == pytest.approx(0.25, abs=0.01)
    assert {v[0] for v in w.values()} == {1}
    assert set(w) == {(clean.id, "pkg/widgets.py", n) for n in weight}


def test_weights_add_up_and_skip_other_repositories(conn, clean):
    write(conn, *(use(clean) for _ in range(3)))
    with conn.cursor() as cur:
        uses, weight, last = feedback.weights(cur, [clean.id], 30)[(clean.id, "pkg/widgets.py", "spin_widget")]
        assert uses == 3 and weight == pytest.approx(3.0, abs=0.01) and last is not None
        assert feedback.weights(cur, [], 30) == {}
        assert feedback.weights(cur, [-1], 30) == {}


# --- resolving a result's identity ---


def test_resolve_finds_symbols_and_pages(conn, fb):
    with conn.cursor() as cur:
        assert feedback.resolve(cur, fb.id, "pkg/widgets.py", "spin_widget") == "function"
        assert feedback.resolve(cur, fb.id, "pkg/widgets.py", "Widget.run") == "method"
        assert feedback.resolve(cur, fb.id, "pkg", "pkg") == "page"
        assert feedback.resolve(cur, fb.id, "pkg/other.py", "spin_widget") is None
        assert feedback.resolve(cur, fb.id, "pkg/widgets.py", "no_such_name") is None


# --- the bounded boost in search (US2, research R5) ---

QUESTION = "spin a widget"
CAP = feedback.boost(1e9, 0.2)


def hammer(conn, repo, name, path="pkg/widgets.py", kind="function", count=1000):
    write(conn, *(use(repo, name, path, kind=kind) for _ in range(count)))


def search(conn, repo, question=QUESTION, **kwargs):
    return semantic.search(conn, question, [repo.name], env=kwargs.pop("env", {}), **kwargs)


def symbols(results):
    return [r for r in results if r.kind not in ("page", "fact")]


def test_every_hit_explains_its_boost(conn, clean):
    write(conn, use(clean), use(clean, "turn_helper", how="opened"))
    hits = search(conn, clean, limit=50)
    assert hits and all(r.score == pytest.approx(r.relevance * (1 + r.feedback)) for r in hits)
    by_name = {r.name: r for r in hits}
    assert by_name["spin_widget"].feedback == pytest.approx(0.04, abs=1e-3)
    assert by_name["turn_helper"].feedback == pytest.approx(0.02, abs=1e-3)


@pytest.mark.parametrize("count", [1, 5, 1000])
def test_the_boost_never_passes_the_cap(conn, clean, count):
    hammer(conn, clean, "spin_widget", count=count)
    assert all(0 <= r.feedback <= CAP for r in search(conn, clean, limit=50))


def test_a_used_symbol_passes_only_what_is_within_twenty_percent(conn, clean):
    before = symbols(search(conn, clean, limit=50))
    target = before[-1]  # the weakest symbol found: everything else starts above it
    hammer(conn, clean, target.name, target.path, target.kind)
    after = symbols(search(conn, clean, limit=50))
    moved = {r.name: i for i, r in enumerate(after)}
    for r in before[:-1]:
        if r.relevance > target.relevance * (1 + CAP):
            assert moved[r.name] < moved[target.name], r.name
        elif r.relevance < target.relevance * (1 + CAP) and r.score > 0:  # not an unfound identifier definition
            assert moved[r.name] > moved[target.name], r.name
    assert [r.relevance for r in after if r.name == target.name] == [target.relevance]


def test_feedback_never_adds_a_hit(conn, clean):
    without = {r.name for r in search(conn, clean, limit=50)}
    hammer(conn, clean, "f3", "pkg/m3.py")
    assert {r.name for r in search(conn, clean, limit=50)} == without


def test_identifier_definitions_stay_first(conn, clean):
    hammer(conn, clean, "spin_widget")
    assert search(conn, clean, "turn_helper")[0].name == "turn_helper"


def test_a_used_page_is_placed_by_its_boosted_similarity(conn, clean):
    hammer(conn, clean, "pkg", "pkg", kind="page")
    hits = search(conn, clean, "what does the pkg package do with widgets", limit=50)
    [at] = [i for i, r in enumerate(hits) if r.kind == "page"]
    page = hits[at]
    assert page.feedback == pytest.approx(CAP) and page.score == pytest.approx(page.relevance * (1 + CAP))
    placed = min(1.0, page.similarity * (1 + CAP))
    assert all(h.similarity is not None and h.similarity >= placed for h in hits[:at] if h.kind != "page")
    assert all(r.feedback == 0 for r in hits if r.kind == "fact")


def test_switching_feedback_off_ranks_as_without_uses(conn, clean):
    baseline = search(conn, clean, limit=50)
    hammer(conn, clean, baseline[-1].name, baseline[-1].path, baseline[-1].kind)
    assert search(conn, clean, limit=50, feedback=False) == baseline
    assert search(conn, clean, limit=50, env={"CCCTX_FEEDBACK_CAP": "0"}) == baseline
    assert all(r.feedback == 0 and r.relevance == r.score for r in baseline)


def test_feedback_adds_no_visible_latency(conn, clean):
    """SC-007: the median of 5 searches with feedback is within 10% (plus 10 ms of timer noise) of the median without."""
    import statistics
    import time

    hammer(conn, clean, "spin_widget", count=20)

    def median(**kwargs):
        times = []
        for _ in range(5):
            started = time.perf_counter()
            search(conn, clean, **kwargs)
            times.append(time.perf_counter() - started)
        return statistics.median(times)

    median()  # warm up
    assert median() <= median(feedback=False) * 1.10 + 0.010


# --- listing, adding and resetting (US3, research R9) ---


def test_listing_shows_decay_by_weight(conn, clean):
    feedback.add(conn, clean.name, "spin_widget")
    feedback.add(conn, clean.name, "turn_helper", days_ago=30)
    feedback.add(conn, clean.name, "Widget.run", days_ago=60)
    rows = feedback.listing(conn, clean.name)
    assert [(r.name, r.kind, r.path, r.uses, round(r.weight, 2), round(r.boost, 2), r.indexed) for r in rows] == [
        ("spin_widget", "function", "pkg/widgets.py", 1, 1.0, 0.04, True),
        ("turn_helper", "function", "pkg/widgets.py", 1, 0.5, 0.02, True),
        ("Widget.run", "method", "pkg/widgets.py", 1, 0.25, 0.01, True)]
    assert all(r.last_used is not None for r in rows)


def test_add_gives_each_use_its_own_cli_session(conn, clean):
    assert feedback.add(conn, clean.name, "spin_widget", count=3) == ("pkg/widgets.py", "function")
    with conn.cursor() as cur:
        cur.execute("select client, session_id from ccctx_feedback where repo_id = :r", r=clean.id)
        rows = cur.fetchall()
    assert len({s for _, s in rows}) == 3 and all(c == "cli" and s.startswith("cli:") for c, s in rows)


def test_add_finds_pages_and_refuses_what_it_cannot_place(conn, clean, tmp_path):
    assert feedback.add(conn, clean.name, "pkg") == ("pkg", "page")
    with pytest.raises(ValueError, match="must not be negative"):
        feedback.add(conn, clean.name, "spin_widget", days_ago=-1)
    with pytest.raises(ValueError, match="not found: no indexed symbol or page named 'nothing_here'"):
        feedback.add(conn, clean.name, "nothing_here")
    with pytest.raises(ValueError, match="not found"):
        feedback.add(conn, clean.name, "spin_widget", path="pkg/m1.py")
    with pytest.raises(ValueError, match="no repository named 't-nowhere'"):
        feedback.add(conn, "t-nowhere", "spin_widget")


def test_a_name_at_several_paths_needs_a_path(conn, tmp_path):
    repo = make_repo(conn, tmp_path / "twice", {"a/x.py": "def same():\n    return 1\n",
                                                "b/x.py": "def same():\n    return 2\n"})
    try:
        with pytest.raises(ValueError, match="'same' is at several paths; pass --path:\na/x.py\nb/x.py"):
            feedback.add(conn, repo.name, "same")
        assert feedback.add(conn, repo.name, "same", path="b/x.py") == ("b/x.py", "function")
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from ccctx_repos where id = :i", i=repo.id)
        conn.commit()


def test_a_use_of_removed_code_lists_unindexed_and_boosts_nothing(conn, clean):
    write(conn, use(clean, "gone_widget"))
    [row] = feedback.listing(conn, clean.name)
    assert (row.name, row.indexed) == ("gone_widget", False)
    assert "gone_widget" not in {r.name for r in search(conn, clean, "gone widget", limit=50)}


def test_reset_one_repository(conn, clean, tmp_path):
    other = make_repo(conn, tmp_path / "other", {"pkg/o.py": "def other_fn():\n    return 1\n"})
    try:
        feedback.add(conn, clean.name, "spin_widget", count=2)
        feedback.add(conn, other.name, "other_fn")
        assert feedback.reset(conn, clean.name) == 2
        assert feedback.listing(conn, clean.name) == [] and len(feedback.listing(conn, other.name)) == 1
        # reset(conn, None) empties every repository's uses: not run here, it would wipe the real ones too
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from ccctx_repos where id = :i", i=other.id)
        conn.commit()
