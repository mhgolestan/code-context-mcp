"""Retrieval evaluation set (constitution VI): queries against the real flask and gin index -> expected locations.

Versioned with the code: a change to ranking, chunking or embedding that lowers these numbers must be justified.
Skips unless the hybrid index is built and both sample repositories are ingested.
"""

import pytest

from golestan_code_mcp import db, registry, semantic
from golestan_code_mcp.config import load

EVAL_VERSION = 1

# SC-002: identifier -> its definitions (repo, path, symbol name); any one at rank 1 is a hit
IDENTIFIERS = {
    "url_for": {("flask", "src/flask/app.py", "Flask.url_for"), ("flask", "src/flask/helpers.py", "url_for")},
    "make_response": {("flask", "src/flask/app.py", "Flask.make_response"),
                      ("flask", "src/flask/helpers.py", "make_response")},
    "render_template": {("flask", "src/flask/templating.py", "render_template")},
    "jsonify": {("flask", "src/flask/json/__init__.py", "jsonify")},
    "send_file": {("flask", "src/flask/helpers.py", "send_file")},
    "full_dispatch_request": {("flask", "src/flask/app.py", "Flask.full_dispatch_request")},
    "handleHTTPRequest": {("gin", "gin.go", "Engine.handleHTTPRequest")},
    "addRoute": {("gin", "gin.go", "Engine.addRoute"), ("gin", "tree.go", "node.addRoute")},
    "getValue": {("gin", "tree.go", "node.getValue")},
    "Next": {("gin", "context.go", "Context.Next")},
}

# SC-003: routing or dispatch code, per repository
ROUTING_QUESTION = "how are incoming requests routed to handlers"
ROUTING = {
    "flask": {"src/flask/app.py", "src/flask/sansio/app.py", "src/flask/sansio/scaffold.py", "src/flask/views.py"},
    "gin": {"gin.go", "routergroup.go", "tree.go"},
}


@pytest.fixture(scope="module")
def conn():
    try:
        c = db.connect(load())
        semantic.require_model(c)
    except Exception:
        pytest.skip("database unreachable or hybrid index not built")
    indexed = {r.name for r in registry.list_repos(c) if r.symbols}
    if not {"flask", "gin"} <= indexed:
        c.close()
        pytest.skip("flask and gin are not both ingested")
    yield c
    c.close()


def where(results):
    return [(r.repo, r.path, r.name) for r in results]


def test_url_for_definitions_first(conn):  # SC-001
    found = where(semantic.search(conn, "url_for"))
    assert set(found[:2]) == IDENTIFIERS["url_for"]


def test_identifier_definitions_at_rank_one(conn):  # SC-002
    top = {q: where(semantic.search(conn, q))[:5] for q in IDENTIFIERS}
    top1 = [q for q, hits in top.items() if hits and hits[0] in IDENTIFIERS[q]]
    top5 = [q for q, hits in top.items() if IDENTIFIERS[q] & set(hits)]
    print(f"eval v{EVAL_VERSION}: top-1 {len(top1)}/{len(top)}, top-5 {len(top5)}/{len(top)}")
    assert len(top1) >= 9, f"rank-1 misses: {sorted(set(top) - set(top1))}"


def test_routing_question_finds_both_repositories(conn):  # SC-003
    found = where(semantic.search(conn, ROUTING_QUESTION))
    assert {repo for repo, path, _ in found if path in ROUTING.get(repo, ())} == set(ROUTING)


def test_repository_scope(conn):  # SC-004
    assert {r.repo for r in semantic.search(conn, "middleware chain", ["gin"])} == {"gin"}


def test_out_of_domain_question(conn):  # SC-005
    assert len(semantic.search(conn, "recipe for sourdough bread")) <= 3
