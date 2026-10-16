# CLAUDE.md

Guidance for AI coding agents (and people) working in this repository.

## What this is

A code context MCP server backed by Oracle AI Database 26ai: it indexes codebases, searches them, follows their calls,
keeps a module wiki and an agent memory, and serves it all over MCP. See [README.md](README.md) and [docs/](docs/).
The source in `src/` is the package `golestan_code_mcp`; the CLI is `ccctx`, and `ccctx mcp` runs the server.

## Commands

```sh
uv sync                                     # install; close MCP clients first, they lock ccctx.exe on Windows
docker compose up -d                        # the database, container oracle26ai on port 1521
uv run golestan-check-db                    # is the database up and is VECTOR working?
uv run pytest                               # the suite; database tests skip with a reason when it is down
uv run pytest tests/search/test_query.py -q # one file
uv run ccctx --help                         # the CLI; ccctx mcp is the MCP server over stdio
uv run ccctx eval --gold gold.json          # retrieval recall@10, after any ranking or chunking change
```

Settings come from the environment, then `.env` at the project root. Never print, log or commit `.env`.

## Rules

- **Small files.** No `.py` file under `src/` or `tests/` over 300 lines; each opens with a docstring stating its
  one concern. A long function is not cut in two to meet the limit.
- **Layout list.** Every source and test file is listed with its purpose in the Layout section below.
  `tests/test_layout.py` fails otherwise. A new `src/` folder is also a package in `pyproject.toml`.
- **Imports.** Absolute. Functions and module-level values from another file are used through their module
  (`runs.record`), so a test's patch hits what the code calls; classes may be imported by name. No import cycles.
- **Tests mirror `src/`.** The tests of `src/<area>/<file>.py` are in `tests/<area>/test_<file>.py`. Shared setup is
  in `tests/conftest.py` (the `conn` fixture) and `tests/support.py`; a folder's own fixtures are in its `conftest.py`.
- **Test first.** A behavior change starts with a failing test. Every MCP tool has contract tests in `tests/server/`.
- **Contracts.** MCP tool names, inputs and results are public: a breaking change needs a major version and a note
  in [docs/mcp-server.md](docs/mcp-server.md). Schema changes are new, re-runnable migrations in `src/schema.py`.
- **Private by default.** Indexing (chunking, embeddings, search, graph) never sends code out of the database. Only
  the optional memory model may, and only what [docs/memory.md](docs/memory.md) lists.
- **Simple.** No new dependency, layer or abstraction without a need the spec states.

## Spec Kit

Features are built with [GitHub Spec Kit](https://github.github.com/spec-kit/): `/speckit-specify`, then
`/speckit-plan`, `/speckit-tasks` and `/speckit-implement`, each feature in `specs/NNN-name/` on its own branch. The
principles in `.specify/memory/constitution.md` govern every plan. Earlier specs are records and keep their old paths.

## Layout

```text
README.md                       what the project is, the quick start, links to the docs
LICENSE                         the MIT License
CLAUDE.md                       this file
docs/                           the reference, one page per topic
  setup.md                      prerequisites, the database, installing, checking that data persists
  indexing.md                   registering and ingesting codebases
  search.md                     searching, usage feedback, evaluating retrieval
  graph.md                      callers, calls and impact
  wiki.md                       the module wiki
  memory.md                     agent memory
  mcp-server.md                 the MCP tools, registering clients, the change history
compose.yaml                    Oracle AI Database 26ai Free, named volume oradata
db/startup/01_app_user.sh       every start: ensures DB_USER exists in FREEPDB1
db/startup/02_vector_memory.sh  first start: enables the vector memory pool, restarts once
src/                            the package golestan_code_mcp, installed from here (see pyproject.toml)
  __init__.py                   the package and its version
  config.py                     settings from the environment, with .env as a fallback
  db.py                         the database connection, and golestan-check-db
  registry.py                   register and list repositories (shared by the CLI and the MCP server)
  samples.py                    golestan-clone-samples: clone the sample repositories
  schema.py                     ordered, re-runnable migrations
  parsing/                      tree-sitter chunks, calls, imports and docs; line-window fallback
    __init__.py                 the area
    tables.py                   the language tables, and register() for each language
    common.py                   the chunk, call and import types; tree and doc helpers every language shares
    chunks.py                   detect a file's language, parse it into chunks, or cut it into line windows
    languages/                  one file per language
      __init__.py               registers every language, in order
      python.py                 Python's rules
      go.py                     Go's rules
      javascript.py             JavaScript's rules
      csharp.py                 C#'s rules
      typescript.py             TypeScript's and TSX's rules
      c_family.py               C's and C++'s rules
      java.py                   Java's rules
      kotlin.py                 Kotlin's rules
      rust.py                   Rust's rules
      php.py                    PHP's rules
      ruby.py                   Ruby's rules
      swift.py                  Swift's rules
  indexing/                     index one repository, in full or incrementally
    __init__.py                 the area
    files.py                    list files with read-only git; skip rules; read the ones to index
    write.py                    store symbols, calls and imports; keep unchanged symbols
    ingest.py                   ingest: plan, parse, write in one transaction, then the graph, wiki and memory
  search/                       hybrid / text / vector search, usage feedback, the evaluation
    __init__.py                 the area
    model.py                    the embedding model, the hybrid index and the code lexer
    candidates.py               the keyword query, and the symbol, page and definition rows a search starts from
    ranking.py                  fuse the two sides, apply the usage boost, place wiki pages
    query.py                    the search, its language and count filters, and its results
    feedback.py                 usage feedback: uses, weights and decay, the bounded boost
    evaluation.py               the gold evaluation (`ccctx eval`)
  graph/                        call and import edges from stored refs; queries via GRAPH_TABLE
    __init__.py                 the area
    imports.py                  resolve imports to indexed files, per language; outside packages
    build.py                    resolve calls to definitions, and rewrite a repository's edges
    queries.py                  callers, calls, impact and definitions
  wiki/                         module pages from the index: compile, cite, mark stale, recompile, read
    __init__.py                 the area
    text.py                     safe sentences from documentation, and readable lists of names
    compile.py                  write one directory's page with its citations
    store.py                    the pages' hybrid index, and writing pages
    build.py                    build a repository's pages, mark them stale, recompile them
    read.py                     list pages, and read one with its citations' current locations
  memory/                       agent memory: record runs, distil facts, re-check, recall
    __init__.py                 the area
    store.py                    the AI Agent Memory package, its schema, and the configured model
    runs.py                     record every tool call off the answering path, and list the runs
    findings.py                 what one run says about its subject
    distill.py                  a finding's text, from the template or a configured model
    consolidate.py              turn pending runs into facts
    recheck.py                  after an ingest, mark facts stale, repoint or re-ask them
    recall.py                   list, recall and forget facts; fact hits in search
  server/                       the nine MCP tools (`ccctx mcp`)
    __init__.py                 loads the tools in listing order and refuses undeclared arguments
    __main__.py                 `python -m golestan_code_mcp.server`
    app.py                      the FastMCP app, and what every tool shares
    search_tools.py             search_code and get_symbol
    graph_tools.py              find_callers, find_callees and impact_of
    wiki_tools.py               read_wiki
    repository_tools.py         list_repositories
    memory_tools.py             recall and mark_used
  cli/                          the ccctx command
    __init__.py                 the ccctx entry point
    __main__.py                 `python -m golestan_code_mcp.cli`
    entry.py                    parse, refuse bad input, connect, migrate and run a command
    parser.py                   every subcommand and its options
    output.py                   plain-text tables and the ingest summary
    index_commands.py           repo, ingest, symbols, stats, model load, db migrate
    search_commands.py          search, callers, calls, impact, eval
    wiki_commands.py            wiki build, list, show, citations
    memory_commands.py          memory list, consolidate, facts; feedback list, add, reset
tests/                          the pytest suite; its folders mirror src/
  conftest.py                   the shared `conn` fixture, and keeping tool calls out of real repositories' memory
  support.py                    git and file helpers several folders use
  test_layout.py                file size, docstrings, this Layout list, the package list
  test_docs.py                  every relative link in the README, CLAUDE.md and docs/ resolves
  test_config.py                settings and their refusals
  test_db.py                    the database check
  test_registry.py              registering and listing repositories
  test_samples.py               cloning the sample repositories
  fixtures/                     parser input files, one folder per language
  parsing/
    helpers.py                  the fixtures folder, and a parse's outline, calls, imports and docs
    test_chunks.py              chunks, line windows, language detection, what a parse returns
    test_languages.py           the files each language is detected for, and files that must not parse
    languages/
      helpers.py                parse a fixture file and read its definitions, docs and calls
      test_python.py            Python's rules
      test_go.py                Go's rules
      test_javascript.py        JavaScript's rules
      test_csharp.py            C#'s rules
      test_typescript.py        TypeScript's and TSX's rules
      test_c_family.py          C's and C++'s rules
      test_java.py              Java's rules
      test_kotlin.py            Kotlin's rules
      test_rust.py              Rust's rules
      test_php.py               PHP's rules
      test_ruby.py              Ruby's rules
      test_swift.py             Swift's rules
  indexing/
    conftest.py                 the throwaway repositories the ingest tests ingest
    helpers.py                  sample files, and readers of what an ingest stored
    test_files.py               skip reasons, the vendor setting, read-only git
    test_ingest.py              a full ingest, re-ingesting, failures and refusals
    test_ingest_incremental.py  an incremental ingest re-reads only what changed
    test_ingest_extraction.py   graph edges, docs, visibility and namespaces; re-reading old extractions
    test_ingest_wiki.py         pages marked stale and recompiled by an ingest
  search/
    conftest.py                 throwaway repositories and pages, and a repository for feedback
    helpers.py                  throwaway symbols, a ledger of pages, readers of hits
    feedback_helpers.py         recording uses, and searching with and without their boost
    test_model.py               the model and the indexes
    test_candidates.py          the keyword query
    test_query.py               refused input, ranking, modes and scope
    test_query_filters.py       language, limit, snippet and defaults
    test_ranking.py             where wiki pages are placed among the hits
    test_ranking_feedback.py    the usage boost in ranking
    test_feedback.py            uses, weights, listing, adding and resetting
    test_evaluation.py          retrieval quality, and the gold set
  graph/
    conftest.py                 the ingested repositories
    helpers.py                  two tiny repositories, and readers of the stored edges
    test_imports.py             import resolution
    test_build.py               call resolution
    test_queries.py             callers, calls, impact and definitions
    test_languages.py           the graph for the languages beyond Python, Go and JavaScript
  wiki/
    conftest.py                 a repository with its wiki built, and a fresh one per test
    helpers.py                  sample repositories, and readers of pages and citations
    test_compile.py             which directories get a page, what it says, its citations
    test_compile_languages.py   pages for the other languages, and their test files last
    test_build.py               building, refusing, marking stale and recompiling
    test_read.py                pages and read, and their refusals
  memory/
    conftest.py                 a connection that needs the model, and a fresh repository
    helpers.py                  sample runs and repositories, tool calls, readers of runs and facts
    test_store.py               the package client, the model, what importing memory loads
    test_runs.py                what a run keeps, and the uses linked to runs
    test_findings.py            what one run says about its subject
    test_distill.py             the model's text, its failures, unknown names
    test_consolidate.py         merging, confidence, contradictions, idempotence, scale
    test_recheck.py             facts after removed, moved and edited code
    test_recall.py              recall, facts in search, forgetting
  server/
    conftest.py                 pages, recorded runs, and a used repository with a fresh session
    helpers.py                  calling tools as a client does, the tool contracts, the core results
    test_app.py                 the listing, refused arguments, outages, logs and recorded runs
    test_search_tools.py        search_code and get_symbol
    test_graph_tools.py         find_callers, find_callees and impact_of
    test_wiki_tools.py          read_wiki
    test_repository_tools.py    list_repositories
    test_memory_tools.py        recall and mark_used
  cli/
    conftest.py                 the database, the model, and prepared repositories
    helpers.py                  run ccctx in-process, sample summaries, table readers
    test_entry.py               ccctx mcp
    test_output.py              tables and ingest summaries
    test_index_commands.py      repo, ingest, symbols, stats, model load, db migrate
    test_search_commands.py     search
    test_graph_commands.py      callers, calls, impact and eval
    test_wiki_commands.py       the wiki commands
    test_memory_commands.py     the memory and feedback commands
  schema/
    conftest.py                 a throwaway repository row
    helpers.py                  the expected tables, and row inserts
    test_core.py                migrations, repositories, files, symbols, revisions, language columns
    test_graph.py               the code graph's tables
    test_wiki.py                the wiki's tables
    test_memory.py              the memory and feedback tables
gold.json                       the retrieval evaluation set (`ccctx eval`)
```
