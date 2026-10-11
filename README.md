# golestan_code_mcp

A code context MCP server backed by Oracle AI Database 26ai. It indexes codebases into the database, searches them by name and meaning, follows their calls, and serves all of it to AI coding tools over MCP (`ccctx mcp`).

## Prerequisites

- Docker (tested with Rancher Desktop), with about 12 GB free for the database image and data
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 for you if needed)
- git
- Node.js, only for the MCP Inspector (`npx`)

## Setup

Every command below is the same in PowerShell and a POSIX shell unless shown otherwise. Run them from the project root.

**1. Create your `.env`.** Copy the example, then replace both passwords. Oracle passwords need at least 8 characters with an upper-case letter, a lower-case letter and a digit.

```powershell
Copy-Item .env.example .env     # PowerShell
```
```sh
cp .env.example .env            # POSIX
```

`.env` is ignored by git. No script prints its passwords.

**2. Start the database.** It runs as the container `oracle26ai` on port 1521. The first start creates the database and the application user, which takes a minute or two.

```sh
docker compose up -d
docker ps
docker logs oracle26ai | tail -5     # look for DATABASE IS READY TO USE!
```

If port 1521 is taken, set `DB_PORT` in `.env` before starting.

**3. Install the Python project.**

```sh
uv sync
```

**4. Check the database.** This waits up to 5 minutes for a first start, then prints the connection, the version, the vector check and the smoke table's row count.

```sh
uv run golestan-check-db
```

Exit codes: `0` ok, `1` a setting is missing, `2` unreachable or still starting, `3` connected but the `VECTOR` check failed.

**5. Clone the sample repositories.** Flask (Python) and Gin (Go) go to `../ccctx-repos` by default (set `CCCTX_REPOS_DIR` to change it; it must be outside this project). Running it again skips what is already there.

```sh
uv run golestan-clone-samples
```

**6. Run the tests.** Database tests are skipped, with a reason, when the database is down. The parser tests always run.

```sh
uv run pytest
```

## Registering codebases

`ccctx` records the codebases to index. The schema (tables `ccctx_repos`, `ccctx_files`, `ccctx_symbols` and `ccctx_migrations`) is created the first time any `ccctx` command runs, or explicitly with `uv run ccctx db migrate`.

```sh
uv run ccctx repo add ../ccctx-repos/flask --name flask
uv run ccctx repo add ../ccctx-repos/gin --name gin
uv run ccctx repo list
```

```text
NAME   PATH                         FILES  SYMBOLS
flask  /home/you/ccctx-repos/flask      0        0
gin    /home/you/ccctx-repos/gin        0        0
```

Paths are stored absolute, with `~` and `..` resolved. Names are 1 to 64 lowercase letters, digits, `-` or `_`. A name or directory can be registered only once. Registering only checks that the path is a directory; it never reads or runs anything inside it.

Exit codes: `0` ok, `1` refused input or a missing setting, `2` the database is unreachable or still starting.

## Ingesting a codebase

`ccctx ingest <name>` indexes a registered git repository as it is on disk. It needs the embedding model and the hybrid index (see [Searching code](#searching-code)) and refuses to run without it. The first ingest indexes everything and prints a summary:

```sh
uv run ccctx ingest flask
```

```text
Scanned   236 files
Skipped     5 files (5 binary)
Indexed 1,353 symbols in 653.3s
Fallback  148 files as line windows (148 no grammar, 0 parse errors)
```

Almost all of that time is the database embedding every symbol for the hybrid index when the ingest commits. On the Free edition container this is CPU-limited, about half a second per symbol.

Later ingests only do the work the changes require. Each repository records the commit it was last indexed at. The next ingest asks git what differs from that commit, adds untracked files and paths that have gone, and compares each file's content hash with what is stored. Only added and changed files are parsed again, and inside a changed file only the functions and classes whose code changed are embedded again: the rest keep their rows and just move to their new lines. Files that were deleted, renamed away, newly ignored, or now too large or binary lose their symbols, so no search result can point at code that is no longer there. Everything else is left alone.

```text
Changed     9 files (2 added, 6 changed, 1 removed)
Unchanged 227 files
Skipped     0 files
Indexed    84 symbols in 6.3s
```

A rename counts as one file removed and one added. `Skipped` and `Fallback` count only the files that run read. When nothing changed:

```text
flask is already up to date at d086db856be1; run with --full to re-index anyway
```

`ccctx ingest <name> --full` throws the repository's index away and rebuilds it, like the first ingest. Use it after changing `CCCTX_VENDOR_DIRS` or `CCCTX_MAX_FILE_BYTES`, or after an upgrade that changes how code is chunked: an incremental run only looks at files whose content changed, so it will not re-apply new rules to the rest. Ingest also rebuilds in full on its own, with a note on stderr, when no revision is recorded (a repository indexed before this feature) or when the recorded commit no longer exists (history rewritten, or the clone replaced).

A failed ingest changes nothing: the index and the recorded commit stay as they were.

How files are chosen:

- **Ignored**: files matched by the repository's `.gitignore` rules, nested ones included, as git applies them. Also symbolic links. `.git` is never read.
- **Vendored**: files under a directory named in `CCCTX_VENDOR_DIRS`, comma-separated, default `vendor,node_modules,third_party`.
- **Too large**: files over `CCCTX_MAX_FILE_BYTES`, default 1048576.
- **Binary**: files that contain a NUL byte or are not valid UTF-8.

Python, Go and JavaScript files are cut into one chunk per function, method and class (in Go, also struct and interface types). Each chunk has its exact line range, decorators included, and methods are named `Type.method`. Other text files, and code with a syntax error, are cut into 60-line windows that overlap by 10 lines, with kind `chunk`. Every symbol records the commit the repository was at when its file was last read. A working tree with uncommitted changes is indexed as it is on disk, and a warning is printed. One JSON record per run goes to stderr, with the mode (`full`, `incremental` or `current`) and the added, changed, removed and unchanged counts.

Ingestion only reads files. Git runs with the repository's fsmonitor and hooks disabled, so nothing inside the repository is executed.

Inspect the result:

```sh
uv run ccctx symbols --repo flask --limit 5      # parsed symbols first; --kind chunk shows windows
uv run ccctx stats --by-language                 # files, symbols and fallback chunks per repo and language
```

## Searching code

Search combines keyword matching and semantic matching, both served by one hybrid vector index in the database. The database embeds code itself with Oracle's prebuilt ONNX `all_MiniLM_L12_v2` model, loaded as `ALL_MINILM_L12_V2`. There is no API key and no network call at search time, and source code never leaves the database to be embedded.

**1. Set up the model and the index once.** This step:

- downloads the 133 MB model file into `.cache/models/` (git-ignored)
- loads it with `DBMS_VECTOR.LOAD_ONNX_MODEL` and records its name and dimension
- builds the hybrid vector index over every symbol's kind, name, path and source

Building the index takes several minutes for flask and gin together. Running the command again does nothing. If you already ran it before hybrid search existed, run it once more: that builds the new index, and the symbols you already ingested are searchable without re-ingesting.

```sh
uv run ccctx model load
```

Check it from SQL:

```sql
SELECT model_name FROM user_mining_models;                                   -- ALL_MINILM_L12_V2
SELECT VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING 'hello world' as data) AS v;  -- 384 numbers
```

`CCCTX_MODEL_URL` and `CCCTX_MODEL_CACHE` change where the file comes from and where it is kept.

The index's vector half lives in the database's vector memory pool. The first container start after vector search was added sets the pool to 256 MB and restarts the database once (`db/startup/02_vector_memory.sh`). Later starts leave it alone. The index updates when an ingest commits, so a finished ingest is searchable at once.

**2. Search.**

```sh
uv run ccctx search "url_for"                                       # the definition comes first
uv run ccctx search "url_for" --mode text                           # only symbols containing url_for
uv run ccctx search "how are incoming requests routed to handlers"  # by meaning
uv run ccctx search "middleware chain" --repo gin                   # one repository
uv run ccctx search "routing" --repo flask,gin                      # or several (also: --repo flask --repo gin)
uv run ccctx search "middleware chain" --language go --limit 3      # one language, three results
```

```text
RANK  SCORE  KEYWORD  SEMANTIC  REPO   KIND      NAME           PATH                     LINES
   1  24.38        5         4  flask  method    Flask.url_for  src/flask/app.py         1102-1222
   2  21.82        6         7  flask  function  url_for        src/flask/helpers.py     200-251
   4  29.69        1        11  flask  class     TestUrlFor     tests/test_helpers.py    94-161
```

Modes, chosen with `--mode`:

- **`hybrid`** (default): keyword and semantic results fused with reciprocal rank fusion (`DBMS_HYBRID_VECTOR.SEARCH` with `RRF` and `UNION`). A symbol that both rank highly rises to the top. The keyword side matches any word of the query, and plain words also match their stems (`routed` finds `route`).
- **`text`**: keyword matching only. Every word of the query must appear.
- **`vector`**: meaning only.

How results are ranked:

- **Definitions first**: when the query is an identifier, the symbols named exactly that, or ending in `.<identifier>` such as `Flask.url_for`, come before everything else in `hybrid` and `text` mode. Matching ignores case.
- **Similarity floor**: a result found only by meaning must have a cosine similarity of at least 0.3, so an unrelated question returns nothing rather than ten weak matches. Keyword matches always count.
- **One window per file**: of a file's fallback line windows, only the best one is shown.
- **10 results by default**, `--limit` 1 to 50. `--language` keeps one language (`python`, `go`, `javascript`, `markdown` and the other languages the chunker assigns). Query text is always taken literally: `_`, `-`, `&`, quotes and parentheses are never search operators.
- **Why each result ranked**: `SCORE` is the fused score (or the single side's score in `text` and `vector` mode). `KEYWORD` and `SEMANTIC` are its rank on each side among the repositories searched, with `-` when that side did not find it.

Fusion settings:

| Setting | Default | Meaning |
|---|---|---|
| `CCCTX_TEXT_WEIGHT` | `1` | Weight of the keyword side in the fusion |
| `CCCTX_VECTOR_WEIGHT` | `1` | Weight of the semantic side in the fusion |
| `CCCTX_SEARCH_CANDIDATES` | `100` | Rows asked of the index before scoping and the result cap; widened automatically, up to 4000, when `--repo` filters most of them out |

## Code relationships

Ingest records three relationships from the parse tree of every Python, Go and JavaScript file:

- **defines**: the file each symbol is in.
- **imports**: the files of the same repository a file imports.
- **calls**: the symbol (or, for a module-level call, the file) making each call, the symbol called, and the line.

They form `ccctx_graph`, a SQL property graph (`CREATE PROPERTY GRAPH`) over the existing tables: files and symbols are its vertices, and imports, calls and defines its edges. Nothing is copied. Three commands query it with `GRAPH_TABLE … MATCH`:

```sh
uv run ccctx callers url_for                 # what calls this?
uv run ccctx calls url_for                   # what does this call?
uv run ccctx impact url_for --depth 3        # what could break: transitive callers, with hop counts
```

```text
CALLER          KIND      REPO   PATH                              LINE  CALLS
login_required  function  flask  examples/tutorial/flaskr/auth.py    25  url_for
...
url_for         function  flask  src/flask/helpers.py               244  Flask.url_for

FROM           LINE  CALLEE                    KIND    REPO   PATH                 LINES
Flask.url_for  1182  Flask.create_url_adapter  method  flask  src/flask/app.py     507-558
url_for         244  Flask.url_for             method  flask  src/flask/app.py     1102-1222

HOPS  NAME            KIND      REPO   PATH                              LINES
   1  login_required  function  flask  examples/tutorial/flaskr/auth.py  19-29
```

- **Names** are case-sensitive. A bare name (`url_for`) matches every function, method, class or type named `url_for` or `<something>.url_for`. A qualified name (`Flask.url_for`) matches only that definition. `CALLS` says which definition each call reaches.
- **`--depth`** is 1 to 5, default 3. Each symbol appears once, at its shortest distance, nearest first. Recursion and cycles are safe.
- **`--repo`** limits a command to the named repositories, as in search. Calls never cross repositories.
- The model is not needed, and ingest keeps the graph current: an incremental ingest leaves the same relationships as `--full`.

On flask, `ccctx callers url_for` finds all 51 calls to `url_for` inside functions and methods. `grep -rn url_for` finds 187 lines, but 136 of them are documentation, templates, comments, docstrings, strings and imports, and none of those appear. `impact url_for` stays at 36 symbols from depth 1 to 5: they are tests and view functions, which pytest and the router call, never other code. `impact Flask.url_for` grows from 1 symbol at depth 1 to 36 at depth 3.

Calls are matched to definitions by name, not by type inference:

- `x()` reaches functions and classes named `x`.
- `module.x()` on an imported module of the repository is treated the same way.
- `self.x()` prefers the method of the enclosing class.
- `obj.x()` reaches every method named `x`.
- A definition in the same file wins, then one in a file the caller imports, then any in the repository.

Dynamic calls, calls through variables, and the code a framework calls for you are not edges. Files indexed as line windows have no relationships.

The first ingest after upgrading re-reads every file for its relationships. Unchanged symbols keep their rows, so nothing is re-embedded: flask took 1.7 s.

## Checking that data persists

`uv run pytest` adds a row to the `ccctx_smoke` table on each run. Note the count, restart the container, and check it again:

```sh
uv run golestan-check-db
docker compose stop
docker compose start
uv run golestan-check-db --wait 300
```

The second count matches the first. `docker compose down` also keeps the data. Only `docker compose down -v` deletes it.

## The MCP server

`ccctx mcp` runs the server over standard input and output. There is no network port and no login: the client starts it as a child process. It finds its database settings the same way the CLI does, from the environment and then `.env` in this checkout, whatever directory the client starts it in.

**Install the command once**, from the project root, so a client can start `ccctx` from any directory:

```sh
uv tool install --editable .
```

The editable install keeps `.env` in reach. With a non-editable install there is no `.env` next to the package, so give the settings to the client instead: `claude mcp add ccctx -e DB_USER=... -e DB_PASSWORD=... -- ccctx mcp`.

**Tools.** Each tool's description says what it answers and when to use one of the others instead. They are all read-only.

| Tool | Parameters | Returns |
|---|---|---|
| `search_code` | `query`, `repo`, `language`, `limit` (default 10, 1 to 50) | Hits with repository, path, symbol name, kind, language, line range, a snippet (the first 10 lines, at most 800 characters) and the fused, keyword and semantic scores. The same hits as `ccctx search`. |
| `get_symbol` | `name`, `repo` | Up to 5 definitions with that name, each with its full indexed source, path and line range. A source over 400 lines is cut and marked `cut`. `total` and `truncated` say whether more matched. |
| `find_callers` | `name`, `repo` | The rows of `ccctx callers`. |
| `find_callees` | `name`, `repo` | The rows of `ccctx calls`. |
| `impact_of` | `name`, `depth` (default 3, 1 to 5), `repo` | The rows of `ccctx impact`, with hop counts. |
| `list_repositories` | none | The registered repositories, the same as `ccctx repo list`. |

- **Names** work as in the graph commands: a bare `url_for` matches every `url_for`, and `Flask.url_for` only that method.
- **Limits:** the graph tools return at most 200 rows, with `truncated` set when there were more.
- **Refusals:** bad input, an unknown repository or language, a missing model and a stopped database all come back as `"status": "unavailable"` with a reason. A call with an argument the tool does not take is refused.
- **Logging:** each call writes one line to standard error with the tool, time, status and result count.

**Changed in 1.0.0.**
- **Renamed tools:** `search` is now `search_code`, `callers` is now `find_callers`, `calls` is now `find_callees`, and `impact` is now `impact_of`.
- **Removed:** `health`. A failed call says when the database is unreachable.
- **Parameters:** the `repos` list became one `repo`, and `search` lost `question` (now `query`) and `mode` (always hybrid).
- **Old registrations:** `golestan-code-mcp` still starts the same server.

**MCP Inspector**: lists the tools and calls them by hand.

```sh
npx @modelcontextprotocol/inspector ccctx mcp
```

Open the URL it prints, connect, then **Tools → List Tools**. To compare with the CLI, call `search_code` with `{"query": "how are requests routed", "repo": "flask"}`: it returns the same 10 symbols in the same order as `ccctx search "how are requests routed" --repo flask`.

**Claude Code**:

```sh
claude mcp add ccctx -- ccctx mcp
```

Start a session in any other directory and ask, for example, "using ccctx, how does Flask decide which view function handles a request?".

**Which tool a client picks.** Five questions were run through Claude Code with `claude -p`, its built-in tools turned off and only this server connected. Each question was written so one tool is the right first step. The client called that tool first every time:

| Question | First call | Then |
|---|---|---|
| using ccctx, how does Flask decide which view function handles a request? | `search_code` | `get_symbol` ×6, a narrower `search_code`, `find_callers` |
| show me the full source of Flask.dispatch_request | `get_symbol` | none |
| what calls url_for in flask? | `find_callers` | none |
| if I change Flask.url_for, what could break? | `impact_of` | `get_symbol`, `find_callers`, `search_code` |
| what does Flask.full_dispatch_request call? | `find_callees` | `get_symbol` |

Its routing answer traced registration through `App.add_url_rule` into `url_map` and `view_functions`. It then traced each request through `Flask.wsgi_app`, `AppContext.match_request` and `Flask.dispatch_request`, citing paths and lines. It never asked to open a file.

## Layout

```text
compose.yaml              Oracle AI Database 26ai Free, named volume oradata
db/startup/01_app_user.sh every start: ensures DB_USER exists in FREEPDB1
db/startup/02_vector_memory.sh first start: enables the vector memory pool, restarts once
src/golestan_code_mcp/    config, db check, sample cloner, MCP server
  schema.py               ordered, re-runnable migrations
  registry.py             register and list repositories (shared by CLI and MCP)
  chunker.py              tree-sitter chunks for Python, Go, JavaScript; line-window fallback
  ingest.py               list files with read-only git, skip rules, full or incremental re-index of one repository
  semantic.py             model and hybrid index setup, hybrid / text / vector search, language and count filters
  graph.py                call and import edges from stored refs; callers, calls, impact via GRAPH_TABLE; definitions
  server.py               the six MCP tools (`ccctx mcp`)
  cli.py                  the ccctx command
tests/                    pytest suite and chunker fixtures
```
