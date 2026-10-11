# golestan_code_mcp

A code context MCP server backed by Oracle AI Database 26ai. This step sets up the local development environment: the database, the Python project, sample repositories and a stub server with one `health` tool.

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

`ccctx ingest <name>` indexes a registered git repository as it is on disk and replaces whatever was indexed before. It needs the embedding model and the hybrid index (see [Searching code](#searching-code)) and refuses to run without it. Each run prints a summary:

```sh
uv run ccctx ingest flask
```

```text
Scanned   236 files
Skipped     5 files (5 binary)
Indexed 1,353 symbols in 125.4s
Fallback  148 files as line windows (148 no grammar, 0 parse errors)
```

Most of that time is the database embedding every symbol for the hybrid index when the ingest commits.

How files are chosen:

- **Ignored**: files matched by the repository's `.gitignore` rules, nested ones included, as git applies them. Also symbolic links. `.git` is never read.
- **Vendored**: files under a directory named in `CCCTX_VENDOR_DIRS`, comma-separated, default `vendor,node_modules,third_party`.
- **Too large**: files over `CCCTX_MAX_FILE_BYTES`, default 1048576.
- **Binary**: files that contain a NUL byte or are not valid UTF-8.

Python, Go and JavaScript files are cut into one chunk per function, method and class (in Go, also struct and interface types). Each chunk has its exact line range, decorators included, and methods are named `Type.method`. Other text files, and code with a syntax error, are cut into 60-line windows that overlap by 10 lines, with kind `chunk`. Every symbol records the commit the repository was at. A working tree with uncommitted changes is indexed as it is on disk, and a warning is printed. One JSON record per run goes to stderr.

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
- **At most 10 results.** Query text is always taken literally: `_`, `-`, `&`, quotes and parentheses are never search operators.
- **Why each result ranked**: `SCORE` is the fused score (or the single side's score in `text` and `vector` mode). `KEYWORD` and `SEMANTIC` are its rank on each side among the repositories searched, with `-` when that side did not find it.

Fusion settings:

| Setting | Default | Meaning |
|---|---|---|
| `CCCTX_TEXT_WEIGHT` | `1` | Weight of the keyword side in the fusion |
| `CCCTX_VECTOR_WEIGHT` | `1` | Weight of the semantic side in the fusion |
| `CCCTX_SEARCH_CANDIDATES` | `100` | Rows asked of the index before scoping and the cap of 10; widened automatically, up to 4000, when `--repo` filters most of them out |

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

The server talks MCP over standard input and output. It has three tools:

- `health` returns the server name, version and whether the database is reachable.
- `list_repositories` returns the registered repositories with their paths and file and symbol counts, the same as `ccctx repo list`.
- `search` takes a `question`, an optional `mode` (`hybrid`, `text` or `vector`) and optional `repos` (or a single `repo`), and returns the same results as `ccctx search`: `status` `ok` or `unavailable`, and each result's `score`, `keyword_rank`, `semantic_rank` and `similarity`.

A stopped database gives `"database": "unreachable"` with a reason, not an error.

**MCP Inspector:**

```sh
uv run mcp dev src/golestan_code_mcp/server.py
```

Open the URL it prints, connect, then **Tools → List Tools → health → Run Tool**.

**Claude Code:** add the server from the project root:

```powershell
claude mcp add golestan_code_mcp -- uv run --directory "$PWD" golestan-code-mcp    # PowerShell
```
```sh
claude mcp add golestan_code_mcp -- uv run --directory "$(pwd)" golestan-code-mcp  # POSIX
```

Or put this in a `.mcp.json` (use your absolute project path):

```json
{
  "mcpServers": {
    "golestan_code_mcp": {
      "command": "uv",
      "args": ["run", "--directory", "<absolute project path>", "golestan-code-mcp"]
    }
  }
}
```

Then run `/mcp` in Claude Code and ask it to call the `health` tool.

## Layout

```text
compose.yaml              Oracle AI Database 26ai Free, named volume oradata
db/startup/01_app_user.sh every start: ensures DB_USER exists in FREEPDB1
db/startup/02_vector_memory.sh first start: enables the vector memory pool, restarts once
src/golestan_code_mcp/    config, db check, sample cloner, MCP server
  schema.py               ordered, re-runnable migrations
  registry.py             register and list repositories (shared by CLI and MCP)
  chunker.py              tree-sitter chunks for Python, Go, JavaScript; line-window fallback
  ingest.py               list files with read-only git, skip rules, store one repository's index
  semantic.py             model and hybrid index setup, hybrid / text / vector search
  cli.py                  the ccctx command
tests/                    pytest suite and chunker fixtures
```
