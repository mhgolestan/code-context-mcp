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

**2. Start the database.** The first start creates the database and the application user, which takes a few minutes.

```sh
docker compose up -d
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

The server talks MCP over standard input and output. It has one tool, `health`, which returns the server name, version and whether the database is reachable. A stopped database gives `"database": "unreachable"` with a reason, not an error.

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
src/golestan_code_mcp/    config, db check, sample cloner, MCP server
tests/                    pytest suite and parser fixtures
```
