"""golestan_code_mcp stdio MCP server. Logs go to stderr; stdout carries the protocol."""

from typing import Literal

import oracledb
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from golestan_code_mcp import __version__, db
from golestan_code_mcp.config import ConfigError, load

NAME = "golestan_code_mcp"
mcp = FastMCP(NAME)


class Health(BaseModel):
    server: str
    version: str
    database: Literal["reachable", "unreachable"]
    reason: str | None


@mcp.tool(description="Report the server name, its version, and whether the database is reachable.")
def health() -> Health:
    try:
        db.connect(load()).close()
    except ConfigError as e:
        return Health(server=NAME, version=__version__, database="unreachable", reason=str(e))
    except oracledb.Error as e:
        return Health(server=NAME, version=__version__, database="unreachable", reason=db.describe(e))
    return Health(server=NAME, version=__version__, database="reachable", reason=None)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
