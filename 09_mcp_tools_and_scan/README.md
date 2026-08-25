# mcp_tools_and_scan

**Theme.** Tool descriptions are instructions the model reads and the user
never sees, and nothing in the MCP protocol binds an approval decision to
the exact artifact approved - a time-of-check to time-of-use (TOCTOU) gap
that lets metadata change after approval, turning an unpinned description
update into indirect prompt injection. With exposed MCP servers numbering in
the hundreds of thousands, the defence pattern is a gateway that treats tool
metadata like a dependency lockfile: pin the hash of everything the model
will read at approval time, verify on every connection, hard-fail on drift.

**Stack.** The official `mcp` Python SDK on both sides - two real FastMCP
servers (`server.py` clean, `poisoned_server.py` carrying an inert
instruction payload in one tool description) spoken to over stdio by a real
client in `main.py` - plus `mcp-scan` available via uvx for production-grade
scanning. No API key required; this exercise is pure protocol.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q                 # runs fully offline

    docker build -t mcp_tools .
    docker run --rm mcp_tools

    # Optional: scan a real MCP client config on your machine
    uvx mcp-scan

## What to look for

Pin the clean server: two hashes land in `tools.lock.json`. Verify the same
server: clean. Verify the poisoned variant - which has identical names,
schemas and code, differing only in one description - and the gateway logic
reports `get_price` drifted while `list_stock` stays trusted; the heuristic
pass separately flags the instruction-shaped phrases. The transferable
lesson: a changed description is a changed prompt, so approval must bind to
content hashes, not to server names.
