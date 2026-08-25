"""MCP tool trust: pin what you approved, detect when it drifts.

Every MCP server a model can reach extends the attack surface: tool
descriptions are instructions the model reads and the user never sees, and
nothing in the protocol binds an approval decision to the exact artifact
approved - a time-of-check to time-of-use (TOCTOU) gap that lets metadata
change AFTER approval, turning an unpinned description update into indirect
prompt injection. The 2026 defence pattern is a gateway that treats tool
metadata like a dependency lockfile:

  PIN     - connect to the approved server, hash each tool's
            (name + description + schema) into tools.lock.json
  VERIFY  - on every later connection, re-hash and compare; ANY drift is a
            hard failure, because a changed description is a changed prompt
  FLAG    - independent of drift, run heuristics for instruction-shaped
            language hiding in descriptions

Both servers here are real MCP servers spoken to over stdio with the real
client library. mcp-scan is the production scanner covering the
same ground plus server-side analysis; see the README.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
LOCKFILE = HERE / "tools.lock.json"

# Phrases that are instruction-shaped rather than descriptive. Heuristics
# are a tripwire, not a proof: the pin/verify mechanism is the hard control.
SUSPICIOUS = [
    r"ignore (all )?previous", r"do not tell", r"before answering",
    r"<important>", r"include every", r"secretly", r"hide this",
    r"email address", r"send .* to",
]


async def list_tools(server_script: str) -> list[dict]:
    """Connect to an MCP server over stdio and fetch its advertised tools.
    This is the same handshake a real client (an IDE, an agent) performs."""
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / server_script)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "schema": t.inputSchema,
                }
                for t in result.tools
            ]


def tool_hash(tool: dict) -> str:
    """One stable hash over everything the model will read about the tool."""
    canonical = json.dumps(
        {"name": tool["name"], "description": tool["description"], "schema": tool["schema"]},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def pin(server_script: str) -> dict:
    """Approve the server AS IT IS NOW: record a hash per tool."""
    tools = asyncio.run(list_tools(server_script))
    lock = {t["name"]: tool_hash(t) for t in tools}
    LOCKFILE.write_text(json.dumps(lock, indent=2))
    return lock


def verify(server_script: str) -> dict:
    """Re-fetch the server's tools and compare against the pinned hashes.

    Returns {"ok": bool, "drifted": [names], "added": [names], "removed": [names]}.
    Drift in a DESCRIPTION is treated identically to drift in a schema:
    both change what the model will do.
    """
    lock = json.loads(LOCKFILE.read_text())
    tools = asyncio.run(list_tools(server_script))
    current = {t["name"]: tool_hash(t) for t in tools}
    drifted = [n for n in lock if n in current and current[n] != lock[n]]
    added = [n for n in current if n not in lock]
    removed = [n for n in lock if n not in current]
    return {"ok": not (drifted or added or removed),
            "drifted": drifted, "added": added, "removed": removed}


def heuristic_flags(server_script: str) -> dict[str, list[str]]:
    """Scan tool descriptions for instruction-shaped phrases."""
    tools = asyncio.run(list_tools(server_script))
    flags: dict[str, list[str]] = {}
    for t in tools:
        hits = [p for p in SUSPICIOUS if re.search(p, t["description"], re.IGNORECASE)]
        if hits:
            flags[t["name"]] = hits
    return flags


# -------------------------------------------------------------------- walk --
def walk() -> None:
    print("=== 1. PIN the approved server (server.py) ===")
    lock = pin("server.py")
    for name, digest in lock.items():
        print(f"  pinned {name}: {digest[:16]}...")

    print("\n=== 2. VERIFY the same server: no drift expected ===")
    print(f"  {verify('server.py')}")

    print("\n=== 3. Post-approval drift: same tools, one description changed ===")
    print("  (poisoned_server.py stands in for 'the server's metadata mutated after approval')")
    result = verify("poisoned_server.py")
    print(f"  {result}")
    print("  >> Drift detected. A changed description IS a changed prompt; the "
          "gateway refuses the connection until a human re-approves.")

    print("\n=== 4. Heuristic scan of the poisoned server's descriptions ===")
    for name, hits in heuristic_flags("poisoned_server.py").items():
        print(f"  {name}: matched {hits}")

    print("\nProduction note: mcp-scan runs without installing it here; point it at "
          "a real client config (uvx mcp-scan) for the same pinning idea plus "
          "hosted analysis of tool poisoning, and see the README for the gateway "
          "pattern this exercise miniaturises.")


if __name__ == "__main__":
    walk()
