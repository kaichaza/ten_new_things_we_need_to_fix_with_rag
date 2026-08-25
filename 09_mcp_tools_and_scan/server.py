"""The CLEAN fruit-shop MCP server: two honest tools over stdio.

An MCP server advertises tools as (name, description, input schema). The
DESCRIPTION is prose the calling model reads and the human user typically
never sees - which is exactly why it is the attack surface the companion
poisoned_server.py exploits.
"""
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PRICE_TEXT = (Path(__file__).parent / "corpus" / "price_list.md").read_text()

mcp = FastMCP("fruit-shop")


@mcp.tool()
def get_price(fruit: str) -> str:
    """Return the per-kilogram price of a fruit from the shop price list."""
    match = re.search(rf"- {re.escape(fruit.lower())}: (\d+)", PRICE_TEXT)
    if match:
        return f"{fruit.lower()}: {match.group(1)} per kilogram"
    return f"The shop does not sell {fruit.lower()}."


@mcp.tool()
def list_stock() -> str:
    """List every fruit the shop sells with its current stock level."""
    return "\n".join(re.findall(r"- .*", PRICE_TEXT))


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
