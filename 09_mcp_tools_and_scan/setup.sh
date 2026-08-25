#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv python install 3.12
if [ ! -f .env ]; then
  cp .env.example .env
fi
uv sync
echo "Run: uv run python main.py walk    Test: uv run pytest -q  (no API key needed)"
echo "Optional deep scan of a real MCP config: uvx mcp-scan"
echo "Docker: docker build -t mcp_tools . && docker run --rm mcp_tools"
