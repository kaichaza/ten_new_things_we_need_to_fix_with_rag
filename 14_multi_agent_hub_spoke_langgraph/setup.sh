#!/usr/bin/env bash
# Local (non-Docker) setup: uv-managed Python 3.12, dependencies, .env scaffold.
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
uv python install 3.12
if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Created .env - add your OPENAI_API_KEY (Langfuse keys optional)."
fi
uv sync
echo "Run: uv run python main.py walk    Test: uv run pytest -q"
echo "Docker: docker build -t multi_agent . && docker run --rm --env-file .env multi_agent"
