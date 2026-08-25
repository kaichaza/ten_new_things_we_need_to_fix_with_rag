#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
# Python 3.11 here, deliberately: diskannpy 0.7.0 ships no 3.12 wheel.
uv python install 3.11
if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Created .env - add your OPENAI_API_KEY (only the corpus test needs it)."
fi
uv sync
echo "Run: uv run python main.py walk    Test: uv run pytest -q"
echo "Docker: docker build -t disk_ann . && docker run --rm --env-file .env disk_ann"
