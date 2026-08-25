#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv python install 3.12

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Created .env - add your key before running anything that costs money."
fi

uv sync

echo
echo "Check the setup (free, no model calls):"
echo "  uv run python main.py doctor"
echo
echo "Full run (indexes for real - a few minutes, a few cents):"
echo "  uv run python main.py walk"
echo
echo "Inspect an existing index (free):"
echo "  uv run python main.py graph"
echo
echo "Containerised:"
echo "  docker build -t graphrag_fruit ."
echo "  docker run --rm graphrag_fruit                          # doctor, free"
echo "  docker run --rm --env-file .env graphrag_fruit uv run python main.py walk"
