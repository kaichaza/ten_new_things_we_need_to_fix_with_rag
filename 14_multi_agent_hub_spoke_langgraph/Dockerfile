# Fully standalone image: uv + Python 3.12 + all dependencies installed here.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_LINK_MODE=copy
# Install dependencies first so Docker layer caching survives code edits.
COPY pyproject.toml ./
RUN uv sync
COPY . .
RUN uv sync
CMD ["uv", "run", "python", "main.py", "walk"]
