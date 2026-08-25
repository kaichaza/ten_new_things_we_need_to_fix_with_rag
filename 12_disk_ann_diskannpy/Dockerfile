# Fully standalone image. NOTE: Python 3.11, not 3.12 - diskannpy 0.7.0 has
# no cp312 wheel and no sdist on PyPI, so 3.11 is the newest supported
# interpreter. This is the only exercise in the series with this exception.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml ./
RUN uv sync
COPY . .
RUN uv sync
CMD ["uv", "run", "python", "main.py", "walk"]
