# Everything installs inside the image. Nothing is expected on the host
# beyond Docker itself.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy

# Dependencies first, so editing the corpus or main.py does not invalidate
# the layer that installs graphrag and its (substantial) dependency tree.
COPY pyproject.toml ./
RUN uv sync

COPY . .
RUN uv sync

# Config check by default: it makes no model calls, so "docker run" without
# an API key tells you whether the image is sound rather than failing on auth.
CMD ["uv", "run", "python", "main.py", "doctor"]
