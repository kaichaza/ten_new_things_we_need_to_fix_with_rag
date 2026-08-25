# Fully standalone image: uv + Python 3.12 + all dependencies installed here.
# The llm-guard PromptInjection scanner runs a local HuggingFace classifier;
# we download it AT BUILD TIME so first run needs no network for the model.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml ./
RUN uv sync
# Pre-fetch the classifier weights into the image (several hundred MB).
RUN uv run python -c "from llm_guard.input_scanners import PromptInjection; PromptInjection()"
COPY . .
RUN uv sync
CMD ["uv", "run", "python", "main.py", "walk"]
