# observability_langfuse_otel

**Theme.** When an AI system misbehaves, the question is never "what was the
answer" but "what happened": which context was retrieved, which model ran,
with how many tokens, under which request. Systems that log those steps
across disjoint tools with no shared key cannot answer it. The fix is one
trace ID end to end, with model calls described in the OpenTelemetry GenAI
semantic conventions - the shared vocabulary any backend can read - and
Langfuse as the LLM-native UI riding the same plumbing (its Python SDK v3
is built on OpenTelemetry).

**Stack.** opentelemetry-sdk (TracerProvider, spans with `gen_ai.*`
attributes, console exporter for the walkthrough, in-memory exporter for the
tests - injected, not monkeypatched), Langfuse v4 client (optional: activates
only when keys are present), OpenAI `gpt-4.1-mini`, uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q

    docker build -t observability .
    docker run --rm --env-file .env observability

## What to look for

The walkthrough prints every span of one real request - root, retrieval
(with hit counts), model call (with `gen_ai.request.model` and token usage) -
then prints the single trace ID they all share: that ID is what a user
quotes in a support ticket and what an auditor asks for by date. With
Langfuse keys in `.env`, the same pipeline ships its trace to Langfuse
Cloud; without them, that half skips cleanly and says so. The tests assert
the two properties that make this observability rather than logging: one
trace ID across all spans, and standard GenAI attributes on the model span.
