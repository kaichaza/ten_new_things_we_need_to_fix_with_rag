"""Who did what: one trace ID across retrieve -> generate, in open conventions.

When something goes wrong in an AI system, the question is not 'what was the
answer' but 'what happened': which context was retrieved, which model was
called, with how many tokens, under which request. Systems logging those
steps across disjoint systems with no shared key cannot answer it. This
exercise instruments a real RAG pipeline with OpenTelemetry so that:

  - ONE trace ID covers the whole request (quote it in a support ticket),
  - the model call is a span carrying GenAI semantic-convention attributes
    (gen_ai.request.model, gen_ai.usage.*), readable by any OTel backend,
  - Langfuse - whose Python SDK v3 is itself built on OpenTelemetry - ships
    the same trace to its UI when keys are present, and is skipped cleanly
    when they are not.

The exporter is injected (configure_tracing), so the tests observe the real
pipeline through a real in-memory exporter rather than monkeypatching.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

load_dotenv()

HERE = Path(__file__).parent
PARAGRAPHS = [p.strip() for p in (HERE / "corpus" / "shop_facts.md").read_text().split("\n\n") if p.strip()]

CHAT_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

_provider: TracerProvider | None = None


def configure_tracing(exporter=None) -> TracerProvider:
    """Install the tracer provider ONCE, with an injectable exporter.

    The walkthrough passes nothing and gets the console exporter; the tests
    pass an InMemorySpanExporter and read the very spans the real pipeline
    emitted. Dependency injection here is what keeps the tests honest.
    """
    global _provider
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create({"service.name": "fruit-shop-rag"})
        )
        trace.set_tracer_provider(_provider)
    _provider.add_span_processor(
        SimpleSpanProcessor(exporter or ConsoleSpanExporter())
    )
    return _provider


def tracer():
    return trace.get_tracer("fruit_shop.pipeline")


# ---------------------------------------------------------- the real pipeline --
def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def retrieve(question: str, k: int = 2) -> list[str]:
    """Retrieval as a child span: which context entered the answer is
    precisely what incident reviews need and rarely have."""
    with tracer().start_as_current_span("retrieve") as span:
        vecs = embed(PARAGRAPHS)
        qvec = embed([question])[0]
        order = np.argsort(-(vecs @ qvec))[:k]
        hits = [PARAGRAPHS[i] for i in order]
        span.set_attribute("retrieval.top_k", k)
        span.set_attribute("retrieval.hit_count", len(hits))
        return hits


def generate(question: str, contexts: list[str]) -> str:
    """The model call as a span carrying OTel GenAI semantic conventions -
    the shared vocabulary that lets ANY backend read model, token usage and
    provider without bespoke parsing."""
    with tracer().start_as_current_span("gen_ai.chat") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "openai")
        span.set_attribute("gen_ai.request.model", CHAT_MODEL)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Answer only from the context, briefly."},
                {"role": "user",
                 "content": "Context:\n" + "\n---\n".join(contexts) + f"\n\nQuestion: {question}"},
            ],
        )
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
        return resp.choices[0].message.content.strip()


def ask(question: str) -> tuple[str, str]:
    """The whole request under ONE root span; returns (answer, trace_id)."""
    with tracer().start_as_current_span("rag.request") as root:
        root.set_attribute("request.question", question)
        contexts = retrieve(question)
        answer = generate(question, contexts)
        trace_id = f"{root.get_span_context().trace_id:032x}"
        return answer, trace_id


# ------------------------------------------------------------------ langfuse --
def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def ship_to_langfuse(question: str) -> str | None:
    """Run the same pipeline under a Langfuse span. Langfuse's SDK v3 is
    built ON OpenTelemetry, so this nests naturally with our spans and the
    trace appears in the Langfuse UI with the model call as a child."""
    if not langfuse_enabled():
        return None
    from langfuse import get_client

    lf = get_client()
    with lf.start_as_current_span(name="fruit-shop-rag-request") as lf_span:
        contexts = retrieve(question)
        answer = generate(question, contexts)
        lf_span.update(input=question, output=answer)
    lf.flush()
    return answer


# -------------------------------------------------------------------- walk --
def walk() -> None:
    configure_tracing()  # console exporter: every span prints below

    q = "What do pears cost, and who usually buys them?"
    print(f"=== Asking with full tracing: {q} ===\n")
    answer, trace_id = ask(q)
    print(f"\nANSWER:   {answer}")
    print(f"TRACE ID: {trace_id}")
    print("\nEvery span above shares that trace ID: the root request, the "
          "retrieval (with hit counts), and the model call (with "
          "gen_ai.request.model and token usage). That ID is what a user "
          "quotes in a ticket and an auditor asks for by date.")

    if langfuse_enabled():
        print("\n=== Langfuse keys found: shipping the same pipeline's trace ===")
        ship_to_langfuse(q)
        print("Sent. Open your Langfuse project to see the trace.")
    else:
        print("\n[langfuse] keys not set - skipped. Add LANGFUSE_PUBLIC_KEY and "
              "LANGFUSE_SECRET_KEY to .env to ship this trace to Langfuse Cloud; "
              "its SDK v3 rides the same OpenTelemetry plumbing you just saw.")


if __name__ == "__main__":
    walk()
