"""The tests observe the REAL pipeline through a real in-memory exporter
(injected via configure_tracing) - no monkeypatching."""
import os

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

exporter = InMemorySpanExporter()
main.configure_tracing(exporter)


@requires_key
def test_one_trace_id_covers_retrieval_and_generation():
    exporter.clear()
    answer, trace_id = main.ask("What do bananas cost?")
    assert "2" in answer

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"rag.request", "retrieve", "gen_ai.chat"} <= names

    # The audit property: every span belongs to the SAME trace.
    trace_ids = {f"{s.context.trace_id:032x}" for s in spans}
    assert trace_ids == {trace_id}


@requires_key
def test_model_span_carries_genai_semantic_conventions():
    exporter.clear()
    main.ask("Does the shop sell grapes?")
    chat = [s for s in exporter.get_finished_spans() if s.name == "gen_ai.chat"][0]

    attrs = dict(chat.attributes)
    assert attrs["gen_ai.request.model"] == main.CHAT_MODEL
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.usage.input_tokens"] > 0
    assert attrs["gen_ai.usage.output_tokens"] > 0


def test_langfuse_client_exposes_what_ship_to_langfuse_calls():
    """The walkthrough crashed on 2026-08-25 because pyproject pinned SDK v4
    while the code called v3's start_as_current_span. Guard the surface: the
    installed client must expose one of the two names the code handles."""
    from langfuse import Langfuse

    v4 = hasattr(Langfuse, "start_as_current_observation")
    v3 = hasattr(Langfuse, "start_as_current_span")
    assert v4 or v3, "langfuse client exposes neither span API this exercise handles"
