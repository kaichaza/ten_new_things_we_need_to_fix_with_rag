"""Real-graph tests. Pure-logic tests (budget policy, cost arithmetic, the
stock tool, the golden set's shape, the Langfuse v4 surface) run keyless;
everything that drives the graph needs the OpenAI key and skips cleanly
without it. Spans are observed through a real in-memory exporter injected
via configure_tracing, exactly as exercise 10 does.
"""
import importlib.metadata
import inspect
import os

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-your"),
    reason="OPENAI_API_KEY not set",
)

exporter = InMemorySpanExporter()
main.configure_tracing(exporter)


# ---- keyless -------------------------------------------------------------------
def test_golden_set_is_well_formed_and_carries_one_sabotage_case():
    cases = main.load_golden()
    assert len(cases) >= 5
    for case in cases:
        assert case["expected_spoke"] in main.SPOKES, case["id"]
        assert case["question"] and case["expected_answer"]
    sabotage = [case for case in cases if case.get("sabotage")]
    assert len(sabotage) == 1
    assert sabotage[0]["force_route"] != sabotage[0]["expected_spoke"]


def test_cost_arithmetic_uses_the_price_table():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    price_in, price_out = main.PRICES_PER_MILLION[main.CHEAP]
    assert main.cost_of(usage, main.CHEAP) == pytest.approx(price_in + price_out)
    assert main.cost_of(usage, "no-such-model") == 0.0


def test_budget_policy_stops_on_hops_and_on_cost():
    state = main.initial_state("q", None)
    assert main.budget_exhausted(state) == (False, "")

    state = main.initial_state("q", None)
    state["hops"] = main.MAX_HOPS
    exhausted, reason = main.budget_exhausted(state)
    assert exhausted and "hop" in reason

    state = main.initial_state("q", None)
    state["cost_usd"] = main.COST_CEILING_USD
    exhausted, reason = main.budget_exhausted(state)
    assert exhausted and "cost" in reason


def test_stock_tool_reads_the_stock_list_keyless():
    assert "3 per kilogram" in main.check_stock.invoke({"fruit": "apple"})
    assert "25 kg" in main.check_stock.invoke({"fruit": "Bananas"})
    assert "does not sell" in main.check_stock.invoke({"fruit": "grapes"})


def test_langfuse_v4_surface_matches_what_main_calls():
    """Guard the three touchpoints against the installed SDK, not the docs.
    The attribute mechanism has moved on every major; this is where the
    next move shows up first."""
    major = int(importlib.metadata.version("langfuse").split(".")[0])
    assert major == 4, f"langfuse major is {major}, this exercise targets v4"

    import langfuse
    from langfuse import Langfuse

    assert hasattr(langfuse, "propagate_attributes"), "v4 attribute mechanism missing"
    params = inspect.signature(Langfuse.__init__).parameters
    for name in ("public_key", "secret_key", "host", "environment"):
        assert name in params, f"Langfuse constructor lacks {name}"

    from langfuse.langchain import CallbackHandler  # noqa: F401


def test_unconfigured_langfuse_degrades_to_null_context():
    """No keys: no client, no handler, a null trace scope, and flush a no-op."""
    saved = {key: os.environ.pop(key, None) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    main._langfuse_client = None
    try:
        assert main.langfuse_enabled() is False
        assert main.langfuse_client() is None
        assert main.langfuse_handler() is None
        with main.trace_scope("t", "test"):
            pass
        main.flush_langfuse()
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        main._langfuse_client = None


# ---- real graph ------------------------------------------------------------------
@requires_key
def test_banana_question_routes_to_banana_spoke_in_one_hop():
    graph = main.build_graph()
    outcome = main.run_question(graph, "What colour is a banana?", thread_id="test-banana", surface="test")
    state = outcome["state"]
    assert outcome["paused"] is False
    assert state["tried"] == ["banana"]
    assert state["hops"] == 1
    assert state["results"][0]["answered"] is True
    assert "yellow" in outcome["final_answer"].lower()
    assert state["cost_usd"] > 0


@requires_key
def test_one_trace_id_covers_hub_turns_and_spoke_runs():
    exporter.clear()
    graph = main.build_graph()
    outcome = main.run_question(graph, "What colour is a banana?", thread_id="test-trace", surface="test")

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert {"hub.request", "hub.turn", "spoke.banana"} <= names

    trace_ids = {f"{span.context.trace_id:032x}" for span in spans}
    assert trace_ids == {outcome["trace_id"]}

    spoke = [span for span in spans if span.name == "spoke.banana"][0]
    attrs = dict(spoke.attributes)
    assert attrs["agent.name"] == "banana"
    assert attrs["gen_ai.usage.input_tokens"] > 0
    assert attrs["agent.cost_usd"] > 0


@requires_key
def test_unanswerable_question_stops_on_the_hop_budget_not_recursion_limit():
    graph = main.build_graph()
    # A GraphRecursionError here would mean the hop budget failed and the
    # backstop caught it; the assertion is that the POLICY stopped the loop.
    outcome = main.run_question(graph, "What colour is a dragon fruit?", thread_id="test-unknown", surface="test")
    state = outcome["state"]
    assert state["degraded"] is True
    assert "hop" in state["degrade_reason"]
    assert state["hops"] == main.MAX_HOPS
    assert len(state["tried"]) == main.MAX_HOPS
    assert all(result["answered"] is False for result in state["results"])
    assert "could not answer" in outcome["final_answer"].lower()


@requires_key
def test_order_gate_pauses_before_writing_and_resume_completes_it():
    if main.ORDERS_LOG.exists():
        main.ORDERS_LOG.unlink()
    graph = main.build_graph()
    outcome = main.run_question(graph, "Are pears in stock? If so, order 2 kg of pears for Matt.",
                                thread_id="test-order", surface="test")

    assert outcome["paused"] is True
    assert outcome["pending"]["action"] == "place_order"
    assert outcome["pending"]["fruit"] == "pear"
    assert outcome["pending"]["kilograms"] == 2
    assert not main.ORDERS_LOG.exists(), "nothing may be written before approval"

    final = main.resume(graph, outcome["config"], approved=True, surface="test")
    assert main.ORDERS_LOG.exists()
    log = main.ORDERS_LOG.read_text()
    assert "fruit=pears" in log and "kg=2" in log
    assert "order placed" in final["final_answer"].lower()


@requires_key
def test_declined_order_writes_nothing():
    if main.ORDERS_LOG.exists():
        main.ORDERS_LOG.unlink()
    graph = main.build_graph()
    outcome = main.run_question(graph, "Order 3 kg of apples for Matt.", thread_id="test-decline", surface="test")
    assert outcome["paused"] is True
    final = main.resume(graph, outcome["config"], approved=False, surface="test")
    assert not main.ORDERS_LOG.exists()
    assert "not approved" in final["final_answer"].lower()


@requires_key
def test_sabotaged_route_is_caught_by_the_trajectory_check_not_the_judge():
    graph = main.build_graph()
    case = [case for case in main.load_golden() if case.get("sabotage")][0]
    rows = main.evaluate(graph, cases=[case])
    row = rows[0]
    assert row["first_spoke"] == case["force_route"]
    assert row["trajectory_ok"] is False
    # The hub retries after the wrong spoke declines, so the answer usually
    # still passes - which is exactly why trajectory is graded separately.
    assert row["hops"] >= 2


@requires_key
def test_honest_golden_cases_pass_both_checks():
    graph = main.build_graph()
    honest = [case for case in main.load_golden() if not case.get("sabotage")]
    rows = main.evaluate(graph, cases=honest)
    for row in rows:
        assert row["trajectory_ok"], f"{row['id']}: routed to {row['first_spoke']}"
        assert row["answer_ok"], f"{row['id']}: {row['judge_reason']} / {row['answer']}"
