"""Real-agent tests: the graph, the model, and the tools are all genuine."""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@requires_key
def test_gate_pauses_before_the_order_and_resume_completes_it():
    if main.ORDERS_LOG.exists():
        main.ORDERS_LOG.unlink()
    graph = main.build_graph()

    outcome = main.run_task(
        graph, "Check the price of apples, then order 2 kg of apples for Matt.",
        thread_id="test-gate",
    )

    # The consequential tool must be PENDING, not executed: no log entry yet.
    assert outcome["paused"] is True
    assert outcome["pending"][0]["name"] == "place_order"
    assert not main.ORDERS_LOG.exists()

    # Approval resumes from the checkpoint and the order finally lands.
    main.resume(graph, outcome["config"])
    assert main.ORDERS_LOG.exists()
    log = main.ORDERS_LOG.read_text()
    assert "fruit=apples" in log and "kg=2" in log


@requires_key
def test_safe_tools_run_without_any_gate():
    graph = main.build_graph()
    outcome = main.run_task(graph, "What do pears cost per kilogram?", thread_id="test-safe")
    assert outcome["paused"] is False
    state = graph.get_state(outcome["config"])
    final = state.values["messages"][-1].content
    assert "4" in final
