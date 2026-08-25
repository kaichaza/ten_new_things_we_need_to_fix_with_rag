"""Bounded autonomy with LangGraph: budgets, checkpoints, and a human gate.

An agent that is 95 per cent reliable per step is roughly 36 per cent
reliable over twenty steps, and an ungated agent that can act on the world
turns every failure into a consequence. The credible 2026 deployment shape is
bounded automation: typed tools, a step budget, checkpointed state, and an
INTERRUPTION POINT before any consequential action. This exercise builds
exactly that: the agent may read prices freely, but the graph is compiled to
pause before the order-placing tool executes; the checkpointer preserves the
paused state; and the run resumes only after approval.

The walkthrough is unattended (no stdin): it auto-approves after printing
what would, in production, be a human's decision screen.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

HERE = Path(__file__).parent
ORDERS_LOG = HERE / "data" / "orders.log"
PRICE_TEXT = (HERE / "corpus" / "price_list.md").read_text()

MODEL = os.getenv("MODEL_STRONG", "gpt-4.1")

# A hard budget on agent loop length. recursion_limit counts graph steps, so
# this bounds runaway tool loops regardless of what the model decides.
STEP_BUDGET = 12


# ------------------------------------------------------------------- tools --
@tool
def lookup_price(fruit: str) -> str:
    """Look up the per-kilogram price of a fruit from the shop price list."""
    match = re.search(rf"- {re.escape(fruit.lower())}: (\d+)", PRICE_TEXT)
    if match:
        return f"{fruit.lower()} cost {match.group(1)} per kilogram."
    return f"The shop does not sell {fruit.lower()}."


@tool
def place_order(fruit: str, kilograms: int) -> str:
    """CONSEQUENTIAL: place an order on Matt's account. Gated behind approval."""
    ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_LOG, "a") as f:
        f.write(f"ORDER customer=Matt fruit={fruit.lower()} kg={kilograms}\n")
    return f"Order placed: {kilograms} kg of {fruit.lower()} for Matt."


SAFE_TOOLS = [lookup_price]
GATED_TOOLS = [place_order]
GATED_NAMES = {t.name for t in GATED_TOOLS}


# ------------------------------------------------------------------- graph --
def build_graph():
    """Two tool nodes - safe and gated - with the compile-time interrupt on
    the gated one. The routing function inspects the model's pending tool
    calls and sends consequential ones through the gate."""
    llm = ChatOpenAI(model=MODEL).bind_tools(SAFE_TOOLS + GATED_TOOLS)

    def agent(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    def route(state: MessagesState):
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return END
        if any(c["name"] in GATED_NAMES for c in calls):
            return "gated_tools"
        return "safe_tools"

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("safe_tools", ToolNode(SAFE_TOOLS))
    g.add_node("gated_tools", ToolNode(GATED_TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["safe_tools", "gated_tools", END])
    g.add_edge("safe_tools", "agent")
    g.add_edge("gated_tools", "agent")

    # The two controls that make this a BOUNDED agent:
    #   - MemorySaver: every step is checkpointed, so a paused run is resumable
    #   - interrupt_before: the graph halts before the gated tool node executes
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["gated_tools"])


def pending_gated_calls(graph, config) -> list[dict]:
    """Read the checkpointed state and report the tool calls awaiting approval."""
    state = graph.get_state(config)
    if not state.next or "gated_tools" not in state.next:
        return []
    last = state.values["messages"][-1]
    return [c for c in (getattr(last, "tool_calls", None) or []) if c["name"] in GATED_NAMES]


def run_task(graph, task: str, thread_id: str) -> dict:
    """Run until completion or until the gate pauses the graph.

    Returns {"paused": bool, "pending": [...], "config": ...} so the caller
    (a human approval surface, or the walkthrough's auto-approver) decides
    what happens next. Resuming is graph.invoke(None, config) - the
    checkpointer supplies the saved state.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": STEP_BUDGET}
    graph.invoke({"messages": [HumanMessage(task)]}, config)
    pending = pending_gated_calls(graph, config)
    return {"paused": bool(pending), "pending": pending, "config": config}


def resume(graph, config) -> str:
    """Approve: continue the paused run from its checkpoint."""
    result = graph.invoke(None, config)
    return result["messages"][-1].content


# -------------------------------------------------------------------- walk --
def walk() -> None:
    if ORDERS_LOG.exists():
        ORDERS_LOG.unlink()
    graph = build_graph()

    print("=== Task: 'Check the price of apples, then order 2 kg for Matt.' ===")
    outcome = run_task(graph, "Check the price of apples, then order 2 kg of apples for Matt.",
                       thread_id="walk-1")

    if outcome["paused"]:
        print("\n[GATE] The graph paused BEFORE executing a consequential tool.")
        for call in outcome["pending"]:
            print(f"[GATE] pending: {call['name']}({call['args']})")
        print(f"[GATE] orders.log exists yet? {ORDERS_LOG.exists()} (it must not)")
        print("[GATE] In production this is a human approval screen. Auto-approving now...")
        final = resume(graph, outcome["config"])
        print(f"\n[agent] {final}")
        print(f"[log] {ORDERS_LOG.read_text().strip()}")
    else:
        print("[unexpected] the model completed without attempting an order")

    print(f"\nStep budget for every run: recursion_limit={STEP_BUDGET}. A looping "
          "agent hits that ceiling and stops with an error instead of spending "
          "unbounded tokens - the difference between an incident and an invoice.")


if __name__ == "__main__":
    walk()
