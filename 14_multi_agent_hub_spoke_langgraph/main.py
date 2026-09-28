"""Hub and spoke agents on LangGraph: budgets, a gate, one trace, a judged golden set.

Exercises 07, 08, 09 and 10 each hold one piece of what a production
multi-agent loop needs - cost accounting, a bounded agent with a gate,
tool trust, one trace ID - and none of them assembles the pieces. This
exercise does the assembly, on the shop's simplest questions, with a hub
and three spokes:

  hub      decides which specialist a question belongs to (structured
           output on the cheap model), checks the budget BEFORE every
           delegation, retries the next specialist when one declines, and
           routes consequential actions through the gate
  spokes   apple, banana and pear specialists, each its own compiled
           subgraph with its own state schema and one tool (check_stock);
           each receives a brief, never the parent conversation, and they
           never talk to each other
  gate     interrupt() before an order is written; resumed with
           Command(resume=True) - the newer idiom next to 08's compile-time
           interrupt_before
  budgets  three independent stops: a hop counter, a dollar ceiling from
           usage_metadata, and recursion_limit
  trace    one OpenTelemetry trace ID over hub turns and spoke runs, plus
           Langfuse SDK v4 (CallbackHandler on the invocation,
           propagate_attributes() entered before it) when keys are present
  eval     a golden set (question, expected answer, expected spoke) judged
           OFFLINE: trajectory by exact match on the first spoke chosen,
           answer by an LLM judge - the judge never runs inside the graph

Run me:  uv run python main.py walk      the guided walkthrough (real calls)
         uv run python main.py doctor    config check, no model calls, free
         uv run python main.py eval      the golden set only
         uv run python main.py clean     remove data/orders.log
"""
from __future__ import annotations

import inspect
import json
import operator
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, Field

load_dotenv()

HERE = Path(__file__).parent
STOCK_FILE = HERE / "corpus" / "stock.md"
GOLDEN_FILE = HERE / "golden" / "questions.json"
ORDERS_LOG = HERE / "data" / "orders.log"

CHEAP = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")

# USD per million tokens, keyed by model name so cost_of() stays generic.
# Read from .env so the numbers can be corrected without touching code.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    CHEAP: (
        float(os.getenv("PRICE_IN_CHEAP", "0.40")),
        float(os.getenv("PRICE_OUT_CHEAP", "1.60")),
    ),
}

# ---- the topology, in constants ----------------------------------------------
# One place to change the specialists. The Literal below must match SPOKES;
# LangGraph reads the Literal on each Command return type to draw the edges,
# so the assertion after it keeps the two in step.
SpokeName = Literal["apple", "banana", "pear"]
SPOKES: tuple[str, ...] = ("apple", "banana", "pear")
PLURALS = {"apple": "apples", "banana": "bananas", "pear": "pears"}
assert set(SPOKES) == set(PLURALS), "SPOKES and PLURALS must name the same fruits"

# ---- the budgets, in constants -----------------------------------------------
# Three independent stops. Any one of them fails to trigger in SOME
# situation, which is why there are three.
MAX_HOPS = 3                  # hub -> spoke delegations per question
COST_CEILING_USD = 0.05       # dollars per question, from usage_metadata
STEP_BUDGET = 30              # recursion_limit: graph steps, the last resort

# A spoke that is asked about the wrong fruit replies with exactly this token.
# Same shape as 07's ESCALATE signal: a self-declared, cheaply parsed decision.
CANNOT_ANSWER = "CANNOT_ANSWER"

# ---- prompts -------------------------------------------------------------------
ROUTER_PROMPT = """You are the front desk of a small fruit shop with three specialists:
apple, banana and pear. Read the customer's question and pick the specialist
whose fruit it is about. If it is about none of those three fruits, pick
"unknown". If the customer also asks to order some fruit, set order_kg to the
number of kilograms they want; otherwise leave order_kg unset."""

SPOKE_PROMPT = """You are the {fruit} specialist at a small fruit shop. You answer only
questions about {plural}: their colour, shape, taste, whether they are in
stock and what they cost. Use the check_stock tool for stock levels and
prices; never guess those. Answer in one or two short sentences a young child
would understand.
If the question is not about {plural}, reply with exactly the single word
{token} and nothing else."""

JUDGE_PROMPT = """You are grading a fruit shop assistant's answer against a reference answer.

Question: {question}
Reference answer: {expected}
Assistant answer: {answer}

Pass if the assistant's answer conveys the same fact as the reference, even
with different wording or extra correct detail. Fail if it contradicts the
reference, omits the fact, or says it cannot answer."""


# ---- OpenTelemetry (same shape as exercise 10) ----------------------------------
_provider: TracerProvider | None = None
_default_exporter: InMemorySpanExporter | None = None


def configure_tracing(exporter=None) -> TracerProvider:
    """Install the tracer provider ONCE, with an injectable exporter.

    The walkthrough passes nothing and gets an in-memory exporter it reads
    back to print a compact span summary; the tests pass their own
    InMemorySpanExporter and assert on the real spans the graph emitted.
    """
    global _provider, _default_exporter
    if _provider is None:
        _provider = TracerProvider(
            resource=Resource.create({"service.name": "fruit-shop-hub-spoke"})
        )
        trace.set_tracer_provider(_provider)
    if exporter is None:
        if _default_exporter is None:
            _default_exporter = InMemorySpanExporter()
        exporter = _default_exporter
    _provider.add_span_processor(SimpleSpanProcessor(exporter))
    return _provider


def tracer():
    return trace.get_tracer("fruit_shop.hub_spoke")


# ---- Langfuse SDK v4 ---------------------------------------------------------------
# The whole Langfuse footprint is three touchpoints, deliberately:
#   1. the lazy shared client, with `environment` on the CONSTRUCTOR (v4)
#   2. the bare LangChain CallbackHandler, passed on the invocation config
#   3. propagate_attributes(), entered BEFORE the invocation so every child
#      observation the handler creates inherits user, session and metadata
# The attribute mechanism is the one thing that has moved across majors
# (v2: handler constructor; v3: langfuse_-prefixed invocation metadata;
# v4: propagate_attributes). Everything here is verified against the
# INSTALLED SDK at run time rather than trusting a docs page: if a name is
# missing the feature degrades and says so, and the turn still runs.
_langfuse_client = None


def langfuse_enabled() -> bool:
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public or not secret:
        return False
    if "replace-me" in public or "replace-me" in secret:
        return False
    return True


def langfuse_client():
    """The shared v4 client, constructed once, or None when not configured."""
    global _langfuse_client
    if not langfuse_enabled():
        return None
    if _langfuse_client is None:
        from langfuse import Langfuse

        kwargs = {
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        }
        # v4 takes the process-level environment on the constructor. Guard it
        # by signature so an SDK without the parameter does not break startup.
        if "environment" in inspect.signature(Langfuse.__init__).parameters:
            kwargs["environment"] = os.getenv("APP_ENV", "practice")
        _langfuse_client = Langfuse(**kwargs)
    return _langfuse_client


def langfuse_handler():
    """The bare LangChain CallbackHandler, or None.

    Constructed after the shared client exists so the handler binds to it.
    v4's only handler change was removing a parameter this code never
    passed, so the bare construction stands.
    """
    if langfuse_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as error:
        print(f"[langfuse] CallbackHandler not importable against the installed "
              f"langchain-core ({error}) - Langfuse tracing skipped, OTel still on")
        return None
    return CallbackHandler()


def trace_scope(thread_id: str, surface: str):
    """propagate_attributes() as a context manager, or a null context.

    v4 constrains propagated metadata to string values of at most 200
    characters; every value here is a short identifier.
    """
    if langfuse_client() is None:
        return nullcontext()
    try:
        from langfuse import propagate_attributes
    except ImportError as error:
        print(f"[langfuse] propagate_attributes not available ({error}) - "
              f"traces will lack user/session attributes")
        return nullcontext()
    return propagate_attributes(
        user_id="matt",
        session_id=thread_id,
        metadata={"exercise": "14_multi_agent_hub_spoke", "surface": surface},
    )


def flush_langfuse() -> None:
    """Flush is unchanged in v4. Called once at the end of a walkthrough."""
    client = langfuse_client()
    if client is not None:
        client.flush()


# ---- cost accounting ------------------------------------------------------------------
def cost_of(usage: dict, model: str) -> float:
    """Dollars for one usage dict {input_tokens, output_tokens} on one model.

    Unknown models cost 0.0 rather than raising: a price table gap must
    never stop a customer turn, but it is reported by doctor().
    """
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return 0.0
    price_in, price_out = prices
    return (usage.get("input_tokens", 0) * price_in
            + usage.get("output_tokens", 0) * price_out) / 1_000_000


def sum_usage(messages: list[BaseMessage]) -> dict:
    """Total usage over the AIMessages in a message list.

    langchain-openai fills AIMessage.usage_metadata from the API response;
    that is the only source of truth for tokens here - nothing estimated.
    """
    total = {"input_tokens": 0, "output_tokens": 0}
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            total["input_tokens"] += message.usage_metadata.get("input_tokens", 0)
            total["output_tokens"] += message.usage_metadata.get("output_tokens", 0)
    return total


# ---- the one tool --------------------------------------------------------------------
def plural(fruit: str) -> str:
    return PLURALS.get(fruit, fruit)


@tool
def check_stock(fruit: str) -> str:
    """Look up the stock level and price per kilogram of one fruit in the shop's stock list."""
    name = fruit.strip().lower()
    if not name.endswith("s"):
        name = name + "s"
    text = STOCK_FILE.read_text()
    match = re.search(rf"^- {re.escape(name)}: (.+)$", text, re.MULTILINE)
    if match:
        return f"{name}: {match.group(1)}"
    return f"The shop does not sell {name}."


# ---- a spoke: its own subgraph, its own state -------------------------------------------
class SpokeState(TypedDict):
    """A spoke's private state. It shares nothing with the hub's schema on
    purpose: the hub passes a brief in and reads a structured result out."""
    messages: Annotated[list[BaseMessage], add_messages]


def build_spoke(fruit: str):
    """One specialist as a compiled subgraph: agent node, tool node, loop.

    The same shape as 08's single agent, factored so the three spokes are
    the same code with different prompts. Each spoke sees only the brief
    the hub gives it - the parent conversation never enters this graph.
    """
    llm = ChatOpenAI(model=CHEAP, temperature=0).bind_tools([check_stock])
    system = SystemMessage(SPOKE_PROMPT.format(fruit=fruit, plural=plural(fruit), token=CANNOT_ANSWER))

    def agent(state: SpokeState):
        return {"messages": [llm.invoke([system] + state["messages"])]}

    def route(state: SpokeState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(SpokeState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([check_stock]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---- the hub: state, router, budgets, gate ----------------------------------------------
class SpokeResult(TypedDict):
    """What a spoke reports back. Small on purpose: the hub keeps summaries,
    never spoke transcripts, so its own context does not grow with every hop."""
    spoke: str
    answered: bool
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class HubState(TypedDict):
    """The hub's typed state.

    Reducers are the point: `tried`, `results`, `hops` and `cost_usd` are
    APPENDED or ADDED by every node that touches them, so a node returns a
    delta and never has to know what the others wrote. Everything else is
    last-write-wins.
    """
    question: str
    forced_route: Optional[str]          # test hook for the sabotage case; not a prompt
    route: Optional[str]
    tried: Annotated[list[str], operator.add]
    results: Annotated[list[SpokeResult], operator.add]
    hops: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    pending_order: Optional[dict]
    order_done: bool
    order_approved: bool
    degraded: bool
    degrade_reason: str
    final_answer: str


class RouteDecision(BaseModel):
    """The router's structured decision. Structured output, not free text,
    so the route is a value the code checks rather than a phrase it parses."""
    fruit: Literal["apple", "banana", "pear", "unknown"]
    order_kg: Optional[int] = Field(default=None, description="Kilograms to order, if the customer asked to order.")


def decide_route(question: str) -> tuple[RouteDecision, dict]:
    """One cheap structured-output call. include_raw=True keeps the raw
    AIMessage so the router's own tokens are counted against the budget."""
    llm = ChatOpenAI(model=CHEAP, temperature=0).with_structured_output(RouteDecision, include_raw=True)
    response = llm.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(question)])
    decision = response["parsed"]
    usage = sum_usage([response["raw"]])
    return decision, usage


def budget_exhausted(state: HubState) -> tuple[bool, str]:
    """The policy, as a pure function so it is testable without a model.

    Checked by the hub BEFORE every delegation. recursion_limit is the
    third stop and lives on the invocation config, not here: it is the
    backstop for the case where this function has a bug.
    """
    if state["hops"] >= MAX_HOPS:
        return True, f"hop budget of {MAX_HOPS} reached"
    if state["cost_usd"] >= COST_CEILING_USD:
        return True, f"cost ceiling of ${COST_CEILING_USD:.2f} reached"
    return False, ""


def hub(state: HubState) -> Command[Literal["apple", "banana", "pear", "order_gate", "finalize"]]:
    """The orchestrator. Every decision is a Command: route plus state delta.

    Order of checks matters:
      1. a spoke answered -> gate if an order is pending, else finalize
      2. budget exhausted -> finalize, degraded
      3. first hop -> ask the router (one model call)
         later hops -> next untried spoke, deterministically, no model call
    """
    with tracer().start_as_current_span("hub.turn") as span:
        results = state["results"]
        last_answered = bool(results) and results[-1]["answered"]
        span.set_attribute("hub.hops_so_far", state["hops"])
        span.set_attribute("hub.cost_so_far_usd", state["cost_usd"])

        if last_answered:
            if state.get("pending_order") and not state["order_done"]:
                span.set_attribute("hub.decision", "order_gate")
                return Command(goto="order_gate")
            span.set_attribute("hub.decision", "finalize")
            return Command(goto="finalize")

        exhausted, reason = budget_exhausted(state)
        if exhausted:
            span.set_attribute("hub.decision", "finalize_degraded")
            span.set_attribute("hub.degrade_reason", reason)
            return Command(goto="finalize", update={"degraded": True, "degrade_reason": reason})

        update: dict = {"hops": 1}
        if not state["tried"]:
            decision, usage = decide_route(state["question"])
            update["cost_usd"] = cost_of(usage, CHEAP)
            if state["forced_route"]:
                fruit = state["forced_route"]
            elif decision.fruit == "unknown":
                fruit = SPOKES[0]
            else:
                fruit = decision.fruit
            if decision.order_kg and decision.fruit != "unknown":
                update["pending_order"] = {"fruit": decision.fruit, "kilograms": decision.order_kg}
            span.set_attribute("hub.router_choice", decision.fruit)
        else:
            remaining = [name for name in SPOKES if name not in state["tried"]]
            if not remaining:
                return Command(goto="finalize",
                               update={"degraded": True, "degrade_reason": "every specialist tried"})
            fruit = remaining[0]

        update["route"] = fruit
        update["tried"] = [fruit]
        span.set_attribute("hub.decision", fruit)
        return Command(goto=fruit, update=update)


def make_spoke_node(fruit: str, subgraph):
    """Wrap a compiled spoke as a hub node.

    The wrapper is where the state boundary lives: brief in (the question
    only), SpokeResult out. The subgraph never sees HubState.
    """
    def node(state: HubState) -> Command[Literal["hub"]]:
        with tracer().start_as_current_span(f"spoke.{fruit}") as span:
            span.set_attribute("agent.name", fruit)
            span.set_attribute("gen_ai.request.model", CHEAP)
            out = subgraph.invoke({"messages": [HumanMessage(state["question"])]})
            usage = sum_usage(out["messages"])
            final = out["messages"][-1].content
            if isinstance(final, list):
                # Some providers return content blocks; flatten text parts.
                final = " ".join(part.get("text", "") for part in final if isinstance(part, dict))
            final = final.strip()
            answered = not final.startswith(CANNOT_ANSWER)
            cost = cost_of(usage, CHEAP)
            result = SpokeResult(
                spoke=fruit, answered=answered, answer=final,
                input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                cost_usd=cost,
            )
            span.set_attribute("agent.answered", answered)
            span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
            span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
            span.set_attribute("agent.cost_usd", cost)
            return Command(goto="hub", update={"results": [result], "cost_usd": cost})
    return node


def place_order(fruit: str, kilograms: int) -> None:
    """The consequential action. Only ever called after the gate resumes."""
    ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_LOG, "a") as handle:
        handle.write(f"ORDER customer=Matt fruit={plural(fruit)} kg={kilograms}\n")


def order_gate(state: HubState) -> Command[Literal["hub"]]:
    """interrupt() before the write. The graph pauses here; the checkpointer
    keeps the state; a later invoke with Command(resume=<value>) re-runs
    this node from the top and interrupt() returns that value.

    The payload is what an approval screen would show. Nothing below the
    interrupt line executes until a human (or the walkthrough's
    auto-approver) resumes.
    """
    order = state["pending_order"]
    approved = interrupt({
        "action": "place_order",
        "customer": "Matt",
        "fruit": order["fruit"],
        "kilograms": order["kilograms"],
    })
    if approved is True:
        place_order(order["fruit"], order["kilograms"])
    return Command(goto="hub", update={"order_done": True, "order_approved": approved is True})


def finalize(state: HubState):
    """Compose the customer-facing answer, including the degraded case."""
    answered = [result for result in state["results"] if result["answered"]]
    if answered:
        text = answered[-1]["answer"]
    else:
        reason = state.get("degrade_reason") or "no specialist answered"
        text = ("I could not answer that. None of our specialists (apple, banana, pear) "
                f"covers it, and I stopped looking when the {reason}.")
    if state["order_done"]:
        order = state["pending_order"]
        if state["order_approved"]:
            text += f" Order placed: {order['kilograms']} kg of {plural(order['fruit'])} for Matt."
        else:
            text += " The order was not approved, so nothing was placed."
    return {"final_answer": text}


def build_graph():
    """Hub, three spokes, gate, finalize. Edges come from the Literal on
    each Command return type; only START and finalize->END are explicit."""
    graph = StateGraph(HubState)
    graph.add_node("hub", hub)
    for fruit in SPOKES:
        graph.add_node(fruit, make_spoke_node(fruit, build_spoke(fruit)))
    graph.add_node("order_gate", order_gate)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "hub")
    graph.add_edge("finalize", END)
    # MemorySaver: every step checkpointed, which is what makes interrupt()
    # resumable. In production this is a Postgres or Redis checkpointer.
    return graph.compile(checkpointer=MemorySaver())


# ---- running one question --------------------------------------------------------------------
def initial_state(question: str, forced_route: Optional[str]) -> HubState:
    return HubState(
        question=question, forced_route=forced_route, route=None,
        tried=[], results=[], hops=0, cost_usd=0.0,
        pending_order=None, order_done=False, order_approved=False,
        degraded=False, degrade_reason="", final_answer="",
    )


def run_question(graph, question: str, thread_id: str,
                 forced_route: Optional[str] = None, surface: str = "walk") -> dict:
    """Run until completion or until the gate pauses.

    Returns a dict the caller (an approval screen, the walkthrough's
    auto-approver, or the evaluator) decides what to do with:
      paused, pending, final_answer, trace_id, config, state
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": STEP_BUDGET}
    handler = langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]

    with trace_scope(thread_id, surface):
        with tracer().start_as_current_span("hub.request") as root:
            root.set_attribute("request.question", question)
            root.set_attribute("request.thread_id", thread_id)
            result = graph.invoke(initial_state(question, forced_route), config)
            trace_id = f"{root.get_span_context().trace_id:032x}"

    pending = None
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        pending = interrupts[0].value

    return {
        "paused": pending is not None,
        "pending": pending,
        "final_answer": result.get("final_answer", ""),
        "trace_id": trace_id,
        "config": config,
        "state": result,
    }


def resume(graph, config: dict, approved: bool = True, surface: str = "walk") -> dict:
    """Approve (or decline) and continue the paused run from its checkpoint."""
    thread_id = config["configurable"]["thread_id"]
    with trace_scope(thread_id, surface):
        with tracer().start_as_current_span("hub.resume"):
            return graph.invoke(Command(resume=approved), config)


# ---- evaluation: golden set, trajectory check, LLM judge ------------------------------------
class Verdict(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text())


def judge(question: str, expected: str, answer: str) -> Verdict:
    """LLM-as-judge for the final answer. Semantic, not lexical: 'Yellow'
    and 'Bananas are yellow when ripe' must both pass. Lives here, outside
    the graph, and is only ever called by evaluate() and the tests."""
    llm = ChatOpenAI(model=CHEAP, temperature=0).with_structured_output(Verdict)
    return llm.invoke(JUDGE_PROMPT.format(question=question, expected=expected, answer=answer))


def evaluate(graph, cases: Optional[list[dict]] = None) -> list[dict]:
    """Run every golden case through the REAL graph and grade two things
    separately: the trajectory (first spoke chosen == expected spoke) and
    the answer (judge verdict). A case can pass one and fail the other,
    which is the whole reason they are separate columns."""
    if cases is None:
        cases = load_golden()
    rows = []
    for index, case in enumerate(cases):
        outcome = run_question(graph, case["question"], thread_id=f"eval-{index}",
                               forced_route=case.get("force_route"), surface="eval")
        state = outcome["state"]
        first_spoke = None
        if state["results"]:
            first_spoke = state["results"][0]["spoke"]
        verdict = judge(case["question"], case["expected_answer"], outcome["final_answer"])
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_spoke": case["expected_spoke"],
            "first_spoke": first_spoke,
            "trajectory_ok": first_spoke == case["expected_spoke"],
            "answer_ok": verdict.verdict == "pass",
            "judge_reason": verdict.reason,
            "hops": state["hops"],
            "cost_usd": state["cost_usd"],
            "sabotage": bool(case.get("sabotage")),
            "answer": outcome["final_answer"],
            "state": state,
        })
    return rows


def print_eval_table(rows: list[dict]) -> None:
    """One compact row per case, then the two answers under it.

    The row grades the trajectory (expected spoke versus the first spoke the
    hub chose) and the answer (judge verdict) separately, because a case can
    pass one and fail the other. The answers are printed underneath rather
    than in columns so a reader can compare wording without the row wrapping.
    """
    header = (f"\n{'case':24s} {'exp_spoke':9s} {'first':9s} {'traj':5s} "
              f"{'answer':7s} {'hops':4s} {'cost_usd':9s}")
    print(header)
    for row in rows:
        if row["trajectory_ok"]:
            traj = "ok"
        else:
            traj = "WRONG"
        if row["answer_ok"]:
            answer = "pass"
        else:
            answer = "FAIL"
        marker = ""
        if row["sabotage"]:
            marker = "  <- sabotage"
        print(f"{row['id']:24s} {row['expected_spoke']:9s} {str(row['first_spoke']):9s} "
              f"{traj:5s} {answer:7s} {row['hops']:<4d} {row['cost_usd']:<9.5f}{marker}")
        print(f"    expected : {row['expected_answer']}")
        print(f"    actual   : {row['answer']}")
        # The judge's reasoning only earns screen space when it disagreed.
        if not row["answer_ok"]:
            print(f"    judge    : {row['judge_reason']}")


def cost_by_spoke(states: list[dict]) -> dict[str, dict]:
    """Aggregate spoke spend across runs: the number that tells you which
    specialist is eating the budget."""
    totals: dict[str, dict] = {name: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                               for name in SPOKES}
    for state in states:
        for result in state["results"]:
            bucket = totals[result["spoke"]]
            bucket["calls"] += 1
            bucket["input_tokens"] += result["input_tokens"]
            bucket["output_tokens"] += result["output_tokens"]
            bucket["cost_usd"] += result["cost_usd"]
    return totals


def print_span_summary(exporter: InMemorySpanExporter, trace_id: str) -> None:
    """Compact view of one trace's spans, instead of the console exporter's
    full JSON dump: name, agent and cost attributes, all under one ID."""
    spans = [span for span in exporter.get_finished_spans()
             if f"{span.context.trace_id:032x}" == trace_id]
    print(f"  {len(spans)} spans under trace {trace_id}:")
    for span in sorted(spans, key=lambda span: span.start_time):
        attrs = dict(span.attributes)
        detail = ""
        if "hub.decision" in attrs:
            detail = f"decision={attrs['hub.decision']}"
        elif "agent.name" in attrs:
            detail = (f"answered={attrs.get('agent.answered')} "
                      f"tokens={attrs.get('gen_ai.usage.input_tokens')}/{attrs.get('gen_ai.usage.output_tokens')} "
                      f"cost=${attrs.get('agent.cost_usd', 0):.5f}")
        print(f"    {span.name:14s} {detail}")


# ---- doctor ------------------------------------------------------------------------------------
def doctor() -> None:
    """Check the setup without spending anything."""
    print("=== doctor: configuration check, no model calls ===\n")

    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-your"):
        print(f"  OpenAI API key        : present ({key[:6]}...{key[-4:]})")
    else:
        print("  OpenAI API key        : MISSING - copy .env.example to .env and add it")

    print(f"  cheap model           : {CHEAP}")
    prices = PRICES_PER_MILLION.get(CHEAP)
    if prices is None:
        print("  price table           : NO ENTRY for the cheap model - cost ceiling cannot bite")
    else:
        print(f"  price table           : ${prices[0]:.2f} in / ${prices[1]:.2f} out per million tokens")
    print(f"  budgets               : hops={MAX_HOPS} cost=${COST_CEILING_USD:.2f} recursion_limit={STEP_BUDGET}")

    import importlib.metadata as metadata
    for package in ("langgraph", "langchain-core", "langchain-openai", "langfuse", "opentelemetry-sdk"):
        try:
            print(f"  {package:22s}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"  {package:22s}: NOT INSTALLED")

    if langfuse_enabled():
        print(f"  Langfuse              : configured for {os.getenv('LANGFUSE_HOST')} (env {os.getenv('APP_ENV', 'practice')})")
        try:
            from langfuse.langchain import CallbackHandler  # noqa: F401
            print("  Langfuse handler      : importable")
        except ImportError as error:
            print(f"  Langfuse handler      : NOT importable ({error}) - OTel path still works")
        try:
            from langfuse import propagate_attributes  # noqa: F401
            print("  propagate_attributes  : available (v4 attribute mechanism)")
        except ImportError:
            print("  propagate_attributes  : NOT available - check the installed major")
    else:
        print("  Langfuse              : not configured (placeholder or missing keys) - OTel only")

    print(f"  golden cases          : {len(load_golden())}")
    print(f"  stock list            : {len(re.findall(r'^- ', STOCK_FILE.read_text(), re.MULTILINE))} fruits")
    print(f"  orders log            : {'exists' if ORDERS_LOG.exists() else 'absent'} ({ORDERS_LOG.relative_to(HERE)})")


# ---- walk -------------------------------------------------------------------------------------
def walk() -> None:
    configure_tracing()
    exporter = _default_exporter
    if ORDERS_LOG.exists():
        ORDERS_LOG.unlink()
    graph = build_graph()
    states: list[dict] = []

    print("=== 1. A banana question: one hop, one trace ===")
    question = "What colour is a banana?"
    outcome = run_question(graph, question, thread_id="walk-banana")
    state = outcome["state"]
    states.append(state)
    print(f"Q: {question}")
    print(f"A: {outcome['final_answer']}")
    print(f"   route={state['route']} tried={state['tried']} hops={state['hops']} cost=${state['cost_usd']:.5f}")
    print_span_summary(exporter, outcome["trace_id"])

    print("\n=== 2. A pear order: the spoke answers, then the GATE pauses the graph ===")
    question = "Are pears in stock? If so, order 2 kg of pears for Matt."
    outcome = run_question(graph, question, thread_id="walk-order")
    print(f"Q: {question}")
    if outcome["paused"]:
        print(f"[GATE] paused before a consequential action: {outcome['pending']}")
        print(f"[GATE] orders.log exists yet? {ORDERS_LOG.exists()} (it must not)")
        print("[GATE] In production this is an approval screen. Auto-approving now...")
        final = resume(graph, outcome["config"], approved=True)
        states.append(final)
        print(f"A: {final['final_answer']}")
        print(f"[log] {ORDERS_LOG.read_text().strip()}")
    else:
        states.append(outcome["state"])
        print(f"[unexpected] the run completed without pausing: {outcome['final_answer']}")

    print("\n=== 3. A question no spoke covers: the HOP budget stops it, not recursion_limit ===")
    question = "What colour is a dragon fruit?"
    outcome = run_question(graph, question, thread_id="walk-unknown")
    state = outcome["state"]
    states.append(state)
    print(f"Q: {question}")
    print(f"A: {outcome['final_answer']}")
    print(f"   tried={state['tried']} hops={state['hops']}/{MAX_HOPS} degraded={state['degraded']} "
          f"reason='{state['degrade_reason']}' cost=${state['cost_usd']:.5f}")

    print("\n=== 4. The golden set, judged offline: trajectory and answer graded separately ===")
    rows = evaluate(graph)
    print_eval_table(rows)
    for row in rows:
        states.append(row["state"])
    sabotage = [row for row in rows if row["sabotage"]]
    if sabotage and sabotage[0]["answer_ok"] and not sabotage[0]["trajectory_ok"]:
        print("\nThe sabotage row answered correctly AFTER the hub retried, so an answer-only "
              "eval would have passed it. Only the trajectory column saw the wrong first "
              "route - and the hops and cost columns show what the mistake cost.")

    print("\n=== 5. Cost per spoke across everything above ===")
    print(f"{'spoke':8s} {'calls':5s} {'in_tok':7s} {'out_tok':7s} {'cost_usd':9s}")
    for name, bucket in cost_by_spoke(states).items():
        print(f"{name:8s} {bucket['calls']:<5d} {bucket['input_tokens']:<7d} {bucket['output_tokens']:<7d} {bucket['cost_usd']:<9.5f}")
    total = sum(state["cost_usd"] for state in states)
    print(f"\ntotal spend this walkthrough: ${total:.5f} (router calls included)")

    if langfuse_enabled():
        flush_langfuse()
        print(f"\n[langfuse] runs shipped to {os.getenv('LANGFUSE_HOST')}; filter by session_id "
              f"(walk-banana, walk-order, walk-unknown, eval-*) and user_id 'matt'.")
    else:
        print("\n[langfuse] keys not set - skipped. Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
              "to .env to see these runs in your Langfuse project.")

    print("\nThe assembly is the lesson: a hub that decides and budgets, spokes that see only a "
          "brief, a gate in code before the write, one trace ID across all of it, and a judge "
          "that lives in the evaluator - never in the customer's path.")


def eval_only() -> None:
    configure_tracing()
    graph = build_graph()
    rows = evaluate(graph)
    print_eval_table(rows)
    flush_langfuse()


def clean() -> None:
    if ORDERS_LOG.exists():
        ORDERS_LOG.unlink()
        print(f"[clean] removed {ORDERS_LOG.relative_to(HERE)}")
    else:
        print("[clean] nothing to remove")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "walk":
        walk()
    elif command == "doctor":
        doctor()
    elif command == "eval":
        eval_only()
    elif command == "clean":
        clean()
    else:
        print("Usage: main.py [walk | doctor | eval | clean]")
