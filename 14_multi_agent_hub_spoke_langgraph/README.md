# multi_agent_hub_spoke_langgraph

**Theme.** Exercises 07, 08, 09 and 10 each hold one piece of a production
multi-agent loop (cost accounting, a bounded agent with a gate, tool trust,
one trace ID) and none of them assembles the pieces. This exercise does the
assembly on the shop's simplest questions - colours, shapes, what is in
stock - with a **hub and three spokes**:

* the **hub** decides which specialist a question belongs to (structured
  output on the cheap model), checks the budget before every delegation,
  retries the next specialist when one declines, and routes consequential
  actions through the gate;
* the **spokes** (apple, banana, pear) are each their own compiled subgraph
  with their own state schema and one tool, `check_stock`. They receive a
  brief, never the parent conversation, and never talk to each other;
* the **gate** is `interrupt()` before an order is written, resumed with
  `Command(resume=True)` - the newer idiom, alongside 08's compile-time
  `interrupt_before`;
* three **budgets** stop the loop independently: a hop counter, a dollar
  ceiling computed from `usage_metadata`, and `recursion_limit`;
* **one trace** covers hub turns and spoke runs (OpenTelemetry, as in 10),
  and Langfuse SDK v4 ships the same run when keys are present
  (`CallbackHandler` on the invocation, `propagate_attributes()` entered
  before it);
* a **golden set** is judged offline: trajectory by exact match on the
  first spoke chosen, answer correctness by an LLM judge. The judge never
  runs inside the customer-facing graph.

**Stack.** LangGraph 1.x (`StateGraph`, `Command`, `interrupt`, subgraphs,
reducers), langchain-openai `gpt-4.1-mini` everywhere, OpenTelemetry SDK,
Langfuse SDK v4 (optional), uv on Python 3.12.

**Scenario note.** The repository's shop sells apples, bananas and pears.
The design discussion behind this exercise used oranges as the third
specialist; pears are used here so the stock facts stay consistent with
every other exercise. `SPOKES` in `main.py` is the one place to change it.

## Run it

    ./setup.sh
    uv run python main.py doctor   # config check, no model calls, free
    uv run python main.py walk     # the guided walkthrough (real calls, cents)
    uv run python main.py eval     # the golden set only
    uv run pytest -q               # pure-logic tests run keyless; the rest skip

    docker build -t multi_agent .
    docker run --rm --env-file .env multi_agent

## What to look for

1. A banana question: the hub picks the banana spoke in one hop, the spoke
   answers, and the printed trace ID is shared by the hub turn and the
   spoke run.
2. A pear order: the spoke answers the stock question, then the graph
   **pauses** at the gate with the order pending - `data/orders.log` does
   not exist at that moment - and only the resume writes it.
3. A dragon fruit question: every spoke declines, the hub retries each one,
   and the run stops on the **hop budget** with a degraded answer rather
   than on `recursion_limit`. The tests assert which budget stopped it.
4. The golden set: each row shows the first spoke chosen versus the
   expected one, the judge's verdict, hops and cost. One row is a
   deliberate sabotage (the route is forced wrong): the answer can still
   pass after the retry, and only the trajectory column catches it. That is
   why trajectory and answer are graded separately.
5. Cost per spoke and per question, as dollars.

Langfuse: with real keys in `.env`, every run is also visible in your
Langfuse project, filterable by the propagated `session_id` (the thread)
and `user_id`. Without keys, the walkthrough says so and carries on.
