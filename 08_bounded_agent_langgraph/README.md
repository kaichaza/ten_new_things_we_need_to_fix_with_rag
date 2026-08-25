# bounded_agent_langgraph

**Theme.** Per-step reliability multiplies away over long chains, and an
ungated agent converts every failure into a consequence. The credible
deployment shape is bounded automation: typed tools, a hard step budget,
checkpointed state, and an interruption point before anything that spends
money, sends messages or changes records. This exercise builds that shape
with LangGraph: the price-lookup tool runs freely; the order-placing tool
sits behind a compile-time interrupt; MemorySaver checkpoints make the
paused run resumable; and `recursion_limit` caps loop length.

**Stack.** LangGraph (StateGraph, ToolNode, MemorySaver, `interrupt_before`),
langchain-openai with `gpt-4.1`, uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q

    docker build -t bounded_agent .
    docker run --rm --env-file .env bounded_agent

## What to look for

The walkthrough asks the agent to check a price and place an order. Watch the
sequence: the lookup executes immediately; the graph then **pauses with the
order still pending** - `data/orders.log` provably does not exist at that
moment - prints what a human approval screen would show, and only after
approval does the resumed run write the order. The tests assert exactly that
ordering against the real agent. The unattended walkthrough auto-approves;
the point is where the pause sits, not who clicks.
