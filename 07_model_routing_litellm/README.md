# model_routing_litellm

**Theme.** Token spend is cost of goods sold. Routing everything through the
strongest model pays reasoning rates and seconds of latency for questions a
small model answers identically; routing everything through the cheap model
quietly fails the hard ones. The workable pattern is a cascade: cheap tier
first, escalation only when the task earns it, and per-request cost as a
first-class number.

**Stack.** litellm (uniform provider interface, `completion_cost` accounting),
OpenAI `gpt-4.1-mini` (cheap) and `gpt-4.1` (strong), uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q

    docker build -t model_routing .
    docker run --rm --env-file .env model_routing

## What to look for

The walkthrough answers one easy question (list the fruits) and one genuinely
multi-step one (cheapest single swap for Matt's basket - the right answer is
swapping apples to bananas for a total of 10) under three strategies, then
prints the spend table. The cascade's escalation signal here is self-declared
by the cheap model; production systems add measured signals (confidence,
task class, retrieval scores), but the shape - a policy deciding which tier a
request deserves - is exactly this.
