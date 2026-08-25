# temporal_bitemporal_graphiti

**Theme.** Two retrieved documents disagree because months passed between
them, and a similarity retriever presents both as simultaneous truth. The
countermeasure is BI-TEMPORAL modelling: every fact carries valid time (when
it was true in the world) and transaction time (when the system learned it),
so "is the shop selling grapes" has three different correct answers across
the timeline - and gets them.

**Stack.** graphiti-core (the library named in the countermeasures sheet for
this pitfall) on its OWN Neo4j 5 container, ports 7475/7688, so it never
collides with 03's graph. OpenAI gpt-4.1-mini, uv on Python 3.12.

## Run it

    ./setup_temporal_graphiti.sh   # from the project root
    uv run python main.py doctor   # config check, no model calls, free
    uv run python main.py walk     # ingests the timeline: minutes, and cents
    uv run python main.py graph    # inspect the temporal graph, free
    uv run pytest -q               # free tests; paid one is opt-in below
    uv run python main.py clean    # wipe the graph (container survives)

Browse the temporal graph at http://localhost:7475 while it exists.

## The corpus, and the two traps

Six dated documents replaying one storyline in order: grapes on sale
(January), inspections fail and the contract ends (2 March), the owner
explains (10 March), a FALSE flyer claims an allergy (1 April), a new
supplier is appointed (20 May), grapes return (June). Two traps:

* the false flyer postdates the true documents, so prefer-the-newest picks
  the lie - recency is not validity;
* the final state contradicts the middle state, so a retriever without
  valid time can only ever give one of the three correct answers.

The walkthrough asks the same question as of three dates, shows the
validity filtering that makes the answers differ, asks the evolution and
recency-trap questions, and then proves superseded facts still exist with
their invalidation timestamps - history preserved, not overwritten.

Whether Graphiti's contradiction handling resists the flyer or invalidates
the true reason in favour of the fresher false one is an empirical result;
the walkthrough prints the evidence either way, and either outcome belongs
in the whitepaper: temporal machinery inherits the provenance problem.

## Cost

`doctor` and `graph` are free. `walk` ingests six documents through
Graphiti's extraction (several model calls per document) and then answers
five questions: a few minutes, cents. Re-runs skip ingestion.

The expensive test is opt-in:

    TEMPORAL_RUN_INDEX=1 uv run pytest -q
