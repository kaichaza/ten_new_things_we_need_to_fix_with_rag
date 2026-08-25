#!/usr/bin/env bash
#
# setup_temporal_graphiti.sh
#
# One idempotent script that builds the whole 13_temporal_bitemporal_graphiti
# exercise: its OWN Neo4j 5 docker container on shifted ports (7475/7688,
# separate from 03's container), uv, Python 3.12, dependencies, source files,
# the dated corpus, and tests.
#
# Run it from the directory that should CONTAIN the project folder:
#
#     cd /home/kai/gen_ai_topics_practice
#     ./setup_temporal_graphiti.sh
#
# Safe to run repeatedly. Source files are rewritten each run; your .env and
# the Neo4j data volume are never touched once they exist. Containers are
# stopped elsewhere, never removed - same policy as the rest of the repo.
#
# Knobs (env vars, all optional):
#     BASE_DIR            where the project folder goes   (default: $PWD)
#     NEO4J_PASSWORD      database password               (default: temporalgraph2026)
#     NEO4J_HTTP_PORT     browser port                    (default: 7475)
#     NEO4J_BOLT_PORT     bolt port                       (default: 7688)
#     SKIP_DOCKER=1       write files only, no container
#     SKIP_UV=1           write files only, no uv / sync
#
set -euo pipefail

BASE_DIR="${BASE_DIR:-$PWD}"
PROJECT_DIR="$BASE_DIR/13_temporal_bitemporal_graphiti"

NEO4J_IMAGE="neo4j:5.26"
NEO4J_CONTAINER="neo4j-temporal-graph"
NEO4J_VOLUME="neo4j-temporal-graph-data"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-temporalgraph2026}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7475}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7688}"

SKIP_DOCKER="${SKIP_DOCKER:-0}"
SKIP_UV="${SKIP_UV:-0}"

say()  { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }

## Preflight checks
say "Preflight"
note "project dir : $PROJECT_DIR"
note "container   : $NEO4J_CONTAINER (ports $NEO4J_HTTP_PORT/$NEO4J_BOLT_PORT - 03's container is untouched)"

if [ "$SKIP_DOCKER" != "1" ]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker is not installed or not on PATH." >&2
        echo "Install it first (Ubuntu: sudo apt-get install docker.io), then rerun." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: docker is installed but the daemon is not reachable." >&2
        echo "Start it (sudo systemctl start docker) or fix group permissions, then rerun." >&2
        exit 1
    fi
    note "docker      : ok"
fi

## uv and Python 3.12
if [ "$SKIP_UV" != "1" ]; then
    say "uv + Python 3.12"
    if ! command -v uv >/dev/null 2>&1; then
        note "uv not found - installing"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv install did not land on PATH. Open a new shell and rerun." >&2
        exit 1
    fi
    note "uv          : $(uv --version)"
    uv python install 3.12
fi

## Project files
say "Writing project files (sources rewritten every run; .env preserved)"
mkdir -p "$PROJECT_DIR/corpus" "$PROJECT_DIR/tests"
cd "$PROJECT_DIR"

cat > "pyproject.toml" << 'PYPROJECT_EOF'
[project]
name = "temporal-bitemporal-graphiti"
version = "0.1.0"
description = "The index has no sense of time: bi-temporal facts with Graphiti on Neo4j"
requires-python = ">=3.12,<3.13"
dependencies = [
    # Pinned to the version named in the countermeasures sheet and verified
    # against this exercise's code (constructor, add_episode, search,
    # EntityEdge validity fields all checked on 0.29.x).
    "graphiti-core>=0.29,<0.30",
    "neo4j>=5.19",
    "openai>=1.0",
    "python-dotenv>=1.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
PYPROJECT_EOF

cat > ".env.example" << 'ENV_EXAMPLE_EOF'
# Copy this file to .env and add your OpenAI key.
#
#     cp .env.example .env
#
# Same OpenAI key convention as every other exercise in this repository.
OPENAI_API_KEY=sk-your-key-here

# THIS EXERCISE'S OWN Neo4j - a second container, separate from 03's, on
# shifted ports. Community Edition runs one database per instance and
# Graphiti owns its own indices, so the two graph exercises never share.
NEO4J_URI=bolt://localhost:__BOLT_PORT__
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=__NEO4J_PASSWORD__

# Model. The repository's standard cheap tier; passed to Graphiti explicitly
# because the library's own default is a stronger model than this needs.
CHAT_MODEL=gpt-4.1-mini

# graphiti-core sends anonymised usage telemetry by default; off, on principle.
GRAPHITI_TELEMETRY_ENABLED=false
ENV_EXAMPLE_EOF


# Bake the actual ports/password of THIS setup into the example, so a fresh
# .env copied from it just works. sed on placeholders, not on user files.
sed -i "s/__BOLT_PORT__/$NEO4J_BOLT_PORT/" .env.example
sed -i "s/__NEO4J_PASSWORD__/$NEO4J_PASSWORD/" .env.example

if [ ! -f .env ]; then
    cp .env.example .env
    note ".env created - add your OPENAI_API_KEY before running anything that costs money"
else
    note ".env exists - left untouched"
fi

cat > "README.md" << 'README_EOF'
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
README_EOF

cat > "corpus/2025-01-15_shop_overview.txt" << 'CORPUS_EOF'
THE FRUIT SHOP - WINTER OVERVIEW, 15 January 2025

The Fruit Shop on Storgatan sells apples, pears, bananas and grapes. Grapes
are supplied by Vindruva AB and have been one of the most popular items all
winter. Margit Lindqvist, the owner, samples every delivery herself before
it goes on the shelves.
CORPUS_EOF

cat > "corpus/2025-03-02_supplier_notice.txt" << 'CORPUS_EOF'
SUPPLIER NOTICE - VINDRUVA AB, 2 March 2025

Formal notice regarding grape consignments from Vindruva AB. Three
consecutive shipments have failed inspection: mold was found in two
consignments and the third was improperly packed. Effective immediately,
The Fruit Shop terminates its supply contract with Vindruva AB. No
replacement grape supplier has been appointed.
CORPUS_EOF

cat > "corpus/2025-03-10_owner_letter.txt" << 'CORPUS_EOF'
LETTER FROM THE OWNER, 10 March 2025

To our customers: you will have noticed there are no grapes on our shelves.
I will not sell fruit I would not eat myself. After the failed inspections
of the Vindruva deliveries I ended that contract, and until I find a grower
I trust, The Fruit Shop will not sell grapes. - Margit Lindqvist
CORPUS_EOF

cat > "corpus/2025-04-01_noticeboard_flyer.txt" << 'CORPUS_EOF'
NEIGHBOURHOOD NOTICEBOARD - FRUIT GOSSIP, 1 April 2025

Why does The Fruit Shop refuse to sell grapes? The real story: the owner,
Margit Lindqvist, is allergic to grapes and cannot handle them. She
developed the allergy after a childhood incident at a vineyard in France.
That is why you will never see grapes in that shop again.
CORPUS_EOF

cat > "corpus/2025-05-20_new_supplier.txt" << 'CORPUS_EOF'
ANNOUNCEMENT - NEW GRAPE SUPPLIER, 20 May 2025

The Fruit Shop has appointed Solvik Vineyards as its new grape supplier.
Solvik's trial consignments passed every inspection, and Margit Lindqvist
has approved the partnership after visiting the vineyard in person. Grapes
return to the shelves on 1 June 2025.
CORPUS_EOF

cat > "corpus/2025-06-05_shop_update.txt" << 'CORPUS_EOF'
SHOP UPDATE, 5 June 2025

Grapes are back. The first Solvik Vineyards deliveries arrived on 1 June
and sold out within two days. The Fruit Shop now sells apples, pears,
bananas and grapes again, and Margit Lindqvist says the wait was worth it.
CORPUS_EOF

cat > "main.py" << 'MAIN_PY_EOF'
"""The index has no sense of time: bi-temporal facts with Graphiti on Neo4j.

Two retrieved documents disagree because months passed between them, and a
similarity retriever has no idea: it presents both as simultaneous truth. The
countermeasure demonstrated here is BI-TEMPORAL modelling - every extracted
fact carries two independent time axes:

  valid time        when the fact was true in the world
                    (grapes were on sale until early March)
  transaction time  when the system learned it
                    (the flyer arrived on 1 April claiming something that
                    was never true at any point)

Graphiti (graphiti-core) does this per episode: facts become graph edges
with valid_at / invalid_at intervals, and when new knowledge contradicts old
knowledge it INVALIDATES the prior edge with a timestamp instead of deleting
it. History stays queryable - which is also the audit story.

The corpus is the fruit shop again, now as a six-document TIMELINE ingested
in date order, including one document 03 never had: a resolution (a new
supplier in May, grapes back in June). Two traps are built in:

  - the false flyer (owner "allergic to grapes") arrives AFTER the true
    documents, so any prefer-the-newest heuristic picks the lie;
  - the final state contradicts the middle state, so "does the shop sell
    grapes" has three different correct answers depending on the date asked
    about. A retriever without valid time can only give one.

The walkthrough asks the same question AS OF three dates and shows the
validity filtering that makes the answers differ; then it asks the evolution
question and the recency-trap question; then it proves the superseded edges
still exist, invalidation timestamps and all.

Run me:  uv run python main.py walk     ingests if needed, then queries
         uv run python main.py doctor   config check, no model calls, free
         uv run python main.py graph    inspect the temporal graph, free
         uv run python main.py clean    remove everything this wrote
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"

# A SECOND Neo4j container, separate from 03's: Community Edition runs one
# database per instance, and Graphiti owns its own indices and constraints -
# pointing it at the container holding 03's paid-for index would collide.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "temporalgraph2026")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# The three as-of dates the point-in-time question is asked against, plus
# the two free-text questions. Every document date is in the filename, so
# the timeline is visible in a directory listing.
ASOF_DATES = ["2025-02-01", "2025-04-15", "2025-06-15"]
POINT_QUESTION = "Is the shop selling grapes?"
EVOLUTION_QUESTION = "How did the shop's grape policy change during 2025?"
TRAP_QUESTION = "Why did the shop stop selling grapes?"

SYNTHESIS_PROMPT = """You are answering a question about a small shop using FACTS extracted from
a temporal knowledge graph. Every fact carries validity metadata:

  valid_at    when the fact became true in the world (may be unknown)
  invalid_at  when it stopped being true (empty means still current)

{asof_clause}

FACTS:
{facts}

QUESTION: {question}

Answer from the facts and their validity intervals only. When facts
contradict each other, use the intervals to order them in time rather than
blending them into one claim. Answer in a short paragraph."""


# ---- clients ----------------------------------------------------------------
def get_graphiti():
    """One Graphiti client, wired to this exercise's own container and to the
    repository's standard cheap model. Imported lazily so 'doctor' can report
    a missing package instead of dying on the import line.

    Verified against graphiti-core 0.29.3: constructor (uri, user, password,
    llm_client, ...), OpenAIClient(config=LLMConfig(...)). Without an explicit
    model, graphiti's default is a stronger model than this exercise needs.
    """
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient

    llm = OpenAIClient(config=LLMConfig(model=CHAT_MODEL, small_model=CHAT_MODEL))
    return Graphiti(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, llm_client=llm)


def get_driver():
    """Plain bolt driver for the free commands (doctor, graph, clean) so they
    never construct an LLM client. Notifications off, as everywhere else."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        notifications_min_severity="OFF",
    )


def run_cypher(driver, query, **params):
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def openai_answer(prompt: str) -> str:
    """One direct completion for the synthesis step. Graphiti extracts and
    stores facts; the ANSWERING from validity-filtered facts is this
    exercise's own code, kept in the open on purpose."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ---- ingestion ---------------------------------------------------------------
def timeline_documents() -> list[tuple[datetime, Path]]:
    """Corpus files are named YYYY-MM-DD_topic.txt; the date prefix becomes
    each episode's reference_time - the transaction-time axis. Sorted, so
    ingestion replays history in the order the shop lived it."""
    documents = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        date_part = path.name.split("_", 1)[0]
        reference = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        documents.append((reference, path))
    return documents


def already_ingested(driver) -> bool:
    rows = run_cypher(driver, "MATCH (e:Episodic) RETURN count(e) AS n")
    return rows[0]["n"] > 0


async def ingest() -> None:
    """Feed the timeline through Graphiti in date order.

    Each document becomes an episode with reference_time set to its date.
    Graphiti extracts entities and facts, assigns validity intervals, and -
    the part no other exercise in this repository does - INVALIDATES earlier
    edges that a later episode contradicts, stamping invalid_at rather than
    deleting anything.
    """
    from graphiti_core.nodes import EpisodeType

    graphiti = get_graphiti()
    try:
        await graphiti.build_indices_and_constraints()

        documents = timeline_documents()
        print(f"[ingest] {len(documents)} documents in date order - real model calls")
        for reference, path in documents:
            print(f"   {reference.date()}  {path.name}")
            await graphiti.add_episode(
                name=path.stem,
                episode_body=path.read_text(),
                source_description=f"shop document dated {reference.date()}",
                reference_time=reference,
                source=EpisodeType.text,
            )
        print("[ingest] complete")
    finally:
        await graphiti.close()


# ---- temporal querying --------------------------------------------------------
def edge_line(edge) -> str:
    """One printable line per fact with its validity interval."""
    valid = edge.valid_at.date().isoformat() if edge.valid_at else "unknown"
    if edge.invalid_at:
        invalid = edge.invalid_at.date().isoformat()
    else:
        invalid = "current"
    return f"[valid {valid} -> {invalid}] {edge.fact}"


def valid_as_of(edges, as_of: datetime):
    """The validity filter that IS this exercise's point: a fact counts on a
    date only if it had become true by then and had not yet stopped being
    true. Facts with unknown valid_at are kept (excluding them would hide
    background facts the extractor could not date)."""
    kept = []
    for edge in edges:
        starts_ok = edge.valid_at is None or edge.valid_at <= as_of
        ends_ok = edge.invalid_at is None or edge.invalid_at > as_of
        if starts_ok and ends_ok:
            kept.append(edge)
    return kept


async def ask(question: str, as_of_date: str | None = None) -> str:
    """Search the temporal graph, filter by validity when a date is given,
    then synthesise an answer from the surviving facts."""
    graphiti = get_graphiti()
    try:
        edges = await graphiti.search(question, num_results=14)
    finally:
        await graphiti.close()

    if as_of_date is not None:
        as_of = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        before = len(edges)
        edges = valid_as_of(edges, as_of)
        print(f"   (validity filter for {as_of_date}: {before} facts -> {len(edges)})")
        asof_clause = f"Answer AS OF {as_of_date}: only what was true on that date counts."
    else:
        asof_clause = "No as-of date: describe the situation across time where relevant."

    if not edges:
        return "(no facts survived the validity filter - has the walk been run?)"

    facts = "\n".join(edge_line(edge) for edge in edges)
    return openai_answer(SYNTHESIS_PROMPT.format(
        asof_clause=asof_clause, facts=facts, question=question))


# ---- inspection ----------------------------------------------------------------
def inspect_graph() -> None:
    """Read back the temporal graph without model calls. The headline is the
    superseded-but-preserved edges: invalidation as an audit trail."""
    driver = get_driver()
    try:
        if not already_ingested(driver):
            print("[graph] nothing ingested yet - run 'main.py walk' first")
            return

        print("\n=== What ingestion built ===")
        rows = run_cypher(
            driver,
            "MATCH (e:Episodic) WITH count(e) AS episodes "
            "MATCH (n:Entity) WITH episodes, count(n) AS entities "
            "OPTIONAL MATCH ()-[r:RELATES_TO]->() "
            "RETURN episodes, entities, count(r) AS facts",
        )
        if rows:
            print(f"\n  Episodes ingested : {rows[0]['episodes']}")
            print(f"  Entities          : {rows[0]['entities']}")
            print(f"  Fact edges        : {rows[0]['facts']}")

        # The assertion no other exercise makes: history is preserved, not
        # overwritten. Superseded facts remain, stamped with invalid_at.
        superseded = run_cypher(
            driver,
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.invalid_at IS NOT NULL
            RETURN r.fact AS fact, r.valid_at AS valid_at, r.invalid_at AS invalid_at
            ORDER BY r.invalid_at
            """,
        )
        print(f"\n  Superseded facts still present (the audit trail): {len(superseded)}")
        for row in superseded[:8]:
            print(f"    invalidated {str(row['invalid_at'])[:10]}: {row['fact']}")
        if not superseded:
            print("    (none yet - the extractor found no contradictions to invalidate)")
    finally:
        driver.close()


# ---- doctor ---------------------------------------------------------------------
def doctor() -> None:
    """Check the setup without spending anything."""
    print("=== doctor: configuration check, no model calls ===\n")

    documents = timeline_documents()
    print(f"  corpus documents      : {len(documents)} (dated "
          f"{documents[0][0].date()} .. {documents[-1][0].date()})" if documents
          else "  corpus documents      : 0")

    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-your"):
        print(f"  OpenAI API key        : present ({key[:6]}...{key[-4:]})")
    else:
        print("  OpenAI API key        : MISSING - copy .env.example to .env and add it")

    telemetry = os.getenv("GRAPHITI_TELEMETRY_ENABLED", "true")
    print(f"  graphiti telemetry    : {telemetry} (.env sets false; the library "
          f"phones home by default)")

    try:
        import graphiti_core  # noqa: F401
        import importlib.metadata as md
        print(f"  graphiti-core         : {md.version('graphiti-core')}")
    except Exception as error:
        print(f"  graphiti-core         : NOT INSTALLED ({error})")
        return

    try:
        driver = get_driver()
        run_cypher(driver, "RETURN 1")
        print(f"  Neo4j reachable       : yes ({NEO4J_URI})")
        ingested = already_ingested(driver)
        print(f"  already ingested      : {ingested}")
        driver.close()
    except Exception as error:
        print(f"  Neo4j reachable       : NO - {type(error).__name__}: {error}")
        print("                          is the TEMPORAL container running? docker ps")


# ---- walk -----------------------------------------------------------------------
async def walk() -> None:
    driver = get_driver()
    try:
        if already_ingested(driver):
            print("[ingest] existing temporal graph found - skipping ingestion "
                  "(main.py clean to redo)")
        else:
            await ingest()
    finally:
        driver.close()

    inspect_graph()

    print("\n=== POINT-IN-TIME: one question, three dates, three answers ===")
    print(f"Q: {POINT_QUESTION}\n")
    for as_of in ASOF_DATES:
        print(f"-- as of {as_of} --")
        print(await ask(POINT_QUESTION, as_of_date=as_of))
        print()

    print("=== EVOLUTION: the answer must be a sequence, not a blend ===")
    print(f"Q: {EVOLUTION_QUESTION}\n")
    print(await ask(EVOLUTION_QUESTION))

    print("\n=== RECENCY TRAP: the freshest claim about this is the false one ===")
    print(f"Q: {TRAP_QUESTION}\n")
    print(await ask(TRAP_QUESTION))

    print("\n" + "=" * 70)
    print("The flyer (2025-04-01) claims an allergy and postdates the true")
    print("documents, so prefer-the-newest picks the lie. What matters above is")
    print("whether the validity intervals ordered the contradiction correctly -")
    print("and check the audit section: if Graphiti invalidated the TRUE reason")
    print("in favour of the fresher false one, that is a finding too. Temporal")
    print("machinery inherits the provenance problem; it does not solve it.")


def clean() -> None:
    """Remove everything this exercise wrote to ITS OWN Neo4j. The container
    and volume survive; only the data goes."""
    driver = get_driver()
    try:
        run_cypher(driver, "MATCH (n) DETACH DELETE n")
        print("[clean] removed all nodes and relationships from the temporal graph")
        print("[clean] Graphiti's indexes remain; re-ingestion recreates content")
    finally:
        driver.close()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "walk":
        asyncio.run(walk())
    elif command == "doctor":
        doctor()
    elif command == "graph":
        inspect_graph()
    elif command == "clean":
        clean()
    else:
        print("Usage: main.py [walk | doctor | graph | clean]")
MAIN_PY_EOF

cat > "tests/conftest.py" << 'CONFTEST_EOF'
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
CONFTEST_EOF

cat > "tests/test_temporal.py" << 'TESTS_EOF'
"""Tests split by cost: the free ones always run, the expensive one is opt-in
via TEMPORAL_RUN_INDEX=1 (ingestion makes many model calls)."""
import os
from datetime import datetime, timezone

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="no API key set",
)
requires_ingest = pytest.mark.skipif(
    os.getenv("TEMPORAL_RUN_INDEX") != "1",
    reason="set TEMPORAL_RUN_INDEX=1 to run the ingestion test (costs money)",
)


def neo4j_reachable() -> bool:
    try:
        driver = main.get_driver()
        main.run_cypher(driver, "RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


requires_neo4j = pytest.mark.skipif(
    not neo4j_reachable(),
    reason="temporal Neo4j not reachable - start the container first",
)


def test_timeline_is_dated_ordered_and_trapped():
    documents = main.timeline_documents()
    assert len(documents) == 6
    dates = [reference for reference, _ in documents]
    assert dates == sorted(dates), "corpus must replay in date order"

    # The recency trap: the false flyer postdates the true documents...
    flyer = main.CORPUS_DIR / "2025-04-01_noticeboard_flyer.txt"
    assert "allergic" in flyer.read_text().lower()
    notice = main.CORPUS_DIR / "2025-03-02_supplier_notice.txt"
    assert "inspection" in notice.read_text().lower()
    assert "allerg" not in notice.read_text().lower()
    # ...and the final state contradicts the middle state.
    update = main.CORPUS_DIR / "2025-06-05_shop_update.txt"
    assert "grapes are back" in update.read_text().lower()


def test_validity_filter_logic():
    """The as-of filter is this exercise's core mechanism; pin it with a
    synthetic edge so the paid test is not the only thing testing it."""
    class FakeEdge:
        def __init__(self, valid_at, invalid_at):
            self.valid_at = valid_at
            self.invalid_at = invalid_at
            self.fact = "fake"

    def day(iso):
        return datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sells = FakeEdge(day("2025-01-01"), day("2025-03-02"))   # true then ended
    stopped = FakeEdge(day("2025-03-02"), day("2025-06-01")) # the gap
    back = FakeEdge(day("2025-06-01"), None)                 # current
    undated = FakeEdge(None, None)                           # kept always
    edges = [sells, stopped, back, undated]

    assert main.valid_as_of(edges, day("2025-02-01")) == [sells, undated]
    assert main.valid_as_of(edges, day("2025-04-15")) == [stopped, undated]
    assert main.valid_as_of(edges, day("2025-06-15")) == [back, undated]


def test_the_package_exposes_what_this_exercise_calls():
    from graphiti_core import Graphiti  # noqa: F401
    from graphiti_core.llm_client.config import LLMConfig  # noqa: F401
    from graphiti_core.llm_client.openai_client import OpenAIClient  # noqa: F401
    from graphiti_core.nodes import EpisodeType

    assert hasattr(EpisodeType, "text")
    assert hasattr(Graphiti, "build_indices_and_constraints")


@requires_neo4j
def test_inspect_graph_is_safe_with_no_ingest(capsys):
    main.inspect_graph()
    assert capsys.readouterr().out


@requires_key
@requires_neo4j
@requires_ingest
@pytest.mark.asyncio
async def test_ingest_and_the_three_dates_answer_differently():
    driver = main.get_driver()
    try:
        if not main.already_ingested(driver):
            await main.ingest()
        assert main.already_ingested(driver)
    finally:
        driver.close()

    february = (await main.ask(main.POINT_QUESTION, as_of_date="2025-02-01")).lower()
    april = (await main.ask(main.POINT_QUESTION, as_of_date="2025-04-15")).lower()
    assert february != april, "as-of answers must differ across the timeline"
TESTS_EOF


note "corpus files   : $(ls corpus | wc -l) (dated, replayed in order)"

## Neo4j container - the exercise's own, on shifted ports
if [ "$SKIP_DOCKER" != "1" ]; then
    say "Neo4j container ($NEO4J_IMAGE, ports $NEO4J_HTTP_PORT/$NEO4J_BOLT_PORT)"

    if docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
        if docker ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
            note "container already running"
        else
            note "container exists but stopped - starting it"
            docker start "$NEO4J_CONTAINER" >/dev/null
        fi
    else
        note "creating container (data persists in volume $NEO4J_VOLUME)"
        # Graphiti needs neither GDS nor APOC on 0.29.x, but APOC costs
        # nothing and spares a rebuild if a future version reaches for it -
        # the exact trap the 03 exercise hit.
        docker run -d \
            --name "$NEO4J_CONTAINER" \
            -p "$NEO4J_HTTP_PORT:7474" \
            -p "$NEO4J_BOLT_PORT:7687" \
            -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" \
            -e NEO4J_PLUGINS='["apoc"]' \
            -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
            -v "$NEO4J_VOLUME:/data" \
            "$NEO4J_IMAGE" >/dev/null
    fi

    note "waiting for Neo4j to answer on bolt (first boot can take ~2 min)"
    ready=0
    for i in $(seq 1 60); do
        if docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
                "RETURN 1" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 3
    done
    if [ "$ready" != "1" ]; then
        echo "ERROR: Neo4j did not become ready. Inspect with:" >&2
        echo "    docker logs $NEO4J_CONTAINER" >&2
        echo "A common cause is a password mismatch with an old data volume;" >&2
        echo "a clean rebuild is:" >&2
        echo "    docker rm -f $NEO4J_CONTAINER && docker volume rm $NEO4J_VOLUME" >&2
        exit 1
    fi
    note "Neo4j is up"
fi

## Python dependencies
if [ "$SKIP_UV" != "1" ]; then
    say "Python dependencies (uv sync)"
    uv sync
fi

## Doctor
if [ "$SKIP_UV" != "1" ]; then
    say "Doctor (free, no model calls)"
    uv run python main.py doctor || true
fi

say "Done"
note "Next (from $PROJECT_DIR):"
note "  1. put your OpenAI key in .env               (only needed once)"
note "  2. uv run python main.py walk                (ingests the timeline, ~cents)"
note "  3. uv run python main.py graph               (inspect, free)"
note "  4. browse http://localhost:$NEO4J_HTTP_PORT  (login neo4j / your password)"
note ""
note "Suite registration (one line each, when you want it in the sweeps):"
note "  - add \"13_temporal_bitemporal_graphiti\" to EXERCISES in both suite scripts"
note "  - export TEMPORAL_RUN_INDEX=1 next to GRAPHRAG_RUN_INDEX in the paid suite"
note ""
note "Container management:"
note "  stop:          docker stop $NEO4J_CONTAINER"
note "  start again:   docker start $NEO4J_CONTAINER   (or rerun this script)"
note "  full rebuild:  docker rm -f $NEO4J_CONTAINER && docker volume rm $NEO4J_VOLUME"
