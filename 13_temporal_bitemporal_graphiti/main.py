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
import logging
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

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

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
