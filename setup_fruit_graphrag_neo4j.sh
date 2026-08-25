#!/usr/bin/env bash
#
# setup_fruit_graphrag_neo4j.sh
#
# One idempotent script that builds the whole 03_graphrag_fruit_graph
# exercise on Neo4j: docker container (Neo4j 5 + Graph Data Science plugin),
# uv, Python 3.12, dependencies, source files, corpus, tests.
#
# Run it from the directory that should CONTAIN the project folder:
#
#     cd /home/kai/gen_ai_topics_practice
#     ./setup_fruit_graphrag_neo4j.sh
#
# Safe to run repeatedly. Source files are rewritten each run (so rerunning
# picks up script updates); your .env and the Neo4j data volume are never
# touched once they exist.
#
# Knobs (env vars, all optional):
#     BASE_DIR            where the project folder goes   (default: $PWD)
#     NEO4J_PASSWORD      database password               (default: fruitgraph2026)
#     NEO4J_HTTP_PORT     browser port                    (default: 7474)
#     NEO4J_BOLT_PORT     bolt port                       (default: 7687)
#     SKIP_DOCKER=1       write files only, no container
#     SKIP_UV=1           write files only, no uv / sync
#
set -euo pipefail

BASE_DIR="${BASE_DIR:-$PWD}"
PROJECT_DIR="$BASE_DIR/03_graphrag_fruit_graph"

NEO4J_IMAGE="neo4j:5.26"
NEO4J_CONTAINER="neo4j-fruit-graph"
NEO4J_VOLUME="neo4j-fruit-graph-data"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-fruitgraph2026}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"

SKIP_DOCKER="${SKIP_DOCKER:-0}"
SKIP_UV="${SKIP_UV:-0}"

say()  { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }

## Preflight checks
say "Preflight"
note "project dir : $PROJECT_DIR"

if [ "$SKIP_DOCKER" != "1" ]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker is not installed or not on PATH." >&2
        echo "Install it first (Ubuntu: sudo apt-get install docker.io," >&2
        echo "or follow https://docs.docker.com/engine/install/), then rerun." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: docker is installed but the daemon is not reachable." >&2
        echo "Start it (sudo systemctl start docker) or fix group permissions" >&2
        echo "(sudo usermod -aG docker \$USER, then log out and back in), then rerun." >&2
        exit 1
    fi
    note "docker      : ok"
fi

## uv and Python 3.12
if [ "$SKIP_UV" != "1" ]; then
    say "uv + Python 3.12"
    if ! command -v uv >/dev/null 2>&1; then
        # uv installs to ~/.local/bin by default; extend PATH for this run.
        note "uv not found - installing"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv install did not land on PATH. Open a new shell and rerun." >&2
        exit 1
    fi
    note "uv          : $(uv --version)"
    # Idempotent: no-op when 3.12 is already managed by uv.
    uv python install 3.12
fi

## Project files
say "Writing project files (sources rewritten every run; .env preserved)"
mkdir -p "$PROJECT_DIR/corpus" "$PROJECT_DIR/tests"
cd "$PROJECT_DIR"

cat > "pyproject.toml" << 'PYPROJECT_EOF'
[project]
name = "graphrag-fruit-graph"
version = "0.2.0"
description = "Corpus-level questions via an entity graph, hierarchical Leiden and community reports - on Neo4j"
requires-python = ">=3.12,<3.13"
dependencies = [
    # First-party Neo4j GraphRAG package. Loose upper bound: the KG builder
    # lives under an experimental namespace and parameter names have moved
    # between releases, which is why main.py introspects signatures instead
    # of pinning hard. Tighten this once the whitepaper run is frozen.
    "neo4j-graphrag[openai]>=1.6,<2",
    "neo4j>=5.19",
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

# Neo4j connection. The defaults match what setup builds; only change these
# if you changed the ports or password in the setup script.
NEO4J_URI=bolt://localhost:__BOLT_PORT__
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=__NEO4J_PASSWORD__

# Models. gpt-4.1-mini and text-embedding-3-small match the Microsoft-era
# settings.yaml so results stay comparable for the whitepaper.
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
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
# graphrag_fruit_graph (Neo4j edition)

**Theme.** Top-k retrieval is structurally local. "What connects the regular
customers?" is a property of the whole corpus, and a chunk retriever answers
it from an arbitrary sample. GraphRAG extracts an entity graph at index time,
clusters it with hierarchical Leiden, writes a report for every community,
and answers global questions from summaries of summaries.

**Stack.** This is the same methodology as the Microsoft graphrag version of
this exercise, rebuilt on maintained components after Microsoft moved
graphrag into maintenance mode:

* Neo4j 5 (docker) with the Graph Data Science plugin
* neo4j-graphrag (first-party Neo4j package): extraction, entity resolution,
  local retrieval, answer generation
* GDS `gds.leiden` with `includeIntermediateCommunities`: the hierarchical
  Leiden clustering
* plain Python + Cypher for community reports and global search - the one
  step no surviving framework packages, kept in the open on purpose
* OpenAI `gpt-4.1-mini` and `text-embedding-3-small`, uv on Python 3.12

## Run it

    ./setup_fruit_graphrag_neo4j.sh   # from the parent directory
    uv run python main.py doctor    # config check, no model calls, free
    uv run python main.py walk      # indexes for real: minutes, and cents
    uv run python main.py graph     # inspect an existing index, free
    uv run pytest -q                # free tests; see below for the paid one
    uv run python main.py clean     # wipe the graph (container survives)

Browse the graph at http://localhost:7474 while it exists - the grape
community sitting apart from the allergy flyer is the whitepaper screenshot.

## The corpus, and the distractor

Nine short documents about one fruit shop. Eight are ordinary records. The
ninth, `grape_rumour_flyer.txt`, restates the grape question almost word for
word and answers it with a mechanism that is wrong - an allergy. The real
reason is spread across `supplier_notice.txt` and `owner_letter.txt`, neither
of which mentions allergy at all.

The flyer will win on lexical similarity, because it contains the question.
Whether the answer says "allergy" or "failed inspections" tells you whether
the graph assembled the chain or the retriever simply matched the
nearest-looking text.

## Cost

`doctor` and `graph` are free. `walk` extracts, clusters and summarises with
real model calls: a few minutes and a few cents. The walkthrough skips
re-extraction when the graph already holds chunks, and skips re-summarising
communities that already have reports, so subsequent runs only pay for the
two queries.

The expensive test is opt-in:

    GRAPHRAG_RUN_INDEX=1 uv run pytest -q
README_EOF

cat > "corpus/bruised_basket.txt" << 'CORPUS_EOF'
BRUISED FRUIT POLICY
Fruit that arrives bruised is sold at half price from a basket by the door
rather than discarded. This is the only exception to the fixed pricing.

The basket is refilled on delivery days and is usually empty by closing.
Margit considers waste a worse outcome than a thin margin.
CORPUS_EOF

cat > "corpus/customer_ingrid.txt" << 'CORPUS_EOF'
CUSTOMER RECORD - INGRID
Ingrid has shopped here every Monday for eight years, almost always for
apples. She bakes, and once taught the Saturday staff her apple cake recipe.

She says our apples remind her of her aunt's orchard outside Uppsala. She has
never once complained about the price.
CORPUS_EOF

cat > "corpus/customer_kwon.txt" << 'CORPUS_EOF'
CUSTOMER RECORD - KWON
Kwon buys bananas in bulk for his running club, usually six kilograms at a
time, on Friday afternoons before the weekend long run.

He jokes that the club runs on bananas and that we should sponsor their
annual race. He has never bought a pear.
CORPUS_EOF

cat > "corpus/customer_tendai.txt" << 'CORPUS_EOF'
CUSTOMER RECORD - TENDAI
Tendai prefers pears above everything and once bought an entire crate before
a family gathering.

Tendai asks about grapes on most visits and always receives the same answer.
He has been coming since the shop opened.
CORPUS_EOF

cat > "corpus/delivery_schedule.txt" << 'CORPUS_EOF'
DELIVERY SCHEDULE
Deliveries arrive on Tuesdays and Fridays. Apples come from Uppsala, bananas
through the Gothenburg wholesaler, and pears from a co-operative in Skane.

Friday deliveries are larger because weekend trade is heavier, which is why
bulk orders are collected on Friday afternoons.
CORPUS_EOF

cat > "corpus/grape_rumour_flyer.txt" << 'CORPUS_EOF'
NEIGHBOURHOOD NOTICEBOARD - FRUIT GOSSIP
DATE: 2025-04-01

Several people have asked why the shop refuses to sell grapes.

The answer is simple: the owner is allergic to grapes and cannot handle them.
This allergy is why the shop refuses to sell grapes, and it has nothing to do
with suppliers or with quality. The allergy developed after a childhood
incident involving a vineyard in France.

There is no truth to the rumour about a terminated supply contract.
CORPUS_EOF

cat > "corpus/owner_letter.txt" << 'CORPUS_EOF'
LETTER FROM THE OWNER
DATE: 2025-03-10

Customers keep asking about grapes, so let me be plain about it.

I will not sell fruit I would not eat myself. After what arrived from
Vindruva last winter I would not put those grapes in my own kitchen, and I am
not going to put them in yours. Until I find a grower whose crates I trust,
this shop does not sell grapes.

Margit
CORPUS_EOF

cat > "corpus/shop_overview.txt" << 'CORPUS_EOF'
THE FRUIT SHOP - OVERVIEW
The Fruit Shop has traded on Bergsgatan for twelve years under its founder,
Margit Lindqvist. It sells exactly three fruits: apples at 3 per kilogram,
bananas at 2 per kilogram, and pears at 4 per kilogram.

Prices are deliberately simple so children can do the arithmetic. There is no
discount tier and no loyalty scheme; what is on the sign is what everyone
pays, including regulars.
CORPUS_EOF

cat > "corpus/supplier_notice.txt" << 'CORPUS_EOF'
SUPPLIER NOTICE - VINDRUVA AB
DATE: 2025-03-02

Vindruva AB regrets that three consecutive grape consignments to The Fruit
Shop failed inspection on arrival. Two crates showed mould at the stem, and
one had been packed at the wrong temperature during transit.

The Fruit Shop has terminated its supply contract with Vindruva AB. No
replacement grape supplier has been appointed.
CORPUS_EOF

cat > "main.py" << 'MAIN_PY_EOF'
"""Corpus-level questions via GraphRAG on Neo4j, driven from Python.

Top-k retrieval is structurally local. A question whose answer is a property
of the WHOLE corpus - "what connects the regular customers?" - gets answered
from an arbitrary sample of chunks. The GraphRAG answer is to extract an
entity graph at index time, cluster it with hierarchical Leiden, and
pre-generate a summary for every community, so global questions are answered
from summaries of summaries.

This version keeps that methodology but retires Microsoft's graphrag library.
The moving parts are:

  extraction        neo4j-graphrag SimpleKGPipeline (first-party Neo4j package)
  entity resolution neo4j-graphrag exact-match resolver (merges duplicate
                    entities across documents; without it every document gets
                    its own "The Fruit Shop" node and Leiden just rediscovers
                    the document boundaries)
  communities       Neo4j GDS gds.leiden with includeIntermediateCommunities,
                    which is the same hierarchical Leiden the paper used
  summaries         a plain loop over communities with one LLM call each,
                    stored on (:Community) nodes - the one step no surviving
                    framework packages, so it is written out in the open here
  local search      neo4j-graphrag VectorCypherRetriever: vector match on
                    chunks, then expand through shared entities to the chunks
                    those entities also appear in
  global search     fetch every community summary at the final Leiden level
                    and synthesise one answer from them

The corpus contains one deliberate distractor - a noticeboard flyer that
restates the grape question almost verbatim and answers it with the wrong
mechanism. Watch what each query mode does with it.

Run me:  uv run python main.py walk     full run, indexes if needed
         uv run python main.py doctor   config check, no model calls, free
         uv run python main.py graph    inspect an existing index, free
         uv run python main.py clean    remove generated state from Neo4j
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"

# Connection details come from .env so the same code runs against the docker
# container the setup script starts, or against any other Neo4j you point it
# at. The defaults match the setup script.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fruitgraph2026")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small

VECTOR_INDEX = "chunk_embeddings"
GDS_GRAPH = "fruit_graph"

# The same entity types the settings.yaml declared in the Microsoft version.
ENTITY_TYPES = ["Person", "Organization", "Product", "Policy", "Location"]

GLOBAL_QUESTION = "What connects the shop's regular customers, and what do their habits have in common?"
LOCAL_QUESTION = "Why does the shop refuse to sell grapes?"

# The community prompt keeps the spirit of the original "butterfly effect"
# prompt from the mini_graph_rag project: name the mechanism, not the list.
COMMUNITY_SUMMARY_PROMPT = """You are analyzing one community from a knowledge graph built over a small
document corpus about a fruit shop.

Community members: {members}
Relationships inside the community:
{triples}

TASK:
Write a short report (3-5 sentences) on what this community is about.
1. Focus on the specific, tangible chain of events or the shared mechanism
   connecting these entities.
2. State causes explicitly when the relationships support them.
3. Do not speculate beyond what the relationships imply.
Return ONLY the report text."""

GLOBAL_SYNTHESIS_PROMPT = """You are answering a question about a document corpus as a whole. You are
given the complete set of community reports produced at index time from the
corpus entity graph. Answer ONLY from these reports.

QUESTION: {question}

COMMUNITY REPORTS:
{reports}

Answer in multiple paragraphs. Cite which communities support each claim by
their id, like [community 3]."""


# ---- version-drift guard --------------------------------------------------
def call_with_supported_args(func, **candidates):
    """Call func with only the keyword arguments it actually accepts.

    Carried over from the Microsoft version, where it earned its keep. It is
    kept here for one reason: neo4j-graphrag's SimpleKGPipeline renamed its
    schema parameters (entities/relations -> node_types/relationship_types)
    between releases. Pass both spellings as candidates and let the installed
    version pick. Anything dropped is reported rather than silently ignored.
    """
    signature = inspect.signature(func)
    accepted = set(signature.parameters)

    used = {}
    dropped = []
    for key, value in candidates.items():
        if key in accepted:
            used[key] = value
        else:
            dropped.append(key)

    if dropped:
        print(f"   (not accepted by this version, dropped: {', '.join(sorted(dropped))})")
    return func(**used)


def pick_param_name(func, value, *names):
    """Return {name: value} for the first name func accepts, else {}.

    Used where the SAME value has been known under different parameter names
    across neo4j-graphrag releases; passing both would double-supply it.
    """
    accepted = set(inspect.signature(func).parameters)
    for name in names:
        if name in accepted:
            return {name: value}
    return {}


# ---- clients ----------------------------------------------------------------
def get_driver():
    """One bolt driver per process. Imported lazily so 'doctor' can report a
    missing package instead of dying on the import line."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_llm():
    from neo4j_graphrag.llm import OpenAILLM

    return OpenAILLM(model_name=CHAT_MODEL, model_params={"temperature": 0})


def get_embedder():
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def run_cypher(driver, query, **params):
    """Small convenience wrapper: run one statement, return list of records."""
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


# ---- indexing: extraction ---------------------------------------------------
def already_indexed(driver) -> bool:
    rows = run_cypher(driver, "MATCH (c:Chunk) RETURN count(c) AS n")
    return rows[0]["n"] > 0


def ensure_vector_index(driver) -> None:
    """Vector index over chunk embeddings, used by local search.

    Raw Cypher with IF NOT EXISTS rather than the package helper, because the
    statement is stable across Neo4j 5.x while helper signatures have moved.
    """
    run_cypher(
        driver,
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
    )


async def extract(driver) -> None:
    """Extract entities and relationships from every corpus document.

    SimpleKGPipeline does per-document what graphrag's extract_graph did:
    chunking (400/50, matching the old settings.yaml), embedding, LLM entity
    and relationship extraction, and writing the lexical graph plus the
    entity graph into Neo4j.
    """
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

    llm = get_llm()
    embedder = get_embedder()

    # Chunking to the same shape as the Microsoft version's settings.yaml.
    # The splitter class has kept its home so far; if a future release moves
    # it, fall back to the pipeline default and say so.
    text_splitter = None
    try:
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
            FixedSizeSplitter,
        )

        text_splitter = FixedSizeSplitter(chunk_size=400, chunk_overlap=50)
    except ImportError as error:
        print(f"   (FixedSizeSplitter not importable in this version, using default chunking: {error})")

    kwargs = {
        "llm": llm,
        "driver": driver,
        "embedder": embedder,
        "from_pdf": False,
    }
    if text_splitter is not None:
        kwargs["text_splitter"] = text_splitter
    # Schema parameter renamed across releases: node_types (new) vs entities (old).
    kwargs.update(pick_param_name(SimpleKGPipeline.__init__, ENTITY_TYPES, "node_types", "entities"))

    pipeline = SimpleKGPipeline(**kwargs)

    documents = sorted(CORPUS_DIR.glob("*.txt"))
    print(f"[extract] {len(documents)} documents - this makes real model calls")
    for document in documents:
        print(f"   extracting {document.name}")
        await pipeline.run_async(text=document.read_text())

    counts = run_cypher(
        driver,
        "MATCH (e:__Entity__) WITH count(e) AS entities "
        "MATCH (c:Chunk) RETURN entities, count(c) AS chunks",
    )
    if counts:
        print(f"[extract] complete - {counts[0]['entities']} entities, {counts[0]['chunks']} chunks")


async def resolve_entities(driver) -> None:
    """Merge duplicate entities across documents.

    The old mini_graph_rag did this with embedding clustering (step_02). The
    first-party equivalent here is the exact-match resolver: entities sharing
    the same name become one node. Without resolution the graph is nine
    disconnected islands and Leiden has nothing interesting to find.

    Note: SimpleKGPipeline already runs this resolver itself after each
    document (perform_entity_resolution defaults to True, verified against
    v1.18.0), and the resolver matches over the WHOLE database, so entities
    are merged across documents as extraction proceeds. This explicit pass is
    belt-and-braces - a no-op when the pipeline has already done the work -
    and it keeps the step visible for the whitepaper narrative.
    """
    try:
        from neo4j_graphrag.experimental.components.resolver import (
            SinglePropertyExactMatchResolver,
        )
    except ImportError as error:
        print(f"[resolve] resolver not importable in this version ({error}) - "
              f"entity duplicates will remain and communities will be weaker")
        return

    resolver = SinglePropertyExactMatchResolver(driver)
    result = await resolver.run()
    print(f"[resolve] entity resolution complete: {result}")


# ---- indexing: communities --------------------------------------------------
def detect_communities(driver) -> None:
    """Hierarchical Leiden over the entity graph, via Neo4j GDS.

    This is the step that used to be graphrag's cluster_graph (and cdlib in
    the mini project, with its int/string mapping workaround - GDS handles
    node identity itself, so that whole fix disappears).

    includeIntermediateCommunities gives the hierarchy: the property on each
    entity becomes a LIST of community ids, one per Leiden level, with the
    final partition last. gamma is the resolution knob - the equivalent of
    the old max_cluster_size hack; higher gamma means more, smaller
    communities, which a nine-document corpus needs to produce more than one.
    concurrency 1 plus a fixed randomSeed makes the partition reproducible,
    which matters for a whitepaper.
    """
    # Idempotency: drop any previous projection and community nodes first.
    run_cypher(driver, f"CALL gds.graph.drop('{GDS_GRAPH}', false)")
    run_cypher(driver, "MATCH (c:Community) DETACH DELETE c")

    # Project only the entity-entity graph. Chunk and Document nodes are
    # excluded by the node label filter, so lexical edges never leak in.
    run_cypher(
        driver,
        f"""
        CALL gds.graph.project(
            '{GDS_GRAPH}',
            '__Entity__',
            {{ALL: {{type: '*', orientation: 'UNDIRECTED'}}}}
        )
        """,
    )

    run_cypher(
        driver,
        f"""
        CALL gds.leiden('{GDS_GRAPH}', {{
            writeProperty: 'leiden',
            includeIntermediateCommunities: true,
            gamma: 1.5,
            concurrency: 1,
            randomSeed: 42
        }})
        YIELD communityCount, modularity
        RETURN communityCount, modularity
        """,
    )

    # Materialise (:Community) nodes per level and connect members. Storing
    # communities as nodes is what makes them addressable by the summariser
    # and inspectable in Neo4j Browser.
    run_cypher(
        driver,
        """
        MATCH (e:__Entity__) WHERE e.leiden IS NOT NULL
        UNWIND range(0, size(e.leiden) - 1) AS level
        MERGE (c:Community {level: level, community_id: e.leiden[level]})
        MERGE (e)-[:IN_COMMUNITY {level: level}]->(c)
        """,
    )

    rows = run_cypher(
        driver,
        """
        MATCH (c:Community)
        RETURN c.level AS level, count(c) AS communities
        ORDER BY level
        """,
    )
    for row in rows:
        print(f"[leiden] level {row['level']}: {row['communities']} communities")
    print("[leiden] the last level is the final Leiden partition; global search reads that one")


def final_level(driver) -> int | None:
    rows = run_cypher(driver, "MATCH (c:Community) RETURN max(c.level) AS level")
    if rows and rows[0]["level"] is not None:
        return int(rows[0]["level"])
    return None


def summarize_communities(driver) -> None:
    """Write an LLM report for every community at every level.

    This is the step no surviving framework packages, so it lives here in
    plain sight: gather members and internal relationships, one model call,
    store the text on the Community node. It is also the direct descendant of
    step_03 in the mini_graph_rag project.
    """
    llm = get_llm()

    communities = run_cypher(
        driver,
        """
        MATCH (c:Community) WHERE c.summary IS NULL
        RETURN c.level AS level, c.community_id AS community_id
        ORDER BY level, community_id
        """,
    )
    if not communities:
        print("[summaries] nothing to do - every community already has a report")
        return

    print(f"[summaries] writing reports for {len(communities)} communities - real model calls")
    for community in communities:
        detail = run_cypher(
            driver,
            """
            MATCH (c:Community {level: $level, community_id: $cid})<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) AS members
            UNWIND members AS a
            UNWIND members AS b
            OPTIONAL MATCH (a)-[r]->(b)
            WITH members,
                 collect(DISTINCT CASE
                     WHEN r IS NULL THEN NULL
                     ELSE coalesce(a.name, a.id) + ' -[' + type(r) + ']-> ' + coalesce(b.name, b.id)
                 END) AS raw_triples
            RETURN [m IN members | coalesce(m.name, m.id)] AS names,
                   [t IN raw_triples WHERE t IS NOT NULL] AS triples
            """,
            level=community["level"],
            cid=community["community_id"],
        )
        if not detail:
            continue
        names = detail[0]["names"]
        triples = detail[0]["triples"]

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            members=", ".join(names),
            triples="\n".join(triples) if triples else "(no internal relationships recorded)",
        )
        response = llm.invoke(prompt)
        summary = response.content.strip()

        run_cypher(
            driver,
            """
            MATCH (c:Community {level: $level, community_id: $cid})
            SET c.summary = $summary
            """,
            level=community["level"],
            cid=community["community_id"],
            summary=summary,
        )
        print(f"   level {community['level']} community {community['community_id']}: "
              f"{len(names)} members, report written")


# ---- inspection --------------------------------------------------------------
def inspect_graph() -> None:
    """Read back what indexing built, without making any model calls."""
    driver = get_driver()
    try:
        if not already_indexed(driver):
            print("[graph] nothing indexed yet - run 'main.py walk' first")
            return

        print("\n=== What the indexing pipeline built ===")

        rows = run_cypher(
            driver,
            "MATCH (e:__Entity__) WITH count(e) AS entities "
            "MATCH (c:Chunk) WITH entities, count(c) AS chunks "
            "OPTIONAL MATCH (:__Entity__)-[r]->(:__Entity__) "
            "RETURN entities, chunks, count(r) AS relationships",
        )
        if rows:
            print(f"\n  Chunks embedded:    {rows[0]['chunks']}")
            print(f"  Entities extracted: {rows[0]['entities']}")
            print(f"  Relationships:      {rows[0]['relationships']}")

        degree = run_cypher(
            driver,
            """
            MATCH (e:__Entity__)-[r]-(:__Entity__)
            RETURN coalesce(e.name, e.id) AS name, count(r) AS degree
            ORDER BY degree DESC LIMIT 6
            """,
        )
        if degree:
            print("\n  Most connected entities (degree = edges touching the node):")
            for row in degree:
                print(f"    {row['degree']:>3}  {row['name']}")

        levels = run_cypher(
            driver,
            """
            MATCH (c:Community)<-[:IN_COMMUNITY]-(e)
            WITH c, count(e) AS members
            RETURN c.level AS level, count(c) AS communities,
                   collect(members) AS sizes
            ORDER BY level
            """,
        )
        if not levels:
            print("\n  no communities found - the community step has not run")
        else:
            print(f"\n  Leiden communities across {len(levels)} levels:")
            for row in levels:
                sizes = ", ".join(str(s) for s in sorted(row["sizes"], reverse=True))
                print(f"    level {row['level']}: {row['communities']:>2} communities  (entities each: {sizes})")
            print("\n  The last level is the final partition; earlier levels are intermediate.")

        reports = run_cypher(
            driver,
            """
            MATCH (c:Community) WHERE c.summary IS NOT NULL
            RETURN c.level AS level, c.community_id AS community_id, c.summary AS summary
            ORDER BY c.level DESC, c.community_id LIMIT 1
            """,
        )
        if reports:
            total = run_cypher(driver, "MATCH (c:Community) WHERE c.summary IS NOT NULL RETURN count(c) AS n")
            print(f"\n  Community reports written: {total[0]['n']}")
            row = reports[0]
            text = row["summary"].strip().replace("\n", "\n    ")
            print(f"\n  One of them (level {row['level']}, community {row['community_id']}):")
            print(f"    {text[:600]}")
    finally:
        driver.close()


# ---- queries ------------------------------------------------------------------
def ask_global(question: str) -> str:
    """Answer from community summaries - the whole set at the final level.

    This is the mode that can speak to the whole corpus, because the reports
    it reads were written at index time from the entire graph rather than
    retrieved in response to the question. The original map-reduce rated each
    report's relevance in a separate call before reducing; at nine documents
    that is ceremony, so all reports go into one synthesis call. The place to
    reintroduce map-reduce at scale is exactly here, and nowhere else.
    """
    driver = get_driver()
    try:
        level = final_level(driver)
        if level is None:
            return "(no communities found - run 'main.py walk' first)"

        reports = run_cypher(
            driver,
            """
            MATCH (c:Community {level: $level}) WHERE c.summary IS NOT NULL
            RETURN c.community_id AS community_id, c.summary AS summary
            ORDER BY community_id
            """,
            level=level,
        )
        if not reports:
            return "(no community reports found - run 'main.py walk' first)"

        report_text = "\n\n".join(
            f"[community {r['community_id']}]\n{r['summary']}" for r in reports
        )
        llm = get_llm()
        response = llm.invoke(GLOBAL_SYNTHESIS_PROMPT.format(question=question, reports=report_text))
        return response.content.strip()
    finally:
        driver.close()


# Local search: vector match on chunks, then expand through the entities each
# chunk mentions to OTHER chunks those entities appear in. This expansion is
# the graph doing its job on the distractor: the flyer chunk wins the vector
# match because it contains the question, but its entities (grapes, the shop)
# are shared with the supplier notice and the owner letter, so those chunks
# arrive as graph context alongside it.
LOCAL_RETRIEVAL_QUERY = """
WITH node, score
OPTIONAL MATCH (node)-[:FROM_CHUNK]-(e:__Entity__)-[rel]-(nb:__Entity__)-[:FROM_CHUNK]-(other:Chunk)
WHERE other <> node
WITH node, score,
     collect(DISTINCT coalesce(e.name, e.id))[..12] AS entities,
     collect(DISTINCT other.text)[..6] AS linked_chunks
RETURN node.text AS text,
       score,
       entities,
       linked_chunks
"""


def ask_local(question: str) -> str:
    """Answer from an entity neighbourhood plus the underlying text units.

    Closer to conventional retrieval, and the right mode for a question about
    one specific thing rather than about the corpus as a whole.
    """
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.retrievers import VectorCypherRetriever

    driver = get_driver()
    try:
        retriever = VectorCypherRetriever(
            driver=driver,
            index_name=VECTOR_INDEX,
            retrieval_query=LOCAL_RETRIEVAL_QUERY,
            embedder=get_embedder(),
        )
        rag = GraphRAG(retriever=retriever, llm=get_llm())
        result = rag.search(query_text=question, retriever_config={"top_k": 4})
        return result.answer
    finally:
        driver.close()


# ---- doctor --------------------------------------------------------------------
def doctor() -> None:
    """Check the setup without spending anything."""
    print("=== doctor: configuration check, no model calls ===\n")

    print(f"  corpus documents      : {len(list(CORPUS_DIR.glob('*.txt')))}")

    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-your"):
        print(f"  OpenAI API key        : present ({key[:6]}...{key[-4:]})")
    else:
        print("  OpenAI API key        : MISSING - copy .env.example to .env and add it")

    try:
        import neo4j_graphrag
        print(f"  neo4j-graphrag        : {getattr(neo4j_graphrag, '__version__', 'unknown')}")
    except ImportError as error:
        print(f"  neo4j-graphrag        : NOT INSTALLED ({error})")
        return

    try:
        driver = get_driver()
    except Exception as error:
        print(f"  bolt driver           : FAILED to construct - {error}")
        return

    try:
        run_cypher(driver, "RETURN 1")
        print(f"  Neo4j reachable       : yes ({NEO4J_URI})")
    except Exception as error:
        print(f"  Neo4j reachable       : NO - {type(error).__name__}: {error}")
        print("                          is the container running? docker ps")
        driver.close()
        return

    try:
        rows = run_cypher(driver, "RETURN gds.version() AS version")
        print(f"  GDS plugin            : {rows[0]['version']}")
    except Exception as error:
        print(f"  GDS plugin            : NOT AVAILABLE - {error}")
        print("                          Leiden needs it; the setup script installs it")

    rows = run_cypher(
        driver,
        "SHOW INDEXES YIELD name WHERE name = $name RETURN count(*) AS n",
        name=VECTOR_INDEX,
    )
    if rows and rows[0]["n"] > 0:
        print(f"  vector index          : {VECTOR_INDEX} present")
    else:
        print(f"  vector index          : {VECTOR_INDEX} absent (walk creates it)")

    print(f"  already indexed       : {already_indexed(driver)}")
    communities = run_cypher(driver, "MATCH (c:Community) RETURN count(c) AS n")
    print(f"  communities           : {communities[0]['n']}")
    driver.close()


# ---- walk ------------------------------------------------------------------------
async def walk() -> None:
    driver = get_driver()
    try:
        ensure_vector_index(driver)

        if already_indexed(driver):
            print("[extract] existing graph found - skipping extraction (main.py clean to redo)")
        else:
            await extract(driver)
            await resolve_entities(driver)

        if final_level(driver) is None:
            detect_communities(driver)
        else:
            print("[leiden] communities already present - skipping detection")

        summarize_communities(driver)
    finally:
        driver.close()

    inspect_graph()

    print("\n=== GLOBAL search: a question about the corpus as a whole ===")
    print(f"Q: {GLOBAL_QUESTION}\n")
    print(ask_global(GLOBAL_QUESTION))

    print("\n=== LOCAL search: a question about one specific thing ===")
    print(f"Q: {LOCAL_QUESTION}\n")
    print(ask_local(LOCAL_QUESTION))

    print("\n" + "=" * 70)
    print("The corpus contains a noticeboard flyer claiming the owner is allergic")
    print("to grapes - a document that restates the question almost word for word")
    print("and answers it with the wrong mechanism. The supplier notice and the")
    print("owner's letter carry the real reason, and they never mention allergy.")
    print("Whether the answer above says 'allergy' or 'failed inspections' tells")
    print("you whether the graph assembled the chain or the retriever matched")
    print("the question to the nearest-looking text.")


def clean() -> None:
    """Remove everything this exercise wrote to Neo4j. The container and its
    volume survive; only the data goes."""
    driver = get_driver()
    try:
        try:
            run_cypher(driver, f"CALL gds.graph.drop('{GDS_GRAPH}', false)")
        except Exception:
            pass  # GDS not installed or no projection - nothing to drop
        run_cypher(driver, "MATCH (n) DETACH DELETE n")
        run_cypher(driver, f"DROP INDEX {VECTOR_INDEX} IF EXISTS")
        print("[clean] removed all nodes, relationships and the vector index")
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

cat > "tests/test_graphrag.py" << 'TESTS_EOF'
"""Tests split by cost: the free ones always run, the expensive one is opt-in.

Indexing this corpus takes minutes and spends real money, so it is not run
on every invocation. Set GRAPHRAG_RUN_INDEX=1 to include it. Tests that only
need a database (no model calls) are skipped automatically when Neo4j is not
reachable, so the suite stays green on a machine without docker running.
"""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="no API key set",
)
requires_index = pytest.mark.skipif(
    os.getenv("GRAPHRAG_RUN_INDEX") != "1",
    reason="set GRAPHRAG_RUN_INDEX=1 to run the indexing test (costs money)",
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
    reason="Neo4j not reachable - start the container first",
)


def test_corpus_is_present_and_includes_the_distractor():
    documents = list(main.CORPUS_DIR.glob("*.txt"))
    assert len(documents) >= 8

    flyer = (main.CORPUS_DIR / "grape_rumour_flyer.txt").read_text().lower()
    assert "allergic" in flyer, "the distractor must offer a wrong mechanism"

    truth = (main.CORPUS_DIR / "supplier_notice.txt").read_text().lower()
    assert "inspection" in truth
    assert "allerg" not in truth, "the true documents must never mention allergy"


def test_the_packages_expose_what_this_exercise_calls():
    """Guard against a version bump renaming things out from under us.

    The Microsoft version needed this for graphrag.api; the Neo4j version
    needs it for the handful of neo4j-graphrag classes main.py imports.
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings          # noqa: F401
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.generation import GraphRAG                  # noqa: F401
    from neo4j_graphrag.llm import OpenAILLM                        # noqa: F401
    from neo4j_graphrag.retrievers import VectorCypherRetriever     # noqa: F401

    # The schema parameter has been renamed across releases; whichever name
    # the installed version uses, main.py must be able to find one of them.
    picked = main.pick_param_name(SimpleKGPipeline.__init__, ["Person"], "node_types", "entities")
    assert picked, "SimpleKGPipeline accepts neither node_types nor entities in this version"


def test_prompts_carry_the_ethos():
    """The whitepaper claims stand on these prompts; keep them honest."""
    assert "{members}" in main.COMMUNITY_SUMMARY_PROMPT
    assert "{triples}" in main.COMMUNITY_SUMMARY_PROMPT
    assert "{reports}" in main.GLOBAL_SYNTHESIS_PROMPT
    assert "{question}" in main.GLOBAL_SYNTHESIS_PROMPT


@requires_neo4j
def test_neo4j_answers_and_gds_is_installed():
    driver = main.get_driver()
    try:
        rows = main.run_cypher(driver, "RETURN gds.version() AS version")
        assert rows and rows[0]["version"], "GDS plugin missing - Leiden cannot run"
    finally:
        driver.close()


@requires_neo4j
def test_inspect_graph_is_safe_with_no_index(capsys):
    main.inspect_graph()
    assert capsys.readouterr().out


@requires_key
@requires_neo4j
@requires_index
@pytest.mark.asyncio
async def test_index_builds_and_both_query_modes_answer():
    driver = main.get_driver()
    try:
        main.ensure_vector_index(driver)
        if not main.already_indexed(driver):
            await main.extract(driver)
            await main.resolve_entities(driver)
        assert main.already_indexed(driver)

        if main.final_level(driver) is None:
            main.detect_communities(driver)
        assert main.final_level(driver) is not None, "Leiden found nothing"

        main.summarize_communities(driver)
        reports = main.run_cypher(
            driver, "MATCH (c:Community) WHERE c.summary IS NOT NULL RETURN count(c) AS n"
        )
        assert reports[0]["n"] > 0
    finally:
        driver.close()

    answer = main.ask_global(main.GLOBAL_QUESTION).lower()
    assert any(name in answer for name in ("ingrid", "kwon", "tendai"))
TESTS_EOF


note "corpus files   : $(ls corpus | wc -l)"

## Neo4j container
if [ "$SKIP_DOCKER" != "1" ]; then
    say "Neo4j container ($NEO4J_IMAGE, GDS plugin, ports $NEO4J_HTTP_PORT/$NEO4J_BOLT_PORT)"

    if docker ps -a --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
        if docker ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
            note "container already running"
        else
            note "container exists but stopped - starting it"
            docker start "$NEO4J_CONTAINER" >/dev/null
        fi
    else
        note "creating container (data persists in volume $NEO4J_VOLUME)"
        # NEO4J_PLUGINS makes the image download and enable Graph Data Science
        # at first boot. The unrestricted line is belt-and-braces: newer
        # images set it when the plugin installs, older ones did not.
        docker run -d \
            --name "$NEO4J_CONTAINER" \
            -p "$NEO4J_HTTP_PORT:7474" \
            -p "$NEO4J_BOLT_PORT:7687" \
            -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" \
            -e NEO4J_PLUGINS='["graph-data-science"]' \
            -e NEO4J_dbms_security_procedures_unrestricted='gds.*' \
            -v "$NEO4J_VOLUME:/data" \
            "$NEO4J_IMAGE" >/dev/null
    fi

    # Wait for bolt to answer. First boot downloads the GDS jar, so allow
    # a couple of minutes before declaring failure.
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

    # Verify the GDS plugin actually loaded - Leiden depends on it.
    if docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
            "RETURN gds.version()" >/dev/null 2>&1; then
        gds_version="$(docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain 'RETURN gds.version()' 2>/dev/null | tail -1)"
        note "GDS plugin  : ok ($gds_version)"
    else
        echo "WARNING: GDS did not answer. The plugin downloads on first boot and" >&2
        echo "needs network access from the container. Check: docker logs $NEO4J_CONTAINER" >&2
        echo "Leiden (main.py walk) will fail until this is resolved." >&2
    fi
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
note "  2. uv run python main.py walk                (indexes, ~cents)"
note "  3. uv run python main.py graph               (inspect, free)"
note "  4. browse http://localhost:$NEO4J_HTTP_PORT  (login neo4j / your password)"
note ""
note "Container management:"
note "  stop:          docker stop $NEO4J_CONTAINER"
note "  start again:   docker start $NEO4J_CONTAINER   (or rerun this script)"
note "  full rebuild:  docker rm -f $NEO4J_CONTAINER && docker volume rm $NEO4J_VOLUME"
