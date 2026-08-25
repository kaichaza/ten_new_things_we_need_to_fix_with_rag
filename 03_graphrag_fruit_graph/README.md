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
