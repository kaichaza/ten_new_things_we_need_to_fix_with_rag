# graphrag_fruit_graph

**Theme.** Top-k retrieval is structurally local. "What themes run across the
customers?" is a property of the whole corpus, and a chunk retriever answers
it from an arbitrary sample. Microsoft GraphRAG's fix: extract an entity graph
at index time, cluster it with the hierarchical Leiden algorithm,
pre-summarise each community, and answer global questions from summaries of
summaries while local questions use the entity neighbourhood.

**Stack.** graphrag 3.1.1 (pinned; driven via its CLI from `main.py`),
pandas for reading back the parquet artifacts, OpenAI `gpt-4.1-mini` for
extraction and synthesis, `text-embedding-3-small`, uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk      # first run indexes: a few cents, a few minutes
    uv run python main.py graph     # inspect an existing index, no model calls
    uv run pytest -q                # end-to-end plus graph-structure tests
    uv run python main.py clean     # remove generated state for a fresh run

    docker build -t graphrag_fruit .
    docker run --rm --env-file .env graphrag_fruit

## What the walkthrough shows

After indexing, `inspect_graph()` reads the parquet artifacts back and prints:

- **Entities and relationships** extracted from the corpus, with the most
  connected entities ranked by degree - degree is counted from the
  relationships table rather than read from a column, because graphrag does
  not emit it consistently across versions.
- **The Leiden community hierarchy**: how many communities exist at each
  level, and how many entities sit in each. Level 0 is the coarsest
  partition; deeper levels subdivide it as cluster affinity narrows.
- **One community report in full** - an LLM-written summary of a cluster,
  generated once at index time. This is the artifact every global query is
  actually built from.

That last one is the point of the exercise. A global answer is not retrieved,
it is synthesised from summaries that already existed before the question was
asked.

## Notes

- graphrag reads its API key from `GRAPHRAG_API_KEY`; `main.py` mirrors your
  `OPENAI_API_KEY` into it, so `.env` needs only the one line.
- `main.py` patches only the model names inside the scaffolded
  `settings.yaml`, leaving the structure to the pinned graphrag version. If
  you bump the pin, re-run `clean` and let it re-scaffold.
- Output tables are located by filename suffix, not exact name: older graphrag
  releases prefixed everything with `create_final_`, and column names vary
  too, so the inspection code checks which columns exist before using them.
- Indexing artifacts land in `output/` as parquet; the walkthrough skips
  re-indexing when they exist.
