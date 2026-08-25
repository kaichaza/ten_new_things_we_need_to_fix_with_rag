# graphrag_fruit_graph

**Theme.** Top-k retrieval is structurally local. "What connects the regular
customers?" is a property of the whole corpus, and a chunk retriever answers
it from an arbitrary sample. GraphRAG extracts an entity graph at index time,
clusters it with hierarchical Leiden, writes a report for every community,
and answers global questions from summaries of summaries.

**Stack.** graphrag 3.1.1 called through its Python API, pandas for reading
the parquet artifacts back, OpenAI `gpt-4.1-mini` and `text-embedding-3-small`,
uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py doctor    # config check, no model calls, free
    uv run python main.py walk      # indexes for real: minutes, and cents
    uv run python main.py graph     # inspect an existing index, free
    uv run pytest -q                # free tests; see below for the paid one
    uv run python main.py clean     # remove generated state

    docker build -t graphrag_fruit .
    docker run --rm graphrag_fruit                                    # doctor
    docker run --rm --env-file .env graphrag_fruit \
        uv run python main.py walk

## Two things this version does differently

**No CLI, no subprocess.** `main.py` imports `graphrag.api` and awaits
`build_index`, `global_search` and `local_search` directly.

**settings.yaml is committed, not scaffolded.** `graphrag init` prompts
interactively - invisible when output is captured, so it hangs - refuses to
run when a `.env` exists, and its `--force` flag overwrites `.env` and
destroys the API key. Every value it generated was overwritten immediately
anyway, so the file lives under version control instead.

Query calls are assembled by introspecting the real function signatures at
runtime. GraphRAG's API has moved parameter names between releases; passing
only what a function accepts survives that, and anything dropped is reported
rather than silently ignored.

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

`doctor` and `graph` are free. `walk` indexes the corpus with real extraction
and summarisation calls: a few minutes and a few cents. The walkthrough skips
re-indexing when `output/` already holds parquet, so subsequent runs only pay
for the two queries.

The expensive test is opt-in:

    GRAPHRAG_RUN_INDEX=1 uv run pytest -q
