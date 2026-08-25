# stale_index_replay

**Theme.** A vector index is a derived view of operational data. Batch-built
indexes go stale the moment the source changes, and the answers they ground
are then perfectly cited and wrong. The correct fix is replay - rebuild
derived state from the canonical source - never hand-editing the index. The
storage layer's versioning turns "what did the system believe last Tuesday?"
into a checkout instead of an argument.

**Stack.** Postgres 16 (source of truth, via Docker Compose), LanceDB
(derived index with version history), OpenAI `text-embedding-3-small` +
`gpt-4.1-mini`, uv on Python 3.12.

## Run it

    ./setup.sh                    # uv + deps + .env scaffold; add your key
    docker compose up -d db       # start Postgres
    uv run python main.py walk    # the guided walkthrough
    uv run pytest -q              # tests against the real app

Fully containerised alternative (all installs inside the image):

    docker compose up --build app

## What the walkthrough shows

1. Seed prices in Postgres (apples 3, bananas 2, pears 4) and build the index.
2. Ask the price of apples: correct.
3. Update apples to 5 in Postgres only. Ask again: the answer is grounded in
   the index and **wrong** - the stale-index failure, reproduced on demand.
4. Rebuild by replaying from source; the answer converges.
5. LanceDB `list_versions` / `checkout`: read what the old index believed.

In production, step 4 is triggered by change data capture from the database's
transaction log (Debezium or native logical replication), not by a human
remembering to run a script.
