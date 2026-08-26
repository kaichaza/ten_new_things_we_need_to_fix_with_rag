# Pitfall Practice: thirteen standalone exercises

Thirteen self-contained Python exercises that reinforce, through working code,
key Gen AI themes I have observed as the technology matures: index freshness and replay, hybrid retrieval and reranking, graph
retrieval, evaluation, PII erasure, injection defence, routing and cost, bounded
agents, MCP tool trust, end-to-end observability, chunking, disk-based vector
search, and bi-temporal (time-aware) retrieval.

Every folder stands alone: its own `README.md`, `pyproject.toml`, `corpus/`
documents, a commented CLI (`main.py`) and a pytest suite that exercises the
real application (no monkeypatching; tests that need an API key or a database
skip cleanly when either is absent). Most folders also ship a `Dockerfile` and
`setup.sh`; the two graph exercises (`03`, `13`) are instead built by
root-level setup scripts, because their state lives in long-lived Neo4j
containers rather than in files.

## Numbering

Folders `01` to `10` are numbered to match the ten-problem one-pager, so
`07_model_routing_litellm` is the exercise behind problem 7 on that page.
Folders `11` to `13` are the three the one-pager does not cover, parked at
the end so they sort last. `folder_numbering.txt` records the mapping, the
renumbering history and the whitepaper pitfall each exercise leads with.

## Shared conventions

- **Python 3.12 via uv** everywhere, except `12_disk_ann_diskannpy`, which uses
  uv-managed Python 3.11 because diskannpy 0.7.0 ships no 3.12 wheel and no
  source distribution (its README explains).
- **Models**: OpenAI `gpt-4.1-mini` as the cheap tier and `gpt-4.1` as the
  strong tier, `text-embedding-3-small` for embeddings; all overridable via
  environment variables `MODEL_CHEAP`, `MODEL_STRONG`, `EMBED_MODEL`.
  Two exceptions: `09_mcp_tools_and_scan` makes no model calls at all, and
  `06_injection_defence_llm_guard` runs a small bundled HuggingFace classifier
  locally alongside its OpenAI generation step.
- **Secrets**: copy `.env.example` to `.env` in each folder and add your keys.
  Nothing is read from stdin; every walkthrough runs unattended.
- **Services**: `01` starts Postgres via its own docker compose file. `03`
  and `13` each own a long-lived Neo4j 5 container created by their setup
  scripts - `neo4j-fruit-graph` (GDS + APOC, ports 7474/7687) and
  `neo4j-temporal-graph` (APOC, ports 7475/7688) - one container each
  because Neo4j Community runs a single database per instance. The suites
  and setup scripts only ever STOP these containers, never remove them or
  their volumes: the volumes hold indexes that cost real model calls to
  build.
- **Scenario**: a small fruit shop selling apples, bananas and pears (never
  grapes), with customers Ingrid, Kwon and Tendai on record, operated for the
  current customer, Matt. Prices are small integers on purpose.

## Running any exercise

    cd <exercise>
    ./setup.sh                 # local: uv python install + uv sync + .env scaffold
    uv run python main.py walk # the guided walkthrough
    uv run pytest -q           # the test suite

    # or fully containerised (all installs happen inside the Dockerfile):
    docker build -t <exercise> .
    docker run --rm --env-file .env <exercise>

`01_stale_index_replay` additionally uses Docker Compose for its Postgres
service. For `03` and `13`, run the root-level setup script once
(`./setup_fruit_graphrag_neo4j.sh`, `./setup_temporal_graphiti.sh`) - each
is idempotent and brings up its exercise's container, files and
dependencies in one go.

To test everything at once, two suites run from the root:

    ./run_level_00_all_basic_test_suite.sh     # free tier: every pytest suite,
                                               # services managed, no paid gates
    ./run_level_01_integration_test_suite.sh   # paid tier: full walkthroughs with
                                               # real model calls plus the gated
                                               # expensive tests; asks before
                                               # spending; transcript lands in
                                               # logs/integration_run_<stamp>.log

## The thirteen

| Folder | Library focus | Theme |
|---|---|---|
| 01_stale_index_replay | lancedb, postgres | Stale indexes; everything downstream is a rebuildable view |
| 02_hybrid_rerank_retrieval | rank-bm25, openai | BM25 and dense legs fused by RRF, then a validated listwise rerank |
| 03_graphrag_fruit_graph | neo4j-graphrag, graphdatascience | Corpus-level questions via Leiden communities on Neo4j |
| 04_rag_eval_ragas | ragas | Faithfulness as a regression metric |
| 05_pii_erasure_replay | lancedb | Right-to-erasure by redact-and-replay |
| 06_injection_defence_llm_guard | llm-guard | The corpus as an attack surface |
| 07_model_routing_litellm | litellm | Cost tiers, escalation, token spend as COGS |
| 08_bounded_agent_langgraph | langgraph | Budgets, checkpoints, a gate before consequential actions |
| 09_mcp_tools_and_scan | mcp, mcp-scan | Tool-description pinning and drift detection |
| 10_observability_langfuse_otel | langfuse, opentelemetry-sdk | One trace ID from question to answer |
| 11_semantic_chunking_two_llms | openai | A cheaper second LLM proposes chunk boundaries |
| 12_disk_ann_diskannpy | diskannpy | Disk-first ANN vs memory-priced serving |
| 13_temporal_bitemporal_graphiti | graphiti-core | Bi-temporal facts; three dates, three correct answers |

Exercises 11 to 13 are not on the ten-problem one-pager. Chunking was folded
into the retrieval precision problem and disk-first ANN into the cost problem,
on the grounds that a client hears each pair as one complaint rather than two;
time-blind indexing (whitepaper pitfall 6) was added after the page was cut.
All three exercises still stand on their own.

## Other files in this repository

- `exercise_summary.md` — what each exercise does, its key libraries, how it
  calls the model, and a representative code extract.
- `folder_numbering.txt` — the numbering scheme, why it is what it is, the
  renumbering history, and the whitepaper pitfall each exercise leads with.
- `print_project_files.sh` — flattens the repository into a single
  `project_contents.txt` for review. Binary files, lock files and caches are
  listed by name only, and `.env` contents are withheld.
- `renumber_to_deck.sh` — the two-pass rename that produced the current
  numbering. Kept as a record; it has already been run.
- `setup_fruit_graphrag_neo4j.sh` / `setup_temporal_graphiti.sh` — idempotent
  one-shot builders for the two Neo4j-backed exercises (`03`, `13`).
- `run_level_00_all_basic_test_suite.sh` — the free test sweep across all
  thirteen exercises; transcript in `test_run.log`.
- `run_level_01_integration_test_suite.sh` — the paid integration suite:
  every walkthrough end to end with real model calls, plus the gated
  expensive tests; timestamped transcripts under `logs/`.
