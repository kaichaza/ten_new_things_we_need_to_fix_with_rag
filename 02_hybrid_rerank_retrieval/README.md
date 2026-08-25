# 12_hybrid_rerank_retrieval

**Theme.** Similarity is not relevance. Dense vectors systematically miss
exact identifiers (an invoice number embeds to almost nothing distinctive),
negation ("we do not sell grapes" versus enthusiastic chatter about grapes),
and numerals - precisely the precision queries enterprises ask most. The
production-standard fix is layered: a lexical leg (BM25) and a semantic leg
merged with Reciprocal Rank Fusion, then a reranking layer that reads query
and passages *together* - wide cheap recall, narrow expensive precision.

**Stack - OpenAI models throughout, per the series convention.**
`text-embedding-3-small` for the dense leg; `gpt-4.1-mini` as a
RankGPT-style listwise reranker, because OpenAI offers no cross-encoder
endpoint - the shortlist goes to the model with the query and comes back as
a validated JSON ranking (malformed output falls back to the fused order).
rank-bm25 provides the lexical leg; it is a pure algorithm, not a model.
The self-hosted alternative for the reranking layer is a cross-encoder such
as MS MARCO MiniLM via sentence-transformers - worth naming in conversation
as the lower-latency, fixed-cost option. uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q          # the BM25 test runs keyless; the rest skip without a key

    docker build -t hybrid_rerank .
    docker run --rm --env-file .env hybrid_rerank

## What to look for

The corpus is booby-trapped: invoices with real identifiers, a policy
passage that *answers* the grape question, and a customer-notes passage that
*mentions* grapes far more often. Three benchmark questions run through four
strategies (dense-only, BM25-only, hybrid RRF, hybrid+rerank) with a top-1
score out of 3 printed at the end - hybrid+rerank should score 3/3, and the
single legs should each drop at least one. The tests assert the guarantees
of the design (the lexical leg catches `INV-2041`; the reranked pipeline
answers everything; fusion never does worse than either leg) rather than
asserting that dense fails, so they survive embedder upgrades.

Two details worth remembering: RRF fuses by *rank*, not score, because BM25
scores and cosine similarities live on incomparable scales (the constant 60
is the standard damping value from the original paper); and the reranker's
output is validated before use - an LLM in the serving path is an untrusted
optimiser, the same rule as the semantic-chunking exercise.
