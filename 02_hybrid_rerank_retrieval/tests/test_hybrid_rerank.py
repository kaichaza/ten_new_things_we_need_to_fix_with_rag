"""Tests against the real index and real OpenAI calls. The BM25 leg is a
pure algorithm and tests keyless; everything touching embeddings or the
reranker requires the key and skips cleanly without it.

We assert what the layered design GUARANTEES (the lexical leg catches exact
identifiers; the reranked pipeline answers all three benchmark questions;
fusion never does worse than either leg) rather than asserting that dense
retrieval fails - an embedding-model upgrade could make dense better, and
the tests should survive that.
"""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

PASSAGES = main.load_passages()


def _bm25_only_index():
    """Build ONLY the lexical leg, keyless, by sidestepping __init__'s
    embedding call - the object is real, just partially constructed."""
    index = main.HybridIndex.__new__(main.HybridIndex)
    index.passages = PASSAGES
    index.bm25 = main.BM25Okapi([main.tokenise(p["text"]) for p in PASSAGES])
    return index


def test_bm25_catches_the_exact_identifier_keyless():
    index = _bm25_only_index()
    ids = index.bm25_search("What is on invoice INV-2041?", k=3)
    assert "INV-2041" in index.passages[ids[0]]["text"]


@requires_key
def test_hybrid_keeps_the_identifier_in_its_shortlist():
    index = main.HybridIndex(PASSAGES)
    ids = index.hybrid_search("What is on invoice INV-2041?", k=5)
    assert any("INV-2041" in index.passages[i]["text"] for i in ids)


@requires_key
def test_reranker_resolves_the_negation_question():
    index = main.HybridIndex(PASSAGES)
    ids = index.hybrid_rerank_search("Does the shop sell grapes?", k=3)
    top = index.passages[ids[0]]
    assert "does not sell grapes" in top["text"]
    # And provenance: the answer must come from policy, not customer chatter.
    assert top["source"] == "policy.md"


@requires_key
def test_full_pipeline_answers_every_benchmark_question_at_top1():
    index = main.HybridIndex(PASSAGES)
    for question, expected in main.BENCHMARK:
        ids = index.hybrid_rerank_search(question, k=3)
        assert expected in index.passages[ids[0]]["text"], question


@requires_key
def test_full_pipeline_is_at_least_as_good_as_either_single_leg():
    index = main.HybridIndex(PASSAGES)
    wins = main.evaluate(index)
    assert wins["hybrid_rerank"] >= wins["dense_only"]
    assert wins["hybrid_rerank"] >= wins["bm25_only"]
    assert wins["hybrid_rerank"] == 3
