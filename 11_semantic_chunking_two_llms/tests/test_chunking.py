"""Real-application tests: the chunker call and retrieval are genuine."""
import os
import re

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def test_naive_chunks_cover_document_but_cut_blindly():
    chunks = main.naive_chunks(main.DOC, size=240)
    # Coverage: rejoining the windows yields the original text exactly.
    assert "".join(chunks) == main.DOC
    # Blindness: at least one boundary falls mid-word (no trailing space/newline).
    assert any(not c.endswith((" ", "\n", ".")) for c in chunks[:-1])


@requires_key
def test_semantic_chunks_are_verbatim_and_coherent():
    chunks = main.semantic_chunks(main.DOC)
    assert 3 <= len(chunks) <= 8
    # The validation contract: nothing added, nothing removed.
    joined = re.sub(r"\s+", " ", "".join(chunks)).strip()
    original = re.sub(r"\s+", " ", main.DOC).strip()
    assert joined == original


@requires_key
def test_semantic_retrieval_finds_the_grape_policy():
    chunks = main.semantic_chunks(main.DOC)
    vecs = main.embed(chunks)
    chunk, _score = main.top1("Why does the shop refuse to sell grapes?", chunks, vecs)
    assert "grape" in chunk.lower()
