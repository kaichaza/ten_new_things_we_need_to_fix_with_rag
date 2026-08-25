"""The structural test uses random vectors and needs NO API key: the index
build, disk layout and recall measurement are all real. The corpus test adds
real embeddings when a key is present."""
import os

import numpy as np
import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def test_index_builds_on_disk_and_recall_is_high_on_random_data():
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((64, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    main.build_index(vectors)
    # The index must exist as FILES on disk - that is the entire point.
    files = list(main.INDEX_DIR.iterdir())
    assert len(files) >= 2 and sum(f.stat().st_size for f in files) > 0

    queries = vectors[:8] + rng.standard_normal((8, 32)).astype(np.float32) * 0.01
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    recall = main.recall_at_k(vectors, queries, k=3)
    assert recall >= 0.9, f"recall@3 was {recall}"


@requires_key
def test_corpus_query_finds_the_banana_price():
    vectors = main.embed(main.SENTENCES)
    main.build_index(vectors)
    index = main.open_index()
    qvec = main.embed(["how much do bananas cost"])[0]
    ids = main.ann_search(index, qvec, k=2)
    assert any("Bananas cost 2" in main.SENTENCES[i] for i in ids)
