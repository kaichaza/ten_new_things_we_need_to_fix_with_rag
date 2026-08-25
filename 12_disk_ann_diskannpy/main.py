"""Disk-first vector search with DiskANN: index files on disk, not RAM prices.

First-generation vector serving held graph indexes in memory for latency, so
cost scaled with corpus size at RAM prices - and teams quietly embedded less
to afford the bill. The disk-first correction (the DiskANN lineage, the Lance
format) keeps the index on NVMe and pays a few milliseconds for an order of
magnitude on cost. This exercise builds a real DiskANN index over the corpus
embeddings, lists the index FILES and their sizes (the point: this lives on
disk), queries it, and measures recall against exact brute-force search so
the accuracy trade is a number, not a slogan.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import diskannpy
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).parent
INDEX_DIR = HERE / "data" / "diskann_index"
SENTENCES = [
    line.strip()
    for line in (HERE / "corpus" / "shop_sentences.md").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def embed(texts: list[str]) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    # Normalised vectors: cosine similarity becomes a dot product, and the
    # index can use the "mips" (maximum inner product) metric for cosine.
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ------------------------------------------------------------------- index --
def build_index(vectors: np.ndarray) -> None:
    """Build the on-disk index. Parameters are sized for a tiny corpus:

    - complexity: candidate-list size during build (quality knob)
    - graph_degree: edges per node in the proximity graph
    - search_memory_maximum / build_memory_maximum (GB): DiskANN's defining
      levers - hard caps on RAM, with the graph living on disk.
    """
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True)
    diskannpy.build_disk_index(
        data=vectors,
        distance_metric="mips",
        index_directory=str(INDEX_DIR),
        complexity=32,
        graph_degree=16,
        search_memory_maximum=0.01,   # ~10 MB serving RAM budget
        build_memory_maximum=1.0,
        num_threads=2,
        pq_disk_bytes=0,
    )


def open_index() -> diskannpy.StaticDiskIndex:
    """Open the index for serving. num_nodes_to_cache is the small hot set
    kept in memory; everything else is read from disk per query."""
    return diskannpy.StaticDiskIndex(
        index_directory=str(INDEX_DIR),
        num_threads=2,
        num_nodes_to_cache=4,
    )


def ann_search(index: diskannpy.StaticDiskIndex, qvec: np.ndarray, k: int = 3) -> list[int]:
    result = index.search(query=qvec, k_neighbors=k, complexity=32)
    return list(result.identifiers)


def brute_force(vectors: np.ndarray, qvec: np.ndarray, k: int = 3) -> list[int]:
    """Exact search: the ground truth ANN is measured against."""
    return list(np.argsort(-(vectors @ qvec))[:k])


def recall_at_k(vectors: np.ndarray, queries: np.ndarray, k: int = 3) -> float:
    """Fraction of exact top-k the ANN index reproduces, averaged over queries."""
    index = open_index()
    hits = 0
    for qvec in queries:
        approx = set(ann_search(index, qvec, k))
        exact = set(brute_force(vectors, qvec, k))
        hits += len(approx & exact) / k
    return hits / len(queries)


def show_disk_footprint() -> None:
    print(f"[disk] index files in {INDEX_DIR.relative_to(HERE)}:")
    for f in sorted(INDEX_DIR.iterdir()):
        print(f"  {f.name:40s} {f.stat().st_size:>10,} bytes")


# -------------------------------------------------------------------- walk --
def walk() -> None:
    print(f"=== Embedding {len(SENTENCES)} corpus sentences ===")
    vectors = embed(SENTENCES)

    print("\n=== Building the DiskANN index (on disk, tiny RAM budget) ===")
    build_index(vectors)
    show_disk_footprint()

    print("\n=== Query the index from disk ===")
    index = open_index()
    for q in ["how much do bananas cost", "who likes pears the most", "grape policy"]:
        qvec = embed([q])[0]
        ids = ann_search(index, qvec, k=2)
        print(f"\nQ: {q}")
        for i in ids:
            print(f"   -> {SENTENCES[i]}")

    print("\n=== Recall against exact brute force (the honest accuracy number) ===")
    queries = embed(["price of apples", "running club fruit", "half price basket"])
    r = recall_at_k(vectors, queries, k=3)
    print(f"recall@3 = {r:.2f}")
    print("\nThe files above ARE the index; serving RAM was capped near 10 MB. On a "
          "corpus of millions this is the difference between NVMe economics and "
          "RAM economics - paid for with the recall gap you just measured.")


if __name__ == "__main__":
    walk()
