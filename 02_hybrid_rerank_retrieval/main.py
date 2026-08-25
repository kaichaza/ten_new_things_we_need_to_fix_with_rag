"""Similarity is not relevance: hybrid retrieval with LLM reranking, on OpenAI.

Dense vectors are a similarity instrument, and similarity is a proxy that
fails in three systematic places enterprises care about most:

  - EXACT IDENTIFIERS - 'INV-2041' embeds to almost nothing distinctive; a
    keyword index matches it exactly.
  - NEGATION - 'we do not sell grapes' and a chat ABOUT grapes look nearly
    identical to an embedding model, because it encodes query and document
    separately and negation barely moves the vector.
  - NUMERALS - '3 per kilogram' carries little geometric weight.

The production-standard fix is layered, and every model here is OpenAI:

  1. BM25 keyword search (rank-bm25) - the lexical leg; pure algorithm,
     no model involved
  2. Dense search (text-embedding-3-small) - the semantic leg
  3. Reciprocal Rank Fusion - merge the two lists without score voodoo
  4. LLM reranking (gpt-4.1-mini) - OpenAI offers no cross-encoder
     endpoint, so reranking is done RankGPT-style: the model reads the
     query and the WHOLE shortlist together and returns a ranking, which is
     what finally distinguishes 'answers it' from 'resembles it'. The
     ranking is validated before use; on invalid output we keep the fused
     order rather than trusting a malformed reply.

The walkthrough runs three adversarial questions through four strategies
and scores each against a labelled gold answer, so the win is a number.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

HERE = Path(__file__).parent

CHEAP_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

# The benchmark: (question, substring the correct passage must contain).
# Each targets one systematic dense-retrieval failure.
BENCHMARK = [
    ("What is on invoice INV-2041?", "INV-2041"),                # exact identifier
    ("Does the shop sell grapes?", "does not sell grapes"),      # negation
    ("What do apples cost per kilogram?", "cost 3 per kilogram"),# numerals/price
]


# ------------------------------------------------------------------ corpus --
def load_passages() -> list[dict]:
    """One passage per paragraph, with provenance."""
    passages = []
    for f in sorted((HERE / "corpus").glob("*.md")):
        for para in f.read_text().split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                passages.append({"source": f.name, "text": para})
    return passages


def tokenise(text: str) -> list[str]:
    """Lowercased word tokens. Production analyzers add stemming and
    identifier-aware tokenisation; this is the honest minimum."""
    return re.findall(r"[a-z0-9-]+", text.lower())


def embed(texts: list[str]) -> np.ndarray:
    """OpenAI embeddings, normalised so dot product equals cosine."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# --------------------------------------------------------------- the index --
class HybridIndex:
    """The three retrieval legs plus the LLM reranker, over one passage list.

    The BM25 statistics and the dense passage matrix are both built once in
    __init__ - index-time artifacts, exactly as in a real system. Only
    queries and reranking happen at request time.
    """

    def __init__(self, passages: list[dict]):
        self.passages = passages
        texts = [p["text"] for p in passages]

        # Lexical leg: BM25 over token statistics. No model, no API call.
        self.bm25 = BM25Okapi([tokenise(t) for t in texts])

        # Semantic leg: one embedding call for the whole corpus.
        self.dense_vecs = embed(texts)

    # -- single-leg searches: each returns passage indices, best first -----
    def bm25_search(self, query: str, k: int = 5) -> list[int]:
        scores = self.bm25.get_scores(tokenise(query))
        return list(np.argsort(-scores)[:k])

    def dense_search(self, query: str, k: int = 5) -> list[int]:
        qvec = embed([query])[0]
        return list(np.argsort(-(self.dense_vecs @ qvec))[:k])

    # -- fusion ------------------------------------------------------------
    def hybrid_search(self, query: str, k: int = 5) -> list[int]:
        """Reciprocal Rank Fusion: rrf(d) = sum over legs of 1/(60 + rank).

        RRF merges ranked lists using only RANK, sidestepping the fact that
        BM25 scores and cosine similarities live on incomparable scales.
        The constant 60 is the standard damping value from the RRF paper.
        """
        rrf: dict[int, float] = {}
        for leg in (self.bm25_search(query, k * 2), self.dense_search(query, k * 2)):
            for rank, idx in enumerate(leg):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        return sorted(rrf, key=rrf.get, reverse=True)[:k]

    # -- reranking ---------------------------------------------------------
    def llm_rerank(self, query: str, candidate_ids: list[int], k: int = 3) -> list[int]:
        """RankGPT-style listwise reranking with gpt-4.1-mini.

        The model sees the query and every shortlisted passage TOGETHER and
        returns a ranking as JSON. That joint reading is what resolves
        negation: 'does not sell grapes' can finally outrank passages that
        merely mention grapes often. The reply is VALIDATED (integers, in
        range, deduplicated); if malformed, we keep the fused order - an
        LLM in the serving path is an untrusted optimiser, same rule as the
        chunking exercise.
        """
        numbered = "\n".join(
            f"[{n}] {self.passages[i]['text']}" for n, i in enumerate(candidate_ids)
        )
        resp = client.chat.completions.create(
            model=CHEAP_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": "You are a search reranker. Rank the numbered passages "
                            "by how well each ANSWERS the query - a passage that "
                            "directly answers it outranks one that merely mentions "
                            "related words. Reply as JSON: "
                            "{\"ranking\": [best_passage_number, next, ...]} "
                            "covering every passage number exactly once."},
                {"role": "user", "content": f"Query: {query}\n\nPassages:\n{numbered}"},
            ],
        )
        try:
            ranking = json.loads(resp.choices[0].message.content)["ranking"]
            seen: list[int] = []
            for n in ranking:
                if isinstance(n, int) and 0 <= n < len(candidate_ids) and n not in seen:
                    seen.append(n)
            if len(seen) == len(candidate_ids):
                return [candidate_ids[n] for n in seen[:k]]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        print("[rerank] invalid ranking from the model - keeping fused order")
        return candidate_ids[:k]

    def hybrid_rerank_search(self, query: str, k: int = 3) -> list[int]:
        """The full production shape: cheap recall wide, expensive precision
        narrow - fuse a broad shortlist, rerank it, return the top."""
        return self.llm_rerank(query, self.hybrid_search(query, k=8), k=k)


# -------------------------------------------------------------------- walk --
def evaluate(index: HybridIndex) -> dict[str, int]:
    """Top-1 hits per strategy over the benchmark - the remembered number."""
    strategies = {
        "dense_only": lambda q: index.dense_search(q, 3),
        "bm25_only": lambda q: index.bm25_search(q, 3),
        "hybrid_rrf": lambda q: index.hybrid_search(q, 3),
        "hybrid_rerank": lambda q: index.hybrid_rerank_search(q, 3),
    }
    wins = {name: 0 for name in strategies}
    for question, expected in BENCHMARK:
        for name, search in strategies.items():
            top = index.passages[search(question)[0]]["text"]
            if expected in top:
                wins[name] += 1
    return wins


def walk() -> None:
    passages = load_passages()
    print(f"=== Index built over {len(passages)} passages: BM25 + OpenAI embeddings ===")
    index = HybridIndex(passages)

    for question, expected in BENCHMARK:
        print(f"\nQ: {question}   (correct passage contains: '{expected}')")
        for name, ids in [
            ("dense ", index.dense_search(question, 3)),
            ("bm25  ", index.bm25_search(question, 3)),
            ("hybrid", index.hybrid_search(question, 3)),
            ("rerank", index.hybrid_rerank_search(question, 3)),
        ]:
            top = index.passages[ids[0]]
            hit = "HIT " if expected in top["text"] else "miss"
            print(f"  [{name}] {hit} top-1 from {top['source']}: {top['text'][:80]}")

    print("\n=== Top-1 score out of 3 questions ===")
    for name, score in evaluate(index).items():
        print(f"  {name:14s} {score}/3")

    print("\nRead the grape question closely: the customer-notes passage MENTIONS "
          "grapes constantly, and the policy passage ANSWERS the question. Cosine "
          "cannot tell those apart reliably; the reranker, reading query and "
          "shortlist together, can. Wide cheap recall, narrow expensive precision - "
          "that division of labour is the whole pattern.")


if __name__ == "__main__":
    walk()
