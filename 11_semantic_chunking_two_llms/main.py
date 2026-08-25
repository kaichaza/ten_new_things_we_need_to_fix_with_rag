"""Semantic chunking with a cheaper second LLM, versus naive fixed-size slicing.

Fixed-size chunking severs sentences from their subjects and policies from
their reasons; the retriever then returns fragments never meant to stand
alone. This exercise uses an inexpensive model (gpt-4.1-mini) as a CHUNKER -
it reads the document once and proposes topic-coherent boundaries - and then
compares retrieval quality against naive 240-character windows on the same
questions. The chunker's output is validated: chunks must reassemble into the
original text, so the cheap model can only choose boundaries, never rewrite
content.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).parent
DOC = (HERE / "corpus" / "shop_story.md").read_text()

CHEAP_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

QUESTIONS = [
    ("Why does the shop refuse to sell grapes?", "grape"),
    ("Who founded the shop and when?", "founded"),
    ("What is the only exception to fixed pricing?", "half price"),
]


# ---------------------------------------------------------------- chunkers --
def naive_chunks(text: str, size: int = 240) -> list[str]:
    """Fixed-size character windows: cheap, instant, and meaning-blind.

    Note how these routinely cut mid-sentence - that is the point.
    """
    return [text[i : i + size] for i in range(0, len(text), size)]


def _normalise(s: str) -> str:
    """Whitespace-insensitive form used to verify chunk reassembly."""
    return re.sub(r"\s+", " ", s).strip()


def _paragraph_chunks(text: str) -> list[str]:
    """Split on blank lines while KEEPING the separators.

    The naive version of this (text.split("\\n\\n")) throws the separators
    away, so "".join(chunks) no longer reproduces the document: two
    paragraphs run together with nothing between them. That breaks the same
    reassembly guarantee the verification above exists to enforce, which
    makes the safe fallback quietly unsafe. Keeping each separator attached
    to the paragraph it followed means "".join(chunks) == text exactly.
    """
    parts = re.split(r"(\n[ \t]*\n)", text)
    chunks: list[str] = []
    pending = ""
    for index in range(0, len(parts), 2):
        body = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        piece = pending + body + separator
        pending = ""
        if piece.strip():
            chunks.append(piece)
        elif chunks:
            # A blank run with no text of its own belongs to the chunk before it.
            chunks[-1] += piece
        else:
            # Leading whitespace, held over until there is a chunk to join.
            pending = piece
    if pending:
        if chunks:
            chunks[-1] += pending
        else:
            chunks = [pending]
    return chunks


def semantic_chunks(text: str) -> list[str]:
    """Ask the CHEAP model for topic-coherent boundaries, then verify.

    The contract: return the document as a JSON list of verbatim, contiguous
    chunks. We verify that the chunks reassemble into the original text
    (whitespace-insensitively); if the model paraphrased or dropped anything,
    we refuse its output and fall back to paragraph splitting. The chunker is
    an untrusted optimiser, and validation is what makes it safe to use.
    """
    resp = client.chat.completions.create(
        model=CHEAP_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": "Split the user's document into 4-6 topic-coherent chunks. "
                        "Each chunk must be a VERBATIM contiguous span of the document, "
                        "in order, covering the whole document with nothing added or "
                        "removed. Reply as JSON: {\"chunks\": [\"...\", \"...\"]}"},
            {"role": "user", "content": text},
        ],
    )
    chunks = json.loads(resp.choices[0].message.content)["chunks"]
    if _normalise("".join(chunks)) == _normalise(text):
        return chunks
    # Paraphrase or loss detected: refuse the LLM output, fall back safely.
    # The fallback must honour the same reassembly contract the model just
    # failed, otherwise refusing the model buys nothing.
    print("[chunker] verification FAILED - falling back to paragraph split")
    return _paragraph_chunks(text)


# --------------------------------------------------------------- retrieval --
def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def top1(question: str, chunks: list[str], chunk_vecs: np.ndarray) -> tuple[str, float]:
    """Cosine top-1 retrieval (vectors are pre-normalised, so dot = cosine)."""
    qvec = embed([question])[0]
    scores = chunk_vecs @ qvec
    best = int(np.argmax(scores))
    return chunks[best], float(scores[best])


# -------------------------------------------------------------------- walk --
def walk() -> None:
    print("=== Chunking the same document two ways ===")
    naive = naive_chunks(DOC)
    semantic = semantic_chunks(DOC)
    print(f"naive: {len(naive)} fixed windows | semantic: {len(semantic)} LLM-chosen chunks")
    print("\nA naive cut, mid-thought:")
    print(f"  ...{naive[1][:110]}...")
    print("\nA semantic chunk, whole thought:")
    print(f"  {semantic[min(2, len(semantic)-1)][:160]}...")

    naive_vecs = embed(naive)
    sem_vecs = embed(semantic)

    print("\n=== Retrieval head-to-head (top-1 chunk per question) ===")
    for question, expected in QUESTIONS:
        n_chunk, n_score = top1(question, naive, naive_vecs)
        s_chunk, s_score = top1(question, semantic, sem_vecs)
        n_hit = expected.lower() in n_chunk.lower()
        s_hit = expected.lower() in s_chunk.lower()
        print(f"\nQ: {question}")
        print(f"  naive    score={n_score:.3f} contains '{expected}': {n_hit}")
        print(f"  semantic score={s_score:.3f} contains '{expected}': {s_hit}")

    print("\nThe cheap model spent a fraction of a cent to choose boundaries; the "
          "expensive part of the pipeline (retrieval quality) is where it pays back. "
          "Validation kept it honest: it may only cut, never rewrite.")


if __name__ == "__main__":
    walk()
