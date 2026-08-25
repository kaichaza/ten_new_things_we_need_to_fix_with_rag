"""Faithfulness as a regression metric: Ragas scoring a real RAG pipeline.

Retrieval reduces hallucination; it does not remove it. The model can exceed,
contradict or mis-cite its sources while the citations make it LOOK
trustworthy. The countermeasure is to make 'does the answer follow from the
retrieved passages' a NUMBER, computed on every pipeline change, exactly like
a regression test. This exercise runs a small real pipeline (retrieve ->
generate), scores it with Ragas (faithfulness, response relevancy, context
precision), and then scores a deliberately unfaithful answer ('the shop has
grapes on offer') against the same contexts to show the metric catching it.

The judge is itself an LLM, so scores are estimates with an error rate:
mature teams calibrate them against a periodically human-audited sample.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
# Must come before any ragas import: ragas loads a langchain-community
# module that no longer exists. See _compat.py for the full explanation.
import _compat  # noqa: F401  (import for side effect, before ragas)

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

load_dotenv()

HERE = Path(__file__).parent
PARAGRAPHS = [p.strip() for p in (HERE / "corpus" / "shop_facts.md").read_text().split("\n\n") if p.strip()]

CHAT_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()

QUESTIONS = [
    "What do bananas cost per kilogram?",
    "Does the shop sell grapes?",
    "Which customer buys apples for baking?",
]


# ---------------------------------------------------------- the real pipeline --
def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


PARA_VECS = None  # computed lazily so importing this module makes no API call


def retrieve(question: str, k: int = 2) -> list[str]:
    global PARA_VECS
    if PARA_VECS is None:
        PARA_VECS = embed(PARAGRAPHS)
    qvec = embed([question])[0]
    order = np.argsort(-(PARA_VECS @ qvec))
    return [PARAGRAPHS[i] for i in order[:k]]


def generate(question: str, contexts: list[str]) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer only from the provided context, briefly."},
            {"role": "user", "content": "Context:\n" + "\n---\n".join(contexts) + f"\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------ evaluation --
def build_dataset(include_sabotage: bool = True) -> EvaluationDataset:
    """Run the real pipeline for each question; optionally append the sabotage
    sample - an answer that contradicts its own contexts - as the canary."""
    samples = []
    for q in QUESTIONS:
        contexts = retrieve(q)
        samples.append(SingleTurnSample(
            user_input=q, retrieved_contexts=contexts, response=generate(q, contexts)
        ))
    if include_sabotage:
        contexts = retrieve("Does the shop sell grapes?")
        samples.append(SingleTurnSample(
            user_input="Does the shop sell grapes?",
            retrieved_contexts=contexts,
            response="Yes - the shop has grapes on special offer this week at 1 per kilogram.",
        ))
    return EvaluationDataset(samples=samples)


def score(dataset: EvaluationDataset):
    """Ragas needs a judge LLM and embeddings; we wrap the same OpenAI models."""
    judge = LangchainLLMWrapper(ChatOpenAI(model=CHAT_MODEL))
    embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))
    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=judge,
        embeddings=embedder,
    )


def walk() -> None:
    print("=== Running the real pipeline and scoring it with Ragas ===")
    dataset = build_dataset(include_sabotage=True)
    result = score(dataset)
    df = result.to_pandas()

    for _, row in df.iterrows():
        print(f"\nQ: {row['user_input']}")
        print(f"A: {str(row['response'])[:100]}")
        print(f"   faithfulness={row['faithfulness']:.2f}  "
              f"relevancy={row['answer_relevancy']:.2f}  "
              f"context_precision={row['llm_context_precision_without_reference']:.2f}")

    print("\nThe last sample is the deliberate sabotage: an answer contradicting "
          "its own contexts. Its faithfulness score should sit far below the "
          "honest samples - that gap is what you regression-test on every "
          "pipeline change (new chunking, new embedder, new model version).")


if __name__ == "__main__":
    walk()
