"""Token spend is cost of goods sold; buy capability at the cheapest adequate tier.

Routing everything through the strongest model pays reasoning rates for
questions a small model answers identically. This exercise runs the same two
questions three ways - cheap-only, strong-only, and a CASCADE (cheap first,
escalate on a self-declared ESCALATE signal) - and prints what each strategy
actually cost, using litellm's per-call cost accounting. litellm also gives
us a single provider-agnostic interface, which is the portability half of the
same lesson.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
CONTEXT = (HERE / "corpus" / "price_list.md").read_text()

CHEAP = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
STRONG = os.getenv("MODEL_STRONG", "gpt-4.1")

EASY_Q = "Which three fruits does the shop sell?"
HARD_Q = ("Matt buys 2 kg of apples and 3 kg of bananas. He can swap exactly one "
          "of those purchases (same kilograms) for a different fruit the shop "
          "sells. Which single swap minimises his total bill, and what is that "
          "minimal total?")

ESCALATE_TOKEN = "ESCALATE"


def call(model: str, question: str, allow_escalation: bool = False) -> tuple[str, float]:
    """One completion via litellm; returns (answer, cost_in_usd).

    litellm.completion is a uniform interface over providers, and
    litellm.completion_cost prices the exact request from its token usage -
    which is how token spend becomes a number on a dashboard instead of a
    surprise on an invoice.
    """
    system = ("Answer strictly from the context. Keep numbers exact.\n"
              f"Context:\n{CONTEXT}")
    if allow_escalation:
        system += (f"\nIf the question needs multi-step arithmetic or comparison of "
                   f"alternatives, reply with exactly the single word {ESCALATE_TOKEN} "
                   f"and nothing else.")
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    cost = litellm.completion_cost(completion_response=resp)
    return answer, cost


def cascade(question: str) -> dict:
    """Cheap first; escalate to the strong tier only when the cheap tier asks.

    The escalation signal here is self-declared by the cheap model. In
    production you would add measured signals (logprob confidence, task
    class, retrieval score) - but the shape is identical: a policy decides
    which tier a request deserves, and most requests deserve the cheap one.
    """
    answer, cost = call(CHEAP, question, allow_escalation=True)
    if answer.strip() == ESCALATE_TOKEN:
        strong_answer, strong_cost = call(STRONG, question)
        return {"model_used": STRONG, "answer": strong_answer,
                "cost": cost + strong_cost, "escalated": True}
    return {"model_used": CHEAP, "answer": answer, "cost": cost, "escalated": False}


def walk() -> None:
    total = {"cheap_only": 0.0, "strong_only": 0.0, "cascade": 0.0}
    for label, q in [("EASY", EASY_Q), ("HARD", HARD_Q)]:
        print(f"\n=== {label}: {q[:70]}... ===" if len(q) > 70 else f"\n=== {label}: {q} ===")

        a, c = call(CHEAP, q)
        total["cheap_only"] += c
        print(f"[cheap-only  {CHEAP}] cost=${c:.6f}\n  {a[:160]}")

        a, c = call(STRONG, q)
        total["strong_only"] += c
        print(f"[strong-only {STRONG}] cost=${c:.6f}\n  {a[:160]}")

        r = cascade(q)
        total["cascade"] += r["cost"]
        print(f"[cascade -> {r['model_used']}, escalated={r['escalated']}] "
              f"cost=${r['cost']:.6f}\n  {r['answer'][:160]}")

    print("\n=== Total spend by strategy (2 questions) ===")
    for k, v in total.items():
        print(f"  {k:12s} ${v:.6f}")
    print("\nThe cascade should sit near cheap-only on the easy question and pay "
          "strong-tier rates only where the task earned them. At production "
          "volume, this difference is a gross-margin line, not a rounding error.")


if __name__ == "__main__":
    walk()
