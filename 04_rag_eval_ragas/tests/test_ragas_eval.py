"""Real pipeline, real judge: the faithfulness metric must separate an honest
answer from a fabricated one over identical contexts."""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@requires_key
def test_faithfulness_separates_honest_from_fabricated():
    dataset = main.build_dataset(include_sabotage=True)
    df = main.score(dataset).to_pandas()

    honest = df.iloc[:-1]["faithfulness"].mean()
    sabotage = df.iloc[-1]["faithfulness"]

    # LLM-judged scores are estimates, so assert a wide, stable margin rather
    # than exact values.
    assert sabotage < 0.5, f"fabricated answer scored {sabotage}, expected low"
    assert honest > sabotage + 0.3, f"honest mean {honest} not clearly above sabotage {sabotage}"


@requires_key
def test_retrieval_grounds_the_grape_question():
    contexts = main.retrieve("Does the shop sell grapes?")
    assert any("does not sell grapes" in c for c in contexts)
