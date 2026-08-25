"""Tests split by cost: the free ones always run, the expensive one is opt-in.

Indexing this corpus takes minutes and spends real money, so it is not run
on every invocation. Set GRAPHRAG_RUN_INDEX=1 to include it.
"""
import inspect
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("GRAPHRAG_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="no API key set",
)
requires_index = pytest.mark.skipif(
    os.getenv("GRAPHRAG_RUN_INDEX") != "1",
    reason="set GRAPHRAG_RUN_INDEX=1 to run the indexing test (costs money)",
)


def test_corpus_is_present_and_includes_the_distractor():
    documents = list(main.CORPUS_DIR.glob("*.txt"))
    assert len(documents) >= 8

    flyer = (main.CORPUS_DIR / "grape_rumour_flyer.txt").read_text().lower()
    assert "allergic" in flyer, "the distractor must offer a wrong mechanism"

    truth = (main.CORPUS_DIR / "supplier_notice.txt").read_text().lower()
    assert "inspection" in truth
    assert "allerg" not in truth, "the true documents must never mention allergy"


def test_settings_parse_without_calling_a_model():
    """The config must load offline. This is the check that used to be a
    scaffolding step, and it costs nothing."""
    config = main.load_graphrag_config()
    assert config is not None
    assert getattr(config, "models", None), "settings.yaml declared no models"


def test_the_api_exposes_what_this_exercise_calls():
    """Guard against a version bump renaming the functions out from under us."""
    import graphrag.api as api

    for name in ("build_index", "global_search", "local_search"):
        assert hasattr(api, name), f"graphrag.api has no {name} in this version"
        assert inspect.signature(getattr(api, name)).parameters


def test_prepare_input_stages_the_corpus():
    main.prepare_input()
    staged = list(main.INPUT_DIR.glob("*.txt"))
    assert len(staged) == len(list(main.CORPUS_DIR.glob("*.txt")))


def test_inspect_graph_is_safe_with_no_index(capsys):
    main.inspect_graph()
    assert capsys.readouterr().out


@requires_key
@requires_index
@pytest.mark.asyncio
async def test_index_builds_and_both_query_modes_answer():
    main.prepare_input()
    if not main.already_indexed():
        await main.build()
    assert main.already_indexed()

    communities = main._table("communities")
    assert communities is not None and len(communities) > 0, "Leiden found nothing"

    reports = main._table("community_reports")
    assert reports is not None and len(reports) > 0

    answer = (await main.ask_global(main.GLOBAL_QUESTION)).lower()
    assert any(name in answer for name in ("ingrid", "kwon", "tendai"))
