"""One end-to-end test: indexing is the expensive step, so it runs once and
every assertion shares it. Requires a real key; costs a few cents."""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@requires_key
def test_index_builds_and_both_query_modes_answer():
    main.prepare_input()
    main.init_and_configure()
    if not main.OUTPUT_DIR.exists() or not list(main.OUTPUT_DIR.rglob("*.parquet")):
        main.index()
    assert list(main.OUTPUT_DIR.rglob("*.parquet")), "indexing must produce artifacts"

    global_answer = main.query(
        "global", "What themes run across the shop's customers and their buying habits?"
    ).lower()
    assert any(word in global_answer for word in ("fruit", "apple", "banana", "pear"))

    local_answer = main.query("local", "What does Ingrid buy?").lower()
    assert "apple" in local_answer


@requires_key
def test_leiden_communities_were_built():
    """The graph is the point of this exercise, so assert it exists rather
    than only that a query returned prose."""
    if not main.OUTPUT_DIR.exists() or not list(main.OUTPUT_DIR.rglob("*.parquet")):
        pytest.skip("nothing indexed yet")

    communities = main._load("communities")
    assert communities is not None, "indexing must produce a communities table"
    assert len(communities) > 0, "Leiden must find at least one community"

    entities = main._load("entities")
    assert entities is not None and len(entities) > 0

    reports = main._load("community_reports")
    assert reports is not None and len(reports) > 0, "each community needs a summary"


def test_inspect_graph_is_safe_without_an_index(capsys):
    """Inspection must not explode when there is nothing to inspect."""
    main.inspect_graph()
    captured = capsys.readouterr().out
    assert captured  # it says something either way
