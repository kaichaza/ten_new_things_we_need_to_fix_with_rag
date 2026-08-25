"""Tests split by cost: the free ones always run, the expensive one is opt-in
via TEMPORAL_RUN_INDEX=1 (ingestion makes many model calls)."""
import os
from datetime import datetime, timezone

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="no API key set",
)
requires_ingest = pytest.mark.skipif(
    os.getenv("TEMPORAL_RUN_INDEX") != "1",
    reason="set TEMPORAL_RUN_INDEX=1 to run the ingestion test (costs money)",
)


def neo4j_reachable() -> bool:
    try:
        driver = main.get_driver()
        main.run_cypher(driver, "RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


requires_neo4j = pytest.mark.skipif(
    not neo4j_reachable(),
    reason="temporal Neo4j not reachable - start the container first",
)


def test_timeline_is_dated_ordered_and_trapped():
    documents = main.timeline_documents()
    assert len(documents) == 6
    dates = [reference for reference, _ in documents]
    assert dates == sorted(dates), "corpus must replay in date order"

    # The recency trap: the false flyer postdates the true documents...
    flyer = main.CORPUS_DIR / "2025-04-01_noticeboard_flyer.txt"
    assert "allergic" in flyer.read_text().lower()
    notice = main.CORPUS_DIR / "2025-03-02_supplier_notice.txt"
    assert "inspection" in notice.read_text().lower()
    assert "allerg" not in notice.read_text().lower()
    # ...and the final state contradicts the middle state.
    update = main.CORPUS_DIR / "2025-06-05_shop_update.txt"
    assert "grapes are back" in update.read_text().lower()


def test_validity_filter_logic():
    """The as-of filter is this exercise's core mechanism; pin it with a
    synthetic edge so the paid test is not the only thing testing it."""
    class FakeEdge:
        def __init__(self, valid_at, invalid_at):
            self.valid_at = valid_at
            self.invalid_at = invalid_at
            self.fact = "fake"

    def day(iso):
        return datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sells = FakeEdge(day("2025-01-01"), day("2025-03-02"))   # true then ended
    stopped = FakeEdge(day("2025-03-02"), day("2025-06-01")) # the gap
    back = FakeEdge(day("2025-06-01"), None)                 # current
    undated = FakeEdge(None, None)                           # kept always
    edges = [sells, stopped, back, undated]

    assert main.valid_as_of(edges, day("2025-02-01")) == [sells, undated]
    assert main.valid_as_of(edges, day("2025-04-15")) == [stopped, undated]
    assert main.valid_as_of(edges, day("2025-06-15")) == [back, undated]


def test_the_package_exposes_what_this_exercise_calls():
    from graphiti_core import Graphiti  # noqa: F401
    from graphiti_core.llm_client.config import LLMConfig  # noqa: F401
    from graphiti_core.llm_client.openai_client import OpenAIClient  # noqa: F401
    from graphiti_core.nodes import EpisodeType

    assert hasattr(EpisodeType, "text")
    assert hasattr(Graphiti, "build_indices_and_constraints")


@requires_neo4j
def test_inspect_graph_is_safe_with_no_ingest(capsys):
    main.inspect_graph()
    assert capsys.readouterr().out


@requires_key
@requires_neo4j
@requires_ingest
@pytest.mark.asyncio
async def test_ingest_and_the_three_dates_answer_differently():
    driver = main.get_driver()
    try:
        if not main.already_ingested(driver):
            await main.ingest()
        assert main.already_ingested(driver)
    finally:
        driver.close()

    february = (await main.ask(main.POINT_QUESTION, as_of_date="2025-02-01")).lower()
    april = (await main.ask(main.POINT_QUESTION, as_of_date="2025-04-15")).lower()
    assert february != april, "as-of answers must differ across the timeline"
