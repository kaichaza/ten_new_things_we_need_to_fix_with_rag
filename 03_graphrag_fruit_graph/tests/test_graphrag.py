"""Tests split by cost: the free ones always run, the expensive one is opt-in.

Indexing this corpus takes minutes and spends real money, so it is not run
on every invocation. Set GRAPHRAG_RUN_INDEX=1 to include it. Tests that only
need a database (no model calls) are skipped automatically when Neo4j is not
reachable, so the suite stays green on a machine without docker running.
"""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="no API key set",
)
requires_index = pytest.mark.skipif(
    os.getenv("GRAPHRAG_RUN_INDEX") != "1",
    reason="set GRAPHRAG_RUN_INDEX=1 to run the indexing test (costs money)",
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
    reason="Neo4j not reachable - start the container first",
)


def test_corpus_is_present_and_includes_the_distractor():
    documents = list(main.CORPUS_DIR.glob("*.txt"))
    assert len(documents) >= 8

    flyer = (main.CORPUS_DIR / "grape_rumour_flyer.txt").read_text().lower()
    assert "allergic" in flyer, "the distractor must offer a wrong mechanism"

    truth = (main.CORPUS_DIR / "supplier_notice.txt").read_text().lower()
    assert "inspection" in truth
    assert "allerg" not in truth, "the true documents must never mention allergy"


def test_the_packages_expose_what_this_exercise_calls():
    """Guard against a version bump renaming things out from under us.

    The Microsoft version needed this for graphrag.api; the Neo4j version
    needs it for the handful of neo4j-graphrag classes main.py imports.
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings          # noqa: F401
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.generation import GraphRAG                  # noqa: F401
    from neo4j_graphrag.llm import OpenAILLM                        # noqa: F401
    from neo4j_graphrag.retrievers import VectorCypherRetriever     # noqa: F401

    # The schema parameter has been renamed across releases; whichever name
    # the installed version uses, main.py must be able to find one of them.
    picked = main.pick_param_name(SimpleKGPipeline.__init__, ["Person"], "node_types", "entities")
    assert picked, "SimpleKGPipeline accepts neither node_types nor entities in this version"


def test_prompts_carry_the_ethos():
    """The whitepaper claims stand on these prompts; keep them honest."""
    assert "{members}" in main.COMMUNITY_SUMMARY_PROMPT
    assert "{triples}" in main.COMMUNITY_SUMMARY_PROMPT
    assert "{reports}" in main.GLOBAL_SYNTHESIS_PROMPT
    assert "{question}" in main.GLOBAL_SYNTHESIS_PROMPT


@requires_neo4j
def test_neo4j_answers_and_gds_is_installed():
    driver = main.get_driver()
    try:
        rows = main.run_cypher(driver, "RETURN gds.version() AS version")
        assert rows and rows[0]["version"], "GDS plugin missing - Leiden cannot run"
    finally:
        driver.close()


@requires_neo4j
def test_inspect_graph_is_safe_with_no_index(capsys):
    main.inspect_graph()
    assert capsys.readouterr().out


@requires_key
@requires_neo4j
@requires_index
@pytest.mark.asyncio
async def test_index_builds_and_both_query_modes_answer():
    driver = main.get_driver()
    try:
        main.ensure_vector_index(driver)
        if not main.already_indexed(driver):
            await main.extract(driver)
            await main.resolve_entities(driver)
        assert main.already_indexed(driver)

        if main.final_level(driver) is None:
            main.detect_communities(driver)
        assert main.final_level(driver) is not None, "Leiden found nothing"

        main.summarize_communities(driver)
        reports = main.run_cypher(
            driver, "MATCH (c:Community) WHERE c.summary IS NOT NULL RETURN count(c) AS n"
        )
        assert reports[0]["n"] > 0
    finally:
        driver.close()

    answer = main.ask_global(main.GLOBAL_QUESTION).lower()
    assert any(name in answer for name in ("ingrid", "kwon", "tendai"))
