"""Tests run the REAL application against real Postgres and real OpenAI.

Assertions are made on retrieved index content (deterministic given the
corpus) rather than on free-form LLM answers, so the tests are stable.
"""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def _db_available() -> bool:
    try:
        import psycopg
        with psycopg.connect(main.DATABASE_URL, connect_timeout=3):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable (docker compose up -d db)")


@requires_key
@requires_db
def test_index_goes_stale_and_replay_fixes_it():
    main.init_db()
    main.build_index()

    fresh = " ".join(main.retrieve("price of apples per kilogram"))
    assert "apples at a price of 3" in fresh

    # Change the source of truth; the derived index must now disagree with it.
    main.update_price("apples", 5)
    stale = " ".join(main.retrieve("price of apples per kilogram"))
    assert "apples at a price of 3" in stale, "index should still hold the OLD fact"

    # Replay: rebuild from source. The index converges to the new truth.
    main.build_index()
    rebuilt = " ".join(main.retrieve("price of apples per kilogram"))
    assert "apples at a price of 5" in rebuilt


@requires_key
@requires_db
def test_time_travel_preserves_history():
    main.init_db()
    v1 = main.build_index()
    main.update_price("apples", 5)
    v2 = main.build_index()
    assert v2 > v1, "each rebuild must create a new inspectable version"
