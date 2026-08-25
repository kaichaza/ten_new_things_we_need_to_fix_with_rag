"""Detection and redaction are pure text operations - tested keyless.
Index round-trip tests use real embeddings when a key is present."""
import os

import pytest

import main

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

NOTES = (main.CORPUS / "customer_notes.md").read_text()


def test_detection_is_scoped_to_the_subject():
    pii = main.detect_subject_pii(NOTES, "Kwon")
    assert "Kwon" in pii and "kwon@example.com" in pii
    # Scoping: other subjects' PII must NOT be swept up.
    assert "ingrid@example.com" not in pii
    assert "tendai@example.com" not in pii


def test_redaction_removes_subject_and_preserves_others():
    redacted = main.redact(NOTES, "Kwon")
    assert "Kwon" not in redacted
    assert "kwon@example.com" not in redacted
    assert "+46 70 444 55 66" not in redacted
    assert "ingrid@example.com" in redacted  # untouched
    assert "[ERASED-DATA-SUBJECT]" in redacted


@requires_key
def test_replay_purges_the_subject_from_the_index():
    main.build_index(main.CORPUS)
    assert "kwon@example.com" in main.all_indexed_text()

    main.write_redacted_corpus("Kwon")
    main.build_index(main.REDACTED, purge_history=True)

    remaining = main.all_indexed_text()
    assert "Kwon" not in remaining
    assert "kwon@example.com" not in remaining
    assert "ingrid@example.com" in remaining
