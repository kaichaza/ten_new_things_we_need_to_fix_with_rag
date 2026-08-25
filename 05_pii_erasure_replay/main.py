"""Right to erasure in a RAG index: redact the source, replay the build, purge history.

Embeddings of personal data are personal data. A naive pipeline flattens the
customer file into an index, and an erasure request ('delete everything about
Kwon') then collides with three facts: the index is derived, the vectors
themselves are sensitive, and versioned storage keeps history by design.
The defensible sequence, demonstrated end to end here:

  1. DETECT  - find the subject's PII (name, email, phone) in the raw corpus.
  2. REDACT  - write a redacted copy of the source; raw text stays canonical,
               so redaction is a transformation, not surgery on vectors.
  3. REPLAY  - rebuild the index from the redacted source.
  4. PURGE   - versioned storage still holds the old rows; erasure is only
               complete when history is dropped too. We drop and recreate the
               table, and the walkthrough proves the old content is gone.
  5. VERIFY  - an LLM pass over the redacted text hunts for anything missed.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import lancedb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
REDACTED = HERE / "corpus_redacted"
DATA_DIR = HERE / "data" / "lance"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")

client = OpenAI()

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s]{7,}\d")


# --------------------------------------------------------------- detection --
def detect_subject_pii(text: str, subject: str) -> list[str]:
    """Find PII belonging to THE SUBJECT: their name, plus emails and phones
    appearing in paragraphs that mention them. Paragraph scoping is what
    keeps Ingrid's contact details untouched when Kwon asks for erasure."""
    found: list[str] = []
    for para in text.split("\n\n"):
        if subject.lower() in para.lower():
            found.append(subject)
            found += EMAIL_RE.findall(para)
            found += PHONE_RE.findall(para)
    return sorted(set(found))


def redact(text: str, subject: str) -> str:
    """Replace every occurrence of the subject's PII with a marker. The
    marker keeps the document readable and makes redaction auditable."""
    for item in detect_subject_pii(text, subject):
        text = text.replace(item, "[ERASED-DATA-SUBJECT]")
    return text


def write_redacted_corpus(subject: str) -> Path:
    """Produce the redacted copy of the whole corpus. Raw stays canonical -
    in production the raw copy itself is then deleted or retention-locked
    per your legal basis; the pipeline mechanics are identical."""
    shutil.rmtree(REDACTED, ignore_errors=True)
    REDACTED.mkdir()
    for f in sorted(CORPUS.glob("*.md")):
        (REDACTED / f.name).write_text(redact(f.read_text(), subject))
    return REDACTED


# ------------------------------------------------------------------- index --
def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_index(source_dir: Path, purge_history: bool = False) -> int:
    """Index one paragraph per row from the given source directory.

    purge_history=True DROPS the table first: overwrite alone keeps prior
    versions readable via checkout, which after an erasure request is a
    liability, not a feature. Rebuild-from-source is what makes the drop
    cheap and safe.
    """
    paragraphs = []
    for f in sorted(source_dir.glob("*.md")):
        paragraphs += [(f.name, p.strip()) for p in f.read_text().split("\n\n") if p.strip()]
    vectors = embed([p for _n, p in paragraphs])
    db = lancedb.connect(str(DATA_DIR))
    if purge_history and "notes" in db.table_names():
        db.drop_table("notes")
    table = db.create_table(
        "notes",
        data=[{"source": n, "text": p, "vector": v} for (n, p), v in zip(paragraphs, vectors)],
        mode="overwrite",
    )
    return table.count_rows()


def search(query: str, k: int = 2) -> list[str]:
    db = lancedb.connect(str(DATA_DIR))
    table = db.open_table("notes")
    qvec = embed([query])[0]
    return [h["text"] for h in table.search(qvec).limit(k).to_list()]


def all_indexed_text() -> str:
    """Every row currently in the index - the erasure verification surface."""
    db = lancedb.connect(str(DATA_DIR))
    table = db.open_table("notes")
    return " ".join(r["text"] for r in table.search().limit(1000).to_list())


# ---------------------------------------------------------------- LLM check --
def llm_verify_erasure(subject: str) -> str:
    """A second pair of eyes: ask the model whether any trace of the subject
    survives in the redacted corpus. Regex catches formats; the model catches
    phrasings ('the silver-card runner') that regex never will."""
    text = "\n\n".join(f.read_text() for f in sorted(REDACTED.glob("*.md")))
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",
             "content": "You are a data-protection reviewer. List any remaining "
                        "personal data or identifying references to the named "
                        "subject in the text. If none, reply exactly: CLEAN"},
            {"role": "user", "content": f"Subject: {subject}\n\nText:\n{text}"},
        ],
    )
    return resp.choices[0].message.content.strip()


# -------------------------------------------------------------------- walk --
def walk() -> None:
    subject = "Kwon"

    print("=== 1. Build the index from the raw corpus ===")
    rows = build_index(CORPUS)
    print(f"[index] {rows} paragraphs indexed")
    print("\nSearch 'silver loyalty card':")
    for hit in search("who holds a silver loyalty card"):
        print(f"  -> {hit[:100]}")

    print(f"\n=== 2. Erasure request received for data subject: {subject} ===")
    pii = detect_subject_pii((CORPUS / "customer_notes.md").read_text(), subject)
    print(f"[detect] subject PII found in source: {pii}")

    print("\n=== 3. Redact the source and REPLAY the index build (history purged) ===")
    write_redacted_corpus(subject)
    rows = build_index(REDACTED, purge_history=True)
    print(f"[index] rebuilt from redacted source: {rows} paragraphs, old versions dropped")

    print("\n=== 4. Prove it ===")
    remaining = all_indexed_text()
    print(f"  '{subject}' still in index:            {subject in remaining}")
    print(f"  'kwon@example.com' still in index:   {'kwon@example.com' in remaining}")
    print(f"  Ingrid's record untouched:           {'ingrid@example.com' in remaining}")

    print("\n=== 5. LLM verification pass over the redacted corpus ===")
    print(f"[verify] {llm_verify_erasure(subject)}")

    print("\nThe load-bearing design choice: raw text canonical, index derived. "
          "Erasure became redact-and-replay - a rehearsed pipeline operation - "
          "instead of surgery on vectors. And note step 3 dropped table history: "
          "versioning is an audit feature until an erasure request makes it a "
          "liability, and a real deployment schedules the purge explicitly.")


if __name__ == "__main__":
    walk()
