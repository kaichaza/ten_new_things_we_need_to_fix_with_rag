"""Stale indexes, rebuild-by-replay, and LanceDB time travel.

The scenario: prices live in Postgres (the operational database, the single
source of truth). A vector index in LanceDB is a DERIVED VIEW of those rows
plus the corpus documents. When a price changes in Postgres, the index does
not know - it keeps serving the world as it was. The fix demonstrated here is
the one the whitepaper argues for: never hand-edit derived state; rebuild it
by replaying from the source, and use the storage layer's versioning to see
(and audit) what the index believed at any point in time.

Run me:  uv run python main.py walk
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import lancedb
import psycopg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "lance"
CORPUS_DIR = HERE / "corpus"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fruit:fruit@localhost:5433/fruitshop")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")

client = OpenAI()  # reads OPENAI_API_KEY from the environment


# ---------------------------------------------------------------- database --
def init_db() -> None:
    """Create the products table and seed the version-1 prices.

    Small numbers on purpose: apples 3, bananas 2, pears 4. Grapes are not a
    row because the shop does not sell them - absence in the source of truth
    is itself information the index must reflect.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS products")
        conn.execute("CREATE TABLE products (name TEXT PRIMARY KEY, price INT)")
        conn.execute(
            "INSERT INTO products (name, price) VALUES ('apples', 3), ('bananas', 2), ('pears', 4)"
        )
    print("[db] products table created: apples=3 bananas=2 pears=4")


def product_sentences() -> list[str]:
    """Read the source of truth and phrase each row as an embeddable sentence."""
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute("SELECT name, price FROM products ORDER BY name").fetchall()
    return [f"The shop sells {name} at a price of {price} per kilogram." for name, price in rows]


def update_price(name: str, new_price: int) -> None:
    """Change a price in the SOURCE OF TRUTH only. The index is now stale."""
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("UPDATE products SET price = %s WHERE name = %s", (new_price, name))
    print(f"[db] {name} price updated to {new_price} - the vector index does NOT know yet")


# --------------------------------------------------------------- embedding --
def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured OpenAI embedding model."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ------------------------------------------------------------------- index --
def build_index() -> int:
    """(Re)build the LanceDB table from source: corpus files + database rows.

    This is the 'replay' operation: derived state is thrown away and
    reconstructed from canonical inputs. Because LanceDB versions every write,
    the old state remains inspectable (see time_travel) - useful for audit,
    dangerous for erasure (see the pii_erasure_replay exercise).
    """
    texts = [p.read_text() for p in sorted(CORPUS_DIR.glob("*.md"))]
    texts += product_sentences()
    vectors = embed(texts)
    db = lancedb.connect(str(DATA_DIR))
    records = [{"text": t, "vector": v} for t, v in zip(texts, vectors)]
    # mode="overwrite" replaces the table contents while PRESERVING version
    # history - the old state remains readable via checkout (see time_travel).
    table = db.create_table("shop", data=records, mode="overwrite")
    print(f"[index] rebuilt with {len(records)} rows - now at version {table.version}")
    return table.version


def retrieve(question: str, k: int = 2) -> list[str]:
    """Vector search over the CURRENT index version. Returns the raw chunks."""
    db = lancedb.connect(str(DATA_DIR))
    table = db.open_table("shop")
    qvec = embed([question])[0]
    hits = table.search(qvec).limit(k).to_list()
    return [h["text"] for h in hits]


def ask(question: str) -> str:
    """Retrieve, then answer strictly from the retrieved context."""
    context = "\n---\n".join(retrieve(question))
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",
             "content": "Answer ONLY from the provided context. If the context "
                        "does not contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def time_travel() -> None:
    """Show LanceDB's version history and read a superseded version.

    Every write created a version. An auditor's question - 'what did the
    system believe on Tuesday?' - becomes a checkout, not an argument.
    """
    db = lancedb.connect(str(DATA_DIR))
    table = db.open_table("shop")
    versions = table.list_versions()
    print(f"[time-travel] the table has {len(versions)} versions")
    if len(versions) >= 2:
        oldest = versions[0]["version"]
        table.checkout(oldest)
        old_rows = [r["text"] for r in table.search().limit(20).to_list() if "apples" in r["text"]]
        print(f"[time-travel] version {oldest} said: {old_rows}")
        table.checkout_latest()
        new_rows = [r["text"] for r in table.search().limit(20).to_list() if "apples" in r["text"]]
        print(f"[time-travel] latest version says: {new_rows}")


# -------------------------------------------------------------------- walk --
def walk() -> None:
    """The guided sequence: fresh -> stale -> replay -> time travel."""
    print("\n=== 1. Seed the source of truth and build the index ===")
    init_db()
    build_index()

    q = "What do apples cost per kilogram?"
    print(f"\n=== 2. Ask while fresh ===\nQ: {q}\nA: {ask(q)}")

    print("\n=== 3. The world changes: apples go from 3 to 5 in Postgres ===")
    update_price("apples", 5)
    print(f"Q: {q}\nA (STALE - answered from the old index): {ask(q)}")
    print(">> Perfectly grounded, perfectly cited, and wrong: the definition of a stale index.")

    print("\n=== 4. The fix is replay, not repair: rebuild from source ===")
    build_index()
    print(f"Q: {q}\nA (fresh): {ask(q)}")

    print("\n=== 5. Time travel: what did the index believe before? ===")
    time_travel()
    print("\nDone. In production, step 4 is triggered by change data capture "
          "(the database's transaction log streaming into the pipeline), not by a human.")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "walk":
        walk()
    elif command == "ask" and len(sys.argv) > 2:
        print(ask(" ".join(sys.argv[2:])))
    else:
        print("Usage: main.py [walk | ask <question>]")
