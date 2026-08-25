"""Corpus-level questions via Microsoft GraphRAG: entities, communities, summaries.

Top-k chunk retrieval is structurally local: a question whose answer is a
property of the WHOLE corpus ('what themes run across the customers?') gets
assembled from an arbitrary sample. GraphRAG's answer is to extract an entity
graph from the corpus, cluster it with the hierarchical Leiden algorithm, and
pre-summarise each community, so global questions are answered from summaries
of summaries and local questions from the entity neighbourhood.

This exercise drives the pinned graphrag CLI (init -> configure -> index ->
inspect -> query) over the fruit-shop corpus, then reads the parquet artifacts
back so the graph it built is visible rather than merely asserted. Indexing
makes real model calls: expect a few cents and a few minutes on this corpus.

Run me:  uv run python main.py walk
         uv run python main.py graph   (inspect an existing index, no cost)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
INPUT_DIR = HERE / "input"       # graphrag's expected input location
SETTINGS = HERE / "settings.yaml"
OUTPUT_DIR = HERE / "output"

CHAT_MODEL = os.getenv("MODEL_CHEAP", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# graphrag reads GRAPHRAG_API_KEY; mirror the OpenAI key into it so the user
# only maintains one variable in .env.
os.environ.setdefault("GRAPHRAG_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def sh(*args: str) -> subprocess.CompletedProcess:
    """Run a graphrag CLI command in this project root, echoing it first."""
    print(f"$ {' '.join(args)}")
    result = subprocess.run(args, cwd=HERE, text=True, capture_output=True)
    if result.returncode != 0:
        # Print before raising. Capturing output keeps the walkthrough
        # readable, but a swallowed stderr turns a one-line cause into an
        # unexplained non-zero exit.
        print(f"[graphrag] command failed, exit {result.returncode}")
        if result.stdout.strip():
            print("--- graphrag stdout ---")
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("--- graphrag stderr ---")
            print(result.stderr.rstrip())
        raise subprocess.CalledProcessError(
            result.returncode, args, output=result.stdout, stderr=result.stderr
        )
    return result


def prepare_input() -> None:
    """graphrag indexes .txt files under input/; copy the corpus across."""
    INPUT_DIR.mkdir(exist_ok=True)
    for md in sorted((HERE / "corpus").glob("*.md")):
        (INPUT_DIR / (md.stem + ".txt")).write_text(md.read_text())
    print(f"[prepare] {len(list(INPUT_DIR.glob('*.txt')))} corpus files staged in input/")


def init_and_configure() -> None:
    """Scaffold graphrag config once, then pin our models into settings.yaml.

    We patch only model names (chat + embedding) and leave the scaffold's
    structure alone, which keeps this resilient to minor config-schema
    changes within the pinned graphrag version.
    """
    if not SETTINGS.exists():
        # "graphrag init" refuses to overwrite existing files, and this
        # directory already has a .env holding the API key. Its --force flag
        # would overwrite settings.yaml, prompts/ AND .env, losing the key.
        #
        # So init runs in a throwaway directory and only the generated
        # settings.yaml and default prompts are copied back. Our .env is
        # never in scope.
        with tempfile.TemporaryDirectory() as scratch:
            sh(sys.executable, "-m", "graphrag", "init", "--root", scratch)

            generated = Path(scratch) / "settings.yaml"
            if not generated.exists():
                raise RuntimeError(
                    f"graphrag init produced no settings.yaml in {scratch}"
                )
            shutil.copy2(generated, SETTINGS)

            # graphrag reads its prompt templates from prompts/ if present.
            # Copy the defaults across once; never overwrite edited ones.
            scratch_prompts = Path(scratch) / "prompts"
            if scratch_prompts.is_dir() and not (HERE / "prompts").exists():
                shutil.copytree(scratch_prompts, HERE / "prompts")

            print("[configure] scaffolded settings.yaml (.env left untouched)")
    cfg = yaml.safe_load(SETTINGS.read_text())
    for name, model_cfg in cfg.get("models", {}).items():
        if "embedding" in str(model_cfg.get("type", "")) or "embedding" in name:
            model_cfg["model"] = EMBED_MODEL
        else:
            model_cfg["model"] = CHAT_MODEL
    SETTINGS.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[configure] settings.yaml pinned to chat={CHAT_MODEL} embed={EMBED_MODEL}")


def index() -> None:
    """Run the indexing pipeline: extraction -> graph -> Leiden -> summaries."""
    sh(sys.executable, "-m", "graphrag", "index", "--root", ".")
    artifacts = list(OUTPUT_DIR.rglob("*.parquet"))
    print(f"[index] complete - {len(artifacts)} parquet artifacts in output/")


# --------------------------------------------------------------- inspection --
# graphrag writes its results as parquet tables. Filenames have changed across
# releases (older builds prefixed everything with create_final_), so these are
# located by suffix rather than by exact name.

def _find_table(suffix: str) -> Path | None:
    """Locate one output table by filename suffix, newest match wins."""
    matches = sorted(OUTPUT_DIR.rglob(f"*{suffix}.parquet"))
    if not matches:
        return None
    return matches[-1]


def _load(suffix: str) -> pd.DataFrame | None:
    path = _find_table(suffix)
    if path is None:
        return None
    return pd.read_parquet(path)


def _first_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first candidate column that this graphrag version emitted."""
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def show_entities() -> None:
    """The nodes: what the extraction step pulled out of the corpus."""
    entities = _load("entities")
    relationships = _load("relationships")
    if entities is None:
        print("[graph] no entities table found")
        return

    title_col = _first_column(entities, "title", "name")
    print(f"\n  Entities extracted: {len(entities)}")

    if relationships is not None:
        print(f"  Relationships:      {len(relationships)}")
        source_col = _first_column(relationships, "source")
        target_col = _first_column(relationships, "target")
        if source_col and target_col:
            # Degree is how many edges touch a node. graphrag sometimes stores
            # this on the entity table and sometimes not, so it is counted here
            # from the relationships table, which is always present.
            degree = (
                pd.concat([relationships[source_col], relationships[target_col]])
                .value_counts()
            )
            print("\n  Most connected entities (degree = edges touching the node):")
            for name, count in degree.head(6).items():
                print(f"    {count:>3}  {name}")
    elif title_col:
        print("\n  Sample entities:")
        for name in entities[title_col].head(6):
            print(f"    - {name}")


def show_communities() -> None:
    """The Leiden hierarchy: clusters of entities, recursively subdivided.

    graphrag runs hierarchical Leiden over the entity graph, clustering until
    a community-size threshold is reached. Communities are strictly
    hierarchical - level 0 is the coarsest partition, and each deeper level
    subdivides its parent as cluster affinity narrows.
    """
    communities = _load("communities")
    if communities is None:
        print("[graph] no communities table found - has indexing run?")
        return

    level_col = _first_column(communities, "level")
    members_col = _first_column(communities, "entity_ids", "entity_id")

    if level_col is None:
        print(f"\n  Communities: {len(communities)} (no level column in this version)")
        return

    levels = sorted(communities[level_col].unique())
    print(f"\n  Leiden communities: {len(communities)} across {len(levels)} levels")

    for level in levels:
        group = communities[communities[level_col] == level]
        if members_col is not None:
            sizes = sorted((len(ids) for ids in group[members_col]), reverse=True)
            spread = ", ".join(str(s) for s in sizes)
            print(f"    level {level}: {len(group):>2} communities  "
                  f"(entities per community: {spread})")
        else:
            print(f"    level {level}: {len(group):>2} communities")

    print("\n  Level 0 is the coarsest partition; deeper levels subdivide it.")


def show_a_summary() -> None:
    """One actual community report - the thing global queries are built from.

    This is the payoff of the whole indexing run: an LLM-written summary of a
    cluster the Leiden algorithm found, generated once at index time and
    reused by every global query afterwards.
    """
    reports = _load("community_reports")
    if reports is None:
        print("\n[graph] no community_reports table found")
        return

    title_col = _first_column(reports, "title")
    body_col = _first_column(reports, "summary", "full_content", "explanation")
    level_col = _first_column(reports, "level")

    # Prefer a coarse community: those summaries span the most of the corpus.
    row = reports.iloc[0]
    if level_col is not None:
        coarse = reports[reports[level_col] == reports[level_col].min()]
        if len(coarse):
            row = coarse.iloc[0]

    print(f"\n  Community reports written: {len(reports)}")
    print("\n  One of them, in full:")
    if title_col:
        print(f"    title: {row[title_col]}")
    if body_col:
        text = str(row[body_col]).strip().replace("\n", "\n    ")
        print(f"    {text}")


def inspect_graph() -> None:
    """Read back what indexing built: entities, Leiden communities, summaries."""
    if not OUTPUT_DIR.exists() or not list(OUTPUT_DIR.rglob("*.parquet")):
        print("[graph] nothing indexed yet - run 'main.py walk' first")
        return

    print("\n=== What the indexing pipeline actually built ===")
    tables = sorted(p.name for p in OUTPUT_DIR.rglob("*.parquet"))
    print(f"  Artifacts: {', '.join(tables)}")

    show_entities()
    show_communities()
    show_a_summary()


def query(method: str, question: str) -> str:
    """Query the built graph. 'global' uses community summaries (corpus-level);
    'local' uses the entity neighbourhood (entity-level)."""
    result = sh(sys.executable, "-m", "graphrag", "query",
                "--root", ".", "--method", method, "--query", question)
    return result.stdout.strip()


def walk() -> None:
    prepare_input()
    init_and_configure()
    if not OUTPUT_DIR.exists() or not list(OUTPUT_DIR.rglob("*.parquet")):
        print("\n=== Indexing (real model calls; a few minutes on this corpus) ===")
        index()
    else:
        print("\n[index] existing output/ found - skipping re-index (delete output/ to redo)")

    inspect_graph()

    print("\n=== GLOBAL query: a corpus-level question no top-k retriever can answer ===")
    print(query("global", "What themes run across the shop's customers and their buying habits?"))

    print("\n=== LOCAL query: an entity-neighbourhood question ===")
    print(query("local", "What does Ingrid buy, and why does she value the shop?"))

    print("\nThe global answer was synthesised from the community summaries printed "
          "above, built once at index time - the structural fix for questions whose "
          "answer is a property of the whole corpus rather than any single chunk.")


def clean() -> None:
    """Remove generated state for a fresh run."""
    for path in (INPUT_DIR, OUTPUT_DIR, HERE / "cache", HERE / "logs"):
        shutil.rmtree(path, ignore_errors=True)
    SETTINGS.unlink(missing_ok=True)
    (HERE / ".env.graphrag").unlink(missing_ok=True)
    print("[clean] removed input/, output/, cache/, logs/, settings.yaml")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "walk":
        walk()
    elif command == "graph":
        inspect_graph()
    elif command == "clean":
        clean()
    else:
        print("Usage: main.py [walk | graph | clean]")
