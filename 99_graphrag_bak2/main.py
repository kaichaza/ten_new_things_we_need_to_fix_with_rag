"""Corpus-level questions via Microsoft GraphRAG, driven from Python.

Top-k retrieval is structurally local. A question whose answer is a property
of the WHOLE corpus - "what connects the regular customers?" - gets answered
from an arbitrary sample of chunks. GraphRAG's answer is to extract an entity
graph at index time, cluster it with hierarchical Leiden, and pre-generate a
summary for every community, so global questions are answered from summaries
of summaries.

This exercise uses GraphRAG's Python API. There is no subprocess and no CLI:
config is loaded with load_config, indexing is await api.build_index, and the
two query modes are await api.global_search and await api.local_search.

The corpus contains one deliberate distractor - a noticeboard flyer that
restates the grape question almost verbatim and answers it with the wrong
mechanism. Watch what each query mode does with it.

Run me:  uv run python main.py walk     full run, indexes if needed
         uv run python main.py doctor   config check, no model calls, free
         uv run python main.py graph    inspect an existing index, free
         uv run python main.py clean    remove generated state
"""
from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "corpus"
INPUT_DIR = HERE / "input"
OUTPUT_DIR = HERE / "output"
SETTINGS = HERE / "settings.yaml"

# GraphRAG reads GRAPHRAG_API_KEY. Mirror OPENAI_API_KEY into it so the .env
# in this folder can follow the same convention as every other exercise.
if not os.getenv("GRAPHRAG_API_KEY") and os.getenv("OPENAI_API_KEY"):
    os.environ["GRAPHRAG_API_KEY"] = os.environ["OPENAI_API_KEY"]

GLOBAL_QUESTION = "What connects the shop's regular customers, and what do their habits have in common?"
LOCAL_QUESTION = "Why does the shop refuse to sell grapes?"


# ------------------------------------------------------------------ config --
def load_graphrag_config():
    """Load settings.yaml into a GraphRagConfig.

    Imported inside the function rather than at module scope so that
    "main.py doctor" can report a helpful message if graphrag is missing,
    instead of dying on an ImportError at the top of the file.
    """
    from graphrag.config.load_config import load_config

    return load_config(HERE)


def call_with_supported_args(func, **candidates):
    """Call func with only the keyword arguments it actually accepts.

    GraphRAG's query API has changed parameter names across releases. Rather
    than pin to one shape and break on the next bump, ask the function what
    it takes. Anything it does not accept is dropped and reported, so a
    silently ignored argument cannot go unnoticed.
    """
    signature = inspect.signature(func)
    accepted = set(signature.parameters)

    used = {k: v for k, v in candidates.items() if k in accepted}
    dropped = sorted(set(candidates) - set(used))

    required = [
        name for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and name not in used
    ]
    if required:
        raise TypeError(
            f"{func.__name__} needs arguments this exercise did not supply: "
            f"{required}. The installed graphrag version expects a different "
            f"signature; run 'main.py doctor' to see it."
        )

    if dropped:
        print(f"   (not accepted by this version, dropped: {', '.join(dropped)})")
    return func(**used)


# ----------------------------------------------------------------- indexing --
def prepare_input() -> None:
    """GraphRAG reads .txt files from input/. Stage the corpus there."""
    INPUT_DIR.mkdir(exist_ok=True)
    for source in sorted(CORPUS_DIR.glob("*.txt")):
        (INPUT_DIR / source.name).write_text(source.read_text())
    print(f"[prepare] {len(list(INPUT_DIR.glob('*.txt')))} documents staged in input/")


def already_indexed() -> bool:
    return OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.parquet"))


async def build() -> None:
    """Run the whole pipeline in-process: extract, cluster, summarise.

    One call does entity extraction from each text unit, graph construction,
    hierarchical Leiden clustering over that graph, and an LLM-written report
    for every community found. Everything lands in output/ as parquet.
    """
    import graphrag.api as api

    config = load_graphrag_config()
    print("[index] running the pipeline - this makes real model calls")

    results = await api.build_index(config=config)

    failures = [r for r in results if getattr(r, "errors", None)]
    for result in results:
        name = getattr(result, "workflow", "?")
        errors = getattr(result, "errors", None)
        if errors:
            print(f"   FAILED  {name}: {errors}")
        else:
            print(f"   ok      {name}")

    if failures:
        raise RuntimeError(f"{len(failures)} workflow(s) failed - see above")

    artifacts = sorted(p.name for p in OUTPUT_DIR.rglob("*.parquet"))
    print(f"[index] complete - {len(artifacts)} parquet artifacts")


# --------------------------------------------------------------- inspection --
def _table(suffix: str) -> pd.DataFrame | None:
    """Locate one output table by filename suffix.

    Suffix rather than exact name: older GraphRAG releases prefixed every
    artifact with create_final_, and this survives either convention.
    """
    matches = sorted(OUTPUT_DIR.rglob(f"*{suffix}.parquet"))
    if not matches:
        return None
    return pd.read_parquet(matches[-1])


def _column(frame: pd.DataFrame, *candidates: str) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def inspect_graph() -> None:
    """Read back what indexing built, without making any model calls."""
    if not already_indexed():
        print("[graph] nothing indexed yet - run 'main.py walk' first")
        return

    print("\n=== What the indexing pipeline built ===")
    print("  artifacts:", ", ".join(sorted(p.name for p in OUTPUT_DIR.rglob("*.parquet"))))

    entities = _table("entities")
    relationships = _table("relationships")
    if entities is not None:
        print(f"\n  Entities extracted: {len(entities)}")
    if relationships is not None:
        print(f"  Relationships:      {len(relationships)}")
        source = _column(relationships, "source")
        target = _column(relationships, "target")
        if source and target:
            degree = pd.concat([relationships[source], relationships[target]]).value_counts()
            print("\n  Most connected entities (degree = edges touching the node):")
            for name, count in degree.head(6).items():
                print(f"    {count:>3}  {name}")

    communities = _table("communities")
    if communities is None:
        print("\n  no communities table found")
    else:
        level = _column(communities, "level")
        members = _column(communities, "entity_ids", "entity_id")
        if level is None:
            print(f"\n  Communities: {len(communities)}")
        else:
            levels = sorted(communities[level].unique())
            print(f"\n  Leiden communities: {len(communities)} across {len(levels)} levels")
            for value in levels:
                group = communities[communities[level] == value]
                if members is not None:
                    sizes = sorted((len(ids) for ids in group[members]), reverse=True)
                    print(f"    level {value}: {len(group):>2} communities  "
                          f"(entities each: {', '.join(str(s) for s in sizes)})")
                else:
                    print(f"    level {value}: {len(group):>2} communities")
            print("\n  Level 0 is the coarsest partition; deeper levels subdivide it.")

    reports = _table("community_reports")
    if reports is not None:
        title = _column(reports, "title")
        body = _column(reports, "summary", "full_content", "explanation")
        print(f"\n  Community reports written: {len(reports)}")
        if len(reports):
            row = reports.iloc[0]
            print("\n  One of them:")
            if title:
                print(f"    title: {row[title]}")
            if body:
                text = str(row[body]).strip().replace("\n", "\n    ")
                print(f"    {text[:600]}")


# ------------------------------------------------------------------ queries --
async def ask_global(question: str) -> str:
    """Answer from community summaries - map over all of them, then reduce.

    This is the mode that can speak to the whole corpus, because the reports
    it reads were written at index time from the entire graph rather than
    retrieved in response to the question.
    """
    import graphrag.api as api

    config = load_graphrag_config()
    result = call_with_supported_args(
        api.global_search,
        config=config,
        entities=_table("entities"),
        communities=_table("communities"),
        community_reports=_table("community_reports"),
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=question,
    )
    response, _context = await result
    return response


async def ask_local(question: str) -> str:
    """Answer from an entity neighbourhood plus the underlying text units.

    Closer to conventional retrieval, and the right mode for a question about
    one specific thing rather than about the corpus as a whole.
    """
    import graphrag.api as api

    config = load_graphrag_config()
    result = call_with_supported_args(
        api.local_search,
        config=config,
        entities=_table("entities"),
        communities=_table("communities"),
        community_reports=_table("community_reports"),
        text_units=_table("text_units"),
        relationships=_table("relationships"),
        covariates=None,
        community_level=2,
        response_type="Multiple Paragraphs",
        query=question,
    )
    response, _context = await result
    return response


# ------------------------------------------------------------------ doctor --
def doctor() -> None:
    """Check the setup without spending anything."""
    print("=== doctor: configuration check, no model calls ===\n")

    print(f"  settings.yaml present : {SETTINGS.exists()}")
    print(f"  corpus documents      : {len(list(CORPUS_DIR.glob('*.txt')))}")
    print(f"  already indexed       : {already_indexed()}")

    key = os.getenv("GRAPHRAG_API_KEY", "")
    if key and not key.startswith("sk-your"):
        print(f"  API key               : present ({key[:6]}...{key[-4:]})")
    else:
        print("  API key               : MISSING - copy .env.example to .env and add it")

    try:
        import graphrag
        print(f"  graphrag version      : {getattr(graphrag, '__version__', 'unknown')}")
    except ImportError as error:
        print(f"  graphrag              : NOT INSTALLED ({error})")
        return

    try:
        config = load_graphrag_config()
        print("  settings.yaml parses  : yes")
        models = getattr(config, "models", {})
        for name in models:
            model = models[name]
            print(f"      {name}: {getattr(model, 'model', '?')}")
    except Exception as error:
        print(f"  settings.yaml parses  : NO - {type(error).__name__}: {error}")
        return

    # Report the signatures this version actually exposes. If a future release
    # renames a parameter, this is where you will see it.
    import graphrag.api as api
    for name in ("build_index", "global_search", "local_search"):
        func = getattr(api, name, None)
        if func is None:
            print(f"  api.{name:<14}: MISSING from this version")
        else:
            params = ", ".join(inspect.signature(func).parameters)
            print(f"  api.{name:<14}: ({params})")


# -------------------------------------------------------------------- walk --
async def walk() -> None:
    prepare_input()

    if already_indexed():
        print("[index] existing output/ found - skipping re-index (main.py clean to redo)")
    else:
        await build()

    inspect_graph()

    print("\n=== GLOBAL search: a question about the corpus as a whole ===")
    print(f"Q: {GLOBAL_QUESTION}\n")
    print(await ask_global(GLOBAL_QUESTION))

    print("\n=== LOCAL search: a question about one specific thing ===")
    print(f"Q: {LOCAL_QUESTION}\n")
    print(await ask_local(LOCAL_QUESTION))

    print("\n" + "=" * 70)
    print("The corpus contains a noticeboard flyer claiming the owner is allergic")
    print("to grapes - a document that restates the question almost word for word")
    print("and answers it with the wrong mechanism. The supplier notice and the")
    print("owner's letter carry the real reason, and they never mention allergy.")
    print("Whether the answer above says 'allergy' or 'failed inspections' tells")
    print("you whether the graph assembled the chain or the retriever matched")
    print("the question to the nearest-looking text.")


def clean() -> None:
    for path in (INPUT_DIR, OUTPUT_DIR, HERE / "cache", HERE / "logs"):
        shutil.rmtree(path, ignore_errors=True)
    print("[clean] removed input/, output/, cache/, logs/ (settings.yaml kept)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if command == "walk":
        asyncio.run(walk())
    elif command == "doctor":
        doctor()
    elif command == "graph":
        inspect_graph()
    elif command == "clean":
        clean()
    else:
        print("Usage: main.py [walk | doctor | graph | clean]")
