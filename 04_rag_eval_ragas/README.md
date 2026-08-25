# rag_eval_ragas

**Theme.** Grounded hallucination: the model can contradict or exceed its
retrieved sources while the citations make the answer look trustworthy. The
countermeasure is to make faithfulness a *number* computed on every pipeline
change - a regression metric, not a vibe. Ragas decomposes RAG quality into
faithfulness (does the answer follow from the contexts), response relevancy
(does it address the question) and context precision (was retrieval clean),
which also tells you *where* to fix a regression: generation or retrieval.

**Stack.** Ragas 0.4 (judge: `gpt-4.1-mini` wrapped via langchain-openai),
a real retrieve-and-generate pipeline over the corpus, uv on Python 3.12.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q

    docker build -t rag_eval .
    docker run --rm --env-file .env rag_eval

## What to look for

Three honest samples run through the real pipeline, then a sabotage sample:
"Yes - the shop has grapes on special offer" scored against contexts that say
the opposite. Its faithfulness lands far below the honest samples - the gap
the tests assert. Because the judge is itself an LLM, scores are estimates
with an error rate; the tests therefore assert wide margins, and production
teams calibrate judge scores against a periodically human-audited sample.
