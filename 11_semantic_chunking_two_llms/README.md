# semantic_chunking_two_llms

**Theme.** Fixed-size chunking destroys the meaning it is meant to deliver:
windows cut mid-sentence, sever policies from their reasons, and the
retriever then faithfully returns amputated fragments. This exercise uses a
*cheaper second LLM* (`gpt-4.1-mini`) purely as a chunker - it proposes
topic-coherent boundaries - and validates its output mechanically: the chunks
must reassemble verbatim into the original document, so the model may only
cut, never rewrite. Retrieval quality is then compared head-to-head against
naive 240-character windows on the same questions.

**Stack.** OpenAI `gpt-4.1-mini` (chunker) + `text-embedding-3-small`
(retrieval), numpy cosine search, uv on Python 3.12. No vector database on
purpose: the lesson is the chunking, not the store.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q

    docker build -t semantic_chunking .
    docker run --rm --env-file .env semantic_chunking

## What to look for

The naive cut printed mid-walkthrough will end mid-thought; the semantic
chunk will be a whole one. On the three benchmark questions the semantic
chunks should contain the expected fact in the top-1 hit consistently; the
naive windows sometimes do and sometimes return the neighbouring fragment.
The validation step is the transferable lesson: an LLM in the data pipeline
is an untrusted optimiser, and a mechanical check (verbatim reassembly) is
what makes it safe to deploy.
