# pii_erasure_replay

**Theme.** Embeddings of personal data are personal data, and a naive
pipeline flattens the customer file into an index anyone can query. When the
erasure request arrives, three facts collide: the index is derived, the
vectors themselves are sensitive, and versioned storage keeps history by
design. The defensible sequence is detect -> redact the source -> replay the
index build -> purge version history -> verify - which only works if raw
text was kept canonical and the index treated as rebuildable from day one.

**Stack.** LanceDB (index; `drop_table` for the history purge), regex
detection scoped per paragraph to the data subject, an LLM verification pass
(`gpt-4.1-mini`) hunting for phrasings regex misses, OpenAI
`text-embedding-3-small`, uv on Python 3.12. Microsoft Presidio is the
production-grade detector this exercise's regex stands in for.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q       # detection/redaction tests run keyless

    docker build -t pii_erasure .
    docker run --rm --env-file .env pii_erasure

## What to look for

Before: searching "silver loyalty card" surfaces Kwon's record, email and
phone. After the erasure request: his PII is gone from source and index,
history is dropped (LanceDB's versioning would otherwise happily serve the
old rows via checkout - an audit feature that becomes a liability the moment
erasure is owed), and Ingrid's and Tendai's records are provably untouched.
The LLM verification pass is the second pair of eyes: regex catches formats,
the model catches phrasings like "the silver-card runner".
