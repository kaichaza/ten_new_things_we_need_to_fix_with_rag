# injection_defence_llm_guard

**Theme.** Any writable source feeding a RAG corpus can carry instructions
addressed to the model rather than the reader - indirect prompt injection,
SQL injection's descendant, ranked the top LLM application risk by OWASP
since the list existed. There is no parameterised-query equivalent for
natural language, so the defence is layered containment: scan what enters
the context, filter what leaves it, and give the generation step the least
authority possible.

**Stack.** llm-guard 0.3.16 (`PromptInjection` input scanner - a local
HuggingFace classifier, downloaded at Docker build time), a regex output
filter for emails (llm-guard's `Sensitive` scanner is the fuller model-based
version), OpenAI `gpt-4.1-mini`, uv on Python 3.12.

## Run it

    ./setup.sh                    # first run downloads the classifier (~700 MB)
    uv run python main.py walk
    uv run pytest -q              # scanner tests run fully offline

    docker build -t injection_defence .   # classifier baked into the image
    docker run --rm --env-file .env injection_defence

## What to look for

Step 1 scans each corpus file in isolation: the poisoned review scores high
risk, the clean documents pass. Step 2 runs the pipeline unguarded - the
poison provably reaches the context (whether the model obeys it is luck plus
instruction-hierarchy training, neither of which is a control). Step 3 runs
guarded: the flagged document is dropped before generation and the output
filter redacts any email that slips through. The control this exercise
cannot demonstrate but the whitepaper insists on: a model that reads
untrusted content must not also hold consequential tools.
