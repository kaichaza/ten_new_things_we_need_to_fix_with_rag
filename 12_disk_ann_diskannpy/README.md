# disk_ann_diskannpy

**Theme.** First-generation vector serving held graph indexes in RAM, so
cost scaled with corpus size at memory prices - and teams quietly embedded
less to afford the bill. The disk-first correction (the DiskANN lineage,
disk-native columnar formats) keeps the index on NVMe under a hard serving
RAM budget, trading a few milliseconds and a measurable sliver of recall for
an order of magnitude on cost. This exercise builds a real DiskANN index,
shows you the index *files* and the tiny RAM cap, and measures recall
against exact brute force so the trade is a number.

**Python version exception.** This is the one exercise in the series on
**Python 3.11**, not 3.12: diskannpy 0.7.0 publishes wheels for CPython
3.9-3.11 only and no source distribution on PyPI, so no 3.12 install path
exists. The `Dockerfile` and `setup.sh` pin 3.11 accordingly and say why.

**Stack.** diskannpy 0.7.0 (`build_disk_index`, `StaticDiskIndex`, `mips`
metric over normalised vectors = cosine), OpenAI `text-embedding-3-small`
(corpus test only - the structural test runs keyless on random vectors),
uv on Python 3.11.

## Run it

    ./setup.sh
    uv run python main.py walk
    uv run pytest -q          # the structural test needs no API key

    docker build -t disk_ann .
    docker run --rm --env-file .env disk_ann

## What to look for

The walkthrough prints the index directory listing - those files are the
index, and `search_memory_maximum` capped serving RAM near 10 MB - then
queries it and reports recall@3 against brute force. On this corpus recall
should be at or near 1.0; at scale, the recall-latency-cost triangle is
tuned with `complexity`, `graph_degree` and the cache size, which is exactly
the engineering conversation the whitepaper says replaces "buy more RAM".
