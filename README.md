# Idea Evolution Paper

A resumable pipeline for collecting scholarly papers from arXiv and PMC and
converting them into compact JSON records with structured sections, cited-paper
abstracts, and in-text citation contexts. The repository contains code only;
downloaded papers, parsed datasets, caches, credentials, and scheduler logs are
intentionally excluded.

## Output schema

Each accepted paper is written as one JSON file with these top-level fields:

```text
title, abstract, introduction, field, venue, year, method, references
```

Every reference contains a stable identifier when available, title, complete
cited-paper abstract, and one or more citation contexts from the source paper.
The default quality gate requires at least five resolved references.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
mkdir -p discovery shards logs
```

Set a real contact address before using public scholarly APIs:

```bash
export PAPER_MINING_ROOT="$PWD"
export PAPER_MINING_CONTACT_EMAIL="you@example.org"
```

An optional Semantic Scholar API key can be supplied with
`SEMANTIC_SCHOLAR_API_KEY` or `SEMANTIC_SCHOLAR_API_KEY_FILE`.

## Build a candidate cache

OAI-PMH is the preferred metadata harvesting route for large arXiv collections:

```bash
export PYTHONPATH="$PWD/runtime"
export PAPER_CANDIDATE_TARGET=320000
python scripts/expand_candidate_cache_oai.py
```

The alternative rate-limited arXiv API harvester is:

```bash
python scripts/expand_candidate_cache_api.py
```

Both harvesters resume from progress files under `discovery/`.

## Download and parse

Run a small local trial:

```bash
export PAPER_MINING_OUTPUT_DIR="$PWD/shards/shard_0"
python -m paper_mining.bulk_download \
  --download-only \
  --source-cache "$PWD/discovery/all_papers_cache.json" \
  --limit 100
```

For a sharded run, give each worker the same `--shard-count` and a distinct
zero-based `--shard-index`. Slurm examples are provided in `scripts/*.sbatch`.
Run `mkdir -p logs discovery shards` before submitting them, and set
`PAPER_MINING_PYTHON` if `python` is not the desired interpreter.

## Important environment variables

| Variable | Purpose |
| --- | --- |
| `PAPER_MINING_ROOT` | Repository/data root used by helper scripts |
| `PAPER_MINING_OUTPUT_DIR` | Output directory for one worker/shard |
| `PAPER_MINING_REFERENCE_METADATA_PATHS` | Additional local metadata indexes, separated by the OS path separator |
| `PAPER_MINING_MIN_REFERENCES` | Minimum resolved cited papers; default `5` |
| `PAPER_MINING_MAX_TITLE_LOOKUPS` | Maximum remote title lookups per paper; default `12` |
| `PAPER_MINING_DISABLE_GATEWAY` | Set to `1` to disable publisher gateway fallback |
| `PAPER_MINING_CONTACT_EMAIL` | Contact address placed in API requests |

## Data and API policy

Respect the terms, licenses, and rate limits of arXiv, PMC, Crossref, PubMed,
Semantic Scholar, and each publisher. Do not commit generated corpora or API
credentials. For very large collections, prefer the providers' official bulk
data channels instead of increasing per-paper request concurrency.
