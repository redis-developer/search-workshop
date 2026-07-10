# Product Search Relevance with RedisVL

One-hour engineering workshop for building and evaluating ecommerce product search with RedisVL, Redis Search, WANDS relevance judgments, and Redis Retrieval Optimizer.

Participants prepare real product-search data, embed product records, index them in Redis, compare retrieval patterns, and use graded relevance labels to choose the next experiment.

## What You Will Build

- A deterministic WANDS product-search sample for live workshop use.
- A `search_text` field built from product names, classes, hierarchy, descriptions, and features.
- A RedisVL index with text, tag, numeric, and vector fields.
- Vector, filtered vector, hybrid, faceted, numeric, and SQL-like query examples.
- A relevance loop with nDCG@10, Recall@25, and query time.
- A Redis Retrieval Optimizer search study with custom parameterized methods.
- A practical `FT.HYBRID` comparison across RRF and linear text/vector weights.
- A production-oriented recommendation for the next benchmark.

## 60-Minute Flow

| Time | Section | Outcome |
|---:|---|---|
| 0-10 min | Setup and WANDS sample | Data roles are clear: corpus, queries, and qrels. |
| 10-25 min | Embeddings and RedisVL index | Products are embedded, loaded as Redis hashes, and indexed. |
| 25-40 min | Query patterns | Participants compare vector, filtered, hybrid, facets, numeric filters, and SQL-like lookup. |
| 40-55 min | Evaluation and optimizer | Results move from visual inspection to measured relevance. |
| 55-60 min | Recommendation | The group leaves with a concrete next production experiment. |

## Prerequisites

- Python 3.11 or 3.12
- `uv`
- Docker, for the local Redis 8.4 server
- Redis 8.4+ or Redis Cloud with `FT.HYBRID` support for the hybrid query cells

## Quick Start

```bash
uv sync
cp .env.example .env
docker compose up -d redis
uv run python scripts/prep_wands.py
uv run python scripts/validate.py
uv run jupyter lab notebook.ipynb
```

Run the notebook top-to-bottom after Redis is reachable. Redis is required because the lab exercises RedisVL and Redis Search directly.

The notebook uses `REDIS_URL` for every Redis client. For a shared classroom Redis instance, give each participant a unique `WORKSHOP_RUN_ID`; that value scopes the index name, key prefix, and embedding cache names.

## Redis Cloud

To run against Redis Cloud, update `.env`:

```bash
REDIS_URL=rediss://:<password>@<host>:<port>
WORKSHOP_RUN_ID=<your-name-or-seat>
```

The validator loads `.env` and pings the target from `REDIS_URL`:

```bash
uv run python scripts/validate.py
```

## Data and Modes

WANDS is the Wayfair ANnotation Dataset for product search relevance assessment.

- Official repo: <https://github.com/wayfair/WANDS>
- Products: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv>
- Queries: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv>
- Judgments: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv>

The source files are tab-separated even though they use a `.csv` extension. The full dataset has 42,994 products, 480 queries, and 233,448 raw relevance judgments.

The default workshop mode uses a deterministic 600-product, 24-query sample. The sampler is query-first: it prioritizes judged queries and keeps at least one relevant product per query when the product budget allows it. Use this mode for the 60-minute delivery.

Full mode exists for longer validation runs:

```bash
WORKSHOP_DATASET=full
uv run python scripts/prep_wands.py --refresh --full
```

Full mode indexes all 42,994 products and evaluates all loaded queries unless `EVAL_QUERY_LIMIT` is set.

WANDS often has many relevant products per query. The notebook shows judgment-density statistics before scoring retrieval methods so Recall@25 is interpreted as a coverage guardrail, not a standalone quality grade.

Generated WANDS files live under `data/`, which is local and ignored by git.

## Data Prep Commands

```bash
# Build or reuse the default local sample
uv run python scripts/prep_wands.py

# Show source URLs
uv run python scripts/prep_wands.py --list-sources

# Rebuild processed outputs from local or downloaded raw files
uv run python scripts/prep_wands.py --refresh
```

Adjust `SAMPLE_PRODUCT_COUNT` and `SAMPLE_QUERY_COUNT` in `.env` for a larger local sample. Larger samples improve evaluation realism but increase embedding and indexing time.

## Environment Knobs

| Variable | Purpose |
|---|---|
| `REDIS_URL` | Redis connection string for local Redis or Redis Cloud. |
| `WORKSHOP_RUN_ID` | Namespaces the Redis index, product keys, and embedding caches. |
| `WORKSHOP_DATASET` | `sample` for the live workshop, `full` for full WANDS runs. |
| `REDIS_LOAD_BATCH_SIZE` | Products written to Redis per retryable load chunk. |
| `EMBEDDING_CHUNK_SIZE` | Progress-logging chunk size around embedding generation. |
| `EMBEDDING_BATCH_SIZE` | Batch size passed into the embedding model. |
| `EVAL_QUERY_LIMIT` | Blank means 8 search-study queries in sample mode and all loaded queries in full mode. |
| `HF_MODEL` | HuggingFace sentence-transformer used for query and product embeddings. |

## Operational Notes

- The notebook recreates the workshop index with `overwrite=True, drop=True`; use workshop-specific names only.
- RedisVL's embedding cache is stored in Redis with no TTL. Reruns are faster after the first embedding pass, and `WORKSHOP_RUN_ID` prevents participant collisions.
- The live path uses `FLAT` vector search because it is exact and easy to explain on a small sample. The notebook includes `HNSW` and `SVS-VAMANA` attributes as follow-on benchmark options.
- Redis Retrieval Optimizer's built-in methods are fixed, but the notebook uses its `search_method_map` extension point to register parameterized methods. That keeps BM25, vector search, `FT.HYBRID` RRF, and `FT.HYBRID` linear weight comparisons inside one search study.

## Validate

```bash
uv run python scripts/validate.py
```

The validator checks the repo shape, parses the notebook, compiles the scripts, verifies required `.env.example` keys, and pings Redis when the Python Redis client is installed.

If `data/` is absent, validation remains valid and prints an informational message. Run `uv run python scripts/prep_wands.py` before the notebook to create the local WANDS artifacts.
