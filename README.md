# Product Search Relevance with RedisVL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redis-developer/search-workshop/blob/colab-migration/notebook.ipynb)
[View the notebook on GitHub](https://github.com/redis-developer/search-workshop/blob/colab-migration/notebook.ipynb)

One-hour engineering workshop for building and evaluating e-commerce product search with RedisVL, Redis Query Engine, WANDS relevance judgments, and Redis Retrieval Optimizer.

Participants prepare real product-search data, embed product records, index them in Redis, compare retrieval patterns, and use graded relevance labels to choose the next experiment.

## What You Will Build

- The complete WANDS product corpus, query set, and relevance judgments.
- A `search_text` field built from product names, classes, hierarchy, descriptions, and features.
- A practical model-sourcing comparison across hosted APIs, open-weight models, and fine-tuning.
- A RedisVL index with text, tag, numeric, and vector fields.
- A clear under-the-hood comparison of `FLAT`, `HNSW`, and `SVS-VAMANA`.
- Vector, tag-filtered vector, numeric-filtered vector, and hybrid query examples.
- One relevance scorecard with nDCG@10, Recall@25, Precision@25, and Redis query time.
- A Redis Retrieval Optimizer search study with imported, parameterized method adapters.
- A practical `FT.HYBRID` comparison across RRF and linear text/vector weights.
- A production-oriented recommendation for the next benchmark.

## 60-Minute Flow

| Time | Section | Outcome |
|---:|---|---|
| 0-8 min | Setup and WANDS data | Data roles are clear: corpus, queries, and qrels. |
| 8-16 min | Embeddings | Participants understand the vector representation and model sourcing choices. |
| 16-25 min | Vector indexes and load | Participants compare index internals, then build the live `FLAT` index. |
| 25-37 min | Query patterns | Participants compare vector, tag- and numeric-filtered vector, and hybrid retrieval. |
| 37-40 min | Production benchmark plan | The group separates ANN Recall@k from product relevance and defines a fair index comparison. |
| 40-55 min | Evaluation and optimizer | Results move from visual inspection to measured relevance. |
| 55-60 min | Recommendation | The group leaves with a concrete next production experiment. |

## Run in Google Colab

Open the notebook directly from GitHub:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redis-developer/search-workshop/blob/colab-migration/notebook.ipynb)

Run the notebook from the first cell. The bootstrap section:

1. Clones the `colab-migration` branch into `/content/search-workshop`, including `pyproject.toml`, `.env.example`, the data-preparation code, and the search-study adapter.
2. Installs the project dependencies from `pyproject.toml`.
3. Runs `scripts/setup_colab_redis.py` to install the latest Redis `8.6.*` patch, start it inside the Colab runtime, and verify Search, RedisJSON, `FT.HYBRID`, and `JSON.GET`.
4. Downloads and prepares WANDS under the cloned repository before the workshop begins.

Colab storage and the local Redis process are ephemeral. After a runtime reset, start again from the bootstrap section. Rerunning the bootstrap cells within one live runtime is safe and reuses the existing repository clone and a healthy Redis process.

The workshop always processes the full WANDS corpus. The first run downloads the dataset and embedding model, then embeds and indexes 42,994 products; a CPU runtime can take several minutes and must remain connected. Redis embedding caches speed up reruns only within the same Colab runtime.

## Run Locally

Local prerequisites are Python 3.11 or 3.12, `uv`, Docker, and Redis 8.6 with `FT.HYBRID` support. The notebook detects that it is outside Colab and leaves Redis startup to Docker Compose or `REDIS_URL`.

```bash
uv sync
cp .env.example .env
docker compose up -d redis
uv run python scripts/prep_wands.py
uv run python scripts/validate.py
uv run jupyter lab notebook.ipynb
```

Run the notebook top-to-bottom after Redis is reachable. The Colab-only setup cell is a no-op locally.

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

## Data

WANDS is the Wayfair ANnotation Dataset for product search relevance assessment.

- Official repo: <https://github.com/wayfair/WANDS>
- Products: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv>
- Queries: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv>
- Judgments: <https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv>

The source files are tab-separated even though they use a `.csv` extension. The full dataset has 42,994 products, 480 queries, and 233,448 raw label rows. Those rows represent 231,873 unique query-product pairs; preparation keeps the highest observed grade when duplicate rows disagree because qrels require one grade per pair.

The workshop always indexes all 42,994 products and evaluates all 480 queries with their available relevance judgments. Dataset size and evaluation coverage are not configurable workshop options.

WANDS often has many relevant products per query. The notebook defines Recall@25 as a coverage guardrail and explains its natural ceiling before scoring retrieval methods.

Generated WANDS files live under `data/`, which is local and ignored by git.

## Data Prep Commands

```bash
# Build or reuse the complete local dataset
uv run python scripts/prep_wands.py

# Show source URLs
uv run python scripts/prep_wands.py --list-sources

# Rebuild processed outputs from local or downloaded raw files
uv run python scripts/prep_wands.py --refresh
```

## Environment Knobs

| Variable | Purpose |
|---|---|
| `REDIS_URL` | Redis connection string for local Redis or Redis Cloud. |
| `WORKSHOP_RUN_ID` | Namespaces the Redis index, product keys, and embedding caches. |
| `REDIS_LOAD_BATCH_SIZE` | Products written to Redis per retryable load chunk. |
| `EMBEDDING_CHUNK_SIZE` | Progress-logging chunk size around embedding generation. |
| `EMBEDDING_BATCH_SIZE` | Batch size passed into the embedding model. |
| `HF_MODEL` | Hugging Face Sentence Transformer used for query and product embeddings. Changing it normally requires re-embedding and rebuilding the index. |

## Operational Notes

- The notebook recreates the workshop index with `overwrite=True, drop=True`; use workshop-specific names only.
- RedisVL's embedding cache is stored in Redis with no TTL. Reruns are faster after the first embedding pass, and `WORKSHOP_RUN_ID` prevents participant collisions.
- Both the Colab APT setup and local Docker service stay on Redis `8.6.*` while accepting patch updates within that minor release. The executable path builds `FLAT` across full WANDS so retrieval-method comparisons use exact nearest neighbors; the notebook explains HNSW and SVS-VAMANA and provides their schema attributes for a follow-on benchmark.
- Hosted and local embedding providers are introduced as model sourcing choices. The executable path remains local and credential-free with `sentence-transformers/all-MiniLM-L6-v2`.
- Fine-tuning is presented as a measured follow-on only when held-out judgments show repeatable domain errors; it is not part of the one-hour execution path.
- Intel-specific SVS-VAMANA compression benefits depend on the Redis edition and CPU. Record the environment in any index benchmark.
- Redis Retrieval Optimizer's built-in methods are fixed, so `scripts/workshop_search_study.py` uses its `search_method_map` extension point for the workshop candidates. The notebook stays focused on the experiment and imports the RedisVL-to-`ranx` adapter instead of redefining it inline.

## Validate

```bash
uv run python scripts/validate.py
```

The validator checks the repo shape, Colab launch contract and setup order, project dependency groups, notebook content, scripts, required `.env.example` keys, and the Redis version and Search capability when Redis is reachable.

If `data/` is absent, validation remains valid and prints an informational message. Run `uv run python scripts/prep_wands.py` before the notebook to create the local WANDS artifacts.
