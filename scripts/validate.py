#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    "notebook.ipynb",
    "scripts/prep_wands.py",
    "scripts/validate.py",
]

ALLOWED_TOP_LEVEL = {
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    "notebook.ipynb",
    "data",
    "scripts",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(ROOT / ".env")


def check_paths() -> None:
    for rel in REQUIRED_PATHS:
        path = ROOT / rel
        if not path.exists():
            fail(f"Missing required path: {rel}")

    extras = {
        path.name
        for path in ROOT.iterdir()
        if path.name not in ALLOWED_TOP_LEVEL
        and path.name not in {".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".env", ".DS_Store"}
    }
    if extras:
        fail(f"Unexpected top-level paths: {', '.join(sorted(extras))}")

    pycache_dirs = [
        path
        for path in ROOT.rglob("__pycache__")
        if path.is_dir() and ".venv" not in path.relative_to(ROOT).parts
    ]
    if pycache_dirs:
        fail("Remove generated __pycache__ directories before sharing the workshop repo")

    processed = ROOT / "data" / "processed"
    generated = [
        processed / "corpus_sample.json",
        processed / "corpus_sample.jsonl",
        processed / "queries_sample.json",
        processed / "qrels_sample.json",
        processed / "manifest.json",
    ]
    if not all(path.exists() for path in generated):
        print("INFO: WANDS data artifacts are not present. Generate them with `python scripts/prep_wands.py`.")


def check_scripts_compile() -> None:
    for rel in ("scripts/prep_wands.py", "scripts/validate.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        compile(source, str(ROOT / rel), "exec")


def check_notebook() -> None:
    notebook_path = ROOT / "notebook.ipynb"
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    if notebook.get("nbformat") != 4:
        fail("notebook.ipynb must be nbformat 4")

    text = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook.get("cells", [])
    )
    required_terms = [
        "WANDS",
        "search_text",
        "VectorQuery",
        "filtered vector",
        "hybrid",
        "faceting",
        "SQLQuery",
        "SVS-VAMANA",
        "nDCG@10",
        "Recall@25",
        "WORKSHOP_RUN_ID",
        "Production checklist",
        "recommendation",
    ]
    missing = [term for term in required_terms if term not in text]
    if missing:
        fail(f"Notebook is missing workshop terms: {', '.join(missing)}")


def check_env_example() -> None:
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "REDIS_URL",
        "WORKSHOP_RUN_ID",
        "WORKSHOP_DATASET",
        "SAMPLE_PRODUCT_COUNT",
        "SAMPLE_QUERY_COUNT",
        "REDIS_LOAD_BATCH_SIZE",
        "EMBEDDING_CHUNK_SIZE",
        "EMBEDDING_BATCH_SIZE",
        "EVAL_QUERY_LIMIT",
        "HF_MODEL",
    ):
        if f"{key}=" not in env_text:
            fail(f".env.example missing {key}")


def check_redis() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis
    except ModuleNotFoundError:
        print("WARN: Python redis package is not installed. Run `uv sync` before the workshop.")
        return

    try:
        client = redis.from_url(redis_url)
        client.ping()
    except Exception:
        print("WARN: Redis unavailable. Start it with `docker compose up -d redis` or check REDIS_URL before running Redis-backed notebook cells.")
        return

    print("OK: Redis reachable from REDIS_URL")


def main() -> None:
    load_local_env()
    check_paths()
    check_scripts_compile()
    check_notebook()
    check_env_example()
    check_redis()
    print("OK: repo shape and workshop files validated")


if __name__ == "__main__":
    try:
        main()
    except SyntaxError as exc:
        fail(f"Python syntax error: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"Invalid notebook JSON: {exc}")
