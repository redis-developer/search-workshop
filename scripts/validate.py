#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
COLAB_NOTEBOOK_URL = "https://colab.research.google.com/github/redis-developer/search-workshop/blob/colab-migration/notebook.ipynb"
GITHUB_NOTEBOOK_URL = "https://github.com/redis-developer/search-workshop/blob/colab-migration/notebook.ipynb"

REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    "notebook.ipynb",
    "scripts/prep_wands.py",
    "scripts/setup_colab_redis.py",
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
        and not path.name.endswith(".egg-info")
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
        processed / "corpus.json",
        processed / "corpus.jsonl",
        processed / "queries.json",
        processed / "qrels.json",
        processed / "manifest.json",
    ]
    if not all(path.exists() for path in generated):
        print("INFO: WANDS data artifacts are not present. Generate them with `python scripts/prep_wands.py`.")


def check_scripts_compile() -> None:
    for rel in (
        "scripts/prep_wands.py",
        "scripts/setup_colab_redis.py",
        "scripts/validate.py",
    ):
        source = (ROOT / rel).read_text(encoding="utf-8")
        compile(source, str(ROOT / rel), "exec")


def check_notebook() -> None:
    notebook_path = ROOT / "notebook.ipynb"
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    if notebook.get("nbformat") != 4:
        fail("notebook.ipynb must be nbformat 4")

    cell_text = [
        "".join(cell.get("source", ""))
        for cell in notebook.get("cells", [])
    ]
    text = "\n".join(cell_text)
    required_terms = [
        COLAB_NOTEBOOK_URL,
        "https://github.com/redis-developer/search-workshop.git",
        "REPO_REF = 'colab-migration'",
        "/content/search-workshop",
        "%pip install -q .",
        "scripts/setup_colab_redis.py",
        "from scripts.setup_colab_redis import",
        "setup_colab_redis()",
        "JSON.GET",
        "WANDS",
        "search_text",
        "VectorQuery",
        "filtered vector",
        "Numeric Filter",
        "hybrid",
        "SVS-VAMANA",
        "Hosted embedding API",
        "Fine-tuned embedding model",
        "ANN Recall@k",
        "Redis Query Engine",
        "Read the Scorecard",
        "Reference Guide",
        "nDCG@10",
        "Recall@25",
        "WORKSHOP_RUN_ID",
        "Production checklist",
        "recommendation",
    ]
    missing = [term for term in required_terms if term not in text]
    if missing:
        fail(f"Notebook is missing workshop terms: {', '.join(missing)}")

    clone_cell = next((i for i, value in enumerate(cell_text) if "REPO_URL = " in value), None)
    install_cell = next((i for i, value in enumerate(cell_text) if "%pip install -q ." in value), None)
    redis_setup_cell = next((i for i, value in enumerate(cell_text) if "setup_colab_redis()" in value), None)
    support_import_cell = next((i for i, value in enumerate(cell_text) if "from scripts.prep_wands import" in value), None)
    ordered_cells = (clone_cell, install_cell, redis_setup_cell, support_import_cell)
    if any(index is None for index in ordered_cells):
        fail("Notebook is missing one or more ordered Colab setup cells")
    if list(ordered_cells) != sorted(ordered_cells):
        fail("Notebook must clone artifacts, install dependencies, start Redis, and then import support code")

    display_name = notebook.get("metadata", {}).get("kernelspec", {}).get("display_name")
    if display_name != "Python 3":
        fail("Notebook kernelspec display name must be Python 3 for Colab portability")


def check_colab_redis_script() -> None:
    setup_script = (ROOT / "scripts" / "setup_colab_redis.py").read_text(
        encoding="utf-8"
    )
    for term in (
        "def setup_colab_redis(",
        "Pin: version 6:8.6.*",
        "/usr/lib/redis/modules/rejson.so",
        "/usr/lib/redis/modules/redisearch.so",
        '"FT.HYBRID"',
        '"JSON.GET"',
        'REQUIRED_MODULES = {"search", "rejson"}',
    ):
        if term not in setup_script:
            fail(f"Colab Redis setup script is missing required term: {term}")


def check_documentation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for term in (
        COLAB_NOTEBOOK_URL,
        GITHUB_NOTEBOOK_URL,
        "## Run in Google Colab",
        "## Run Locally",
        "/content/search-workshop",
        "pyproject.toml",
        "8.6.*",
    ):
        if term not in readme:
            fail(f"README is missing Colab documentation term: {term}")


def check_project_metadata() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project_config = tomllib.load(handle)

    runtime_dependencies = project_config.get("project", {}).get("dependencies", [])
    if any(dependency.lower().startswith("jupyterlab") for dependency in runtime_dependencies):
        fail("JupyterLab must not be a Colab runtime dependency")

    dev_dependencies = project_config.get("dependency-groups", {}).get("dev", [])
    if not any(dependency.lower().startswith("jupyterlab") for dependency in dev_dependencies):
        fail("The default uv development dependency group must include JupyterLab")

    build_system = project_config.get("build-system", {})
    if build_system.get("build-backend") != "setuptools.build_meta":
        fail("pyproject.toml must define the setuptools build backend for `pip install .`")

    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    if "image: redis:8.6" not in compose_text:
        fail("Docker Compose must remain pinned to Redis 8.6")


def check_env_example() -> None:
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "REDIS_URL",
        "WORKSHOP_RUN_ID",
        "REDIS_LOAD_BATCH_SIZE",
        "EMBEDDING_CHUNK_SIZE",
        "EMBEDDING_BATCH_SIZE",
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

    redis_version = client.info("server")["redis_version"]
    redis_major_minor = tuple(int(part) for part in redis_version.split(".")[:2])
    if redis_major_minor < (8, 6):
        fail(f"Redis 8.6+ is required; connected to {redis_version}")

    module_names = set()
    for module in client.module_list():
        name = module.get(b"name", module.get("name"))
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name:
            module_names.add(str(name).lower())
    missing_modules = {"search", "rejson"} - module_names
    if missing_modules:
        fail(
            "Redis Search and RedisJSON are required; "
            f"missing={sorted(missing_modules)}, loaded={sorted(module_names)}"
        )

    for required_command in ("FT.HYBRID", "JSON.GET"):
        if not client.execute_command("COMMAND", "INFO", required_command):
            fail(f"{required_command} is required but unavailable")

    print(
        f"OK: Redis {redis_version} with Search, RedisJSON, "
        "FT.HYBRID, and JSON.GET reachable from REDIS_URL"
    )


def main() -> None:
    load_local_env()
    check_paths()
    check_scripts_compile()
    check_notebook()
    check_colab_redis_script()
    check_documentation()
    check_project_metadata()
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
