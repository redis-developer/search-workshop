#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


WANDS_BASE_URL = "https://github.com/wayfair/WANDS"
WANDS_URLS = {
    "product.csv": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv",
    "query.csv": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv",
    "label.csv": "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv",
}

EXPECTED_COLUMNS = {
    "product.csv": {
        "product_id",
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
    },
    "query.csv": {"query_id", "query"},
    "label.csv": {"id", "query_id", "product_id", "label"},
}

RELEVANCE_MAP = {
    "Exact": 2,
    "Partial": 1,
    "Irrelevant": 0,
}
SAMPLE_STRATEGY_VERSION = 2


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("|", "; ").replace("/", " > ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def clean_int(value: Any, default: int = 0) -> int:
    return int(clean_float(value, float(default)))


def build_search_text(row: pd.Series) -> str:
    fields = [
        row.get("product_name"),
        row.get("product_class"),
        row.get("category_hierarchy"),
        row.get("product_description"),
        row.get("product_features"),
    ]
    return " ".join(clean_text(field) for field in fields if clean_text(field))


def file_stats(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def download_wands(raw_dir: Path, force: bool = False) -> dict[str, dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict[str, Any]] = {}
    for filename, url in WANDS_URLS.items():
        target = raw_dir / filename
        if target.exists() and not force:
            print(f"Using existing {target}")
        else:
            print(f"Downloading {filename} from {url}")
            with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir) as tmp:
                tmp_path = Path(tmp.name)
            try:
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(target)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        sources[filename] = {"url": url, **file_stats(target)}
    return sources


def validate_columns(name: str, frame: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS[name] - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing expected columns: {sorted(missing)}")


def load_wands(raw_dir: Path, download: bool = True, force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sources = download_wands(raw_dir, force=force_download) if download else {}
    missing = [name for name in WANDS_URLS if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing WANDS files in {raw_dir}: {', '.join(missing)}. "
            "Run `python scripts/prep_wands.py --refresh` to download them."
        )

    products = pd.read_csv(raw_dir / "product.csv", sep="\t")
    queries = pd.read_csv(raw_dir / "query.csv", sep="\t")
    labels = pd.read_csv(raw_dir / "label.csv", sep="\t")

    validate_columns("product.csv", products)
    validate_columns("query.csv", queries)
    validate_columns("label.csv", labels)

    products = products.rename(columns={"category hierarchy": "category_hierarchy"})

    for frame in (products, queries, labels):
        for column in ("id", "query_id", "product_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    products["search_text"] = products.apply(build_search_text, axis=1)
    labels["relevance"] = labels["label"].map(RELEVANCE_MAP).fillna(0).astype(int)

    if not sources:
        sources = {name: {"url": url, **file_stats(raw_dir / name)} for name, url in WANDS_URLS.items()}

    source_summary = {
        "homepage": WANDS_BASE_URL,
        "files": sources,
        "raw_counts": {
            "products": int(len(products)),
            "queries": int(len(queries)),
            "judgments": int(len(labels)),
        },
        "columns": {
            "products": list(products.columns),
            "queries": list(queries.columns),
            "labels": list(labels.columns),
        },
        "note": "WANDS files use tab-separated values even though the filenames end in .csv.",
    }
    return products, queries, labels, source_summary


def choose_sample(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    max_products: int,
    max_queries: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if max_products < 1:
        raise ValueError("max_products must be at least 1")
    if max_queries < 1:
        raise ValueError("max_queries must be at least 1")

    positives = labels[labels["relevance"] > 0]
    query_priority = (
        positives.groupby("query_id")["relevance"]
        .agg(["max", "sum", "count"])
        .sort_values(["max", "sum", "count"], ascending=False)
    )
    selected_query_ids = query_priority.head(max_queries).index.tolist()
    selected_positive_labels = positives[positives["query_id"].isin(selected_query_ids)].copy()

    required_product_ids: set[str] = set()
    usable_query_ids: list[str] = []
    for query_id in selected_query_ids:
        query_positives = selected_positive_labels[selected_positive_labels["query_id"] == query_id]
        if query_positives.empty:
            continue
        best_label = query_positives.sort_values(
            ["relevance", "product_id"],
            ascending=[False, True],
        ).iloc[0]
        product_id = str(best_label["product_id"])
        if len(required_product_ids) >= max_products and product_id not in required_product_ids:
            continue
        required_product_ids.add(product_id)
        usable_query_ids.append(str(query_id))

    sample_labels = labels[labels["query_id"].isin(usable_query_ids)].copy()

    product_priority = (
        sample_labels.groupby("product_id")["relevance"]
        .agg(["max", "sum", "count"])
        .sort_values(["max", "sum", "count"], ascending=False)
    )
    selected_product_ids = set(required_product_ids)
    for product_id in product_priority.index.astype(str):
        if len(selected_product_ids) >= max_products:
            break
        selected_product_ids.add(product_id)

    sample_labels = sample_labels[sample_labels["product_id"].isin(selected_product_ids)].copy()

    usable_query_ids = set(sample_labels[sample_labels["relevance"] > 0]["query_id"].unique())
    sample_queries = queries[queries["query_id"].isin(usable_query_ids)].copy()
    sample_labels = sample_labels[sample_labels["query_id"].isin(set(sample_queries["query_id"]))].copy()
    sample_products = products[products["product_id"].isin(selected_product_ids)].copy()
    return sample_products, sample_queries, sample_labels


def product_record(row: pd.Series) -> dict[str, Any]:
    return {
        "id": str(row["product_id"]),
        "_id": str(row["product_id"]),
        "product_id": str(row["product_id"]),
        "product_name": clean_text(row.get("product_name")),
        "product_class": clean_text(row.get("product_class")),
        "category_hierarchy": clean_text(row.get("category_hierarchy")),
        "product_description": clean_text(row.get("product_description")),
        "product_features": clean_text(row.get("product_features")),
        "average_rating": clean_float(row.get("average_rating")),
        "review_count": clean_int(row.get("review_count")),
        "search_text": clean_text(row.get("search_text")),
        "text": clean_text(row.get("search_text")),
    }


def make_corpus(products: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["product_id"]): product_record(row) for _, row in products.iterrows()}


def make_queries(queries: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["query_id"]): clean_text(row["query"])
        for _, row in queries.sort_values("query_id").iterrows()
    }


def make_qrels(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for _, row in labels.iterrows():
        qrels.setdefault(str(row["query_id"]), {})[str(row["product_id"])] = int(row["relevance"])
    return qrels


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def existing_manifest(
    processed_dir: Path,
    max_products: int,
    max_queries: int,
) -> dict[str, Any] | None:
    manifest_path = processed_dir / "manifest.json"
    required = [
        processed_dir / "corpus_sample.json",
        processed_dir / "corpus_sample.jsonl",
        processed_dir / "queries_sample.json",
        processed_dir / "qrels_sample.json",
        manifest_path,
    ]
    if not all(path.exists() for path in required):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = manifest.get("sample", {})
    if sample.get("requested_products") != max_products:
        return None
    if sample.get("requested_queries") != max_queries:
        return None
    if manifest.get("sample_strategy_version") != SAMPLE_STRATEGY_VERSION:
        return None
    return manifest


def write_structured_split(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    processed_dir: Path,
    suffix: str,
) -> dict[str, str]:
    corpus = make_corpus(products)
    query_map = make_queries(queries)
    qrels = make_qrels(labels)

    corpus_json = processed_dir / f"corpus{suffix}.json"
    corpus_jsonl = processed_dir / f"corpus{suffix}.jsonl"
    queries_json = processed_dir / f"queries{suffix}.json"
    qrels_json = processed_dir / f"qrels{suffix}.json"

    write_json(corpus_json, corpus)
    write_jsonl(corpus_jsonl, list(corpus.values()))
    write_json(queries_json, query_map)
    write_json(qrels_json, qrels)

    return {
        "corpus": portable_path(corpus_json),
        "corpus_jsonl": portable_path(corpus_jsonl),
        "queries": portable_path(queries_json),
        "qrels": portable_path(qrels_json),
        "products": str(len(corpus)),
        "queries_count": str(len(query_map)),
        "qrels_count": str(sum(len(items) for items in qrels.values())),
    }


def prepare_wands(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    max_products: int | None = None,
    max_queries: int | None = None,
    download: bool = True,
    force_download: bool = False,
    refresh: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    max_products = max_products or int(os.getenv("SAMPLE_PRODUCT_COUNT", "600"))
    max_queries = max_queries or int(os.getenv("SAMPLE_QUERY_COUNT", "24"))

    if not refresh and not force_download and not full:
        manifest = existing_manifest(processed_path, max_products, max_queries)
        if manifest is not None:
            return manifest

    products, queries, labels, source_summary = load_wands(
        raw_path,
        download=download,
        force_download=force_download,
    )

    sample_products, sample_queries, sample_labels = choose_sample(
        products=products,
        queries=queries,
        labels=labels,
        max_products=max_products,
        max_queries=max_queries,
    )

    sample_files = write_structured_split(
        sample_products,
        sample_queries,
        sample_labels,
        processed_path,
        "_sample",
    )

    files: dict[str, Any] = {"sample": sample_files}
    if full:
        files["full"] = write_structured_split(products, queries, labels, processed_path, "")

    manifest = {
        "dataset": "WANDS",
        "source": source_summary,
        "raw_dir": portable_path(raw_path),
        "processed_dir": portable_path(processed_path),
        "sample": {
            "enabled": True,
            "requested_products": max_products,
            "requested_queries": max_queries,
            "products": int(sample_files["products"]),
            "queries": int(sample_files["queries_count"]),
            "qrels": int(sample_files["qrels_count"]),
        },
        "full_outputs_written": full,
        "sample_strategy_version": SAMPLE_STRATEGY_VERSION,
        "sample_note": (
            "The sampler prioritizes judged queries and keeps at least one relevant product per query "
            "when the product budget allows it. Actual query count can still be lower than requested "
            "if the product budget is smaller than the query budget or the source data has too few judged positives."
        ),
        "relevance_map": RELEVANCE_MAP,
        "files": files,
    }
    write_json(processed_path / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and structure WANDS for the RedisVL workshop.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--sample-products", type=int, default=None)
    parser.add_argument("--sample-queries", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Also write full corpus/query/qrels outputs.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Rebuild processed artifacts even if sample outputs already exist.")
    parser.add_argument("--list-sources", action="store_true", help="Print WANDS source URLs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_sources:
        print(json.dumps({"homepage": WANDS_BASE_URL, "files": WANDS_URLS}, indent=2))
        return

    manifest = prepare_wands(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_products=args.sample_products,
        max_queries=args.sample_queries,
        download=not args.no_download,
        force_download=args.force_download,
        refresh=args.refresh,
        full=args.full,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
