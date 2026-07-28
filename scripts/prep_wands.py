from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    "label.csv": {"query_id", "product_id", "label"},
}

RELEVANCE_MAP = {
    "Exact": 2,
    "Partial": 1,
    "Irrelevant": 0,
}


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
    cleaned_fields = [clean_text(field) for field in fields]
    return " ".join(field for field in cleaned_fields if field)


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


def load_wands(
    raw_dir: Path, download: bool = True, force_download: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
        for column in ("query_id", "product_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    products["search_text"] = products.apply(build_search_text, axis=1)
    labels["relevance"] = labels["label"].map(RELEVANCE_MAP).fillna(0).astype(int)

    if not sources:
        sources = {
            name: {"url": url, **file_stats(raw_dir / name)} for name, url in WANDS_URLS.items()
        }

    source_summary = {
        "homepage": WANDS_BASE_URL,
        "files": sources,
        "raw_counts": {
            "products": len(products),
            "queries": len(queries),
            "judgments": len(labels),
        },
        "columns": {
            "products": list(products.columns),
            "queries": list(queries.columns),
            "labels": list(labels.columns),
        },
        "note": "WANDS files use tab-separated values even though the filenames end in .csv.",
    }
    return products, queries, labels, source_summary


def product_record(row: pd.Series) -> dict[str, Any]:
    product_id = str(row["product_id"])
    return {
        "product_id": product_id,
        "product_name": clean_text(row.get("product_name")),
        "product_class": clean_text(row.get("product_class")),
        "category_hierarchy": clean_text(row.get("category_hierarchy")),
        "average_rating": clean_float(row.get("average_rating")),
        "review_count": clean_int(row.get("review_count")),
        "search_text": clean_text(row.get("search_text")),
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
    pair_relevance = labels.groupby(["query_id", "product_id"])["relevance"].max()
    for (query_id, product_id), relevance in pair_relevance.items():
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(relevance)
    return qrels


def write_json(path: Path, payload: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        handle.write("\n")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def existing_manifest(processed_dir: Path) -> dict[str, Any] | None:
    manifest_path = processed_dir / "manifest.json"
    required = [
        processed_dir / "corpus.json",
        processed_dir / "queries.json",
        processed_dir / "qrels.json",
        manifest_path,
    ]
    if not all(path.exists() for path in required):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 4:
        return None
    if set(manifest.get("files", {})) != {"full"}:
        return None
    return manifest


def write_structured_data(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    processed_dir: Path,
) -> dict[str, str | int]:
    corpus = make_corpus(products)
    query_map = make_queries(queries)
    qrels = make_qrels(labels)

    corpus_json = processed_dir / "corpus.json"
    queries_json = processed_dir / "queries.json"
    qrels_json = processed_dir / "qrels.json"

    write_json(corpus_json, corpus)
    write_json(queries_json, query_map)
    write_json(qrels_json, qrels)

    return {
        "corpus": portable_path(corpus_json),
        "queries": portable_path(queries_json),
        "qrels": portable_path(qrels_json),
        "products": len(corpus),
        "queries_count": len(query_map),
        "qrels_count": sum(len(items) for items in qrels.values()),
    }


def prepare_wands(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    download: bool = True,
    force_download: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    (processed_path / "corpus.jsonl").unlink(missing_ok=True)
    if not refresh and not force_download:
        manifest = existing_manifest(processed_path)
        if manifest is not None:
            return manifest

    products, queries, labels, source_summary = load_wands(
        raw_path,
        download=download,
        force_download=force_download,
    )

    full_files = write_structured_data(
        products,
        queries,
        labels,
        processed_path,
    )

    manifest = {
        "manifest_version": 4,
        "dataset": "WANDS",
        "source": source_summary,
        "raw_dir": portable_path(raw_path),
        "processed_dir": portable_path(processed_path),
        "relevance_map": RELEVANCE_MAP,
        "files": {"full": full_files},
    }
    write_json(processed_path / "manifest.json", manifest, pretty=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and structure WANDS for the RedisVL workshop."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild processed artifacts even if full outputs already exist.",
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="Print WANDS source URLs and exit."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_sources:
        print(json.dumps({"homepage": WANDS_BASE_URL, "files": WANDS_URLS}, indent=2))
        return

    manifest = prepare_wands(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        download=not args.no_download,
        force_download=args.force_download,
        refresh=args.refresh,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
