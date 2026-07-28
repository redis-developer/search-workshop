#!/usr/bin/env python3
"""Workshop-specific adapters around Redis Retrieval Optimizer.

The notebook owns the experiment and its interpretation. This module owns the
mechanical boundary between RedisVL query results, ranx runs, and the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable
import warnings

from numba.core.errors import NumbaTypeSafetyWarning
import pandas as pd
from ranx import Qrels, Run, evaluate
from redis import Redis
from redis.commands.json.path import Path as RedisJSONPath
from redis_retrieval_optimizer.schema import SearchMethodOutput
from redis_retrieval_optimizer.search_methods.base import run_search_w_time
from redis_retrieval_optimizer.search_study import run_search_study
from redisvl.query import HybridQuery, TextQuery, VectorQuery


SCORECARD_METRICS = ("ndcg@10", "recall@25", "precision@25")
SCORECARD_COLUMNS = (
    "search_method",
    *SCORECARD_METRICS,
    "avg_redis_query_ms",
)
LINEAR_TEXT_WEIGHTS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class WorkshopStudyResult:
    scorecard: pd.DataFrame
    redis_key: str
    search_methods: tuple[str, ...]


def _query_text(raw_query: str | dict[str, Any]) -> str:
    return str(raw_query["query"]) if isinstance(raw_query, dict) else str(raw_query)


def _clean_doc_id(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value) if value is not None else None


def _rows_to_scores(
    rows: list[dict[str, Any]],
    id_field_name: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, row in enumerate(rows, start=1):
        document_id = _clean_doc_id(row.get(id_field_name))
        if document_id is None:
            continue
        # ranx needs unique scores that preserve the order returned by Redis.
        scores[document_id] = 1.0 / rank
    return scores or {"no_match": 0.0}


def _run_query_method(
    search_input: Any,
    query_factory: Callable[[Any, str], Any],
) -> SearchMethodOutput:
    ranked_results: dict[str, dict[str, float]] = {}

    for query_id, raw_query in search_input.raw_queries.items():
        query = query_factory(search_input, _query_text(raw_query))
        rows = run_search_w_time(
            search_input.index,
            query,
            search_input.query_metrics,
        )
        ranked_results[str(query_id)] = _rows_to_scores(
            rows,
            id_field_name=search_input.id_field_name,
        )

    return SearchMethodOutput(
        run=Run(ranked_results),
        query_metrics=search_input.query_metrics,
    )


def _bm25_text_method(search_input: Any) -> SearchMethodOutput:
    return _run_query_method(
        search_input,
        query_factory=lambda inputs, text: TextQuery(
            text=text,
            text_field_name=inputs.text_field_name,
            return_fields=[inputs.id_field_name, inputs.text_field_name],
            num_results=inputs.ret_k,
        ),
    )


def _vector_method(search_input: Any) -> SearchMethodOutput:
    def make_query(inputs: Any, text: str) -> VectorQuery:
        vector = inputs.emb_model.embed(
            text,
            as_buffer=True,
            normalize_embeddings=True,
        )
        return VectorQuery(
            vector=vector,
            vector_field_name=inputs.vector_field_name,
            return_fields=[inputs.id_field_name, inputs.text_field_name],
            num_results=inputs.ret_k,
        )

    return _run_query_method(
        search_input,
        make_query,
    )


def _make_hybrid_method(
    combination_method: str = "LINEAR",
    text_weight: float = 0.5,
) -> Callable[[Any], SearchMethodOutput]:
    def hybrid_method(search_input: Any) -> SearchMethodOutput:
        def make_query(inputs: Any, text: str) -> HybridQuery:
            vector = inputs.emb_model.embed(
                text,
                as_buffer=True,
                normalize_embeddings=True,
            )
            query_kwargs: dict[str, Any] = {
                "text": text,
                "text_field_name": inputs.text_field_name,
                "vector": vector,
                "vector_field_name": inputs.vector_field_name,
                "vector_search_method": "KNN",
                "knn_ef_runtime": None,
                "combination_method": combination_method,
                "yield_combined_score_as": "hybrid_score",
                "return_fields": [
                    inputs.id_field_name,
                    inputs.text_field_name,
                ],
                "num_results": inputs.ret_k,
            }
            if combination_method == "LINEAR":
                query_kwargs["linear_alpha"] = text_weight
            else:
                query_kwargs["rrf_window"] = max(inputs.ret_k, 20)
                query_kwargs["rrf_constant"] = 60
            return HybridQuery(**query_kwargs)

        return _run_query_method(
            search_input,
            make_query,
        )

    return hybrid_method


def _capture_method(
    name: str,
    method: Callable[[Any], SearchMethodOutput],
    outputs: dict[str, SearchMethodOutput],
) -> Callable[[Any], SearchMethodOutput]:
    def wrapped(search_input: Any) -> SearchMethodOutput:
        output = method(search_input)
        outputs[name] = output
        return output

    return wrapped


def _build_search_method_map(
    outputs: dict[str, SearchMethodOutput],
) -> dict[str, Callable[[Any], SearchMethodOutput]]:
    methods: dict[str, Callable[[Any], SearchMethodOutput]] = {
        "bm25_text": _bm25_text_method,
        "vector_cosine": _vector_method,
        "hybrid_rrf": _make_hybrid_method("RRF"),
    }
    for text_weight in LINEAR_TEXT_WEIGHTS:
        weight_label = int(round(text_weight * 100))
        methods[f"hybrid_linear_text_{weight_label:03d}"] = _make_hybrid_method(
            "LINEAR",
            text_weight=text_weight,
        )

    return {
        name: _capture_method(name, method, outputs)
        for name, method in methods.items()
    }


def _load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _persist_scorecard(
    scorecard: pd.DataFrame,
    redis_url: str,
    redis_key: str,
) -> None:
    payload = {
        column: [_json_value(value) for value in scorecard[column].tolist()]
        for column in SCORECARD_COLUMNS
    }
    Redis.from_url(redis_url).json().set(
        redis_key,
        RedisJSONPath.root_path(),
        payload,
    )


def run_workshop_search_study(
    *,
    redis_url: str,
    study_id: str,
    index_name: str,
    queries_path: str | Path,
    qrels_path: str | Path,
    embedding_model: str,
    embedding_dims: int,
    embedding_cache_name: str,
    ret_k: int = 25,
    id_field_name: str = "product_id",
    text_field_name: str = "search_text",
    vector_field_name: str = "embedding",
) -> WorkshopStudyResult:
    """Run the fixed workshop candidate set and return its canonical scorecard.

    The helper deliberately fixes the evaluation depth at 25. Redis Retrieval
    Optimizer supplies the method lifecycle and timings; the captured runs are
    evaluated at the scorecard's exact cutoffs before those same columns are
    persisted to Redis.
    """
    if ret_k != 25:
        raise ValueError("The workshop scorecard is defined for ret_k=25.")

    captured_outputs: dict[str, SearchMethodOutput] = {}
    search_method_map = _build_search_method_map(captured_outputs)
    search_methods = tuple(search_method_map)

    config = {
        "study_id": study_id,
        "index_name": index_name,
        "queries": str(queries_path),
        "qrels": str(qrels_path),
        "search_methods": list(search_methods),
        "ret_k": ret_k,
        "id_field_name": id_field_name,
        "text_field_name": text_field_name,
        "vector_field_name": vector_field_name,
        "embedding_model": {
            "type": "hf",
            "model": embedding_model,
            "dim": embedding_dims,
            "embedding_cache_name": embedding_cache_name,
            "dtype": "float32",
        },
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*experimental.*future versions.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*experimental method hybrid_search.*",
            category=UserWarning,
        )
        optimizer_metrics = run_search_study(
            redis_url=redis_url,
            config=config,
            search_method_map=search_method_map,
        )

        missing_outputs = set(search_methods) - set(captured_outputs)
        if missing_outputs:
            raise RuntimeError(
                f"Search study did not capture methods: {sorted(missing_outputs)}"
            )

        qrels = Qrels(_load_qrels(qrels_path))
        cutoff_rows = []
        for method_name in search_methods:
            cutoff_metrics = evaluate(
                qrels,
                captured_outputs[method_name].run,
                metrics=list(SCORECARD_METRICS),
                make_comparable=True,
            )
            cutoff_rows.append(
                {"search_method": method_name, **cutoff_metrics}
            )

    cutoff_metrics = pd.DataFrame(cutoff_rows)
    redis_latency = optimizer_metrics[["search_method", "avg_query_time"]].copy()
    redis_latency["avg_redis_query_ms"] = (
        redis_latency.pop("avg_query_time") * 1000
    ).round(3)

    scorecard = cutoff_metrics.merge(
        redis_latency,
        on="search_method",
        how="inner",
        validate="one_to_one",
    )
    if len(scorecard) != len(search_methods):
        raise RuntimeError(
            "Optimizer timing rows do not match the registered search methods."
        )
    scorecard = scorecard.round(
        {metric: 6 for metric in SCORECARD_METRICS}
    )
    scorecard = scorecard.sort_values(
        ["ndcg@10", "recall@25", "avg_redis_query_ms"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    scorecard = scorecard[list(SCORECARD_COLUMNS)]

    redis_key = f"study:{study_id}"
    _persist_scorecard(scorecard, redis_url, redis_key)
    return WorkshopStudyResult(
        scorecard=scorecard,
        redis_key=redis_key,
        search_methods=search_methods,
    )
