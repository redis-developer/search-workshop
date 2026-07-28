"""Direct, inspectable retrieval evaluation for the WANDS workshop."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from redisvl.query import HybridQuery, TextQuery, VectorQuery

DEFAULT_CANDIDATES = {
    "bm25_text": ("text", None),
    "vector_cosine": ("vector", None),
    "hybrid_rrf": ("rrf", None),
    "hybrid_linear_text_025": ("linear", 0.25),
    "hybrid_linear_text_050": ("linear", 0.50),
    "hybrid_linear_text_075": ("linear", 0.75),
}

SCORECARD_COLUMNS = (
    "search_method",
    "ndcg@10",
    "recall@25",
    "precision@25",
    "ndcg_wins",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
)

STRATEGY_LABELS = {
    "bm25_text": "BM25 text",
    "vector_cosine": "Vector cosine",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_linear_text_025": "Linear · 25% text",
    "hybrid_linear_text_050": "Linear · 50% text",
    "hybrid_linear_text_075": "Linear · 75% text",
}

STRATEGY_COLORS = {
    "vector_cosine": "#59636E",
    "hybrid_rrf": "#DC382D",
    "hybrid_linear_text_025": "#F7A58D",
    "hybrid_linear_text_050": "#EE7257",
    "hybrid_linear_text_075": "#B92D24",
}


def _discounted_gain(relevance: Sequence[float]) -> float:
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(relevance, start=1))


def ndcg_at_k(
    judgments: Mapping[str, int],
    ranked_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Compute nDCG@k."""
    observed = [float(judgments.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted(
        (float(grade) for grade in judgments.values() if grade > 0),
        reverse=True,
    )[:k]
    ideal_gain = _discounted_gain(ideal)
    return _discounted_gain(observed) / ideal_gain if ideal_gain else 0.0


def recall_at_k(
    judgments: Mapping[str, int],
    ranked_ids: Sequence[str],
    k: int = 25,
) -> float:
    """Compute Recall@k."""
    relevant = {doc_id for doc_id, grade in judgments.items() if grade > 0}
    return len(relevant & set(ranked_ids[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(
    judgments: Mapping[str, int],
    ranked_ids: Sequence[str],
    k: int = 25,
) -> float:
    """Compute Precision@k."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(judgments.get(doc_id, 0) > 0 for doc_id in ranked_ids[:k]) / k


RANKING_METRICS = {
    "ndcg@10": partial(ndcg_at_k, k=10),
    "recall@25": partial(recall_at_k, k=25),
    "precision@25": partial(precision_at_k, k=25),
}


def latency_bands_ms(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Return mean and percentile latency bands in milliseconds."""
    values = np.asarray(latencies_ms, dtype=float)
    if values.size == 0:
        raise ValueError("At least one latency measurement is required.")
    return {
        "mean_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
    }


def query_win_counts(per_query: pd.DataFrame) -> dict[str, int]:
    """Count queries where each strategy ties for the best nDCG@10."""
    query_best = per_query.groupby("query_id")["ndcg@10"].transform("max")
    winners = per_query[np.isclose(per_query["ndcg@10"], query_best)]
    return {
        str(name): int(count) for name, count in winners.groupby("search_method").size().items()
    }


def plot_query_ndcg_deltas(
    per_query: pd.DataFrame,
    baseline: str = "bm25_text",
) -> Figure:
    """Plot per-query nDCG@10 changes relative to one baseline strategy."""
    required = {"query_id", "search_method", "ndcg@10"}
    if missing := required - set(per_query.columns):
        raise ValueError(f"Per-query results are missing columns: {sorted(missing)}")

    scores = per_query.pivot(
        index="query_id",
        columns="search_method",
        values="ndcg@10",
    )
    if baseline not in scores:
        raise ValueError(f"Baseline strategy is missing: {baseline}")

    comparisons = [name for name in DEFAULT_CANDIDATES if name != baseline and name in scores]
    if not comparisons:
        raise ValueError("At least one non-baseline strategy is required.")

    deltas = scores[comparisons].subtract(scores[baseline], axis=0)
    order = deltas.mean().sort_values().index.tolist()
    values = [deltas[name].dropna().to_numpy() for name in order]
    positions = np.arange(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    boxes = ax.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        tick_labels=[STRATEGY_LABELS.get(name, name) for name in order],
        medianprops={"color": "#161616", "linewidth": 1.5},
        whiskerprops={"color": "#737373", "linewidth": 1.1},
        capprops={"color": "#737373", "linewidth": 1.1},
    )
    for box, name in zip(boxes["boxes"], order):
        box.set_facecolor(STRATEGY_COLORS.get(name, "#DC382D"))
        box.set_alpha(0.82)
        box.set_edgecolor("none")

    means = [float(deltas[name].mean()) for name in order]
    ax.scatter(
        means,
        positions,
        color="#161616",
        marker="D",
        s=30,
        zorder=3,
        label="Mean",
    )
    ax.axvline(0, color="#161616", linewidth=1.2)
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "How each strategy changes first-page ranking quality",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        "Per-query Δ nDCG@10 versus BM25  ·  positive values are better",
        labelpad=10,
    )
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _document_ids(rows: Sequence[Mapping[str, Any]], id_field: str) -> list[str]:
    values = (row.get(id_field) for row in rows)
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
        if value is not None
    ]


def _build_query(
    strategy: str,
    text_weight: float | None,
    text: str,
    vector: bytes,
    *,
    id_field: str,
    text_field: str,
    vector_field: str,
    ret_k: int,
) -> TextQuery | VectorQuery | HybridQuery:
    common = {"return_fields": [id_field], "num_results": ret_k}
    if strategy == "text":
        return TextQuery(
            text=text,
            text_field_name=text_field,
            stopwords=None,
            **common,
        )
    if strategy == "vector":
        return VectorQuery(
            vector=vector,
            vector_field_name=vector_field,
            **common,
        )
    if strategy not in {"rrf", "linear"}:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    hybrid_options: dict[str, Any] = {
        "combination_method": strategy.upper(),
        "yield_combined_score_as": "hybrid_score",
        "stopwords": None,
    }
    if strategy == "linear":
        hybrid_options["linear_alpha"] = text_weight
    else:
        hybrid_options.update({"rrf_window": ret_k, "rrf_constant": 60})
    return HybridQuery(
        text=text,
        text_field_name=text_field,
        vector=vector,
        vector_field_name=vector_field,
        vector_search_method="KNN",
        knn_ef_runtime=None,
        **hybrid_options,
        **common,
    )


def _scorecard(
    per_query: pd.DataFrame,
    candidates: Mapping[str, tuple[str, float | None]],
) -> pd.DataFrame:
    wins = query_win_counts(per_query)
    rows = []
    for name in candidates:
        method = per_query[per_query["search_method"] == name]
        rows.append(
            {
                "search_method": name,
                **{metric: float(method[metric].mean()) for metric in RANKING_METRICS},
                "ndcg_wins": wins.get(name, 0),
                **latency_bands_ms(method["latency_ms"]),
            }
        )

    scorecard = pd.DataFrame(rows).sort_values(
        ["ndcg@10", "recall@25", "p95_latency_ms"],
        ascending=[False, False, True],
    )
    scorecard = scorecard.reset_index(drop=True)[list(SCORECARD_COLUMNS)]
    return scorecard.round(
        {
            **{metric: 6 for metric in RANKING_METRICS},
            **{column: 3 for column in SCORECARD_COLUMNS if column.endswith("_ms")},
        }
    )


def run_search_comparison(
    *,
    index: Any,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    query_vectors: Mapping[str, bytes],
    candidates: Mapping[str, tuple[str, float | None]] = DEFAULT_CANDIDATES,
    id_field: str = "product_id",
    text_field: str = "search_text",
    vector_field: str = "embedding",
    ret_k: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy over the same queries; return scorecard and query rows."""
    if ret_k != 25:
        raise ValueError("The workshop comparison is defined at ret_k=25.")
    query_ids = set(queries)
    if missing := query_ids - set(qrels):
        raise ValueError(f"Missing qrels for query IDs: {sorted(missing)[:5]}")
    if missing := query_ids - set(query_vectors):
        raise ValueError(f"Missing query vectors for query IDs: {sorted(missing)[:5]}")

    rows = []
    with warnings.catch_warnings():
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
        candidate_items = list(candidates.items())
        print(
            f"Running {len(candidate_items)} strategies over "
            f"{len(queries):,} queries in balanced order..."
        )
        for query_number, (query_id, text) in enumerate(queries.items()):
            offset = query_number % len(candidate_items)
            balanced_items = candidate_items[offset:] + candidate_items[:offset]
            for name, (strategy, text_weight) in balanced_items:
                query = _build_query(
                    strategy,
                    text_weight,
                    str(text),
                    query_vectors[query_id],
                    id_field=id_field,
                    text_field=text_field,
                    vector_field=vector_field,
                    ret_k=ret_k,
                )
                started = perf_counter()
                query_rows = index.query(query)
                latency_ms = (perf_counter() - started) * 1000
                ranked_ids = _document_ids(query_rows, id_field)
                rows.append(
                    {
                        "search_method": name,
                        "query_id": query_id,
                        **{
                            metric: function(qrels[query_id], ranked_ids)
                            for metric, function in RANKING_METRICS.items()
                        },
                        "latency_ms": latency_ms,
                    }
                )
            completed = query_number + 1
            if completed % 100 == 0 or completed == len(queries):
                print(f"- completed {completed:,}/{len(queries):,} queries")

    per_query = pd.DataFrame(rows)
    return _scorecard(per_query, candidates), per_query


def summarize_comparison(scorecard: pd.DataFrame) -> list[str]:
    """Return concise evidence-based conclusions from the scorecard."""
    winner = scorecard.iloc[0]
    fastest = scorecard.loc[scorecard["p95_latency_ms"].idxmin()]
    broadest = scorecard.loc[scorecard["ndcg_wins"].idxmax()]
    baseline = scorecard[scorecard["search_method"] == "bm25_text"].iloc[0]
    return [
        (
            f"{winner['search_method']} leads with nDCG@10="
            f"{winner['ndcg@10']:.4f}, a "
            f"{winner['ndcg@10'] - baseline['ndcg@10']:+.4f} change from BM25."
        ),
        (
            f"{broadest['search_method']} ties for best nDCG@10 on "
            f"{int(broadest['ndcg_wins'])} queries, the broadest result."
        ),
        (
            f"{fastest['search_method']} has the lowest p95 Redis latency at "
            f"{fastest['p95_latency_ms']:.3f} ms in this local sequential run."
        ),
        (
            "Treat small gaps as ties until they repeat on refreshed judgments "
            "and representative production traffic."
        ),
    ]
