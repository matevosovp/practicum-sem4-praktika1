"""Evaluation utilities for ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.constants import PRODUCT_COLUMNS


def precision_at_k(y_true: np.ndarray, recommendations: np.ndarray, k: int) -> float:
    hits = 0.0
    for index, recs in enumerate(recommendations):
        true_items = set(np.flatnonzero(y_true[index]))
        if not recs.size:
            continue
        hits += sum(int(item in true_items) for item in recs[:k]) / k
    return hits / len(recommendations)


def recall_at_k(y_true: np.ndarray, recommendations: np.ndarray, k: int) -> float:
    total = 0.0
    valid_rows = 0
    for index, recs in enumerate(recommendations):
        true_items = set(np.flatnonzero(y_true[index]))
        if not true_items:
            continue
        valid_rows += 1
        total += sum(int(item in true_items) for item in recs[:k]) / len(true_items)
    return total / valid_rows if valid_rows else 0.0


def map_at_k(y_true: np.ndarray, recommendations: np.ndarray, k: int) -> float:
    total = 0.0
    valid_rows = 0
    for index, recs in enumerate(recommendations):
        true_items = set(np.flatnonzero(y_true[index]))
        if not true_items:
            continue
        valid_rows += 1
        hits = 0
        ap = 0.0
        for rank, item_idx in enumerate(recs[:k], start=1):
            if item_idx in true_items:
                hits += 1
                ap += hits / rank
        total += ap / min(len(true_items), k)
    return total / valid_rows if valid_rows else 0.0


def ndcg_at_k(y_true: np.ndarray, recommendations: np.ndarray, k: int) -> float:
    total = 0.0
    valid_rows = 0
    for index, recs in enumerate(recommendations):
        true_items = set(np.flatnonzero(y_true[index]))
        if not true_items:
            continue
        valid_rows += 1
        dcg = 0.0
        for rank, item_idx in enumerate(recs[:k], start=1):
            if item_idx in true_items:
                dcg += 1.0 / math.log2(rank + 1)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(true_items), k) + 1))
        total += dcg / ideal if ideal else 0.0
    return total / valid_rows if valid_rows else 0.0


def build_recommendation_indices(scores: np.ndarray, current_products: np.ndarray, k: int) -> np.ndarray:
    masked_scores = scores.copy()
    masked_scores[current_products == 1] = -1.0
    top_indices = np.argsort(-masked_scores, axis=1)[:, :k]
    return top_indices


def evaluate_rankings(
    y_true: np.ndarray,
    scores: np.ndarray,
    current_products: np.ndarray,
    k: int,
) -> dict[str, float]:
    recommendations = build_recommendation_indices(scores, current_products, k)
    return {
        f"precision_at_{k}": precision_at_k(y_true, recommendations, k),
        f"recall_at_{k}": recall_at_k(y_true, recommendations, k),
        f"map_at_{k}": map_at_k(y_true, recommendations, k),
        f"ndcg_at_{k}": ndcg_at_k(y_true, recommendations, k),
        "rows": float(len(y_true)),
        "rows_with_target": float((y_true.sum(axis=1) > 0).sum()),
    }


def build_error_analysis_frame(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int,
    top_n: int = 30,
) -> pd.DataFrame:
    """Return interpretable examples where true added products were missed."""

    y_true = df[[f"target__{column}" for column in PRODUCT_COLUMNS]].to_numpy()
    current = df[PRODUCT_COLUMNS].to_numpy()
    recs = build_recommendation_indices(scores, current, k)
    rows: list[dict[str, object]] = []

    for row_idx, row in enumerate(recs):
        true_items = set(np.flatnonzero(y_true[row_idx]))
        predicted_items = set(row.tolist())
        missed_items = true_items - predicted_items
        if not missed_items:
            continue
        rows.append(
            {
                "fecha_dato": str(df.iloc[row_idx]["fecha_dato"]),
                "target_month": str(df.iloc[row_idx]["target_month"]),
                "ncodpers": int(df.iloc[row_idx]["ncodpers"]),
                "segmento": df.iloc[row_idx].get("segmento"),
                "products_total": float(df.iloc[row_idx].get("products_total", 0)),
                "true_products": [PRODUCT_COLUMNS[index] for index in sorted(true_items)],
                "recommended_products": [PRODUCT_COLUMNS[index] for index in row.tolist()],
                "missed_products": [PRODUCT_COLUMNS[index] for index in sorted(missed_items)],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["fecha_dato", "target_month", "ncodpers", "segmento", "products_total"])

    return pd.DataFrame(rows).head(top_n)
