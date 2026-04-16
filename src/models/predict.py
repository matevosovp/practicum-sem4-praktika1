"""Prediction helpers shared by training code and API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import PRODUCT_COLUMNS, PRODUCT_NAME_MAP
from src.models.evaluate import build_recommendation_indices


def score_dataframe(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    if hasattr(model, "predict_scores"):
        return np.asarray(model.predict_scores(df), dtype=float)
    scores = model.predict_proba(df[bundle["feature_columns"]])
    if isinstance(scores, list):
        return np.column_stack([column[:, 1] for column in scores])
    return np.asarray(scores, dtype=float)


def recommend_from_scores(scores: np.ndarray, current_products: np.ndarray, k: int) -> list[list[dict[str, Any]]]:
    indices = build_recommendation_indices(scores=scores, current_products=current_products, k=k)
    recommendations: list[list[dict[str, Any]]] = []

    for row_idx, row in enumerate(indices):
        row_recommendations = []
        for product_idx in row:
            product_code = PRODUCT_COLUMNS[product_idx]
            row_recommendations.append(
                {
                    "product_code": product_code,
                    "product_name": PRODUCT_NAME_MAP[product_code],
                    "score": float(scores[row_idx, product_idx]),
                }
            )
        recommendations.append(row_recommendations)
    return recommendations
