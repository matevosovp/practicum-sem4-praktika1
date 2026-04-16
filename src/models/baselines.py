"""Baseline recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.constants import PRODUCT_COLUMNS


class GlobalPopularityRecommender:
    """Recommend products by overall next-month adoption frequency."""

    def __init__(self) -> None:
        self.product_scores_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "GlobalPopularityRecommender":
        target_columns = [f"target__{column}" for column in PRODUCT_COLUMNS]
        self.product_scores_ = train_df[target_columns].mean().to_numpy(dtype=float)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.product_scores_ is None:
            raise RuntimeError("Baseline model is not fitted")
        return np.tile(self.product_scores_, (len(df), 1))


class SegmentPopularityRecommender:
    """Recommend by smoothed product popularity inside client segments."""

    def __init__(self, segment_column: str = "segmento", smoothing: float = 50.0) -> None:
        self.segment_column = segment_column
        self.smoothing = smoothing
        self.global_scores_: pd.Series | None = None
        self.segment_scores_: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "SegmentPopularityRecommender":
        target_columns = [f"target__{column}" for column in PRODUCT_COLUMNS]
        self.global_scores_ = train_df[target_columns].mean()
        grouped = train_df.groupby(self.segment_column, dropna=False)[target_columns].agg(["mean", "count"])

        rows = []
        for segment_value, row in grouped.iterrows():
            segment_count = float(row[(target_columns[0], "count")])
            segment_scores = {}
            for target_column in target_columns:
                segment_mean = float(row[(target_column, "mean")])
                global_mean = float(self.global_scores_[target_column])
                smoothed = (segment_mean * segment_count + global_mean * self.smoothing) / (segment_count + self.smoothing)
                segment_scores[target_column] = smoothed
            segment_scores[self.segment_column] = segment_value
            rows.append(segment_scores)

        self.segment_scores_ = pd.DataFrame(rows)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self.segment_scores_ is None or self.global_scores_ is None:
            raise RuntimeError("SegmentPopularityRecommender is not fitted")

        target_columns = [f"target__{column}" for column in PRODUCT_COLUMNS]
        scored = df[[self.segment_column]].merge(self.segment_scores_, on=self.segment_column, how="left")
        for column in target_columns:
            scored[column] = scored[column].fillna(float(self.global_scores_[column]))
        return scored[target_columns].to_numpy(dtype=float)
