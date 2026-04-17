"""CatBoost-based multilabel recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.data.constants import PRODUCT_COLUMNS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ProductModelInfo:
    """Single product model metadata."""

    product_code: str
    is_constant: bool
    constant_score: float | None = None
    train_rows: int = 0
    positive_rate: float = 0.0
    best_iteration: int | None = None
    validation_rows: int = 0


class MultiProductCatBoostRecommender:
    """One-vs-rest CatBoost models trained per product."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        categorical_feature_columns: list[str],
        model_params: dict[str, Any],
        random_state: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.categorical_feature_columns = categorical_feature_columns
        self.model_params = model_params
        self.random_state = random_state
        self.models_: dict[str, CatBoostClassifier] = {}
        self.model_info_: dict[str, ProductModelInfo] = {}

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df[self.feature_columns].copy()
        for column in self.categorical_feature_columns:
            if column in frame.columns:
                frame[column] = frame[column].fillna("UNKNOWN").astype(str)
        return frame

    def fit(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        *,
        eval_df: pd.DataFrame | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "MultiProductCatBoostRecommender":
        frame = self._prepare_frame(df)
        eval_frame = self._prepare_frame(eval_df) if eval_df is not None else None
        total_products = len(PRODUCT_COLUMNS)
        stage_started_at = time.perf_counter()

        for product_idx, (product_code, target_column) in enumerate(zip(PRODUCT_COLUMNS, target_columns), start=1):
            product_started_at = time.perf_counter()
            product_frame = frame.loc[df[product_code] == 0].copy()
            product_target = df.loc[df[product_code] == 0, target_column].astype(int)
            positive_rate = float(product_target.mean()) if len(product_target) else 0.0
            logger.info(
                "CatBoost product %s/%s | product=%s | train_rows=%s | positive_rate=%.5f",
                product_idx,
                total_products,
                product_code,
                len(product_target),
                positive_rate,
            )

            if product_target.nunique() < 2:
                self.model_info_[product_code] = ProductModelInfo(
                    product_code=product_code,
                    is_constant=True,
                    constant_score=positive_rate,
                    train_rows=int(len(product_target)),
                    positive_rate=positive_rate,
                    best_iteration=None,
                    validation_rows=0,
                )
                logger.info(
                    "CatBoost product %s/%s finished | product=%s | constant predictor | elapsed=%.1fs",
                    product_idx,
                    total_products,
                    product_code,
                    time.perf_counter() - product_started_at,
                )
                continue

            fit_kwargs: dict[str, Any] = {"cat_features": self.categorical_feature_columns}
            validation_rows = 0
            if eval_df is not None and eval_frame is not None:
                eval_mask = eval_df[product_code] == 0
                product_eval_frame = eval_frame.loc[eval_mask].copy()
                product_eval_target = eval_df.loc[eval_mask, target_column].astype(int)
                validation_rows = int(len(product_eval_target))
                if validation_rows and product_eval_target.nunique() >= 2:
                    fit_kwargs["eval_set"] = (product_eval_frame, product_eval_target)
                    fit_kwargs["use_best_model"] = True
                    if early_stopping_rounds is not None:
                        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

            model = CatBoostClassifier(
                **self.model_params,
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(product_frame, product_target, **fit_kwargs)
            self.models_[product_code] = model
            best_iteration = model.get_best_iteration()
            if best_iteration is None or best_iteration < 0:
                best_iteration = model.tree_count_
            self.model_info_[product_code] = ProductModelInfo(
                product_code=product_code,
                is_constant=False,
                constant_score=None,
                train_rows=int(len(product_target)),
                positive_rate=positive_rate,
                best_iteration=int(best_iteration),
                validation_rows=validation_rows,
            )
            elapsed = time.perf_counter() - product_started_at
            average_elapsed = (time.perf_counter() - stage_started_at) / product_idx
            remaining = average_elapsed * (total_products - product_idx)
            logger.info(
                "CatBoost product %s/%s finished | product=%s | elapsed=%.1fs | best_iteration=%s | eta_remaining=%.1fs",
                product_idx,
                total_products,
                product_code,
                elapsed,
                best_iteration,
                remaining,
            )

        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        frame = self._prepare_frame(df)
        scores = np.zeros((len(frame), len(PRODUCT_COLUMNS)), dtype=float)
        for column_idx, product_code in enumerate(PRODUCT_COLUMNS):
            info = self.model_info_[product_code]
            if info.is_constant:
                scores[:, column_idx] = float(info.constant_score or 0.0)
                continue
            scores[:, column_idx] = self.models_[product_code].predict_proba(frame)[:, 1]
        return scores

    def get_feature_importance_frame(self, top_n: int = 20) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for product_code in PRODUCT_COLUMNS:
            info = self.model_info_[product_code]
            if info.is_constant:
                rows.append(
                    {
                        "product": product_code,
                        "feature_name": "constant_predictor",
                        "importance": 0.0,
                        "train_rows": info.train_rows,
                        "positive_rate": info.positive_rate,
                    }
                )
                continue

            model = self.models_[product_code]
            importance = model.get_feature_importance(prettified=True).head(top_n)
            for _, row in importance.iterrows():
                rows.append(
                    {
                        "product": product_code,
                        "feature_name": str(row["Feature Id"]),
                        "importance": float(row["Importances"]),
                        "train_rows": info.train_rows,
                        "positive_rate": info.positive_rate,
                    }
                )

        return pd.DataFrame(rows)


class RegisteredBundlePyFuncModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper around the saved recommendation bundle."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.bundle = joblib.load(context.artifacts["bundle_path"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        model = self.bundle["model"]
        scores = model.predict_scores(model_input)
        return pd.DataFrame(scores, columns=self.bundle["product_columns"])
