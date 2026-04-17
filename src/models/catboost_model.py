"""CatBoost-based multilabel recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
import gc
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


def prepare_catboost_frame(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> pd.DataFrame:
    """Return a compact feature frame ready for CatBoost."""

    frame = df.loc[:, feature_columns].copy()
    categorical_set = set(categorical_feature_columns)

    for column in feature_columns:
        if column in categorical_set:
            frame[column] = frame[column].fillna("UNKNOWN").astype("string")
            continue

        series = frame[column]
        if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            frame[column] = pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")
            continue

        frame[column] = pd.to_numeric(series, errors="coerce").astype("float32")

    return frame


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
    model_filename: str | None = None


class MultiProductCatBoostRecommender:
    """One-vs-rest CatBoost models trained per product."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        categorical_feature_columns: list[str],
        model_params: dict[str, Any],
        random_state: int,
        model_artifact_dir: Path | None = None,
        inference_cache_enabled: bool = False,
    ) -> None:
        self.feature_columns = feature_columns
        self.categorical_feature_columns = categorical_feature_columns
        self.model_params = model_params
        self.random_state = random_state
        self.model_artifact_dir = Path(model_artifact_dir) if model_artifact_dir is not None else None
        self.inference_cache_enabled = inference_cache_enabled
        self.models_: dict[str, CatBoostClassifier] = {}
        self.model_info_: dict[str, ProductModelInfo] = {}
        self._loaded_model_cache: dict[str, CatBoostClassifier] = {}

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_catboost_frame(
            df,
            feature_columns=self.feature_columns,
            categorical_feature_columns=self.categorical_feature_columns,
        )

    def _product_model_path(self, product_code: str) -> Path:
        if self.model_artifact_dir is None:
            raise RuntimeError("Model artifact directory is not configured")
        return self.model_artifact_dir / f"{product_code}.cbm"

    def rebase_artifact_dir(self, new_dir: Path) -> None:
        self.model_artifact_dir = Path(new_dir)
        self.clear_inference_cache()

    def clear_inference_cache(self) -> None:
        self._loaded_model_cache.clear()
        gc.collect()

    def enable_inference_cache(self, preload: bool = False) -> None:
        self.inference_cache_enabled = True
        if preload:
            for product_code in PRODUCT_COLUMNS:
                if not self.model_info_[product_code].is_constant:
                    self._load_product_model(product_code, use_cache=True)

    def _load_product_model(self, product_code: str, *, use_cache: bool | None = None) -> CatBoostClassifier:
        cache_enabled = self.inference_cache_enabled if use_cache is None else use_cache

        if product_code in self.models_:
            return self.models_[product_code]
        if cache_enabled and product_code in self._loaded_model_cache:
            return self._loaded_model_cache[product_code]

        info = self.model_info_[product_code]
        if info.model_filename is None:
            raise RuntimeError(f"Missing model artifact for product {product_code}")
        if self.model_artifact_dir is None:
            raise RuntimeError("Model artifact directory is not configured")

        model = CatBoostClassifier()
        model.load_model(str(self.model_artifact_dir / info.model_filename))
        if cache_enabled:
            self._loaded_model_cache[product_code] = model
        return model

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

        self.models_.clear()
        self.model_info_.clear()
        self.clear_inference_cache()

        if self.model_artifact_dir is not None:
            self.model_artifact_dir.mkdir(parents=True, exist_ok=True)
            for existing_model in self.model_artifact_dir.glob("*.cbm"):
                existing_model.unlink()

        for product_idx, (product_code, target_column) in enumerate(zip(PRODUCT_COLUMNS, target_columns), start=1):
            product_started_at = time.perf_counter()
            train_mask = df[product_code].to_numpy(copy=False) == 0
            product_frame = frame.loc[train_mask]
            product_target = df.loc[train_mask, target_column].astype("int8")
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
                    model_filename=None,
                )
                logger.info(
                    "CatBoost product %s/%s finished | product=%s | constant predictor | elapsed=%.1fs",
                    product_idx,
                    total_products,
                    product_code,
                    time.perf_counter() - product_started_at,
                )
                del product_frame, product_target, train_mask
                gc.collect()
                continue

            fit_kwargs: dict[str, Any] = {"cat_features": self.categorical_feature_columns}
            validation_rows = 0
            product_eval_frame: pd.DataFrame | None = None
            product_eval_target: pd.Series | None = None

            if eval_df is not None and eval_frame is not None:
                eval_mask = eval_df[product_code].to_numpy(copy=False) == 0
                product_eval_frame = eval_frame.loc[eval_mask]
                product_eval_target = eval_df.loc[eval_mask, target_column].astype("int8")
                validation_rows = int(len(product_eval_target))
                if validation_rows and product_eval_target.nunique() >= 2:
                    fit_kwargs["eval_set"] = (product_eval_frame, product_eval_target)
                    fit_kwargs["use_best_model"] = True
                    if early_stopping_rounds is not None:
                        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                del eval_mask

            model = CatBoostClassifier(
                **self.model_params,
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(product_frame, product_target, **fit_kwargs)
            best_iteration = model.get_best_iteration()
            if best_iteration is None or best_iteration < 0:
                best_iteration = model.tree_count_

            model_filename: str | None = None
            if self.model_artifact_dir is None:
                self.models_[product_code] = model
            else:
                model_path = self._product_model_path(product_code)
                model.save_model(str(model_path))
                model_filename = model_path.name

            self.model_info_[product_code] = ProductModelInfo(
                product_code=product_code,
                is_constant=False,
                constant_score=None,
                train_rows=int(len(product_target)),
                positive_rate=positive_rate,
                best_iteration=int(best_iteration),
                validation_rows=validation_rows,
                model_filename=model_filename,
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

            del product_frame, product_target, train_mask
            if product_eval_frame is not None:
                del product_eval_frame
            if product_eval_target is not None:
                del product_eval_target
            if self.model_artifact_dir is not None:
                del model
            gc.collect()

        del frame
        if eval_frame is not None:
            del eval_frame
        gc.collect()
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        frame = self._prepare_frame(df)
        scores = np.zeros((len(frame), len(PRODUCT_COLUMNS)), dtype=np.float32)
        use_cache = self.inference_cache_enabled

        for column_idx, product_code in enumerate(PRODUCT_COLUMNS):
            info = self.model_info_[product_code]
            if info.is_constant:
                scores[:, column_idx] = float(info.constant_score or 0.0)
                continue

            model = self._load_product_model(product_code, use_cache=use_cache)
            scores[:, column_idx] = model.predict_proba(frame)[:, 1].astype("float32")
            if not use_cache and product_code not in self.models_:
                del model
                gc.collect()

        del frame
        gc.collect()
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

            model = self._load_product_model(product_code, use_cache=False)
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
            if product_code not in self.models_:
                del model
                gc.collect()

        return pd.DataFrame(rows)


class RegisteredBundlePyFuncModel(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper around the saved recommendation bundle."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.bundle = joblib.load(context.artifacts["bundle_path"])
        model = self.bundle["model"]
        if hasattr(model, "rebase_artifact_dir"):
            model.rebase_artifact_dir(Path(context.artifacts["model_storage_dir"]))
            model.enable_inference_cache(preload=True)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        model = self.bundle["model"]
        scores = model.predict_scores(model_input)
        return pd.DataFrame(scores, columns=self.bundle["product_columns"])
