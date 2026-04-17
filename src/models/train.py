"""Training entrypoint for staged CatBoost recommendation experiments."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.data.build_dataset import build_eda_artifacts, build_monthly_modeling_dataset, load_modeling_month
from src.data.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PRODUCT_COLUMNS, PRODUCT_NAME_MAP
from src.models.baselines import GlobalPopularityRecommender, SegmentPopularityRecommender
from src.models.catboost_model import MultiProductCatBoostRecommender, RegisteredBundlePyFuncModel
from src.models.evaluate import build_error_analysis_frame, evaluate_rankings
from src.utils.config import ProjectConfig, ensure_directories
from src.utils.logging_utils import get_logger

ENGINEERED_NUMERIC_FEATURES = [
    "month_number",
    "customer_since_months",
    "prev_products_total",
    "products_total",
    "products_added_prev_month",
    "products_dropped_prev_month",
    "has_any_new_product",
]

BASIC_NUMERIC_FEATURES = [*NUMERIC_COLUMNS, *PRODUCT_COLUMNS]
BASIC_CATEGORICAL_FEATURES = [*CATEGORICAL_COLUMNS]
FE_NUMERIC_FEATURES = [*ENGINEERED_NUMERIC_FEATURES, *NUMERIC_COLUMNS, *PRODUCT_COLUMNS]
FE_CATEGORICAL_FEATURES = [*CATEGORICAL_COLUMNS]

logger = get_logger(__name__)


@dataclass(slots=True)
class SplitMetrics:
    split_name: str
    metrics: dict[str, float]


@dataclass(slots=True)
class ExperimentResult:
    stage_name: str
    model_name: str
    model: Any | None
    feature_columns: list[str]
    categorical_feature_columns: list[str]
    valid_metrics: dict[str, float]
    test_metrics: dict[str, float]
    run_id: str | None = None
    params: dict[str, Any] | None = None
    notes: str | None = None
    artifacts: dict[str, str] | None = None
    registry_version: str | None = None


def primary_metric_key(config: ProjectConfig) -> str:
    return f"map_at_{config.top_k}"


def primary_metric_value(config: ProjectConfig, metrics: dict[str, float]) -> float:
    return float(metrics[primary_metric_key(config)])


def sample_evaluation_month(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df

    positives = df.loc[df["target_count"] > 0]
    negatives = df.loc[df["target_count"] == 0]
    positive_sample = min(len(positives), sample_size // 2)
    negative_sample = min(len(negatives), sample_size - positive_sample)

    sampled = pd.concat(
        [
            positives.sample(n=positive_sample, random_state=seed) if positive_sample else positives.iloc[0:0],
            negatives.sample(n=negative_sample, random_state=seed + 1) if negative_sample else negatives.iloc[0:0],
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def sample_training_rows(df: pd.DataFrame, negative_ratio: float, random_state: int) -> pd.DataFrame:
    positives = df.loc[df["target_count"] > 0]
    negatives = df.loc[df["target_count"] == 0]

    if positives.empty:
        sample_size = min(len(negatives), 100_000)
        return negatives.sample(n=sample_size, random_state=random_state)

    negative_sample_size = min(len(negatives), int(len(positives) * negative_ratio))
    sampled_negatives = negatives.sample(n=negative_sample_size, random_state=random_state) if negative_sample_size else negatives.iloc[0:0]
    sampled = pd.concat([positives, sampled_negatives], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def cap_training_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)

    positives = df.loc[df["target_count"] > 0]
    negatives = df.loc[df["target_count"] == 0]

    if len(positives) >= max_rows:
        return positives.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    remaining_negative_budget = max_rows - len(positives)
    sampled_negatives = negatives.sample(n=min(len(negatives), remaining_negative_budget), random_state=random_state + 1)
    capped = pd.concat([positives, sampled_negatives], ignore_index=True)
    return capped.sample(frac=1.0, random_state=random_state + 2).reset_index(drop=True)


def load_sampled_training_frame(
    config: ProjectConfig,
    *,
    months: tuple[str, ...] | None = None,
    negative_ratio: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    selected_months = months or config.train_months
    selected_negative_ratio = config.negative_sample_ratio if negative_ratio is None else negative_ratio

    frames = []
    for offset, month in enumerate(selected_months):
        monthly_df = load_modeling_month(config, month)
        frames.append(sample_training_rows(monthly_df, selected_negative_ratio, config.random_state + offset))

    train_df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        uncapped_rows = len(train_df)
        train_df = cap_training_rows(train_df, max_rows=max_rows, random_state=config.random_state)
        if len(train_df) < uncapped_rows:
            logger.info(
                "Capped sampled training frame | original_rows=%s | capped_rows=%s | max_rows=%s",
                uncapped_rows,
                len(train_df),
                max_rows,
            )

    for column in CATEGORICAL_COLUMNS:
        if column in train_df.columns:
            train_df[column] = train_df[column].fillna("UNKNOWN").astype(str)
    return train_df.reset_index(drop=True)


def load_sampled_evaluation_frames(
    config: ProjectConfig,
    months: tuple[str, ...],
    *,
    sample_size: int,
    seed: int,
) -> list[tuple[str, pd.DataFrame]]:
    monthly_frames: list[tuple[str, pd.DataFrame]] = []

    for offset, month in enumerate(months):
        monthly_df = load_modeling_month(config, month)
        monthly_df = sample_evaluation_month(monthly_df, sample_size=sample_size, seed=seed + offset)
        for column in CATEGORICAL_COLUMNS:
            if column in monthly_df.columns:
                monthly_df[column] = monthly_df[column].fillna("UNKNOWN").astype(str)
        monthly_frames.append((month, monthly_df.reset_index(drop=True)))

    return monthly_frames


def concat_sampled_frames(monthly_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat([frame for _, frame in monthly_frames], ignore_index=True)


def fit_global_baseline(config: ProjectConfig) -> GlobalPopularityRecommender:
    columns = ["target_count", *[f"target__{column}" for column in PRODUCT_COLUMNS]]
    monthly_frames = [load_modeling_month(config, month)[columns] for month in config.train_months]
    return GlobalPopularityRecommender().fit(pd.concat(monthly_frames, ignore_index=True))


def fit_segment_baseline(config: ProjectConfig) -> SegmentPopularityRecommender:
    columns = ["segmento", *[f"target__{column}" for column in PRODUCT_COLUMNS]]
    monthly_frames = [load_modeling_month(config, month)[columns] for month in config.train_months]
    return SegmentPopularityRecommender().fit(pd.concat(monthly_frames, ignore_index=True))


def build_reference_stats(config: ProjectConfig) -> dict[str, Any]:
    train_sample = load_sampled_training_frame(config, max_rows=config.train_max_rows)
    reference_stats: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for column in FE_NUMERIC_FEATURES:
        series = pd.to_numeric(train_sample[column], errors="coerce")
        reference_stats["numeric"][column] = {
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }

    for column in FE_CATEGORICAL_FEATURES:
        top_values = train_sample[column].fillna("UNKNOWN").astype(str).value_counts(normalize=True).head(10)
        reference_stats["categorical"][column] = top_values.to_dict()

    return reference_stats


def build_catboost_model(
    *,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
    params: dict[str, Any],
    random_state: int,
) -> MultiProductCatBoostRecommender:
    return MultiProductCatBoostRecommender(
        feature_columns=feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        model_params=params,
        random_state=random_state,
    )


def predict_scores(model_like: Any, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if hasattr(model_like, "predict_scores"):
        return np.asarray(model_like.predict_scores(df), dtype=float)

    scores = model_like.predict_proba(df[feature_columns])
    if isinstance(scores, list):
        return np.column_stack([column[:, 1] for column in scores])
    return np.asarray(scores, dtype=float)


def evaluate_model_on_frames(
    config: ProjectConfig,
    model_like: Any,
    feature_columns: list[str],
    monthly_frames: list[tuple[str, pd.DataFrame]],
    split_name: str,
) -> tuple[SplitMetrics, pd.DataFrame]:
    metrics_rows = []
    error_frames = []

    for month, monthly_df in monthly_frames:
        scores = predict_scores(model_like, monthly_df, feature_columns)
        metrics = evaluate_rankings(
            y_true=monthly_df[list(config.target_columns)].to_numpy(),
            scores=np.asarray(scores, dtype=float),
            current_products=monthly_df[PRODUCT_COLUMNS].to_numpy(),
            k=config.top_k,
        )
        metrics["snapshot_month"] = month
        metrics_rows.append(metrics)
        error_frame = build_error_analysis_frame(monthly_df, np.asarray(scores, dtype=float), k=config.top_k)
        error_frame["split_name"] = split_name
        error_frame["snapshot_month"] = month
        error_frames.append(error_frame)

    metrics_df = pd.DataFrame(metrics_rows)
    aggregated_metrics = {key: float(metrics_df[key].mean()) for key in metrics_df.columns if key != "snapshot_month"}
    aggregated_metrics["months"] = float(len(monthly_frames))
    return SplitMetrics(split_name=split_name, metrics=aggregated_metrics), pd.concat(error_frames, ignore_index=True)


def evaluate_model_on_months(
    config: ProjectConfig,
    model_like: Any,
    feature_columns: list[str],
    months: tuple[str, ...],
    split_name: str,
    *,
    sample_size_override: int | None = None,
    seed_offset: int = 0,
) -> tuple[SplitMetrics, pd.DataFrame]:
    monthly_frames = load_sampled_evaluation_frames(
        config,
        months,
        sample_size=sample_size_override or config.eval_month_sample_size,
        seed=config.random_state + seed_offset,
    )
    return evaluate_model_on_frames(config, model_like, feature_columns, monthly_frames, split_name)


def _metric_payload(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if isinstance(value, float) and key not in {"rows", "rows_with_target", "months"}
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def release_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()


def save_supporting_artifacts(config: ProjectConfig, reference_stats: dict[str, Any]) -> dict[str, Path]:
    reference_stats_path = config.models_dir / "reference_stats.json"
    product_mapping_path = config.models_dir / "product_mapping.json"

    reference_stats_path.write_text(json.dumps(reference_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    product_mapping_path.write_text(json.dumps(PRODUCT_NAME_MAP, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "reference_stats_path": reference_stats_path,
        "product_mapping_path": product_mapping_path,
    }


def build_split_summary_frame(config: ProjectConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_map = {
        "train": config.train_months,
        "validation": config.valid_months,
        "test": config.test_months,
    }

    for split_name, months in split_map.items():
        for month in months:
            monthly_df = load_modeling_month(config, month)
            rows.append(
                {
                    "split": split_name,
                    "snapshot_month": month,
                    "rows": int(len(monthly_df)),
                    "unique_clients": int(monthly_df["ncodpers"].nunique()),
                    "rows_with_target": int((monthly_df["target_count"] > 0).sum()),
                    "target_events": int(monthly_df["target_count"].sum()),
                    "positive_rate": float((monthly_df["target_count"] > 0).mean()),
                }
            )

    return pd.DataFrame(rows)


def build_experiment_leaderboard_frame(config: ProjectConfig, results: list[ExperimentResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_key = primary_metric_key(config)

    for result in results:
        rows.append(
            {
                "stage_name": result.stage_name,
                "model_name": result.model_name,
                "val_map_at_3": float(result.valid_metrics.get(metric_key, 0.0)),
                "val_precision_at_3": float(result.valid_metrics.get(f"precision_at_{config.top_k}", 0.0)),
                "val_recall_at_3": float(result.valid_metrics.get(f"recall_at_{config.top_k}", 0.0)),
                "val_ndcg_at_3": float(result.valid_metrics.get(f"ndcg_at_{config.top_k}", 0.0)),
                "test_map_at_3": float(result.test_metrics.get(metric_key, 0.0)),
                "test_precision_at_3": float(result.test_metrics.get(f"precision_at_{config.top_k}", 0.0)),
                "test_recall_at_3": float(result.test_metrics.get(f"recall_at_{config.top_k}", 0.0)),
                "test_ndcg_at_3": float(result.test_metrics.get(f"ndcg_at_{config.top_k}", 0.0)),
                "run_id": result.run_id,
                "notes": result.notes,
            }
        )

    return pd.DataFrame(rows).sort_values("val_map_at_3", ascending=False).reset_index(drop=True)


def save_stage_bundle(
    config: ProjectConfig,
    *,
    stage_name: str,
    model_name: str,
    model: Any,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
    reference_stats: dict[str, Any],
    params: dict[str, Any],
    notes: str,
) -> Path:
    bundle_path = config.models_dir / f"{stage_name}.joblib"
    bundle = {
        "model": model,
        "bundle_type": model_name,
        "stage_name": stage_name,
        "feature_columns": feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "product_columns": PRODUCT_COLUMNS,
        "top_k": config.top_k,
        "reference_stats": reference_stats,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "primary_metric_name": config.primary_metric_name,
        "notes": notes,
    }
    save_bundle(bundle, bundle_path)
    return bundle_path


def log_baseline_run(
    config: ProjectConfig,
    *,
    stage_name: str,
    model_name: str,
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
    notes: str,
    model_path: Path,
) -> ExperimentResult:
    with mlflow.start_run(run_name=stage_name) as run:
        mlflow.log_params(
            {
                "stage_name": stage_name,
                "model_family": "baseline",
                "model_type": model_name,
                "top_k": config.top_k,
                "primary_metric_name": config.primary_metric_name,
                "train_months": ",".join(config.train_months),
                "valid_months": ",".join(config.valid_months),
                "test_months": ",".join(config.test_months),
                "eval_month_sample_size": config.eval_month_sample_size,
                "notes": notes,
            }
        )
        mlflow.log_metrics(_metric_payload("val", valid_metrics))
        mlflow.log_metrics(_metric_payload("test", test_metrics))
        mlflow.log_metric(config.primary_metric_name, primary_metric_value(config, valid_metrics))
        mlflow.log_artifact(str(model_path))

        return ExperimentResult(
            stage_name=stage_name,
            model_name=model_name,
            model=None,
            feature_columns=[],
            categorical_feature_columns=[],
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            run_id=run.info.run_id,
            params={},
            notes=notes,
            artifacts={"bundle_path": str(model_path)},
        )


def log_catboost_run(
    config: ProjectConfig,
    *,
    stage_name: str,
    model_name: str,
    model: MultiProductCatBoostRecommender,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
    valid_metrics: dict[str, float],
    test_metrics: dict[str, float],
    valid_errors: pd.DataFrame,
    test_errors: pd.DataFrame,
    reference_stats: dict[str, Any],
    params: dict[str, Any],
    notes: str,
    extra_artifacts: dict[str, Path] | None = None,
) -> ExperimentResult:
    bundle_path = save_stage_bundle(
        config,
        stage_name=stage_name,
        model_name=model_name,
        model=model,
        feature_columns=feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        reference_stats=reference_stats,
        params=params,
        notes=notes,
    )

    feature_importance_path = config.models_dir / f"{stage_name}_feature_importance.csv"
    valid_errors_path = config.models_dir / f"{stage_name}_validation_errors.csv"
    test_errors_path = config.models_dir / f"{stage_name}_test_errors.csv"
    feature_list_path = config.models_dir / f"{stage_name}_feature_list.json"
    categorical_features_path = config.models_dir / f"{stage_name}_categorical_features.json"

    model.get_feature_importance_frame().to_csv(feature_importance_path, index=False)
    valid_errors.to_csv(valid_errors_path, index=False)
    test_errors.to_csv(test_errors_path, index=False)
    feature_list_path.write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
    categorical_features_path.write_text(json.dumps(categorical_feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_paths: dict[str, Path] = {
        "bundle_path": bundle_path,
        "feature_importance_path": feature_importance_path,
        "validation_errors_path": valid_errors_path,
        "test_errors_path": test_errors_path,
        "feature_list_path": feature_list_path,
        "categorical_features_path": categorical_features_path,
    }
    if extra_artifacts:
        artifact_paths.update(extra_artifacts)

    with mlflow.start_run(run_name=stage_name) as run:
        mlflow.log_params(
            {
                "stage_name": stage_name,
                "model_family": "catboost",
                "model_type": model_name,
                "top_k": config.top_k,
                "primary_metric_name": config.primary_metric_name,
                "train_months": ",".join(config.train_months),
                "valid_months": ",".join(config.valid_months),
                "test_months": ",".join(config.test_months),
                "eval_month_sample_size": config.eval_month_sample_size,
                "fit_eval_size": config.catboost_fit_eval_size,
                "early_stopping_rounds": config.catboost_early_stopping_rounds,
                "num_features": len(feature_columns),
                "num_categorical_features": len(categorical_feature_columns),
                "num_train_rows": sum(info.train_rows for info in model.model_info_.values()),
                "notes": notes,
                **params,
            }
        )
        mlflow.log_metrics(_metric_payload("val", valid_metrics))
        mlflow.log_metrics(_metric_payload("test", test_metrics))
        mlflow.log_metric(config.primary_metric_name, primary_metric_value(config, valid_metrics))
        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(str(artifact_path))

        return ExperimentResult(
            stage_name=stage_name,
            model_name=model_name,
            model=None,
            feature_columns=feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            run_id=run.info.run_id,
            params=params,
            notes=notes,
            artifacts={name: str(path) for name, path in artifact_paths.items()},
        )


def build_tuning_candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "iterations": 160,
            "depth": 5,
            "learning_rate": 0.08,
            "l2_leaf_reg": 4.0,
            "min_data_in_leaf": 48,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.0,
        },
        {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.06,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 32,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
        {
            "iterations": 240,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6.0,
            "min_data_in_leaf": 48,
            "random_strength": 1.2,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
        },
        {
            "iterations": 220,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 32,
            "random_strength": 1.5,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
        },
        {
            "iterations": 150,
            "depth": 6,
            "learning_rate": 0.09,
            "l2_leaf_reg": 7.0,
            "min_data_in_leaf": 64,
            "random_strength": 0.7,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.2,
        },
    ]


def tuning_common_params(config: ProjectConfig) -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "auto_class_weights": "Balanced",
        "thread_count": config.catboost_thread_count,
        "used_ram_limit": config.catboost_ram_limit,
    }


def flatten_tuning_result(
    config: ProjectConfig,
    *,
    candidate_id: int,
    candidate_name: str,
    search_stage: str,
    params: dict[str, Any],
    valid_metrics: dict[str, float],
    fit_rows: int,
    eval_rows: int,
    fit_seconds: float,
    run_id: str | None,
    screening_rank: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "search_stage": search_stage,
        "run_id": run_id,
        "fit_rows": fit_rows,
        "eval_rows": eval_rows,
        "fit_seconds": round(fit_seconds, 2),
        config.primary_metric_name: primary_metric_value(config, valid_metrics),
        f"val_precision_at_{config.top_k}": float(valid_metrics[f"precision_at_{config.top_k}"]),
        f"val_recall_at_{config.top_k}": float(valid_metrics[f"recall_at_{config.top_k}"]),
        f"val_ndcg_at_{config.top_k}": float(valid_metrics[f"ndcg_at_{config.top_k}"]),
    }
    if screening_rank is not None:
        payload["screening_rank"] = screening_rank
    payload.update(params)
    return payload


def log_tuning_candidate_run(
    config: ProjectConfig,
    *,
    candidate_name: str,
    search_stage: str,
    params: dict[str, Any],
    valid_metrics: dict[str, float],
    fit_rows: int,
    eval_rows: int,
    fit_seconds: float,
    notes: str,
    screening_rank: int | None = None,
) -> str:
    with mlflow.start_run(run_name=candidate_name) as run:
        mlflow.set_tags(
            {
                "run_kind": "tuning_candidate",
                "search_stage": search_stage,
                "selection_metric": config.primary_metric_name,
            }
        )
        mlflow.log_params(
            {
                "stage_name": "stage_04_catboost_tuned",
                "model_family": "catboost",
                "model_type": "catboost_tuning_candidate",
                "candidate_name": candidate_name,
                "search_stage": search_stage,
                "top_k": config.top_k,
                "primary_metric_name": config.primary_metric_name,
                "fit_rows": fit_rows,
                "eval_rows": eval_rows,
                "fit_eval_size": config.catboost_fit_eval_size,
                "early_stopping_rounds": config.catboost_early_stopping_rounds,
                "notes": notes,
                **({"screening_rank": screening_rank} if screening_rank is not None else {}),
                **params,
            }
        )
        mlflow.log_metrics(_metric_payload("val", valid_metrics))
        mlflow.log_metric(config.primary_metric_name, primary_metric_value(config, valid_metrics))
        mlflow.log_metric("fit_seconds", float(fit_seconds))
        return run.info.run_id


def run_two_stage_tuning(
    config: ProjectConfig,
    *,
    full_train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> tuple[MultiProductCatBoostRecommender, dict[str, Any], SplitMetrics, pd.DataFrame, dict[str, Path]]:
    common_params = tuning_common_params(config)
    stage_a_candidates = build_tuning_candidate_grid()

    logger.info(
        "Stage A tuning | months=%s | max_rows=%s | candidate_count=%s",
        ",".join(config.tuning_stage_a_months),
        config.tuning_stage_a_max_rows,
        len(stage_a_candidates),
    )
    stage_a_train_df = load_sampled_training_frame(
        config,
        months=config.tuning_stage_a_months,
        max_rows=config.tuning_stage_a_max_rows,
    )
    stage_a_eval_frames = load_sampled_evaluation_frames(
        config,
        config.valid_months,
        sample_size=config.tuning_stage_a_eval_size,
        seed=config.random_state + 2_000,
    )
    stage_a_eval_df = concat_sampled_frames(stage_a_eval_frames)

    stage_a_results: list[dict[str, Any]] = []
    metric_column = config.primary_metric_name

    for candidate_idx, candidate in enumerate(stage_a_candidates, start=1):
        candidate_name = f"stage_04a_candidate_{candidate_idx:02d}"
        candidate_params = {**common_params, **candidate}
        logger.info("Stage A candidate %s/%s | %s", candidate_idx, len(stage_a_candidates), json.dumps(candidate, ensure_ascii=False))

        model = build_catboost_model(
            feature_columns=feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            params=candidate_params,
            random_state=config.random_state,
        )
        fit_started_at = time.perf_counter()
        model.fit(
            stage_a_train_df,
            list(config.target_columns),
            eval_df=stage_a_eval_df,
            early_stopping_rounds=config.catboost_early_stopping_rounds,
        )
        fit_seconds = time.perf_counter() - fit_started_at
        valid_metrics, _ = evaluate_model_on_frames(config, model, feature_columns, stage_a_eval_frames, "valid")
        run_id = log_tuning_candidate_run(
            config,
            candidate_name=candidate_name,
            search_stage="stage_a",
            params=candidate_params,
            valid_metrics=valid_metrics.metrics,
            fit_rows=len(stage_a_train_df),
            eval_rows=len(stage_a_eval_df),
            fit_seconds=fit_seconds,
            notes="Fast screening on recent train months with sampled validation and early stopping.",
        )
        stage_a_results.append(
            flatten_tuning_result(
                config,
                candidate_id=candidate_idx,
                candidate_name=candidate_name,
                search_stage="stage_a",
                params=candidate,
                valid_metrics=valid_metrics.metrics,
                fit_rows=len(stage_a_train_df),
                eval_rows=len(stage_a_eval_df),
                fit_seconds=fit_seconds,
                run_id=run_id,
            )
        )
        release_memory(model, valid_metrics)

    stage_a_df = pd.DataFrame(stage_a_results).sort_values(metric_column, ascending=False).reset_index(drop=True)
    stage_a_df["screening_rank"] = np.arange(1, len(stage_a_df) + 1)
    top_stage_a = stage_a_df.head(config.tuning_stage_b_top_n).copy()

    logger.info(
        "Stage B tuning | top_n=%s | candidates=%s",
        config.tuning_stage_b_top_n,
        ", ".join(top_stage_a["candidate_name"].tolist()),
    )
    stage_b_eval_frames = load_sampled_evaluation_frames(
        config,
        config.valid_months,
        sample_size=config.eval_month_sample_size,
        seed=config.random_state + 3_000,
    )
    stage_b_eval_df = concat_sampled_frames(stage_b_eval_frames)

    stage_b_results: list[dict[str, Any]] = []
    best_model: MultiProductCatBoostRecommender | None = None
    best_params: dict[str, Any] | None = None
    best_valid_metrics: SplitMetrics | None = None
    best_valid_errors: pd.DataFrame | None = None
    best_metric = float("-inf")

    for _, stage_a_row in top_stage_a.iterrows():
        candidate_name = stage_a_row["candidate_name"].replace("stage_04a", "stage_04b")
        candidate = stage_a_candidates[int(stage_a_row["candidate_id"]) - 1]
        candidate_params = {**common_params, **candidate}
        logger.info(
            "Stage B candidate | source=%s | screening_rank=%s",
            stage_a_row["candidate_name"],
            int(stage_a_row["screening_rank"]),
        )

        model = build_catboost_model(
            feature_columns=feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            params=candidate_params,
            random_state=config.random_state,
        )
        fit_started_at = time.perf_counter()
        model.fit(
            full_train_df,
            list(config.target_columns),
            eval_df=stage_b_eval_df,
            early_stopping_rounds=config.catboost_early_stopping_rounds,
        )
        fit_seconds = time.perf_counter() - fit_started_at
        valid_metrics, valid_errors = evaluate_model_on_frames(config, model, feature_columns, stage_b_eval_frames, "valid")
        run_id = log_tuning_candidate_run(
            config,
            candidate_name=candidate_name,
            search_stage="stage_b",
            params=candidate_params,
            valid_metrics=valid_metrics.metrics,
            fit_rows=len(full_train_df),
            eval_rows=len(stage_b_eval_df),
            fit_seconds=fit_seconds,
            notes="Full confirmation pass on the complete train sample with the same time-based validation.",
            screening_rank=int(stage_a_row["screening_rank"]),
        )
        stage_b_results.append(
            flatten_tuning_result(
                config,
                candidate_id=int(stage_a_row["candidate_id"]),
                candidate_name=candidate_name,
                search_stage="stage_b",
                params=candidate,
                valid_metrics=valid_metrics.metrics,
                fit_rows=len(full_train_df),
                eval_rows=len(stage_b_eval_df),
                fit_seconds=fit_seconds,
                run_id=run_id,
                screening_rank=int(stage_a_row["screening_rank"]),
            )
        )

        candidate_metric = primary_metric_value(config, valid_metrics.metrics)
        if candidate_metric > best_metric:
            if best_model is not None and best_valid_metrics is not None and best_valid_errors is not None:
                release_memory(best_model, best_valid_metrics, best_valid_errors)
            best_metric = candidate_metric
            best_model = model
            best_params = candidate_params
            best_valid_metrics = valid_metrics
            best_valid_errors = valid_errors
        else:
            release_memory(model, valid_metrics, valid_errors)

    stage_a_path = config.models_dir / "stage_04_stage_a_leaderboard.csv"
    stage_b_path = config.models_dir / "stage_04_stage_b_leaderboard.csv"
    stage_summary_path = config.models_dir / "stage_04_tuning_summary.json"
    stage_a_df.to_csv(stage_a_path, index=False)
    pd.DataFrame(stage_b_results).sort_values(metric_column, ascending=False).to_csv(stage_b_path, index=False)
    stage_summary_path.write_text(
        json.dumps(
            {
                "strategy": "two_stage_manual_screening",
                "primary_metric_name": config.primary_metric_name,
                "stage_a_train_months": list(config.tuning_stage_a_months),
                "stage_a_candidate_count": len(stage_a_candidates),
                "stage_b_top_n": config.tuning_stage_b_top_n,
                "stage_a_sample_rows": int(len(stage_a_train_df)),
                "stage_a_eval_rows": int(len(stage_a_eval_df)),
                "stage_b_eval_rows": int(len(stage_b_eval_df)),
                "best_candidate_metric": float(best_metric),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if best_model is None or best_params is None or best_valid_metrics is None or best_valid_errors is None:
        raise RuntimeError("Two-stage tuning did not produce a final CatBoost candidate")

    release_memory(stage_a_train_df, stage_a_eval_df, stage_b_eval_df, stage_a_df, top_stage_a)
    return (
        best_model,
        best_params,
        best_valid_metrics,
        best_valid_errors,
        {
            "stage_a_leaderboard_path": stage_a_path,
            "stage_b_leaderboard_path": stage_b_path,
            "tuning_summary_path": stage_summary_path,
        },
    )


def register_best_model(
    config: ProjectConfig,
    result: ExperimentResult,
    metadata_path: Path,
    input_example: pd.DataFrame,
) -> str:
    if result.run_id is None:
        raise RuntimeError("Cannot register a model without MLflow run_id")
    if not result.artifacts or "bundle_path" not in result.artifacts:
        raise RuntimeError("Cannot register a model without saved bundle artifact")

    client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
    bundle_path = result.artifacts["bundle_path"]
    bundle = joblib.load(bundle_path)
    output_example = pd.DataFrame(bundle["model"].predict_scores(input_example), columns=PRODUCT_COLUMNS)
    signature = infer_signature(input_example, output_example)

    logger.info("Logging registry-ready pyfunc model into winning run | run_id=%s", result.run_id)
    with mlflow.start_run(run_id=result.run_id):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RegisteredBundlePyFuncModel(),
            artifacts={"bundle_path": str(bundle_path)},
            input_example=input_example,
            signature=signature,
        )

    release_memory(bundle, output_example)

    model_uri = f"runs:/{result.run_id}/model"
    registration = mlflow.register_model(model_uri=model_uri, name=config.mlflow_registered_model_name)
    client.set_registered_model_alias(config.mlflow_registered_model_name, config.mlflow_model_alias, registration.version)
    client.set_model_version_tag(config.mlflow_registered_model_name, registration.version, "stage_name", result.stage_name)
    client.set_model_version_tag(config.mlflow_registered_model_name, registration.version, "primary_metric_name", config.primary_metric_name)
    client.set_model_version_tag(
        config.mlflow_registered_model_name,
        registration.version,
        config.primary_metric_name,
        str(primary_metric_value(config, result.valid_metrics)),
    )
    client.set_model_version_tag(config.mlflow_registered_model_name, registration.version, "metadata_path", str(metadata_path))
    return str(registration.version)


def build_metadata(
    config: ProjectConfig,
    results: list[ExperimentResult],
    selected_result: ExperimentResult,
) -> Path:
    stages = {}
    for result in results:
        stages[result.stage_name] = {
            "model_name": result.model_name,
            "valid": result.valid_metrics,
            "test": result.test_metrics,
            "run_id": result.run_id,
            "params": result.params,
            "notes": result.notes,
            "artifacts": result.artifacts,
            "registry_version": result.registry_version,
        }

    metadata = {
        "selected_model": selected_result.stage_name,
        "primary_metric_name": config.primary_metric_name,
        "registered_model_name": config.mlflow_registered_model_name,
        "registered_model_alias": config.mlflow_model_alias,
        "stages": stages,
    }
    metadata_path = config.models_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def run_training(config: ProjectConfig, force_prepare: bool = False) -> dict[str, Any]:
    started_at = time.perf_counter()
    ensure_directories(config)
    logger.info("Preparing directories and monthly datasets...")
    build_monthly_modeling_dataset(config, force=force_prepare)
    build_eda_artifacts(config)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    logger.info(
        "Loading sampled training frame | train_months=%s | negative_sample_ratio=%.3f | eval_month_sample_size=%s",
        ",".join(config.train_months),
        config.negative_sample_ratio,
        config.eval_month_sample_size,
    )
    train_df = load_sampled_training_frame(config, max_rows=config.train_max_rows)
    reference_stats = build_reference_stats(config)
    support_artifacts = save_supporting_artifacts(config, reference_stats)
    fit_eval_frames = load_sampled_evaluation_frames(
        config,
        config.valid_months,
        sample_size=config.catboost_fit_eval_size,
        seed=config.random_state + 1_000,
    )
    fit_eval_df = concat_sampled_frames(fit_eval_frames)

    experiment_results: list[ExperimentResult] = []
    logger.info("Training frame ready | rows=%s | columns=%s", len(train_df), len(train_df.columns))

    total_stages = 5

    def log_stage(stage_idx: int, stage_name: str, message: str) -> None:
        elapsed = time.perf_counter() - started_at
        logger.info("[%s/%s] %s | %s | elapsed=%s", stage_idx, total_stages, stage_name, message, format_seconds(elapsed))

    log_stage(1, "stage_00_global_popularity", "fitting baseline")
    global_baseline = fit_global_baseline(config)
    global_valid_metrics, _ = evaluate_model_on_months(config, global_baseline, [], config.valid_months, "valid")
    global_test_metrics, _ = evaluate_model_on_months(config, global_baseline, [], config.test_months, "test")
    global_bundle_path = save_stage_bundle(
        config,
        stage_name="stage_00_global_popularity",
        model_name="global_popularity",
        model=global_baseline,
        feature_columns=[],
        categorical_feature_columns=[],
        valid_metrics=global_valid_metrics.metrics,
        test_metrics=global_test_metrics.metrics,
        reference_stats=reference_stats,
        params={},
        notes="Global popularity baseline for new product adoption.",
    )
    experiment_results.append(
        log_baseline_run(
            config,
            stage_name="stage_00_global_popularity",
            model_name="global_popularity",
            valid_metrics=global_valid_metrics.metrics,
            test_metrics=global_test_metrics.metrics,
            notes="Global popularity baseline for new product adoption.",
            model_path=global_bundle_path,
        )
    )
    log_stage(1, "stage_00_global_popularity", f"done | {config.primary_metric_name}={primary_metric_value(config, global_valid_metrics.metrics):.5f}")
    release_memory(global_baseline)

    log_stage(2, "stage_01_segment_popularity", "fitting baseline")
    segment_baseline = fit_segment_baseline(config)
    segment_valid_metrics, _ = evaluate_model_on_months(config, segment_baseline, [], config.valid_months, "valid")
    segment_test_metrics, _ = evaluate_model_on_months(config, segment_baseline, [], config.test_months, "test")
    segment_bundle_path = save_stage_bundle(
        config,
        stage_name="stage_01_segment_popularity",
        model_name="segment_popularity",
        model=segment_baseline,
        feature_columns=[],
        categorical_feature_columns=[],
        valid_metrics=segment_valid_metrics.metrics,
        test_metrics=segment_test_metrics.metrics,
        reference_stats=reference_stats,
        params={"segment_column": "segmento"},
        notes="Segment-smoothed popularity baseline by `segmento`.",
    )
    experiment_results.append(
        log_baseline_run(
            config,
            stage_name="stage_01_segment_popularity",
            model_name="segment_popularity",
            valid_metrics=segment_valid_metrics.metrics,
            test_metrics=segment_test_metrics.metrics,
            notes="Segment-smoothed popularity baseline by `segmento`.",
            model_path=segment_bundle_path,
        )
    )
    log_stage(2, "stage_01_segment_popularity", f"done | {config.primary_metric_name}={primary_metric_value(config, segment_valid_metrics.metrics):.5f}")
    release_memory(segment_baseline)

    basic_feature_columns = [*BASIC_NUMERIC_FEATURES, *BASIC_CATEGORICAL_FEATURES]
    basic_params = {
        "iterations": 160,
        "depth": 6,
        "learning_rate": 0.07,
        "l2_leaf_reg": 4.0,
        "min_data_in_leaf": 48,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        **tuning_common_params(config),
    }
    log_stage(3, "stage_02_catboost_basic", f"training CatBoost with {len(basic_feature_columns)} features")
    basic_model = build_catboost_model(
        feature_columns=basic_feature_columns,
        categorical_feature_columns=BASIC_CATEGORICAL_FEATURES,
        params=basic_params,
        random_state=config.random_state,
    )
    basic_model.fit(
        train_df,
        list(config.target_columns),
        eval_df=fit_eval_df,
        early_stopping_rounds=config.catboost_early_stopping_rounds,
    )
    basic_valid_metrics, basic_valid_errors = evaluate_model_on_months(config, basic_model, basic_feature_columns, config.valid_months, "valid")
    basic_test_metrics, basic_test_errors = evaluate_model_on_months(config, basic_model, basic_feature_columns, config.test_months, "test")
    experiment_results.append(
        log_catboost_run(
            config,
            stage_name="stage_02_catboost_basic",
            model_name="catboost_basic",
            model=basic_model,
            feature_columns=basic_feature_columns,
            categorical_feature_columns=BASIC_CATEGORICAL_FEATURES,
            valid_metrics=basic_valid_metrics.metrics,
            test_metrics=basic_test_metrics.metrics,
            valid_errors=basic_valid_errors,
            test_errors=basic_test_errors,
            reference_stats=reference_stats,
            params=basic_params,
            notes="CatBoost on raw profile and current products with a fixed validation eval_set and early stopping.",
        )
    )
    log_stage(3, "stage_02_catboost_basic", f"done | {config.primary_metric_name}={primary_metric_value(config, basic_valid_metrics.metrics):.5f}")
    release_memory(basic_model, basic_valid_errors, basic_test_errors)

    fe_feature_columns = [*FE_NUMERIC_FEATURES, *FE_CATEGORICAL_FEATURES]
    fe_params = {
        "iterations": 260,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "min_data_in_leaf": 32,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        **tuning_common_params(config),
    }
    log_stage(4, "stage_03_catboost_feature_engineering", f"training CatBoost with {len(fe_feature_columns)} features")
    fe_model = build_catboost_model(
        feature_columns=fe_feature_columns,
        categorical_feature_columns=FE_CATEGORICAL_FEATURES,
        params=fe_params,
        random_state=config.random_state,
    )
    fe_model.fit(
        train_df,
        list(config.target_columns),
        eval_df=fit_eval_df,
        early_stopping_rounds=config.catboost_early_stopping_rounds,
    )
    fe_valid_metrics, fe_valid_errors = evaluate_model_on_months(config, fe_model, fe_feature_columns, config.valid_months, "valid")
    fe_test_metrics, fe_test_errors = evaluate_model_on_months(config, fe_model, fe_feature_columns, config.test_months, "test")
    experiment_results.append(
        log_catboost_run(
            config,
            stage_name="stage_03_catboost_feature_engineering",
            model_name="catboost_feature_engineering",
            model=fe_model,
            feature_columns=fe_feature_columns,
            categorical_feature_columns=FE_CATEGORICAL_FEATURES,
            valid_metrics=fe_valid_metrics.metrics,
            test_metrics=fe_test_metrics.metrics,
            valid_errors=fe_valid_errors,
            test_errors=fe_test_errors,
            reference_stats=reference_stats,
            params=fe_params,
            notes="CatBoost after temporal and portfolio-delta feature engineering, still using time-based eval_set and early stopping.",
        )
    )
    log_stage(4, "stage_03_catboost_feature_engineering", f"done | {config.primary_metric_name}={primary_metric_value(config, fe_valid_metrics.metrics):.5f}")
    release_memory(fe_model, fe_valid_errors, fe_test_errors)

    log_stage(5, "stage_04_catboost_tuned", "running two-stage hyperparameter selection")
    if config.catboost_tuning_enabled:
        tuned_model, tuned_params, tuned_valid_metrics, tuned_valid_errors, tuning_artifacts = run_two_stage_tuning(
            config,
            full_train_df=train_df,
            feature_columns=fe_feature_columns,
            categorical_feature_columns=FE_CATEGORICAL_FEATURES,
        )
        tuned_notes = (
            "Two-stage CatBoost tuning: Stage A screens a small manual candidate set on recent train months, "
            "Stage B confirms top candidates on the full train sample. All candidates are logged to MLflow and "
            f"the final choice is made by {config.primary_metric_name}."
        )
        tuned_result_params = {
            **tuned_params,
            "tuning_strategy": "two_stage_manual_screening",
            "tuning_stage_a_candidate_count": len(build_tuning_candidate_grid()),
            "tuning_stage_b_top_n": config.tuning_stage_b_top_n,
            "tuning_stage_a_months": ",".join(config.tuning_stage_a_months),
        }
    else:
        tuned_model = build_catboost_model(
            feature_columns=fe_feature_columns,
            categorical_feature_columns=FE_CATEGORICAL_FEATURES,
            params=fe_params,
            random_state=config.random_state,
        )
        tuned_model.fit(
            train_df,
            list(config.target_columns),
            eval_df=fit_eval_df,
            early_stopping_rounds=config.catboost_early_stopping_rounds,
        )
        tuned_valid_metrics, tuned_valid_errors = evaluate_model_on_months(config, tuned_model, fe_feature_columns, config.valid_months, "valid")
        tuning_artifacts = {}
        tuned_notes = "Hyperparameter search disabled. Final CatBoost stage reuses the engineered-feature configuration with early stopping."
        tuned_result_params = {**fe_params, "tuning_strategy": "disabled"}

    tuned_test_metrics, tuned_test_errors = evaluate_model_on_months(config, tuned_model, fe_feature_columns, config.test_months, "test")
    tuned_result = log_catboost_run(
        config,
        stage_name="stage_04_catboost_tuned",
        model_name="catboost_tuned",
        model=tuned_model,
        feature_columns=fe_feature_columns,
        categorical_feature_columns=FE_CATEGORICAL_FEATURES,
        valid_metrics=tuned_valid_metrics.metrics,
        test_metrics=tuned_test_metrics.metrics,
        valid_errors=tuned_valid_errors,
        test_errors=tuned_test_errors,
        reference_stats=reference_stats,
        params=tuned_result_params,
        notes=tuned_notes,
        extra_artifacts={**tuning_artifacts, **support_artifacts},
    )
    experiment_results.append(tuned_result)
    log_stage(5, "stage_04_catboost_tuned", f"done | {config.primary_metric_name}={primary_metric_value(config, tuned_valid_metrics.metrics):.5f}")
    release_memory(tuned_model, tuned_valid_errors, tuned_test_errors)

    split_summary_path = config.models_dir / "split_summary.csv"
    experiment_leaderboard_path = config.models_dir / "experiment_leaderboard.csv"
    build_split_summary_frame(config).to_csv(split_summary_path, index=False)
    build_experiment_leaderboard_frame(config, experiment_results).to_csv(experiment_leaderboard_path, index=False)

    catboost_results = [result for result in experiment_results if result.model_name.startswith("catboost")]
    selected_result = max(catboost_results, key=lambda item: primary_metric_value(config, item.valid_metrics))
    logger.info(
        "Selecting best CatBoost stage | stage=%s | %s=%.5f",
        selected_result.stage_name,
        config.primary_metric_name,
        primary_metric_value(config, selected_result.valid_metrics),
    )

    metadata_path = build_metadata(config, experiment_results, selected_result)
    logger.info("Registering model in MLflow Model Registry | name=%s | alias=%s", config.mlflow_registered_model_name, config.mlflow_model_alias)
    registry_input_example = train_df[selected_result.feature_columns].head(10).copy()
    selected_result.registry_version = register_best_model(config, selected_result, metadata_path, registry_input_example)
    metadata_path = build_metadata(config, experiment_results, selected_result)
    release_memory(registry_input_example, fit_eval_df)

    selected_bundle_path = Path(selected_result.artifacts["bundle_path"]) if selected_result.artifacts else config.models_dir / f"{selected_result.stage_name}.joblib"
    best_bundle = joblib.load(selected_bundle_path)
    best_bundle["selected_model"] = selected_result.stage_name
    best_bundle["registered_model_name"] = config.mlflow_registered_model_name
    best_bundle["registered_model_alias"] = config.mlflow_model_alias
    best_bundle["registered_model_version"] = selected_result.registry_version
    best_bundle["metrics"] = json.loads(metadata_path.read_text(encoding="utf-8"))["stages"]
    best_bundle["artifact_paths"] = {
        **{name: str(path) for name, path in support_artifacts.items()},
        "split_summary_path": str(split_summary_path),
        "experiment_leaderboard_path": str(experiment_leaderboard_path),
    }
    save_bundle(best_bundle, config.models_dir / "best_model.joblib")

    feature_list_path = config.models_dir / "feature_list.json"
    categorical_features_path = config.models_dir / "categorical_features.json"
    feature_list_path.write_text(json.dumps(selected_result.feature_columns, indent=2, ensure_ascii=False), encoding="utf-8")
    categorical_features_path.write_text(json.dumps(selected_result.categorical_feature_columns, indent=2, ensure_ascii=False), encoding="utf-8")

    release_memory(train_df)

    total_elapsed = time.perf_counter() - started_at
    logger.info(
        "Training pipeline completed | selected=%s | registry_version=%s | total_elapsed=%s",
        selected_result.stage_name,
        selected_result.registry_version,
        format_seconds(total_elapsed),
    )

    return best_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train staged CatBoost recommendation experiments.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild intermediate datasets from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    try:
        bundle = run_training(config=config, force_prepare=args.force_prepare)
    except Exception:
        logger.exception("Training pipeline failed")
        raise
    print(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
