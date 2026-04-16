"""Training entrypoint for the recommendation model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.build_dataset import build_eda_artifacts, build_monthly_modeling_dataset, load_modeling_month
from src.data.constants import PRODUCT_COLUMNS
from src.models.baselines import GlobalPopularityRecommender, SegmentPopularityRecommender
from src.models.evaluate import build_error_analysis_frame, evaluate_rankings
from src.models.predict import score_dataframe
from src.utils.config import ProjectConfig, ensure_directories


@dataclass(slots=True)
class SplitMetrics:
    split_name: str
    metrics: dict[str, float]


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


def build_estimator(config: ProjectConfig) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(config.numeric_feature_columns)),
            ("categorical", categorical_transformer, list(config.categorical_feature_columns)),
        ]
    )

    classifier = OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            class_weight="balanced",
            max_iter=30,
            tol=1e-3,
            random_state=config.random_state,
        ),
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


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


def load_sampled_training_frame(config: ProjectConfig) -> pd.DataFrame:
    frames = []
    for offset, month in enumerate(config.train_months):
        monthly_df = load_modeling_month(config, month)
        frames.append(sample_training_rows(monthly_df, config.negative_sample_ratio, config.random_state + offset))
    return pd.concat(frames, ignore_index=True)


def fit_global_baseline(config: ProjectConfig) -> GlobalPopularityRecommender:
    columns = ["target_count", *[f"target__{column}" for column in PRODUCT_COLUMNS]]
    monthly_frames = [load_modeling_month(config, month)[columns] for month in config.train_months]
    baseline = GlobalPopularityRecommender().fit(pd.concat(monthly_frames, ignore_index=True))
    return baseline


def fit_segment_baseline(config: ProjectConfig) -> SegmentPopularityRecommender:
    columns = ["segmento", *[f"target__{column}" for column in PRODUCT_COLUMNS]]
    monthly_frames = [load_modeling_month(config, month)[columns] for month in config.train_months]
    baseline = SegmentPopularityRecommender().fit(pd.concat(monthly_frames, ignore_index=True))
    return baseline


def evaluate_model_on_months(
    config: ProjectConfig,
    model_like: Any,
    months: tuple[str, ...],
    split_name: str,
) -> tuple[SplitMetrics, pd.DataFrame]:
    metrics_rows = []
    error_frames = []

    for offset, month in enumerate(months):
        monthly_df = load_modeling_month(config, month)
        monthly_df = sample_evaluation_month(monthly_df, config.eval_month_sample_size, config.random_state + offset)
        if hasattr(model_like, "predict_scores"):
            scores = model_like.predict_scores(monthly_df)
        else:
            scores = model_like.predict_proba(monthly_df[list(config.feature_columns)])
            if isinstance(scores, list):
                scores = np.column_stack([column[:, 1] for column in scores])

        metrics = evaluate_rankings(
            y_true=monthly_df[list(config.target_columns)].to_numpy(),
            scores=np.asarray(scores, dtype=float),
            current_products=monthly_df[PRODUCT_COLUMNS].to_numpy(),
            k=config.top_k,
        )
        metrics["snapshot_month"] = month
        metrics_rows.append(metrics)
        error_frames.append(build_error_analysis_frame(monthly_df, np.asarray(scores, dtype=float), k=config.top_k))

    metrics_df = pd.DataFrame(metrics_rows)
    aggregated_metrics = {
        key: float(metrics_df[key].mean())
        for key in metrics_df.columns
        if key != "snapshot_month"
    }
    aggregated_metrics["months"] = float(len(months))
    return SplitMetrics(split_name=split_name, metrics=aggregated_metrics), pd.concat(error_frames, ignore_index=True)


def build_reference_stats(config: ProjectConfig) -> dict[str, Any]:
    train_sample = load_sampled_training_frame(config)
    reference_stats: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for column in config.numeric_feature_columns:
        series = pd.to_numeric(train_sample[column], errors="coerce")
        reference_stats["numeric"][column] = {
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }

    for column in config.categorical_feature_columns:
        top_values = train_sample[column].fillna("UNKNOWN").astype("string").value_counts(normalize=True).head(10)
        reference_stats["categorical"][column] = top_values.to_dict()

    return reference_stats


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    rows = []
    for target_name, estimator in zip(PRODUCT_COLUMNS, classifier.estimators_):
        if not hasattr(estimator, "coef_"):
            rows.append(
                {
                    "product": target_name,
                    "feature_name": "constant_predictor",
                    "coefficient": 0.0,
                    "abs_coefficient": 0.0,
                }
            )
            continue
        coef = estimator.coef_.ravel()
        top_indices = np.argsort(np.abs(coef))[-15:][::-1]
        for idx in top_indices:
            rows.append(
                {
                    "product": target_name,
                    "feature_name": str(feature_names[idx]),
                    "coefficient": float(coef[idx]),
                    "abs_coefficient": float(abs(coef[idx])),
                }
            )
    return pd.DataFrame(rows)


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def run_training(config: ProjectConfig, force_prepare: bool = False) -> dict[str, Any]:
    ensure_directories(config)
    build_monthly_modeling_dataset(config, force=force_prepare)
    build_eda_artifacts(config)

    train_df = load_sampled_training_frame(config)
    X_train = train_df[list(config.feature_columns)]
    y_train = train_df[list(config.target_columns)]

    global_baseline = fit_global_baseline(config)
    segment_baseline = fit_segment_baseline(config)
    model = build_estimator(config)
    model.fit(X_train, y_train)

    global_valid_metrics, _ = evaluate_model_on_months(config, global_baseline, config.valid_months, "valid")
    global_test_metrics, _ = evaluate_model_on_months(config, global_baseline, config.test_months, "test")
    segment_valid_metrics, _ = evaluate_model_on_months(config, segment_baseline, config.valid_months, "valid")
    segment_test_metrics, _ = evaluate_model_on_months(config, segment_baseline, config.test_months, "test")
    model_valid_metrics, valid_errors = evaluate_model_on_months(config, model, config.valid_months, "valid")
    model_test_metrics, test_errors = evaluate_model_on_months(config, model, config.test_months, "test")

    reference_stats = build_reference_stats(config)
    feature_importance = extract_feature_importance(model)

    comparison_metrics = {
        "global_baseline_valid": global_valid_metrics.metrics,
        "global_baseline_test": global_test_metrics.metrics,
        "segment_baseline_valid": segment_valid_metrics.metrics,
        "segment_baseline_test": segment_test_metrics.metrics,
        "supervised_sgd_valid": model_valid_metrics.metrics,
        "supervised_sgd_test": model_test_metrics.metrics,
    }

    supervised_bundle = {
        "model": model,
        "bundle_type": "supervised_sgd",
        "feature_columns": list(config.feature_columns),
        "numeric_feature_columns": list(config.numeric_feature_columns),
        "categorical_feature_columns": list(config.categorical_feature_columns),
        "product_columns": PRODUCT_COLUMNS,
        "top_k": config.top_k,
        "reference_stats": reference_stats,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": comparison_metrics,
        "selected_model": "supervised_sgd",
    }
    save_bundle(supervised_bundle, config.models_dir / "supervised_model.joblib")

    candidate_models = {
        "global_popularity": (global_baseline, global_valid_metrics.metrics[f"map_at_{config.top_k}"]),
        "segment_popularity": (segment_baseline, segment_valid_metrics.metrics[f"map_at_{config.top_k}"]),
        "supervised_sgd": (model, model_valid_metrics.metrics[f"map_at_{config.top_k}"]),
    }
    selected_model_name = max(candidate_models, key=lambda item: candidate_models[item][1])
    selected_model = candidate_models[selected_model_name][0]

    best_bundle = {
        "model": selected_model,
        "bundle_type": selected_model_name,
        "feature_columns": list(config.feature_columns),
        "numeric_feature_columns": list(config.numeric_feature_columns),
        "categorical_feature_columns": list(config.categorical_feature_columns),
        "product_columns": PRODUCT_COLUMNS,
        "top_k": config.top_k,
        "reference_stats": reference_stats,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": comparison_metrics,
        "selected_model": selected_model_name,
    }
    save_bundle(best_bundle, config.models_dir / "best_model.joblib")

    feature_importance_path = config.models_dir / "feature_importance.csv"
    valid_errors_path = config.models_dir / "validation_errors.csv"
    test_errors_path = config.models_dir / "test_errors.csv"
    metadata_path = config.models_dir / "model_metadata.json"
    reference_stats_path = config.models_dir / "reference_stats.json"

    feature_importance.to_csv(feature_importance_path, index=False)
    valid_errors.to_csv(valid_errors_path, index=False)
    test_errors.to_csv(test_errors_path, index=False)
    metadata_path.write_text(
        json.dumps({"selected_model": selected_model_name, **comparison_metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reference_stats_path.write_text(json.dumps(reference_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name="logreg_ovr_ranking") as run:
        mlflow.log_params(
            {
                "random_state": config.random_state,
                "negative_sample_ratio": config.negative_sample_ratio,
                "top_k": config.top_k,
                "train_months": ",".join(config.train_months),
                "valid_months": ",".join(config.valid_months),
                "test_months": ",".join(config.test_months),
                "num_features": len(config.feature_columns),
                "num_train_rows": len(train_df),
                "eval_month_sample_size": config.eval_month_sample_size,
                "model_type": "OneVsRest(SGDClassifier log_loss)",
            }
        )
        mlflow.log_metrics({f"global_valid_{k}": v for k, v in global_valid_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"global_test_{k}": v for k, v in global_test_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"segment_valid_{k}": v for k, v in segment_valid_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"segment_test_{k}": v for k, v in segment_test_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"valid_{k}": v for k, v in model_valid_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"test_{k}": v for k, v in model_test_metrics.metrics.items() if isinstance(v, float)})
        mlflow.log_param("selected_model", selected_model_name)
        mlflow.log_artifact(str(feature_importance_path))
        mlflow.log_artifact(str(valid_errors_path))
        mlflow.log_artifact(str(test_errors_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(reference_stats_path))

        input_example = X_train.head(5)
        signature = infer_signature(input_example, model.predict_proba(input_example))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        supervised_bundle["mlflow_run_id"] = run.info.run_id
        save_bundle(supervised_bundle, config.models_dir / "supervised_model.joblib")

    if selected_model_name != "supervised_sgd":
        with mlflow.start_run(run_name=f"{selected_model_name}_best") as run:
            mlflow.log_param("model_type", selected_model_name)
            mlflow.log_param("selected_for_deployment", 1)
            if selected_model_name == "global_popularity":
                selected_valid_metrics = global_valid_metrics.metrics
                selected_test_metrics = global_test_metrics.metrics
            else:
                selected_valid_metrics = segment_valid_metrics.metrics
                selected_test_metrics = segment_test_metrics.metrics
            mlflow.log_metrics({f"valid_{k}": v for k, v in selected_valid_metrics.items() if isinstance(v, float)})
            mlflow.log_metrics({f"test_{k}": v for k, v in selected_test_metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(str(config.models_dir / "best_model.joblib"))
            mlflow.log_artifact(str(metadata_path))
            best_bundle["mlflow_run_id"] = run.info.run_id
            save_bundle(best_bundle, config.models_dir / "best_model.joblib")

    return best_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bank product recommendation model.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild intermediate datasets from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    bundle = run_training(config=config, force_prepare=args.force_prepare)
    print(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
