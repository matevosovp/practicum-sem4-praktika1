"""Central configuration for the project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.data.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PRODUCT_COLUMNS


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(slots=True)
class ProjectConfig:
    """Project-wide paths and modeling parameters."""

    random_state: int = int(_env("PROJECT_RANDOM_STATE", "42"))
    raw_data_path: Path = Path(_env("RAW_DATA_PATH", "train_ver2.csv"))
    processed_dir: Path = Path(_env("PROCESSED_DIR", "data/processed"))
    eda_dir: Path = Path(_env("EDA_DIR", "data/eda_artifacts"))
    models_dir: Path = Path(_env("MODELS_DIR", "models"))
    mlruns_dir: Path = Path(_env("MLRUNS_DIR", "mlruns"))
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow_experiment: str = _env("MLFLOW_EXPERIMENT", "bank-product-recommendations")
    train_months: tuple[str, ...] = tuple(
        _env(
            "TRAIN_MONTHS",
            "2015-02-28,2015-03-28,2015-04-28,2015-05-28,2015-06-28,"
            "2015-07-28,2015-08-28,2015-09-28,2015-10-28,2015-11-28",
        ).split(",")
    )
    valid_months: tuple[str, ...] = tuple(_env("VALID_MONTHS", "2015-12-28,2016-01-28,2016-02-28").split(","))
    test_months: tuple[str, ...] = tuple(_env("TEST_MONTHS", "2016-03-28,2016-04-28").split(","))
    chunk_size: int = int(_env("CSV_CHUNK_SIZE", "200000"))
    negative_sample_ratio: float = float(_env("NEGATIVE_SAMPLE_RATIO", "0.8"))
    eval_month_sample_size: int = int(_env("EVAL_MONTH_SAMPLE_SIZE", "120000"))
    top_k: int = int(_env("TOP_K", "3"))
    monthly_dir_name: str = "monthly"
    modeling_dir_name: str = "modeling"
    feature_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "month_number",
            "customer_since_months",
            "prev_products_total",
            "products_total",
            "products_added_prev_month",
            "products_dropped_prev_month",
            "has_any_new_product",
            *NUMERIC_COLUMNS,
            *CATEGORICAL_COLUMNS,
            *PRODUCT_COLUMNS,
        )
    )
    numeric_feature_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "month_number",
            "customer_since_months",
            "prev_products_total",
            "products_total",
            "products_added_prev_month",
            "products_dropped_prev_month",
            "has_any_new_product",
            *NUMERIC_COLUMNS,
            *PRODUCT_COLUMNS,
        )
    )
    categorical_feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(CATEGORICAL_COLUMNS))
    target_columns: tuple[str, ...] = field(default_factory=lambda: tuple(f"target__{col}" for col in PRODUCT_COLUMNS))
    api_model_path: Path = Path(_env("API_MODEL_PATH", "models/best_model.joblib"))
    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = int(_env("API_PORT", "8000"))
    prometheus_enabled: bool = _env("PROMETHEUS_ENABLED", "true").lower() == "true"

    @property
    def monthly_dir(self) -> Path:
        return self.processed_dir / self.monthly_dir_name

    @property
    def modeling_dir(self) -> Path:
        return self.processed_dir / self.modeling_dir_name


def ensure_directories(config: ProjectConfig) -> None:
    """Create required directories."""

    for path in [
        config.processed_dir,
        config.eda_dir,
        config.models_dir,
        config.mlruns_dir,
        config.monthly_dir,
        config.modeling_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
