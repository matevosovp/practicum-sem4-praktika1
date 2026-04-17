"""Central configuration for the project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.data.constants import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PRODUCT_COLUMNS


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ProjectConfig:
    """Project-wide paths and modeling parameters."""

    project_root: Path = PROJECT_ROOT
    random_state: int = int(_env("PROJECT_RANDOM_STATE", "42"))
    raw_data_path: Path = Path(_env("RAW_DATA_PATH", "train_ver2.csv"))
    processed_dir: Path = Path(_env("PROCESSED_DIR", "data/processed"))
    eda_dir: Path = Path(_env("EDA_DIR", "data/eda_artifacts"))
    models_dir: Path = Path(_env("MODELS_DIR", "models"))
    mlruns_dir: Path = Path(_env("MLRUNS_DIR", "mlruns"))
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow_experiment: str = _env("MLFLOW_EXPERIMENT", "bank-product-recommendations")
    mlflow_registered_model_name: str = _env("MLFLOW_REGISTERED_MODEL_NAME", "bank-product-recommendations-catboost")
    mlflow_model_alias: str = _env("MLFLOW_MODEL_ALIAS", "champion")
    primary_metric_name: str = _env("PRIMARY_METRIC_NAME", "val_map_at_3")
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
    negative_sample_ratio: float = float(_env("NEGATIVE_SAMPLE_RATIO", "0.6"))
    train_max_rows: int = int(_env("TRAIN_MAX_ROWS", "350000"))
    eval_month_sample_size: int = int(_env("EVAL_MONTH_SAMPLE_SIZE", "120000"))
    top_k: int = int(_env("TOP_K", "3"))
    catboost_thread_count: int = int(_env("CATBOOST_THREAD_COUNT", "4"))
    catboost_tuning_enabled: bool = _env("CATBOOST_TUNING_ENABLED", "true").lower() == "true"
    catboost_early_stopping_rounds: int = int(_env("CATBOOST_EARLY_STOPPING_ROUNDS", "40"))
    catboost_fit_eval_size: int = int(_env("CATBOOST_FIT_EVAL_SIZE", "40000"))
    catboost_ram_limit: str = _env("CATBOOST_RAM_LIMIT", "12gb")
    tuning_stage_a_month_count: int = int(_env("TUNING_STAGE_A_MONTH_COUNT", "4"))
    tuning_stage_a_max_rows: int = int(_env("TUNING_STAGE_A_MAX_ROWS", "180000"))
    tuning_stage_a_eval_size: int = int(_env("TUNING_STAGE_A_EVAL_SIZE", "40000"))
    tuning_stage_b_top_n: int = int(_env("TUNING_STAGE_B_TOP_N", "2"))
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

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.raw_data_path = self._resolve_path(self.raw_data_path)
        self.processed_dir = self._resolve_path(self.processed_dir)
        self.eda_dir = self._resolve_path(self.eda_dir)
        self.models_dir = self._resolve_path(self.models_dir)
        self.mlruns_dir = self._resolve_path(self.mlruns_dir)
        self.api_model_path = self._resolve_path(self.api_model_path)

    def _resolve_path(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def monthly_dir(self) -> Path:
        return self.processed_dir / self.monthly_dir_name

    @property
    def modeling_dir(self) -> Path:
        return self.processed_dir / self.modeling_dir_name

    @property
    def tuning_stage_a_months(self) -> tuple[str, ...]:
        month_count = max(1, min(self.tuning_stage_a_month_count, len(self.train_months)))
        return self.train_months[-month_count:]


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
