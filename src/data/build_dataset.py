"""Build monthly modeling datasets and EDA artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import CATEGORICAL_COLUMNS, MONTHLY_DATASET_COLUMNS, NUMERIC_COLUMNS, PRODUCT_COLUMNS
from src.data.load import dataset_month_path, load_month_frame, split_raw_to_monthly_files
from src.data.preprocess import optimize_modeling_frame_dtypes
from src.features.feature_engineering import add_prev_month_deltas, add_snapshot_features
from src.features.target_builder import build_multilabel_target
from src.utils.config import ProjectConfig, ensure_directories


def list_available_months(config: ProjectConfig) -> list[str]:
    split_raw_to_monthly_files(config)
    return sorted(path.stem for path in config.monthly_dir.glob("*.csv"))


def modeling_target_columns() -> list[str]:
    return [f"target__{column}" for column in PRODUCT_COLUMNS]


def project_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [*MONTHLY_DATASET_COLUMNS, *modeling_target_columns(), "target_count"]
    projected = df.loc[:, [column for column in required_columns if column in df.columns]].copy()
    return optimize_modeling_frame_dtypes(projected)


def build_monthly_modeling_dataset(config: ProjectConfig, force: bool = False) -> list[Path]:
    """Create one parquet file per current month with features and next-month targets."""

    ensure_directories(config)
    months = list_available_months(config)
    dataset_paths = sorted(config.modeling_dir.glob("*.parquet"))
    if dataset_paths and not force:
        return dataset_paths

    if force:
        for file_path in dataset_paths:
            file_path.unlink()

    for index, month in enumerate(months[:-1]):
        output_path = dataset_month_path(config, month)
        if output_path.exists() and not force:
            continue

        current_df = load_month_frame(config, month)
        prev_df = load_month_frame(config, months[index - 1]) if index > 0 else None
        next_df = load_month_frame(config, months[index + 1])

        current_df = add_snapshot_features(current_df)
        current_df = add_prev_month_deltas(current_df, prev_df)
        current_df = build_multilabel_target(current_df, next_df)
        current_df = project_modeling_frame(current_df)
        current_df.to_parquet(output_path, index=False)

    return sorted(config.modeling_dir.glob("*.parquet"))


def load_modeling_month(config: ProjectConfig, month: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = dataset_month_path(config, month)
    if not path.exists():
        raise FileNotFoundError(f"Modeling month dataset not found: {path}")
    return pd.read_parquet(path, columns=columns)


def load_split_dataframe(config: ProjectConfig, months: tuple[str, ...], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [load_modeling_month(config, month, columns=columns) for month in months]
    if not frames:
        raise ValueError("No months provided for split loading")
    return pd.concat(frames, ignore_index=True)


def _missing_profile(monthly_df: pd.DataFrame) -> pd.Series:
    return monthly_df.isna().mean().sort_values(ascending=False)


def build_eda_artifacts(config: ProjectConfig) -> dict[str, Path]:
    """Generate lightweight aggregated artifacts for notebooks and README."""

    ensure_directories(config)
    build_monthly_modeling_dataset(config)

    monthly_stats: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    additions_rows: list[dict[str, object]] = []

    for month in list_available_months(config):
        monthly_df = load_month_frame(config, month)
        monthly_df["snapshot_month"] = month
        monthly_df["products_total"] = monthly_df[PRODUCT_COLUMNS].sum(axis=1)

        monthly_stats.append(
            {
                "snapshot_month": month,
                "rows": int(len(monthly_df)),
                "unique_clients": int(monthly_df["ncodpers"].nunique()),
                "active_clients_share": float(monthly_df["ind_actividad_cliente"].fillna(0).gt(0).mean()),
                "avg_products_total": float(monthly_df["products_total"].mean()),
                "median_renta": float(monthly_df["renta"].median(skipna=True)),
            }
        )

        missing_profile = _missing_profile(monthly_df[["age", "renta", "fecha_alta", "cod_prov", "nomprov", "segmento"]])
        for feature_name, missing_share in missing_profile.items():
            missing_rows.append(
                {
                    "snapshot_month": month,
                    "feature_name": feature_name,
                    "missing_share": float(missing_share),
                }
            )

        segment_summary = (
            monthly_df.groupby("segmento", dropna=False)
            .agg(
                clients=("ncodpers", "nunique"),
                avg_age=("age", "mean"),
                avg_renta=("renta", "mean"),
                avg_products_total=("products_total", "mean"),
            )
            .reset_index()
        )
        segment_summary["snapshot_month"] = month
        segment_rows.append(segment_summary)

        product_share = monthly_df[PRODUCT_COLUMNS].mean().reset_index()
        product_share.columns = ["product", "penetration_share"]
        product_share["snapshot_month"] = month
        product_rows.append(product_share)

        modeling_path = dataset_month_path(config, month)
        if modeling_path.exists():
            modeling_df = pd.read_parquet(
                modeling_path,
                columns=["target_month", "target_count", *modeling_target_columns()],
            )
            additions_rows.append(
                pd.DataFrame(
                    {
                        "snapshot_month": month,
                        "target_month": [str(modeling_df["target_month"].iloc[0])],
                        "new_products_total": [int(modeling_df["target_count"].sum())],
                        "share_with_new_product": [float((modeling_df["target_count"] > 0).mean())],
                    },
                )
            )

    paths = {
        "monthly_overview": config.eda_dir / "monthly_overview.csv",
        "missing_profile": config.eda_dir / "missing_profile.csv",
        "segment_profile": config.eda_dir / "segment_profile.csv",
        "product_penetration": config.eda_dir / "product_penetration.csv",
        "new_product_dynamics": config.eda_dir / "new_product_dynamics.csv",
        "dataset_summary": config.eda_dir / "dataset_summary.json",
    }

    pd.DataFrame(monthly_stats).to_csv(paths["monthly_overview"], index=False)
    pd.DataFrame(missing_rows).to_csv(paths["missing_profile"], index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(paths["segment_profile"], index=False)
    pd.concat(product_rows, ignore_index=True).to_csv(paths["product_penetration"], index=False)
    pd.concat(additions_rows, ignore_index=True).to_csv(paths["new_product_dynamics"], index=False)

    product_penetration_df = pd.concat(product_rows, ignore_index=True)
    latest_month = max(list_available_months(config))

    summary_payload = {
        "months_available": list_available_months(config),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "product_columns": PRODUCT_COLUMNS,
        "top_products_last_month": (
            product_penetration_df.loc[product_penetration_df["snapshot_month"] == latest_month]
            .sort_values("penetration_share", ascending=False)
            .head(5)[["product", "penetration_share"]]
            .to_dict(orient="records")
        ),
    }
    paths["dataset_summary"].write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
