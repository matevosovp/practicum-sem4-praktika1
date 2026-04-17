"""Cleaning helpers for the Santander product recommendation dataset."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.data.constants import CATEGORICAL_COLUMNS, DATE_COLUMNS, NUMERIC_COLUMNS, PRODUCT_COLUMNS


def _strip_object_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
            df[column] = df[column].replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA})
    return df


def clean_raw_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one raw CSV chunk and standardize dtypes."""

    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    df = _strip_object_columns(df, object_columns)

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in PRODUCT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(0, 1).astype("int8")

    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=18, upper=100)
    if "antiguedad" in df.columns:
        df["antiguedad"] = df["antiguedad"].clip(lower=0)
    if "renta" in df.columns:
        df.loc[df["renta"] < 0, "renta"] = pd.NA

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("UNKNOWN").astype("string")

    return df


def optimize_modeling_frame_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast modeling columns to keep parquet files and fit frames compact."""

    optimized = df.copy()

    float32_columns = {
        "customer_since_months",
        "age",
        "ind_nuevo",
        "antiguedad",
        "indrel",
        "tipodom",
        "cod_prov",
        "ind_actividad_cliente",
        "renta",
    }
    int8_columns = {
        "month_number",
        "prev_products_total",
        "products_total",
        "products_added_prev_month",
        "products_dropped_prev_month",
        "has_any_new_product",
        "target_count",
        *PRODUCT_COLUMNS,
        *[f"target__{column}" for column in PRODUCT_COLUMNS],
    }

    if "ncodpers" in optimized.columns:
        optimized["ncodpers"] = pd.to_numeric(optimized["ncodpers"], errors="coerce").astype("int32")

    for column in float32_columns:
        if column in optimized.columns:
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype("float32")

    for column in int8_columns:
        if column in optimized.columns:
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").fillna(0).astype("int8")

    for column in CATEGORICAL_COLUMNS:
        if column in optimized.columns:
            optimized[column] = optimized[column].fillna("UNKNOWN").astype("string")

    return optimized


def read_clean_csv(path: str | pd.io.common.FilePath, **kwargs: object) -> pd.DataFrame:
    """Read a prepared monthly CSV with consistent parsing."""

    df = pd.read_csv(path, parse_dates=["fecha_dato", "fecha_alta", "ult_fec_cli_1t"], low_memory=False, **kwargs)
    return clean_raw_chunk(df)
