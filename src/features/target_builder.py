"""Target generation for new product adoption."""

from __future__ import annotations

import pandas as pd

from src.data.constants import PRODUCT_COLUMNS


def build_multilabel_target(current_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    """Attach next-month product additions to the current month snapshot."""

    next_products = next_df[["ncodpers", *PRODUCT_COLUMNS]].copy()
    next_products = next_products.rename(columns={column: f"next__{column}" for column in PRODUCT_COLUMNS})
    merged = current_df.merge(next_products, on="ncodpers", how="left")

    for column in PRODUCT_COLUMNS:
        merged[f"next__{column}"] = merged[f"next__{column}"].fillna(merged[column]).astype("int8")
        target_column = f"target__{column}"
        merged[target_column] = (merged[f"next__{column}"] - merged[column]).clip(lower=0).astype("int8")

    merged["target_count"] = merged[[f"target__{column}" for column in PRODUCT_COLUMNS]].sum(axis=1).astype("int8")
    merged["target_month"] = next_df["fecha_dato"].iloc[0]
    merged = merged.drop(columns=[f"next__{column}" for column in PRODUCT_COLUMNS])
    return merged
