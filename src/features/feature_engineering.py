"""Feature engineering for recommendation modeling."""

from __future__ import annotations

import pandas as pd

from src.data.constants import PRODUCT_COLUMNS


def add_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-snapshot features derived from the current client state."""

    enriched = df.copy()
    enriched["month_number"] = enriched["fecha_dato"].dt.month.astype("int8")
    customer_since = (enriched["fecha_dato"] - enriched["fecha_alta"]).dt.days.div(30)
    enriched["customer_since_months"] = customer_since.fillna(enriched["antiguedad"]).clip(lower=0)
    enriched["products_total"] = enriched[PRODUCT_COLUMNS].sum(axis=1).astype("int16")
    return enriched


def add_prev_month_deltas(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> pd.DataFrame:
    """Merge previous month product states and compute recent deltas."""

    if prev_df is None:
        enriched = df.copy()
        enriched["prev_products_total"] = 0
        enriched["products_added_prev_month"] = 0
        enriched["products_dropped_prev_month"] = 0
        enriched["has_any_new_product"] = 0
        return enriched

    prev_products = prev_df[["ncodpers", *PRODUCT_COLUMNS]].copy()
    prev_products = prev_products.rename(columns={column: f"prev__{column}" for column in PRODUCT_COLUMNS})
    enriched = df.merge(prev_products, on="ncodpers", how="left")

    prev_columns = [f"prev__{column}" for column in PRODUCT_COLUMNS]
    enriched[prev_columns] = enriched[prev_columns].fillna(0).astype("int8")

    diff = enriched[PRODUCT_COLUMNS].to_numpy() - enriched[prev_columns].to_numpy()
    enriched["prev_products_total"] = enriched[prev_columns].sum(axis=1).astype("int16")
    enriched["products_added_prev_month"] = (diff > 0).sum(axis=1).astype("int8")
    enriched["products_dropped_prev_month"] = (diff < 0).sum(axis=1).astype("int8")
    enriched["has_any_new_product"] = (enriched["products_added_prev_month"] > 0).astype("int8")
    enriched = enriched.drop(columns=prev_columns)
    return enriched
