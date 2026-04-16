"""Time-based split helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def split_by_months(df: pd.DataFrame, months: Iterable[str]) -> pd.DataFrame:
    """Return a copy filtered to the requested snapshot months."""

    requested = set(months)
    snapshot_month = df["fecha_dato"].dt.strftime("%Y-%m-%d")
    return df.loc[snapshot_month.isin(requested)].copy()
