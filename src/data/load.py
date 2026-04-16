"""Chunked data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.data.constants import SELECTED_COLUMNS
from src.data.preprocess import clean_raw_chunk, read_clean_csv
from src.utils.config import ProjectConfig, ensure_directories


def iter_raw_chunks(config: ProjectConfig) -> Iterator[pd.DataFrame]:
    """Yield cleaned chunks from the raw CSV."""

    for chunk in pd.read_csv(
        config.raw_data_path,
        usecols=SELECTED_COLUMNS,
        dtype="string",
        chunksize=config.chunk_size,
        low_memory=False,
    ):
        yield clean_raw_chunk(chunk)


def month_file_path(config: ProjectConfig, month: str) -> Path:
    return config.monthly_dir / f"{month}.csv"


def dataset_month_path(config: ProjectConfig, month: str) -> Path:
    return config.modeling_dir / f"{month}.parquet"


def split_raw_to_monthly_files(config: ProjectConfig, force: bool = False) -> list[Path]:
    """Split the big CSV into cleaned per-month files."""

    ensure_directories(config)
    existing_files = sorted(config.monthly_dir.glob("*.csv"))
    if existing_files and not force:
        return existing_files

    if force:
        for file_path in existing_files:
            file_path.unlink()

    for chunk in iter_raw_chunks(config):
        for month, month_df in chunk.groupby(chunk["fecha_dato"].dt.strftime("%Y-%m-%d"), dropna=False):
            if pd.isna(month):
                continue
            file_path = month_file_path(config, str(month))
            month_df.to_csv(file_path, mode="a", header=not file_path.exists(), index=False)

    return sorted(config.monthly_dir.glob("*.csv"))


def load_month_frame(config: ProjectConfig, month: str) -> pd.DataFrame:
    """Load one prepared monthly snapshot."""

    path = month_file_path(config, month)
    if not path.exists():
        raise FileNotFoundError(f"Monthly snapshot does not exist: {path}")
    return read_clean_csv(path)
