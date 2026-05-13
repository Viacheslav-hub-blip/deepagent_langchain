"""Функции загрузки входных файлов для pipeline поиска инсайтов.

Содержит:
- load_dataframe: загружает CSV, Excel, Parquet, JSON или JSONL в pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataframe(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Загружает табличный файл в pandas DataFrame.

    Args:
        path: Путь к входному файлу `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json` или `.jsonl`.
        **read_kwargs: Дополнительные параметры, которые будут переданы функции pandas.

    Returns:
        pandas DataFrame с содержимым входного файла.

    Raises:
        FileNotFoundError: Если входной файл не найден.
        ValueError: Если расширение файла не поддерживается.
    """

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path, **read_kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source_path, **read_kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(source_path, **read_kwargs)
    if suffix == ".jsonl":
        return pd.read_json(source_path, lines=True, **read_kwargs)
    if suffix == ".json":
        return pd.read_json(source_path, **read_kwargs)

    raise ValueError(
        "Неподдерживаемый формат файла. "
        "Поддерживаются .csv, .xlsx, .xls, .parquet, .json и .jsonl."
    )
