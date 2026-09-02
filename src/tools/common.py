"""Validation and JSON-normalization shared by deterministic tools."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MAX_SAMPLE_ITEMS, MAX_UPLOAD_BYTES


class ToolInputError(ValueError):
    """A clear, user-safe tool validation error."""


def load_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path).resolve()
    if source.suffix.lower() != ".csv":
        raise ToolInputError(f"Only CSV files are supported: {source.name}")
    if not source.is_file():
        raise ToolInputError(f"Dataset does not exist: {source.name}")
    if source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ToolInputError(f"Dataset exceeds the 50 MB MVP limit: {source.name}")
    try:
        frame = pd.read_csv(source)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ToolInputError(f"Could not parse {source.name} as CSV: {exc}") from exc
    if frame.columns.empty:
        raise ToolInputError(f"Dataset has no columns: {source.name}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ToolInputError(f"{dataset} is missing column(s): {', '.join(missing)}")


def apply_filters(frame: pd.DataFrame, conditions: list[dict[str, Any]] | None) -> pd.DataFrame:
    filtered = frame
    for condition in conditions or []:
        column = str(condition.get("column", ""))
        operator = str(condition.get("operator", "eq")).lower()
        if column not in filtered.columns:
            raise ToolInputError(f"Filter column does not exist: {column}")
        value = condition.get("value")
        series = filtered[column]
        operations = {
            "eq": lambda: series == value,
            "ne": lambda: series != value,
            "gt": lambda: series > value,
            "gte": lambda: series >= value,
            "lt": lambda: series < value,
            "lte": lambda: series <= value,
            "in": lambda: series.isin(value if isinstance(value, list) else [value]),
            "not_in": lambda: ~series.isin(value if isinstance(value, list) else [value]),
            "is_null": lambda: series.isna(),
            "not_null": lambda: series.notna(),
        }
        if operator not in operations:
            raise ToolInputError(f"Unsupported filter operator: {operator}")
        try:
            filtered = filtered.loc[operations[operator]()].copy()
        except TypeError as exc:
            raise ToolInputError(f"Invalid filter for {column}: {exc}") from exc
    return filtered


def aggregate(frame: pd.DataFrame, column: str | None, aggregation: str) -> float | int:
    operation = aggregation.lower()
    if operation not in {"sum", "count", "mean"}:
        raise ToolInputError(f"Unsupported aggregation: {aggregation}")
    if operation == "count" and column is None:
        return int(len(frame))
    if column is None or column not in frame.columns:
        raise ToolInputError(f"Metric column does not exist: {column}")
    if operation in {"sum", "mean"} and not pd.api.types.is_numeric_dtype(frame[column]):
        raise ToolInputError(f"Metric column must be numeric for {operation}: {column}")
    result = getattr(frame[column], operation)()
    return int(result) if operation == "count" else round(float(result), 6)


def compact_values(values: pd.Series) -> list[Any]:
    return [json_safe(value) for value in values.drop_duplicates().head(MAX_SAMPLE_ITEMS).tolist()]


def json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
