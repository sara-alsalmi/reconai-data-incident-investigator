"""Dataset profiling tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import DatasetProfile
from src.tools.common import json_safe, load_csv


def _date_columns(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        series = frame[column]
        name_hint = any(token in column.lower() for token in ("date", "time", "timestamp"))
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(column)
        elif name_hint and not series.dropna().empty:
            parsed = pd.to_datetime(series, errors="coerce")
            if float(parsed.notna().mean()) >= 0.8:
                candidates.append(column)
    return candidates


def profile_dataset(path: str | Path) -> dict:
    """Return a compact, deterministic profile without exposing full rows."""
    frame = load_csv(path)
    rows = len(frame)
    unique_counts = {column: int(frame[column].nunique(dropna=True)) for column in frame.columns}
    identifiers = [
        column
        for column in frame.columns
        if rows > 0
        and (
            column.lower() == "id"
            or column.lower().endswith("_id")
            or unique_counts[column] / rows >= 0.98
        )
    ]
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    summaries: dict[str, dict[str, float | None]] = {}
    for column in numeric_columns:
        series = frame[column]
        summaries[column] = {
            "min": json_safe(series.min()),
            "max": json_safe(series.max()),
            "mean": json_safe(series.mean()),
            "sum": json_safe(series.sum()),
        }
    profile = DatasetProfile(
        dataset=Path(path).name,
        path=str(Path(path).resolve()),
        row_count=rows,
        columns=frame.columns.tolist(),
        data_types={column: str(dtype) for column, dtype in frame.dtypes.items()},
        null_counts={column: int(value) for column, value in frame.isna().sum().items()},
        duplicate_count=int(frame.duplicated(keep=False).sum()),
        unique_counts=unique_counts,
        numeric_summaries=summaries,
        possible_identifier_columns=identifiers,
        possible_date_columns=_date_columns(frame),
        numeric_columns=numeric_columns,
    )
    return profile.model_dump(mode="json")


def infer_relationships(profiles: list[DatasetProfile]) -> None:
    """Add shared-key candidates together with their observed dataset grain."""
    for profile in profiles:
        relationships: list[dict] = []
        for other in profiles:
            if other.dataset == profile.dataset:
                continue
            shared = set(profile.possible_identifier_columns) & set(
                other.possible_identifier_columns
            )
            for column in sorted(shared):
                local_eligible = profile.row_count - profile.null_counts.get(column, 0)
                other_eligible = other.row_count - other.null_counts.get(column, 0)
                local_unique = profile.unique_counts.get(column, 0) == local_eligible
                other_unique = other.unique_counts.get(column, 0) == other_eligible
                if local_unique and other_unique:
                    cardinality = "one_to_one"
                elif local_unique:
                    cardinality = "one_to_many"
                elif other_unique:
                    cardinality = "many_to_one"
                else:
                    cardinality = "many_to_many"
                relationships.append(
                    {
                        "dataset": other.dataset,
                        "local_column": column,
                        "other_column": column,
                        "cardinality": cardinality,
                        "local_key_is_unique": local_unique,
                        "other_key_is_unique": other_unique,
                        "basis": (
                            "shared identifier column name with cardinality inferred from "
                            "non-null key uniqueness"
                        ),
                    }
                )
        profile.possible_relationships = relationships
