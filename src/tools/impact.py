"""Evidence-bounded business impact calculation tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.common import ToolInputError, apply_filters, load_csv, require_columns


def calculate_business_impact(
    dataset_path: str | Path,
    amount_column: str | None = None,
    date_column: str | None = None,
    segment_column: str | None = None,
    filter_conditions: list[dict[str, Any]] | None = None,
    match_against_path: str | Path | None = None,
    local_key: str | None = None,
    other_key: str | None = None,
    only_unmatched: bool = False,
) -> dict:
    frame = load_csv(dataset_path)
    total_records = len(frame)
    affected = apply_filters(frame, filter_conditions)
    if only_unmatched:
        if not all((match_against_path, local_key, other_key)):
            raise ToolInputError(
                "only_unmatched requires match_against_path, local_key, and other_key"
            )
        other = load_csv(match_against_path)
        require_columns(affected, [str(local_key)], Path(dataset_path).name)
        require_columns(other, [str(other_key)], Path(match_against_path).name)
        affected = affected.loc[
            affected[str(local_key)].notna()
            & ~affected[str(local_key)].isin(other[str(other_key)].dropna())
        ].copy()

    result: dict[str, Any] = {
        "dataset": Path(dataset_path).name,
        "affected_record_count": int(len(affected)),
        "total_record_count": int(total_records),
        "affected_percentage": 0.0
        if total_records == 0
        else len(affected) / total_records * 100,
        "affected_amount_sum": None,
        "affected_date_range": None,
        "segment_column": segment_column,
        "primary_affected_segment": None,
    }
    if amount_column:
        require_columns(affected, [amount_column], Path(dataset_path).name)
        if not pd.api.types.is_numeric_dtype(affected[amount_column]):
            raise ToolInputError(f"Amount column must be numeric: {amount_column}")
        result["affected_amount_sum"] = round(float(affected[amount_column].sum()), 2)
    if date_column:
        require_columns(affected, [date_column], Path(dataset_path).name)
        dates = pd.to_datetime(affected[date_column], errors="coerce").dropna()
        if not dates.empty:
            result["affected_date_range"] = {
                "start": dates.min().isoformat(),
                "end": dates.max().isoformat(),
            }
    if segment_column:
        require_columns(affected, [segment_column], Path(dataset_path).name)
        counts = affected[segment_column].value_counts(dropna=False)
        if not counts.empty:
            top = counts.index[0]
            result["primary_affected_segment"] = {
                "segment": None if pd.isna(top) else str(top),
                "record_count": int(counts.iloc[0]),
                "percentage_of_affected": float(counts.iloc[0] / len(affected) * 100),
            }
    return result
