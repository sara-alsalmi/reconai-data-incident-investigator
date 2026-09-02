"""Flexible, bounded segment and time analysis tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.common import (
    ToolInputError,
    aggregate,
    apply_filters,
    json_safe,
    load_csv,
    require_columns,
)


def segment_analysis(
    dataset_path: str | Path,
    group_by_column: str | None,
    metric_column: str | None = None,
    aggregation: str = "count",
    filter_conditions: list[dict[str, Any]] | None = None,
    date_column: str | None = None,
    date_frequency: str = "D",
    match_against_path: str | Path | None = None,
    local_key: str | None = None,
    other_key: str | None = None,
    only_unmatched: bool = False,
    top_n: int = 20,
) -> dict:
    """Aggregate a filtered segment, optionally restricting it to unmatched rows."""
    frame = load_csv(dataset_path)
    frame = apply_filters(frame, filter_conditions)
    match_context = None
    if only_unmatched:
        if not all((match_against_path, local_key, other_key)):
            raise ToolInputError(
                "only_unmatched requires match_against_path, local_key, and other_key"
            )
        other = load_csv(match_against_path)
        require_columns(frame, [str(local_key)], Path(dataset_path).name)
        require_columns(other, [str(other_key)], Path(match_against_path).name)
        before = len(frame)
        frame = frame.loc[
            frame[str(local_key)].notna()
            & ~frame[str(local_key)].isin(other[str(other_key)].dropna())
        ].copy()
        match_context = {
            "comparison_dataset": Path(match_against_path).name,
            "records_before_matching_filter": before,
            "unmatched_records_analyzed": len(frame),
        }

    grouping: list[str] = []
    if group_by_column:
        require_columns(frame, [group_by_column], Path(dataset_path).name)
        grouping.append(group_by_column)
    period_column = None
    if date_column:
        require_columns(frame, [date_column], Path(dataset_path).name)
        parsed = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.loc[parsed.notna()].copy()
        period_column = "__period"
        frame[period_column] = parsed.loc[parsed.notna()].dt.to_period(date_frequency).astype(str)
        grouping.append(period_column)
    if not grouping:
        raise ToolInputError("Provide group_by_column or date_column")
    if metric_column is not None:
        require_columns(frame, [metric_column], Path(dataset_path).name)

    records: list[dict[str, Any]] = []
    grouper = grouping[0] if len(grouping) == 1 else grouping
    for key, group in frame.groupby(grouper, dropna=False):
        keys = (key,) if len(grouping) == 1 else key
        record = {
            (date_column if name == period_column else name): json_safe(value)
            for name, value in zip(grouping, keys)
        }
        record["value"] = aggregate(group, metric_column, aggregation)
        record["record_count"] = int(len(group))
        records.append(record)
    records.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    limit = min(max(int(top_n), 1), 100)
    return {
        "dataset": Path(dataset_path).name,
        "aggregation": aggregation.lower(),
        "metric_column": metric_column,
        "grouping": [date_column if name == period_column else name for name in grouping],
        "filtered_record_count": int(len(frame)),
        "result_count": len(records),
        "results": records[:limit],
        "results_truncated": len(records) > limit,
        "match_context": match_context,
    }

