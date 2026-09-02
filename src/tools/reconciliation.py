"""Aggregate, matching, and duplicate analysis tools."""

from __future__ import annotations

from pathlib import Path

from src.tools.common import (
    ToolInputError,
    aggregate,
    compact_values,
    load_csv,
    require_columns,
)


def compare_aggregates(
    dataset_a_path: str | Path,
    dataset_b_path: str | Path,
    metric_column_a: str | None = None,
    metric_column_b: str | None = None,
    aggregation: str = "sum",
) -> dict:
    frame_a = load_csv(dataset_a_path)
    frame_b = load_csv(dataset_b_path)
    result_a = aggregate(frame_a, metric_column_a, aggregation)
    result_b = aggregate(frame_b, metric_column_b, aggregation)
    difference = round(float(result_a) - float(result_b), 6)
    denominator = abs(float(result_b))
    percentage = (
        None if denominator == 0 else round(difference / denominator * 100, 6)
    )
    return {
        "dataset_a": Path(dataset_a_path).name,
        "dataset_b": Path(dataset_b_path).name,
        "aggregation": aggregation.lower(),
        "metric_column_a": metric_column_a,
        "metric_column_b": metric_column_b,
        "dataset_a_result": result_a,
        "dataset_b_result": result_b,
        "absolute_difference": round(abs(difference), 6),
        "signed_difference_a_minus_b": difference,
        "percentage_difference_vs_b": percentage,
    }


def find_unmatched_records(
    dataset_a_path: str | Path,
    dataset_b_path: str | Path,
    key_column_a: str,
    key_column_b: str,
) -> dict:
    """Find rows in A whose non-null key is absent from B."""
    frame_a = load_csv(dataset_a_path)
    frame_b = load_csv(dataset_b_path)
    require_columns(frame_a, [key_column_a], Path(dataset_a_path).name)
    require_columns(frame_b, [key_column_b], Path(dataset_b_path).name)
    valid_a = frame_a.loc[frame_a[key_column_a].notna()].copy()
    keys_b = set(frame_b[key_column_b].dropna().tolist())
    unmatched = valid_a.loc[~valid_a[key_column_a].isin(keys_b)]
    matched_count = int(len(valid_a) - len(unmatched))
    unmatched_count = int(len(unmatched))
    total = len(valid_a)
    return {
        "direction": f"{Path(dataset_a_path).name}_without_{Path(dataset_b_path).name}",
        "dataset_a": Path(dataset_a_path).name,
        "dataset_b": Path(dataset_b_path).name,
        "key_column_a": key_column_a,
        "key_column_b": key_column_b,
        "eligible_record_count": total,
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "unmatched_percentage": 0.0 if total == 0 else unmatched_count / total * 100,
        "sample_unmatched_identifiers": compact_values(unmatched[key_column_a]),
        "null_key_count_a": int(frame_a[key_column_a].isna().sum()),
    }


def find_duplicates(
    dataset_path: str | Path, key_columns: list[str] | None = None
) -> dict:
    frame = load_csv(dataset_path)
    if key_columns:
        require_columns(frame, key_columns, Path(dataset_path).name)
    subset = key_columns or frame.columns.tolist()
    mask = frame.duplicated(subset=subset, keep=False)
    duplicates = frame.loc[mask]
    count = int(mask.sum())
    if key_columns:
        sample = duplicates[key_columns].drop_duplicates().head(10).to_dict("records")
    else:
        sample = duplicates.index.to_series().head(10).astype(int).tolist()
    return {
        "dataset": Path(dataset_path).name,
        "checked_columns": subset,
        "duplicate_row_count": count,
        "duplicate_percentage": 0.0 if len(frame) == 0 else count / len(frame) * 100,
        "sample_duplicate_identifiers": sample,
    }
