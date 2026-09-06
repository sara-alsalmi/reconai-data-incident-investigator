"""Aggregate, matching, and duplicate analysis tools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.tools.common import (
    ToolInputError,
    aggregate,
    compact_values,
    json_safe,
    load_csv,
    require_columns,
)
from src.config import MAX_SAMPLE_ITEMS


_DETAIL_COLUMN_HINTS = (
    "status",
    "state",
    "timestamp",
    "date",
    "time",
    "amount",
    "total",
    "value",
    "price",
    "revenue",
    "cost",
    "balance",
    "channel",
    "region",
    "segment",
    "category",
    "type",
    "method",
    "source",
)


def _automatic_detail_columns(frame: pd.DataFrame, key_column: str) -> list[str]:
    """Choose a small, useful context set without requiring question-specific input."""
    candidates = [column for column in frame.columns if column != key_column]

    def rank(column: str) -> tuple[int, int, int, str]:
        lowered = column.lower()
        is_date_like = lowered.endswith("_at")
        semantic_rank = next(
            (
                index
                for index, hint in enumerate(_DETAIL_COLUMN_HINTS)
                if hint in lowered
            ),
            len(_DETAIL_COLUMN_HINTS),
        )
        if is_date_like:
            semantic_rank = min(semantic_rank, _DETAIL_COLUMN_HINTS.index("timestamp"))
        identifier_penalty = 1 if lowered == "id" or lowered.endswith("_id") else 0
        primary_time_penalty = (
            0
            if any(
                hint in lowered
                for hint in ("created", "purchase", "occurred", "event", "transaction")
            )
            else 1
        )
        return identifier_penalty, semantic_rank, primary_time_penalty, lowered

    candidates.sort(key=rank)
    useful = [
        column
        for column in candidates
        if column.lower().endswith("_at")
        or any(hint in column.lower() for hint in _DETAIL_COLUMN_HINTS)
    ]
    return useful[:4]


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
    detail_columns: list[str] | None = None,
) -> dict:
    """Find rows in A whose non-null key is absent from B.

    Results distinguish unmatched rows from unmatched unique key values. A bounded
    record preview includes useful context automatically, or caller-selected detail
    columns when supplied.
    """
    frame_a = load_csv(dataset_a_path)
    frame_b = load_csv(dataset_b_path)
    require_columns(frame_a, [key_column_a], Path(dataset_a_path).name)
    require_columns(frame_b, [key_column_b], Path(dataset_b_path).name)
    if detail_columns is None:
        selected_details = _automatic_detail_columns(frame_a, key_column_a)
    else:
        if not all(isinstance(column, str) and column for column in detail_columns):
            raise ToolInputError("Detail columns must be non-empty strings")
        selected_details = list(dict.fromkeys(detail_columns))
        require_columns(frame_a, selected_details, Path(dataset_a_path).name)
        selected_details = [
            column for column in selected_details if column != key_column_a
        ][:8]
    valid_a = frame_a.loc[frame_a[key_column_a].notna()].copy()
    keys_b = set(frame_b[key_column_b].dropna().tolist())
    unmatched = valid_a.loc[~valid_a[key_column_a].isin(keys_b)]
    matched_count = int(len(valid_a) - len(unmatched))
    unmatched_count = int(len(unmatched))
    unmatched_unique_key_count = int(unmatched[key_column_a].nunique(dropna=True))
    total = len(valid_a)
    preview_frame = unmatched.drop_duplicates(subset=[key_column_a]).head(
        MAX_SAMPLE_ITEMS
    )
    preview_columns = [key_column_a, *selected_details]
    sample_records = [
        {column: json_safe(value) for column, value in record.items()}
        for record in preview_frame[preview_columns].to_dict("records")
    ]
    return {
        "direction": f"{Path(dataset_a_path).name}_without_{Path(dataset_b_path).name}",
        "dataset_a": Path(dataset_a_path).name,
        "dataset_b": Path(dataset_b_path).name,
        "key_column_a": key_column_a,
        "key_column_b": key_column_b,
        "eligible_record_count": total,
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "unmatched_unique_key_count": unmatched_unique_key_count,
        "unmatched_percentage": 0.0 if total == 0 else unmatched_count / total * 100,
        "sample_unmatched_identifiers": compact_values(unmatched[key_column_a]),
        "detail_columns": selected_details,
        "sample_unmatched_records": sample_records,
        "sample_is_complete": unmatched_unique_key_count <= MAX_SAMPLE_ITEMS,
        "sample_limit": MAX_SAMPLE_ITEMS,
        "null_key_count_a": int(frame_a[key_column_a].isna().sum()),
    }


def find_duplicates(
    dataset_path: str | Path, key_columns: list[str] | None = None
) -> dict:
    """Find exact duplicate rows or records participating in repeated keys.

    Repeated relationship keys are not automatically data errors: they can be
    the expected grain of a one-to-many table. Exact full-row duplicates are a
    stronger anomaly signal, so the result states which analysis was performed.
    """
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
        "analysis_type": "repeated_key" if key_columns else "exact_row",
        "checked_columns": subset,
        "duplicate_row_count": count,
        "duplicate_percentage": 0.0 if len(frame) == 0 else count / len(frame) * 100,
        "sample_duplicate_identifiers": sample,
        "interpretation": (
            "Records participate in repeated keys; repeated keys are not automatically "
            "errors, so determine relationship grain before classifying them."
            if key_columns
            else "Records are exact duplicates across every column."
        ),
    }


def compare_record_values(
    dataset_a_path: str | Path,
    dataset_b_path: str | Path,
    key_column_a: str,
    key_column_b: str,
    value_column_a: str,
    value_column_b: str,
    tolerance: float = 0.0,
) -> dict:
    """Compare scalar values for unique keys present in both datasets."""
    if tolerance < 0:
        raise ToolInputError("Tolerance must be zero or greater")
    if key_column_a == value_column_a or key_column_b == value_column_b:
        raise ToolInputError("Key and value columns must be different")
    frame_a = load_csv(dataset_a_path)
    frame_b = load_csv(dataset_b_path)
    require_columns(
        frame_a, [key_column_a, value_column_a], Path(dataset_a_path).name
    )
    require_columns(
        frame_b, [key_column_b, value_column_b], Path(dataset_b_path).name
    )
    numeric_values = pd.api.types.is_numeric_dtype(
        frame_a[value_column_a]
    ) and pd.api.types.is_numeric_dtype(frame_b[value_column_b])

    valid_a = frame_a.loc[frame_a[key_column_a].notna()].copy()
    valid_b = frame_b.loc[frame_b[key_column_b].notna()].copy()
    duplicate_a = valid_a[key_column_a].duplicated(keep=False)
    duplicate_b = valid_b[key_column_b].duplicated(keep=False)
    unique_a = valid_a.loc[~duplicate_a, [key_column_a, value_column_a]].rename(
        columns={key_column_a: "__key", value_column_a: "__value_a"}
    )
    unique_b = valid_b.loc[~duplicate_b, [key_column_b, value_column_b]].rename(
        columns={key_column_b: "__key", value_column_b: "__value_b"}
    )
    matched = unique_a.merge(unique_b, on="__key", how="inner", validate="one_to_one")
    both_null = matched["__value_a"].isna() & matched["__value_b"].isna()
    one_null = matched["__value_a"].isna() ^ matched["__value_b"].isna()
    if numeric_values:
        numeric_delta = matched["__value_a"] - matched["__value_b"]
        value_mismatch = numeric_delta.abs().gt(float(tolerance)).fillna(False)
    else:
        numeric_delta = None
        value_mismatch = (~matched["__value_a"].eq(matched["__value_b"])) & ~both_null
    mismatch_mask = one_null | value_mismatch
    mismatches = matched.loc[mismatch_mask].copy()

    sample = []
    for _, row in mismatches.head(10).iterrows():
        sample.append(
            {
                "key": row["__key"],
                "dataset_a_value": json_safe(row["__value_a"]),
                "dataset_b_value": json_safe(row["__value_b"]),
                "difference_a_minus_b": None
                if not numeric_values
                or pd.isna(row["__value_a"])
                or pd.isna(row["__value_b"])
                else round(float(row["__value_a"] - row["__value_b"]), 6),
            }
        )
    comparable_differences = (
        numeric_delta.loc[value_mismatch].dropna() if numeric_values else None
    )
    matched_count = int(len(matched))
    mismatch_count = int(mismatch_mask.sum())
    return {
        "dataset_a": Path(dataset_a_path).name,
        "dataset_b": Path(dataset_b_path).name,
        "key_column_a": key_column_a,
        "key_column_b": key_column_b,
        "value_column_a": value_column_a,
        "value_column_b": value_column_b,
        "tolerance": float(tolerance),
        "matched_unique_key_count": matched_count,
        "equal_value_count": matched_count - mismatch_count,
        "value_mismatch_count": mismatch_count,
        "value_mismatch_percentage": 0.0
        if matched_count == 0
        else mismatch_count / matched_count * 100,
        "null_value_mismatch_count": int(one_null.sum()),
        "signed_difference_a_minus_b": None
        if comparable_differences is None
        else round(float(comparable_differences.sum()), 6),
        "absolute_difference_sum": None
        if comparable_differences is None
        else round(float(comparable_differences.abs().sum()), 6),
        "duplicate_key_record_count_a": int(duplicate_a.sum()),
        "duplicate_key_record_count_b": int(duplicate_b.sum()),
        "both_null_equal_count": int(both_null.sum()),
        "sample_mismatches": sample,
        "comparison_type": "numeric" if numeric_values else "exact",
    }


def compare_rows_by_key(
    dataset_a_path: str | Path,
    dataset_b_path: str | Path,
    key_column_a: str,
    key_column_b: str,
    columns: list[str] | None = None,
    tolerance: float = 0.0,
) -> dict:
    """Compare several same-named columns for unique keys present in both datasets."""
    if tolerance < 0:
        raise ToolInputError("Tolerance must be zero or greater")
    frame_a = load_csv(dataset_a_path)
    frame_b = load_csv(dataset_b_path)
    require_columns(frame_a, [key_column_a], Path(dataset_a_path).name)
    require_columns(frame_b, [key_column_b], Path(dataset_b_path).name)
    selected = columns or [
        column
        for column in frame_a.columns
        if column in frame_b.columns and column not in {key_column_a, key_column_b}
    ]
    if not selected:
        raise ToolInputError("No shared value columns are available for row comparison")
    if not all(isinstance(column, str) and column for column in selected):
        raise ToolInputError("Comparison columns must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ToolInputError("Comparison columns must be unique")
    if any(column in {key_column_a, key_column_b} for column in selected):
        raise ToolInputError("Comparison columns must not include either key column")
    require_columns(frame_a, selected, Path(dataset_a_path).name)
    require_columns(frame_b, selected, Path(dataset_b_path).name)

    valid_a = frame_a.loc[frame_a[key_column_a].notna()].copy()
    valid_b = frame_b.loc[frame_b[key_column_b].notna()].copy()
    duplicate_a = valid_a[key_column_a].duplicated(keep=False)
    duplicate_b = valid_b[key_column_b].duplicated(keep=False)
    rename_a = {key_column_a: "__key"} | {
        column: f"__a_{column}" for column in selected
    }
    rename_b = {key_column_b: "__key"} | {
        column: f"__b_{column}" for column in selected
    }
    unique_a = valid_a.loc[~duplicate_a, [key_column_a, *selected]].rename(
        columns=rename_a
    )
    unique_b = valid_b.loc[~duplicate_b, [key_column_b, *selected]].rename(
        columns=rename_b
    )
    matched = unique_a.merge(unique_b, on="__key", how="inner", validate="one_to_one")
    any_mismatch = pd.Series(False, index=matched.index)
    mismatches_by_column: dict[str, int] = {}
    for column in selected:
        left = matched[f"__a_{column}"]
        right = matched[f"__b_{column}"]
        both_null = left.isna() & right.isna()
        one_null = left.isna() ^ right.isna()
        if pd.api.types.is_numeric_dtype(frame_a[column]) and pd.api.types.is_numeric_dtype(
            frame_b[column]
        ):
            mismatch = one_null | (left - right).abs().gt(float(tolerance)).fillna(False)
        else:
            mismatch = one_null | ((~left.eq(right)) & ~both_null)
        mismatches_by_column[column] = int(mismatch.sum())
        any_mismatch |= mismatch
    mismatched = matched.loc[any_mismatch]
    return {
        "dataset_a": Path(dataset_a_path).name,
        "dataset_b": Path(dataset_b_path).name,
        "key_column_a": key_column_a,
        "key_column_b": key_column_b,
        "columns_compared": selected,
        "tolerance": float(tolerance),
        "matched_unique_key_count": int(len(matched)),
        "record_mismatch_count": int(any_mismatch.sum()),
        "mismatches_by_column": mismatches_by_column,
        "duplicate_key_record_count_a": int(duplicate_a.sum()),
        "duplicate_key_record_count_b": int(duplicate_b.sum()),
        "sample_mismatched_keys": compact_values(mismatched["__key"]),
    }
