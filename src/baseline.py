"""Deterministic baseline reconciliation performed before LLM investigation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.agents.tool_adapter import ToolAuditor
from src.models import DatasetProfile, InvestigationState
from src.state import add_trace
from src.tools import (
    calculate_business_impact,
    compare_aggregates,
    compare_record_values,
    compare_rows_by_key,
    find_duplicates,
    find_unmatched_records,
    segment_analysis,
)
from src.tools.common import ToolInputError


MAX_BASELINE_RELATIONSHIPS = 2


def _normalized_words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def collect_question_required_evidence(state: InvestigationState) -> int:
    """Collect an explicitly requested numeric comparison during an agent attempt.

    This is a controller reliability gate: an LLM may not finish a question about a
    net value difference without a deterministic sum comparison for the named or
    most relevant shared metric.
    """
    if state.attempt_count < 1 or len(state.dataset_profiles) != 2:
        return 0

    question = _normalized_words(state.question)
    comparison_requested = any(
        term in question for term in ("difference", "differ", "disagree", "mismatch")
    )
    numeric_requested = any(
        term in question
        for term in ("net", "amount", "total", "sum", "revenue", "balance", "value")
    )
    if not comparison_requested or not numeric_requested:
        return 0

    left, right = state.dataset_profiles
    shared_numeric = [
        column
        for column in left.numeric_columns
        if column in right.numeric_columns
        and column.lower() != "id"
        and not column.lower().endswith("_id")
    ]
    if not shared_numeric:
        return 0

    explicitly_named = [
        column
        for column in shared_numeric
        if f" {_normalized_words(column)} " in f" {question} "
    ]
    semantic_candidates = [
        column
        for column in shared_numeric
        if any(
            token in column.lower()
            for token in ("amount", "total", "revenue", "balance", "value", "price", "cost")
        )
    ]
    metric = (explicitly_named or semantic_candidates or shared_numeric)[0]

    datasets = {Path(path).name: path for path in state.dataset_paths}
    if left.dataset not in datasets or right.dataset not in datasets:
        return 0
    auditor = ToolAuditor(state, datasets)
    collected = 0

    sum_already_collected = any(
        item.tool == "compare_aggregates"
        and item.supporting_details.get("aggregation") == "sum"
        and item.supporting_details.get("dataset_a") == left.dataset
        and item.supporting_details.get("dataset_b") == right.dataset
        and item.supporting_details.get("metric_column_a") == metric
        and item.supporting_details.get("metric_column_b") == metric
        for item in state.evidence
    )
    if not sum_already_collected:
        arguments = {
            "dataset_a": left.dataset,
            "dataset_b": right.dataset,
            "metric_column_a": metric,
            "metric_column_b": metric,
            "aggregation": "sum",
        }
        try:
            auditor.execute(
                "compare_aggregates",
                arguments,
                [left.dataset, right.dataset],
                lambda: compare_aggregates(
                    datasets[left.dataset],
                    datasets[right.dataset],
                    metric,
                    metric,
                    "sum",
                ),
            )
            collected += 1
            add_trace(
                state,
                "evidence",
                f"Controller required a `{metric}` sum comparison to answer the question",
            )
        except (ToolInputError, ValueError) as exc:
            add_trace(
                state,
                "quality",
                f"Required numeric comparison could not be collected: {exc}",
            )

    localization_requested = any(
        term in question
        for term in (
            "which",
            "where",
            "segment",
            "category",
            "priority",
            "channel",
            "region",
            "concentrat",
        )
    )
    if not localization_requested:
        return collected

    shared_dimensions = [
        column
        for column in left.columns
        if column in right.columns
        and column != metric
        and column not in left.numeric_columns
        and column not in right.numeric_columns
        and column not in left.possible_date_columns
        and column not in right.possible_date_columns
        and column.lower() != "id"
        and not column.lower().endswith("_id")
        and 1 < left.unique_counts.get(column, 0) <= 100
        and 1 < right.unique_counts.get(column, 0) <= 100
    ]
    named_dimensions = [
        column
        for column in shared_dimensions
        if f" {_normalized_words(column)} " in f" {question} "
    ]
    semantic_dimensions = [
        column
        for column in shared_dimensions
        if any(
            token in column.lower()
            for token in ("segment", "category", "priority", "channel", "region", "status")
        )
    ]
    candidates = named_dimensions or semantic_dimensions
    if not candidates:
        return collected
    dimension = candidates[0]

    segment_ids: list[str] = []
    for profile in (left, right):
        existing = next(
            (
                item
                for item in state.evidence
                if item.tool == "segment_analysis"
                and item.supporting_details.get("dataset") == profile.dataset
                and item.supporting_details.get("grouping") == [dimension]
                and item.supporting_details.get("metric_column") == metric
                and item.supporting_details.get("aggregation") == "sum"
            ),
            None,
        )
        if existing:
            segment_ids.append(existing.evidence_id)
            continue
        arguments = {
            "dataset": profile.dataset,
            "group_by_column": dimension,
            "metric_column": metric,
            "aggregation": "sum",
            "filters": [],
            "date_column": "",
            "date_frequency": "D",
            "match_against_dataset": "",
            "local_key": "",
            "other_key": "",
            "only_unmatched": False,
        }
        try:
            response = json.loads(
                auditor.execute(
                    "segment_analysis",
                    arguments,
                    [profile.dataset],
                    lambda profile=profile: segment_analysis(
                        datasets[profile.dataset],
                        dimension,
                        metric,
                        "sum",
                    ),
                )
            )
            collected += 1
            segment_ids.append(response["evidence_id"])
        except (ToolInputError, ValueError) as exc:
            add_trace(
                state,
                "quality",
                f"Required `{dimension}` segment comparison could not be collected: {exc}",
            )
    if len(segment_ids) == 2:
        add_trace(
            state,
            "evidence",
            f"Controller required paired `{dimension}` totals ({'/'.join(segment_ids)})",
        )
    return collected


def _question_detail_columns(
    question: str, profile: DatasetProfile, key: str
) -> list[str] | None:
    """Return columns named by the user; otherwise let the tool choose context."""
    normalized_question = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    padded_question = f" {normalized_question} "
    requested: list[str] = []
    for column in profile.columns:
        if column == key:
            continue
        normalized_column = re.sub(r"[^a-z0-9]+", " ", column.lower()).strip()
        if normalized_column and f" {normalized_column} " in padded_question:
            requested.append(column)
    return requested[:8] or None


def _key_score(
    key: str, left: DatasetProfile, right: DatasetProfile
) -> tuple[float, int, str]:
    left_ratio = left.unique_counts.get(key, 0) / max(left.row_count, 1)
    right_ratio = right.unique_counts.get(key, 0) / max(right.row_count, 1)
    name_score = 0 if key.lower() == "id" or key.lower().endswith("_id") else 1
    return (-(left_ratio + right_ratio), name_score, key.lower())


def _relationships(profiles: list[DatasetProfile]) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str, float, int, str]] = []
    for index, left in enumerate(profiles):
        for right in profiles[index + 1 :]:
            shared = set(left.possible_identifier_columns) & set(
                right.possible_identifier_columns
            )
            for key in shared:
                score = _key_score(key, left, right)
                candidates.append(
                    (left.dataset, right.dataset, key, score[0], score[1], score[2])
                )
    candidates.sort(key=lambda item: item[3:])
    selected: list[tuple[str, str, str]] = []
    used_pairs: set[tuple[str, str]] = set()
    for left, right, key, *_ in candidates:
        pair = (left, right)
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        selected.append((left, right, key))
        if len(selected) >= MAX_BASELINE_RELATIONSHIPS:
            break
    return selected


def collect_baseline_evidence(state: InvestigationState) -> int:
    """Run mandatory count, key, exact-duplicate, and row checks."""
    datasets = {Path(path).name: path for path in state.dataset_paths}
    auditor = ToolAuditor(state, datasets)
    relationships = _relationships(state.dataset_profiles)
    if not relationships:
        add_trace(
            state,
            "quality",
            "No shared high-confidence identifier was inferred; investigator must map keys.",
        )
        return 0

    before = len(state.evidence)

    def execute(name, arguments, dataset_names, function) -> dict | None:
        try:
            return json.loads(
                auditor.execute(name, arguments, dataset_names, function)
            )
        except (ToolInputError, ValueError) as exc:
            add_trace(state, "quality", f"Baseline {name} check skipped: {exc}")
            return None

    for left, right, key in relationships:
        execute(
            "compare_aggregates",
            {
                "dataset_a": left,
                "dataset_b": right,
                "metric_column_a": "",
                "metric_column_b": "",
                "aggregation": "count",
            },
            [left, right],
            lambda left=left, right=right: compare_aggregates(
                datasets[left], datasets[right], None, None, "count"
            ),
        )
        unmatched_results = []
        profile_by_name = {item.dataset: item for item in state.dataset_profiles}
        for dataset_a, dataset_b in ((left, right), (right, left)):
            detail_columns = _question_detail_columns(
                state.question, profile_by_name[dataset_a], key
            )
            unmatched_result = execute(
                "find_unmatched_records",
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "key_column_a": key,
                    "key_column_b": key,
                    "detail_columns": detail_columns,
                },
                [dataset_a, dataset_b],
                lambda dataset_a=dataset_a, dataset_b=dataset_b,
                detail_columns=detail_columns: find_unmatched_records(
                    datasets[dataset_a], datasets[dataset_b], key, key, detail_columns
                ),
            )
            if unmatched_result:
                unmatched_results.append(unmatched_result)
        for dataset in (left, right):
            execute(
                "find_duplicates",
                {"dataset": dataset, "key_columns": []},
                [dataset],
                lambda dataset=dataset: find_duplicates(datasets[dataset], None),
            )
        row_result = execute(
            "compare_rows_by_key",
            {
                "dataset_a": left,
                "dataset_b": right,
                "key_column_a": key,
                "key_column_b": key,
                "columns": [],
                "tolerance": 0.0,
            },
            [left, right],
            lambda left=left, right=right: compare_rows_by_key(
                datasets[left], datasets[right], key, key
            ),
        )
        if row_result:
            differing_columns = [
                column
                for column, count in row_result.get("mismatches_by_column", {}).items()
                if count > 0
            ]
            left_profile = next(item for item in state.dataset_profiles if item.dataset == left)
            right_profile = next(item for item in state.dataset_profiles if item.dataset == right)
            numeric_shared = set(left_profile.numeric_columns) & set(
                right_profile.numeric_columns
            )
            for column in [item for item in differing_columns if item in numeric_shared][:3]:
                execute(
                    "compare_record_values",
                    {
                        "dataset_a": left,
                        "dataset_b": right,
                        "key_column_a": key,
                        "key_column_b": key,
                        "value_column_a": column,
                        "value_column_b": column,
                        "tolerance": 0.0,
                    },
                    [left, right],
                    lambda left=left, right=right, column=column: compare_record_values(
                        datasets[left],
                        datasets[right],
                        key,
                        key,
                        column,
                        column,
                    ),
                )
        for unmatched_result in unmatched_results:
            if unmatched_result.get("unmatched_count", 0) <= 0:
                continue
            dataset = unmatched_result["dataset_a"]
            other = unmatched_result["dataset_b"]
            profile = profile_by_name[dataset]
            amount_candidates = [
                column
                for column in profile.numeric_columns
                if any(
                    token in column.lower()
                    for token in ("amount", "total", "price", "revenue", "cost", "balance")
                )
            ]
            dimensions = [
                column
                for column in profile.columns
                if column not in profile.possible_identifier_columns
                and column not in profile.possible_date_columns
                and column not in profile.numeric_columns
                and 1 < profile.unique_counts.get(column, 0) <= 100
            ]
            dimensions.sort(
                key=lambda column: (
                    0
                    if any(
                        token in column.lower()
                        for token in ("channel", "region", "segment", "priority", "status")
                    )
                    else 1,
                    column,
                )
            )
            execute(
                "calculate_business_impact",
                {
                    "dataset": dataset,
                    "amount_column": amount_candidates[0] if amount_candidates else "",
                    "date_column": profile.possible_date_columns[0]
                    if profile.possible_date_columns
                    else "",
                    "segment_column": dimensions[0] if dimensions else "",
                    "filters": [],
                    "match_against_dataset": other,
                    "local_key": key,
                    "other_key": key,
                    "only_unmatched": True,
                },
                [dataset, other],
                lambda dataset=dataset, other=other, profile=profile,
                amount_candidates=amount_candidates, dimensions=dimensions: calculate_business_impact(
                    datasets[dataset],
                    amount_candidates[0] if amount_candidates else None,
                    profile.possible_date_columns[0]
                    if profile.possible_date_columns
                    else None,
                    dimensions[0] if dimensions else None,
                    [],
                    datasets[other],
                    key,
                    key,
                    True,
                ),
            )
    collected = len(state.evidence) - before
    add_trace(
        state,
        "evidence",
        f"Deterministic baseline collected {collected} evidence item(s) before LLM reasoning",
    )
    return collected
