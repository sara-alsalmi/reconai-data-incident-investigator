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
)
from src.tools.common import ToolInputError


MAX_BASELINE_RELATIONSHIPS = 2


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
