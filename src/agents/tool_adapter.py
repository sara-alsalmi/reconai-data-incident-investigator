"""Audited CrewAI wrappers around the six deterministic tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.models import EvidenceItem, InvestigationState
from src.state import add_trace
from src.tools import (
    calculate_business_impact,
    compare_aggregates,
    find_duplicates,
    find_unmatched_records,
    profile_dataset,
    segment_analysis,
)


def _safe_json(value: str, expected: type) -> Any:
    parsed = json.loads(value or ("[]" if expected is list else "{}"))
    if not isinstance(parsed, expected):
        raise ValueError(f"Expected JSON {expected.__name__}")
    return parsed


class ToolAuditor:
    def __init__(self, state: InvestigationState, datasets: dict[str, str]):
        self.state = state
        self.datasets = datasets

    def path(self, name: str) -> str:
        if name not in self.datasets:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {', '.join(sorted(self.datasets))}"
            )
        return self.datasets[name]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        datasets: list[str],
        function: Callable[[], dict],
    ) -> str:
        canonical = json.dumps(arguments, sort_keys=True, default=str)
        signature = hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()[:16]
        if signature in self.state.tool_call_signatures:
            add_trace(
                self.state,
                "decision",
                f"Skipped repeated {tool_name} call; retry must collect different evidence",
            )
            return json.dumps(
                {
                    "error": "This exact tool call was already made. Choose a different "
                    "tool, filter, grouping, metric, direction, or time grain."
                }
            )
        self.state.tool_call_signatures.append(signature)
        result = function()
        evidence_id = f"E{len(self.state.evidence) + 1:03d}"
        finding, metric, value = _summarize(tool_name, result)
        self.state.evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                attempt=self.state.attempt_count,
                finding=finding,
                tool=tool_name,
                datasets=datasets,
                metric=metric,
                value=value,
                supporting_details=result,
                call_signature=signature,
            )
        )
        add_trace(
            self.state,
            "tool",
            f"{tool_name} produced {evidence_id}: {finding}",
        )
        response = dict(result)
        response["evidence_id"] = evidence_id
        return json.dumps(response, default=str)


def _summarize(tool: str, result: dict) -> tuple[str, str | None, Any]:
    if tool == "profile_dataset":
        return (
            f"Profiled {result['dataset']}: {result['row_count']} rows and "
            f"{len(result['columns'])} columns",
            "row_count",
            result["row_count"],
        )
    if tool == "compare_aggregates":
        return (
            f"{result['aggregation']} differs by {result['absolute_difference']} between "
            f"{result['dataset_a']} and {result['dataset_b']}",
            "absolute_difference",
            result["absolute_difference"],
        )
    if tool == "find_unmatched_records":
        return (
            f"Found {result['unmatched_count']} unmatched records "
            f"({result['unmatched_percentage']:.2f}%) in {result['direction']}",
            "unmatched_count",
            result["unmatched_count"],
        )
    if tool == "find_duplicates":
        return (
            f"Found {result['duplicate_row_count']} duplicate rows in {result['dataset']}",
            "duplicate_row_count",
            result["duplicate_row_count"],
        )
    if tool == "segment_analysis":
        leader = result["results"][0] if result["results"] else None
        return (
            f"Analyzed {result['filtered_record_count']} records across "
            f"{result['result_count']} segments; leading result: {leader}",
            "filtered_record_count",
            result["filtered_record_count"],
        )
    return (
        f"Calculated impact for {result['affected_record_count']} affected records in "
        f"{result['dataset']}",
        "affected_record_count",
        result["affected_record_count"],
    )


def build_crewai_tools(state: InvestigationState) -> list[Any]:
    from crewai.tools import tool

    datasets = {Path(path).name: path for path in state.dataset_paths}
    auditor = ToolAuditor(state, datasets)

    @tool("profile_dataset")
    def profile_dataset_tool(dataset: str) -> str:
        """Profile a named uploaded CSV. Returns compact schema, quality, and summary facts."""
        args = {"dataset": dataset}
        return auditor.execute(
            "profile_dataset", args, [dataset], lambda: profile_dataset(auditor.path(dataset))
        )

    @tool("compare_aggregates")
    def compare_aggregates_tool(
        dataset_a: str,
        dataset_b: str,
        metric_column_a: str = "",
        metric_column_b: str = "",
        aggregation: str = "sum",
    ) -> str:
        """Compare sum, count, or mean across two named datasets. Blank metric is valid only for row count."""
        args = locals().copy()
        return auditor.execute(
            "compare_aggregates",
            args,
            [dataset_a, dataset_b],
            lambda: compare_aggregates(
                auditor.path(dataset_a),
                auditor.path(dataset_b),
                metric_column_a or None,
                metric_column_b or None,
                aggregation,
            ),
        )

    @tool("find_unmatched_records")
    def find_unmatched_records_tool(
        dataset_a: str, dataset_b: str, key_column_a: str, key_column_b: str
    ) -> str:
        """Find records in dataset A whose key does not occur in dataset B. Direction matters."""
        args = locals().copy()
        return auditor.execute(
            "find_unmatched_records",
            args,
            [dataset_a, dataset_b],
            lambda: find_unmatched_records(
                auditor.path(dataset_a), auditor.path(dataset_b), key_column_a, key_column_b
            ),
        )

    @tool("find_duplicates")
    def find_duplicates_tool(dataset: str, key_columns_json: str = "[]") -> str:
        """Find duplicate rows in a dataset, optionally using a JSON list of key columns."""
        keys = _safe_json(key_columns_json, list)
        args = {"dataset": dataset, "key_columns": keys}
        return auditor.execute(
            "find_duplicates",
            args,
            [dataset],
            lambda: find_duplicates(auditor.path(dataset), keys or None),
        )

    @tool("segment_analysis")
    def segment_analysis_tool(
        dataset: str,
        group_by_column: str = "",
        metric_column: str = "",
        aggregation: str = "count",
        filters_json: str = "[]",
        date_column: str = "",
        date_frequency: str = "D",
        match_against_dataset: str = "",
        local_key: str = "",
        other_key: str = "",
        only_unmatched: bool = False,
    ) -> str:
        """Group filtered records by a segment and/or date. Can analyze only rows unmatched to another dataset."""
        filters = _safe_json(filters_json, list)
        args = {
            "dataset": dataset,
            "group_by_column": group_by_column,
            "metric_column": metric_column,
            "aggregation": aggregation,
            "filters": filters,
            "date_column": date_column,
            "date_frequency": date_frequency,
            "match_against_dataset": match_against_dataset,
            "local_key": local_key,
            "other_key": other_key,
            "only_unmatched": only_unmatched,
        }
        return auditor.execute(
            "segment_analysis",
            args,
            [name for name in (dataset, match_against_dataset) if name],
            lambda: segment_analysis(
                auditor.path(dataset),
                group_by_column or None,
                metric_column or None,
                aggregation,
                filters,
                date_column or None,
                date_frequency,
                auditor.path(match_against_dataset) if match_against_dataset else None,
                local_key or None,
                other_key or None,
                only_unmatched,
            ),
        )

    @tool("calculate_business_impact")
    def calculate_business_impact_tool(
        dataset: str,
        amount_column: str = "",
        date_column: str = "",
        segment_column: str = "",
        filters_json: str = "[]",
        match_against_dataset: str = "",
        local_key: str = "",
        other_key: str = "",
        only_unmatched: bool = False,
    ) -> str:
        """Calculate affected count, percentage, exact amount, date range, and top segment from filtered/unmatched data."""
        filters = _safe_json(filters_json, list)
        args = {
            "dataset": dataset,
            "amount_column": amount_column,
            "date_column": date_column,
            "segment_column": segment_column,
            "filters": filters,
            "match_against_dataset": match_against_dataset,
            "local_key": local_key,
            "other_key": other_key,
            "only_unmatched": only_unmatched,
        }
        return auditor.execute(
            "calculate_business_impact",
            args,
            [name for name in (dataset, match_against_dataset) if name],
            lambda: calculate_business_impact(
                auditor.path(dataset),
                amount_column or None,
                date_column or None,
                segment_column or None,
                filters,
                auditor.path(match_against_dataset) if match_against_dataset else None,
                local_key or None,
                other_key or None,
                only_unmatched,
            ),
        )

    return [
        profile_dataset_tool,
        compare_aggregates_tool,
        find_unmatched_records_tool,
        find_duplicates_tool,
        segment_analysis_tool,
        calculate_business_impact_tool,
    ]

