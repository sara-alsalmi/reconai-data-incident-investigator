"""Hybrid runtime with LLM investigation/verification and deterministic support stages."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.agents.investigator import create_investigator_agent
from src.agents.terminal_listener import ensure_terminal_listener
from src.agents.tool_adapter import build_crewai_tools
from src.agents.verifier import create_verifier_agent
from src.config import Settings
from src.models import (
    BusinessReportDraft,
    DatasetProfile,
    InvestigationAttempt,
    InvestigationState,
    VerificationResult,
)


class ProfileReview(BaseModel):
    datasets_reviewed: list[str]
    quality_observations: list[str] = Field(default_factory=list)
    relationship_candidates: list[str] = Field(default_factory=list)


def build_openrouter_llm(settings: Settings) -> Any:
    from crewai import LLM

    model = settings.openrouter_model
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    return LLM(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        max_tokens=8192,
    )


_DIMENSION_PRIORITY = (
    "channel",
    "region",
    "customer_segment",
    "segment",
    "platform",
    "source",
    "origin",
    "product",
    "category",
    "country",
    "payment_method",
    "status",
)


def _dimension_rank(column: str) -> tuple[int, str]:
    lowered = column.lower()
    for index, keyword in enumerate(_DIMENSION_PRIORITY):
        if keyword in lowered:
            return index, lowered
    return len(_DIMENSION_PRIORITY), lowered


def evidence_quality_guide(profiles: list[DatasetProfile], question: str) -> str:
    """Build dataset-aware guidance without prescribing a fixed tool path."""
    lines: list[str] = []
    for profile in profiles:
        excluded = set(profile.possible_identifier_columns)
        excluded.update(profile.possible_date_columns)
        excluded.update(profile.numeric_columns)
        informative = [
            column
            for column in profile.columns
            if column not in excluded
            and 1 < profile.unique_counts.get(column, 0) <= 100
        ]
        informative.sort(key=_dimension_rank)
        constants = [
            column
            for column in profile.columns
            if profile.unique_counts.get(column, 0) <= 1
        ]
        lines.append(
            f"- {profile.dataset}: ranked informative dimensions={informative or ['none']}; "
            f"constant/non-informative columns={constants or ['none']}; "
            f"date candidates={profile.possible_date_columns or ['none']}"
        )
    relationship_summaries: set[str] = set()
    for profile in profiles:
        for relationship in profile.possible_relationships:
            relationship_summaries.add(
                f"- Observed relationship: {profile.dataset}."
                f"{relationship['local_column']} to {relationship['dataset']}."
                f"{relationship['other_column']} is "
                f"{relationship.get('cardinality', 'unknown-cardinality')}."
            )
    lines.extend(sorted(relationship_summaries))
    lowered_question = question.lower()
    comparison_terms = (
        "match",
        "mismatch",
        "differ",
        "difference",
        "reconcile",
        "revenue",
        "total",
    )
    if any(term in lowered_question for term in comparison_terms):
        lines.extend(
            [
                "- This appears to be a cross-dataset reconciliation question.",
                "- First establish whether a discrepancy exists: compare the relevant count "
                "or aggregate, test shared keys in the relevant direction(s), and compare "
                "values for matched keys when relevant. For a broad whole-row question, use "
                "compare_rows_by_key once instead of comparing every field separately.",
                "- If those checks show no discrepancy, stop and report that result. Do not "
                "search for a segment, date, impact, or cause that the evidence does not show.",
                "- Only after proving a discrepancy, test duplicates if useful, locate it in "
                "an informative business dimension and time range, and calculate only directly "
                "measurable impact.",
                "- A constant field such as a one-value status is not an affected segment. "
                "Prefer the earliest relevant ranked dimension and explain any skipped target.",
            ]
        )
    return "\n".join(lines)


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def _relationship_cardinality(
    state: InvestigationState, dataset_a: str, dataset_b: str
) -> str | None:
    for profile in state.dataset_profiles:
        if profile.dataset != dataset_a:
            continue
        for relationship in profile.possible_relationships:
            if relationship.get("dataset") == dataset_b:
                return relationship.get("cardinality")
    reverse = {
        "one_to_many": "many_to_one",
        "many_to_one": "one_to_many",
        "one_to_one": "one_to_one",
        "many_to_many": "many_to_many",
    }
    for profile in state.dataset_profiles:
        if profile.dataset != dataset_b:
            continue
        for relationship in profile.possible_relationships:
            if relationship.get("dataset") == dataset_a:
                return reverse.get(relationship.get("cardinality"))
    return None


def _question_focus_tools(question: str) -> set[str] | None:
    """Return tools relevant to a focused question, or None for broad investigations."""
    lowered = question.lower()
    tools: set[str] = set()
    if any(
        term in lowered
        for term in ("missing", "unmatched", "absent", "coverage", "correspond")
    ):
        tools.update({"find_unmatched_records", "calculate_business_impact"})
    duplicate_forbidden = "do not" in lowered and any(
        term in lowered for term in ("duplicate", "repeated", "repeat")
    )
    if not duplicate_forbidden and any(
        term in lowered for term in ("duplicate", "repeated", "repeat")
    ):
        tools.add("find_duplicates")
    value_analysis = any(
        term in lowered
        for term in ("amount", "total", "revenue", "balance", "sum", "field", "column")
    ) or (
        "value" in lowered
        and any(term in lowered for term in ("differ", "mismatch"))
    )
    if value_analysis:
        tools.update(
            {
                "compare_aggregates",
                "compare_record_values",
                "compare_rows_by_key",
                "calculate_business_impact",
            }
        )
    if any(
        term in lowered
        for term in (
            "segment",
            "region",
            "channel",
            "category",
            "priority",
            "where",
            "when",
            "period",
            "date",
            "month",
            "year",
        )
    ):
        tools.update({"segment_analysis", "calculate_business_impact"})
    return tools or None


def _report_evidence(state: InvestigationState) -> list[Any]:
    focus_tools = _question_focus_tools(state.question)
    candidates = state.evidence if focus_tools is None else [
        item for item in state.evidence if item.tool in focus_tools
    ]
    narrower_issue_exists = any(
        (
            item.tool == "find_unmatched_records"
            and item.supporting_details.get("unmatched_count", 0) > 0
        )
        or (
            item.tool == "compare_record_values"
            and item.supporting_details.get("value_mismatch_count", 0) > 0
        )
        or (
            item.tool == "compare_rows_by_key"
            and item.supporting_details.get("record_mismatch_count", 0) > 0
        )
        for item in state.evidence
    )
    relevant = []
    lowered_question = state.question.lower()
    contextual_impact_requested = any(
        term in lowered_question
        for term in (
            "amount",
            "value",
            "total",
            "impact",
            "where",
            "when",
            "period",
            "date",
            "time",
            "segment",
            "region",
            "channel",
            "category",
            "priority",
        )
    )
    duplicate_requested = any(
        term in lowered_question for term in ("duplicate", "repeated", "repeat")
    ) and not any(
        phrase in lowered_question
        for phrase in ("do not classify", "not duplicate", "are valid")
    )
    for item in candidates:
        if (
            item.tool == "find_duplicates"
            and item.supporting_details.get("analysis_type") == "repeated_key"
            and not duplicate_requested
        ):
            continue
        if (
            item.tool == "find_duplicates"
            and item.supporting_details.get("analysis_type") != "repeated_key"
            and item.supporting_details.get("duplicate_row_count", 0) == 0
            and not duplicate_requested
        ):
            continue
        if item.tool == "compare_aggregates" and item.supporting_details.get(
            "aggregation"
        ) == "count":
            cardinality = _relationship_cardinality(
                state,
                item.supporting_details.get("dataset_a", ""),
                item.supporting_details.get("dataset_b", ""),
            )
            if cardinality and cardinality != "one_to_one":
                continue
        if (
            item.tool == "calculate_business_impact"
            and focus_tools is not None
            and focus_tools <= {"find_unmatched_records", "calculate_business_impact"}
            and not contextual_impact_requested
        ):
            continue
        if (
            item.tool == "calculate_business_impact"
            and narrower_issue_exists
            and item.supporting_details.get("affected_record_count")
            == item.supporting_details.get("total_record_count")
        ):
            continue
        relevant.append(item)
    return relevant


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_header(column: str) -> str:
    """Turn source column names into readable, reliably separated table headers."""
    words = column.replace("_", " ").split()
    if not words:
        return column
    readable = ["ID" if word.lower() == "id" else word.lower() for word in words]
    readable[0] = readable[0] if readable[0] == "ID" else readable[0].capitalize()
    return " ".join(readable)


def _affected_details_markdown(report_evidence: list[Any]) -> str:
    """Render bounded affected-record previews without implying a sample is complete."""
    sections: list[str] = []
    for item in report_evidence:
        if item.tool != "find_unmatched_records":
            continue
        details = item.supporting_details
        records = details.get("sample_unmatched_records") or []
        if not records:
            continue
        unique_count = int(
            details.get("unmatched_unique_key_count", details.get("unmatched_count", 0))
        )
        columns: list[str] = []
        for record in records:
            for column in record:
                if column not in columns:
                    columns.append(column)
        columns = columns[:3]
        dataset_a = details.get("dataset_a", "dataset A")
        dataset_b = details.get("dataset_b", "dataset B")
        completeness = (
            "Showing the affected record"
            if unique_count == 1 and details.get("sample_is_complete", True)
            else f"Showing all {unique_count} affected key values"
            if details.get("sample_is_complete", len(records) >= unique_count)
            else f"Showing {len(records)} of {unique_count} affected key values"
        )
        header = "| " + " | ".join(_markdown_header(column) for column in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = [
            "| "
            + " | ".join(_markdown_cell(record.get(column)) for column in columns)
            + " |"
            for record in records
        ]
        sections.append(
            f"### `{dataset_a}` · {item.evidence_id}\n\n"
            f"{completeness}. No matching key was found in `{dataset_b}`.\n\n"
            f"{header}\n{separator}\n" + "\n".join(rows)
        )
    for item in report_evidence:
        if item.tool != "compare_record_values":
            continue
        details = item.supporting_details
        records = details.get("sample_mismatches") or []
        if not records:
            continue
        mismatch_count = int(details.get("value_mismatch_count", len(records)))
        columns = list(records[0])
        header = "| " + " | ".join(_markdown_header(column) for column in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = [
            "| "
            + " | ".join(_markdown_cell(record.get(column)) for column in columns)
            + " |"
            for record in records
        ]
        completeness = (
            f"All {mismatch_count} mismatched key value"
            + ("" if mismatch_count == 1 else "s")
            if len(records) >= mismatch_count
            else f"Preview of {len(records)} out of {mismatch_count} mismatched key values"
        )
        sections.append(
            f"### Matched-key value differences · {item.evidence_id}\n\n"
            f"{completeness}:\n\n"
            f"{header}\n{separator}\n" + "\n".join(rows)
        )
    return "\n\n".join(sections)


def _direct_answer(
    state: InvestigationState, conclusive: bool, report_evidence: list[Any]
) -> str:
    if not conclusive:
        return (
            "The available evidence was not sufficient for a reliable conclusion. "
            "Review the evidence below before taking corrective action."
        )
    hypothesis = state.hypothesis or "The performed checks did not establish a discrepancy."
    focus_tools = _question_focus_tools(state.question)
    key_gap_conclusion = (
        "key gap" in hypothesis.lower()
        or "absent from" in hypothesis.lower()
        or (
            focus_tools is not None
            and focus_tools <= {"find_unmatched_records", "calculate_business_impact"}
        )
    )
    key_checks = [
        item for item in report_evidence if item.tool == "find_unmatched_records"
    ]
    if not key_gap_conclusion or not key_checks:
        return hypothesis

    positive: list[str] = []
    zero: list[str] = []
    for item in key_checks:
        details = item.supporting_details
        count = int(
            details.get("unmatched_unique_key_count", details.get("unmatched_count", 0))
        )
        key = details.get("key_column_a", "key")
        dataset_a = details.get("dataset_a", "dataset A")
        dataset_b = details.get("dataset_b", "dataset B")
        if count:
            positive.append(
                f"`{dataset_a}` has **{count}** `{key}` "
                f"{'value' if count == 1 else 'values'} with no matching record in "
                f"`{dataset_b}` ({item.evidence_id})."
            )
        else:
            zero.append(
                f"All `{key}` values in `{dataset_a}` have a matching record in "
                f"`{dataset_b}` ({item.evidence_id})."
            )
    return " ".join(
        [
            *positive,
            *zero,
            "This confirms the data-level gap; the uploaded files do not show why it happened.",
        ]
    )


def _contextual_actions(report_evidence: list[Any], fallback: list[str]) -> list[str]:
    """Create actionable follow-up steps when an affected key is known."""
    unmatched = [
        item
        for item in report_evidence
        if item.tool == "find_unmatched_records"
        and item.supporting_details.get("unmatched_count", 0) > 0
    ]
    if not unmatched:
        return fallback[:3]

    item = unmatched[0]
    details = item.supporting_details
    dataset_a = details.get("dataset_a", "the affected dataset")
    dataset_b = details.get("dataset_b", "the comparison dataset")
    key = details.get("key_column_a", "key")
    records = details.get("sample_unmatched_records") or []
    unique_count = int(
        details.get("unmatched_unique_key_count", details.get("unmatched_count", 0))
    )

    if unique_count == 1 and records and records[0].get(key) is not None:
        key_value = _markdown_cell(records[0][key])
        first = (
            f"Search the source system behind `{dataset_b}` for `{key}={key_value}` "
            "and confirm whether the corresponding record exists elsewhere."
        )
    else:
        first = (
            f"Reconcile the {unique_count} affected `{key}` values from `{dataset_a}` "
            f"against the source system behind `{dataset_b}`."
        )

    dated_context = None
    if records:
        dated_context = next(
            (
                (column, value)
                for column, value in records[0].items()
                if column != key
                and value is not None
                and (
                    column.lower().endswith("_at")
                    or any(
                        hint in column.lower()
                        for hint in ("date", "time", "timestamp")
                    )
                )
            ),
            None,
        )
    if dated_context:
        column, value = dated_context
        second = (
            f"Review ingestion and processing logs around `{_markdown_cell(value)}` "
            f"from `{dataset_a}` (`{column}`)."
        )
    else:
        second = "Review source-system and ingestion logs to locate where the relationship was lost."

    return [
        first,
        second,
        "Correct or backfill data only after confirming the system of record.",
    ]


def _concise_evidence(report_evidence: list[Any]) -> list[str]:
    """Translate tool records into short analyst-facing evidence statements."""
    lines: list[str] = []
    exact_duplicate_zeros: list[Any] = []
    key_checks = [
        item for item in report_evidence if item.tool == "find_unmatched_records"
    ]
    has_single_record_preview = any(
        item.supporting_details.get(
            "unmatched_unique_key_count",
            item.supporting_details.get("unmatched_count", 0),
        )
        == 1
        and bool(item.supporting_details.get("sample_unmatched_records"))
        for item in key_checks
    )
    if key_checks:
        keys = sorted(
            {
                item.supporting_details.get("key_column_a", "key")
                for item in key_checks
            }
        )
        datasets = sorted(
            {
                str(dataset)
                for item in key_checks
                for dataset in (
                    item.supporting_details.get("dataset_a"),
                    item.supporting_details.get("dataset_b"),
                )
                if dataset
            }
        )
        evidence_ids = "/".join(item.evidence_id for item in key_checks)
        lines.append(
            f"{evidence_ids} — Checked `{', '.join(keys)}` coverage in both directions "
            f"between " + " and ".join(f"`{dataset}`" for dataset in datasets) + "."
        )
    for item in report_evidence:
        details = item.supporting_details
        if item.tool == "find_unmatched_records":
            continue
        if item.tool == "find_duplicates":
            if details.get("analysis_type") == "repeated_key":
                lines.append(
                    f"{item.evidence_id} — {details.get('duplicate_row_count', 0)} rows "
                    "participate in repeated keys; this alone is not a duplicate error."
                )
            elif details.get("duplicate_row_count", 0) == 0:
                exact_duplicate_zeros.append(item)
            else:
                lines.append(f"{item.evidence_id} — {item.finding}.")
            continue
        if item.tool == "calculate_business_impact":
            if has_single_record_preview and details.get("affected_amount_sum") is None:
                continue
            measured = ["affected record count"]
            if details.get("affected_date_range"):
                measured.append("date range")
            if details.get("affected_amount_sum") is not None:
                measured.append("numerical impact")
            if (
                details.get("primary_affected_segment")
                and details.get("affected_record_count", 0) > 1
            ):
                measured.append("leading segment")
            if len(measured) > 1:
                lines.append(
                    f"{item.evidence_id} — Measured " + ", ".join(measured) + "."
                )
            continue
        line = f"{item.evidence_id} — {item.finding.rstrip('.')}."
        if line not in lines:
            lines.append(line)
    if exact_duplicate_zeros:
        datasets = [
            f"`{item.supporting_details.get('dataset', 'dataset')}`"
            for item in exact_duplicate_zeros
        ]
        evidence_ids = "/".join(item.evidence_id for item in exact_duplicate_zeros)
        lines.append(
            f"{evidence_ids} — No exact duplicate rows were found in "
            + " or ".join(datasets)
            + "."
        )
    return lines


def _useful_scope_value(value: str) -> bool:
    lowered = value.lower()
    return not lowered.startswith("not ") and "not established" not in lowered


def _render_report(
    draft: BusinessReportDraft,
    state: InvestigationState,
    conclusive: bool,
) -> str:
    status = "Finding confirmed from uploaded files" if conclusive else "Inconclusive"
    verifier_confidence = state.verification.confidence if state.verification else 0.0
    confidence = (
        "High"
        if conclusive and verifier_confidence >= 0.85
        else "Medium"
        if conclusive and verifier_confidence >= 0.65
        else "Low"
    )
    report_evidence = _report_evidence(state)
    fallback_actions = draft.recommended_actions or [
        "Collect the missing source-system evidence."
    ]
    actions = _contextual_actions(report_evidence, fallback_actions)
    uncertainty = list(draft.remaining_uncertainty)
    technical_uncertainty = "The underlying technical mechanism is not proven by CSV evidence."
    if not any("technical" in item.lower() for item in uncertainty):
        uncertainty.append(technical_uncertainty)
    impact = next(
        (
            item
            for item in reversed(report_evidence)
            if item.tool == "calculate_business_impact"
            and (
                not isinstance(
                    item.supporting_details.get("total_record_count"), (int, float)
                )
                or item.supporting_details.get("affected_record_count", 0)
                < item.supporting_details["total_record_count"]
            )
        ),
        None,
    )
    affected_records = "Not established from the performed checks."
    focus_tools = _question_focus_tools(state.question)
    strict_scope = "only" in state.question.lower() or (
        focus_tools is not None
        and focus_tools <= {"find_unmatched_records", "calculate_business_impact"}
    )
    segment_requested = any(
        term in state.question.lower()
        for term in ("segment", "region", "channel", "category", "priority", "where")
    )
    time_requested = any(
        term in state.question.lower()
        for term in ("when", "period", "date", "month", "year", "time")
    )
    value_requested = any(
        term in state.question.lower()
        for term in ("amount", "value", "total", "revenue", "balance", "sum", "impact")
    )
    affected_segment = (
        "Not requested in the investigation question."
        if strict_scope and not segment_requested
        else "Not established from the performed checks."
    )
    affected_period = (
        "Not requested in the investigation question."
        if strict_scope and not time_requested
        else "Not established from the performed checks."
    )
    measurable_impact = (
        "Not requested in the investigation question."
        if strict_scope and not value_requested
        else "Not calculated from the available columns."
    )
    count_descriptions: list[str] = []
    seen_count_kinds: set[tuple[Any, ...]] = set()
    count_specs = (
        ("find_unmatched_records", "unmatched_count", "unmatched rows"),
        ("find_duplicates", "duplicate_row_count", "exact duplicate rows"),
        ("compare_record_values", "value_mismatch_count", "matched-key rows with value mismatches"),
        ("compare_rows_by_key", "record_mismatch_count", "matched-key rows with field mismatches"),
    )
    for tool, field, label in count_specs:
        for item in report_evidence:
            if item.tool != tool:
                continue
            if (
                tool == "find_duplicates"
                and item.supporting_details.get("analysis_type") == "repeated_key"
            ):
                continue
            value = item.supporting_details.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            count = int(value)
            dataset = item.supporting_details.get("dataset_a") or item.supporting_details.get(
                "dataset"
            )
            kind = (
                "matched_mismatch" if tool.startswith("compare_") else tool,
                dataset,
                count,
            )
            if kind in seen_count_kinds:
                continue
            seen_count_kinds.add(kind)
            location = f" in {dataset}" if dataset else ""
            if tool == "find_unmatched_records":
                unique_count = int(
                    item.supporting_details.get("unmatched_unique_key_count", count)
                )
                key = item.supporting_details.get("key_column_a", "key")
                value_word = "value" if unique_count == 1 else "values"
                row_word = "row" if count == 1 else "rows"
                count_descriptions.append(
                    f"{unique_count} unmatched {key} {value_word}{location} "
                    f"({count} {row_word}; {item.evidence_id})"
                )
            else:
                label = label[:-1] if count == 1 and label.endswith("s") else label
                count_descriptions.append(
                    f"{count} {label}{location} ({item.evidence_id})"
                )
    if count_descriptions:
        affected_records = "; ".join(count_descriptions[:3])

    if impact:
        details = impact.supporting_details
        if not count_descriptions:
            affected_records = f"{details['affected_record_count']} records ({impact.evidence_id})"
        segment = details.get("primary_affected_segment")
        if (
            segment
            and details.get("affected_record_count", 0) > 1
            and (not strict_scope or segment_requested)
        ):
            affected_segment = (
                f"{segment['segment']} — {segment['record_count']} records "
                f"({impact.evidence_id})"
            )
        period = details.get("affected_date_range")
        if period and (not strict_scope or time_requested):
            affected_period = (
                f"{period['start']} to {period['end']} ({impact.evidence_id})"
            )
        amount = details.get("affected_amount_sum")
        if amount is not None and (not strict_scope or value_requested):
            measurable_impact = f"{amount:,.2f} in affected value ({impact.evidence_id})"
    segment_evidence = [
        item
        for item in report_evidence
        if item.tool == "segment_analysis"
        and len(item.supporting_details.get("grouping", [])) == 1
        and not item.supporting_details.get("results_truncated")
    ]
    for index, left in enumerate(segment_evidence):
        left_details = left.supporting_details
        grouping = left_details["grouping"]
        for right in segment_evidence[index + 1 :]:
            right_details = right.supporting_details
            if (
                left_details.get("dataset") == right_details.get("dataset")
                or grouping != right_details.get("grouping")
                or left_details.get("metric_column") != right_details.get("metric_column")
                or left_details.get("aggregation") != right_details.get("aggregation")
            ):
                continue
            dimension = grouping[0]
            left_values = {
                row.get(dimension): float(row["value"])
                for row in left_details.get("results", [])
            }
            right_values = {
                row.get(dimension): float(row["value"])
                for row in right_details.get("results", [])
            }
            differing = [
                segment
                for segment in left_values.keys() | right_values.keys()
                if round(left_values.get(segment, 0.0) - right_values.get(segment, 0.0), 6)
                != 0
            ]
            if len(differing) == 1 and (not strict_scope or segment_requested):
                affected_segment = (
                    f"{dimension} = {differing[0]} (localized by paired totals, "
                    f"{left.evidence_id}/{right.evidence_id})"
                )
                break
        if affected_segment != "Not established from the performed checks.":
            break

    numeric_mismatches = [
        item
        for item in report_evidence
        if item.tool == "compare_record_values"
        and item.supporting_details.get("value_mismatch_count", 0) > 0
        and isinstance(
            item.supporting_details.get("signed_difference_a_minus_b"), (int, float)
        )
    ]
    sum_differences = [
        item
        for item in report_evidence
        if item.tool == "compare_aggregates"
        and item.supporting_details.get("aggregation") == "sum"
        and isinstance(
            item.supporting_details.get("signed_difference_a_minus_b"), (int, float)
        )
    ]
    difference_evidence = sum_differences[-1] if sum_differences else (
        numeric_mismatches[-1] if numeric_mismatches else None
    )
    if difference_evidence is not None and (not strict_scope or value_requested):
        details = difference_evidence.supporting_details
        measurable_impact = (
            f"{details['signed_difference_a_minus_b']:,.2f} signed difference "
            f"({details['dataset_a']} minus {details['dataset_b']}, "
            f"{difference_evidence.evidence_id})"
        )
    answer = _direct_answer(state, conclusive, report_evidence)
    evidence_lines = _concise_evidence(report_evidence)
    evidence_md = "\n".join(f"- {item}" for item in evidence_lines)
    affected_details_md = _affected_details_markdown(report_evidence)
    previewed_unique_count = sum(
        int(
            item.supporting_details.get(
                "unmatched_unique_key_count",
                item.supporting_details.get("unmatched_count", 0),
            )
        )
        for item in report_evidence
        if item.tool == "find_unmatched_records"
        and item.supporting_details.get("sample_unmatched_records")
    )
    affected_heading = (
        "Affected record" if previewed_unique_count == 1 else "Affected records"
    )
    affected_details_section = (
        f"## {affected_heading}\n\n{affected_details_md}\n\n"
        if affected_details_md
        else ""
    )
    scope_rows: list[tuple[str, str]] = []
    if not affected_details_md and _useful_scope_value(affected_records):
        scope_rows.append(("Affected records", affected_records))
    if _useful_scope_value(affected_segment):
        scope_rows.append(("Affected segment", affected_segment))
    single_record_preview = bool(affected_details_md) and any(
        item.tool == "find_unmatched_records"
        and item.supporting_details.get("unmatched_unique_key_count", 0) == 1
        for item in report_evidence
    )
    if _useful_scope_value(affected_period) and not single_record_preview:
        scope_rows.append(("Affected period", affected_period))
    if _useful_scope_value(measurable_impact):
        scope_rows.append(("Measurable impact", measurable_impact))
    scope_section = ""
    if scope_rows:
        scope_md = "\n".join(
            f"| {label} | {value} |" for label, value in scope_rows
        )
        scope_section = (
            "## Impact and scope\n\n"
            "| Measure | Result |\n| --- | --- |\n"
            f"{scope_md}\n\n"
        )

    unique_uncertainty: list[str] = []
    root_cause_uncertainty = (
        "Technical cause: Not established from the uploaded CSV files."
        if conclusive
        else "Root Cause: Inconclusive. The uploaded CSV files do not establish the "
        "underlying technical mechanism."
    )
    for item in [root_cause_uncertainty, *uncertainty]:
        normalized = item.strip().rstrip(".")
        if not normalized:
            continue
        if "technical" in normalized.lower() and any(
            "technical" in existing.lower() for existing in unique_uncertainty
        ):
            continue
        if normalized.lower() not in {value.lower() for value in unique_uncertainty}:
            unique_uncertainty.append(normalized + ".")
    uncertainty_md = "\n".join(f"- {item}" for item in unique_uncertainty[:3])
    actions_md = "\n".join(
        f"{index}. {item}" for index, item in enumerate(actions[:3], 1)
    )
    evidence_section = (
        f"## Supporting evidence\n\n{evidence_md}\n\n" if evidence_md else ""
    )
    uncertainty_section = (
        f"## What remains unknown\n\n{uncertainty_md}\n\n"
        if uncertainty_md
        else ""
    )
    actions_section = (
        f"## Recommended next steps\n\n{actions_md}\n" if actions_md else ""
    )
    confidence_label = "Confidence in data finding" if conclusive else "Confidence"
    return f"""# Investigation Result

> **Status:** {status} · **{confidence_label}:** {confidence}

## Answer

{answer}

{affected_details_section}{scope_section}{evidence_section}{uncertainty_section}{actions_section}
"""


class CrewAIRuntime:
    """Run two independent LLM agents with deterministic support stages."""

    def __init__(self, state: InvestigationState, settings: Settings | None = None):
        # ReconAI prints its own safe, local terminal trace. Suppress CrewAI's
        # optional hosted-tracing prompt so CLI and Gradio runs never pause for
        # interactive input or suggest uploading investigation traces.
        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
        try:
            from crewai.events.listeners.tracing.utils import (
                set_suppress_tracing_messages,
            )

            set_suppress_tracing_messages(True)
        except Exception:
            pass
        try:
            ensure_terminal_listener()
        except Exception as exc:
            # CrewAI's event classes are an optional observability layer and can
            # change between releases. ReconAI's state trace still prints live,
            # so an event-listener problem must never stop an investigation.
            print(
                "[ReconAI] [TRACE] CrewAI event listener unavailable; "
                f"ReconAI trace remains enabled ({type(exc).__name__}).",
                flush=True,
            )
        self.settings = settings or Settings.from_env(require_api_key=True)
        self.llm = build_openrouter_llm(self.settings)
        tools = build_crewai_tools(state)
        self.investigator = create_investigator_agent(self.llm, tools)
        self.verifier = create_verifier_agent(self.llm)

    def review_profiles(self, state: InvestigationState) -> ProfileReview:
        quality = []
        relationships = []
        for profile in state.dataset_profiles:
            if profile.duplicate_count:
                quality.append(
                    f"{profile.dataset} contains {profile.duplicate_count} duplicate rows."
                )
            null_total = sum(profile.null_counts.values())
            if null_total:
                quality.append(
                    f"{profile.dataset} contains {null_total} null cells across profiled columns."
                )
            for relationship in profile.possible_relationships:
                relationships.append(
                    f"{profile.dataset}.{relationship['local_column']} -> "
                    f"{relationship['dataset']}.{relationship['other_column']}"
                )
        return ProfileReview(
            datasets_reviewed=[profile.dataset for profile in state.dataset_profiles],
            quality_observations=quality,
            relationship_candidates=sorted(set(relationships)),
        )

    def investigate(
        self, state: InvestigationState, previous_verification: VerificationResult | None
    ) -> InvestigationAttempt:
        from src.baseline import collect_question_required_evidence

        collect_question_required_evidence(state)
        evidence = [
            _normalize_numbers(item.model_dump(mode="json")) for item in state.evidence
        ]
        feedback = (
            previous_verification.model_dump(mode="json")
            if previous_verification is not None
            else None
        )
        prompt = f"""
Investigate this enterprise data question adaptively.

Question: {state.question}
Attempt: {state.attempt_count} of 3
Dataset profiles: {json.dumps([p.model_dump(mode='json') for p in state.dataset_profiles])}
Previously collected tool evidence: {json.dumps(evidence, default=str)}
Previous verifier feedback: {json.dumps(feedback, default=str)}

Dataset-aware evidence quality guide:
{evidence_quality_guide(state.dataset_profiles, state.question)}

The controller has already run mandatory count, bidirectional key, exact-duplicate, and
full-shared-row checks for every inferred relationship. Interpret that existing evidence
before calling anything. If it proves no discrepancy, stop without more tools and state only
the checked scope. If it proves a discrepancy, use at most the few additional tools needed to
localize its segment, time range, or measurable impact.

The controller may also have added a question-required comparison during this attempt. Treat
any evidence item whose attempt number matches the current attempt as fresh mandatory evidence,
cite its evidence ID, and do not repeat that exact call.

The user does not need to describe table cardinality or prescribe tools. Infer intent from
the natural-language question, profiles, filenames, shared keys, and observed cardinality.
Unmatched-key evidence contains a bounded affected-record preview with automatically selected
status/date/value context. When the user asks which records are affected, include the actual
identifiers from that evidence; if the preview is truncated, explicitly call it a preview.

Choose useful tools yourself. All important numerical facts must come from tool calls.
Treat the user's requested scope and explicit exclusions as binding requirements. Answer the
question asked; do not elevate unrelated baseline observations into the conclusion.
Do not repeat an exact prior call. On a retry, address the missing evidence and explore a
different grouping, direction, filter, metric, or time grain. Use evidence IDs returned by
tools. For a clean result, limit the conclusion to the keys, measures, and fields actually
checked; never claim that every column is identical without comparing every column. The
compare_record_values tool accepts numeric or categorical values. filters_json must be a JSON
list. compare_rows_by_key compares all shared non-key fields when columns_json is an empty
list. Date frequencies may be D/day, W/week, M/month, Q/quarter, or Y/year. Before
segmenting, inspect profile cardinality: a constant status field cannot explain
concentration. Prefer business-origin dimensions (for example channel, region, customer
segment, platform, or source) over statuses and methods when profiles support them. End with
findings, a cautious data-level root-cause hypothesis, remaining uncertainty, and the evidence
IDs that support it. The hypothesis itself must stop at the data-level explanation. Never put
an ETL, pipeline, software, deletion, ingestion, or operational failure into the hypothesis
unless direct evidence proves it; put possible technical mechanisms only in uncertainty or
recommended follow-up. Association must not be stated as proven causation.

Tool-choice rules for ambiguous cases:
- Use the observed relationship cardinality in the dataset profiles. Raw row counts are not
  directly comparable across one-to-many or many-to-many tables. A repeated foreign key on
  the many side is expected grain, not evidence of a duplicate error.
- find_duplicates with a blank key list checks exact full-row duplicates. With key columns it
  reports records participating in repeated keys and does not prove those repetitions are
  erroneous. Respect any question instruction that declares repeated keys valid.
- Before finishing, check every quantity explicitly requested by the user against the existing
  evidence. If the question requests a net source-versus-target value difference and there is
  no sum comparison for that value column, call compare_aggregates with aggregation=sum for
  the two datasets. Do not substitute the row-count comparison for a value comparison.
- When missing keys and duplicate keys coexist, describe both observed conditions. Do not
  assume duplicate-key rows are identical copies or assign part of an aggregate difference to
  them unless direct evidence proves that. A whole-dataset aggregate comparison may establish
  the net difference independently without decomposing it into unsupported components.
- If the question asks which category contains a value discrepancy but does not ask when,
  call segment_analysis once for each dataset with the same category, metric, and aggregation;
  leave date_column blank. Compare the corresponding category totals. Do not add a time grain
  that the question did not request.
"""
        return self._run_structured(self.investigator, prompt, InvestigationAttempt)

    def verify(
        self, state: InvestigationState, attempt: InvestigationAttempt
    ) -> VerificationResult:
        cited = set(attempt.evidence_ids)
        evidence = [
            _normalize_numbers(item.model_dump(mode="json"))
            for item in state.evidence
            if not cited or item.evidence_id in cited
        ]
        prompt = f"""
Critically verify whether the proposed hypothesis answers the question and is supported by
the supplied deterministic evidence. Reject unsupported numbers, causal overreach, circular
reasoning, or evidence that shows only a correlation. ACCEPT only when the evidence is
sufficient for a carefully worded evidence-backed hypothesis. Return a confidence from 0-1,
a concise reason, and specific missing evidence when rejecting.

Question: {state.question}
Hypothesis: {attempt.hypothesis}
Findings: {json.dumps(attempt.findings)}
Evidence: {json.dumps(evidence, default=str)}
Remaining uncertainty: {json.dumps(attempt.remaining_uncertainty)}

Dataset-aware evidence quality guide:
{evidence_quality_guide(state.dataset_profiles, state.question)}

Verification rules:
- Treat the question's requested scope and explicit exclusions as acceptance criteria. REJECT
  a conclusion that answers a broader or different question or classifies a condition as an
  error after the user explicitly declared it valid.
- When the question asks which records or identifiers are affected, REJECT an answer that
  reports only a count when the supplied evidence contains their identifiers or row details.
- Use the profiled relationship cardinality. Do not treat raw row-count differences across
  different grains as reconciliation incidents, and do not treat repeated keys on the many
  side of a one-to-many relationship as duplicate errors. Exact full-row duplicates remain a
  separate finding.
- Judge the data-level explanation separately from the underlying technical mechanism.
- REJECT any hypothesis that attributes the discrepancy to an ETL, pipeline, software,
  deletion, ingestion, or operational failure without direct evidence. Hedging with words such
  as "may", "likely", or "possibly" does not make that causal claim supported.
- Do not reject a well-supported data-level explanation merely because application logs are
  required to prove the deeper technical mechanism; record that as remaining uncertainty.
- Key-absence evidence proves only that no matching key appears in the uploaded comparison
  file. It does not prove that a real-world transaction never occurred or that an entry was
  deleted. Require the hypothesis to preserve that dataset-level boundary.
- A claimed affected segment is meaningful only when its source column has more than one
  value and tool evidence shows concentration. A constant status such as all "Captured" is
  not an affected segment.
- For a reconciliation mismatch, check whether the numerical gap, key-level mismatch,
  informative segment, time range, and impact form a consistent explanation. Request the
  specific missing check when they do not.
- Do not demand an unsupported decomposition of a net aggregate difference. When deterministic
  evidence proves missing keys and duplicate keys coexist, and an independently calculated
  whole-dataset sum proves the requested net difference, that evidence is sufficient for a
  cautious data-level explanation of those observed conditions. The hypothesis must not claim
  how much each condition contributed unless that contribution was measured.
- Matching grouped totals from both datasets are valid evidence that other displayed segments
  do not contribute to a discrepancy. Do not require an additional row-level segment check when
  the paired full-dataset group results already isolate exactly one differing segment.
"""
        return self._run_structured(self.verifier, prompt, VerificationResult)

    def report(self, state: InvestigationState, conclusive: bool) -> str:
        draft = BusinessReportDraft(
            executive_summary=state.hypothesis or "The investigation was inconclusive.",
            data_level_cause=state.hypothesis or "Inconclusive",
            underlying_technical_cause="Not established by CSV evidence.",
            confidence="Low",
            main_discrepancy="Derived from deterministic evidence below.",
            affected_records="Derived from deterministic evidence below.",
            affected_segment="Derived from deterministic evidence below.",
            affected_period="Derived from deterministic evidence below.",
            measurable_impact="Derived from deterministic evidence below.",
            recommended_actions=[
                "Review the source and target records identified by the evidence.",
                "Inspect source-system and pipeline logs before assigning a technical cause.",
                "Correct only records confirmed against the system of record.",
                "Add an automated reconciliation check for the affected keys and measures.",
            ],
            remaining_uncertainty=[
                "The uploaded CSV files do not establish the underlying technical mechanism."
            ],
        )
        return _render_report(draft, state, conclusive)

    @staticmethod
    def _run_structured(agent: Any, description: str, model: type[BaseModel]) -> Any:
        from crewai import Crew, Process, Task

        schema = json.dumps(model.model_json_schema(), default=str)
        last_error: Exception | None = None
        for structured_attempt in range(1, 3):
            recovery = ""
            if structured_attempt == 2:
                recovery = f"""

IMPORTANT OUTPUT RECOVERY:
The preceding execution did not finish with the required structured result. Complete the
original task using the supplied evidence. You may use a genuinely necessary tool, but never
finish on a tool observation. Your final response must be one object matching this JSON schema:
{schema}
"""
                print(
                    f"[ReconAI] [TRACE] {agent.role} is retrying malformed "
                    f"{model.__name__} output once.",
                    flush=True,
                )
            task = Task(
                description=description + recovery,
                expected_output=(
                    f"One valid structured {model.__name__} object. Never return a tool "
                    "result as the final answer."
                ),
                agent=agent,
                output_pydantic=model,
            )
            try:
                output = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=False,
                    tracing=False,
                ).kickoff()
                parsed = getattr(output, "pydantic", None)
                if parsed is None and getattr(output, "tasks_output", None):
                    parsed = getattr(output.tasks_output[-1], "pydantic", None)
                if isinstance(parsed, model):
                    return parsed
                raw = getattr(output, "raw", str(output))
                return model.model_validate_json(raw)
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                structured_error = isinstance(
                    exc, (ValidationError, json.JSONDecodeError)
                ) or any(
                    marker in error_text
                    for marker in (
                        "validation error",
                        "valid json",
                        "json decode",
                        "output_pydantic",
                    )
                )
                if structured_attempt == 2 or not structured_error:
                    raise
        raise RuntimeError(f"Unable to obtain {model.__name__}") from last_error
