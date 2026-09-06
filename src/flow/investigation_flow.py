"""Bounded profiler → investigator → verifier → retry/report orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from src.agents.runtime import CrewAIRuntime
from src.baseline import collect_baseline_evidence
from src.config import MAX_INVESTIGATION_ATTEMPTS
from src.models import (
    DatasetProfile,
    InvestigationAttempt,
    InvestigationResult,
    InvestigationState,
    Verdict,
    VerificationResult,
)
from src.state import add_trace, render_trace
from src.tools.common import ToolInputError
from src.tools.profiling import infer_relationships, profile_dataset


class AgentRuntime(Protocol):
    def review_profiles(self, state: InvestigationState): ...

    def investigate(
        self, state: InvestigationState, previous_verification: VerificationResult | None
    ) -> InvestigationAttempt: ...

    def verify(
        self, state: InvestigationState, attempt: InvestigationAttempt
    ) -> VerificationResult: ...

    def report(self, state: InvestigationState, conclusive: bool) -> str: ...


_UNSUPPORTED_TECHNICAL_CAUSES = (
    "pipeline",
    "etl",
    "software bug",
    "ingestion failure",
    "loading failure",
    "deployment",
    "system outage",
)

_DIAGNOSTIC_TOOLS = {
    "compare_aggregates",
    "find_unmatched_records",
    "compare_record_values",
    "compare_rows_by_key",
    "find_duplicates",
    "segment_analysis",
    "calculate_business_impact",
}


def _segment_localization_requested(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in ("which", "where", "segment", "category", "priority", "channel", "region")
    ) and any(
        term in lowered
        for term in ("amount", "total", "value", "revenue", "balance", "difference", "mismatch")
    )


def _paired_segment_localization(
    state: InvestigationState, attempt: InvestigationAttempt
) -> tuple[str, list[str], list[str]] | None:
    """Return cited segments whose paired full-dataset totals differ."""
    profile_rows = {profile.dataset: profile.row_count for profile in state.dataset_profiles}
    segment_evidence = [
        item
        for item in state.evidence
        if item.tool == "segment_analysis"
        and item.attempt > 0
        and len(item.supporting_details.get("grouping", [])) == 1
        and not item.supporting_details.get("results_truncated")
        and item.supporting_details.get("filtered_record_count")
        == profile_rows.get(item.supporting_details.get("dataset"))
    ]
    cited = set(attempt.evidence_ids)
    for index, left in enumerate(segment_evidence):
        left_details = left.supporting_details
        for right in segment_evidence[index + 1 :]:
            right_details = right.supporting_details
            if (
                left_details.get("dataset") == right_details.get("dataset")
                or left_details.get("grouping") != right_details.get("grouping")
                or left_details.get("metric_column") != right_details.get("metric_column")
                or left_details.get("aggregation") != right_details.get("aggregation")
                or left.evidence_id not in cited
                or right.evidence_id not in cited
            ):
                continue
            dimension = left_details["grouping"][0]
            left_values = {
                str(row.get(dimension)): float(row["value"])
                for row in left_details.get("results", [])
            }
            right_values = {
                str(row.get(dimension)): float(row["value"])
                for row in right_details.get("results", [])
            }
            differing = sorted(
                segment
                for segment in left_values.keys() | right_values.keys()
                if round(
                    left_values.get(segment, 0.0) - right_values.get(segment, 0.0),
                    6,
                )
                != 0
            )
            if differing:
                return dimension, differing, [left.evidence_id, right.evidence_id]
    return None


def _complete_localized_value_reconciliation(
    state: InvestigationState,
    attempt: InvestigationAttempt,
    localization: tuple[str, list[str], list[str]],
) -> bool:
    """Check that paired segment totals fully localize a matched-key value drift."""
    _, differing_segments, _ = localization
    hypothesis = attempt.hypothesis.lower()
    if not all(segment.lower() in hypothesis for segment in differing_segments):
        return False
    value_mismatches = any(
        item.tool == "compare_record_values"
        and item.supporting_details.get("value_mismatch_count", 0) > 0
        for item in state.evidence
    )
    matching_keys = all(
        item.supporting_details.get("unmatched_count") == 0
        for item in state.evidence
        if item.tool == "find_unmatched_records"
    )
    no_exact_duplicates = all(
        item.supporting_details.get("duplicate_row_count") == 0
        for item in state.evidence
        if item.tool == "find_duplicates"
        and item.supporting_details.get("analysis_type") == "exact_row"
    )
    return value_mismatches and matching_keys and no_exact_duplicates


def _complete_offsetting_reconciliation(
    state: InvestigationState, attempt: InvestigationAttempt
) -> bool:
    """Recognize a fully evidenced missing-plus-duplicate reconciliation."""
    evidence = state.evidence
    count_checks = [
        item
        for item in evidence
        if item.tool == "compare_aggregates"
        and item.supporting_details.get("aggregation") == "count"
        and item.supporting_details.get("absolute_difference") == 0
    ]
    missing = [
        item
        for item in evidence
        if item.tool == "find_unmatched_records"
        and item.supporting_details.get("unmatched_count", 0) > 0
    ]
    matched_directions = [
        item
        for item in evidence
        if item.tool == "find_unmatched_records"
        and item.supporting_details.get("unmatched_count") == 0
    ]
    duplicates = [
        item
        for item in evidence
        if item.tool == "find_duplicates"
        and item.supporting_details.get("analysis_type") == "exact_row"
        and item.supporting_details.get("duplicate_row_count", 0) > 0
    ]
    matching_rows = [
        item
        for item in evidence
        if item.tool == "compare_rows_by_key"
        and item.supporting_details.get("record_mismatch_count") == 0
    ]
    sum_checks = [
        item
        for item in evidence
        if item.tool == "compare_aggregates"
        and item.attempt > 0
        and item.supporting_details.get("aggregation") == "sum"
        and isinstance(
            item.supporting_details.get("signed_difference_a_minus_b"),
            (int, float),
        )
    ]
    if not all(
        (count_checks, missing, matched_directions, duplicates, matching_rows, sum_checks)
    ):
        return False

    cited = set(attempt.evidence_ids)
    required_ids = {
        missing[0].evidence_id,
        duplicates[0].evidence_id,
        matching_rows[0].evidence_id,
        sum_checks[-1].evidence_id,
    }
    if not required_ids <= cited:
        return False

    hypothesis = attempt.hypothesis.lower()
    if not any(term in hypothesis for term in ("missing", "absent", "unmatched")):
        return False
    if not any(term in hypothesis for term in ("duplicate", "duplicated")):
        return False
    difference = float(
        sum_checks[-1].supporting_details["signed_difference_a_minus_b"]
    )
    numeric_forms = {
        f"{difference:.2f}",
        f"{difference:,.2f}",
        f"{difference:g}",
    }
    return any(value in attempt.hypothesis for value in numeric_forms)


def apply_verification_guard(
    state: InvestigationState,
    attempt: InvestigationAttempt,
    verification: VerificationResult,
) -> VerificationResult:
    """Apply non-LLM acceptance rules after the verifier responds."""
    lowered = attempt.hypothesis.lower()
    unsupported = [
        term for term in _UNSUPPORTED_TECHNICAL_CAUSES if term in lowered
    ]
    if unsupported:
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=min(verification.confidence, 0.4),
            reason=(
                "Controller rejected an unsupported technical mechanism in the "
                f"hypothesis: {', '.join(unsupported)}."
            ),
            missing_evidence=[
                "State only the data-level discrepancy; keep possible technical mechanisms "
                "in remaining uncertainty unless direct logs prove them."
            ],
        )
    localization = (
        _paired_segment_localization(state, attempt)
        if _segment_localization_requested(state.question)
        else None
    )
    if _segment_localization_requested(state.question):
        if localization is None and verification.verdict == Verdict.ACCEPT:
            return VerificationResult(
                verdict=Verdict.REJECT,
                confidence=min(verification.confidence, 0.3),
                reason=(
                    "Controller rejected a segment conclusion without cited paired "
                    "full-dataset segment evidence."
                ),
                missing_evidence=[
                    "Compare the requested metric by the same segment in both datasets."
                ],
            )
        if localization is not None:
            dimension, differing_segments, _ = localization
            missing_segments = [
                segment
                for segment in differing_segments
                if segment.lower() not in lowered
            ]
            if missing_segments and verification.verdict == Verdict.ACCEPT:
                return VerificationResult(
                    verdict=Verdict.REJECT,
                    confidence=min(verification.confidence, 0.3),
                    reason=(
                        f"Controller rejected the `{dimension}` conclusion because it "
                        "does not match the paired segment totals."
                    ),
                    missing_evidence=[
                        f"Use the supported differing segment(s): {', '.join(differing_segments)}."
                    ],
                )
            if (
                verification.verdict == Verdict.REJECT
                and _complete_localized_value_reconciliation(
                    state, attempt, localization
                )
            ):
                return VerificationResult(
                    verdict=Verdict.ACCEPT,
                    confidence=max(verification.confidence, 0.9),
                    reason=(
                        f"Controller confirmed that paired `{dimension}` totals isolate "
                        f"the discrepancy to {', '.join(differing_segments)}."
                    ),
                )
    if (
        verification.verdict == Verdict.REJECT
        and _complete_offsetting_reconciliation(state, attempt)
    ):
        return VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=max(verification.confidence, 0.9),
            reason=(
                "Controller confirmed a complete deterministic evidence chain: "
                "balanced row counts, one-directional missing keys, exact duplicates, "
                "matching shared rows, and an independently measured net sum."
            ),
        )
    question = state.question.lower()
    duplicate_constraint = "do not" in question and any(
        term in question for term in ("duplicate", "repeated", "repeat")
    )
    duplicate_claim = any(
        term in lowered
        for term in (
            "duplicate records caused",
            "duplicate rows caused",
            "caused by duplicate",
            "caused by repeated",
            "repeated keys explain",
            "duplicates explain",
        )
    )
    if duplicate_constraint and duplicate_claim:
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=min(verification.confidence, 0.3),
            reason=(
                "Controller rejected a duplicate-error claim that conflicts with an "
                "explicit instruction in the investigation question."
            ),
            missing_evidence=[
                "Respect the stated relationship grain and answer only the requested "
                "unique-key coverage question."
            ],
        )
    cited = set(attempt.evidence_ids)
    diagnostic_citations = [
        item
        for item in state.evidence
        if item.evidence_id in cited and item.tool in _DIAGNOSTIC_TOOLS
    ]
    if verification.verdict == Verdict.ACCEPT and not diagnostic_citations:
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=min(verification.confidence, 0.3),
            reason="Controller rejected acceptance without cited diagnostic evidence.",
            missing_evidence=[
                "Cite at least one deterministic reconciliation evidence ID."
            ],
        )
    fresh_diagnostic_evidence = [
        item
        for item in state.evidence
        if item.attempt > 0 and item.tool in _DIAGNOSTIC_TOOLS
    ]
    if verification.verdict == Verdict.ACCEPT and not fresh_diagnostic_evidence:
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=min(verification.confidence, 0.3),
            reason=(
                "Controller rejected acceptance because the agentic investigation "
                "collected no fresh diagnostic evidence."
            ),
            missing_evidence=[
                "Use a deterministic tool to collect the unresolved evidence requested "
                "by the question."
            ],
        )
    return verification


def deterministic_clean_match(state: InvestigationState) -> bool:
    """Return true only when baseline evidence proves complete equality for a pair."""
    if len(state.dataset_profiles) != 2:
        return False
    left, right = state.dataset_profiles
    if left.row_count != right.row_count or set(left.columns) != set(right.columns):
        return False
    row_checks = [
        item for item in state.evidence if item.tool == "compare_rows_by_key"
    ]
    if len(row_checks) != 1:
        return False
    row_details = row_checks[0].supporting_details
    if row_details.get("record_mismatch_count") != 0:
        return False
    if row_details.get("matched_unique_key_count") != left.row_count:
        return False
    expected_columns = set(left.columns) - {
        row_details.get("key_column_a"),
        row_details.get("key_column_b"),
    }
    if set(row_details.get("columns_compared", [])) != expected_columns:
        return False
    required_zero_fields = {
        "compare_aggregates": "absolute_difference",
        "find_unmatched_records": "unmatched_count",
        "find_duplicates": "duplicate_row_count",
    }
    for tool, field in required_zero_fields.items():
        matching = [item for item in state.evidence if item.tool == tool]
        required_count = 1 if tool == "compare_aggregates" else 2
        if len(matching) < required_count:
            return False
        if any(item.supporting_details.get(field) != 0 for item in matching):
            return False
    unmatched = [
        item for item in state.evidence if item.tool == "find_unmatched_records"
    ]
    if any(item.supporting_details.get("null_key_count_a") != 0 for item in unmatched):
        return False
    return True


def _key_coverage_hypothesis(key_checks: list) -> str:
    clauses: list[str] = []
    record_details: list[str] = []
    for item in key_checks:
        details = item.supporting_details
        count = int(
            details.get(
                "unmatched_unique_key_count", details.get("unmatched_count", 0)
            )
        )
        value_word = "value" if count == 1 else "values"
        verb = "is" if count == 1 else "are"
        clauses.append(
            f"{count} {details.get('key_column_a', 'key')} {value_word} in "
            f"{details.get('dataset_a')} {verb} absent from {details.get('dataset_b')}"
        )
        for sample in details.get("sample_unmatched_records", []):
            rendered = ", ".join(
                f"{column}={value}" for column, value in sample.items()
            )
            record_details.append(
                f"{details.get('dataset_a')} ({rendered}; {item.evidence_id})"
            )
    detail_sentence = ""
    if record_details:
        label = "Affected record" if len(record_details) == 1 else "Affected records"
        detail_sentence = f" {label}: " + "; ".join(record_details) + "."
    return (
        "Within the uploaded files, " + "; ".join(clauses) + "."
        + detail_sentence
        + " This proves the dataset-level key gap, but not why it occurred."
    )


def deterministic_baseline_conclusion(
    state: InvestigationState,
) -> tuple[str, str] | None:
    """Return a proven baseline conclusion for common single-cause reconciliations."""
    if deterministic_clean_match(state):
        return (
            "No discrepancy detected within the validated scope.",
            "Equal schemas, row counts, key coverage, duplicate checks, and shared row values.",
        )
    if len(state.dataset_profiles) != 2:
        return None
    lowered_question = state.question.lower()
    coverage_requested = any(
        term in lowered_question
        for term in ("missing", "unmatched", "absent", "coverage", "correspond")
    )
    duplicate_forbidden = "do not" in lowered_question and any(
        term in lowered_question for term in ("duplicate", "repeated", "repeat")
    )
    value_analysis_requested = any(
        term in lowered_question
        for term in (
            "amount",
            "total",
            "revenue",
            "balance",
            "sum",
        )
    ) or (
        "value" in lowered_question
        and any(term in lowered_question for term in ("differ", "mismatch"))
    )
    other_analysis_requested = value_analysis_requested or any(
        term in lowered_question
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
    ) or (
        not duplicate_forbidden
        and any(
            term in lowered_question for term in ("duplicate", "repeated", "repeat")
        )
    )
    key_checks = [
        item for item in state.evidence if item.tool == "find_unmatched_records"
    ]
    if coverage_requested and not other_analysis_requested and len(key_checks) == 2:
        return (
            _key_coverage_hypothesis(key_checks),
            "Bidirectional deterministic key checks fully answer the requested coverage scope.",
        )
    count_checks = [
        item for item in state.evidence if item.tool == "compare_aggregates"
    ]
    row_checks = [
        item for item in state.evidence if item.tool == "compare_rows_by_key"
    ]
    unmatched = [
        item
        for item in state.evidence
        if item.tool == "find_unmatched_records"
        and item.supporting_details.get("unmatched_count", 0) > 0
    ]
    duplicates = [
        item
        for item in state.evidence
        if item.tool == "find_duplicates"
        and item.supporting_details.get("duplicate_row_count", 0) > 0
        and item.supporting_details.get("analysis_type") != "repeated_key"
    ]
    broad_investigation = any(
        term in lowered_question
        for term in ("investigate", "disagree", "reconcile", "affected record")
    )
    has_key_gap = any(
        item.supporting_details.get(
            "unmatched_unique_key_count",
            item.supporting_details.get("unmatched_count", 0),
        )
        > 0
        for item in key_checks
    )
    no_null_key_ambiguity = all(
        item.supporting_details.get("null_key_count_a", 0) == 0
        for item in key_checks
    )
    if (
        broad_investigation
        and not other_analysis_requested
        and len(key_checks) == 2
        and has_key_gap
        and no_null_key_ambiguity
        and not duplicates
    ):
        return (
            _key_coverage_hypothesis(key_checks),
            "Deterministic bidirectional key checks and affected-record details establish "
            "the observed discrepancy; raw row counts are interpreted using relationship grain.",
        )
    if len(count_checks) != 1 or len(row_checks) != 1:
        return None
    count_details = count_checks[0].supporting_details
    row_details = row_checks[0].supporting_details
    count_gap = float(count_details.get("absolute_difference", 0))
    row_mismatches = int(row_details.get("record_mismatch_count", 0))

    if len(unmatched) == 1 and not duplicates and row_mismatches == 0:
        details = unmatched[0].supporting_details
        missing_count = int(details["unmatched_count"])
        if count_gap == missing_count:
            hypothesis = (
                f"{missing_count} records in {details['dataset_a']} have {details['key_column_a']} "
                f"values absent from {details['dataset_b']}, fully explaining the row-count gap."
            )
            return hypothesis, "A one-directional key gap exactly equals the row-count difference."

    if len(duplicates) == 1 and not unmatched and row_mismatches == 0 and count_gap > 0:
        details = duplicates[0].supporting_details
        duplicate_records = int(details["duplicate_row_count"])
        duplicate_dataset = details["dataset"]
        profile_rows = {
            profile.dataset: profile.row_count for profile in state.dataset_profiles
        }
        other_dataset = next(name for name in profile_rows if name != duplicate_dataset)
        if profile_rows[duplicate_dataset] - profile_rows[other_dataset] == count_gap:
            hypothesis = (
                f"{duplicate_dataset} is {int(count_gap)} rows larger and contains "
                f"{duplicate_records} exact duplicate rows; no unmatched "
                "keys or differences among the remaining unique-key rows were found."
            )
            return hypothesis, "Exact full-row duplicates fully explain the row-count gap."

    localization_terms = (
        "where",
        "which",
        "concentrat",
        "segment",
        "priority",
        "region",
        "channel",
        "when",
        "month",
        "period",
    )
    localization_requested = any(
        term in state.question.lower() for term in localization_terms
    )
    localization_evidence = any(
        item.tool in {"segment_analysis", "calculate_business_impact"}
        for item in state.evidence
    )
    if (
        not unmatched
        and not duplicates
        and count_gap == 0
        and row_mismatches > 0
        and (not localization_requested or localization_evidence)
    ):
        differing = [
            column
            for column, count in row_details.get("mismatches_by_column", {}).items()
            if count > 0
        ]
        if differing:
            hypothesis = (
                f"{row_mismatches} matched keys contain value mismatches in: "
                f"{', '.join(differing)}. Row counts and key coverage otherwise match."
            )
            return hypothesis, "Complete shared-row comparison isolates a value-level discrepancy."
    return None


class ReconAIInvestigationFlow:
    """A deterministic control loop containing bounded CrewAI agent tasks."""

    def __init__(
        self,
        question: str,
        dataset_paths: list[str],
        runtime: AgentRuntime | None = None,
    ):
        clean_question = question.strip()
        if not clean_question:
            raise ToolInputError("Enter an investigation question.")
        if not dataset_paths:
            raise ToolInputError("Upload at least one CSV dataset.")
        resolved = [str(Path(path).resolve()) for path in dataset_paths]
        self.state = InvestigationState(
            investigation_id=str(uuid4()),
            question=clean_question,
            dataset_paths=resolved,
        )
        self.runtime = runtime

    def run(self) -> InvestigationResult:
        self._profile()
        collect_baseline_evidence(self.state)
        runtime = self.runtime or CrewAIRuntime(self.state)
        add_trace(self.state, "agent", "Deterministic Profiler started profile review")
        runtime.review_profiles(self.state)
        add_trace(self.state, "agent", "Deterministic Profiler reviewed dataset profiles")

        baseline_conclusion = deterministic_baseline_conclusion(self.state)
        if baseline_conclusion:
            self.state.hypothesis, baseline_reason = baseline_conclusion
            self.state.verification = VerificationResult(
                verdict=Verdict.ACCEPT,
                confidence=1.0,
                reason=baseline_reason,
            )
            add_trace(
                self.state,
                "verification",
                "Controller accepted a fully explained deterministic baseline without an LLM call",
            )
            report = runtime.report(self.state, conclusive=True)
            self.state.final_report = report
            add_trace(self.state, "complete", "Investigation finished")
            return InvestigationResult(
                state=self.state, report=report, trace=render_trace(self.state)
            )

        previous_verification: VerificationResult | None = None
        accepted = False
        for attempt_number in range(1, MAX_INVESTIGATION_ATTEMPTS + 1):
            self.state.attempt_count = attempt_number
            add_trace(
                self.state,
                "agent",
                f"Data Investigator Agent started attempt {attempt_number}",
            )
            evidence_before = len(self.state.evidence)
            attempt = runtime.investigate(self.state, previous_verification)
            valid_evidence_ids = {item.evidence_id for item in self.state.evidence}
            invalid_ids = [
                evidence_id
                for evidence_id in attempt.evidence_ids
                if evidence_id not in valid_evidence_ids
            ]
            attempt.evidence_ids = [
                evidence_id
                for evidence_id in attempt.evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            if invalid_ids:
                add_trace(
                    self.state,
                    "quality",
                    "Ignored unsupported evidence reference(s): "
                    + ", ".join(invalid_ids),
                )
            add_trace(
                self.state,
                "evidence",
                f"Attempt {attempt_number} collected "
                f"{len(self.state.evidence) - evidence_before} new evidence item(s)",
            )
            self.state.hypothesis = attempt.hypothesis
            add_trace(
                self.state,
                "decision",
                f"Investigator proposed hypothesis on attempt {attempt_number}",
            )
            add_trace(
                self.state,
                "agent",
                f"Evidence Verifier Agent started review for attempt {attempt_number}",
            )
            verification = runtime.verify(self.state, attempt)
            guarded_verification = apply_verification_guard(
                self.state, attempt, verification
            )
            if guarded_verification != verification:
                add_trace(
                    self.state,
                    "quality",
                    f"Controller overruled verifier: {guarded_verification.reason}",
                )
            verification = guarded_verification
            self.state.verification = verification
            previous_verification = verification
            add_trace(
                self.state,
                "verification",
                f"Evidence Verifier returned {verification.verdict.value} "
                f"with {verification.confidence:.0%} confidence: {verification.reason}",
            )
            if verification.verdict == Verdict.ACCEPT:
                accepted = True
                break
            if attempt_number < MAX_INVESTIGATION_ATTEMPTS:
                detail = "; ".join(verification.missing_evidence) or verification.reason
                add_trace(
                    self.state,
                    "decision",
                    f"Retry requested with different evidence: {detail}",
                )

        add_trace(
            self.state,
            "agent",
            "Evidence Report Renderer started structured report generation",
        )
        report = runtime.report(self.state, conclusive=accepted)
        if not accepted:
            required = "Root Cause: Inconclusive\nConfidence: Low\n\n"
            if "Root Cause: Inconclusive" not in report or "Confidence: Low" not in report:
                report = required + report
        self.state.final_report = report
        add_trace(self.state, "agent", "Evidence Report Renderer generated the final report")
        add_trace(self.state, "complete", "Investigation finished")
        return InvestigationResult(
            state=self.state, report=report, trace=render_trace(self.state)
        )

    def _profile(self) -> None:
        seen_names: set[str] = set()
        for path in self.state.dataset_paths:
            name = Path(path).name
            if name in seen_names:
                raise ToolInputError(f"Uploaded filenames must be unique: {name}")
            seen_names.add(name)
            self.state.dataset_profiles.append(
                DatasetProfile.model_validate(profile_dataset(path))
            )
            add_trace(self.state, "tool", f"Profiled {name}")
        infer_relationships(self.state.dataset_profiles)
