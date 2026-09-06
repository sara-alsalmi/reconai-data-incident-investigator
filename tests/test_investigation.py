from __future__ import annotations

import pandas as pd

from src.agents.runtime import CrewAIRuntime
from src.flow import ReconAIInvestigationFlow
from src.flow.investigation_flow import apply_verification_guard
from src.models import (
    DatasetProfile,
    EvidenceItem,
    InvestigationAttempt,
    InvestigationState,
    Verdict,
    VerificationResult,
)


def test_controller_rejects_agentic_acceptance_without_fresh_evidence():
    investigation_state = InvestigationState(
        investigation_id="fresh-evidence-guard",
        question="Explain the net amount difference.",
        dataset_paths=["source.csv", "target.csv"],
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=0,
                finding="Baseline count comparison",
                tool="compare_aggregates",
                supporting_details={"aggregation": "count"},
            )
        ],
    )
    attempt = InvestigationAttempt(
        findings=["The amount differs."],
        hypothesis="The uploaded datasets have a net amount difference.",
        evidence_ids=["E001"],
    )
    accepted = VerificationResult(
        verdict=Verdict.ACCEPT,
        confidence=0.9,
        reason="The conclusion is supported.",
    )

    guarded = apply_verification_guard(investigation_state, attempt, accepted)

    assert guarded.verdict == Verdict.REJECT
    assert "no fresh diagnostic evidence" in guarded.reason


def test_controller_accepts_complete_offsetting_evidence_chain():
    evidence_specs = [
        ("E001", 0, "compare_aggregates", {"aggregation": "count", "absolute_difference": 0.0}),
        ("E002", 0, "find_unmatched_records", {"unmatched_count": 40}),
        ("E003", 0, "find_unmatched_records", {"unmatched_count": 0}),
        (
            "E004",
            0,
            "find_duplicates",
            {"analysis_type": "exact_row", "duplicate_row_count": 80},
        ),
        ("E005", 0, "compare_rows_by_key", {"record_mismatch_count": 0}),
        (
            "E006",
            1,
            "compare_aggregates",
            {"aggregation": "sum", "signed_difference_a_minus_b": 726530.49},
        ),
    ]
    state = InvestigationState(
        investigation_id="complete-offsetting",
        question="Identify every cause and calculate the net amount difference.",
        dataset_paths=["source.csv", "target.csv"],
        evidence=[
            EvidenceItem(
                evidence_id=evidence_id,
                attempt=attempt_number,
                finding="deterministic evidence",
                tool=tool,
                supporting_details=details,
            )
            for evidence_id, attempt_number, tool, details in evidence_specs
        ],
    )
    attempt = InvestigationAttempt(
        findings=["Missing records and exact duplicates coexist."],
        hypothesis=(
            "Missing source records and exact duplicate target rows coexist; the "
            "measured source-minus-target amount difference is 726,530.49."
        ),
        evidence_ids=["E002", "E004", "E005", "E006"],
    )
    over_demanding_rejection = VerificationResult(
        verdict=Verdict.REJECT,
        confidence=0.4,
        reason="A decomposition of the independently measured net sum is required.",
    )

    guarded = apply_verification_guard(state, attempt, over_demanding_rejection)

    assert guarded.verdict == Verdict.ACCEPT
    assert guarded.confidence == 0.9
    assert "complete deterministic evidence chain" in guarded.reason

    attempt.hypothesis = "Missing records and duplicates produce a difference of 1.00."
    still_rejected = apply_verification_guard(
        state, attempt, over_demanding_rejection
    )
    assert still_rejected.verdict == Verdict.REJECT


def test_controller_requires_and_validates_paired_segment_evidence():
    profiles = [
        DatasetProfile(
            dataset=name,
            path=name,
            row_count=2,
            columns=["record_id", "priority", "amount"],
            data_types={"record_id": "int64", "priority": "object", "amount": "float64"},
            null_counts={"record_id": 0, "priority": 0, "amount": 0},
            duplicate_count=0,
            unique_counts={"record_id": 2, "priority": 2, "amount": 2},
            possible_identifier_columns=["record_id"],
            numeric_columns=["record_id", "amount"],
        )
        for name in ("source.csv", "target.csv")
    ]
    base_evidence = [
        EvidenceItem(
            evidence_id="E001",
            attempt=0,
            finding="45 value mismatches",
            tool="compare_record_values",
            supporting_details={"value_mismatch_count": 45},
        ),
        *[
            EvidenceItem(
                evidence_id=f"E00{index}",
                attempt=0,
                finding="No unmatched rows",
                tool="find_unmatched_records",
                supporting_details={"unmatched_count": 0},
            )
            for index in (2, 3)
        ],
        *[
            EvidenceItem(
                evidence_id=f"E00{index}",
                attempt=0,
                finding="No exact duplicates",
                tool="find_duplicates",
                supporting_details={
                    "analysis_type": "exact_row",
                    "duplicate_row_count": 0,
                },
            )
            for index in (4, 5)
        ],
    ]
    segment_results = (
        [{"priority": "2-HIGH", "value": 100.0}, {"priority": "1-URGENT", "value": 50.0}],
        [{"priority": "2-HIGH", "value": 123.5}, {"priority": "1-URGENT", "value": 50.0}],
    )
    segment_evidence = [
        EvidenceItem(
            evidence_id=f"E00{index}",
            attempt=1,
            finding="Paired priority totals",
            tool="segment_analysis",
            supporting_details={
                "dataset": dataset,
                "grouping": ["priority"],
                "metric_column": "amount",
                "aggregation": "sum",
                "filtered_record_count": 2,
                "results": results,
                "results_truncated": False,
            },
        )
        for index, dataset, results in zip(
            (6, 7), ("source.csv", "target.csv"), segment_results
        )
    ]
    state = InvestigationState(
        investigation_id="segment-guard",
        question="Which priority contains the amount difference?",
        dataset_paths=["source.csv", "target.csv"],
        dataset_profiles=profiles,
        evidence=[*base_evidence, *segment_evidence],
    )
    accepted = VerificationResult(
        verdict=Verdict.ACCEPT,
        confidence=0.95,
        reason="Accepted by the verifier.",
    )
    wrong_attempt = InvestigationAttempt(
        hypothesis="The amount difference is in 1-URGENT.",
        evidence_ids=["E001", "E006", "E007"],
    )

    guarded_wrong = apply_verification_guard(state, wrong_attempt, accepted)

    assert guarded_wrong.verdict == Verdict.REJECT
    assert "does not match the paired segment totals" in guarded_wrong.reason

    correct_attempt = InvestigationAttempt(
        hypothesis="The amount difference is isolated to 2-HIGH.",
        evidence_ids=["E001", "E006", "E007"],
    )
    over_demanding_rejection = VerificationResult(
        verdict=Verdict.REJECT,
        confidence=0.4,
        reason="More localization is required.",
    )
    guarded_correct = apply_verification_guard(
        state, correct_attempt, over_demanding_rejection
    )

    assert guarded_correct.verdict == Verdict.ACCEPT
    assert "paired `priority` totals isolate" in guarded_correct.reason


class AcceptingRuntime:
    def review_profiles(self, state):
        return None

    def investigate(self, state, previous_verification):
        evidence_id = f"E{len(state.evidence) + 1:03d}"
        state.evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                attempt=state.attempt_count,
                finding="Deterministic reconciliation found unmatched payment records",
                tool="find_unmatched_records",
                datasets=["payments.csv", "orders.csv"],
                metric="unmatched_count",
                value=2,
                supporting_details={"unmatched_count": 2},
            )
        )
        return InvestigationAttempt(
            findings=["Unmatched records exist"],
            hypothesis="The discrepancy is associated with payments lacking order rows.",
            evidence_ids=[evidence_id],
        )

    def verify(self, state, attempt):
        if state.attempt_count == 1:
            return VerificationResult(
                verdict=Verdict.REJECT,
                confidence=0.4,
                reason="A segment check is still needed.",
                missing_evidence=["Group unmatched records by segment."],
            )
        return VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=0.86,
            reason="Two attempts supplied consistent evidence.",
        )

    def report(self, state, conclusive):
        assert conclusive is True
        return "# Investigation Report\n\nRoot Cause: Evidence-backed hypothesis"


class RejectingRuntime(AcceptingRuntime):
    def verify(self, state, attempt):
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=0.2,
            reason="Evidence remains insufficient.",
            missing_evidence=["Provide upstream event logs."],
        )

    def report(self, state, conclusive):
        assert conclusive is False
        return "# Investigation Report\n\nAdditional data is required."


def test_end_to_end_state_transitions_and_report(datasets):
    orders, payments = datasets
    flow = ReconAIInvestigationFlow(
        "Why do payment and order totals differ?",
        [orders, payments],
        runtime=AcceptingRuntime(),
    )
    result = flow.run()
    assert result.state.attempt_count == 2
    assert len(result.state.dataset_profiles) == 2
    assert len(result.state.evidence) == 10
    assert result.state.verification.verdict == Verdict.ACCEPT
    assert result.state.final_report
    assert "Verifier returned ACCEPT" in result.trace


def test_attempt_limit_produces_inconclusive(datasets):
    orders, payments = datasets
    result = ReconAIInvestigationFlow(
        "What caused the discrepancy?",
        [orders, payments],
        runtime=RejectingRuntime(),
    ).run()
    assert result.state.attempt_count == 3
    assert result.state.verification.verdict == Verdict.REJECT
    assert "Root Cause: Inconclusive" in result.report
    assert "Confidence: Low" in result.report


def test_controller_rejects_unsupported_technical_cause(datasets):
    class UnsafeRuntime(AcceptingRuntime):
        def investigate(self, state, previous_verification):
            attempt = super().investigate(state, previous_verification)
            attempt.hypothesis = "An ETL pipeline failure caused the missing records."
            return attempt

        def verify(self, state, attempt):
            return VerificationResult(
                verdict=Verdict.ACCEPT,
                confidence=0.95,
                reason="Accepted by the model verifier.",
            )

        def report(self, state, conclusive):
            return "Additional evidence is required."

    orders, payments = datasets
    result = ReconAIInvestigationFlow(
        "Why do the files disagree?", [orders, payments], runtime=UnsafeRuntime()
    ).run()
    assert result.state.attempt_count == 3
    assert result.state.verification.verdict == Verdict.REJECT
    assert "Root Cause: Inconclusive" in result.report
    assert "Controller overruled verifier" in result.trace


def test_complete_clean_match_skips_llm_roles(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    frame = pd.DataFrame(
        {
            "record_id": [1, 2],
            "status": ["Open", "Closed"],
            "amount": [10.0, 20.0],
        }
    )
    frame.to_csv(source, index=False)
    frame.to_csv(target, index=False)

    class CleanRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("Investigator must not run for a proven clean match")

        def verify(self, state, attempt):
            raise AssertionError("Verifier must not run for a proven clean match")

        def report(self, state, conclusive):
            assert conclusive is True
            return "Root Cause: No discrepancy detected within the validated scope."

    result = ReconAIInvestigationFlow(
        "Do these files disagree?",
        [str(source), str(target)],
        runtime=CleanRuntime(),
    ).run()
    assert result.state.attempt_count == 0
    assert result.state.verification.verdict == Verdict.ACCEPT
    assert "without an LLM call" in result.trace


def test_proven_missing_records_skip_llm_roles(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1], "amount": [10.0]}
    ).to_csv(target, index=False)

    class NoLLMRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("Investigator must not run for proven missing records")

        def verify(self, state, attempt):
            raise AssertionError("Verifier must not run for proven missing records")

        def report(self, state, conclusive):
            assert conclusive is True
            return state.hypothesis

    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=NoLLMRuntime(),
    ).run()
    assert result.state.attempt_count == 0
    assert "absent" in result.state.hypothesis


def test_proven_duplicate_records_skip_llm_roles(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1, 2, 2], "amount": [10.0, 20.0, 20.0]}
    ).to_csv(target, index=False)

    class NoLLMRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("Investigator must not run for proven duplicates")

        def verify(self, state, attempt):
            raise AssertionError("Verifier must not run for proven duplicates")

        def report(self, state, conclusive):
            assert conclusive is True
            return state.hypothesis

    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=NoLLMRuntime(),
    ).run()
    assert result.state.attempt_count == 0
    assert "duplicate" in result.state.hypothesis


def test_proven_value_drift_skips_llm_roles(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 25.0]}
    ).to_csv(target, index=False)

    class NoLLMRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("Investigator must not run for proven value drift")

        def verify(self, state, attempt):
            raise AssertionError("Verifier must not run for proven value drift")

        def report(self, state, conclusive):
            assert conclusive is True
            return state.hypothesis

    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=NoLLMRuntime(),
    ).run()
    assert result.state.attempt_count == 0
    assert "amount" in result.state.hypothesis


def test_localization_question_uses_agent_after_value_drift(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {
            "record_id": [1, 2],
            "order_priority": ["1-URGENT", "2-HIGH"],
            "amount": [10.0, 20.0],
        }
    ).to_csv(source, index=False)
    pd.DataFrame(
        {
            "record_id": [1, 2],
            "order_priority": ["1-URGENT", "2-HIGH"],
            "amount": [10.0, 25.0],
        }
    ).to_csv(target, index=False)

    class LocalizationRuntime:
        def __init__(self):
            self.investigate_called = False

        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            self.investigate_called = True
            evidence_ids = []
            for dataset, high_value in (("source.csv", 20.0), ("target.csv", 25.0)):
                evidence_id = f"E{len(state.evidence) + 1:03d}"
                evidence_ids.append(evidence_id)
                state.evidence.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        attempt=state.attempt_count,
                        finding=f"Amount totals by priority in {dataset}",
                        tool="segment_analysis",
                        datasets=[dataset],
                        value=2,
                        supporting_details={
                            "dataset": dataset,
                            "grouping": ["order_priority"],
                            "metric_column": "amount",
                            "aggregation": "sum",
                            "filtered_record_count": 2,
                            "results": [
                                {"order_priority": "1-URGENT", "value": 10.0},
                                {"order_priority": "2-HIGH", "value": high_value},
                            ],
                            "results_truncated": False,
                        },
                    )
                )
            return InvestigationAttempt(
                findings=["2-HIGH contains the amount drift"],
                hypothesis="The amount mismatch is concentrated in 2-HIGH.",
                evidence_ids=evidence_ids,
            )

        def verify(self, state, attempt):
            return VerificationResult(
                verdict=Verdict.ACCEPT,
                confidence=0.9,
                reason="The requested localization is supported.",
            )

        def report(self, state, conclusive):
            return state.hypothesis

    runtime = LocalizationRuntime()
    result = ReconAIInvestigationFlow(
        "Which order_priority contains the amount discrepancy?",
        [str(source), str(target)],
        runtime=runtime,
    ).run()
    assert runtime.investigate_called is True
    assert result.state.attempt_count == 1
    assert "2-HIGH" in result.state.hypothesis


def test_one_to_many_key_coverage_sticks_to_requested_scope(tmp_path):
    entities = tmp_path / "entities.csv"
    events = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "record_id": ["A", "B", "C"],
            "status": ["Complete", "Complete", "Complete"],
            "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    ).to_csv(entities, index=False)
    pd.DataFrame(
        {
            "record_id": ["A", "A", "B"],
            "event_sequence": [1, 2, 1],
            "event_value": [10.0, 5.0, 20.0],
        }
    ).to_csv(events, index=False)

    class CoverageRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("A complete key-coverage question should not need an LLM")

        def verify(self, state, attempt):
            raise AssertionError("A complete key-coverage question should not need an LLM")

        def report(self, state, conclusive):
            assert conclusive is True
            return CrewAIRuntime.report(self, state, conclusive)

    question = (
        "Compare unique record_id coverage only. Repeated record_id values in events.csv "
        "are valid, so do not classify them as duplicate errors. Identify entities with "
        "no corresponding event and event IDs with no entity."
    )
    result = ReconAIInvestigationFlow(
        question, [str(entities), str(events)], runtime=CoverageRuntime()
    ).run()

    relationships = result.state.dataset_profiles[0].possible_relationships
    assert relationships[0]["cardinality"] == "one_to_many"
    assert result.state.attempt_count == 0
    assert "`entities.csv` has **1** `record_id` value" in result.report
    assert "All `record_id` values in `events.csv` have a matching record" in result.report
    assert "| Record ID | Status | Created at |" in result.report
    assert "| C | Complete | 2024-01-03 |" in result.report
    assert "Technical cause: Not established" in result.report
    assert "count differs" not in result.report
    assert "exact duplicate" not in result.report
    assert "Affected segment" not in result.report
    assert "Affected period" not in result.report
    assert all(
        item.supporting_details.get("analysis_type") != "repeated_key"
        for item in result.state.evidence
        if item.tool == "find_duplicates"
    )


def test_normal_key_question_returns_identifier_and_context_without_schema_instructions(
    tmp_path,
):
    orders = tmp_path / "orders.csv"
    payments = tmp_path / "payments.csv"
    pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "order_status": ["shipped", "delivered"],
            "order_purchase_timestamp": ["2024-01-01 10:00:00", "2024-01-02 11:00:00"],
        }
    ).to_csv(orders, index=False)
    pd.DataFrame(
        {
            "order_id": ["A", "A"],
            "payment_sequence": [1, 2],
            "payment_value": [5.0, 10.0],
        }
    ).to_csv(payments, index=False)

    class DeterministicRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("The baseline evidence fully answers this question")

        def verify(self, state, attempt):
            raise AssertionError("The baseline evidence fully answers this question")

        def report(self, state, conclusive):
            return CrewAIRuntime.report(self, state, conclusive)

    result = ReconAIInvestigationFlow(
        "Which order IDs have no corresponding payment, and which payment IDs have no order?",
        [str(orders), str(payments)],
        runtime=DeterministicRuntime(),
    ).run()

    assert result.state.attempt_count == 0
    assert "`orders.csv` has **1** `order_id` value" in result.report
    assert "| Order ID | Order status | Order purchase timestamp |" in result.report
    assert "| --- | --- | --- |" in result.report
    assert "| B | delivered | 2024-01-02 11:00:00 |" in result.report
    assert "All `order_id` values in `payments.csv` have a matching record" in result.report
    assert "repeated keys" not in result.report
    assert "Technical cause: Not established" in result.report
    assert "Search the source system behind `payments.csv` for `order_id=B`" in result.report
    assert "Review ingestion and processing logs around `2024-01-02 11:00:00`" in result.report


def test_broad_question_resolves_simple_one_to_many_key_gap_without_llm(tmp_path):
    orders = tmp_path / "orders.csv"
    payments = tmp_path / "payments.csv"
    pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "order_status": ["shipped", "delivered"],
            "order_purchase_timestamp": ["2024-01-01", "2024-01-02"],
        }
    ).to_csv(orders, index=False)
    pd.DataFrame(
        {
            "order_id": ["A", "A"],
            "payment_sequence": [1, 2],
            "payment_value": [5.0, 10.0],
        }
    ).to_csv(payments, index=False)

    class DeterministicRuntime:
        def review_profiles(self, state):
            return None

        def investigate(self, state, previous_verification):
            raise AssertionError("A simple verified key gap should not depend on an LLM")

        def verify(self, state, attempt):
            raise AssertionError("A simple verified key gap should not depend on an LLM")

        def report(self, state, conclusive):
            return CrewAIRuntime.report(self, state, conclusive)

    result = ReconAIInvestigationFlow(
        "Investigate why these datasets disagree and show the affected records.",
        [str(orders), str(payments)],
        runtime=DeterministicRuntime(),
    ).run()

    assert result.state.attempt_count == 0
    assert "`orders.csv` has **1** `order_id` value" in result.report
    assert "| B | delivered | 2024-01-02 |" in result.report
    assert "Finding confirmed from uploaded files" in result.report
    assert "Confidence in data finding" in result.report
    assert "participating in repeated keys" not in result.report
    assert "exact duplicate" not in result.report
    assert "## Main Discrepancy" not in result.report
    assert "Scoped 1 affected record" not in result.report
