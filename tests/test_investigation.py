from __future__ import annotations

import pandas as pd

from src.agents.runtime import CrewAIRuntime
from src.flow import ReconAIInvestigationFlow
from src.flow.investigation_flow import apply_verification_guard
from src.models import (
    EvidenceItem,
    InvestigationAttempt,
    InvestigationState,
    Verdict,
    VerificationResult,
)


def test_controller_allows_acceptance_from_cited_baseline_evidence():
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

    assert guarded == accepted


def test_controller_rejects_acceptance_without_a_valid_evidence_citation():
    state = InvestigationState(
        investigation_id="citation-contract",
        question="Why do these files differ?",
        dataset_paths=["source.csv", "target.csv"],
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=0,
                finding="Deterministic comparison",
                tool="compare_aggregates",
            )
        ],
    )
    attempt = InvestigationAttempt(
        hypothesis="The files differ.", evidence_ids=["E999"]
    )
    accepted = VerificationResult(
        verdict=Verdict.ACCEPT, confidence=0.9, reason="Supported."
    )

    guarded = apply_verification_guard(state, attempt, accepted)

    assert guarded.verdict == Verdict.REJECT
    assert "valid evidence citation" in guarded.reason


def test_controller_leaves_semantic_judgment_to_verifier():
    state = InvestigationState(
        investigation_id="semantic-verifier",
        question="Which priority contains the amount difference?",
        dataset_paths=["source.csv", "target.csv"],
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=1,
                finding="Paired priority totals",
                tool="segment_analysis",
            )
        ],
    )
    attempt = InvestigationAttempt(
        hypothesis="The amount difference is in 1-URGENT.", evidence_ids=["E001"]
    )
    rejection = VerificationResult(
        verdict=Verdict.REJECT,
        confidence=0.9,
        reason="The cited totals contradict the conclusion.",
    )
    acceptance = VerificationResult(
        verdict=Verdict.ACCEPT,
        confidence=0.9,
        reason="The cited totals support the conclusion.",
    )

    assert apply_verification_guard(state, attempt, rejection) == rejection
    assert apply_verification_guard(state, attempt, acceptance) == acceptance


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


def test_verifier_rejects_unsupported_technical_cause(datasets):
    class UnsafeRuntime(AcceptingRuntime):
        def investigate(self, state, previous_verification):
            attempt = super().investigate(state, previous_verification)
            attempt.hypothesis = "An ETL pipeline failure caused the missing records."
            return attempt

        def verify(self, state, attempt):
            return VerificationResult(
                verdict=Verdict.REJECT,
                confidence=0.2,
                reason="The technical mechanism is not supported by CSV evidence.",
                missing_evidence=["State only the supported data-level finding."],
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
    assert "Controller overruled verifier" not in result.trace


class BaselineUsingRuntime:
    def __init__(self, hypothesis: str):
        self.hypothesis = hypothesis
        self.review_calls = 0
        self.investigate_calls = 0
        self.verify_calls = 0

    def review_profiles(self, state):
        self.review_calls += 1

    def investigate(self, state, previous_verification):
        self.investigate_calls += 1
        return InvestigationAttempt(
            findings=[self.hypothesis],
            hypothesis=self.hypothesis,
            evidence_ids=[item.evidence_id for item in state.evidence],
        )

    def verify(self, state, attempt):
        self.verify_calls += 1
        return VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=0.95,
            reason="The cited baseline evidence supports the data-level finding.",
        )

    def report(self, state, conclusive):
        return CrewAIRuntime.report(self, state, conclusive)


def test_complete_clean_match_runs_both_agents_without_extra_tools(tmp_path):
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

    runtime = BaselineUsingRuntime("No discrepancy detected within the validated scope.")
    result = ReconAIInvestigationFlow(
        "Do these files disagree?",
        [str(source), str(target)],
        runtime=runtime,
    ).run()
    assert result.state.attempt_count == 1
    assert result.state.verification.verdict == Verdict.ACCEPT
    assert runtime.investigate_calls == 1
    assert runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)
    assert "collected 0 new evidence item(s)" in result.trace


def test_proven_missing_records_run_both_agents_without_extra_tools(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1], "amount": [10.0]}
    ).to_csv(target, index=False)

    runtime = BaselineUsingRuntime(
        "One record_id value in source.csv is absent from target.csv."
    )
    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=runtime,
    ).run()
    assert result.state.attempt_count == 1
    assert "absent" in result.state.hypothesis
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)


def test_proven_duplicate_records_run_both_agents_without_extra_tools(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1, 2, 2], "amount": [10.0, 20.0, 20.0]}
    ).to_csv(target, index=False)

    runtime = BaselineUsingRuntime(
        "target.csv contains an extra exact duplicate record."
    )
    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=runtime,
    ).run()
    assert result.state.attempt_count == 1
    assert "duplicate" in result.state.hypothesis
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)


def test_proven_value_drift_runs_both_agents_without_extra_tools(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 20.0]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"record_id": [1, 2], "amount": [10.0, 25.0]}
    ).to_csv(target, index=False)

    runtime = BaselineUsingRuntime(
        "One matched record contains a different amount value."
    )
    result = ReconAIInvestigationFlow(
        "Why do the files disagree?",
        [str(source), str(target)],
        runtime=runtime,
    ).run()
    assert result.state.attempt_count == 1
    assert "amount" in result.state.hypothesis
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)


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

    runtime = BaselineUsingRuntime(
        "One record_id value in entities.csv is absent from events.csv; all event "
        "record_id values have a matching entity."
    )
    question = (
        "Compare unique record_id coverage only. Repeated record_id values in events.csv "
        "are valid, so do not classify them as duplicate errors. Identify entities with "
        "no corresponding event and event IDs with no entity."
    )
    result = ReconAIInvestigationFlow(
        question, [str(entities), str(events)], runtime=runtime
    ).run()

    relationships = result.state.dataset_profiles[0].possible_relationships
    assert relationships[0]["cardinality"] == "one_to_many"
    assert result.state.attempt_count == 1
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)
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

    runtime = BaselineUsingRuntime(
        "One order_id value in orders.csv is absent from payments.csv; all payment "
        "order_id values have a matching order."
    )
    result = ReconAIInvestigationFlow(
        "Which order IDs have no corresponding payment, and which payment IDs have no order?",
        [str(orders), str(payments)],
        runtime=runtime,
    ).run()

    assert result.state.attempt_count == 1
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)
    assert "`orders.csv` has **1** `order_id` value" in result.report
    assert "| Order ID | Order status | Order purchase timestamp |" in result.report
    assert "| --- | --- | --- |" in result.report
    assert "| B | delivered | 2024-01-02 11:00:00 |" in result.report
    assert "All `order_id` values in `payments.csv` have a matching record" in result.report
    assert "repeated keys" not in result.report
    assert "Technical cause: Not established" in result.report
    assert "Search the source system behind `payments.csv` for `order_id=B`" in result.report
    assert "Review ingestion and processing logs around `2024-01-02 11:00:00`" in result.report


def test_broad_question_resolves_simple_one_to_many_key_gap_with_both_agents(tmp_path):
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

    runtime = BaselineUsingRuntime(
        "One order_id value in orders.csv is absent from payments.csv; all payment "
        "order_id values have a matching order."
    )
    result = ReconAIInvestigationFlow(
        "Investigate why these datasets disagree and show the affected records.",
        [str(orders), str(payments)],
        runtime=runtime,
    ).run()

    assert result.state.attempt_count == 1
    assert runtime.investigate_calls == runtime.verify_calls == 1
    assert all(item.attempt == 0 for item in result.state.evidence)
    assert "`orders.csv` has **1** `order_id` value" in result.report
    assert "| B | delivered | 2024-01-02 |" in result.report
    assert "Finding confirmed from uploaded files" in result.report
    assert "Confidence in data finding" in result.report
    assert "participating in repeated keys" not in result.report
    assert "exact duplicate" not in result.report
    assert "## Main Discrepancy" not in result.report
    assert "Scoped 1 affected record" not in result.report
