from __future__ import annotations

import pandas as pd

from src.agents.runtime import CrewAIRuntime
from src.flow import ReconAIInvestigationFlow
from src.models import (
    EvidenceItem,
    InvestigationAttempt,
    Verdict,
    VerificationResult,
)


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
            evidence_id = f"E{len(state.evidence) + 1:03d}"
            state.evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    attempt=state.attempt_count,
                    finding="Amount drift is concentrated in 2-HIGH",
                    tool="segment_analysis",
                    datasets=["source.csv", "target.csv"],
                    value=1,
                    supporting_details={"segment": "2-HIGH"},
                )
            )
            return InvestigationAttempt(
                findings=["2-HIGH contains the amount drift"],
                hypothesis="The amount mismatch is concentrated in 2-HIGH.",
                evidence_ids=[evidence_id],
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
