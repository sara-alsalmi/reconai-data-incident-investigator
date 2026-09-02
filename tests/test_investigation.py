from __future__ import annotations

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
    assert len(result.state.evidence) == 2
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

