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


def apply_verification_guard(
    state: InvestigationState,
    attempt: InvestigationAttempt,
    verification: VerificationResult,
) -> VerificationResult:
    """Enforce only the mechanical evidence contract around verifier decisions."""
    known_evidence_ids = {item.evidence_id for item in state.evidence}
    cited_evidence_ids = set(attempt.evidence_ids)
    if verification.verdict == Verdict.ACCEPT and not (
        known_evidence_ids & cited_evidence_ids
    ):
        return VerificationResult(
            verdict=Verdict.REJECT,
            confidence=min(verification.confidence, 0.3),
            reason="Controller rejected acceptance without a valid evidence citation.",
            missing_evidence=[
                "Cite at least one existing deterministic evidence ID."
            ],
        )
    return verification


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
                    f"Retry requested: {detail}",
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
