"""CrewAI-backed implementation of the four agent roles."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.agents.investigator import create_investigator_agent
from src.agents.profiler import create_profiler_agent
from src.agents.reporter import create_reporter_agent
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
        max_tokens=4096,
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
                "- Coverage targets (choose order and tools adaptively): quantify the gap; "
                "test shared keys in relevant directions; test duplicates if useful; "
                "locate affected records in a ranked informative business dimension; "
                "locate the onset/time range; and calculate only directly measurable impact.",
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


def _render_report(
    draft: BusinessReportDraft,
    state: InvestigationState,
    conclusive: bool,
) -> str:
    status = "Completed — Evidence-supported" if conclusive else "Completed — Inconclusive"
    confidence = draft.confidence if conclusive else "Low"
    technical_cause = (
        draft.underlying_technical_cause
        if conclusive
        else "Inconclusive — the available CSV evidence does not prove the technical mechanism."
    )
    root_cause_label = draft.data_level_cause if conclusive else "Inconclusive"
    evidence = draft.key_evidence or [
        f"{item.evidence_id}: {item.finding}" for item in state.evidence
    ]
    actions = draft.recommended_actions or ["Collect the missing source-system evidence."]
    uncertainty = draft.remaining_uncertainty or ["No additional uncertainty was supplied."]
    evidence_md = "\n".join(f"- {item}" for item in evidence)
    actions_md = "\n".join(f"{index}. {item}" for index, item in enumerate(actions, 1))
    uncertainty_md = "\n".join(f"- {item}" for item in uncertainty)
    return f"""# Investigation Report

> **Status:** {status}  
> **Confidence:** {confidence}

## Executive Summary

{draft.executive_summary}

## Main Discrepancy

{draft.main_discrepancy}

## Root Cause Assessment

**Root Cause: {root_cause_label}**

| Assessment level | Conclusion |
|---|---|
| What the data proves | {draft.data_level_cause} |
| Underlying technical cause | {technical_cause} |

**Confidence: {confidence}**

## Scope and Business Impact

| Measure | Result |
|---|---|
| Affected records | {draft.affected_records} |
| Affected segment | {draft.affected_segment} |
| Affected period | {draft.affected_period} |
| Measurable impact | {draft.measurable_impact} |

## Key Evidence

{evidence_md}

## Recommended Next Actions

{actions_md}

## Remaining Uncertainty

{uncertainty_md}
"""


class CrewAIRuntime:
    """Construct exactly four agents and run one bounded task per role invocation."""

    def __init__(self, state: InvestigationState, settings: Settings | None = None):
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
        self.profiler = create_profiler_agent(self.llm)
        self.investigator = create_investigator_agent(self.llm, tools)
        self.verifier = create_verifier_agent(self.llm)
        self.reporter = create_reporter_agent(self.llm)

    def review_profiles(self, state: InvestigationState) -> ProfileReview:
        compact = [profile.model_dump(mode="json") for profile in state.dataset_profiles]
        prompt = f"""
Review these deterministic dataset profiles. Do not calculate new statistics or invent
relationships. Mention a relationship only when its supplied basis supports it.

Profiles:
{json.dumps(compact, default=str)}
"""
        return self._run_structured(self.profiler, prompt, ProfileReview)

    def investigate(
        self, state: InvestigationState, previous_verification: VerificationResult | None
    ) -> InvestigationAttempt:
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

Choose useful tools yourself. All important numerical facts must come from tool calls.
Do not repeat an exact prior call. On a retry, address the missing evidence and explore a
different grouping, direction, filter, metric, or time grain. Use evidence IDs returned by
tools. Before segmenting, inspect profile cardinality: a constant status field cannot explain
concentration. Prefer business-origin dimensions (for example channel, region, customer
segment, platform, or source) over statuses and methods when profiles support them. End with
findings, a cautious data-level root-cause hypothesis, remaining uncertainty, and the evidence
IDs that support it. Association must not be stated as proven causation.
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
- Judge the data-level explanation separately from the underlying technical mechanism.
- Do not reject a well-supported data-level explanation merely because application logs are
  required to prove the deeper technical mechanism; record that as remaining uncertainty.
- A claimed affected segment is meaningful only when its source column has more than one
  value and tool evidence shows concentration. A constant status such as all "Captured" is
  not an affected segment.
- For a reconciliation mismatch, check whether the numerical gap, key-level mismatch,
  informative segment, time range, and impact form a consistent explanation. Request the
  specific missing check when they do not.
"""
        return self._run_structured(self.verifier, prompt, VerificationResult)

    def report(self, state: InvestigationState, conclusive: bool) -> str:
        status_instruction = (
            "The verifier accepted the evidence-backed hypothesis."
            if conclusive
            else (
                "Maximum attempts were reached. The report MUST state exactly "
                "'Root Cause: Inconclusive' and 'Confidence: Low'."
            )
        )
        prompt = f"""
Prepare a structured enterprise investigation report for an investigator. Use plain English,
short sentences, and evidence IDs beside factual claims. Separate what the CSV data proves
(the data-level cause) from the deeper technical mechanism, which may remain unknown. An
affected segment must be an informative dimension with multiple possible values; never label
a constant status as the affected segment. Never add a number not present in evidence and
never estimate losses. Provide 3-5 specific next actions, ordered by priority.
{status_instruction}

Question: {state.question}
Hypothesis: {state.hypothesis}
Verification: {json.dumps(state.verification.model_dump(mode='json') if state.verification else None)}
Dataset quality guide: {evidence_quality_guide(state.dataset_profiles, state.question)}
Evidence: {json.dumps([_normalize_numbers(item.model_dump(mode='json')) for item in state.evidence], default=str)}
"""
        draft = self._run_structured(self.reporter, prompt, BusinessReportDraft)
        return _render_report(draft, state, conclusive)

    @staticmethod
    def _run_structured(agent: Any, description: str, model: type[BaseModel]) -> Any:
        from crewai import Crew, Process, Task

        task = Task(
            description=description,
            expected_output=f"Valid structured output matching {model.__name__}",
            agent=agent,
            output_pydantic=model,
        )
        output = Crew(
            agents=[agent], tasks=[task], process=Process.sequential, verbose=False
        ).kickoff()
        parsed = getattr(output, "pydantic", None)
        if parsed is None and getattr(output, "tasks_output", None):
            parsed = getattr(output.tasks_output[-1], "pydantic", None)
        if isinstance(parsed, model):
            return parsed
        raw = getattr(output, "raw", str(output))
        return model.model_validate_json(raw)
