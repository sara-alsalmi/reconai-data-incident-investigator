"""Structured contracts shared by agents, tools, and orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    dataset: str
    path: str
    row_count: int
    columns: list[str]
    data_types: dict[str, str]
    null_counts: dict[str, int]
    duplicate_count: int
    unique_counts: dict[str, int]
    numeric_summaries: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    possible_identifier_columns: list[str] = Field(default_factory=list)
    possible_date_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    possible_relationships: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    attempt: int
    finding: str
    tool: str
    datasets: list[str] = Field(default_factory=list)
    metric: str | None = None
    value: Any = None
    supporting_details: dict[str, Any] = Field(default_factory=dict)
    call_signature: str = ""


class InvestigationAttempt(BaseModel):
    findings: list[str] = Field(default_factory=list)
    hypothesis: str
    remaining_uncertainty: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class BusinessReportDraft(BaseModel):
    """Structured reporter output rendered deterministically as readable Markdown."""

    executive_summary: str
    data_level_cause: str
    underlying_technical_cause: str
    confidence: Literal["High", "Medium", "Low"]
    main_discrepancy: str
    key_evidence: list[str] = Field(default_factory=list)
    affected_records: str
    affected_segment: str
    affected_period: str
    measurable_impact: str
    recommended_actions: list[str] = Field(default_factory=list)
    remaining_uncertainty: list[str] = Field(default_factory=list)


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class VerificationResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    missing_evidence: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    sequence: int
    category: str
    message: str


class InvestigationState(BaseModel):
    investigation_id: str
    question: str
    dataset_paths: list[str]
    dataset_profiles: list[DatasetProfile] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    hypothesis: str | None = None
    verification: VerificationResult | None = None
    attempt_count: int = 0
    final_report: str | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    tool_call_signatures: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    state: InvestigationState
    report: str
    trace: str
