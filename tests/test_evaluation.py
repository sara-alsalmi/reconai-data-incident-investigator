from __future__ import annotations

from types import SimpleNamespace

from scripts.run_tpch_evaluation import score_run
from src.models import (
    EvidenceItem,
    InvestigationState,
    Verdict,
    VerificationResult,
)


def _result(hypothesis: str):
    state = InvestigationState(
        investigation_id="eval",
        question="Why are records missing?",
        dataset_paths=["source.csv", "warehouse.csv"],
        hypothesis=hypothesis,
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=1,
                finding="Found 60 unmatched records",
                tool="find_unmatched_records",
                datasets=["source.csv", "warehouse.csv"],
                supporting_details={
                    "dataset_a": "source.csv",
                    "unmatched_count": 60,
                },
            )
        ],
    )
    return SimpleNamespace(state=state, report="60 unmatched records are missing.")


def test_evaluation_accepts_supported_data_level_hypothesis():
    expected = {
        "required_terms": ["missing", "unmatched"],
        "forbidden_hypothesis_terms": ["pipeline", "etl"],
        "checks": [
            {
                "tool": "find_unmatched_records",
                "field": "unmatched_count",
                "value": 60,
                "dataset_a": "source.csv",
            }
        ],
    }
    score = score_run(_result("60 source records are missing from the warehouse."), expected)
    assert score["passed"] is True


def test_evaluation_rejects_unsupported_technical_cause():
    expected = {
        "required_terms": ["missing", "unmatched"],
        "forbidden_hypothesis_terms": ["pipeline", "etl"],
        "checks": [
            {
                "tool": "find_unmatched_records",
                "field": "unmatched_count",
                "value": 60,
                "dataset_a": "source.csv",
            }
        ],
    }
    score = score_run(
        _result("A pipeline failure caused 60 records to be missing."), expected
    )
    assert score["evidence_checks_passed"] is True
    assert score["causal_safety_passed"] is False
    assert score["passed"] is False


def test_agentic_evaluation_rejects_baseline_only_result():
    expected = {
        "required_all_terms": ["missing"],
        "min_attempts": 1,
        "min_agent_tool_calls": 1,
        "required_agent_tools": {"compare_aggregates": 1},
        "checks": [
            {
                "tool": "find_unmatched_records",
                "field": "unmatched_count",
                "value": 60,
            }
        ],
    }
    result = _result("60 records are missing.")
    score = score_run(result, expected)
    assert score["evidence_checks_passed"] is True
    assert score["agentic_requirements_passed"] is False
    assert score["passed"] is False


def test_agentic_evaluation_requires_matching_agent_tool_details():
    result = _result("Missing records and duplicate records offset one another.")
    result.state.attempt_count = 1
    result.state.verification = VerificationResult(
        verdict=Verdict.ACCEPT,
        confidence=0.9,
        reason="Agent evidence supports both observed causes.",
    )
    result.state.evidence.append(
        EvidenceItem(
            evidence_id="E002",
            attempt=1,
            finding="Compared total_amount sums",
            tool="compare_aggregates",
            datasets=["source.csv", "warehouse.csv"],
            supporting_details={
                "aggregation": "sum",
                "metric_column_a": "total_amount",
                "metric_column_b": "total_amount",
                "signed_difference_a_minus_b": 125.0,
            },
        )
    )
    expected = {
        "required_all_terms": ["missing", "duplicate"],
        "min_attempts": 1,
        "min_agent_tool_calls": 1,
        "required_agent_tools": {"compare_aggregates": 1},
        "required_verdict": "ACCEPT",
        "checks": [
            {
                "tool": "compare_aggregates",
                "field": "signed_difference_a_minus_b",
                "value": 125.0,
                "attempt_min": 1,
                "details": {
                    "aggregation": "sum",
                    "metric_column_a": "total_amount",
                    "metric_column_b": "total_amount",
                },
            }
        ],
    }
    score = score_run(result, expected)
    assert score["agentic_requirements_passed"] is True
    assert score["verdict_requirement_passed"] is True
    assert score["evidence_checks_passed"] is True
    assert score["passed"] is True
