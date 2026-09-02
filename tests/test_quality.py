from __future__ import annotations

from src.agents.runtime import _render_report, evidence_quality_guide
from src.models import (
    BusinessReportDraft,
    DatasetProfile,
    InvestigationState,
)
from src.state import add_trace


def test_quality_guide_prefers_channel_and_flags_constant_status():
    profile = DatasetProfile(
        dataset="payments.csv",
        path="payments.csv",
        row_count=100,
        columns=["payment_id", "channel", "payment_method", "payment_status", "amount"],
        data_types={
            "payment_id": "object",
            "channel": "object",
            "payment_method": "object",
            "payment_status": "object",
            "amount": "float64",
        },
        null_counts={column: 0 for column in [
            "payment_id", "channel", "payment_method", "payment_status", "amount"
        ]},
        duplicate_count=0,
        unique_counts={
            "payment_id": 100,
            "channel": 3,
            "payment_method": 3,
            "payment_status": 1,
            "amount": 90,
        },
        possible_identifier_columns=["payment_id"],
        numeric_columns=["amount"],
    )
    guide = evidence_quality_guide([profile], "Why do totals not match?")
    assert guide.index("channel") < guide.index("payment_method")
    assert "payment_status" in guide
    assert "constant/non-informative" in guide


def test_report_renderer_is_consistent_and_readable():
    state = InvestigationState(
        investigation_id="12345678-0000-0000-0000-000000000000",
        question="Why do totals differ?",
        dataset_paths=["orders.csv", "payments.csv"],
    )
    draft = BusinessReportDraft(
        executive_summary="The mismatch is explained by orphan payment rows (E001).",
        data_level_cause="Payments exist without matching order rows (E001).",
        underlying_technical_cause="Source-system logs are required.",
        confidence="High",
        main_discrepancy="Payment total exceeds order total (E001).",
        key_evidence=["E001: 10 unmatched payments."],
        affected_records="10 records (E001)",
        affected_segment="Mobile (E002)",
        affected_period="2024-08-24 onward (E003)",
        measurable_impact="1,000.00 in directly affected value (E004)",
        recommended_actions=["Reconcile the missing order IDs."],
        remaining_uncertainty=["The technical failure mechanism is unknown."],
    )
    report = _render_report(draft, state, conclusive=True)
    assert "## Root Cause Assessment" in report
    assert "| Affected segment | Mobile (E002) |" in report
    assert "Underlying technical cause" in report
    assert "## Recommended Next Actions" in report


def test_trace_prints_live_safe_event(capsys):
    state = InvestigationState(
        investigation_id="abcdef12-0000-0000-0000-000000000000",
        question="Test",
        dataset_paths=["test.csv"],
    )
    add_trace(state, "tool", "Profiled test.csv")
    output = capsys.readouterr().out
    assert "[ReconAI:abcdef12]" in output
    assert "[TOOL]" in output
    assert "Profiled test.csv" in output

