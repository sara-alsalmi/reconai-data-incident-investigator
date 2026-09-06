from __future__ import annotations

from src.agents.runtime import _render_report, evidence_quality_guide
from src.models import (
    BusinessReportDraft,
    DatasetProfile,
    EvidenceItem,
    InvestigationState,
    Verdict,
    VerificationResult,
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
        hypothesis="Payments exist without matching order rows.",
        verification=VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=0.9,
            reason="Supported by deterministic evidence.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=1,
                finding="Calculated impact for 10 affected records in payments.csv",
                tool="calculate_business_impact",
                datasets=["payments.csv", "orders.csv"],
                value=10,
                supporting_details={
                    "affected_record_count": 10,
                    "affected_amount_sum": 1000.0,
                    "affected_date_range": {
                        "start": "2024-08-24",
                        "end": "2024-08-30",
                    },
                    "primary_affected_segment": {
                        "segment": "Mobile",
                        "record_count": 8,
                        "percentage_of_affected": 80.0,
                    },
                },
            )
        ],
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
    assert "## Answer" in report
    assert "## Root Cause Assessment" not in report
    assert "## Impact and scope" in report
    assert "| Affected segment | Mobile — 8 records (E001) |" in report
    assert "## What remains unknown" in report
    assert "Technical cause: Not established" in report
    assert "1,000.00 in affected value (E001)" in report
    assert "## Recommended next steps" in report


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


def test_report_ignores_unfiltered_whole_dataset_impact():
    state = InvestigationState(
        investigation_id="report-guard",
        question="Which priority contains the amount discrepancy?",
        dataset_paths=["source.csv", "warehouse.csv"],
        hypothesis="45 amount mismatches are localized to 2-HIGH.",
        verification=VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=0.95,
            reason="The paired segment totals isolate the mismatch.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=0,
                finding="Found 45 value mismatches",
                tool="compare_record_values",
                datasets=["source.csv", "warehouse.csv"],
                value=45,
                supporting_details={
                    "dataset_a": "source.csv",
                    "dataset_b": "warehouse.csv",
                    "value_mismatch_count": 45,
                    "signed_difference_a_minus_b": -1057.5,
                },
            ),
            EvidenceItem(
                evidence_id="E002",
                attempt=1,
                finding="Calculated impact for all 15000 rows",
                tool="calculate_business_impact",
                datasets=["source.csv"],
                value=15000,
                supporting_details={
                    "dataset": "source.csv",
                    "affected_record_count": 15000,
                    "total_record_count": 15000,
                    "affected_amount_sum": 999999.0,
                    "primary_affected_segment": {
                        "segment": "2-HIGH",
                        "record_count": 3065,
                    },
                },
            ),
            EvidenceItem(
                evidence_id="E003",
                attempt=1,
                finding="Source totals by priority",
                tool="segment_analysis",
                datasets=["source.csv"],
                supporting_details={
                    "dataset": "source.csv",
                    "aggregation": "sum",
                    "metric_column": "total_amount",
                    "grouping": ["order_priority"],
                    "results": [
                        {"order_priority": "2-HIGH", "value": 100.0},
                        {"order_priority": "3-MEDIUM", "value": 50.0},
                    ],
                    "results_truncated": False,
                },
            ),
            EvidenceItem(
                evidence_id="E004",
                attempt=1,
                finding="Warehouse totals by priority",
                tool="segment_analysis",
                datasets=["warehouse.csv"],
                supporting_details={
                    "dataset": "warehouse.csv",
                    "aggregation": "sum",
                    "metric_column": "total_amount",
                    "grouping": ["order_priority"],
                    "results": [
                        {"order_priority": "2-HIGH", "value": 1157.5},
                        {"order_priority": "3-MEDIUM", "value": 50.0},
                    ],
                    "results_truncated": False,
                },
            ),
        ],
    )
    draft = BusinessReportDraft(
        executive_summary="unused",
        data_level_cause="unused",
        underlying_technical_cause="unknown",
        confidence="High",
        main_discrepancy="unused",
        affected_records="unused",
        affected_segment="unused",
        affected_period="unused",
        measurable_impact="unused",
    )
    report = _render_report(draft, state, conclusive=True)
    assert "45 matched-key rows with value mismatches" in report
    assert "order_priority = 2-HIGH" in report
    assert "-1,057.50 signed difference" in report
    assert "| Affected records | 15000 records" not in report
    assert "Calculated impact for all 15000 rows" not in report


def test_report_respects_key_coverage_scope_and_one_to_many_grain():
    question = (
        "Compare unique record_id coverage only. Repeated record_id values are valid; "
        "do not classify them as duplicate errors."
    )
    entity_profile = DatasetProfile(
        dataset="entities.csv",
        path="entities.csv",
        row_count=3,
        columns=["record_id", "status"],
        data_types={"record_id": "object", "status": "object"},
        null_counts={"record_id": 0, "status": 0},
        duplicate_count=0,
        unique_counts={"record_id": 3, "status": 1},
        possible_identifier_columns=["record_id"],
        possible_relationships=[
            {
                "dataset": "events.csv",
                "local_column": "record_id",
                "other_column": "record_id",
                "cardinality": "one_to_many",
            }
        ],
    )
    event_profile = DatasetProfile(
        dataset="events.csv",
        path="events.csv",
        row_count=7,
        columns=["record_id", "sequence"],
        data_types={"record_id": "object", "sequence": "int64"},
        null_counts={"record_id": 0, "sequence": 0},
        duplicate_count=0,
        unique_counts={"record_id": 3, "sequence": 3},
        possible_identifier_columns=["record_id"],
    )
    state = InvestigationState(
        investigation_id="scope",
        question=question,
        dataset_paths=["entities.csv", "events.csv"],
        dataset_profiles=[entity_profile, event_profile],
        hypothesis=(
            "Within the uploaded files, one entity key has no corresponding event key."
        ),
        verification=VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=1.0,
            reason="Bidirectional key checks answer the requested scope.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=0,
                finding="count differs by 4 between events.csv and entities.csv",
                tool="compare_aggregates",
                datasets=["events.csv", "entities.csv"],
                value=4,
                supporting_details={
                    "dataset_a": "events.csv",
                    "dataset_b": "entities.csv",
                    "aggregation": "count",
                },
            ),
            EvidenceItem(
                evidence_id="E002",
                attempt=0,
                finding="Found 1 unmatched record in entities.csv",
                tool="find_unmatched_records",
                datasets=["entities.csv", "events.csv"],
                value=1,
                supporting_details={
                    "dataset_a": "entities.csv",
                    "dataset_b": "events.csv",
                    "unmatched_count": 1,
                },
            ),
            EvidenceItem(
                evidence_id="E003",
                attempt=0,
                finding="Found 4 records participating in repeated keys",
                tool="find_duplicates",
                datasets=["events.csv"],
                value=4,
                supporting_details={
                    "dataset": "events.csv",
                    "analysis_type": "repeated_key",
                    "duplicate_row_count": 4,
                },
            ),
        ],
    )
    draft = BusinessReportDraft(
        executive_summary="unused",
        data_level_cause="unused",
        underlying_technical_cause="unknown",
        confidence="High",
        main_discrepancy="unused",
        affected_records="unused",
        affected_segment="unused",
        affected_period="unused",
        measurable_impact="unused",
    )

    report = _render_report(draft, state, conclusive=True)

    assert "`entities.csv` has **1** `key` value" in report
    assert "count differs by 4" not in report
    assert "participating in repeated keys" not in report
    assert "Affected segment" not in report
    assert "Affected period" not in report


def test_report_never_labels_repeated_relationship_keys_as_exact_duplicates():
    state = InvestigationState(
        investigation_id="repeated-key-report",
        question="Investigate why these datasets disagree and show affected records.",
        dataset_paths=["entities.csv", "events.csv"],
        hypothesis="One entity key is absent from events.csv.",
        verification=VerificationResult(
            verdict=Verdict.ACCEPT,
            confidence=1.0,
            reason="Supported by key evidence.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="E001",
                attempt=1,
                finding=(
                    "Found 7 records participating in repeated keys; repeated keys are "
                    "not automatically errors"
                ),
                tool="find_duplicates",
                datasets=["events.csv"],
                value=7,
                supporting_details={
                    "dataset": "events.csv",
                    "analysis_type": "repeated_key",
                    "duplicate_row_count": 7,
                },
            )
        ],
    )
    draft = BusinessReportDraft(
        executive_summary="unused",
        data_level_cause="unused",
        underlying_technical_cause="unknown",
        confidence="High",
        main_discrepancy="unused",
        affected_records="unused",
        affected_segment="unused",
        affected_period="unused",
        measurable_impact="unused",
    )

    report = _render_report(draft, state, conclusive=True)

    assert "participating in repeated keys" not in report
    assert "7 exact duplicate rows" not in report
