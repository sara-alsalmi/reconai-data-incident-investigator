from __future__ import annotations

from src.baseline import (
    collect_baseline_evidence,
    collect_question_required_evidence,
)
from src.flow import ReconAIInvestigationFlow


def test_baseline_collects_mandatory_reconciliation_evidence(datasets):
    orders, payments = datasets
    flow = ReconAIInvestigationFlow("Why do the files disagree?", [orders, payments])
    flow._profile()
    collected = collect_baseline_evidence(flow.state)
    assert collected == 8
    tools = [item.tool for item in flow.state.evidence]
    assert tools.count("find_unmatched_records") == 2
    assert tools.count("find_duplicates") == 2
    assert "compare_aggregates" in tools
    assert "compare_rows_by_key" in tools
    assert tools.count("calculate_business_impact") == 2


def test_controller_collects_explicitly_requested_sum_once(datasets):
    orders, payments = datasets
    flow = ReconAIInvestigationFlow(
        "Calculate the net source-minus-target amount difference.",
        [orders, payments],
    )
    flow._profile()
    collect_baseline_evidence(flow.state)
    flow.state.attempt_count = 1

    assert collect_question_required_evidence(flow.state) == 1
    evidence = flow.state.evidence[-1]
    assert evidence.attempt == 1
    assert evidence.tool == "compare_aggregates"
    assert evidence.supporting_details["aggregation"] == "sum"
    assert evidence.supporting_details["metric_column_a"] == "amount"
    assert evidence.supporting_details["signed_difference_a_minus_b"] == -300.0
    assert collect_question_required_evidence(flow.state) == 0


def test_controller_collects_paired_named_segment_totals(datasets):
    orders, payments = datasets
    flow = ReconAIInvestigationFlow(
        "Which channel contains the amount difference?",
        [orders, payments],
    )
    flow._profile()
    collect_baseline_evidence(flow.state)
    flow.state.attempt_count = 1

    assert collect_question_required_evidence(flow.state) == 3
    segment_evidence = [
        item
        for item in flow.state.evidence
        if item.attempt == 1 and item.tool == "segment_analysis"
    ]
    assert len(segment_evidence) == 2
    assert {item.supporting_details["dataset"] for item in segment_evidence} == {
        "orders.csv",
        "payments.csv",
    }
    assert all(
        item.supporting_details["grouping"] == ["channel"]
        and item.supporting_details["metric_column"] == "amount"
        and item.supporting_details["aggregation"] == "sum"
        for item in segment_evidence
    )
