from __future__ import annotations

import pandas as pd
import pytest

from src.baseline import collect_baseline_evidence
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


@pytest.mark.parametrize(
    "question",
    [
        "Calculate the net source-minus-target amount difference.",
        "Which channel contains the amount difference?",
        "Explain what is causing the amount difference and identify affected records.",
    ],
)
def test_baseline_does_not_preempt_adaptive_agent_analysis(datasets, question):
    orders, payments = datasets
    flow = ReconAIInvestigationFlow(question, [orders, payments])
    flow._profile()
    collect_baseline_evidence(flow.state)
    assert all(item.attempt == 0 for item in flow.state.evidence)
    assert not any(
        item.tool == "compare_aggregates"
        and item.supporting_details.get("aggregation") == "sum"
        for item in flow.state.evidence
    )
    assert not any(
        item.tool in {"segment_analysis", "reconcile_record_set_contributions"}
        for item in flow.state.evidence
    )


def test_offsetting_contribution_bridge_is_not_mandatory_baseline_evidence(tmp_path):
    ledger = tmp_path / "ledger.csv"
    archive = tmp_path / "archive.csv"
    pd.DataFrame(
        {"invoice_id": ["A", "B", "C"], "balance": [100.0, 200.0, 300.0]}
    ).to_csv(ledger, index=False)
    pd.DataFrame(
        {"invoice_id": ["A", "B", "B"], "balance": [100.0, 200.0, 200.0]}
    ).to_csv(archive, index=False)
    flow = ReconAIInvestigationFlow(
        "Explain what is causing the balance totals to differ, identify affected records, "
        "and calculate the net difference.",
        [str(ledger), str(archive)],
    )
    flow._profile()
    collect_baseline_evidence(flow.state)
    assert all(item.attempt == 0 for item in flow.state.evidence)
    assert not any(
        item.tool == "reconcile_record_set_contributions"
        for item in flow.state.evidence
    )
