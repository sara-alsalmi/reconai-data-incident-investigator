from __future__ import annotations

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
