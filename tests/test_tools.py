from __future__ import annotations

import pytest

from src.tools import (
    calculate_business_impact,
    compare_aggregates,
    find_duplicates,
    find_unmatched_records,
    profile_dataset,
    segment_analysis,
)


def test_profile_dataset(datasets):
    orders, _ = datasets
    result = profile_dataset(orders)
    assert result["row_count"] == 4
    assert result["duplicate_count"] == 2
    assert "order_date" in result["possible_date_columns"]
    assert result["numeric_summaries"]["amount"]["sum"] == 900.0


def test_compare_aggregates(datasets):
    orders, payments = datasets
    result = compare_aggregates(orders, payments, "amount", "amount", "sum")
    assert result["dataset_a_result"] == 900.0
    assert result["dataset_b_result"] == 1200.0
    assert result["absolute_difference"] == 300.0
    assert result["percentage_difference_vs_b"] == pytest.approx(-25.0)


def test_find_unmatched_records(datasets):
    orders, payments = datasets
    result = find_unmatched_records(payments, orders, "order_id", "order_id")
    assert result["matched_count"] == 2
    assert result["unmatched_count"] == 2
    assert result["unmatched_percentage"] == 50.0
    assert result["sample_unmatched_identifiers"] == ["O4", "O5"]


def test_find_duplicates(datasets):
    orders, _ = datasets
    result = find_duplicates(orders, ["order_id"])
    assert result["duplicate_row_count"] == 2
    assert result["duplicate_percentage"] == 50.0
    assert result["sample_duplicate_identifiers"] == [{"order_id": "O3"}]


def test_segment_analysis_for_unmatched_rows(datasets):
    orders, payments = datasets
    result = segment_analysis(
        payments,
        group_by_column="channel",
        aggregation="count",
        match_against_path=orders,
        local_key="order_id",
        other_key="order_id",
        only_unmatched=True,
    )
    assert result["filtered_record_count"] == 2
    assert {row["channel"]: row["value"] for row in result["results"]} == {
        "Mobile": 1,
        "Partner": 1,
    }


def test_calculate_business_impact(datasets):
    orders, payments = datasets
    result = calculate_business_impact(
        payments,
        amount_column="amount",
        date_column="payment_date",
        segment_column="channel",
        match_against_path=orders,
        local_key="order_id",
        other_key="order_id",
        only_unmatched=True,
    )
    assert result["affected_record_count"] == 2
    assert result["affected_percentage"] == 50.0
    assert result["affected_amount_sum"] == 900.0
    assert result["affected_date_range"]["start"].startswith("2024-08-25")

