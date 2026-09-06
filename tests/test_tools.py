from __future__ import annotations

import pandas as pd
import pytest

from src.tools import (
    calculate_business_impact,
    compare_aggregates,
    compare_record_values,
    compare_rows_by_key,
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
    assert result["unmatched_unique_key_count"] == 2
    assert result["unmatched_percentage"] == 50.0
    assert result["sample_unmatched_identifiers"] == ["O4", "O5"]
    assert result["sample_is_complete"] is True
    assert result["sample_unmatched_records"][0] == {
        "order_id": "O4",
        "payment_date": "2024-08-25",
        "amount": 400.0,
        "channel": "Mobile",
    }


def test_find_unmatched_records_distinguishes_rows_from_unique_keys(tmp_path):
    entities = tmp_path / "entities.csv"
    events = tmp_path / "events.csv"
    pd.DataFrame({"record_id": ["A"]}).to_csv(entities, index=False)
    pd.DataFrame(
        {
            "record_id": ["A", "B", "B"],
            "status": ["posted", "failed", "retried"],
            "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    ).to_csv(events, index=False)

    result = find_unmatched_records(
        events,
        entities,
        "record_id",
        "record_id",
        detail_columns=["status", "created_at"],
    )

    assert result["unmatched_count"] == 2
    assert result["unmatched_unique_key_count"] == 1
    assert result["sample_unmatched_records"] == [
        {"record_id": "B", "status": "failed", "created_at": "2024-01-02"}
    ]


def test_compare_record_values(datasets):
    orders, payments = datasets
    result = compare_record_values(
        orders, payments, "order_id", "order_id", "amount", "amount"
    )
    assert result["matched_unique_key_count"] == 2
    assert result["value_mismatch_count"] == 0
    assert result["duplicate_key_record_count_a"] == 2


def test_compare_record_values_detects_drift(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"id": [1, 2], "amount": [10.0, 20.0]}).to_csv(left, index=False)
    pd.DataFrame({"id": [1, 2], "amount": [10.0, 25.5]}).to_csv(right, index=False)
    result = compare_record_values(left, right, "id", "id", "amount", "amount")
    assert result["value_mismatch_count"] == 1
    assert result["signed_difference_a_minus_b"] == -5.5
    assert result["absolute_difference_sum"] == 5.5


def test_compare_record_values_supports_categories(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"id": [1, 2], "status": ["Open", "Closed"]}).to_csv(left, index=False)
    pd.DataFrame({"id": [1, 2], "status": ["Open", "Pending"]}).to_csv(right, index=False)
    result = compare_record_values(left, right, "id", "id", "status", "status")
    assert result["comparison_type"] == "exact"
    assert result["value_mismatch_count"] == 1
    assert result["signed_difference_a_minus_b"] is None


def test_compare_rows_by_key_checks_all_shared_fields(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame(
        {"id": [1, 2], "status": ["Open", "Closed"], "amount": [10.0, 20.0]}
    ).to_csv(left, index=False)
    pd.DataFrame(
        {"id": [1, 2], "status": ["Open", "Pending"], "amount": [10.0, 25.0]}
    ).to_csv(right, index=False)
    result = compare_rows_by_key(left, right, "id", "id")
    assert result["record_mismatch_count"] == 1
    assert result["mismatches_by_column"] == {"status": 1, "amount": 1}


def test_find_duplicates(datasets):
    orders, _ = datasets
    result = find_duplicates(orders, ["order_id"])
    assert result["duplicate_row_count"] == 2
    assert result["duplicate_percentage"] == 50.0
    assert result["sample_duplicate_identifiers"] == [{"order_id": "O3"}]
    assert result["analysis_type"] == "repeated_key"


def test_repeated_relationship_keys_are_distinct_from_exact_duplicates(tmp_path):
    payments = tmp_path / "payments.csv"
    pd.DataFrame(
        {
            "record_id": ["A", "A", "B"],
            "sequence": [1, 2, 1],
            "value": [10.0, 5.0, 20.0],
        }
    ).to_csv(payments, index=False)

    repeated_keys = find_duplicates(payments, ["record_id"])
    exact_rows = find_duplicates(payments)

    assert repeated_keys["analysis_type"] == "repeated_key"
    assert repeated_keys["duplicate_row_count"] == 2
    assert "not automatically" in repeated_keys["interpretation"]
    assert exact_rows["analysis_type"] == "exact_row"
    assert exact_rows["duplicate_row_count"] == 0


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


def test_segment_analysis_accepts_plain_language_date_frequency(datasets):
    orders, _ = datasets
    result = segment_analysis(
        orders,
        group_by_column=None,
        date_column="order_date",
        date_frequency="day",
    )
    assert result["result_count"] == 3


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
