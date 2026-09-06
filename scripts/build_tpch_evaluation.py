"""Build reproducible ReconAI evaluation cases from the standard TPC-H schema."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tpch-evaluation"


def _write_case(
    root: Path,
    case_id: str,
    question: str,
    source: pd.DataFrame,
    warehouse: pd.DataFrame,
    expected: dict,
) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(case_dir / "source_orders.csv", index=False)
    warehouse.to_csv(case_dir / "warehouse_orders.csv", index=False)
    manifest = {
        "case_id": case_id,
        "benchmark": "TPC-H",
        "question": question,
        "datasets": ["source_orders.csv", "warehouse_orders.csv"],
        "expected": expected,
    }
    (case_dir / "ground_truth.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build(output: Path, scale_factor: float = 0.01) -> list[str]:
    if output.exists():
        resolved = output.resolve()
        expected_parent = (PROJECT_ROOT / "data").resolve()
        if expected_parent not in resolved.parents:
            raise ValueError("Evaluation output must remain inside the project data directory")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(":memory:")
    connection.execute("INSTALL tpch")
    connection.execute("LOAD tpch")
    connection.execute("CALL dbgen(sf = ?)", [scale_factor])
    orders = connection.execute(
        """
        SELECT
            o_orderkey AS order_id,
            o_custkey AS customer_id,
            CAST(o_orderdate AS VARCHAR) AS order_date,
            o_orderstatus AS order_status,
            o_orderpriority AS order_priority,
            CAST(o_totalprice AS DOUBLE) AS total_amount,
            o_clerk AS clerk
        FROM orders
        ORDER BY o_orderkey
        """
    ).df()
    connection.close()

    _write_case(
        output,
        "clean_control",
        "Do the source and warehouse order datasets disagree?",
        orders,
        orders.copy(),
        {
            "outcome": "no_discrepancy",
            "required_terms": ["no discrepancy", "match", "zero"],
            "forbidden_hypothesis_terms": ["pipeline", "etl", "software", "ingestion failure"],
            "checks": [
                {"tool": "compare_aggregates", "field": "absolute_difference", "value": 0.0},
                {"tool": "find_unmatched_records", "field": "unmatched_count", "value": 0},
                {"tool": "compare_rows_by_key", "field": "record_mismatch_count", "value": 0},
            ],
        },
    )

    missing_candidates = orders.loc[orders["order_priority"] == "1-URGENT"].sort_values(
        ["order_date", "order_id"]
    )
    missing = missing_candidates.tail(60)
    missing_ids = set(missing["order_id"].tolist())
    missing_warehouse = orders.loc[~orders["order_id"].isin(missing_ids)].copy()
    _write_case(
        output,
        "missing_urgent_orders",
        "Why does the warehouse contain fewer orders than the source, and where is the gap concentrated?",
        orders,
        missing_warehouse,
        {
            "outcome": "missing_records",
            "required_terms": ["missing", "unmatched", "absent"],
            "forbidden_hypothesis_terms": [
                "pipeline",
                "etl",
                "software",
                "ingestion",
                "loading issue",
            ],
            "affected_record_count": len(missing),
            "affected_amount": round(float(missing["total_amount"].sum()), 2),
            "affected_segment": "1-URGENT",
            "affected_period": {
                "start": str(missing["order_date"].min()),
                "end": str(missing["order_date"].max()),
            },
            "checks": [
                {
                    "tool": "find_unmatched_records",
                    "field": "unmatched_count",
                    "value": len(missing),
                    "dataset_a": "source_orders.csv",
                }
            ],
        },
    )

    duplicate_seed = orders.loc[orders["order_status"] == "F"].head(40).copy()
    duplicate_warehouse = pd.concat([orders, duplicate_seed], ignore_index=True)
    _write_case(
        output,
        "duplicate_fulfilled_orders",
        "Why are warehouse order counts and totals higher than the source?",
        orders,
        duplicate_warehouse,
        {
            "outcome": "duplicate_records",
            "required_terms": ["duplicate", "duplicated"],
            "forbidden_hypothesis_terms": ["pipeline", "etl", "software", "ingestion"],
            "extra_record_count": len(duplicate_seed),
            "duplicate_record_count_including_originals": len(duplicate_seed) * 2,
            "excess_amount": round(float(duplicate_seed["total_amount"].sum()), 2),
            "affected_segment": "F",
            "checks": [
                {
                    "tool": "find_duplicates",
                    "field": "duplicate_row_count",
                    "value": len(duplicate_seed) * 2,
                    "dataset": "warehouse_orders.csv",
                }
            ],
        },
    )

    drift_candidates = orders.loc[orders["order_priority"] == "2-HIGH"].tail(50)
    drift_ids = set(drift_candidates["order_id"].tolist())
    drift_warehouse = orders.copy()
    drift_warehouse.loc[
        drift_warehouse["order_id"].isin(drift_ids), "total_amount"
    ] += 17.25
    _write_case(
        output,
        "high_priority_amount_drift",
        "Why do source and warehouse order totals differ even though their order IDs match?",
        orders,
        drift_warehouse,
        {
            "outcome": "value_mismatch",
            "required_terms": ["amount", "value", "mismatch", "differ"],
            "forbidden_hypothesis_terms": ["pipeline", "etl", "software", "ingestion"],
            "affected_record_count": len(drift_candidates),
            "difference_source_minus_warehouse": round(-17.25 * len(drift_candidates), 2),
            "affected_segment": "2-HIGH",
            "checks": [
                {
                    "tool": "compare_record_values",
                    "field": "value_mismatch_count",
                    "value": len(drift_candidates),
                },
                {
                    "tool": "compare_record_values",
                    "field": "signed_difference_a_minus_b",
                    "value": round(-17.25 * len(drift_candidates), 2),
                },
            ],
        },
    )

    offset_missing = missing_candidates.head(40).copy()
    offset_missing_ids = set(offset_missing["order_id"].tolist())
    offset_duplicate_seed = (
        orders.loc[
            (orders["order_status"] == "F")
            & (~orders["order_id"].isin(offset_missing_ids))
        ]
        .head(40)
        .copy()
    )
    offset_warehouse = orders.loc[
        ~orders["order_id"].isin(offset_missing_ids)
    ].copy()
    offset_warehouse = pd.concat(
        [offset_warehouse, offset_duplicate_seed], ignore_index=True
    )
    offset_value_difference = round(
        float(offset_missing["total_amount"].sum())
        - float(offset_duplicate_seed["total_amount"].sum()),
        2,
    )
    _write_case(
        output,
        "agentic_offsetting_missing_duplicates",
        "The row counts match, but the files do not. Identify every record-level cause and calculate the net source-minus-warehouse total_amount difference.",
        orders,
        offset_warehouse,
        {
            "requires_agent": True,
            "outcome": "multiple_causes",
            "required_term_groups": [
                ["missing", "absent", "unmatched"],
                ["duplicate", "repeated key"],
            ],
            "semantic_scope": "hypothesis",
            "required_verdict": "ACCEPT",
            "forbidden_hypothesis_terms": ["pipeline", "etl", "software", "ingestion"],
            "min_attempts": 1,
            "min_agent_tool_calls": 1,
            "required_agent_tools": {"compare_aggregates": 1},
            "missing_record_count": len(offset_missing),
            "duplicate_record_count_including_originals": len(offset_duplicate_seed) * 2,
            "difference_source_minus_warehouse": offset_value_difference,
            "checks": [
                {
                    "tool": "compare_aggregates",
                    "field": "absolute_difference",
                    "value": 0.0,
                    "details": {"aggregation": "count"},
                },
                {
                    "tool": "find_unmatched_records",
                    "field": "unmatched_count",
                    "value": len(offset_missing),
                    "dataset_a": "source_orders.csv",
                },
                {
                    "tool": "find_duplicates",
                    "field": "duplicate_row_count",
                    "value": len(offset_duplicate_seed) * 2,
                    "dataset": "warehouse_orders.csv",
                },
                {
                    "tool": "compare_aggregates",
                    "field": "signed_difference_a_minus_b",
                    "value": offset_value_difference,
                    "attempt_min": 1,
                    "details": {
                        "aggregation": "sum",
                        "metric_column_a": "total_amount",
                        "metric_column_b": "total_amount",
                    },
                },
            ],
        },
    )

    localized_drift = orders.loc[orders["order_priority"] == "2-HIGH"].head(45)
    localized_drift_ids = set(localized_drift["order_id"].tolist())
    localized_warehouse = orders.copy()
    localized_warehouse.loc[
        localized_warehouse["order_id"].isin(localized_drift_ids), "total_amount"
    ] += 23.5
    _write_case(
        output,
        "agentic_priority_localization",
        "Counts and order IDs match, but total_amount differs. Which order_priority contains the discrepancy?",
        orders,
        localized_warehouse,
        {
            "requires_agent": True,
            "outcome": "localized_value_mismatch",
            "required_all_terms": ["2-high"],
            "semantic_scope": "hypothesis",
            "required_verdict": "ACCEPT",
            "forbidden_hypothesis_terms": ["pipeline", "etl", "software", "ingestion"],
            "min_attempts": 1,
            "min_agent_tool_calls": 2,
            "required_agent_tools": {"segment_analysis": 2},
            "affected_record_count": len(localized_drift),
            "affected_segment": "2-HIGH",
            "difference_source_minus_warehouse": round(-23.5 * len(localized_drift), 2),
            "checks": [
                {
                    "tool": "compare_record_values",
                    "field": "value_mismatch_count",
                    "value": len(localized_drift),
                },
                {
                    "tool": "segment_analysis",
                    "field": "filtered_record_count",
                    "value": len(orders),
                    "dataset": "source_orders.csv",
                    "attempt_min": 1,
                    "details": {
                        "grouping": ["order_priority"],
                        "metric_column": "total_amount",
                        "aggregation": "sum",
                    },
                },
                {
                    "tool": "segment_analysis",
                    "field": "filtered_record_count",
                    "value": len(orders),
                    "dataset": "warehouse_orders.csv",
                    "attempt_min": 1,
                    "details": {
                        "grouping": ["order_priority"],
                        "metric_column": "total_amount",
                        "aggregation": "sum",
                    },
                },
            ],
        },
    )
    return sorted(path.name for path in output.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale-factor", type=float, default=0.01)
    args = parser.parse_args()
    cases = build(args.output, args.scale_factor)
    print(f"Built {len(cases)} TPC-H evaluation cases in {args.output}")
    for case in cases:
        print(f"- {case}")


if __name__ == "__main__":
    main()
