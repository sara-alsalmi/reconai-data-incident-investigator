"""Generate deterministic ReconAI demo data with one hidden reconciliation incident."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


SEED = 20240824
ORDER_COUNT = 6000
CUSTOMER_COUNT = 1400


def generate(output_dir: Path | None = None) -> dict[str, Path]:
    random.seed(SEED)
    target = output_dir or Path(__file__).resolve().parents[1] / "sample_data"
    target.mkdir(parents=True, exist_ok=True)

    regions = ["North", "South", "East", "West"]
    segments = ["SMB", "Mid-Market", "Enterprise"]
    channels = ["Web", "Mobile", "Partner"]
    methods = ["Card", "Wallet", "Bank Transfer"]

    customers = []
    for index in range(1, CUSTOMER_COUNT + 1):
        customers.append(
            {
                "customer_id": f"C{index:05d}",
                "region": random.choice(regions),
                "customer_segment": random.choices(segments, weights=[55, 30, 15])[0],
                "signup_date": (
                    date(2023, 1, 1) + timedelta(days=random.randrange(550))
                ).isoformat(),
            }
        )
    customer_by_id = {row["customer_id"]: row for row in customers}

    orders = []
    payments = []
    start = date(2024, 7, 1)
    for index in range(1, ORDER_COUNT + 1):
        customer_id = f"C{random.randint(1, CUSTOMER_COUNT):05d}"
        order_date = start + timedelta(days=random.randrange(77))
        channel = random.choices(channels, weights=[48, 42, 10])[0]
        amount = round(random.uniform(18, 1200), 2)
        order_id = f"O{index:07d}"
        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": order_date.isoformat(),
                "channel": channel,
                "region": customer_by_id[customer_id]["region"],
                "amount": amount,
                "status": random.choices(["Completed", "Refunded"], weights=[97, 3])[0],
            }
        )
        payments.append(
            {
                "payment_id": f"P{index:07d}",
                "order_id": order_id,
                "payment_date": (order_date + timedelta(days=random.choice([0, 0, 0, 1]))).isoformat(),
                "payment_method": random.choice(methods),
                "amount": amount,
                "payment_status": "Captured",
                "channel": channel,
            }
        )

    incident_start = date(2024, 8, 24)
    removed_ids = {
        row["order_id"]
        for row in orders
        if row["channel"] == "Mobile"
        and date.fromisoformat(row["order_date"]) >= incident_start
        and random.random() < 0.72
    }
    visible_orders = [row for row in orders if row["order_id"] not in removed_ids]

    paths = {
        "orders": target / "orders.csv",
        "payments": target / "payments.csv",
        "customers": target / "customers.csv",
        "ground_truth": target / "ground_truth.json",
    }
    pd.DataFrame(visible_orders).to_csv(paths["orders"], index=False)
    pd.DataFrame(payments).to_csv(paths["payments"], index=False)
    pd.DataFrame(customers).to_csv(paths["customers"], index=False)
    paths["ground_truth"].write_text(
        json.dumps(
            {
                "incident": "Missing order records",
                "primary_affected_segment": "Mobile",
                "approximate_start": "2024-08-24",
                "observable_symptom": "Payments exist without matching orders",
                "removed_order_count": len(removed_ids),
                "removed_order_ids": sorted(removed_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


if __name__ == "__main__":
    generated = generate()
    print(f"Generated demo data in {generated['orders'].parent}")

