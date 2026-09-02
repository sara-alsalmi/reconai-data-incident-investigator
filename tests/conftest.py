from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def datasets(tmp_path):
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O3"],
            "order_date": ["2024-08-23", "2024-08-24", "2024-08-25", "2024-08-25"],
            "channel": ["Web", "Mobile", "Mobile", "Mobile"],
            "amount": [100.0, 200.0, 300.0, 300.0],
        }
    )
    payments = pd.DataFrame(
        {
            "payment_id": ["P1", "P2", "P3", "P4"],
            "order_id": ["O1", "O2", "O4", "O5"],
            "payment_date": ["2024-08-23", "2024-08-24", "2024-08-25", "2024-08-26"],
            "channel": ["Web", "Mobile", "Mobile", "Partner"],
            "amount": [100.0, 200.0, 400.0, 500.0],
        }
    )
    order_path = tmp_path / "orders.csv"
    payment_path = tmp_path / "payments.csv"
    orders.to_csv(order_path, index=False)
    payments.to_csv(payment_path, index=False)
    return str(order_path), str(payment_path)

