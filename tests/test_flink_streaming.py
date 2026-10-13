"""Unit tests for Flink transaction normalization and window features."""
import json

import pytest

from src.streaming.flink_fraud_processor import (
    compute_window_features,
    parse_transaction,
)


def test_parse_transaction_normalizes_event_time_and_optional_device():
    event = parse_transaction(json.dumps({
        "transaction_id": "txn-1",
        "customer_id": "customer-7",
        "transaction_amount": "12.50",
        "event_timestamp": "2026-01-01T00:00:00Z",
    }))

    assert event["transaction_id"] == "txn-1"
    assert event["customer_id"] == "customer-7"
    assert event["transaction_amount"] == 12.5
    assert event["device_id"] is None
    assert event["event_time_ms"] == 1767225600000


@pytest.mark.parametrize("missing", ["transaction_id", "customer_id", "event_timestamp"])
def test_parse_transaction_rejects_missing_required_fields(missing):
    event = {
        "transaction_id": "txn-1", "customer_id": "customer-7",
        "transaction_amount": 1.0, "event_timestamp": "2026-01-01T00:00:00Z",
    }
    del event[missing]
    with pytest.raises(ValueError):
        parse_transaction(json.dumps(event))


def test_compute_window_features_counts_velocity_amount_and_device_switches():
    events = [
        {"transaction_id": "b", "event_time_ms": 2000,
         "transaction_amount": 10.0, "device_id": "device-b"},
        {"transaction_id": "a", "event_time_ms": 1000,
         "transaction_amount": 2.25, "device_id": "device-a"},
        {"transaction_id": "c", "event_time_ms": 3000,
         "transaction_amount": 3.5, "device_id": "device-b"},
    ]

    features = compute_window_features(events, 10000)

    assert features == {
        "transaction_velocity_10s": 3,
        "rolling_amount_10s": 15.75,
        "device_switching_frequency_10s": 2,
        "window_end_ms": 10000,
    }


def test_parse_transaction_rejects_negative_amount():
    with pytest.raises(ValueError, match="cannot be negative"):
        parse_transaction(json.dumps({
            "transaction_id": "txn-1", "customer_id": "customer-7",
            "transaction_amount": -1, "event_timestamp": "2026-01-01T00:00:00Z",
        }))


def test_window_features_are_independent_of_input_order():
    events = [
        {"transaction_id": "t2", "event_time_ms": 2000,
         "transaction_amount": 4.0, "device_id": "b"},
        {"transaction_id": "t1", "event_time_ms": 1000,
         "transaction_amount": 3.0, "device_id": "a"},
    ]

    assert compute_window_features(events, 3000) == compute_window_features(
        reversed(events), 3000
    )
