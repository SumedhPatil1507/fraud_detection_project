import pytest
from src.validation import validate_transaction, REQUIRED_FIELDS


def test_valid_transaction():
    txn = {"transaction_amount": 100.0, "distance_from_home_km": 10.0, "hour": 12}
    result = validate_transaction(txn)
    assert result.valid is True
    assert len(result.errors) == 0


def test_missing_required_field():
    txn = {"distance_from_home_km": 10.0}
    result = validate_transaction(txn)
    assert result.valid is False
    assert any("transaction_amount" in e for e in result.errors)


def test_out_of_range_hour():
    txn = {"transaction_amount": 100.0, "distance_from_home_km": 10.0, "hour": 25}
    result = validate_transaction(txn)
    assert result.valid is False
    assert any("hour" in e for e in result.errors)


def test_negative_amount():
    txn = {"transaction_amount": -50.0, "distance_from_home_km": 10.0}
    result = validate_transaction(txn)
    assert result.valid is False


def test_business_rule_warning():
    txn = {"transaction_amount": 60000.0, "distance_from_home_km": 0.5}
    result = validate_transaction(txn)
    assert len(result.warnings) > 0
