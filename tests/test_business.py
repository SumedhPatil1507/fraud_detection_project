import numpy as np
import pytest
from src.business import compute_business_cost


@pytest.fixture
def perfect_preds():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    return y, probs


@pytest.fixture
def all_wrong():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    return y, probs


def test_perfect_predictions_zero_cost(perfect_preds):
    y, probs = perfect_preds
    result = compute_business_cost(y, probs, threshold=0.5)
    assert result["estimated_cost_usd"] == 0.0
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0


def test_all_wrong_max_cost(all_wrong):
    y, probs = all_wrong
    result = compute_business_cost(y, probs, threshold=0.5)
    assert result["false_negative"] == 2
    assert result["false_positive"] == 2
    assert result["estimated_cost_usd"] == 2 * 5000 + 2 * 200


def test_custom_costs():
    y = np.array([1])
    probs = np.array([0.1])
    result = compute_business_cost(y, probs, threshold=0.5, fn_cost=1000, fp_cost=100)
    assert result["estimated_cost_usd"] == 1000.0


def test_net_impact_positive_when_savings_exceed_cost(perfect_preds):
    y, probs = perfect_preds
    result = compute_business_cost(y, probs, threshold=0.5)
    assert result["net_impact_usd"] >= 0


def test_precision_recall_range(perfect_preds):
    y, probs = perfect_preds
    result = compute_business_cost(y, probs, threshold=0.5)
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
