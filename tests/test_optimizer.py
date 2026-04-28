import pytest
import numpy as np
from src.optimizer import optimize_threshold, roi_projection


def test_optimize_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    result = optimize_threshold(y_true, probs, fn_cost=5000, fp_cost=200, step=0.1)
    assert "optimal_threshold" in result
    assert 0.0 < result["optimal_threshold"] < 1.0
    assert "max_net_impact_usd" in result


def test_roi_projection():
    result = roi_projection(
        daily_transactions=10000,
        fraud_rate=0.02,
        avg_fraud_amount=2000,
        detection_rate=0.85,
        fp_rate=0.01,
        fp_cost=200,
        months=12,
    )
    assert result["monthly_savings_usd"] > 0
    assert result["annual_net_usd"] > 0
    assert result["daily_fraud_caught"] > 0
