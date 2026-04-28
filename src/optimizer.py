"""
Dynamic Cost-Benefit Optimizer
Finds the optimal decision threshold by maximizing net financial impact.
Also supports custom cost matrices and ROI projections.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def optimize_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    fn_cost: float = 5000,
    fp_cost: float = 200,
    step: float = 0.01,
) -> dict:
    """Sweep thresholds and return the one with best net impact."""
    thresholds = np.arange(0.01, 1.0, step)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        savings = tp * fn_cost
        cost    = fp * fp_cost + fn * fn_cost
        net     = savings - cost
        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1        = 2 * precision * recall / (precision + recall + 1e-10)
        results.append({
            "threshold": round(float(t), 3),
            "net_impact_usd": round(float(net), 2),
            "savings_usd": round(float(savings), 2),
            "cost_usd": round(float(cost), 2),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    df = pd.DataFrame(results)
    best_row = df.loc[df["net_impact_usd"].idxmax()]
    return {
        "optimal_threshold": float(best_row["threshold"]),
        "max_net_impact_usd": float(best_row["net_impact_usd"]),
        "at_optimal": best_row.to_dict(),
        "sweep_df": df,
    }


def roi_projection(
    daily_transactions: int,
    fraud_rate: float,
    avg_fraud_amount: float,
    detection_rate: float,
    fp_rate: float,
    fp_cost: float = 200,
    months: int = 12,
) -> dict:
    """Project monthly and annual ROI from deploying the model."""
    daily_fraud   = daily_transactions * fraud_rate
    daily_legit   = daily_transactions * (1 - fraud_rate)
    daily_caught  = daily_fraud * detection_rate
    daily_missed  = daily_fraud * (1 - detection_rate)
    daily_fp      = daily_legit * fp_rate

    monthly_savings = daily_caught * avg_fraud_amount * 30
    monthly_cost    = (daily_fp * fp_cost + daily_missed * avg_fraud_amount) * 30
    monthly_net     = monthly_savings - monthly_cost

    return {
        "monthly_savings_usd": round(monthly_savings, 2),
        "monthly_cost_usd": round(monthly_cost, 2),
        "monthly_net_usd": round(monthly_net, 2),
        "annual_net_usd": round(monthly_net * months, 2),
        "daily_fraud_caught": round(daily_caught, 1),
        "daily_false_positives": round(daily_fp, 1),
    }
