import numpy as np


def compute_business_cost(y_true, probs, threshold, fn_cost=5000, fp_cost=200):
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    cost = fn * fn_cost + fp * fp_cost
    savings = tp * fn_cost  # fraud caught = money saved

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "estimated_cost_usd": float(cost),
        "estimated_savings_usd": float(savings),
        "net_impact_usd": float(savings - cost),
    }
