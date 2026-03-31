def compute_business_cost(y_true, probs, threshold):

    preds = (probs > threshold).astype(int)

    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    cost = fn * 5000 + fp * 200

    return {
        "false_positive": int(fp),
        "false_negative": int(fn),
        "estimated_cost": float(cost)
    }