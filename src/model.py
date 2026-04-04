import pickle
import os
import numpy as np
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from xgboost import XGBClassifier

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(df):
    y = df['label']
    X = df.drop(columns=['label', 'financial_loss'], errors='ignore')
    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1

    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=42,
    )
    model = XGBClassifier(**params)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring='roc_auc', n_jobs=-1)

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = float(thresholds[np.argmax(f1)])
    roc = roc_auc_score(y_test, probs)

    report = classification_report(
        y_test, (probs >= best_threshold).astype(int),
        target_names=["Legit", "Fraud"], output_dict=True
    )

    # MLflow logging
    try:
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", roc)
            mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
            mlflow.log_metric("best_threshold", best_threshold)
            mlflow.log_metric("fraud_precision", report["Fraud"]["precision"])
            mlflow.log_metric("fraud_recall", report["Fraud"]["recall"])
            mlflow.xgboost.log_model(model, "model")
    except Exception:
        pass  # MLflow optional — don't break training if unavailable

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(X.columns.tolist(), open(FEATURE_PATH, "wb"))
    pickle.dump(X.mean(), open(MEAN_PATH, "wb"))

    return model, X_test, y_test, probs, best_threshold, roc, report, cv_scores
