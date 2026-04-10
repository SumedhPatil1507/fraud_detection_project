import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.ensemble import StackingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except Exception:
    _OPTUNA_AVAILABLE = False

try:
    import mlflow
    import mlflow.xgboost
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH, TRAIN_DIST_PATH, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)


def _optuna_tune(X_train, y_train, n_trials=15):
    """Tune XGBoost hyperparameters with Optuna."""
    import optuna
    # Use a subsample for tuning speed — full data used for final fit
    sample_size = min(5000, len(X_train))
    idx = np.random.default_rng(42).choice(len(X_train), sample_size, replace=False)
    X_s = X_train.iloc[idx]
    y_s = y_train.iloc[idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    neg, pos = (y_s == 0).sum(), (y_s == 1).sum()
    scale = neg / pos if pos > 0 else 1

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
            scale_pos_weight=scale,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_s, y_s, cv=cv,
                                 scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _build_ensemble(best_params, scale):
    """Stacking ensemble: XGBoost + LightGBM → Logistic Regression meta."""
    xgb = XGBClassifier(**best_params, scale_pos_weight=scale,
                        eval_metric='logloss', random_state=42, n_jobs=-1)
    estimators = [("xgb", xgb)]
    if _LGBM_AVAILABLE:
        lgbm = LGBMClassifier(
            n_estimators=100, learning_rate=0.1,
            scale_pos_weight=scale, random_state=42, verbose=-1, n_jobs=-1
        )
        estimators.append(("lgbm", lgbm))
    meta = LogisticRegression(max_iter=500, random_state=42)
    return StackingClassifier(estimators=estimators, final_estimator=meta,
                              cv=3, passthrough=False, n_jobs=-1)


def train_model(df, use_optuna=True, n_trials=15):
    y = df['label']
    X = df.drop(columns=['label', 'financial_loss'], errors='ignore')
    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    if use_optuna and _OPTUNA_AVAILABLE:
        best_params = _optuna_tune(X_train, y_train, n_trials=n_trials)
    else:
        best_params = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                           n_jobs=-1)

    # ── Stacking ensemble ──────────────────────────────────────────────────────
    ensemble = _build_ensemble(best_params, scale)

    # ── Cross-validation (3-fold for speed) ───────────────────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv,
                                scoring='roc_auc', n_jobs=-1)

    ensemble.fit(X_train, y_train)

    # ── Calibrate probabilities ────────────────────────────────────────────────
    calibrated = CalibratedClassifierCV(ensemble, cv="prefit", method="isotonic")
    calibrated.fit(X_test, y_test)

    probs = calibrated.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = float(thresholds[np.argmax(f1)])
    roc = roc_auc_score(y_test, probs)

    report = classification_report(
        y_test, (probs >= best_threshold).astype(int),
        target_names=["Legit", "Fraud"], output_dict=True
    )

    # ── Anomaly detection layer (Isolation Forest) ─────────────────────────────
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    anomaly_scores = -iso.score_samples(X_test)  # higher = more anomalous

    # ── Save training distribution for drift detection ─────────────────────────
    train_dist = {col: {"mean": float(X_train[col].mean()),
                        "std": float(X_train[col].std() + 1e-10)}
                  for col in X_train.columns}
    pickle.dump(train_dist, open(TRAIN_DIST_PATH, "wb"))

    # ── MLflow logging ─────────────────────────────────────────────────────────
    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("fraud_detection")
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metric("roc_auc", roc)
                mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
                mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
                mlflow.log_metric("best_threshold", best_threshold)
                mlflow.log_metric("fraud_precision", report["Fraud"]["precision"])
                mlflow.log_metric("fraud_recall", report["Fraud"]["recall"])
        except Exception:
            pass

    pickle.dump(calibrated, open(MODEL_PATH, "wb"))
    pickle.dump(X.columns.tolist(), open(FEATURE_PATH, "wb"))
    pickle.dump(X.mean(), open(MEAN_PATH, "wb"))

    return (calibrated, X_test, y_test, probs, best_threshold,
            roc, report, cv_scores, anomaly_scores, best_params)
