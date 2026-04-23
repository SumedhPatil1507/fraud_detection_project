import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.ensemble import IsolationForest
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

# Fast defaults for Streamlit Cloud (weak CPU)
_FAST_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    n_jobs=1,  # single thread — more stable on cloud
)


def _optuna_tune(X_train, y_train, n_trials=10):
    import optuna
    sample_size = min(3000, len(X_train))
    idx = np.random.default_rng(42).choice(len(X_train), sample_size, replace=False)
    X_s, y_s = X_train.iloc[idx], y_train.iloc[idx]
    neg, pos = (y_s == 0).sum(), (y_s == 1).sum()
    scale = neg / pos if pos > 0 else 1
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 150),
            max_depth=trial.suggest_int("max_depth", 3, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            scale_pos_weight=scale, eval_metric='logloss',
            random_state=42, n_jobs=1,
        )
        return cross_val_score(XGBClassifier(**params), X_s, y_s,
                               cv=cv, scoring='roc_auc').mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_model(df, use_optuna=False, n_trials=10):
    y = df['label']
    X = df.drop(columns=['label', 'financial_loss'], errors='ignore')
    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1

    # ── Hyperparameters ────────────────────────────────────────────────────────
    if use_optuna and _OPTUNA_AVAILABLE:
        best_params = _optuna_tune(X_train, y_train, n_trials=n_trials)
    else:
        best_params = _FAST_PARAMS.copy()

    # ── Single XGBoost (fast) + optional LightGBM soft-vote ───────────────────
    xgb = XGBClassifier(**best_params, scale_pos_weight=scale,
                        eval_metric='logloss', random_state=42)

    if _LGBM_AVAILABLE:
        from sklearn.ensemble import VotingClassifier
        lgbm = LGBMClassifier(n_estimators=80, learning_rate=0.1,
                              scale_pos_weight=scale, random_state=42,
                              verbose=-1, n_jobs=1)
        model = VotingClassifier(
            estimators=[("xgb", xgb), ("lgbm", lgbm)],
            voting='soft', n_jobs=1
        )
    else:
        model = xgb

    # ── 3-fold CV on a subsample for speed ────────────────────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_sample = min(5000, len(X_train))
    idx = np.random.default_rng(0).choice(len(X_train), cv_sample, replace=False)
    cv_scores = cross_val_score(model, X_train.iloc[idx], y_train.iloc[idx],
                                cv=cv, scoring='roc_auc')

    model.fit(X_train, y_train)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    try:
        calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        calibrated.fit(X_test, y_test)
        final_model = calibrated
    except Exception:
        # Fallback if calibration fails — use model directly
        final_model = model

    probs = final_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = float(thresholds[np.argmax(f1)])
    roc = roc_auc_score(y_test, probs)

    report = classification_report(
        y_test, (probs >= best_threshold).astype(int),
        target_names=["Legit", "Fraud"], output_dict=True
    )

    # ── Isolation Forest on subsample ─────────────────────────────────────────
    iso_sample = min(3000, len(X_train))
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train.iloc[:iso_sample])
    anomaly_scores = -iso.score_samples(X_test)

    # ── Save artifacts with versioning ────────────────────────────────────────
    from datetime import datetime
    import glob
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")

    train_dist = {col: {"mean": float(X_train[col].mean()),
                        "std": float(X_train[col].std() + 1e-10)}
                  for col in X_train.columns}
    pickle.dump(train_dist, open(TRAIN_DIST_PATH, "wb"))
    pickle.dump(final_model, open(MODEL_PATH, "wb"))
    pickle.dump(final_model, open(versioned_path, "wb"))
    pickle.dump(X.columns.tolist(), open(FEATURE_PATH, "wb"))
    pickle.dump(X.mean(), open(MEAN_PATH, "wb"))

    # Keep only last 3 versioned models
    versioned = sorted(glob.glob(os.path.join(MODEL_DIR, "model_*.pkl")))
    for old in versioned[:-3]:
        try:
            os.remove(old)
        except Exception:
            pass

    return (final_model, X_test, y_test, probs, best_threshold,
            roc, report, cv_scores, anomaly_scores, best_params)
