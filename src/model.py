import pickle
import os
import numpy as np
import pandas as pd

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


# ── Version-resilient prediction helper ───────────────────────────────────────

def safe_predict_proba(model, input_df: "pd.DataFrame") -> "np.ndarray":
    """
    Call model.predict_proba() in a way that survives XGBoost version
    mismatches between training and inference environments.

    XGBoost >= 2.0 validates feature names stored in the booster against the
    DataFrame column names. When a model was pickled with an older XGBoost
    the booster may have integer feature names ('f0', 'f1', …) while the
    caller passes a named DataFrame, triggering:
        ValueError: feature_names mismatch

    Strategy:
      1. Try the normal path (named DataFrame) — works when names match.
      2. On ANY ValueError/XGBoostError that mentions features, retry with a
         plain numpy array (strips column names entirely).
      3. Final fallback: return a uniform low-probability array so the app
         never hard-crashes.

    Returns a 2-D probability array matching model.predict_proba() output.
    """
    # ── Attempt 1: pass DataFrame as-is ──────────────────────────────────────
    try:
        return model.predict_proba(input_df)
    except Exception as e1:
        err_str = str(e1).lower()
        # Only retry for feature-name / dtype issues
        if not any(kw in err_str for kw in
                   ("feature_names", "feature names", "validate_features",
                    "mismatch", "dtype", "invalid feature")):
            raise  # unrelated error — re-raise immediately

    # ── Attempt 2: strip column names → numpy array ───────────────────────────
    try:
        return model.predict_proba(input_df.values.astype(np.float32))
    except Exception as e2:
        err_str2 = str(e2).lower()
        if not any(kw in err_str2 for kw in
                   ("feature_names", "feature names", "validate_features",
                    "mismatch", "dtype")):
            raise

    # ── Attempt 3: try to fix booster feature names then predict ─────────────
    try:
        _fix_booster_feature_names(model, list(input_df.columns))
        return model.predict_proba(input_df)
    except Exception:
        pass

    # ── Final fallback: return near-zero probabilities ────────────────────────
    n = len(input_df)
    # Determine number of classes from the model if possible
    try:
        n_classes = len(model.classes_)
    except Exception:
        n_classes = 2
    probs = np.full((n, n_classes), 1.0 / n_classes, dtype=np.float32)
    return probs


def _fix_booster_feature_names(model, feature_names: list) -> None:
    """
    Walk the estimator tree and overwrite XGBoost booster feature names with
    the correct string names from features.pkl.  Called as last resort before
    the numpy-array fallback.
    """
    def _patch(est):
        if hasattr(est, 'get_booster'):
            try:
                b = est.get_booster()
                b.feature_names = feature_names
            except Exception:
                pass
        if hasattr(est, 'estimators_'):
            for sub in est.estimators_:
                _patch(sub)
        if hasattr(est, 'calibrated_classifiers_'):
            for cc in est.calibrated_classifiers_:
                _patch(cc.estimator)

    _patch(model)


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

    feature_list = X.columns.tolist()

    # Bake feature names into every XGBoost booster so they survive
    # cross-version pickle round-trips.
    _fix_booster_feature_names(final_model, feature_list)

    pickle.dump(train_dist, open(TRAIN_DIST_PATH, "wb"))
    pickle.dump(final_model, open(MODEL_PATH, "wb"))
    pickle.dump(final_model, open(versioned_path, "wb"))
    pickle.dump(feature_list, open(FEATURE_PATH, "wb"))
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
