import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(df):

    y = df['label']
    X = df.drop(columns=['label'], errors='ignore')

    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=10,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2*(precision*recall)/(precision+recall+1e-10)

    best_threshold = thresholds[np.argmax(f1)]

    print("ROC:", roc_auc_score(y_test, probs))

    # Save artifacts
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(X.columns.tolist(), open(FEATURE_PATH, "wb"))
    pickle.dump(X.mean(), open(MEAN_PATH, "wb"))

    return model, X_test, y_test, probs, best_threshold