import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "enterprise_fraud_dataset.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "features.pkl")
MEAN_PATH = os.path.join(MODEL_DIR, "means.pkl")
TRAIN_DIST_PATH = os.path.join(MODEL_DIR, "train_dist.pkl")  # for drift detection

PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "outputs", "predictions_audit.csv")
