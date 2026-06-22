import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH   = os.path.join(BASE_DIR, "data", "enterprise_fraud_dataset.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_data.csv")

MODEL_DIR        = os.path.join(BASE_DIR, "outputs", "model")
MODEL_PATH       = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_PATH     = os.path.join(MODEL_DIR, "features.pkl")
MEAN_PATH        = os.path.join(MODEL_DIR, "means.pkl")
TRAIN_DIST_PATH  = os.path.join(MODEL_DIR, "train_dist.pkl")

PLOT_DIR       = os.path.join(BASE_DIR, "outputs", "plots")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "outputs", "predictions_audit.csv")

# ── Infrastructure config (read from env, fall back to local dev defaults) ──
DATABASE_URL      = os.environ.get("DATABASE_URL", "")          # asyncpg PostgreSQL
REDIS_URL         = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
AWS_S3_BUCKET     = os.environ.get("AWS_S3_BUCKET", "")
MINIO_ENDPOINT    = os.environ.get("MINIO_ENDPOINT", "")
NEO4J_URI         = os.environ.get("NEO4J_URI", "")
NEPTUNE_ENDPOINT  = os.environ.get("NEPTUNE_ENDPOINT", "")
PROMETHEUS_PORT   = int(os.environ.get("PROMETHEUS_PORT", "9090"))
