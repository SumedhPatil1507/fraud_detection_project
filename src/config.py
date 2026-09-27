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

# ── Fraud Investigation Copilot ─────────────────────────────────────────────
# Vector store (ChromaDB) — persisted on local disk by default
CHROMA_PERSIST_DIR   = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(BASE_DIR, "outputs", "chroma_db"),
)
CHROMA_AUDIT_COLLECTION  = os.environ.get("CHROMA_AUDIT_COLLECTION",  "audit_log")
CHROMA_SAR_COLLECTION    = os.environ.get("CHROMA_SAR_COLLECTION",    "sar_reports")

# Embedding model (sentence-transformers, runs locally — no API key needed)
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# LLM used by the copilot writer agent (reuses existing GROQ_API_KEY)
COPILOT_LLM_MODEL = os.environ.get("COPILOT_LLM_MODEL", "llama3-8b-8192")

# How many historical cases the retriever surfaces per query
COPILOT_TOP_K = int(os.environ.get("COPILOT_TOP_K", "5"))

# Minimum fraud probability for a transaction to enter the copilot pipeline
COPILOT_MIN_FRAUD_PROB = float(os.environ.get("COPILOT_MIN_FRAUD_PROB", "0.5"))
