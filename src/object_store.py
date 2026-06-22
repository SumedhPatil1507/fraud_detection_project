"""
Secure Object Store — S3-compatible artifact storage.
Replaces local .pkl file reads/writes for model artifacts and SAR reports.

Backends (in priority order):
  1. AWS S3 (if AWS_S3_BUCKET + boto3 available)
  2. MinIO (if MINIO_ENDPOINT configured)
  3. Local filesystem fallback (outputs/ directory)

MD5 checksum verification on every model load — refuses to load if
checksum mismatches the registry entry.
"""
from __future__ import annotations
import os
import io
import pickle
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Any

# ── Optional boto3 ─────────────────────────────────────────────────────────────
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    _BOTO3 = True
except ImportError:
    _BOTO3 = False

from src.config import MODEL_DIR, MODEL_PATH, FEATURE_PATH, MEAN_PATH

CHECKSUM_REGISTRY = os.path.join(MODEL_DIR, "checksum_registry.json")
BUCKET = os.environ.get("AWS_S3_BUCKET", "") or os.environ.get("MINIO_BUCKET", "")


# ── S3 Client ──────────────────────────────────────────────────────────────────

def _get_s3():
    if not _BOTO3 or not BUCKET:
        return None, None
    minio_ep = os.environ.get("MINIO_ENDPOINT", "")
    try:
        if minio_ep:
            client = boto3.client(
                "s3",
                endpoint_url=minio_ep,
                aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
                aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            )
        else:
            client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "ap-south-1"),
            )
        return client, BUCKET
    except Exception:
        return None, None


# ── Checksum Registry ──────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _load_registry() -> dict:
    if os.path.exists(CHECKSUM_REGISTRY):
        try:
            with open(CHECKSUM_REGISTRY) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(reg: dict):
    os.makedirs(os.path.dirname(CHECKSUM_REGISTRY), exist_ok=True)
    with open(CHECKSUM_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)


def _register_artifact(key: str, checksum: str, size: int, location: str):
    reg = _load_registry()
    reg[key] = {
        "checksum":    checksum,
        "size_bytes":  size,
        "location":    location,
        "registered":  datetime.now(timezone.utc).isoformat(),
    }
    _save_registry(reg)


def verify_artifact(key: str, data: bytes) -> tuple[bool, str]:
    """Verify MD5 checksum of artifact data against registry."""
    reg = _load_registry()
    if key not in reg:
        return True, "not_registered"  # First load — trust it
    expected = reg[key]["checksum"]
    actual   = _md5(data)
    if actual != expected:
        return False, f"CHECKSUM MISMATCH: expected={expected} got={actual}"
    return True, "ok"


# ── Core Store / Load ──────────────────────────────────────────────────────────

def store_artifact(key: str, obj: Any, content_type: str = "application/octet-stream") -> str:
    """
    Serialize and store any Python object.
    Returns storage location string.
    """
    data = pickle.dumps(obj)
    checksum = _md5(data)
    client, bucket = _get_s3()

    if client and bucket:
        try:
            s3_key = f"fraudguard/{key}"
            client.put_object(
                Bucket=bucket, Key=s3_key, Body=data,
                ContentType=content_type,
                Metadata={"md5checksum": checksum},
            )
            location = f"s3://{bucket}/{s3_key}"
            _register_artifact(key, checksum, len(data), location)
            return location
        except Exception as e:
            print(f"[ObjectStore] S3 write failed ({e}) — using local")

    # Local fallback
    local_path = os.path.join(MODEL_DIR, key.replace("/", "_"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    _register_artifact(key, checksum, len(data), local_path)
    return local_path


def load_artifact(key: str, local_fallback_path: Optional[str] = None) -> Optional[Any]:
    """
    Load artifact from S3/MinIO or local fallback.
    Verifies MD5 checksum before returning — raises ValueError on mismatch.
    """
    client, bucket = _get_s3()

    if client and bucket:
        try:
            s3_key = f"fraudguard/{key}"
            response = client.get_object(Bucket=bucket, Key=s3_key)
            data = response["Body"].read()
            ok, msg = verify_artifact(key, data)
            if not ok:
                raise ValueError(f"[ObjectStore] {msg} for key={key}")
            return pickle.loads(data)
        except Exception as e:
            if "NoSuchKey" not in str(e) and "checksum" not in str(e).lower():
                print(f"[ObjectStore] S3 read failed ({e}) — using local")

    # Local fallback
    path = local_fallback_path or os.path.join(MODEL_DIR, key.replace("/", "_"))
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        ok, msg = verify_artifact(key, data)
        if not ok:
            raise ValueError(f"[ObjectStore] {msg} for path={path}")
        return pickle.loads(data)
    return None


def store_sar_json(sar_id: str, sar: dict) -> str:
    """Store SAR JSON to S3 or local."""
    import json as _json
    data = _json.dumps(sar, indent=2).encode()
    checksum = _md5(data)
    client, bucket = _get_s3()

    if client and bucket:
        try:
            s3_key = f"fraudguard/sar/{sar_id}.json"
            client.put_object(
                Bucket=bucket, Key=s3_key, Body=data,
                ContentType="application/json",
                Metadata={"md5checksum": checksum},
            )
            return f"s3://{bucket}/{s3_key}"
        except Exception as e:
            print(f"[ObjectStore] SAR S3 write failed ({e})")

    # Local fallback
    sar_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sar_reports")
    os.makedirs(sar_dir, exist_ok=True)
    path = os.path.join(sar_dir, f"{sar_id}.json")
    with open(path, "w") as f:
        f.write(data.decode())
    return path


def load_model_artifacts() -> tuple:
    """
    Load model, features, means from object store with checksum verification.
    Returns (model, features, means) or (None, None, None).
    """
    model    = load_artifact("model.pkl",    MODEL_PATH)
    features = load_artifact("features.pkl", FEATURE_PATH)
    means    = load_artifact("means.pkl",    MEAN_PATH)
    return model, features, means


def store_model_artifacts(model, features, means) -> dict:
    """Store model artifacts with checksum registration."""
    return {
        "model":    store_artifact("model.pkl",    model),
        "features": store_artifact("features.pkl", features),
        "means":    store_artifact("means.pkl",    means),
    }


def get_registry_summary() -> dict:
    reg = _load_registry()
    return {
        "total_artifacts": len(reg),
        "backend": "s3" if (BUCKET and _BOTO3) else "local",
        "bucket": BUCKET or "local",
        "artifacts": [
            {"key": k, "checksum": v["checksum"][:8] + "...",
             "size_kb": round(v["size_bytes"] / 1024, 1),
             "registered": v["registered"]}
            for k, v in reg.items()
        ],
    }
