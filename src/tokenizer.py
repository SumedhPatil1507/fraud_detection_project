"""
Cryptographic Key Rotation & Tokenization Engine
Implements:
  - Format-Preserving Tokenization (FPT) for PII fields
  - Key rotation with versioned key store
  - Vault API pattern (local mock + Supabase pgcrypto schema)
  - Role-gated decryption (only admin/analyst can detokenize)
"""
from __future__ import annotations
import os
import json
import hmac
import hashlib
import secrets
import base64
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
from cryptography.fernet import Fernet, MultiFernet

# ── Key Store ──────────────────────────────────────────────────────────────────
KEY_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "key_store.json"
)
TOKEN_VAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "token_vault.json"
)


def _load_key_store() -> dict:
    if os.path.exists(KEY_STORE_PATH):
        try:
            with open(KEY_STORE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"keys": [], "active_version": None}


def _save_key_store(store: dict):
    os.makedirs(os.path.dirname(KEY_STORE_PATH), exist_ok=True)
    with open(KEY_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


def _load_token_vault() -> dict:
    if os.path.exists(TOKEN_VAULT_PATH):
        try:
            with open(TOKEN_VAULT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_token_vault(vault: dict):
    os.makedirs(os.path.dirname(TOKEN_VAULT_PATH), exist_ok=True)
    with open(TOKEN_VAULT_PATH, "w") as f:
        json.dump(vault, f, indent=2)


# ── Key Lifecycle ──────────────────────────────────────────────────────────────

def generate_key(purpose: str = "encryption") -> dict:
    """Generate a new versioned Fernet key and add to key store."""
    store = _load_key_store()
    version = len(store["keys"]) + 1
    new_key = {
        "version":    version,
        "key":        Fernet.generate_key().decode(),
        "purpose":    purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rotated_at": None,
        "status":     "active",
    }
    # Mark previous active key as rotated
    for k in store["keys"]:
        if k["status"] == "active":
            k["status"] = "rotated"
            k["rotated_at"] = datetime.now(timezone.utc).isoformat()

    store["keys"].append(new_key)
    store["active_version"] = version
    _save_key_store(store)
    return new_key


def rotate_keys() -> dict:
    """Rotate encryption keys — generates new key, keeps old for decryption."""
    new_key = generate_key(purpose="rotation")
    return {
        "rotated_at": datetime.now(timezone.utc).isoformat(),
        "new_version": new_key["version"],
        "message": "Key rotated. Old keys retained for decryption of existing tokens.",
    }


def get_active_fernet() -> Fernet:
    """Return Fernet instance with active key. Falls back to env var."""
    store = _load_key_store()
    active = next(
        (k for k in store["keys"] if k["status"] == "active"), None
    )
    if active:
        return Fernet(active["key"].encode())

    # Fallback to env var
    env_key = os.environ.get("ENCRYPTION_KEY", "")
    if env_key:
        return Fernet(env_key.encode())

    # Auto-generate dev key
    new_key = generate_key(purpose="auto-init")
    return Fernet(new_key["key"].encode())


def get_multi_fernet() -> MultiFernet:
    """
    MultiFernet supports decryption with any key version.
    Encrypts with the active key, decrypts with any stored key.
    """
    store = _load_key_store()
    fernets = []

    # Active key first (used for encryption)
    for k in sorted(store["keys"], key=lambda x: x["version"], reverse=True):
        try:
            fernets.append(Fernet(k["key"].encode()))
        except Exception:
            pass

    if not fernets:
        fernets = [get_active_fernet()]

    return MultiFernet(fernets)


def get_key_store_summary() -> dict:
    store = _load_key_store()
    return {
        "total_keys": len(store["keys"]),
        "active_version": store.get("active_version"),
        "keys": [
            {
                "version": k["version"],
                "status": k["status"],
                "created_at": k["created_at"],
                "rotated_at": k["rotated_at"],
            }
            for k in store["keys"]
        ],
    }


# ── Tokenization Engine ────────────────────────────────────────────────────────

def _make_token(value: str, field: str) -> str:
    """
    Generate a deterministic, format-preserving token.
    Token = TKN_ + first 2 chars of field + HMAC-SHA256 truncated to 12 hex chars.
    """
    secret = os.environ.get("TOKEN_SECRET", "fraudguard-token-secret-dev")
    h = hmac.new(secret.encode(), f"{field}:{value}".encode(), hashlib.sha256)
    return f"TKN_{field[:2].upper()}_{h.hexdigest()[:12].upper()}"


def tokenize(value: str, field: str) -> str:
    """
    Tokenize a PII value. Stores encrypted original in vault.
    Returns a stable token that can be used in place of the real value.
    """
    if not value or value.startswith("TKN_"):
        return value

    token = _make_token(value, field)
    vault = _load_token_vault()

    if token not in vault:
        fernet = get_active_fernet()
        encrypted = fernet.encrypt(value.encode()).decode()
        vault[token] = {
            "encrypted_value": encrypted,
            "field": field,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_token_vault(vault)

    return token


def detokenize(token: str, role: str = "viewer") -> str:
    """
    Reverse tokenization — requires admin or analyst role.
    Returns original plaintext or '[REDACTED]' if unauthorized.
    """
    allowed_roles = {"admin", "analyst"}
    if role not in allowed_roles:
        return "[REDACTED — insufficient role]"

    if not token.startswith("TKN_"):
        return token

    vault = _load_token_vault()
    entry = vault.get(token)
    if not entry:
        return "[TOKEN_NOT_FOUND]"

    try:
        mf = get_multi_fernet()
        return mf.decrypt(entry["encrypted_value"].encode()).decode()
    except Exception:
        return "[DECRYPTION_FAILED]"


def tokenize_dataframe(df: pd.DataFrame,
                        fields: Optional[list] = None) -> pd.DataFrame:
    """Tokenize PII columns in a DataFrame."""
    if fields is None:
        fields = ["customer_id", "merchant_id", "device_id"]
    df = df.copy()
    for col in fields:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda v: tokenize(v, col)
            )
    return df


def detokenize_dataframe(df: pd.DataFrame, role: str = "viewer",
                          fields: Optional[list] = None) -> pd.DataFrame:
    """Detokenize PII columns — role-gated."""
    if fields is None:
        fields = ["customer_id", "merchant_id", "device_id"]
    df = df.copy()
    for col in fields:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda v: detokenize(v, role)
            )
    return df


# ── Supabase pgcrypto Schema (SQL) ─────────────────────────────────────────────

PGCRYPTO_SCHEMA_SQL = """
-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encryption key table (store hashed keys, never plaintext)
CREATE TABLE IF NOT EXISTS vault_keys (
    id          SERIAL PRIMARY KEY,
    version     INTEGER UNIQUE NOT NULL,
    key_hash    TEXT NOT NULL,          -- SHA-256 of the key (for audit)
    purpose     TEXT DEFAULT 'encryption',
    status      TEXT DEFAULT 'active',  -- active | rotated | revoked
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    rotated_at  TIMESTAMPTZ
);

-- Token vault table
CREATE TABLE IF NOT EXISTS token_vault (
    token           TEXT PRIMARY KEY,
    field_name      TEXT NOT NULL,
    encrypted_value BYTEA NOT NULL,     -- pgp_sym_encrypt(value, key)
    key_version     INTEGER REFERENCES vault_keys(version),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Row-level security: only admin/analyst roles can decrypt
ALTER TABLE token_vault ENABLE ROW LEVEL SECURITY;

CREATE POLICY analyst_read ON token_vault
    FOR SELECT
    USING (auth.jwt() ->> 'role' IN ('admin', 'analyst'));

-- Encrypt function (call from application layer)
-- SELECT pgp_sym_encrypt('sensitive_value', 'your_key') AS encrypted;

-- Decrypt function (role-gated via RLS)
-- SELECT pgp_sym_decrypt(encrypted_value, 'your_key')::TEXT
--   FROM token_vault WHERE token = 'TKN_CU_XXXX';

-- Predictions table with tokenized PII
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    customer_token      TEXT REFERENCES token_vault(token),
    merchant_token      TEXT REFERENCES token_vault(token),
    transaction_amount  NUMERIC(12,2),
    distance_from_home_km NUMERIC(8,2),
    hour                SMALLINT,
    is_foreign          BOOLEAN,
    is_new_device       BOOLEAN,
    vpn_detected        BOOLEAN,
    fraud_probability   NUMERIC(6,4),
    is_fraud            BOOLEAN,
    threshold_used      NUMERIC(4,3),
    graph_risk_score    NUMERIC(8,4),
    ring_member         BOOLEAN DEFAULT FALSE
);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud ON predictions(is_fraud);
"""


def get_pgcrypto_schema() -> str:
    """Return the Supabase pgcrypto schema SQL."""
    return PGCRYPTO_SCHEMA_SQL
