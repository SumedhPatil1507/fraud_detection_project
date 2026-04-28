"""
Data Encryption at Rest
Uses Fernet symmetric encryption (AES-128-CBC) for sensitive fields.
Key is loaded from ENCRYPTION_KEY env var or auto-generated (dev mode).
"""
import os
import base64
import pandas as pd
from cryptography.fernet import Fernet


def _get_key() -> bytes:
    key = os.environ.get("ENCRYPTION_KEY", "")
    if key:
        # Accept raw 32-byte base64url key or full Fernet key
        try:
            return key.encode()
        except Exception:
            pass
    # Dev fallback — generate and warn
    generated = Fernet.generate_key()
    print("[WARN] ENCRYPTION_KEY not set — using ephemeral key (data won't persist across restarts)")
    return generated


_fernet = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_get_key())
    return _fernet


def encrypt_value(value: str) -> str:
    """Encrypt a string value, returns base64 ciphertext."""
    if not value:
        return value
    return _get_fernet().encrypt(value.encode()).decode()


def decrypt_value(token: str) -> str:
    """Decrypt a Fernet token back to plaintext."""
    if not token:
        return token
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except Exception:
        return "[DECRYPTION_FAILED]"


# Columns that should be encrypted at rest
SENSITIVE_COLUMNS = ["customer_id", "merchant_id", "device_id"]


def encrypt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encrypt sensitive columns in a DataFrame before storage."""
    df = df.copy()
    for col in SENSITIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(encrypt_value)
    return df


def decrypt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Decrypt sensitive columns after loading from storage."""
    df = df.copy()
    for col in SENSITIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(decrypt_value)
    return df


def generate_new_key() -> str:
    """Generate a new Fernet key — store this in ENCRYPTION_KEY env var."""
    return Fernet.generate_key().decode()
