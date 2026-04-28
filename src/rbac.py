"""
Role-Based Access Control (RBAC)
Roles: admin > analyst > viewer
Permissions are checked via decorators and helper functions.
"""
from __future__ import annotations
import os
import hashlib
import hmac
from enum import Enum
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader


class Role(str, Enum):
    ADMIN   = "admin"    # full access: train, retrain, manage users, view audit
    ANALYST = "analyst"  # predict, view metrics, HITL review, view audit
    VIEWER  = "viewer"   # read-only: view metrics, predictions (no PII)


# Permission map — what each role can do
PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN:   {"predict", "batch_predict", "train", "retrain", "audit", "hitl",
                   "manage_users", "view_pii", "sar_generate", "shadow_deploy",
                   "model_info", "drift"},
    Role.ANALYST: {"predict", "batch_predict", "audit", "hitl", "sar_generate",
                   "model_info", "drift"},
    Role.VIEWER:  {"model_info"},
}

# API key → (hashed_key, role) registry
# In production, store in DB. Here we load from env vars:
#   API_KEY_ADMIN=<key>   API_KEY_ANALYST=<key>   API_KEY_VIEWER=<key>
def _build_key_registry() -> dict[str, Role]:
    registry: dict[str, Role] = {}
    defaults = {
        Role.ADMIN:   os.environ.get("API_KEY_ADMIN",   "admin-dev-key"),
        Role.ANALYST: os.environ.get("API_KEY_ANALYST", "analyst-dev-key"),
        Role.VIEWER:  os.environ.get("API_KEY_VIEWER",  "viewer-dev-key"),
    }
    for role, key in defaults.items():
        if key:
            registry[key] = role
    return registry


_KEY_REGISTRY: dict[str, Role] = _build_key_registry()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_role(api_key: str = Security(api_key_header)) -> Role:
    """FastAPI dependency — resolves API key to a Role."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required.")
    role = _KEY_REGISTRY.get(api_key)
    if role is None:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return role


def require_permission(permission: str):
    """FastAPI dependency factory — raises 403 if role lacks permission."""
    def _check(role: Role = Depends(get_role)) -> Role:
        if permission not in PERMISSIONS.get(role, set()):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' lacks permission: '{permission}'."
            )
        return role
    return _check


def has_permission(role: Role, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, set())


# ── Streamlit RBAC helpers ─────────────────────────────────────────────────────
def streamlit_get_role() -> Optional[Role]:
    """Get current user role from Streamlit session state."""
    try:
        import streamlit as st
        return st.session_state.get("user_role", Role.VIEWER)
    except Exception:
        return Role.VIEWER


def streamlit_require(permission: str) -> bool:
    """Return True if current Streamlit user has permission, else show error."""
    try:
        import streamlit as st
        role = streamlit_get_role()
        if not has_permission(role, permission):
            st.error(f"🔒 Access denied — requires '{permission}' permission (your role: {role}).")
            return False
        return True
    except Exception:
        return False
