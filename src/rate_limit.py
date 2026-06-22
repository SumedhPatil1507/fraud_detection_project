"""
API Rate Limiting using slowapi (Starlette-compatible limiter).
Falls back gracefully if slowapi is not installed.
"""
from __future__ import annotations
import os

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _SLOWAPI = True
except ImportError:
    _SLOWAPI = False


def make_limiter():
    if _SLOWAPI:
        return Limiter(key_func=get_remote_address)
    return None


LIMITER = make_limiter()

# Default rate limits per role (requests / window)
RATE_LIMITS = {
    "admin":   os.environ.get("RATE_LIMIT_ADMIN",   "1000/minute"),
    "analyst": os.environ.get("RATE_LIMIT_ANALYST", "200/minute"),
    "viewer":  os.environ.get("RATE_LIMIT_VIEWER",  "30/minute"),
    "default": os.environ.get("RATE_LIMIT_DEFAULT", "60/minute"),
}


def get_limit_for_role(role: str) -> str:
    return RATE_LIMITS.get(str(role), RATE_LIMITS["default"])
