"""
Data Validation Layer
Validates incoming transaction data for type correctness, range bounds,
and business-rule consistency before model inference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


FIELD_RULES: dict[str, dict] = {
    "transaction_amount":      {"type": float, "min": 0.01,  "max": 1_000_000},
    "distance_from_home_km":   {"type": float, "min": 0.0,   "max": 50_000},
    "hour":                    {"type": int,   "min": 0,     "max": 23},
    "is_foreign":              {"type": int,   "min": 0,     "max": 1},
    "is_new_device":           {"type": int,   "min": 0,     "max": 1},
    "vpn_detected":            {"type": int,   "min": 0,     "max": 1},
    "transaction_velocity_1h": {"type": float, "min": 0.0,   "max": 500},
    "amount_deviation_ratio":  {"type": float, "min": 0.0,   "max": 1000},
    "threshold":               {"type": float, "min": 0.0,   "max": 1.0},
}

REQUIRED_FIELDS = {"transaction_amount", "distance_from_home_km"}


def validate_transaction(data: dict[str, Any]) -> ValidationResult:
    result = ValidationResult(valid=True)

    # Required fields
    for f in REQUIRED_FIELDS:
        if f not in data or data[f] is None:
            result.add_error(f"Missing required field: '{f}'")

    # Per-field rules
    for fname, rules in FIELD_RULES.items():
        val = data.get(fname)
        if val is None:
            continue
        try:
            val = rules["type"](val)
        except (TypeError, ValueError):
            result.add_error(f"'{fname}' must be {rules['type'].__name__}, got {type(val).__name__}")
            continue
        if "min" in rules and val < rules["min"]:
            result.add_error(f"'{fname}' = {val} is below minimum {rules['min']}")
        if "max" in rules and val > rules["max"]:
            result.add_error(f"'{fname}' = {val} exceeds maximum {rules['max']}")

    # Business-rule cross-checks
    amount = data.get("transaction_amount", 0)
    distance = data.get("distance_from_home_km", 0)
    if amount and distance:
        if float(amount) > 50_000 and float(distance) < 1:
            result.add_warning("Very high amount with near-zero distance — unusual pattern")
    if data.get("vpn_detected") and data.get("is_foreign"):
        result.add_warning("VPN + foreign transaction combination — elevated risk")

    return result


def validate_batch(records: list[dict]) -> list[ValidationResult]:
    return [validate_transaction(r) for r in records]
