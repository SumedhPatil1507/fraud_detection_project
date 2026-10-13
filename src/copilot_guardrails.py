"""Fail-closed citation, grounding, and SAR output checks."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

FINCEN_NARRATIVE_MAX_CHARS = 20_000
FINCEN_STANDARD_DEADLINE_DAYS = 30
FINCEN_NO_SUSPECT_DEADLINE_DAYS = 60
_CITATION = re.compile(r"\[chunk_id:([^\]\s]+)\]")
_WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]*")
_STOP_WORDS = {
    "about", "after", "again", "also", "among", "and", "are", "around", "as",
    "because", "been", "before", "being", "between", "both", "but", "case",
    "could", "customer", "during", "each", "from", "had", "have", "into", "its",
    "more", "most", "other", "over", "same", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "through", "under", "was",
    "were", "when", "where", "which", "while", "with", "would",
}


def _working_day_deadline(concluded_at: datetime, days: int = 7) -> datetime:
    """Add working days; optional RBI_HOLIDAY_CALENDAR is newline YYYY-MM-DD."""
    holidays = set()
    calendar_path = os.getenv("RBI_HOLIDAY_CALENDAR")
    if calendar_path:
        holidays = {line.strip() for line in Path(calendar_path).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}
    current = concluded_at
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5 and current.date().isoformat() not in holidays:
            added += 1
    return current.replace(hour=23, minute=59, second=59, microsecond=999999)


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_lerner_price(pricing: dict, checks: dict, errors: list[str]) -> None:
    """Enforce configured L=(price-cost)/price cap for proposed prices."""
    if not pricing:
        checks["lerner_pricing_status"] = "not_applicable"
        return
    if "proposed_price" not in pricing or "marginal_cost" not in pricing:
        errors.append("pricing requires proposed_price and marginal_cost for Lerner validation")
        checks["lerner_pricing_status"] = "missing_inputs"
        return
    try:
        price = float(pricing["proposed_price"])
        marginal_cost = float(pricing["marginal_cost"])
        # The cap is deployment policy; input data cannot relax it.
        cap = float(os.getenv("LERNER_INDEX_MAX", "0.30"))
        if not all(map(lambda value: value == value and abs(value) != float("inf"),
                       (price, marginal_cost, cap))):
            raise ValueError("non-finite pricing input")
        if price <= 0 or marginal_cost < 0 or not 0 <= cap < 1:
            raise ValueError("price, marginal cost, or Lerner cap is outside its valid range")
        index = (price - marginal_cost) / price
        price_ceiling = marginal_cost / (1.0 - cap)
        checks.update({"lerner_index": round(index, 6), "lerner_index_max": cap,
                       "lerner_price_ceiling": round(price_ceiling, 6)})
        if index > cap + 1e-12:
            errors.append("proposed price exceeds the configured Lerner-index cap")
            checks["lerner_pricing_status"] = "over_cap"
        else:
            checks["lerner_pricing_status"] = "within_cap"
        # Use Guardrails AI's numeric range validator when the optional package
        # is installed; the direct check above remains the deterministic control.
        try:
            from guardrails import Guard
            from guardrails_ai.valid_range import ValidRange
            Guard().use(ValidRange(min=0, max=price_ceiling, on_fail="exception")).validate(price)
        except ImportError:
            errors.append("Guardrails AI ValidRange validator is unavailable for pricing validation")
            checks["lerner_guardrails_status"] = "unavailable"
        except Exception as exc:
            errors.append(f"Guardrails AI Lerner price validation failed ({type(exc).__name__})")
            checks["lerner_guardrails_status"] = "rejected"
        else:
            checks["lerner_guardrails_status"] = "passed"
    except (TypeError, ValueError, OverflowError):
        errors.append("pricing inputs must be finite numeric values with a valid Lerner cap")
        checks["lerner_pricing_status"] = "invalid_inputs"


def validate_sar_narrative(narrative: str, retrieved_cases: list[dict],
                           transaction: dict | None = None,
                           *, now: datetime | None = None,
                           pricing: dict | None = None,
                           require_filing_dates: bool = False) -> dict:
    """Return {valid, errors, checks}; no citation is trusted unless retrieved."""
    transaction = transaction or {}
    errors: list[str] = []
    jurisdiction = str(transaction.get("jurisdiction")
                       or os.getenv("SAR_JURISDICTION", "US")).upper()
    max_chars = FINCEN_NARRATIVE_MAX_CHARS
    fincen_applies = jurisdiction in {"US", "USA", "FINCEN", "US_FINCEN"}
    if fincen_applies:
        if len(narrative) > max_chars:
            errors.append(f"narrative exceeds FinCEN {max_chars:,}-character limit")
    # Guardrails AI validates the character constraint; the direct FinCEN
    # bound above stays explicit and easy to audit.
    try:
        from guardrails import Guard
        from guardrails_ai.valid_length import ValidLength
        Guard().use(ValidLength(min=1, max=max_chars if fincen_applies else None,
                                on_fail="exception")).validate(narrative)
    except ImportError as exc:
        errors.append(f"Guardrails AI or its ValidLength validator is unavailable ({type(exc).__name__})")
    except Exception as exc:
        # Validator exceptions may include the rejected narrative; retain only
        # the exception type in returned state and logs.
        errors.append(f"Guardrails AI validation failed ({type(exc).__name__})")

    by_id = {str(hit.get("id")): str(hit.get("text", ""))
             for hit in retrieved_cases if hit.get("id") is not None}
    cited_ids = set(_CITATION.findall(narrative))
    if not cited_ids:
        errors.append("narrative must cite at least one retrieved vector chunk ID")
    invalid_ids = cited_ids - by_id.keys()
    if invalid_ids:
        errors.append("narrative cites chunk IDs absent from retrieved context")

    # Every non-empty sentence is treated as a factual assertion. A cited
    # source must support its meaningful terms, not merely be named.
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "<decimal>", narrative)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    for index, sentence in enumerate(sentences, 1):
        refs = _CITATION.findall(sentence)
        if not refs:
            errors.append(f"sentence {index} has no vector chunk citation")
            continue
        words = {word.lower() for word in _WORD.findall(_CITATION.sub("", sentence))
                 if len(word) > 2 and word.lower() not in _STOP_WORDS}
        sources = [by_id[ref] for ref in refs if ref in by_id]
        source_words = {word.lower() for source in sources for word in _WORD.findall(source)
                        if len(word) > 2 and word.lower() not in _STOP_WORDS}
        overlap = words & source_words
        if len(words) >= 3 and (len(overlap) < 2 or len(overlap) / len(words) < 0.20):
            errors.append(f"sentence {index} is not grounded in its cited vector context")

    checks = {"jurisdiction": jurisdiction, "narrative_characters": len(narrative),
              "fincen_narrative_limit": max_chars if fincen_applies else None,
              "cited_chunk_ids": sorted(cited_ids)}
    normalized_now = now or datetime.now(timezone.utc)
    if normalized_now.tzinfo is None:
        normalized_now = normalized_now.replace(tzinfo=timezone.utc)
    else:
        normalized_now = normalized_now.astimezone(timezone.utc)
    if fincen_applies:
        detection = transaction.get("initial_detection_at")
        if detection:
            try:
                detected_at = _parse_datetime(detection)
                raw_no_suspect = transaction.get("no_suspect_identified_at_detection", False)
                no_suspect = (raw_no_suspect is True or
                              str(raw_no_suspect).strip().lower() in {"1", "true", "yes"})
                days = FINCEN_NO_SUSPECT_DEADLINE_DAYS if no_suspect else FINCEN_STANDARD_DEADLINE_DAYS
                deadline = (detected_at + timedelta(days=days)).replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
                checks["fincen_filing_due_at"] = deadline.isoformat()
                checks["fincen_filing_deadline_days"] = days
                checks["fincen_filing_deadline_status"] = (
                    "within_deadline" if normalized_now <= deadline else "overdue_escalate"
                )
                if detected_at > normalized_now:
                    errors.append("FinCEN initial_detection_at cannot be in the future")
                if normalized_now > deadline:
                    errors.append("FinCEN SAR filing deadline has passed; escalate for compliance review")
            except (TypeError, ValueError, OverflowError):
                errors.append("FinCEN initial_detection_at is invalid")
                checks["fincen_filing_deadline_status"] = "invalid_detection_time"
        else:
            checks["fincen_filing_deadline_status"] = "not_evaluated_missing_initial_detection_at"
            if require_filing_dates:
                errors.append("FinCEN initial_detection_at is required to verify the SAR filing deadline")
    conclusion = transaction.get("suspicion_concluded_at")
    if jurisdiction in {"IN", "INDIA", "RBI"} and conclusion:
        try:
            concluded = _parse_datetime(conclusion)
            deadline = _working_day_deadline(concluded)
            if now is None:
                now = datetime.now(timezone.utc)
            elif now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            else:
                now = now.astimezone(timezone.utc)
            checks["rbi_str_due_at"] = deadline.isoformat()
            checks["rbi_str_deadline_status"] = "within_deadline" if now <= deadline else "overdue_escalate"
            checks["rbi_holiday_calendar"] = "configured" if os.getenv("RBI_HOLIDAY_CALENDAR") else "missing"
            if not os.getenv("RBI_HOLIDAY_CALENDAR"):
                errors.append("RBI_HOLIDAY_CALENDAR is required to verify working-day deadlines")
            if now > deadline:
                errors.append("RBI STR seven-working-day filing deadline has passed; escalate to the Principal Officer")
            if not str(transaction.get("principal_officer_reason", "")).strip():
                errors.append("RBI principal_officer_reason is required for Principal Officer review")
        except (TypeError, ValueError, OSError) as exc:
            errors.append(f"RBI suspicion_concluded_at is invalid: {exc}")
    elif jurisdiction in {"IN", "INDIA", "RBI"}:
        checks["rbi_str_deadline_status"] = "not_evaluated_missing_suspicion_concluded_at"
        errors.append("RBI suspicion_concluded_at is required to verify the STR filing deadline")

    _validate_lerner_price(pricing or transaction.get("pricing", {}), checks, errors)
    return {"valid": not errors, "errors": errors, "checks": checks}
