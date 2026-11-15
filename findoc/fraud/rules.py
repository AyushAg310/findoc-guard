"""Rule-based validation checks for cheques (FR-FRD-01..03, signature presence).

Each rule returns a :class:`~findoc.types.CheckResult` whether or not it fired, so the
report shows what was verified as well as what failed. ``severity`` (how bad if true) and
``confidence`` (how sure we are) feed the Phase 5 risk score.

Rules never guess: if a field couldn't be read, the rule reports it as unverifiable
(``*_UNREADABLE``) instead of passing or failing it.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from findoc.recognize.parsers import add_months
from findoc.types import CheckResult, ExtractedField

STALE_MONTHS = 3  # RBI: cheques are valid for 3 months from the date of issue
AMOUNT_TOLERANCE = Decimal("0.50")


def _by_name(fields: list[ExtractedField]) -> dict[str, ExtractedField]:
    return {f.name: f for f in fields}


def check_amount(fields: dict[str, ExtractedField]) -> CheckResult:
    words, figures = fields.get("amount_words"), fields.get("amount_figures")
    if not words or not figures or words.normalized is None or figures.normalized is None:
        missing = [
            n for n, f in (("words", words), ("figures", figures)) if not f or f.normalized is None
        ]
        return CheckResult(
            id="AMOUNT_UNREADABLE",
            fired=True,
            severity=0.4,
            confidence=1.0,
            reason=f"Could not read the amount in {' and '.join(missing)}; verify manually.",
        )
    a, b = Decimal(words.normalized), Decimal(figures.normalized)
    conf = round(min(words.confidence, figures.confidence), 4)
    if abs(a - b) > AMOUNT_TOLERANCE:
        return CheckResult(
            id="AMOUNT_MISMATCH",
            fired=True,
            severity=1.0,
            confidence=conf,
            reason=f"Amount in words ({a:,.2f}) differs from amount in figures ({b:,.2f}).",
            evidence={"words": str(a), "figures": str(b)},
        )
    return CheckResult(
        id="AMOUNT_MISMATCH",
        fired=False,
        confidence=conf,
        reason=f"Words and figures agree ({b:,.2f}).",
    )


def check_date(fields: dict[str, ExtractedField], today: date) -> CheckResult:
    f = fields.get("date")
    if not f or f.normalized is None:
        return CheckResult(
            id="DATE_UNREADABLE",
            fired=True,
            severity=0.5,
            confidence=1.0,
            reason="Cheque date is missing or not a valid calendar date.",
        )
    d = date.fromisoformat(f.normalized)
    if d > today:
        return CheckResult(
            id="DATE_POSTDATED",
            fired=True,
            severity=0.6,
            confidence=f.confidence,
            reason=f"Post-dated cheque: dated {d:%d %b %Y}, cannot be presented before then.",
            evidence={"date": d.isoformat()},
        )
    expiry = add_months(d, STALE_MONTHS)
    if today > expiry:
        return CheckResult(
            id="DATE_STALE",
            fired=True,
            severity=0.9,
            confidence=f.confidence,
            reason=f"Stale cheque: dated {d:%d %b %Y}, valid only until {expiry:%d %b %Y}.",
            evidence={"date": d.isoformat(), "valid_until": expiry.isoformat()},
        )
    return CheckResult(
        id="DATE_VALIDITY",
        fired=False,
        confidence=f.confidence,
        reason=f"Dated {d:%d %b %Y}; valid until {expiry:%d %b %Y}.",
    )


def check_formats(fields: dict[str, ExtractedField]) -> list[CheckResult]:
    results = []
    # Format rules are sanity checks (often tripped by a single OCR error), not evidence of
    # fraud on their own, so their severity is low; the risk score (Phase 5) weighs them.
    specs = [
        ("ifsc", "IFSC_INVALID", "IFSC must be 4 letters, '0', then 6 letters/digits.", 0.4),
        ("account_number", "ACCOUNT_INVALID", "Account number must be 9-18 digits.", 0.4),
        ("micr", "MICR_INVALID", "MICR line must have 6+9+6+2 digits.", 0.3),
    ]
    for name, check_id, rule, severity in specs:
        f = fields.get(name)
        if f is None:
            continue
        ok = f.normalized is not None
        results.append(
            CheckResult(
                id=check_id,
                fired=not ok,
                severity=severity if not ok else 0.0,
                confidence=f.confidence if ok else 0.8,
                reason=f"{name} '{f.value}' {'is valid' if ok else 'is invalid'}. {rule}",
            )
        )
    return results


def check_signature(fields: dict[str, ExtractedField]) -> CheckResult:
    f = fields.get("signature")
    absent = f is None or f.value == "absent"
    return CheckResult(
        id="SIGNATURE_MISSING",
        fired=absent,
        severity=0.8 if absent else 0.0,
        confidence=f.confidence if f else 1.0,
        reason="No signature found in the signature area."
        if absent
        else "Signature area contains ink (presence only; similarity check in Phase 5).",
    )


def run_cheque_rules(
    fields: list[ExtractedField], today: date | None = None, text_read: bool = True
) -> list[CheckResult]:
    """All Phase 2 cheque rules. Text rules are skipped when OCR didn't run."""
    by_name = _by_name(fields)
    results = [check_signature(by_name)]
    if text_read:
        results = [
            check_amount(by_name),
            check_date(by_name, today or date.today()),
            *check_formats(by_name),
            *results,
        ]
    return results
