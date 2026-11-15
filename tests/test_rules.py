from __future__ import annotations

from datetime import date

from findoc.fraud.rules import run_cheque_rules
from findoc.types import ExtractedField

TODAY = date(2026, 11, 15)


def f(name: str, value: str, normalized: str | None, conf: float = 0.9) -> ExtractedField:
    return ExtractedField(name=name, value=value, normalized=normalized, confidence=conf)


def base_fields(**overrides: tuple[str, str | None]) -> list[ExtractedField]:
    fields = {
        "date": ("01112026", "2026-11-01"),
        "amount_words": ("Twenty Five Thousand Only", "25000.00"),
        "amount_figures": ("25,000/-", "25000.00"),
        "ifsc": ("ARVL0001234", "ARVL0001234"),
        "account_number": ("12345678901", "12345678901"),
        "micr": ("123456 302912282 834713 10", "123456 302912282 834713 10"),
        "signature": ("present", "0.05"),
    }
    fields.update(overrides)
    return [f(n, v, norm) for n, (v, norm) in fields.items()]


def fired(fields: list[ExtractedField], **kw) -> set[str]:  # type: ignore[no-untyped-def]
    return {c.id for c in run_cheque_rules(fields, today=TODAY, **kw) if c.fired}


def test_clean_cheque_passes_all() -> None:
    results = run_cheque_rules(base_fields(), today=TODAY)
    assert not [c for c in results if c.fired]
    assert {c.id for c in results} >= {"AMOUNT_MISMATCH", "DATE_VALIDITY", "SIGNATURE_MISSING"}


def test_amount_mismatch() -> None:
    fields = base_fields(amount_words=("Two Lakh Fifty Thousand", "250000.00"))
    assert fired(fields) == {"AMOUNT_MISMATCH"}
    check = next(c for c in run_cheque_rules(fields, today=TODAY) if c.id == "AMOUNT_MISMATCH")
    assert check.severity == 1.0
    assert check.evidence == {"words": "250000.00", "figures": "25000.00"}


def test_amount_tolerance_allows_rounding() -> None:
    fields = base_fields(amount_figures=("25,000.25", "25000.25"))
    assert "AMOUNT_MISMATCH" not in fired(fields)


def test_unreadable_amount_is_reported_not_passed() -> None:
    assert "AMOUNT_UNREADABLE" in fired(base_fields(amount_figures=("~~", None)))


def test_stale_and_postdated() -> None:
    assert fired(base_fields(date=("01072026", "2026-07-01"))) == {"DATE_STALE"}
    assert fired(base_fields(date=("01122026", "2026-12-01"))) == {"DATE_POSTDATED"}
    assert fired(base_fields(date=("15082026", "2026-08-15"))) == set()  # exactly 3 months
    assert fired(base_fields(date=("99", None))) == {"DATE_UNREADABLE"}


def test_format_rules() -> None:
    fields = base_fields(ifsc=("ARVL1234", None), account_number=("123", None), micr=("1", None))
    assert fired(fields) == {"IFSC_INVALID", "ACCOUNT_INVALID", "MICR_INVALID"}


def test_missing_signature() -> None:
    assert fired(base_fields(signature=("absent", "0.0"))) == {"SIGNATURE_MISSING"}


def test_without_ocr_only_signature_runs() -> None:
    results = run_cheque_rules(base_fields(), today=TODAY, text_read=False)
    assert [c.id for c in results] == ["SIGNATURE_MISSING"]
