from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from findoc.recognize.parsers import (
    add_months,
    amount_in_words,
    indian_grouping,
    normalize_account,
    normalize_ifsc,
    number_to_words,
    parse_amount_figures,
    parse_cheque_date,
    parse_micr,
    words_to_amount,
)


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (15, "fifteen"),
        (105, "one hundred five"),
        (25_000, "twenty five thousand"),
        (1_20_500, "one lakh twenty thousand five hundred"),
        (5_20_00_000, "five crore twenty lakh"),
        (1_00_00_00_000, "one hundred crore"),
    ],
)
def test_number_to_words_indian_system(n: int, words: str) -> None:
    assert number_to_words(n) == words


def test_words_round_trip_random() -> None:
    rng = np.random.default_rng(0)
    for n in rng.integers(1, 10**10, 2000):
        assert words_to_amount(number_to_words(int(n))).value == Decimal(int(n))


def test_amount_in_words_with_paise() -> None:
    text = amount_in_words(Decimal("25000.50"))
    assert text == "Twenty Five Thousand and Fifty Paise Only"
    assert words_to_amount(text).value == Decimal("25000.50")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Twenty Five Thousand Only", "25000"),
        ("Rupees Twenty-Five Thousand Only", "25000"),
        ("One Lac Twenty Thousand", "120000"),
        ("Two Lakhs", "200000"),
        ("Ten Crores", "100000000"),
        ("Hundred", "100"),
        ("Thousand", "1000"),
    ],
)
def test_words_to_amount_variants(text: str, value: str) -> None:
    assert words_to_amount(text).value == Decimal(value)


def test_words_to_amount_repairs_ocr_typos() -> None:
    parsed = words_to_amount("Twenty Fve Thousnd Rupees Only")
    assert parsed.value == Decimal("25000")
    assert ("thousnd", "thousand") in parsed.corrections
    assert 0 < parsed.confidence < 1


def test_words_to_amount_rejects_non_numbers() -> None:
    parsed = words_to_amount("Pay to the order of")
    assert parsed.value is None
    assert parsed.confidence == 0.0
    assert parsed.unknown


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Rs. 2,50,000.00/-", "250000.00"),
        ("₹ 8,40,515/-", "840515.00"),
        ("25,OOO", "25000.00"),  # letter O read for zero
        ("1.234.56", "1234.56"),  # only the last dot is decimal
    ],
)
def test_parse_amount_figures(text: str, value: str) -> None:
    assert parse_amount_figures(text) == Decimal(value)


def test_parse_amount_figures_invalid() -> None:
    assert parse_amount_figures("abc") is None
    assert parse_amount_figures("") is None


@pytest.mark.parametrize(
    ("n", "text"),
    [(5, "5"), (1000, "1,000"), (840515, "8,40,515"), (123456789, "12,34,56,789")],
)
def test_indian_grouping(n: int, text: str) -> None:
    assert indian_grouping(n) == text
    assert parse_amount_figures(text) == Decimal(n)


def test_parse_cheque_date() -> None:
    assert parse_cheque_date("15112026") == date(2026, 11, 15)
    assert parse_cheque_date("1 5 1 1 2 0 2 6") == date(2026, 11, 15)
    assert parse_cheque_date("15/11/26") == date(2026, 11, 15)
    assert parse_cheque_date("3l1O2026") == date(2026, 10, 31)  # OCR letter confusions
    assert parse_cheque_date("31112026") is None  # 31 November doesn't exist
    assert parse_cheque_date("12345") is None


def test_add_months_clamps_day() -> None:
    assert add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
    assert add_months(date(2026, 1, 15), 3) == date(2026, 4, 15)


def test_ifsc() -> None:
    assert normalize_ifsc("ARVL0001234") == "ARVL0001234"
    assert normalize_ifsc("arvl O001234") == "ARVL0001234"  # O in the zero position
    assert normalize_ifsc("4RVL0001234") == "ARVL0001234"  # digit in the letter part
    assert normalize_ifsc("ARVL1001234") is None  # 5th character must be 0
    assert normalize_ifsc("SHORT") is None


def test_account_number() -> None:
    assert normalize_account("0878 5834 713") == "08785834713"
    assert normalize_account("12345") is None


def test_micr() -> None:
    micr = parse_micr("123456 302912282 834713 10")
    assert micr is not None
    assert micr.cheque_number == "123456" and micr.micr_code == "302912282"
    assert micr.account_id == "834713" and micr.transaction_code == "10"
    assert str(micr) == "123456 302912282 834713 10"
    assert parse_micr("123") is None
