"""Parsers that turn noisy OCR text into validated values (FR-EXT, FR-FRD-01..03).

OCR output is never trusted as-is. Each parser:

1. **Normalises** text (case, spaces, currency symbols).
2. **Repairs common OCR confusions** that depend on the field's type, e.g. the letter ``O``
   read in a numeric field is almost certainly the digit ``0``.
3. **Validates** against the field's format and returns ``None`` when it can't produce a
   trustworthy value, so rules can report "unreadable" instead of comparing garbage.

Amounts use the Indian numbering system: 1,00,000 = one lakh; 1,00,00,000 = one crore.
"""

from __future__ import annotations

import calendar
import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

# --------------------------------------------------------------------------- numbers -> words

ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]  # fmt: skip
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_hundred(n: int) -> list[str]:
    if n < 20:
        return [ONES[n]] if n else []
    return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
    return words + _below_hundred(n % 100)


def number_to_words(n: int) -> str:
    """Indian-system words for a non-negative integer, e.g. 1,25,000 -> 'one lakh twenty five
    thousand'."""
    if n < 0:
        raise ValueError("negative amounts are not supported")
    if n == 0:
        return "zero"
    words: list[str] = []
    crore, rest = divmod(n, 10_000_000)
    if crore:
        words += number_to_words(crore).split() + ["crore"]
    lakh, rest = divmod(rest, 100_000)
    if lakh:
        words += _below_hundred(lakh) + ["lakh"]
    thousand, rest = divmod(rest, 1000)
    if thousand:
        words += _below_hundred(thousand) + ["thousand"]
    words += _below_thousand(rest)
    return " ".join(words)


def amount_in_words(amount: Decimal | int) -> str:
    """Cheque style: 'Twenty Five Thousand Only' or '... and Fifty Paise Only'."""
    amount = Decimal(amount).quantize(Decimal("0.01"))
    rupees = int(amount)
    paise = int((amount - rupees) * 100)
    text = number_to_words(rupees).title()
    if paise:
        text += " and " + number_to_words(paise).title() + " Paise"
    return text + " Only"


def indian_grouping(amount: Decimal | int) -> str:
    """12345678 -> '1,23,45,678' (last 3 digits, then groups of 2)."""
    amount = Decimal(amount).quantize(Decimal("0.01"))
    rupees, paise = divmod(int(amount * 100), 100)
    s = str(rupees)
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join([*groups, tail])
    return f"{text}.{paise:02d}" if paise else text


# --------------------------------------------------------------------------- words -> number

SMALL = {w: i for i, w in enumerate(ONES) if w} | {w: i * 10 for i, w in enumerate(TENS) if w}
MULTIPLIERS = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
}
FILLER = {"and", "only", "rupees", "rupee", "rs", "inr"}
PAISE = {"paise", "paisa"}
VOCAB = sorted(set(SMALL) | set(MULTIPLIERS) | FILLER | PAISE)


@dataclass
class AmountParse:
    value: Decimal | None
    corrections: list[tuple[str, str]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """1.0 if every token was an exact dictionary word; lower with each repair."""
        if self.value is None:
            return 0.0
        penalty = 0.15 * len(self.corrections) + 0.3 * len(self.unknown)
        return max(0.0, 1.0 - penalty)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("-", " "))


def _closest(tok: str) -> list[str]:
    """Nearest number word, if close in both spelling and length.

    Two-letter tokens ("to", "of") are never repaired; three-letter tokens only to words at
    most one letter longer ("fve" -> "five", but not "the" -> "three").
    """
    if len(tok) < 3:
        return []
    max_diff, cutoff = (1, 0.8) if len(tok) == 3 else (2, 0.75)
    candidates = [w for w in VOCAB if abs(len(w) - len(tok)) <= max_diff]
    return difflib.get_close_matches(tok, candidates, n=1, cutoff=cutoff)


def _correct(tokens: list[str]) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Snap each token to the closest number word (fixes OCR typos like 'thousnd')."""
    out, corrections, unknown = [], [], []
    for tok in tokens:
        if tok in VOCAB:
            out.append(tok)
            continue
        close = _closest(tok)
        if close:
            out.append(close[0])
            corrections.append((tok, close[0]))
        elif len(tok) > 1:  # single stray letters are OCR noise, not words
            unknown.append(tok)
    return out, corrections, unknown


def _words_value(words: list[str]) -> int | None:
    total, current, seen = 0, 0, False
    for w in words:
        if w in SMALL:
            current += SMALL[w]
            seen = True
        elif w == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif w in ("thousand", "lakh", "lakhs", "lac", "lacs"):
            total += max(current, 1) * MULTIPLIERS[w]
            current, seen = 0, True
        elif w in ("crore", "crores"):
            total = (total + max(current, 1)) * MULTIPLIERS[w]
            current, seen = 0, True
    return total + current if seen else None


def words_to_amount(text: str) -> AmountParse:
    """Parse 'Twenty Five Thousand and Fifty Paise Only' -> Decimal('25000.50')."""
    words, corrections, unknown = _correct(_tokens(text))
    number_words = [w for w in words if w not in FILLER]
    if len(unknown) > len(number_words):  # mostly not an amount at all
        return AmountParse(None, corrections, unknown)
    paise = 0
    if any(w in PAISE for w in words):
        idx = max(i for i, w in enumerate(words) if w in PAISE)
        start = idx
        while start > 0 and words[start - 1] in SMALL:
            start -= 1
        paise = _words_value(words[start:idx]) or 0
        words = words[:start] + words[idx + 1 :]
    rupees = _words_value([w for w in words if w not in FILLER])
    if rupees is None and not paise:
        return AmountParse(None, corrections, unknown)
    value = Decimal(rupees or 0) + Decimal(paise) / 100
    return AmountParse(value.quantize(Decimal("0.01")), corrections, unknown)


# --------------------------------------------------------------------------- figures, dates, ids

DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1",
                             "|": "1", "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2",
                             "z": "2", "G": "6", "g": "9", "T": "7"})  # fmt: skip
LETTER_FIXES = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"})


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text.translate(DIGIT_FIXES))


def parse_amount_figures(text: str) -> Decimal | None:
    """'Rs. 2,50,000.00/-' -> Decimal('250000.00')."""
    t = re.sub(r"(?i)rs\.?|inr|₹|/-", "", text).strip()
    t = t.translate(DIGIT_FIXES)
    t = re.sub(r"[^\d.,]", "", t).strip(".,")
    # A decimal point is only ever followed by 1-2 digits (paise) at the end. Any other dot is
    # a misread comma: "30,00.189" is 30,00,189, not 3000.189.
    head, dot, tail = t.rpartition(".")
    if dot and 1 <= len(tail) <= 2 and tail.isdigit():
        t = head.replace(".", "").replace(",", "") + "." + tail
    else:
        t = t.replace(".", "").replace(",", "")
    if not t:
        return None
    try:
        return Decimal(t).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def parse_cheque_date(text: str) -> date | None:
    """Date boxes 'DDMMYYYY' (separators ignored); 'DDMMYY' assumes 20YY."""
    d = digits_only(text)
    if len(d) == 6:
        d = d[:4] + "20" + d[4:]
    if len(d) != 8:
        return None
    try:
        return date(int(d[4:]), int(d[2:4]), int(d[:2]))
    except ValueError:
        return None


def add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year, month = d.year + month_index // 12, month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def normalize_ifsc(text: str) -> str | None:
    """IFSC: 4 letters (bank), '0', 6 alphanumerics (branch), e.g. ARVL0001234."""
    t = re.sub(r"[^A-Za-z0-9]", "", text.upper().replace("IFSC", ""))
    if len(t) != 11:
        return None
    t = t[:4].translate(LETTER_FIXES) + t[4].replace("O", "0") + t[5:]
    return t if IFSC_RE.match(t) else None


def normalize_account(text: str) -> str | None:
    """Indian bank account numbers are 9-18 digits."""
    d = digits_only(text)
    return d if 9 <= len(d) <= 18 else None


@dataclass
class MicrLine:
    cheque_number: str  # 6 digits
    micr_code: str  # 9 digits: city (3) + bank (3) + branch (3)
    account_id: str  # 6 digits
    transaction_code: str  # 2 digits

    def __str__(self) -> str:
        return f"{self.cheque_number} {self.micr_code} {self.account_id} {self.transaction_code}"


def parse_micr(text: str) -> MicrLine | None:
    d = digits_only(text)
    if len(d) != 23:
        return None
    return MicrLine(d[:6], d[6:15], d[15:21], d[21:])
