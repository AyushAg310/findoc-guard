"""GSTIN: India's 15-character GST identification number, and its check digit (Phase 7).

    27  AAPFU0939F  1  Z  V
    |   |          |  |  check character (mod-36 checksum of the first 14)
    |   |          |  'Z' by default
    |   |          entity number for this PAN in this state (1-9, then A-Z)
    |   PAN of the business: 5 letters, 4 digits, 1 letter
    state code (01-38, 97 other territory, 99 centre jurisdiction)

**Check digit** (the GSTN algorithm, a Luhn variant in base 36): map each of the first 14
characters to its value in ``0-9A-Z`` (0..35); multiply alternately by 1 and 2 (1 for the first
character); for each product add its base-36 "digits" (``p // 36 + p % 36``); the check value is
``(36 - sum % 36) % 36``. Any single-character typo and most swaps of neighbours change it, so
an invalid check character means the number was mistyped, misread, or made up.

The state code also decides the **tax type**: supplier and buyer in the same state pay
CGST + SGST (half each); different states pay IGST.
"""

from __future__ import annotations

import re

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^(\d{2})([A-Z]{5}\d{4}[A-Z])([1-9A-Z])Z([0-9A-Z])$")
STATES = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh",
    "24": "Gujarat", "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu",
    "34": "Puducherry", "35": "Andaman and Nicobar Islands", "36": "Telangana",
    "37": "Andhra Pradesh", "38": "Ladakh", "97": "Other Territory", "99": "Centre Jurisdiction",
}  # fmt: skip
# OCR confusions to undo where the format says a digit or a letter must be
_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8",
                           "Z": "2", "G": "6"})  # fmt: skip
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"})


def check_char(first14: str) -> str:
    """The check character for the first 14 characters of a GSTIN."""
    total = 0
    for i, ch in enumerate(first14.upper()):
        p = CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return CHARS[(36 - total % 36) % 36]


def clean(text: str) -> str:
    """Upper case, separators removed, and OCR confusions fixed position by position (digits
    where the format requires digits, letters where it requires letters)."""
    t = re.sub(r"[^0-9A-Za-z]", "", text).upper()
    if len(t) != 15:
        return t
    parts = [
        t[0:2].translate(_TO_DIGIT),
        t[2:7].translate(_TO_LETTER),
        t[7:11].translate(_TO_DIGIT),
        t[11].translate(_TO_LETTER),
        t[12:],
    ]
    return "".join(parts)


def validate(text: str) -> tuple[str | None, str]:
    """(normalised GSTIN or None, reason). Valid means: format, known state code, check digit."""
    g = clean(text)
    if not GSTIN_RE.match(g):
        return None, f"'{text}' is not in GSTIN format (2 digits, PAN, entity, Z, check)"
    if g[:2] not in STATES:
        return None, f"'{g}': unknown state code {g[:2]}"
    expected = check_char(g[:14])
    if g[14] != expected:
        return None, f"'{g}': check character is {g[14]}, should be {expected}"
    return g, "valid"


# characters OCR commonly confuses, in both directions (digits with digits, letters with
# letters, and across, which matters where the format allows both: entity and check)
CONFUSABLE = {
    "0": "O8Q",
    "O": "0QD",
    "Q": "O0",
    "D": "0O",
    "1": "7IL",
    "7": "1",
    "I": "1L",
    "L": "1I",
    "3": "8",
    "8": "3B0",
    "B": "8",
    "5": "6S",
    "6": "5G",
    "S": "5",
    "G": "6C",
    "C": "G",
    "2": "Z",
    "Z": "2",
    "E": "F",
    "F": "E",
    "M": "N",
    "N": "M",
    "U": "V",
    "V": "U",
}


def misread_fixes(text: str) -> list[str]:
    """Valid GSTINs one common OCR confusion away from ``text`` (empty if ``text`` is valid or
    no single confusion explains it)."""
    g = clean(text)
    if len(g) != 15 or validate(g)[0]:
        return []
    out = []
    for i, ch in enumerate(g):
        for alt in CONFUSABLE.get(ch, ""):
            cand = g[:i] + alt + g[i + 1 :]
            if validate(cand)[0] and cand not in out:
                out.append(cand)
    return out


def make(state: str, pan: str, entity: str = "1") -> str:
    """A valid GSTIN from its parts (for synthetic data and tests)."""
    first = f"{state}{pan}{entity}Z"
    return first + check_char(first)


def state_of(gstin: str) -> str | None:
    return STATES.get(gstin[:2]) if gstin else None
