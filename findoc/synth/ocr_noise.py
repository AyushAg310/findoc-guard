"""Simulated OCR: ground-truth text with controlled, realistic character errors.

Used to measure how robust the parsers and rules are to OCR mistakes, independently of any
particular OCR model ("if OCR gets 5% of characters wrong, do we still catch an amount
mismatch?"). Real-OCR results come from running the same evaluation with EasyOCR.

Errors follow typical OCR confusions: visually similar substitutions (0/O, 1/l/I, 5/S, 8/B,
rn/m, e/c), deletions, and spurious spaces. ``cer`` is the per-character error probability
(Character Error Rate).
"""

from __future__ import annotations

import numpy as np

CONFUSIONS = {
    "0": "O", "O": "0", "o": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5", "8": "B",
    "B": "8", "2": "Z", "6": "G", "e": "c", "c": "e", "a": "o", "n": "m", "m": "n", "u": "v",
    "h": "b", "i": "l", "t": "f", "r": "n",
}  # fmt: skip
# With a digits-only allowlist the recogniser must output a digit, so confusions are
# between similar-looking digits instead of digit/letter pairs.
DIGIT_CONFUSIONS = {
    "0": "8", "8": "0", "1": "7", "7": "1", "3": "8", "5": "6", "6": "5", "2": "7", "9": "4",
    "4": "9",
}  # fmt: skip


def corrupt(text: str, cer: float, rng: np.random.Generator, allowlist: str | None = None) -> str:
    digits_only = allowlist is not None and not any(c.isalpha() for c in allowlist)
    table = DIGIT_CONFUSIONS if digits_only else CONFUSIONS
    out = []
    for ch in text:
        r = rng.random()
        if r >= cer:
            out.append(ch)
        elif r < cer * 0.6:  # substitution
            out.append(table.get(ch, ch))
        elif r < cer * 0.85:  # deletion
            continue
        else:  # spurious space
            out.extend([ch, " "])
    return "".join(out)


class OracleOcr:
    """Implements ``read_field``: returns the true text for a field, corrupted at ``cer``."""

    device = "simulated"

    def __init__(self, truth_fields: dict[str, str], cer: float = 0.0, seed: int = 0) -> None:
        self.truth = truth_fields
        self.cer = cer
        self.rng = np.random.default_rng(seed)

    def read_field(self, name: str, img: np.ndarray, allowlist: str | None) -> tuple[str, float]:
        text = corrupt(self.truth.get(name, ""), self.cer, self.rng, allowlist)
        if allowlist:
            text = "".join(c for c in text if c in allowlist or c.isspace())
        return text, max(0.0, 1.0 - 3 * self.cer)

    def read_line(self, img: np.ndarray, allowlist: str | None = None) -> tuple[str, float]:
        raise NotImplementedError("OracleOcr only reads named fields")

    def read(self, img: np.ndarray) -> list:
        return []
