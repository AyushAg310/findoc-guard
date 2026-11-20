"""Explainable risk score from the fired checks (FR-FRD-09, SRS 10.6).

Every check that fired adds ``points x severity x confidence`` to the score, capped at 100:

* ``points`` - how much this kind of problem alone can contribute (its weight, out of 100);
* ``severity`` (0..1) - how bad this instance is, set by the check;
* ``confidence`` (0..1) - how sure the check is (OCR confidence, detector score, ...).

The report lists each contribution, so anyone can recompute the score by hand. Bands:
below ``risk_medium`` (30) **low**, below ``risk_high`` (70) **medium** (review), otherwise
**high** (escalate). A few findings are critical on their own and force **high** when the
check is confident: a duplicate presentation, words and figures that disagree, a tampered
amount.

Why points rather than weights that sum to 1 (as first written in the SRS): with weights
summing to 1 over ~12 checks, a single proven fraud (amount in words differs from figures)
could contribute at most ~10 points and the cheque would come out "low". Points state directly
"this alone is worth review / escalation", which is how reviewers think about it.
"""

from __future__ import annotations

from dataclasses import dataclass

from findoc.types import CheckResult

# Points per check id: the most it alone can add to the score.
POINTS: dict[str, float] = {
    "AMOUNT_MISMATCH": 100,
    "DUPLICATE": 100,
    "TAMPER_REGION": 90,
    "SIGNATURE_MISMATCH": 80,
    "SIGNATURE_MISSING": 75,
    "DATE_STALE": 70,
    "DATE_POSTDATED": 60,
    "DATE_UNREADABLE": 40,
    "AMOUNT_UNREADABLE": 40,
    "IFSC_INVALID": 25,
    "ACCOUNT_INVALID": 25,
    "MICR_INVALID": 20,
    "INVOICE_MATH": 80,
    "GSTIN_INVALID": 60,
    "TAX_TYPE_MISMATCH": 60,
    "INVOICE_UNVERIFIED": 30,
}
DEFAULT_POINTS = 30.0  # a check without its own entry
# check id -> minimum confidence at which it forces the "high" band by itself
CRITICAL: dict[str, float] = {"AMOUNT_MISMATCH": 0.9, "DUPLICATE": 0.8, "TAMPER_REGION": 0.8}
BANDS = ("low", "medium", "high")


@dataclass
class Risk:
    score: int
    band: str
    contributions: list[dict]  # [{"check", "points", "severity", "confidence", "contribution"}]
    forced_by: str | None = None  # critical check that forced the band, if any

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "band": self.band,
            "forced_by": self.forced_by,
            "contributions": self.contributions,
        }


def risk_score(
    checks: list[CheckResult],
    medium: int = 30,
    high: int = 70,
    points: dict[str, float] | None = None,
) -> Risk:
    table = POINTS | (points or {})
    contributions = []
    for c in checks:
        if not c.fired:
            continue
        p = float(table.get(c.id, DEFAULT_POINTS))
        value = p * c.severity * c.confidence
        contributions.append(
            {
                "check": c.id,
                "points": p,
                "severity": round(c.severity, 3),
                "confidence": round(c.confidence, 3),
                "contribution": round(value, 1),
            }
        )
    contributions.sort(key=lambda x: -x["contribution"])
    score = int(round(min(100.0, sum(x["contribution"] for x in contributions))))
    band = "high" if score >= high else "medium" if score >= medium else "low"
    forced = None
    for c in checks:
        if c.fired and c.id in CRITICAL and c.confidence >= CRITICAL[c.id] and band != "high":
            band, forced = "high", c.id
            break
    return Risk(score, band, contributions, forced)
