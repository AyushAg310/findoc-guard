"""Validation rules for GST invoices (Phase 7, SRS 10.5 INVOICE_MATH and FORMAT_INVALID).

All rules work on the normalised invoice (:meth:`findoc.recognize.invoice.Invoice.normalized`)
and, like the cheque rules, report what they could not verify instead of guessing.

* ``GSTIN_INVALID`` (vendor, buyer): format, state code and check character
  (:mod:`findoc.recognize.gstin`). A wrong check character means the number was mistyped,
  misread or invented; a fake vendor GSTIN is a classic way to claim input tax credit.
* ``INVOICE_MATH``: every line's quantity x rate = amount; the amounts add up to the subtotal;
  the tax equals the subtotal times a valid GST rate (0, 0.25, 3, 5, 12, 18, 28 %), with CGST =
  SGST; subtotal + tax = total. Differences up to ``TOLERANCE`` (rounding) are allowed;
  severity grows with the relative error. An invoice says most numbers twice (the subtotal is
  also the sum of the lines; the total is also written in words), so a number that disagrees
  while everything else agrees with its other derivation is treated as an **OCR misread** and
  corrected, not reported as fraud. Without that, one misread digit in 40% of clean invoices
  (at 1% character errors) raised a false alarm; see the Phase 7 guide.
* ``TAX_TYPE_MISMATCH``: CGST + SGST must be charged within one state (same first two GSTIN
  digits), IGST between states. The wrong type is a common error and a way to shift tax
  between the centre and the states.
"""

from __future__ import annotations

from decimal import Decimal

from findoc.recognize import gstin as gst
from findoc.types import CheckResult

TOLERANCE = Decimal("1.00")  # rupees: rounding of each line or tax
GST_RATES = (
    Decimal("0"),
    Decimal("0.25"),
    Decimal("3"),
    Decimal("5"),
    Decimal("12"),
    Decimal("18"),
    Decimal("28"),
)


def _d(v: str | None) -> Decimal | None:
    return None if v is None else Decimal(v)


def _severity(diff: Decimal, base: Decimal) -> float:
    """0.5 for a small relative error, rising to 1.0 at 5% or more."""
    rel = float(abs(diff) / max(abs(base), Decimal("1")))
    return round(min(1.0, 0.5 + 10 * rel), 3)


def check_gstins(fields: dict[str, str | None]) -> list[CheckResult]:
    out = []
    for name, who in (("vendor_gstin", "Vendor"), ("buyer_gstin", "Buyer")):
        raw = fields.get(name)
        if raw is None:
            continue
        valid, why = gst.validate(raw)
        fixes = gst.misread_fixes(raw) if valid is None else []
        if valid is not None:
            reason, conf = f"{who} GSTIN {raw}: valid.", 0.9
        elif fixes:
            # one OCR confusion away from a valid number: more likely misread than invalid
            reason = (
                f"{who} GSTIN {raw}: {why}; one misread character away from {fixes[0]} "
                "(valid), so probably an OCR error: check the printed number."
            )
            conf = 0.3
        else:
            reason, conf = f"{who} GSTIN {raw}: {why}.", 0.9
        out.append(
            CheckResult(
                id="GSTIN_INVALID",
                fired=valid is None,
                severity=0.7 if valid is None else 0.0,
                confidence=conf,
                reason=reason,
                evidence={"party": name.split("_")[0], "gstin": raw, "misread_fixes": fixes},
            )
        )
    return out


DIGIT_CONFUSIONS = {
    "0": "86",
    "1": "7",
    "2": "7",
    "3": "8",
    "4": "9",
    "5": "6",
    "6": "58",
    "7": "12",
    "8": "3069",
    "9": "48",
}


def _variants(value: Decimal) -> list[Decimal]:
    """``value`` with one digit replaced by a digit OCR commonly confuses it with."""
    s = format(value, "f")
    return [
        Decimal(s[:i] + alt + s[i + 1 :])
        for i, ch in enumerate(s)
        for alt in DIGIT_CONFUSIONS.get(ch, "")
    ]


def _one_confusion_fits(q: Decimal, r: Decimal, a: Decimal) -> bool:
    """Would one commonly confused digit in the quantity, the rate or the amount make
    quantity x rate equal the amount?"""
    return (
        any(abs(v * r - a) <= TOLERANCE for v in _variants(q))
        or any(abs(q * v - a) <= TOLERANCE for v in _variants(r))
        or any(abs(q * r - v) <= TOLERANCE for v in _variants(a))
    )


def check_math(inv: dict) -> CheckResult:
    """Lines, subtotal, tax and total, using the invoice's redundancy to tell a misread number
    from a wrong one: every amount can be computed two ways, so a number that disagrees while
    everything else agrees with the other way of computing it was misread, and is corrected
    rather than reported (the correction is listed in the evidence)."""
    f, items = inv["fields"], inv["items"]
    problems: list[str] = []
    misreads: list[str] = []
    sev = 0.0
    sub, total, words = _d(f.get("subtotal")), _d(f.get("total")), _d(f.get("amount_words"))
    cgst, sgst, igst = _d(f.get("cgst")), _d(f.get("sgst")), _d(f.get("igst"))

    # 1. lines: quantity x rate = amount
    amounts = [_d(it.get("amount")) for it in items]
    known = None not in amounts and bool(items)
    lines_ok = True  # every line read and self-consistent: only then is their sum evidence
    for i, it in enumerate(items):
        q, r, a = _d(it.get("qty")), _d(it.get("rate")), amounts[i]
        if None in (q, r, a) or abs(q * r - a) <= TOLERANCE:
            continue
        if known and sub is not None:
            with_qr = sum(amounts) - a + q * r
            if abs(with_qr - sub) <= TOLERANCE * len(items):
                misreads.append(
                    f"line {i + 1} amount read {a:,.2f}, quantity x rate {q * r:,.2f} "
                    "fits the subtotal"
                )
                amounts[i] = q * r
                continue
        if _one_confusion_fits(q, r, a):
            misreads.append(f"line {i + 1}: one misread digit explains {q} x {r} vs {a:,.2f}")
            lines_ok = False
            continue
        lines_ok = False
        problems.append(f"line {i + 1}: {q} x {r} = {q * r:,.2f}, not {a:,.2f}")
        sev = max(sev, _severity(q * r - a, a))

    # 2. tax: CGST = SGST
    if cgst is not None and sgst is not None and abs(cgst - sgst) > TOLERANCE:
        base = sub if sub is not None else (sum(amounts) if known else None)
        fixed = None
        if base:
            for half in (cgst, sgst):
                fits_rate = any(abs(base * rt / 200 - half) <= TOLERANCE for rt in GST_RATES)
                fits_total = total is None or abs(base + 2 * half - total) <= TOLERANCE
                if fits_rate and fits_total:
                    fixed = half
        if fixed is not None:
            misreads.append(f"CGST/SGST read {cgst:,.2f}/{sgst:,.2f}; {fixed:,.2f} fits the rate")
            cgst = sgst = fixed
        else:
            problems.append(f"CGST {cgst:,.2f} and SGST {sgst:,.2f} differ (they must be equal)")
            sev = max(sev, _severity(cgst - sgst, cgst))
    tax = None
    if cgst is not None or sgst is not None or igst is not None:
        tax = sum(x for x in (cgst, sgst, igst) if x is not None)

    # 3. subtotal = sum of lines (only when every line was read and is self-consistent: a row
    # damaged by OCR would otherwise look like a wrong subtotal)
    if known and lines_ok and sub is not None and abs(sum(amounts) - sub) > TOLERANCE * len(items):
        s = sum(amounts)
        if tax is not None and total is not None and abs(s + tax - total) <= TOLERANCE:
            misreads.append(f"subtotal read {sub:,.2f}; the lines ({s:,.2f}) fit the total")
            sub = s
        else:
            problems.append(f"line amounts add to {s:,.2f}, subtotal says {sub:,.2f}")
            sev = max(sev, _severity(s - sub, sub))

    # 4. tax rate
    rate = None
    if tax is not None and sub:
        implied = tax / sub * 100
        rate = min(GST_RATES, key=lambda r: abs(r - implied))
        if abs(sub * rate / 100 - tax) > TOLERANCE * 2:
            problems.append(f"tax {tax:,.2f} is {implied:.2f}% of the subtotal, not a GST rate")
            sev = max(sev, _severity(sub * rate / 100 - tax, tax))

    # 5. total = subtotal + tax (the amount in words settles a disagreement)
    if None not in (total, sub, tax) and abs(sub + tax - total) > TOLERANCE:
        computed = sub + tax
        if words is not None and abs(words - computed) <= TOLERANCE:
            misreads.append(
                f"total read {total:,.2f}; subtotal + tax and the amount in words "
                f"agree on {computed:,.2f}"
            )
        else:
            why = (
                " (the amount in words agrees with the printed total)"
                if words is not None and abs(words - total) <= TOLERANCE
                else ""
            )
            problems.append(f"subtotal + tax = {computed:,.2f}, total says {total:,.2f}{why}")
            sev = max(sev, _severity(computed - total, total))

    verified = sub is not None and total is not None and tax is not None
    evidence = {
        "problems": problems,
        "probable_misreads": misreads,
        "gst_rate": str(rate) if rate is not None else None,
    }
    if problems:
        return CheckResult(
            id="INVOICE_MATH",
            fired=True,
            severity=sev,
            confidence=0.85,
            reason="Invoice arithmetic does not add up: " + "; ".join(problems) + ".",
            evidence=evidence,
        )
    note = f" (probable misreads corrected: {'; '.join(misreads)})" if misreads else ""
    return CheckResult(
        id="INVOICE_MATH" if verified else "INVOICE_UNVERIFIED",
        fired=not verified,
        severity=0.0 if verified else 0.3,
        confidence=0.85 if verified else 1.0,
        reason=(
            f"Lines, subtotal, {rate}% GST and total add up{note}."
            if verified
            else "Could not read the subtotal, tax and total; arithmetic not verified."
        ),
        evidence=evidence,
    )


def check_tax_type(fields: dict[str, str | None]) -> CheckResult | None:
    v, b = fields.get("vendor_gstin"), fields.get("buyer_gstin")
    has_split = fields.get("cgst") is not None or fields.get("sgst") is not None
    has_igst = fields.get("igst") is not None
    if not v or not b or has_split == has_igst:
        return None  # unknown parties, or no tax read (or both: reported by INVOICE_MATH)
    same = v[:2] == b[:2]
    wrong = (same and has_igst) or (not same and has_split)
    where = (
        f"both in {gst.state_of(v) or v[:2]}"
        if same
        else f"{gst.state_of(v) or v[:2]} to {gst.state_of(b) or b[:2]}"
    )
    expected = "CGST + SGST" if same else "IGST"
    charged = "IGST" if has_igst else "CGST + SGST"
    return CheckResult(
        id="TAX_TYPE_MISMATCH",
        fired=wrong,
        severity=0.6 if wrong else 0.0,
        confidence=0.9,
        reason=f"Supply {where}: {expected} expected, {charged} charged"
        + (" (wrong tax type)." if wrong else "."),
        evidence={"intra_state": same, "charged": charged},
    )


def run_invoice_rules(inv: dict) -> list[CheckResult]:
    f = inv["fields"]
    out = check_gstins(f)
    out.append(check_math(inv))
    tt = check_tax_type(f)
    if tt is not None:
        out.append(tt)
    return out
