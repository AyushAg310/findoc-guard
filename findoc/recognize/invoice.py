"""Invoice field extraction from OCR words (Phase 7, FR-EXT-04).

Input: OCR words with boxes on the flat page. Output: header fields (vendor, GSTINs, invoice
number, date, subtotal, taxes, total) and line items (description, HSN, quantity, rate,
amount), each normalised (``normalize_field``) so values can be checked and exported.

Two extractors share everything except the labelling step:

* ``RuleExtractor`` (always available): keyword anchors and page geometry. Values follow their
  keys on the same line ("Invoice No: GST/2026-27/0457"); amounts are the right-most number on
  a "Sub Total" / "CGST" / "Grand Total" line; GSTINs are found by pattern and assigned to the
  buyer when they sit in the buyer's block; the vendor is the largest text near the top; the
  line-item table starts at a header line naming at least three of Description / HSN / Qty /
  Rate / Amount, its columns are placed under the header words, and rows run until the
  subtotal line.
* LayoutLMv3 (:mod:`findoc.recognize.layout`, trained with ``findoc invoice train``) labels
  every word with a field tag, using the text, the box and the page image together; its labels
  go through the same ``assemble`` step.

``split_phrases`` first breaks OCR phrases into words (EasyOCR returns phrases such as
"Invoice No: GST/2026-27/0457"), dividing the box in proportion to character counts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from findoc.recognize import gstin as gst
from findoc.recognize.parsers import DIGIT_FIXES, parse_amount_figures, words_to_amount
from findoc.types import BBox, OcrWord

HEADER_FIELDS = (
    "vendor",
    "vendor_gstin",
    "buyer",
    "buyer_gstin",
    "invoice_no",
    "date",
    "subtotal",
    "cgst",
    "sgst",
    "igst",
    "total",
)
ITEM_FIELDS = ("desc", "hsn", "qty", "rate", "amount")
LABEL_TO_FIELD = {
    "VENDOR": "vendor",
    "VENDOR_GSTIN": "vendor_gstin",
    "BUYER": "buyer",
    "BUYER_GSTIN": "buyer_gstin",
    "INV_NO": "invoice_no",
    "DATE": "date",
    "SUBTOTAL": "subtotal",
    "CGST": "cgst",
    "SGST": "sgst",
    "IGST": "igst",
    "TOTAL": "total",
    "ITEM_DESC": "desc",
    "ITEM_HSN": "hsn",
    "ITEM_QTY": "qty",
    "ITEM_RATE": "rate",
    "ITEM_AMOUNT": "amount",
}
GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z0-9]{10}[0-9A-Z]Z[0-9A-Z]\b")


@dataclass
class Invoice:
    fields: dict[str, str] = field(default_factory=dict)  # raw text per header field
    items: list[dict[str, str]] = field(default_factory=list)  # raw text per item column
    method: str = "rules"

    def normalized(self) -> dict:
        return {
            "fields": {k: normalize_field(k, v) for k, v in self.fields.items()},
            "items": [{k: normalize_field(k, v) for k, v in it.items()} for it in self.items],
        }


# --------------------------------------------------------------------------- normalisation

MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


def parse_invoice_date(text: str) -> date | None:
    t = text.strip().replace(",", " ")
    m = re.search(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})", t)
    if m:
        d, mo, y = (int(x) for x in m.groups())
    else:
        m = re.search(r"(\d{1,2})\s*([A-Za-z]{3,9})\s*(\d{2,4})", t)
        if not m or m.group(2)[:3].lower() not in MONTHS:
            return None
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], int(m.group(3))
    y = y + 2000 if y < 100 else y
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def parse_money(text: str):  # noqa: ANN201 - Decimal | None
    """An invoice amount. Invoices print paise, so the amount ends in a separator and two digits;
    Indian grouping never ends in a two-digit group (1,29,492.00), so a final ``,dd`` is a
    decimal point misread as a comma. If OCR glued two numbers into one cell, the last counts."""
    tokens = [tok for tok in text.split() if any(ch.isdigit() for ch in tok.translate(DIGIT_FIXES))]
    if not tokens:
        return None
    t = tokens[-1].translate(DIGIT_FIXES)
    t = re.sub(r"[,.](\d{2})(/-)?$", r".\1", t)
    return parse_amount_figures(t)


def normalize_field(name: str, text: str | None) -> str | None:
    """Canonical value of a field, or None if it does not parse."""
    if text is None:
        return None
    t = text.strip()
    if name in ("vendor_gstin", "buyer_gstin"):
        g = gst.clean(t)
        return g if gst.GSTIN_RE.match(g) else None  # format only: the check is a fraud rule
    if name == "amount_words":
        v = words_to_amount(t).value
        return None if v is None else str(v)
    if name in ("subtotal", "cgst", "sgst", "igst", "total", "rate", "amount"):
        v = parse_money(t)
        return None if v is None else str(v)
    if name == "qty":
        v = parse_amount_figures(t)
        if v is None:
            return None
        return str(int(v)) if v == v.to_integral() else str(v.normalize())
    if name == "date":
        d = parse_invoice_date(t)
        return d.isoformat() if d else None
    if name == "hsn":
        digits = re.sub(r"\D", "", t)
        return digits if 4 <= len(digits) <= 8 else None
    if name == "invoice_no":
        return re.sub(r"\s+", "", t).upper() or None
    return re.sub(r"\s+", " ", t).strip().lower() or None  # names


# ------------------------------------------------------------------------------- geometry


def split_phrases(words: list[OcrWord]) -> list[OcrWord]:
    """OCR phrases -> words, each with a share of the phrase box proportional to its length."""
    out = []
    for w in words:
        parts = w.text.split()
        if len(parts) <= 1:
            out.append(w)
            continue
        total = sum(len(p) for p in parts) + len(parts) - 1
        x0, x1 = w.bbox.x0, w.bbox.x1
        step = (x1 - x0) / max(total, 1)
        pos = 0
        for p in parts:
            a = x0 + pos * step
            out.append(
                OcrWord(
                    text=p,
                    confidence=w.confidence,
                    bbox=BBox(x0=a, y0=w.bbox.y0, x1=a + len(p) * step, y1=w.bbox.y1),
                )
            )
            pos += len(p) + 1
    return out


def lines_of(words: list[OcrWord], tol: float = 0.55) -> list[list[OcrWord]]:
    """Group words into text lines (vertical centre within ``tol`` x height), left to right."""
    items = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2)
    lines: list[list[OcrWord]] = []
    for w in items:
        cy = (w.bbox.y0 + w.bbox.y1) / 2
        hgt = max(1.0, w.bbox.y1 - w.bbox.y0)
        if lines:
            ref = lines[-1]
            rcy = sum((v.bbox.y0 + v.bbox.y1) / 2 for v in ref) / len(ref)
            if abs(cy - rcy) < tol * hgt:
                ref.append(w)
                continue
        lines.append([w])
    return [sorted(ln, key=lambda w: w.bbox.x0) for ln in lines]


def segments(line: list[OcrWord], gap: float = 2.5) -> list[list[OcrWord]]:
    """A text line cut where the gap between words exceeds ``gap`` x text height (columns)."""
    out = [[line[0]]]
    for w in line[1:]:
        h = max(1.0, out[-1][-1].bbox.y1 - out[-1][-1].bbox.y0)
        if w.bbox.x0 - out[-1][-1].bbox.x1 > gap * h:
            out.append([w])
        else:
            out[-1].append(w)
    return out


def _gstins(line: list[OcrWord]) -> list[str]:
    """GSTIN-shaped strings on a line; OCR may split one across up to three tokens."""
    toks = [re.sub(r"[^0-9A-Za-z]", "", w.text) for w in line]
    found = []
    for i in range(len(toks)):
        cand = ""
        for j in range(i, min(i + 3, len(toks))):
            cand += toks[j]
            if len(cand) == 15:
                g = gst.clean(cand)
                if GSTIN_PATTERN.match(g) and g not in found:
                    found.append(g)
            if len(cand) >= 15:
                break
    return found


def _text(ws: list[OcrWord]) -> str:
    return " ".join(w.text for w in ws)


# -------------------------------------------------------------------------------- assembly


def assemble(labelled: list[tuple[OcrWord, str]]) -> Invoice:
    """Words with field labels (``LABEL_TO_FIELD`` keys or "O") -> an Invoice. Header fields:
    the labelled words in reading order (for repeated GSTIN-like fields, the first group). Line
    items: item-labelled words grouped by text line, one row per line with a description or
    amount."""
    inv = Invoice(method="labels")
    header: dict[str, list[OcrWord]] = {}
    item_words: list[tuple[OcrWord, str]] = []
    for w, lab in labelled:
        name = LABEL_TO_FIELD.get(lab)
        if name is None:
            continue
        if name in ITEM_FIELDS:
            item_words.append((w, name))
        else:
            header.setdefault(name, []).append(w)
    for name, ws in header.items():
        lines = lines_of(ws)
        # a field is one phrase: keep its first line (a repeated label elsewhere is noise)
        inv.fields[name] = _text(lines[0])
    if item_words:
        by_line = lines_of([w for w, _ in item_words])
        lab_of = {id(w): n for w, n in item_words}
        for ln in by_line:
            row: dict[str, list[str]] = {}
            for w in ln:
                row.setdefault(lab_of[id(w)], []).append(w.text)
            if "desc" in row or "amount" in row:
                inv.items.append({k: " ".join(v) for k, v in row.items()})
    return inv


# --------------------------------------------------------------------------------- scoring


def field_counts(pred: dict, truth: dict) -> tuple[int, int, int, int, int]:
    """Scoring against the truth (both normalised): (correct header fields, header fields
    predicted, header fields in the truth, correct line-item cells, line-item cells in the
    truth). Line items are compared row by row in order."""
    pf = {k: v for k, v in pred["fields"].items() if k in HEADER_FIELDS}
    tf = truth["fields"]
    tp = sum(1 for k, v in pf.items() if v is not None and tf.get(k) == v)
    npred = sum(1 for v in pf.values() if v is not None)
    cells = ok = 0
    for i, tr in enumerate(truth["items"]):
        pr = pred["items"][i] if i < len(pred["items"]) else {}
        for k, v in tr.items():
            cells += 1
            ok += pr.get(k) == v
    return tp, npred, len(tf), ok, cells


def summarise(counts: Sequence[float]) -> dict:
    tp, npred, ntrue, ok, cells = counts
    p, r = tp / max(npred, 1), tp / max(ntrue, 1)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
        "item_cell_accuracy": round(float(ok / max(cells, 1)), 4),
    }


# ---------------------------------------------------------------------------- rule-based

KEYS = {
    "invoice_no": [r"invoice\s*(no|number|#)", r"bill\s*no", r"inv\.?\s*no"],
    "date": [r"invoice\s*date", r"\bdated?\b"],
    "subtotal": [r"sub\s*-?\s*total", r"taxable\s*(value|amount)", r"total\s*taxable"],
    "cgst": [r"\bcgst\b"],
    "sgst": [r"\bsgst\b|\butgst\b"],
    "igst": [r"\bigst\b"],
    "total": [
        r"grand\s*total",
        r"total\s*amount",
        r"invoice\s*total",
        r"amount\s*payable",
        r"net\s*amount",
    ],
}
BUYER_KEYS = r"bill\s*to|buyer|receiver|billed\s*to|customer|consignee"
NUMBER = re.compile(r"^[₹Rs.]*\d[\d,]*(\.\d{1,2})?/?-?$")
LETTER_FOR_DIGIT = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "6": "g", "2": "z"})


def numeric(text: str) -> bool:
    """Looks like a number once typical OCR letter-for-digit confusions are undone, and has
    mostly digits to begin with (so words like 'Sub' do not count)."""
    t = text.replace(" ", "")
    digits = sum(ch.isdigit() for ch in t)
    return digits >= max(1, len(t) // 2) and bool(NUMBER.match(t.translate(DIGIT_FIXES)))


def _keytext(text: str) -> str:
    """Lower case with digit-for-letter confusions undone, for matching key words."""
    return text.lower().translate(LETTER_FOR_DIGIT)


def _key_match(line: list[OcrWord], pattern: str) -> int | None:
    """Index of the first word after the key phrase on this line, or None."""
    text = ""
    ends = []
    for w in line:
        text += (" " if text else "") + w.text.lower()
        ends.append(len(text))
    m = re.search(pattern, _keytext(text))
    if not m:
        return None
    end = m.end()
    for i, e in enumerate(ends):
        if e >= end:
            return i + 1
    return len(line)


def _value_after(line: list[OcrWord], start: int) -> list[OcrWord]:
    """Words after a key, skipping a ':' and stopping at a wide gap (the next column)."""
    ws = [w for w in line[start:] if w.text not in (":", "-", "#")]
    if not ws:
        return []
    out = [ws[0]]
    h = max(1.0, ws[0].bbox.y1 - ws[0].bbox.y0)
    for w in ws[1:]:
        if w.bbox.x0 - out[-1].bbox.x1 > 2.5 * h:
            break
        out.append(w)
    return out


def amount_in_words_text(words: list[OcrWord]) -> str | None:
    """The total written in words ("Rupees Five Lakh ... Only"): the line after an "Amount in
    words" label, or a line starting with "Rupees". Both extractors add it: the invoice checks
    use it to tell a misread total from a wrong one."""
    lines = lines_of(split_phrases(words))
    for i, ln in enumerate(lines):
        low = _keytext(_text(ln))
        if re.search(r"amount\s*(chargeable\s*)?in\s*words", low):
            rest = _text(ln).split(":", 1)[1].strip() if ":" in _text(ln) else ""
            if len(rest.split()) >= 2:
                return rest
            if i + 1 < len(lines):
                return _text(lines[i + 1])
        if low.startswith("rupees") and len(ln) >= 3:
            return _text(ln)
    return None


class RuleExtractor:
    def extract(self, words: list[OcrWord]) -> Invoice:
        words = split_phrases(words)
        lines = lines_of(words)
        inv = Invoice(method="rules")
        page_h = max((w.bbox.y1 for w in words), default=1.0)

        # key: value fields and amount lines
        sub_line = None
        for li, ln in enumerate(lines):
            low = _text(ln).lower()
            for name, pats in KEYS.items():
                if name in inv.fields:
                    continue
                for pat in pats:
                    start = _key_match(ln, pat)
                    if start is None:
                        continue
                    if name in ("subtotal", "cgst", "sgst", "igst", "total"):
                        nums = [w for w in ln[start:] if numeric(w.text)]
                        if nums and "words" not in low:
                            inv.fields[name] = nums[-1].text
                            if name == "subtotal":
                                sub_line = li
                    else:
                        val = _value_after(ln, start)
                        if val and not (name == "date" and parse_invoice_date(_text(val)) is None):
                            inv.fields[name] = _text(val)
                    break

        # GSTINs: by pattern; the one in the buyer's block is the buyer's
        buyer_zone_start = None
        for li, ln in enumerate(lines):
            if re.search(BUYER_KEYS, _text(ln).lower()):
                buyer_zone_start = li
                break
        for li, ln in enumerate(lines):
            low = _text(ln).lower()
            is_buyer = bool(re.search(BUYER_KEYS, low)) or (
                buyer_zone_start is not None and buyer_zone_start <= li <= buyer_zone_start + 4
            )
            for g in _gstins(ln):
                key = "buyer_gstin" if is_buyer else "vendor_gstin"
                if key in inv.fields:
                    key = "buyer_gstin" if key == "vendor_gstin" else "vendor_gstin"
                inv.fields.setdefault(key, g)

        # vendor: the tallest text segment in the top quarter that isn't a title or a label
        top = [seg for ln in lines if ln[0].bbox.y0 < 0.25 * page_h for seg in segments(ln)]
        skip = r"invoice|gstin|bill|date|no\.|tax|" + BUYER_KEYS
        cands = [seg for seg in top if not re.search(skip, _text(seg).lower())]
        if cands:
            best = max(cands, key=lambda seg: sum(w.bbox.y1 - w.bbox.y0 for w in seg) / len(seg))
            inv.fields["vendor"] = _text(best)

        # buyer: after a buyer key on its line, else the next line
        if buyer_zone_start is not None:
            ln = lines[buyer_zone_start]
            start = _key_match(ln, BUYER_KEYS)
            after = [
                w
                for w in _value_after(ln, start or len(ln))
                if not GSTIN_PATTERN.match(gst.clean(w.text))
            ]
            if after and "gstin" not in _text(after).lower():
                inv.fields["buyer"] = _text(after)
            elif buyer_zone_start + 1 < len(lines):
                nxt = lines[buyer_zone_start + 1]
                inv.fields["buyer"] = _text(_value_after(nxt, 0))

        inv.items = self._table(lines, sub_line)
        return inv

    def _table(self, lines: list[list[OcrWord]], end: int | None) -> list[dict[str, str]]:
        """Line items under the table header, up to the subtotal line."""
        head = None
        for i, ln in enumerate(lines):
            found = {COLUMN_NAMES[k] for k in (_norm(w.text) for w in ln) if k in COLUMN_NAMES}
            if len(found) >= 3:
                head = i
                break
        if head is None:
            return []
        cols: dict[str, tuple[float, float]] = {}
        for w in lines[head]:
            k = _norm(w.text)
            if k in COLUMN_NAMES and COLUMN_NAMES[k] not in cols:
                cols[COLUMN_NAMES[k]] = (w.bbox.x0, w.bbox.x1)
        numeric_cols = {k: v for k, v in cols.items() if k in ("qty", "rate", "amount")}
        text_cols = sorted(((v[0], k) for k, v in cols.items() if k in ("desc", "hsn")))
        stop_re = "|".join(KEYS["subtotal"] + KEYS["total"])
        rows = []
        for ln in lines[head + 1 : end if end is not None else len(lines)]:
            if re.search(stop_re, _keytext(_text(ln))):
                break
            row: dict[str, list[str]] = {}
            for w in ln:
                cx = (w.bbox.x0 + w.bbox.x1) / 2
                name = None
                if numeric_cols and numeric(w.text):
                    # numbers are right-aligned under their header: nearest right edge
                    k, (_, x1) = min(numeric_cols.items(), key=lambda kv: abs(w.bbox.x1 - kv[1][1]))
                    if abs(w.bbox.x1 - x1) < 0.6 * (x1 - min(v[0] for v in cols.values())):
                        name = k
                if name is None or (name == "qty" and "hsn" in cols and cx < cols["qty"][0] - 20):
                    # text (or the HSN code): the text column whose left edge it follows
                    left = [k for x0, k in text_cols if cx >= x0 - 25]
                    name = left[-1] if left else None
                if name is not None:
                    row.setdefault(name, []).append(w.text)
            if any(numeric(t) for t in row.get("amount", [])):
                rows.append({k: " ".join(v) for k, v in row.items()})
        return rows


COLUMN_NAMES = {
    "description": "desc",
    "item": "desc",
    "particulars": "desc",
    "hsn": "hsn",
    "hsn/sac": "hsn",
    "sac": "hsn",
    "qty": "qty",
    "quantity": "qty",
    "rate": "rate",
    "price": "rate",
    "amount": "amount",
    "value": "amount",
}


def _norm(t: str) -> str:
    return _keytext(t).strip(".:")
