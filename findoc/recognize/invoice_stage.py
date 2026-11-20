"""Phase 7 pipeline stage: invoice fields, line items and invoice checks (after OCR)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from findoc.config import get_settings
from findoc.fraud.invoice_rules import run_invoice_rules
from findoc.recognize.invoice import Invoice, RuleExtractor, amount_in_words_text
from findoc.types import DocType, Document, ExtractedField

INVOICE_CUES = re.compile(r"tax\s*invoice|gstin|\bhsn\b|\bcgst\b|\bigst\b")


@lru_cache(maxsize=1)
def _layout_extractor(path: Path):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from findoc.recognize.layout import LayoutExtractor

    return LayoutExtractor(path)


def looks_like_invoice(text: str) -> bool:
    """At least two invoice cues ("Tax Invoice", GSTIN, HSN, CGST, IGST) in the OCR text."""
    return len(set(INVOICE_CUES.findall(text.lower()))) >= 2


def extract_invoice(doc: Document) -> Invoice:
    s = get_settings()
    use_layout = s.invoice_extractor == "layoutlmv3" or (
        s.invoice_extractor == "auto" and (s.invoice_model / "config.json").is_file()
    )
    if use_layout:
        img = doc.enhanced if doc.enhanced is not None else doc.flat
        try:
            return _layout_extractor(s.invoice_model).extract(doc.words, img)
        except ImportError as exc:  # transformers not installed: say so and use the rules
            doc.messages.append(f"LayoutLMv3 unavailable ({exc}); rule-based extraction used")
    return RuleExtractor().extract(doc.words)


def invoice_stage(doc: Document) -> Document:
    if doc.doc_type is DocType.CHEQUE or not doc.words:
        return doc
    if doc.doc_type is not DocType.INVOICE:
        if not looks_like_invoice(doc.full_text):
            return doc
        doc.doc_type = DocType.INVOICE
    inv = extract_invoice(doc)
    words_total = amount_in_words_text(doc.words)
    if words_total and "amount_words" not in inv.fields:
        inv.fields["amount_words"] = words_total
    norm = inv.normalized()
    conf = sum(w.confidence for w in doc.words) / len(doc.words)
    doc.fields = [
        ExtractedField(
            name=k,
            value=v,
            raw=v,
            normalized=norm["fields"].get(k),
            method=inv.method,
            confidence=round(conf, 4),
        )
        for k, v in inv.fields.items()
    ]
    doc.line_items = [
        {"row": i + 1, **{k: v for k, v in it.items()}, "normalized": norm["items"][i]}
        for i, it in enumerate(inv.items)
    ]
    checks = run_invoice_rules(norm)
    doc.checks.extend(checks)
    fired = [c.id for c in checks if c.fired]
    doc.messages.append(
        f"Invoice ({inv.method}): {len(inv.fields)} fields, {len(inv.items)} line items; "
        f"checks flagged {fired or 'nothing'}"
    )
    return doc
