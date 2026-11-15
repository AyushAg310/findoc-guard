"""Phase 2 pipeline stage: validation rules on extracted cheque fields."""

from __future__ import annotations

from findoc.fraud.rules import run_cheque_rules
from findoc.types import DocType, Document


def rules_stage(doc: Document) -> Document:
    if doc.doc_type is not DocType.CHEQUE or not doc.fields:
        return doc
    text_read = bool(doc.quality.get("fields_read_with_ocr"))
    doc.checks.extend(run_cheque_rules(doc.fields, text_read=text_read))
    fired = [c.id for c in doc.checks if c.fired]
    doc.messages.append(f"Rule checks: {len(doc.checks)} run, {len(fired)} flagged {fired or ''}")
    return doc
