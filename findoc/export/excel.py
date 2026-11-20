"""Excel export of processed documents (Phase 7, FR-APP-04).

``write_workbook`` turns result JSON files (``findoc process ... --out result.json``) into one
.xlsx workbook that a finance team can filter and sort:

* **Documents**: one row per document: file, type, status, risk score and band, bank or
  vendor, number, date, total, and how many checks fired;
* **Fields**: every extracted field, as read and as validated, with confidence;
* **Line items**: invoice lines (description, HSN, quantity, rate, amount), validated values;
* **Checks**: every check with its result, severity, confidence and reason.

Numbers are written as numbers (not text) so Excel can add them up; the header row is bold and
frozen, and columns are sized to their content.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

SHEETS = {
    "Documents": [
        "file",
        "type",
        "status",
        "risk_score",
        "risk_band",
        "bank_or_vendor",
        "number",
        "date",
        "total",
        "checks_fired",
    ],
    "Fields": ["file", "field", "value", "validated", "confidence", "method"],
    "Line items": ["file", "row", "description", "hsn", "qty", "rate", "amount"],
    "Checks": ["file", "check", "fired", "severity", "confidence", "reason"],
}


def _num(v: str | None) -> float | str | None:
    if v is None:
        return None
    try:
        return float(Decimal(str(v)))
    except (InvalidOperation, ValueError):
        return v


def rows_for(result: dict) -> dict[str, list[list]]:
    """The rows one result JSON contributes to each sheet."""
    name = Path(result.get("source", "")).name
    fields = {f["name"]: f for f in result.get("fields", [])}

    def val(*names: str) -> str | None:
        for n in names:
            f = fields.get(n)
            if f and (f.get("normalized") or f.get("value")):
                return f.get("normalized") or f.get("value")
        return None

    fired = [c for c in result.get("checks", []) if c.get("fired")]
    risk = result.get("risk") or {}
    out: dict[str, list[list]] = {k: [] for k in SHEETS}
    out["Documents"].append(
        [
            name,
            result.get("doc_type"),
            result.get("status"),
            result.get("risk_score"),
            risk.get("band"),
            result.get("template_id") or val("vendor"),
            val("invoice_no", "micr"),
            val("date"),
            _num(val("total", "amount_figures")),
            len(fired),
        ]
    )
    for f in result.get("fields", []):
        out["Fields"].append(
            [
                name,
                f["name"],
                f.get("value"),
                f.get("normalized"),
                f.get("confidence"),
                f.get("method"),
            ]
        )
    for it in result.get("line_items", []):
        n = it.get("normalized") or {}
        out["Line items"].append(
            [
                name,
                it.get("row"),
                it.get("desc"),
                n.get("hsn"),
                _num(n.get("qty")),
                _num(n.get("rate")),
                _num(n.get("amount")),
            ]
        )
    for c in result.get("checks", []):
        out["Checks"].append(
            [
                name,
                c["id"],
                "FLAG" if c["fired"] else "pass",
                c.get("severity"),
                c.get("confidence"),
                c.get("reason"),
            ]
        )
    return out


def write_workbook(results: list[dict], path: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    sheets = {name: wb.create_sheet(name) for name in SHEETS}
    for name, ws in sheets.items():
        ws.append(SHEETS[name])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "A2"
    for r in results:
        for name, rows in rows_for(r).items():
            for row in rows:
                sheets[name].append(row)
    for ws in sheets.values():
        for i, col in enumerate(ws.iter_cols(values_only=True), 1):
            width = max(len(str(v)) if v is not None else 0 for v in col)
            ws.column_dimensions[get_column_letter(i)].width = min(60, max(8, width + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def load_results(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]
