"""Phase 7: GSTIN, synthetic invoices, extraction, invoice checks, classifier, Excel export."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from findoc.cli import app
from findoc.config import get_settings
from findoc.fraud.invoice_rules import check_tax_type, run_invoice_rules
from findoc.fraud.risk import risk_score
from findoc.recognize import gstin as gst
from findoc.recognize.invoice import (
    RuleExtractor,
    assemble,
    field_counts,
    lines_of,
    normalize_field,
    parse_invoice_date,
    split_phrases,
    summarise,
)
from findoc.recognize.invoice_stage import invoice_stage, looks_like_invoice
from findoc.recognize.ocr import words_to_text
from findoc.synth.invoice import (
    FRAUDS,
    LAYOUTS,
    PAGE,
    generate_invoices,
    render_invoice,
    simulate_ocr,
    truth_normalized,
)
from findoc.synth.receipt import render_receipt
from findoc.types import BBox, DocType, Document, InputKind, OcrWord

runner = CliRunner()


def ocr_of(words: list, seed: int = 0, cer: float = 0.0) -> list[OcrWord]:
    return simulate_ocr(words, np.random.default_rng(seed), cer=cer)


# ------------------------------------------------------------------------------------ GSTIN


def test_gstin_validation() -> None:
    assert gst.validate("27AAPFU0939F1ZV") == ("27AAPFU0939F1ZV", "valid")  # published example
    assert gst.validate("29AAGCB7383J1Z4")[0] == "29AAGCB7383J1Z4"
    assert gst.validate("27AAPFU0939F1ZW")[0] is None  # wrong check character
    assert "should be V" in gst.validate("27AAPFU0939F1ZW")[1]
    assert gst.validate("27-AAPFU-O939F-1ZV")[0] == "27AAPFU0939F1ZV"  # O read for 0, dashes
    assert "state code" in gst.validate("55AAPFU0939F1ZV")[1]
    assert "format" in gst.validate("27AAPFU0939")[1]
    g = gst.make("08", "ABCDE1234F", "2")
    assert gst.validate(g)[0] == g and gst.state_of(g) == "Rajasthan"


# ---------------------------------------------------------------------------- synthesis


@pytest.mark.parametrize("layout", LAYOUTS)
def test_render_invoice_layouts(layout: str) -> None:
    page, truth, words = render_invoice(np.random.default_rng(1), layout)
    assert page.shape == (PAGE[1], PAGE[0], 3) and truth.layout == layout
    labels = {w.label for w in words}
    assert {
        "VENDOR",
        "VENDOR_GSTIN",
        "BUYER_GSTIN",
        "INV_NO",
        "DATE",
        "TOTAL",
        "ITEM_AMOUNT",
    } <= labels
    assert gst.validate(truth.vendor_gstin)[0] and gst.validate(truth.buyer_gstin)[0]
    assert (truth.cgst is None) == (truth.igst is not None)
    for w in words:
        x0, y0, x1, y1 = w.box
        assert 0 <= x0 < x1 <= PAGE[0] and 0 <= y0 < y1 <= PAGE[1]


def test_fraud_variants_break_what_they_should() -> None:
    from decimal import Decimal

    for kind in FRAUDS:
        _, t, _ = render_invoice(np.random.default_rng(3), "classic", kind)
        sub, total = Decimal(t.subtotal), Decimal(t.total)
        tax = sum(Decimal(x) for x in (t.cgst, t.sgst, t.igst) if x)
        lines_ok = all(abs(i.qty * Decimal(i.rate) - Decimal(i.amount)) < 1 for i in t.items)
        assert (sub + tax == total) == (kind != "total")
        assert lines_ok == (kind != "lineitem")
        assert (t.cgst is not None) == (t.intra_state != (kind == "taxsplit"))
        assert (gst.validate(t.vendor_gstin)[0] is None) == (kind == "gstin")


def test_generate_invoices_and_receipt(tmp_path: Path) -> None:
    paths = generate_invoices(2, tmp_path, seed=4, fraud_rate=0.5)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    labels = json.loads((tmp_path / "labels.json").read_text())
    assert set(labels) == {p.name for p in paths} and labels[paths[0].name]["words"]
    r = render_receipt(np.random.default_rng(0))
    assert r.shape[0] > 2 * r.shape[1]


# --------------------------------------------------------------------------- extraction


def test_normalisation() -> None:
    assert parse_invoice_date("21 Sep 2026") == date(2026, 9, 21)
    assert parse_invoice_date("05.04.25") == date(2025, 4, 5)
    assert parse_invoice_date("31/02/2026") is None
    assert normalize_field("total", "Rs. 1,29,492.00") == "129492.00"
    assert normalize_field("qty", "30") == "30"
    assert normalize_field("hsn", "HSN 8414") == "8414" and normalize_field("hsn", "12") is None
    assert normalize_field("invoice_no", "gst/2026-27/ 0457") == "GST/2026-27/0457"
    assert normalize_field("vendor_gstin", "not a gstin") is None
    assert normalize_field("vendor", "  Thar   Logistics ") == "thar logistics"
    assert normalize_field("date", None) is None


def test_split_phrases_and_lines() -> None:
    w = OcrWord(text="Invoice No: GST/1", confidence=0.9, bbox=BBox(x0=0, y0=0, x1=170, y1=20))
    parts = split_phrases([w])
    assert [p.text for p in parts] == ["Invoice", "No:", "GST/1"]
    assert parts[0].bbox.x0 == 0 and parts[-1].bbox.x1 == pytest.approx(170)
    assert len(lines_of(parts)) == 1


@pytest.mark.parametrize("layout", LAYOUTS)
def test_assemble_true_labels_gives_truth(layout: str) -> None:
    for seed in range(3):
        _, truth, words = render_invoice(np.random.default_rng((seed, 7)), layout)
        ocr = [
            OcrWord(
                text=w.text,
                confidence=1.0,
                bbox=BBox(x0=w.box[0], y0=w.box[1], x1=w.box[2], y1=w.box[3]),
            )
            for w in words
        ]
        inv = assemble([(o, w.label) for o, w in zip(ocr, words, strict=True)])
        assert inv.normalized() == truth_normalized(truth)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rule_extractor_on_clean_ocr(layout: str) -> None:
    total = np.zeros(5)
    for seed in range(4):
        _, truth, words = render_invoice(np.random.default_rng((seed, 11)), layout)
        pred = RuleExtractor().extract(ocr_of(words, seed)).normalized()
        total += field_counts(pred, truth_normalized(truth))
    s = summarise(total)
    assert s["f1"] >= 0.95 and s["item_cell_accuracy"] >= 0.95


# -------------------------------------------------------------------------------- checks


def test_invoice_rules_catch_each_fraud() -> None:
    expected = {
        "total": "INVOICE_MATH",
        "lineitem": "INVOICE_MATH",
        "taxsplit": "TAX_TYPE_MISMATCH",
        "gstin": "GSTIN_INVALID",
    }
    for kind in (None, *FRAUDS):
        _, truth, words = render_invoice(np.random.default_rng(21), "banner", kind)
        fired = {c.id for c in run_invoice_rules(truth_normalized(truth)) if c.fired}
        assert fired == ({expected[kind]} if kind else set())


def test_invoice_rules_edge_cases() -> None:
    unread = run_invoice_rules({"fields": {"vendor": "x"}, "items": []})
    assert [c.id for c in unread] == ["INVOICE_UNVERIFIED"] and unread[0].fired
    odd = {
        "fields": {"subtotal": "1000.00", "cgst": "90.00", "sgst": "80.00", "total": "1170.00"},
        "items": [],
    }
    math = next(c for c in run_invoice_rules(odd) if c.id == "INVOICE_MATH")
    assert math.fired and "must be equal" in math.reason  # 90 + 80 = 170 fits the total
    misread = {
        "fields": {"subtotal": "1000.00", "cgst": "90.00", "sgst": "30.00", "total": "1180.00"},
        "items": [],
    }
    fixed = next(c for c in run_invoice_rules(misread) if c.id == "INVOICE_MATH")
    assert not fixed.fired and fixed.evidence["probable_misreads"]  # 90 fits rate and total
    words = {
        "fields": {
            "subtotal": "1000.00",
            "igst": "180.00",
            "total": "1130.00",
            "amount_words": "1180.00",
        },
        "items": [],
    }
    assert not next(c for c in run_invoice_rules(words) if c.id == "INVOICE_MATH").fired
    words["fields"]["amount_words"] = "1130.00"  # words agree with the wrong total: genuine
    assert next(c for c in run_invoice_rules(words) if c.id == "INVOICE_MATH").fired
    rate = {"fields": {"subtotal": "1000.00", "igst": "150.00", "total": "1150.00"}, "items": []}
    assert next(c for c in run_invoice_rules(rate) if c.id == "INVOICE_MATH").fired
    ok = check_tax_type({"vendor_gstin": "08X", "buyer_gstin": "27Y", "igst": "1"})
    assert ok is not None and not ok.fired
    assert check_tax_type({"vendor_gstin": "08X", "buyer_gstin": "08Y"}) is None  # no tax read
    assert risk_score([c for c in run_invoice_rules(odd) if c.fired]).band != "low"


def test_invoice_stage(tmp_path: Path) -> None:
    _, truth, words = render_invoice(np.random.default_rng(5), "compact", "total")
    doc = Document(source=tmp_path / "x.jpg", kind=InputKind.IMAGE)
    doc.words = ocr_of(words)
    doc.full_text = words_to_text(doc.words)
    assert looks_like_invoice(doc.full_text) and not looks_like_invoice("Dear Sir, thank you")
    doc = invoice_stage(doc)
    assert doc.doc_type is DocType.INVOICE and doc.line_items
    assert {f.name: f.normalized for f in doc.fields}["total"] == truth_normalized(truth)["fields"][
        "total"
    ]
    assert any(c.id == "INVOICE_MATH" and c.fired for c in doc.checks)
    letter = Document(source=tmp_path / "y.jpg", kind=InputKind.IMAGE)
    letter.words = [OcrWord(text="Hello", confidence=1, bbox=BBox(x0=0, y0=0, x1=5, y1=5))]
    letter.full_text = "Hello"
    assert invoice_stage(letter).doc_type is DocType.UNKNOWN


# ---------------------------------------------------------------------------- classifier


def test_doctype_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from findoc.classify.doctype import CLASSES, DocTypeModel, page_features, synthetic_page, train
    from findoc.classify.stages import classify_stage

    assert page_features(synthetic_page("receipt", 0)).shape == (641,)
    out = tmp_path / "doctype.pkl"
    model = train(out, per_class=8, val_per_class=3, workers=1, log=lambda *_: None)
    assert set(model.meta["confusion"]) == set(CLASSES) and 0 <= model.meta["accuracy"] <= 1
    label, conf, probs = DocTypeModel.load(out).predict(synthetic_page("invoice", 77))
    assert label in CLASSES and probs[label] == conf

    monkeypatch.setenv("FINDOC_DOCTYPE_MODEL", str(out))
    get_settings.cache_clear()
    doc = Document(source=tmp_path / "z.jpg", kind=InputKind.IMAGE)
    doc.flat = synthetic_page("invoice", 78)
    doc = classify_stage(doc)
    assert "doc_type_confidence" in doc.quality and any("Document type" in m for m in doc.messages)


# -------------------------------------------------------------------------------- export


def test_excel_export(tmp_path: Path) -> None:
    from openpyxl import load_workbook

    _, truth, words = render_invoice(np.random.default_rng(8), "classic")
    doc = Document(source=tmp_path / "inv.jpg", kind=InputKind.IMAGE)
    doc.words = ocr_of(words)
    doc.full_text = words_to_text(doc.words)
    doc = invoice_stage(doc)
    res = tmp_path / "inv.json"
    res.write_text(json.dumps(doc.result()))
    out = tmp_path / "out.xlsx"
    r = runner.invoke(app, ["export", str(res), "--out", str(out)])
    assert r.exit_code == 0, r.output
    wb = load_workbook(out)
    assert wb.sheetnames == ["Documents", "Fields", "Line items", "Checks"]
    docs = list(wb["Documents"].values)
    assert docs[1][0] == "inv.jpg" and docs[1][1] == "invoice"
    assert docs[1][8] == pytest.approx(float(truth.total))  # a number, not text
    assert wb["Line items"].max_row == len(truth.items) + 1


def test_synth_invoice_cli(tmp_path: Path) -> None:
    r = runner.invoke(app, ["synth", "--kind", "invoice", "--n", "1", "-o", str(tmp_path)])
    assert r.exit_code == 0, r.output and (tmp_path / "labels.json").exists()


# ------------------------------------------------------------------------------ LayoutLMv3


def test_layoutlmv3_tiny_train_and_extract(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from findoc.recognize.layout import BIO, LayoutExtractor, bio_tags, synthetic_sample, train

    _, _, words = render_invoice(np.random.default_rng(0), "classic")
    tags = bio_tags(words)
    assert tags.count("B-VENDOR") == 1 and all(t in BIO for t in tags)
    rec = train(
        tmp_path / "lm",
        base="tiny",
        samples=8,
        val_samples=2,
        epochs=1,
        batch_size=4,
        log=lambda *_: None,
    )
    assert rec["history"] and (tmp_path / "lm" / "config.json").exists()
    img, ws, _, _ = synthetic_sample(3)
    inv = LayoutExtractor(tmp_path / "lm").extract(ws, img)
    assert inv.method == "layoutlmv3"


def test_gstin_position_repairs() -> None:
    g = gst.make("19", "JOYFS6682F", "1")  # 19JOYFS6682F1Z?
    assert gst.validate(g[:13] + "2" + g[14])[0] == g  # "1Z" read as "12": 14th is always Z
    assert gst.validate(g[:12] + "I" + g[13:])[0] == g  # entity 1 read as I
    q = gst.make("33", "BQPFV7973X", "1")
    o = q[:3] + "O" + q[4:]  # EasyOCR reads Q as O
    assert gst.validate(o)[0] is None and gst.misread_fixes(o) == [q]


def test_gstin_second_look() -> None:
    import numpy as np

    from findoc.recognize.invoice_stage import reread_gstin, word_containing
    from findoc.types import BBox, OcrWord

    true = gst.make("29", "ORJPV2195L", "1")
    first = true[:5] + "O" + true[6:]  # one wrong letter, invalid
    words = [
        OcrWord(text="Bill To:", confidence=0.9, bbox=BBox(x0=0, y0=0, x1=50, y1=10)),
        OcrWord(text=f"GSTIN: {first}", confidence=0.6, bbox=BBox(x0=0, y0=20, x1=200, y1=40)),
    ]
    w = word_containing(words, first)
    assert w is words[1]

    class Engine:
        def __init__(self, text: str) -> None:
            self.text = text

        def read_line(self, img: np.ndarray, allowlist: str | None = None) -> tuple[str, float]:
            assert allowlist and img.shape[0] > 40  # enlarged crop
            return self.text, 0.9

    img = np.full((100, 300, 3), 255, np.uint8)
    assert reread_gstin(Engine(f"GSTIN {true}"), img, w.bbox, first) == true
    assert reread_gstin(Engine(first), img, w.bbox, first) is None  # same misread: stays
    other = gst.make("07", "ABCDE1234F", "1")  # valid but a different number: not taken
    assert reread_gstin(Engine(other), img, w.bbox, first) is None
