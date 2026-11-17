from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

from findoc.localize.align import FeatureMatcher, identify, register, resize_to_width
from findoc.localize.templates import ChequeTemplate, TemplateStore, builtin_store, draw_fields
from findoc.pipeline import process
from findoc.recognize import ocr
from findoc.recognize.fields import crop_fields, ink_mask, read_cheque_fields
from findoc.recognize.parsers import words_to_amount
from findoc.synth.cheque import BANKS, TEMPLATE_SIZE, render_blank, render_cheque
from findoc.synth.dataset import cheque_photo, generate_cheques
from findoc.synth.ocr_noise import OracleOcr, corrupt
from findoc.types import DocType

TODAY = date.today()

# ---------- generator ----------


def test_blank_templates_differ_between_banks() -> None:
    blanks = [render_blank(style) for style in BANKS.values()]
    assert all(b.shape == (TEMPLATE_SIZE[1], TEMPLATE_SIZE[0], 3) for b in blanks)
    assert not np.array_equal(blanks[0], blanks[1])


def test_render_cheque_truth_is_consistent() -> None:
    img, truth = render_cheque(np.random.default_rng(1), bank="ARVL", today=TODAY)
    assert img.shape[:2] == (TEMPLATE_SIZE[1], TEMPLATE_SIZE[0])
    assert words_to_amount(truth.amount_words).value == Decimal(truth.amount)
    assert truth.ifsc.startswith("ARVL0") and len(truth.ifsc) == 11
    assert truth.signed


@pytest.mark.parametrize("flag", ["mismatch", "stale", "postdated", "unsigned"])
def test_fraud_flags_create_the_condition(flag: str) -> None:
    _, t = render_cheque(np.random.default_rng(2), today=TODAY, **{flag: True})
    written = date.fromisoformat(t.date)
    if flag == "mismatch":
        assert words_to_amount(t.amount_words).value != Decimal(t.amount)
    elif flag == "stale":
        assert written < TODAY - timedelta(days=95)
    elif flag == "postdated":
        assert written > TODAY
    else:
        assert not t.signed


def test_generate_cheques_writes_labels(tmp_path: Path) -> None:
    paths = generate_cheques(3, tmp_path, seed=5, fraud_rate=1.0)
    import json

    labels = json.loads((tmp_path / "labels.json").read_text())
    assert len(paths) == 3 and set(labels) == {p.name for p in paths}
    assert all(any(v["fraud"].values()) for v in labels.values())
    assert len(labels[paths[0].name]["page_to_photo"]) == 3


# ---------- templates and alignment ----------


def test_template_store_round_trip(tmp_path: Path) -> None:
    store = builtin_store()
    store.save(tmp_path)
    loaded = TemplateStore.from_dir(tmp_path)
    assert sorted(t.id for t in loaded) == sorted(BANKS)
    t = loaded.get("ARVL")
    assert t.size == TEMPLATE_SIZE
    assert t.field("payee").box == store.get("ARVL").field("payee").box
    with pytest.raises(KeyError):
        t.field("nope")


@pytest.mark.parametrize("bank", sorted(BANKS))
def test_identify_every_bank_from_a_photo(bank: str) -> None:
    photo, truth, _ = cheque_photo(seed=40, bank=bank)
    doc = process_photo(photo)
    assert doc.doc_type is DocType.CHEQUE
    assert doc.template_id == truth.bank
    assert doc.quality["template_margin"] > 1.3


def test_orb_matcher_also_identifies() -> None:
    img, _ = render_cheque(np.random.default_rng(3), bank="DCCN", today=TODAY)
    result = identify(img, builtin_store(), method="orb", min_inliers=20)
    assert result is not None and result.template_id == "DCCN"
    with pytest.raises(ValueError):
        FeatureMatcher("magic")


def test_non_cheque_is_not_matched(synth_photo_path: Path) -> None:
    doc = process(synth_photo_path, ocr=False)  # Phase 1 invoice-like page
    assert doc.doc_type is DocType.UNKNOWN
    assert doc.template_id is None
    assert any("No cheque template matched" in m for m in doc.messages)


def _polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return inter / union


def test_field_boxes_land_on_ground_truth() -> None:
    """Map each template field box into the photo via our estimate and via ground truth."""
    photo, truth, page_to_photo = cheque_photo(seed=21)
    doc = process_photo(photo)
    photo_to_flat = np.array(doc.homography)
    flat_to_tmpl = np.array(doc.template_homography)
    tmpl_to_photo = np.linalg.inv(flat_to_tmpl @ photo_to_flat)
    template = builtin_store().get(doc.template_id)
    for spec in template.fields:
        x0, y0, x1, y1 = spec.box
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
        est = cv2.perspectiveTransform(box, tmpl_to_photo).reshape(-1, 2)
        gt = cv2.perspectiveTransform(box, page_to_photo).reshape(-1, 2)
        assert _polygon_iou(est, gt) > 0.85, spec.name


# ---------- template subtraction and fields ----------


def test_template_subtraction_removes_preprinted_content() -> None:
    template = builtin_store().get("INDV")
    blank_mask = ink_mask(template.reference, template.reference)
    assert blank_mask.mean() < 0.5  # (0-255 scale) essentially nothing on a blank leaf
    filled, _ = render_cheque(np.random.default_rng(4), bank="INDV", today=TODAY)
    crops = crop_fields(filled, template)
    assert crops["payee"].ink_ratio > 0.01
    assert crops["signature"].ink_ratio > 0.006


def test_compact_cells_respaces_date_digits() -> None:
    from findoc.recognize.fields import compact_cells

    template = builtin_store().get("ARVL")
    filled, truth = render_cheque(np.random.default_rng(9), bank="ARVL", today=TODAY)
    spec = template.field("date")
    assert spec.cells == 8
    date_crop = crop_fields(filled, template)["date"].image
    compact = compact_cells(date_crop, spec.cells)
    assert compact.shape[0] == date_crop.shape[0]
    assert compact.shape[1] < 0.75 * date_crop.shape[1]  # wide gaps removed
    ink_cols = (cv2.cvtColor(compact, cv2.COLOR_BGR2GRAY) < 200).any(axis=0)
    runs = np.count_nonzero(np.diff(ink_cols.astype(int)) == 1) + int(ink_cols[0])
    assert runs == len(truth.fields["date"])  # one ink run per digit, in order

    blank = np.full_like(date_crop, 255)
    assert compact_cells(blank, 8) is blank  # nothing written: returned unchanged


def test_read_fields_without_ocr_reports_signature_only() -> None:
    template = builtin_store().get("ARVL")
    filled, _ = render_cheque(np.random.default_rng(6), bank="ARVL", unsigned=True, today=TODAY)
    fields = read_cheque_fields(filled, template, engine=None)
    assert [f.name for f in fields] == ["signature"]
    assert fields[0].value == "absent"


def test_read_fields_with_oracle_ocr() -> None:
    template = builtin_store().get("KVRY")
    filled, truth = render_cheque(np.random.default_rng(8), bank="KVRY", today=TODAY)
    fields = {f.name: f for f in read_cheque_fields(filled, template, OracleOcr(truth.fields))}
    assert fields["amount_words"].normalized == truth.amount
    assert fields["amount_figures"].normalized == truth.amount
    assert fields["date"].normalized == truth.date
    assert fields["ifsc"].normalized == truth.ifsc
    assert fields["account_number"].normalized == truth.account_number
    assert fields["micr"].normalized == truth.micr
    assert fields["payee"].value == truth.payee


def test_corrupt_rate() -> None:
    rng = np.random.default_rng(0)
    text = "TWENTY FIVE THOUSAND ONLY " * 40
    assert corrupt(text, 0.0, rng) == text
    noisy = corrupt(text, 0.1, rng)
    assert noisy != text


def test_draw_fields_and_register() -> None:
    template = builtin_store().get("ARVL")
    out = draw_fields(template.reference, template)
    assert out.shape == template.reference.shape
    small, scale = resize_to_width(template.reference, 812)
    assert small.shape[1] == 812 and scale[0, 0] == pytest.approx(0.5)
    back = register(small, np.linalg.inv(scale), template)
    assert back.shape == template.reference.shape


# ---------- full pipeline with simulated OCR ----------


@pytest.mark.parametrize(
    ("fraud", "expected"),
    [
        (None, set()),
        ("mismatch", {"AMOUNT_MISMATCH"}),
        ("stale", {"DATE_STALE"}),
        ("postdated", {"DATE_POSTDATED"}),
        ("unsigned", {"SIGNATURE_MISSING"}),
    ],
)
def test_pipeline_flags_fraud_with_perfect_ocr(
    fraud: str | None, expected: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    photo, truth, _ = cheque_photo(seed=33, fraud=fraud)
    monkeypatch.setattr(ocr, "get_engine", lambda: OracleOcr(truth.fields))
    doc = process_photo(photo, ocr_on=True)
    assert {c.id for c in doc.checks if c.fired} == expected
    assert doc.template_id == truth.bank


def process_photo(photo: np.ndarray, ocr_on: bool = False):  # type: ignore[no-untyped-def]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cheque.jpg"
        cv2.imwrite(str(path), photo)
        return process(path, ocr=ocr_on)


def test_custom_template_dir_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from findoc.config import get_settings
    from findoc.localize.templates import default_store

    only = TemplateStore([builtin_store().get("ARVL")])
    only.save(tmp_path)
    assert [t.id for t in default_store(tmp_path)] == ["ARVL"]
    assert len(default_store(tmp_path / "missing")) == len(BANKS)
    assert isinstance(next(iter(default_store(None))), ChequeTemplate)
    get_settings.cache_clear()
