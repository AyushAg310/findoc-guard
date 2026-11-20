"""Phase 6: risk score, tamper detection, duplicate index, signature comparison."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from findoc.cli import app
from findoc.config import get_settings
from findoc.fraud.duplicates import DuplicateIndex, ink_features, verify
from findoc.fraud.risk import risk_score
from findoc.fraud.signature import SignatureStore, describe, distance, normalise
from findoc.fraud.tamper import (
    BLOCK,
    FEATURE_NAMES,
    TamperModel,
    block_features,
    classical_heatmap,
    field_scores,
    load_model,
    page_mask,
    to_flat,
    upsample,
)
from findoc.io import save_image
from findoc.pipeline import process
from findoc.synth.cheque import render_cheque
from findoc.synth.dataset import CHEQUE_FILL, CHEQUE_PHOTO_SIZE
from findoc.synth.photo import photograph
from findoc.synth.signature import Signer
from findoc.synth.tamper import KINDS, box_quad, forged_sample, tamper
from findoc.types import CheckResult

runner = CliRunner()


def check(id_: str, fired: bool = True, severity: float = 1.0, conf: float = 1.0) -> CheckResult:
    return CheckResult(id=id_, fired=fired, severity=severity, confidence=conf)


# ------------------------------------------------------------------------------- risk score


def test_risk_score_contributions_and_bands() -> None:
    r = risk_score([check("DATE_POSTDATED", severity=0.6, conf=0.9), check("IFSC_INVALID", False)])
    assert r.score == round(60 * 0.6 * 0.9) and r.band == "medium"
    assert [c["check"] for c in r.contributions] == ["DATE_POSTDATED"]
    assert risk_score([check("MICR_INVALID", severity=0.3, conf=0.8)]).band == "low"
    capped = risk_score([check("DATE_STALE"), check("SIGNATURE_MISSING")])
    assert capped.score == 100 and capped.band == "high"
    custom = risk_score([check("DATE_POSTDATED", severity=0.5)], points={"DATE_POSTDATED": 20})
    assert custom.score == 10 and custom.band == "low"


def test_critical_checks_force_high() -> None:
    weak = risk_score([check("TAMPER_REGION", severity=0.3, conf=0.95)])
    assert weak.score < 70 and weak.band == "high" and weak.forced_by == "TAMPER_REGION"
    unsure = risk_score([check("AMOUNT_MISMATCH", severity=0.4, conf=0.5)])
    assert unsure.band == "low" and unsure.forced_by is None
    assert risk_score([]).as_dict() == {
        "score": 0,
        "band": "low",
        "forced_by": None,
        "contributions": [],
    }


# ---------------------------------------------------------------------------- forgeries


@pytest.fixture(scope="module")
def capture() -> dict:
    return forged_sample(123, forged=False)


@pytest.mark.parametrize("kind", KINDS)
def test_tamper_kinds_change_only_the_field(capture: dict, kind: str) -> None:
    rng = np.random.default_rng(0)
    hmat = capture["page_to_image"]
    quad = box_quad((1386, 352, 1586, 420), hmat)
    src = forged_sample(124, forged=False)
    edited, mask = tamper(
        capture["image"],
        quad,
        kind,
        rng,
        donor_quad=box_quad((300, 200, 500, 268), hmat),
        source=src["image"],
        source_quad=box_quad((1386, 352, 1586, 420), src["page_to_image"]),
    )
    assert edited.shape == capture["image"].shape and mask.any()
    field = cv2.dilate(page_mask(mask.shape, quad, shrink=0).astype(np.uint8), np.ones((9, 9)))
    assert not (mask & (field == 0)).any()  # the ground truth stays inside the field


def test_tamper_rejects_bad_arguments(capture: dict) -> None:
    quad = box_quad((1386, 352, 1586, 420), capture["page_to_image"])
    rng = np.random.default_rng(0)
    for kind in ("splice", "copymove", "blur"):
        with pytest.raises(ValueError):
            tamper(capture["image"], quad, kind, rng)


def test_forged_sample_records_what_was_done() -> None:
    s = forged_sample(5, forged=True)
    assert s["kind"] in KINDS and s["field"] and s["mask"].any() and s["page"][s["mask"]].all()


# --------------------------------------------------------------------------- tamper maps


def test_block_features_and_classical_map(capture: dict) -> None:
    img = capture["image"]
    f = block_features(img, capture["page"])
    assert f.shape == (img.shape[0] // BLOCK, img.shape[1] // BLOCK, len(FEATURE_NAMES))
    assert np.isfinite(f).all()
    heat = classical_heatmap(img, capture["page"])
    assert heat.shape == f.shape[:2] and heat.min() >= 0 and heat.max() <= 1
    assert upsample(heat, img.shape[:2]).shape == img.shape[:2]


def test_to_flat_homography_and_map() -> None:
    heat = np.zeros((100, 200), np.float32)
    heat[40:60, 50:70] = 1
    shift = np.array([[1, 0, -50], [0, 1, -40], [0, 0, 1]], float)
    flat = to_flat(heat, (20, 20), homography=shift)
    assert flat.mean() > 0.9
    mx, my = np.meshgrid(np.arange(20, dtype=np.float32) + 50, np.arange(20, dtype=np.float32) + 40)
    assert np.allclose(to_flat(heat, (20, 20), flat_map=(mx, my)), 1)
    with pytest.raises(ValueError):
        to_flat(heat, (20, 20))
    scores = field_scores(np.pad(np.ones((5, 5)), 5), {"hot": (5, 5, 10, 10), "cold": (0, 0, 4, 4)})
    assert scores == {"hot": 1.0, "cold": 0.0}


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from findoc.fraud.tamper_train import train

    out = tmp_path_factory.mktemp("model") / "tamper.pkl"
    train(out, samples=4, val_samples=2, workers=1, max_iter=40, log=lambda *_: None)
    return out


def test_train_save_load(tiny_model: Path, capture: dict) -> None:
    m = TamperModel.load(tiny_model)
    assert set(m.meta) >= {"image_auc", "detection_rate", "false_alarm_rate", "threshold"}
    assert 0.5 <= m.threshold <= 0.99
    prob = m.block_probability(capture["image"], capture["page"])
    assert prob.shape == (capture["image"].shape[0] // BLOCK, capture["image"].shape[1] // BLOCK)
    assert prob.min() >= 0 and prob.max() <= 1
    bad = tiny_model.parent / "old.pkl"
    import pickle

    bad.write_bytes(pickle.dumps({"version": 0, "features": [], "model": None}))
    with pytest.raises(ValueError):
        TamperModel.load(bad)


def test_tamper_stage_with_and_without_model(
    tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = forged_sample(77, forged=True, kind="retype", field="amount_figures")
    path = tmp_path / "forged.png"
    save_image(path, s["image"])
    doc = process(path, ocr=False)
    tc = next(c for c in doc.checks if c.id == "TAMPER_REGION")
    assert not tc.fired and "not trained" in tc.reason and doc.tamper_map is not None
    assert doc.tamper_map.shape == doc.flat.shape[:2]

    monkeypatch.setenv("FINDOC_TAMPER_MODEL", str(tiny_model))
    get_settings.cache_clear()
    load_model.cache_clear()
    doc = process(path, ocr=False)
    tc = next(c for c in doc.checks if c.id == "TAMPER_REGION")
    assert set(tc.evidence["field_scores"]) >= {"amount_figures", "payee", "date"}
    assert doc.risk["band"] in {"low", "medium", "high"} and doc.risk_score is not None

    monkeypatch.setenv("FINDOC_TAMPER", "off")
    get_settings.cache_clear()
    assert not any(c.id == "TAMPER_REGION" for c in process(path, ocr=False).checks)


# ---------------------------------------------------------------------------- duplicates


def _photo(seed: int, shot: int, tmp: Path) -> Path:
    page, _ = render_cheque(np.random.default_rng(seed), bank="ARVL")
    p, _ = photograph(
        page,
        np.random.default_rng((seed, shot)),
        size=CHEQUE_PHOTO_SIZE,
        fill=CHEQUE_FILL,
        blur_sigma=0.5 * shot,
    )
    path = tmp / f"c{seed}_{shot}.jpg"
    save_image(path, p)
    return path


def test_duplicate_index_roundtrip(tmp_path: Path) -> None:
    ink = np.full((200, 400), 255, np.uint8)
    cv2.putText(ink, "Ravi Kumar 25,000", (10, 120), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, 0, 2)
    other = np.full((200, 400), 255, np.uint8)
    cv2.putText(other, "Neha 1,200/-", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    n, r = verify(ink_features(ink), ink_features(ink))
    assert n > 20 and r > 0.9
    idx = DuplicateIndex(tmp_path / "idx")
    idx.add("a.jpg", "ARVL", ink, micr="123456 302123456 654321 10")
    idx = DuplicateIndex(tmp_path / "idx")  # reloaded from disk
    assert len(idx) == 1
    assert idx.search("KVRY", ink) == []  # another bank is never compared
    same_micr = idx.search("ARVL", other, micr="123456 302123456 654321 10")
    assert same_micr[0].same_micr and same_micr[0].duplicate
    assert not idx.search("ARVL", other)[0].duplicate


def test_duplicate_stage_flags_second_photo(tmp_path: Path) -> None:
    first, second, unrelated = (
        _photo(1, 0, tmp_path),
        _photo(1, 1, tmp_path),
        _photo(2, 0, tmp_path),
    )
    doc = process(first, ocr=False, register=True)
    assert not any(c.id == "DUPLICATE" for c in doc.checks)  # empty index: nothing to compare
    assert any("Added to the duplicate index" in m for m in doc.messages)
    doc = process(unrelated, ocr=False)
    assert not next(c for c in doc.checks if c.id == "DUPLICATE").fired
    doc = process(second, ocr=False, register=True)
    dup = next(c for c in doc.checks if c.id == "DUPLICATE")
    assert dup.fired and dup.evidence["source"] == first.name
    assert doc.risk["band"] == "high" and any("Not added" in m for m in doc.messages)


# ---------------------------------------------------------------------------- signatures


def test_signature_descriptors() -> None:
    assert normalise(np.zeros((50, 100), np.uint8)) is None
    rng = np.random.default_rng(0)
    signer, other = Signer.random(rng), Signer.random(rng)

    def sig(who: Signer, skilled: bool = False) -> np.ndarray:
        img = np.full((130, 490, 3), 255, np.uint8)
        who.draw(img, (0, 0, 490, 130), rng, skilled)
        return normalise((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8))

    ref = describe(sig(signer))
    assert distance(ref, ref) == pytest.approx(0, abs=1e-4)
    assert distance(ref, describe(sig(signer))) < distance(ref, describe(sig(other)))


def test_signature_stage_and_cli(tmp_path: Path) -> None:
    signer = Signer.random(np.random.default_rng(3))
    forger = Signer.random(np.random.default_rng(4))
    paths = {}
    for name, who, seed in (
        ("r1", signer, 10),
        ("r2", signer, 11),
        ("ok", signer, 12),
        ("bad", forger, 13),
    ):
        page, _ = render_cheque(np.random.default_rng(seed), signer=who)
        p, _ = photograph(
            page, np.random.default_rng(seed), size=CHEQUE_PHOTO_SIZE, fill=CHEQUE_FILL
        )
        paths[name] = tmp_path / f"{name}.jpg"
        save_image(paths[name], p)
    res = runner.invoke(
        app, ["signatures", "add", "1234567890", str(paths["r1"]), str(paths["r2"])]
    )
    assert res.exit_code == 0, res.output
    assert SignatureStore(get_settings().signatures_dir).accounts() == {"1234567890": 2}
    assert "1234567890: 2" in runner.invoke(app, ["signatures", "list"]).output

    ok = process(paths["ok"], ocr=False, account="1234567890")
    bad = process(paths["bad"], ocr=False, account="1234567890")
    c_ok = next(c for c in ok.checks if c.id == "SIGNATURE_MISMATCH")
    c_bad = next(c for c in bad.checks if c.id == "SIGNATURE_MISMATCH")
    assert not c_ok.fired and c_bad.fired
    assert c_ok.evidence["distance"] < c_bad.evidence["distance"]
    unknown = process(paths["ok"], ocr=False, account="999")
    assert not any(c.id == "SIGNATURE_MISMATCH" for c in unknown.checks)
    assert any("No reference signatures" in m for m in unknown.messages)


# -------------------------------------------------------------------------------- CLI


def test_fraud_cli_commands(tmp_path: Path) -> None:
    photo = _photo(3, 0, tmp_path)
    res = runner.invoke(
        app, ["process", str(photo), "--no-ocr", "--register", "--save-dir", str(tmp_path / "out")]
    )
    assert res.exit_code == 0, res.output
    assert "Risk score" in res.output and (tmp_path / "out" / f"{photo.stem}_tamper.jpg").exists()
    assert "c3_0.jpg" in runner.invoke(app, ["index", "list"]).output
    res = runner.invoke(app, ["tamper", "check", str(photo), "--out", str(tmp_path / "t.jpg")])
    assert res.exit_code == 0 and "no model trained" in res.output and (tmp_path / "t.jpg").exists()
    assert runner.invoke(app, ["index", "clear", "--yes"]).exit_code == 0
    assert "already empty" in runner.invoke(app, ["index", "clear", "--yes"]).output
