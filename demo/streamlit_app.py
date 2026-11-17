"""FinDoc Guard demo UI.

Run from the repository root:

    pip install -e ".[ui]"          # add ",ocr" to also read text
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from findoc import __version__
from findoc.config import get_settings
from findoc.localize.templates import default_store, draw_fields
from findoc.pipeline import process
from findoc.preprocess.detect import draw_detection
from findoc.recognize.fields import crop_fields
from findoc.synth.dataset import FRAUD_KINDS, cheque_photo
from findoc.synth.photo import photograph, render_page

st.set_page_config(page_title="FinDoc Guard", layout="wide")
st.title("FinDoc Guard")
st.caption(
    f"v{__version__} · quality gate, page detection, perspective correction, "
    "cheque template matching, field extraction, rule checks, video tracking and keyframes"
)


def rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV images are BGR; Streamlit expects RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Photo or short video of a cheque or document",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webm"],
    )
    st.write("or generate a synthetic input:")
    kind = st.radio("Synthetic input", ["Cheque", "Cheque video", "Generic page"], horizontal=True)
    seed = st.number_input("Seed", 0, 10_000, 7)
    blur = 0.0
    if kind != "Cheque video":
        blur = st.slider("Camera shake (blur sigma)", 0.0, 6.0, 0.0, 0.5)
    fraud = None
    if kind in ("Cheque", "Cheque video"):
        choice = st.selectbox("Fraud condition", ["none", *FRAUD_KINDS])
        fraud = None if choice == "none" else choice
    use_synth = st.button("Generate synthetic input")
    run_ocr = st.checkbox("Run OCR (needs the [ocr] extra)", value=True)

tmp_dir = Path(tempfile.gettempdir()) / "findoc_demo"
tmp_dir.mkdir(exist_ok=True)
source: Path | None = None

if uploaded is not None:
    source = tmp_dir / uploaded.name
    source.write_bytes(uploaded.getvalue())
elif use_synth:
    if kind == "Cheque":
        photo, truth, _ = cheque_photo(int(seed), fraud, blur_sigma=float(blur))
        st.session_state["truth"] = truth
    elif kind == "Cheque video":
        from findoc.synth.video import cheque_video, write_video

        with st.spinner("Rendering a 3 s hand-held video..."):
            frames, _, truth = cheque_video(int(seed), fraud, seconds=3.0)
        st.session_state["truth"] = truth
        source = write_video(tmp_dir / f"synthetic_video_{seed}_{fraud}.mp4", frames, 30.0)
    else:
        rng = np.random.default_rng(int(seed))
        photo, _ = photograph(render_page(rng), rng, blur_sigma=float(blur))
        st.session_state.pop("truth", None)
    if kind != "Cheque video":
        source = tmp_dir / f"synthetic_{kind}_{seed}_{blur}_{fraud}.jpg"
        cv2.imwrite(str(source), photo)

if source is None:
    st.info("Upload a photo or video, or generate a synthetic one from the sidebar.")
    st.stop()

with st.spinner("Processing..."):
    doc = process(source, ocr=run_ocr)

status = doc.status.value
flagged = [c for c in doc.checks if c.fired]
if status == "ok" and flagged:
    st.error(f"Processed: {len(flagged)} check(s) flagged")
elif status == "ok":
    st.success("Processed successfully" + (": all checks passed" if doc.checks else ""))
elif status == "rejected_quality":
    st.warning("Rejected by the quality gate")
else:
    st.error("Processing failed")
for msg in doc.messages:
    st.write("- " + msg)

if doc.frames:
    from findoc.track.viz import keyframe_sheet, write_tracking_video

    st.subheader("0. Video tracking (KLT) and keyframes")
    st.json(doc.tracking)
    st.line_chart(
        {
            "sharpness": [t.sharpness for t in doc.track],
            "tracked points": [t.tracks for t in doc.track],
        }
    )
    if doc.keyframes:
        st.caption(
            f"{len(doc.keyframes)} keyframes (sharp, spread viewpoints) for 3D reconstruction; "
            f"best frame #{doc.best_frame} goes to the 2D pipeline below"
        )
        st.image(rgb(keyframe_sheet(doc)), width="stretch")
    with st.expander("Tracking video"):
        st.caption("Browsers may not play mp4v; download and open it in VLC if it doesn't play.")
        tracked = write_tracking_video(doc, tmp_dir / f"{source.stem}_tracking.mp4")
        st.download_button("Download tracking video", tracked.read_bytes(), tracked.name)

cols = st.columns(3)
if doc.original is not None:
    shown = doc.original
    if doc.corners is not None:
        shown = draw_detection(doc.original, np.array(doc.corners, dtype=np.float32))
    cols[0].subheader("1. Detected document" + (" (best frame)" if doc.frames else ""))
    cols[0].image(rgb(shown), width="stretch")
if doc.flat is not None:
    cols[1].subheader("2. Perspective corrected")
    cols[1].image(rgb(doc.flat), width="stretch")
if doc.enhanced is not None:
    cols[2].subheader("3. Lighting normalised")
    cols[2].image(rgb(doc.enhanced), width="stretch")

if doc.registered is not None and doc.template_id:
    template = default_store(get_settings().templates_dir).get(doc.template_id)
    st.subheader(f"4. Registered to template: {template.name}")
    st.image(rgb(draw_fields(doc.registered, template)), width="stretch")

    with st.expander("What OCR sees: field crops after template subtraction"):
        for crop in crop_fields(doc.registered, template).values():
            st.caption(f"{crop.name} · ink {crop.ink_ratio:.1%}")
            st.image(rgb(crop.image))

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted fields")
        st.dataframe(
            [
                {
                    "field": f.name,
                    "read": f.value,
                    "validated": f.normalized or "invalid",
                    "confidence": round(f.confidence, 2),
                }
                for f in doc.fields
            ],
            hide_index=True,
        )
        truth = st.session_state.get("truth")
        if truth is not None and uploaded is None:
            with st.expander("Ground truth (synthetic)"):
                st.json({k: v for k, v in truth.__dict__.items() if k != "fields"})
    with right:
        st.subheader("Checks")
        for c in doc.checks:
            icon = "🚩" if c.fired else "✅"
            st.write(f"{icon} **{c.id}**: {c.reason}")

left, right = st.columns(2)
with left:
    st.subheader("Quality and matching metrics")
    st.json(doc.quality)
with right:
    st.subheader("Stage timings (ms)")
    st.bar_chart({s.stage: s.ms for s in doc.stages})
    if doc.full_text:
        st.subheader("Recognised text")
        st.text(doc.full_text)

with st.expander("Full JSON result"):
    st.code(json.dumps(doc.result(), indent=2), language="json")
