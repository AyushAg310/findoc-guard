"""FinDoc Guard demo UI (Phase 1).

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
from findoc.pipeline import process
from findoc.preprocess.detect import draw_detection
from findoc.synth.photo import photograph, render_page

st.set_page_config(page_title="FinDoc Guard", layout="wide")
st.title("FinDoc Guard")
st.caption(f"v{__version__} · Phase 1: quality gate, page detection, perspective correction, OCR")


def rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV images are BGR; Streamlit expects RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Photo of a document", type=["jpg", "jpeg", "png", "webp"])
    st.write("or")
    seed = st.number_input("Synthetic photo seed", 0, 10_000, 7)
    blur = st.slider("Simulated camera shake (blur sigma)", 0.0, 6.0, 0.0, 0.5)
    use_synth = st.button("Generate synthetic photo")
    run_ocr = st.checkbox("Run OCR (needs the [ocr] extra)", value=True)

tmp_dir = Path(tempfile.gettempdir()) / "findoc_demo"
tmp_dir.mkdir(exist_ok=True)
source: Path | None = None

if uploaded is not None:
    source = tmp_dir / uploaded.name
    source.write_bytes(uploaded.getvalue())
elif use_synth:
    rng = np.random.default_rng(int(seed))
    photo, _ = photograph(render_page(rng), rng, blur_sigma=float(blur))
    source = tmp_dir / f"synthetic_{seed}_{blur}.jpg"
    cv2.imwrite(str(source), photo)

if source is None:
    st.info("Upload a photo or generate a synthetic one from the sidebar.")
    st.stop()

with st.spinner("Processing..."):
    doc = process(source, ocr=run_ocr)

status = doc.status.value
if status == "ok":
    st.success("Processed successfully")
elif status == "rejected_quality":
    st.warning("Rejected by the quality gate")
else:
    st.error("Processing failed")
for msg in doc.messages:
    st.write("- " + msg)

cols = st.columns(3)
if doc.original is not None:
    shown = doc.original
    if doc.corners is not None:
        shown = draw_detection(doc.original, np.array(doc.corners, dtype=np.float32))
    cols[0].subheader("1. Detected page")
    cols[0].image(rgb(shown), width="stretch")
if doc.flat is not None:
    cols[1].subheader("2. Perspective corrected")
    cols[1].image(rgb(doc.flat), width="stretch")
if doc.enhanced is not None:
    cols[2].subheader("3. Lighting normalised")
    cols[2].image(rgb(doc.enhanced), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Quality metrics")
    st.json(doc.quality)
    st.subheader("Stage timings (ms)")
    st.bar_chart({s.stage: s.ms for s in doc.stages})
with right:
    st.subheader("Recognised text")
    st.text(doc.full_text or "(no text: OCR disabled, not installed, or nothing found)")

with st.expander("Full JSON result"):
    st.code(json.dumps(doc.result(), indent=2), language="json")
