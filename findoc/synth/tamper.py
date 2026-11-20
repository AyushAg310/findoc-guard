"""Synthetic digital forgeries of captured cheques, with exact masks (Phase 6).

The threat modelled here is the common one for remote cheque deposit: the fraudster edits the
**photo or video frame** before submitting it, e.g. changing the amount, payee or date. Three
standard manipulations from the image-forensics literature, each applied to a field region
given as a quadrilateral in *image* coordinates (where the field appears in the capture):

* ``retype`` (erase and rewrite): the ink in the field is removed by inpainting and new text
  is drawn digitally. The new pixels never went through the camera, so they lack its sensor
  noise and JPEG history.
* ``splice``: the same field is cut from **another** capture (a different cheque, different
  lighting, noise, blur and compression) and pasted in, resampled to fit.
* ``copymove``: a region of the **same** image (another written line) is copied, slightly
  rescaled and rotated, and pasted over the field. Same camera noise, but the resampling
  smooths it.

All edited images are then saved again as JPEG, as an edited upload would be. The returned mask
marks the pixels that were changed (before that final compression), which is the ground truth
for pixel-level evaluation.
"""

from __future__ import annotations

import cv2
import numpy as np

KINDS = ("retype", "splice", "copymove")
INK = (110, 35, 20)  # BGR, the synthetic ballpoint colour


def _quad_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 255)
    return m


def _rect_to_quad(w: int, h: int, quad: np.ndarray) -> np.ndarray:
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.getPerspectiveTransform(src, quad.astype(np.float32))


def _feather(mask: np.ndarray, px: float) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), px)[..., None]


def _text_patch(text: str, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Dark-ink text on transparent background as an (h, w) alpha map, filling the box."""
    alpha = np.zeros((h, w), np.uint8)
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    scale = 1.0
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    scale = min(0.92 * w / max(tw, 1), 0.75 * h / max(th, 1))
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    org = (int(rng.uniform(0.02, 0.08) * w), int((h + th) / 2))
    cv2.putText(alpha, text, org, font, scale, 255, max(1, int(round(scale * 2))), cv2.LINE_AA)
    return alpha


def retype(
    img: np.ndarray, quad: np.ndarray, rng: np.random.Generator, text: str | None = None
) -> np.ndarray:
    """Erase the ink inside ``quad`` (inpainting) and write ``text`` over it."""
    h, w = img.shape[:2]
    region = _quad_mask((h, w), quad)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = np.median(gray[region > 0])
    ink = ((gray < paper - 40) & (region > 0)).astype(np.uint8) * 255
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    out = cv2.inpaint(img, ink, 5, cv2.INPAINT_TELEA)
    bw = int(np.linalg.norm(quad[1] - quad[0]))
    bh = int(np.linalg.norm(quad[3] - quad[0]))
    if text is None:
        text = f"{int(rng.integers(10, 99))},{int(rng.integers(100, 999))}/-"
    alpha = _text_patch(text, max(bw, 8), max(bh, 8), rng)
    alpha = cv2.warpPerspective(alpha, _rect_to_quad(bw, bh, quad), (w, h))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6).astype(np.float32)[..., None] / 255.0
    ink_colour = np.array(INK, np.float32) * rng.uniform(0.8, 1.1)
    out = out.astype(np.float32) * (1 - alpha) + ink_colour * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def paste(
    img: np.ndarray, quad: np.ndarray, source: np.ndarray, source_quad: np.ndarray
) -> np.ndarray:
    """Warp ``source_quad`` of ``source`` onto ``quad`` of ``img`` with a soft edge."""
    h, w = img.shape[:2]
    hmat = cv2.getPerspectiveTransform(source_quad.astype(np.float32), quad.astype(np.float32))
    warped = cv2.warpPerspective(source, hmat, (w, h), flags=cv2.INTER_LINEAR)
    m = _feather(_quad_mask((h, w), quad), 1.5)
    # match the overall brightness of the target so the edit is not obvious to the eye
    region = _quad_mask((h, w), quad) > 0
    gain = (img[region].astype(np.float32).mean(0) + 1) / (
        warped[region].astype(np.float32).mean(0) + 1
    )
    warped = np.clip(warped.astype(np.float32) * gain, 0, 255)
    out = img.astype(np.float32) * (1 - m) + warped * m
    return np.clip(out, 0, 255).astype(np.uint8)


def copymove(
    img: np.ndarray, quad: np.ndarray, donor_quad: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Copy ``donor_quad`` of the same image onto ``quad``, slightly rescaled and rotated."""
    c = donor_quad.mean(0)
    s = rng.uniform(0.85, 1.15)
    a = np.deg2rad(rng.uniform(-3, 3))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    donor = (donor_quad - c) @ rot.T * s + c
    return paste(img, quad, img, donor)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def changed_mask(before: np.ndarray, after: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Ground truth: pixels the edit changed, closed into regions, inside the field quad."""
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > 4
    m = cv2.morphologyEx(diff.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    region = cv2.dilate(_quad_mask(m.shape, quad), np.ones((5, 5), np.uint8))
    return (m > 0) & (region > 0)


def tamper(
    img: np.ndarray,
    quad: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    donor_quad: np.ndarray | None = None,
    source: np.ndarray | None = None,
    source_quad: np.ndarray | None = None,
    quality: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one manipulation and re-save as JPEG. Returns (edited image, changed-pixel mask)."""
    if kind == "retype":
        edited = retype(img, quad, rng)
    elif kind == "splice":
        if source is None or source_quad is None:
            raise ValueError("splice needs a source image and quad")
        edited = paste(img, quad, source, source_quad)
    elif kind == "copymove":
        if donor_quad is None:
            raise ValueError("copymove needs a donor quad")
        edited = copymove(img, quad, donor_quad, rng)
    else:
        raise ValueError(f"unknown tamper kind {kind!r}; expected one of {KINDS}")
    mask = changed_mask(img, edited, quad)
    return jpeg(edited, quality or int(rng.integers(85, 96))), mask


# ------------------------------------------------------------------------- training samples

FIELDS = ("amount_figures", "payee", "date", "amount_words_1", "amount_words_2")
DONOR_BOX = (300, 200, 500, 268)  # part of the payee line: handwriting to copy from


def box_quad(box: tuple[int, int, int, int], page_to_image: np.ndarray) -> np.ndarray:
    """A template/page box -> its quadrilateral in the capture."""
    x0, y0, x1, y1 = box
    pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, page_to_image).reshape(-1, 2)


def capture(
    seed: int, bank: str | None = None, quality: int | None = None, blur: float = 0.0
) -> tuple[np.ndarray, object, np.ndarray]:
    """A synthetic cheque photo with a random camera JPEG quality (80-95).
    Returns (photo, cheque truth, page -> photo homography)."""
    from findoc.synth.cheque import render_cheque
    from findoc.synth.dataset import CHEQUE_FILL, CHEQUE_PHOTO_SIZE
    from findoc.synth.photo import photograph

    rng = np.random.default_rng(seed)
    page, truth = render_cheque(rng, bank=bank)
    q = quality or int(rng.integers(80, 96))
    photo, quad = photograph(
        page, rng, size=CHEQUE_PHOTO_SIZE, fill=CHEQUE_FILL, blur_sigma=blur, jpeg_quality=q
    )
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return photo, truth, cv2.getPerspectiveTransform(src, quad.astype(np.float32))


def forged_sample(
    seed: int, forged: bool, kind: str | None = None, field: str | None = None
) -> dict:
    """One synthetic capture, clean or forged, with everything an evaluation needs:
    ``image``, ``mask`` (changed pixels), ``page`` (document pixels), ``page_to_image``,
    ``bank``, ``kind``, ``field``. Clean samples are sometimes re-saved as JPEG too, as an
    innocent re-upload would be, so "was re-compressed" alone is not evidence."""
    from findoc.synth.cheque import BANKS, field_specs

    rng = np.random.default_rng((seed, 7))
    photo, truth, hmat = capture(seed)
    h, w = photo.shape[:2]
    page = cv2.warpPerspective(np.ones((736, 1624), np.uint8), hmat, (w, h)) > 0
    out = {"page_to_image": hmat, "bank": truth.bank, "page": page, "kind": None, "field": None}
    if not forged:
        if rng.random() < 0.5:
            photo = jpeg(photo, int(rng.integers(85, 96)))
        return out | {"image": photo, "mask": np.zeros((h, w), bool)}
    kind = kind or KINDS[seed % len(KINDS)]
    field = field or FIELDS[seed % len(FIELDS)]
    specs = {f.name: f for f in field_specs(BANKS[truth.bank])}
    quad = box_quad(specs[field].box, hmat)
    src, _, h2 = capture(seed + 77_777, bank=truth.bank, blur=float(rng.uniform(0, 1)))
    donor = (170, 276, 700, 342) if field == "payee" else DONOR_BOX
    image, mask = tamper(
        photo,
        quad,
        kind,
        rng,
        donor_quad=box_quad(donor, hmat),
        source=src,
        source_quad=box_quad(specs[field].box, h2),
    )
    return out | {"image": image, "mask": mask, "kind": kind, "field": field}
