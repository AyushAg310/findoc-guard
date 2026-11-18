"""Template identification and registration by feature matching (CO-3, FR-EXT-02/03).

Given a flattened photo of a cheque, decide **which bank template** it is and compute the
**homography** that maps the photo onto that template, so every field box is known.

Pipeline for each template:

1. **Keypoints + descriptors** (SIFT by default, ORB optional). SIFT finds blob-like points
   at many scales (difference-of-Gaussians extrema) and describes each with a 128-D histogram
   of gradient orientations, normalised for rotation, scale and brightness.
2. **Matching**: for each photo descriptor, the two nearest template descriptors; keep the match
   only if the nearest is clearly better than the second (Lowe's ratio test, 0.75). Repetitive
   structures like the security pattern produce ambiguous matches and are rejected here.
3. **RANSAC homography**: repeatedly fit a homography to 4 random matches and count how many
   other matches agree within 5 px (inliers); keep the best model, then refit on its inliers.
   Wrong matches (outliers) cannot outvote a consistent geometric explanation.
4. **Score** = number of inliers. The template with the most inliers wins, if it has at least
   ``min_inliers`` and the homography is geometrically sane.

Template descriptors are computed once and cached. The SuperPoint + LightGlue matcher arrives
with the 3D module in Phase 5 and can be swapped in behind :class:`FeatureMatcher`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from findoc.localize.templates import ChequeTemplate, TemplateStore

RATIO = 0.75
RANSAC_PX = 5.0
MIN_INLIERS = 40


@dataclass
class AlignResult:
    template_id: str
    homography: np.ndarray  # maps input-image pixels -> template pixels
    inliers: int
    matches: int
    scores: dict[str, int]  # inliers per candidate template

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


class FeatureMatcher:
    """Detects and matches local features. ``method`` is 'sift' or 'orb'."""

    def __init__(self, method: str = "sift", n_features: int = 4000) -> None:
        self.method = method
        if method == "sift":
            self._detector = cv2.SIFT_create(nfeatures=n_features)
            self._norm = cv2.NORM_L2
        elif method == "orb":
            self._detector = cv2.ORB_create(nfeatures=n_features)
            self._norm = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unknown feature method: {method}")
        self._bf = cv2.BFMatcher(self._norm)

    def detect(self, img: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Contrast-limited histogram equalisation makes faint print easier to describe.
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return self._detector.detectAndCompute(gray, None)

    def match(self, des_a: np.ndarray | None, des_b: np.ndarray | None) -> list[cv2.DMatch]:
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            return []
        good = []
        for pair in self._bf.knnMatch(des_a, des_b, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                good.append(pair[0])
        return good


@lru_cache(maxsize=4)
def get_matcher(method: str = "sift") -> FeatureMatcher:
    return FeatureMatcher(method)


_template_cache: dict[tuple[str, str, int], tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}


def template_features(
    template: ChequeTemplate, matcher: FeatureMatcher
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    key = (template.id, matcher.method, id(template.reference))
    if key not in _template_cache:
        _template_cache[key] = matcher.detect(template.reference)
    return _template_cache[key]


def homography_is_sane(h_img_to_tmpl: np.ndarray, img_size: tuple[int, int],
                       tmpl_size: tuple[int, int]) -> bool:  # fmt: skip
    """Reject degenerate fits: the template must map to a convex, reasonably sized region."""
    tw, th = tmpl_size
    corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32).reshape(-1, 1, 2)
    try:
        inv = np.linalg.inv(h_img_to_tmpl)
    except np.linalg.LinAlgError:
        return False
    pts = cv2.perspectiveTransform(corners, inv).reshape(-1, 2)
    if not cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(pts.astype(np.float32))
    iw, ih = img_size
    return 0.2 * iw * ih < area < 5.0 * iw * ih


def align_to_template(
    img: np.ndarray, template: ChequeTemplate, matcher: FeatureMatcher | None = None
) -> tuple[np.ndarray | None, int, int]:
    """Homography (img -> template), inlier count and match count for one template."""
    matcher = matcher or get_matcher()
    kp_img, des_img = matcher.detect(img)
    return _align(kp_img, des_img, img, template, matcher)


def _align(
    kp_img: list[cv2.KeyPoint],
    des_img: np.ndarray | None,
    img: np.ndarray,
    template: ChequeTemplate,
    matcher: FeatureMatcher,
) -> tuple[np.ndarray | None, int, int]:
    kp_t, des_t = template_features(template, matcher)
    matches = matcher.match(des_img, des_t)
    if len(matches) < 10:
        return None, 0, len(matches)
    src = np.float32([kp_img[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=5000)
    if homography is None or mask is None:
        return None, 0, len(matches)
    inliers = int(mask.sum())
    h, w = img.shape[:2]
    if not homography_is_sane(homography, (w, h), template.size):
        return None, 0, len(matches)
    return homography, inliers, len(matches)


def identify(
    img: np.ndarray,
    store: TemplateStore,
    method: str = "sift",
    min_inliers: int = MIN_INLIERS,
) -> AlignResult | None:
    """Find the best-matching template, or None if nothing matches convincingly."""
    matcher = get_matcher(method)
    kp_img, des_img = matcher.detect(img)
    best: AlignResult | None = None
    scores: dict[str, int] = {}
    for template in store:
        homography, inliers, n_matches = _align(kp_img, des_img, img, template, matcher)
        scores[template.id] = inliers
        if homography is not None and (best is None or inliers > best.inliers):
            best = AlignResult(template.id, homography, inliers, n_matches, scores)
    if best is None or best.inliers < min_inliers:
        return None
    best.scores = scores
    return best


def register(img: np.ndarray, homography: np.ndarray, template: ChequeTemplate) -> np.ndarray:
    """Warp the photo into the template's coordinate frame (same size as the reference)."""
    return cv2.warpPerspective(
        img, homography, template.size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def resize_to_width(img: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize to ``width`` keeping aspect ratio; returns the image and the 3x3 scale matrix."""
    h, w = img.shape[:2]
    s = width / w
    out = cv2.resize(img, (width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return out, np.diag([s, s, 1.0])
