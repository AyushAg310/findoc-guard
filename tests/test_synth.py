from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from findoc.synth.photo import generate, photograph, render_page


def test_render_page_shape_and_content() -> None:
    page = render_page(np.random.default_rng(0))
    assert page.shape == (1100, 850, 3)
    assert page.min() < 100  # has dark text
    assert np.median(page) > 200  # mostly white paper


def test_photograph_is_deterministic() -> None:
    a, qa = photograph(render_page(np.random.default_rng(5)), np.random.default_rng(5))
    b, qb = photograph(render_page(np.random.default_rng(5)), np.random.default_rng(5))
    np.testing.assert_array_equal(qa, qb)
    assert np.array_equal(a, b)


def test_corners_inside_frame() -> None:
    rng = np.random.default_rng(9)
    photo, quad = photograph(render_page(rng), rng, size=(800, 1000))
    assert photo.shape == (1000, 800, 3)
    assert (quad[:, 0] >= 0).all() and (quad[:, 0] < 800).all()
    assert (quad[:, 1] >= 0).all() and (quad[:, 1] < 1000).all()


def test_generate_writes_images_and_labels(tmp_path: Path) -> None:
    paths = generate(3, tmp_path, seed=11, size=(600, 800))
    assert [p.name for p in paths] == ["synth_0011.jpg", "synth_0012.jpg", "synth_0013.jpg"]
    labels = json.loads((tmp_path / "labels.json").read_text())
    assert set(labels) == {p.name for p in paths}
    assert len(labels["synth_0011.jpg"]["corners"]) == 4
