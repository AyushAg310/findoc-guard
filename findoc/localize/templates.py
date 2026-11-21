"""Cheque templates: a blank reference image plus the boxes where each field is written.

Cheques are fixed-layout documents (CTS-2010), so once we know which bank's leaf we are
looking at and how the photo maps onto that leaf, every field's position is known. This is
cheaper and more reliable than learning to find fields, which is why cheques use templates
and variable-layout invoices use a learned model (Phase 7).

Templates live on disk as::

    <templates_dir>/<ID>/reference.png
    <templates_dir>/<ID>/template.json   {"id", "name", "fields": [{name, box, kind, allowlist}]}

so an administrator can register a new bank from a scanned blank cheque (FR-APP-08): with
``findoc templates``, or in the web UI's template editor (Phase 8). The built-in set is the
twelve fictional banks from :mod:`findoc.synth.cheque`; templates in ``templates_dir`` are
added to it (a custom template with a built-in's id replaces it). Every template is scaled to
the standard leaf width (1624 px), because registration compares a photo against all of them
at one scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from findoc.io import load_image, save_image
from findoc.synth.cheque import BANKS, TEMPLATE_SIZE, FieldSpec, field_specs, render_blank


@dataclass
class ChequeTemplate:
    id: str
    name: str
    reference: np.ndarray  # BGR blank leaf
    fields: list[FieldSpec]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the reference image."""
        h, w = self.reference.shape[:2]
        return w, h

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def save(self, root: Path) -> Path:
        folder = root / self.id
        folder.mkdir(parents=True, exist_ok=True)
        save_image(folder / "reference.png", self.reference)
        meta = {"id": self.id, "name": self.name, "fields": [asdict(f) for f in self.fields]}
        (folder / "template.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> ChequeTemplate:
        meta = json.loads((folder / "template.json").read_text(encoding="utf-8"))
        fields = [
            FieldSpec(f["name"], tuple(f["box"]), f["kind"], f.get("allowlist"), f.get("cells", 1))  # type: ignore[arg-type]
            for f in meta["fields"]
        ]
        return cls(meta["id"], meta["name"], load_image(folder / "reference.png"), fields).scaled()

    def scaled(self, width: int = TEMPLATE_SIZE[0]) -> ChequeTemplate:
        """This template at the standard leaf width (boxes scaled with the image)."""
        w, h = self.size
        if w == width:
            return self
        s = width / w
        ref = cv2.resize(self.reference, (width, round(h * s)), interpolation=cv2.INTER_AREA)
        fields = [
            FieldSpec(f.name, tuple(round(v * s) for v in f.box), f.kind, f.allowlist, f.cells)  # type: ignore[arg-type]
            for f in self.fields
        ]
        return ChequeTemplate(self.id, self.name, ref, fields)


class TemplateStore:
    def __init__(self, templates: list[ChequeTemplate]) -> None:
        self.templates = {t.id: t for t in templates}

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.templates.values())

    def __len__(self) -> int:
        return len(self.templates)

    def get(self, template_id: str) -> ChequeTemplate:
        return self.templates[template_id]

    @classmethod
    def from_dir(cls, root: Path) -> TemplateStore:
        folders = sorted(p for p in root.iterdir() if (p / "template.json").exists())
        return cls([ChequeTemplate.load(p) for p in folders])

    def save(self, root: Path) -> list[Path]:
        return [t.save(root) for t in self]


@lru_cache(maxsize=1)
def builtin_store() -> TemplateStore:
    """Templates for the fictional synthetic banks (rendered once, then cached)."""
    return TemplateStore(
        [
            ChequeTemplate(code, style.name, render_blank(style), field_specs(style))
            for code, style in BANKS.items()
        ]
    )


def _dir_signature(root: Path) -> tuple:
    return tuple(
        (p.parent.name, p.stat().st_mtime_ns) for p in sorted(root.glob("*/template.json"))
    )


@lru_cache(maxsize=4)
def _custom_store(root: Path, signature: tuple) -> TemplateStore:  # noqa: ARG001 - cache key
    return TemplateStore.from_dir(root)


def custom_store(templates_dir: Path | None) -> TemplateStore:
    """Templates saved in ``templates_dir`` (re-read only when one is added or changed)."""
    if not templates_dir or not templates_dir.is_dir():
        return TemplateStore([])
    return _custom_store(templates_dir.resolve(), _dir_signature(templates_dir))


def default_store(templates_dir: Path | None = None) -> TemplateStore:
    """The built-in banks plus the templates in ``templates_dir`` (same id: custom wins)."""
    custom = custom_store(templates_dir)
    if not len(custom):
        return builtin_store()
    merged = {t.id: t for t in builtin_store()} | {t.id: t for t in custom}
    return TemplateStore(list(merged.values()))


def draw_fields(img: np.ndarray, template: ChequeTemplate) -> np.ndarray:
    """Overlay the template's field boxes and names on a registered cheque image."""
    out = img.copy()
    colours = {"handwritten": (0, 140, 255), "printed": (200, 120, 0), "signature": (0, 160, 0)}
    for f in template.fields:
        x0, y0, x1, y1 = f.box
        c = colours.get(f.kind, (0, 0, 255))
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        cv2.putText(out, f.name, (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out
