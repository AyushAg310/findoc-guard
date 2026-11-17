"""Cheque templates: a blank reference image plus the boxes where each field is written.

Cheques are fixed-layout documents (CTS-2010), so once we know which bank's leaf we are
looking at and how the photo maps onto that leaf, every field's position is known. This is
cheaper and more reliable than learning to find fields, which is why cheques use templates
and variable-layout invoices use a learned model (Phase 6).

Templates live on disk as::

    <templates_dir>/<ID>/reference.png
    <templates_dir>/<ID>/template.json   {"id", "name", "fields": [{name, box, kind, allowlist}]}

so an administrator can register a new bank from a scanned blank cheque (FR-APP-08).
The built-in set is the four fictional banks from :mod:`findoc.synth.cheque`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from findoc.io import load_image, save_image
from findoc.synth.cheque import BANKS, FieldSpec, field_specs, render_blank


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
        return cls(meta["id"], meta["name"], load_image(folder / "reference.png"), fields)


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


def default_store(templates_dir: Path | None = None) -> TemplateStore:
    """Templates from ``templates_dir`` if it has any, otherwise the built-in set."""
    if templates_dir and templates_dir.is_dir():
        store = TemplateStore.from_dir(templates_dir)
        if len(store):
            return store
    return builtin_store()


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
