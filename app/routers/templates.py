"""Cheque templates: list the banks the system knows, and let an admin add one (FR-APP-08).

An admin uploads a blank leaf and places the field boxes in the UI. Boxes arrive as fractions
of the image (0..1), so the editor doesn't care about pixel sizes; the image is scaled to the
standard leaf width (1624 px) before saving. The template is written to the pipeline's
``templates_dir`` (``data/templates`` by default, git-ignored) and is used from the next
upload on. Built-in (fictional) banks can't be changed or deleted.

Reference images are served with the account-number and MICR areas covered, because a
personalised leaf carries the holder's account number.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import audit, require
from app.models import Role, User
from findoc.config import get_settings
from findoc.localize.templates import (
    ChequeTemplate,
    builtin_store,
    custom_store,
    default_store,
)
from findoc.synth.cheque import BANKS, TEMPLATE_SIZE, FieldSpec, field_specs

router = APIRouter(prefix="/api/v1/templates", tags=["templates"])

# kind, allowlist and cells per field name, from the standard (CTS-2010) layout
STANDARD = {f.name: f for f in field_specs(next(iter(BANKS.values())))}
REQUIRED = {"payee", "amount_words_1", "amount_figures", "date", "signature"}


class BoxIn(BaseModel):
    name: str
    box: list[float] = Field(min_length=4, max_length=4, description="x0, y0, x1, y1 in 0..1")


def _custom_dir() -> Path:
    d = get_settings().templates_dir
    if d is None:
        raise HTTPException(409, "Set FINDOC_TEMPLATES_DIR to add templates")
    return d


def _summary(t: ChequeTemplate, builtin: bool) -> dict[str, Any]:
    w, h = t.size
    return {
        "id": t.id,
        "name": t.name,
        "builtin": builtin,
        "size": [w, h],
        "fields": [
            {
                "name": f.name,
                "kind": f.kind,
                "box": [f.box[0] / w, f.box[1] / h, f.box[2] / w, f.box[3] / h],
            }
            for f in t.fields
        ],
    }


@router.get("")
def list_templates(_: User = Depends(require(Role.REVIEWER))) -> list[dict[str, Any]]:
    """Every bank the cheque pipeline can recognise; custom ones first."""
    custom = {t.id for t in custom_store(get_settings().templates_dir)}
    store = default_store(get_settings().templates_dir)
    items = [_summary(t, t.id not in custom) for t in store]
    return sorted(items, key=lambda x: (x["builtin"], x["name"]))


@router.get("/standard-layout")
def standard_layout(_: User = Depends(require(Role.ADMIN))) -> list[dict[str, Any]]:
    """The CTS-2010 field boxes as fractions: the editor's starting point."""
    w, h = TEMPLATE_SIZE
    return [
        {
            "name": f.name,
            "kind": f.kind,
            "box": [f.box[0] / w, f.box[1] / h, f.box[2] / w, f.box[3] / h],
        }
        for f in STANDARD.values()
    ]


@router.get("/{template_id}/image")
def template_image(template_id: str, _: User = Depends(require(Role.REVIEWER))) -> Response:
    """The blank leaf, with the account-number and MICR areas covered."""
    from app.worker import REDACT, _cover

    store = default_store(get_settings().templates_dir)
    try:
        t = store.get(template_id)
    except KeyError as exc:
        raise HTTPException(404, "Template not found") from exc
    quads = []
    for f in t.fields:
        if f.name in REDACT:
            x0, y0, x1, y1 = f.box
            xc = x0 + (x1 - x0) * REDACT[f.name]
            quads.append(np.array([[x0, y0], [xc, y0], [xc, y1], [x0, y1]], np.float32))
    img = _cover(t.reference, quads) if quads else t.reference
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(
        buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"}
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_template(
    name: str = Form(..., min_length=2, max_length=60),
    code: str = Form(..., description="4 capital letters, like an IFSC bank code"),
    fields: str = Form(..., description='JSON list of {"name", "box": [x0, y0, x1, y1]} in 0..1'),
    image: UploadFile = File(..., description="A blank cheque leaf, photographed flat or scanned"),
    db: Session = Depends(get_db),
    user: User = Depends(require(Role.ADMIN)),
) -> dict[str, Any]:
    """Add a bank from a blank leaf and its field boxes."""
    code = code.strip().upper()
    if not re.fullmatch(r"[A-Z]{4}", code):
        raise HTTPException(422, "The code must be 4 letters (A-Z)")
    if code in BANKS:
        raise HTTPException(409, f"{code} is a built-in bank; choose another code")
    try:
        boxes = [BoxIn.model_validate(b) for b in json.loads(fields)]
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, f"fields is not valid: {exc}") from exc
    names = {b.name for b in boxes}
    if unknown := names - set(STANDARD):
        raise HTTPException(422, f"Unknown field(s): {', '.join(sorted(unknown))}")
    if missing := REQUIRED - names:
        raise HTTPException(422, f"Missing field(s): {', '.join(sorted(missing))}")

    data = await image.read(20 * 2**20 + 1)
    if len(data) > 20 * 2**20:
        raise HTTPException(422, "The image is larger than 20 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "The image could not be decoded")
    h, w = img.shape[:2]
    if not 1.6 <= w / h <= 2.8:
        raise HTTPException(
            422,
            f"A cheque leaf is about 2.2 times as wide as tall; this image "
            f"is {w}x{h}. Crop it to the leaf.",
        )
    width = TEMPLATE_SIZE[0]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    specs = []
    for b in boxes:
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in b.box)
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            raise HTTPException(422, f"The {b.name} box is too small")
        s = STANDARD[b.name]
        specs.append(
            FieldSpec(
                b.name,
                (round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)),
                s.kind,
                s.allowlist,
                s.cells,
            )
        )
    template = ChequeTemplate(code, name.strip(), img, specs)
    template.save(_custom_dir())
    audit(db, user, "template.created", "template", None, code=code, name=template.name)
    db.commit()
    return _summary(template, builtin=False)


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(
    template_id: str, db: Session = Depends(get_db), user: User = Depends(require(Role.ADMIN))
) -> None:
    if template_id in {t.id for t in builtin_store()} and template_id not in {
        t.id for t in custom_store(get_settings().templates_dir)
    }:
        raise HTTPException(409, "Built-in banks can't be deleted")
    folder = _custom_dir() / template_id
    if not (folder / "template.json").is_file() or not re.fullmatch(r"[A-Z]{4}", template_id):
        raise HTTPException(404, "Template not found")
    shutil.rmtree(folder)
    audit(db, user, "template.deleted", "template", None, code=template_id)
    db.commit()
