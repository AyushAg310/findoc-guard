"""Phase 8: the web service, job queue and access control.

Runs on SQLite by default; set FINDOC_TEST_DATABASE_URL to a PostgreSQL URL to run the same
tests there (CI does both).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

PASSWORD = "correct horse"


def _clear_caches() -> None:
    from app import db, settings, worker

    settings.get_api_settings.cache_clear()
    if db.get_engine.cache_info().currsize:
        db.get_engine().dispose()
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    worker.ocr_enabled.cache_clear()


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    url = os.environ.get("FINDOC_TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv("FINDOC_API_DATABASE_URL", url)
    monkeypatch.setenv("FINDOC_API_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("FINDOC_API_FRONTEND_DIR", str(tmp_path / "no_ui"))
    monkeypatch.setenv("FINDOC_API_INLINE_WORKER", "false")
    monkeypatch.setenv("FINDOC_API_OCR", "off")
    monkeypatch.setenv("FINDOC_API_JWT_SECRET", "test-secret-" + "x" * 32)
    monkeypatch.setenv("FINDOC_API_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("FINDOC_API_ADMIN_PASSWORD", PASSWORD)
    _clear_caches()
    from app.db import Base, get_engine

    Base.metadata.drop_all(get_engine())  # a clean PostgreSQL database per test
    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client
    Base.metadata.drop_all(get_engine())
    _clear_caches()


def login(client: TestClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    r = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def users(api: TestClient) -> dict[str, dict[str, str]]:
    admin = login(api, "admin@example.com")
    out = {"admin": admin}
    for role in ("submitter", "reviewer"):
        email = f"{role}@example.com"
        r = api.post(
            "/api/v1/users",
            headers=admin,
            json={"email": email, "role": role, "password": PASSWORD},
        )
        assert r.status_code == 201, r.text
        out[role] = login(api, email)
    r = api.post(
        "/api/v1/users",
        headers=admin,
        json={"email": "other@example.com", "role": "submitter", "password": PASSWORD},
    )
    out["other"] = login(api, "other@example.com")
    return out


def png_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def cheque_jpg() -> bytes:
    from findoc.synth.dataset import cheque_photo

    photo, _, _ = cheque_photo(7)
    return png_bytes(photo, ".jpg")


def upload(client: TestClient, headers: dict, data: bytes, name: str = "cheque.jpg") -> str:
    r = client.post("/api/v1/documents", headers=headers, files={"file": (name, data)})
    assert r.status_code == 202, r.text
    return r.json()["id"]


FAKE_RESULT = {
    "template_id": "synthbank_a",
    "doc_type": "cheque",
    "fields": [
        {
            "name": "payee",
            "value": "Ravi Kumar",
            "normalized": "Ravi Kumar",
            "confidence": 0.9,
            "bbox": {"x0": 10, "y0": 20, "x1": 300, "y1": 60},
        },
        {
            "name": "amount_figures",
            "value": "2500.00",
            "normalized": "2500.00",
            "confidence": 0.95,
            "bbox": None,
        },
        {
            "name": "account_number",
            "value": "123456789012",
            "normalized": "123456789012",
            "confidence": 0.9,
            "bbox": None,
        },
    ],
    "checks": [
        {
            "id": "AMOUNT_MISMATCH",
            "fired": True,
            "severity": 0.8,
            "confidence": 0.9,
            "reason": "Words say 25,000",
            "evidence": {},
        }
    ],
    "risk": {
        "score": 72,
        "band": "high",
        "contributions": [{"check": "AMOUNT_MISMATCH", "contribution": 72.0}],
    },
    "stages": [{"stage": "load", "ms": 12.5, "ok": True, "note": ""}],
}


def put_result(doc_id: str, status: str = "needs_review") -> None:
    """Give a document a finished result without running OCR (for review-flow tests)."""
    from app.db import get_sessionmaker
    from app.models import Document

    with get_sessionmaker()() as s:
        row = s.get(Document, doc_id)
        row.status, row.result = status, FAKE_RESULT
        row.risk_score, row.risk_band, row.doc_type = 72, "high", "cheque"
        s.commit()


# --------------------------------------------------------------------------- unit pieces


def test_security_primitives() -> None:
    from app.security import (
        api_key_prefix,
        create_token,
        decode_token,
        hash_password,
        new_api_key,
        verify_password,
    )

    h = hash_password("s3cret-pass")
    assert h.startswith("scrypt$") and verify_password("s3cret-pass", h)
    assert not verify_password("wrong", h) and not verify_password("x", "garbage")
    assert hash_password("same") != hash_password("same")  # salted
    tok = create_token("u1", "reviewer", "k" * 32, 5)
    assert decode_token(tok, "k" * 32)["role"] == "reviewer"
    with pytest.raises(Exception):  # noqa: B017
        decode_token(tok, "other-key" * 4)
    key, prefix, digest = new_api_key()
    assert api_key_prefix(key) == prefix and len(digest) == 64
    assert api_key_prefix("not-a-key") is None


def test_upload_sniffing(tmp_path: Path) -> None:
    from app.uploads import UploadError, sniff, store_upload

    img = np.full((300, 400, 3), 200, np.uint8)
    s = store_upload(png_bytes(img, ".jpg"), tmp_path / "a", 10**7)
    assert s.media_type == "image/jpeg" and s.path.suffix == ".png" and s.kind == "image"
    assert cv2.imread(str(s.path)).shape == (300, 400, 3)
    assert sniff(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4).kind == "video"
    assert sniff(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 4).media_type == "video/quicktime"
    for bad, msg in [
        (b"", "empty"),
        (b"%PDF-1.7 hello", "Unsupported"),
        (b"\xff\xd8\xff" + b"junk" * 10, "decoded"),
        (png_bytes(np.zeros((50, 50, 3), np.uint8)), "outside"),
    ]:
        with pytest.raises(UploadError, match=msg):
            store_upload(bad, tmp_path / "b", 10**7)
    with pytest.raises(UploadError, match="larger"):
        store_upload(png_bytes(img), tmp_path / "c", 100)


def test_mask_digits() -> None:
    from app.routers.documents import mask_digits

    assert mask_digits("123456789012") == "••••••••9012"
    assert mask_digits("⑈123⑈ 456 7890") == "⑈•••⑈ ••• 7890"
    assert mask_digits(None) is None and mask_digits("12") == "12"


# --------------------------------------------------------------------------- auth and roles


def test_auth_and_problem_details(api: TestClient, users: dict) -> None:
    r = api.get("/api/v1/auth/me")
    assert r.status_code == 401 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["title"] == "Unauthorized" and r.json()["instance"] == "/api/v1/auth/me"
    bad = api.post("/api/v1/auth/login", data={"username": "admin@example.com", "password": "x"})
    assert bad.status_code == 401
    me = api.get("/api/v1/auth/me", headers=users["reviewer"]).json()
    assert me["role"] == "reviewer" and me["email"] == "reviewer@example.com"
    assert api.get("/api/v1/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = api.post("/api/v1/users", headers=users["admin"], json={"email": "x", "password": "short"})
    assert r.status_code == 422 and r.json()["errors"]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_roles_are_enforced(api: TestClient, users: dict) -> None:
    sub, rev = users["submitter"], users["reviewer"]
    assert api.get("/api/v1/users", headers=sub).status_code == 403
    assert api.get("/api/v1/users", headers=rev).status_code == 403
    assert api.get("/api/v1/audit", headers=rev).status_code == 403
    assert api.get("/api/v1/documents/stats", headers=sub).status_code == 403
    assert api.get("/api/v1/documents/stats", headers=rev).status_code == 200
    assert api.post("/api/v1/exports", headers=sub, json={}).status_code == 403
    assert api.get("/api/v1/rules", headers=rev).json()["bands"]["high"] == 70
    users_list = api.get("/api/v1/users", headers=users["admin"]).json()
    admin_id = next(u["id"] for u in users_list if u["role"] == "admin")
    r = api.patch(f"/api/v1/users/{admin_id}", headers=users["admin"], json={"active": False})
    assert r.status_code == 409  # can't lock yourself out
    sub_id = next(u["id"] for u in users_list if u["email"] == "submitter@example.com")
    r = api.patch(f"/api/v1/users/{sub_id}", headers=users["admin"], json={"active": False})
    assert r.json()["active"] is False
    assert api.get("/api/v1/auth/me", headers=sub).status_code == 401  # disabled at once


def test_api_keys(api: TestClient, users: dict, cheque_jpg: bytes) -> None:
    r = api.post("/api/v1/api-keys", headers=users["submitter"], json={"name": "erp"})
    assert r.status_code == 201
    key = r.json()["key"]
    listed = api.get("/api/v1/api-keys", headers=users["submitter"]).json()
    assert listed[0]["key"] is None and listed[0]["prefix"] in key  # secret not shown again
    doc_id = upload(api, {"X-API-Key": key}, cheque_jpg)
    assert api.get(f"/api/v1/documents/{doc_id}", headers={"X-API-Key": key}).status_code == 200
    assert api.get("/api/v1/auth/me", headers={"X-API-Key": key + "x"}).status_code == 401
    api.delete(f"/api/v1/api-keys/{r.json()['id']}", headers=users["submitter"])
    assert api.get("/api/v1/auth/me", headers={"X-API-Key": key}).status_code == 401


# --------------------------------------------------------------------------- documents


def test_upload_process_and_view(api: TestClient, users: dict, cheque_jpg: bytes) -> None:
    from app.worker import run_once

    sub = users["submitter"]
    r = api.post("/api/v1/documents", headers=sub, files={"file": ("x.pdf", b"%PDF-1.4 x")})
    assert r.status_code == 422 and "Unsupported" in r.json()["detail"]
    doc_id = upload(api, sub, cheque_jpg, "../../etc/cheque 1.jpg")
    d = api.get(f"/api/v1/documents/{doc_id}", headers=sub).json()
    assert d["status"] == "queued" and d["filename"] == ".._.._etc_cheque 1.jpg"
    again = api.post("/api/v1/documents", headers=sub, files={"file": ("c.jpg", cheque_jpg)})
    assert again.json()["duplicate_of"] == doc_id

    assert run_once() == doc_id
    d = api.get(f"/api/v1/documents/{doc_id}", headers=sub).json()
    assert d["status"] in ("processed", "needs_review"), d["error"]
    assert d["doc_type"] == "cheque" and d["page_frame"] == "template" and d["progress"] == 1.0
    assert {"page", "flat", "original", "annotated"} <= set(d["images"])
    assert [s["stage"] for s in d["stages"]][:2] == ["load", "video"]
    img = api.get(f"/api/v1/documents/{doc_id}/image?variant=page", headers=sub)
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    # the account number is covered in the image too, not just masked in the JSON
    from app.db import get_sessionmaker
    from app.models import Document
    from app.worker import REDACT
    from findoc.config import get_settings
    from findoc.localize.templates import default_store

    with get_sessionmaker()() as s:
        template_id = s.get(Document, doc_id).result["template_id"]
    template = default_store(get_settings().templates_dir).get(template_id)
    page = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    for f in template.fields:
        if f.name in REDACT:
            x0, y0, x1, y1 = f.box
            covered = page[y0 + 3 : y1 - 3, x0 + 3 : int(x0 + (x1 - x0) * REDACT[f.name]) - 3]
            assert covered.min() >= 160 and covered.mean() > 190  # grey hatching, no dark digits
    assert (
        api.get(f"/api/v1/documents/{doc_id}/image?variant=bogus", headers=sub).status_code == 422
    )

    # another submitter can't see it (404, not 403); reviewers can
    assert api.get(f"/api/v1/documents/{doc_id}", headers=users["other"]).status_code == 404
    assert api.get("/api/v1/documents", headers=users["other"]).json()["total"] == 0
    listed = api.get(
        "/api/v1/documents?status=processed&status=needs_review", headers=users["reviewer"]
    ).json()
    assert listed["total"] == 1 and listed["items"][0]["owner_email"] == "submitter@example.com"
    assert api.get("/api/v1/documents?q=nomatch", headers=users["reviewer"]).json()["total"] == 0
    stats = api.get("/api/v1/documents/stats", headers=users["reviewer"]).json()
    assert stats["total"] == 2 and stats["by_status"]["queued"] == 1
    assert "findoc_stage_seconds_count" in api.get("/metrics").text
    assert api.get("/health").json()["queued"] == 1


def test_blurry_photo_needs_retake(api: TestClient, users: dict, blurry_photo_path: Path) -> None:
    from app.worker import run_once

    doc_id = upload(api, users["submitter"], blurry_photo_path.read_bytes())
    run_once()
    d = api.get(f"/api/v1/documents/{doc_id}", headers=users["submitter"]).json()
    assert d["status"] == "rejected_quality" and d["allowed_decisions"] == []


def test_review_flow_and_audit(api: TestClient, users: dict, cheque_jpg: bytes) -> None:
    sub, rev, adm = users["submitter"], users["reviewer"], users["admin"]
    doc_id = upload(api, sub, cheque_jpg)
    base = f"/api/v1/documents/{doc_id}"
    assert (
        api.patch(f"{base}/fields", headers=rev, json={"fields": {"payee": "x"}}).status_code == 409
    )
    put_result(doc_id)

    d = api.get(base, headers=sub).json()
    acct = next(f for f in d["fields"] if f["name"] == "account_number")
    assert acct["value"] == "••••••••9012"  # masked
    assert d["fields"][0]["bbox"] == [10, 20, 300, 60]
    assert d["checks"][0]["contribution"] == 72.0 and d["allowed_decisions"] == []

    assert (
        api.patch(f"{base}/fields", headers=sub, json={"fields": {"payee": "x"}}).status_code == 403
    )
    r = api.patch(f"{base}/fields", headers=rev, json={"fields": {"nonsense": "x"}})
    assert r.status_code == 422
    r = api.patch(f"{base}/fields", headers=rev, json={"fields": {"amount_figures": "25000.00"}})
    f = next(f for f in r.json()["fields"] if f["name"] == "amount_figures")
    assert f["corrected"] == "25000.00" and f["value"] == "2500.00"
    assert r.json()["field_edits"][0]["user_email"] == "reviewer@example.com"

    assert set(api.get(base, headers=rev).json()["allowed_decisions"]) == {
        "approved",
        "rejected",
        "escalated",
    }
    r = api.post(f"{base}/decision", headers=rev, json={"decision": "escalated"})
    assert r.status_code == 422  # comment required
    r = api.post(
        f"{base}/decision",
        headers=rev,
        json={"decision": "escalated", "comment": "Amount looks altered"},
    )
    assert r.json()["status"] == "escalated"
    r = api.post(f"{base}/decision", headers=rev, json={"decision": "approved"})
    assert r.status_code == 403  # escalations go to an admin
    r = api.post(f"{base}/decision", headers=adm, json={"decision": "rejected", "comment": "fraud"})
    assert r.json()["status"] == "rejected" and len(r.json()["decisions"]) == 2
    r = api.post(f"{base}/decision", headers=adm, json={"decision": "approved"})
    assert r.status_code == 409

    actions = [
        a["action"] for a in api.get(f"/api/v1/audit?target_id={doc_id}", headers=adm).json()
    ]
    assert actions[::-1] == [
        "document.submitted",
        "field.corrected",
        "document.escalated",
        "document.rejected",
    ]


def test_exports(api: TestClient, users: dict, cheque_jpg: bytes) -> None:
    rev = users["reviewer"]
    doc_id = upload(api, users["submitter"], cheque_jpg)
    put_result(doc_id)
    api.patch(
        f"/api/v1/documents/{doc_id}/fields", headers=rev, json={"fields": {"payee": "Ravi K."}}
    )
    r = api.post("/api/v1/exports", headers=rev, json={"format": "json", "band": ["high"]})
    assert r.status_code == 201 and r.json()["count"] == 1
    records = api.get(r.json()["download"], headers=rev).json()
    fields = {f["name"]: f for f in records[0]["fields"]}
    assert fields["payee"]["value"] == "Ravi K." and fields["payee"]["method"] == "reviewer"
    assert fields["account_number"]["value"] == "••••••••9012"
    r = api.post("/api/v1/exports", headers=rev, json={"format": "xlsx"})
    xl = api.get(r.json()["download"], headers=rev)
    assert xl.status_code == 200 and xl.content[:2] == b"PK"
    assert api.post("/api/v1/exports", headers=rev, json={"band": ["low"]}).json()["count"] == 0
    assert api.get("/api/v1/exports/nope", headers=rev).status_code == 404


def test_retries_then_fails(
    api: TestClient, users: dict, cheque_jpg: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    import findoc.pipeline
    from app.worker import run_once

    def broken(**_: object) -> None:
        raise RuntimeError("model file corrupt")

    monkeypatch.setattr(findoc.pipeline, "default_pipeline", broken)
    doc_id = upload(api, users["submitter"], cheque_jpg)
    base = f"/api/v1/documents/{doc_id}"
    for attempt in (1, 2):
        run_once()
        d = api.get(base, headers=users["reviewer"]).json()
        assert d["status"] == "queued" and d["attempts"] == attempt
    run_once()
    d = api.get(base, headers=users["reviewer"]).json()
    assert d["status"] == "failed" and "model file corrupt" in d["error"]
    assert "Traceback" not in (api.get(base, headers=users["submitter"]).json()["error"] or "")
    assert run_once() is None  # queue empty
    r = api.post(f"{base}/retry", headers=users["reviewer"])
    assert r.json()["status"] == "queued" and r.json()["attempts"] == 0
    assert api.post(f"{base}/retry", headers=users["reviewer"]).status_code == 409


def test_claim_is_exclusive_and_stale_jobs_requeue(
    api: TestClient, users: dict, cheque_jpg: bytes
) -> None:
    from app.db import get_sessionmaker
    from app.models import Document, now
    from app.worker import claim_next, requeue_stale

    doc_id = upload(api, users["submitter"], cheque_jpg)
    with get_sessionmaker()() as a, get_sessionmaker()() as b:
        assert claim_next(a) == doc_id
        assert claim_next(b) is None
        assert requeue_stale(b) == 0
        row = b.get(Document, doc_id)
        row.started_at = now() - timedelta(hours=2)
        b.commit()
        assert requeue_stale(b) == 1
        assert claim_next(b) == doc_id


def test_delete_and_purge(api: TestClient, users: dict, cheque_jpg: bytes) -> None:
    from typer.testing import CliRunner

    from app.db import get_sessionmaker
    from app.models import Document, now
    from app.storage import doc_dir, source_path
    from findoc.cli import app as cli

    old = upload(api, users["submitter"], cheque_jpg)
    put_result(old, "approved")
    with get_sessionmaker()() as s:
        s.get(Document, old).created_at = now() - timedelta(days=40)
        s.commit()
    assert source_path(old) is not None
    out = CliRunner().invoke(cli, ["api", "purge", "--days", "30"])
    assert out.exit_code == 0 and "media of 1" in out.output
    assert (
        source_path(old) is None
        and api.get(f"/api/v1/documents/{old}", headers=users["reviewer"]).status_code == 200
    )
    assert api.delete(f"/api/v1/documents/{old}", headers=users["reviewer"]).status_code == 403
    assert api.delete(f"/api/v1/documents/{old}", headers=users["admin"]).status_code == 204
    assert not doc_dir(old).exists()
    assert api.get(f"/api/v1/documents/{old}", headers=users["admin"]).status_code == 404


def test_cli_create_user_and_worker_once(api: TestClient) -> None:
    from typer.testing import CliRunner

    from findoc.cli import app as cli

    r = CliRunner().invoke(
        cli,
        [
            "api",
            "create-user",
            "--email",
            "Rev2@Example.com",
            "--role",
            "reviewer",
            "--password",
            "longenough",
        ],
    )
    assert r.exit_code == 0, r.output
    assert login(api, "rev2@example.com", "longenough")
    r = CliRunner().invoke(cli, ["api", "worker", "--once"])
    assert r.exit_code == 0 and "Processed 0" in r.output


def test_serves_built_frontend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api) -> None:  # noqa: ANN001
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>review ui</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    from fastapi import FastAPI

    from app.main import mount_frontend

    app = FastAPI()
    mount_frontend(app, dist)
    c = TestClient(app)
    assert "review ui" in c.get("/documents/abc").text  # client-side route
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert "review ui" in c.get("/../../etc/passwd").text  # no escaping the folder
    assert c.get("/api/v1/nothing").status_code == 404
