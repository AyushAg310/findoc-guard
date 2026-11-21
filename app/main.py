"""The FastAPI application: routers, errors as RFC 9457 problem details, security headers,
metrics, start-up (tables, first admin, in-process worker) and the built React app."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import func, select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.db import get_sessionmaker, init_db
from app.deps import audit
from app.models import Role, User, new_id
from app.routers import documents, other
from app.security import hash_password
from app.settings import get_api_settings
from findoc.logging import get_logger

log = get_logger(__name__)

PROBLEM = "application/problem+json"


def problem(status: int, title: str, detail: object, request: Request, **extra: object) -> Response:
    """RFC 9457: ``type``, ``title``, ``status``, ``detail``, ``instance``."""
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM)


def bootstrap_admin() -> None:
    s = get_api_settings()
    if not (s.admin_email and s.admin_password):
        return
    with get_sessionmaker()() as db:
        if db.scalar(select(func.count()).select_from(User)):
            return
        user = User(
            id=new_id(),
            email=s.admin_email.lower(),
            name="Admin",
            role=Role.ADMIN.value,
            password_hash=hash_password(s.admin_password),
        )
        db.add(user)
        audit(
            db,
            None,
            "user.created",
            "user",
            user.id,
            email=user.email,
            role=user.role,
            via="FINDOC_API_ADMIN_EMAIL",
        )
        db.commit()
        log.info("Created admin user %s", user.email)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    bootstrap_admin()
    worker = None
    if get_api_settings().inline_worker:
        from app.worker import start_inline_worker

        worker = start_inline_worker()
        log.info("In-process worker started")
    yield
    if worker:
        worker[1].set()
        worker[0].join(timeout=5)


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinDoc Guard API",
        version=__version__,
        description="Upload cheque and invoice photos or videos, follow their processing, and "
        "review the extracted fields and fraud checks. Authenticate with a JWT from "
        "`/api/v1/auth/login` or an `X-API-Key` header.",
        lifespan=lifespan,
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        titles = {
            400: "Bad request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not found",
            405: "Method not allowed",
            409: "Conflict",
            413: "Too large",
            422: "Unprocessable",
            429: "Too many requests",
        }
        resp = problem(exc.status_code, titles.get(exc.status_code, "Error"), exc.detail, request)
        for k, v in (getattr(exc, "headers", None) or {}).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return problem(422, "Invalid request", "See errors", request, errors=errors)

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception) -> Response:
        log.exception("Unhandled error on %s", request.url.path)
        return problem(500, "Internal error", "Something went wrong; it has been logged", request)

    @app.middleware("http")
    async def headers_and_metrics(request: Request, call_next):  # noqa: ANN001, ANN202
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "other")
        other.METRICS.observe(request.method, path, response.status_code, time.perf_counter() - t0)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    app.include_router(other.auth)
    app.include_router(documents.router)
    app.include_router(other.exports)
    app.include_router(other.admin)
    app.include_router(other.system)
    mount_frontend(app, get_api_settings().frontend_dir)
    return app


def mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built React app; any non-API path returns index.html (client-side routes)."""
    index = dist / "index.html"
    if not index.is_file():

        @app.get("/", include_in_schema=False)
        def no_ui() -> dict[str, str]:
            return {
                "message": "FinDoc Guard API. The review UI isn't built: run `npm ci && "
                "npm run build` in frontend/. API docs: /docs"
            }

        return
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith(("api/", "docs", "openapi.json", "health", "metrics")):
            raise HTTPException(404, "Not found")
        target = (root / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


app = create_app()
