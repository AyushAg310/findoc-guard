"""Service settings (Phase 8), read from ``FINDOC_API_*`` environment variables or ``.env``.

The pipeline's own settings stay in :mod:`findoc.config` (``FINDOC_*``).
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINDOC_API_", env_file=".env", extra="ignore")

    # SQLite for a single machine; PostgreSQL in Docker Compose, e.g.
    # postgresql+psycopg://findoc:findoc@db:5432/findoc
    database_url: str = "sqlite:///data/findoc.db"
    # Uploaded media and rendered images (a Docker volume in Compose)
    storage_dir: Path = Path("data/storage")
    # Built React app served at "/" when it exists (``npm run build`` in frontend/)
    frontend_dir: Path = Path("frontend/dist")

    # Auth: set a fixed secret in production, or every restart logs everyone out
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_minutes: int = 8 * 60
    # Created at start-up when the database has no users (demo / first run)
    admin_email: str | None = None
    admin_password: str | None = None

    # Uploads (NFR-SEC-05)
    max_upload_mb: int = 200

    # Jobs: run a worker thread inside the API process (single machine) or separate
    # ``findoc worker`` processes (Compose)
    inline_worker: bool = True
    max_attempts: int = 3  # NFR-REL-02
    poll_seconds: float = 1.0
    stale_minutes: int = 30  # a job "processing" longer than this is assumed lost and requeued
    # OCR in the worker: "auto" = when EasyOCR is installed
    ocr: str = "auto"

    # Privacy (NFR-PRV-01): raw media older than this is deleted by ``findoc api purge``
    retention_days: int = 30


@lru_cache
def get_api_settings() -> ApiSettings:
    return ApiSettings()
