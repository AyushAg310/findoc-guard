"""``findoc api ...``: run the service and workers, manage users, purge old media.

Imports of FastAPI and SQLAlchemy happen inside the commands, so ``findoc --help`` works
without the ``[api]`` extra installed.
"""

from __future__ import annotations

import typer
from rich.console import Console

api_app = typer.Typer(help="Web service, workers and users (Phase 8).", no_args_is_help=True)
console = Console()


def _need_api() -> None:
    try:
        import fastapi  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter(f'{exc}. Install with: pip install -e ".[api]"') from exc


@api_app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="0.0.0.0 to accept other machines"),
    port: int = typer.Option(8000),
    worker: bool = typer.Option(True, "--worker/--no-worker", help="Process jobs in-process"),
) -> None:
    """Run the API (and, by default, a worker thread). Open http://127.0.0.1:8000"""
    _need_api()
    import os

    import uvicorn

    os.environ["FINDOC_API_INLINE_WORKER"] = "true" if worker else "false"
    uvicorn.run("app.main:app", host=host, port=port, proxy_headers=True)


@api_app.command("worker")
def worker(
    once: bool = typer.Option(False, "--once", help="Process the queue until empty, then exit"),
) -> None:
    """Run a worker process: takes queued documents and runs the pipeline."""
    _need_api()
    from app.db import init_db
    from app.worker import run_forever, run_once

    init_db()
    if once:
        n = 0
        while run_once():
            n += 1
        console.print(f"Processed {n} document(s)")
        return
    try:
        run_forever()
    except KeyboardInterrupt:
        console.print("Worker stopped")


@api_app.command("create-user")
def create_user(
    email: str = typer.Option(..., prompt=True),
    role: str = typer.Option("reviewer", help="submitter, reviewer or admin"),
    name: str = typer.Option(""),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True),
) -> None:
    """Create a user (the first admin, for example)."""
    _need_api()
    from sqlalchemy import func, select

    from app.db import get_sessionmaker, init_db
    from app.deps import audit
    from app.models import Role, User, new_id
    from app.security import hash_password

    if role not in {r.value for r in Role}:
        raise typer.BadParameter("role must be submitter, reviewer or admin")
    if len(password) < 8:
        raise typer.BadParameter("password must have at least 8 characters")
    init_db()
    with get_sessionmaker()() as db:
        if db.scalar(select(User).where(func.lower(User.email) == email.lower())):
            raise typer.BadParameter(f"{email} already exists")
        user = User(
            id=new_id(),
            email=email.lower(),
            name=name,
            role=role,
            password_hash=hash_password(password),
        )
        db.add(user)
        audit(db, None, "user.created", "user", user.id, email=user.email, role=role, via="cli")
        db.commit()
    console.print(f"Created {role} {email}")


@api_app.command("purge")
def purge(
    days: int | None = typer.Option(None, help="Default: FINDOC_API_RETENTION_DAYS (30)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Delete uploaded media older than the retention period (NFR-PRV-01). Results, rendered
    images and the audit trail are kept; decided or failed documents only."""
    _need_api()
    from datetime import timedelta

    from sqlalchemy import select

    from app.db import get_sessionmaker, init_db
    from app.deps import audit
    from app.models import DocStatus, Document, now
    from app.settings import get_api_settings
    from app.storage import delete_source

    keep = days if days is not None else get_api_settings().retention_days
    cutoff = now() - timedelta(days=keep)
    busy = (DocStatus.QUEUED.value, DocStatus.PROCESSING.value)
    init_db()
    with get_sessionmaker()() as db:
        rows = db.scalars(
            select(Document).where(
                Document.created_at < cutoff,
                Document.media_deleted.is_(False),
                Document.status.not_in(busy),
            )
        ).all()
        for row in rows:
            if not dry_run:
                delete_source(row.id)
                row.media_deleted = True
                audit(db, None, "document.media_purged", "document", row.id, days=keep)
        db.commit()
    console.print(
        f"{'Would delete' if dry_run else 'Deleted'} media of {len(rows)} document(s) "
        f"older than {keep} days"
    )
