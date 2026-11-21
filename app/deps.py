"""Request dependencies: who is calling, and may they (FR-APP-07, NFR-SEC-03).

A caller authenticates with either ``Authorization: Bearer <JWT>`` (people, after
``/api/v1/auth/login``) or ``X-API-Key: fdk_...`` (systems). An API key acts as the user who
owns it, with that user's role. Unknown or expired credentials give 401; a known caller
without the right role gives 403.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ApiKey, AuditLog, Role, User, now
from app.security import api_key_prefix, decode_token, hash_api_key
from app.settings import get_api_settings

ROLE_RANK = {Role.SUBMITTER.value: 0, Role.REVIEWER.value: 1, Role.ADMIN.value: 2}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    key = request.headers.get("x-api-key")
    if key:
        prefix = api_key_prefix(key)
        row = (
            db.scalar(select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked.is_(False)))
            if prefix
            else None
        )
        if row is None or not hmac.compare_digest(row.key_hash, hash_api_key(key)):
            raise _unauthorized("Invalid API key")
        row.last_used_at = now()
        db.commit()
        user = row.user
    else:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise _unauthorized("Not authenticated")
        try:
            claims = decode_token(auth[7:], get_api_settings().jwt_secret)
        except jwt.PyJWTError as exc:
            raise _unauthorized("Invalid or expired token") from exc
        user = db.get(User, claims.get("sub"))
    if user is None or not user.active:
        raise _unauthorized("Unknown or disabled user")
    request.state.user = user
    return user


def require(role: Role) -> Callable[..., User]:
    """Dependency: the caller's role must be ``role`` or higher (admin > reviewer >
    submitter)."""

    def check(user: User = Depends(current_user)) -> User:
        if ROLE_RANK[user.role] < ROLE_RANK[role.value]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires the {role.value} role")
        return user

    return check


def audit(
    db: Session,
    user: User | None,
    action: str,
    target_type: str,
    target_id: str | None = None,
    **detail: object,
) -> None:
    db.add(
        AuditLog(
            actor_id=user.id if user else None,
            actor_email=user.email if user else None,
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
        )
    )
