"""Passwords, JWTs and API keys (NFR-SEC-02).

* Passwords: scrypt (memory-hard, in Python's standard library), random 16-byte salt,
  compared in constant time. Stored as ``scrypt$n$r$p$salt$hash`` so the cost can be raised
  later without breaking old hashes.
* Users get a JWT (HS256) after logging in: ``sub`` = user id, ``role``, ``exp``.
* API clients get a key ``fdk_<prefix>_<secret>``. Only SHA-256(key) is stored (a fast hash is
  fine here: the key is 32 random bytes, not a guessable password); the prefix finds the row.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    got = hashlib.scrypt(
        password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p)
    )
    return hmac.compare_digest(got, base64.b64decode(digest))


def create_token(user_id: str, role: str, secret: str, minutes: int) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user_id, "role": role, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(claims, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict:
    """Claims of a valid, unexpired token; raises ``jwt.PyJWTError`` otherwise."""
    return jwt.decode(token, secret, algorithms=["HS256"])


def new_api_key() -> tuple[str, str, str]:
    """(full key shown once, prefix stored in clear, SHA-256 hex stored)."""
    prefix = secrets.token_hex(4)
    key = f"fdk_{prefix}_{secrets.token_urlsafe(32)}"
    return key, prefix, hash_api_key(key)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def api_key_prefix(key: str) -> str | None:
    parts = key.split("_", 2)
    return parts[1] if len(parts) == 3 and parts[0] == "fdk" else None
