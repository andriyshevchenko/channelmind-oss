"""User account registry — the tenant boundary for a multi-tenant deployment.

Every bot, corpus and source belongs to exactly one account, so all data access
is scoped by ``owner_id``. Accounts are created on first Google sign-in and keyed
internally by an opaque id; the Google ``sub`` claim is the stable external
identity we upsert against (email can change, ``sub`` does not).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json


@dataclass
class User:
    id: str
    google_sub: str
    email: str
    name: str
    picture: str
    created_at: str
    plan: str = "beta"  # billing/quota tier; see plans.py. Default keeps pre-plan records loadable.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserStore:
    """JSON-backed account registry, keyed by internal id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "accounts.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def get(self, user_id: str) -> User | None:
        rec = self._load().get(user_id)
        return User(**rec) if rec else None

    def get_by_sub(self, google_sub: str) -> User | None:
        for rec in self._load().values():
            if rec.get("google_sub") == google_sub:
                return User(**rec)
        return None

    def delete(self, user_id: str) -> User | None:
        """Remove the account record (the last step of account erasure).

        Returns the deleted user, or ``None`` if no such record existed (so a
        double-submit erasure is idempotent rather than an error).
        """
        with file_lock(self._registry):
            data = self._load()
            rec = data.pop(user_id, None)
            if rec is None:
                return None
            atomic_write_json(self._registry, data)
            return User(**rec)

    def list(self) -> list[User]:
        return [User(**rec) for rec in self._load().values()]

    def upsert_google(
        self, google_sub: str, email: str, name: str, picture: str = ""
    ) -> User:
        """Create the account on first sign-in, or refresh its profile fields."""
        with file_lock(self._registry):
            data = self._load()
            for uid, rec in data.items():
                if rec.get("google_sub") == google_sub:
                    rec["email"] = email or rec.get("email", "")
                    rec["name"] = name or rec.get("name", "")
                    rec["picture"] = picture or rec.get("picture", "")
                    data[uid] = rec
                    atomic_write_json(self._registry, data)
                    return User(**rec)
            user = User(
                id=uuid.uuid4().hex[:12],
                google_sub=google_sub,
                email=email or "",
                name=name or "",
                picture=picture or "",
                created_at=_now(),
            )
            data[user.id] = asdict(user)
            atomic_write_json(self._registry, data)
            return user
