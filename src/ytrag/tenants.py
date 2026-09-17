"""Per-tenant entitlements — plan and daily usage counters, keyed by user id.

One record per ``User`` (the tenant boundary). This is the data layer only:
plan selection and daily-message accounting live here; the policy that *enforces*
those limits is applied elsewhere. The daily counter carries the date it applies
to so a new UTC day resets usage without a background job (lazy rollover).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

PLAN_BASIC = "basic"
PLAN_PREMIUM = "premium"
PLANS = frozenset({PLAN_BASIC, PLAN_PREMIUM})


@dataclass
class Tenant:
    user_id: str
    created_at: str
    updated_at: str
    plan: str = PLAN_BASIC
    daily_count: int = 0
    daily_date: str = ""  # YYYY-MM-DD (UTC) the daily_count applies to


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class TenantStore:
    """JSON-backed entitlements registry, keyed by user id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "tenants.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def get(self, user_id: str) -> Tenant | None:
        rec = self._load().get(user_id)
        return Tenant(**rec) if rec else None

    def delete(self, user_id: str) -> bool:
        """Drop a tenant's entitlement record (account erasure). True if one went."""
        with file_lock(self._registry):
            data = self._load()
            if data.pop(user_id, None) is None:
                return False
            atomic_write_json(self._registry, data)
            return True

    def get_or_create(self, user_id: str) -> Tenant:
        """Return the tenant, creating a basic-plan record on first sight."""
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id)
            if rec is not None:
                return Tenant(**rec)
            now = _now()
            tenant = Tenant(user_id=user_id, created_at=now, updated_at=now)
            data[user_id] = asdict(tenant)
            atomic_write_json(self._registry, data)
            return tenant

    def set_plan(self, user_id: str, plan: str) -> Tenant | None:
        if plan not in PLANS:
            raise ValueError(f"Unknown plan: {plan}")
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id)
            if rec is None:
                return None
            rec["plan"] = plan
            rec["updated_at"] = _now()
            data[user_id] = rec
            atomic_write_json(self._registry, data)
            return Tenant(**rec)

    def usage_today(self, user_id: str) -> int:
        """Messages counted for today; 0 if the stored counter is from a past day."""
        rec = self._load().get(user_id)
        if rec is None:
            return 0
        return int(rec.get("daily_count", 0)) if rec.get("daily_date") == _today() else 0

    def record_message(self, user_id: str) -> int:
        """Increment today's counter (resetting first on a new UTC day); return the new count."""
        today = _today()
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id)
            if rec is None:
                now = _now()
                rec = asdict(Tenant(user_id=user_id, created_at=now, updated_at=now))
            if rec.get("daily_date") != today:
                rec["daily_date"] = today
                rec["daily_count"] = 0
            rec["daily_count"] = int(rec.get("daily_count", 0)) + 1
            rec["updated_at"] = _now()
            data[user_id] = rec
            atomic_write_json(self._registry, data)
            return rec["daily_count"]
