"""Named-plan policy registry — the single source of truth for one tenant's limits.

A user's plan is ONE object (:class:`Plan`) that owns everything the rest of the
app must respect for that tenant: the shared-capacity quota limits (bots,
sources, concurrent ingests, videos per channel), the guest/private-share caps
(daily message cap, rate limits, distinct-visitor cap, message/history sizes),
the monthly managed spend budget, and the feature toggles (sharing / BYOK /
managed). Consumers READ from the plan — they never hardcode a constant — so the
whole offering is tuned in this one file.

**Decision (2026-08-07): NO pricing for the MVP — all users are beta.** Every
plan ships with ``price=None``; the field exists so tiers/prices become a data
change here, not a refactor. The two plans are ``BETA_USER`` (the bounded
default) and ``DEVELOPER`` (generous/unlimited, for the operator and devs).

The *effective* plan can be overridden per-email via ``YTRAG_DEVELOPER_EMAILS``
(and the legacy ``YTRAG_ALPHA_EMAILS`` alias) so the operator can grant developer
access without editing the account store.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .accounts import User

# Registry keys / plan names.
BETA_PLAN = "BETA_USER"
DEVELOPER_PLAN = "DEVELOPER"
DEFAULT_PLAN = BETA_PLAN

# Old ``User.plan`` values (and the previous env alias) map onto the new names so
# accounts created before this registry keep loading and resolving correctly.
_LEGACY_NAMES: dict[str, str] = {
    "beta": BETA_PLAN,
    "alpha": DEVELOPER_PLAN,
    "developer": DEVELOPER_PLAN,
}


@dataclass(frozen=True)
class PlanLimits:
    """Shared-capacity quota caps (unchanged shape — the plan widget reads this)."""
    max_bots: int
    max_sources_per_bot: int
    max_sources_total: int
    max_active_jobs: int
    max_videos_per_channel: int


@dataclass(frozen=True)
class Plan:
    """One named plan: the single source of truth for its whole policy.

    ``guest_*``/``distinct_guest_cap`` are the private-share caps that used to be
    module constants in ``share_limits.py``; callers now pass these down from the
    resolved (bot-owner's) plan. ``max_monthly_cost_usd`` is the managed spend
    ceiling (``None`` = unlimited). The feature toggles gate sharing / BYOK /
    managed per plan. ``price`` is ``None`` for every beta plan.
    """
    name: str
    limits: PlanLimits
    # --- guest / private-share caps (billed to the bot owner) ---
    guest_daily_message_cap: int
    guest_min_interval_sec: float
    guest_hourly_limit: int
    distinct_guest_cap: int
    guest_message_max_chars: int
    guest_history_max_turns: int
    guest_history_max_chars: int
    # --- budget: monthly managed spend ceiling; None = unlimited ---
    max_monthly_cost_usd: float | None
    # --- proxy bandwidth: monthly per-user transcript-proxy ceiling in MB;
    # None = unlimited. Sized to the shared proxy budget so a handful of beta
    # users can't drain it in a day (BUG bandwidth-ceiling). Metered crudely from
    # downloaded subtitle bytes into usage.json and enforced at ingest enqueue /
    # mid-import. Lives here so it's tuned as data, never hardcoded. ---
    max_monthly_proxy_mb: int | None = None
    # --- feature toggles ---
    sharing_enabled: bool = True
    byok_enabled: bool = True
    managed_enabled: bool = True
    # Billing/upgrade UI visibility. OFF for the whole beta (no pricing anywhere);
    # flipping it on later is a registry data change, not a UI refactor — the SPA
    # keys the Upgrade/checkout section off this toggle, not a build-time constant.
    upgrades_enabled: bool = False
    # --- pricing (None for beta — a field for the future, not used yet) ---
    price: float | None = None


PLANS: dict[str, Plan] = {
    BETA_PLAN: Plan(
        name=BETA_PLAN,
        limits=PlanLimits(
            max_bots=5,
            max_sources_per_bot=5,
            max_sources_total=15,
            max_active_jobs=3,
            max_videos_per_channel=300,
        ),
        # Bounded defaults == the previous ``share_limits`` module constants.
        guest_daily_message_cap=50,
        guest_min_interval_sec=3.0,
        guest_hourly_limit=30,
        distinct_guest_cap=10,
        guest_message_max_chars=2000,
        guest_history_max_turns=10,
        guest_history_max_chars=8000,
        # A modest monthly managed budget for beta users.
        max_monthly_cost_usd=5.0,
        # Per-user monthly transcript-proxy ceiling. The shared proxy plan is
        # ~3GB/mo; 500MB/user lets ~4 friends import comfortably with headroom for
        # auto-sync + retries, and one user can't drain the whole budget alone.
        max_monthly_proxy_mb=500,
        sharing_enabled=True,
        byok_enabled=True,
        managed_enabled=True,
        upgrades_enabled=False,  # no billing/upgrade UI during the beta
        price=None,
    ),
    DEVELOPER_PLAN: Plan(
        name=DEVELOPER_PLAN,
        limits=PlanLimits(
            max_bots=100,
            max_sources_per_bot=100,
            max_sources_total=1000,
            max_active_jobs=20,
            max_videos_per_channel=20000,
        ),
        guest_daily_message_cap=1000,
        guest_min_interval_sec=0.0,
        guest_hourly_limit=1000,
        distinct_guest_cap=1000,
        guest_message_max_chars=8000,
        guest_history_max_turns=40,
        guest_history_max_chars=64000,
        # Unlimited spend.
        max_monthly_cost_usd=None,
        # Unlimited proxy bandwidth for developers/operator.
        max_monthly_proxy_mb=None,
        sharing_enabled=True,
        byok_enabled=True,
        managed_enabled=True,
        upgrades_enabled=False,  # no billing/upgrade UI during the beta
        price=None,
    ),
}


def _normalize_name(name: str | None) -> str:
    """Map a stored/legacy plan name onto a live registry key (fallback: default)."""
    if not name:
        return DEFAULT_PLAN
    key = name.strip()
    key = _LEGACY_NAMES.get(key.lower(), key)
    return key if key in PLANS else DEFAULT_PLAN


def _developer_emails() -> set[str]:
    """Emails promoted to DEVELOPER via env (new var + legacy ``YTRAG_ALPHA_EMAILS``)."""
    raw = ",".join(
        (os.getenv("YTRAG_DEVELOPER_EMAILS", ""), os.getenv("YTRAG_ALPHA_EMAILS", ""))
    )
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def plan_for(user: User) -> Plan:
    """The :class:`Plan` actually enforced for this user — the single accessor.

    An email on the developer allowlist is promoted to ``DEVELOPER`` (env grant,
    no store edit); otherwise the user's stored plan name (normalized) decides,
    defaulting to ``BETA_USER``.
    """
    if user.email and user.email.strip().lower() in _developer_emails():
        return PLANS[DEVELOPER_PLAN]
    return PLANS[_normalize_name(user.plan)]


# ---- backward-compat helpers (derive from the registry) --------------------
def effective_plan(user: User) -> str:
    """The NAME of the plan enforced for this user (legacy string API)."""
    return plan_for(user).name


def limits_for(plan: "str | Plan") -> PlanLimits:
    """Limits for a plan NAME or a :class:`Plan` object (legacy string API).

    Unknown names fall back to the default tier, matching the old behavior.
    """
    if isinstance(plan, Plan):
        return plan.limits
    return PLANS[_normalize_name(plan)].limits
