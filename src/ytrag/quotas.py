"""Quota accounting and enforcement for per-user plan limits.

Counts a user's live consumption (bots, sources, active ingest jobs) against
their effective plan's :class:`~ytrag.plans.PlanLimits` and turns an over-limit
condition into a clear, user-facing rejection message. Kept out of the HTTP
layer so the counting lives in one place and the endpoints stay thin: each
``check_*`` returns ``None`` when allowed or an error string when the caller
should reject (HTTP 429).
"""
from __future__ import annotations

from dataclasses import dataclass

from .accounts import User
from .bots import Bot, BotStore
from .config import Config
from .ingest_jobs import STATUS_QUEUED, STATUS_RUNNING, JobStore
from .plans import plan_for
from .usage import UsageStore


def _bots(cfg: Config, user_id: str) -> list[Bot]:
    return BotStore(cfg.data_dir).list_for(user_id)


def count_bots(cfg: Config, user_id: str) -> int:
    return len(_bots(cfg, user_id))


def count_sources_total(cfg: Config, user_id: str) -> int:
    return sum(len(b.sources) for b in _bots(cfg, user_id))


def count_active_jobs(cfg: Config, user_id: str) -> int:
    """Ingest jobs that occupy the shared queue: queued + running."""
    jobs = JobStore(cfg.data_dir).list_for_user(user_id)
    return sum(1 for j in jobs if j.status in (STATUS_QUEUED, STATUS_RUNNING))


@dataclass(frozen=True)
class Usage:
    bots: int
    sources: int
    active_jobs: int


def usage_for(cfg: Config, user_id: str) -> Usage:
    """A single snapshot of the user's consumption for the plan widget."""
    bots = _bots(cfg, user_id)
    return Usage(
        bots=len(bots),
        sources=sum(len(b.sources) for b in bots),
        active_jobs=count_active_jobs(cfg, user_id),
    )


# ---- enforcement (return None = allowed, str = reject with this message) ----
def check_can_create_bot(cfg: Config, user: User) -> str | None:
    plan = plan_for(user)
    limits = plan.limits
    used = count_bots(cfg, user.id)
    if used >= limits.max_bots:
        return (
            f"Your {plan.name} plan allows up to {limits.max_bots} bots "
            f"(you have {used}). Delete a bot or upgrade to add more."
        )
    return None


def check_can_add_source(cfg: Config, user: User, bot: Bot) -> str | None:
    plan = plan_for(user)
    limits = plan.limits
    if len(bot.sources) >= limits.max_sources_per_bot:
        return (
            f"This bot already has {len(bot.sources)} sources, the maximum for "
            f"your {plan.name} plan ({limits.max_sources_per_bot} per bot)."
        )
    total = count_sources_total(cfg, user.id)
    if total >= limits.max_sources_total:
        return (
            f"You've reached your {plan.name} plan's total of "
            f"{limits.max_sources_total} sources across all bots (you have {total})."
        )
    return None


def check_can_enqueue_ingest(cfg: Config, user: User) -> str | None:
    plan = plan_for(user)
    limits = plan.limits
    active = count_active_jobs(cfg, user.id)
    if active >= limits.max_active_jobs:
        return (
            f"You already have {active} ingests queued or running, the maximum for "
            f"your {plan.name} plan ({limits.max_active_jobs}). Wait for one to finish "
            f"before starting another."
        )
    if over_bandwidth(cfg, user):
        return (
            f"You've reached your {plan.name} plan's monthly indexing limit — "
            f"you've indexed the maximum videos allowed for this month. New "
            f"imports are paused until next month; your existing bots and chat still work."
        )
    return None


def monthly_proxy_mb(cfg: Config, user_id: str) -> float:
    """This user's transcript-proxy bandwidth used this calendar month, in MB."""
    return UsageStore(cfg.data_dir).monthly_proxy_mb(user_id)


def over_bandwidth(cfg: Config, user: User) -> bool:
    """Whether the user has hit their plan's monthly transcript-proxy ceiling.

    A plan with ``max_monthly_proxy_mb = None`` (e.g. ``DEVELOPER``) is never over.
    Enforced at ingest ENQUEUE (:func:`check_can_enqueue_ingest`) and, for a single
    large channel that crosses the line mid-run, by the ingest worker's per-video
    probe — so hitting it stops further ingest with a clear message while existing
    bots and chat keep working.
    """
    cap = plan_for(user).max_monthly_proxy_mb
    if cap is None:
        return False
    return monthly_proxy_mb(cfg, user.id) >= cap


def cap_video_limit(user: User, requested: int | None) -> int:
    """Clamp the per-channel video count to the plan cap.

    ``requested`` is ``None`` (caller asked for *all* videos) or a positive int;
    either way the effective limit never exceeds the plan's cap, so one channel
    can't queue unbounded work.
    """
    cap = plan_for(user).limits.max_videos_per_channel
    if requested is None or requested <= 0:
        return cap
    return min(requested, cap)


def over_budget(cfg: Config, user: User) -> bool:
    """Whether the user has hit their plan's monthly managed spend ceiling.

    Enforced BEFORE any model call on every chat path via the shared
    ``user_settings.resolve_chat_for_user`` gate: the web authed/guest endpoints
    (``app.py``) and the Telegram poller (``telegram_bot.py``) all reject a
    Managed chat with a 429 / friendly message once the owner is over budget. A
    BYOK chat runs on the user's own key and is never budget-gated. A plan with
    ``max_monthly_cost_usd = None`` (e.g. ``DEVELOPER``) is never over budget.
    """
    limit = plan_for(user).max_monthly_cost_usd
    if limit is None:
        return False
    spent = UsageStore(cfg.data_dir).monthly_cost_usd(user.id)
    return spent >= limit
