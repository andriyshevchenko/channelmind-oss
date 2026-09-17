"""Passive token-usage & cost accounting, keyed by bot owner.

This is bookkeeping only: it records what each tenant spent so the operator can
*see* it — there is no limit, quota or gating anywhere. Usage is captured at the
single chat choke point and rolled up per user (and per model) in ``usage.json``.

Keep this module free of an ``llm`` import: ``llm.py`` imports ``Usage`` from
here, so a reverse import would create a cycle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

logger = logging.getLogger(__name__)

# Approximate public list prices, USD per 1M tokens: (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-opus-4-7": (15.0, 75.0),
    "anthropic/claude-opus-4.8": (15.0, 75.0),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "gpt-4o": (2.5, 10.0),
}

# Fail-SAFE fallback for an unknown/unpriced token model, USD per 1M tokens:
# (input, output). Deliberately set at/above the most expensive known model so an
# unpriced Managed model still ACCRUES cost — otherwise a zero price would
# silently disable ``quotas.over_budget`` no matter the volume. Erring high keeps
# the budget gate armed; known models below are never affected.
_DEFAULT_PRICE_PER_1M: tuple[float, float] = (15.0, 75.0)

# Unpriced models already warned about — so a hot loop of unpriced-model calls
# logs ONE warning per distinct model, not one per call. Bounded by the (small)
# number of distinct model names ever seen.
_warned_unpriced: set[str] = set()

# Approximate public list prices for speech-to-text, USD per audio MINUTE. Whisper
# is billed by audio duration (not tokens), so transcription cost is metered
# separately from the token-priced chat/vision models above. Unknown model -> a
# sensible non-zero default so audio spend still accrues toward the owner budget.
TRANSCRIPTION_PRICES: dict[str, float] = {
    "whisper-large-v3": 0.00185,  # Groq
    "whisper-large-v3-turbo": 0.0004,  # Groq
    "whisper-1": 0.006,  # OpenAI
}
_DEFAULT_TRANSCRIPTION_PRICE_PER_MIN = 0.006


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Sum two usages (Inc1.4c: fold the router call into the turn's total).

        Lets the chat core combine the cheap understanding call's tokens with the
        answer call's tokens into ONE usage the caller bills exactly once, without a
        second accounting record.
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


# Rough characters-per-token ratio for English-ish text. Used ONLY as a
# mid-stream disconnect fallback (see ``estimate_output_tokens``) — never for a
# clean call, which always carries the provider's authoritative token counts.
_CHARS_PER_TOKEN = 4


def estimate_output_tokens(char_count: int) -> int:
    """Approximate output-token count from produced character count (~4 chars/token).

    A streaming provider reports real usage ONLY in its terminal chunk, so if an
    SSE client disconnects mid-answer that chunk never arrives and the produced
    tokens would otherwise bill as ZERO. This gives the billing fallback a
    non-zero, order-correct estimate from the count of characters streamed so far.
    Deliberately a floor-ish heuristic (undercounts hidden reasoning tokens);
    a non-positive count yields 0.
    """
    if char_count <= 0:
        return 0
    return max(1, char_count // _CHARS_PER_TOKEN)


def estimate_cost(model: str, usage: Usage) -> float:
    """USD cost for a call; unknown model -> conservative non-zero fallback.

    A known/priced model is costed exactly as before. An UNKNOWN model must never
    cost 0.0 — that would silently disable the monthly budget gate
    (``quotas.over_budget``) for an unpriced Managed model at any volume. Instead
    we fall back to a deliberately-high default price so spend still accrues, and
    log a masked warning (model name only, never keys/secrets) so the operator can
    add a real price. Never crashes.
    """
    priced = PRICES.get(model)
    if priced is None:
        if model not in _warned_unpriced:
            _warned_unpriced.add(model)
            logger.warning(
                "usage: no price for model %r; using conservative fallback so budget "
                "still accrues — add it to PRICES for accurate costing", model,
            )
        in_price, out_price = _DEFAULT_PRICE_PER_1M
    else:
        in_price, out_price = priced
    cost = usage.prompt_tokens / 1e6 * in_price + usage.completion_tokens / 1e6 * out_price
    return round(cost, 6)


def estimate_transcription_cost(model: str, seconds: float) -> float:
    """USD cost to transcribe ``seconds`` of audio with ``model``.

    Whisper is priced per audio minute. An unknown model falls back to a sensible
    non-zero per-minute default so audio spend is never silently free; negative or
    missing durations clamp to zero.
    """
    per_min = TRANSCRIPTION_PRICES.get(model, _DEFAULT_TRANSCRIPTION_PRICE_PER_MIN)
    return round(max(0.0, seconds) / 60.0 * per_min, 6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _month_key(now: datetime | None = None) -> str:
    """The current UTC calendar month as ``"YYYY-MM"`` — the budget window key."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


class UsageStore:
    """JSON-backed per-user usage/cost aggregate, keyed by user id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "usage.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def get(self, user_id: str) -> dict | None:
        return self._load().get(user_id)

    def delete(self, user_id: str) -> bool:
        """Drop a user's usage aggregate (account erasure). True if a record went."""
        with file_lock(self._registry):
            data = self._load()
            if data.pop(user_id, None) is None:
                return False
            atomic_write_json(self._registry, data)
            return True

    def record(self, user_id: str, model: str, usage: Usage) -> None:
        """Add one call's tokens+cost to the user's aggregate and per-model bucket."""
        self._add(
            user_id, model, usage.prompt_tokens, usage.completion_tokens,
            estimate_cost(model, usage),
        )

    def record_transcription(self, user_id: str, model: str, seconds: float) -> None:
        """Add one audio-transcription's cost to the owner's aggregate/budget.

        Whisper is duration-priced with no token counts, so this records zero
        tokens and the per-minute cost against the SAME lifetime + monthly + per
        model rollups ``record`` uses — so audio spend counts toward the plan's
        monthly ceiling exactly like a chat call.
        """
        self._add(user_id, model, 0, 0, estimate_transcription_cost(model, seconds))

    def record_proxy_mb(self, user_id: str, mb: float) -> None:
        """Add transcript-proxy bandwidth (in MB) to a user's monthly + lifetime
        rollups. Metered crudely from downloaded subtitle bytes; arms the per-plan
        monthly proxy ceiling (``quotas.over_bandwidth``). Negative/NaN clamps to 0.
        """
        try:
            mb = float(mb)
        except (TypeError, ValueError):
            return
        if not mb or mb != mb or mb < 0:  # 0, NaN, or negative -> nothing to add
            return
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {
                "total_cost_usd": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "by_model": {},
            }
            rec["proxy_mb"] = round(rec.get("proxy_mb", 0.0) + mb, 4)
            months = rec.setdefault("by_month", {})
            month = _month_key()
            mrec = months.get(month) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            }
            mrec["proxy_mb"] = round(mrec.get("proxy_mb", 0.0) + mb, 4)
            months[month] = mrec
            rec["updated_at"] = _now()
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def monthly_proxy_mb(self, user_id: str, month: str | None = None) -> float:
        """This user's transcript-proxy bandwidth (MB) for a UTC calendar month
        (default: current). ``0.0`` when nothing recorded. Read by the plan's
        ``max_monthly_proxy_mb`` ceiling."""
        month = month or _month_key()
        rec = self.get(user_id) or {}
        mrec = (rec.get("by_month") or {}).get(month) or {}
        return round(mrec.get("proxy_mb", 0.0), 4)

    def _add(
        self, user_id: str, model: str, prompt_tokens: int, completion_tokens: int,
        cost: float,
    ) -> None:
        """Accumulate one metered call (tokens + precomputed cost) for a user.

        The single write path behind ``record`` / ``record_transcription`` so both
        land in the identical lifetime, per-month (budget window) and per-model
        rollups under one file lock.
        """
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {
                "total_cost_usd": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "by_model": {},
            }
            rec["prompt_tokens"] += prompt_tokens
            rec["completion_tokens"] += completion_tokens
            rec["total_cost_usd"] = round(rec["total_cost_usd"] + cost, 6)
            rec["calls"] += 1
            # Per-calendar-month rollup (UTC), added ALONGSIDE the lifetime totals
            # above so the existing UI aggregates are untouched. This is the window
            # the plan's monthly budget reads from.
            months = rec.setdefault("by_month", {})
            month = _month_key()
            mrec = months.get(month) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            }
            mrec["prompt_tokens"] += prompt_tokens
            mrec["completion_tokens"] += completion_tokens
            mrec["cost_usd"] = round(mrec["cost_usd"] + cost, 6)
            mrec["calls"] += 1
            months[month] = mrec
            bucket = rec["by_model"].get(model) or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            }
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 6)
            bucket["calls"] += 1
            rec["by_model"][model] = bucket
            rec["updated_at"] = _now()
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def monthly_cost_usd(self, user_id: str, month: str | None = None) -> float:
        """This user's spend for a UTC calendar month (default: current month).

        Returns ``0.0`` for a user/month with no recorded usage. Used by the plan
        budget (``max_monthly_cost_usd``); the lifetime ``total_cost_usd`` the UI
        already shows is left untouched.
        """
        month = month or _month_key()
        rec = self.get(user_id) or {}
        mrec = (rec.get("by_month") or {}).get(month) or {}
        return round(mrec.get("cost_usd", 0.0), 6)

    def summary(self) -> dict:
        """The full per-user map plus a computed grand total."""
        data = self._load()
        grand = {
            "total_cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "calls": 0,
        }
        for rec in data.values():
            grand["total_cost_usd"] += rec.get("total_cost_usd", 0.0)
            grand["prompt_tokens"] += rec.get("prompt_tokens", 0)
            grand["completion_tokens"] += rec.get("completion_tokens", 0)
            grand["calls"] += rec.get("calls", 0)
        grand["total_cost_usd"] = round(grand["total_cost_usd"], 6)
        return {"users": data, "grand_total": grand}
