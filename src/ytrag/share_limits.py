"""Guest-traffic caps and rate limits for the private-share endpoint (Phase A).

A guest reaches a bot through an unguessable link with **no login**, and every
answer runs on — and is billed to — the bot OWNER's quota/key. These limits are
the backstop that stops one shared link from draining the owner's budget or the
shared box, and they double as the legal "small circle" enforcement.

The app runs as a **single replica** (see ``PRODUCTION.md`` — Telegram's
``getUpdates`` forbids horizontal scale), so simple in-process counters are
sufficient: no Redis, no cross-process coordination. Counters live in memory and
reset on restart, which is fine for a daily cap on a small friends circle.

The thresholds now live on the bot OWNER's plan (``plans.Plan`` guest caps); the
caller resolves the owner's plan and passes the caps into :meth:`ShareLimiter.check`
and :func:`sanitize_guest_history`. The module-level constants below remain ONLY
as fallback defaults for callers that don't pass a plan (and to keep old imports
working) — the plan is the source of truth.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# ---- tunable policy ---------------------------------------------------
# Per-bot daily cap on guest messages (billed to the owner). A bot may override
# this downward via ``Bot.share_message_cap``; 0 there means "use this default".
GUEST_DAILY_MESSAGE_CAP = 50
# Per-client (IP) pacing: minimum seconds between two messages…
GUEST_MIN_INTERVAL_SEC = 3.0
# …and a rolling-hour ceiling per client.
GUEST_HOURLY_LIMIT = 30
# Per-bot soft cap on how many distinct clients (IPs) may chat in a day —
# the "small circle" guard. Beyond it, new visitors are denied gracefully.
DISTINCT_GUEST_CAP = 10
# Reject a single guest message longer than this (memory/prompt-abuse guard).
GUEST_MESSAGE_MAX_CHARS = 2000
# Guest-supplied chat history is clamped on THREE axes before it reaches the
# owner-billed prompt: turns, per-turn size, and total size. The count clamp
# alone is not enough — a guest could send 10 turns of 1 MB each and burn the
# owner's tokens, so we also cap each turn and the concatenated whole.
GUEST_HISTORY_MAX_TURNS = 10
GUEST_HISTORY_MAX_CHARS = 8000

_DAY_SECONDS = 86_400
_HOUR_SECONDS = 3_600


def sanitize_guest_history(
    history: object,
    *,
    max_turns: int = GUEST_HISTORY_MAX_TURNS,
    max_item_chars: int = GUEST_MESSAGE_MAX_CHARS,
    max_total_chars: int = GUEST_HISTORY_MAX_CHARS,
) -> list[dict]:
    """Make guest-supplied chat history safe to splice into the owner-billed prompt.

    A guest controls this payload entirely, so it is both a prompt-injection
    surface and a token-burn lever. We:

    1. Whitelist ``role`` to ``user``/``assistant`` only — any ``system`` or
       unknown-role turn (which could rewrite the grounding rules) or turn with
       no content is dropped.
    2. Coerce ``content`` to a string and truncate each turn to ``max_item_chars``.
    3. Keep only the most recent ``max_turns`` turns, then drop the oldest ones
       until the concatenated size fits ``max_total_chars``.

    The authed chat path does not go through here, so it is unaffected.
    """
    if not isinstance(history, list):
        return []
    cleaned: list[dict] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = item.get("content")
        if content is None:
            continue
        content = str(content)
        if not content:
            continue
        if len(content) > max_item_chars:
            content = content[:max_item_chars]
        cleaned.append({"role": role, "content": content})
    # Cap the number of turns first (keep the most recent).
    cleaned = cleaned[-max_turns:]
    # Then bound the total concatenated size, dropping oldest turns until fit.
    total = sum(len(m["content"]) for m in cleaned)
    while cleaned and total > max_total_chars:
        total -= len(cleaned.pop(0)["content"])
    return cleaned


@dataclass(frozen=True)
class Decision:
    """Outcome of a limit check. ``ok`` gates the chat; ``status``/``error`` shape
    the JSON rejection sent to the guest when it doesn't pass."""
    ok: bool
    status: int = 200
    error: str = ""


class ShareLimiter:
    """In-process, thread-safe counters for guest traffic on shared bots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # bot_id -> {"day": int, "count": int, "ips": set[str]}
        self._bot: dict[str, dict] = {}
        # (bot_id, ip) -> {"last": float, "hits": list[float]}
        self._client: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _day(now: float) -> int:
        return int(now // _DAY_SECONDS)

    def _bucket(self, bot_id: str, day: int) -> dict:
        b = self._bot.get(bot_id)
        if b is None or b["day"] != day:
            b = {"day": day, "count": 0, "ips": set()}
            self._bot[bot_id] = b
        return b

    def today_count(self, bot_id: str, now: float | None = None) -> int:
        """Guest messages recorded for this bot today — read-only, for the UI."""
        now = time.time() if now is None else now
        day = self._day(now)
        with self._lock:
            b = self._bot.get(bot_id)
            return b["count"] if b is not None and b["day"] == day else 0

    def check(
        self,
        bot_id: str,
        client_ip: str,
        *,
        daily_cap: int = GUEST_DAILY_MESSAGE_CAP,
        min_interval_sec: float = GUEST_MIN_INTERVAL_SEC,
        hourly_limit: int = GUEST_HOURLY_LIMIT,
        distinct_guest_cap: int = DISTINCT_GUEST_CAP,
        now: float | None = None,
    ) -> Decision:
        """Atomically apply rate limit → distinct-guest cap → daily cap, and only
        record the message (advancing all counters) if every check passes.

        The caps come from the bot owner's resolved plan (the caller passes them);
        the parameter defaults mirror the module constants as a fallback.
        """
        now = time.time() if now is None else now
        day = self._day(now)
        with self._lock:
            ckey = (bot_id, client_ip)
            c = self._client.get(ckey) or {"last": 0.0, "hits": []}

            # 1) per-client pacing
            if now - c["last"] < min_interval_sec:
                return Decision(False, 429,
                                "You're sending messages too quickly — give it a few seconds.")
            hits = [t for t in c["hits"] if now - t < _HOUR_SECONDS]
            if len(hits) >= hourly_limit:
                return Decision(False, 429,
                                "Hourly message limit reached for this link. Try again later.")

            # 2) distinct-guest soft cap (small-circle guard)
            b = self._bucket(bot_id, day)
            if client_ip not in b["ips"] and len(b["ips"]) >= distinct_guest_cap:
                return Decision(False, 429,
                                "This shared link has reached its visitor limit for today.")

            # 3) per-bot daily message cap (billed to the owner)
            if b["count"] >= max(1, daily_cap):
                return Decision(False, 429,
                                "This bot has reached its daily message limit. "
                                "Please try again tomorrow.")

            # record — all checks passed
            hits.append(now)
            c["last"] = now
            c["hits"] = hits
            self._client[ckey] = c
            b["ips"].add(client_ip)
            b["count"] += 1
            return Decision(True)


# One process-wide limiter (single replica → shared in-memory state is correct).
limiter = ShareLimiter()
