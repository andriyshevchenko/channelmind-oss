"""Per-bot beta engagement metrics, keyed by (bot, UTC day).

This is measurement only — no limits, no gating anywhere. It records, per bot and
per calendar day, how many chat turns happened, how many tokens they cost, and how
each turn was routed (a real grounded ``question``, casual ``smalltalk``, or a
retrieval-floor ``decline``). Those three counters let us read the friends-beta's
two key signals: per-bot **active days** + engagement, and the **deflect rate**
(smalltalk + decline over all turns) that will later gate the self-fallthrough.

Turns are captured at the SINGLE chat choke point (``bot_service.chat`` /
``chat_stream``) exactly once each, the same place owner token usage is billed, so
every surface (web authed + guest, Telegram text/voice, stream + non-stream) is
covered by one hook. Persistence mirrors :mod:`ytrag.usage`: a small JSON file
committed atomically under a per-path lock, so it is durable across restarts and
safe under the app's concurrent writers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

# The three ways a chat turn is routed (see rag.Assistant / bot_service kind
# derivation). ``question`` = a grounded answer with sources; ``smalltalk`` = the
# greeting/thanks bypass; ``decline`` = the retrieval-floor "not covered" reply.
KIND_QUESTION = "question"
KIND_SMALLTALK = "smalltalk"
KIND_DECLINE = "decline"

# Map each turn ``kind`` to its per-day counter key. Kept explicit (rather than
# pluralizing on the fly) so the stored schema is stable and a typo can't silently
# create a stray counter. The set of valid kinds is exactly these keys.
_KIND_COUNTER = {
    KIND_QUESTION: "questions",
    KIND_SMALLTALK: "smalltalk",
    KIND_DECLINE: "declines",
}
# A turn that DEFLECTS instead of answering from the corpus — smalltalk or a
# decline. Their share of all turns is the deflect_rate (self-fallthrough signal).
_DEFLECT_COUNTERS = ("smalltalk", "declines")

# The numeric fields every day row and the totals carry, so a fresh row starts
# zeroed and a rollup can sum a fixed, known set of keys.
_COUNTER_KEYS = ("messages", "tokens", "questions", "smalltalk", "declines")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    """The current UTC calendar day as ``"YYYY-MM-DD"`` — the metric bucket key."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _empty_day() -> dict:
    return {key: 0 for key in _COUNTER_KEYS}


class MetricsStore:
    """JSON-backed per-bot engagement aggregate, keyed by bot id then UTC day."""

    def __init__(self, data_dir: Path):
        self._root = Path(data_dir)
        self._registry = self._root / "metrics.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def record_turn(self, bot_id: str, *, kind: str, tokens: int) -> None:
        """Add ONE completed chat turn to ``bot_id``'s counters for today (UTC).

        Increments the day's ``messages`` and ``tokens`` and the per-``kind`` counter
        under a single file lock (read-modify-write), so concurrent chat surfaces can
        record turns without losing an update. An unknown ``kind`` is ignored (a turn
        with no derivable route is not counted) and negative tokens clamp to zero, so a
        bad caller can never corrupt the aggregate. Never raises on a valid call.
        """
        counter = _KIND_COUNTER.get(kind)
        if counter is None:
            return
        tokens = max(0, int(tokens))
        day = _today()
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id) or {"days": {}}
            days = rec.setdefault("days", {})
            row = days.get(day) or _empty_day()
            row["messages"] += 1
            row["tokens"] += tokens
            row[counter] += 1
            days[day] = row
            rec["updated_at"] = _now()
            data[bot_id] = rec
            atomic_write_json(self._registry, data)

    def summary(self, bot_id: str) -> dict:
        """Per-day rows + totals + active_days + deflect_rate for one bot.

        ``days`` is the per-UTC-day rows sorted oldest-first; ``totals`` sums every
        counter across them; ``active_days`` is the count of distinct days with at
        least one message (the beta's retention signal); ``deflect_rate`` is
        (smalltalk + declines) / messages, ``0.0`` when the bot has no turns yet. A
        bot with nothing recorded returns the same shape with empty/zero values.
        """
        rec = self._load().get(bot_id) or {}
        days_map = rec.get("days") or {}
        rows = [{"date": day, **days_map[day]} for day in sorted(days_map)]
        totals = {key: 0 for key in _COUNTER_KEYS}
        active_days = 0
        for row in rows:
            for key in _COUNTER_KEYS:
                totals[key] += row.get(key, 0)
            if row.get("messages", 0) >= 1:
                active_days += 1
        messages = totals["messages"]
        deflected = sum(totals[key] for key in _DEFLECT_COUNTERS)
        deflect_rate = round(deflected / messages, 4) if messages else 0.0
        return {
            "bot_id": bot_id,
            "days": rows,
            "totals": totals,
            "active_days": active_days,
            "deflect_rate": deflect_rate,
        }
