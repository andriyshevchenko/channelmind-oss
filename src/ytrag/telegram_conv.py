"""Durable rolling conversation buffer for the Telegram bot.

Telegram chats were stateless: every inbound message was answered in isolation
with no memory of prior turns. This store gives each Telegram conversation a
short, bounded memory so follow-ups ("and when did that happen?") resolve —
without ballooning cost.

Design (see ``docs/telegram-conversation-memory-plan.md``):

* Keyed by ``(telegram_chat_id, bot_id)`` so every chat, on every bot, is
  isolated.
* Stores ONLY CLEAN turns — ``{role:"user", content:<raw question>}`` and
  ``{role:"assistant", content:<final answer text>}``. It NEVER stores the
  retrieved transcript excerpts / RAG context block: that big block is injected
  only for the CURRENT question by ``rag.Assistant.answer``. Keeping history to
  short Q&A text is what keeps input size (and cost) bounded even for strong
  models.
* Bounded on write via :func:`share_limits.sanitize_guest_history` — the SAME
  cleaner + caps the web guest path uses, resolved from the bot OWNER's plan
  (turns / per-item chars / total chars). Oldest turns are trimmed first. The
  role whitelist there also guarantees no ``system`` text can be persisted.
* Durable JSON (atomic write + per-path file lock, mirroring ``JobStore`` /
  settings) so a single-worker app keeps memory across restarts. ``/reset``
  (alias ``/new``) clears one conversation.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json
from .share_limits import sanitize_guest_history

# Cap on the NUMBER of distinct (chat_id:bot_id) conversations kept in the single
# JSON store. Per-conversation turns are already capped; this bounds the count of
# conversations so an unbounded population of chats/bots can't grow the file (and
# the whole-file rewrite on every append) without limit. Oldest-by-``updated_at``
# conversations are evicted on write once the cap is exceeded.
_DEFAULT_MAX_CONVERSATIONS = 5000

# User-facing answer styles.  Keep the durable value deliberately English and
# stable; Telegram renders the localized labels in ``telegram_bot``.
MODE_REFERENCE = "reference"
MODE_THINKING = "thinking"
_MODES = {MODE_REFERENCE, MODE_THINKING}


def _env_max_conversations() -> int:
    raw = (os.getenv("YTRAG_TELEGRAM_MAX_CONVERSATIONS", "") or "").strip()
    try:
        val = int(raw) if raw else _DEFAULT_MAX_CONVERSATIONS
    except ValueError:
        return _DEFAULT_MAX_CONVERSATIONS
    return val if val > 0 else _DEFAULT_MAX_CONVERSATIONS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelegramConvStore:
    """JSON-backed rolling buffer of clean turns, keyed by ``chat_id:bot_id``."""

    def __init__(self, root: Path, max_conversations: int | None = None):
        self._registry = Path(root) / "telegram_conv.json"
        self._max_conversations = (
            max_conversations if max_conversations is not None
            else _env_max_conversations()
        )

    @staticmethod
    def _key(chat_id: object, bot_id: str) -> str:
        return f"{chat_id}:{bot_id}"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    @staticmethod
    def _turns_of(rec: object) -> list[dict]:
        """Pull the turn list out of a stored record (tolerating shapes)."""
        if isinstance(rec, dict):
            turns = rec.get("turns")
        else:
            turns = rec
        return turns if isinstance(turns, list) else []

    def history(self, chat_id: object, bot_id: str) -> list[dict]:
        """The stored clean turns for this conversation (empty when none)."""
        return self._turns_of(self._load().get(self._key(chat_id, bot_id)))

    def mode(self, chat_id: object, bot_id: str) -> str:
        """Selected answer style for this chat/bot.

        OSS build: the «Мислення»/Thinking general-advisor mode is intentionally
        disabled, so every turn resolves to the grounded «Довідник»/Reference path
        regardless of any stored value. (The mode machinery is left in place but
        inert; the /mode toggle is not exposed in this build.)"""
        return MODE_REFERENCE

    def set_mode(self, chat_id: object, bot_id: str, mode: str) -> str:
        """Persist an answer style without disturbing the rolling history."""
        chosen = mode if mode in _MODES else MODE_REFERENCE
        with file_lock(self._registry):
            data = self._load()
            key = self._key(chat_id, bot_id)
            rec = data.get(key)
            rec = dict(rec) if isinstance(rec, dict) else {"turns": self._turns_of(rec)}
            rec["mode"] = chosen
            rec["updated_at"] = _now()
            data[key] = rec
            self._evict(data, keep_key=key)
            atomic_write_json(self._registry, data)
        return chosen

    def append(
        self,
        chat_id: object,
        bot_id: str,
        user_text: str,
        assistant_text: str,
        *,
        max_turns: int,
        max_item_chars: int,
        max_total_chars: int,
    ) -> list[dict]:
        """Append the new user + assistant turn, then clean + trim to the caps.

        Trimming reuses :func:`sanitize_guest_history` so only ``user``/
        ``assistant`` roles survive (no excerpts / system text can leak into
        stored history) and the buffer respects the owner plan's turn-count and
        char caps, dropping the OLDEST turns first. Returns the persisted turns.
        """
        with file_lock(self._registry):
            data = self._load()
            key = self._key(chat_id, bot_id)
            turns = self._turns_of(data.get(key))
            turns = turns + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            turns = sanitize_guest_history(
                turns,
                max_turns=max_turns,
                max_item_chars=max_item_chars,
                max_total_chars=max_total_chars,
            )
            # Preserve the selected answer mode across turns. Appending a turn must
            # NOT reset the chat back to «Довідник» — without this carry-over the
            # record is rewritten as turns-only, so the very next message reads the
            # default mode and the user's «Мислення» selection silently drops.
            prior = data.get(key)
            prior_mode = prior.get("mode") if isinstance(prior, dict) else None
            rec: dict = {"turns": turns, "updated_at": _now()}
            if prior_mode in _MODES:
                rec["mode"] = prior_mode
            data[key] = rec
            self._evict(data, keep_key=key)
            atomic_write_json(self._registry, data)
            return turns

    def _evict(self, data: dict, keep_key: str) -> None:
        """Drop the oldest conversations until ``data`` fits ``max_conversations``.

        Eviction is by ``updated_at`` (oldest first); a legacy record without a
        stamp sorts oldest and is evicted first. ``keep_key`` — the conversation
        just written — is NEVER evicted, so the current write always survives even
        if it happens to be the very oldest by an out-of-order clock.
        """
        cap = self._max_conversations
        if cap <= 0 or len(data) <= cap:
            return

        def _stamp(k: str) -> str:
            rec = data.get(k)
            return (rec.get("updated_at") or "") if isinstance(rec, dict) else ""

        evictable = sorted((k for k in data if k != keep_key), key=_stamp)
        excess = len(data) - cap
        for k in evictable[:excess]:
            data.pop(k, None)

    def clear(self, chat_id: object, bot_id: str) -> bool:
        """Drop conversation history while preserving a selected answer mode."""
        with file_lock(self._registry):
            data = self._load()
            key = self._key(chat_id, bot_id)
            if key not in data:
                return False
            rec = data.get(key)
            mode = rec.get("mode") if isinstance(rec, dict) else None
            if mode in _MODES:
                data[key] = {"turns": [], "mode": mode, "updated_at": _now()}
            else:
                data.pop(key, None)
            atomic_write_json(self._registry, data)
            return True
