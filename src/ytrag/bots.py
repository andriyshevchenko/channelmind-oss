"""Bot registry — a bot is the first-class unit users create and talk to.

Each bot owns exactly one isolated corpus (its own Chroma collection, via
``CorpusStore``) and its own persona. A bot can ingest MANY sources (several
YouTube channels and reference documents) which all feed that one corpus, so the
assistant reasons across everything the bot was taught. Sources are stored inline
on the bot record because they are small metadata rows mutated together with the
bot; the heavy transcript/vector data lives under the corpus.

Every bot belongs to an ``owner_id`` (a ``User``) — the multi-tenant boundary.
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

TYPE_YOUTUBE = "youtube"
TYPE_DOCUMENT = "document"

SRC_PENDING = "pending"
SRC_BUILDING = "building"
SRC_READY = "ready"
SRC_ERROR = "error"


@dataclass
class BotSource:
    id: str
    type: str            # youtube | document
    key: str             # normalized de-dup key (channel handle / filename)
    label: str           # display name
    author: str          # attribution stamped onto every chunk
    # A YouTube source is a whole ``channel``, a single ``playlist``, or a single
    # ``video`` (all keep type="youtube"; ``kind`` is the sub-dimension). Fixed at
    # creation from the add form. Dataclass default keeps pre-playlist records
    # loadable.
    kind: str = "channel"
    status: str = SRC_PENDING
    item_count: int = 0
    chunk_count: int = 0
    added_at: str = ""
    channel_title: str = ""
    channel_avatar: str = ""
    # ---- auto-sync (Phase K) -----------------------------------------
    # ``sync_freq`` is one of off|daily|weekly|monthly, or "" meaning "inherit the
    # owner's user-level default" (resolved at scan time in ``autosync``). Only
    # meaningful for YouTube sources. ``last_sync_at`` stamps the last incremental
    # sync attempt so the scheduler doesn't re-scan an unchanged channel every
    # tick. Empty defaults keep pre-Phase-K source records loadable.
    sync_freq: str = ""
    last_sync_at: str = ""
    # Stamped ONCE, when this source's first ingest job completes successfully
    # (BUG-020). Auto-sync is gated on it: a just-added, still-importing, or
    # cancelled channel has an empty value and is therefore NEVER auto-synced,
    # and the sync interval is measured from here until the first real sync.
    # Empty default keeps pre-existing source records loadable.
    first_indexed_at: str = ""
    # Comma-joined subtitle languages chosen at import time ("" = English only,
    # "all" = every language the channel offers). Persisted so auto-sync re-syncs
    # with the SAME selection instead of silently falling back to en-only. Empty
    # default keeps pre-existing source records loadable.
    sync_langs: str = ""


@dataclass
class Bot:
    id: str
    owner_id: str
    name: str
    persona: str
    description: str
    corpus_id: str
    language: str = ""   # bot's language: drives reply language AND service phrases
    # ---- persona (Phase J) -------------------------------------------
    # ``persona`` is the builder-generated identity. ``custom_prompt`` is an
    # optional free-form system prompt that, when non-empty, OVERRIDES the builder
    # persona at chat time (precedence lives in ``bot_service.effective_persona``).
    # Both persist independently: clearing ``custom_prompt`` back to "" reverts to
    # the builder persona. Empty default keeps pre-Phase-J bot records loadable.
    custom_prompt: str = ""
    telegram_token: str = ""
    telegram_username: str = ""
    created_at: str = ""
    sources: list[BotSource] = field(default_factory=list)
    # ---- private share link (Phase A) --------------------------------
    # An unguessable token IS the credential: friends need no Google account and
    # the owner can revoke or rotate it without touching the (guessable) bot id.
    # Empty token = sharing off. ``share_revoked`` kills a token without minting a
    # new one; ``share_message_cap`` (0 = use the global default) caps guest
    # traffic per day. Dataclass defaults keep pre-share bot records loadable.
    share_token: str = ""
    share_created_at: str = ""
    share_revoked: bool = False
    share_message_cap: int = 0
    # ---- follow-up suggestions (Fable feel-win #2) -------------------
    # When True the grounded chat prompt may end a substantive answer with ONE subtle
    # line suggesting 2-3 next questions (the Gemini proactivity pattern, done inside
    # the SAME answer call — no extra cost). Owner-controllable per bot so it can never
    # annoy: False removes the guidance entirely. Default True keeps the feature on for
    # new bots; the bool default keeps pre-existing bot records loadable.
    suggest_followups: bool = True
    # ---- YouTube Shorts inclusion (BUG-009) --------------------------
    # When False (the default) channel enumeration EXCLUDES Shorts so a bot is
    # trained on the channel's substantive long-form videos, not throwaway clips.
    # Owner-flippable per bot: True re-includes Shorts on the next ingest/sync.
    # Default False makes new bots skip Shorts; the bool default keeps pre-BUG-009
    # bot records loadable.
    include_shorts: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bot_from_rec(rec: dict) -> Bot:
    rec = dict(rec)
    rec["sources"] = [BotSource(**s) for s in rec.get("sources", [])]
    return Bot(**rec)


class BotStore:
    """JSON-backed registry of bots, keyed by id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "bots.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    # ---- reads --------------------------------------------------------
    def get(self, bot_id: str) -> Bot | None:
        rec = self._load().get(bot_id)
        return _bot_from_rec(rec) if rec else None

    def list_for(self, owner_id: str) -> list[Bot]:
        rows = [
            _bot_from_rec(rec)
            for rec in self._load().values()
            if rec.get("owner_id") == owner_id
        ]
        return sorted(rows, key=lambda b: b.created_at)

    def all(self) -> list[Bot]:
        """Every bot in the registry, oldest first (owner-independent).

        Used by the auto-sync scheduler, which scans across all tenants."""
        rows = [_bot_from_rec(rec) for rec in self._load().values()]
        return sorted(rows, key=lambda b: b.created_at)

    def owned(self, bot_id: str, owner_id: str) -> Bot | None:
        """Return the bot only if it belongs to this owner (tenant guard)."""
        bot = self.get(bot_id)
        if bot is None or bot.owner_id != owner_id:
            return None
        return bot

    def find_source_by_key(self, bot_id: str, type_: str, key: str) -> BotSource | None:
        bot = self.get(bot_id)
        if bot is None:
            return None
        for s in bot.sources:
            if s.type == type_ and s.key == key:
                return s
        return None

    def find_by_share_token(self, token: str) -> Bot | None:
        """Resolve an *active* share token to its bot — scans ALL bots (owner-independent).

        Returns ``None`` for an empty, unknown, or revoked token so a guest link
        that was turned off (or never existed) simply fails to resolve.
        """
        if not token:
            return None
        for rec in self._load().values():
            if secrets.compare_digest(rec.get("share_token") or "", token) and not rec.get("share_revoked", False):
                return _bot_from_rec(rec)
        return None

    # ---- writes -------------------------------------------------------
    def create(
        self, owner_id: str, name: str, persona: str, description: str, corpus_id: str,
        language: str = "",
    ) -> Bot:
        bot = Bot(
            id=uuid.uuid4().hex[:12],
            owner_id=owner_id,
            name=name,
            persona=persona or "",
            description=description or "",
            corpus_id=corpus_id,
            language=language or "",
            created_at=_now(),
        )
        with file_lock(self._registry):
            data = self._load()
            data[bot.id] = asdict(bot)
            atomic_write_json(self._registry, data)
        return bot

    def update(self, bot_id: str, **fields) -> Bot | None:
        allowed = {"name", "persona", "custom_prompt", "description", "language",
                   "telegram_token", "telegram_username", "suggest_followups",
                   "include_shorts"}
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            for k, v in fields.items():
                if k in allowed and v is not None:
                    rec[k] = v
            data[bot_id] = rec
            atomic_write_json(self._registry, data)
            return _bot_from_rec(rec)

    def delete(self, bot_id: str) -> Bot | None:
        with file_lock(self._registry):
            data = self._load()
            rec = data.pop(bot_id, None)
            if rec is None:
                return None
            atomic_write_json(self._registry, data)
            return _bot_from_rec(rec)

    # ---- share token --------------------------------------------------
    def create_share_token(self, bot_id: str) -> Bot | None:
        """Return the bot holding an active share token, minting one if needed.

        Idempotent: an already-active token is returned unchanged. A revoked or
        absent token is (re)generated, which re-activates sharing on a fresh,
        unguessable credential.
        """
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            if not rec.get("share_token") or rec.get("share_revoked", False):
                rec["share_token"] = secrets.token_urlsafe(24)
                rec["share_created_at"] = _now()
                rec["share_revoked"] = False
                data[bot_id] = rec
                atomic_write_json(self._registry, data)
            return _bot_from_rec(rec)

    def rotate_share_token(self, bot_id: str) -> Bot | None:
        """Mint a brand-new token, invalidating the old link immediately."""
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            rec["share_token"] = secrets.token_urlsafe(24)
            rec["share_created_at"] = _now()
            rec["share_revoked"] = False
            data[bot_id] = rec
            atomic_write_json(self._registry, data)
            return _bot_from_rec(rec)

    def revoke_share_token(self, bot_id: str) -> Bot | None:
        """Turn sharing off — the existing link stops resolving at once."""
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            rec["share_revoked"] = True
            data[bot_id] = rec
            atomic_write_json(self._registry, data)
            return _bot_from_rec(rec)

    # ---- source rows --------------------------------------------------
    def add_source(
        self, bot_id: str, type_: str, key: str, label: str, author: str,
        kind: str = "channel",
    ) -> BotSource | None:
        src = BotSource(
            id=uuid.uuid4().hex[:12],
            type=type_,
            key=key,
            label=label,
            author=author,
            kind=kind,
            status=SRC_PENDING,
            added_at=_now(),
        )
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            rec.setdefault("sources", []).append(asdict(src))
            data[bot_id] = rec
            atomic_write_json(self._registry, data)
            return src

    def update_source(self, bot_id: str, source_id: str, **fields) -> BotSource | None:
        allowed = {"status", "item_count", "chunk_count", "label", "author",
                   "channel_title", "channel_avatar", "sync_freq", "last_sync_at",
                   "first_indexed_at", "sync_langs"}
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            for s in rec.get("sources", []):
                if s.get("id") == source_id:
                    for k, v in fields.items():
                        if k in allowed and v is not None:
                            s[k] = v
                    data[bot_id] = rec
                    atomic_write_json(self._registry, data)
                    return BotSource(**s)
            return None

    def remove_source(self, bot_id: str, source_id: str) -> BotSource | None:
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(bot_id)
            if rec is None:
                return None
            kept, removed = [], None
            for s in rec.get("sources", []):
                if s.get("id") == source_id:
                    removed = BotSource(**s)
                else:
                    kept.append(s)
            if removed is None:
                return None
            rec["sources"] = kept
            data[bot_id] = rec
            atomic_write_json(self._registry, data)
            return removed
