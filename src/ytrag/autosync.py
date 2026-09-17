"""Per-channel auto-sync (Phase K): incremental re-ingest of new channel videos.

A YouTube source can be set to sync ``off``/``daily``/``weekly``/``monthly``. One
asyncio background task ticks hourly and, for every source whose interval has
elapsed since its last sync, triggers an INCREMENTAL sync: list the channel's
current videos, diff them against the transcripts already ingested for that
source, and — only if something is new — enqueue the SAME channel ingest job on
the existing single FIFO ingest worker.

The FIFO worker already skips any transcript it has on disk (see
``ingest.download_transcripts``), so re-running a channel downloads ONLY the new
videos; nothing existing is re-fetched. We never spawn a second download worker —
every real ingest funnels through the one FIFO queue in ``ingest_queue.py``.

Frequency resolution is layered: a source's own ``sync_freq`` wins; a blank value
inherits the owner's user-level default (``UserSettings.sync_freq``); ``off`` (or
an unknown value) disables auto-sync for that source. Heavy modules are imported
lazily inside functions so ``user_settings`` can import the freq constants from
here without an import cycle.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

# Frequencies a source (or the user-level default) may take. "off" disables it.
SYNC_OFF = "off"
SYNC_DAILY = "daily"
SYNC_WEEKLY = "weekly"
SYNC_MONTHLY = "monthly"
SYNC_FREQS = (SYNC_OFF, SYNC_DAILY, SYNC_WEEKLY, SYNC_MONTHLY)
DEFAULT_SYNC_FREQ = SYNC_OFF

# Interval in seconds for each ACTIVE frequency (a month is approximated as 30d).
_INTERVAL_SECONDS = {
    SYNC_DAILY: 24 * 3600,
    SYNC_WEEKLY: 7 * 24 * 3600,
    SYNC_MONTHLY: 30 * 24 * 3600,
}

# How often the scheduler wakes to scan sources for due syncs.
TICK_SECONDS = 3600


def normalize_freq(freq: str | None) -> str:
    """Lower-case + validate a frequency. Unknown/blank -> "" (means inherit)."""
    f = (freq or "").strip().lower()
    return f if f in SYNC_FREQS else ""


def effective_freq(source_freq: str | None, user_default: str | None) -> str:
    """Resolve the frequency actually in effect for a source.

    A source's own frequency wins; a blank/unknown source frequency inherits the
    user-level default; if that too is blank/unknown it falls back to ``off``."""
    f = normalize_freq(source_freq)
    if f:
        return f
    return normalize_freq(user_default) or SYNC_OFF


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    # A naive timestamp is treated as UTC so the comparison below never raises.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def interval_elapsed(freq: str, last_sync_at: str | None, now: datetime) -> bool:
    """True when a source on ``freq`` is due for a sync.

    A never-synced active source is due immediately. ``off`` / unknown freq is
    never due. An unparseable ``last_sync_at`` is treated as never-synced.

    NOTE: the SCHEDULER no longer uses this directly — see :func:`source_is_due`,
    which additionally gates on the initial import having completed (BUG-020).
    This helper is kept for the raw interval maths and its callers/tests."""
    secs = _INTERVAL_SECONDS.get(normalize_freq(freq))
    if secs is None:
        return False
    last = _parse_iso(last_sync_at)
    if last is None:
        return True
    return now - last >= timedelta(seconds=secs)


def source_is_due(freq: str, first_indexed_at: str | None,
                  last_sync_at: str | None, now: datetime) -> bool:
    """True when a YouTube source is due for an AUTO-sync (BUG-020).

    Unlike :func:`interval_elapsed`, this NEVER returns True for a channel whose
    initial import has not completed successfully: a channel is gated off until it
    has EITHER a ``first_indexed_at`` (stamped when the first ingest job finishes)
    OR a ``last_sync_at`` (only ever stamped by a real sync, which itself implies a
    completed import). A just-added, in-flight, or cancelled channel has neither and
    is never auto-synced. The non-empty ``last_sync_at`` fallback also keeps
    channels that completed and synced BEFORE ``first_indexed_at`` existed (no
    backfill) auto-syncing after deploy. When due, the interval is measured from the
    last sync (or, if it has never synced, the first successful import) — so an empty
    ``last_sync_at`` no longer means "due now"."""
    secs = _INTERVAL_SECONDS.get(normalize_freq(freq))
    if secs is None:
        return False
    # Gate: the initial import must have completed at least once — proven by a
    # first-import stamp OR a prior successful sync.
    if not first_indexed_at and not last_sync_at:
        return False
    baseline = _parse_iso(last_sync_at) or _parse_iso(first_indexed_at)
    if baseline is None:
        return False
    return now - baseline >= timedelta(seconds=secs)


def diff_new_video_ids(current_videos, ingested_keys) -> list[str]:
    """Return the ids of ``current_videos`` not already present in ``ingested_keys``.

    ``current_videos`` is a list of ``{"id": ...}`` dicts (as ``list_video_ids``
    returns) or bare id strings; ``ingested_keys`` is the set of ingested
    ``<video_id>`` stems on disk. Both sides are normalized with the same
    filename-safe transform used when transcripts are written, so ids compare on
    equal footing. Order-preserving and de-duplicated; the RAW id is returned."""
    from .transcript_ingest import _safe_id

    have = set(ingested_keys or ())
    out: list[str] = []
    seen: set[str] = set()
    for v in current_videos or ():
        vid = v.get("id") if isinstance(v, dict) else v
        if not vid:
            continue
        key = _safe_id(str(vid))
        if key in have or key in seen:
            continue
        seen.add(key)
        out.append(vid)
    return out


def ingested_keys(cfg, bot, source) -> set[str]:
    """The set of ``<video_id>`` stems already ingested for a source (on disk)."""
    from . import bot_service

    tdir = bot_service.source_transcripts_dir(cfg, bot.corpus_id, source.id)
    if not tdir.exists():
        return set()
    return {p.stem for p in tdir.glob("*.json")}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp_last_sync(cfg, bot_id: str, source_id: str) -> None:
    from . import bot_service

    bot_service.bot_store(cfg).update_source(bot_id, source_id, last_sync_at=_now_iso())


def _has_active_job(cfg, source_id: str) -> bool:
    """True when a queued/running ingest job already covers this source."""
    from .ingest_jobs import JobStore

    return any(j.source_id == source_id for j in JobStore(cfg.data_dir).list_active())


def incremental_sync(cfg, bot, source, *, user_id: str,
                     langs=None, limit: int | None = None) -> int:
    """List the channel, diff against ingested transcripts, and enqueue only new.

    Lists the channel's current videos, diffs them against what's already ingested
    for this source, and — when new videos exist and no ingest is already in
    flight for the source — enqueues the channel ingest on the single FIFO worker
    (which downloads only the missing videos). Always stamps ``last_sync_at`` so a
    channel with no new content isn't re-scanned every tick. Enforces the OWNER's
    plan the same way the manual import paths do: the enqueue is capped to the
    plan's per-channel video limit (:func:`quotas.cap_video_limit`) and skipped
    when the owner is already at their active-ingest cap
    (:func:`quotas.check_can_enqueue_ingest`). Returns the count of new videos
    queued (0 for a non-YouTube source, nothing new, an ingest already in progress,
    or when the owner's active-job cap is reached)."""
    from . import ingest, quotas
    from .accounts import User, UserStore
    from .bots import TYPE_YOUTUBE
    from .ingest_queue import manager as ingest_manager

    if source.type != TYPE_YOUTUBE:
        return 0
    current = ingest.list_video_ids(
        source.key,
        cfg.transcript_proxy or None,
        include_shorts=getattr(bot, "include_shorts", False),
    )
    new_ids = diff_new_video_ids(current, ingested_keys(cfg, bot, source))
    _stamp_last_sync(cfg, bot.id, source.id)
    if not new_ids or _has_active_job(cfg, source.id):
        return 0
    # Auto-sync must respect the same plan caps as the manual import paths
    # (app.py import + retry). Resolve the bot OWNER and clamp the per-channel
    # video count to their plan; a missing record (never happens in prod — a bot
    # always has an owner) falls back to the default bounded plan so we NEVER
    # enqueue an uncapped download. Honor the owner's active-job cap too, so a
    # plan-capped user can't queue past their concurrent-ingest limit via sync.
    owner = UserStore(cfg.data_dir).get(user_id) or User(
        id=user_id, google_sub="", email="", name="", picture="", created_at="",
    )
    if quotas.check_can_enqueue_ingest(cfg, owner):
        return 0
    # Reuse the langs chosen at import time (persisted on the source, L5) so a
    # channel imported with "all" re-syncs every language instead of en-only. An
    # explicit ``langs`` arg still wins; a source with no stored selection falls
    # back to None → the prior en-only behavior (never the 200-language storm).
    if langs is None:
        langs = getattr(source, "sync_langs", "") or None
    ingest_manager.enqueue(
        cfg, bot, source,
        user_id=user_id,
        langs=langs, limit=quotas.cap_video_limit(owner, limit),
        base_delay=2.0, max_delay=900.0,
        max_retries=None, cookies_browser=None,
        origin="auto",
    )
    return len(new_ids)


def due_syncs(cfg, now: datetime | None = None):
    """Yield ``(bot, source, freq)`` for every YouTube source currently due.

    Frequency is resolved per source against the OWNER's user-level default. Only
    sources on an active, interval-elapsed frequency are yielded."""
    from . import bot_service
    from .bots import TYPE_YOUTUBE
    from .user_settings import UserSettingsStore

    now = now or datetime.now(timezone.utc)
    bstore = bot_service.bot_store(cfg)
    ustore = UserSettingsStore(cfg.data_dir)
    for bot in bstore.all():
        user_default = ustore.get(bot.owner_id).sync_freq
        for source in bot.sources:
            if source.type != TYPE_YOUTUBE:
                continue
            freq = effective_freq(getattr(source, "sync_freq", ""), user_default)
            if freq == SYNC_OFF:
                continue
            if source_is_due(freq, getattr(source, "first_indexed_at", ""),
                              getattr(source, "last_sync_at", ""), now):
                yield bot, source, freq


class AutoSyncScheduler:
    """A single asyncio task that periodically triggers due per-source syncs.

    Started on app startup and cancelled on shutdown. Each tick runs the blocking
    scan/enqueue work in a thread so the event loop is never blocked, and a failed
    tick can never kill the loop. All real download work still goes through the one
    FIFO ingest worker — the scheduler only decides WHAT to enqueue and WHEN."""

    def __init__(self, cfg_loader, tick_seconds: int = TICK_SECONDS) -> None:
        self._cfg_loader = cfg_loader
        self._tick_seconds = tick_seconds
        self._task: asyncio.Task | None = None

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="autosync-scheduler")
        return self._task

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._tick_seconds)
                await asyncio.to_thread(self.tick)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad tick must never kill the loop
                pass

    def tick(self, now: datetime | None = None) -> int:
        """Run one scan pass (blocking). Returns the number of sources triggered."""
        cfg = self._cfg_loader()
        triggered = 0
        for bot, source, _freq in due_syncs(cfg, now):
            try:
                incremental_sync(cfg, bot, source, user_id=bot.owner_id)
                triggered += 1
            except Exception:  # noqa: BLE001 - one bad source must not stop the scan
                pass
        return triggered
