"""Single-worker FIFO queue for server-side YouTube channel ingests.

Channel ingests are processed strictly one at a time by ONE daemon thread. This
protects the shared proxy and YouTube (no parallel scraping) and honours Chroma's
single-writer constraint (only this thread rebuilds a bot's collection). Jobs are
persisted via the ``JobStore`` so an ingest survives a process restart: on
startup ``resume_pending`` re-enqueues everything still queued/running, oldest
first, and a worker picks up where it left off.

The mechanics of a single job mirror ``web.jobs.JobManager._run_bot_youtube``:
``bot_service.ingest_channel(...)`` then ``bot_service.rebuild_bot(...)``. The
only difference is serialization and durable status in the JobStore.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable

from . import bot_service, ingest_status, notify, youtube_count
from .bots import Bot, BotSource
from .config import Config, load_config
from .ingest_jobs import STATUS_CANCELLED, IngestJob, JobStore
from .logging_setup import proxy_diagnostic
from .throttle import ThrottleConfig, humanize_seconds

logger = logging.getLogger(__name__)


class IngestCancelled(Exception):
    """Raised inside the worker when a user cancels an in-flight ingest."""


def _langs_to_str(langs: Iterable[str] | str | None) -> str:
    """Normalize a langs list/str into the comma-joined form the JobStore stores."""
    if not langs:
        return ""
    if isinstance(langs, str):
        return langs
    return ",".join(str(x).strip() for x in langs if str(x).strip())


def _langs_from_str(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _find_source(bot: Bot | None, source_id: str) -> BotSource | None:
    if bot is None:
        return None
    for s in bot.sources:
        if s.id == source_id:
            return s
    return None


class IngestQueue:
    """In-process FIFO queue drained by a single lazily-started daemon worker."""

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    # ---- worker lifecycle --------------------------------------------
    def _ensure_worker(self) -> None:
        """Start the single worker thread once (restart it if it ever died)."""
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name="ingest-queue-worker", daemon=True
                )
                self._worker.start()

    # ---- public API ---------------------------------------------------
    def enqueue(
        self,
        cfg: Config,
        bot: Bot,
        source: BotSource,
        *,
        user_id: str,
        langs: Iterable[str] | str | None,
        limit: int | None,
        base_delay: float,
        max_delay: float,
        max_retries: int | None,
        cookies_browser: str | None,
        origin: str = "manual",
    ) -> str:
        """Persist a queued job, push it onto the FIFO, and return its id (non-blocking)."""
        store = JobStore(cfg.data_dir)
        job = store.create(
            user_id=user_id,
            channel_url=source.key or source.label,
            bot_id=bot.id,
            source_id=source.id,
            langs=_langs_to_str(langs),
            limit=limit,
            base_delay=base_delay,
            max_delay=max_delay,
            max_retries=max_retries,
            cookies_browser=cookies_browser or "",
            origin=origin,
        )
        logger.info(
            "job ENQUEUED job=%s origin=%s bot=%s source=%s kind=%s key=%s",
            job.job_id, origin, bot.id, source.id,
            getattr(source, "kind", "") or getattr(source, "type", ""),
            source.key or source.label,
        )
        self._q.put(job.job_id)
        self._ensure_worker()
        return job.job_id

    def resume_pending(self, cfg: Config) -> int:
        """Re-enqueue jobs left queued/running by a prior process, OLDEST FIRST.

        A job whose bot or source no longer exists is marked error rather than
        re-run. Returns the number of jobs re-enqueued."""
        store = JobStore(cfg.data_dir)
        bstore = bot_service.bot_store(cfg)
        # list_active() is newest-first; drain oldest-first to preserve FIFO order.
        active = sorted(store.list_active(), key=lambda j: j.created_at)
        resumed = 0
        for job in active:
            bot = bstore.get(job.bot_id) if job.bot_id else None
            source = _find_source(bot, job.source_id)
            if bot is None or source is None:
                logger.warning("job %s not resumable: bot or source no longer exists",
                               job.job_id)
                store.mark_error(job.job_id, "bot or source no longer exists")
                continue
            self._q.put(job.job_id)
            resumed += 1
        if resumed:
            logger.info("resuming %d pending ingest job(s) after restart", resumed)
            self._ensure_worker()
        return resumed

    # ---- worker loop --------------------------------------------------
    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._process(job_id)
            except Exception:  # noqa: BLE001 - a bad job must never kill the worker
                logger.exception("ingest worker: unexpected error processing job=%s", job_id)
            finally:
                self._q.task_done()

    def _process(self, job_id: str) -> None:
        cfg = load_config()
        store = JobStore(cfg.data_dir)
        job = store.get(job_id)
        if job is None:
            logger.warning("job %s vanished before it could run", job_id)
            return
        # Cancelled while it sat queued (tombstone) or flagged before we started.
        if job.status == STATUS_CANCELLED or job.cancel_requested:
            logger.info("job CANCELLED job=%s (before start)", job_id)
            store.mark_cancelled(job_id)
            return
        started = time.monotonic()
        logger.info(
            "job STARTED job=%s origin=%s bot=%s source=%s key=%s | transcript proxy: %s",
            job_id, job.origin, job.bot_id, job.source_id, job.channel_url,
            proxy_diagnostic(cfg.transcript_proxy),
        )
        # Guard everything below so a job never hangs in `running` and is always
        # driven to a terminal state, even if bot/source reconstruction throws.
        try:
            # Always reload bot+source fresh — never trust objects captured at enqueue.
            bstore = bot_service.bot_store(cfg)
            bot = bstore.get(job.bot_id) if job.bot_id else None
            source = _find_source(bot, job.source_id)
            if bot is None or source is None:
                store.mark_error(job_id, "bot or source no longer exists")
                return

            def on_progress(ev: dict) -> None:
                # Cheap per-video cancellation check: bail out of the download loop.
                fresh = store.get(job_id)
                if fresh is None or fresh.cancel_requested or fresh.status == STATUS_CANCELLED:
                    raise IngestCancelled
                stage = ev.get("stage")
                # Meter transcript-proxy bandwidth toward the owner's monthly
                # ceiling. Any event may carry a `proxy_mb` estimate (a flat
                # per-video cost on every proxied attempt, plus the measured
                # subtitle bytes on a save); 'skip' (already on disk) carries none.
                pmb = ev.get("proxy_mb")
                if pmb:
                    try:
                        from .usage import UsageStore
                        UsageStore(cfg.data_dir).record_proxy_mb(bot.owner_id, pmb)
                    except Exception:  # noqa: BLE001 - metering must never fail a job
                        pass
                if stage == "listed":
                    total = int(ev.get("total", 0))
                    logger.info("job %s listed %d video(s) to process", job_id, total)
                    # BUG-005: seed videos_done from the videos already downloaded in
                    # prior passes so a resumed job's progress reflects durable
                    # on-disk state and never regresses on reload (the 56→33 desync).
                    try:
                        seed = bot_service.completed_video_count(cfg, bot, source)
                    except Exception:  # noqa: BLE001 - a seed probe must never fail a job
                        seed = 0
                    store.mark_running(job_id, videos_total=total, videos_done=seed)
                    # BUG-001 determinate progress: window_total = what we index this
                    # run (exact bar denominator); channel_total = approx channel size
                    # (YouTube Data API v3 when YTRAG_YOUTUBE_API_KEY is set, else the
                    # window count as a floor — works with NO key). Floored to the
                    # window so "142 / 300, Channel ~300" is never internally smaller.
                    try:
                        approx = youtube_count.channel_total_videos(
                            job.channel_url, fallback=lambda: total
                        )
                    except Exception:  # noqa: BLE001 - count lookup must never fail a job
                        approx = total
                    store.set_window(
                        job_id, window_total=total, channel_total=max(approx, total)
                    )
                elif stage == "channel":
                    # Persist channel name/avatar for the UI. Pass None for empties
                    # so update_source's None-skip guard won't clobber a stored value.
                    # A metadata persist failure must never abort the ingest.
                    try:
                        bstore.update_source(
                            bot.id, source.id,
                            channel_title=(ev.get("title") or None),
                            channel_avatar=(ev.get("avatar") or None),
                        )
                    except Exception:  # noqa: BLE001 - metadata is best-effort
                        pass
                elif stage == "saved":
                    # A NEW transcript written this pass — the only event that bumps
                    # videos_done (added on top of the on-disk seed above).
                    store.bump_done(job_id)
                elif stage == "skip":
                    # Already on disk from a PRIOR pass: it is already reflected in the
                    # seeded videos_done, so DON'T bump — bumping here is exactly the
                    # BUG-005 double-count/regress on resume. Intentionally a no-op.
                    pass
                elif stage in ("nosub", "skip_permanent"):
                    # No captions / permanently unavailable: a SKIP, not a failure —
                    # counted separately so the UI labels it "skipped (no captions)"
                    # rather than red-flagging it as failed (BUG-016 / G3).
                    store.bump_skipped(job_id)
                elif stage == "error":
                    store.bump_failed(job_id)

            def _over_bandwidth() -> bool:
                # Per-video probe: stop starting NEW downloads once the owner has
                # crossed their plan's monthly proxy ceiling (partial import still
                # finalizes + rebuilds, so the bot is never empty and chat works).
                try:
                    from . import quotas
                    from .accounts import UserStore
                    owner = UserStore(cfg.data_dir).get(bot.owner_id)
                    return owner is not None and quotas.over_bandwidth(cfg, owner)
                except Exception:  # noqa: BLE001 - a probe failure must never abort ingest
                    return False

            bot_service.ingest_channel(
                cfg,
                bot,
                source,
                langs=_langs_from_str(job.langs),
                limit=job.limit,
                throttle=ThrottleConfig(base_delay=job.base_delay, max_delay=job.max_delay),
                max_retries=job.max_retries,
                cookies_browser=job.cookies_browser or None,
                progress=on_progress,
                job_id=job_id,
                budget_probe=_over_bandwidth,
            )
            # Pass a cancel probe so a rebuild that races an account/bot delete
            # (which flags cancel_requested / removes the job) aborts cleanly
            # instead of re-creating an orphaned collection.
            def _cancel_requested() -> bool:
                fresh = store.get(job_id)
                return (
                    fresh is None
                    or fresh.cancel_requested
                    or fresh.status == STATUS_CANCELLED
                )

            # INCREMENTAL rebuild (the default): this is the routine sync path
            # (manual import, auto-sync, "index more"), so embed + append ONLY the
            # new videos' chunks — never drop + re-embed the whole corpus (BUG-021).
            # On-disk dedup already downloaded only the new videos above.
            bot_service.rebuild_bot(cfg, bot, should_abort=_cancel_requested)
            # If a cancel landed during the rebuild, finalize as cancelled rather
            # than falsely reporting the (aborted, empty) build as done.
            if _cancel_requested():
                logger.info("job CANCELLED job=%s (during rebuild)", job_id)
                store.mark_cancelled(job_id)
                return
            done = store.mark_done(job_id)
            # BUG-020: stamp the first successful import completion so the
            # auto-sync scheduler only ever fires for a channel whose initial
            # import actually finished (never a just-added / in-flight /
            # cancelled one). Stamp ONCE and never overwrite, so the sync
            # interval baseline stays stable across later re-imports.
            try:
                from . import autosync
                fresh_src = _find_source(bstore.get(bot.id), source.id)
                if fresh_src is not None and not getattr(fresh_src, "first_indexed_at", ""):
                    bstore.update_source(bot.id, source.id,
                                         first_indexed_at=autosync._now_iso())
            except Exception:  # noqa: BLE001 - stamp is best-effort, never fail a DONE job
                pass
            logger.info(
                "job DONE job=%s done=%d failed=%d total=%d elapsed=%s",
                job_id,
                getattr(done, "videos_done", 0),
                getattr(done, "videos_failed", 0),
                getattr(done, "videos_total", 0),
                humanize_seconds(time.monotonic() - started),
            )
            # Best-effort completion email. Wrap the call site too: without this,
            # an exception here would fall through to the outer handler and wrongly
            # flip a DONE job to error (notify is internally defensive already).
            try:
                notify.notify_ingest_complete(cfg, job_id)
            except Exception:  # noqa: BLE001 - email is a side effect, never fail the job
                pass
        except IngestCancelled:
            # Non-destructive: leave the source PENDING, do not rebuild or error.
            logger.info("job CANCELLED job=%s (during download)", job_id)
            store.mark_cancelled(job_id)
            return
        except Exception as exc:  # noqa: BLE001 - persist failure, keep worker alive
            # EXCEPTION (with traceback): this is the line that finally explains a
            # job stuck at 0% — proxy down, hard-block, extractor failure, etc.
            logger.exception("job ERROR job=%s: %s", job_id, exc)
            # BUG-017: never leave an empty bot. If SOME videos were downloaded
            # before the failure, index them so the successfully-fetched ones are
            # usable and chat works — a partial, usable bot beats a blank one.
            self._index_partial(cfg, job)
            # G3: surface a calm, user-facing failure sentence (raw kept for logs).
            store.mark_error(
                job_id, str(exc), friendly=ingest_status.friendly_failure(str(exc))
            )
            # Best-effort failure email; must never raise out of the worker.
            try:
                notify.notify_ingest_complete(cfg, job_id)
            except Exception:  # noqa: BLE001 - email is a side effect, never fail the job
                pass

    def _index_partial(self, cfg: Config, job: IngestJob) -> None:
        """Index whatever transcripts already downloaded, so a FAILED import still
        leaves a usable (never empty) bot (BUG-017).

        Only rebuilds when at least one transcript is on disk — a job that failed
        before downloading anything (e.g. a channel-level block) has nothing to
        surface and its source is left un-built, unchanged. Fully best-effort: any
        error here is swallowed so it can never turn a mark_error into a crash."""
        try:
            bstore = bot_service.bot_store(cfg)
            bot = bstore.get(job.bot_id) if job.bot_id else None
            source = _find_source(bot, job.source_id)
            if bot is None or source is None:
                return
            if bot_service.completed_video_count(cfg, bot, source) <= 0:
                return  # nothing downloaded — nothing partial to surface
            bot_service.rebuild_bot(cfg, bot)
            logger.info("job %s: indexed partial results before erroring", job.job_id)
        except Exception:  # noqa: BLE001 - partial finalize must never fail the job
            logger.exception("job %s: partial finalize failed (non-fatal)", job.job_id)


# Module-level singleton — one queue, one worker thread, per process.
manager = IngestQueue()
