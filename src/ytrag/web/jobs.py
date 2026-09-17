"""In-memory background job registry for the local web UI.

Jobs ingest a YouTube channel into a specific source folder or just rebuild the
shared index. After any channel ingest the whole index is rebuilt so a source's
author metadata and any deletions are always reflected.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import bot_service
from ..bots import Bot, BotSource
from ..config import Config, load_config
from ..indexer import rebuild_index
from ..ingest import download_transcripts
from ..ingest_jobs import JobStore
from ..safestore import atomic_write_json, file_lock, read_json
from ..sources import Source, SourceStore
from ..throttle import ThrottleConfig

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    label: str
    # Owning tenant (user id). Set at creation so ``GET /api/jobs/{id}`` can scope
    # a rebuild's label + logs to its owner; ``None`` means no owner is known (a
    # pre-owner persisted record or an internal/test job) and is treated as
    # unreadable by any authenticated caller — fail closed, not open.
    owner_id: str | None = None
    status: str = "running"  # running | done | error
    stage: str = "starting"
    total: int = 0
    processed: int = 0
    saved: int = 0
    indexed: int = 0
    log: list[str] = field(default_factory=list)
    error: str | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "stage": self.stage,
            "total": self.total,
            "processed": self.processed,
            "saved": self.saved,
            "indexed": self.indexed,
            "log": self.log[-200:],
            "error": self.error,
        }


class RebuildJobStore:
    """Durable set of IN-FLIGHT index-rebuild jobs, keyed by job id.

    :class:`JobManager` tracks a rebuild's live progress in memory, but that state
    is lost on restart — a rebuild left running when the process died would never
    finish, so the SPA (polling ``GET /api/jobs/{id}``) would wait forever and the
    index would stay stale. This JSON file is a minimal durable mirror: a record
    is written when a rebuild STARTS and removed the instant it reaches a terminal
    state (done/error). The file therefore only ever holds rebuilds that were
    genuinely in flight, and on startup :meth:`JobManager.resume_pending_rebuilds`
    re-drives exactly those — mirroring ``ingest_queue.IngestQueue.resume_pending``.
    """

    def __init__(self, root: Path):
        self._registry = Path(root) / "rebuild_jobs.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def add(self, job_id: str, bot_id: str, label: str, owner_id: str | None = None) -> None:
        with file_lock(self._registry):
            data = self._load()
            data[job_id] = {
                "job_id": job_id,
                "bot_id": bot_id,
                "label": label,
                "owner_id": owner_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(self._registry, data)

    def remove(self, job_id: str) -> None:
        with file_lock(self._registry):
            data = self._load()
            if data.pop(job_id, None) is not None:
                atomic_write_json(self._registry, data)

    def list_pending(self) -> list[dict]:
        """Persisted in-flight rebuilds, OLDEST FIRST (restart recovery order)."""
        recs = list(self._load().values())
        recs.sort(key=lambda r: r.get("created_at", ""))
        return recs


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # Per-bot rebuild coalescing. The whole app is single process (the
        # ingest FIFO worker, the Telegram poll thread and request threads all
        # live here), so concurrent rebuilds of the SAME bot would drop+rebuild
        # the same Chroma collection at once — a race. ``_active_rebuilds`` is a
        # cheap coalesce guard: a redundant 2nd rebuild request for a bot already
        # rebuilding is skipped (a rebuild is an idempotent drop +
        # rebuild-from-sources, so a concurrent duplicate would only duplicate
        # work). The HARD cross-driver backstop lives one level down in
        # ``bot_service.rebuild_bot``, which takes a process-wide per-bot lock
        # (``bot_service.rebuild_lock``) around the actual drop+rebuild — so even
        # the ingest worker's rebuild path, which never touches this JobManager,
        # can never overlap a user-triggered rebuild of the same collection.
        self._active_rebuilds: set[str] = set()

    def _begin_rebuild(self, bot_id: str) -> bool:
        """Reserve the single in-flight rebuild slot for ``bot_id``.

        Returns ``False`` if a rebuild for this bot is already in flight — the
        caller should then coalesce (skip) instead of spawning a concurrent one.
        """
        with self._lock:
            if bot_id in self._active_rebuilds:
                return False
            self._active_rebuilds.add(bot_id)
            return True

    def _end_rebuild(self, bot_id: str) -> None:
        with self._lock:
            self._active_rebuilds.discard(bot_id)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _new(self, label: str, owner_id: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], label=label, owner_id=owner_id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    # ---- channel ingest ----------------------------------------------
    def start_youtube(self, source: Source, langs, limit, base_delay, max_delay,
                      max_retries=None, cookies_browser=None) -> Job:
        job = self._new(source.label)
        t = threading.Thread(
            target=self._run_youtube,
            args=(job, source, langs, limit, base_delay, max_delay,
                  max_retries, cookies_browser),
            daemon=True,
        )
        t.start()
        return job

    def _run_youtube(self, job, source, langs, limit, base_delay, max_delay,
                     max_retries, cookies_browser):
        try:
            cfg = load_config()
            sstore = SourceStore(cfg.data_dir)

            def on_progress(ev: dict):
                job.stage = ev.get("stage", job.stage)
                if "total" in ev:
                    job.total = ev["total"]
                if "index" in ev:
                    job.processed = ev["index"]
                if ev.get("stage") == "saved":
                    job.saved += 1
                msg = ev.get("message")
                if msg:
                    job.log.append(msg)

            download_transcripts(
                source.key,
                sstore.transcripts_dir(source.id),
                langs=langs,
                limit=limit,
                throttle=ThrottleConfig(base_delay=base_delay, max_delay=max_delay),
                max_retries=max_retries,
                cookies_browser=cookies_browser,
                progress=on_progress,
                proxy=cfg.transcript_proxy or None,
            )

            self._rebuild(cfg, sstore, job)
            job.stage = "done"
            job.status = "done"
            job.log.append("All done.")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.log.append(f"ERROR: {exc}")

    # ---- rebuild only -------------------------------------------------
    def start_rebuild(self) -> Job:
        job = self._new("rebuild")
        t = threading.Thread(target=self._run_rebuild, args=(job,), daemon=True)
        t.start()
        return job

    def _run_rebuild(self, job):
        try:
            cfg = load_config()
            sstore = SourceStore(cfg.data_dir)
            self._rebuild(cfg, sstore, job)
            job.stage = "done"
            job.status = "done"
            job.log.append("All done.")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.log.append(f"ERROR: {exc}")

    # ---- per-bot ingest ----------------------------------------------
    def start_bot_youtube(self, bot: Bot, source: BotSource, *, langs, limit,
                          base_delay, max_delay, max_retries=None,
                          cookies_browser=None) -> Job:
        job = self._new(f"{bot.name}: {source.label}")
        t = threading.Thread(
            target=self._run_bot_youtube,
            args=(job, bot, source, langs, limit, base_delay, max_delay,
                  max_retries, cookies_browser),
            daemon=True,
        )
        t.start()
        return job

    def _run_bot_youtube(self, job, bot, source, langs, limit, base_delay,
                         max_delay, max_retries, cookies_browser):
        try:
            cfg = load_config()

            def on_progress(ev: dict):
                job.stage = ev.get("stage", job.stage)
                if "total" in ev:
                    job.total = ev["total"]
                if "index" in ev:
                    job.processed = ev["index"]
                if ev.get("stage") == "saved":
                    job.saved += 1
                msg = ev.get("message")
                if msg:
                    job.log.append(msg)

            bot_service.ingest_channel(
                cfg, bot, source,
                langs=langs, limit=limit,
                throttle=ThrottleConfig(base_delay=base_delay, max_delay=max_delay),
                max_retries=max_retries, cookies_browser=cookies_browser,
                progress=on_progress,
            )
            self._rebuild_bot(cfg, bot, job)
            job.stage = "done"
            job.status = "done"
            job.log.append("All done.")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.log.append(f"ERROR: {exc}")

    def start_bot_rebuild(self, bot: Bot, owner_id: str | None = None) -> Job:
        job = self._new(f"{bot.name}: rebuild", owner_id=owner_id)
        if not self._begin_rebuild(bot.id):
            # Coalesce: an in-flight rebuild for THIS bot already covers this
            # request. A rebuild is an idempotent drop + rebuild-from-sources, so
            # a redundant concurrent one would only duplicate work and race on the
            # same Chroma collection. Resolve this job immediately (no durable
            # record, no thread) rather than spawning a second rebuild.
            job.stage = job.status = "done"
            job.log.append(
                "A rebuild for this bot is already in progress — skipped (coalesced)."
            )
            return job
        # Persist BEFORE spawning so a crash between here and completion is
        # recoverable — the record is cleared the moment the job finishes.
        self._persist_rebuild(job, bot.id)
        self._spawn_bot_rebuild(job, bot)
        return job

    def _spawn_bot_rebuild(self, job: Job, bot: Bot) -> None:
        t = threading.Thread(target=self._run_bot_rebuild, args=(job, bot), daemon=True)
        t.start()

    def _run_bot_rebuild(self, job, bot):
        try:
            # The real drop+rebuild for THIS bot is serialized inside
            # ``bot_service.rebuild_bot`` via the shared per-bot ``rebuild_lock``,
            # so two rebuilds of the same collection can never overlap even across
            # drivers (this JobManager and the ingest worker). Runs in a daemon
            # thread, so any wait there never blocks the request thread / event
            # loop.
            cfg = load_config()
            self._rebuild_bot(cfg, bot, job)
            job.stage = "done"
            job.status = "done"
            job.log.append("All done.")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            job.log.append(f"ERROR: {exc}")
        finally:
            # Terminal (done OR error) -> drop the durable record so restart never
            # re-runs a finished rebuild. The in-memory Job stays for SPA polling.
            self._forget_rebuild(job.id)
            # Release the coalesce slot LAST so a rebuild queued/clicked while this
            # one ran is free to start now that the collection is settled.
            self._end_rebuild(bot.id)

    # ---- durable rebuild tracking (restart recovery) ------------------
    @staticmethod
    def _persist_rebuild(job: Job, bot_id: str) -> None:
        try:
            RebuildJobStore(load_config().data_dir).add(
                job.id, bot_id, job.label, job.owner_id)
        except Exception:  # noqa: BLE001 - persistence is best-effort, never block a rebuild
            logger.exception("could not persist rebuild job %s for restart resume", job.id)

    @staticmethod
    def _forget_rebuild(job_id: str) -> None:
        try:
            RebuildJobStore(load_config().data_dir).remove(job_id)
        except Exception:  # noqa: BLE001 - a failed cleanup only risks a redundant resume
            logger.exception("could not clear persisted rebuild job %s", job_id)

    def resume_pending_rebuilds(self, cfg: Config) -> int:
        """Re-drive index-rebuild jobs left in flight by a prior process.

        Mirrors ``ingest_queue.resume_pending``: reconstruct each persisted rebuild
        and run it to completion under its ORIGINAL job id, so a SPA still polling
        ``GET /api/jobs/{id}`` sees it finish. Two non-resume cases are finalized
        cleanly (record cleared, in-memory Job marked terminal so the UI stops
        polling a dead job):

        * bot no longer exists -> mark ``error`` (a rebuild has nothing to target);
        * bot has a queued/running INGEST -> that ingest ends in its own bot
          rebuild, so running a standalone rebuild for the same bot now would
          double-process the collection. Defer to the ingest (mark ``done``).

        Returns the number of rebuilds actually re-driven."""
        store = RebuildJobStore(cfg.data_dir)
        bstore = bot_service.bot_store(cfg)
        # Bots already covered by an in-flight ingest whose terminal step rebuilds
        # the same collection — never rebuild those concurrently here.
        ingest_bot_ids = {
            j.bot_id for j in JobStore(cfg.data_dir).list_active() if j.bot_id
        }
        resumed = deferred = 0
        for rec in store.list_pending():
            job_id = rec.get("job_id") or ""
            bot_id = rec.get("bot_id") or ""
            label = rec.get("label") or "rebuild"
            owner_id = rec.get("owner_id")
            if not job_id:
                continue
            # Re-materialize the in-memory Job under its ORIGINAL id so a poll resolves.
            job = Job(id=job_id, label=label, owner_id=owner_id)
            with self._lock:
                self._jobs[job_id] = job
            bot = bstore.get(bot_id) if bot_id else None
            if bot is None:
                job.status = job.stage = "error"
                job.error = "bot no longer exists"
                job.log.append("ERROR: bot no longer exists — cannot resume rebuild.")
                store.remove(job_id)
                continue
            if bot_id in ingest_bot_ids:
                job.status = job.stage = "done"
                job.log.append("Deferred: a pending import for this bot will rebuild the index.")
                store.remove(job_id)
                deferred += 1
                continue
            if not self._begin_rebuild(bot_id):
                # Another pre-crash record for the SAME bot is already being
                # re-driven; coalesce this duplicate rather than run a concurrent
                # rebuild of the same collection.
                job.status = job.stage = "done"
                job.log.append("Coalesced: another rebuild for this bot is already in progress.")
                store.remove(job_id)
                continue
            self._spawn_bot_rebuild(job, bot)
            resumed += 1
        if resumed:
            logger.info("resuming %d pending rebuild job(s) after restart", resumed)
        if deferred:
            logger.info(
                "deferred %d pending rebuild job(s) to an in-flight ingest for the same bot",
                deferred,
            )
        return resumed

    @staticmethod
    def _rebuild_bot(cfg, bot: Bot, job: Job):
        # A JobManager rebuild is a whole-collection RECONCILE — the explicit
        # "Rebuild" button and the post-source-removal rebuild both need every
        # source re-materialized and any removed source's vectors gone, which only
        # a FULL drop-and-rebuild guarantees. The routine incremental append path
        # (BUG-021) lives on the FIFO ingest worker instead (auto-sync / re-import).
        job.stage = "building"
        job.log.append("Rebuilding this bot's index from its sources...")

        def on_progress(ev: dict):
            msg = ev.get("message")
            if msg:
                job.log.append(msg)

        job.indexed = bot_service.rebuild_bot(cfg, bot, progress=on_progress, full=True)
        job.log.append(f"Index ready: {job.indexed} chunks.")

    @staticmethod
    def _rebuild(cfg, sstore, job: Job):
        job.stage = "building"
        job.log.append("Rebuilding vector index from all sources...")

        def on_progress(ev: dict):
            msg = ev.get("message")
            if msg:
                job.log.append(msg)

        job.indexed = rebuild_index(cfg, sstore, progress=on_progress)
        job.log.append(f"Index ready: {job.indexed} chunks.")


manager = JobManager()
