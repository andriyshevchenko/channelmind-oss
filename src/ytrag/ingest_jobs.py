"""Server-side transcript ingestion jobs — queued work items, keyed by job id.

One record per ingestion request (a user asking to ingest a channel's
transcripts). This is the data layer only: it tracks queue status and
per-video progress counters so a worker can pick up ``queued`` jobs and resume
``running`` ones after a restart. The worker loop, API, and UI live elsewhere.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .safestore import atomic_write_json, file_lock, read_json

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUSES = frozenset(
    {STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED}
)


@dataclass
class IngestJob:
    job_id: str
    user_id: str
    channel_url: str
    status: str
    created_at: str
    updated_at: str
    videos_total: int = 0
    videos_done: int = 0
    videos_failed: int = 0
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    # What a worker needs to reconstruct and run the job after a restart.
    bot_id: str = ""
    source_id: str = ""
    langs: str = ""  # comma-joined subtitle languages ("" = original source language; "all" = every language)
    limit: int | None = None
    base_delay: float = 2.0
    max_delay: float = 900.0
    max_retries: int | None = None
    cookies_browser: str = ""
    # How the job was started: ``manual`` (a user clicked Import / Retry) or
    # ``auto`` (the auto-sync scheduler, or the "Sync now" incremental sync).
    # Surfaced verbatim in GET responses so the UI can badge auto-sync rows.
    # Dataclass default keeps pre-origin job records loadable as ``manual``.
    origin: str = "manual"
    # Set by ``request_cancel`` on a running job; the worker observes it and
    # finalizes to ``cancelled``. LAST original field so old records still load.
    cancel_requested: bool = False
    # BUG-016 / G3: videos that had NO captions are SKIPPED, not failed — counted
    # separately so the UI can label them "skipped (no captions)" rather than
    # lumping them into the red "failed" tally. New field with a default keeps
    # pre-1.3 records loadable.
    videos_skipped: int = 0
    # BUG-001 determinate progress (backend fields; the UI renders them in 1.5):
    #  * ``window_total`` — how many videos THIS run will index (= the enumerated,
    #    cap-clamped total). The progress bar denominator ("142 / 300").
    #  * ``channel_total`` — the channel's APPROX total public video count (from the
    #    YouTube Data API v3 when a key is set, else the window count as a floor).
    #    Drives the context line ("Channel ~2,500, indexing the 300 newest").
    # Both default 0 so old records load; both are set once at the ``listed`` stage.
    window_total: int = 0
    channel_total: int = 0
    # G3: a calm, user-facing sentence for a FAILED job (e.g. "YouTube temporarily
    # blocked the import."). ``error`` keeps the raw detail for logs/debugging;
    # this is what the UI shows. Empty when the job didn't fail. New field with a
    # default keeps old records loadable.
    error_friendly: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """JSON-backed ingestion queue, keyed by job id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "ingest_jobs.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def create(
        self,
        user_id: str,
        channel_url: str,
        *,
        bot_id: str = "",
        source_id: str = "",
        langs: str = "",
        limit: int | None = None,
        base_delay: float = 2.0,
        max_delay: float = 900.0,
        max_retries: int | None = None,
        cookies_browser: str = "",
        origin: str = "manual",
    ) -> IngestJob:
        """Enqueue a new job in ``queued`` state and return it.

        The optional keyword args capture everything a worker needs to run and,
        after a restart, resume the job without the original caller's context.
        ``origin`` labels manual vs auto-sync starts for the UI.
        """
        with file_lock(self._registry):
            data = self._load()
            now = _now()
            job = IngestJob(
                job_id=uuid4().hex[:12],
                user_id=user_id,
                channel_url=channel_url,
                status=STATUS_QUEUED,
                created_at=now,
                updated_at=now,
                bot_id=bot_id,
                source_id=source_id,
                langs=langs,
                limit=limit,
                base_delay=base_delay,
                max_delay=max_delay,
                max_retries=max_retries,
                cookies_browser=cookies_browser,
                origin=origin,
            )
            data[job.job_id] = asdict(job)
            atomic_write_json(self._registry, data)
            return job

    def get(self, job_id: str) -> IngestJob | None:
        rec = self._load().get(job_id)
        return IngestJob(**rec) if rec else None

    def list_for_user(self, user_id: str) -> list[IngestJob]:
        """All jobs owned by ``user_id``, newest first."""
        jobs = [
            IngestJob(**rec)
            for rec in self._load().values()
            if rec.get("user_id") == user_id
        ]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def delete_for_user(self, user_id: str) -> int:
        """Remove ALL job records owned by ``user_id`` (account erasure).

        Returns how many were removed. Active jobs should be ``request_cancel``-ed
        first so the running worker bails; once the record is gone the worker's
        next ``get`` returns ``None`` and it stops touching the (also-deleted) bot.
        """
        with file_lock(self._registry):
            data = self._load()
            doomed = [jid for jid, rec in data.items() if rec.get("user_id") == user_id]
            for jid in doomed:
                data.pop(jid, None)
            if doomed:
                atomic_write_json(self._registry, data)
            return len(doomed)

    def delete(self, job_id: str) -> bool:
        """Atomically remove ONE job record. Returns True if it existed.

        Used to dismiss a finished job row, and by Retry to replace an errored
        record with a fresh queued one rather than leaving a duplicate."""
        with file_lock(self._registry):
            data = self._load()
            if job_id not in data:
                return False
            data.pop(job_id, None)
            atomic_write_json(self._registry, data)
            return True

    def list_active(self) -> list[IngestJob]:
        """Queued or running jobs, newest first (for restart recovery)."""
        jobs = [
            IngestJob(**rec)
            for rec in self._load().values()
            if rec.get("status") in (STATUS_QUEUED, STATUS_RUNNING)
        ]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def _update(self, job_id: str, **changes) -> IngestJob | None:
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(job_id)
            if rec is None:
                return None
            status = changes.get("status")
            if status is not None and status not in STATUSES:
                raise ValueError(f"Unknown status: {status}")
            rec.update(changes)
            rec["updated_at"] = _now()
            data[job_id] = rec
            atomic_write_json(self._registry, data)
            return IngestJob(**rec)

    def mark_running(
        self, job_id: str, videos_total: int, *, videos_done: int = 0
    ) -> IngestJob | None:
        # Reset the per-pass failed/skipped counters and SEED videos_done to the
        # number of videos already completed on disk from prior passes (0 on a
        # fresh run). This is the BUG-005 fix: a resumed pass re-lists the channel
        # and re-emits "skip" for already-downloaded videos, but those skips no
        # longer bump the counter (they are already reflected in this seed), so the
        # done count reflects durable, absolute on-disk state and never visibly
        # regresses across a reload/restart (the 56→33 desync). failed/skipped
        # zero because they are re-derived fresh each pass (a caption-less video is
        # re-attempted every pass), which keeps them from inflating on resume.
        return self._update(
            job_id,
            status=STATUS_RUNNING,
            videos_total=videos_total,
            videos_done=max(0, int(videos_done)),
            videos_failed=0,
            videos_skipped=0,
            started_at=_now(),
        )

    def set_window(
        self, job_id: str, *, window_total: int, channel_total: int
    ) -> IngestJob | None:
        """Record the BUG-001 determinate-progress totals for a running job.

        ``window_total`` is how many videos this run will index (the exact bar
        denominator); ``channel_total`` is the channel's approximate total public
        video count for the context line. Both are non-negative."""
        return self._update(
            job_id,
            window_total=max(0, int(window_total)),
            channel_total=max(0, int(channel_total)),
        )

    def _bump(self, job_id: str, field: str) -> IngestJob | None:
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(job_id)
            if rec is None:
                return None
            rec[field] = int(rec.get(field, 0)) + 1
            rec["updated_at"] = _now()
            data[job_id] = rec
            atomic_write_json(self._registry, data)
            return IngestJob(**rec)

    def bump_done(self, job_id: str) -> IngestJob | None:
        return self._bump(job_id, "videos_done")

    def bump_failed(self, job_id: str) -> IngestJob | None:
        return self._bump(job_id, "videos_failed")

    def bump_skipped(self, job_id: str) -> IngestJob | None:
        # A no-captions video: counted separately from failures so the UI can
        # label it "skipped (no captions)" rather than red-flagging it (BUG-016).
        return self._bump(job_id, "videos_skipped")

    def mark_done(self, job_id: str) -> IngestJob | None:
        return self._update(job_id, status=STATUS_DONE, finished_at=_now())

    def mark_error(self, job_id: str, error: str, *, friendly: str = "") -> IngestJob | None:
        # ``error`` keeps the raw detail (logs/debug); ``friendly`` is the calm G3
        # sentence the UI surfaces. Callers that don't pass one leave it blank.
        return self._update(
            job_id, status=STATUS_ERROR, finished_at=_now(),
            error=error, error_friendly=friendly,
        )

    def mark_cancelled(self, job_id: str) -> IngestJob | None:
        return self._update(
            job_id, status=STATUS_CANCELLED, finished_at=_now()
        )

    def request_cancel(self, job_id: str) -> IngestJob | None:
        """Idempotently ask for a job to be cancelled.

        - missing → None
        - terminal (done/error/cancelled) → returned unchanged
        - queued → tombstoned to ``cancelled`` now (worker skips it when popped)
        - running → flag ``cancel_requested``; worker finalizes on its next check
        """
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(job_id)
            if rec is None:
                return None
            status = rec.get("status")
            if status in (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED):
                return IngestJob(**rec)
            if status == STATUS_QUEUED:
                rec["status"] = STATUS_CANCELLED
                rec["finished_at"] = _now()
            else:  # running
                rec["cancel_requested"] = True
            rec["updated_at"] = _now()
            data[job_id] = rec
            atomic_write_json(self._registry, data)
            return IngestJob(**rec)
