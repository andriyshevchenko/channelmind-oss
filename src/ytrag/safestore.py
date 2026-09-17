"""Atomic, lockable JSON persistence shared by settings and the corpus registry.

Concurrent writers (web UI thread, admin API, background build jobs) can all
touch these small JSON files, so we serialize read-modify-write cycles with a
per-path lock and commit via a temp file + os.replace so a crash mid-write never
leaves a truncated file. Files are chmod 0600 (best-effort on Windows).

Readers take the SAME per-path lock (see ``read_json``): a read can therefore
never overlap the writer's ``os.replace`` window. That closes two related
Windows races — the writer's replace failing with a sharing violation
(WinError 5/32) because an unlocked reader held the file open, and a reader
observing a transient empty/partial file and reporting it as an empty document
(which made the SPA job-poller drop a live job). The lock is RE-ENTRANT so the
common "read_json while already holding file_lock" write pattern never deadlocks.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Per-path RE-ENTRANT locks. Re-entrancy matters: writers hold file_lock across a
# whole read-modify-write and call read_json() inside it — a plain Lock would
# self-deadlock the moment read_json also acquired the lock.
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def file_lock(path: Path) -> threading.RLock:
    """Return the process-wide RE-ENTRANT lock guarding a given file path.

    Hold this across a full read-modify-write to avoid lost updates. Being an
    RLock, a writer already holding it may call ``read_json`` (which re-acquires
    it) without deadlocking, while a reader on ANOTHER thread still fully excludes
    the writer's ``os.replace`` window."""
    key = str(Path(path).resolve())
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _locks[key] = lock
        return lock


def read_json(path: Path, default: Any = None) -> Any:
    """Read + parse a JSON file under the SAME per-path lock as the writer.

    Holding ``file_lock`` here means a read on another thread can never overlap the
    writer's ``os.replace``: it sees either the fully-committed old file or waits
    for the committed new one — never a truncated/empty in-between. That removes the
    Windows sharing violation the replace used to hit AND the transient empty read
    that made callers (e.g. the SPA job store) treat live state as gone.

    A DIFFERENT process replacing the file is outside our in-process lock, so on the
    rare transient ``OSError`` (sharing violation) or a momentarily empty/partial
    read we retry briefly rather than immediately surfacing the file as ``default``
    (an empty result would drop live state). Only a genuinely unreadable/undecodable
    file after the bounded retries falls back to ``default``.
    """
    p = Path(path)
    with file_lock(p):
        for attempt in range(12):
            if not p.exists():
                return default
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                # Transient cross-process sharing violation mid-replace — retry.
                if attempt < 11:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                return default
            if text.strip() == "":
                # File exists but momentarily blank (a cross-process replace in
                # flight). Registry files are never legitimately empty ("{}" at
                # minimum), so never accept this as a valid empty document — retry.
                if attempt < 11:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                return default
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Partial content from a cross-process writer — retry for the
                # committed version before giving up.
                if attempt < 11:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                return default
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _replace_with_retry(tmp: str, path: Path) -> None:
    """``os.replace`` with a short backoff retry for Windows sharing violations.

    On Windows ``os.replace`` fails with ``PermissionError`` (WinError 5 / 32)
    when another thread or process holds the destination open for reading at that
    instant — e.g. a concurrent unlocked reader of a registry file, or an AV
    scanner briefly locking the freshly written temp file. The rename is atomic
    and idempotent, so retrying after a tiny delay lets the transient lock clear
    instead of surfacing a 500. POSIX ``os.replace`` never raises this, so the
    first attempt succeeds there and the loop is a no-op.
    """
    last: Exception | None = None
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # WinError 5/32 — sharing violation
            last = exc
            if attempt < 11:
                time.sleep(0.05 * (attempt + 1))
    if last is not None:
        raise last
