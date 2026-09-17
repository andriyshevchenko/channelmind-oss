"""Narrow, append-only audit log for legally-significant events (Phase B).

We record ONLY events that matter for a dispute or a compliance ask — a user
attesting they have the rights to content they add, share-token lifecycle,
takedowns — NOT page views or chat queries (that would be GDPR over-collection).

Storage is one JSON object per line in ``data/audit.log`` (JSONL): append-only,
trivial to tail, and never rewritten. We reuse the ``safestore`` per-path lock so
two writers never interleave a half-written line. This is deliberately tiny — a
richer hash-chained store is the later B5 work; here we just need a durable,
timestamped record of each acceptance.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .safestore import file_lock

# Version tag for the rights-attestation copy the user clicks through. Bump this
# whenever the attestation wording materially changes so old acceptances stay
# pinned to the text that was actually shown.
CONTENT_RIGHTS_ATTEST_VERSION = "content-rights-attest v1"


class AuditWriteError(Exception):
    """Raised when an audit entry could not be durably written.

    Callers guarding a legally-significant action (a rights attestation) must
    treat this as fail-closed: refuse the action rather than proceed with no
    record of the acceptance.
    """


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(
    data_dir: str | os.PathLike,
    *,
    actor: str,
    action: str,
    target: str,
    doc_version: str = "",
    outcome: str = "accepted",
    ip: str = "",
    **extra: object,
) -> dict:
    """Append one audit entry and return it.

    ``actor`` is the acting user id, ``action`` a stable event slug (e.g.
    ``"source_attestation"``), ``target`` the thing acted on (bot id + source
    key), ``doc_version`` the version of any document accepted, ``outcome`` the
    result. Extra keyword args are merged in for event-specific context.
    """
    path = Path(data_dir) / "audit.log"
    entry: dict = {
        "ts": _utc_now_iso(),
        "actor": actor,
        "action": action,
        "target": target,
        "doc_version": doc_version,
        "outcome": outcome,
        "ip": ip,
    }
    if extra:
        entry.update(extra)
    line = json.dumps(entry, ensure_ascii=False)
    try:
        with file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            try:
                os.chmod(path, 0o600)  # best-effort; no-op on Windows
            except OSError:
                pass
    except OSError as exc:
        # Fail-closed: the caller guards a legally-significant action and must
        # refuse it rather than proceed with no durable record of acceptance.
        raise AuditWriteError(str(exc)) from exc
    return entry
