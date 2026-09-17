"""Friendly, user-facing labels for import failure/skip states (G3 + BUG-016).

Import outcomes must read like a helpful status, not a raw error. Two distinct
per-video terminal conditions were previously conflated under one unfriendly
"Failed":

* a video with NO captions/subtitles is not a failure — nothing was wrong, there
  was simply nothing to transcribe. It is a **skip**, labelled
  "skipped (no captions)" (BUG-016).
* a real error (a proxy/network/extractor failure, a private/removed video) is a
  genuine **failure** and keeps a friendly one-line explanation (G3).

These are BACKEND status/label VALUES only — the SPA renders them in
sub-increment 1.5. Keeping the strings here (one source of truth) means the web
and Telegram surfaces, and the job record, all speak the same words.
"""
from __future__ import annotations

# Per-video skip label: a no-captions video is skipped, never "Failed".
LABEL_SKIPPED_NO_CAPTIONS = "skipped (no captions)"
# Per-video hard-failure label (a real error, not a missing-captions skip).
LABEL_FAILED = "failed"

# Friendly whole-job failure states (G3). The mapping below turns a raw
# exception string into one of these calm, human sentences.
FAILURE_BAD_URL = "That doesn't look like a valid YouTube channel or video link."
FAILURE_PRIVATE = "This channel or video is private or unavailable."
FAILURE_NO_VIDEOS = "No videos with captions were found to index."
FAILURE_BLOCKED = (
    "YouTube temporarily blocked the import. Please try again in a little while."
)
FAILURE_GENERIC = "The import didn't finish. Please try again."


def friendly_failure(raw_error: str) -> str:
    """Map a raw ingest error string to a calm, user-facing failure sentence (G3).

    Falls back to a generic-but-friendly message for anything unrecognized, so a
    surfaced failure never shows a stack-trace fragment or bare exception text.
    """
    text = (raw_error or "").strip().lower()
    if not text:
        return FAILURE_GENERIC
    if "429" in text or "block" in text or "rate" in text or "sign in to confirm" in text:
        return FAILURE_BLOCKED
    if "private" in text or "unavailable" in text or "members-only" in text:
        return FAILURE_PRIVATE
    if "not a valid" in text or "invalid" in text or "unsupported url" in text \
            or "no such channel" in text:
        return FAILURE_BAD_URL
    if "no videos" in text or "0 videos" in text or "empty" in text:
        return FAILURE_NO_VIDEOS
    return FAILURE_GENERIC
