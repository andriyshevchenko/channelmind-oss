"""The ingest seam: turn already-extracted transcript items into corpus files.

This is the single choke point every ingest path funnels through, and the reason
the backend is ready for a future "thick client" (browser extension). The public
HTTP endpoint accepts a list of transcript items (id + title + url + text +
optional timed segments) and writes them here; the server-side yt-dlp fetcher
produces the exact same on-disk shape into the same directory. So whether a
transcript is scraped on the server or extracted in the user's browser and POSTed
up, everything downstream (chunk → embed → the bot's collection) is identical and
no backend change is needed when the extension ships.

Each item is written as ``<video_id>.json`` and stamped with the source author so
the rebuild can attribute every chunk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _safe_id(raw: str) -> str:
    """A filesystem-safe id; fall back to a hash-ish slug for odd inputs."""
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", (raw or "").strip())
    return s[:120] or "item"


def normalize_item(raw: dict, author: str, source_label: str) -> dict:
    """Coerce a loosely-shaped client item into the canonical transcript doc.

    Accepts the same fields ``download_transcripts`` writes plus ``author``/
    ``source`` stamps. ``segments`` (timed captions) are optional — without them
    the chunker falls back to untimed prose (no deep-link timecodes)."""
    vid = _safe_id(str(raw.get("id") or raw.get("video_id") or raw.get("url") or ""))
    segments = []
    for s in raw.get("segments") or []:
        try:
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", start) or start)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        segments.append({"start": start, "end": end, "text": str(s.get("text", ""))})
    # A client (e.g. the browser extension) may send timed segments only; derive
    # the flat text from them so the item isn't dropped as "empty".
    text = str(raw.get("text") or "").strip()
    if not text and segments:
        text = " ".join(s["text"] for s in segments if s["text"]).strip()
    return {
        "id": vid,
        "title": str(raw.get("title") or vid),
        "url": str(raw.get("url") or ""),
        "upload_date": raw.get("upload_date"),
        "duration": raw.get("duration"),
        "text": text,
        "segments": segments,
        "author": author,
        "source": source_label,
    }


def write_items(
    transcripts_dir: Path, items: list[dict], author: str, source_label: str
) -> int:
    """Write normalized transcript items to ``<id>.json``. Returns count written.

    Items whose text is empty are skipped. Existing files are overwritten (a
    re-POST refreshes the transcript)."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for raw in items or []:
        doc = normalize_item(raw, author, source_label)
        if not doc["text"].strip():
            continue
        dest = transcripts_dir / f"{doc['id']}.json"
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    return written
