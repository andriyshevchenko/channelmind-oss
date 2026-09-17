"""Parse subtitle files (VTT/SRT) into timestamped, deduplicated segments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import webvtt

_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")


@dataclass
class Segment:
    """A caption cue: its start/end offset (seconds) and cleaned text."""

    start: float
    end: float
    text: str


def _clean_line(line: str) -> str:
    line = _TAG_RE.sub("", line)  # strip inline timing tags like <00:00:01.000>
    return line.strip()


def _to_seconds(ts: str) -> float:
    m = _TS_RE.search(ts or "")
    if not m:
        return 0.0
    hours, minutes, seconds, millis = m.groups()
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    if millis:
        total += int(millis.ljust(3, "0")) / 1000.0
    return float(total)


def _caption_bounds(caption) -> tuple[float, float]:
    start = getattr(caption, "start_in_seconds", None)
    end = getattr(caption, "end_in_seconds", None)
    if start is None:
        start = _to_seconds(getattr(caption, "start", ""))
    if end is None:
        end = _to_seconds(getattr(caption, "end", ""))
    return float(start), float(end)


def parse_segments(path: Path) -> list[Segment]:
    """Read a .vtt (or .srt) file into timestamped segments.

    YouTube auto-captions roll: each cue repeats the previous line then appends a
    new one, so we dedupe consecutive identical lines and attach each cue's start
    time to the new text it introduces."""
    suffix = path.suffix.lower()
    if suffix == ".srt":
        captions = webvtt.from_srt(str(path))
    else:
        captions = webvtt.read(str(path))

    segments: list[Segment] = []
    last: str | None = None
    for caption in captions:
        new_lines: list[str] = []
        for raw in caption.text.splitlines():
            cleaned = _clean_line(raw)
            if not cleaned or cleaned == last:
                continue
            new_lines.append(cleaned)
            last = cleaned
        if not new_lines:
            continue
        start, end = _caption_bounds(caption)
        segments.append(Segment(start=start, end=end, text=" ".join(new_lines)))
    return segments


def segments_to_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)


def parse_subtitle_file(path: Path) -> str:
    """Backwards-compatible prose extraction (no timing)."""
    return segments_to_text(parse_segments(path))
