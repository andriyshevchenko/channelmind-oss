"""Registry of content sources feeding the local single-index library.

A "source" is one YouTube channel or one uploaded document. Every source has an
author (whose advice it represents) that is stamped onto each chunk's metadata so
the assistant can attribute claims. Sources share ONE vector index — the whole
point is to blend several pickup artists and reference docs into one corpus and
let the assistant reason across them.

The registry is the source of truth for de-duplication (never ingest the same
channel or file twice) and for showing the user what has been ingested.
"""
from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

TYPE_YOUTUBE = "youtube"
TYPE_DOCUMENT = "document"


@dataclass
class Source:
    id: str
    type: str            # youtube | document
    author: str          # e.g. "Ігор" — stamped onto every chunk
    label: str           # channel handle/name or document filename
    key: str             # normalized de-dup key
    added_at: str
    item_count: int = 0  # videos for a channel, 1 for a document
    chunk_count: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# YouTube hostnames we strip when parsing a channel reference (with trailing "/").
_YT_HOSTS = ("www.youtube.com/", "m.youtube.com/", "youtube.com/", "youtu.be/")

# A bare channel id pasted without the ``/channel/`` prefix — the ``UC…`` (canonical
# channel) or ``UU…`` (uploads) form. These are case-sensitive identity tokens, so
# they must fold to ``channel/<id>`` (NOT be lower-cased as a handle would be). The
# 20+ tail keeps this from matching ordinary handles like ``UChannel``/``UCLA``.
_BARE_CHANNEL_ID_RE = re.compile(r"^(?:UC|UU)[A-Za-z0-9_-]{20,}$")


def _parse_channel_ref(channel: str) -> tuple[str, str]:
    """Split a channel reference into (kind, identity).

    ``kind`` is one of "handle", "channel", "user", "c". Recognizes the canonical
    channel URL forms and their bare/handle equivalents, ignoring scheme, host, and
    any trailing tab suffix (/videos, /shorts, ...):

      ``@Igor`` / ``Igor`` / ``.../@Igor/videos``  -> ("handle", "Igor")
      ``.../channel/UCabc/shorts``                 -> ("channel", "UCabc")
      ``.../user/Foo``                             -> ("user", "Foo")
      ``.../c/Bar``                                -> ("c", "Bar")

    Handles are case-insensitive on YouTube, but the ``UC…`` channel id and the
    legacy user/custom names are identity-bearing tokens, so this returns them
    verbatim — the caller decides how to fold case. (Mirrors the parser in
    ingest.py, which builds the fetch URL from the same forms.)
    """
    raw = (channel or "").strip().rstrip("/")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    low = raw.lower()
    for host in _YT_HOSTS:
        if low.startswith(host):
            raw = raw[len(host):]
            break
    raw = raw.lstrip("/")
    segments = [s for s in raw.split("/") if s]
    if not segments:
        return ("handle", "")
    first = segments[0]
    if first.startswith("@"):
        return ("handle", first[1:])
    if first.lower() in ("channel", "user", "c") and len(segments) >= 2:
        return (first.lower(), segments[1])
    if _BARE_CHANNEL_ID_RE.match(first):
        return ("channel", first)
    return ("handle", first)


def normalize_youtube_key(channel: str) -> str:
    """Collapse the many spellings of a channel to one de-dup key.

    Every tab/scheme/host of ONE channel folds to a single key, while two DIFFERENT
    channels never collide:

      ``@Igor`` / ``Igor`` / ``https://youtube.com/@Igor/videos``  -> "@igor"
      ``.../channel/UCabc/videos`` / ``.../channel/UCabc/shorts``  -> "channel/UCabc"
      ``.../user/Foo``                                             -> "user/Foo"
      ``.../c/Bar``                                                -> "c/Bar"

    Handles are lower-cased (YouTube treats them case-insensitively); the ``UC…``
    id and legacy user/custom names keep their case, since those are case-sensitive
    identity tokens.
    """
    kind, ident = _parse_channel_ref(channel)
    if not ident:
        return ""
    if kind == "handle":
        return "@" + ident.lower()
    return f"{kind}/{ident}"


# A playlist id sits in the ``list=`` query param of a playlist/watch URL.
_PLAYLIST_ID_RE = re.compile(r"[?&]list=([\w-]+)")
# Personal / auto-generated lists that are NOT real shareable playlists and must
# be refused: Liked (LL), Watch Later (WL), radio/mix (RD…), user uploads (UL).
_REJECTED_PLAYLIST_RE = re.compile(r"^(LL|WL|RD|UL)")
# A bare playlist id pasted without a URL wrapper — only the real, addable
# prefixes (regular PL, uploads-as-playlist UU, old OL, favorites FL).
_BARE_PLAYLIST_ID_RE = re.compile(r"^(PL|UU|OL|FL)[\w-]*$")


def normalize_playlist_key(url: str) -> str:
    """Collapse a playlist reference to one de-dup key ``playlist/<PLID>``.

    Extracts the ``list=`` id from a playlist/watch URL (or accepts a bare
    ``PL…``/``UU…``/``OL…``/``FL…`` id). Playlist ids are CASE-SENSITIVE, so the
    id is returned verbatim — never lower-cased. Returns "" when no list id is
    present OR the id is a personal/auto-generated list (Liked ``LL``, Watch
    Later ``WL``, radio/mix ``RD``, uploads ``UL``); the caller rejects those
    with a 400."""
    raw = (url or "").strip()
    m = _PLAYLIST_ID_RE.search(raw)
    plid = m.group(1) if m else (raw if _BARE_PLAYLIST_ID_RE.match(raw) else "")
    if not plid or _REJECTED_PLAYLIST_RE.match(plid):
        return ""
    return f"playlist/{plid}"


# A single video is identified by its 11-char id, extracted from any of the URL
# forms YouTube uses to point at ONE video: a watch URL (``watch?v=<id>``), a
# short link (``youtu.be/<id>``), a Short (``/shorts/<id>``), or an embed
# (``/embed/<id>`` / legacy ``/v/<id>``). Video ids are ``[0-9A-Za-z_-]{11}``.
_VIDEO_ID_RE = re.compile(
    r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/v/)([0-9A-Za-z_-]{11})"
)


def normalize_video_key(url: str) -> str:
    """Collapse a single-video reference to one de-dup key ``video/<VIDEOID>``.

    Extracts the 11-char video id from a watch URL (``watch?v=<id>``), a short
    link (``youtu.be/<id>``), a Short (``/shorts/<id>``), or an embed
    (``/embed/<id>``). Video ids are CASE-SENSITIVE, so the id is returned
    verbatim — never lower-cased. Returns "" when no video id is present (the
    caller rejects that with a 400). The ``video/`` namespace never collides with
    a channel (``@handle`` / ``channel/UC…``) or playlist (``playlist/…``) key."""
    m = _VIDEO_ID_RE.search((url or "").strip())
    return f"video/{m.group(1)}" if m else ""


def normalize_document_key(filename: str) -> str:
    return (filename or "").strip().lower()


class SourceStore:
    """JSON-backed registry of content sources, keyed by id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "sources.json"

    # ---- layout -------------------------------------------------------
    def source_dir(self, source_id: str) -> Path:
        return self._root / "sources" / source_id

    def transcripts_dir(self, source_id: str) -> Path:
        return self.source_dir(source_id) / "transcripts"

    def documents_dir(self, source_id: str) -> Path:
        return self.source_dir(source_id) / "documents"

    # ---- reads --------------------------------------------------------
    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def list(self) -> list[Source]:
        rows = [Source(**rec) for rec in self._load().values()]
        return sorted(rows, key=lambda s: s.added_at)

    def get(self, source_id: str) -> Source | None:
        rec = self._load().get(source_id)
        return Source(**rec) if rec else None

    def find_by_key(self, type_: str, key: str) -> Source | None:
        for rec in self._load().values():
            if rec.get("type") == type_ and rec.get("key") == key:
                return Source(**rec)
        return None

    # ---- writes -------------------------------------------------------
    def add(self, type_: str, author: str, label: str, key: str) -> Source:
        source = Source(
            id=uuid.uuid4().hex[:12],
            type=type_,
            author=author,
            label=label,
            key=key,
            added_at=_now(),
        )
        with file_lock(self._registry):
            data = self._load()
            data[source.id] = asdict(source)
            atomic_write_json(self._registry, data)
        return source

    def update(self, source_id: str, **fields) -> Source | None:
        allowed = {"author", "label", "item_count", "chunk_count"}
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(source_id)
            if rec is None:
                return None
            for k, v in fields.items():
                if k in allowed and v is not None:
                    rec[k] = v
            data[source_id] = rec
            atomic_write_json(self._registry, data)
            return Source(**rec)

    def delete(self, source_id: str) -> Source | None:
        """Remove the record AND its payload files. The caller rebuilds the index."""
        with file_lock(self._registry):
            data = self._load()
            rec = data.pop(source_id, None)
            if rec is None:
                return None
            atomic_write_json(self._registry, data)
        shutil.rmtree(self.source_dir(source_id), ignore_errors=True)
        return Source(**rec)
