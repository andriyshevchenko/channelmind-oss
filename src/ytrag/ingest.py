"""Enumerate a channel's videos and download their transcripts via yt-dlp.

Ingestion is patient and resumable: it paces requests with an adaptive throttle,
backs off hard on any rate-limit/block, retries the same video, and skips videos
whose transcript JSON already exists — so a run can be left going for days and
resumed after interruption without re-downloading finished work.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlencode, urlsplit, urlunsplit

from yt_dlp import YoutubeDL

from .throttle import (
    AdaptiveThrottle,
    ThrottleConfig,
    humanize_seconds,
    is_block_error,
    is_permanent_skip,
    is_transient_transport_error,
)
from .logging_setup import proxy_diagnostic
from .safestore import atomic_write_json
from .transcripts import parse_segments, segments_to_text

logger = logging.getLogger(__name__)

ProgressFn = Callable[[dict], None]

# Emit an aggregate INFO progress heartbeat every this-many videos processed.
# Per-video detail stays at DEBUG so INFO reads as a low-noise progress stream.
_PROGRESS_EVERY = 10

# Poll interval, in seconds, while sleeping off a block cool-off. A long cool-off
# (up to ~30 min) is slept in short slices so a cancel requested mid-backoff is
# honoured within ~this many seconds instead of only between videos. Each slice
# fires a message-less "backoff" heartbeat whose only job is to run the progress
# callback's cancel probe (which raises to abort the download); UI consumers just
# refresh the stage — no log line, no counter bump.
_BACKOFF_POLL_SEC = 1.0


def _sleep_with_cancel(
    seconds: float,
    emit: Callable[..., None],
    *,
    index: int,
    total: int,
    video_id: str,
    attempt: int,
) -> None:
    """Sleep ``seconds`` in ``_BACKOFF_POLL_SEC`` slices, firing a heartbeat after
    each so a mid-backoff cancel (surfaced by the progress callback raising) aborts
    promptly instead of only at video boundaries."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_BACKOFF_POLL_SEC, remaining))
        emit(stage="backoff", index=index, total=total, video_id=video_id,
             attempt=attempt)


# YouTube hostnames we strip when parsing a channel reference (with trailing "/").
_YT_HOSTS = ("www.youtube.com/", "m.youtube.com/", "youtube.com/", "youtu.be/")

# A bare channel id (``UC…``/``UU…``) pasted without the ``/channel/`` prefix — a
# case-sensitive identity token that must build a ``/channel/<id>`` fetch URL rather
# than a ``/@<id>`` handle URL. Mirrors the same detector in ``sources.py`` so the
# de-dup key and the fetch URL agree for a bare id.
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
    verbatim — the caller decides how to fold case.
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


def _is_single_video_url(channel: str) -> bool:
    """True for a URL that points at ONE video, not a channel to enumerate.

    Matches a watch URL (``/watch?v=`` or ``youtu.be/<id>``) or a single Short
    (``/shorts/<id>``) — but NOT the bare ``/shorts`` channel tab. Such URLs must
    flow through as-is so single-video ingest paths keep working.
    """
    c = channel.strip().lower()
    if "watch?v=" in c or "/watch?" in c or "youtu.be/" in c:
        return True
    if re.search(r"/shorts/[\w-]{5,}", c):
        return True
    return False


def _is_playlist_ref(ref: str) -> bool:
    """True for a playlist reference — our ``playlist/<id>`` de-dup key or any URL
    carrying a ``list=`` query param.

    Checked BEFORE the single-video test in ``extract_channel`` so a
    ``watch?v=…&list=…`` URL enumerates the WHOLE playlist rather than the one
    video it happens to point at."""
    r = (ref or "").strip()
    if r.startswith("playlist/"):
        return True
    return bool(re.search(r"[?&]list=[\w-]+", r))


def _playlist_url(ref: str) -> str:
    """Build the canonical playlist URL from a playlist ref.

    ``playlist/<PLID>`` -> ``https://www.youtube.com/playlist?list=<PLID>``.
    A raw URL that already carries ``list=`` is normalized to the SAME canonical
    ``playlist?list=<PLID>`` form (dropping any ``watch?v=`` video so the entire
    playlist is listed)."""
    r = (ref or "").strip()
    if r.startswith("playlist/"):
        plid = r[len("playlist/"):]
    else:
        m = re.search(r"[?&]list=([\w-]+)", r)
        plid = m.group(1) if m else ""
    return f"https://www.youtube.com/playlist?list={plid}"


def _is_video_key(ref: str) -> bool:
    """True for our ``video/<id>`` single-video de-dup key."""
    return (ref or "").strip().startswith("video/")


def _video_url(ref: str) -> str:
    """Canonical watch URL for a ``video/<id>`` de-dup key.

    ``video/<VIDEOID>`` -> ``https://www.youtube.com/watch?v=<VIDEOID>``. A ref
    that is already a URL is returned as-is (defensive; single-video URLs flow
    through unchanged)."""
    r = (ref or "").strip()
    if r.startswith("video/"):
        return f"https://www.youtube.com/watch?v={r[len('video/'):]}"
    return r


def _channel_base(channel: str) -> str:
    """Return the tab-less canonical channel URL so callers can attach a tab.

    Accepts any recognized channel reference — a handle (``@Igor`` / ``Igor``) or a
    canonical URL form (``/channel/UC…``, ``/user/Name``, ``/c/Name``), possibly
    already pointing at a specific tab — and rebuilds its tab-less base so callers
    can attach ``/videos`` and ``/shorts`` themselves:

      ``@Igor``            -> ``https://www.youtube.com/@Igor``
      ``/channel/UCabc``   -> ``https://www.youtube.com/channel/UCabc``
      ``/user/Foo``        -> ``https://www.youtube.com/user/Foo``
      ``/c/Bar``           -> ``https://www.youtube.com/c/Bar``
    """
    kind, ident = _parse_channel_ref(channel)
    if kind == "handle":
        return f"https://www.youtube.com/@{ident}"
    return f"https://www.youtube.com/{kind}/{ident}"


def _normalize_channel_url(channel: str) -> str:
    """Back-compat: the channel's ``/videos`` tab URL, or a single-video URL as-is."""
    if _is_single_video_url(channel):
        return channel.strip()
    return _channel_base(channel) + "/videos"


def channel_meta_from_info(info: dict | None, is_playlist: bool = False) -> dict:
    """Extract {title, avatar} metadata from a yt-dlp channel/playlist info dict.

    Fully defensive: any missing/None/malformed input yields empty strings and
    never raises. For a CHANNEL, ``title`` prefers the channel/uploader name over
    ``info["title"]`` because a channel's /videos page reports a title like
    "Name - Videos". For a PLAYLIST, ``info["title"]`` IS the playlist's real
    name, so it's preferred (falling back to channel/uploader). ``avatar`` picks
    the best thumbnail URL: an entry whose ``id`` mentions "avatar" wins, else
    the widest thumbnail (by ``width``, then ``preference``)."""
    if not isinstance(info, dict):
        return {"title": "", "avatar": ""}
    if is_playlist:
        title = info.get("title") or info.get("channel") or info.get("uploader") or ""
    else:
        title = info.get("channel") or info.get("uploader") or ""
    thumbnails = info.get("thumbnails") or []
    avatar = ""
    if isinstance(thumbnails, list) and thumbnails:
        thumbs = [t for t in thumbnails if isinstance(t, dict) and t.get("url")]
        chosen = None
        for t in thumbs:
            if "avatar" in str(t.get("id") or "").lower():
                chosen = t
                break
        if chosen is None and thumbs:
            def _rank(t: dict) -> tuple[int, int]:
                w = t.get("width")
                p = t.get("preference")
                return (
                    w if isinstance(w, int) else -1,
                    p if isinstance(p, int) else -1,
                )
            chosen = max(thumbs, key=_rank)
        if chosen is not None:
            avatar = chosen.get("url") or ""
    return {"title": title, "avatar": avatar}


def _videos_from_entries(info: dict | None) -> list[dict]:
    """Flatten a yt-dlp channel-tab info dict into {id, title, url} entries."""
    if not isinstance(info, dict):
        return []
    videos: list[dict] = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        videos.append(
            {
                "id": e["id"],
                "title": e.get("title") or e["id"],
                "url": e.get("url") or f"https://youtu.be/{e['id']}",
            }
        )
    return videos


class EnumerationBlocked(RuntimeError):
    """yt-dlp swallowed a hard extraction failure (returned ``None``) under
    ``ignoreerrors=True`` — a channel-level 429/bot-block, a network/DNS error, or
    an absent/404 tab. Signalled as an ``error`` by the extract helpers so the
    caller treats a swallowed block like a raised one instead of a false empty."""


def _extract_tab(
    url: str, opts: dict, is_playlist: bool = False
) -> tuple[dict, list[dict], Exception | None]:
    """Extract ONE channel tab (or single-video URL) → (channel_meta, videos, error).

    Never raises. ``error`` is set when extraction FAILED — either because it threw
    (proxy down, DNS/network error) OR because yt-dlp, running with
    ``ignoreerrors=True``, swallowed a hard failure and returned ``None`` (a
    channel-level 429/bot-block or an absent/404 tab), which we surface as
    :class:`EnumerationBlocked`. A reachable tab always yields a playlist *dict*
    (with ``entries``, possibly empty), so a ``None`` result is a "couldn't fetch"
    signal — NOT a legitimately empty tab. The caller uses ``error`` to tell a real
    failure apart from an absent-but-fine tab: it degrades gracefully while at least
    one tab succeeds, but surfaces an error when they ALL fail instead of silently
    reporting zero videos. A genuinely empty channel/tab (a real dict with no
    entries) returns empty meta and ``error=None`` and so still succeeds.
    """
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - a dead tab must not fail the whole ingest
        return {"title": "", "avatar": ""}, [], exc
    if not info:
        return {"title": "", "avatar": ""}, [], EnumerationBlocked(
            f"yt-dlp returned no info for {url} "
            "(channel-level block/429, network error, or absent tab)"
        )
    return (
        channel_meta_from_info(info, is_playlist=is_playlist),
        _videos_from_entries(info),
        None,
    )


def _extract_single_video(
    url: str, opts: dict
) -> tuple[dict, list[dict], Exception | None]:
    """Extract ONE video → (meta, [single entry], error). Never raises.

    A single-video URL resolves to the VIDEO's own info dict, which (unlike a
    channel/playlist tab) has no ``entries``. We wrap that dict as a one-item
    ``{id, title, url}`` list so downstream ingest treats it as a one-video
    channel, and set ``meta.title`` to the VIDEO title (a video source lists as
    the video itself, not its uploader). ``error`` is set only on a real
    extractor failure, mirroring ``_extract_tab``."""
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - surfaced by the caller for single video
        return {"title": "", "avatar": ""}, [], exc
    if not info:
        # ``ignoreerrors=True`` turns a blocked/unreachable video into a ``None``
        # return rather than a raise; surface it so the single-video caller errors
        # instead of silently completing with zero transcripts.
        return {"title": "", "avatar": ""}, [], EnumerationBlocked(
            f"yt-dlp returned no info for {url} (block/429 or unreachable video)"
        )
    entries = _videos_from_entries(info)
    if not entries and info.get("id"):
        entries = [{
            "id": info["id"],
            "title": info.get("title") or info["id"],
            "url": info.get("webpage_url") or url,
        }]
    base_meta = channel_meta_from_info(info, is_playlist=False)
    title = info.get("title") or base_meta.get("title") or ""
    return {"title": title, "avatar": base_meta.get("avatar", "")}, entries, None


# ---- YouTube Shorts detection (BUG-009) -----------------------------------
# A bot excludes Shorts by default (``Bot.include_shorts``); these helpers decide
# which enumerated entries ARE Shorts so ``extract_channel`` can drop them before
# the newest-N cap. Two signals, layered like the direct→proxy enumeration
# fallback: the precise YouTube Data API duration (a Short runs <= 60s) when a
# ``YTRAG_YOUTUBE_API_KEY`` is set, and a zero-network heuristic — the entry came
# off the channel's ``/shorts`` tab, or its URL carries ``/shorts/`` — when there
# is no key or the API lookup fails.
_DATA_API_ROOT = "https://www.googleapis.com/youtube/v3"
_DATA_API_TIMEOUT_SEC = 8.0
_DATA_API_BATCH = 50  # videos.list caps ``id`` at 50 per request.
_SHORTS_MAX_SECONDS = 60

# Injected HTTP getter: (url) -> parsed JSON dict. Swapped in tests so the
# duration path is exercised offline against a canned Data API response.
JsonGetter = Callable[[str], dict]


def _default_data_api_get(url: str) -> dict:
    """GET ``url`` and parse the JSON body. Raises on transport/HTTP/JSON error."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_DATA_API_TIMEOUT_SEC) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _parse_iso8601_duration(text: str | None) -> int | None:
    """Parse a YouTube ``contentDetails.duration`` (ISO-8601, e.g. ``PT59S``,
    ``PT1M2S``, ``PT1H2M3S``) into whole seconds. Returns None when the input is
    missing or not a duration, so a malformed value never mis-classifies a video."""
    if not isinstance(text, str):
        return None
    m = re.fullmatch(
        r"P(?:(?P<d>\d+)D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?",
        text.strip(),
    )
    if not m or not any(m.groups()):
        return None
    d, h, mi, s = (int(m.group(k) or 0) for k in ("d", "h", "m", "s"))
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _short_ids_by_duration(
    ids: list[str], api_key: str, http_get: JsonGetter
) -> set[str] | None:
    """Return the subset of ``ids`` whose Data API duration is <= 60s (a Short).

    Batches ``ids`` in groups of 50 (the ``videos.list`` limit). Returns None on
    ANY HTTP/parse failure so the caller falls back to the tab heuristic instead
    of letting a lookup error break enumeration — the same degrade-don't-fail
    contract the rest of the ingest path keeps."""
    if not ids:
        return set()
    shorts: set[str] = set()
    for i in range(0, len(ids), _DATA_API_BATCH):
        batch = ids[i : i + _DATA_API_BATCH]
        params = {"part": "contentDetails", "id": ",".join(batch), "key": api_key}
        url = f"{_DATA_API_ROOT}/videos?{urlencode(params)}"
        try:
            payload = http_get(url)
        except Exception as exc:  # noqa: BLE001 - a lookup failure must never fail ingest
            logger.warning("shorts detect: Data API lookup failed: %s", exc)
            return None
        for item in payload.get("items") or []:
            vid = item.get("id")
            secs = _parse_iso8601_duration(
                (item.get("contentDetails") or {}).get("duration")
            )
            if vid and secs is not None and secs <= _SHORTS_MAX_SECONDS:
                shorts.add(vid)
    return shorts


def _heuristic_short_ids(videos: list[dict], shorts_tab_ids: set[str]) -> set[str]:
    """Zero-network Shorts heuristic: an entry yt-dlp listed under the channel's
    ``/shorts`` tab, or whose URL contains ``/shorts/``."""
    ids = set(shorts_tab_ids)
    for v in videos:
        if "/shorts/" in (v.get("url") or ""):
            ids.add(v["id"])
    return ids


def detect_short_ids(
    videos: list[dict],
    shorts_tab_ids: set[str],
    *,
    api_key: str | None = None,
    http_get: JsonGetter | None = None,
) -> set[str]:
    """Return the ids in ``videos`` that are YouTube Shorts.

    The tab/URL heuristic is ALWAYS applied — an entry off the channel's ``/shorts``
    tab IS a Short, whatever its length. When a Data API key is set, the duration
    signal is UNIONED on top (a <= 60s video the heuristic missed), never used to
    replace it: YouTube now allows Shorts up to 3 min, so a 61–180s Short still on
    the ``/shorts`` tab must not be re-classified long-form just because a key is
    present. The 60s duration threshold is kept deliberately conservative so the
    API path never false-positives a genuine 2–3 min long-form video. A failed
    lookup simply contributes nothing (the heuristic still stands). ``http_get``
    is injectable so the duration path is testable offline."""
    ids = _heuristic_short_ids(videos, shorts_tab_ids)
    if api_key:
        by_dur = _short_ids_by_duration(
            [v["id"] for v in videos], api_key, http_get or _default_data_api_get
        )
        if by_dur is not None:
            ids |= by_dur
    return ids


def extract_channel(
    channel: str,
    proxy: str | None = None,
    limit: int | None = None,
    include_shorts: bool = True,
) -> tuple[dict, list[dict]]:
    """Enumerate a channel's videos → (channel_meta, videos).

    Enumerates BOTH the ``/videos`` and ``/shorts`` tabs and merges them into one
    flat list of {id, title, url}, DEDUPed by video id (a video seen on both tabs
    appears once, keeping the /videos title). Channel {title, avatar} comes from
    the /videos extraction, falling back to /shorts when /videos is empty. Each
    ``extract_flat`` call is cheap; the Shorts tab adds at most one network call.

    ``include_shorts`` (BUG-009): when False, Shorts are dropped from the merged
    list BEFORE the ``limit`` cap, so a bot indexes the newest N *long-form*
    videos (cap + newest-first ordering preserved). Default True keeps the legacy
    merge-everything behaviour for callers that don't opt in. A single-video
    ``/shorts/<id>`` link is NOT affected — an explicit single-video request is
    always honoured (it never reaches the tab-merge branch below).

    A single-video URL (a watch URL or a ``/shorts/<id>`` link) is passed through
    unchanged for single-video ingest paths.
    """
    enumeration_limit = limit if limit is not None and limit > 0 else None
    opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
        **({"proxy": proxy} if proxy else {}),
        # Bound yt-dlp's flat playlist expansion.  This applies only to cheap
        # enumeration, never to subtitle download options.
        **({"playlistend": enumeration_limit} if enumeration_limit else {}),
    }
    logger.info(
        "enumeration start source=%s limit=%s proxy=%s",
        channel, enumeration_limit if enumeration_limit is not None else "unbounded",
        proxy_diagnostic(proxy),
    )

    # A playlist ref is enumerated as ONE flat list (its own ``list=`` page); there
    # is no /videos+/shorts merge and no fallback tab, so any extractor error is
    # surfaced. Checked FIRST so a ``watch?v=…&list=…`` URL lists the whole
    # playlist rather than being treated as a single video below.
    if _is_playlist_ref(channel):
        purl = _playlist_url(channel)
        meta, entries, error = _extract_tab(purl, opts, is_playlist=True)
        if error is not None:
            logger.error("enumerate playlist FAILED %s: %s", purl, error)
            raise RuntimeError(f"Failed to extract playlist {purl}: {error}") from error
        entries = entries[:enumeration_limit] if enumeration_limit else entries
        logger.info("enumerated playlist %s: %d videos", purl, len(entries))
        return meta, entries

    # A single-video source — either the ``video/<id>`` de-dup key or a raw watch /
    # youtu.be / ``/shorts/<id>`` URL — extracts exactly ONE video (its own info
    # dict, meta = the video title). There is no fallback tab, so any extractor
    # error is surfaced rather than silently completing with zero transcripts.
    if _is_video_key(channel) or _is_single_video_url(channel):
        single = _video_url(channel) if _is_video_key(channel) else channel.strip()
        meta, entries, error = _extract_single_video(single, opts)
        if error is not None:
            logger.error("enumerate single video FAILED %s: %s", single, error)
            raise RuntimeError(f"Failed to extract {single}: {error}") from error
        logger.info("enumerated single video %s: %d entr%s",
                    single, len(entries), "y" if len(entries) == 1 else "ies")
        return meta, entries[:enumeration_limit] if enumeration_limit else entries

    base = _channel_base(channel)
    videos_meta, videos_entries, videos_error = _extract_tab(base + "/videos", opts)
    shorts_meta, shorts_entries, shorts_error = _extract_tab(base + "/shorts", opts)

    # Degrade gracefully as long as ONE tab returned normally (a live /videos with a
    # 404/empty /shorts still succeeds). Only when BOTH tabs failed — proxy down,
    # network/DNS error, or a swallowed YouTube hard-block/429 that yt-dlp turned
    # into a None return (EnumerationBlocked) — do we re-raise so the ingest job is
    # marked errored rather than reporting a false "done, 0 videos". A channel that
    # is genuinely empty still reaches its tab (a real dict with no entries →
    # error=None), so it merges cleanly to [] instead of erroring.
    if videos_error is not None and shorts_error is not None:
        logger.error(
            "enumerate channel FAILED %s (both /videos and /shorts): %s",
            channel, videos_error,
        )
        raise RuntimeError(
            f"Failed to list videos for {channel}: {videos_error}"
        ) from videos_error

    # One dead tab is tolerated (degrade to the other) but worth a WARNING — a
    # channel scraping only half its videos is a clue when counts look low.
    if videos_error is not None:
        logger.warning("enumerate channel %s: /videos tab failed (%s) — using /shorts only",
                       channel, videos_error)
    elif shorts_error is not None:
        logger.warning("enumerate channel %s: /shorts tab failed (%s) — using /videos only",
                       channel, shorts_error)

    merged: list[dict] = []
    seen: set[str] = set()
    for v in (*videos_entries, *shorts_entries):
        vid = v["id"]
        if vid in seen:
            continue
        seen.add(vid)
        merged.append(v)

    # Prefer /videos channel meta; fall back to /shorts when /videos is empty.
    meta = videos_meta if videos_entries else (shorts_meta or videos_meta)

    # BUG-009: drop Shorts BEFORE the cap so the cap keeps the newest N *long-form*
    # videos (not the newest N of a shorts-diluted list). Done here, ahead of the
    # slice below, so both the 300/BETA cap and yt-dlp's newest-first ordering are
    # preserved. Shorts are detected via the Data API duration when a key is set,
    # else the /shorts-tab / URL heuristic (see ``detect_short_ids``).
    excluded_shorts = 0
    if not include_shorts and merged:
        from .config import youtube_api_key

        # A dead /shorts tab means the most reliable Shorts signal (tab-of-origin)
        # is missing, so filtering degrades to the URL/duration signals only — note
        # it so a channel that leaks a Short is explainable from the log.
        if shorts_error is not None:
            logger.warning(
                "enumerate channel %s: /shorts tab failed — Shorts filtering "
                "degraded to URL/duration signals only", channel,
            )
        shorts_tab_ids = {e["id"] for e in shorts_entries}
        short_ids = detect_short_ids(merged, shorts_tab_ids, api_key=youtube_api_key())
        if short_ids:
            before = len(merged)
            merged = [v for v in merged if v["id"] not in short_ids]
            excluded_shorts = before - len(merged)

    # /videos wins over /shorts before the total cap is applied.  A positive
    # playlistend limits each yt-dlp tab; this final cap prevents the two tabs
    # together from exceeding the caller's requested total.
    if enumeration_limit:
        merged = merged[:enumeration_limit]
    logger.info(
        "enumerated channel %s: %d videos (videos=%d, shorts=%d, shorts_excluded=%d)",
        channel, len(merged), len(videos_entries), len(shorts_entries), excluded_shorts,
    )
    return meta, merged


def enumerate_with_fallback(
    channel: str,
    proxy: str | None = None,
    limit: int | None = None,
    include_shorts: bool = True,
) -> tuple[dict, list[dict]]:
    """Enumerate a channel DIRECT (no proxy) first, falling back to the proxy
    only if the direct attempt is blocked (BUG-012).

    Channel/playlist enumeration is cheap flat metadata that does NOT need the
    residential transcript proxy; routing it through the proxy burned gigabytes of
    the monthly bandwidth budget for zero anti-block benefit. We therefore list via
    the server's own IP first and only retry through ``proxy`` when the direct
    listing genuinely fails (YouTube hard-block / 429 / network error surfaced as a
    ``RuntimeError``/``EnumerationBlocked``). The subtitle DOWNLOAD itself still
    always goes through the proxy — this only changes the listing step."""
    try:
        return extract_channel(channel, proxy=None, limit=limit, include_shorts=include_shorts)
    except RuntimeError:
        if not proxy:
            raise
        logger.warning(
            "direct enumeration failed for %s — retrying through the transcript proxy",
            channel,
        )
        return extract_channel(channel, proxy=proxy, limit=limit, include_shorts=include_shorts)


def list_video_ids(
    channel: str, proxy: str | None = None, include_shorts: bool = True
) -> list[dict]:
    """Return a flat list of {id, title, url} for every video on the channel.

    Enumeration goes direct-first with a proxy fallback (BUG-012) so periodic
    auto-sync listing doesn't burn residential-proxy bandwidth. ``include_shorts``
    (BUG-009) mirrors the ingest filter: with it False, auto-sync's "what's new"
    diff excludes Shorts too, so a Shorts-off bot doesn't re-detect and re-enqueue
    Shorts every tick only for the ingest to drop them."""
    from .config import test_mode

    if test_mode():
        from .testmode import canned_video_ids

        return canned_video_ids(channel)
    return enumerate_with_fallback(channel, proxy, include_shorts=include_shorts)[1]


def download_transcripts(
    channel: str,
    out_dir: Path,
    langs: Iterable[str] | None = None,
    limit: int | None = None,
    throttle: ThrottleConfig | None = None,
    max_retries: int | None = None,
    cookies_browser: str | None = None,
    progress: ProgressFn | None = None,
    proxy: str | None = None,
    job_id: str | None = None,
    budget_probe: "Callable[[], bool] | None" = None,
    include_shorts: bool = True,
) -> list[dict]:
    """Download subtitles for every video, patiently and resumably.

    Writes one <video_id>.json per video with clean transcript text. Returns the
    list of transcript metadata dicts that now exist on disk for this channel.

    ``max_retries``: None or <= 0 means unbounded — a blocked video is retried
    with escalating cool-offs, because YouTube's rate-limit window is unknown and
    may last hours. Even so, a per-video ceiling (``YTRAG_MAX_BLOCK_ATTEMPTS``
    attempts OR ``YTRAG_MAX_BLOCK_BACKOFF_SEC`` cumulative backoff seconds,
    whichever trips first) marks a persistently blocked video failed and moves on,
    so one wedged video can't monopolize the single global ingest worker. Only a
    *permanent* per-video condition (age-restricted, private, removed, ...) skips
    immediately. ``langs``: empty or None defaults to the video's ORIGINAL
    source-language auto-caption (yt-dlp's ``<lang>-orig`` track, selected via
    the ``".*-orig"`` regex) — one track per video, so a Russian channel yields
    Russian transcripts instead of a machine-translated ``en`` (or nothing).
    An explicit ``["auto"]`` or ``["all"]`` downloads every subtitle language
    the channel offers; a fixed default of one original track avoids yt-dlp
    fetching ~200 auto-translated tracks per video (an instant YouTube 429).
    The YouTube player client is pinned to a single lightweight one
    (``android_vr`` by default, see :func:`_player_client_opt`) so each video
    costs one player request instead of three and never hits the ``web``
    client's Subtitles PO-Token wall. ``cookies_browser`` (e.g. "chrome",
    "edge", "firefox") pulls logged-in cookies from that browser so
    age-restricted videos download and a signed-in session gets throttled less.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test-mode seam: skip yt-dlp entirely and materialize in-repo transcript
    # fixtures into out_dir, emitting the same progress stages the real fetcher
    # does so the queue's counts/statuses fire. The import then flows through the
    # REAL load -> chunk -> embed -> Chroma pipeline. Imported lazily so nothing
    # test-only loads on a production path.
    from .config import test_mode

    if test_mode():
        from .testmode import fake_download_transcripts

        return fake_download_transcripts(
            channel, out_dir, langs=langs, limit=limit, progress=progress
        )

    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    def emit(**kw):
        if progress:
            progress(kw)

    emit(stage="listing", message=f"Listing videos for {channel}")
    meta, videos = enumerate_with_fallback(
        channel, proxy=proxy, limit=limit, include_shorts=include_shorts
    )
    total = len(videos)
    emit(stage="listed", total=total, message=f"Found {total} videos")
    emit(stage="channel", title=meta.get("title", ""), avatar=meta.get("avatar", ""))

    tb = AdaptiveThrottle(throttle)
    unlimited = max_retries is None or max_retries <= 0
    retry_cap = "∞" if unlimited else str(max_retries)
    # Even in unbounded mode, cap how long ONE video may wedge the single global
    # FIFO worker with block cool-offs (H3): whichever of a cumulative backoff
    # wall-clock ceiling or a max-attempt count trips first ends the retry loop
    # for that video (marked failed + skipped), so a channel YouTube persistently
    # blocks can't starve every other tenant's queued ingest. Finite max_retries
    # behaviour is unchanged.
    from .config import max_block_attempts, max_block_backoff_sec, proxy_mb_per_video

    block_attempt_ceiling = max_block_attempts()
    block_backoff_ceiling = max_block_backoff_sec()
    # Flat per-video proxy-bandwidth estimate (MB) added to metered attempts when a
    # proxy is in use; the ceiling meter would otherwise wildly under-count real
    # traffic. Zero when running direct (no proxy) so metering stays honest.
    per_video_mb = proxy_mb_per_video() if proxy else 0.0
    tag = f"job={job_id} " if job_id else ""
    sub_langs = _resolve_langs(langs)
    player_client = _player_client_opt()
    logger.info(
        "%ssubtitle fetch configuration source=%s languages=%s client=%s proxy=%s cookies=%s",
        tag, channel, ",".join(sub_langs),
        ",".join(player_client["extractor_args"]["youtube"]["player_client"])
        if player_client else "yt-dlp default",
        proxy_diagnostic(proxy),
        "enabled" if _cookies_opt(cookies_browser) else "disabled",
    )
    opts = _subtitle_opts(sub_langs, cookies_browser, proxy, raw_dir)

    logger.info("%sdownloading transcripts for %s: %d video(s)%s",
                tag, channel, total, f" (limit={limit})" if limit is not None else "")

    saved: list[dict] = []
    failed = 0
    skipped = 0
    for i, v in enumerate(videos, 1):
        # Periodic aggregate heartbeat so a long run shows life at INFO without a
        # line per video (per-video detail lives at DEBUG/WARNING below).
        if total and i % _PROGRESS_EVERY == 0:
            logger.info("%sprogress %d/%d (saved=%d, failed=%d, skipped=%d) channel=%s",
                        tag, i, total, len(saved), failed, skipped, channel)
        vid = v["id"]
        dest = out_dir / f"{vid}.json"
        if dest.exists():
            # A finalized transcript = the resume skip (BUG-018). Guard the read:
            # a truncated/corrupt json left by a crash mid-write must NOT crash the
            # whole channel run — treat it as not-yet-saved and re-fetch below.
            try:
                cached = json.loads(dest.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                logger.warning("%scorrupt cached transcript %s — re-fetching [%d/%d]",
                               tag, vid, i, total)
            else:
                saved.append(_meta_summary(cached))
                skipped += 1
                logger.debug("%sskip existing %s [%d/%d]", tag, vid, i, total)
                emit(stage="skip", index=i, total=total, video_id=vid, title=v["title"],
                     message=f"[{i}/{total}] skip (already have) {v['title']}")
                continue

        logger.debug("%sfetching %s [%d/%d] %s", tag, vid, i, total, v["title"])
        # Per-user monthly proxy-bandwidth ceiling (crude): a video not yet on disk
        # is about to cost proxy bandwidth. If the owner has crossed their ceiling,
        # stop starting NEW downloads and finalize with what's already saved — the
        # bot is never left empty and chat keeps working.
        if budget_probe is not None and budget_probe():
            logger.info("%sbandwidth ceiling reached — stopping further downloads "
                        "for %s at [%d/%d] (saved=%d)", tag, channel, i, total,
                        len(saved))
            emit(stage="bandwidth_stop", index=i, total=total,
                 message=f"Monthly indexing limit reached — stopped after indexing "
                         f"{len(saved)} video(s) this month; your existing bot and chat still work")
            break
        attempt = 0
        block_backoff_total = 0.0
        transport_attempt = 0
        while True:
            attempt += 1
            # Sticky session normally keys on the video id (one exit IP per video,
            # so the player response and its caption GET share an IP — the 429 fix
            # in _apply_sticky_session). After a transient TRANSPORT failure the
            # exit itself is dead, so rotate to a FRESH session key: the retry then
            # leaves via a DIFFERENT exit IP instead of hammering the broken one.
            session_key = vid if transport_attempt == 0 else f"{vid}-t{transport_attempt}"
            video_proxy = _apply_sticky_session(proxy, session_key)
            video_opts = opts if video_proxy == proxy else {**opts, "proxy": video_proxy}
            tb.wait(operation="subtitle")
            try:
                with YoutubeDL(video_opts) as ydl:
                    info = ydl.extract_info(v["url"], download=True)
                tb.on_success(operation="subtitle")
                break
            except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
                # Permanent per-video conditions never recover — skip at once so
                # unbounded mode can't loop forever on one age-gated video.
                if is_permanent_skip(exc):
                    failed += 1
                    logger.warning("%svideo %s unavailable [%d/%d]: %s (failed=%d)",
                                   tag, vid, i, total, exc, failed)
                    emit(stage="skip_permanent", index=i, total=total, video_id=vid,
                         title=v["title"],
                         message=f"[{i}/{total}] skip (unavailable) {v['title']}: {exc}")
                    info = None
                    break
                if is_block_error(exc):
                    # Retry the block *unless* a ceiling is hit. Finite mode is
                    # bounded by max_retries (unchanged). Unbounded mode is bounded
                    # by a hard per-video attempt count AND cumulative backoff
                    # wall-clock — whichever trips first — so a persistently blocked
                    # video can't wedge the one global worker for every tenant.
                    if unlimited:
                        exhausted = (attempt >= block_attempt_ceiling
                                     or block_backoff_total >= block_backoff_ceiling)
                    else:
                        exhausted = attempt > max_retries
                    if not exhausted:
                        cooloff = tb.on_block(operation="subtitle")
                        emit(stage="backoff", index=i, total=total, video_id=vid,
                             attempt=attempt, cooloff=round(cooloff, 1),
                             delay=round(tb.current_delay, 1),
                             message=f"[{i}/{total}] blocked, cooling off "
                                     f"{humanize_seconds(cooloff)} "
                                     f"(attempt {attempt}/{retry_cap})")
                        block_backoff_total += cooloff
                        _sleep_with_cancel(cooloff, emit, index=i, total=total,
                                           video_id=vid, attempt=attempt)
                        continue
                    # Ceiling reached — fail THIS video only (record the block
                    # reason), then fall through so the loop continues to the next
                    # video / the worker moves to the next job. Never hangs the run
                    # and never touches sibling videos.
                    failed += 1
                    logger.warning(
                        "%svideo %s blocked past ceiling [%d/%d] after %d attempt(s), "
                        "%s of backoff: %s (failed=%d)",
                        tag, vid, i, total, attempt,
                        humanize_seconds(block_backoff_total), exc, failed)
                    emit(stage="error", index=i, total=total, video_id=vid,
                         title=v["title"],
                         message=f"[{i}/{total}] gave up on {v['title']} after "
                                 f"{attempt} attempt(s) / "
                                 f"{humanize_seconds(block_backoff_total)} blocked: {exc}")
                    info = None
                    break
                if is_transient_transport_error(exc):
                    # A dropped socket / TLS EOF from a flaky residential exit — NOT
                    # a block. Retry with a FRESH proxy session (new exit IP), bounded
                    # by the SAME per-video ceilings as a block so one persistently
                    # broken exit can't wedge the single global worker. A short local
                    # backoff (not the heavy block cool-off) — the exit changed, so no
                    # long penance is warranted.
                    if unlimited:
                        exhausted = (attempt >= block_attempt_ceiling
                                     or block_backoff_total >= block_backoff_ceiling)
                    else:
                        exhausted = attempt > max_retries
                    if not exhausted:
                        transport_attempt += 1
                        cooloff = min(2.0 * (2 ** (transport_attempt - 1)), 30.0)
                        block_backoff_total += cooloff
                        emit(stage="backoff", index=i, total=total, video_id=vid,
                             attempt=attempt, cooloff=round(cooloff, 1),
                             delay=round(tb.current_delay, 1),
                             message=f"[{i}/{total}] network hiccup, retrying via a "
                                     f"fresh proxy session in {humanize_seconds(cooloff)} "
                                     f"(attempt {attempt}/{retry_cap})")
                        _sleep_with_cancel(cooloff, emit, index=i, total=total,
                                           video_id=vid, attempt=attempt)
                        continue
                    # Ceiling reached — fall through to fail THIS video only.
                failed += 1
                logger.warning("%svideo %s error [%d/%d]: %s (failed=%d)",
                               tag, vid, i, total, exc, failed)
                emit(stage="error", index=i, total=total, video_id=vid,
                     message=f"[{i}/{total}] error on {v['title']}: {exc}")
                info = None
                break

        # A fetch failure (info is None) ALREADY emitted its own error/skip_permanent
        # event above. Emitting a second "nosub" here would double-count the video —
        # the progress driver bumps videos_failed on BOTH events. Skip straight to the
        # next video; only a clean fetch that yielded no subtitle file is a real nosub.
        # Every video that reached the download loop transited the proxy at least
        # once (success, caption-less, or blocked) — meter the flat per-video
        # bandwidth estimate toward the owner's ceiling regardless of outcome.
        if per_video_mb:
            emit(stage="proxy_usage", index=i, total=total, video_id=vid,
                 proxy_mb=per_video_mb)
        if info is None:
            continue
        sub_file = _find_subtitle_file(raw_dir, vid)
        if sub_file is None:
            failed += 1
            logger.info("%sno captions for %s [%d/%d] (failed=%d)",
                        tag, vid, i, total, failed)
            emit(stage="nosub", index=i, total=total, video_id=vid, title=v["title"],
                 message=f"[{i}/{total}] no captions for {v['title']}")
            continue

        segments = parse_segments(sub_file)
        text = segments_to_text(segments)
        # Bytes fetched through the proxy for this video (crude bandwidth meter):
        # the downloaded .vtt size, captured before it's purged below.
        try:
            sub_bytes = sub_file.stat().st_size
        except OSError:
            sub_bytes = 0
        # The .vtt is now parsed into memory and disposable. Purge the video's
        # staged subtitle file(s) so <source>/scrape/_raw/ doesn't grow without
        # bound (wasted disk + an ever-slower glob on later runs) — L6.
        _purge_raw_subtitles(raw_dir, vid)
        if not text.strip():
            failed += 1
            logger.info("%sempty captions for %s [%d/%d] (failed=%d)",
                        tag, vid, i, total, failed)
            emit(stage="nosub", index=i, total=total, video_id=vid, title=v["title"],
                 message=f"[{i}/{total}] empty captions for {v['title']}")
            continue

        meta = {
            "id": vid,
            "title": info.get("title") or v["title"],
            "url": v["url"],
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "text": text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in segments
            ],
        }
        # Atomic write (temp + os.replace) so a crash mid-write can never leave a
        # truncated json that would crash the next resume (BUG-018 robustness).
        atomic_write_json(dest, meta)
        saved.append(_meta_summary(meta))
        emit(stage="saved", index=i, total=total, video_id=vid, title=meta["title"],
             delay=round(tb.current_delay, 1), bytes=sub_bytes,
             proxy_mb=round(sub_bytes / 1_000_000, 6),
             message=f"[{i}/{total}] saved {meta['title']}")

    logger.info("%sfinished %s: %d/%d transcript(s) available (failed=%d, skipped=%d)",
                tag, channel, len(saved), total, failed, skipped)
    emit(stage="done", total=total, saved=len(saved),
         message=f"Done. {len(saved)} transcripts available.")
    return saved


_KNOWN_BROWSERS = {"chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi", "safari"}


def _cookies_opt(cookies_browser: str | None) -> dict:
    """yt-dlp option to read logged-in cookies from a local browser, if requested."""
    b = (cookies_browser or "").strip().lower()
    if not b or b in ("none", "off"):
        return {}
    if b not in _KNOWN_BROWSERS:
        return {}
    return {"cookiesfrombrowser": (b,)}


_DEFAULT_PLAYER_CLIENT = "android_vr"

# Username-suffix template (must contain "{s}") that pins a rotating residential
# proxy to a per-video sticky exit IP. Empty/unset = no session pinning.
_STICKY_ENV = "YTRAG_TRANSCRIPT_PROXY_STICKY_FORMAT"


def _apply_sticky_session(proxy: str | None, session_key: str) -> str | None:
    """Pin a rotating residential proxy to a per-key sticky exit IP.

    A rotating gateway hands out a NEW exit IP on every request. yt-dlp fetches a
    video's player response and then its timedtext caption on SEPARATE requests,
    so under rotation each leaves via a different IP — YouTube binds the player
    session to the first IP and 429s the caption GET that arrives from another,
    which is exactly the failure the real-e2e log showed (``session=rotating``).

    This injects a deterministic session token derived from ``session_key`` (the
    video id) into the proxy username, so every request for ONE video shares ONE
    exit IP while DIFFERENT videos still rotate to different IPs. That is the
    correct residential pattern at scale: it removes the session-binding 429
    without collapsing all traffic onto a single hammered IP (which a statically
    sticky username would).

    Controlled by ``YTRAG_TRANSCRIPT_PROXY_STICKY_FORMAT`` — a username-suffix
    template containing ``{s}``. For a typical rotating-residential proxy the
    value is simply ``-{s}``: many gateways have NO dashboard toggle for sticky
    sessions; a session is a numeric id appended to the username
    (``user-<number>`` = same exit IP for ~10 min, ``user-rotate`` = rotate per
    request). Accordingly the injected token is a plain integer (such gateways reject
    alphanumeric session ids) and any trailing ``-rotate`` on the base username is
    replaced. Empty/unset (the default) returns the proxy unchanged, so behaviour
    is identical until the operator opts in.
    """
    fmt = os.getenv(_STICKY_ENV, "").strip()
    if not proxy or not fmt or "{s}" not in fmt:
        return proxy
    try:
        parts = urlsplit(proxy)
        username = parts.username
        hostname = parts.hostname
        password = parts.password
        port = parts.port  # may raise ValueError on a non-numeric port
    except ValueError:
        # A malformed proxy string must NEVER crash the ingest job here (this runs
        # before the YoutubeDL try/except) — fail open and leave the proxy as-is.
        return proxy
    if not username or not hostname:
        return proxy
    # Most residential proxy gateways require a NUMERIC session token;
    # a hex digest would fail proxy auth. Derive a stable integer from the key.
    sid = str(int(hashlib.sha1(session_key.encode("utf-8")).hexdigest(), 16) % 1_000_000_000)
    # A rotating username ends in "-rotate"; the session token takes its place.
    base_user = re.sub(r"-rotate$", "", username, flags=re.IGNORECASE)
    auth = base_user + fmt.replace("{s}", sid)
    if password is not None:
        auth += f":{password}"
    # Re-bracket an IPv6 literal so the rebuilt netloc stays parseable.
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        host += f":{port}"
    netloc = f"{auth}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _player_client_opt() -> dict:
    """yt-dlp ``extractor_args`` pinning the YouTube player client(s).

    Why this exists — YouTube's default yt-dlp client set is
    ``(visionos, android_vr, web)``: yt-dlp issues a separate player request per
    client, so every video costs THREE player round-trips — triple the exposure
    to YouTube's rate limiter and "Sign in to confirm you're not a bot" gate,
    which is exactly what tripped the real-e2e 429s through the residential
    proxy. Worse, the ``web`` client now requires a Subtitles PO Token to return
    captions at all (without one they are silently discarded), so it burns a
    request and yields nothing for our subtitle-only workload.

    Pinning a single lightweight client that returns automatic captions WITHOUT
    a PO Token and WITHOUT a JS challenge (``android_vr``: ``REQUIRE_JS_PLAYER``
    is False) cuts each video to ONE player request and sidesteps the web
    PO-Token subtitle wall entirely. Override with the
    ``YTRAG_YTDLP_PLAYER_CLIENT`` env var (a comma list, e.g. ``"ios"`` or
    ``"android_vr,ios"``); set it empty or ``"default"`` to fall back to
    yt-dlp's own default client selection.
    """
    raw = os.getenv("YTRAG_YTDLP_PLAYER_CLIENT", _DEFAULT_PLAYER_CLIENT)
    clients = [c.strip() for c in raw.split(",") if c.strip()]
    if not clients or clients == ["default"]:
        return {}
    return {"extractor_args": {"youtube": {"player_client": clients}}}


def _subtitle_opts(
    sub_langs: list[str],
    cookies_browser: str | None,
    proxy: str | None,
    raw_dir: Path,
) -> dict:
    """Build the yt-dlp options for a subtitle-only download.

    Pins the YouTube player client (:func:`_player_client_opt`) so each video
    costs a single player request that returns captions without a PO Token, and
    requests only ``sub_langs`` (the original source language by default) so a
    run never fans out to ~200 auto-translated tracks. ``ignoreerrors`` is off
    on purpose: the caller wants the exception so it can back off on a block."""
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": sub_langs,
        "subtitlesformat": "vtt",
        "ignoreerrors": False,  # we want exceptions so we can back off
        **_player_client_opt(),
        **_cookies_opt(cookies_browser),
        **({"proxy": proxy} if proxy else {}),
        "outtmpl": str(raw_dir / "%(id)s.%(ext)s"),
    }


def _resolve_langs(langs: Iterable[str] | None) -> list[str]:
    """Resolve the requested subtitle languages for yt-dlp.

    An explicit ``"all"``/``"auto"`` request maps to yt-dlp's all-languages
    wildcard. Explicit language codes pass through (lower-cased, blanks dropped).

    An empty/unspecified request defaults to the video's ORIGINAL source
    language via the ``".*-orig"`` regex — the single auto-caption track yt-dlp
    keys as ``<lang>-orig`` (e.g. ``ru-orig`` for a Russian video, ``en-orig``
    for an English one). This is deliberately NOT a fixed ``["en"]``: forcing
    English silently pulls a machine-TRANSLATED caption for a non-English
    channel (or nothing when no English translation exists), which is poor
    grounding material for a personal bot built from, say, a Russian channel.
    Selecting the original keeps one track per video (so it never regresses to
    the ~200-track ``"all"`` fetch that instantly trips YouTube's HTTP 429) while
    capturing the true transcript regardless of language. Callers that genuinely
    want every language must still ask for it explicitly with ``"all"``.
    """
    items = [str(x).strip().lower() for x in (langs or []) if str(x).strip()]
    if "auto" in items or "all" in items:
        return ["all"]
    if not items:
        return [".*-orig"]
    return items


def _meta_summary(meta: dict) -> dict:
    return {k: meta[k] for k in ("id", "title", "url")}


def _find_subtitle_file(raw_dir: Path, video_id: str) -> Path | None:
    matches = sorted(raw_dir.glob(f"{video_id}*.vtt"))
    manual = [m for m in matches if ".auto." not in m.name]
    if manual:
        return manual[0]
    return matches[0] if matches else None


def _purge_raw_subtitles(raw_dir: Path, video_id: str) -> None:
    """Delete a video's staged ``.vtt`` file(s) once parsed to JSON (L6).

    yt-dlp writes raw subtitle files into ``<source>/scrape/_raw/`` (one or more
    per video/language). Left in place they accumulate without bound — wasted disk
    plus an ever-slower :func:`_find_subtitle_file` glob on every later run. Once
    the transcript is parsed into memory the ``.vtt`` is disposable. Best-effort:
    any unlink failure is ignored (a leftover file is harmless)."""
    for f in raw_dir.glob(f"{video_id}*.vtt"):
        try:
            f.unlink()
        except OSError:
            pass


def load_transcripts(out_dir: Path) -> list[dict]:
    """Load every transcript json in ``out_dir``, ordered NEWEST-FIRST.

    yt-dlp stamps ``upload_date`` as a zero-padded ``YYYYMMDD`` string, which sorts
    correctly lexicographically. Ordering by it descending means a rebuild embeds
    the most recent videos first, so a freshly-imported (or capped) channel becomes
    useful for its latest content quickly while older videos backfill — instead of
    the arbitrary alphabetical-by-video-id order a plain filename sort gave. A
    blank/missing date is the empty string, which sorts LAST under the descending
    order (an undated doc is treated as oldest). The two-pass sort keeps the
    ordering deterministic: an ``id``-ascending pass first makes equal-date docs a
    stable, reproducible tie-break."""
    docs = [
        json.loads(f.read_text(encoding="utf-8"))
        for f in out_dir.glob("*.json")
    ]
    docs.sort(key=lambda d: str(d.get("id") or ""))
    docs.sort(key=lambda d: str(d.get("upload_date") or ""), reverse=True)
    return docs
