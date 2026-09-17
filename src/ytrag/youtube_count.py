"""Approximate channel video-count helper for determinate import progress (BUG-001).

The import UI wants two numbers for an honest, never-stuck progress display
(operator-approved "option B"):

* the **window total** — how many videos THIS run will index, i.e.
  ``min(channel_total, per-channel BETA cap)`` — which is exactly the count the
  enumeration step already produces, and
* the channel's **approximate total** — used only for the context line
  ("Channel ~2,500 videos, indexing the 300 newest").

Only the approximate total needs an external lookup, and this module is the one
place that does it. When a ``YTRAG_YOUTUBE_API_KEY`` is configured we ask the
YouTube Data API v3 for the channel's public ``statistics.videoCount`` (one cheap
request). When NO key is set — the default — we DO NOT fail: we fall back to a
caller-supplied estimate (the enumerated window count), so import and progress
work fully offline with no key and no network dependency on this path.

The count is deliberately labelled "~": ``videoCount`` includes Shorts and lags
private/unlisted churn, so it is a context hint, never a precise denominator. The
progress BAR is always driven by the exact window total, never this number.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Callable

from .config import youtube_api_key
from .ingest import _parse_channel_ref

logger = logging.getLogger(__name__)

_API_ROOT = "https://www.googleapis.com/youtube/v3"
_HTTP_TIMEOUT_SEC = 8.0

# Injected HTTP getter type: (url) -> parsed JSON dict. Swapped in tests so the
# with-key path is exercised deterministically with a mocked API response.
JsonGetter = Callable[[str], dict]


def _default_http_get(url: str) -> dict:
    """GET ``url`` and parse the JSON body. Raises on transport/HTTP/JSON error."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _query(part_params: dict, key: str) -> str:
    """Build a Data API URL for ``channels.list`` with the given selector params."""
    params = {"part": "statistics", "key": key, **part_params}
    return f"{_API_ROOT}/channels?{urllib.parse.urlencode(params)}"


def _selector(channel_url: str) -> dict | None:
    """Map a channel reference to the ``channels.list`` selector params, or None.

    ``@handle`` → ``forHandle``; a ``UC…`` channel id → ``id``. Legacy ``user``
    names use ``forUsername``. A custom ``/c/<name>`` slug has no direct
    ``channels.list`` selector (it needs a resolve step we deliberately skip to
    keep this to one request), so it returns None → the caller falls back.
    """
    kind, ident = _parse_channel_ref(channel_url)
    if not ident:
        return None
    if kind == "channel":
        return {"id": ident}
    if kind == "handle":
        return {"forHandle": f"@{ident}"}
    if kind == "user":
        return {"forUsername": ident}
    return None


def _video_count_from_api(channel_url: str, key: str, http_get: JsonGetter) -> int | None:
    """Fetch the channel's public ``statistics.videoCount`` via the Data API.

    Returns the integer count, or None on any error / empty result so the caller
    can fall back rather than propagate a network failure into the ingest path."""
    selector = _selector(channel_url)
    if selector is None:
        return None
    try:
        payload = http_get(_query(selector, key))
    except Exception as exc:  # noqa: BLE001 - a lookup failure must never fail ingest
        logger.warning("youtube count: API lookup failed for %s: %s", channel_url, exc)
        return None
    items = payload.get("items") or []
    if not items:
        return None
    stats = items[0].get("statistics") or {}
    raw = stats.get("videoCount")
    if raw is None:
        return None
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def channel_total_videos(
    channel_url: str,
    *,
    fallback: Callable[[], int] | None = None,
    http_get: JsonGetter = _default_http_get,
) -> int:
    """Best-effort APPROXIMATE total public video count for a channel.

    With ``YTRAG_YOUTUBE_API_KEY`` set, asks the YouTube Data API v3 for the
    channel's ``statistics.videoCount``. With NO key (the default) — or if the
    lookup fails / can't resolve the channel — falls back to ``fallback()`` (the
    enumerated window count), so this ALWAYS returns a usable number with no key.
    Returns 0 only when there is no key AND no fallback.

    ``http_get`` is injectable so the with-key path is testable offline against a
    mocked API response.
    """
    key = youtube_api_key()
    if key:
        count = _video_count_from_api(channel_url, key, http_get)
        if count is not None:
            return count
    if fallback is not None:
        try:
            return int(fallback())
        except Exception:  # noqa: BLE001 - a bad fallback must not crash the caller
            return 0
    return 0
