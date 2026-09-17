"""Central, idempotent logging configuration for the whole app.

One ``StreamHandler`` is attached to the ``ytrag`` namespace logger and writes
structured, timestamped lines to **stdout** so they show up in ``docker logs`` /
under uvicorn right next to (but visually distinct from) uvicorn's own HTTP
access lines. The level comes from ``YTRAG_LOG_LEVEL`` (default ``INFO``; set
``DEBUG`` to see per-video fetch/throttle detail).

Why the ``ytrag`` namespace and not the root logger:

* Every app module already does ``logging.getLogger(__name__)`` — under the
  ``ytrag`` package those are children of the ``ytrag`` logger, so a single
  handler here lights all of them up.
* We set ``propagate = False`` so app records never bubble to the root logger.
  Uvicorn configures its own ``uvicorn`` / ``uvicorn.access`` loggers (also
  non-propagating), so app logs and access logs stay separate and **nothing is
  double-configured or duplicated**.

``setup_logging`` is idempotent: calling it twice (repeated imports, uvicorn
reload, test re-entry) never stacks a second handler — it just re-applies the
requested level.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from urllib.parse import urlsplit

APP_LOGGER_NAME = "ytrag"
_DEFAULT_LEVEL_NAME = "INFO"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Marker set on both the logger and the handler we own, so a second call is a
# cheap no-op and we only ever remove/replace the handler we installed (never
# one some host — uvicorn, pytest's caplog — attached).
_OWNED = "_ytrag_logging_owned"


def resolve_level(raw: str | None = None) -> int:
    """Map ``YTRAG_LOG_LEVEL`` (or an explicit value) to a ``logging`` level int.

    Accepts a level name ("DEBUG", "info", ...) or a numeric string. Anything
    blank/unknown falls back to ``INFO`` so a typo can never silence the app.
    """
    if raw is None:
        raw = os.getenv("YTRAG_LOG_LEVEL", "")
    name = str(raw).strip().upper()
    if not name:
        name = _DEFAULT_LEVEL_NAME
    if name.isdigit():
        return int(name)
    level = getattr(logging, name, None)
    return level if isinstance(level, int) else logging.INFO


def setup_logging(level: str | int | None = None, *, force: bool = False) -> logging.Logger:
    """Configure the ``ytrag`` logger once; return it. Safe to call repeatedly.

    ``level`` overrides ``YTRAG_LOG_LEVEL`` when given. On a repeat call the
    existing handler is kept (no duplicate lines) and only the level is
    re-applied, unless ``force`` rebuilds the handler from scratch.
    """
    logger = logging.getLogger(APP_LOGGER_NAME)
    resolved = resolve_level(level)

    already = getattr(logger, _OWNED, False)
    if already and not force:
        logger.setLevel(resolved)
        return logger

    # Drop only the handler(s) we previously installed (idempotent under force).
    for h in list(logger.handlers):
        if getattr(h, _OWNED, False):
            logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    setattr(handler, _OWNED, True)
    logger.addHandler(handler)
    logger.setLevel(resolved)
    # Namespaced: do not bubble to root (avoids duplicate lines if a dependency
    # configured the root logger, and keeps app logs distinct from access logs).
    logger.propagate = False
    setattr(logger, _OWNED, True)
    return logger


def mask_proxy(proxy: str | None) -> str:
    """Describe a proxy URL for logs WITHOUT leaking credentials or the full URL.

    Returns ``"disabled"`` for an empty value, otherwise ``"enabled (host=…)"``
    — never the ``user:pass@`` userinfo, path, or query. Falls back to a bare
    ``"enabled"`` when no host can be parsed.
    """
    p = (proxy or "").strip()
    if not p:
        return "disabled"
    try:
        host = urlsplit(p if "://" in p else "//" + p).hostname or ""
    except ValueError:
        host = ""
    return f"enabled (host={host})" if host else "enabled"


def proxy_diagnostic(proxy: str | None) -> str:
    """Return safe, useful proxy connection details for operational logs.

    Credentials, paths, query strings, and a sticky-session identifier are never
    included.  ``session`` describes only the routing mode: ``per-video-sticky``
    when ``YTRAG_TRANSCRIPT_PROXY_STICKY_FORMAT`` pins each video to its own exit
    IP, ``sticky`` when the base username carries a ``-session-`` marker, else
    ``rotating``.
    """
    p = (proxy or "").strip()
    if not p:
        return "disabled"
    try:
        parsed = urlsplit(p if "://" in p else "//" + p)
        host = parsed.hostname or ""
        port = parsed.port
        scheme = parsed.scheme or "unknown"
        username = parsed.username or ""
    except ValueError:
        return "enabled (unparseable)"

    # Residential gateways conventionally encode sticky sessions in the username.
    # Deliberately report only the mode, never the account or session token.
    #
    # Three routing modes, in priority order:
    #   * "per-video-sticky" — YTRAG_TRANSCRIPT_PROXY_STICKY_FORMAT is set to a
    #     "{s}" template, so ingest pins each video to its own exit IP at download
    #     time (see ingest._apply_sticky_session). The BASE username still ends in
    #     "-rotate" here, so the marker heuristic below would mislabel it
    #     "rotating" — check the env first so the log reflects the real mechanism.
    #   * "sticky" — the base username itself carries a "-session-" marker.
    #   * "rotating" — a fresh exit IP per request.
    sticky_fmt = os.getenv("YTRAG_TRANSCRIPT_PROXY_STICKY_FORMAT", "").strip()
    if sticky_fmt and "{s}" in sticky_fmt:
        session_mode = "per-video-sticky"
    elif re.search(r"(?:^|[-_])session(?:[-_]|$)", username, re.I):
        session_mode = "sticky"
    else:
        session_mode = "rotating"
    details = [f"scheme={scheme}"]
    if host:
        details.append(f"host={host}")
    if port is not None:
        details.append(f"port={port}")
    details.append(f"session={session_mode}")
    return "enabled (" + ", ".join(details) + ")"
