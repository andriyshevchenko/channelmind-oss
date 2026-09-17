"""FastAPI web UI for a multi-tenant "My Bots" RAG app.

Each user signs in with Google and manages their own bots. A bot owns an isolated
corpus and can ingest several YouTube channels + documents, be chatted with in the
browser, and be exposed as its own Telegram bot via a pasted token.

The transcript-ingest endpoint (``POST /api/bots/{id}/transcripts``) is the seam
that keeps the backend ready for a future browser extension: it accepts
already-extracted transcript text, exactly what the extension will send, while the
server-side yt-dlp fetcher writes the identical shape. No backend change is needed
when the thick client ships. Like the other content-adding endpoints it enforces
the server-side rights attestation (an explicit ``consent``/``attest`` flag, logged
to the audit trail) before creating a source, so the extension seam can't bypass it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import secrets
import threading
import time
from pathlib import Path

# Python's default mimetypes table has no entry for woff2 (and often not woff),
# so StaticFiles would serve the vendored fonts as ``application/octet-stream``.
# Register the correct types once, at import, so ``font/woff2`` is sent — cleaner
# for proxies/caches and unambiguous under ``X-Content-Type-Options: nosniff``.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import Headers, MutableHeaders

from .. import audit, autosync, bot_service, oauth, quotas, telegram_bot
from ..accounts import User, UserStore
from ..tenants import TenantStore
from ..bot import interview_persona
from ..bots import TYPE_DOCUMENT, TYPE_YOUTUBE
from ..config import env_startup_warning, is_production, load_config, test_guest_daily_cap
from ..logging_setup import mask_proxy, setup_logging
from ..corpora import CorpusStore
from ..metrics import MetricsStore
from ..documents import is_supported
from ..plans import DEFAULT_PLAN, DEVELOPER_PLAN, PLANS, plan_for
from ..rag import RETRIEVAL_TOP_K, _deep_link
from ..settings_store import FIELDS, masked_settings, save_settings
from ..user_settings import (
    CHAT_PROVIDERS,
    EMBED_PROVIDERS,
    KEY_PROVIDERS,
    MODE_BYOK,
    MODE_MANAGED,
    TRANSCRIPTION_PROVIDERS,
    VISION_PROVIDERS,
    KeystoreUnavailable,
    UserSettingsStore,
    resolve_chat_for_user,
)
from ..share_limits import sanitize_guest_history
from ..share_limits import limiter as share_limiter
from ..sources import (
    normalize_document_key,
    normalize_playlist_key,
    normalize_video_key,
    normalize_youtube_key,
)
from ..usage import UsageStore
from ..ingest_jobs import (
    JobStore,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from ..ingest_queue import manager as ingest_manager
from .jobs import manager

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

# The buildless Vue 3 SPA (vendored from the prototype). Its assets are served
# under /spa/* and its production shell at "/" (see _spa_index_production).
SPA_DIR = BASE / "spa"

# Configure the ``ytrag`` namespace logger once, at import — this module IS the
# ASGI entrypoint uvicorn loads (``uvicorn ytrag.web.app:app``), so wiring it here
# guarantees app logs stream to stdout in prod alongside uvicorn's access lines
# without double-configuring either. Idempotent; honours ``YTRAG_LOG_LEVEL``.
setup_logging()
logger = logging.getLogger("ytrag.web")


# ---- M: cache-control for static assets -------------------------------
class CachedStaticFiles(StaticFiles):
    """``StaticFiles`` that stamps a deploy-appropriate ``Cache-Control`` (Phase M).

    The build is BUILDLESS — there is no asset-hashing step to lean on — so caching
    is keyed on which directory an asset lives in, not a content hash:

      * ``vendor/*`` (pinned Vue / vue-router builds) and ``fonts/*`` (content-stable
        woff2) only ever change together with their filename, so they are cached
        hard and ``immutable`` for a year — the browser skips revalidation entirely.
      * everything else — the app's own UNHASHED JS/CSS/views and the ``/spa/`` shell
        — MUST revalidate on every load; otherwise a redeploy leaves a client pinned
        to stale JS talking to a changed API. ``no-cache`` (store, but always
        revalidate via ETag/Last-Modified) gives exactly that without re-downloading
        when nothing changed.

    Only response headers are touched, so the ``/spa`` design-render mount and the
    conditional-request (304) handling both keep working unchanged.
    """

    @staticmethod
    def _cache_control_for(rel_path: str) -> str:
        p = rel_path.replace("\\", "/").lstrip("/").lower()
        if p.startswith("vendor/") or p.startswith("fonts/"):
            return "public, max-age=31536000, immutable"
        return "no-cache"

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        # ``path`` is the resolved relative asset path (e.g. "vendor/vue…js",
        # "fonts/…woff2", "styles.css", or "." → index.html). Stamp on success and
        # on 304s; a 404/405 response is harmless to tag too.
        response.headers["Cache-Control"] = self._cache_control_for(path)
        return response


app = FastAPI(title="yt-channel-rag")
app.mount("/static", CachedStaticFiles(directory=str(BASE / "static")), name="static")
# SPA static tree. The mount ALWAYS serves the asset files under /spa/* (vendored
# JS, styles, fonts, views) — the production shell at "/" references them by
# absolute /spa/* URLs, so this mount is load-bearing in prod too. Only the
# html=True DIRECTORY-INDEX behavior (auto-serving spa/index.html at /spa/) is
# development-only: in production it would serve the preview shell as a second,
# CSP-incompatible entry point, so we gate it off. Assets are unaffected either way.
app.mount(
    "/spa",
    CachedStaticFiles(directory=str(SPA_DIR), html=not is_production()),
    name="spa",
)
oauth.install(app)


# ---- SPA production shell ---------------------------------------------
# The repo's spa/index.html keeps <meta cm-mode="preview"> and relative "./" asset
# refs so it renders straight from disk (file://) or via the /spa/ mount for design
# review. Production must (a) boot in real-API mode and (b) load assets from /spa/*
# even though the shell is served at "/". We derive that variant once, in memory.
_spa_index_html: str | None = None

# A sentinel baked into the cached production shell in place of the real CSP nonce.
# ``spa_root`` ``.replace()``s it with a fresh per-request nonce, so we never rebuild
# the whole shell per request — only substitute this token. Chosen so it can't
# collide with any real shell content (asset URL, importmap key, or bootstrap code).
_NONCE_PLACEHOLDER = "__CM_CSP_NONCE__"


def _spa_index_production() -> str:
    global _spa_index_html
    if _spa_index_html is None:
        raw = (SPA_DIR / "index.html").read_text(encoding="utf-8")
        # Boot the real API + Google login instead of the mock adapter.
        raw = raw.replace('content="preview"', 'content="production"')
        # No-login self-host mode: when dev-login is enabled (YTRAG_DEV_LOGIN), tell
        # the SPA so a fresh visitor is auto-signed-in as the local dev user (via
        # /auth/dev) with NO Google wall. Off by default, so a real OAuth deployment
        # is unaffected. dev_login_enabled() is env-constant, so baking it into the
        # once-built cached shell is correct.
        if oauth.dev_login_enabled():
            raw = raw.replace(
                '<meta name="cm-mode" content="production">',
                '<meta name="cm-mode" content="production">\n'
                '<meta name="cm-dev-login" content="1">',
            )
        # Rewrite the shell's own relative asset refs (link href, importmap, and the
        # entry module imports) to absolute /spa/* URLs the mount serves. Only the
        # shell uses "./" refs; the chained module imports resolve relative to /spa/.
        raw = raw.replace('"./', '"/spa/').replace("'./", "'/spa/")
        # Give BOTH inline <script> blocks (the importmap and the module bootstrap) a
        # CSP nonce slot. The placeholder is substituted with this response's real
        # nonce in spa_root so the served scripts match the response's CSP header.
        raw = raw.replace(
            '<script type="importmap">',
            f'<script type="importmap" nonce="{_NONCE_PLACEHOLDER}">',
        )
        raw = raw.replace(
            '<script type="module">',
            f'<script type="module" nonce="{_NONCE_PLACEHOLDER}">',
        )
        _spa_index_html = raw
    return _spa_index_html


# ---- M: production security headers + Content-Security-Policy ----------
# CSP-relevant note on the shell: the SPA has exactly two INLINE <script> blocks —
# the importmap and the module bootstrap. Rather than allow all inline script
# (``'unsafe-inline'``), we allow them with a per-request NONCE: a fresh random
# token is minted per response, written into both inline <script> tags AND the
# response's own ``script-src``. A nonce (not a sha256 hash) is the spec mechanism
# that browsers honor robustly for an inline ``<script type="importmap">`` — a hash
# on an importmap is not reliably accepted, and hash-allowing modules would demand
# per-module integrity/modulepreload that breaks the SPA's on-demand view loading.
def _new_csp_nonce() -> str:
    """A fresh, cryptographically-random per-response CSP nonce (128-bit, base64url)."""
    return secrets.token_urlsafe(16)


def _content_security_policy(nonce: str) -> str:
    """The app's Content-Security-Policy for a SINGLE response, bound to ``nonce``.

    Built per request (not cached) because ``script-src`` carries this response's
    unique ``'nonce-…'`` — the same nonce written into the served shell's two inline
    <script> tags, so header and body always agree.

    Two intentional relaxations, both FORCED by the buildless runtime-compiled Vue
    SPA — not laziness — and each contained so it doesn't reopen XSS:

      * ``script-src`` includes ``'unsafe-eval'``: the vendored full Vue build
        compiles every component's ``template`` string into a render function via
        ``new Function`` at runtime. With no build step to pre-compile templates
        there is no way around it. It is contained: inline script is allowed ONLY by
        the per-request nonce (NOT ``'unsafe-inline'``), so an injected <script>
        element — which can't guess the nonce — is still blocked, and user text
        reaches the DOM only through Vue's escaped ``{{ }}`` interpolation.
      * ``style-src`` includes ``'unsafe-inline'``: the design uses hundreds of
        static ``style="…"`` and ``:style`` bindings across the views. Vue applies
        most via the CSSOM (already allowed by ``'self'``), but permitting inline
        styles removes any browser-specific risk of the look breaking — the hard
        "do not change the design" constraint — at low marginal risk.

    ``script-src`` KEEPS ``'self'`` (and deliberately omits ``'strict-dynamic'``):
    the vendored Vue/router and every hash-routed view are same-origin ES modules
    loaded by bare-specifier import, and must stay allowed by ``'self'``;
    ``'strict-dynamic'`` would make the browser ignore ``'self'`` and block them.

    ``img-src`` allows ``https:`` for Google account avatars; ``connect-src 'self'``
    because the SPA only calls its own ``/api``; ``font-src 'self'`` since fonts are
    now vendored. HSTS is deliberately NOT emitted here — the TLS/proxy layer
    (Caddy) owns ``Strict-Transport-Security``.
    """
    return "; ".join([
        "default-src 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        f"script-src 'self' 'unsafe-eval' 'nonce-{nonce}'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: https:",
        "font-src 'self'",
        "connect-src 'self'",
    ])


# Static hardening headers sent alongside the CSP in production. (X-Frame-Options is
# belt-and-suspenders next to the CSP ``frame-ancestors 'none'`` for old browsers.)
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
}


class SecurityHeadersMiddleware:
    """Attach the CSP + hardening headers to every response, in production only.

    Pure ASGI (like the other guards) so it forwards ``receive`` untouched and never
    wraps it in a task group — the outer body-cap abort still propagates cleanly. It
    only ADDS response headers on the ``http.response.start`` message; it never
    blocks, buffers, or rewrites a body, so it cannot break the SPA boot, dev-login,
    the ``/spa`` design-render, or guest share pages. Gated on ``is_production()`` at
    request time so dev keeps headers off (frictionless mock/design render) and a
    test can flip the env var without re-importing the app.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not is_production():
            await self.app(scope, receive, send)
            return
        # Mint ONE nonce for this response, stash it on the scope so the route
        # (spa_root) injects the SAME value into the served shell's inline <script>
        # tags, and bind the CSP header to it. Header and body therefore always
        # carry an identical nonce for this exact response.
        nonce = _new_csp_nonce()
        scope["csp_nonce"] = nonce
        csp = _content_security_policy(nonce)

        async def _send(message):
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Content-Security-Policy"] = csp
                for key, value in _SECURITY_HEADERS.items():
                    headers[key] = value
            await send(message)

        await self.app(scope, receive, _send)


# ---- E1: CSRF defense-in-depth (Origin/Referer check) -----------------
# The app is cookie-session + JSON/multipart POSTs. ``SameSite=Lax`` on the
# session cookie already stops a cross-site page from sending the cookie on a
# state-changing request, so cross-site POST/PUT/PATCH/DELETE arrive
# unauthenticated and bounce at ``require_user``. This middleware is a second,
# server-side lock that does not depend on the browser honouring SameSite: for
# state-changing methods it rejects any request whose ``Origin`` (or, failing
# that, ``Referer``) host is not our own host.
#
# Deliberately permissive in two ways so it never breaks a legitimate flow:
#   * a request with NO Origin AND no Referer is allowed — a browser always
#     sends Origin on a cross-site state-changing request, so its absence means a
#     non-browser client (curl, the future extension, health checks), which
#     carries no ambient session cookie to forge in the first place;
#   * the cookieless guest endpoint (``/api/s/{token}/chat``) is exempt — the
#     unguessable token IS the credential, there is no cookie to ride, so CSRF is
#     meaningless there and a friend may open the link from anywhere.
_CSRF_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _request_hostname_from_headers(headers) -> str:
    """Our own canonical hostname for the CSRF Origin comparison.

    Prefers the operator-configured canonical host (``cfg.app_domain`` — from
    ``YTRAG_APP_DOMAIN``/``APP_DOMAIN`` or derived from the OAuth redirect URL).
    That value is server-controlled, so a request carrying a spoofed
    ``X-Forwarded-Host`` can't make an attacker's Origin look same-origin. Only
    when no canonical host is configured (dev) do we fall back to the
    client-suppliable Host/X-Forwarded-Host header — the previous behavior.
    """
    try:
        canonical = load_config().app_domain
    except Exception:  # noqa: BLE001 - a broken config must not break the guard
        canonical = ""
    if canonical:
        return canonical
    host = headers.get("x-forwarded-host") or headers.get("host") or ""
    host = host.split(",")[0].strip()
    return host.rsplit(":", 1)[0].lower() if host else ""


def _request_hostname(request: Request) -> str:
    """Thin ``Request`` wrapper around :func:`_request_hostname_from_headers`."""
    return _request_hostname_from_headers(request.headers)


def _origin_hostname(value: str) -> str:
    from urllib.parse import urlsplit

    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


class CSRFOriginGuardMiddleware:
    """Pure-ASGI CSRF Origin/Referer guard (see the E1 rationale below).

    Deliberately a pure ASGI middleware, NOT a ``BaseHTTPMiddleware``: the latter
    reads/streams the request body inside an anyio task group, which would re-wrap
    the outer body-size guard's abort signal in a ``BaseExceptionGroup`` and break
    its clean 413. A pure ASGI middleware forwards ``receive`` untouched, so that
    abort propagates cleanly up to :class:`MaxBodySizeMiddleware`.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") == "http"
            and scope.get("method", "").upper() in _CSRF_METHODS
            and not scope.get("path", "").startswith("/api/s/")
        ):
            headers = Headers(scope=scope)
            source = headers.get("origin") or headers.get("referer")
            if source and _origin_hostname(source) != _request_hostname_from_headers(headers):
                response = JSONResponse(
                    {"error": "Cross-site request blocked."}, status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---- E-fix 1: hard request-body cap (outermost ASGI guard) ------------
# Without this, an UNAUTHENTICATED multi-GB POST to the multipart document
# endpoint is fully streamed into Starlette's parser (and spooled to disk) BEFORE
# FastAPI runs auth or our per-file Content-Length precheck — so disk-DoS was
# stopped only by Caddy's request_body cap and collapsed entirely when the app is
# hit directly (dev/no proxy). This middleware sits OUTERMOST — ahead of the
# session, CSRF, routing and the form parser — and enforces a hard whole-body
# ceiling for state-changing methods, independent of any reverse proxy.
_BODY_CAP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_DEFAULT_MAX_BODY_BYTES = 25 * 1024 * 1024


class _BodyTooLarge(BaseException):
    """Internal signal: the streamed request body crossed the hard cap.

    Inherits ``BaseException`` (not ``Exception``) on purpose: it is raised from
    inside the wrapped ``receive`` while the downstream form/JSON parser is reading
    the body. Frameworks (FastAPI/Starlette) catch ``Exception`` during body
    parsing and would turn it into a generic 400 before it could reach this
    middleware; a ``BaseException`` slips past those catch-alls so we can convert it
    into a precise 413 in ``MaxBodySizeMiddleware`` below.
    """


def _max_body_cap() -> int:
    """Config-driven whole-body ceiling in bytes (never raises)."""
    try:
        return load_config().max_body_bytes
    except Exception:  # noqa: BLE001 - a broken config must not disable the guard
        return _DEFAULT_MAX_BODY_BYTES


def _scope_content_length(scope) -> int | None:
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value.decode("latin-1").strip())
            except (ValueError, AttributeError):
                return None
    return None


def _is_body_too_large(exc: BaseException) -> bool:
    """True when ``exc`` is (or wholly wraps) our body-size abort signal.

    Our own guards are pure ASGI, so normally the abort arrives raw. But if any
    ``BaseHTTPMiddleware`` is ever added to the chain it reads the body inside an
    anyio task group, which re-wraps whatever ``receive`` raises in a
    ``BaseExceptionGroup``. Unwrap it so the abort is still recognized regardless.
    Only a group whose *every* leaf is ``_BodyTooLarge`` counts — a mixed group
    carries a real error too and must propagate untouched.
    """
    if isinstance(exc, _BodyTooLarge):
        return True
    if isinstance(exc, BaseExceptionGroup):
        _matched, rest = exc.split(_BodyTooLarge)
        return _matched is not None and rest is None
    return False


async def _send_413(send) -> None:
    body = b'{"error":"Request body too large."}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class MaxBodySizeMiddleware:
    """Reject an over-cap request body BEFORE it reaches routing/auth/parsing.

    Two layers, so neither a declared nor a spoofed size gets through:

      * a declared ``Content-Length`` over the cap → 413 immediately, before the
        body is even read;
      * otherwise ``receive`` is wrapped to count streamed bytes and raise the
        moment the running total exceeds the cap. Because the parser reads the
        body THROUGH this wrapper, the stream is cut off after at most ~cap bytes
        — the multi-GB blob is never fully read and never fully spooled to disk.

    Only state-changing methods are inspected; GET/HEAD/OPTIONS (and non-HTTP
    scopes like lifespan/websocket) pass straight through. ``max_bytes=None``
    (the default) reads the cap from config per request so it stays config-driven;
    a fixed value can be injected for tests.
    """

    def __init__(self, app, max_bytes: int | None = None):
        self.app = app
        self._max_bytes = max_bytes

    def _cap(self) -> int:
        return self._max_bytes if self._max_bytes is not None else _max_body_cap()

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method", "").upper() not in _BODY_CAP_METHODS:
            await self.app(scope, receive, send)
            return

        max_bytes = self._cap()

        declared = _scope_content_length(scope)
        if declared is not None and declared > max_bytes:
            await _send_413(send)
            return

        seen = 0
        started = False

        async def _guarded_receive():
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > max_bytes:
                    raise _BodyTooLarge()
            return message

        async def _tracking_send(message):
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, _guarded_receive, _tracking_send)
        except BaseException as exc:  # noqa: BLE001 - re-raised unless it's our abort
            # We cut the body off mid-parse. The abort may arrive raw, or wrapped in
            # a BaseExceptionGroup by an inner BaseHTTPMiddleware's task group — both
            # are recognized. Anything else propagates untouched. If the app hasn't
            # started a response (the normal case — parsing precedes the reply), send
            # our own 413; otherwise we can't override an in-flight response.
            if not _is_body_too_large(exc):
                raise
            if not started:
                await _send_413(send)


# Middleware registration order (add_middleware inserts at the FRONT, so the LAST
# call is OUTERMOST). Session is installed first by oauth.install(app); then the
# security-headers stamp; then the CSRF guard; then the body cap LAST so it stays
# OUTERMOST. Resulting request flow:
#   MaxBodySize → CSRFOriginGuard → SecurityHeaders → SessionMiddleware → routing
# The body cap MUST remain outermost so it runs before the form parser can spool a
# flood (and so no BaseHTTPMiddleware ever wraps its receive). SecurityHeaders sits
# just inside it: it still tags every NORMAL response — the SPA shell, static
# assets, API, error pages all flow out through it from routing — while leaving the
# body-cap/CSRF rejection responses (rare error paths that need no CSP) untouched.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CSRFOriginGuardMiddleware)
app.add_middleware(MaxBodySizeMiddleware)


# ---- E2: authed-chat input clamps -------------------------------------
# Bound what an authenticated chat request can push into the owner-billed prompt
# (mirrors the guest-path caps in ``share_limits``). A too-large payload is a
# token-burn/DoS lever even for a logged-in user.
AUTHED_MESSAGE_MAX_CHARS = 8000
AUTHED_HISTORY_MAX_TURNS = 20
AUTHED_HISTORY_MAX_CHARS = 24000
CHAT_TOP_K_MIN = 1
CHAT_TOP_K_MAX = 20
# Wider default retrieval breadth for cross-channel synthesis (BUG-022). Sourced from
# rag.RETRIEVAL_TOP_K so the web default tracks the single retrieval-breadth knob.
CHAT_TOP_K_DEFAULT = RETRIEVAL_TOP_K


def _clamp_top_k(value: object) -> int:
    """Coerce a client-supplied ``top_k`` into ``[CHAT_TOP_K_MIN, CHAT_TOP_K_MAX]``."""
    try:
        k = int(value)
    except (TypeError, ValueError):
        return CHAT_TOP_K_DEFAULT
    return max(CHAT_TOP_K_MIN, min(CHAT_TOP_K_MAX, k))


_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


async def _read_upload_capped(file: UploadFile, max_bytes: int) -> bytes | None:
    """Read an upload into memory but never more than ``max_bytes``.

    Reads in fixed chunks and aborts the moment the accumulated size would exceed
    the cap, so a client that lies about (or omits) Content-Length still cannot
    make us buffer an unbounded blob. Returns the bytes on success, or ``None`` if
    the stream exceeds the cap (the caller turns that into a 413).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@app.on_event("startup")
def _startup():
    # One clear "who am I" line at boot: env, effective log level, and whether a
    # transcript proxy is wired (MASKED — never the URL/creds).
    try:
        cfg = load_config()
        logger.info(
            "startup: env=%s log_level=%s transcript_proxy=%s data_dir=%s",
            "production" if is_production() else "development",
            logging.getLevelName(logging.getLogger("ytrag").getEffectiveLevel()),
            mask_proxy(cfg.transcript_proxy),
            cfg.data_dir,
        )
    except Exception:  # noqa: BLE001 - a boot banner must never block startup
        logger.warning("startup: could not load config for boot banner", exc_info=True)
    # Fail-loud when YTRAG_ENV is ambiguous: an unset env silently defaults to
    # development and would honor YTRAG_TEST_MODE (offline fakes). Don't block boot.
    _env_warning = env_startup_warning()
    if _env_warning:
        logger.warning("startup: %s", _env_warning)
    try:
        telegram_bot.manager.start_all()
    except Exception:  # noqa: BLE001 - never block app startup on Telegram
        logger.exception("startup: telegram bots failed to start")
    try:
        ingest_manager.resume_pending(load_config())
    except Exception:  # noqa: BLE001 - never block app startup on the ingest queue
        logger.exception("startup: ingest queue resume_pending failed")
    try:
        manager.resume_pending_rebuilds(load_config())
    except Exception:  # noqa: BLE001 - never block app startup on the rebuild resume
        logger.exception("startup: rebuild-job resume failed")


# One process-wide auto-sync scheduler (Phase K). It only DECIDES what to enqueue;
# all real download work runs through the single FIFO ingest worker above.
autosync_scheduler = autosync.AutoSyncScheduler(load_config)


@app.on_event("startup")
async def _startup_autosync():
    try:
        autosync_scheduler.start()
        logger.info("startup: auto-sync scheduler started (tick=%ss)",
                    autosync.TICK_SECONDS)
    except Exception:  # noqa: BLE001 - never block startup on the scheduler
        logger.exception("startup: auto-sync scheduler failed to start")


@app.on_event("shutdown")
async def _shutdown_autosync():
    try:
        await autosync_scheduler.stop()
    except Exception:  # noqa: BLE001 - best-effort clean cancel on shutdown
        pass


# ---- reliability: health probe + generic error handling (Phase D2/D3) -
def _app_version() -> str:
    """Build/version string for the health probe (best-effort, never raises)."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("yt-channel-rag")
    except Exception:  # noqa: BLE001 - fall back to the FastAPI-declared version
        return app.version


def _health_snapshot(cfg) -> tuple[dict, bool]:
    """Cheap liveness/readiness probe. Returns ``(payload, ok)``.

    Two critical checks: the JSON stores / data dir is writable (write + delete a
    tiny temp file) and Chroma opens on its configured dir. The payload exposes
    ONLY booleans + build info — never config values, paths, or secrets.

    This does the REAL I/O every call; callers on the request path should go
    through ``_cached_health_snapshot`` so unauthenticated bursts don't repeat it.
    """
    checks: dict[str, bool] = {}
    # A *successful write* is the writability signal. Cleanup (unlink) is
    # best-effort in its own try: a failed unlink must neither flip the result
    # to False (spurious 503) nor leave the probe file accumulating on disk.
    probe = None
    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        probe = cfg.data_dir / f".healthz-{secrets.token_hex(4)}"
        probe.write_text("ok", encoding="utf-8")
        checks["data_dir_writable"] = True
    except Exception:  # noqa: BLE001 - a failed write IS the signal
        checks["data_dir_writable"] = False
    if probe is not None:
        try:
            probe.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - cleanup failure must not flip the result
            pass
    try:
        import chromadb

        chromadb.PersistentClient(path=str(cfg.chroma_dir)).list_collections()
        checks["chroma"] = True
    except Exception:  # noqa: BLE001
        checks["chroma"] = False
    ok = all(checks.values())
    payload = {
        "status": "ok" if ok else "degraded",
        "service": "channelmind",
        "version": _app_version(),
        "checks": checks,
    }
    return payload, ok


# The probe writes to disk and opens a fresh Chroma client on every call — too
# expensive to run per request on an UNAUTHENTICATED endpoint (DoS angle; cost
# also grows with the Chroma collection count). Cache the result for a short TTL
# so a burst of liveness checks collapses to a single real probe. Keyed on the
# config dirs as well as the clock: if the data/chroma dir changes the key
# changes and we re-probe immediately (correctness) instead of serving a stale
# snapshot. ``time.monotonic()`` — never wall-clock — so clock changes can't
# freeze or expire the cache incorrectly.
_HEALTH_TTL_SECONDS = 5.0
_health_lock = threading.Lock()
_health_cache: tuple | None = None  # (key, monotonic_ts, (payload, ok))


def _cached_health_snapshot(cfg) -> tuple[dict, bool]:
    """TTL-cached wrapper around :func:`_health_snapshot`.

    Within ``_HEALTH_TTL_SECONDS`` of the last real probe *for the same config
    dirs*, returns the cached ``(payload, ok)`` without touching disk or Chroma.
    """
    global _health_cache
    key = (str(cfg.data_dir), str(cfg.chroma_dir))
    now = time.monotonic()
    with _health_lock:
        cached = _health_cache
    if cached is not None:
        ckey, ts, result = cached
        if ckey == key and (now - ts) < _HEALTH_TTL_SECONDS:
            return result
    payload, ok = _health_snapshot(cfg)
    with _health_lock:
        _health_cache = (key, time.monotonic(), (payload, ok))
    return payload, ok


@app.get("/healthz")
def healthz():
    """Unauthenticated liveness/readiness probe (external monitor + Caddy).

    200 when the data dir is writable and Chroma opens; 503 when a critical check
    fails. Deliberately outside ``require_user`` and secret-free so a monitor and
    the reverse proxy can reach it without credentials.
    """
    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001 - a broken config is itself a degraded state
        logger.exception("healthz: load_config failed")
        return JSONResponse(
            {"status": "degraded", "service": "channelmind",
             "version": _app_version(), "checks": {"config": False}},
            status_code=503,
        )
    payload, ok = _cached_health_snapshot(cfg)
    return JSONResponse(payload, status_code=200 if ok else 503)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for UNHANDLED (non-HTTP) exceptions — stop leaking internals.

    The full traceback is logged server-side under a short generated request id;
    the client only ever sees a generic message + that id. ``/api/*`` callers get
    JSON, everything else gets the branded ``error.html`` 500 page. HTTPExceptions
    (auth redirects, 404s, explicit 4xx) keep Starlette's own handling — this only
    fires for genuine bugs/provider failures that would otherwise 500 with a stack.
    """
    request_id = secrets.token_hex(4)
    logger.exception("unhandled error id=%s %s %s", request_id,
                     request.method, request.url.path)
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            {"error": "Something went wrong", "request_id": request_id},
            status_code=500,
        )
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "code": 500,
            "request_id": request_id,
            "message": "An unexpected error occurred on our end. Please try "
                       f"again in a moment. (id: {request_id})",
        },
        status_code=500,
    )


@app.exception_handler(oauth.APIUnauthorized)
async def _api_unauthorized_handler(request: Request, exc: oauth.APIUnauthorized):
    """Unauthenticated /api/* → 401 JSON so the SPA can route to its login view."""
    return JSONResponse({"error": "unauthenticated"}, status_code=401)


def require_user(request: Request) -> User:
    return oauth.login_required(request)


def require_developer(user: User = Depends(require_user)) -> User:
    """Gate operator/admin-only routes to the DEVELOPER plan.

    Reuses the existing plan registry (``plans.py``) as the admin concept — no
    separate is-admin flag. The operator and devs resolve to ``DEVELOPER`` (via
    ``YTRAG_DEVELOPER_EMAILS`` or a stored plan name); every other authenticated
    user is ``BETA_USER`` and gets a 403. Applied to routes that write the
    SERVER-GLOBAL config (``settings_store`` FIELDS: provider api keys, llm/embed
    models+providers, transcript_proxy, oauth client id/secret, session_secret),
    which any tenant could otherwise overwrite.
    """
    if plan_for(user).name != DEVELOPER_PLAN:
        raise HTTPException(status_code=403, detail="developer access required")
    return user


# ---- pages ------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    error = request.query_params.get("error")
    return templates.TemplateResponse(
        request,
        "login.html",
        {"dev_login": oauth.dev_login_enabled(), "error": error},
    )


@app.get("/", response_class=HTMLResponse)
def spa_root(request: Request):
    """Serve the Vue SPA shell (production mode) at the app root.

    Auth is now handled client-side: the shell boots, calls /api/me, and on the
    401-JSON (login_required) routes itself to #/login. The hash router serves all
    app views from this one document, so no per-route server rewrites are needed.
    The old Jinja page routes (/bots/{id}, /settings, /account, /about, /terms,
    /privacy, /p/{id}) now 302-redirect to their SPA hash routes (Phase L) so old
    bookmarks don't 404; their templates were moved to templates/legacy/. Only
    error.html (exception handler + /s/{token} 404) and login.html (OAuth entry)
    remain live Jinja templates.

    The two inline <script> tags carry a per-request CSP nonce. In production
    SecurityHeadersMiddleware minted it, stashed it on the scope, and put the SAME
    nonce in this response's ``script-src`` — we substitute it into the cached shell
    here. In dev the middleware is off (no CSP at all), so a throwaway nonce is used;
    without a CSP the attribute is inert. The nonce'd shell is ``no-cache`` so a
    response minted for one visitor's nonce can never be reused for another.
    """
    nonce = request.scope.get("csp_nonce") or _new_csp_nonce()
    html = _spa_index_production().replace(_NONCE_PLACEHOLDER, nonce)
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-cache"},  # nonce'd shell must never be cached
    )


# ---- retired Jinja page routes (Phase L) ------------------------------
# These views moved into the Vue SPA (served at "/", hash-routed). The routes are
# kept only so old bookmarks 302-redirect to the matching hash route instead of
# 404ing; the SPA gates auth itself (boots, calls /api/me, bounces to #/login),
# so no Depends(require_user) here. Their templates live in templates/legacy/.
@app.get("/bots/{bot_id}")
def bot_detail_page(bot_id: str):
    """Retired → SPA #/bots/{id} (old bookmark compatibility)."""
    return RedirectResponse(url=f"/#/bots/{bot_id}", status_code=302)


@app.get("/settings")
def settings_page():
    """Retired → SPA #/settings (old bookmark compatibility)."""
    return RedirectResponse(url="/#/settings", status_code=302)


@app.get("/about")
def about_page():
    """Retired → SPA landing at #/about (old bookmark compatibility)."""
    return RedirectResponse(url="/#/about", status_code=302)


@app.get("/terms")
def terms_page():
    """Retired → SPA #/terms (old bookmark compatibility)."""
    return RedirectResponse(url="/#/terms", status_code=302)


@app.get("/privacy")
def privacy_page():
    """Retired → SPA #/privacy (old bookmark compatibility)."""
    return RedirectResponse(url="/#/privacy", status_code=302)


@app.get("/account")
def account_page():
    """Retired → SPA #/account (old bookmark compatibility)."""
    return RedirectResponse(url="/#/account", status_code=302)


@app.get("/p/{bot_id}")
def public_chat_page(bot_id: str):
    """Retired legacy public-chat shell.

    A bare bot id never granted access — sharing is private and token-based
    (``GET /s/{token}`` → SPA ``#/s/{token}``). Old ``/p/{id}`` links redirect to
    the SPA's private-share explanation so the visitor is told to ask the owner
    for a real share link rather than hitting a dead page.
    """
    return RedirectResponse(url="/#/error/private-share", status_code=302)


@app.get("/s/{token}")
def share_chat_page(token: str, request: Request):
    """Compatibility redirect for a private share link — no login required.

    The guest UI now lives in the SPA at the hash route ``/#/s/{token}`` (H4), so
    an old-format ``/s/{token}`` link is redirected there and the SPA takes over
    (it loads ``GET /api/s/{token}`` for the header + quota, then chats via
    ``POST /api/s/{token}/chat``). An unknown/revoked/rotated token resolves to
    nothing → a friendly 404 (never a redirect to a dead SPA page, and — matching
    the pre-SPA behavior — never an auth redirect).
    """
    cfg = load_config()
    bot = bot_service.bot_store(cfg).find_by_share_token(token)
    if bot is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "code": 404,
                "title": "Link not available",
                "message": "This share link is invalid or was turned off by its owner. "
                           "Ask them for a fresh link.",
            },
            status_code=404,
        )
    # Valid token → hand off to the SPA guest route. The fragment (#/s/…) keeps the
    # hash router in charge; the shell is served at "/" so the browser loads it and
    # PublicChat.js resolves the token against GET /api/s/{token}.
    return RedirectResponse(url=f"/#/s/{token}", status_code=302)


@app.get("/api/s/{token}")
def api_share_info(token: str, request: Request):
    """Guest-facing share-info (no login): what PublicChat.js renders before chat.

    Returns the bot name, an avatar hint, the one-line source summary, and the
    live guest quota (today's count vs the owner-plan cap). An unknown/revoked
    token is a clean 404 JSON — the SPA renders its "link no longer active" state
    from it — and NEVER an auth redirect (this route is deliberately outside
    ``require_user``; the token is the only credential a guest has)."""
    cfg = load_config()
    bot = bot_service.bot_store(cfg).find_by_share_token(token)
    if bot is None:
        return JSONResponse(
            {"error": "This share link is no longer active."}, status_code=404
        )
    owner = UserStore(cfg.data_dir).get(bot.owner_id)
    plan = plan_for(owner) if owner is not None else PLANS[DEFAULT_PLAN]
    cap = _effective_guest_cap(bot, plan)
    avatar = next(
        (s.channel_avatar for s in bot.sources if getattr(s, "channel_avatar", "")),
        "",
    )
    return {
        "active": True,
        "bot_name": bot.name,
        "avatar": avatar,
        "source_summary": _share_source_summary(bot),
        "quota": {
            "used_today": share_limiter.today_count(bot.id),
            "daily_cap": cap,
        },
    }


# ---- account / status ------------------------------------------------
@app.get("/api/me")
def api_me(user: User = Depends(require_user)):
    return {"user": {"id": user.id, "email": user.email, "name": user.name,
                     "picture": user.picture}}


@app.get("/api/me/plan")
def api_me_plan(user: User = Depends(require_user)):
    cfg = load_config()
    plan = plan_for(user)
    usage = quotas.usage_for(cfg, user.id)
    monthly_spend = UsageStore(cfg.data_dir).monthly_cost_usd(user.id)
    # ``plan`` (name string) and ``limits``/``usage`` keep their historical shape
    # so the existing plan widget (SPA Account/MyBots views) is unaffected; the
    # toggles/guest_caps/budget/price blocks below are purely additive.
    return {
        "plan": plan.name,
        "limits": asdict(plan.limits),
        "usage": {
            "bots_used": usage.bots,
            "sources_used": usage.sources,
            "active_jobs": usage.active_jobs,
        },
        "toggles": {
            "sharing_enabled": plan.sharing_enabled,
            "byok_enabled": plan.byok_enabled,
            "managed_enabled": plan.managed_enabled,
            "upgrades_enabled": plan.upgrades_enabled,  # billing/upgrade UI (OFF for beta)
        },
        "guest_caps": {
            "daily_message_cap": plan.guest_daily_message_cap,
            "min_interval_sec": plan.guest_min_interval_sec,
            "hourly_limit": plan.guest_hourly_limit,
            "distinct_guest_cap": plan.distinct_guest_cap,
            "message_max_chars": plan.guest_message_max_chars,
            "history_max_turns": plan.guest_history_max_turns,
            "history_max_chars": plan.guest_history_max_chars,
        },
        "budget": {
            "used_usd": round(monthly_spend, 6),
            "limit_usd": plan.max_monthly_cost_usd,  # None = unlimited
        },
        "price": plan.price,  # None during the beta
    }


@app.get("/api/me/usage")
def api_me_usage(user: User = Depends(require_user)):
    """The current user's LLM usage/cost aggregate (chat only; zeros if none)."""
    cfg = load_config()
    store = UsageStore(cfg.data_dir)
    rec = store.get(user.id) or {}
    by_model = []
    for model, bucket in (rec.get("by_model") or {}).items():
        by_model.append({
            "model": model,
            "prompt_tokens": bucket.get("prompt_tokens", 0),
            "completion_tokens": bucket.get("completion_tokens", 0),
            "cost_usd": bucket.get("cost_usd", 0.0),
            "calls": bucket.get("calls", 0),
        })
    by_model.sort(key=lambda m: m["cost_usd"], reverse=True)
    prompt_tokens = rec.get("prompt_tokens", 0)
    completion_tokens = rec.get("completion_tokens", 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": rec.get("total_cost_usd", 0.0),
        "monthly_cost_usd": store.monthly_cost_usd(user.id),
        "calls": rec.get("calls", 0),
        "by_model": by_model,
    }


@app.get("/api/me/settings")
def api_get_my_settings(user: User = Depends(require_user)):
    """This user's chat mode + BYOK key status (masked) plus the plan toggles.

    Never returns a raw key — only ``mode`` and a per-provider ``keys_set``
    yes/no, alongside ``byok_enabled``/``managed_enabled`` so the UI can hide the
    BYOK controls when the plan disables them.
    """
    cfg = load_config()
    plan = plan_for(user)
    return UserSettingsStore(cfg.data_dir).masked(user.id, plan)


@app.post("/api/me/settings")
async def api_save_my_settings(request: Request, user: User = Depends(require_user)):
    """Set this user's chat mode, per-capability provider choices, and/or BYOK key(s).

    Body: ``{"mode": "managed"|"byok", "llm_provider": <chat provider>,
    "embed_provider"/"transcription_provider"/"vision_provider": <provider>,
    "keys": {<provider>: <key>}}`` (all optional). Each per-capability provider is
    validated against its own capability set; a blank clears the choice (server
    default). Guards:

    * ``mode=byok`` is rejected when the plan has ``byok_enabled`` false —
      a user can't force BYOK onto a plan that disables it.
    * a blank key value clears that provider's stored key.
    * storing a key with no ``YTRAG_KEYSTORE_SECRET`` returns a clean error and
      writes NOTHING (never a plaintext key).

    Always responds with the masked view — raw keys never leave the server.
    """
    cfg = load_config()
    plan = plan_for(user)
    store = UserSettingsStore(cfg.data_dir)
    data = await request.json()
    if not isinstance(data, dict):
        return JSONResponse({"error": "invalid body"}, status_code=400)

    mode = data.get("mode")
    if mode is not None:
        if mode not in (MODE_MANAGED, MODE_BYOK):
            return JSONResponse({"error": "mode must be 'managed' or 'byok'"},
                                status_code=400)
        if mode == MODE_BYOK and not plan.byok_enabled:
            return JSONResponse(
                {"error": "Your plan does not allow bring-your-own-key mode."},
                status_code=400,
            )
        if mode == MODE_MANAGED and not plan.managed_enabled:
            return JSONResponse(
                {"error": "Your plan does not allow managed mode."},
                status_code=400,
            )
        store.set_mode(user.id, mode)

    provider = data.get("llm_provider")
    if provider is not None:
        provider = (provider or "").strip().lower()
        if provider and provider not in CHAT_PROVIDERS:
            return JSONResponse({"error": "unsupported chat provider"}, status_code=400)
        store.set_llm_provider(user.id, provider)

    # Per-capability BYOK provider CHOICES (embed pins new corpora; vision/
    # transcription swap provider+model+key at resolution). Each is validated
    # against its own capability set; a blank clears the choice (server default).
    for cap, valid in (
        ("embed", EMBED_PROVIDERS),
        ("transcription", TRANSCRIPTION_PROVIDERS),
        ("vision", VISION_PROVIDERS),
    ):
        raw = data.get(f"{cap}_provider")
        if raw is not None:
            raw = (raw or "").strip().lower()
            if raw and raw not in valid:
                return JSONResponse(
                    {"error": f"unsupported {cap} provider"}, status_code=400
                )
            store.set_capability_provider(user.id, cap, raw)

    # Default auto-sync frequency for NEW YouTube sources (Phase K). Must be an
    # explicit off|daily|weekly|monthly — blank/unknown is rejected rather than
    # silently coerced, so the UI can't quietly change the meaning of the default.
    sync_freq = data.get("sync_freq")
    if sync_freq is not None:
        if sync_freq not in autosync.SYNC_FREQS:
            return JSONResponse(
                {"error": "sync_freq must be one of off, daily, weekly, monthly"},
                status_code=400,
            )
        store.set_sync_freq(user.id, sync_freq)

    # User-facing notification-email toggle (backend for the 1.5 UI). Must be an
    # explicit boolean; absent leaves the stored value unchanged. Default (ON) is
    # applied at read time in user_settings, so legacy users keep getting emails.
    email_notifications = data.get("email_notifications")
    if email_notifications is not None:
        if not isinstance(email_notifications, bool):
            return JSONResponse(
                {"error": "email_notifications must be a boolean"},
                status_code=400,
            )
        store.set_email_notifications(user.id, email_notifications)

    keys = data.get("keys")
    if keys is not None:
        if not isinstance(keys, dict):
            return JSONResponse({"error": "keys must be an object"}, status_code=400)
        # Only the plan may bring keys at all (defense in depth: even if the UI is
        # shown, a plan without BYOK can't seed a key).
        if any((v or "").strip() for v in keys.values()) and not plan.byok_enabled:
            return JSONResponse(
                {"error": "Your plan does not allow bring-your-own-key mode."},
                status_code=400,
            )
        for p, v in keys.items():
            if p not in KEY_PROVIDERS:
                return JSONResponse({"error": f"unsupported provider: {p}"},
                                    status_code=400)
            try:
                store.set_key(user.id, p, v or "")
            except KeystoreUnavailable:
                return JSONResponse(
                    {"error": "Key storage isn't configured on this server "
                              "(YTRAG_KEYSTORE_SECRET). Your key was not saved."},
                    status_code=503,
                )
    return {"ok": True, "settings": store.masked(user.id, plan)}


@app.delete("/api/me")
def api_delete_me(request: Request, user: User = Depends(require_user)):
    """Self-serve account erasure (GDPR right to erasure) — replaces the stub.

    Full cascade for the current user, in a deliberate, best-effort order:

    1. For each owned bot: cancel its queued/running ingest jobs (so no worker
       races the delete), stop its Telegram poller, then ``bot_service.delete_bot``
       — the SAME path the per-bot delete uses, which drops the bot's Chroma
       collection + corpus record + on-disk transcript/document dirs. Reusing it is
       what guarantees no orphaned collection is left behind.
    2. A final defensive sweep re-drops each bot's Chroma collection + corpus dir
       in case a worker mid-rebuild re-created one after its own delete.
    3. Remove the user's remaining ingest-job records, usage aggregate, and tenant
       entitlement record.
    4. Delete the account record itself — the one hard requirement (while it
       exists the user can log back in, i.e. erasure did not happen). A failure
       here is the only 500, and it happens BEFORE the audit write so a failed
       delete never leaves a false "account_deleted" record.
    5. Write the ``account_deleted`` audit entry (best-effort: an audit-write
       failure must NOT block a data-subject erasure), then clear the session.

    Robustness: a per-bot/sub-step failure is collected and the cascade continues
    (a half-deletion that still lets the user log in is worse than a stray file).
    The collected errors are surfaced in the audit entry (``error_count`` + a
    truncated sample) so a partial failure is visible.
    """
    cfg = load_config()
    errors: list[str] = []
    bstore = bot_service.bot_store(cfg)
    job_store = JobStore(cfg.data_dir)

    bots = bstore.list_for(user.id)
    for bot in bots:
        try:
            for job in job_store.list_for_user(user.id):
                if job.bot_id == bot.id and job.status in (STATUS_QUEUED, STATUS_RUNNING):
                    job_store.request_cancel(job.job_id)
        except Exception as exc:  # noqa: BLE001 - best-effort, keep cascading
            errors.append(f"cancel-jobs bot={bot.id}: {exc}")
        try:
            telegram_bot.manager.stop(bot.id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stop-telegram bot={bot.id}: {exc}")
        try:
            bot_service.delete_bot(cfg, bot)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"delete-bot bot={bot.id}: {exc}")

    # Defensive final sweep: a worker mid-rebuild may have re-created a just-
    # dropped collection between our request_cancel and delete_bot. Re-drop each
    # deleted bot's collection + corpus dir so no orphan survives erasure.
    for bot in bots:
        try:
            bot_service.sweep_corpus(cfg, bot.corpus_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sweep-corpus bot={bot.id}: {exc}")

    try:
        job_store.delete_for_user(user.id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"delete-jobs: {exc}")
    try:
        UsageStore(cfg.data_dir).delete(user.id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"delete-usage: {exc}")
    try:
        TenantStore(cfg.data_dir).delete(user.id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"delete-tenant: {exc}")
    try:
        # Per-user chat settings + encrypted BYOK keys (Phase C2/C3) are tenant
        # data — erase them with the rest of the cascade.
        UserSettingsStore(cfg.data_dir).delete(user.id)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"delete-user-settings: {exc}")

    # The account record is the hard requirement: while it exists the user can log
    # back in, i.e. the erasure did not happen. A failure here is the only 500.
    # Do it BEFORE the audit write so a failed delete never leaves a false
    # "account_deleted" record (nor a double-log when the user retries).
    try:
        UserStore(cfg.data_dir).delete(user.id)
    except Exception:  # noqa: BLE001
        return JSONResponse(
            {"error": "We couldn't fully delete your account. Please try again."},
            status_code=500,
        )

    # Audit AFTER the record is truly gone, capturing the real outcome. Unlike the
    # fail-closed attestation write, erasure must never be blocked by an audit
    # failure — log-and-continue. Surface any collected sub-step errors so a
    # partial failure is visible.
    try:
        audit.record(
            cfg.data_dir,
            actor=user.id,
            action="account_deleted",
            target=f"user:{user.id}",
            outcome="deleted_with_errors" if errors else "deleted",
            ip=_client_ip(request),
            bots_deleted=len(bots),
            error_count=len(errors),
            errors=errors[:5],
        )
    except Exception:  # noqa: BLE001 - never block erasure on the audit log
        pass

    request.session.clear()
    return {"ok": True}


@app.get("/api/status")
def api_status(user: User = Depends(require_user)):
    cfg = load_config()
    return {
        "llm_provider": cfg.llm_provider,
        "llm_model": cfg.llm_model,
        "embed_provider": cfg.embed_provider,
        "embed_model": cfg.embed_model,
    }


@app.post("/api/settings")
async def api_save_settings(request: Request, user: User = Depends(require_developer)):
    data = await request.json()
    clean = {k: v for k, v in data.items() if k in FIELDS}
    save_settings(clean)
    return {"ok": True, "settings": masked_settings()}


# ---- bots -------------------------------------------------------------
def _bot_summary(cfg, bot) -> dict:
    corpus = CorpusStore(cfg.data_dir).get(bot.corpus_id)
    return {
        "id": bot.id,
        "name": bot.name,
        "description": bot.description,
        "persona": bot.persona,
        "custom_prompt": bot.custom_prompt,
        "language": bot.language,
        "suggest_followups": bot.suggest_followups,
        "include_shorts": bot.include_shorts,
        "telegram_username": bot.telegram_username,
        "source_count": len(bot.sources),
        "chunk_count": corpus.chunk_count if corpus else 0,
        "status": corpus.status if corpus else "empty",
    }


def _bot_detail(cfg, bot) -> dict:
    d = _bot_summary(cfg, bot)
    d["sources"] = [asdict(s) for s in bot.sources]
    return d


def _get_owned(cfg, bot_id: str, user: User):
    return bot_service.bot_store(cfg).owned(bot_id, user.id)


# ---- private share helpers -------------------------------------------
def _trusted_proxy_count() -> int:
    """How many trusted proxies sit in front of the app (default: 1, i.e. Caddy)."""
    try:
        return max(1, int(os.environ.get("YTRAG_TRUSTED_PROXIES", "1")))
    except (TypeError, ValueError):
        return 1


def _client_ip(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    ``X-Forwarded-For`` is client-controlled: a guest can PREPEND fake IPs to the
    left to dodge the per-client rate/distinct/daily caps. Our trusted proxy
    (Caddy, single replica) APPENDS the real peer IP to the RIGHT of whatever the
    client sent, so we read from the right — skipping ``YTRAG_TRUSTED_PROXIES``
    trusted hops — instead of trusting the spoofable leftmost value. Falls back to
    the socket peer when no XFF header is present.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            idx = min(_trusted_proxy_count(), len(parts))
            return parts[-idx]
    return request.client.host if request.client else "unknown"


def _record_attestation(cfg, request: Request, *, actor: str, target: str):
    """Fail-closed rights-attestation audit write.

    Writes the ``source_attestation`` audit entry BEFORE the source is added.
    Returns ``None`` on success; on an audit write failure returns a clean JSON
    503 the caller must return, so the source is never added without a durable
    record of the acceptance.
    """
    try:
        audit.record(
            cfg.data_dir,
            actor=actor,
            action="source_attestation",
            target=target,
            doc_version=audit.CONTENT_RIGHTS_ATTEST_VERSION,
            outcome="accepted",
            ip=_client_ip(request),
        )
    except audit.AuditWriteError:
        return JSONResponse(
            {"error": "Couldn't record your rights confirmation, please retry."},
            status_code=503,
        )
    return None


def _share_source_summary(bot) -> str:
    """One line describing what a shared bot is grounded in, for the guest page."""
    authors: list[str] = []
    for s in bot.sources:
        a = (s.author or "").strip()
        if a and a not in authors:
            authors.append(a)
    if not authors:
        return "Answers grounded in this bot's sources — every reply cites where it came from."
    shown = ", ".join(authors[:3])
    more = f" +{len(authors) - 3} more" if len(authors) > 3 else ""
    return f"Answers grounded in {shown}{more} — every reply cites its sources."


def _effective_guest_cap(bot, plan) -> int:
    """The per-bot guest daily message cap actually enforced and displayed.

    Starts from the OWNER plan's cap, applies the bot's own ``share_message_cap``
    override (a non-zero value wins, downward only), then — offline TEST MODE only
    — the fail-closed ``test_guest_daily_cap`` override so the e2e suite can reach
    the daily-cap 429 deterministically. The test override is inert in production
    (see :func:`ytrag.config.test_guest_daily_cap`), so real caps are never lowered.
    """
    cap = (
        min(bot.share_message_cap, plan.guest_daily_message_cap)
        if bot.share_message_cap
        else plan.guest_daily_message_cap
    )
    override = test_guest_daily_cap()
    if override is not None:
        cap = min(cap, override)
    return cap


def _share_info(request: Request, bot, owner: User) -> dict:
    """The share state the UI renders: active flag, link, and today's usage vs cap.

    The default daily cap comes from the bot OWNER's plan; a per-bot
    ``share_message_cap`` override (non-zero) still wins.
    """
    active = bool(bot.share_token) and not bot.share_revoked
    _plan = plan_for(owner)
    cap = _effective_guest_cap(bot, _plan)
    path = f"/s/{bot.share_token}" if active else ""
    url = (str(request.base_url).rstrip("/") + path) if active else ""
    return {
        "active": active,
        "path": path,
        "url": url,
        "created_at": bot.share_created_at or "",
        "daily_cap": cap,
        "today_count": share_limiter.today_count(bot.id) if active else 0,
    }


@app.get("/api/bots")
def api_list_bots(user: User = Depends(require_user)):
    cfg = load_config()
    bots = bot_service.bot_store(cfg).list_for(user.id)
    return {"bots": [_bot_summary(cfg, b) for b in bots]}


@app.post("/api/bots")
async def api_create_bot(request: Request, user: User = Depends(require_user)):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    description = (data.get("description") or "").strip()
    persona = (data.get("persona") or "").strip()
    language = (data.get("language") or "").strip()
    cfg = load_config()
    limit_error = quotas.check_can_create_bot(cfg, user)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)
    bot = bot_service.create_bot(cfg, user.id, name, persona, description, language)
    return {"bot": _bot_detail(cfg, bot)}


@app.get("/api/bots/{bot_id}")
def api_get_bot(bot_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"bot": _bot_detail(cfg, bot)}


@app.get("/api/bots/{bot_id}/stats")
def api_bot_stats(bot_id: str, user: User = Depends(require_user)):
    """Owner-scoped per-bot beta engagement metrics (per-day turns, active days,
    deflect rate). Ownership-checked exactly like the other bot endpoints; a bot
    with no chat turns yet returns the same shape with empty/zero values."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"stats": MetricsStore(cfg.data_dir).summary(bot.id)}


@app.patch("/api/bots/{bot_id}")
async def api_update_bot(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    # ``custom_prompt`` (Phase J): present with a non-empty string = save/update;
    # present as "" = clear (reverts chat to the builder persona). It's included in
    # the tuple so an empty string round-trips through update() (which only skips
    # None), letting the SPA clear a saved custom prompt.
    # Coerce to str and cap length so a malformed client (non-string, or a huge
    # prompt that would inflate every LLM call) can't persist a value that later
    # 500s at chat time or blows up token cost. Empty string still clears.
    _MAX_FIELD_CHARS = 8000
    fields = {
        k: str(data[k])[:_MAX_FIELD_CHARS]
        for k in ("name", "persona", "custom_prompt", "description", "language")
        if k in data and data[k] is not None
    }
    # ``suggest_followups`` (Fable feel-win #2) is a BOOL toggle, not a text field — keep
    # it OUT of the string-coercion map above (which would turn it into "True"/"False")
    # and coerce it to a real bool so the per-bot on/off switch round-trips cleanly.
    if "suggest_followups" in data and data["suggest_followups"] is not None:
        fields["suggest_followups"] = bool(data["suggest_followups"])
    # ``include_shorts`` (BUG-009) is likewise a BOOL toggle the 1.5 UI wires to a
    # per-bot switch — coerce to a real bool (default-off lives on the model, not here).
    if "include_shorts" in data and data["include_shorts"] is not None:
        fields["include_shorts"] = bool(data["include_shorts"])
    updated = bot_service.bot_store(cfg).update(bot_id, **fields)
    return {"bot": _bot_detail(cfg, updated)}


@app.delete("/api/bots/{bot_id}")
def api_delete_bot(bot_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    telegram_bot.manager.stop(bot_id)
    bot_service.delete_bot(cfg, bot)
    return {"ok": True}


@app.post("/api/bots/{bot_id}/persona/interview")
async def api_interview_persona(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    description = (data.get("description") or "").strip()
    if not description:
        return JSONResponse({"error": "description is required"}, status_code=400)
    answers = data.get("answers") or []
    if not isinstance(answers, list):
        return JSONResponse({"error": "answers must be a list"}, status_code=400)
    try:
        result = interview_persona(cfg, description, answers)
    except Exception:  # noqa: BLE001 - provider/internal detail stays in the logs
        request_id = secrets.token_hex(4)
        logger.exception("interview_persona failed id=%s bot=%s", request_id, bot_id)
        return JSONResponse(
            {"error": "Couldn't generate persona suggestions right now. "
                      "Please try again.", "request_id": request_id},
            status_code=500,
        )
    return result


# ---- sources ----------------------------------------------------------
@app.post("/api/bots/{bot_id}/sources/youtube")
async def api_add_youtube(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    channel = (data.get("channel") or "").strip()
    author = (data.get("author") or "").strip()
    if not channel:
        return JSONResponse({"error": "channel is required"}, status_code=400)
    if not author:
        return JSONResponse({"error": "author is required"}, status_code=400)
    # Rights attestation is enforced SERVER-SIDE — the checkbox in the UI is a
    # convenience, not the gate. Reject unless an explicit boolean true is sent.
    if data.get("consent") is not True and data.get("attest") is not True:
        return JSONResponse(
            {"error": "You must confirm you own this channel or have the rights "
                      "to use its content before adding it."},
            status_code=400,
        )

    # A YouTube source is a whole channel, a single playlist, or a single video
    # (all type=youtube, distinguished by ``kind``). The de-dup key is derived per
    # kind: channels fold their many URL spellings; a playlist keys off its
    # (case-sensitive) list= id; a video keys off its 11-char video id.
    kind = (data.get("kind") or "channel").strip().lower()
    if kind not in ("channel", "playlist", "video"):
        return JSONResponse(
            {"error": "kind must be 'channel', 'playlist', or 'video'"},
            status_code=400)
    if kind == "playlist":
        key = normalize_playlist_key(channel)
        if not key:
            return JSONResponse(
                {"error": "Personal or auto-generated mixes can't be added."},
                status_code=400,
            )
    elif kind == "video":
        key = normalize_video_key(channel)
        if not key:
            return JSONResponse(
                {"error": "That doesn't look like a YouTube video URL."},
                status_code=400,
            )
    else:
        key = normalize_youtube_key(channel)
    if bot_service.bot_store(cfg).find_source_by_key(bot_id, TYPE_YOUTUBE, key):
        return JSONResponse({"error": f"“{key}” is already a source of this bot."},
                            status_code=409)
    limit_error = quotas.check_can_add_source(cfg, user, bot)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)
    err = _record_attestation(cfg, request, actor=user.id,
                              target=f"bot:{bot_id} youtube:{key}")
    if err is not None:
        return err
    source = bot_service.add_youtube_source(cfg, bot, key, key, author, kind=kind)
    # Stamp the source's auto-sync frequency (Phase K): an explicit choice in the
    # add form wins; otherwise inherit the user's default. An unknown value falls
    # back to inherit rather than erroring the add.
    requested = autosync.normalize_freq(data.get("sync_freq") or data.get("auto_sync"))
    if kind == "video":
        # A single video never gains new items, so a schedule is pointless: default
        # a video source to "off" unless the caller explicitly picked a frequency.
        freq = requested or "off"
    else:
        freq = requested or UserSettingsStore(cfg.data_dir).get(user.id).sync_freq
    bot_service.bot_store(cfg).update_source(bot.id, source.id, sync_freq=freq)
    return {"source_id": source.id, "sync_freq": freq, "kind": kind}


@app.post("/api/bots/{bot_id}/sources/{source_id}/import")
async def api_start_import(
    bot_id: str, source_id: str, request: Request, user: User = Depends(require_user)
):
    """Second step of the two-step flow: enqueue an ingest for an existing source."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    source = next((s for s in bot.sources if s.id == source_id), None)
    if source is None:
        return JSONResponse({"error": "source not found"}, status_code=404)
    if source.type != TYPE_YOUTUBE:
        return JSONResponse({"error": "only YouTube sources can be imported"},
                            status_code=400)

    active = [
        j for j in JobStore(cfg.data_dir).list_for_user(user.id)
        if j.source_id == source_id and j.status in (STATUS_QUEUED, STATUS_RUNNING)
    ]
    if active:
        return JSONResponse(
            {"error": "An import for this source is already in progress."},
            status_code=409,
        )

    limit_error = quotas.check_can_enqueue_ingest(cfg, user)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)

    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - empty/invalid body → all defaults
        data = {}
    if not isinstance(data, dict):
        data = {}

    langs = [x.strip() for x in (data.get("langs") or "").split(",") if x.strip()]
    limit = data.get("limit")
    limit = int(limit) if limit not in (None, "", 0, "0") else None
    limit = quotas.cap_video_limit(user, limit)
    raw_retries = data.get("max_retries")
    max_retries = int(raw_retries) if str(raw_retries).strip() not in ("", "None") else None
    if max_retries is not None and max_retries <= 0:
        max_retries = None

    job_id = ingest_manager.enqueue(
        cfg, bot, source,
        user_id=user.id,
        langs=langs, limit=limit,
        base_delay=float(data.get("base_delay", 2.0)),
        max_delay=float(data.get("max_delay", 900.0)),
        max_retries=max_retries,
        cookies_browser=(data.get("cookies_browser") or "").strip() or None,
    )
    # Remember the chosen langs on the source so auto-sync re-syncs the SAME
    # selection (e.g. "all") instead of silently defaulting to en-only (L5).
    bot_service.bot_store(cfg).update_source(
        bot_id, source.id, sync_langs=",".join(langs),
    )
    return {"job_id": job_id, "source_id": source.id}


@app.post("/api/bots/{bot_id}/sources/{source_id}/sync-freq")
async def api_set_source_sync_freq(
    bot_id: str, source_id: str, request: Request, user: User = Depends(require_user)
):
    """Update a YouTube source's auto-sync frequency (Phase K).

    Body: ``{"sync_freq": "off"|"daily"|"weekly"|"monthly"|""}``. A blank value
    resets the source to inherit the user's default. Only YouTube sources carry a
    frequency."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    source = next((s for s in bot.sources if s.id == source_id), None)
    if source is None:
        return JSONResponse({"error": "source not found"}, status_code=404)
    if source.type != TYPE_YOUTUBE:
        return JSONResponse({"error": "only YouTube sources can be auto-synced"},
                            status_code=400)
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - empty/invalid body
        data = {}
    if not isinstance(data, dict):
        data = {}
    raw = data.get("sync_freq")
    # "" is a valid reset-to-inherit; any other non-empty value must be a known freq.
    if raw not in ("", None) and raw not in autosync.SYNC_FREQS:
        return JSONResponse(
            {"error": "sync_freq must be one of off, daily, weekly, monthly (or blank)"},
            status_code=400,
        )
    freq = autosync.normalize_freq(raw) if raw else ""
    updated = bot_service.bot_store(cfg).update_source(
        bot_id, source_id, sync_freq=freq)
    if updated is None:
        return JSONResponse({"error": "source not found"}, status_code=404)
    return {"source_id": source_id, "sync_freq": updated.sync_freq}


@app.post("/api/bots/{bot_id}/sources/{source_id}/sync")
async def api_sync_source_now(
    bot_id: str, source_id: str, user: User = Depends(require_user)
):
    """Incremental "Sync now" for a YouTube source (Phase K).

    Lists the channel, diffs against already-ingested videos, and enqueues ONLY
    the new ones on the existing single FIFO ingest worker (which skips anything
    already on disk). Returns the number of new videos queued."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    source = next((s for s in bot.sources if s.id == source_id), None)
    if source is None:
        return JSONResponse({"error": "source not found"}, status_code=404)
    if source.type != TYPE_YOUTUBE:
        return JSONResponse({"error": "only YouTube sources can be synced"},
                            status_code=400)
    # Same guards as a manual import: no double-run, and respect the plan's
    # per-user in-flight ingest cap.
    active = [
        j for j in JobStore(cfg.data_dir).list_for_user(user.id)
        if j.source_id == source_id and j.status in (STATUS_QUEUED, STATUS_RUNNING)
    ]
    if active:
        return JSONResponse(
            {"error": "An import for this source is already in progress."},
            status_code=409,
        )
    limit_error = quotas.check_can_enqueue_ingest(cfg, user)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)
    try:
        new_count = autosync.incremental_sync(cfg, bot, source, user_id=user.id)
    except Exception as exc:  # noqa: BLE001 - surface a clean error, never 500 the UI
        return JSONResponse({"error": f"Sync failed: {exc}"}, status_code=502)
    return {"source_id": source_id, "new": new_count}


@app.post("/api/bots/{bot_id}/sources/document")
async def api_add_document(
    bot_id: str,
    request: Request,
    file: UploadFile = File(...),
    author: str = Form(...),
    consent: str = Form(""),
    user: User = Depends(require_user),
):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    filename = Path(file.filename or "").name
    author = (author or "").strip()
    if not filename:
        return JSONResponse({"error": "file is required"}, status_code=400)
    if not author:
        return JSONResponse({"error": "author is required"}, status_code=400)
    # Rights attestation, enforced SERVER-SIDE (the multipart form sends the flag
    # as a string). Anything but an explicit truthy value is rejected.
    if consent.strip().lower() not in ("true", "1", "on", "yes"):
        return JSONResponse(
            {"error": "You must confirm you have the rights to use this document "
                      "before adding it."},
            status_code=400,
        )
    if not is_supported(filename):
        return JSONResponse({"error": "Unsupported type. Use txt, md, pdf, or docx."},
                            status_code=400)

    # Upload size cap (memory-DoS guard). Fast-reject on a declared Content-Length
    # over the cap; Caddy's request_body cap is the coarse edge guard in front.
    max_bytes = cfg.max_upload_bytes
    max_mb = max_bytes // (1024 * 1024)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                return JSONResponse(
                    {"error": f"File too large (max {max_mb} MB)."}, status_code=413)
        except ValueError:
            pass

    key = normalize_document_key(filename)
    if bot_service.bot_store(cfg).find_source_by_key(bot_id, TYPE_DOCUMENT, key):
        return JSONResponse({"error": "This document is already a source of this bot."},
                            status_code=409)
    limit_error = quotas.check_can_add_source(cfg, user, bot)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)
    # Streaming, capped read: never buffer more than the cap even if the client
    # lied about (or omitted) Content-Length above.
    data_bytes = await _read_upload_capped(file, max_bytes)
    if data_bytes is None:
        return JSONResponse({"error": f"File too large (max {max_mb} MB)."}, status_code=413)
    if not data_bytes:
        return JSONResponse({"error": "file is empty"}, status_code=400)
    err = _record_attestation(cfg, request, actor=user.id,
                              target=f"bot:{bot_id} document:{key}")
    if err is not None:
        return err
    source = bot_service.add_document_source(cfg, bot, key, filename, author, data_bytes)
    job = manager.start_bot_rebuild(bot, owner_id=user.id)
    return {"job_id": job.id, "source_id": source.id}


@app.post("/api/bots/{bot_id}/transcripts")
async def api_ingest_transcripts(bot_id: str, request: Request, user: User = Depends(require_user)):
    """The ingest seam: accept already-extracted transcript items (extension-ready).

    This is a content-adding seam like the youtube/document endpoints, so it is
    gated by the SAME server-side rights attestation: the caller (browser or the
    future extension) must send an explicit ``consent``/``attest`` true, and the
    acceptance is written to the audit log (fail-closed) BEFORE any source is
    created. The request/response shape is otherwise unchanged.
    """
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    items = data.get("items") or []
    if not isinstance(items, list) or not items:
        return JSONResponse({"error": "items (a non-empty list) is required"}, status_code=400)
    author = (data.get("author") or "").strip()
    if not author:
        return JSONResponse({"error": "author is required"}, status_code=400)
    # Rights attestation is enforced SERVER-SIDE here too — this endpoint adds
    # content, so it cannot bypass the gate the youtube/document seams enforce.
    if data.get("consent") is not True and data.get("attest") is not True:
        return JSONResponse(
            {"error": "You must confirm you own this channel or have the rights "
                      "to use its content before adding it."},
            status_code=400,
        )
    label = (data.get("label") or data.get("key") or author).strip()
    key = normalize_youtube_key(data.get("key") or label) or label

    err = _record_attestation(cfg, request, actor=user.id,
                              target=f"bot:{bot_id} youtube:{key}")
    if err is not None:
        return err

    bstore = bot_service.bot_store(cfg)
    source = bstore.find_source_by_key(bot_id, TYPE_YOUTUBE, key)
    if source is None:
        source = bstore.add_source(bot_id, TYPE_YOUTUBE, key, label, author)
    written = bot_service.ingest_text_items(cfg, bot, source, items)
    job = manager.start_bot_rebuild(bstore.get(bot_id), owner_id=user.id)
    return {"job_id": job.id, "source_id": source.id, "written": written}


@app.delete("/api/bots/{bot_id}/sources/{source_id}")
def api_delete_source(bot_id: str, source_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    removed = bot_service.remove_source(cfg, bot, source_id)
    if removed is None:
        return JSONResponse({"error": "source not found"}, status_code=404)
    job = manager.start_bot_rebuild(bot_service.bot_store(cfg).get(bot_id), owner_id=user.id)
    return {"ok": True, "job_id": job.id}


@app.post("/api/bots/{bot_id}/rebuild")
def api_rebuild(bot_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    job = manager.start_bot_rebuild(bot, owner_id=user.id)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str, user: User = Depends(require_user)):
    job = manager.get(job_id)
    # 404 (not 403) for a non-owner: mirror the ingest-job endpoints so a caller
    # can't use the status route as an existence oracle for another tenant's
    # rebuild, and never leak another tenant's label / log lines via snapshot().
    if not job or job.owner_id != user.id:
        return JSONResponse({"error": "not found"}, status_code=404)
    return job.snapshot()


@app.get("/api/ingest/jobs")
def api_ingest_jobs(user: User = Depends(require_user)):
    cfg = load_config()
    jobs = JobStore(cfg.data_dir).list_for_user(user.id)
    return {"jobs": [asdict(j) for j in jobs]}


@app.get("/api/ingest/jobs/{job_id}")
def api_ingest_job(job_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    job = JobStore(cfg.data_dir).get(job_id)
    if job is None or job.user_id != user.id:
        return JSONResponse({"error": "not found"}, status_code=404)
    return asdict(job)


@app.post("/api/ingest/jobs/{job_id}/cancel")
def api_ingest_job_cancel(job_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    store = JobStore(cfg.data_dir)
    job = store.get(job_id)
    if job is None or job.user_id != user.id:
        return JSONResponse({"error": "not found"}, status_code=404)
    updated = store.request_cancel(job_id)
    if updated is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return asdict(updated)


@app.post("/api/ingest/jobs/{job_id}/retry")
def api_ingest_job_retry(job_id: str, user: User = Depends(require_user)):
    """Requeue a failed import: enqueue a FRESH job reusing the old one's params,
    then delete the errored record so the queue row is replaced, not duplicated.

    Only a terminal FAILED job (error/cancelled) can be retried — a done job has
    nothing to redo, and an active one is already running. The FIFO worker holds
    its queue in memory, so flipping a stored status would never re-run the job:
    we must go back through ``enqueue``."""
    cfg = load_config()
    store = JobStore(cfg.data_dir)
    job = store.get(job_id)
    if job is None or job.user_id != user.id:
        return JSONResponse({"error": "not found"}, status_code=404)
    if job.status not in (STATUS_ERROR, STATUS_CANCELLED):
        return JSONResponse(
            {"error": "Only failed imports can be retried."}, status_code=409)

    # Reconstruct the bot + source the job targeted; either may have been deleted.
    bstore = bot_service.bot_store(cfg)
    bot = bstore.get(job.bot_id) if job.bot_id else None
    source = next((s for s in bot.sources if s.id == job.source_id), None) if bot else None
    if bot is None or source is None:
        return JSONResponse(
            {"error": "The bot or source for this import no longer exists."},
            status_code=409,
        )

    # Don't double-run: if an import for this source is already queued/running
    # (e.g. the user retried twice), reject rather than pile on.
    active = [
        j for j in store.list_for_user(user.id)
        if j.source_id == job.source_id and j.status in (STATUS_QUEUED, STATUS_RUNNING)
    ]
    if active:
        return JSONResponse(
            {"error": "An import for this source is already in progress."},
            status_code=409,
        )
    limit_error = quotas.check_can_enqueue_ingest(cfg, user)
    if limit_error:
        return JSONResponse({"error": limit_error}, status_code=429)

    new_id = ingest_manager.enqueue(
        cfg, bot, source,
        user_id=user.id,
        # Re-apply the CURRENT plan cap: the stored limit was capped under the
        # plan in force when the job was first enqueued, which may since have been
        # downgraded. Retrying must never run at a stale, higher video cap.
        langs=job.langs, limit=quotas.cap_video_limit(user, job.limit),
        base_delay=job.base_delay, max_delay=job.max_delay,
        max_retries=job.max_retries,
        cookies_browser=job.cookies_browser or None,
        origin="manual",
    )
    # Keep the source's persisted langs aligned with this (re)import (L5).
    bot_service.bot_store(cfg).update_source(
        bot.id, source.id, sync_langs=job.langs or "",
    )
    # Replace the errored row with the fresh queued one.
    store.delete(job_id)
    new_job = store.get(new_id)
    if new_job is None:
        return JSONResponse({"error": "retry failed"}, status_code=500)
    return asdict(new_job)


@app.delete("/api/ingest/jobs/{job_id}")
def api_ingest_job_dismiss(job_id: str, user: User = Depends(require_user)):
    """Drop a FINISHED job row (done/error/cancelled) so it stops being reported.

    An active (queued/running) job must be cancelled first — dismissing it would
    orphan a running worker, so we refuse with a 409."""
    cfg = load_config()
    store = JobStore(cfg.data_dir)
    job = store.get(job_id)
    if job is None or job.user_id != user.id:
        return JSONResponse({"error": "not found"}, status_code=404)
    if job.status not in (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED):
        return JSONResponse(
            {"error": "Cancel the import before dismissing it."}, status_code=409)
    store.delete(job_id)
    return {"ok": True}


# ---- chat -------------------------------------------------------------
def _resolve_chat(cfg, billed_user: User):
    """Resolve the effective chat Config for ``billed_user`` and enforce budget.

    Returns ``(effective_cfg, None)`` when the chat may proceed — ``effective_cfg``
    is the server config for Managed, or the user's BYOK provider/key swapped in.
    Returns ``(None, (status, message))`` when it must be rejected:

    * Managed mode + the user is at/over their plan's monthly budget -> 429.
    * No usable mode for the plan (e.g. managed disabled, no BYOK key) -> 400.

    The budget is checked BEFORE any model call, and only for Managed — a BYOK
    chat runs on the user's own key and is never budget-gated. The resolution +
    budget gate lives in the shared ``resolve_chat_for_user`` helper so the web
    and Telegram paths enforce it identically.
    """
    plan = plan_for(billed_user)
    resolved = resolve_chat_for_user(cfg, billed_user, plan)
    if resolved.status == "over_budget":
        return None, (429, resolved.error or "Monthly usage limit reached for this plan.")
    if resolved.config is None:
        return None, (400, resolved.error or "Chat is unavailable for your plan.")
    return resolved.config, None


# Web chat modes mirror the Telegram inline modes (see telegram_conv): the SPA sends
# "reference" (Довідник) or "thinking" (Мислення) per chat. Anything else — an older
# client that sends no mode, or a bad value — maps to None, i.e. TODAY'S default web
# pipeline, byte-for-byte. «Мислення» routes to the v2 general-advisor base
# (rag.Assistant); «Довідник» / None keep the grounded pipeline.
_WEB_ANSWER_MODES = {"reference", "thinking"}


def _answer_mode(data: dict) -> str | None:
    """The per-request chat mode from the request body, or None when absent/unknown
    (which preserves the historical web behavior — no answer_mode passed)."""
    mode = (data.get("mode") or "").strip().lower()
    return mode if mode in _WEB_ANSWER_MODES else None


@app.post("/api/bots/{bot_id}/chat")
async def api_chat(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    # Bound the inputs before they reach the owner-billed prompt (token-burn/DoS
    # guard, mirrors the guest path). Over-long messages are rejected; history is
    # sanitized + capped by the shared guest sanitizer; top_k is clamped to range.
    if len(message) > AUTHED_MESSAGE_MAX_CHARS:
        return JSONResponse(
            {"error": f"Message too long (max {AUTHED_MESSAGE_MAX_CHARS} characters)."},
            status_code=400,
        )
    history = sanitize_guest_history(
        data.get("history"),
        max_turns=AUTHED_HISTORY_MAX_TURNS,
        max_item_chars=AUTHED_MESSAGE_MAX_CHARS,
        max_total_chars=AUTHED_HISTORY_MAX_CHARS,
    )
    top_k = _clamp_top_k(data.get("top_k", CHAT_TOP_K_DEFAULT))
    # Managed/BYOK resolution + monthly budget enforcement for the acting user
    # (who, for an owned bot, is also the billed owner). BEFORE any model call.
    chat_cfg, err = _resolve_chat(cfg, user)
    if err is not None:
        return JSONResponse({"error": err[1]}, status_code=err[0])
    try:
        # PLAN §5b: awaited async chat — the ~7s of model calls no longer block the
        # single event loop (the old sync call here serialized EVERY request).
        result = await bot_service.achat(
            chat_cfg, bot, message, history=history, top_k=top_k,
            answer_mode=_answer_mode(data),
        )
    except Exception:  # noqa: BLE001 - provider/key/path detail stays in the logs
        request_id = secrets.token_hex(4)
        logger.exception("chat failed id=%s bot=%s", request_id, bot_id)
        return JSONResponse(
            {"error": "Something went wrong while answering. Please try again.",
             "request_id": request_id},
            status_code=500,
        )
    sources = _dedupe_sources(result)
    # BUG-007: surface the "still indexing" flag the gate sets so the 1.5 UI can
    # render an indexing state instead of a normal answer bubble (data-only).
    resp = {"answer": result["answer"], "sources": sources}
    if result.get("indexing"):
        resp["indexing"] = True
    return resp


# Idle gap after which the SSE loop emits a keep-alive comment. Comfortably under
# common proxy read-idle timeouts (nginx/Caddy default ~60s) so a long agentic
# round with no user-facing delta can't look like a dead connection.
SSE_HEARTBEAT_SECS = 15.0


def _sse_frame(event: str, payload: dict) -> str:
    """One Server-Sent-Events frame: a named event plus its JSON data line."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/bots/{bot_id}/chat/stream")
async def api_chat_stream(bot_id: str, request: Request, user: User = Depends(require_user)):
    """Streaming (SSE) twin of :func:`api_chat` (Inc1.4b) — ADDITIVE.

    The non-stream ``/chat`` endpoint is untouched (the Vue SPA still uses it; SSE
    consumption is deferred to 1.5). This mirrors ``/chat`` exactly for validation,
    the BUG-007 indexing gate, and the Managed/BYOK monthly-budget gate — all of
    which run BEFORE any token is streamed — then emits ``delta`` frames as the
    answer arrives and a final ``done`` frame carrying deduped sources."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if len(message) > AUTHED_MESSAGE_MAX_CHARS:
        return JSONResponse(
            {"error": f"Message too long (max {AUTHED_MESSAGE_MAX_CHARS} characters)."},
            status_code=400,
        )
    history = sanitize_guest_history(
        data.get("history"),
        max_turns=AUTHED_HISTORY_MAX_TURNS,
        max_item_chars=AUTHED_MESSAGE_MAX_CHARS,
        max_total_chars=AUTHED_HISTORY_MAX_CHARS,
    )
    top_k = _clamp_top_k(data.get("top_k", CHAT_TOP_K_DEFAULT))
    # Budget/BYOK gate BEFORE opening the stream — a 429 must arrive as a normal
    # JSON response, never mid-stream (the same ordering as the non-stream path).
    chat_cfg, err = _resolve_chat(cfg, user)
    if err is not None:
        return JSONResponse({"error": err[1]}, status_code=err[0])

    async def _events():
        # PLAN §5b: an ASYNC generator on the event loop — before this, the sync
        # generator was iterated on anyio's ~40-thread pool, which was the hard
        # concurrency ceiling per worker for streamed chats.
        agen = bot_service.achat_stream(
            chat_cfg, bot, message, history=history, top_k=top_k,
            answer_mode=_answer_mode(data),
        )
        ait = agen.__aiter__()
        # The next event, fetched as a Task so we can wait on it WITHOUT cancelling
        # it on a heartbeat tick — cancelling ``__anext__`` would tear the billing
        # stream down mid-answer. Only teardown (below) cancels it.
        pending: asyncio.Future | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(ait.__anext__())
                # Wait up to SSE_HEARTBEAT_SECS for the next event; on timeout the
                # Task keeps running (not cancelled) and we emit a keep-alive so a
                # long buffered agentic round (#42 withholds the final answer until
                # it is grounding-validated → possibly no delta for seconds) can't
                # trip a proxy idle-timeout or freeze the client.
                done, _ = await asyncio.wait({pending}, timeout=SSE_HEARTBEAT_SECS)
                if not done:
                    yield ": keep-alive\n\n"  # SSE comment — clients ignore it
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                finally:
                    pending = None
                etype = event.get("type")
                if etype == "delta":
                    yield _sse_frame("delta", {"text": event.get("text", "")})
                elif etype == "status":
                    # Agentic progress (searching/round). The classic path emits
                    # none of these, so with the flag OFF this branch never fires
                    # and the wire stays byte-for-byte classic.
                    yield _sse_frame("status", {
                        "stage": event.get("stage"),
                        "query": event.get("query"),
                        "round": event.get("round"),
                    })
                elif etype == "done":
                    done_payload = {"sources": _dedupe_sources(event)}
                    if event.get("indexing"):
                        done_payload["indexing"] = True
                    yield _sse_frame("done", done_payload)
        except Exception:  # noqa: BLE001 - provider/key/path detail stays in the logs
            request_id = secrets.token_hex(4)
            logger.exception("chat stream failed id=%s bot=%s", request_id, bot_id)
            yield _sse_frame(
                "error",
                {"error": "Something went wrong while answering. Please try again.",
                 "request_id": request_id},
            )
        finally:
            # Deterministic teardown on client disconnect (GeneratorExit is NOT
            # caught above) OR normal exit. An in-flight ``__anext__`` Task is
            # cancelled and awaited FIRST so its CancelledError propagates through
            # bot_service.achat_stream's finally (exactly-once billing of the
            # produced-so-far tokens) before we aclose — never left to GC.
            if pending is not None:
                pending.cancel()
                try:
                    await pending
                except BaseException:  # noqa: BLE001 - cancel/teardown noise
                    pass
            await agen.aclose()

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _dedupe_sources(result: dict) -> list[dict]:
    # Dedupe by the video's stable identity (id/url), NOT its title: two distinct
    # videos can share a title, and keying on title collapses them to one entry
    # keeping the first video's (wrong) link. Fall back to url, then title, and
    # use .get throughout so a row missing any key never 500s the surface.
    #
    # Preserve the TIMECODE DEEP-LINK: hits are ordered most-relevant-first, so the
    # FIRST hit kept per identity carries the best moment. When that hit has a
    # `start` offset we surface `<url>?t=<start>` (the deep-link into the video at
    # the cited moment) instead of the bare video url — the product promise the UI
    # then renders as a clickable citation. Sources with no `start` (e.g. a
    # document, or the synthetic rows in verify_low_hardening) keep their plain url.
    sources, seen = [], set()
    for h in result.get("sources", []):
        m = h.get("meta") or {}
        ident = m.get("video_id") or m.get("url") or m.get("title") or ""
        if ident and ident in seen:
            continue
        if ident:
            seen.add(ident)
        base_url = m.get("url", "") or ""
        start = m.get("start")
        url = _deep_link(base_url, start) if (base_url and start is not None) else base_url
        sources.append({"title": m.get("title", ""), "url": url})
    return sources


# ---- private share (owner management) --------------------------------
@app.get("/api/bots/{bot_id}/share")
def api_get_share(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _share_info(request, bot, user)


@app.post("/api/bots/{bot_id}/share")
def api_create_share(bot_id: str, request: Request, user: User = Depends(require_user)):
    """Create-or-return the private share link (idempotent)."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    updated = bot_service.bot_store(cfg).create_share_token(bot_id)
    return _share_info(request, updated, user)


@app.post("/api/bots/{bot_id}/share/rotate")
def api_rotate_share(bot_id: str, request: Request, user: User = Depends(require_user)):
    """Mint a fresh token — the previous link stops working immediately."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    updated = bot_service.bot_store(cfg).rotate_share_token(bot_id)
    return _share_info(request, updated, user)


@app.delete("/api/bots/{bot_id}/share")
def api_revoke_share(bot_id: str, request: Request, user: User = Depends(require_user)):
    """Revoke sharing — the link stops resolving at once."""
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    updated = bot_service.bot_store(cfg).revoke_share_token(bot_id)
    return _share_info(request, updated, user)


# ---- private share (guest chat — no login) ---------------------------
@app.post("/api/s/{token}/chat")
async def api_share_chat(token: str, request: Request):
    """Token-scoped guest chat. Runs on and is billed to the bot OWNER (usage is
    recorded under ``bot.owner_id`` inside ``bot_service.chat``), guarded by the
    per-bot daily cap + per-client rate limit. No authentication required."""
    cfg = load_config()
    bot = bot_service.bot_store(cfg).find_by_share_token(token)
    if bot is None:
        return JSONResponse({"error": "This share link is no longer active."}, status_code=404)

    # Every guest cap is billed to — and therefore sized by — the bot OWNER's
    # plan. Resolve it once; fall back to the default plan only if the owner
    # record is somehow missing (never trust a None into plan_for).
    owner = UserStore(cfg.data_dir).get(bot.owner_id)
    plan = plan_for(owner) if owner is not None else PLANS[DEFAULT_PLAN]

    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if len(message) > plan.guest_message_max_chars:
        return JSONResponse(
            {"error": f"Message too long (max {plan.guest_message_max_chars} characters)."},
            status_code=400,
        )

    cap = _effective_guest_cap(bot, plan)
    decision = share_limiter.check(
        bot.id,
        _client_ip(request),
        daily_cap=cap,
        min_interval_sec=plan.guest_min_interval_sec,
        hourly_limit=plan.guest_hourly_limit,
        distinct_guest_cap=plan.distinct_guest_cap,
    )
    if not decision.ok:
        return JSONResponse({"error": decision.error}, status_code=decision.status)

    # Guest-supplied context is sanitized on all three axes (role whitelist,
    # per-turn size, total size + turn count) before it reaches the owner-billed
    # prompt — see ``sanitize_guest_history`` — using the owner plan's caps.
    history = sanitize_guest_history(
        data.get("history"),
        max_turns=plan.guest_history_max_turns,
        max_item_chars=plan.guest_message_max_chars,
        max_total_chars=plan.guest_history_max_chars,
    )
    # The guest runs on — and is billed to — the OWNER: resolve the OWNER's
    # Managed/BYOK key and enforce the OWNER's monthly Managed budget BEFORE any
    # model call. A missing owner record (should not happen) degrades to the
    # server key with no budget rather than blocking the guest.
    chat_cfg = cfg
    if owner is not None:
        chat_cfg, err = _resolve_chat(cfg, owner)
        if err is not None:
            # Neutral, guest-facing wording — never expose the owner's plan /
            # billing state to an anonymous visitor.
            if err[0] == 429:
                return JSONResponse(
                    {"error": "This bot has reached its usage limit for now. "
                              "Please try again later."},
                    status_code=429,
                )
            return JSONResponse(
                {"error": "The bot is temporarily unavailable. Please try again in a moment."},
                status_code=503,
            )
    try:
        # PLAN §5b: awaited async chat — guest chats stop blocking the event loop too.
        result = await bot_service.achat(
            chat_cfg, bot, message, history=history, top_k=CHAT_TOP_K_DEFAULT,
            answer_mode=_answer_mode(data),
        )
    except Exception:  # noqa: BLE001 - never leak provider/internal errors to a guest
        return JSONResponse(
            {"error": "The bot is temporarily unavailable. Please try again in a moment."},
            status_code=502,
        )
    resp = {"answer": result["answer"], "sources": _dedupe_sources(result)}
    if result.get("indexing"):
        resp["indexing"] = True
    return resp


# ---- telegram ---------------------------------------------------------
@app.post("/api/bots/{bot_id}/telegram")
async def api_telegram_connect(bot_id: str, request: Request, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    data = await request.json()
    token = (data.get("token") or "").strip()
    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)
    try:
        username = telegram_bot.manager.start(bot_id, token)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    bot_service.bot_store(cfg).update(bot_id, telegram_token=token, telegram_username=username)
    return {"ok": True, "username": username}


@app.delete("/api/bots/{bot_id}/telegram")
def api_telegram_disconnect(bot_id: str, user: User = Depends(require_user)):
    cfg = load_config()
    bot = _get_owned(cfg, bot_id, user)
    if bot is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    telegram_bot.manager.stop(bot_id)
    bot_service.bot_store(cfg).update(bot_id, telegram_token="", telegram_username="")
    return {"ok": True}


def run(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)
