"""Google OAuth 2.0 login and signed-cookie sessions for the multi-tenant web app.

Authentication is the tenant boundary: every request must resolve to a ``User``
(via ``current_user`` / ``login_required``) so downstream data access can be scoped
to an owner. We keep this self-contained — Authlib drives the OpenID Connect flow,
Starlette's ``SessionMiddleware`` holds only the internal ``user_id`` in a signed
cookie, and a dev-login escape hatch keeps the app usable locally before the
operator configures Google credentials.
"""
from __future__ import annotations

import logging
import os
import secrets

from authlib.integrations.base_client.errors import MismatchingStateError, OAuthError
from authlib.integrations.starlette_client import OAuth
from starlette.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse

from . import settings_store
from .accounts import User, UserStore
from .config import is_production, load_config
from .tenants import TenantStore

logger = logging.getLogger(__name__)

_GOOGLE_METADATA = "https://accounts.google.com/.well-known/openid-configuration"
_DEFAULT_REDIRECT = "http://127.0.0.1:8000/auth/callback"
_TRUTHY = {"1", "true", "yes"}
_FALSY = {"0", "false", "no"}

_oauth: OAuth | None = None

# One-shot marker for the login-retry safety net (see ``auth_callback``). A
# state/CSRF mismatch on the FIRST login is almost always the OAuth-state cookie
# failing to round-trip on the initial hop (the flow starts on one host — e.g.
# ``localhost`` — but the callback lands on another — ``127.0.0.1`` — so the
# cookie holding the state isn't in that host's jar yet). We re-run the login
# ONCE; this cookie bounds the retry so a genuinely broken login still lands on
# the error page instead of looping forever.
_OAUTH_RETRY_COOKIE = "ytrag_oauth_retry"


def _is_state_mismatch(exc: Exception) -> bool:
    """True when a callback failed a CSRF/state check (recoverable by a clean
    re-login) rather than on a real auth failure (bad/expired code, denied
    consent, server restart mid-flow)."""
    if isinstance(exc, MismatchingStateError):
        return True
    if isinstance(exc, OAuthError):
        return (getattr(exc, "error", "") or "") in {
            "mismatching_state", "missing_state", "csrf_error",
        }
    return False


def _store() -> UserStore:
    return UserStore(load_config().data_dir)


def _ensure_tenant(user_id: str) -> None:
    # Best-effort: a tenant record must exist for every signed-in user, but a
    # store hiccup must never block login.
    try:
        TenantStore(load_config().data_dir).get_or_create(user_id)
    except Exception:
        logger.exception("failed to ensure tenant record for %s", user_id)


def dev_login_enabled() -> bool:
    # The dev escape hatch is never available in production, regardless of flags.
    if is_production():
        return False
    settings_store.apply_to_env()
    if os.getenv("YTRAG_DEV_LOGIN", "").strip().lower() in _TRUTHY:
        return True
    return not os.getenv("GOOGLE_CLIENT_ID")


def _secure_cookies() -> bool:
    """Whether session cookies get the Secure flag (https-only)."""
    override = os.getenv("YTRAG_SECURE_COOKIES", "").strip().lower()
    if override in _TRUTHY:
        return True
    if override in _FALSY:
        return False
    return is_production()


def _dev_session_secret() -> str:
    """Development only: reuse a persisted secret so restarts don't drop sessions."""
    secret = settings_store.load_settings().get("session_secret")
    if secret:
        return str(secret)
    secret = secrets.token_urlsafe(32)
    settings_store.save_settings({"session_secret": secret})
    return secret


def _require_production_env() -> None:
    """Refuse to boot in production without the mandatory secrets/OAuth config."""
    missing: list[str] = []
    if not os.getenv("YTRAG_SESSION_SECRET"):
        missing.append("YTRAG_SESSION_SECRET (persistent signing key for session cookies)")
    if not os.getenv("GOOGLE_CLIENT_ID"):
        missing.append("GOOGLE_CLIENT_ID (Google OAuth client id)")
    if not os.getenv("GOOGLE_CLIENT_SECRET"):
        missing.append("GOOGLE_CLIENT_SECRET (Google OAuth client secret)")
    redirect = (os.getenv("YTRAG_OAUTH_REDIRECT_URL") or "").strip()
    if not redirect:
        missing.append("YTRAG_OAUTH_REDIRECT_URL (public https:// callback URL)")
    else:
        low = redirect.lower()
        if not low.startswith("https://") or "localhost" in low or "127.0.0.1" in low:
            missing.append(
                "YTRAG_OAUTH_REDIRECT_URL must be an https:// URL and not "
                f"localhost/127.0.0.1 (got: {redirect})"
            )
    if missing:
        raise RuntimeError(
            "YTRAG_ENV=production but required configuration is missing or invalid:\n  - "
            + "\n  - ".join(missing)
            + "\n\nSet these as real environment variables (see .env.example / "
            "PRODUCTION.md) and restart."
        )


class APIUnauthorized(Exception):
    """Raised by ``login_required`` for unauthenticated ``/api/*`` requests.

    The SPA's fetch adapter (``spa/api/real.js``) expects a machine-readable
    ``401 {"error": "unauthenticated"}`` for API calls, not the 307→``/login``
    HTML redirect that browser navigations get. ``app.py`` registers a handler
    that renders this as that JSON so the client-side router can bounce the user
    to ``#/login`` itself.
    """


def current_user(request) -> User | None:
    user_id = request.session.get("user_id")
    return _store().get(user_id) if user_id else None


def login_required(request) -> User:
    user = current_user(request)
    if user is not None:
        return user
    # API callers (the SPA) get 401 JSON so their fetch interceptor can route to
    # the login view; page navigations keep the 307→/login redirect.
    if request.url.path.startswith("/api/"):
        raise APIUnauthorized()
    raise HTTPException(status_code=307, headers={"Location": "/login"})


def install(app) -> None:
    settings_store.apply_to_env()

    if is_production():
        _require_production_env()

    secret = os.getenv("YTRAG_SESSION_SECRET")
    if not secret:
        # Only reachable in development: production is guaranteed a secret above.
        secret = _dev_session_secret()
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        https_only=_secure_cookies(),
        same_site="lax",
    )

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    global _oauth
    _oauth = OAuth()
    if client_id and client_secret:
        _oauth.register(
            name="google",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=_GOOGLE_METADATA,
            client_kwargs={"scope": "openid email profile"},
        )

    def _redirect_uri() -> str:
        return os.getenv("YTRAG_OAUTH_REDIRECT_URL") or _DEFAULT_REDIRECT

    @app.get("/auth/google")
    async def auth_google(request: Request):
        client = _oauth.create_client("google") if _oauth else None
        if client is None:
            return RedirectResponse("/login?error=oauth_not_configured", status_code=302)
        return await client.authorize_redirect(request, _redirect_uri())

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        client = _oauth.create_client("google") if _oauth else None
        if client is None:
            return RedirectResponse("/login?error=oauth_not_configured", status_code=302)
        already_retried = request.cookies.get(_OAUTH_RETRY_COOKIE) == "1"
        try:
            token = await client.authorize_access_token(request)
            info = token.get("userinfo") or await client.userinfo(token=token)
            user = _store().upsert_google(
                google_sub=info["sub"],
                email=info.get("email", ""),
                name=info.get("name", ""),
                picture=info.get("picture", ""),
            )
            _ensure_tenant(user.id)
        except Exception as exc:
            # A CSRF/state mismatch on the FIRST hop is almost always the OAuth-state
            # cookie not surviving the initial cross-site round-trip (the login began
            # on one host but the callback landed on another, so the state cookie
            # isn't in this host's jar yet). Rather than dumping a first-time user on
            # an error page, transparently re-run ONE clean login — by then the
            # cookie is anchored to the callback host and validates. A one-shot
            # marker cookie bounds this to a single retry, so a genuine, repeated
            # failure still lands on the error page and can never loop.
            if _is_state_mismatch(exc) and not already_retried:
                logger.info("oauth state mismatch on first attempt — retrying login once")
                retry = RedirectResponse("/auth/google", status_code=302)
                retry.set_cookie(
                    _OAUTH_RETRY_COOKIE, "1",
                    max_age=300, path="/", httponly=True,
                    samesite="lax", secure=_secure_cookies(),
                )
                return retry
            # Stale/replayed state after we already retried, expired code, or a
            # server restart mid-flow must not surface as a 500 — bounce the user
            # back to a clean login page with an error.
            logger.exception("oauth callback failed")
            failed = RedirectResponse("/login?error=login_failed", status_code=302)
            failed.delete_cookie(_OAUTH_RETRY_COOKIE, path="/")
            return failed
        request.session["user_id"] = user.id
        done = RedirectResponse("/", status_code=302)
        done.delete_cookie(_OAUTH_RETRY_COOKIE, path="/")
        return done

    @app.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=302)

    @app.get("/auth/dev")
    async def auth_dev(request: Request):
        if not dev_login_enabled():
            raise HTTPException(status_code=404)
        user = _store().upsert_google(
            google_sub="dev-local",
            email="dev@local",
            name="Dev User",
        )
        _ensure_tenant(user.id)
        request.session["user_id"] = user.id
        return RedirectResponse("/", status_code=302)
