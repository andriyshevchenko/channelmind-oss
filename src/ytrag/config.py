from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_LLM_MODEL = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-sonnet-4.5",
}
_DEFAULT_EMBED_MODEL = {
    "voyage": "voyage-3",
    "openai": "text-embedding-3-large",
    "openrouter": "openai/text-embedding-3-large",
}
_DEFAULT_TRANSCRIPTION_MODEL = {
    "groq": "whisper-large-v3",
    "openai": "whisper-1",
}
_DEFAULT_VISION_MODEL = {
    "openrouter": "anthropic/claude-sonnet-4.5",
    "openai": "gpt-4o",
}
_OPENAI_COMPAT_BASE = {
    "openai": None,  # SDK default: https://api.openai.com/v1
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
}


def default_llm_model(provider: str) -> str:
    """The default chat model for a provider (used by per-user BYOK resolution).

    Falls back to the anthropic default for an unknown provider so a caller never
    ends up with an empty model string.
    """
    return _DEFAULT_LLM_MODEL.get(provider, _DEFAULT_LLM_MODEL["anthropic"])


def default_embed_model(provider: str) -> str:
    """The default embedding model for a provider (BYOK corpus pinning). Falls back
    to the voyage default for an unknown provider so a model string is never empty."""
    return _DEFAULT_EMBED_MODEL.get(provider, _DEFAULT_EMBED_MODEL["voyage"])


def default_transcription_model(provider: str) -> str:
    """The default transcription model for a provider (per-capability BYOK swap)."""
    return _DEFAULT_TRANSCRIPTION_MODEL.get(provider, _DEFAULT_TRANSCRIPTION_MODEL["groq"])


def default_vision_model(provider: str) -> str:
    """The default vision model for a provider (per-capability BYOK swap)."""
    return _DEFAULT_VISION_MODEL.get(provider, _DEFAULT_VISION_MODEL["openrouter"])


def is_production() -> bool:
    """True when running under YTRAG_ENV=production.

    This single switch flips 12-factor env precedence, secure cookies, the
    dev-login escape hatch, and fail-fast startup validation.
    """
    return os.getenv("YTRAG_ENV", "development").strip().lower() == "production"


def env_startup_warning() -> str | None:
    """A loud boot warning when ``YTRAG_ENV`` is not explicitly set (else None).

    The test-mode fakes are only prod-safe because :func:`is_production` forces them
    off, and that gate keys on ``YTRAG_ENV``. When the var is unset the process
    silently defaults to ``development`` and would honor ``YTRAG_TEST_MODE`` — a real
    deployment that forgets the var could boot serving offline fakes. Returning a
    warning string (rather than raising) keeps dev/test working while making the
    ambiguity impossible to miss in the logs; the caller logs it at WARNING.
    """
    if os.getenv("YTRAG_ENV", "").strip():
        return None
    return (
        "YTRAG_ENV is not set — defaulting to 'development'. A real deployment MUST "
        "set YTRAG_ENV=production explicitly; otherwise YTRAG_TEST_MODE would be "
        "honored and offline test fakes could be served on a production path."
    )


def test_mode() -> bool:
    """True when the offline deterministic test mode is active.

    Honoured ONLY outside production and only when ``YTRAG_TEST_MODE`` is truthy —
    the SAME fail-closed posture as the dev-login escape hatch: production forces
    it OFF regardless of the env var, and it defaults OFF everywhere else. When on,
    the three non-deterministic edges (chat/persona LLM, embeddings, YouTube/yt-dlp
    source) are swapped for in-repo fakes at their factory boundaries so the real
    app runs fully offline with no network and no API keys. The test-mode fakes in
    ``ytrag.testmode`` are imported LAZILY behind this gate, so nothing test-only
    is ever loaded or instantiated on a production path.
    """
    if is_production():
        return False
    return os.getenv("YTRAG_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def test_guest_daily_cap() -> int | None:
    """TEST-ONLY override for the per-bot guest daily message cap, or None.

    Honoured ONLY when :func:`test_mode` is true — which is itself forced OFF in
    production — so this can NEVER lower a real deployment's guest caps (same
    fail-closed posture as the other test seams). It lets the offline e2e server
    force a tiny cap via ``YTRAG_TEST_GUEST_DAILY_CAP`` so the guest daily-cap 429
    is reachable deterministically, instead of sending 50+ messages. A missing,
    non-integer, or non-positive value is ignored (returns None → normal caps).
    """
    if not test_mode():
        return None
    raw = os.getenv("YTRAG_TEST_GUEST_DAILY_CAP", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class Config:
    llm_provider: str
    llm_model: str
    embed_provider: str
    embed_model: str
    transcription_provider: str
    transcription_model: str
    vision_provider: str
    vision_model: str
    anthropic_api_key: str
    openai_api_key: str
    openrouter_api_key: str
    voyage_api_key: str
    groq_api_key: str
    data_dir: Path
    chroma_dir: Path
    transcript_proxy: str
    # Max accepted document-upload size (bytes). Enforced in the web upload
    # handler (413) and mirrored at the edge by Caddy's request_body cap. A new
    # field with a default keeps existing Config(...) call sites working.
    max_upload_bytes: int = 20 * 1024 * 1024
    # Hard ceiling on the WHOLE request body (bytes), enforced by the outermost
    # ASGI middleware BEFORE routing/auth/form-parsing. Sits a bit above the
    # per-file upload cap so multipart overhead fits; independent of Caddy so the
    # disk-DoS guard holds even when the app is hit directly (dev/no proxy).
    max_body_bytes: int = 25 * 1024 * 1024
    # Canonical app hostname for the CSRF Origin comparison. When set, the guard
    # trusts THIS over the client-suppliable Host/X-Forwarded-Host headers. Empty
    # in dev → the guard falls back to the request Host (previous behavior).
    app_domain: str = ""
    # Model for the pre-answer understanding/router call ONLY (intent + query rewrite),
    # on the SAME provider/key as the chat model (BUG-029 item 4). Empty → fall back to
    # ``llm_model`` (today's behavior: router shares the answer model). Set a small, fast
    # model (e.g. google/gemini-3.5-flash-lite) so the router — which runs BEFORE the
    # first token on every turn — is sub-second instead of a full-flagship round-trip.
    router_model: str = ""
    # Agentic RAG (YTRAG_AGENTIC_RAG). OFF (default) = today's exact single-shot
    # grounded pipeline (router → retrieve → one answer call). ON = a grounded
    # "question" turn is answered by an AGENT LOOP: the answer model drives retrieval
    # itself via a ``search_channel`` tool (1..N refining calls), reasons on top of
    # the material, and cites with clickable deep-links. Smalltalk/decline paths are
    # unchanged either way. Same rollout posture as ``router_model``: default-off so
    # prod behavior only changes when the operator flips the Variable.
    agentic_rag: bool = False

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def llm_key(self) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
        }[self.llm_provider]

    def embed_key(self) -> str:
        return {
            "voyage": self.voyage_api_key,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
        }[self.embed_provider]

    def transcription_key(self) -> str:
        return {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
        }[self.transcription_provider]

    def vision_key(self) -> str:
        return {
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
        }[self.vision_provider]

    def openai_compat_base_url(self, provider: str) -> str | None:
        return _OPENAI_COMPAT_BASE.get(provider)


def load_config() -> Config:
    from .settings_store import apply_to_env

    apply_to_env()  # dev: settings.json overrides env; prod: env wins, settings fills gaps
    llm_provider = os.getenv("YTRAG_LLM_PROVIDER", "anthropic").lower()
    embed_provider = os.getenv("YTRAG_EMBED_PROVIDER", "voyage").lower()
    transcription_provider = os.getenv("YTRAG_TRANSCRIPTION_PROVIDER", "groq").lower()
    vision_provider = os.getenv("YTRAG_VISION_PROVIDER", "openrouter").lower()
    if llm_provider not in _DEFAULT_LLM_MODEL:
        raise ValueError(f"Unknown YTRAG_LLM_PROVIDER: {llm_provider}")
    if embed_provider not in _DEFAULT_EMBED_MODEL:
        raise ValueError(f"Unknown YTRAG_EMBED_PROVIDER: {embed_provider}")
    if transcription_provider not in _DEFAULT_TRANSCRIPTION_MODEL:
        raise ValueError(f"Unknown YTRAG_TRANSCRIPTION_PROVIDER: {transcription_provider}")
    if vision_provider not in _DEFAULT_VISION_MODEL:
        raise ValueError(f"Unknown YTRAG_VISION_PROVIDER: {vision_provider}")

    _max_upload = _max_upload_bytes()
    return Config(
        llm_provider=llm_provider,
        llm_model=os.getenv("YTRAG_LLM_MODEL") or _DEFAULT_LLM_MODEL[llm_provider],
        embed_provider=embed_provider,
        embed_model=os.getenv("YTRAG_EMBED_MODEL") or _DEFAULT_EMBED_MODEL[embed_provider],
        transcription_provider=transcription_provider,
        transcription_model=os.getenv("YTRAG_TRANSCRIPTION_MODEL")
        or _DEFAULT_TRANSCRIPTION_MODEL[transcription_provider],
        vision_provider=vision_provider,
        vision_model=os.getenv("YTRAG_VISION_MODEL") or _DEFAULT_VISION_MODEL[vision_provider],
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        data_dir=Path(os.getenv("YTRAG_DATA_DIR", "data")),
        chroma_dir=Path(os.getenv("YTRAG_CHROMA_DIR", "chroma")),
        transcript_proxy=os.getenv("YTRAG_TRANSCRIPT_PROXY", ""),
        max_upload_bytes=_max_upload,
        max_body_bytes=_max_body_bytes(_max_upload),
        app_domain=_app_domain(),
        # Strip so a whitespace-only Variable (e.g. "  " from a fat-fingered GitHub
        # Variable) is treated as UNSET — otherwise it's truthy and would build the
        # router on a blank slug, silently 400ing every router call (the router's
        # try/except then swallows it, disabling rewrite/smalltalk on every turn).
        router_model=os.getenv("YTRAG_ROUTER_MODEL", "").strip(),
        agentic_rag=_env_truthy("YTRAG_AGENTIC_RAG"),
    )


def _env_truthy(name: str) -> bool:
    """True when env var ``name`` is a truthy flag (same accepted spellings as
    ``YTRAG_TEST_MODE``). Anything else — unset, blank, "0", garbage — is False,
    so a flag can never be flipped ON by accident."""
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _max_upload_bytes() -> int:
    """Document-upload ceiling in bytes from ``YTRAG_MAX_UPLOAD_MB`` (default 20).

    Guards the memory-DoS on the upload handler. A non-numeric or non-positive
    value falls back to the default rather than disabling the limit.
    """
    raw = os.getenv("YTRAG_MAX_UPLOAD_MB", "").strip()
    try:
        mb = float(raw) if raw else 20.0
    except ValueError:
        mb = 20.0
    if mb <= 0:
        mb = 20.0
    return int(mb * 1024 * 1024)


def _max_body_bytes(max_upload_bytes: int) -> int:
    """Hard whole-body ceiling in bytes from ``YTRAG_MAX_BODY_MB`` (default 25).

    A non-numeric/non-positive value falls back to the default. The result is
    never allowed below ``max_upload_bytes`` + 5 MB of multipart overhead, so a
    document upload right at the per-file cap can never be rejected by this coarse
    outer guard (it exists to stop multi-GB floods, not to duplicate the precise
    per-file 413).
    """
    raw = os.getenv("YTRAG_MAX_BODY_MB", "").strip()
    try:
        mb = float(raw) if raw else 25.0
    except ValueError:
        mb = 25.0
    if mb <= 0:
        mb = 25.0
    cap = int(mb * 1024 * 1024)
    floor = max_upload_bytes + 5 * 1024 * 1024
    return max(cap, floor)


def max_block_backoff_sec() -> float:
    """Per-video cumulative block-backoff wall-clock ceiling, in seconds.

    Even in "unbounded" retry mode (``max_retries`` None/<=0) a single video that
    YouTube persistently 429s/bot-checks must not monopolize the one global FIFO
    ingest worker forever and starve every other tenant's queued job. Once a video
    has spent this many cumulative seconds *sleeping in block cool-offs*, it is
    marked failed and skipped so the worker moves on. From
    ``YTRAG_MAX_BLOCK_BACKOFF_SEC`` (default 1800 = 30 min). A non-numeric or
    non-positive value falls back to the default rather than disabling the ceiling.
    """
    raw = os.getenv("YTRAG_MAX_BLOCK_BACKOFF_SEC", "").strip()
    try:
        secs = float(raw) if raw else 1800.0
    except ValueError:
        secs = 1800.0
    if secs <= 0:
        secs = 1800.0
    return secs


def max_block_attempts() -> int:
    """Per-video hard cap on block/backoff attempts in unbounded retry mode.

    The companion to :func:`max_block_backoff_sec` — whichever ceiling trips first
    ends the retry loop for that one video (marked failed, skipped) so the single
    global worker can never be wedged on a persistently blocked video. From
    ``YTRAG_MAX_BLOCK_ATTEMPTS`` (default 8). A non-numeric/non-positive value
    falls back to the default.
    """
    raw = os.getenv("YTRAG_MAX_BLOCK_ATTEMPTS", "").strip()
    try:
        n = int(raw) if raw else 8
    except ValueError:
        n = 8
    if n <= 0:
        n = 8
    return n


def proxy_mb_per_video() -> float:
    """Estimated transcript-proxy bandwidth per PROXIED video, in MB.

    The residential proxy carries far more than the final subtitle file for each
    video (the watch page, the player JSON, and the caption track all transit the
    proxy), so metering the tiny ``.vtt`` size alone under-counts real proxy
    traffic by 1-2 orders of magnitude and would make the per-user monthly ceiling
    (``plans.max_monthly_proxy_mb``) far looser than its label. This flat per-video
    estimate is ADDED to the measured subtitle bytes on every proxied attempt
    (including caption-less / failed ones) so the ceiling tracks reality. Crude on
    purpose. From ``YTRAG_PROXY_MB_PER_VIDEO`` (default 0.5). A non-numeric or
    negative value falls back to the default.
    """
    raw = os.getenv("YTRAG_PROXY_MB_PER_VIDEO", "").strip()
    try:
        mb = float(raw) if raw else 0.5
    except ValueError:
        mb = 0.5
    if mb < 0:
        mb = 0.5
    return mb


def max_retrieval_distance() -> float:
    """Retrieval relevance FLOOR: the max vector distance a hit may have to still
    count as "relevant" context, from ``YTRAG_MAX_RETRIEVAL_DISTANCE``.

    The Chroma collections use cosine space, so ``distance`` is ``1 - cosine`` —
    LOWER means CLOSER (0 = identical, 1 = orthogonal, 2 = opposite). A hit is kept
    only when ``distance <= this ceiling``; if NOTHING clears it the assistant
    declines deterministically instead of hallucinating over irrelevant chunks.

    The default (1.0) is deliberately CONSERVATIVE — it rejects only hits that are
    orthogonal-or-worse to the question (clearly nothing relevant), so genuine
    on-topic queries are never dropped and the existing prompt-level grounding
    still handles the softer "not quite covered" cases. Operators who want a
    stricter floor lower this (e.g. 0.6). A non-numeric or non-positive value
    falls back to the default rather than disabling the floor.
    """
    raw = os.getenv("YTRAG_MAX_RETRIEVAL_DISTANCE", "").strip()
    try:
        d = float(raw) if raw else 1.0
    except ValueError:
        d = 1.0
    # Guard both non-positive AND non-finite (nan/inf): `nan <= 0` is False, so a
    # bare `d <= 0` check would let nan through and `dist <= nan` is always False —
    # the floor would then reject EVERY hit and the bot would decline everything.
    if not math.isfinite(d) or d <= 0:
        d = 1.0
    return d


# Default answer budget (output tokens). The old hardcoded 1024 truncated longer
# grounded answers mid-thought — the single biggest "the bot feels dumb" fix
# (Inc1.4b). Raised to a roomier default; operators can still tune it.
DEFAULT_MAX_OUTPUT_TOKENS = 3000


def max_output_tokens() -> int:
    """Max output tokens for a chat/vision answer, from ``YTRAG_MAX_OUTPUT_TOKENS``.

    Defaults to :data:`DEFAULT_MAX_OUTPUT_TOKENS` (3000) so answers are no longer
    clipped at the old 1024. A non-numeric or non-positive value falls back to the
    default rather than disabling the budget (a 0/negative cap would make the
    provider reject or truncate every answer)."""
    raw = os.getenv("YTRAG_MAX_OUTPUT_TOKENS", "").strip()
    try:
        n = int(raw) if raw else DEFAULT_MAX_OUTPUT_TOKENS
    except ValueError:
        n = DEFAULT_MAX_OUTPUT_TOKENS
    return n if n > 0 else DEFAULT_MAX_OUTPUT_TOKENS


# OpenRouter's unified "reasoning" effort levels. Anything outside this set (incl.
# the empty string, "none", "off") means reasoning is DISABLED — today's behavior.
_VALID_REASONING_EFFORTS = ("low", "medium", "high")


def llm_reasoning_effort() -> str | None:
    """OpenRouter reasoning effort from ``YTRAG_LLM_REASONING_EFFORT``, or None (OFF).

    OFF by default so the request is byte-for-byte today's request. When set to
    ``low``/``medium``/``high`` the OpenAI-compatible (OpenRouter) chat path adds a
    ``reasoning: {effort: ...}`` field so the model "thinks" before answering. Any
    other value (incl. ``none``/``off``/blank) is treated as OFF — the flag never
    silently enables a mode the operator didn't ask for."""
    raw = os.getenv("YTRAG_LLM_REASONING_EFFORT", "").strip().lower()
    return raw if raw in _VALID_REASONING_EFFORTS else None


# Per-mode reasoning effort (BUG-033). The two Telegram answer modes have opposite
# cost/quality postures, so they get opposite reasoning defaults: «Мислення»
# (``thinking``) is the deliberately SLOW, smart mode → ``medium``; «Довідник»
# (``reference``) is the fast strict lookup → ``low``. Each is env-overridable to any
# valid effort; an unset or invalid value falls back to the per-mode default. Any
# OTHER answer_mode — including ``None``, the default flag-gated pipeline — returns
# None, so the global :func:`llm_reasoning_effort` knob and today's hardness gate stay
# byte-for-byte unchanged on every path that is not one of the two named modes.
_MODE_REASONING_DEFAULTS = {"thinking": "medium", "reference": "low"}
_MODE_REASONING_ENV = {
    "thinking": "YTRAG_REASONING_EFFORT_THINKING",
    "reference": "YTRAG_REASONING_EFFORT_REFERENCE",
}


def mode_reasoning_effort(answer_mode: str | None) -> str | None:
    """The reasoning effort for a Telegram answer mode, or None for any other path.

    Returns ``"medium"`` for ``thinking`` and ``"low"`` for ``reference`` by default
    (see :data:`_MODE_REASONING_DEFAULTS`), each overridable via its env var
    (:data:`_MODE_REASONING_ENV`). An unrecognized ``answer_mode`` — crucially
    ``None`` — returns None so the no-mode path's reasoning stays exactly today's."""
    default = _MODE_REASONING_DEFAULTS.get(answer_mode or "")
    if default is None:
        return None
    raw = os.getenv(_MODE_REASONING_ENV[answer_mode], "").strip().lower()
    return raw if raw in _VALID_REASONING_EFFORTS else default


def youtube_api_key() -> str:
    """YouTube Data API v3 key for the channel video-count helper, or "".

    Read from ``YTRAG_YOUTUBE_API_KEY``. This key is used ONLY to fetch a
    channel's approximate public video count (``channels.list`` /
    ``statistics.videoCount``) so the UI can show a determinate context line
    ("Channel ~2,500, indexing 300 newest") — see :mod:`ytrag.youtube_count`.
    It is entirely OPTIONAL: when unset, the count helper falls back to a
    scrape-based estimate, so import + progress work unchanged with no key.
    """
    return os.getenv("YTRAG_YOUTUBE_API_KEY", "").strip()


def telegram_api_base() -> str:
    """Base URL for the Telegram Bot API, from ``YTRAG_TELEGRAM_API_BASE``.

    REQUIRED — there is deliberately NO default. Bot-API calls build
    ``{base}/bot<token>/<method>`` and file downloads build
    ``{base}/file/bot<token>/<path>`` from THIS single base, so one override
    redirects the whole Telegram surface (a local Bot API server, an operator
    proxy, or a fake in tests). We REFUSE to fall back to the real
    ``https://api.telegram.org``: the operator works from a work laptop and must
    never accidentally hit live Telegram, so an unset/blank value RAISES instead
    of defaulting. Called LAZILY at Telegram-operation time (URL construction),
    so the app still boots when Telegram is unused — nothing on the import or
    startup path forces this. Any trailing slash is normalized away so callers
    can safely append ``/bot...`` without producing a double slash.
    """
    raw = os.getenv("YTRAG_TELEGRAM_API_BASE", "").strip()
    if not raw:
        raise RuntimeError(
            "YTRAG_TELEGRAM_API_BASE must be set — refusing to default to the "
            "real Telegram API"
        )
    return raw.rstrip("/")


def telegram_file_api_base() -> str:
    """Base URL for Telegram file downloads (``getFile`` result paths).

    Derived from :func:`telegram_api_base` as ``{base}/file`` so the single
    ``YTRAG_TELEGRAM_API_BASE`` override redirects downloads too — which means
    the "must be set, no real-API default" requirement propagates here: when
    neither var is set, this raises via :func:`telegram_api_base`. An optional
    ``YTRAG_TELEGRAM_FILE_BASE`` overrides only the file host when a deployment
    genuinely splits the two; when unset it tracks the main base. Any trailing
    slash is normalized away.
    """
    raw = os.getenv("YTRAG_TELEGRAM_FILE_BASE", "").strip()
    return (raw or f"{telegram_api_base()}/file").rstrip("/")


def _app_domain() -> str:
    """Canonical app hostname for the CSRF Origin check (or "" when unconfigured).

    Prefers an explicit ``YTRAG_APP_DOMAIN``/``APP_DOMAIN`` (a bare host or a full
    URL); otherwise derives the host from ``YTRAG_OAUTH_REDIRECT_URL`` so the one
    canonical URL the operator already sets doubles as the CSRF anchor. Returns a
    lowercase hostname with any scheme/port/path stripped.
    """
    from urllib.parse import urlsplit

    def _host(value: str) -> str:
        value = (value or "").strip()
        if not value:
            return ""
        # urlsplit needs a netloc: give a bare "example.com[:port]" a "//" prefix.
        if "://" not in value and not value.startswith("//"):
            value = "//" + value
        try:
            return (urlsplit(value).hostname or "").lower()
        except ValueError:
            return ""

    explicit = os.getenv("YTRAG_APP_DOMAIN") or os.getenv("APP_DOMAIN") or ""
    host = _host(explicit)
    if host:
        return host
    return _host(os.getenv("YTRAG_OAUTH_REDIRECT_URL", ""))
