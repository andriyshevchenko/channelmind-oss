"""Per-user chat settings: Managed vs BYOK, and the user's own provider keys.

This is the tenant-scoped counterpart to the operator-only ``settings_store.py``.
Where ``settings.json`` holds the *server's* global provider keys (operator scope),
this module holds each USER's choice of:

* ``mode``: ``"managed"`` (default — runs on the server key, plan budget applies)
  or ``"byok"`` (bring-your-own-key — runs on the user's key, no plan budget); and
* their own chat-provider API key(s).

**Security (Phase C3).** User keys are credentials, so they are encrypted AT REST
with Fernet (AES-128-CBC + HMAC). The Fernet key is derived from a DEDICATED env
var ``YTRAG_KEYSTORE_SECRET`` (never the session secret). If that secret is unset
we *refuse to store* a BYOK key — we never write a plaintext key to disk — so a
misconfigured box degrades to Managed-only rather than leaking credentials. A lost
/ rotated ``YTRAG_KEYSTORE_SECRET`` simply makes every stored key undecryptable,
which we treat as "no key set" (the user re-enters it). Keys are masked to a
boolean flag in every API response and are never logged.

Storage mirrors ``settings_store``/``usage``: one JSON file under the data dir,
keyed by user id, written through the ``safestore`` ``file_lock`` +
``atomic_write_json`` pattern (atomic_write_json chmods 0600 best-effort).
"""
from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

from .accounts import User
from .autosync import DEFAULT_SYNC_FREQ, normalize_freq
from .config import (
    Config,
    default_embed_model,
    default_llm_model,
    default_transcription_model,
    default_vision_model,
)
from .plans import Plan
from .safestore import atomic_write_json, file_lock, read_json

# BYOK is per-CAPABILITY: a user may bring their own key for the chat LLM AND for
# embeddings, transcription and vision. Each tuple below lists the providers valid
# for that capability (mirrors config.py's option lists and the SPA selectors).
# Keys are stored per PROVIDER, not per capability — one ``openai`` key serves any
# capability that can use openai — so ``KEY_PROVIDERS`` (the union) is the set
# ``set_key``/``masked`` accept, while the capability tuples drive resolution.
CHAT_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "openrouter")
EMBED_PROVIDERS: tuple[str, ...] = ("voyage", "openai", "openrouter")
TRANSCRIPTION_PROVIDERS: tuple[str, ...] = ("groq", "openai")
VISION_PROVIDERS: tuple[str, ...] = ("openrouter", "openai")
# Every provider a user may store a key for (union of the capability sets).
KEY_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "openrouter", "voyage", "groq")

MODE_MANAGED = "managed"
MODE_BYOK = "byok"

_KEYSTORE_ENV = "YTRAG_KEYSTORE_SECRET"

# Which Config field carries the key for each provider (all capabilities).
_KEY_FIELD = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
    "voyage": "voyage_api_key",
    "groq": "groq_api_key",
}

# Which Config field names the PROVIDER a non-chat capability uses. This field name
# is shared with the matching :class:`UserSettings` attribute (the owner's CHOSEN
# provider) — resolution reads the user's choice first, then falls back to this cfg
# field. Embeddings stay corpus-pinned (only the KEY is swapped so vectors match);
# vision/transcription may swap provider+model+key to the owner's chosen provider.
_CAPABILITY_PROVIDER_FIELD = {
    "embed": "embed_provider",
    "transcription": "transcription_provider",
    "vision": "vision_provider",
}
# Which Config field carries the MODEL for each non-chat capability (used when a
# vision/transcription swap changes the provider — the new provider's default model
# is pinned in).
_CAPABILITY_MODEL_FIELD = {
    "embed": "embed_model",
    "transcription": "transcription_model",
    "vision": "vision_model",
}
# Valid provider set per capability (drives resolution + the persisted-choice
# validation). Mirrors the tuples above so an unknown/foreign choice can't stick.
_CAPABILITY_PROVIDERS: dict[str, tuple[str, ...]] = {
    "embed": EMBED_PROVIDERS,
    "transcription": TRANSCRIPTION_PROVIDERS,
    "vision": VISION_PROVIDERS,
}
# Default-model resolver per capability (the new provider's default when a swap
# changes the provider). Values are the ``config.default_*_model`` helpers.
_CAPABILITY_MODEL_DEFAULT = {
    "embed": default_embed_model,
    "transcription": default_transcription_model,
    "vision": default_vision_model,
}


class KeystoreUnavailable(RuntimeError):
    """Raised when a BYOK key can't be stored because no keystore secret is set.

    Callers turn this into a clean user-facing error — we NEVER fall back to
    writing a plaintext key to disk.
    """


# ---- encryption --------------------------------------------------------
def keystore_configured() -> bool:
    """True when ``YTRAG_KEYSTORE_SECRET`` is set (BYOK key storage is possible)."""
    return bool(os.getenv(_KEYSTORE_ENV, "").strip())


def _fernet():
    """A :class:`Fernet` derived from ``YTRAG_KEYSTORE_SECRET``, or ``None``.

    The secret is a high-entropy operator-supplied value; we derive the 32-byte
    Fernet key deterministically via SHA-256 so the same secret always yields the
    same key (no stored salt to lose). Returns ``None`` when the secret is unset —
    the signal to refuse BYOK storage.
    """
    secret = os.getenv(_KEYSTORE_ENV, "").strip()
    if not secret:
        return None
    from cryptography.fernet import Fernet

    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _encrypt(plaintext: str):
    """Fernet-encrypt a key. Raises :class:`KeystoreUnavailable` if no secret."""
    f = _fernet()
    if f is None:
        raise KeystoreUnavailable(
            f"{_KEYSTORE_ENV} is not set — cannot securely store provider keys."
        )
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> str | None:
    """Decrypt a stored token, or ``None`` if impossible (no secret / wrong key /
    corrupt). A ``None`` return is treated everywhere as "no key set"."""
    if not token:
        return None
    f = _fernet()
    if f is None:
        return None
    try:
        from cryptography.fernet import InvalidToken

        try:
            return f.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken:
            return None
    except Exception:  # noqa: BLE001 - never let a bad token crash the chat path
        return None


# ---- model -------------------------------------------------------------
@dataclass(frozen=True)
class UserSettings:
    """One user's resolved settings snapshot (keys are still ciphertext here)."""
    mode: str = MODE_MANAGED
    # provider -> Fernet ciphertext (never plaintext on disk / in memory here).
    keys: dict[str, str] | None = None
    # Optional preferred BYOK chat provider; blank = auto-pick from stored keys.
    llm_provider: str = ""
    # Owner's CHOSEN per-capability BYOK providers (blank = server default). The
    # embed choice pins the provider of NEW corpora at create time (existing corpora
    # stay pinned to their own provider); vision/transcription choices swap the
    # provider+model+key at resolution time. Each is validated against its capability
    # set on read, so a stale/foreign value degrades to blank (server default).
    embed_provider: str = ""
    transcription_provider: str = ""
    vision_provider: str = ""
    # User-level default auto-sync frequency for NEW YouTube sources (Phase K):
    # off|daily|weekly|monthly. New sources inherit this at add time; a source can
    # override it per channel. Default "off" keeps auto-sync opt-in.
    sync_freq: str = DEFAULT_SYNC_FREQ
    # User-facing notification emails (e.g. ingest-complete via Resend). Default
    # True preserves current behavior — existing users keep receiving them. When
    # False, the single send boundary in ``notify.py`` suppresses them.
    # Transactional/auth mail does NOT flow through this toggle.
    email_notifications: bool = True

    def enc_keys(self) -> dict[str, str]:
        return self.keys or {}


class UserSettingsStore:
    """JSON-backed per-user settings, keyed by user id (``user_settings.json``)."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._registry = self._root / "user_settings.json"

    def _load(self) -> dict:
        return read_json(self._registry, {}) or {}

    def _raw(self, user_id: str) -> dict:
        return self._load().get(user_id) or {}

    def _capability_choice(self, rec: dict, capability: str) -> str:
        """The stored per-capability provider choice, normalized + validated against
        that capability's set (unknown/foreign -> blank = server default)."""
        val = (rec.get(_CAPABILITY_PROVIDER_FIELD[capability]) or "").strip().lower()
        return val if val in _CAPABILITY_PROVIDERS[capability] else ""

    def get(self, user_id: str) -> UserSettings:
        rec = self._raw(user_id)
        mode = rec.get("mode") or MODE_MANAGED
        if mode not in (MODE_MANAGED, MODE_BYOK):
            mode = MODE_MANAGED
        # Notifications default ON: legacy records lack the key, so ``True`` here
        # keeps their current behavior. Only an explicit stored ``False`` suppresses;
        # a non-bool legacy value degrades to the safe default rather than sticking.
        raw_notif = rec.get("email_notifications", True)
        email_notifications = raw_notif if isinstance(raw_notif, bool) else True
        return UserSettings(
            mode=mode,
            keys={k: v for k, v in (rec.get("keys") or {}).items() if v},
            llm_provider=(rec.get("llm_provider") or "").strip().lower(),
            embed_provider=self._capability_choice(rec, "embed"),
            transcription_provider=self._capability_choice(rec, "transcription"),
            vision_provider=self._capability_choice(rec, "vision"),
            sync_freq=normalize_freq(rec.get("sync_freq")) or DEFAULT_SYNC_FREQ,
            email_notifications=email_notifications,
        )

    def has_key(self, user_id: str, provider: str) -> bool:
        """True only if a key for ``provider`` is stored AND currently decryptable."""
        token = self.get(user_id).enc_keys().get(provider)
        return _decrypt(token or "") is not None

    def decrypted_key(self, user_id: str, provider: str) -> str | None:
        return _decrypt(self.get(user_id).enc_keys().get(provider) or "")

    def set_mode(self, user_id: str, mode: str) -> None:
        if mode not in (MODE_MANAGED, MODE_BYOK):
            raise ValueError(f"unknown mode: {mode!r}")
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            rec["mode"] = mode
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def set_key(self, user_id: str, provider: str, key: str) -> None:
        """Store (encrypted) or clear a BYOK key for any supported provider.

        Accepts any provider in :data:`KEY_PROVIDERS` (chat, embedding,
        transcription and vision providers) — keys are stored per provider, so one
        stored key serves every capability that can use that provider. A blank
        ``key`` clears any stored key for that provider. A non-blank key is
        Fernet-encrypted before it touches disk; if the keystore secret is unset
        this raises :class:`KeystoreUnavailable` and writes NOTHING.
        """
        if provider not in KEY_PROVIDERS:
            raise ValueError(f"unsupported provider: {provider!r}")
        key = (key or "").strip()
        # Encrypt BEFORE taking the lock so a KeystoreUnavailable never leaves a
        # half-written record (and never writes plaintext).
        token = _encrypt(key) if key else ""
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            keys = dict(rec.get("keys") or {})
            if token:
                keys[provider] = token
            else:
                keys.pop(provider, None)
            rec["keys"] = keys
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def set_llm_provider(self, user_id: str, provider: str) -> None:
        """Set the preferred BYOK chat provider (blank = auto-pick)."""
        provider = (provider or "").strip().lower()
        if provider and provider not in CHAT_PROVIDERS:
            raise ValueError(f"unsupported chat provider: {provider!r}")
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            rec["llm_provider"] = provider
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def set_capability_provider(self, user_id: str, capability: str, provider: str) -> None:
        """Persist the owner's CHOSEN provider for a non-chat capability (``embed`` /
        ``transcription`` / ``vision``). A blank clears the choice (server default);
        a non-blank value must be valid for that capability."""
        if capability not in _CAPABILITY_PROVIDER_FIELD:
            raise ValueError(f"unknown capability: {capability!r}")
        provider = (provider or "").strip().lower()
        if provider and provider not in _CAPABILITY_PROVIDERS[capability]:
            raise ValueError(f"unsupported {capability} provider: {provider!r}")
        field = _CAPABILITY_PROVIDER_FIELD[capability]
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            rec[field] = provider
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def set_sync_freq(self, user_id: str, freq: str) -> None:
        """Set this user's default auto-sync frequency for new YouTube sources."""
        f = normalize_freq(freq)
        if not f:
            raise ValueError(f"unknown sync frequency: {freq!r}")
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            rec["sync_freq"] = f
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def set_email_notifications(self, user_id: str, enabled: bool) -> None:
        """Enable/disable user-facing notification emails for this user.

        Stored as an explicit bool; the send boundary in ``notify.py`` honors it.
        """
        with file_lock(self._registry):
            data = self._load()
            rec = data.get(user_id) or {}
            rec["email_notifications"] = bool(enabled)
            data[user_id] = rec
            atomic_write_json(self._registry, data)

    def delete(self, user_id: str) -> bool:
        """Drop a user's settings/keys (account erasure). True if a record went."""
        with file_lock(self._registry):
            data = self._load()
            if data.pop(user_id, None) is None:
                return False
            atomic_write_json(self._registry, data)
            return True

    def masked(self, user_id: str, plan: Plan | None = None) -> dict:
        """Display view: mode + per-provider 'key set: yes/no', never a raw key.

        When a ``plan`` is passed, the toggles it exposes (``byok_enabled`` /
        ``managed_enabled``) ride along so the UI can hide BYOK when disabled.
        ``keystore_ready`` tells the UI whether BYOK storage is even possible on
        this deployment.
        """
        us = self.get(user_id)
        # "set" means a decryptable key exists — a key encrypted under a lost
        # secret reads as not-set, which is exactly what the user experiences.
        # Covers every capability's providers (chat + embedding/transcription/
        # vision) so the UI can reflect a "key set" state for each.
        keys_set = {p: self.has_key(user_id, p) for p in KEY_PROVIDERS}
        out: dict = {
            "mode": us.mode,
            "llm_provider": us.llm_provider,
            # Owner's chosen per-capability BYOK providers (blank = server default),
            # so the SPA can hydrate each selector from the server's truth.
            "embed_provider": us.embed_provider,
            "transcription_provider": us.transcription_provider,
            "vision_provider": us.vision_provider,
            "sync_freq": us.sync_freq,
            "email_notifications": us.email_notifications,
            "keys_set": keys_set,
            "keystore_ready": keystore_configured(),
        }
        if plan is not None:
            out["byok_enabled"] = plan.byok_enabled
            out["managed_enabled"] = plan.managed_enabled
        return out


# ---- chat key/provider resolution -------------------------------------
@dataclass(frozen=True)
class ChatResolution:
    """The outcome of resolving which provider/key a chat call must use.

    * ``mode`` — ``"byok"`` or ``"managed"`` (the mode actually applied, after any
      safe fallback), or ``"none"`` when nothing is usable.
    * ``config`` — the effective :class:`Config` to run the chat with (server keys
      for Managed; the user's provider/key swapped in for BYOK). ``None`` on error.
    * ``budget_applies`` — whether the plan's monthly Managed budget must be
      enforced (True only for Managed).
    * ``error`` — a clean, user-facing message when no mode is usable; ``None``
      otherwise.
    """
    mode: str
    config: Config | None
    budget_applies: bool
    error: str | None = None


def _byok_config(cfg: Config, provider: str, key: str) -> Config:
    """A Config with the chat provider/model/key swapped to the user's BYOK key.

    Only the chat LLM fields change — embedding/transcription keep the server
    config (the corpus is pinned to the server's embed model)."""
    model = cfg.llm_model if provider == cfg.llm_provider else default_llm_model(provider)
    return replace(
        cfg,
        llm_provider=provider,
        llm_model=model,
        **{_KEY_FIELD[provider]: key},
    )


def resolve_chat_config(
    cfg: Config,
    user: User,
    plan: Plan,
    store: UserSettingsStore | None = None,
) -> ChatResolution:
    """Decide which provider/key a chat for ``user`` runs on, given their ``plan``.

    Precedence:

    1. **BYOK** — only when the plan enables it AND the user selected ``byok`` AND
       they have a usable (decryptable) key. Runs on the user's key; the plan
       budget does NOT apply.
    2. **Managed** — otherwise, when the plan enables it. Runs on the server key;
       the plan's ``max_monthly_cost_usd`` budget applies.
    3. **Neither usable** — a clean error (e.g. a plan with managed disabled and
       the user has no BYOK key). The caller returns it; no model call happens.

    A mode the plan disables never wins: a ``byok`` selection on a plan without
    ``byok_enabled`` silently falls back to Managed; ``managed`` on a plan without
    ``managed_enabled`` falls back to a usable BYOK key if present.
    """
    store = store or UserSettingsStore(cfg.data_dir)
    us = store.get(user.id)

    def pick_byok_provider() -> str | None:
        # Preferred provider first (if it has a usable key), else the first chat
        # provider the user has a usable key for.
        pref = us.llm_provider
        order = ([pref] if pref in CHAT_PROVIDERS else []) + [
            p for p in CHAT_PROVIDERS if p != pref
        ]
        for p in order:
            if store.decrypted_key(user.id, p):
                return p
        return None

    wants_byok = us.mode == MODE_BYOK
    byok_provider = pick_byok_provider() if plan.byok_enabled else None

    # 1) BYOK when selected + enabled + a usable key exists.
    if plan.byok_enabled and wants_byok and byok_provider:
        key = store.decrypted_key(user.id, byok_provider) or ""
        return ChatResolution(
            mode=MODE_BYOK,
            config=_byok_config(cfg, byok_provider, key),
            budget_applies=False,
        )

    # 2) Managed when the plan allows it.
    if plan.managed_enabled:
        return ChatResolution(mode=MODE_MANAGED, config=cfg, budget_applies=True)

    # 2b) Managed disabled but the user has a usable BYOK key -> use it (fallback).
    if plan.byok_enabled and byok_provider:
        key = store.decrypted_key(user.id, byok_provider) or ""
        return ChatResolution(
            mode=MODE_BYOK,
            config=_byok_config(cfg, byok_provider, key),
            budget_applies=False,
        )

    # 3) Nothing usable — a clean error, never a model call.
    return ChatResolution(
        mode="none",
        config=None,
        budget_applies=False,
        error="Chat is unavailable for your plan until you add your own API key.",
    )


# ---- shared resolve + budget gate (used by web AND Telegram) -----------
@dataclass(frozen=True)
class ResolvedChat:
    """Resolution + monthly-budget outcome for a billed user's chat.

    The single gate both the web endpoints and the Telegram poller run before
    any model call, so BYOK key selection and the Managed monthly-budget 429 are
    enforced identically on every path.

    * ``config`` — the effective :class:`Config` to run the chat with (server keys
      for Managed, the user's BYOK provider/key swapped in), or ``None`` when the
      chat must be blocked.
    * ``status`` — ``"ok"`` (proceed), ``"over_budget"`` (Managed budget hit —
      the web layer maps this to 429), or ``"unavailable"`` (no usable mode for
      the plan — the web layer maps this to 400).
    * ``error`` — a user-facing message when ``status`` is not ``"ok"``.
    """
    config: Config | None
    status: str
    error: str | None = None


def resolve_chat_for_user(
    cfg: Config,
    user: User,
    plan: Plan,
    store: UserSettingsStore | None = None,
) -> ResolvedChat:
    """Resolve which provider/key ``user``'s chat runs on AND enforce the budget.

    Wraps :func:`resolve_chat_config` (Managed vs BYOK key selection) with the
    plan's monthly Managed-spend gate, so callers get one decision:

    * BYOK / Managed both usable and under budget -> ``status="ok"`` with the
      effective ``config``.
    * Managed selected but the user is at/over their plan's monthly budget ->
      ``status="over_budget"`` (a BYOK chat runs on the user's own key and is
      never budget-gated).
    * No usable mode for the plan -> ``status="unavailable"``.

    The budget is checked BEFORE any model call and ONLY for Managed. The budget
    accessor (``quotas.over_budget``) is imported lazily to avoid an import cycle
    (``quotas`` pulls in bots/jobs/usage).
    """
    res = resolve_chat_config(cfg, user, plan, store=store)
    if res.error is not None or res.config is None:
        return ResolvedChat(
            config=None,
            status="unavailable",
            error=res.error or "Chat is unavailable for your plan.",
        )
    if res.budget_applies:
        from . import quotas  # lazy: avoid quotas<->user_settings import cycle

        if quotas.over_budget(cfg, user):
            return ResolvedChat(
                config=None,
                status="over_budget",
                error="Monthly usage limit reached for this plan.",
            )
    return ResolvedChat(config=res.config, status="ok")


# ---- non-chat capability (embed / transcription / vision) BYOK ---------
def _capability_provider(cfg: Config, us: UserSettings, capability: str) -> str:
    """The provider a non-chat ``capability`` resolves to for this owner.

    * **embed** — ALWAYS the ``cfg`` (corpus-pinned) provider. A corpus is pinned to
      one embed provider for the life of its vectors; the owner's *chosen* embed
      provider only pins NEW corpora (see ``bot_service.create_bot``), never an
      existing one, so resolution must not honor it here.
    * **vision / transcription** — the owner's CHOSEN provider when set + valid, else
      the ``cfg`` (server-config) provider.
    """
    if capability == "embed":
        return getattr(cfg, _CAPABILITY_PROVIDER_FIELD["embed"])
    chosen = (getattr(us, _CAPABILITY_PROVIDER_FIELD[capability], "") or "").strip().lower()
    if chosen and chosen in _CAPABILITY_PROVIDERS[capability]:
        return chosen
    return getattr(cfg, _CAPABILITY_PROVIDER_FIELD[capability])


def _resolve_capability(
    cfg: Config, user: User, plan: Plan, capability: str, store: UserSettingsStore | None
) -> tuple[str | None, str | None]:
    """Resolve ``(provider, key)`` for a non-chat capability, or ``(None, None)`` when
    the server key/provider should be used (graceful fallback). A key is returned
    ONLY when the plan enables BYOK, the user selected ``byok`` mode, and a
    decryptable key for the resolved provider exists."""
    if capability not in _CAPABILITY_PROVIDER_FIELD:
        raise ValueError(f"unknown capability: {capability!r}")
    if not plan.byok_enabled:
        return None, None
    store = store or UserSettingsStore(cfg.data_dir)
    us = store.get(user.id)
    if us.mode != MODE_BYOK:
        return None, None
    provider = _capability_provider(cfg, us, capability)
    key = store.decrypted_key(user.id, provider)
    return (provider, key) if key else (None, None)


def byok_capability_key(
    cfg: Config,
    user: User,
    plan: Plan,
    capability: str,
    store: UserSettingsStore | None = None,
) -> str | None:
    """The bot OWNER's usable BYOK key for the provider a non-chat ``capability``
    resolves to, or ``None`` when the server key should be used instead.

    ``capability`` is ``"embed" | "transcription" | "vision"``. For embeddings the
    caller passes a corpus-pinned cfg so the pinned provider governs (only the KEY
    is ever swapped — vectors stay compatible). Vision/transcription resolve to the
    owner's CHOSEN provider when set, else the server-config provider. Returns a key
    ONLY when the plan enables BYOK, the user selected ``byok`` mode, and a
    decryptable key for that provider exists — otherwise ``None`` (graceful fallback).
    """
    return _resolve_capability(cfg, user, plan, capability, store)[1]


def apply_byok_capability_key(
    cfg: Config,
    user: User,
    plan: Plan,
    capability: str,
    store: UserSettingsStore | None = None,
) -> Config:
    """Return ``cfg`` with the owner's BYOK settings swapped in for ``capability``
    when a usable key exists; otherwise ``cfg`` unchanged (server key/provider —
    graceful fallback). BYOK billing stays consistent with chat: a resolved owner
    key runs on the owner's own credentials and is never plan-budget-gated.

    * **embed** — swaps ONLY the KEY; the corpus-pinned provider/model are kept so
      the query is embedded with the same model the vectors were built with.
    * **vision / transcription** — swaps provider + default model + key when the
      owner's chosen provider differs from ``cfg``; when it matches, only the KEY
      changes (preserving any operator model override).
    """
    provider, key = _resolve_capability(cfg, user, plan, capability, store)
    if not key:
        return cfg
    field = _KEY_FIELD.get(provider or "")
    if not field:
        return cfg
    if capability == "embed" or provider == getattr(cfg, _CAPABILITY_PROVIDER_FIELD[capability]):
        # Corpus-pinned embed, or same provider as cfg: swap the KEY only.
        return replace(cfg, **{field: key})
    # Provider changed (vision/transcription): pin the new provider + its default
    # model alongside the owner's key.
    return replace(
        cfg,
        **{
            _CAPABILITY_PROVIDER_FIELD[capability]: provider,
            _CAPABILITY_MODEL_FIELD[capability]: _CAPABILITY_MODEL_DEFAULT[capability](provider),
            field: key,
        },
    )
