"""Optional JSON settings that override environment variables at runtime.

The web UI writes connection settings here so users don't have to edit .env.
Values found in this file take precedence over environment variables.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .safestore import atomic_write_json, file_lock, read_json

SETTINGS_PATH = Path(os.getenv("YTRAG_SETTINGS_FILE", "settings.json"))

# Keys the UI is allowed to manage, mapped to their env var names.
FIELDS = {
    "llm_provider": "YTRAG_LLM_PROVIDER",
    "llm_model": "YTRAG_LLM_MODEL",
    "embed_provider": "YTRAG_EMBED_PROVIDER",
    "embed_model": "YTRAG_EMBED_MODEL",
    "transcription_provider": "YTRAG_TRANSCRIPTION_PROVIDER",
    "transcription_model": "YTRAG_TRANSCRIPTION_MODEL",
    "vision_provider": "YTRAG_VISION_PROVIDER",
    "vision_model": "YTRAG_VISION_MODEL",
    "transcript_proxy": "YTRAG_TRANSCRIPT_PROXY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "voyage_api_key": "VOYAGE_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    # Google OAuth + web session (multi-tenant login).
    "google_client_id": "GOOGLE_CLIENT_ID",
    "google_client_secret": "GOOGLE_CLIENT_SECRET",
    "oauth_redirect_url": "YTRAG_OAUTH_REDIRECT_URL",
    "session_secret": "YTRAG_SESSION_SECRET",
}
_SECRET_FIELDS = {f for f in FIELDS if f.endswith("_api_key")} | {
    "google_client_secret",
    "session_secret",
}

# Bot persona is app data (not an env-mapped connection field). It lives in the
# same settings file but is managed through its own helpers.
_PERSONA_KEY = "bot_persona"
_PERSONA_DESC_KEY = "bot_description"


def load_settings() -> dict:
    return read_json(SETTINGS_PATH, {}) or {}


def load_persona() -> dict:
    data = load_settings()
    return {
        "persona": data.get(_PERSONA_KEY, "") or "",
        "description": data.get(_PERSONA_DESC_KEY, "") or "",
    }


def save_persona(persona: str, description: str | None = None) -> None:
    with file_lock(SETTINGS_PATH):
        current = read_json(SETTINGS_PATH, {}) or {}
        current[_PERSONA_KEY] = persona or ""
        if description is not None:
            current[_PERSONA_DESC_KEY] = description
        atomic_write_json(SETTINGS_PATH, current)


def save_settings(data: dict) -> None:
    with file_lock(SETTINGS_PATH):
        current = read_json(SETTINGS_PATH, {}) or {}
        for key, value in data.items():
            if key not in FIELDS:
                continue
            # Ignore blank secret submissions so existing keys aren't wiped.
            if key in _SECRET_FIELDS and not value:
                continue
            current[key] = value
        atomic_write_json(SETTINGS_PATH, current)


def apply_to_env() -> None:
    """Push saved settings into os.environ so load_config() picks them up.

    Precedence follows 12-factor in production: real environment variables win
    and settings.json only fills gaps (setdefault). In development the UI is the
    source of truth, so settings.json overrides the environment.
    """
    from .config import is_production

    prod = is_production()
    for key, value in load_settings().items():
        env = FIELDS.get(key)
        if env and value:
            if prod:
                os.environ.setdefault(env, str(value))
            else:
                os.environ[env] = str(value)


def _mask_proxy(url: str) -> str:
    """Redact embedded credentials in a proxy URL for display (keep scheme/host/port)."""
    if not url:
        return ""
    return re.sub(r"://[^/@]*@", "://***@", url)


def masked_settings() -> dict:
    """Settings for display: secrets reduced to a boolean 'is set' flag."""
    data = load_settings()
    out = {}
    for key in FIELDS:
        if key in _SECRET_FIELDS:
            out[key + "_set"] = bool(data.get(key))
        elif key == "transcript_proxy":
            out[key] = _mask_proxy(data.get(key, ""))
        else:
            out[key] = data.get(key, "")
    return out
