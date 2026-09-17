"""Bridge each bot to its own Telegram bot via long-poll getUpdates.

No public URL exists (the app runs locally), so webhooks are impossible; we
long-poll getUpdates per token in a daemon thread instead. One Telegram bot maps
to exactly one assistant: the pasted BotFather token is the identity, and every
inbound text is answered by ``bot_service.chat`` so replies stay grounded in that
bot's corpus and persona. Bots are re-read fresh on every message because persona
and sources mutate through the UI while a poller runs.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from telegramify_markdown import convert, split_entities

from . import bot_service
from .config import load_config, telegram_api_base, telegram_file_api_base
from .share_limits import sanitize_guest_history
from .telegram_conv import MODE_REFERENCE, MODE_THINKING, TelegramConvStore
from .transcribe import make_transcriber

_MSG_LIMIT = 4096


def _api_url(token: str, method: str) -> str:
    """Build a Telegram Bot API endpoint from the configurable base."""
    return f"{telegram_api_base()}/bot{token}/{method}"


def _file_url(token: str, file_path: str) -> str:
    """Build a Telegram file-download URL from the configurable file base."""
    return f"{telegram_file_api_base()}/bot{token}/{file_path}"


def _seconds_to_ts(sec: float) -> str:
    s = int(sec or 0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _deep_link(url: str, start: float) -> str:
    """Append a YouTube time offset so the link jumps to the cited moment."""
    if not url or not start or start <= 0:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(start)}s"


BOTFATHER_INSTRUCTIONS = """\
### Connect this bot to Telegram

1. Open **@BotFather** in Telegram.
2. Send **/newbot** and pick a display name, then a username ending in `bot`.
3. Copy the **HTTP API token** BotFather gives you.
4. Paste that token into this bot's **Telegram** panel here and click **Save**.
5. Open your new bot in Telegram and start chatting.
"""


def _call(token: str, method: str, params: dict, timeout: float = 30.0) -> dict:
    """POST to the Telegram Bot API; return the parsed JSON envelope."""
    data = urllib.parse.urlencode(params).encode()
    url = _api_url(token, method)
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get_file_path(token: str, file_id: str) -> str | None:
    """Resolve a Telegram file_id to its server-side file_path via getFile."""
    try:
        res = _call(token, "getFile", {"file_id": file_id}, timeout=15.0)
    except Exception:  # noqa: BLE001 - transient network/API failure
        return None
    if not res.get("ok"):
        return None
    return (res.get("result") or {}).get("file_path")


def _download_file(token: str, file_path: str) -> bytes:
    """Download a Telegram file by its file_path."""
    url = _file_url(token, file_path)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _guess_image_mime(file_path: str) -> str:
    """Map a Telegram file_path extension to an image MIME; default JPEG."""
    lower = (file_path or "").lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _audio_filename(file_path: str) -> str:
    """Basename of the Telegram file_path (carries the real extension, e.g. .oga)
    so the transcription SDK can sniff the content type; safe default otherwise."""
    return (file_path or "").rsplit("/", 1)[-1] or "audio.ogg"


def verify_token(token: str) -> dict:
    """Return getMe bot info on success; raise ValueError on a bad token."""
    try:
        res = _call(token, "getMe", {}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - any failure means unusable token
        raise ValueError("Invalid token") from exc
    if not res.get("ok"):
        raise ValueError("Invalid token")
    return res["result"]


def _preview_options(url: str | None) -> str:
    """Telegram link_preview_options JSON: show the pinned url, else no preview."""
    return json.dumps({"url": url} if url else {"is_disabled": True})


def _send_message(token: str, chat_id: int, text: str, preview_url: str | None = None,
                  *, reply_markup: dict | None = None, timeout: float = 30.0) -> None:
    """Render standard Markdown into Telegram text + entities and send it.

    ``convert`` parses the LLM's CommonMark (## headings, **bold**, lists,
    [links]) into plain text plus a list of Telegram MessageEntity offsets.
    We send those entities instead of a ``parse_mode`` string, which sidesteps
    the whole class of MarkdownV2 escaping/parse failures (e.g. the LLM nesting
    **bold** inside a ### heading yields literal ``**`` under the string path and
    gets rejected). ``split_entities`` chops on the 4096 limit while recomputing
    offsets per chunk. When ``preview_url`` is given we pin the link preview to
    that single most-relevant video on the first chunk (via
    ``link_preview_options.url``, which previews it regardless of link order in the
    text); every other chunk suppresses previews so source links don't each expand
    into a card. Any render/delivery failure falls back to plain text."""
    try:
        rendered, entities = convert(text)
        chunks = split_entities(rendered, entities, _MSG_LIMIT)
    except Exception:  # noqa: BLE001 - fall back to plain text on any render error
        chunks = None

    delivered = 0
    for i, (chunk_text, chunk_entities) in enumerate(chunks or []):
        params = {
            "chat_id": chat_id, "text": chunk_text,
            "link_preview_options": _preview_options(preview_url if i == 0 else None),
        }
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        if chunk_entities:
            params["entities"] = json.dumps([e.to_dict() for e in chunk_entities])
        try:
            res = _call(token, "sendMessage", params, timeout=timeout)
        except Exception:  # noqa: BLE001
            break
        if not res.get("ok"):
            break
        delivered += 1
    if delivered:
        return  # fully or partially sent — don't re-post as plain (avoids dupes)

    try:
        params = {
            "chat_id": chat_id, "text": text[:_MSG_LIMIT],
            "link_preview_options": _preview_options(preview_url),
        }
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        _call(token, "sendMessage", params, timeout=timeout)
    except Exception:  # noqa: BLE001 - transient; loop continues
        pass


def _send_message_async(token: str, chat_id: int, text: str,
                        preview_url: str | None = None, *, reply_markup: dict | None = None,
                        timeout: float = 10.0) -> None:
    """Deliver a short control message OFF the caller thread so a slow Telegram
    sendMessage can never stall the single long-poll dispatcher (L1).

    The poll loop uses this for its own replies (``/start``, ``/reset``, the
    "still thinking" busy note, the spawn-fail error) — none of which need to
    block reading the next update. A short ``timeout`` bounds the spawned worker,
    and every failure (including a rare thread-start failure) is swallowed so the
    poller is never affected. Long answer sends stay on their own worker thread
    via the unchanged synchronous ``_send_message``."""
    try:
        threading.Thread(
            target=_send_message,
            args=(token, chat_id, text),
            kwargs={"preview_url": preview_url, "reply_markup": reply_markup, "timeout": timeout},
            name=f"tg-ctrl-send-{chat_id}", daemon=True,
        ).start()
    except Exception:  # noqa: BLE001 - best-effort control send; never touch the poller
        pass


def _send_chat_action(token: str, chat_id: int, action: str = "typing") -> None:
    """Fire a chat action (e.g. the 'typing…' bubble). Best-effort, never raises."""
    try:
        _call(token, "sendChatAction", {"chat_id": chat_id, "action": action}, timeout=10.0)
    except Exception:  # noqa: BLE001 - cosmetic; ignore any failure
        pass


def _typing_keepalive(token: str, chat_id: int, stop: threading.Event) -> None:
    """Refresh the 'typing…' bubble every ~4s until stopped.

    Telegram clears the indicator after ~5s, but answers take ~10s, so a single
    action would lapse mid-wait and look worse than none. This re-sends until the
    answer is ready."""
    while not stop.is_set():
        _send_chat_action(token, chat_id)
        stop.wait(4.0)


# Progressive-streaming cadence (Inc1.4b). Telegram rate-limits edits, so we edit
# the live message only after ENOUGH new text has arrived AND enough time passed —
# except the very first edit, which fires as soon as there's visible progress so
# the placeholder turns into real text quickly.
_STREAM_EDIT_MIN_CHARS = 64
_STREAM_EDIT_MIN_INTERVAL = 1.2  # seconds


def _should_stream_edit(
    answer_len: int, last_len: int, edits_done: int, now: float, last_edit: float,
) -> bool:
    """Whether to push another progressive edit given growth + elapsed time."""
    if answer_len - last_len < _STREAM_EDIT_MIN_CHARS:
        return False
    if edits_done == 0:
        return True  # first visible progress — don't wait on the interval
    return now - last_edit >= _STREAM_EDIT_MIN_INTERVAL


def _send_placeholder(token: str, chat_id: int, text: str) -> int | None:
    """Send the initial placeholder bubble and return its message_id (or None).

    None means the send failed — the caller then falls back to a single normal
    send at the end instead of trying to edit a message that doesn't exist."""
    try:
        res = _call(token, "sendMessage", {
            "chat_id": chat_id, "text": text[:_MSG_LIMIT],
            "link_preview_options": _preview_options(None),
        }, timeout=15.0)
    except Exception:  # noqa: BLE001 - transient; caller falls back to a full send
        return None
    if not res.get("ok"):
        return None
    return (res.get("result") or {}).get("message_id")


def _edit_message(token: str, chat_id: int, message_id: int, text: str) -> None:
    """Best-effort PLAIN-text edit used for live progress (no entities/preview).

    Plain text mid-stream sidesteps the markdown-parse failures a half-written
    answer would trigger; the final render (with entities + sources) is applied
    once by :func:`_finalize_stream_message`. A 'message is not modified' reply or
    any transient error is swallowed — progress edits are cosmetic."""
    try:
        _call(token, "editMessageText", {
            "chat_id": chat_id, "message_id": message_id, "text": text[:_MSG_LIMIT],
            "link_preview_options": _preview_options(None),
        }, timeout=15.0)
    except Exception:  # noqa: BLE001 - cosmetic live update; never raise
        pass


def _edit_rendered_chunk(token: str, chat_id: int, message_id: int,
                         text: str, entities, preview_url: str | None) -> bool:
    """Edit ``message_id`` to a fully rendered chunk (entities + pinned preview)."""
    params = {
        "chat_id": chat_id, "message_id": message_id, "text": text,
        "link_preview_options": _preview_options(preview_url),
    }
    if entities:
        params["entities"] = json.dumps([e.to_dict() for e in entities])
    try:
        res = _call(token, "editMessageText", params, timeout=30.0)
    except Exception:  # noqa: BLE001
        return False
    return bool(res.get("ok"))


def _finalize_stream_message(token: str, chat_id: int, message_id: int | None,
                             text: str, preview_url: str | None = None) -> None:
    """Replace the live placeholder with the final, fully rendered answer.

    Renders the accumulated Markdown into Telegram text + entities and splits on
    the 4096 limit exactly like :func:`_send_message`. The FIRST chunk edits the
    live message in place (so the streamed bubble becomes the final answer); any
    overflow chunks are sent as follow-ups. Every failure path falls back to a
    plain full send so the user always receives the complete answer."""
    if message_id is None:
        _send_message(token, chat_id, text, preview_url=preview_url)
        return
    try:
        rendered, entities = convert(text)
        chunks = split_entities(rendered, entities, _MSG_LIMIT)
    except Exception:  # noqa: BLE001 - render failed: best-effort plain edit
        chunks = None
    if not chunks:
        _edit_message(token, chat_id, message_id, text)
        return
    first_text, first_entities = chunks[0]
    if not _edit_rendered_chunk(token, chat_id, message_id, first_text, first_entities,
                                preview_url):
        _send_message(token, chat_id, text, preview_url=preview_url)
        return
    for chunk_text, chunk_entities in chunks[1:]:
        params = {
            "chat_id": chat_id, "text": chunk_text,
            "link_preview_options": _preview_options(None),
        }
        if chunk_entities:
            params["entities"] = json.dumps([e.to_dict() for e in chunk_entities])
        try:
            _call(token, "sendMessage", params, timeout=30.0)
        except Exception:  # noqa: BLE001 - one overflow chunk failing stops the tail
            break


def _final_answer_text(answer: str, sources: list | None = None) -> str:
    """Return the model answer without a second, retrieval-generated source footer.

    Citations now live in the model's deliberate one-source quote blocks. ``sources``
    remains an argument because callers still use it to select the Telegram preview,
    but appending it here would duplicate (and can contradict) those cited blocks.
    """
    del sources
    return answer or "(no answer)"


def _preview_url(hits: list) -> str | None:
    """Deep-link to the single most-relevant video: the top-ranked hit with a url.

    Hits arrive best-first from retrieval, so the first one carrying a YouTube url
    is the video worth previewing. Document hits (no url) are skipped."""
    for h in hits or []:
        meta = (h or {}).get("meta") or {}
        url = (meta.get("url") or "").strip()
        if url:
            return _deep_link(url, meta.get("start") or 0)
    return None


# Service phrases the bot shows itself (not model output), keyed by the bot's
# configured language. Blank language keeps the Ukrainian default (matching the
# current bots); a set-but-unlisted language falls back to a generic English
# phrase, while the model's actual replies still honour the language via the
# system prompt.
_BUSY_PHRASES = {
    "": "зачекай, ще думаю",
    "uk": "зачекай, ще думаю",
    "ua": "зачекай, ще думаю",
    "ukrainian": "зачекай, ще думаю",
    "українська": "зачекай, ще думаю",
    "en": "hang on — still thinking",
    "english": "hang on — still thinking",
}
_BUSY_FALLBACK = "hang on — still thinking"


def _busy_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _BUSY_PHRASES.get(lang, _BUSY_FALLBACK)


# Shown when a photo arrives but no vision model is configured (cfg.vision_key()
# is empty). Same language keying as _BUSY_PHRASES.
_NO_VISION_PHRASES = {
    "": "я поки не вмію дивитися на зображення",
    "uk": "я поки не вмію дивитися на зображення",
    "ua": "я поки не вмію дивитися на зображення",
    "ukrainian": "я поки не вмію дивитися на зображення",
    "українська": "я поки не вмію дивитися на зображення",
    "en": "I can't look at images yet.",
    "english": "I can't look at images yet.",
}
_NO_VISION_FALLBACK = "I can't look at images yet."


def _no_vision_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _NO_VISION_PHRASES.get(lang, _NO_VISION_FALLBACK)


# Shown when a voice/audio message arrives but no transcription model is
# configured (cfg.transcription_key() is empty). Same language keying as above.
_NO_AUDIO_PHRASES = {
    "": "я поки не вмію слухати аудіо",
    "uk": "я поки не вмію слухати аудіо",
    "ua": "я поки не вмію слухати аудіо",
    "ukrainian": "я поки не вмію слухати аудіо",
    "українська": "я поки не вмію слухати аудіо",
    "en": "I can't listen to audio yet.",
    "english": "I can't listen to audio yet.",
}
_NO_AUDIO_FALLBACK = "I can't listen to audio yet."


def _no_audio_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _NO_AUDIO_PHRASES.get(lang, _NO_AUDIO_FALLBACK)


# Shown when transcription succeeds but returns no discernible speech.
_EMPTY_AUDIO_PHRASES = {
    "": "я не розчув, що там в аудіо",
    "uk": "я не розчув, що там в аудіо",
    "ua": "я не розчув, що там в аудіо",
    "ukrainian": "я не розчув, що там в аудіо",
    "українська": "я не розчув, що там в аудіо",
    "en": "I couldn't make out any speech in that audio.",
    "english": "I couldn't make out any speech in that audio.",
}
_EMPTY_AUDIO_FALLBACK = "I couldn't make out any speech in that audio."


def _empty_audio_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _EMPTY_AUDIO_PHRASES.get(lang, _EMPTY_AUDIO_FALLBACK)


# Shown when the bot OWNER has hit their plan's monthly Managed budget: the model
# call is skipped and this friendly note is sent instead. Same language keying.
_LIMIT_PHRASES = {
    "": "цей бот вичерпав місячний ліміт — спробуй наступного місяця",
    "uk": "цей бот вичерпав місячний ліміт — спробуй наступного місяця",
    "ua": "цей бот вичерпав місячний ліміт — спробуй наступного місяця",
    "ukrainian": "цей бот вичерпав місячний ліміт — спробуй наступного місяця",
    "українська": "цей бот вичерпав місячний ліміт — спробуй наступного місяця",
    "en": "This bot has reached its monthly usage limit — try again next month.",
    "english": "This bot has reached its monthly usage limit — try again next month.",
}
_LIMIT_FALLBACK = "This bot has reached its monthly usage limit — try again next month."


def _limit_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _LIMIT_PHRASES.get(lang, _LIMIT_FALLBACK)


# Shown when no chat provider/key is usable for the owner's plan (e.g. Managed
# disabled and no BYOK key). Same language keying as above.
_UNAVAILABLE_PHRASES = {
    "": "бот тимчасово недоступний",
    "uk": "бот тимчасово недоступний",
    "ua": "бот тимчасово недоступний",
    "ukrainian": "бот тимчасово недоступний",
    "українська": "бот тимчасово недоступний",
    "en": "This bot is temporarily unavailable.",
    "english": "This bot is temporarily unavailable.",
}
_UNAVAILABLE_FALLBACK = "This bot is temporarily unavailable."


def _unavailable_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _UNAVAILABLE_PHRASES.get(lang, _UNAVAILABLE_FALLBACK)


def _owner_gone(cfg, bot) -> bool:
    """Whether this bot is ORPHANED — no owner it can ever bill.

    True in two definitive cases, both of which should FAIL-FAST (stop the poller,
    send nothing) rather than keep answering on an ungated server key while an
    attacker floods the bot (M6):

      * the bot carries NO owner id (empty/missing) — a legacy or detached bot
        that can never bill anyone; and
      * the bot carries an owner id but no such user record exists (deleted owner).

    Returns False on a TRANSIENT lookup error (a store read blip): the owner may
    well exist, so a hiccup must never tear down a live poll thread — drop the one
    message quietly and let the next update re-resolve once the blip clears."""
    owner_id = (getattr(bot, "owner_id", "") or "").strip()
    if not owner_id:
        return True  # orphaned — no owner to bill, fail-fast
    try:
        from .accounts import UserStore

        return UserStore(cfg.data_dir).get(owner_id) is None
    except Exception:  # noqa: BLE001 - transient error must not kill the poller
        return False


def _resolve_chat_cfg(cfg, bot):
    """Resolve the effective chat Config for a Telegram bot, billed to the OWNER.

    Runs the SAME shared gate the web paths use (``resolve_chat_for_user``):
    Managed/BYOK key selection plus the owner's monthly Managed-budget check —
    so Telegram gets BYOK (the owner's key when ``mode=byok``) and honours the
    429 budget ceiling exactly like the guest path.

    Returns ``(config, None)`` when the model call may proceed, or ``(None,
    message)`` with a friendly, language-appropriate note to send instead — but
    ONLY for a valid, resolvable owner hitting their budget/BYOK gate (over budget
    → limit note; usable-key-less → unavailable note). For an ORPHANED/owner-gone
    owner, or ANY resolution error, it returns ``(None, None)``: drop the message
    SILENTLY (no outbound) rather than either degrading to the ungated SERVER key
    or emitting our own reply per spam message (an amplification vector against a
    hijacked bot). NEVER raises. (A transient error therefore drops this one
    message but does not wedge the poller — the next message re-resolves cleanly
    once the hiccup clears.)"""
    try:
        from .accounts import UserStore
        from .plans import plan_for
        from .user_settings import resolve_chat_for_user

        owner_id = (getattr(bot, "owner_id", "") or "").strip()
        owner = UserStore(cfg.data_dir).get(owner_id) if owner_id else None
        if owner is None:
            return None, None  # orphaned — silent drop, never the server key
        resolved = resolve_chat_for_user(cfg, owner, plan_for(owner))
        if resolved.status == "over_budget":
            return None, _limit_phrase(bot)
        if resolved.config is None:
            return None, _unavailable_phrase(bot)
        return resolved.config, None
    except Exception:  # noqa: BLE001 - drop silently on error, never the server key
        return None, None


def _owner_transcription_cfg(cfg, bot):
    """``cfg`` with the bot OWNER's BYOK transcription key swapped in when set.

    Voice messages are transcribed with the owner's own key (BYOK) when they are
    in BYOK mode and have a key stored for the server's transcription provider;
    otherwise the server key is used (the documented graceful fallback). Never
    raises — any lookup failure degrades to the passed-in ``cfg`` (server key)."""
    try:
        from .accounts import UserStore
        from .plans import plan_for
        from .user_settings import apply_byok_capability_key

        owner_id = (getattr(bot, "owner_id", "") or "").strip()
        owner = UserStore(cfg.data_dir).get(owner_id) if owner_id else None
        if owner is None:
            return cfg
        return apply_byok_capability_key(cfg, owner, plan_for(owner), "transcription")
    except Exception:  # noqa: BLE001 - never break transcription on a resolution hiccup
        return cfg


# Confirmation shown when a user clears the conversation buffer via /reset (or
# its /new alias). Same language keying as the service phrases above.
_RESET_PHRASES = {
    "": "почали з чистого аркуша — я забув попередню розмову.",
    "uk": "почали з чистого аркуша — я забув попередню розмову.",
    "ua": "почали з чистого аркуша — я забув попередню розмову.",
    "ukrainian": "почали з чистого аркуша — я забув попередню розмову.",
    "українська": "почали з чистого аркуша — я забув попередню розмову.",
    "en": "Fresh start — I've cleared our conversation.",
    "english": "Fresh start — I've cleared our conversation.",
}
_RESET_FALLBACK = "Fresh start — I've cleared our conversation."


def _reset_phrase(bot) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _RESET_PHRASES.get(lang, _RESET_FALLBACK)


# Friendly onboarding shown on /start, keyed by the bot's language (same keys as
# the service phrases above; blank keeps the Ukrainian default, an unlisted
# language falls back to English). ``{name}`` is filled with the bot's own name at
# send time so a renamed bot always greets correctly. Kept short and warm: what it
# does (grounded answers with sources), how to use it (ask / voice), and /reset.
_HELP_UK = (
    "Привіт! Я {name} 🤖\n\n"
    "Відповідаю на твої запитання, спираючись на проіндексовані YouTube-канали та "
    "документи — і показую джерела, звідки взяв відповідь.\n\n"
    "Як користуватися: просто постав запитання звичайним повідомленням. Голосові "
    "теж розумію — надішли, і я їх розшифрую.\n\n"
    "\n\n/reset (або /new) очищає пам'ять розмови."
)
_HELP_EN = (
    "Hi! I'm {name} 🤖\n\n"
    "I answer your questions grounded in my indexed YouTube channels and documents "
    "— and I show you the sources I used.\n\n"
    "How to use me: just ask a question in a normal message. Voice notes work too — "
    "send one and I'll transcribe it.\n\n"
    "\n\n/reset (or /new) clears conversation memory."
)
_HELP_TEMPLATES = {
    "": _HELP_UK,
    "uk": _HELP_UK,
    "ua": _HELP_UK,
    "ukrainian": _HELP_UK,
    "українська": _HELP_UK,
    "en": _HELP_EN,
    "english": _HELP_EN,
}
_HELP_FALLBACK = _HELP_EN


def _help_message(bot) -> str:
    """Localized /start help for this bot, with its name filled in."""
    lang = (getattr(bot, "language", "") or "").strip().lower()
    template = _HELP_TEMPLATES.get(lang, _HELP_FALLBACK)
    return template.format(name=getattr(bot, "name", "") or "this bot")


# Telegram command menu (setMyCommands payload) published per bot so /start and
# /reset show up in the client UI. Descriptions localized by the bot's language
# with the same keying/fallback as the phrases above. Command names carry NO
# leading slash — Telegram's BotCommand.command is the bare token.
_COMMAND_MENU_UK = [
    {"command": "start", "description": "Довідка / що вміє цей бот"},
    {"command": "help", "description": "Показати довідку"},
    {"command": "reset", "description": "Очистити пам'ять розмови"},
]
_COMMAND_MENU_EN = [
    {"command": "start", "description": "Help / what this bot does"},
    {"command": "help", "description": "Show help"},
    {"command": "reset", "description": "Clear conversation memory"},
]
_COMMAND_MENUS = {
    "": _COMMAND_MENU_UK,
    "uk": _COMMAND_MENU_UK,
    "ua": _COMMAND_MENU_UK,
    "ukrainian": _COMMAND_MENU_UK,
    "українська": _COMMAND_MENU_UK,
    "en": _COMMAND_MENU_EN,
    "english": _COMMAND_MENU_EN,
}
_COMMAND_MENU_FALLBACK = _COMMAND_MENU_EN


def _command_menu(bot) -> list[dict]:
    """Localized /start + /reset command menu for this bot's language."""
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return _COMMAND_MENUS.get(lang, _COMMAND_MENU_FALLBACK)


_MODE_TEXT_UK = (
    "Обери стиль відповіді:\n\n"
    "Довідник — перевірені факти з посиланнями.\n"
    "Мислення — факти з посиланнями та пояснення моїх висновків.\n\n"
    "Зараз: {current}"
)
_MODE_TEXT_EN = (
    "Choose an answer style:\n\n"
    "Reference — sourced facts.\n"
    "Thinking — sourced facts plus an explanation of my conclusions.\n\n"
    "Current: {current}"
)


def _mode_message(bot, current: str, changed: bool = False) -> str:
    lang = (getattr(bot, "language", "") or "").strip().lower()
    uk = lang in {"", "uk", "ua", "ukrainian", "українська"}
    labels = (
        {MODE_REFERENCE: "Довідник", MODE_THINKING: "Мислення"}
        if uk else {MODE_REFERENCE: "Reference", MODE_THINKING: "Thinking"}
    )
    prefix = ("Готово. " if changed and uk else "Updated. " if changed else "")
    return prefix + (_MODE_TEXT_UK if uk else _MODE_TEXT_EN).format(current=labels[current])


def _mode_labels(bot) -> dict[str, str]:
    """Localized labels for the two durable mode values."""
    lang = (getattr(bot, "language", "") or "").strip().lower()
    return ({MODE_REFERENCE: "Довідник", MODE_THINKING: "Мислення"}
            if lang in {"", "uk", "ua", "ukrainian", "українська"}
            else {MODE_REFERENCE: "Reference", MODE_THINKING: "Thinking"})


def _mode_keyboard(bot, current: str) -> dict:
    """Telegram inline keyboard for mode selection; checkmarks show the active mode."""
    labels = _mode_labels(bot)
    return {"inline_keyboard": [[
        {"text": f"{'✓ ' if current == MODE_REFERENCE else ''}{labels[MODE_REFERENCE]}",
         "callback_data": f"ytrag:mode:{MODE_REFERENCE}"},
        {"text": f"{'✓ ' if current == MODE_THINKING else ''}{labels[MODE_THINKING]}",
         "callback_data": f"ytrag:mode:{MODE_THINKING}"},
    ]]}


def _mode_confirmation(bot, selected: str) -> str:
    """One-line callback confirmation, intentionally free of implementation jargon."""
    uk = (getattr(bot, "language", "") or "").strip().lower() in {
        "", "uk", "ua", "ukrainian", "українська"
    }
    if uk:
        detail = ("лише перевірені факти з посиланнями"
                  if selected == MODE_REFERENCE else "факти з посиланнями та мої висновки")
        return f"Готово: {_mode_labels(bot)[selected]} — {detail}."
    detail = ("sourced facts only" if selected == MODE_REFERENCE
              else "sourced facts plus my conclusions")
    return f"Updated: {_mode_labels(bot)[selected]} — {detail}."


def _mode_choice(arg: str) -> str | None:
    token = (arg or "").strip().lower()
    if token in {"reference", "довідник", "dovidnyk"}:
        return MODE_REFERENCE
    if token in {"thinking", "мислення", "myslennia"}:
        return MODE_THINKING
    return None


def _callback_mode(data: object) -> str | None:
    """Return a mode only for this bot's exact inline-button payloads."""
    if not isinstance(data, str) or not data.startswith("ytrag:mode:"):
        return None
    return _mode_choice(data.removeprefix("ytrag:mode:"))


def _ack_callback(token: str, callback_id: object) -> None:
    """Stop Telegram's button spinner; callback acks are always best-effort."""
    if not callback_id:
        return
    try:
        _call(token, "answerCallbackQuery", {"callback_query_id": callback_id}, timeout=10.0)
    except Exception:  # noqa: BLE001 - an ack failure must not kill polling
        pass


def _edit_mode_confirmation(token: str, chat_id: int, message_id: object,
                            bot, selected: str) -> bool:
    """Replace the mode picker with its short localized selection confirmation."""
    if message_id is None:
        return False
    try:
        res = _call(token, "editMessageText", {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": _mode_confirmation(bot, selected),
            "link_preview_options": _preview_options(None),
            "reply_markup": json.dumps(_mode_keyboard(bot, selected)),
        }, timeout=10.0)
    except Exception:  # noqa: BLE001 - caller sends a separate confirmation instead
        return False
    if res.get("ok"):
        return True
    # Telegram rejects an identical repeat edit with ``message is not modified``.
    # The selected mode is already durable and visible, so don't turn a second
    # tap into a duplicate confirmation message.
    return "message is not modified" in str(res.get("description", "")).lower()


def _register_commands(bot_id: str, token: str) -> None:
    """Publish the /start + /reset command menu (setMyCommands) so Telegram's UI
    lists them.

    Best-effort and localized: the bot is read fresh to key the descriptions on
    its language (English when it can't be read), and the call is routed through
    :func:`_call` — i.e. the configured ``YTRAG_TELEGRAM_API_BASE`` — so it stays
    fake in tests/local and only reaches real Telegram on the VPS. Every failure is
    swallowed: a menu hiccup must never block a poller from starting."""
    try:
        bot = bot_service.bot_store(load_config()).get(bot_id)
    except Exception:  # noqa: BLE001 - registry read hiccup: fall back to English menu
        bot = None
    commands = _command_menu(bot) if bot is not None else _COMMAND_MENU_FALLBACK
    try:
        _call(token, "setMyCommands", {"commands": json.dumps(commands)}, timeout=10.0)
    except Exception:  # noqa: BLE001 - best-effort menu; never break the poller start
        pass


def _history_limits(cfg, bot) -> tuple[int, int, int]:
    """Resolve the OWNER plan's history caps: (max_turns, per-item, total chars).

    A Telegram bot's chat runs on — and is billed to — the OWNER, so the rolling
    buffer reuses the SAME caps the web guest path applies from the owner's plan.
    Any lookup hiccup degrades to the default plan's caps rather than blocking a
    chat (mirrors ``_resolve_chat_cfg``'s never-wedge posture)."""
    from .accounts import UserStore
    from .plans import DEFAULT_PLAN, PLANS, plan_for

    try:
        owner_id = (getattr(bot, "owner_id", "") or "").strip()
        owner = UserStore(cfg.data_dir).get(owner_id) if owner_id else None
        plan = plan_for(owner) if owner is not None else PLANS[DEFAULT_PLAN]
    except Exception:  # noqa: BLE001 - never let a plan lookup kill a chat
        plan = PLANS[DEFAULT_PLAN]
    return (
        plan.guest_history_max_turns,
        plan.guest_message_max_chars,
        plan.guest_history_max_chars,
    )


def _conv_history(cfg, bot, chat_id) -> list[dict]:
    """Load the rolling buffer for (chat_id, bot_id) as clean history to prepend.

    Re-cleans on read (defence in depth) with the owner plan's caps so a
    hand-edited store can never splice excerpts / oversized turns into the
    owner-billed prompt. Never raises — an empty list means "no memory"."""
    try:
        max_turns, max_item, max_total = _history_limits(cfg, bot)
        turns = TelegramConvStore(cfg.data_dir).history(chat_id, bot.id)
        return sanitize_guest_history(
            turns, max_turns=max_turns, max_item_chars=max_item, max_total_chars=max_total
        )
    except Exception:  # noqa: BLE001 - memory is best-effort, never blocks a chat
        return []


def _store_conv_turn(cfg, bot, chat_id, user_text: str, answer_text: str) -> None:
    """Append the CLEAN user+assistant turn to the buffer and trim.

    ``answer_text`` is the model's answer, including any model-produced citation
    quote blocks. Callers never append a separate retrieval-generated Sources
    footer, so unrelated retrieved chunks cannot enter memory. Never raises: a
    failed write must not break the reply the user already got."""
    if not (user_text or "").strip():
        return
    try:
        max_turns, max_item, max_total = _history_limits(cfg, bot)
        TelegramConvStore(cfg.data_dir).append(
            chat_id, bot.id, user_text, answer_text or "",
            max_turns=max_turns, max_item_chars=max_item, max_total_chars=max_total,
        )
    except Exception:  # noqa: BLE001 - accounting-style: never break a chat
        pass


def _image_turn_user_text(caption: str, image_summary: str) -> str:
    """Build the memory USER turn for a «Мислення» photo turn (#62).

    A photo's content is only visible to the vision model that turn, so — for the
    NEXT text turn to still "know" what the screenshot showed — we persist it as
    text: the caption (if any) plus a bracketed ``[Скріншот: …]`` factual description
    of the image (from the same vision call). Never returns an empty string (an empty
    user turn is silently dropped by ``_store_conv_turn``): with no summary it degrades
    to a bare ``[Скріншот]`` marker so the turn — and the assistant answer paired with
    it — still enters memory."""
    cap = (caption or "").strip()
    summ = (image_summary or "").strip()
    tag = f"[Скріншот: {summ}]" if summ else "[Скріншот]"
    return f"{cap}\n{tag}".strip() if cap else tag


def _transcription_seconds(duration: float, transcript: str) -> float:
    """Best-effort audio length (seconds) to meter the transcription cost.

    Prefers Telegram's reported clip ``duration``; when it's missing/zero (some
    clients omit it) it estimates from the transcript at ~150 words/min so a real
    clip is never billed as free. Both paths clamp to a non-negative float."""
    try:
        secs = float(duration or 0)
    except (TypeError, ValueError):
        secs = 0.0
    if secs > 0:
        return secs
    words = len((transcript or "").split())
    return words / 150.0 * 60.0  # ~150 wpm speech


def _record_transcription(trans_cfg, bot, duration: float, transcript: str) -> None:
    """Record the audio-transcription spend against the bot OWNER (M9).

    Bills the ACTUAL model the transcription ran on: ``trans_cfg`` is the config
    after the owner's BYOK transcription swap (provider + default model + key), so
    ``trans_cfg.transcription_model`` is what was really used — billing with the
    server default instead would misprice a BYOK owner on a different provider (e.g.
    OpenAI ``whisper-1`` charged at the groq rate). Mirrors ``bot_service.chat``'s
    passive accounting so it accrues toward the monthly budget just like the follow-on
    chat. Never raises — accounting must not break a reply."""
    try:
        from .usage import UsageStore

        seconds = _transcription_seconds(duration, transcript)
        UsageStore(trans_cfg.data_dir).record_transcription(
            bot.owner_id, trans_cfg.transcription_model, seconds
        )
    except Exception:  # noqa: BLE001 - passive accounting: never break a chat
        pass


class TelegramManager:
    """Owns one daemon long-poll thread per bot_id, keyed by its token."""

    def __init__(self) -> None:
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        # Per-(bot_id, chat_id) generations currently in flight. A chat may hold
        # at most one, so a second message while one runs is refused instead of
        # firing a parallel (expensive) generation.
        self._inflight: set[tuple[str, int]] = set()
        self._inflight_lock = threading.Lock()
        # Per-(bot_id, chat_id) monotonic "reset epoch". ``/reset`` bumps it and
        # clears the buffer under ``_reset_lock``; a generation captures the epoch
        # at spawn time and, at store time, writes its turn ONLY if the epoch is
        # unchanged. So a ``/reset`` that lands WHILE a generation is in flight is
        # never repopulated by that in-flight answer (M7) — without making the
        # user wait for the running generation to finish.
        self._reset_epoch: dict[tuple[str, int], int] = {}
        self._reset_lock = threading.Lock()

    def _current_epoch(self, key: tuple[str, int]) -> int:
        with self._reset_lock:
            return self._reset_epoch.get(key, 0)

    def _bump_reset_epoch(self, key: tuple[str, int]) -> None:
        with self._reset_lock:
            self._reset_epoch[key] = self._reset_epoch.get(key, 0) + 1

    def _store_if_fresh(
        self, key: tuple[str, int], epoch: int | None, cfg, bot, chat_id,
        user_text: str, answer_text: str,
    ) -> None:
        """Persist a completed turn unless a ``/reset`` superseded this generation.

        Holding ``_reset_lock`` across the epoch check AND the write makes it
        atomic against ``/reset``'s bump+clear: either this turn is written and a
        later reset clears it, or the reset wins and this stale-epoch turn is
        dropped — the buffer is never left holding a turn from before the reset.
        ``epoch is None`` (helper called directly, outside the dispatcher) always
        stores, preserving the plain append contract."""
        with self._reset_lock:
            if epoch is not None and self._reset_epoch.get(key, 0) != epoch:
                return
            _store_conv_turn(cfg, bot, chat_id, user_text, answer_text)

    def start(self, bot_id: str, token: str) -> str:
        username = verify_token(token)["username"]
        self.stop(bot_id)  # restart cleanly if a poller already runs
        _register_commands(bot_id, token)  # best-effort: publish the /start+/reset menu
        stop = threading.Event()
        thread = threading.Thread(
            target=self._poll_loop, args=(bot_id, token, stop),
            name=f"tg-{bot_id}", daemon=True,
        )
        with self._lock:
            self._stops[bot_id] = stop
            self._threads[bot_id] = thread
        thread.start()
        return username

    def stop(self, bot_id: str) -> None:
        with self._lock:
            stop = self._stops.pop(bot_id, None)
            self._threads.pop(bot_id, None)
        if stop is not None:
            stop.set()

    def running(self) -> list[str]:
        with self._lock:
            return [bid for bid, t in self._threads.items() if t.is_alive()]

    def start_all(self) -> None:
        """Boot a poller for every bot (any owner) that has a saved token."""
        cfg = load_config()
        store = bot_service.bot_store(cfg)
        for bot_id in _all_bot_ids(cfg):
            bot = store.get(bot_id)
            if bot is None or not bot.telegram_token:
                continue
            try:
                self.start(bot_id, bot.telegram_token)
            except Exception:  # noqa: BLE001 - one bad token must not block others
                continue

    # ---- polling ------------------------------------------------------
    def _poll_loop(self, bot_id: str, token: str, stop: threading.Event) -> None:
        offset = 0
        while not stop.is_set():
            try:
                res = _call(token, "getUpdates", {
                    "offset": offset, "timeout": 25,
                }, timeout=35.0)
            except Exception:  # noqa: BLE001 - transient network error; back off
                time.sleep(3)
                continue
            if not res.get("ok"):
                time.sleep(3)
                continue
            for update in res.get("result", []):
                if stop.is_set():
                    return
                # The offset advance is the one bit that used to live outside the
                # try/except: a malformed update lacking ``update_id`` raised a
                # KeyError here and killed this daemon poll thread → the bot went
                # silent until a full restart. Guard it: skip an id-less update
                # (can't ack what has no id) instead of crashing the loop.
                uid = update.get("update_id")
                if uid is None:
                    continue
                offset = uid + 1
                try:
                    keep = self._handle(bot_id, token, update)
                except Exception:  # noqa: BLE001 - one bad update must never kill the poller
                    continue
                if not keep:
                    return  # bot gone or token changed — end the loop

    def _handle(self, bot_id: str, token: str, update: dict) -> bool:
        """Answer one update. Return False when the poller should stop."""
        callback = update.get("callback_query") or {}
        # A callback has its own message envelope. Acknowledge it before any
        # validation below, including unknown/malformed payloads, so Telegram
        # never leaves a button spinner running.
        if callback:
            _ack_callback(token, callback.get("id"))
        message = (callback.get("message") if callback else None) or \
            update.get("message") or update.get("edited_message") or {}
        text = (message.get("text") or "").strip()
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return True

        cfg = load_config()
        bot = bot_service.bot_store(cfg).get(bot_id)
        if bot is None or bot.telegram_token != token:
            return False  # deleted or re-tokened out from under us

        # Orphaned bot (no owner id, or the owner account was erased) but the
        # poller is still live: it can never bill anyone, so STOP and send nothing
        # rather than answer on an ungated server key OR emit a reply per spam
        # message against a hijacked bot (M6). Only a definitive orphan stops the
        # loop; a transient lookup error keeps it running (see ``_owner_gone``).
        if _owner_gone(cfg, bot):
            return False

        if callback:
            selected = _callback_mode(callback.get("data"))
            if selected is None:
                return True  # another feature's button (or malformed data)
            store = TelegramConvStore(cfg.data_dir)
            selected = store.set_mode(chat_id, bot_id, selected)
            if not _edit_mode_confirmation(token, chat_id, message.get("message_id"), bot, selected):
                _send_message_async(
                    token, chat_id, _mode_confirmation(bot, selected),
                    reply_markup=_mode_keyboard(bot, selected),
                )
            return True

        if text.startswith("/start") or text.startswith("/help"):
            _send_message_async(token, chat_id, _help_message(bot))
            return True

        # /reset (alias /new) clears this conversation's rolling memory. Match on
        # the leading command token only (tolerating "/reset@BotName" and args)
        # so it can't collide with normal text like "/news…".
        cmd = text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""
        if cmd in ("/reset", "/new"):
            # Bump the reset epoch AND clear under one lock so an in-flight
            # generation started before now can't repopulate the cleared buffer
            # (M7): its stale epoch makes ``_store_if_fresh`` drop the late write.
            key = (bot_id, chat_id)
            with self._reset_lock:
                self._reset_epoch[key] = self._reset_epoch.get(key, 0) + 1
                TelegramConvStore(cfg.data_dir).clear(chat_id, bot_id)
            _send_message_async(token, chat_id, _reset_phrase(bot))
            return True

        if cmd in ("/mode", "/menu"):
            store = TelegramConvStore(cfg.data_dir)
            # The picker is the sole mode-changing UX.  Ignore command arguments
            # so a pasted or mistyped ``/mode thinking`` cannot silently change
            # a chat's answer style without the user pressing a visible button.
            current = store.mode(chat_id, bot_id)
            _send_message_async(
                token, chat_id, _mode_message(bot, current),
                reply_markup=_mode_keyboard(bot, current),
            )
            return True

        # Classify: a photo message (answered by the vision path) vs a text
        # message (unchanged path) vs neither (ignore).
        photos = message.get("photo") or []
        if photos:
            caption = (message.get("caption") or "").strip()
            file_id = photos[-1]["file_id"]  # largest available size
            return self._spawn_answer(
                self._answer_image, bot_id, token, chat_id, cfg, bot, file_id, caption,
            )
        voice = message.get("voice") or message.get("audio")
        if voice:
            caption = (message.get("caption") or "").strip()
            file_id = voice["file_id"]
            # Telegram voice/audio objects carry the clip duration (seconds); it
            # meters the transcription cost billed to the owner (M9).
            duration = voice.get("duration") or 0
            return self._spawn_answer(
                self._answer_audio, bot_id, token, chat_id, cfg, bot,
                file_id, caption, duration,
            )
        if text:
            return self._spawn_answer(
                self._answer, bot_id, token, chat_id, cfg, bot, text,
            )
        return True

    def _spawn_answer(self, worker, bot_id, token, chat_id, cfg, bot, *extra) -> bool:
        """Acquire the per-chat anti-spam lock, then spawn ``worker`` off the poll
        loop. Shared by the text and image paths.

        Anti-spam: one in-flight generation per chat. A second message while the
        first runs gets a short "still thinking" note and is dropped (no queue),
        so a user can't fan out several expensive generations at once. Generation
        runs off the poll loop so it keeps reading updates. Guard the spawn itself:
        if it fails we must release the lock (else the chat is stuck "busy" forever)
        and swallow the error (else the poll thread dies)."""
        key = (bot_id, chat_id)
        with self._inflight_lock:
            busy = key in self._inflight
            if not busy:
                self._inflight.add(key)
        if busy:
            _send_message_async(token, chat_id, _busy_phrase(bot))
            return True

        # Snapshot the reset epoch on the poll thread, ordered against any prior
        # ``/reset`` for this chat, so the worker can detect a reset that lands
        # while it runs and drop its (now stale) turn (M7).
        epoch = self._current_epoch(key)
        try:
            threading.Thread(
                target=worker, args=(token, chat_id, key, cfg, bot, *extra),
                kwargs={"epoch": epoch},
                name=f"tg-answer-{chat_id}", daemon=True,
            ).start()
        except Exception:  # noqa: BLE001 - never leak the lock or kill the poller
            with self._inflight_lock:
                self._inflight.discard(key)
            _send_message_async(token, chat_id, "Sorry, I hit an error answering that.")
        return True

    def _answer(
        self, token: str, chat_id: int, key: tuple[str, int], cfg, bot, text: str,
        *, epoch: int | None = None,
    ) -> None:
        """Stream one grounded answer (placeholder + live edits), then release the
        chat's in-flight lock."""
        typing_stop = threading.Event()
        try:
            # Owner-billed Managed/BYOK resolution + monthly-budget gate BEFORE
            # any model call (same shared helper as the web paths).
            chat_cfg, block_msg = _resolve_chat_cfg(cfg, bot)
            if chat_cfg is None:
                if block_msg is not None:
                    _send_message(token, chat_id, block_msg)
                return
            threading.Thread(
                target=_typing_keepalive, args=(token, chat_id, typing_stop), daemon=True,
            ).start()
            history = _conv_history(cfg, bot, chat_id)
            self._stream_answer(
                token, chat_id, key, cfg, bot, chat_cfg, text, history, epoch, typing_stop,
            )
        except Exception:  # noqa: BLE001 - never let one message kill the worker
            _send_message(token, chat_id, "Sorry, I hit an error answering that.")
        finally:
            typing_stop.set()  # ensure the keepalive thread exits on every path
            with self._inflight_lock:
                self._inflight.discard(key)

    def _stream_answer(
        self, token: str, chat_id: int, key: tuple[str, int], cfg, bot, chat_cfg,
        text: str, history: list[dict], epoch: int | None, typing_stop: threading.Event,
    ) -> None:
        """Consume ``bot_service.chat_stream`` and deliver it progressively.

        Sends a placeholder, edits it as tokens arrive (throttled), then applies
        the final rendered answer + sources in place. The CLEAN turn (answer only,
        no sources block) is persisted BEFORE the sources are appended, and only if
        a ``/reset`` hasn't superseded this generation (M7)."""
        placeholder_id = _send_placeholder(token, chat_id, _busy_phrase(bot))
        answer, sources, last_len, edits = "", [], 0, 0
        last_edit = 0.0
        mode = TelegramConvStore(cfg.data_dir).mode(chat_id, bot.id)
        for event in bot_service.chat_stream(
            chat_cfg, bot, text, history=history, answer_mode=mode,
        ):
            if event.get("type") == "delta":
                answer += event.get("text", "")
                now = time.monotonic()
                if placeholder_id is not None and _should_stream_edit(
                    len(answer), last_len, edits, now, last_edit
                ):
                    _edit_message(token, chat_id, placeholder_id, answer)
                    last_len, last_edit, edits = len(answer), now, edits + 1
            elif event.get("type") == "done":
                sources = event.get("sources") or []
        self._store_if_fresh(key, epoch, cfg, bot, chat_id, text, answer)
        typing_stop.set()  # stop before the final render/send
        final = _final_answer_text(answer, sources)
        _finalize_stream_message(
            token, chat_id, placeholder_id, final, preview_url=_preview_url(sources),
        )

    def _answer_image(
        self, token: str, chat_id: int, key: tuple[str, int], cfg, bot,
        file_id: str, caption: str, *, epoch: int | None = None,
    ) -> None:
        """Answer a photo with the vision model, then release the chat's lock.

        «Мислення»: the turn IS stored in conversation memory (#62) — the vision call
        also yields a plain factual ``image_summary`` which, with the caption, becomes
        the stored USER turn text so a later TEXT turn still knows what the screenshot
        showed; the write is epoch-guarded like the text/voice paths so a ``/reset``
        landing mid-generation can't repopulate the cleared buffer (M7).
        «Довідник»/no-mode stays stateless — nothing is stored (unchanged)."""
        try:
            # Gate on the EFFECTIVE vision key — the owner's BYOK vision credential
            # if set, else the server key — resolved via the same owner-capability
            # seam both photo branches use. So an owner with a valid BYOK vision key
            # reaches the image path even when the SERVER has no vision key; the
            # "no vision" message fires ONLY when NEITHER key exists (#59).
            if not bot_service.effective_vision_key(cfg, bot):
                _send_message(token, chat_id, _no_vision_phrase(bot))
                return

            # Owner-billed Managed/BYOK resolution + monthly-budget gate BEFORE
            # the (expensive) vision call.
            chat_cfg, block_msg = _resolve_chat_cfg(cfg, bot)
            if chat_cfg is None:
                if block_msg is not None:
                    _send_message(token, chat_id, block_msg)
                return

            typing_stop = threading.Event()
            placeholder_id: int | None = None
            try:
                threading.Thread(
                    target=_typing_keepalive, args=(token, chat_id, typing_stop),
                    daemon=True,
                ).start()
                # Show a visible "still thinking" bubble up front (like the text
                # path), then replace it in place with the answer — so a photo
                # doesn't sit on a bare "typing…" indicator with no message.
                placeholder_id = _send_placeholder(token, chat_id, _busy_phrase(bot))
                file_path = _get_file_path(token, file_id)
                if file_path is None:
                    typing_stop.set()
                    _finalize_stream_message(
                        token, chat_id, placeholder_id,
                        "Sorry, I hit an error answering that.",
                    )
                    return
                image_bytes = _download_file(token, file_path)
                mime = _guess_image_mime(file_path)
                # Honor the per-chat mode like the text path (was ignored — a photo
                # always ran the grounded/retrieval path). «Мислення» → a VISION + v2
                # base turn that sees the FULL dialogue and reasons with NO retrieval;
                # «Довідник»/no-mode → the unchanged grounded vision path (which takes
                # no history — the photo path stays stateless there).
                mode = TelegramConvStore(cfg.data_dir).mode(chat_id, bot.id)
                history = (
                    _conv_history(cfg, bot, chat_id)
                    if mode == MODE_THINKING else None
                )
                result = bot_service.chat_with_image(
                    chat_cfg, bot, caption, image_bytes, mime,
                    history=history, answer_mode=mode,
                )
                sources = result.get("sources")
                raw_answer = result.get("answer") or ""
                # «Мислення»: persist this photo turn as TEXT so a later text turn
                # retains it. The stored USER turn is the caption + a factual
                # [Скріншот: …] description (from the same vision call), never empty;
                # the assistant turn is the model's answer. Epoch-guarded (M7).
                # «Довідник»/no-mode: stateless — store nothing (unchanged).
                if mode == MODE_THINKING:
                    self._store_if_fresh(
                        key, epoch, cfg, bot, chat_id,
                        _image_turn_user_text(caption, result.get("image_summary") or ""),
                        raw_answer,
                    )
                answer = _final_answer_text(raw_answer, sources)
                typing_stop.set()  # stop before the (possibly multi-chunk) send
                _finalize_stream_message(
                    token, chat_id, placeholder_id, answer or "(no answer)",
                    preview_url=_preview_url(sources),
                )
            except Exception:  # noqa: BLE001 - never let one message kill the worker
                _finalize_stream_message(
                    token, chat_id, placeholder_id,
                    "Sorry, I hit an error answering that.",
                )
            finally:
                typing_stop.set()  # ensure the keepalive thread exits on every path
        finally:
            with self._inflight_lock:
                self._inflight.discard(key)

    def _answer_audio(
        self, token: str, chat_id: int, key: tuple[str, int], cfg, bot,
        file_id: str, caption: str, duration: float = 0.0,
        *, epoch: int | None = None,
    ) -> None:
        """Transcribe a voice/audio message, then run the normal grounded chat on
        the transcript; release the chat's lock on every path."""
        try:
            # Honor the per-chat mode like the text path: a voice note is transcript→
            # text, so it reuses the SAME routing (incl. «Мислення» v2 for thinking).
            mode = TelegramConvStore(cfg.data_dir).mode(chat_id, bot.id)
            # BUG-007 cost guard: block a not-yet-indexed bot BEFORE transcription
            # (which bills the owner) and before the model — otherwise a voice note
            # to an empty bot pays for transcription for nothing. chat()/chat_with_
            # image() keep the gate as the backstop for the text/image paths. The gate
            # is MODE-AWARE (H1): «Мислення» reasons with no retrieval, so a corpus-less
            # bot may answer a voice note in that mode — matching the text path.
            if not bot_service.chat_ready(cfg, bot, mode):
                _send_message(token, chat_id, bot_service.CHAT_NOT_READY_MESSAGE)
                return
            # Resolve the OWNER's BYOK transcription key (falls back to the server
            # key) BEFORE the "is transcription configured?" guard, so a BYOK owner
            # with their own key transcribes even if the server has none.
            trans_cfg = _owner_transcription_cfg(cfg, bot)
            if not trans_cfg.transcription_key():
                _send_message(token, chat_id, _no_audio_phrase(bot))
                return

            # Owner-billed Managed/BYOK resolution + monthly-budget gate BEFORE
            # transcription + the model call (skip both when over budget).
            chat_cfg, block_msg = _resolve_chat_cfg(cfg, bot)
            if chat_cfg is None:
                if block_msg is not None:
                    _send_message(token, chat_id, block_msg)
                return

            typing_stop = threading.Event()
            placeholder_id: int | None = None
            try:
                threading.Thread(
                    target=_typing_keepalive, args=(token, chat_id, typing_stop),
                    daemon=True,
                ).start()
                # Visible "still thinking" bubble up front (like the text path),
                # replaced in place with the answer once transcription + chat run.
                placeholder_id = _send_placeholder(token, chat_id, _busy_phrase(bot))
                file_path = _get_file_path(token, file_id)
                if file_path is None:
                    typing_stop.set()
                    _finalize_stream_message(
                        token, chat_id, placeholder_id,
                        "Sorry, I hit an error answering that.",
                    )
                    return
                audio_bytes = _download_file(token, file_path)
                # No language hint: Whisper auto-detects, more robust than mapping
                # bot.language (which may be "ukrainian"/"" not an ISO-639-1 code).
                transcript = make_transcriber(trans_cfg).transcribe(
                    audio_bytes, _audio_filename(file_path)
                )
                # Bill the transcription itself to the OWNER (M9): it is real spend
                # that must accrue toward the monthly budget, not just the follow-on
                # chat. Bill against trans_cfg (the ACTUAL BYOK/server model that ran),
                # not the server default. Passive accounting — never break the reply.
                _record_transcription(trans_cfg, bot, duration, transcript)
                question = (transcript or "").strip()
                if not question:
                    typing_stop.set()
                    _finalize_stream_message(
                        token, chat_id, placeholder_id, _empty_audio_phrase(bot),
                    )
                    return
                query = f"{caption}\n\n{question}" if caption else question
                history = _conv_history(cfg, bot, chat_id)
                result = bot_service.chat(
                    chat_cfg, bot, query, history=history, answer_mode=mode,
                )
                sources = result.get("sources")
                raw_answer = result.get("answer") or ""
                # Store the clean transcript-derived question + model answer (with
                # any deliberate citation quote blocks, but no generated footer) so a
                # spoken follow-up carries context like a typed one.
                # Skipped if a /reset superseded this generation while it ran (M7).
                self._store_if_fresh(key, epoch, cfg, bot, chat_id, query, raw_answer)
                answer = _final_answer_text(raw_answer, sources)
                typing_stop.set()  # stop before the (possibly multi-chunk) send
                _finalize_stream_message(
                    token, chat_id, placeholder_id, answer or "(no answer)",
                    preview_url=_preview_url(sources),
                )
            except Exception:  # noqa: BLE001 - never let one message kill the worker
                _finalize_stream_message(
                    token, chat_id, placeholder_id,
                    "Sorry, I hit an error answering that.",
                )
            finally:
                typing_stop.set()  # ensure the keepalive thread exits on every path
        finally:
            with self._inflight_lock:
                self._inflight.discard(key)


def _all_bot_ids(cfg) -> list[str]:
    """Read the registry JSON directly — BotStore exposes no list_all."""
    from .safestore import read_json

    data = read_json(cfg.data_dir / "bots.json", {}) or {}
    return list(data.keys())


manager = TelegramManager()
