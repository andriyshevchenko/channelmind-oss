"""Best-effort transactional email when an ingest job reaches a terminal state.

The bot owner may close their browser tab while a channel import runs in the
background worker. This module emails them once the job finishes (done) or fails
(error) so they learn the outcome without watching the UI.

Everything here is a best-effort SIDE EFFECT: it must NEVER affect the ingest
outcome, block the queue meaningfully, or crash the single worker thread. The
entire public function is wrapped in a broad try/except that swallows every
failure — missing config, no user email, network error, bad HTTP response.

HTTP uses stdlib ``urllib.request`` (the project has no requests/httpx dep),
mirroring ``telegram_bot._call``. Config is read directly from the environment
(this module is self-contained like ``usage.py``); nothing is added to Config.

Env vars:
  YTRAG_RESEND_API_KEY  Resend API key. Empty/missing → feature disabled (no-op).
  YTRAG_EMAIL_FROM      From header. Default works without a custom domain.
  YTRAG_APP_BASE_URL    Public base URL; if set, emails link to the bot page.
"""
from __future__ import annotations

import html
import json
import logging
import os
import urllib.request

from . import bot_service
from .accounts import UserStore
from .config import Config
from .ingest_jobs import STATUS_DONE, STATUS_ERROR, JobStore
from .user_settings import UserSettingsStore

_LOG = logging.getLogger(__name__)
_RESEND_URL = "https://api.resend.com/emails"
_DEFAULT_FROM = "yt-channel-rag <onboarding@resend.dev>"


def notify_ingest_complete(cfg: Config, job_id: str) -> None:
    """Email the bot owner that ingest ``job_id`` finished or failed.

    Best-effort and fully defensive: never raises, never affects the ingest.
    """
    try:
        api_key = (os.getenv("YTRAG_RESEND_API_KEY") or "").strip()
        if not api_key:
            return  # feature disabled

        # Re-read the job fresh — the caller's in-memory copy is stale after
        # mark_done/mark_error, so we need the FINAL status and counts.
        job = JobStore(cfg.data_dir).get(job_id)
        if job is None:
            return
        if job.status not in (STATUS_DONE, STATUS_ERROR):
            return  # cancelled or non-terminal: don't email

        user = UserStore(cfg.data_dir).get(job.user_id)
        if user is None or not (user.email or "").strip():
            return
        # Honor the per-user notification-email toggle (default ON). This is the
        # single send boundary for user-facing notifications, so an owner who
        # turned them off is not emailed. Transactional/auth mail does not pass
        # through this module and is unaffected.
        if not UserSettingsStore(cfg.data_dir).get(job.user_id).email_notifications:
            return
        email = user.email.strip()

        bot = None
        try:
            bot = bot_service.bot_store(cfg).get(job.bot_id) if job.bot_id else None
        except Exception:  # noqa: BLE001 - bot lookup is decorative
            bot = None
        bot_name = (getattr(bot, "name", "") or "").strip() or (
            job.channel_url or "your bot"
        )

        subject, text, html_body = _compose(cfg, job, bot_name)
        _send(api_key, email, subject, text, html_body)
    except Exception:  # noqa: BLE001 - best-effort; never affect ingest
        _LOG.warning("ingest completion email failed for job %s", job_id, exc_info=True)


def _base_url() -> str:
    return (os.getenv("YTRAG_APP_BASE_URL") or "").strip().rstrip("/")


def _summary_line(job) -> str:
    line = f"{job.videos_done}/{job.videos_total} videos indexed"
    if job.videos_failed and job.videos_failed > 0:
        line += f" ({job.videos_failed} failed)"
    return line


def _compose(cfg: Config, job, bot_name: str) -> tuple[str, str, str]:
    """Return (subject, text_body, html_body) for a terminal job."""
    channel = job.channel_url or ""
    summary = _summary_line(job)
    base = _base_url()
    link = f"{base}/bots/{job.bot_id}" if base and job.bot_id else ""

    if job.status == STATUS_DONE:
        subject = f"Import finished: {bot_name}"
        text_lines = [
            f"Your import for {bot_name} finished.",
            "",
            f"Channel: {channel}",
            summary,
        ]
    else:  # STATUS_ERROR
        subject = f"Import failed: {bot_name}"
        text_lines = [
            f"Your import for {bot_name} failed.",
            "",
            f"Channel: {channel}",
            summary,
            "",
            f"Error: {job.error or 'unknown error'}",
        ]
    if link:
        text_lines += ["", f"Open your bot: {link}"]
    text = "\n".join(text_lines)

    html_body = _compose_html(job, bot_name, channel, summary, link)
    return subject, text, html_body


def _compose_html(job, bot_name: str, channel: str, summary: str, link: str) -> str:
    """Minimal HTML body. All user-controlled values are html.escape'd."""
    b = html.escape(bot_name)
    ch = html.escape(channel)
    verb = "finished" if job.status == STATUS_DONE else "failed"
    parts = [
        f"<p>Your import for <strong>{b}</strong> {verb}.</p>",
        f"<p>Channel: {ch}<br>{html.escape(summary)}</p>",
    ]
    if job.status == STATUS_ERROR:
        parts.append(f"<p>Error: {html.escape(job.error or 'unknown error')}</p>")
    if link:
        safe_link = html.escape(link)
        parts.append(f'<p><a href="{safe_link}">Open your bot</a></p>')
    return "<html><body>" + "".join(parts) + "</body></html>"


def _send(api_key: str, email: str, subject: str, text: str, html_body: str) -> None:
    payload = json.dumps(
        {
            "from": (os.getenv("YTRAG_EMAIL_FROM") or "").strip() or _DEFAULT_FROM,
            "to": [email],
            "subject": subject,
            "text": text,
            "html": html_body,
        }
    ).encode()
    req = urllib.request.Request(
        _RESEND_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        resp.read()  # drain; a 2xx with no exception means accepted
