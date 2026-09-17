# AGENTS.md — instructions for an AI coding agent

This file tells an autonomous coding agent how to set up, run, and work on
**Channelmind** without human help. Follow the steps in order. Commands assume a
POSIX shell; Windows equivalents are noted inline.

## TL;DR — do this now (fresh clone)

The only thing a human must provide is one `OPENROUTER_API_KEY`. Everything else is
zero-config. Run this, substituting the key:

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                              # then put OPENROUTER_API_KEY=sk-or-... in .env
ytrag ingest @veritasium --limit 5 && ytrag build
ytrag ask "Summarize one recurring theme."        # grounded answer with sources = success
uvicorn ytrag.web.app:app --host 127.0.0.1 --port 8000   # web UI, auto-logged-in, no Google
```

If `ytrag ask` returns a grounded answer with citations, the install is correct. The
sections below explain each step, the file map, and the edit guardrails.

## What this project is

A self-hostable RAG app over a YouTube channel's transcripts. Pipeline:
`ingest (yt-dlp) → chunk → embed → Chroma vector store → grounded LLM chat`.
Python package lives in `src/ytrag/`. CLI entry point is `ytrag` (Typer).
Web app is FastAPI + a Vue SPA under `src/ytrag/web/`. Optional Telegram bot.

## 1. Environment setup

```bash
# Requires Python 3.11+ and ffmpeg on PATH.
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

## 2. Configuration (no cloud accounts needed)

```bash
cp .env.example .env
```

Then set exactly one secret in `.env`:

```
OPENROUTER_API_KEY=sk-or-...     # from https://openrouter.ai/keys
```

Leave `YTRAG_ENV=development` and `YTRAG_DEV_LOGIN=1` as shipped — this enables the
**no-login local mode** (the web app auto-authenticates a local developer, so no
Google OAuth / consent screen is required). Providers default to OpenRouter for both
chat and embeddings, so that single key is sufficient. Do NOT commit `.env`.

## 3. Index a channel and verify chat (CLI)

```bash
ytrag ingest @veritasium --limit 10      # small limit for a fast first run
ytrag build                              # embed into the local Chroma index
ytrag ask "Summarize one recurring theme."   # expect a grounded answer with sources
```

Data lands in `./data` (transcripts + JSON state) and `./chroma` (vectors); both are
gitignored. Re-running `build` is idempotent (upsert by chunk id).

## 4. Run the web app

```bash
uvicorn ytrag.web.app:app --host 127.0.0.1 --port 8000    # http://127.0.0.1:8000
```

Open the URL. In local no-login mode (`YTRAG_DEV_LOGIN=1`) the SPA **auto-signs-in**
as a local `dev@local` user on first load — no Google OAuth, no consent screen, no
click. (A "Continue as local user" button is also shown as a fallback.) Then create a
bot, add a YouTube source, wait for ingest, and chat. Answers are strictly grounded in
the channel, with citations and timecodes. To use real Google login instead, set
`YTRAG_DEV_LOGIN=0` and configure the OAuth vars.

NOTE: `ytrag serve` is NOT the web UI — it starts the agent-facing MCP + admin server
(`/mcp`, `/admin`). Use the `uvicorn ytrag.web.app:app` command above for the web app.
On Windows, run any command that prints answers with `PYTHONUTF8=1` to avoid a
cp1252 console-encoding crash on Unicode glyphs (the Dockerfile already sets this).

## 5. Tests

Verification scripts under `scripts/` are runnable directly:

```bash
python scripts/e2e_contract.py
python scripts/verify_video_ingest.py
python scripts/verify_source_citation_wiring.py
python scripts/verify_source_timecode.py
```

Some scripts honor an offline fake mode via `YTRAG_TEST_MODE=1` (deterministic, no
network / no keys) — check a script's header before running.

## Conventions & guardrails for edits

- **Grounded-only chat.** This build intentionally ships ONLY the grounded
  «Довідник»/Reference answer path. A second "Thinking" general-advisor mode was
  removed at the UI layer: the Telegram `/mode` command and the SPA mode toggles are
  gone, and `TelegramConvStore.mode()` is pinned to `MODE_REFERENCE`. The underlying
  mode machinery is left in place but inert — do not re-expose a thinking/advisor
  toggle unless explicitly asked.
- **Monetization is dormant, not removed.** Plan caps, quotas, usage/billing, BYOK
  keystore, and share/guest tokens remain in the code but are inactive in the
  single-user no-login local mode. Leave them alone unless a task targets them.
- **Secrets.** Never hardcode keys, tokens, emails, or domains. All config comes from
  `.env` / environment. The only place a real key belongs is your local `.env`.
- **Providers are pluggable** via `src/ytrag/config.py` and `llm.py`/`embed.py` — add
  a provider there, don't fork call sites.
- After changing Python, sanity-check imports: `python -c "import ytrag.cli"`.
- After changing SPA JS, syntax-check: `node --check src/ytrag/web/spa/views/<file>.js`.

## Where things live

| Concern            | Files |
|--------------------|-------|
| Ingest / transcripts | `ingest.py`, `transcripts.py`, `transcript_ingest.py`, `transcribe.py`, `chunk.py`, `sources.py` |
| Embed / store      | `embed.py`, `store.py`, `indexer.py` |
| RAG / answering    | `rag.py`, `llm.py`, `intent.py`, `bot_service.py` |
| CLI                | `cli.py` |
| Web app + SPA      | `web/app.py`, `web/spa/` |
| Telegram bot       | `telegram_bot.py`, `telegram_conv.py` |
| Auto-sync / jobs   | `autosync.py`, `ingest_jobs.py`, `ingest_queue.py` |
| Config             | `config.py` |
