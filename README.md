# Channelmind

Index every video transcript from a YouTube channel and chat with an AI assistant
grounded in them — with **clickable source citations and timecodes**. Runs fully on
your own machine: bring your own LLM key, point it at any channel, and chat over the
web UI, the CLI, or a Telegram bot.

```
channel ──yt-dlp──▶ transcripts ──chunk──▶ embeddings ──▶ Chroma vector DB ──▶ LLM RAG chat
```

- **No Google Cloud, no OAuth setup** to run locally — a built-in dev-login opens the app straight into a single-user workspace.
- **No proxy needed** — running on your own residential IP, YouTube fetches work directly.
- **Bring your own key** — one [OpenRouter](https://openrouter.ai) key powers both chat and embeddings (or swap in Anthropic / OpenAI / Voyage / Groq).
- Grounded answers only: the assistant cites the source videos and timecodes it used, and declines when the channel doesn't cover a question.

---

## Quickstart (≈2 minutes)

**Prerequisites:** Python 3.11+ and [`ffmpeg`](https://ffmpeg.org) on your PATH (yt-dlp uses it).

```bash
# 1. Clone and install
git clone <your-fork-url> channelmind && cd channelmind
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .

# 2. Configure — one key is enough
cp .env.example .env
#   then edit .env and set OPENROUTER_API_KEY=sk-or-...   (get one at https://openrouter.ai/keys)

# 3a. CLI path — index a channel and chat
ytrag ingest @veritasium --limit 50      # download transcripts (patient & resumable)
ytrag build                              # chunk + embed into the local Chroma index
ytrag chat                               # interactive grounded chat in your terminal
#   or one-shot:
ytrag ask "What did they say about the double-slit experiment?"

# 3b. Web path — the full app (chat UI, bot manager)
uvicorn ytrag.web.app:app --host 127.0.0.1 --port 8000    # then open http://127.0.0.1:8000
```

That's it. `.env` ships with `YTRAG_ENV=development` and `YTRAG_DEV_LOGIN=1`, so the
web app needs **no Google account and no consent screen**: just open the app — it
**auto-signs you in** to a local workspace (a "Continue as local user" button is there
as a fallback). To use real Google login instead, set `YTRAG_DEV_LOGIN=0`.

> `ytrag serve` is a *different* server — the agent-facing MCP + admin plane at
> `/mcp` — not the web UI. Use the `uvicorn` command above for the web app.

---

## Providers (pluggable)

Both the chat model and the embedding model are swappable via `.env`:

| Role      | Options                               | Default (OSS)                 |
|-----------|---------------------------------------|-------------------------------|
| **LLM**   | `anthropic` · `openai` · `openrouter` | `openrouter`                  |
| **Embed** | `voyage` · `openai` · `openrouter`    | `openrouter`                  |

One OpenRouter key can serve **both** chat and embeddings. To split providers, set
`YTRAG_LLM_PROVIDER` / `YTRAG_EMBED_PROVIDER` and the matching key in `.env`
(see `.env.example` for every option).

## Telegram (optional)

A channel bot can also answer in Telegram. Set `YTRAG_TELEGRAM_API_BASE` in `.env`
and connect a bot from the web UI — see `docs/telegram-connect-howto.md`.

## How it works

1. **ingest** — `yt-dlp` enumerates a channel's videos and downloads subtitles (manual, falling back to auto-generated); VTT is cleaned into plain prose. Patient and resumable: it paces requests, backs off on 429s, and skips already-downloaded videos, so a large channel can run for hours and resume after any interruption.
2. **build** — transcripts are split into overlapping chunks, embedded, and stored in a local [Chroma](https://www.trychroma.com) DB. Re-running `build` upserts by chunk id (idempotent).
3. **chat** — a question is embedded, the closest chunks retrieved, and the LLM answers using only those excerpts — citing the source videos and timecodes.

See `ARCHITECTURE.md` for the module map.

## Configuration

Everything lives in `.env` (copy from `.env.example`). Common knobs:
`YTRAG_LLM_PROVIDER` / `YTRAG_EMBED_PROVIDER`, the matching API key(s),
`YTRAG_LLM_MODEL` / `YTRAG_EMBED_MODEL`, `YTRAG_DATA_DIR`, `YTRAG_CHROMA_DIR`,
and an optional `YTRAG_ROUTER_MODEL` (a small fast model that cuts time-to-first-token).

## Tests

Lightweight verification scripts live under `scripts/` (e.g. `verify_video_ingest.py`,
`verify_source_citation_wiring.py`, `verify_source_timecode.py`, `e2e_contract.py`).
Run one with `python scripts/<name>.py`.

## Legal & responsible use

Channelmind is a **neutral tool**: it ships no transcripts or data, and downloads
nothing until *you* point it at a channel using *your own* API keys.

- **You are responsible** for complying with [YouTube's Terms of Service](https://www.youtube.com/t/terms) and with copyright law in your jurisdiction. Automated download of subtitles may be restricted by those terms.
- Intended for **personal, research, and educational** use. Do not redistribute transcripts you download, and do not use the tool to reproduce or republish creators' content.
- **Rights holders / takedowns:** if you operate a hosted instance and a rights holder objects to a channel being indexed, remove that source and its stored transcripts/vectors. A placeholder takedown contact lives in the app's legal pages — replace it with your own before any public deployment.

## Using an AI coding agent to set it up

Point your agent (Cursor, Claude Code, Codex, etc.) at this repo — it will read
[`AGENTS.md`](AGENTS.md) and know how to install, configure, run, and extend the
project on its own. You only need to hand it one `OPENROUTER_API_KEY`.

## License

[MIT](LICENSE).
