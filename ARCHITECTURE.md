# Architecture — Channelmind (yt-channel-rag)

> Orient in 5 minutes. This maps the DURABLE structure (it changes slowly); the day-to-day
> status/backlog lives in `Implementation Plans/PLAN.md`, the "why" in `DECISIONS.md`, and
> the working conventions in `AGENTS.md`.

## What it is
A per-user tool that ingests a **whole YouTube channel's transcripts**, indexes them, and
serves a **grounded chat bot** (web + Telegram) that answers from that channel's material
with deep-link citations. Stateful single-VPS deploy (Docker: app + Caddy).

## Runtime shape
- **One FastAPI/uvicorn process** (`src/ytrag/web/app.py`, ASGI `app`). It ALSO hosts, in-process:
  the Telegram long-poller, the auto-sync scheduler, and the single FIFO ingest worker.
  → This is why prod runs **one worker** (multiple workers would double-poll Telegram / double-run
  sync). Horizontal scale needs the poller split out first — see PLAN §5b / DECISIONS.
- **Chroma** (SQLite-backed vector store) on disk (`/chroma`); JSON stores for state
  (bots/accounts/usage/ingest_jobs/…) under `/data`.
- **OpenRouter** for chat + embeddings (single key); Voyage optional (default embed provider).
- Self-host = runs locally (or on any VPS/container you control); no managed infra assumed.

## Components (by concern)
**Chat / RAG** — the heart:
- `rag.py` (Assistant) — the answer engine. Classic single-shot path AND the agentic loop
  (behind `YTRAG_AGENTIC_RAG`). Retrieval, prompt build, grounding rules, citations, streaming,
  billing fold. Sync methods (`answer`/`answer_stream`) + async twins (`aanswer`/`aanswer_stream`).
- `intent.py` — the pre-answer **router** (`understand`/`aunderstand`): classify smalltalk vs
  question, rewrite follow-ups to a standalone query + sub-queries. Runs on its OWN model
  (`YTRAG_ROUTER_MODEL`, cheap/fast) or falls back to the answer model.
- `smalltalk.py` — the router's deterministic **veto** (`looks_like_instruction_or_oversized`
  = imperative/oversized/jailbreak-marker) + `looks_like_question` (speculation gate).
- `embed.py` — embedders (Voyage / OpenAI-compat), `embed_queries` (batched query embed).
- `llm.py` — the LLM seam: `OpenAICompatLLM`/`AnthropicLLM`, tool-calling seam
  (`complete_tools`/`stream_tools` + async), `shared_openai_client` / `shared_async_openai_client`
  (warm pooled clients), `make_llm`/`make_router_llm`.
- `store.py` (VectorStore/Chroma), `chunk.py` (transcript→chunks), `sources.py` (Sources block).
- `bot_service.py` — the chat entry the web/Telegram call: `chat`/`chat_stream` + async
  `achat`/`achat_stream`; the BUG-007 ready-gate; exactly-once billing (`_BillOnce`/`_MetricOnce`).

**Ingest** (transcripts in):
- `ingest.py` (yt-dlp transcript fetch, proxy/sticky sessions, player-client, langs), `ingest_queue.py`
  (single FIFO worker), `ingest_jobs.py` (JobStore, resume), `youtube_count.py` (enumerate count),
  `autosync.py` (per-bot Off/Daily/Weekly/Monthly scheduler reusing the FIFO worker).

**Web / SPA**:
- `web/app.py` — all HTTP endpoints (auth, bots CRUD, ingest, `/chat` + `/chat/stream` SSE),
  ASGI middleware (body caps, CSRF), error/health handlers.
- `web/spa/` — buildless Vue SPA (BotDetail, PublicChat, Settings; `mdlite.js` markdown, `api/`).

**Telegram**: `telegram_bot.py` (long-poller, per-chat dispatch, streaming answers, audio/photo),
`telegram_conv.py` (per-chat memory).

**Identity / limits / billing**: `oauth.py` (Google), `accounts.py`/`tenants.py`/`user_settings.py`,
`plans.py` + `share_limits.py`/`quotas.py` (named-plan policy registry, beta caps), `usage.py`
(token accounting + monthly budget 429), `metrics.py` (per-bot beta metrics), `throttle.py`.

**Infra**: `config.py` (all env → `Config`), `safestore.py`/`settings_store.py` (JSON persistence +
Fernet key encryption `YTRAG_KEYSTORE_SECRET`), `logging_setup.py`, `notify.py` (Resend email),
`cli.py`, `testmode.py` (FakeLLM/FakeEmbedder for offline `YTRAG_TEST_MODE=1`).

## Chat request flow (classic)
`bot_service.chat` → `Assistant`: `understand` (router: smalltalk? rewrite) → if question:
`embed_queries` → Chroma retrieve (diversify across sources) → relevance floor (decline if nothing)
→ build grounded prompt → answer LLM (structured-citation when fragments available, #28) →
cited-only Sources with deep-link timecodes. Billing folded once (router + answer).

## Chat request flow (agentic, `YTRAG_AGENTIC_RAG=1`)
Router still gates smalltalk. For a question the answer model drives a `search_channel` tool loop
(≤5 rounds), reasons on top of an **authoritative** base, then the final answer is validated:
fabricated deep-links stripped, cited-only Sources via the fragment registry; **zero valid
citations → fall back to the classic decline** (never ships ungrounded). Streaming buffers the
final answer until validated (tool-round `status` events stream live). Default OFF.

## Testing / CI
- `scripts/verify_*.py` — ~78 deterministic checks. 76 run OFFLINE (`YTRAG_TEST_MODE=1`, fakes,
  0 LLM tokens); 2 need real creds (`verify_secrets`, `verify_google_auth_real`).
- `.github/workflows/`: `verify-offline.yml` (push/PR, the offline lane), `real-e2e.yml`
  (manual, Playwright + real deps), `deploy.yml` (push master → build+push image → VPS).

## Key env flags (see `env.example` + `config.py` for the full list)
`YTRAG_LLM_MODEL` (answer), `YTRAG_ROUTER_MODEL` (cheap router; empty=answer model),
`YTRAG_AGENTIC_RAG` (agentic path on/off, default off), `YTRAG_EMBED_MODEL/PROVIDER`,
`YTRAG_MAX_RETRIEVAL_DISTANCE` (relevance floor), `YTRAG_TRANSCRIPT_PROXY(+_STICKY_FORMAT `{s}`)`,
`YTRAG_YTDLP_PLAYER_CLIENT`, `YTRAG_TEST_MODE`. Secrets: `YTRAG_SESSION_SECRET`,
`YTRAG_KEYSTORE_SECRET`, `OPENROUTER_API_KEY`, Google OAuth, `GROQ_API_KEY`.
