# Use-case map — Channelmind frontend

One entry per use case: the screen/component that owns it, the state involved, the `api.*` calls it makes (adapter interface in `api/index.js`; mock in `mock/api.js`), and what's still mock-only (needs a real endpoint or wiring).

## Runtime modes (config.js)

Set via `<meta name="cm-mode">` in `index.html`:
- `preview` (also the default when the tag is absent) — design-render mode: mock data from `mock/api.js`, no auth, every UI state reachable (incl. the Public Chat state switcher, P4)
- `production` — real API + Google login; mock modules are never imported

Why preview mode exists:
1. **Design review without a backend** — anyone can open the app and see every screen and state live (login errors, quota reached, revoked link, ingestion animations); some states are hard to reproduce with real data.
2. **Parallel development** — frontend work proceeds on fixtures before endpoints exist; the contract is the `mock/api.js` interface.
3. **Design iteration** — UI changes are rendered and verified in preview without touching production systems.

Production mode guarantees none of this (mock data, auth bypass, state switchers) ever reaches users — all of it is disabled by the one meta tag.

Legend: **View** = file in `views/`. All api calls go through the single adapter — integrating the backend = implementing `api/real.js` with the same method signatures.

---

## Auth & session

| # | Use case | View / component | api calls | Notes for integration |
|---|---|---|---|---|
| A1 | Sign in with Google | Login.js | `api.loginWithGoogle()` | Real: redirect to OAuth; on failure land on `/login?error=1` (view already renders the error state) |
| A2 | Restore session on load | app.js (`provide('user')`) | `api.me()` | 401 in production → redirect `/login`. Header + Account read the injected `user` ref |
| A3 | Log out | AppHeader.js menu, Account.js | `api.logout()` | Clear session, go to `/login` |
| A4 | Session expired mid-use | ErrorPage.js (`/error/session-expired`) | — | Wire a global 401 interceptor in `api/real.js` to route here or to `/login` |

## Bots (My Bots — MyBots.js)

| # | Use case | Owner | api calls | Notes |
|---|---|---|---|---|
| B1 | List bots (loading skeleton / empty / error / grid) | MyBots.js `gridState` | `api.listBots()` | Card badges map `status`: ready/pending/error |
| B2 | Create bot (+ at-limit disable & upsell) | MyBots.js create form | `api.createBot({name, description, language})`, limit from `api.plan()` | `atLimit` computed disables form; amber upsell links to `/account` |
| B3 | Delete bot (confirm dialog) | MyBots.js `confirm` | `api.deleteBot(id)` | |
| B4 | Usage widgets (LLM spend/tokens, per-model; plan bars) | MyBots.js | `api.usage()`, `api.plan()` | |
| B5 | Open bot | card → router `/bots/:id` | — | |

## Bot detail (BotDetail.js) — the heaviest view

| # | Use case | Owner (state) | api calls | Notes |
|---|---|---|---|---|
| D1 | Load bot (+ not-found error state) | `bot`, `loadError` | `api.getBot(id)` | Response feeds name, chunks, sources[], custom_prompt, telegram_username |
| D2 | Rename bot | `rename()` | `api.updateBot(id, {name})` | |
| D3 | Persona — builder flow (meta-prompt, 3-question interview, language, generate, read-only preview) | `persona`, `interview` | `api.interviewPersona()` (questions), **generate = mock-only** | Needs endpoint: generate persona from desc+answers+language; response → `persona.sysPrompt` |
| D4 | Persona — custom prompt (save/update, active indicator) | `persona.custom/savedCustom`, `personaSource` | **mock-only** | Needs save/clear custom_prompt endpoint; overrides builder |
| D5 | Add YouTube channel (consent required, auto-sync frequency) | `addYt` | `api.addYouTubeSource(botId, {channel, author, consent, auto_sync})` | Consent checkbox gates the button |
| D6 | Upload document | `addDoc` | **mock-only** | Needs multipart upload endpoint |
| D7 | Start import (pending → building) | `startImport(s)` | `api.startImport(botId, sourceId)` | UI then animates via `simulateImport` — replace with job polling/SSE |
| D8 | Ingestion queue (progress ring, counts, cancel, transcript log) | `queue`, `simulateImport`, `cancelJob` | **mock-only animation** | Real: poll job status; cancel endpoint returns source to `pending` |
| D9 | Sync now (manual incremental sync, animated) | `syncNow(s)` | **mock-only** | Real: trigger incremental sync job; only videos newer than index |
| D10 | Auto-sync per channel (Off/Daily/Weekly/Monthly) | `setAutoSync(s, v)` | **mock-only** | Needs PATCH source {auto_sync}; status line shows last/next sync |
| D11 | Delete source (confirm) | `removeSource(s)` | **mock-only** | Needs DELETE source |
| D12 | Rebuild index (busy state) | `rebuilding` | **mock-only** | Needs rebuild job endpoint |
| D13 | Test chat (grounded answer + cited sources, thinking state) | `chat`, `send()` | **mock-only** | Real: chat endpoint returning text + source citations |
| D14 | Telegram connect (5-step guide, token verify, connected card, disconnect confirm) | `tg` | **mock-only** | Needs token verify + disconnect endpoints |
| D15 | Share — create private link | `share*` state, Share modal | **mock-only** | Needs create-token endpoint; link format `/s/<token>` |
| D16 | Share — copy / rotate (confirm) / revoke (confirm) | Share modal | **mock-only** | Rotate = new token, old dies; revoke = link off |
| D17 | Guest quota display in share modal | `guestUsed` | **mock-only** | Real: per-bot daily guest message count |

## Settings (Settings.js)

| # | Use case | Owner | api calls | Notes |
|---|---|---|---|---|
| S1 | Managed vs BYOK mode switch | `mode` | `api.mySettings()` (load), `api.saveMySettings()` | |
| S2 | BYOK: pick provider per capability + model override | `prov`, `models` | part of `saveMySettings` body | Key fields for unselected providers are locked |
| S3 | BYOK: save keys with required-key validation | `save()`, `errors` | `api.saveMySettings({mode, llm_provider, keys})` | Selected provider without saved/entered key → blocked, field highlighted, error toast |
| S4 | Default auto-sync for new channels | `defaultSync` | **mock-only** | Needs user-setting endpoint; Add-channel form should default to it (today hardcoded weekly) |

## Account (Account.js)

| # | Use case | api calls | Notes |
|---|---|---|---|
| C1 | Profile + log out | injected `user`, `api.logout()` | |
| C2 | Plan usage bars + spend | `api.plan()` | |
| C3 | Upgrade tiers (Free/Pro/BYOK) | **mock-only** | CTA → checkout |
| C4 | Delete account (confirm) | `api.deleteAccount()` | |

## Public guest chat (PublicChat.js, `/s/:token`)

| # | Use case | Notes |
|---|---|---|
| P1 | Active chat with guest quota counter | **mock-only**; real: resolve token → bot, chat endpoint, quota header. Demo routes: `/s/demo` |
| P2 | Quota reached (banner, disabled input) | `/s/limit` demos it |
| P3 | Revoked/invalid link | `/s/revoked` demos it |
| P4 | Preview state switcher (bottom-left) | `IS_PREVIEW` only — never renders in production |

## Static / misc

| # | Use case | View |
|---|---|---|
| M1 | Landing (`/about`) | Landing.js |
| M2 | Terms / Privacy | LegalDoc.js (`kind` prop) |
| M3 | 404 / no-access / session-expired | ErrorPage.js |
| M4 | Theme toggle (dark/light, persisted `cm-theme`) | theme.js + AppHeader.js |
| M5 | Toasts (success/error) | local per view (`toasts`/`toast`) |
| M6 | Confirm dialogs (Escape closes) | local `confirm` per view |
| M7 | Custom dropdown (keyboard + aria) | components/SelectField.js — use for every select |

---

## Integration checklist (mock → real)

1. Implement `api/real.js` with the exact method set of `mock/api.js` (`me, plan, usage, mySettings, saveMySettings, deleteAccount, listBots, createBot, getBot, updateBot, deleteBot, addYouTubeSource, startImport, interviewPersona, logout, loginWithGoogle`).
2. Add the **missing endpoints** flagged `mock-only` above: persona generate/save, document upload, job status+cancel, sync now, auto-sync setting, delete source, rebuild, test chat, telegram, share tokens, guest chat/quota, default auto-sync, checkout.
3. Replace `simulateImport` in BotDetail.js with job polling (the queue-entry shape `{pct, done, failed}` is the contract).
4. Add a 401 interceptor → `/login` (or `/error/session-expired`).
5. `config.js`: production mode picks `api/real.js`; mock code never loads. Ship `index.html` with `<meta name="cm-mode" content="production">` — without it the app boots in preview (mock) mode.
