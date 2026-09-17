# Frontend map — who owns what

Buildless Vue 3 (ES modules, no bundler). Entry: `index.html` → `app.js` → `router.js`.

## Views (screens) — `views/`
| File | Route | Flow it owns |
|---|---|---|
| MyBots.js | `/` | Bot list, create bot, delete confirm, usage/plan/models widgets |
| BotDetail.js | `/bots/:id` | Persona (builder + custom prompt), sources lifecycle (add YT/doc, import, sync, auto-sync), ingestion queue, test chat, Telegram connect, **Share (private link: create/copy/rotate/revoke)** |
| Settings.js | `/settings` | Managed vs BYOK, providers, API keys (+required validation), default auto-sync |
| Account.js | `/account` | Profile, plan/usage, upgrade tiers, delete account |
| Login.js | `/login` | Google sign-in (+error state via `?error`) |
| PublicChat.js | `/s/:token` | Guest chat via private link. States: active / link revoked / guest quota reached (`/s/revoked`, `/s/limit` demo them; preview has a state switcher) |
| Landing.js | `/about` | Marketing page |
| LegalDoc.js | `/terms`, `/privacy` | Terms + Privacy (one component, `kind` prop) |
| ErrorPage.js | `/error/:variant`, catch-all | 404 / no-access / session-expired |

## Shared — `components/`
- `AppHeader.js` — top nav, user menu, theme toggle (only on non-`bare` routes)
- `SelectField.js` — the custom dropdown (keyboard: arrows/Enter/Esc). Use it everywhere; never native `<select>`.

## Infrastructure
- `styles.css` — all design tokens (`:root` dark, `[data-theme="light"]` light) + shared classes. No hardcoded surface colors in views — use tokens.
- `theme.js` — theme state, persisted in localStorage (`cm-theme`)
- `config.js` — `MODE` / `IS_PREVIEW` switch
- `api/index.js` — picks adapter by mode; `mock/api.js` = fixtures for preview; real adapter = backend's job
- Share flow lives ONLY in BotDetail. My Bots has no share UI (removed by design).
- Full use-case → component → api map: see `USECASES.md`.
- Legacy Jinja templates: `legacy/` — reference only, do not ship.
