// Real API adapter — thin fetch wrappers over the FastAPI backend (src/ytrag/web/app.py).
// Same method set (and, where the views pass arguments, the same call signatures)
// as mock/api.js. In production api/index.js imports THIS module; the mock is never
// loaded. Transport only: it maps our JSON endpoints onto the shapes the views read.
import { reconnecting } from './connection.js';
import { DEV_LOGIN } from '../config.js';

// Transient-failure handling (deploy-UX). During a redeploy the app container is
// briefly down while docker-compose pulls + restarts it; Caddy stays up and either
// can't reach the upstream (fetch() rejects) or returns a gateway 502/503/504. Those
// are NOT real errors — they clear in seconds — so req() rides them out: retry a few
// times with short backoff and drive a global "Reconnecting…" pill instead of blowing
// up the view. A genuine response from the app (200, 401, a 4xx/5xx it produced
// itself) means we're connected and is handled exactly as before.
const TRANSIENT_STATUS = new Set([502, 503, 504]);
const BACKOFF_MS = [500, 1000, 2000, 4000]; // 4 retries → up to ~7.5s of bridging
const MAX_RETRIES = BACKOFF_MS.length;
// Only auto-retry methods that are idempotent per HTTP semantics (RFC 7231): a
// re-send is guaranteed to have the SAME effect as a single send. A transient
// failure (fetch reject / gateway 502-504) is ambiguous — the app may have ALREADY
// received and committed the request before the connection was reset (e.g. a POST
// container killed mid-flight during the rollout, or a slow chat that outran the
// grace period). Silently re-sending a POST/PATCH then would DUPLICATE the side
// effect: a second bot/source/import, a double-charged chat completion. So for the
// non-idempotent methods we still surface the redeploy via the reconnect pill but
// throw a transient error instead of re-submitting — the view lets the user retry
// once we're back. GET polls (the loops that most need to ride out a redeploy) and
// idempotent DELETE/PUT keep retrying, which is where the value is anyway.
const IDEMPOTENT_METHODS = new Set(['GET', 'HEAD', 'PUT', 'DELETE', 'OPTIONS']);

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

// Flip the reconnect indicator on and wait out the backoff before the next attempt.
function backoff(attempt) {
  reconnecting.value = true;
  return sleep(BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)]);
}

async function req(path, opts = {}) {
  const retryable = IDEMPOTENT_METHODS.has((opts.method || 'GET').toUpperCase());
  // No-login self-host mode: on the first 401 we transparently establish the local
  // dev session (GET /auth/dev sets the cookie) and retry ONCE, so a fresh visitor
  // is signed in with no Google wall. Guarded so we never loop.
  let triedDevLogin = false;
  for (let attempt = 0; ; attempt++) {
    let res;
    try {
      res = await fetch(path, {
        // Only set a JSON content-type when we actually send a JSON body. Multipart
        // uploads (FormData) must NOT get this header — the browser sets the boundary.
        headers: opts.json ? { 'Content-Type': 'application/json' } : {},
        credentials: 'same-origin',
        ...opts,
        body: opts.json ? JSON.stringify(opts.json) : opts.body,
      });
    } catch (e) {
      // fetch() rejects only on a network-level failure (upstream unreachable, DNS,
      // connection reset) — precisely the redeploy window. Retry it as transient,
      // but only for idempotent requests (see IDEMPOTENT_METHODS).
      if (retryable && attempt < MAX_RETRIES) { await backoff(attempt); continue; }
      // Still unreachable (or a non-retryable method). Show the reconnect pill so the
      // UI reflects reality; the polling loops keep trying and will clear it on success.
      reconnecting.value = true;
      const err = new Error('offline');
      err.transient = true;
      throw err;
    }
    // A gateway 502/503/504 is Caddy telling us the app upstream is down — same class
    // of failure as a network throw. Retry idempotent requests with backoff before
    // treating it as real; non-idempotent ones surface immediately (no silent re-POST).
    if (TRANSIENT_STATUS.has(res.status)) {
      if (retryable && attempt < MAX_RETRIES) { await backoff(attempt); continue; }
      reconnecting.value = true;
      const err = new Error('offline');
      err.status = res.status;
      err.transient = true;
      throw err; // pill stays up; a later poll/action recovers it
    }
    // The app answered — we're connected again. Clear the indicator and handle the
    // response exactly as before.
    reconnecting.value = false;
    // An expired/absent session now returns 401 JSON for /api/* (backend H2 change),
    // not a 307 HTML redirect. Route the hash-router SPA to its login view.
    if (res.status === 401) {
      if (DEV_LOGIN && !triedDevLogin) {
        triedDevLogin = true;
        // No-login self-host: establish the local dev session, then retry the request.
        // /auth/dev sets the session cookie and redirects; we only need the cookie, so
        // the body is ignored. Best-effort — a failure falls through to the login view.
        try { await fetch('/auth/dev', { credentials: 'same-origin' }); } catch (e) {}
        continue;
      }
      if (window.location.hash !== '#/login') window.location.hash = '#/login';
      throw new Error('unauthenticated');
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      // Surface the backend's {"error": …} shape, and attach the status so callers
      // that must branch on it (guest chat 429 → quota state, share-info 404 →
      // revoked state) can read err.status without re-parsing the message.
      const err = new Error(data.error || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }
}

// The real /api/me/plan returns limits as {max_sources_per_bot, max_sources_total}
// (see plans.Limits); the plan widgets (MyBots.js, Account.js) read limits.max_sources
// as the account-wide source ceiling. Bridge the two without dropping the real keys.
function adaptPlan(p) {
  if (p && p.limits && p.limits.max_sources === undefined) {
    p.limits.max_sources = p.limits.max_sources_total;
  }
  return p;
}

// The backend serializes each source with type 'youtube'|'document' (bots.py
// BotSource), but BotDetail.normalizeSource keys its Channel/Playlist tag and its
// icon off type 'yt'|'doc'. Map the type and pass `kind` (channel|playlist)
// through so a playlist source renders as a playlist. Mirrors the mock getBot
// shape; leaves every other field untouched.
function adaptBot(b) {
  if (b && Array.isArray(b.sources)) {
    b.sources = b.sources.map(s => ({
      ...s,
      type: s.type === 'youtube' ? 'yt' : s.type === 'document' ? 'doc' : s.type,
      kind: s.kind || (s.type === 'youtube' ? 'channel' : null),
    }));
  }
  return b;
}

export const api = {
  me: () => req('/api/me').then(d => d.user),
  plan: () => req('/api/me/plan').then(adaptPlan),
  usage: () => req('/api/me/usage'),
  // GET returns the masked settings unwrapped; POST wraps it as {ok, settings}
  // (app.py api_save_my_settings). Unwrap so both hand the views the SAME masked
  // shape (mode/llm_provider/keys_set) that Settings.js hydrate(res) reads. The
  // `(r && r.settings) ? r.settings : r` guard handles BOTH shapes: a bare GET
  // masked payload has no `.settings` key so it passes through untouched.
  mySettings: () => req('/api/me/settings'),
  saveMySettings: (body) => req('/api/me/settings', { method: 'POST', json: body })
    .then(r => (r && r.settings) ? r.settings : r),
  deleteAccount: () => req('/api/me', { method: 'DELETE' }),

  listBots: () => req('/api/bots').then(d => d.bots),
  createBot: (body) => req('/api/bots', { method: 'POST', json: body }).then(d => d.bot),
  getBot: (id) => req(`/api/bots/${id}`).then(d => adaptBot(d.bot)),
  updateBot: (id, body) => req(`/api/bots/${id}`, { method: 'PATCH', json: body }).then(d => d.bot),
  // Persona custom prompt (Phase J): a non-empty text saves/overrides the builder
  // persona at chat time; passing '' clears it (reverts to the builder persona).
  // Rides the same PATCH path as updateBot and returns the updated bot (incl.
  // custom_prompt) so the view can re-derive the active-persona indicator.
  saveCustomPrompt: (id, text) => req(`/api/bots/${id}`, { method: 'PATCH', json: { custom_prompt: text } }).then(d => d.bot),
  deleteBot: (id) => req(`/api/bots/${id}`, { method: 'DELETE' }),

  addYouTubeSource: (botId, body) => req(`/api/bots/${botId}/sources/youtube`, { method: 'POST', json: body }),
  startImport: (botId, sourceId, body = {}) => req(`/api/bots/${botId}/sources/${sourceId}/import`, { method: 'POST', json: body }),
  // Per-source auto-sync (Phase K): set the frequency (off|daily|weekly|monthly,
  // or '' to inherit the user default), and trigger an incremental "Sync now"
  // that enqueues ONLY videos newer than what's already indexed. syncSource
  // returns {new: <count queued>}.
  setSyncFreq: (botId, sourceId, freq) => req(`/api/bots/${botId}/sources/${sourceId}/sync-freq`, { method: 'POST', json: { sync_freq: freq } }),
  syncSource: (botId, sourceId) => req(`/api/bots/${botId}/sources/${sourceId}/sync`, { method: 'POST' }),
  interviewPersona: (botId, body) => req(`/api/bots/${botId}/persona/interview`, { method: 'POST', json: body }),

  // Document upload is multipart/form-data — NEVER a JSON body (the browser must
  // own the boundary), so it goes through `body` (FormData), not `json`. The
  // backend enforces a server-side rights attestation, hence the consent flag.
  uploadDocument: (botId, { file, author, consent }) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('author', author);
    fd.append('consent', consent ? 'true' : 'false');
    return req(`/api/bots/${botId}/sources/document`, { method: 'POST', body: fd });
  },
  deleteSource: (botId, sourceId) => req(`/api/bots/${botId}/sources/${sourceId}`, { method: 'DELETE' }),
  rebuild: (botId) => req(`/api/bots/${botId}/rebuild`, { method: 'POST' }),
  chat: (botId, body) => req(`/api/bots/${botId}/chat`, { method: 'POST', json: body }),

  // Streaming owner chat (item 8b). POST /api/bots/{id}/chat/stream returns a
  // text/event-stream: repeated `event: delta\ndata: {"text":…}` frames, then a
  // terminal `event: done\ndata: {sources,…}`. We read the body incrementally,
  // split SSE frames on the blank-line delimiter, and invoke onDelta per delta /
  // onDone once. NOT routed through req(): streaming needs its own fetch, and a
  // chat send is non-idempotent so it must never be silently re-issued. A 409
  // (backend rejects a 2nd concurrent send per chat) surfaces via err.status so
  // the view can show a friendly note instead of an error bubble.
  chatStream: async (botId, body, { onDelta, onDone, signal } = {}) => {
    let res;
    try {
      res = await fetch(`/api/bots/${botId}/chat/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin', body: JSON.stringify(body), signal,
      });
    } catch (e) {
      reconnecting.value = true;
      const err = new Error('offline'); err.transient = true; throw err;
    }
    if (res.status === 401) {
      if (window.location.hash !== '#/login') window.location.hash = '#/login';
      throw new Error('unauthenticated');
    }
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      const err = new Error(data.error || `HTTP ${res.status}`); err.status = res.status; throw err;
    }
    reconnecting.value = false;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', result = { sources: [] };
    const handleFrame = (frame) => {
      let event = 'message', data = '';
      for (const line of frame.split(/\r\n|\r|\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) return;
      let payload; try { payload = JSON.parse(data); } catch (e) { return; }
      if (event === 'delta') { if (payload.text && onDelta) onDelta(payload.text); }
      else if (event === 'done') { result = payload; if (onDone) onDone(payload); }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line. Our server emits LF, but a proxy may
      // rewrite endings, so accept \n\n, \r\n\r\n and \r\r. A lone trailing \r at a
      // chunk boundary matches nothing and stays buffered until its pair arrives.
      let m;
      while ((m = buf.match(/\r\n\r\n|\n\n|\r\r/))) {
        handleFrame(buf.slice(0, m.index)); buf = buf.slice(m.index + m[0].length);
      }
    }
    if (buf.trim()) handleFrame(buf);
    return result;
  },

  // Ingest queue: list all of the user's jobs, poll one, or cancel one. The queue
  // card in BotDetail derives its {pct, done, failed} rows from the IngestJob
  // fields (videos_total/done/failed, status) these return.
  ingestJobs: () => req('/api/ingest/jobs').then(d => d.jobs || []),
  ingestJob: (jobId) => req(`/api/ingest/jobs/${jobId}`),
  cancelJob: (jobId) => req(`/api/ingest/jobs/${jobId}/cancel`, { method: 'POST' }),
  // Requeue a FAILED import (error-row Retry): enqueues a fresh job reusing the
  // old params and replaces the errored row. Returns the new IngestJob.
  retryJob: (jobId) => req(`/api/ingest/jobs/${jobId}/retry`, { method: 'POST' }),
  // Drop a finished (done/error/cancelled) job row so it stops being reported.
  dismissJob: (jobId) => req(`/api/ingest/jobs/${jobId}`, { method: 'DELETE' }),
  // Rebuild jobs (document add / source delete / rebuild button) live in the
  // in-memory JobManager and are polled via a different endpoint + snapshot shape.
  jobStatus: (jobId) => req(`/api/jobs/${jobId}`),

  // Private share link lifecycle (owner). Each returns the same _share_info shape:
  // {active, path, url, created_at, daily_cap, today_count}.
  getShare: (botId) => req(`/api/bots/${botId}/share`),
  createShare: (botId) => req(`/api/bots/${botId}/share`, { method: 'POST' }),
  rotateShare: (botId) => req(`/api/bots/${botId}/share/rotate`, { method: 'POST' }),
  revokeShare: (botId) => req(`/api/bots/${botId}/share`, { method: 'DELETE' }),

  // Telegram deploy: connect verifies the token server-side and returns the bot
  // username; disconnect stops the poller and clears the stored token.
  connectTelegram: (botId, token) => req(`/api/bots/${botId}/telegram`, { method: 'POST', json: { token } }),
  disconnectTelegram: (botId) => req(`/api/bots/${botId}/telegram`, { method: 'DELETE' }),

  // Guest (no login): share-info for the guest page, and the token-scoped chat.
  // Both hit /api/s/{token}/* — the token IS the credential, no cookie needed.
  shareInfo: (token) => req(`/api/s/${token}`),
  guestChat: (token, body) => req(`/api/s/${token}/chat`, { method: 'POST', json: body }),

  logout: () => { window.location.href = '/logout'; },
  loginWithGoogle: () => { window.location.href = '/auth/google'; },
};
