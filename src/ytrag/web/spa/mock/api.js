// Mock API adapter — same interface as api/real.js, backed by in-memory fixtures.
import * as fx from './fixtures.js';

const delay = (ms = 220) => new Promise(r => setTimeout(r, ms));
let bots = fx.bots.map(b => ({ ...b }));
let nextId = 5;

// ---- in-memory job simulators (preview mode only) --------------------------
// The views now drive their queue/progress off the real job-polling endpoints
// instead of a local animation, so the mock has to hand back *progressing* jobs
// or the preview would never advance. The worker is strictly sequential (FIFO):
// one `running` job at a time, the rest wait as `queued` — matching production.
// Jobs carry `origin: 'manual' | 'auto'` (the small backend flag from the
// ingestion-UX brief) so auto-sync rows can be labelled.
let jobSeq = 1;
const ingestJobs = {}; // job_id -> IngestJob-shaped record
const jobOrder = [];   // FIFO order
const rebuildJobs = {}; // job_id -> Job.snapshot()-shaped record

function newIngestJob(botId, sourceId, channelUrl, opts = {}) {
  const id = 'jmock' + jobSeq++;
  // window_total = videos THIS run will index; channel_total = channel's approx
  // public video count (>= window when the channel is larger than the window).
  const windowTotal = opts.window ?? opts.total ?? 20;
  ingestJobs[id] = {
    job_id: id, bot_id: botId, status: opts.status || 'queued',
    source_id: sourceId, channel_url: channelUrl || sourceId,
    videos_total: windowTotal, window_total: windowTotal,
    channel_total: opts.channel ?? windowTotal,
    videos_done: opts.done ?? 0,
    videos_failed: opts.failed ?? 0,
    videos_skipped: opts.skipped ?? 0,
    origin: opts.origin || 'manual', error: opts.error || null,
  };
  jobOrder.push(id);
  return id;
}
// One tick of the sequential worker: advance the running job; when it finishes
// (or none is running) promote the oldest queued job.
function stepQueue() {
  const jobs = jobOrder.map(id => ingestJobs[id]).filter(Boolean);
  let running = jobs.find(j => j.status === 'running');
  if (!running) { running = jobs.find(j => j.status === 'queued'); if (running) running.status = 'running'; }
  if (!running) return;
  running.videos_done = Math.min(running.videos_total, running.videos_done + Math.max(1, Math.ceil(running.videos_total / 30)));
  // A few videos have no captions — they're SKIPPED, not failed (BUG-016).
  if (Math.random() < 0.35) running.videos_skipped = Math.min(Math.round(running.window_total * 0.04), (running.videos_skipped || 0) + 1);
  if (running.videos_done >= running.videos_total) running.status = 'done';
}
// Seed demo jobs so the preview shows every ingestion state out of the box
// (Kitchen Alchemy / b1): a running import, two waiting (one auto-sync), an
// errored one, and a cancelled one — plus a queued import on Woodshop Mentor
// so the cross-page indicators have something to count.
newIngestJob('b1', 's2', '@alexbakes', { status: 'running', window: 300, channel: 2500, done: 142, failed: 3, skipped: 8 });
newIngestJob('b1', 's6', '@slow-tv-archive', { status: 'queued', window: 500, channel: 1840 });
newIngestJob('b1', 's1', '@kitchenalchemy', { status: 'queued', window: 25, channel: 25, origin: 'auto' });
newIngestJob('b1', 's5', '@retro-restores', { status: 'error', window: 388, channel: 1200, done: 134, failed: 6, skipped: 4, error: 'Transcript proxy rate-limited — try again in a few minutes' });
newIngestJob('b1', 's-old', '@urban-gardening', { status: 'cancelled', window: 210, channel: 210, done: 87 });
newIngestJob('b2', 's-w1', '@woodshop-archive', { status: 'queued', window: 96, channel: 96 });
function newRebuildJob(label) {
  const id = 'rmock' + jobSeq++;
  rebuildJobs[id] = { id, label, status: 'running', stage: 'starting', total: 0, processed: 0, saved: 0, indexed: 0, log: [], error: null };
  return id;
}
function stepRebuild(j) {
  if (j.status !== 'running') return;
  if (j.stage === 'starting') j.stage = 'building';
  else { j.stage = 'done'; j.status = 'done'; j.indexed = 1279; }
}

export const api = {
  me: async () => { await delay(120); return { ...fx.user }; },
  plan: async () => {
    await delay();
    const p = JSON.parse(JSON.stringify({ ...fx.plan, usage: { ...fx.plan.usage, bots_used: bots.length } }));
    // Mirror real.js adaptPlan: bridge the account-wide source ceiling the plan
    // widgets read as limits.max_sources onto the real max_sources_total field.
    if (p.limits && p.limits.max_sources === undefined) p.limits.max_sources = p.limits.max_sources_total;
    return p;
  },
  usage: async () => { await delay(); return JSON.parse(JSON.stringify(fx.usage)); },
  // Realistic managed-mode masked shape (mirrors UserSettingsStore.masked): mode,
  // llm_provider, per-chat-provider keys_set yes/no, keystore + plan toggles.
  mySettings: async () => { await delay(); return { mode: 'managed', llm_provider: 'anthropic', embed_provider: 'voyage', transcription_provider: 'groq', vision_provider: 'openrouter', keys_set: { anthropic: true, openai: false, openrouter: false }, keystore_ready: true, byok_enabled: true, managed_enabled: true, sync_freq: 'weekly', email_notifications: true }; },
  saveMySettings: async (body) => {
    await delay();
    // Build the REAL POST wire envelope {ok, settings:<masked>} (app.py returns
    // {"ok": True, "settings": store.masked(...)}), reflecting any keys just set,
    // then apply the SAME unwrap real.js does. This way preview runs the identical
    // unwrap path as production — if that unwrap regressed, preview would break too
    // (it can't silently hide the "hydrate got the envelope, not the settings" bug).
    const keys_set = { anthropic: true, openai: false, openrouter: false };
    const inKeys = (body && body.keys) || {};
    ['anthropic', 'openai', 'openrouter'].forEach(k => { if (inKeys[k] && inKeys[k].trim()) keys_set[k] = true; });
    const settings = { mode: (body && body.mode) || 'managed', llm_provider: (body && body.llm_provider) || 'anthropic', embed_provider: (body && body.embed_provider) || 'voyage', transcription_provider: (body && body.transcription_provider) || 'groq', vision_provider: (body && body.vision_provider) || 'openrouter', keys_set, keystore_ready: true, byok_enabled: true, managed_enabled: true, sync_freq: (body && body.sync_freq) || 'weekly', email_notifications: (body && typeof body.email_notifications === 'boolean') ? body.email_notifications : true };
    const wire = { ok: true, settings }; // exact real POST envelope
    return (wire && wire.settings) ? wire.settings : wire; // same unwrap as real.js
  },
  deleteAccount: async () => { await delay(); return { ok: true }; },

  listBots: async () => { await delay(350); return bots.map(b => ({ ...b })); },
  createBot: async ({ name, description = '', language = 'English' }) => {
    await delay(300);
    const bot = { id: 'b' + nextId++, name, description: description || 'No description yet.', status: 'pending', source_count: 0, chunk_count: 0, telegram_username: null, persona: '', language, suggest_followups: true };
    bots.push(bot);
    return { ...bot };
  },
  getBot: async (id) => {
    await delay();
    const b = bots.find(x => x.id === id);
    if (!b) throw new Error('not found');
    const sources = id === 'b1' ? [
      { id: 's1', type: 'yt', label: '@kitchenalchemy', meta: 'Kitchen Alchemy · author: Alex Rivera · 340 items · 11,204 chunks', status: 'ready', avatar: 'K', avatarBg: 'linear-gradient(135deg,#f59e0b,#ef4444)', autoSync: 'weekly', lastSynced: '2h ago' },
      { id: 's2', type: 'yt', label: '@alexbakes', meta: 'Alex Bakes · author: Alex Rivera · 118 items · 0 chunks', status: 'building', avatar: 'A', avatarBg: 'linear-gradient(135deg,#6366f1,#8b5cf6)' },
      { id: 's3', type: 'doc', label: 'family-recipes-vol2.pdf', meta: 'author: Alex Rivera · 1 item · 1,279 chunks', status: 'ready' },
      { id: 's7', type: 'yt', kind: 'playlist', label: 'Knife Skills — Full Course', meta: 'playlist · author: Alex Rivera · 24 items · 812 chunks', status: 'ready', avatar: 'P', avatarBg: 'linear-gradient(135deg,#10b981,#6366f1)', autoSync: 'off', lastSynced: '5d ago' },
      { id: 's8', type: 'yt', kind: 'video', label: 'The 3-Hour Sourdough Deep Dive', meta: 'video · author: Alex Rivera · 1 item · 96 chunks', status: 'ready', avatar: 'V', avatarBg: 'linear-gradient(135deg,#ef4444,#f59e0b)', autoSync: 'off' },
      { id: 's5', type: 'yt', label: '@retro-restores', meta: 'Retro Restores · author: Alex Rivera · 134 items · 4,206 chunks', status: 'error', avatar: 'R', avatarBg: 'linear-gradient(135deg,#f59e0b,#f43f5e)', autoSync: 'daily', lastSynced: '3h ago', sync_error: true },
      { id: 's6', type: 'yt', label: '@slow-tv-archive', meta: 'Slow TV Archive · author: Alex Rivera · 540 videos found · 0 chunks', status: 'pending', avatar: 'S', avatarBg: 'linear-gradient(135deg,#06b6d4,#6366f1)' },
      { id: 's4', type: 'yt', label: '@weeknight-meals', meta: 'Weeknight Meals · author: Alex Rivera · 0 items · 0 chunks', status: 'pending' },
    ] : [];
    return { ...b, sources };
  },
  updateBot: async (id, body) => { await delay(); const b = bots.find(x => x.id === id); Object.assign(b, body); return { ...b }; },
  // Phase J: same PATCH-backed persona custom-prompt save/clear as real.js.
  saveCustomPrompt: async (id, text) => { await delay(); const b = bots.find(x => x.id === id); if (b) b.custom_prompt = text; return { ...b, custom_prompt: text }; },
  deleteBot: async (id) => { await delay(280); bots = bots.filter(x => x.id !== id); return { ok: true }; },

  addYouTubeSource: async (botId, body) => { await delay(); return { source_id: 's-mock' + jobSeq, kind: (body && body.kind) || 'channel', sync_freq: (body && (body.sync_freq || body.auto_sync)) || 'off' }; },
  startImport: async (botId, sourceId) => { await delay(); return { job_id: newIngestJob(botId, sourceId, sourceId), source_id: sourceId }; },
  // Phase K per-source auto-sync mock: echo the freq back, and simulate an
  // incremental sync that queues a small batch of "new" videos as an ingest job.
  setSyncFreq: async (botId, sourceId, freq) => { await delay(); return { source_id: sourceId, sync_freq: freq || '' }; },
  syncSource: async (botId, sourceId) => { await delay(); newIngestJob(botId, sourceId, sourceId, { total: 3 }); return { source_id: sourceId, new: 3 }; },
  // The real interview is iterative ({done, question, persona}): up to 3 SHORT
  // description-tailored questions, one per call, then the final system prompt.
  // The mock simulates that loop (2 questions, then done) with an LLM-like delay
  // so the "composing a question" indicator is actually visible in preview.
  interviewPersona: async (botId, body) => {
    await delay(1400);
    const answered = ((body && body.answers) || []).length;
    const qs = [
      'You mentioned explaining the “why” behind techniques — when a viewer just wants a quick answer, should the bot stay brief or still add the reasoning?',
      'How should it handle questions your videos don’t cover — admit the gap, or suggest the closest video you do have?',
    ];
    if (answered < qs.length) return { done: false, question: qs[answered], persona: null };
    const heard = ((body && body.answers) || []).map(a => a.answer).filter(Boolean).join(' ');
    return {
      done: true, question: null,
      persona: `You are the assistant for this creator’s channel.\n\nVoice: warm, practical, never condescending. Explain the “why” behind techniques${heard ? ', adapting depth to the question' : ''}.\n\nRules:\n• Answer ONLY from the indexed transcripts and documents; cite video titles.\n• If a topic isn’t covered, say so and point to the closest related video.\n• Always offer substitutions where relevant.\n• Keep answers concise; expand only when asked.`,
    };
  },

  uploadDocument: async (botId, { author }) => { await delay(); return { job_id: newRebuildJob('doc: ' + (author || 'upload')), source_id: 'd-mock' + jobSeq }; },
  deleteSource: async () => { await delay(280); return { ok: true, job_id: newRebuildJob('rebuild after delete') }; },
  rebuild: async () => { await delay(); return { job_id: newRebuildJob('rebuild') }; },
  chat: async (botId, body) => {
    await delay(1200);
    return {
      answer: "Great question! The key is **temperature control** — brown butter develops its nutty flavor at around 250°F when the milk solids toast.\n\n- Use a *light-colored* pan so you can watch the color turn\n- Swirl constantly once it foams\n- Pull it off heat the moment it smells nutty — it keeps cooking\n\n💡 You could ask: how do I store brown butter, or which sauces start from it?",
      sources: [{ title: 'Brown Butter, Explained', url: 'https://youtu.be/dQw4w9WgXcQ?t=95' }, { title: '5 Sauces Every Cook Should Master', url: 'https://youtu.be/dQw4w9WgXcQ?t=210' }],
    };
  },
  // Streaming owner chat (item 8b): emit a few deltas then a done frame, so the
  // progressive-typing path works offline exactly like the real SSE endpoint.
  chatStream: async (botId, body, { onDelta, onDone } = {}) => {
    const answer = "Great question! The key is **temperature control** — brown butter develops its nutty flavor at around 250°F when the milk solids toast.\n\n- Use a *light-colored* pan so you can watch the color turn\n- Swirl constantly once it foams\n- Pull it off heat the moment it smells nutty — it keeps cooking\n\n💡 You could ask: how do I store brown butter, or which sauces start from it?";
    const sources = [{ title: 'Brown Butter, Explained', url: 'https://youtu.be/dQw4w9WgXcQ?t=95' }, { title: '5 Sauces Every Cook Should Master', url: 'https://youtu.be/dQw4w9WgXcQ?t=210' }];
    await delay(650); // latency before the first delta — shows the typing indicator
    const chunks = answer.match(/[\s\S]{1,22}/g) || [answer];
    for (const c of chunks) { if (onDelta) onDelta(c); await delay(75); }
    const done = { sources, grounded: true, model: 'claude-sonnet-4' };
    if (onDone) onDone(done);
    return done;
  },

  ingestJobs: async () => { await delay(120); stepQueue(); return jobOrder.map(id => ingestJobs[id]).filter(Boolean).map(j => ({ ...j })); },
  ingestJob: async (jobId) => { await delay(120); stepQueue(); const j = ingestJobs[jobId]; return j ? { ...j } : { job_id: jobId, status: 'done', videos_total: 0, videos_done: 0, videos_failed: 0 }; },
  cancelJob: async (jobId) => { await delay(); const j = ingestJobs[jobId]; if (j) j.status = 'cancelled'; return j ? { ...j } : { job_id: jobId, status: 'cancelled' }; },
  // Requeue an errored job (design: error-row Retry). Real endpoint lands with
  // the backend `origin` flag work.
  retryJob: async (jobId) => { await delay(); const j = ingestJobs[jobId]; if (j) { j.status = 'queued'; j.error = null; } return j ? { ...j } : { job_id: jobId, status: 'queued' }; },
  // Drop a finished (error/cancelled) job row so it stops being reported.
  dismissJob: async (jobId) => { await delay(120); delete ingestJobs[jobId]; return { ok: true }; },
  jobStatus: async (jobId) => { await delay(120); const j = rebuildJobs[jobId]; if (j) stepRebuild(j); return j ? { ...j } : { id: jobId, status: 'done', stage: 'done', indexed: 0, log: [] }; },

  getShare: async () => { await delay(); return { active: false, path: '', url: '', created_at: '', daily_cap: 200, today_count: 0 }; },
  createShare: async () => { await delay(); return { active: true, path: '/s/preview-token', url: 'http://localhost:8000/s/preview-token', created_at: '', daily_cap: 200, today_count: 0 }; },
  rotateShare: async () => { await delay(); return { active: true, path: '/s/rotated-token', url: 'http://localhost:8000/s/rotated-token', created_at: '', daily_cap: 200, today_count: 0 }; },
  revokeShare: async () => { await delay(); return { active: false, path: '', url: '', created_at: '', daily_cap: 200, today_count: 0 }; },

  connectTelegram: async (botId, token) => { await delay(1200); return { ok: true, username: '@preview_bot' }; },
  disconnectTelegram: async () => { await delay(); return { ok: true }; },

  shareInfo: async () => { await delay(); return { active: true, bot_name: 'Kitchen Alchemy', avatar: '', source_summary: "Answers grounded in the channel's videos and recipes", quota: { used_today: 12, daily_cap: 200 } }; },
  guestChat: async () => {
    await delay(1200);
    return {
      answer: "The trick is starting with a **cold pan** — the fat renders slowly and the skin crisps without burning.\n\n- Score the skin lightly, salt it 20 minutes ahead\n- Medium-low heat, skin-side down, don't move it\n- Flip only when the skin releases on its own",
      sources: [{ title: 'Crispy Skin, Every Time', url: 'https://youtu.be/dQw4w9WgXcQ?t=132' }, { title: 'Pan Basics: Heat Control', url: 'https://youtu.be/dQw4w9WgXcQ?t=45' }],
    };
  },

  logout: () => { window.location.hash = '#/login'; },
  loginWithGoogle: () => { window.location.hash = '#/'; },
};
