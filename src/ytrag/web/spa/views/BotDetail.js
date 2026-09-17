import { ref, reactive, computed, onMounted, onBeforeUnmount, watch } from 'vue';
import { api } from '../api/index.js';
import { useRoute } from 'vue-router';
import SelectField from '../components/SelectField.js';
import { renderMarkdown } from '../mdlite.js';

const STATUS = { ready: 'Ready', building: 'Building', pending: 'Pending', error: 'Error', running: 'running', queued: 'Queued', cancelled: 'Cancelled' };

export default {
  components: { SelectField },
  setup() {
    const route = useRoute();
    const botId = route.params.id;
    const bot = ref(null);
    const loadError = ref(false);
    const renamed = ref(false);
    // Transient "Saved ✓" flag for the Language field (mirrors `renamed`).
    const langSaved = ref(false);
    let langSavedTimer;  // tracked so we can cancel it on re-save / unmount
    const persona = reactive({ mode: 'builder', desc: '', custom: '', savedCustom: '', lang: '', generating: false, status: '', statusColor: 'var(--text-faint)', sysPrompt: '' });
    // Live conversational interview: `thread` holds the visible back-and-forth
    // ({role: 'ai'|'user', text}); `qa` mirrors it as {question, answer} pairs for
    // the interviewPersona endpoint; `thinking` gates the composing indicator.
    const interview = reactive({ started: false, thinking: false, done: false, answer: '', thread: [], qa: [], qNum: 0, saved: false, saving: false });
    const sources = ref([]);
    const queue = ref([]);
    const addYt = reactive({ open: false, kind: 'channel', url: '', author: '', consent: false, sync: 'weekly', shorts: false });
    // `fileObj` holds the actual File for the multipart upload; `file` stays the
    // display name the template renders. Never expose fileObj to the template.
    const addDoc = reactive({ open: false, file: '', fileObj: null, author: '' });
    const chat = reactive({ input: '', thinking: false, streaming: false, messages: [] });
    // OSS build: the «Мислення»/Thinking general-advisor mode is disabled, so the
    // chat always runs the grounded «Довідник»/Reference path. chatMode is pinned to
    // "reference" and the toggle UI is not rendered. (setChatMode kept as an inert
    // no-op so the existing template binding surface stays intact.)
    // Original: "reference" (default) answers strictly from the channel's sources; "thinking"
    // reasons over the whole dialogue. Persisted per bot so the choice sticks across
    // visits; sent with every chat send so the backend selects the answer path.
    const chatMode = ref('reference');   // pinned; Thinking mode disabled in OSS build
    const setChatMode = (_m) => {};  // inert: toggle removed, grounded Reference only
    const tg = reactive({ connected: false, token: '', verifying: false, username: '' });
    const shareOpen = ref(false);
    const rebuilding = ref(false);
    const shareLink = ref('');
    const guestUsed = ref(0);
    const guestCap = ref(200); // real per-bot/plan daily cap, from the share endpoints
    const copied = ref(false);
    const confirm = ref(null);
    const toasts = ref([]);
    let nextToast = 1;

    const toast = (msg, ok = true) => {
      const id = nextToast++;
      toasts.value.push({ id, msg, ok });
      setTimeout(() => { toasts.value = toasts.value.filter(t => t.id !== id); }, 3200);
    };

    const onEsc = (e) => { if (e.key === 'Escape') { confirm.value = null; shareOpen.value = false; } };
    onMounted(() => document.addEventListener('keydown', onEsc));
    onBeforeUnmount(() => document.removeEventListener('keydown', onEsc));

    onMounted(async () => {
      try {
        const b = await api.getBot(botId);
        bot.value = b;
        // The builder "description" box is a SHORT, one-line brief of the bot —
        // seed it from the real `description` field, NOT the saved generated system
        // prompt (`b.persona`), which belongs in the editable "Generated persona"
        // box below.
        persona.desc = b.description || '';
        persona.sysPrompt = b.persona || '';
        // L1: seed the language field from the bot's stored value; leave it BLANK
        // when the bot never had one, so a persona save can't silently write
        // "English" back to it (see langForPatch()).
        persona.lang = b.language || '';
        // include-Shorts default reflects the bot's current value (BUG-009).
        addYt.shorts = !!b.include_shorts;
        // A previously saved builder persona is surfaced as a completed interview:
        // it populates the editable generated-persona textarea, lights the "Active:
        // generated persona" indicator (personaSource → 'builder'), and enables
        // carry-over into Custom mode — without replaying any interview questions.
        if (b.persona) { interview.done = true; interview.saved = true; }
        sources.value = (b.sources || []).map(normalizeSource);
        // custom_prompt (Phase J1): the backend now returns it. When a custom prompt
        // is saved, open the persona card straight into Custom mode with the saved
        // text prefilled and marked active (savedCustom drives the "Active: custom
        // prompt" indicator). Empty → builder mode, as before. Defensive on absence.
        if (b.custom_prompt) { persona.mode = 'custom'; persona.custom = b.custom_prompt; persona.savedCustom = b.custom_prompt; }
        if (b.telegram_username) { tg.connected = true; tg.username = b.telegram_username; }
        // Rebuild any in-flight ingest state from the server (survives a reload),
        // then poll it live. Replaces the old local simulateImport animation (H3).
        await refreshJobs();
        ensurePolling();
      } catch (e) { loadError.value = true; }
      // Preload the current private-share state (link + live guest quota) so the
      // Share modal opens already reflecting reality.
      try { applyShare(await api.getShare(botId)); } catch (e) { /* no share yet / non-fatal */ }
    });

    // Refresh the share modal's live guest count each time it opens.
    watch(shareOpen, async (open) => {
      if (!open) return;
      try { applyShare(await api.getShare(botId)); } catch (e) { /* non-fatal */ }
    });

    function normalizeSource(s) {
      return {
        id: s.id, type: s.type === 'youtube' || s.type === 'yt' ? 'yt' : 'doc',
        label: s.label || s.key || s.name, meta: s.meta || `author: ${s.author || '—'} · ${s.item_count ?? 0} items · ${(s.chunk_count ?? 0).toLocaleString('en-US')} chunks`,
        status: s.status || 'pending', avatar: s.avatar, avatarBg: s.avatarBg,
        kind: s.kind || (s.type === 'yt' ? 'channel' : null),
        expanded: false, justDone: false,
        syncError: !!(s.sync_error || s.syncError), syncing: false, checked: false,
        autoSync: s.sync_freq || s.auto_sync || s.autoSync || 'off',
        lastSynced: fmtSynced(s.last_sync_at || s.last_synced || s.lastSynced || ''),
      };
    }

    // Render an ISO last-sync stamp as a short local date; blank stays blank so
    // syncStatusText falls back to its "just now" copy.
    function fmtSynced(v) {
      if (!v) return '';
      const d = new Date(v);
      return isNaN(d.getTime()) ? v : d.toLocaleDateString();
    }

    const slug = computed(() => (bot.value?.name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''));
    // Fold a _share_info response into the modal's reactive state (link + real
    // daily cap + today's guest count). Inactive share → empty link.
    const applyShare = (info) => {
      shareLink.value = info && info.active ? (info.url || '') : '';
      guestUsed.value = (info && info.today_count) || 0;
      guestCap.value = (info && info.daily_cap) || guestCap.value;
    };
    const createLink = async () => {
      try { applyShare(await api.createShare(botId)); toast('Private link created'); }
      catch (e) { toast(e.message, false); }
    };
    const rotateLink = async () => {
      try { applyShare(await api.rotateShare(botId)); toast('Link rotated — the old link is disabled'); }
      catch (e) { toast(e.message, false); }
    };
    const revokeLink = () => {
      confirm.value = {
        title: 'Turn off sharing?', msg: 'The current link will stop working immediately. You can create a new one later.', action: 'Revoke',
        onOk: async () => {
          try { applyShare(await api.revokeShare(botId)); toast('Sharing turned off', false); }
          catch (e) { toast(e.message, false); }
          confirm.value = null;
        },
      };
    };

    const personaSource = computed(() => persona.savedCustom ? 'custom' : persona.sysPrompt ? 'builder' : 'none');
    const personaSourceLabel = computed(() =>
      personaSource.value === 'custom' ? 'Active: custom prompt' :
      personaSource.value === 'builder' ? 'Active: generated persona' :
      'No persona yet — bot uses a neutral default');

    const rename = async () => {
      try {
        await api.updateBot(botId, { name: bot.value.name });
        renamed.value = true; setTimeout(() => { renamed.value = false; }, 2200);
      } catch (e) { toast(e.message, false); }
    };

    // BUG-023: persist the bot's reply language. The field lives OUTSIDE the
    // persona mode templates, so it's editable in both Builder and Custom. Fires
    // on blur/Enter (@change). Trim; empty clears to "" (English default). Skips
    // the PATCH when unchanged so a plain blur doesn't re-save. The backend then
    // threads `bot.language` into the system prompt ("Always write your reply in
    // {lang}"), so the bot actually answers in this language.
    // L1: a persona save must only touch `language` when the owner actually chose
    // one. Empty field → omit the key entirely, so a never-set language is never
    // silently overwritten with a default on the backend.
    const langForPatch = () => { const l = (persona.lang || '').trim(); return l ? { language: l } : {}; };

    // Per-bot follow-up suggestions on/off (item 5). Optimistic with rollback.
    const setSuggestFollowups = async (v) => {
      if (!bot.value) return;
      const prev = bot.value.suggest_followups;
      bot.value.suggest_followups = v;
      try { await api.updateBot(botId, { suggest_followups: v }); toast(v ? 'Follow-up suggestions on' : 'Follow-up suggestions off'); }
      catch (e) { bot.value.suggest_followups = prev; toast(e.message, false); }
    };

    const saveLanguage = async () => {
      const lang = (persona.lang || '').trim();
      persona.lang = lang;
      if (!bot.value || lang === (bot.value.language || '')) return;
      try {
        await api.updateBot(botId, { language: lang });
        bot.value.language = lang;
        clearTimeout(langSavedTimer);
        langSaved.value = true;
        langSavedTimer = setTimeout(() => { langSaved.value = false; }, 2200);
      } catch (e) { toast(e.message, false); }
    };

    // Conversational persona builder: each call to the iterative endpoint either
    // returns the next tailored question ({done:false, question}) or finishes with
    // the generated system prompt ({done:true, persona}), which is persisted as
    // the bot's builder persona (NOT the Phase-J custom_prompt) and shown editable.
    const askNext = async () => {
      interview.thinking = true;
      try {
        const result = await api.interviewPersona(botId, { description: (persona.desc || '').trim(), answers: interview.qa.slice() });
        if (result && !result.done && result.question && interview.qNum < 3) {
          interview.qNum++;
          interview.thread.push({ role: 'ai', text: result.question });
        } else {
          const sysPrompt = (result && result.persona) || '';
          if (!sysPrompt) throw new Error('Could not generate a persona — please try again.');
          persona.sysPrompt = sysPrompt;
          await api.updateBot(botId, { persona: sysPrompt, ...langForPatch() });
          const lang = (persona.lang || '').trim();
          if (bot.value && lang) bot.value.language = lang;
          interview.done = true;
          interview.saved = true;
          interview.thread.push({ role: 'ai', text: 'That’s everything I needed — your persona is ready below. Edit it freely; it’s already active.' });
        }
      } catch (e) {
        interview.thread.push({ role: 'ai', text: e.message || 'Something went wrong — try again.', error: true });
      } finally { interview.thinking = false; }
    };
    // Retry after a failed interview call: drop the error row and re-ask with
    // the same answers (no state lost).
    const retryInterview = () => {
      if (interview.thinking) return;
      interview.thread = interview.thread.filter(m => !m.error);
      askNext();
    };
    const generatePersona = () => {
      if (interview.thinking) return;
      const desc = (persona.desc || '').trim();
      if (!desc) { persona.status = 'Add a short description first.'; persona.statusColor = 'var(--red)'; return; }
      persona.status = '';
      Object.assign(interview, { started: true, done: false, thinking: false, answer: '', thread: [], qa: [], qNum: 0, saved: false });
      askNext();
    };
    // Persist manual edits to the generated prompt (same PATCH {persona} field).
    const savePersonaEdit = async () => {
      const text = (persona.sysPrompt || '').trim();
      if (!text || interview.saving) return;
      interview.saving = true;
      const lang = (persona.lang || '').trim();
      try {
        await api.updateBot(botId, { persona: text, ...langForPatch() });
        if (bot.value && lang) bot.value.language = lang;
        interview.saved = true; toast('Persona saved — the bot uses it now');
      }
      catch (e) { toast(e.message, false); }
      finally { interview.saving = false; }
    };

    const timers = [];
    // ---- real ingest/rebuild progress (H3) --------------------------------
    // The queue card is driven entirely by the backend job endpoints, so it
    // survives a reload (state is rebuilt from the jobs list, not from a local
    // animation). Two job families feed it:
    //   * YouTube imports  -> GET /api/ingest/jobs  (IngestJob: videos_*)
    //   * rebuilds (doc add / delete / rebuild btn) -> GET /api/jobs/{id}
    // Tracked rebuild jobs: id -> label, so we can show + finalize them.
    const rebuildJobs = new Map();
    let pollTimer = null;

    const finishedMsg = { done: null, error: 'error', cancelled: 'cancelled' };

    // Turn a full YouTube/handle URL into its @handle for the queue label (BUG-003).
    const handleOf = (v) => {
      if (!v) return v;
      const s = String(v);
      if (s.startsWith('@')) return s;
      const m = s.match(/\/(@[\w.\-]+)/);
      if (m) return m[1];
      if (/youtube\.com\/playlist|[?&]list=/i.test(s)) return 'Playlist';
      if (/youtube\.com\/watch|youtu\.be\//i.test(s)) return 'Video';
      return s;
    };
    const ingestToEntry = (j) => {
      // BUG-001: the denominator is window_total (videos THIS run indexes), never a
      // fake total; channel_total is the channel's approx size for the context line.
      const windowTotal = j.window_total || j.videos_total || 0;
      const channelTotal = j.channel_total || windowTotal;
      const doneN = j.videos_done || 0;
      const pct = windowTotal > 0 ? Math.round((doneN / windowTotal) * 100) : 0;
      const src = sources.value.find(s => s.id === j.source_id);
      return {
        id: 'q' + j.job_id, jid: j.job_id, kind: 'ingest', sid: j.source_id,
        url: handleOf((src && src.label) || j.channel_url || j.source_id),
        pct, doneN, window: windowTotal, channelTotal,
        skipped: j.videos_skipped || 0, failed: j.videos_failed || 0,
        done: `${doneN} / ${windowTotal}`,
        total: windowTotal, status: j.status, origin: j.origin || 'manual', error: j.error || '',
        indeterminate: windowTotal === 0,
      };
    };

    const rebuildPct = (snap) => (snap.stage === 'done' ? 100 : snap.stage === 'building' ? 65 : 15);

    const reloadBot = async () => {
      try {
        const b = await api.getBot(botId);
        bot.value = b;
        sources.value = (b.sources || []).map(normalizeSource);
      } catch (e) { /* transient — keep the current view */ }
    };

    async function refreshJobs() {
      // --- ingest jobs (imports) ---
      let jobs = [];
      try { jobs = await api.ingestJobs(); } catch (e) { jobs = queue.value.length ? null : []; }
      // GET /api/ingest/jobs returns ingest jobs across ALL of the user's bots, so
      // scope to THIS bot at the source before any active/finalize/cancel logic runs.
      // Otherwise another bot's import bleeds into this queue — its completion fires
      // this bot's "ready" toast + reloadBot, and Cancel could cancel its job.
      if (jobs) jobs = jobs.filter(j => j.bot_id === botId);
      if (jobs) {
        const isActive = (j) => j.status === 'queued' || j.status === 'running';
        const activeIds = new Set(jobs.filter(isActive).map(j => j.job_id));
        // FIFO positions among the waiting jobs (the worker is strictly sequential).
        const queuedIds = jobs.filter(j => j.status === 'queued').map(j => j.job_id);
        for (const j of jobs.filter(isActive)) {
          const src = sources.value.find(s => s.id === j.source_id);
          if (src) {
            if (j.status === 'running') {
              // A previously-synced source is *syncing*, not doing a first build —
              // keeps its sync-row visible with the inline "Syncing…" indicator.
              src.syncing = !!src.lastSynced || src.status === 'ready';
              src.status = 'building';
            } else if (j.status === 'queued' && src.status === 'pending') {
              src.status = 'queued'; // waiting ≠ building: distinct chip, no ring %
            }
          }
          const entry = ingestToEntry(j);
          entry.pos = queuedIds.indexOf(j.job_id) + 1; // 0 for the running job
          const existing = queue.value.find(q => q.jid === j.job_id);
          if (existing) { Object.assign(existing, entry); existing.final = false; }
          else queue.value.push(entry);
        }
        // Finalize entries whose job left the active set; keep error/cancelled
        // rows visible (Retry / Dismiss), let done rows linger briefly.
        for (const j of jobs.filter(j => !isActive(j))) {
          const q = queue.value.find(x => x.jid === j.job_id);
          const src = sources.value.find(s => s.id === j.source_id);
          if (q && !q.final) {
            Object.assign(q, ingestToEntry(j), { final: true });
            if (src) {
              src.syncing = false;
              if (j.status === 'done') { src.status = 'ready'; src.syncError = false; src.lastSynced = 'just now'; }
              else if (j.status === 'error') { src.status = 'error'; src.syncError = !!src.lastSynced; }
              else src.status = src.lastSynced ? 'ready' : 'pending';
            }
            if (j.status === 'done') {
              const label = src ? src.label : 'source';
              setTimeout(() => { queue.value = queue.value.filter(x => x.id !== q.id); }, 4000);
              await reloadBot();
              // BUG-010: flash ONLY the source that just finished, not the whole list.
              const ns = sources.value.find(s => s.id === j.source_id);
              if (ns) { ns.justDone = true; setTimeout(() => { const cur = sources.value.find(s => s.id === j.source_id); if (cur) cur.justDone = false; }, 1500); }
              toast(`Import complete — ${label} is ready`);
            } else if (j.status === 'error') toast(`Import failed — ${j.error || 'see logs'}`, false);
            else if (j.status === 'cancelled') toast('Import cancelled', false);
          } else if (!q && (j.status === 'error' || j.status === 'cancelled')) {
            // Jobs that finished before this page opened: surface error/cancelled
            // rows silently (no toast) so the state is never invisible.
            const entry = ingestToEntry(j); entry.final = true;
            queue.value.push(entry);
            if (src && j.status === 'error') { src.status = 'error'; src.syncError = !!src.lastSynced; }
          } else if (!q && j.status === 'done') {
            // A DONE ingest we never observed active — it finished before this page
            // opened, or faster than our first ~1.5s poll (small/offline imports).
            // The finalize block above never ran for it, so its source would stay
            // stuck at "building" until a manual reload. Reconcile it from the
            // AUTHORITATIVE backend state (reloadBot → getBot/normalizeSource) rather
            // than optimistically forcing 'ready', so a job that indexed nothing or
            // errored per-source is reflected correctly. Guard on a non-terminal
            // current status so repeat polls don't reload/toast in a loop.
            if (src && src.status !== 'ready' && src.status !== 'error') {
              await reloadBot();
              const done = sources.value.find(s => s.id === j.source_id);
              if (done && done.status === 'ready') toast(`Import complete — ${done.label} is ready`);
            }
          }
        }
        // Drop entries whose job disappeared server-side (e.g. dismissed elsewhere).
        const known = new Set(jobs.map(j => j.job_id));
        queue.value = queue.value.filter(q => q.kind !== 'ingest' || known.has(q.jid) || q.status === 'done');
      }
      // --- rebuild jobs (doc add / delete / rebuild button) ---
      for (const [jid, label] of [...rebuildJobs]) {
        let snap = null;
        try { snap = await api.jobStatus(jid); } catch (e) { snap = null; }
        if (!snap) { rebuildJobs.delete(jid); queue.value = queue.value.filter(q => q.jid !== jid); continue; }
        const entry = queue.value.find(q => q.jid === jid);
        if (entry) { entry.pct = rebuildPct(snap); entry.done = snap.stage === 'done' ? 'indexed' : 'indexing…'; }
        if (snap.status === 'done' || snap.status === 'error') {
          rebuildJobs.delete(jid);
          queue.value = queue.value.filter(q => q.jid !== jid);
          if (snap.status === 'done') { await reloadBot(); toast(`${label} — index ready`); }
          else toast(`${label} — rebuild failed`, false);
        }
      }
    }

    const anythingActive = () => queue.value.some(q => q.status === 'running' || q.status === 'queued') || rebuildJobs.size > 0;
    function ensurePolling() {
      if (pollTimer) return;
      pollTimer = setInterval(async () => {
        await refreshJobs();
        if (!anythingActive()) { clearInterval(pollTimer); pollTimer = null; }
      }, 1500);
      timers.push(pollTimer);
    }
    // Track a rebuild-family job and show it in the queue card.
    function trackRebuild(jobId, label, sid) {
      if (!jobId) return;
      rebuildJobs.set(jobId, label);
      if (!queue.value.some(q => q.jid === jobId)) {
        queue.value.push({ id: 'q' + jobId, jid: jobId, kind: 'rebuild', sid: sid || '', url: label, pct: 15, done: 'indexing…', failed: 0 });
      }
      ensurePolling();
    }
    onBeforeUnmount(() => { timers.forEach(t => { clearInterval(t); clearTimeout(t); }); clearTimeout(langSavedTimer); });

    // Phase J1/J2: persist the free-form custom prompt via PATCH /api/bots/{id}
    // {custom_prompt}. A non-empty save OVERRIDES the builder persona at chat time
    // (precedence enforced server-side in bot_service.effective_persona); grounding
    // stays layered on top regardless. The builder persona is untouched — both
    // persist independently, so clearing reverts to the builder.
    const saveCustom = async () => {
      const text = persona.custom.trim();
      if (!text) return;
      try {
        const b = await api.saveCustomPrompt(botId, text);
        persona.savedCustom = (b && b.custom_prompt) || text;
        persona.custom = persona.savedCustom;
        toast('Custom prompt saved — it now overrides the builder');
      } catch (e) { toast(e.message, false); }
    };
    // Clear the saved custom prompt (send ''): reverts chat to the builder persona.
    // The textarea draft is kept so switching back to Builder loses nothing.
    const clearCustom = async () => {
      if (!persona.savedCustom) return;
      try {
        await api.saveCustomPrompt(botId, '');
        persona.savedCustom = '';
        toast('Custom prompt cleared — the builder persona is active again', false);
      } catch (e) { toast(e.message, false); }
    };

    const submitAnswer = () => {
      const a = interview.answer.trim();
      if (!a || interview.thinking || interview.done) return;
      const lastQ = [...interview.thread].reverse().find(m => m.role === 'ai');
      interview.qa.push({ question: (lastQ && lastQ.text) || '', answer: a });
      interview.thread.push({ role: 'user', text: a });
      interview.answer = '';
      askNext();
    };
    // Switching to Custom prefills the generated persona as a starting point
    // (never overwrites an existing draft; custom_prompt precedence unchanged).
    const switchPersonaMode = (m) => {
      persona.mode = m;
      if (m === 'custom' && !persona.custom.trim() && persona.sysPrompt) {
        persona.custom = persona.sysPrompt;
        toast('Copied your generated persona as a starting point');
      }
    };

    // Phase K: persist the per-source auto-sync frequency. Optimistic update with
    // rollback so a rejected save doesn't leave the selector out of sync.
    const setAutoSync = async (s, v) => {
      const prev = s.autoSync;
      s.autoSync = v;
      try {
        const res = await api.setSyncFreq(botId, s.id, v);
        s.autoSync = (res && res.sync_freq) || v || 'off';
        toast(v === 'off' ? 'Auto-sync turned off' : 'Auto-sync set to ' + v);
      } catch (e) { s.autoSync = prev; toast(e.message, false); }
    };
    const NEXT_SYNC = { daily: 'tomorrow', weekly: 'next week', monthly: 'next month' };
    const syncedText = (s) => `Last synced ${s.lastSynced || 'just now'}`;
    const nextText = (s) => s.autoSync !== 'off' ? `Next sync ${NEXT_SYNC[s.autoSync] || s.autoSync}` : 'Auto-sync off';

    const startImport = async (s) => {
      try {
        await api.startImport(botId, s.id);
        s.status = 'building';
        await refreshJobs();  // pick up the freshly-queued job into the queue card
        ensurePolling();
        toast('Import started');
      } catch (e) { toast(e.message, false); }
    };
    // Phase K: incremental "Sync now". The backend lists the channel, diffs
    // against the index, and enqueues ONLY new videos on the FIFO ingest worker;
    // the queue card then tracks that job like any other import. When nothing is
    // new, no job is queued and we just refresh the last-synced stamp.
    const syncNow = async (s) => {
      if (s._syncing) return;
      s._syncing = true;
      try {
        const res = await api.syncSource(botId, s.id);
        const n = (res && res.new) || 0;
        if (n > 0) {
          s.status = 'building';
          s.syncing = true;
          await refreshJobs();   // pick up the freshly-queued incremental job
          ensurePolling();
          toast(`Syncing ${n} new video${n === 1 ? '' : 's'} — only newer videos are fetched`);
        } else {
          // Auto-sync/manual check found nothing new: whisper it inline (§5 of
          // the brief) instead of only toasting.
          s.lastSynced = 'just now';
          s.checked = true;
          setTimeout(() => { s.checked = false; }, 6000);
          toast('Already up to date — no new videos');
        }
      } catch (e) { toast(e.message, false); }
      finally { s._syncing = false; }
    };
    const retrySync = (s) => { s.syncError = false; syncNow(s); };
    // Queue-row actions for finished jobs (error → Retry requeues on the FIFO
    // worker; error/cancelled → Dismiss removes the row).
    const retryJob = async (q) => {
      try {
        if (api.retryJob) await api.retryJob(q.jid);
        q.final = false; q.status = 'queued'; q.error = '';
        await refreshJobs(); ensurePolling();
        toast('Import requeued');
      } catch (e) { toast(e.message, false); }
    };
    const dismissJob = async (q) => {
      queue.value = queue.value.filter(x => x.id !== q.id);
      try { if (api.dismissJob) await api.dismissJob(q.jid); } catch (e) { /* non-fatal */ }
    };
    const posText = (p) => p <= 1 ? 'next in line' : (p === 2 ? '2nd' : p === 3 ? '3rd' : p + 'th') + ' in line';
    const qSub = (q) => {
      if (q.status === 'queued') return `${(q.total || 0).toLocaleString('en-US')} videos found · ${posText(q.pos)}`;
      if (q.status === 'error') return q.error || 'Import failed — see logs';
      if (q.status === 'cancelled') return `Stopped at ${q.doneN} of ${q.window} — indexed videos are kept`;
      if (q.status === 'done') return `${q.done} · indexed`;
      return q.done;
    };
    const qRowStyle = (q) =>
      q.status === 'error' ? { border: '1px solid rgba(248,113,113,.25)', background: 'rgba(248,113,113,.04)' } :
      q.status === 'done' ? { border: '1px solid rgba(52,211,153,.25)', background: 'rgba(52,211,153,.05)' } :
      q.status === 'cancelled' ? { border: '1px solid var(--border-soft)', background: 'var(--panel-2)', opacity: .7 } :
      { border: '1px solid var(--border-soft)', background: 'var(--panel-2)' };
    const hasRunning = computed(() => queue.value.some(q => q.status === 'running'));
    // Live entry for the transcripts panel — same data as the ring, no fake numbers.
    const runningEntry = computed(() => queue.value.find(q => q.status === 'running') || null);

    const removeSource = (s) => {
      confirm.value = {
        title: 'Delete source?', msg: `“${s.label}” and its indexed chunks will be removed from this bot.`, action: 'Delete',
        onOk: async () => {
          try {
            const res = await api.deleteSource(botId, s.id);
            sources.value = sources.value.filter(x => x.id !== s.id);
            toast('Source deleted');
            // Deletion triggers a rebuild so chunk counts stay accurate.
            if (res && res.job_id) trackRebuild(res.job_id, 'Rebuild after delete');
          } catch (e) { toast(e.message, false); }
          confirm.value = null;
        },
      };
    };

    const addChannel = async () => {
      if (!addYt.url.trim() || !addYt.author.trim() || !addYt.consent) return;
      const kind = addYt.kind;
      try {
        // A single video never gains new items, so it defaults to no auto-sync.
        const sync = kind === 'video' ? 'off' : addYt.sync;
        const res = await api.addYouTubeSource(botId, { channel: addYt.url.trim(), author: addYt.author.trim(), consent: true, sync_freq: sync, kind, include_shorts: addYt.shorts });
        if (bot.value) bot.value.include_shorts = addYt.shorts;
        sources.value.push({ id: res.source_id, type: 'yt', kind, label: addYt.url.trim(), meta: `author: ${addYt.author.trim()} · 0 items · 0 chunks`, status: 'pending', autoSync: (res && res.sync_freq) || sync });
        Object.assign(addYt, { url: '', author: '', consent: false, sync: 'weekly', kind: 'channel', open: false });
        toast(kind === 'playlist' ? 'Playlist added — start import when ready' : (kind === 'video' ? 'Video added — start import when ready' : 'Channel added — start import when ready'));
      } catch (e) { toast(e.message, false); }
    };

    const onDocFile = (e) => {
      const f = e.target.files[0];
      addDoc.fileObj = f || null;
      addDoc.file = f ? f.name : '';  // display name only
    };
    const uploadDoc = async () => {
      if (!addDoc.fileObj || !addDoc.author.trim()) return;
      const label = addDoc.file;
      try {
        // The document form has no consent checkbox but the backend enforces a
        // server-side rights attestation on every content-adding endpoint, so we
        // send consent=true (uploading one's own file IS the attestation). If the
        // design later adds a doc-consent control, thread it through here.
        const res = await api.uploadDocument(botId, { file: addDoc.fileObj, author: addDoc.author.trim(), consent: true });
        if (res && res.source_id) {
          sources.value.push({ id: res.source_id, type: 'doc', label, meta: `author: ${addDoc.author.trim()} · 1 item · 0 chunks`, status: 'building' });
        }
        // A document add rebuilds the index — track that rebuild job for progress.
        if (res && res.job_id) trackRebuild(res.job_id, label, res.source_id);
        Object.assign(addDoc, { file: '', fileObj: null, author: '', open: false });
        toast('Import started');
      } catch (e) { toast(e.message, false); }
    };

    const cancelJob = (q) => {
      confirm.value = {
        title: 'Cancel import?', msg: "Progress on this channel will be kept; unfetched videos won't be indexed.", action: 'Cancel import',
        onOk: async () => {
          try {
            // Only ingest jobs are cancellable server-side; rebuild jobs are short
            // and have no cancel endpoint, so we just drop the local row.
            if (q.kind === 'ingest' && q.jid) await api.cancelJob(q.jid);
          } catch (e) { toast(e.message, false); }
          const src = sources.value.find(x => x.id === q.sid);
          if (src) { src.syncing = false; src.status = src.lastSynced ? 'ready' : 'pending'; }
          queue.value = queue.value.filter(x => x.id !== q.id);
          rebuildJobs.delete(q.jid);
          confirm.value = null; toast('Import cancelled', false);
        },
      };
    };

    // Owner chat consumes the SSE stream (item 8b): a placeholder assistant bubble
    // fills token-by-token as deltas arrive. Citations are part of the answer itself
    // as deliberately formatted one-source Markdown quote blocks.
    // Send stays disabled for the whole in-flight generation (chat.streaming); a
    // 409 (backend rejects a 2nd concurrent send) shows a friendly note. The
    // non-stream /chat path stays as a fallback.
    const send = async () => {
      const q = chat.input.trim();
      if (!q || chat.thinking || chat.streaming) return;
      const history = chat.messages.map(m => ({ role: m.who === 'You' ? 'user' : 'assistant', content: m.text }));
      chat.messages.push({ who: 'You', text: q });
      chat.input = '';
      chat.thinking = true; chat.streaming = true;
      const msg = reactive({ who: 'Assistant', text: '', streaming: true });
      let started = false;
      const begin = () => { if (!started) { started = true; chat.thinking = false; chat.messages.push(msg); } };
      const nonStream = async () => {
        const res = await api.chat(botId, { message: q, history, mode: chatMode.value });
        begin(); msg.text = res.answer || '';
      };
      try {
        if (api.chatStream) {
          const done = await api.chatStream(botId, { message: q, history, mode: chatMode.value }, {
            onDelta: (t) => { begin(); msg.text += t; },
            onDone: () => {},
          });
          if (!started) { // stream carried no deltas — surface whatever `done` had
            begin();
            if (!msg.text && done && done.answer) msg.text = done.answer;
          }
        } else {
          await nonStream();
        }
      } catch (e) {
        chat.thinking = false;
        if (e.status === 409) {
          if (started) chat.messages = chat.messages.filter(m => m !== msg);
          chat.messages.push({ who: 'Assistant', text: 'This bot is still answering your last message — wait for it to finish before sending another.', error: true });
        } else if (started && msg.text) {
          msg.error = true; // partial answer then the stream dropped — keep what arrived
        } else {
          try { await nonStream(); }               // one fallback attempt on a stream failure
          catch (e2) { begin(); msg.text = e2.message || 'Something went wrong.'; msg.error = true; }
        }
      } finally {
        chat.thinking = false; chat.streaming = false; msg.streaming = false;
      }
    };

    const connectTg = async () => {
      if (!tg.token.trim()) return;
      tg.verifying = true;
      try {
        const res = await api.connectTelegram(botId, tg.token.trim());
        tg.connected = true;
        tg.username = res.username || '';
        tg.token = '';
        toast('Telegram connected');
      } catch (e) { toast(e.message, false); }
      finally { tg.verifying = false; }
    };
    const askDisconnect = () => {
      confirm.value = {
        title: 'Disconnect Telegram?', msg: `${tg.username} will stop answering until you reconnect a token.`, action: 'Disconnect',
        onOk: async () => {
          try { await api.disconnectTelegram(botId); tg.connected = false; toast('Telegram disconnected', false); }
          catch (e) { toast(e.message, false); }
          confirm.value = null;
        },
      };
    };

    const copy = async (text, flag) => {
      try { await navigator.clipboard.writeText(text); } catch (e) {}
      flag.value = true; setTimeout(() => { flag.value = false; }, 1800);
    };

    const dash = (pct) => { const C = 2 * Math.PI * 15; return (C * pct / 100).toFixed(1) + ' ' + C.toFixed(1); };

    const logLines = [
      '[14:02:11] fetch transcript 9Kx2… ok (2,310 words)',
      '[14:02:13] fetch transcript mQ7p… ok (1,842 words)',
      '[14:02:14] fetch transcript zR4t… FAILED (no captions)',
      '[14:02:16] chunk + embed batch 12 → 96 chunks',
      '[14:02:18] fetch transcript aL9s… ok (3,077 words)',
      '[14:02:19] saved 139/142 · queue depth 173',
    ];

    // Manual "Rebuild index" — keeps the button-spinner design (no queue row);
    // polls the rebuild job to completion, then refreshes the chunk counts.
    const rebuildIndex = async () => {
      if (rebuilding.value) return;
      rebuilding.value = true;
      toast('Index rebuild started');
      try {
        const res = await api.rebuild(botId);
        const jid = res && res.job_id;
        if (jid) {
          await new Promise((resolve) => {
            const iv = setInterval(async () => {
              let snap = null;
              try { snap = await api.jobStatus(jid); } catch (e) { snap = null; }
              if (!snap || snap.status === 'done' || snap.status === 'error') {
                clearInterval(iv);
                resolve(snap);
              }
            }, 1500);
            timers.push(iv);
          });
        }
        await reloadBot();
        toast('Index rebuilt — all chunks refreshed');
      } catch (e) { toast(e.message, false); }
      finally { rebuilding.value = false; }
    };

    return {
      bot, loadError, renamed, langSaved, persona, personaSource, personaSourceLabel, interview, sources, queue, addYt, addDoc, chat, chatMode, setChatMode, tg,
      shareOpen, shareLink, guestUsed, guestCap, copied, confirm, toasts, logLines,
      STATUS,
      rename, saveLanguage, generatePersona, savePersonaEdit, switchPersonaMode, retryInterview, saveCustom, clearCustom, submitAnswer, setAutoSync, syncedText, nextText, startImport, syncNow,
      retrySync, retryJob, dismissJob, qSub, qRowStyle, hasRunning, runningEntry,
      removeSource, addChannel, onDocFile, uploadDoc, cancelJob, send, connectTg, askDisconnect, dash,
      createLink, rotateLink, revokeLink, rebuilding, rebuildIndex,
      setSuggestFollowups, renderMd: renderMarkdown,
      metaParts: (s) => (s.meta || '').split(' · ').map(x => x.trim()).filter(Boolean),
      copyLink: () => copy(shareLink.value, copied),
      fmt: (n) => (n || 0).toLocaleString('en-US'),
    };
  },
  template: `
  <main class="page" style="max-width: 784px; padding-top: 28px;">
    <router-link to="/" style="font-size: 13px; color: var(--text-dim); display: inline-flex; align-items: center; gap: 6px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M12 19l-7-7 7-7"></path></svg>
      My Bots
    </router-link>

    <div v-if="loadError" style="margin-top: 32px; display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 56px 32px; border-radius: 12px; border: 1px solid rgba(248,113,113,.25); background: rgba(248,113,113,.04); text-align: center;">
      <span style="font-size: 15px; font-weight: 600; color: var(--red);">Couldn't load this bot</span>
      <router-link to="/">← Back to My Bots</router-link>
    </div>

    <template v-else-if="!bot">
      <div style="margin-top: 24px; display: flex; flex-direction: column; gap: 20px;">
        <div class="skeleton" style="height: 40px; width: 260px; border-radius: 8px;"></div>
        <div class="bd-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
          <div v-for="i in 2" :key="i" class="card skeleton" style="height: 320px; padding: 22px; display: flex; flex-direction: column; gap: 14px;">
            <div class="skel-line" style="width: 40%; height: 15px;"></div>
            <div class="skel-line" style="width: 92%; height: 11px; background: var(--border-soft);"></div>
            <div class="skel-line" style="width: 84%; height: 11px; background: var(--border-soft);"></div>
          </div>
        </div>
      </div>
    </template>

    <template v-else>
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 14px 0 24px; flex-wrap: wrap;">
        <div class="title-row" style="display: flex; align-items: center; gap: 10px;">
          <input v-model="bot.name" data-testid="bot-title-input" class="title-input">
          <button class="btn btn-ghost" data-testid="rename-btn" style="height: 32px; padding: 0 12px; font-size: 12.5px;" @click="rename">Rename</button>
          <span v-if="renamed" style="font-size: 12px; color: var(--green);">Saved ✓</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
          <span class="badge badge-ready"><span class="badge-dot"></span>Ready</span>
          <span class="badge mono" style="background: rgba(138,138,153,.08); color: var(--text-dim); border-color: rgba(138,138,153,.2); white-space: nowrap;">{{ fmt(bot.chunk_count) }} chunks</span>
          <button class="btn btn-secondary btn-sm" data-testid="share-btn" @click="shareOpen = true">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7M16 6l-4-4-4 4M12 2v13"></path></svg>
            Share
          </button>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 16px;">

        <div class="card" style="padding: 22px; display: flex; flex-direction: column; gap: 16px;">
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;">
            <div class="card-head-title" style="display: flex; flex-direction: column; gap: 3px; min-width: 0;">
              <div style="display: flex; align-items: baseline; gap: 8px;"><span style="font-size: 15px; font-weight: 700;">Persona</span><span style="font-size: 12px; color: var(--text-faint);">Identity</span></div>
              <span style="font-size: 11.5px; line-height: 1.4;" :style="{ color: personaSource === 'none' ? 'var(--text-faint)' : 'var(--green)' }">{{ personaSourceLabel }}</span>
            </div>
            <div class="seg" data-testid="persona-mode-toggle">
              <button data-testid="persona-mode-builder" :class="{ on: persona.mode === 'builder' }" @click="switchPersonaMode('builder')" style="display: inline-flex; align-items: center; gap: 6px;">Builder<span v-if="personaSource === 'builder'" style="width: 6px; height: 6px; border-radius: 9999px; background: var(--green);"></span></button>
              <button data-testid="persona-mode-custom" :class="{ on: persona.mode === 'custom' }" @click="switchPersonaMode('custom')" style="display: inline-flex; align-items: center; gap: 6px;">Custom prompt<span v-if="persona.savedCustom" style="width: 6px; height: 6px; border-radius: 9999px; background: var(--green);"></span></button>
            </div>
          </div>
          <label class="field">
            <span class="label" style="font-size: 12.5px;">Language</span>
            <div style="display: flex; align-items: center; gap: 10px;">
              <input class="input" data-testid="persona-lang" style="flex: 1;" v-model="persona.lang" @change="saveLanguage" @keydown.enter.prevent="saveLanguage" placeholder="e.g. English, Spanish, Ukrainian">
              <span v-if="langSaved" data-testid="lang-saved" style="font-size: 12px; color: var(--green); white-space: nowrap;">Saved ✓</span>
            </div>
          </label>
          <template v-if="persona.mode === 'builder'">
          <label class="field">
            <span class="label" style="font-size: 12.5px;">Tell me what you want the bot to look like…</span>
            <textarea class="textarea" data-testid="persona-desc" v-model="persona.desc" rows="3" :disabled="interview.started && !interview.done" placeholder="e.g. Warm and practical, explains the why behind techniques, always offers substitutions, never condescending."></textarea>
          </label>
          <div class="row-wrap" style="display: flex; align-items: center; gap: 12px;">
            <button class="btn btn-primary btn-block-sm" data-testid="persona-generate" style="padding: 0 16px; font-size: 13px;" :disabled="interview.thinking" @click="generatePersona">{{ interview.started ? (interview.done ? 'Regenerate' : 'Start over') : 'Generate persona' }}</button>
            <span style="font-size: 12px;" :style="{ color: persona.statusColor }">{{ persona.status }}</span>
          </div>
          <div v-if="interview.started" style="border-radius: 8px; border: 1px solid var(--border); background: var(--panel-2); padding: 14px; display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
              <span style="font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">Persona interview</span>
              <span v-if="!interview.done" class="mono" style="font-size: 11px; color: var(--text-faint);">Question {{ Math.min(interview.qNum, 3) }} of 3</span>
              <span v-else style="font-size: 11.5px; font-weight: 600; color: var(--green);">Complete ✓</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 10px;">
              <div v-for="(m, i) in interview.thread" :key="i" class="msg-in" data-testid="interview-msg" :data-role="m.role" :style="m.role === 'user' ? { alignSelf: 'flex-end', maxWidth: '85%' } : { alignSelf: 'stretch' }">
                <span v-if="m.role === 'ai'" style="font-size: 13px; line-height: 1.55; min-width: 0;" :style="{ color: m.error ? 'var(--red)' : 'var(--text)' }">{{ m.text }} <button v-if="m.error && i === interview.thread.length - 1" style="border: none; background: none; padding: 0; cursor: pointer; font-family: inherit; font-size: 13px; font-weight: 600; color: var(--red); text-decoration: underline;" @click="retryInterview">Try again</button></span>
                <span v-else style="display: inline-block; font-size: 12.5px; line-height: 1.5; color: var(--text-mid); background: var(--raised); border: 1px solid var(--border-2); border-radius: 10px 10px 3px 10px; padding: 7px 11px;">{{ m.text }}</span>
              </div>
              <div v-if="interview.thinking" class="msg-in" style="display: flex; align-items: center; gap: 9px;">
                <span style="font-size: 11.5px; color: var(--text-faint);">{{ interview.qNum === 0 ? 'Reading your description' : 'Composing the next question' }}<span class="think-ellip"><span>.</span><span>.</span><span>.</span></span></span>
              </div>
            </div>
            <div v-if="!interview.done" style="display: flex; gap: 8px;">
              <input class="input" data-testid="interview-answer" style="flex: 1; background: var(--bg);" v-model="interview.answer" :disabled="interview.thinking" placeholder="Your answer…" @keydown.enter="submitAnswer">
              <button class="btn btn-secondary" data-testid="interview-answer-btn" :disabled="interview.thinking || !interview.answer.trim()" @click="submitAnswer">Answer</button>
            </div>
          </div>
          <div v-if="persona.sysPrompt && interview.done" style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
              <span style="font-size: 12.5px; font-weight: 600; color: var(--text-mid);">Generated persona <span style="font-weight: 500; color: var(--text-faint);">— editable</span></span>
              <span v-if="interview.saved" data-testid="persona-saved" style="font-size: 11.5px; color: var(--green);">Saved ✓</span>
              <span v-else data-testid="persona-unsaved" style="font-size: 11.5px; color: var(--amber);">Unsaved edits</span>
            </div>
            <textarea class="textarea mono" data-testid="persona-generated" style="min-height: 180px; font-size: 12px; line-height: 1.6; background: var(--bg);" v-model="persona.sysPrompt" @input="interview.saved = false"></textarea>
            <div class="row-wrap" style="display: flex; align-items: center; gap: 12px;">
              <button class="btn btn-secondary btn-block-sm" data-testid="persona-save" style="padding: 0 14px; font-size: 12.5px;" :disabled="interview.saved || interview.saving || !persona.sysPrompt.trim()" @click="savePersonaEdit">{{ interview.saving ? 'Saving…' : interview.saved ? 'Saved ✓' : 'Save changes' }}</button>
              <span style="font-size: 12px; color: var(--text-faint);">Switch to Custom prompt to start from this text.</span>
            </div>
          </div>
          </template>
          <template v-else>
            <div v-if="persona.savedCustom" style="display: flex; align-items: center; gap: 9px; padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(52,211,153,.25); background: rgba(52,211,153,.06); font-size: 12.5px; color: var(--green);">
              <span style="width: 7px; height: 7px; border-radius: 9999px; background: var(--green); flex-shrink: 0;"></span>
              Custom prompt is active — the bot uses it instead of the builder.
            </div>
            <label class="field" style="flex: 1;">
              <span class="label" style="font-size: 12.5px;">System prompt <span style="font-weight: 500; color: var(--text-faint);">— used as-is, overrides the builder</span></span>
              <textarea class="textarea mono" data-testid="persona-custom" style="flex: 1; min-height: 220px; font-size: 12px; line-height: 1.6;" v-model="persona.custom" placeholder="You are a helpful assistant for this channel. Answer only from the indexed transcripts…"></textarea>
            </label>
            <div class="row-wrap" style="display: flex; align-items: center; gap: 12px;">
              <button class="btn btn-primary btn-block-sm" data-testid="persona-custom-save" style="padding: 0 16px; font-size: 13px;" :disabled="!persona.custom.trim() || persona.custom.trim() === persona.savedCustom" @click="saveCustom">{{ persona.savedCustom && persona.custom.trim() === persona.savedCustom ? 'Saved ✓' : persona.savedCustom ? 'Update prompt' : 'Save prompt' }}</button>
              <button v-if="persona.savedCustom" class="btn btn-ghost btn-block-sm" style="padding: 0 14px; font-size: 13px; color: var(--text-faint);" @click="clearCustom">Clear</button>
              <span style="font-size: 12px; color: var(--text-faint);">Switch back to Builder any time — your draft is kept.</span>
            </div>
          </template>
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 14px; padding-top: 14px; border-top: 1px solid var(--border-soft);">
            <div style="display: flex; flex-direction: column; gap: 3px; min-width: 0;">
              <span style="font-size: 12.5px; font-weight: 600;">Suggest follow-up questions</span>
              <span style="font-size: 11.5px; color: var(--text-faint); line-height: 1.45;">Adds a “💡 you could ask…” line after answers.</span>
            </div>
            <button type="button" role="switch" data-testid="suggest-followups-toggle" :aria-checked="bot.suggest_followups ? 'true' : 'false'" class="cm-switch" :class="{ on: bot.suggest_followups }" @click="setSuggestFollowups(!bot.suggest_followups)"></button>
          </div>
        </div>

        <div class="card" style="padding: 22px; display: flex; flex-direction: column; gap: 14px;">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: baseline; gap: 8px;"><span style="font-size: 15px; font-weight: 700;">Sources</span><span style="font-size: 12px; color: var(--text-faint);">Knowledge</span></div>
            <button class="btn btn-ghost" data-testid="rebuild-index-btn" style="height: 32px; padding: 0 12px; font-size: 12.5px;" :disabled="rebuilding" @click="rebuildIndex"><span v-if="rebuilding" style="animation: pulse 1.4s ease-in-out infinite;">Rebuilding…</span><span v-else>Rebuild index</span></button>
          </div>
          <div v-if="sources.length" class="src-list" style="display: flex; flex-direction: column; gap: 8px;">
            <div v-for="s in sources" :key="s.id" class="src-card" :class="{ 'src-done-flash': s.justDone }" data-testid="source-row" :data-source-kind="s.type === 'doc' ? 'document' : (s.kind || 'channel')" style="display: flex; flex-direction: column; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border-soft); background: var(--panel-2);">
              <div class="src-row" :class="{ 'src-doc': s.type === 'doc' }" style="display: flex; align-items: center; gap: 12px;">
                <div class="src-id" style="display: flex; align-items: center; gap: 12px; min-width: 0; flex: 1;">
                  <div v-if="s.avatar" style="width: 34px; height: 34px; border-radius: 9999px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #fff; flex-shrink: 0;" :style="{ background: s.avatarBg }">{{ s.avatar }}</div>
                  <div v-else style="width: 34px; height: 34px; border-radius: 8px; background: var(--raised); display: flex; align-items: center; justify-content: center; font-size: 14px; color: var(--text-dim); flex-shrink: 0;">{{ s.type === 'yt' ? '▶' : '▧' }}</div>
                  <div style="display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1;">
                    <span class="src-title" style="display: flex; align-items: center; gap: 6px; min-width: 0;">
                      <span class="src-label" data-testid="source-label" style="font-size: 13px; font-weight: 600;">{{ s.label }}</span>&#32;<span class="src-tag" data-testid="source-tag">{{ s.type === 'doc' ? 'Document' : s.kind === 'playlist' ? 'Playlist' : s.kind === 'video' ? 'Video' : 'Channel' }}</span>
                    </span>
                    <span class="src-meta-seg" data-testid="source-meta" style="font-size: 11.5px; color: var(--text-faint);"><template v-for="(p, pi) in metaParts(s)" :key="pi"><span v-if="pi" class="sep">·</span><span>{{ p }}</span></template></span>
                  </div>
                </div>
                <div class="src-actions" style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
                  <span class="badge" data-testid="source-status" :data-status="s.status" :class="'badge-' + s.status" style="font-size: 11px; padding: 2px 9px;"><span class="badge-dot" style="width: 5px; height: 5px;"></span>{{ STATUS[s.status] || s.status }}</span>
                  <button v-if="s.type === 'yt' && s.status === 'pending'" class="btn btn-secondary" data-testid="start-import-btn" style="height: 30px; padding: 0 12px; font-size: 12px; gap: 6px;" @click="startImport(s)"><svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"></path></svg>Start import</button>
                  <button v-if="s.type === 'yt' && (s.status === 'ready' || s.status === 'error')" class="btn btn-ghost" data-testid="sync-now-btn" style="height: 30px; padding: 0 11px; font-size: 12px;" :disabled="s.syncing" @click="syncNow(s)">Sync now</button>
                  <button class="btn btn-danger-ghost" style="height: 30px; padding: 0 10px; font-size: 12px; color: var(--text-faint);" @click="removeSource(s)">Delete</button>
                </div>
              </div>
              <div v-if="s.type === 'yt' && (s.status === 'ready' || s.syncing || s.syncError)" class="sync-row" style="display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-soft);">
                <label style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-dim); flex-shrink: 0;">
                  <span style="display: flex; align-items: center; gap: 8px; white-space: nowrap;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#5a5a6a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Auto-sync</span>
                  <select-field small data-testid="autosync-select" :disabled="s.syncing" :model-value="s.autoSync" @update:model-value="setAutoSync(s, $event)" :options="[{value:'off',label:'Off'},{value:'daily',label:'Daily'},{value:'weekly',label:'Weekly'},{value:'monthly',label:'Monthly'}]"></select-field>
                </label>
                <span v-if="s.syncing" class="sync-status" style="display: inline-flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--blue); white-space: nowrap;">
                  <svg class="spin" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M21 12a9 9 0 1 1-6.2-8.56"></path></svg>
                  Syncing — only new videos are fetched
                </span>
                <span v-else-if="s.syncError" class="sync-status" style="font-size: 11.5px; color: var(--red); white-space: nowrap;">
                  Last sync failed · <button style="border: none; background: none; padding: 0; cursor: pointer; font-family: inherit; font-size: 11.5px; font-weight: 600; color: var(--red); text-decoration: underline;" @click="retrySync(s)">Retry</button>
                </span>
                <span v-else-if="s.checked" class="sync-status" style="display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--green); white-space: nowrap;">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>
                  Checked · up to date
                </span>
                <span v-else class="sync-status" style="font-size: 11.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" :style="{ color: s.autoSync !== 'off' ? 'var(--text-dim)' : 'var(--text-faint)' }"><span>{{ syncedText(s) }}</span><span class="sync-next"><span class="sync-sep"> · </span>{{ nextText(s) }}</span></span>
              </div>
            </div>
          </div>
          <div v-else style="padding: 28px; border-radius: 8px; border: 1px dashed var(--border-2); background: var(--panel-2); text-align: center; font-size: 13px; color: var(--text-dim);">No sources yet — add a YouTube channel or upload a document below.</div>

          <div data-testid="add-source-panel" style="border-radius: 8px; border: 1px solid var(--border); background: var(--panel-2);">
            <button class="collapse-btn" data-testid="add-source-toggle" @click="addYt.open = !addYt.open">
              <span style="display: flex; align-items: center; gap: 9px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#f87171"><path d="M23 7.5s-.2-1.7-.9-2.4c-.9-1-1.9-1-2.4-1C16.4 3.8 12 3.8 12 3.8s-4.4 0-7.7.3c-.5.1-1.5.1-2.4 1-.7.7-.9 2.4-.9 2.4S.8 9.4.8 11.4v1.8c0 1.9.2 3.9.2 3.9s.2 1.7.9 2.4c.9 1 2 .9 2.5 1 1.8.2 7.6.3 7.6.3s4.4 0 7.7-.3c.5-.1 1.5-.1 2.4-1 .7-.7.9-2.4.9-2.4s.2-1.9.2-3.9v-1.8c0-1.9-.2-3.9-.2-3.9zM9.8 15.3V8.7l6.2 3.3-6.2 3.3z"></path></svg>
                Add YouTube {{ addYt.kind === 'playlist' ? 'playlist' : (addYt.kind === 'video' ? 'video' : 'channel') }}
              </span>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#8a8a99" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" :style="{ transform: addYt.open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform .2s' }"><path d="M6 9l6 6 6-6"></path></svg>
            </button>
            <div v-if="addYt.open" style="padding: 8px 14px 16px; display: flex; flex-direction: column; gap: 10px;">
              <div class="kind-toggle" role="radiogroup" aria-label="Source kind" data-testid="source-kind-toggle" style="display: flex; gap: 2px; padding: 2px; border-radius: 8px; border: 1px solid var(--border-soft); background: var(--bg); align-self: flex-start;">
                <button type="button" role="radio" data-testid="kind-channel" :aria-checked="addYt.kind === 'channel'" class="kind-btn" :class="{ on: addYt.kind === 'channel' }" @click="addYt.kind = 'channel'">Channel</button>
                <button type="button" role="radio" data-testid="kind-playlist" :aria-checked="addYt.kind === 'playlist'" class="kind-btn" :class="{ on: addYt.kind === 'playlist' }" @click="addYt.kind = 'playlist'">Playlist</button>
                <button type="button" role="radio" data-testid="kind-video" :aria-checked="addYt.kind === 'video'" class="kind-btn" :class="{ on: addYt.kind === 'video' }" @click="addYt.kind = 'video'; addYt.sync = 'off'">Video</button>
              </div>
              <input class="input" data-testid="source-url-input" style="background: var(--bg);" v-model="addYt.url" :placeholder="addYt.kind === 'playlist' ? 'Playlist URL (youtube.com/playlist?list=…) *' : (addYt.kind === 'video' ? 'Video URL (youtube.com/watch?v=…) *' : 'Channel URL or @handle *')">
              <div class="row-wrap" style="display: flex; gap: 8px;">
                <input class="input" data-testid="source-author-input" style="flex: 1 1 160px; background: var(--bg);" v-model="addYt.author" placeholder="Author *">
                <button class="btn btn-primary" data-testid="add-source-btn" style="box-shadow: none;" :disabled="!addYt.url.trim() || !addYt.author.trim() || !addYt.consent" @click="addChannel">{{ addYt.kind === 'playlist' ? 'Add playlist' : (addYt.kind === 'video' ? 'Add video' : 'Add channel') }}</button>
              </div>
              <label style="display: flex; align-items: center; justify-content: space-between; gap: 9px; font-size: 12.5px; color: var(--text-mid); padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border-soft); background: var(--bg);">
                <span style="display: flex; align-items: center; gap: 9px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#5a5a6a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Auto-sync new videos</span>
                <select-field small v-model="addYt.sync" :options="[{value:'off',label:'Off'},{value:'daily',label:'Daily'},{value:'weekly',label:'Weekly'},{value:'monthly',label:'Monthly'}]"></select-field>
              </label>
              <label v-if="addYt.kind === 'channel'" style="display: flex; align-items: center; justify-content: space-between; gap: 9px; font-size: 12.5px; color: var(--text-mid); padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border-soft); background: var(--bg);">
                <span style="display: flex; flex-direction: column; gap: 2px; min-width: 0;">
                  <span style="display: flex; align-items: center; gap: 9px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#5a5a6a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"></rect><path d="M10 8l6 4-6 4V8z"></path></svg>Include Shorts</span>
                  <span style="font-size: 11px; color: var(--text-faint); line-height: 1.4;">Off by default — most channels' Shorts add little to the index.</span>
                </span>
                <button type="button" role="switch" data-testid="include-shorts-toggle" :aria-checked="addYt.shorts ? 'true' : 'false'" class="cm-switch" :class="{ on: addYt.shorts }" @click="addYt.shorts = !addYt.shorts"></button>
              </label>
              <div data-testid="source-consent" :data-checked="addYt.consent ? 'true' : 'false'" @click="addYt.consent = !addYt.consent" style="display: flex; align-items: flex-start; gap: 9px; font-size: 12.5px; color: var(--text-mid); line-height: 1.45; cursor: pointer; user-select: none;">
                <span style="width: 16px; height: 16px; flex-shrink: 0; margin-top: 1px; border-radius: 5px; border: 1px solid; display: flex; align-items: center; justify-content: center; transition: background .15s, border-color .15s;" :style="{ background: addYt.consent ? 'var(--accent-grad)' : 'var(--bg)', borderColor: addYt.consent ? 'transparent' : 'var(--border-3)' }">
                  <svg v-if="addYt.consent" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"></path></svg>
                </span>
                <span>I'll use this bot privately and I take responsibility for how I use this {{ addYt.kind === 'playlist' ? "playlist's" : (addYt.kind === 'video' ? "video's" : "channel's") }} content <span style="color: var(--accent-from);">*</span></span>
              </div>
              <span style="font-size: 11px; color: var(--text-faint); line-height: 1.5;">Keep it to a few people you trust. We remove content on a valid rights-holder / DMCA request — see <router-link to="/terms" style="color: var(--text-dim); text-decoration: underline;">Terms</router-link>.</span>
            </div>
          </div>

          <div style="border-radius: 8px; border: 1px solid var(--border); background: var(--panel-2);">
            <button class="collapse-btn" data-testid="add-doc-toggle" @click="addDoc.open = !addDoc.open">
              <span style="display: flex; align-items: center; gap: 9px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path d="M14 2v6h6"></path></svg>
                Add document
              </span>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#8a8a99" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" :style="{ transform: addDoc.open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform .2s' }"><path d="M6 9l6 6 6-6"></path></svg>
            </button>
            <div v-if="addDoc.open" style="padding: 8px 14px 16px; display: flex; flex-direction: column; gap: 10px;">
              <label class="file-drop">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"></path></svg>
                {{ addDoc.file || 'Choose file (txt, md, pdf, docx)' }}
                <input type="file" data-testid="doc-file-input" accept=".txt,.md,.pdf,.docx" @change="onDocFile" style="display: none;">
              </label>
              <div class="row-wrap" style="display: flex; gap: 8px;">
                <input class="input" data-testid="doc-author-input" style="flex: 1 1 160px; background: var(--bg);" v-model="addDoc.author" placeholder="Author *">
                <button class="btn btn-primary" data-testid="doc-upload-btn" style="box-shadow: none;" :disabled="!addDoc.fileObj || !addDoc.author.trim()" @click="uploadDoc">Upload &amp; index</button>
              </div>
            </div>
          </div>

          <div style="display: flex; flex-direction: column; gap: 8px;">
            <span style="font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">Ingestion queue</span>
            <div v-if="!queue.length" style="padding: 14px; border-radius: 8px; border: 1px dashed var(--border-2); background: var(--panel-2); font-size: 12px; color: var(--text-faint); display: flex; align-items: center; gap: 9px;">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"></circle><path d="M12 8v4l2.5 2"></path></svg>
              Queue idle — imports and auto-syncs appear here and keep running even if you close this tab.
            </div>
            <div v-for="q in queue" :key="q.id" class="q-row" data-testid="queue-row" :data-status="q.status" style="display: flex; align-items: center; gap: 14px; padding: 12px 14px; border-radius: 8px;" :style="qRowStyle(q)">
              <svg v-if="q.status === 'running'" width="38" height="38" viewBox="0 0 38 38" style="flex-shrink: 0;">
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--track)" stroke-width="4"></circle>
                <circle cx="19" cy="19" r="15" fill="none" stroke="url(#ringGrad)" stroke-width="4" stroke-linecap="round" :stroke-dasharray="dash(q.pct)" transform="rotate(-90 19 19)"></circle>
                <defs><linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f43f5e"></stop><stop offset="1" stop-color="#6366f1"></stop></linearGradient></defs>
                <text x="19" y="22" text-anchor="middle" font-size="9" font-weight="600" fill="var(--text)" font-family="JetBrains Mono, monospace">{{ q.pct }}%</text>
              </svg>
              <svg v-else-if="q.status === 'queued'" width="38" height="38" viewBox="0 0 38 38" style="flex-shrink: 0; opacity: .65;">
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--track)" stroke-width="4" stroke-dasharray="2.5 5"></circle>
                <g stroke="var(--text-faint)" stroke-width="1.8" stroke-linecap="round" fill="none"><circle cx="19" cy="19" r="7"></circle><path d="M19 15v4l2.6 2"></path></g>
              </svg>
              <svg v-else-if="q.status === 'error'" width="38" height="38" viewBox="0 0 38 38" style="flex-shrink: 0;">
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--track)" stroke-width="4"></circle>
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--red)" stroke-width="4" stroke-linecap="round" :stroke-dasharray="dash(q.pct)" transform="rotate(-90 19 19)" opacity=".55"></circle>
                <g stroke="var(--red)" stroke-width="2" stroke-linecap="round"><path d="M19 12.5v7.5"></path><path d="M19 24.5v.01"></path></g>
              </svg>
              <svg v-else-if="q.status === 'cancelled'" width="38" height="38" viewBox="0 0 38 38" style="flex-shrink: 0;">
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--track)" stroke-width="4"></circle>
                <g stroke="var(--text-faint)" stroke-width="2" stroke-linecap="round"><path d="M14.5 14.5l9 9M23.5 14.5l-9 9"></path></g>
              </svg>
              <svg v-else width="38" height="38" viewBox="0 0 38 38" style="flex-shrink: 0;">
                <circle cx="19" cy="19" r="15" fill="none" stroke="var(--green)" stroke-width="4" opacity=".85"></circle>
                <path d="M13 19.5l4 4 8-8.5" fill="none" stroke="var(--green)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"></path>
              </svg>
              <div class="q-info" style="display: flex; flex-direction: column; gap: 3px; min-width: 0; flex: 1;">
                <span style="display: flex; align-items: center; gap: 8px; min-width: 0;">
                  <span class="mono q-label" style="font-size: 12.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{{ q.url }}</span>
                </span>
                <span v-if="q.status === 'running'" style="font-size: 11.5px; color: var(--text-faint); display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap;"><span v-if="q.origin === 'auto'" style="display: inline-flex; align-items: center; gap: 4px; color: var(--link);"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Auto-sync</span><span class="q-count">{{ q.done }} videos</span><span v-if="q.failed" class="q-count q-failed">· {{ q.failed }} failed</span><span v-if="q.skipped" class="q-count q-skipped">· {{ q.skipped }} skipped (no captions)</span></span>
                <span v-else style="font-size: 11.5px; display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap;" :style="{ color: q.status === 'error' ? 'var(--red)' : 'var(--text-faint)' }"><span v-if="q.origin === 'auto'" style="display: inline-flex; align-items: center; gap: 4px; color: var(--link);"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Auto-sync ·</span><span class="q-count">{{ qSub(q) }}</span><span v-if="q.skipped && q.status !== 'queued'" class="q-count q-skipped">· {{ q.skipped }} skipped</span></span>
                <span v-if="q.channelTotal > q.window && (q.status === 'running' || q.status === 'queued')" data-testid="channel-context" style="font-size: 11px; color: var(--text-ghost); display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                  <span>Channel has ~{{ fmt(q.channelTotal) }} videos; indexing the first {{ fmt(q.window) }}.</span>
                  <button type="button" class="index-more-locked" data-testid="index-more" disabled title="Available after beta"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"></rect><path d="M8 11V7a4 4 0 0 1 8 0v4"></path></svg>Index more</button>
                </span>
              </div>
              <div class="q-actions" style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
                <span v-if="q.status === 'running'" class="badge badge-building" style="font-size: 11px; padding: 2px 9px;"><span class="badge-dot" style="width: 5px; height: 5px;"></span>running</span>
                <span v-else-if="q.status === 'queued'" class="badge badge-queued" style="font-size: 11px; padding: 2px 9px;"><span class="badge-dot" style="width: 5px; height: 5px;"></span>Waiting in queue</span>
                <span v-else-if="q.status === 'error'" class="badge badge-error" style="font-size: 11px; padding: 2px 9px;"><span class="badge-dot" style="width: 5px; height: 5px;"></span>error</span>
                <span v-else-if="q.status === 'cancelled'" class="badge badge-queued" style="font-size: 11px; padding: 2px 9px;">Cancelled</span>
                <span v-else class="badge badge-ready" style="font-size: 11px; padding: 2px 9px;">Ready ✓</span>
                <button v-if="q.status === 'running' || q.status === 'queued'" data-testid="queue-cancel" class="btn btn-ghost" style="height: 30px; padding: 0 10px; font-size: 12px;" @click="cancelJob(q)">Cancel</button>
                <button v-if="q.status === 'error'" data-testid="queue-retry" class="btn btn-secondary" style="height: 30px; padding: 0 12px; font-size: 12px;" @click="retryJob(q)">Retry</button>
                <button v-if="q.status === 'error' || q.status === 'cancelled'" data-testid="queue-dismiss" class="btn btn-danger-ghost" style="height: 30px; padding: 0 10px; font-size: 12px; color: var(--text-faint);" @click="dismissJob(q)">Dismiss</button>
              </div>
            </div>
          </div>

          <div v-if="hasRunning" style="border-radius: 8px; border: 1px solid var(--border-soft); background: var(--panel-2); padding: 14px; display: flex; flex-direction: column; gap: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <span style="font-size: 12.5px; font-weight: 600; color: var(--blue);">Fetching transcripts <span class="mono" style="font-weight: 500; color: var(--text-faint);">{{ runningEntry ? runningEntry.url : '' }}</span></span>
              <span class="mono" style="font-size: 11.5px; color: var(--text-faint);">{{ runningEntry ? runningEntry.done + ' · ' + runningEntry.failed + ' failed' : '' }}</span>
            </div>
            <div class="progress" style="height: 5px;"><div :style="{ width: (runningEntry ? runningEntry.pct : 0) + '%' }"></div></div>
            <div class="mono" style="max-height: 84px; overflow-y: auto; font-size: 10.5px; line-height: 1.7; color: var(--text-faint); background: var(--bg); border-radius: 6px; padding: 8px 10px;">
              <div v-for="line in logLines" :key="line">{{ line }}</div>
            </div>
          </div>
        </div>

        <div class="card" style="padding: 22px; display: flex; flex-direction: column; gap: 14px; min-height: 420px;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 15px; font-weight: 700;">Chat</span>
            <span style="font-size: 12px; color: var(--text-faint);">Try it</span>
          </div>
          <div style="flex: 1; display: flex; flex-direction: column; gap: 12px; overflow-y: auto;">
            <div v-if="!chat.messages.length && !chat.thinking" style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; text-align: center; padding: 32px 20px;">
              <div style="width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, rgba(244,63,94,.15), rgba(99,102,241,.15)); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              </div>
              <span style="font-size: 13px; color: var(--text-dim); max-width: 260px; line-height: 1.5;">Ask this bot something to see grounded answers with sources</span>
            </div>
            <div v-for="(m, i) in chat.messages" :key="i" data-testid="chat-message" :data-role="m.who === 'You' ? 'user' : 'assistant'" :data-error="m.error ? 'true' : 'false'" style="display: flex; flex-direction: column; gap: 4px;" :style="{ alignItems: m.who === 'You' ? 'flex-end' : 'flex-start' }">
              <span style="font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">{{ m.who }}</span>
              <div v-if="m.who === 'You'" data-testid="chat-bubble" style="max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 13px; line-height: 1.55; white-space: pre-wrap; background: var(--bubble-user); border: 1px solid var(--border-3);">{{ m.text }}</div>
              <div v-else data-testid="chat-bubble" class="md-body" style="max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 13px; line-height: 1.55;" :style="{ background: 'var(--panel-2)', border: '1px solid ' + (m.error ? 'rgba(248,113,113,.35)' : 'var(--border)'), color: m.error ? 'var(--red)' : 'var(--text)' }" v-html="renderMd(m.text)"></div>
            </div>
            <div v-if="chat.thinking" style="display: flex; flex-direction: column; gap: 4px;">
              <span style="font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">Assistant</span>
              <div style="align-self: flex-start; padding: 10px 14px; border-radius: 12px; background: var(--panel-2); border: 1px solid var(--border); font-size: 13px; color: var(--text-dim); animation: pulse 1.4s ease-in-out infinite;">…thinking…</div>
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <input class="input" data-testid="chat-input" style="flex: 1;" v-model="chat.input" @keydown.enter="send" :placeholder="chat.streaming ? 'Answering…' : 'Ask anything about the channel…'">
            <button class="btn btn-primary" data-testid="chat-send" style="font-size: 13px;" :disabled="chat.streaming || chat.thinking || !chat.input.trim()" @click="send">Send</button>
          </div>
        </div>

        <div class="card" style="padding: 22px; display: flex; flex-direction: column; gap: 16px;">
          <div style="display: flex; align-items: baseline; gap: 8px;"><span style="font-size: 15px; font-weight: 700;">Telegram</span><span style="font-size: 12px; color: var(--text-faint);">Deploy</span></div>
          <template v-if="!tg.connected">
            <ol style="margin: 0; padding: 0 0 0 2px; list-style: none; display: flex; flex-direction: column; gap: 10px;">
              <li v-for="(text, i) in ['Open @BotFather in Telegram', 'Send /newbot and pick a name + handle', 'Copy the HTTP API token BotFather returns', 'Paste it below and hit Connect', 'Message your bot — it answers from this index']" :key="i" style="display: flex; align-items: flex-start; gap: 10px; font-size: 13px; line-height: 1.5; color: var(--text-mid);">
                <span style="width: 20px; height: 20px; flex-shrink: 0; border-radius: 9999px; background: var(--raised); border: 1px solid var(--border-3); display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: var(--text-dim);">{{ i + 1 }}</span>
                <span>{{ text }}</span>
              </li>
            </ol>
            <div style="display: flex; gap: 8px;">
              <input type="password" class="input mono" style="flex: 1; font-size: 12px;" v-model="tg.token" placeholder="123456:ABC-DEF…">
              <button class="btn btn-primary" style="box-shadow: none;" @click="connectTg">Connect</button>
            </div>
            <span v-if="tg.verifying" style="font-size: 12px; color: var(--blue); animation: pulse 1.4s ease-in-out infinite;">Verifying token…</span>
          </template>
          <div v-else style="display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px; border-radius: 8px; border: 1px solid rgba(52,211,153,.25); background: rgba(52,211,153,.06);">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="width: 8px; height: 8px; border-radius: 9999px; background: var(--green);"></span>
              <span style="font-size: 12.5px; font-weight: 600; color: var(--green);">Connected as <span class="mono" style="font-size: 12px;">{{ tg.username }}</span></span>
            </div>
            <button class="btn btn-danger" style="height: 32px; padding: 0 12px; font-size: 12.5px; font-weight: 600;" @click="askDisconnect">Disconnect</button>
          </div>
        </div>
      </div>
    </template>
  </main>

  <div v-if="shareOpen && bot" class="overlay" @click.self="shareOpen = false">
    <div class="modal" data-testid="share-modal" style="max-width: 460px;">
      <div class="modal-accent"></div>
      <div style="padding: 24px; display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
          <div style="display: flex; flex-direction: column; gap: 4px;">
            <span style="font-size: 17px; font-weight: 700; letter-spacing: -.01em;">Share this bot</span>
            <span style="font-size: 12.5px; color: var(--text-dim);">A private link — no account needed to chat.</span>
          </div>
          <button class="icon-btn" @click="shareOpen = false">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"></path></svg>
          </button>
        </div>
        <template v-if="!shareLink">
          <span style="font-size: 13px; color: var(--text-mid); line-height: 1.55;">Create a private link so a few people you trust can chat with this bot. It runs on <strong>your</strong> quota, so keep the circle small.</span>
          <button class="btn btn-primary" data-testid="share-create" style="width: 100%;" @click="createLink">Create private link</button>
        </template>
        <template v-else>
          <div style="display: flex; gap: 8px;">
            <div class="mono" data-testid="share-link" style="flex: 1; height: 40px; display: flex; align-items: center; padding: 0 12px; border-radius: 8px; border: 1px solid var(--border-3); background: var(--bg); font-size: 12px; color: var(--text-mid); overflow: hidden; white-space: nowrap;">{{ shareLink }}</div>
            <button class="btn btn-secondary" data-testid="share-copy" @click="copyLink" style="min-width: 84px;">{{ copied ? 'Copied ✓' : 'Copy' }}</button>
          </div>
          <div style="display: flex; align-items: center; justify-content: space-between; padding: 0 2px;">
            <span style="font-size: 12.5px; color: var(--text-faint);">Guest messages today</span>
            <span class="mono" data-testid="share-quota" style="font-size: 12.5px; color: var(--text-mid);">{{ guestUsed }} / {{ guestCap }}</span>
          </div>
          <div style="padding: 11px 14px; border-radius: 8px; border: 1px solid rgba(251,191,36,.3); background: rgba(251,191,36,.06); font-size: 12.5px; line-height: 1.5; color: var(--amber);">Anyone with this link can chat with this bot on your quota — share only with people you trust.</div>
          <div style="display: flex; gap: 10px;">
            <button class="btn btn-secondary" data-testid="share-rotate" style="flex: 1;" @click="rotateLink">Rotate link</button>
            <button class="btn btn-danger" data-testid="share-revoke" style="flex: 1;" @click="revokeLink">Revoke</button>
          </div>
          <span style="font-size: 11.5px; color: var(--text-faint); line-height: 1.5;">Rotating makes a new link and instantly disables the old one. Revoking turns sharing off entirely.</span>
        </template>
      </div>
    </div>
  </div>

  <div v-if="confirm" class="overlay" data-testid="confirm-dialog" @click.self="confirm = null">
    <div class="modal" style="max-width: 380px; padding: 24px; display: flex; flex-direction: column; gap: 10px;">
      <span style="font-size: 16px; font-weight: 700;">{{ confirm.title }}</span>
      <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">{{ confirm.msg }}</span>
      <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px;">
        <button class="btn btn-ghost btn-sm" data-testid="confirm-cancel" @click="confirm = null">Cancel</button>
        <button class="btn btn-danger btn-sm" data-testid="confirm-ok" @click="confirm.onOk()">{{ confirm.action }}</button>
      </div>
    </div>
  </div>

  <div class="toasts">
    <div v-for="t in toasts" :key="t.id" class="toast" :class="{ err: !t.ok }"><span class="dot"></span>{{ t.msg }}</div>
  </div>
  `,
};
