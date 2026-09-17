import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue';
import { api } from '../api/index.js';
import SelectField from '../components/SelectField.js';

const STATUS = {
  pending: 'Pending', building: 'Building', ready: 'Ready', error: 'Error',
};
const fmt = (n) => (n || 0).toLocaleString('en-US');

export default {
  components: { SelectField },
  setup() {
    const gridState = ref('loading'); // loading | normal | error
    const bots = ref([]);
    const plan = ref(null);
    const usage = ref(null);
    const showModels = ref(false);
    const form = reactive({ name: '', language: 'English', description: '' });
    const creating = ref(false);
    const created = ref(false);
    const confirm = ref(null);
    const onEsc = (e) => { if (e.key === 'Escape') confirm.value = null; };
    onMounted(() => document.addEventListener('keydown', onEsc));
    onBeforeUnmount(() => document.removeEventListener('keydown', onEsc));
    const toasts = ref([]);
    let nextToast = 1, createdTimer = null;

    const toast = (msg, ok = true) => {
      const id = nextToast++;
      toasts.value.push({ id, msg, ok });
      setTimeout(() => { toasts.value = toasts.value.filter(t => t.id !== id); }, 3200);
    };

    const load = async () => {
      gridState.value = 'loading';
      try {
        const [b, p, u] = await Promise.all([api.listBots(), api.plan(), api.usage()]);
        bots.value = b; plan.value = p; usage.value = u;
        gridState.value = 'normal';
      } catch (e) {
        gridState.value = 'error';
      }
    };
    onMounted(load);

    // Per-card activity badge (ingestion-UX brief §3): poll the cross-bot jobs
    // list so a bot with background work shows "Importing…" / "Syncing…" /
    // "N queued" right on its card. Absent at rest.
    const jobs = ref([]);
    let jobsTimer = null;
    const pollJobs = async () => { try { jobs.value = await api.ingestJobs(); } catch (e) { /* non-fatal */ } };
    onMounted(() => { pollJobs(); jobsTimer = setInterval(pollJobs, 3000); });
    onBeforeUnmount(() => clearInterval(jobsTimer));
    const botActivity = (id) => {
      const mine = jobs.value.filter(j => j.bot_id === id);
      const running = mine.find(j => j.status === 'running');
      const queued = mine.filter(j => j.status === 'queued').length;
      if (running) {
        const label = running.origin === 'auto' ? 'Syncing…' : 'Importing…';
        return { cls: 'badge-building', label: queued ? label + ' · ' + queued + ' queued' : label };
      }
      if (queued) return { cls: 'badge-queued', label: queued + ' queued' };
      return null;
    };

    // Beta ships with billing/upgrade UI hidden behind the plan toggle (Phase I,
    // same gate Account.js uses). When it's off we show the at-limit copy WITHOUT
    // a dead "Upgrade" link into the hidden Account billing section.
    const upgradesEnabled = computed(() => !!(plan.value?.toggles?.upgrades_enabled));
    const botLimit = computed(() => plan.value?.limits?.max_bots ?? 5);
    const atLimit = computed(() => bots.value.length >= botLimit.value);
    const nameOk = computed(() => form.name.trim().length > 0);
    const videoCap = computed(() => plan.value?.limits?.max_videos_per_channel ?? 500);
    const usageBars = computed(() => {
      if (!plan.value) return [];
      const L = plan.value.limits, U = plan.value.usage;
      const bar = (label, used, max) => ({ label, count: `${used} / ${max}`, pct: Math.min(100, max ? used / max * 100 : 0) + '%' });
      return [
        bar('Bots', bots.value.length, L.max_bots),
        bar('Sources', U.sources_used, L.max_sources),
        bar('Active imports', U.active_jobs, L.max_active_jobs),
      ];
    });
    const modelRows = computed(() => (usage.value?.by_model || []).map(m => ({
      name: m.model,
      tokens: fmt(m.prompt_tokens + m.completion_tokens) + ' tok',
      spend: '$' + m.cost_usd.toFixed(3),
      calls: m.calls + ' calls',
    })));

    const createBot = async () => {
      if (!nameOk.value || atLimit.value || creating.value) return;
      creating.value = true;
      try {
        const bot = await api.createBot({ name: form.name.trim(), description: form.description.trim(), language: form.language });
        bots.value.push(bot);
        form.name = ''; form.description = '';
        created.value = true;
        clearTimeout(createdTimer); createdTimer = setTimeout(() => { created.value = false; }, 2400);
        toast('Bot created');
      } catch (e) {
        toast(e.message, false);
      } finally { creating.value = false; }
    };

    const askDelete = (bot) => {
      confirm.value = {
        msg: `“${bot.name}” and its indexed sources will be permanently removed. This can’t be undone.`,
        onOk: async () => {
          confirm.value = null;
          try {
            await api.deleteBot(bot.id);
            bots.value = bots.value.filter(b => b.id !== bot.id);
            toast('Bot deleted');
          } catch (e) { toast(e.message, false); }
        },
      };
    };


    return {
      gridState, bots, plan, usage, showModels, form, creating, created,
      confirm, toasts,
      botLimit, atLimit, upgradesEnabled, nameOk, videoCap, usageBars, modelRows, botActivity,
      load, createBot, askDelete, STATUS, fmt,
      spend: computed(() => '$' + (usage.value?.cost_usd ?? 0).toFixed(4)),
      tokens: computed(() => fmt(usage.value?.total_tokens)),
    };
  },
  template: `
  <main class="page" data-testid="mybots-page">
    <div style="display: flex; flex-direction: column; gap: 5px; margin-bottom: 24px;">
      <span style="font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--link);">Workspace</span>
      <h1 style="margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -.01em;">My Bots</h1>
      <span style="font-size: 13.5px; color: var(--text-dim);">Each bot blends your channels and documents into a private index — with its own persona and Telegram handle.</span>
    </div>

    <div class="card" data-testid="new-bot" style="padding: 20px; margin-bottom: 16px; display: flex; flex-direction: column; gap: 14px;">
      <span style="font-size: 13px; font-weight: 700;">Create a bot</span>
      <div class="grid-2" style="display: grid; grid-template-columns: 1.2fr .8fr; gap: 12px;">
        <label class="field">
          <span class="label">Name <span style="color: var(--accent-from);">*</span></span>
          <input class="input" data-testid="bot-name-input" v-model="form.name" :disabled="atLimit" placeholder="e.g. Cooking Channel Assistant">
        </label>
        <label class="field">
          <span class="label">Language</span>
          <select-field v-model="form.language" :options="['English','Spanish','German','French','Russian','Portuguese']" :class="{ disabled: atLimit }" style="width: 100%;"></select-field>
        </label>
      </div>
      <label class="field">
        <span class="label">Description</span>
        <textarea class="textarea" v-model="form.description" :disabled="atLimit" rows="2" placeholder="What should this bot know and how should it speak?"></textarea>
      </label>
      <div style="display: flex; align-items: center; gap: 12px;">
        <button class="btn btn-primary" data-testid="create-bot-btn" :disabled="!nameOk || atLimit || creating" @click="createBot">{{ creating ? 'Creating…' : 'Create bot' }}</button>
        <span v-if="atLimit && upgradesEnabled" style="font-size: 12.5px; color: var(--amber);">You've reached your plan limit — <router-link to="/account" style="font-weight: 600;">Upgrade</router-link> to add more.</span>
        <span v-else-if="atLimit" style="font-size: 12.5px; color: var(--amber);">You've reached your plan limit.</span>
        <span v-if="created" style="font-size: 12.5px; color: var(--green);">Bot created ✓</span>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; margin-bottom: 28px;">
      <div class="card" style="padding: 18px 20px; display: flex; flex-direction: column; gap: 14px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <span style="font-size: 13px; font-weight: 700;">LLM usage</span>
          <button class="link-btn" @click="showModels = !showModels">
            Show by model
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" :style="{ transform: showModels ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform .2s' }"><path d="M6 9l6 6 6-6"></path></svg>
          </button>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
          <div class="stat-box"><span style="font-size: 11px; color: var(--text-faint);">Spend</span><span class="mono" style="font-size: 17px; font-weight: 500;">{{ spend }}</span></div>
          <div class="stat-box"><span style="font-size: 11px; color: var(--text-faint);">Tokens</span><span class="mono" style="font-size: 17px; font-weight: 500;">{{ tokens }}</span></div>
        </div>
        <div v-if="showModels" style="display: flex; flex-direction: column; gap: 7px; border-top: 1px solid var(--border); padding-top: 12px;">
          <div v-for="m in modelRows" :key="m.name" class="mono" style="display: flex; align-items: center; gap: 10px; font-size: 12px;">
            <span style="color: var(--text-mid); flex: 1;">{{ m.name }}</span>
            <span style="color: var(--text-dim);">{{ m.tokens }}</span><span style="color: var(--text-faint);">·</span>
            <span>{{ m.spend }}</span><span style="color: var(--text-faint);">·</span>
            <span style="color: var(--text-dim);">{{ m.calls }}</span>
          </div>
        </div>
        <span style="font-size: 11px; color: var(--text-faint);">Spend reflects chat LLM calls only — ingestion &amp; embeddings not counted.</span>
      </div>
      <div class="card" data-testid="plan-usage" style="padding: 18px 20px; display: flex; flex-direction: column; gap: 14px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 13px; font-weight: 700;">Your plan</span>
          <span class="badge-plan">{{ plan?.plan ?? '…' }}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
          <div v-for="u in usageBars" :key="u.label" class="stat-box" style="gap: 7px;">
            <span style="font-size: 11px; color: var(--text-faint);">{{ u.label }}</span>
            <span class="mono" style="font-size: 15px; font-weight: 500;">{{ u.count }}</span>
            <div class="progress"><div :style="{ width: u.pct }"></div></div>
          </div>
        </div>
        <span style="font-size: 11px; color: var(--text-faint);">Up to {{ videoCap }} videos per channel.</span>
      </div>
    </div>

    <div v-if="gridState === 'loading'" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;">
      <div v-for="i in 3" :key="i" class="card skeleton" style="height: 196px; padding: 20px; display: flex; flex-direction: column; gap: 12px;">
        <div class="skel-line" style="width: 45%; height: 15px;"></div>
        <div class="skel-line" style="width: 90%; height: 11px; background: var(--border-soft);"></div>
        <div class="skel-line" style="width: 70%; height: 11px; background: var(--border-soft);"></div>
        <div style="flex: 1;"></div>
        <div style="width: 100%; height: 52px; border-radius: 8px; background: var(--panel-2);"></div>
      </div>
    </div>

    <div v-else-if="gridState === 'error'" style="display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 56px 32px; border-radius: 12px; border: 1px solid rgba(248,113,113,.25); background: rgba(248,113,113,.04); text-align: center;">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path></svg>
      <span style="font-size: 15px; font-weight: 600; color: var(--red);">Failed to load bots</span>
      <span style="font-size: 13px; color: var(--text-dim);">Check your connection and try again.</span>
      <button class="btn btn-secondary btn-sm" style="margin-top: 4px;" @click="load">Retry</button>
    </div>

    <div v-else-if="bots.length === 0" style="display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 72px 32px; border-radius: 12px; border: 1px dashed var(--border-2); background: var(--panel-3); text-align: center;">
      <div style="width: 52px; height: 52px; border-radius: 12px; background: linear-gradient(135deg, rgba(244,63,94,.18), rgba(99,102,241,.18)); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--link)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><rect x="4" y="6" width="16" height="12" rx="3"></rect><circle cx="9" cy="12" r="1" fill="#a5b4fc" stroke="none"></circle><circle cx="15" cy="12" r="1" fill="#a5b4fc" stroke="none"></circle></svg>
      </div>
      <div style="display: flex; flex-direction: column; gap: 6px;">
        <span style="font-size: 17px; font-weight: 600;">No bots yet</span>
        <span style="font-size: 13.5px; color: var(--text-dim); max-width: 380px; line-height: 1.55;">Create your first bot above: train it on your YouTube channels and documents, give it a persona, wire it to Telegram.</span>
      </div>
    </div>

    <template v-else>
      <span style="display: block; font-size: 13px; font-weight: 700; margin-bottom: 12px;">Your bots</span>
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;">
        <div v-for="bot in bots" :key="bot.id" class="card bot-card" data-testid="bot-card" :data-bot-id="bot.id" style="display: flex; flex-direction: column; gap: 14px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,.4); transition: border-color .15s, transform .15s;">
          <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;">
            <div style="display: flex; flex-direction: column; gap: 5px; min-width: 0;">
              <span data-testid="bot-card-name" style="font-size: 16px; font-weight: 600; letter-spacing: -.01em; color: var(--text-bright);">{{ bot.name }}</span>
              <span style="font-size: 13px; line-height: 1.5; color: var(--text-dim); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{{ bot.description }}</span>
            </div>
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 6px; flex-shrink: 0;">
              <span class="badge" :class="'badge-' + bot.status">
                <span class="badge-dot"></span>{{ STATUS[bot.status] || bot.status }}
              </span>
              <span v-if="botActivity(bot.id)" class="badge" :class="botActivity(bot.id).cls" style="font-size: 11px; padding: 2px 9px;">
                <span class="badge-dot" style="width: 5px; height: 5px;"></span>{{ botActivity(bot.id).label }}
              </span>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 20px; padding: 12px 14px; border-radius: 8px; background: var(--panel-2); border: 1px solid var(--border-soft);">
            <div style="display: flex; flex-direction: column; gap: 2px;">
              <span class="mono" style="font-size: 15px; font-weight: 500;">{{ bot.source_count }}</span>
              <span style="font-size: 11px; color: var(--text-faint);">Sources</span>
            </div>
            <span class="divider-v"></span>
            <div style="display: flex; flex-direction: column; gap: 2px;">
              <span class="mono" style="font-size: 15px; font-weight: 500;">{{ fmt(bot.chunk_count) }}</span>
              <span style="font-size: 11px; color: var(--text-faint);">Indexed chunks</span>
            </div>
            <div style="margin-left: auto; display: flex; align-items: center; gap: 5px; font-size: 12px;" :style="{ color: bot.telegram_username ? '#8fb8e8' : 'var(--text-faint)' }">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M21.9 4.6c.2-1-.7-1.8-1.6-1.4L2.7 10.1c-1 .4-1 1.9.1 2.2l4.5 1.4 1.7 5.3c.3 1 1.6 1.2 2.2.4l2.4-2.9 4.6 3.4c.9.6 2.1.1 2.3-1l3.4-14.3z" transform="scale(.9) translate(1.2,1.2)"></path></svg>
              {{ bot.telegram_username || 'not connected' }}
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <router-link :to="'/bots/' + bot.id" class="btn btn-secondary btn-sm" data-testid="bot-open" style="flex: 1;">Open</router-link>
            <button class="btn btn-danger-ghost btn-sm" data-testid="bot-delete" @click="askDelete(bot)">Delete</button>
          </div>
        </div>
      </div>
    </template>
  </main>

  <div v-if="confirm" class="overlay" @click.self="confirm = null">
    <div class="modal" style="max-width: 380px; padding: 24px; display: flex; flex-direction: column; gap: 10px;">
      <span style="font-size: 16px; font-weight: 700;">Delete bot?</span>
      <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">{{ confirm.msg }}</span>
      <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px;">
        <button class="btn btn-ghost btn-sm" @click="confirm = null">Cancel</button>
        <button class="btn btn-danger btn-sm" data-testid="bot-delete-confirm" @click="confirm.onOk()">Delete</button>
      </div>
    </div>
  </div>

  <div class="toasts">
    <div v-for="t in toasts" :key="t.id" class="toast" :class="{ err: !t.ok }">
      <span class="dot"></span>{{ t.msg }}
    </div>
  </div>
  `,
};
