import { ref, reactive, computed, onMounted } from 'vue';
import { api } from '../api/index.js';
import SelectField from '../components/SelectField.js';

const PROV = [
  { id: 'llm', label: 'LLM provider', options: ['anthropic', 'openai', 'openrouter'] },
  { id: 'embedding', label: 'Embedding provider', options: ['voyage', 'openai', 'openrouter'] },
  { id: 'transcription', label: 'Transcription provider', options: ['groq', 'openai'] },
  { id: 'vision', label: 'Vision provider', options: ['openrouter', 'openai'] },
];
const KEYS = ['anthropic', 'openai', 'openrouter', 'voyage', 'groq'];
// BYOK is now per-capability: the save payload persists a key for EVERY selected
// provider (not chat-only) plus the embed/transcription/vision provider choices —
// see user_settings per-capability resolution.

export default {
  components: { SelectField },
  setup() {
    const mode = ref('byok');
    const prov = reactive({ llm: 'anthropic', embedding: 'voyage', transcription: 'groq', vision: 'openrouter' });
    const models = reactive({ llm: '', embedding: '', transcription: '', vision: '' });
    const keys = reactive({ anthropic: '', openai: '', openrouter: '', voyage: '', groq: '' });
    // Key-set state is hydrated from GET /api/me/settings on mount (N3) — no more
    // hardcoded prototype values. The backend now tracks a key-set flag for EVERY
    // provider (chat + embed/transcription/vision), so each row reflects real state.
    const saved = reactive({ anthropic: false, voyage: false, groq: false, openai: false, openrouter: false });
    const savedMsg = ref(false);
    const toast = ref(null);
    const errors = ref([]);
    const defaultSync = ref('weekly');
    const emailNotify = ref(true);
    let t = null;
    const showToast = (msg, ok = true) => {
      toast.value = { msg, ok };
      clearTimeout(t); t = setTimeout(() => { toast.value = null; }, 3200);
    };

    const active = computed(() => new Set(Object.values(prov)));
    const keyRows = computed(() => KEYS
      .map(k => ({
        id: k, label: k.charAt(0).toUpperCase() + k.slice(1),
        locked: !active.value.has(k), saved: saved[k],
        required: active.value.has(k) && !saved[k],
        error: errors.value.includes(k),
      }))
      .sort((a, b) => a.locked - b.locked));

    // Hydrate local state from a masked settings payload (mode + per-provider
    // key-set booleans). Used both on mount and after a save so the UI always
    // reflects the server's truth rather than an optimistic guess.
    const hydrate = (s) => {
      if (!s) return;
      if (s.mode) mode.value = s.mode;
      // Per-capability provider choices ride back masked; a blank means "server
      // default", so only a non-blank value overrides the local selector default.
      if (s.llm_provider) prov.llm = s.llm_provider;
      if (s.embed_provider) prov.embedding = s.embed_provider;
      if (s.transcription_provider) prov.transcription = s.transcription_provider;
      if (s.vision_provider) prov.vision = s.vision_provider;
      if (s.sync_freq) defaultSync.value = s.sync_freq;
      if (typeof s.email_notifications === 'boolean') emailNotify.value = s.email_notifications;
      if (s.keys_set) KEYS.forEach(k => { saved[k] = !!s.keys_set[k]; });
    };

    // Phase K: the default-sync control has no dedicated Save button in the
    // design, so persist it immediately on change via the same masked-settings
    // endpoint. Reverts on failure so the selector never lies about server state.
    const setDefaultSync = async (v) => {
      const prev = defaultSync.value;
      defaultSync.value = v;
      try {
        const res = await api.saveMySettings({ sync_freq: v });
        if (res && res.sync_freq) defaultSync.value = res.sync_freq;
        showToast('Default auto-sync set to ' + v);
      } catch (e) { defaultSync.value = prev; showToast(e.message, false); }
    };

    // Email notifications on/off (item 4). No dedicated Save button in the design,
    // so persist immediately via the same masked-settings endpoint; revert on
    // failure so the switch never lies about server state.
    const setEmailNotify = async (v) => {
      const prev = emailNotify.value;
      emailNotify.value = v;
      try {
        const res = await api.saveMySettings({ email_notifications: v });
        if (res && typeof res.email_notifications === 'boolean') emailNotify.value = res.email_notifications;
        showToast(v ? 'Email notifications on' : 'Email notifications off');
      } catch (e) { emailNotify.value = prev; showToast(e.message, false); }
    };

    onMounted(async () => {
      try { hydrate(await api.mySettings()); } catch (e) { /* stay on defaults */ }
    });

    const save = async () => {
      const missing = KEYS.filter(k => active.value.has(k) && !saved[k] && !keys[k].trim());
      if (missing.length) {
        errors.value = missing;
        showToast('Add API keys for: ' + missing.map(k => k.charAt(0).toUpperCase() + k.slice(1)).join(', '), false);
        return;
      }
      errors.value = [];
      try {
        // Send a key for EACH SELECTED provider the user actually typed — BYOK now
        // spans every capability (chat + embed/transcription/vision), so the payload
        // is no longer chat-only. A BLANK value means "clear this key", so we omit
        // untyped fields (that keeps "leave blank to keep the existing one") and
        // skip locked/deselected providers so a stale value can't wipe a saved key.
        const keyPayload = {};
        KEYS.forEach(k => { if (active.value.has(k) && keys[k].trim()) keyPayload[k] = keys[k].trim(); });
        // Persist the per-capability provider choices alongside mode + keys so the
        // owner's embed/transcription/vision selection round-trips (masked) too.
        const res = await api.saveMySettings({
          mode: mode.value,
          llm_provider: prov.llm,
          embed_provider: prov.embedding,
          transcription_provider: prov.transcription,
          vision_provider: prov.vision,
          keys: keyPayload,
          sync_freq: defaultSync.value,
        });
        KEYS.forEach(k => { if (keys[k].trim()) keys[k] = ''; });
        hydrate(res); // reflect the returned masked state (mode + keys_set)
        savedMsg.value = true;
        showToast('Settings saved');
      } catch (e) { showToast(e.message, false); }
    };

    return { mode, prov, models, keys, savedMsg, toast, defaultSync, emailNotify, keyRows, save, setDefaultSync, setEmailNotify, PROV };
  },
  template: `
  <main class="page" data-testid="settings-page" style="max-width: 860px; padding-top: 40px;">
    <h1 style="margin: 0 0 6px; font-size: 22px; font-weight: 700; letter-spacing: -.01em;">Connection settings</h1>
    <p style="margin: 0 0 28px; font-size: 13px; color: var(--text-dim);">Saved to <span class="mono" style="font-size: 12px; color: var(--link);">settings.json</span>, overrides <span class="mono" style="font-size: 12px; color: var(--link);">.env</span>. Leave a key blank to keep the existing one.</p>

    <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px;">
      <span style="font-size: 12.5px; font-weight: 600; color: var(--text-mid);">LLM access mode</span>
      <div style="display: flex; gap: 8px;">
        <button class="btn" data-testid="settings-mode-managed" :class="mode === 'managed' ? 'btn-primary' : 'btn-ghost'" style="font-size: 13px; box-shadow: none;" @click="mode = 'managed'">Managed</button>
        <button class="btn" data-testid="settings-mode-byok" :class="mode === 'byok' ? 'btn-primary' : 'btn-ghost'" style="font-size: 13px; box-shadow: none;" @click="mode = 'byok'">BYOK — bring your own key</button>
      </div>
    </div>

    <div v-if="mode === 'managed'" class="card" style="padding: 28px; display: flex; align-items: center; gap: 14px;">
      <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(244,63,94,.15), rgba(99,102,241,.15)); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
      </div>
      <div style="display: flex; flex-direction: column; gap: 3px;">
        <span style="font-size: 13.5px; font-weight: 600;">Using Channelmind's managed models.</span>
        <span style="font-size: 12.5px; color: var(--text-dim);">No API keys needed — usage counts against your plan quota. Switch to BYOK to run on your own keys.</span>
      </div>
    </div>

    <div v-else class="card" style="padding: 28px;">
      <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 24px; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border-soft); background: var(--panel-2);">
        <span style="font-size: 12.5px; color: var(--text-dim); line-height: 1.5;">In BYOK mode, plan quotas don't apply — model usage runs on your own keys. Pick a provider for each capability below; every selected provider needs an API key.</span>
      </div>
      <div class="grid-2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px 24px;">
        <div v-for="p in PROV" :key="p.id" style="display: flex; flex-direction: column; gap: 7px;">
          <span style="font-size: 12.5px; font-weight: 600; color: var(--text-mid);">{{ p.label }}</span>
          <div style="display: flex; gap: 8px;">
            <select-field style="flex: 1;" :data-testid="'settings-prov-' + p.id" v-model="prov[p.id]" :options="p.options" @update:model-value="savedMsg = false"></select-field>
            <input class="input mono" style="flex: 1.2; font-size: 12px;" v-model="models[p.id]" placeholder="model override (optional)">
          </div>
          <span v-if="p.id === 'embedding'" data-testid="settings-embed-note" style="font-size: 11.5px; color: var(--text-dim); line-height: 1.4;">Applies to new bots — existing bots keep their embedding provider until you rebuild their index.</span>
        </div>
      </div>
      <div style="height: 1px; background: var(--border); margin: 26px 0;"></div>
      <span style="display: block; font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint); margin-bottom: 14px;">API keys</span>
      <div class="grid-2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px 24px;">
        <div v-for="k in keyRows" :key="k.id" style="display: flex; flex-direction: column; gap: 7px;" :style="{ opacity: k.locked ? .45 : 1 }">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 12.5px; font-weight: 600; color: var(--text-mid); display: flex; align-items: center; gap: 6px; white-space: nowrap;">
              {{ k.label }} API key
              <svg v-if="k.locked" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#5a5a6a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"></rect><path d="M8 11V7a4 4 0 0 1 8 0v4"></path></svg>
            </span>
            <span :data-testid="'settings-key-status-' + k.id" :data-saved="k.saved ? 'true' : 'false'" style="font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 9999px; border: 1px solid; white-space: nowrap;" :style="k.saved ? 'color: var(--green); background: rgba(52,211,153,.1); border-color: rgba(52,211,153,.25)' : k.required ? 'color: var(--amber); background: rgba(251,191,36,.1); border-color: rgba(251,191,36,.25)' : 'color: var(--text-faint); background: rgba(138,138,153,.06); border-color: rgba(138,138,153,.15)'">{{ k.saved ? '✓ saved' : k.required ? 'required' : 'not used' }}</span>
          </div>
          <input type="password" class="input mono" :data-testid="'settings-key-' + k.id" :class="{ invalid: k.error }" style="font-size: 12px;" v-model="keys[k.id]" :disabled="k.locked" :placeholder="k.locked ? 'provider not selected' : 'sk-…'">
        </div>
      </div>
      <div class="row-wrap" style="display: flex; align-items: center; gap: 14px; margin-top: 28px;">
        <button class="btn btn-primary btn-block-sm" data-testid="settings-save-btn" @click="save">Save settings</button>
        <span v-if="savedMsg" style="font-size: 12.5px; color: var(--green);">Saved. Keys marked ✓ saved are stored — blank fields keep existing values.</span>
      </div>
    </div>

    <div class="card" style="padding: 24px 28px; margin-top: 20px;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
        <div style="display: flex; flex-direction: column; gap: 3px; min-width: 0;">
          <span style="font-size: 13.5px; font-weight: 600;">Default auto-sync for new channels</span>
          <span style="font-size: 12.5px; color: var(--text-dim); line-height: 1.5;">Applied when you add a YouTube channel. Each channel can override this on its bot page.</span>
        </div>
        <select-field style="flex-shrink: 0; min-width: 160px;" :model-value="defaultSync" @update:model-value="setDefaultSync($event)" :options="[{value:'off',label:'Off — manual only'},{value:'daily',label:'Daily'},{value:'weekly',label:'Weekly'},{value:'monthly',label:'Monthly'}]"></select-field>
      </div>
    </div>

    <div class="card" data-testid="email-notify-card" style="padding: 24px 28px; margin-top: 20px;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
        <div style="display: flex; flex-direction: column; gap: 3px; min-width: 0;">
          <span style="font-size: 13.5px; font-weight: 600;">Email notifications</span>
          <span style="font-size: 12.5px; color: var(--text-dim); line-height: 1.5;">Get an email when an import finishes, a sync fails, or a bot needs attention.</span>
        </div>
        <button type="button" role="switch" data-testid="email-notify-toggle" :aria-checked="emailNotify ? 'true' : 'false'" class="cm-switch" :class="{ on: emailNotify }" @click="setEmailNotify(!emailNotify)"></button>
      </div>
    </div>
  </main>

  <div class="toasts">
    <div v-if="toast" class="toast" :class="{ err: !toast.ok }"><span class="dot"></span>{{ toast.msg }}</div>
  </div>
  `,
};
