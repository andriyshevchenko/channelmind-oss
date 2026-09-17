import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue';
import { useRoute } from 'vue-router';
import { api } from '../api/index.js';
import { IS_PREVIEW } from '../config.js';
import { renderMarkdown } from '../mdlite.js';

const DEFAULT_LIMIT = 200;

export default {
  setup() {
    const route = useRoute();
    const token = route.params.token;
    // In PREVIEW the state is derived from the token so every state is demoable
    // (/s/revoked = revoked, /s/limit = quota reached, else active). In PRODUCTION
    // it is loaded from GET /api/s/{token} on mount.
    const initial = IS_PREVIEW
      ? (token === 'revoked' ? 'revoked' : token === 'limit' ? 'limit' : 'active')
      : 'active';
    const view = ref(initial);
    // Bot header data — real values arrive from share-info; the defaults keep the
    // PREVIEW render identical to the original fixture.
    const guestLimit = ref(DEFAULT_LIMIT);
    const guestUsed = ref(initial === 'limit' ? DEFAULT_LIMIT : (IS_PREVIEW ? 12 : 0));
    const botName = ref('Kitchen Alchemy');
    const avatarLetter = ref('K');
    const sourceSummary = ref("Answers grounded in the channel's videos and recipes");
    const chat = reactive({ input: '', thinking: false, messages: [] });
    // OSS build: «Мислення»/Thinking mode disabled — chat always runs the grounded
    // «Довідник»/Reference path. chatMode is pinned; the toggle UI is not rendered.
    const chatMode = ref('reference');
    const setChatMode = (_m) => {};  // inert: toggle removed, grounded Reference only
    const atLimit = computed(() => view.value === 'limit' || guestUsed.value >= guestLimit.value);

    // PRODUCTION: resolve the token → bot header + live quota. A 404 (unknown or
    // revoked token) drops us into the revoked state; the redirect at /s/{token}
    // only sends valid tokens here, but a token can be revoked between click+load.
    onMounted(async () => {
      if (IS_PREVIEW) return;
      try {
        const info = await api.shareInfo(token);
        botName.value = info.bot_name || botName.value;
        if (info.bot_name) avatarLetter.value = info.bot_name.trim().charAt(0).toUpperCase() || 'K';
        if (info.source_summary) sourceSummary.value = info.source_summary;
        const quota = info.quota || {};
        if (typeof quota.daily_cap === 'number') guestLimit.value = quota.daily_cap;
        if (typeof quota.used_today === 'number') guestUsed.value = quota.used_today;
        view.value = 'active';
      } catch (e) {
        view.value = 'revoked';
      }
    });

    const send = async () => {
      const q = chat.input.trim();
      if (!q || chat.thinking || atLimit.value) return;
      if (IS_PREVIEW) {
        chat.messages.push({ who: 'You', text: q });
        chat.input = ''; chat.thinking = true; guestUsed.value++;
        setTimeout(() => {
          chat.thinking = false;
          chat.messages.push({ who: 'Assistant', text: "The trick is starting with a **cold pan** — the fat renders slowly and the skin crisps without burning.\n\n- Score the skin lightly, salt it 20 minutes ahead\n- Medium-low heat, skin-side down, don't move it\n- Flip only when the skin releases on its own\n\nIn the crispy-skin episode this takes about 8 minutes for duck breast, less for chicken thighs.\n\n> [Crispy Skin, Every Time @ 2:12](https://youtu.be/dQw4w9WgXcQ?t=132)\n\n> [Pan Basics: Heat Control @ 0:45](https://youtu.be/dQw4w9WgXcQ?t=45)" });
        }, 1600);
        return;
      }
      // PRODUCTION: real token-scoped guest chat. History in the sanitizer shape.
      const history = chat.messages.map(m => ({ role: m.who === 'You' ? 'user' : 'assistant', content: m.text }));
      chat.messages.push({ who: 'You', text: q });
      chat.input = ''; chat.thinking = true; guestUsed.value++;
      try {
        const res = await api.guestChat(token, { message: q, history, mode: chatMode.value });
        chat.messages.push({ who: 'Assistant', text: res.answer || '' });
      } catch (e) {
        // 429 → the per-bot daily cap / rate limit — surface the quota-reached
        // state. 404 → the token was revoked/rotated after the page loaded — flip to
        // the same 'revoked' state used when share-info 404s on load, not an error
        // bubble. Anything else → a transient failure message in the thread.
        if (e.status === 429) { view.value = 'limit'; guestUsed.value = guestLimit.value; }
        else if (e.status === 404) { view.value = 'revoked'; }
        else chat.messages.push({ who: 'Assistant', text: e.message || 'The bot is temporarily unavailable.' });
      } finally { chat.thinking = false; }
    };
    const setView = (v) => { view.value = v; guestUsed.value = v === 'limit' ? guestLimit.value : 12; };
    return { view, guestUsed, GUEST_LIMIT: guestLimit, chat, chatMode, setChatMode, atLimit, send, setView, IS_PREVIEW, botName, avatarLetter, sourceSummary, renderMd: renderMarkdown };
  },
  template: `
  <div style="min-height: 100vh; background: radial-gradient(700px 400px at 50% -10%, rgba(99,102,241,.08), transparent), var(--bg); display: flex; flex-direction: column; align-items: center; padding: 32px 24px;">

    <div v-if="view === 'revoked'" data-testid="guest-revoked" style="flex: 1; display: flex; align-items: center; justify-content: center; width: 100%;">
      <div class="modal" style="max-width: 400px;">
        <div class="modal-accent"></div>
        <div style="padding: 40px 32px 32px; display: flex; flex-direction: column; align-items: center; gap: 12px; text-align: center;">
          <div style="width: 44px; height: 44px; border-radius: 13px; background: var(--panel-2); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center;">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="var(--text-faint)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13.73 18H6a4 4 0 0 1-1.6-7.7M9.9 4.24A5 5 0 0 1 17 8h1a4 4 0 0 1 2.6 7.06"></path><path d="M2 2l20 20"></path></svg>
          </div>
          <span style="font-size: 18px; font-weight: 700; letter-spacing: -.01em;">This link is no longer active</span>
          <span style="font-size: 13.5px; color: var(--text-dim); line-height: 1.55; max-width: 300px;">The owner turned sharing off or replaced the link. Ask them for a new one.</span>
        </div>
      </div>
    </div>

    <div v-else data-testid="guest-chat" style="width: 100%; max-width: 680px; flex: 1; display: flex; flex-direction: column; gap: 16px;">
      <div style="display: flex; align-items: center; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border);">
        <div style="width: 40px; height: 40px; border-radius: 9999px; background: linear-gradient(135deg, #f59e0b, #ef4444); display: flex; align-items: center; justify-content: center; font-size: 16px; font-weight: 700; color: #fff;">{{ avatarLetter }}</div>
        <div style="display: flex; flex-direction: column; gap: 1px; flex: 1; min-width: 0;">
          <span style="font-size: 16px; font-weight: 700; letter-spacing: -.01em;">{{ botName }}</span>
          <span style="font-size: 12px; color: var(--text-dim);">{{ sourceSummary }}</span>
        </div>
        <span class="mono" data-testid="guest-cap" style="font-size: 11px; color: var(--text-faint); white-space: nowrap;">{{ guestUsed }} / {{ GUEST_LIMIT }} today</span>
      </div>
      <div style="flex: 1; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; padding: 8px 0;">
        <div v-if="!chat.messages.length && !chat.thinking && !atLimit" style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; text-align: center; padding: 48px 20px;">
          <div style="width: 44px; height: 44px; border-radius: 13px; background: linear-gradient(135deg, rgba(244,63,94,.15), rgba(99,102,241,.15)); border: 1px solid var(--border-2); display: flex; align-items: center; justify-content: center;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--link)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
          </div>
          <span style="font-size: 13.5px; color: var(--text-dim); max-width: 300px; line-height: 1.5;">Ask anything about the channel — answers cite the videos they come from.</span>
        </div>
        <div v-for="(m, i) in chat.messages" :key="i" data-testid="guest-message" :data-role="m.who === 'You' ? 'user' : 'assistant'" style="display: flex; flex-direction: column; gap: 4px;" :style="{ alignItems: m.who === 'You' ? 'flex-end' : 'flex-start' }">
          <span style="font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">{{ m.who }}</span>
          <div v-if="m.who === 'You'" style="max-width: 85%; padding: 11px 15px; border-radius: 12px; font-size: 13.5px; line-height: 1.55; white-space: pre-wrap; background: var(--bubble-user); border: 1px solid var(--border-3);">{{ m.text }}</div>
          <div v-else class="md-body" style="max-width: 85%; padding: 11px 15px; border-radius: 12px; font-size: 13.5px; line-height: 1.55; background: var(--panel-2); border: 1px solid var(--border);" v-html="renderMd(m.text)"></div>
        </div>
        <div v-if="chat.thinking" style="display: flex; flex-direction: column; gap: 4px;">
          <span style="font-size: 10.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-faint);">Assistant</span>
          <div style="align-self: flex-start; padding: 11px 15px; border-radius: 12px; background: var(--panel-2); border: 1px solid var(--border); font-size: 13.5px; color: var(--text-dim); animation: pulse 1.4s ease-in-out infinite;">…thinking…</div>
        </div>
      </div>
      <div v-if="atLimit" data-testid="guest-limit" style="padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(251,191,36,.3); background: rgba(251,191,36,.06); font-size: 12.5px; line-height: 1.5; color: var(--amber); text-align: center;">Guest message limit reached for today ({{ GUEST_LIMIT }}) — come back tomorrow.</div>
      <div style="display: flex; gap: 8px;">
        <input class="input" data-testid="guest-input" style="flex: 1; height: 44px; font-size: 13.5px;" v-model="chat.input" @keydown.enter="send" :disabled="atLimit" :placeholder="atLimit ? 'Chat paused until tomorrow' : 'Ask ' + botName + '…'">
        <button class="btn btn-primary" data-testid="guest-send" style="height: 44px; padding: 0 20px;" :disabled="atLimit" @click="send">Send</button>
      </div>
      <router-link to="/about" style="align-self: center; display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--text-faint);">
        <span style="width: 12px; height: 12px; border-radius: 4px; background: var(--accent-grad); display: inline-block;"></span>
        Powered by Channelmind
      </router-link>
    </div>

    <div v-if="IS_PREVIEW" style="position: fixed; bottom: 16px; left: 16px; z-index: 70;" class="seg">
      <button :class="{ on: view === 'active' }" @click="setView('active')">Active</button>
      <button :class="{ on: view === 'limit' }" @click="setView('limit')">Quota reached</button>
      <button :class="{ on: view === 'revoked' }" @click="setView('revoked')">Revoked</button>
    </div>
  </div>
  `,
};
