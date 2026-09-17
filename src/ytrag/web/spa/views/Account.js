import { ref, inject, computed, onMounted, onBeforeUnmount } from 'vue';
import { api } from '../api/index.js';
import { IS_PREVIEW } from '../config.js';

const TIERS = [
  { name: 'Free', price: '$0', per: '/ month', popular: false, cta: 'Current plan', style: 'ghost',
    features: ['5 bots', '20 sources', '2 active imports', '500 videos per channel'] },
  { name: 'Pro', price: '$19', per: '/ month', popular: true, cta: 'Upgrade to Pro', style: 'primary',
    features: ['25 bots', '200 sources', '10 active imports', '2,000 videos per channel', 'Priority ingestion'] },
  { name: 'BYOK', price: '$9', per: '/ month + your keys', popular: false, cta: 'Switch to BYOK', style: 'secondary',
    features: ['Unlimited LLM usage on your keys', '25 bots', '200 sources', 'Provider & model control'] },
];

export default {
  setup() {
    const user = inject('user', ref(null));
    const upgradeOpen = ref(false);
    // Billing/upgrade UI is gated behind the plan toggle (Phase I). Beta ships it
    // OFF, so the Upgrade button + tier grid render ONLY when the plan says so —
    // the section is hidden entirely (no empty placeholder) otherwise.
    const upgradesEnabled = ref(false);
    const confirm = ref(false);
    const loadError = ref(false);
    const onEsc = (e) => { if (e.key === 'Escape') confirm.value = false; };
    onMounted(() => document.addEventListener('keydown', onEsc));
    onBeforeUnmount(() => document.removeEventListener('keydown', onEsc));
    const toast = ref(null);
    let t = null;
    const showToast = (msg, ok = true) => {
      toast.value = { msg, ok };
      clearTimeout(t); t = setTimeout(() => { toast.value = null; }, 3000);
    };
    const usage = ref([
      { label: 'Bots', count: '4 / 5', pct: '80%', hasBar: true },
      { label: 'Sources', count: '10 / 20', pct: '50%', hasBar: true },
      { label: 'Active imports', count: '1 / 2', pct: '50%', hasBar: true },
      { label: 'Spend', count: '$0.0473', hasBar: false },
      { label: 'Tokens', count: '128,540', hasBar: false },
    ]);
    api.plan().then(p => {
      upgradesEnabled.value = !!(p.toggles && p.toggles.upgrades_enabled);
      const L = p.limits, U = p.usage;
      usage.value = [
        { label: 'Bots', count: `${U.bots_used} / ${L.max_bots}`, pct: Math.min(100, U.bots_used / L.max_bots * 100) + '%', hasBar: true },
        { label: 'Sources', count: `${U.sources_used} / ${L.max_sources}`, pct: Math.min(100, U.sources_used / L.max_sources * 100) + '%', hasBar: true },
        { label: 'Active imports', count: `${U.active_jobs} / ${L.max_active_jobs}`, pct: Math.min(100, U.active_jobs / L.max_active_jobs * 100) + '%', hasBar: true },
        { label: 'Spend', count: '$' + (p.budget?.used_usd ?? 0).toFixed(4), hasBar: false },
      ];
    }).catch(() => { loadError.value = true; });
    const deleteAccount = async () => {
      confirm.value = false;
      try { await api.deleteAccount(); showToast('Account deletion scheduled', false); }
      catch (e) { showToast(e.message, false); }
    };
    // PREVIEW-ONLY plan-state switcher (same pattern as PublicChat's state seg):
    // lets designers flip between the beta state (no billing UI) and the Pro state
    // (upgrade button + tier grid) without a backend. No effect in production.
    const setPreviewPlan = (pro) => {
      upgradesEnabled.value = pro;
      upgradeOpen.value = pro; // open the tier grid immediately so Pro is visible
    };

    return { user, usage, upgradeOpen, upgradesEnabled, confirm, loadError, toast, TIERS, showToast, deleteAccount, logout: () => api.logout(), IS_PREVIEW, setPreviewPlan };
  },
  template: `
  <main class="page" data-testid="account-page" style="max-width: 860px; padding-top: 40px; display: flex; flex-direction: column; gap: 16px;">
    <h1 style="margin: 0 0 8px; font-size: 22px; font-weight: 700; letter-spacing: -.01em;">Account</h1>
    <div v-if="loadError" style="display: flex; align-items: center; gap: 9px; padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(248,113,113,.25); background: rgba(248,113,113,.05); font-size: 12.5px; color: var(--red);">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path></svg>
      Couldn't load your plan usage — refresh to retry. Account details below are unaffected.
    </div>

    <div class="card" style="padding: 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
      <div style="display: flex; align-items: center; gap: 14px;">
        <div class="avatar" style="width: 48px; height: 48px; font-size: 19px;">{{ user?.name ? user.name[0] : '?' }}</div>
        <div style="display: flex; flex-direction: column; gap: 2px;">
          <span style="font-size: 15px; font-weight: 700;">{{ user?.name }}</span>
          <span style="font-size: 13px; color: var(--text-dim);">{{ user?.email }}</span>
        </div>
      </div>
      <button class="btn btn-ghost btn-sm" style="color: var(--text-mid); font-weight: 600;" @click="logout">Log out</button>
    </div>

    <div class="card" data-testid="plan-usage" style="padding: 24px; display: flex; flex-direction: column; gap: 16px;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 14px; font-weight: 700;">Your plan</span>
          <span class="badge-plan">Free</span>
        </div>
        <button v-if="upgradesEnabled" class="btn btn-primary btn-sm" style="font-size: 13px;" @click="upgradeOpen = !upgradeOpen">{{ upgradeOpen ? 'Hide plans' : 'Upgrade plan' }}</button>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;">
        <div v-for="u in usage" :key="u.label" class="stat-box" style="gap: 7px;">
          <span style="font-size: 11px; color: var(--text-faint);">{{ u.label }}</span>
          <span class="mono" style="font-size: 15px; font-weight: 500;">{{ u.count }}</span>
          <div v-if="u.hasBar" class="progress"><div :style="{ width: u.pct }"></div></div>
        </div>
      </div>
      <div v-if="upgradesEnabled && upgradeOpen" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; border-top: 1px solid var(--border); padding-top: 18px;">
        <div v-for="tier in TIERS" :key="tier.name" style="border-radius: 12px; border: 1px solid; background: var(--panel-2); padding: 20px; display: flex; flex-direction: column; gap: 12px; position: relative;" :style="{ borderColor: tier.popular ? 'var(--indigo)' : 'var(--border-2)' }">
          <span v-if="tier.popular" style="position: absolute; top: -9px; left: 16px; font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; padding: 2px 8px; border-radius: 9999px; background: var(--accent-grad); color: #fff;">Popular</span>
          <div style="display: flex; flex-direction: column; gap: 2px;">
            <span style="font-size: 14px; font-weight: 700;">{{ tier.name }}</span>
            <span class="mono" style="font-size: 20px; font-weight: 700;">{{ tier.price }}<span style="font-size: 11px; font-weight: 400; color: var(--text-faint);"> {{ tier.per }}</span></span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 6px;">
            <span v-for="f in tier.features" :key="f" style="font-size: 12.5px; color: var(--text-mid); display: flex; align-items: center; gap: 7px;"><span style="color: var(--green);">✓</span>{{ f }}</span>
          </div>
          <button class="btn btn-sm" style="margin-top: auto; font-size: 13px; box-shadow: none;" :class="'btn-' + tier.style" @click="tier.style !== 'ghost' && showToast('Checkout would open here')">{{ tier.cta }}</button>
        </div>
      </div>
    </div>

    <div style="border-radius: 12px; border: 1px solid rgba(248,113,113,.25); background: rgba(248,113,113,.03); padding: 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
      <div style="display: flex; flex-direction: column; gap: 4px;">
        <span style="font-size: 14px; font-weight: 700; color: var(--red);">Danger zone</span>
        <span style="font-size: 12.5px; color: var(--text-dim); max-width: 480px; line-height: 1.5;">Deleting your account permanently removes all bots, sources, and indexes. This can't be undone.</span>
      </div>
      <button class="btn btn-danger btn-sm" data-testid="account-delete" style="font-weight: 600;" @click="confirm = true">Delete account</button>
    </div>

    <div style="display: flex; gap: 16px; font-size: 12px; padding-top: 4px;">
      <router-link to="/terms" style="color: var(--text-faint);">Terms of Service</router-link>
      <router-link to="/privacy" style="color: var(--text-faint);">Privacy Policy</router-link>
    </div>
  </main>

  <div v-if="confirm" class="overlay" @click.self="confirm = false">
    <div class="modal" style="max-width: 400px; padding: 24px; display: flex; flex-direction: column; gap: 10px;">
      <span style="font-size: 16px; font-weight: 700;">Delete account?</span>
      <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">All your bots, sources, and indexes will be permanently deleted, and connected Telegram bots will stop responding.</span>
      <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px;">
        <button class="btn btn-ghost btn-sm" @click="confirm = false">Cancel</button>
        <button class="btn btn-danger btn-sm" data-testid="account-delete-confirm" @click="deleteAccount">Delete everything</button>
      </div>
    </div>
  </div>

  <div v-if="IS_PREVIEW" style="position: fixed; bottom: 16px; left: 16px; z-index: 70;" class="seg">
    <button :class="{ on: !upgradesEnabled }" @click="setPreviewPlan(false)">Beta</button>
    <button :class="{ on: upgradesEnabled }" @click="setPreviewPlan(true)">Pro</button>
  </div>

  <div class="toasts">
    <div v-if="toast" class="toast" :class="{ err: !toast.ok }"><span class="dot"></span>{{ toast.msg }}</div>
  </div>
  `,
};
