import { ref } from 'vue';
import { useRoute } from 'vue-router';
import { api } from '../api/index.js';
import { IS_PREVIEW, DEV_LOGIN } from '../config.js';

export default {
  setup() {
    const route = useRoute();
    const showError = ref(route.query.error != null);
    return { showError, IS_PREVIEW, DEV_LOGIN, login: () => api.loginWithGoogle() };
  },
  template: `
  <div data-testid="login-page" style="min-height: 100vh; background: radial-gradient(800px 500px at 50% -10%, rgba(99,102,241,.09), transparent), var(--bg); display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px;">
    <div style="width: 100%; max-width: 380px; border-radius: 12px; border: 1px solid var(--border); background: var(--panel); box-shadow: 0 24px 64px rgba(0,0,0,.5); overflow: hidden;">
      <div class="modal-accent"></div>
      <div style="padding: 36px 32px 28px; display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: var(--accent-grad); display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(99,102,241,.35); margin-bottom: 6px;">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><rect x="4" y="6" width="16" height="12" rx="3"></rect><circle cx="9" cy="12" r="1" fill="#fff" stroke="none"></circle><circle cx="15" cy="12" r="1" fill="#fff" stroke="none"></circle></svg>
        </div>
        <span style="font-size: 19px; font-weight: 700; letter-spacing: -.01em;">Channelmind</span>
        <span style="font-size: 13.5px; color: var(--text-dim);">Sign in to continue</span>
        <div v-if="showError" style="width: 100%; margin-top: 10px; padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(248,113,113,.3); background: rgba(248,113,113,.08); color: var(--red); font-size: 12.5px; text-align: left; display: flex; align-items: center; gap: 8px;">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path></svg>
          Sign-in failed. Please try again.
        </div>
        <button @click="login" class="google-btn">
          <svg width="16" height="16" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"></path><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z"></path><path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84z"></path><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15A11 11 0 0 0 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"></path></svg>
          Sign in with Google
        </button>
        <a v-if="DEV_LOGIN" href="/auth/dev" data-testid="dev-login" style="margin-top: 8px; font-size: 12.5px; color: var(--text-dim);">Continue as local user</a>
        <router-link v-else-if="IS_PREVIEW" to="/" style="margin-top: 8px; font-size: 12.5px; color: var(--text-dim);">Continue as local dev user</router-link>
      </div>
    </div>
    <span style="margin-top: 20px; font-size: 12px; color: var(--text-faint);">Grounded chat over your YouTube channel's transcripts.</span>
    <div style="margin-top: 10px; display: flex; gap: 16px; font-size: 11.5px;">
      <router-link to="/terms" style="color: var(--text-faint);">Terms</router-link>
      <router-link to="/privacy" style="color: var(--text-faint);">Privacy</router-link>
    </div>
  </div>
  `,
};
