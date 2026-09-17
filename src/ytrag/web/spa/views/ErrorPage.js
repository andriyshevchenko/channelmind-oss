const VARIANTS = {
  '404': { code: '404', title: 'Page not found', message: "This page doesn't exist or was moved. Check the address or head back to your bots." },
  'no-access': { code: '403', title: 'Bot not found / no access', message: 'This bot doesn\'t exist, was deleted, or belongs to another account.' },
  'session-expired': { code: '401', title: 'Session expired', message: 'Your session has expired — please sign in again to continue.' },
  'private-share': { code: '403', title: 'This share link is private', message: 'Channelmind bots are shared through a private link the owner sends you. Ask the owner of this bot for a fresh share link to start chatting.' },
};

export default {
  props: { variant: { type: String, default: '404' } },
  computed: {
    m() { return VARIANTS[this.variant] || VARIANTS['404']; },
    isSession() { return this.variant === 'session-expired'; },
  },
  template: `
  <div data-testid="error-page" :data-code="m.code" style="min-height: 100vh; background: radial-gradient(700px 400px at 50% -10%, rgba(99,102,241,.08), transparent), var(--bg); display: flex; align-items: center; justify-content: center; padding: 24px;">
    <div class="modal" style="max-width: 400px;">
      <div class="modal-accent"></div>
      <div style="padding: 40px 32px 32px; display: flex; flex-direction: column; align-items: center; gap: 12px; text-align: center;">
        <span class="mono" style="font-size: 12px; font-weight: 500; padding: 3px 10px; border-radius: 9999px; border: 1px solid;" :style="isSession ? 'color: var(--amber); background: rgba(251,191,36,.1); border-color: rgba(251,191,36,.25)' : 'color: var(--red); background: rgba(248,113,113,.1); border-color: rgba(248,113,113,.25)'">{{ m.code }}</span>
        <span style="font-size: 19px; font-weight: 700; letter-spacing: -.01em;">{{ m.title }}</span>
        <span style="font-size: 13.5px; color: var(--text-dim); line-height: 1.55; max-width: 300px;">{{ m.message }}</span>
        <router-link v-if="isSession" to="/login" class="btn btn-primary" style="margin-top: 10px; font-size: 13.5px;">Sign in</router-link>
        <router-link v-else to="/" class="btn btn-secondary" style="margin-top: 10px; font-size: 13.5px;">Back to My Bots</router-link>
      </div>
    </div>
  </div>
  `,
};
