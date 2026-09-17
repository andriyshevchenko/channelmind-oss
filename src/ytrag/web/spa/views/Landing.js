export default {
  template: `
  <div style="min-height: 100vh; background: radial-gradient(900px 500px at 50% -10%, rgba(99,102,241,.12), transparent), var(--bg); display: flex; flex-direction: column;">
    <header style="max-width: 1080px; width: 100%; margin: 0 auto; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; align-items: center; gap: 10px;">
        <div class="logo-mark">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><rect x="4" y="6" width="16" height="12" rx="3"></rect><circle cx="9" cy="12" r="1" fill="#fff" stroke="none"></circle><circle cx="15" cy="12" r="1" fill="#fff" stroke="none"></circle></svg>
        </div>
        <span class="logo-name">Channelmind</span>
      </div>
      <router-link to="/login" class="btn btn-ghost btn-sm" style="color: var(--text); font-weight: 600;">Sign in</router-link>
    </header>
    <main style="max-width: 1080px; width: 100%; margin: 0 auto; padding: 72px 32px 96px; flex: 1;">
      <div style="display: flex; flex-direction: column; align-items: center; text-align: center; gap: 18px;">
        <div style="width: 56px; height: 56px; border-radius: 16px; background: var(--accent-grad); display: flex; align-items: center; justify-content: center; box-shadow: 0 12px 40px rgba(99,102,241,.4);">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><rect x="4" y="6" width="16" height="12" rx="3"></rect><circle cx="9" cy="12" r="1" fill="#fff" stroke="none"></circle><circle cx="15" cy="12" r="1" fill="#fff" stroke="none"></circle></svg>
        </div>
        <h1 style="margin: 0; font-size: 44px; font-weight: 700; letter-spacing: -.02em; line-height: 1.1; max-width: 640px;">Channelmind</h1>
        <p style="margin: 0; font-size: 18px; color: var(--text-dim); max-width: 520px; line-height: 1.5;">Grounded chat over your YouTube channel's transcripts.</p>
        <router-link to="/login" class="google-btn" style="margin-top: 10px; width: auto; height: 44px; padding: 0 22px; font-size: 14px;">
          <svg width="16" height="16" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"></path><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z"></path><path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84z"></path><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15A11 11 0 0 0 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"></path></svg>
          Sign in with Google
        </router-link>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 80px;">
        <div class="card" style="padding: 24px; display: flex; flex-direction: column; gap: 10px;">
          <span style="font-size: 18px; color: var(--red);">▶</span>
          <span style="font-size: 15px; font-weight: 700;">Ingest a channel</span>
          <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">Point Channelmind at your channel or documents — transcripts become a private, searchable index.</span>
        </div>
        <div class="card" style="padding: 24px; display: flex; flex-direction: column; gap: 10px;">
          <span style="font-size: 18px; color: var(--link);">✦</span>
          <span style="font-size: 15px; font-weight: 700;">Give it a persona</span>
          <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">A short interview turns your voice and style into a system prompt — no prompt engineering needed.</span>
        </div>
        <div class="card" style="padding: 24px; display: flex; flex-direction: column; gap: 10px;">
          <span style="font-size: 18px; color: var(--green);">↗</span>
          <span style="font-size: 15px; font-weight: 700;">Share it privately</span>
          <span style="font-size: 13px; color: var(--text-dim); line-height: 1.55;">A private share link for a few people you trust, or your own Telegram bot — they chat, answers stay grounded in your content.</span>
        </div>
      </div>
    </main>
    <footer style="border-top: 1px solid var(--border);">
      <div style="max-width: 1080px; margin: 0 auto; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--text-faint);">
        <span>© 2026 Channelmind</span>
        <div style="display: flex; gap: 18px;">
          <router-link to="/terms" style="color: var(--text-dim);">Terms</router-link>
          <router-link to="/privacy" style="color: var(--text-dim);">Privacy</router-link>
        </div>
      </div>
    </footer>
  </div>
  `,
};
