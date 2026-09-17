export default {
  props: { title: String },
  template: `
  <main class="page" style="display: flex; flex-direction: column; align-items: center; gap: 10px; padding-top: 96px; text-align: center;">
    <span style="font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--link);">Coming up</span>
    <h1 style="margin: 0; font-size: 22px; font-weight: 700;">{{ title }}</h1>
    <span style="font-size: 13.5px; color: var(--text-dim);">This screen hasn't been ported to the new frontend yet.</span>
    <router-link to="/" style="margin-top: 8px;">← Back to My Bots</router-link>
  </main>
  `,
};
