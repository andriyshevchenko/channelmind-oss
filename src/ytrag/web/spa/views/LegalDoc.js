const SECTIONS = {
  terms: {
    title: 'Terms of Service', updated: 'Effective July 17, 2026',
    intro: `These Terms of Service ("Terms") are an agreement between you and Channelmind ("we", "us"), the operator of the service. By signing in and using the service you accept these Terms.`,
    sections: [
      ['1. What Channelmind is', `Channelmind is a personal utility. You build a chat assistant ("bot") from one or a few YouTube channels and documents you provide, and the bot answers questions grounded in that content, with citations. It exists so you can get grounded answers from material you'd otherwise have to watch or read in full. The service is provided as-is.`],
      ['2. Private sharing only', `You may share a bot with a small circle of people you trust using a private, unguessable share link, or by connecting your own Telegram bot. Shared bots run on your account's usage and are protected by per-bot caps and rate limits. Channelmind keeps sharing private — a bot is never placed on an openly reachable page, never inserted into another website, and never listed in a catalogue. You may not use the service to operate a bot as an open or commercial content service.`],
      ['3. Your content and your rights attestation', `You may only add channels and documents you own or otherwise have the rights to use for a private bot shared with a small circle. Each time you add a source you attest that you hold those rights; we record that attestation. You keep ownership of your content and grant us only the limited licence needed to index it, generate grounded answers from it, and serve those answers to the people you privately share the bot with. Removing a source or bot removes its index.`],
      ['4. Acceptable use', `Don't use Channelmind to infringe anyone's rights, to build bots that impersonate a real person without their consent, to redistribute content publicly, or to generate unlawful content. We may suspend or remove bots and accounts that breach these Terms.`],
      ['5. Plans, usage, and keys', `Bot usage — including guests you share with — runs on the account owner. On a managed plan, model usage runs on our keys within your plan's limits. If you bring your own API keys (BYOK), model usage runs on your keys and those plan limits don't apply to model usage. You are responsible for usage on your account, including by people you share a link with.`],
      ['6. Dependence on third-party platforms', `Channelmind fetches YouTube transcripts using third-party tooling that we do not control. Ingesting new content depends on that tooling and on YouTube's own availability, and may be slowed, interrupted, or unavailable at any time. Content you have already ingested keeps working regardless. We make no guarantee that any particular channel can be ingested or re-ingested.`],
      ['7. Copyright & takedowns', `If you are a rights holder and believe content has been added without permission, contact takedown@example.com. On a valid notice we will remove the affected source across the accounts that hold it, delete its stored transcripts, and rebuild the affected bots so they no longer draw on that channel. We record takedown actions we take.`],
      ['8. Termination', `You can stop using the service at any time and request deletion of your account and its data (see the Privacy Policy). We may suspend or terminate accounts that materially breach these Terms.`],
      ['9. Disclaimers & liability', `Bots generate answers from the content you provide and can be wrong or incomplete; don't rely on them where accuracy matters without checking the cited sources. The service is provided "as is" without warranties, and to the extent permitted by law our liability arising from the service is limited. Nothing here excludes liability that cannot be excluded by law.`],
      ['10. Governing law & changes', `These Terms are governed by the laws of the operator's principal place of business. We may update these Terms; material changes will be reflected in the effective date above, and continued use after a change means you accept it. Questions: takedown@example.com.`],
    ],
  },
  privacy: {
    title: 'Privacy Policy', updated: 'Effective July 17, 2026',
    intro: `This policy explains how Channelmind ("we", "us") handles personal data. For any privacy question you can contact us at takedown@example.com.`,
    sections: [
      ['1. What we collect', `Your Google account name and email (for sign-in); the channels and documents you add and the transcripts fetched for them; the personas you generate; chat messages sent to your bots (yours and those of people you share a link with); and basic usage metrics (token counts and spend) used to run the service and enforce limits. We also keep a narrow audit record of legally-significant actions — such as your rights attestation when you add a source, and share-link and takedown actions — not your browsing or individual chats.`],
      ['2. How and why we use it', `We use this data to build and serve your bots: transcripts and documents are split into passages, indexed as vectors, and stored in a private index scoped to your account, and chat messages are sent to the model provider to generate grounded answers. The lawful bases are performance of our agreement with you (running the service you asked for) and our legitimate interests in operating, securing, and supporting it. We do not sell your data or use your content to train models.`],
      ['3. Private sharing', `Sharing is private only. A bot is reachable through an unguessable link you create and can revoke, or through your own Telegram bot — there are no openly reachable bot pages. Anyone with a live link can chat with that bot on your account's usage, so share links only with people you trust. Messages that guests send are processed to answer them and are associated with the bot, not with a guest identity.`],
      ['4. Third-party processors', `To run the service we rely on providers acting on our behalf: a hosting provider for the server; model providers that generate answers and build the search index; and Telegram where you connect a bot. Transcripts are fetched from YouTube using third-party tooling; that dependency means new ingestion may be interrupted or unavailable at times, though content already ingested is unaffected. We share only what each provider needs to perform its function.`],
      ['5. API keys (BYOK)', `If you bring your own provider keys, they are treated as secrets: stored with restricted access, masked in the interface, used only to call the providers you selected, never shown back in full, and never written to logs. You can replace or clear a key at any time.`],
      ['6. Retention', `We keep account data, bots, sources, transcripts, and usage records for as long as your account is active. When you remove a source or bot, its index and stored transcripts are deleted. You can delete your account and all its data yourself at any time from the Account page — this removes your bots, their sources and transcripts, derived indexes, and usage records. A minimal audit record that an action occurred may be retained where we need it to evidence compliance, without retaining the deleted content itself.`],
      ['7. Your rights', `Subject to applicable law you may request access to, correction of, a copy of, or erasure of your personal data, and may object to or restrict certain processing. Erasure is available self-serve from your Account page; for access, correction, or a copy, contact takedown@example.com and we will respond within the time the law requires. You may also complain to your local data protection authority.`],
      ['8. Deletion, takedowns & contact', `Deleting a bot or source permanently removes its associated indexes; to delete your whole account and all its data, use the delete option on your Account page. Rights holders can request removal of a channel via takedown@example.com; we purge the affected source, its transcripts, and its vector chunks across accounts on a valid notice. For any privacy question, contact takedown@example.com.`],
    ],
  },
};

export default {
  props: { kind: { type: String, default: 'terms' } },
  computed: { doc() { return SECTIONS[this.kind]; } },
  template: `
  <div style="min-height: 100vh; background: var(--bg);">
    <header style="border-bottom: 1px solid var(--border);">
      <div style="max-width: 720px; margin: 0 auto; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between;">
        <router-link to="/about" style="display: flex; align-items: center; gap: 10px; color: var(--text);">
          <div style="width: 24px; height: 24px; border-radius: 7px; background: var(--accent-grad);"></div>
          <span style="font-size: 14px; font-weight: 700;">Channelmind</span>
        </router-link>
        <div style="display: flex; gap: 16px; font-size: 12.5px;">
          <router-link to="/terms" :style="kind === 'terms' ? 'color: var(--text); font-weight: 600' : 'color: var(--text-dim)'">Terms</router-link>
          <router-link to="/privacy" :style="kind === 'privacy' ? 'color: var(--text); font-weight: 600' : 'color: var(--text-dim)'">Privacy</router-link>
        </div>
      </div>
    </header>
    <main style="max-width: 720px; margin: 0 auto; padding: 48px 32px 96px;">
      <h1 style="margin: 0 0 6px; font-size: 28px; font-weight: 700; letter-spacing: -.01em;">{{ doc.title }}</h1>
      <p style="margin: 0 0 36px; font-size: 12.5px; color: var(--text-faint);">{{ doc.updated }}</p>
      <div style="display: flex; flex-direction: column; gap: 28px; font-size: 14px; line-height: 1.7; color: var(--text-mid);">
        <p v-if="doc.intro" style="margin: 0;">{{ doc.intro }}</p>
        <section v-for="s in doc.sections" :key="s[0]">
          <h2 style="margin: 0 0 8px; font-size: 16px; font-weight: 700; color: var(--text);">{{ s[0] }}</h2>
          <p style="margin: 0;">{{ s[1] }}</p>
        </section>
      </div>
    </main>
  </div>
  `,
};
