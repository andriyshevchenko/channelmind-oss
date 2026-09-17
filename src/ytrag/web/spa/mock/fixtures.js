// Mock fixtures for preview mode. Mirrors the shapes returned by the real API.
export const user = { id: 'u1', email: 'alex@rivera.dev', name: 'Alex Rivera', picture: '' };

export const bots = [
  { id: 'b1', name: 'Kitchen Alchemy', description: 'Trained on 340 cooking videos and my recipe PDFs. Answers technique questions in my voice, always suggests substitutions.', status: 'ready', source_count: 4, chunk_count: 12483, telegram_username: '@kitchenalchemy_bot', persona: '', language: 'English', suggest_followups: true, include_shorts: false },
  { id: 'b2', name: 'Woodshop Mentor', description: 'Full woodworking channel archive plus tool manuals. Guides beginners through joinery, finishing, and shop safety.', status: 'building', source_count: 2, chunk_count: 3912, telegram_username: '@woodshopmentor_bot', persona: '', language: 'English', suggest_followups: true, include_shorts: false },
  { id: 'b3', name: 'Finance Explainers', description: 'Personal finance series and quarterly newsletter archive.', status: 'pending', source_count: 1, chunk_count: 0, telegram_username: null, persona: '', language: '', suggest_followups: true, include_shorts: false },
  { id: 'b4', name: 'Retro Game Lab', description: 'Console restoration deep-dives and repair guides. Import failed on two playlists — needs a re-run.', status: 'error', source_count: 3, chunk_count: 6027, telegram_username: null, persona: '', language: 'English', suggest_followups: true, include_shorts: true },
];

// Same shape as GET /api/me/plan (api_me_plan): plan/limits/usage/toggles/
// guest_caps/budget/price. `limits` uses the real PlanLimits keys
// (max_sources_per_bot/max_sources_total) — the account-wide max_sources the
// widgets read is bridged in api.plan() exactly like real.js adaptPlan does.
// Default = beta state (upgrades_enabled:false) so preview matches production.
export const plan = {
  plan: 'BETA_USER',
  limits: { max_bots: 5, max_sources_per_bot: 5, max_sources_total: 20, max_active_jobs: 2, max_videos_per_channel: 500 },
  usage: { bots_used: 4, sources_used: 10, active_jobs: 1 },
  toggles: { sharing_enabled: true, byok_enabled: true, managed_enabled: true, upgrades_enabled: false },
  guest_caps: { daily_message_cap: 50, min_interval_sec: 3.0, hourly_limit: 30, distinct_guest_cap: 10, message_max_chars: 2000, history_max_turns: 10, history_max_chars: 8000 },
  budget: { used_usd: 0.0473, limit_usd: 5.0 },
  price: null,
};

export const usage = {
  prompt_tokens: 97400, completion_tokens: 31140, total_tokens: 128540,
  cost_usd: 0.0473, monthly_cost_usd: 0.0473, calls: 46,
  by_model: [
    { model: 'claude-sonnet-4', prompt_tokens: 64000, completion_tokens: 20200, cost_usd: 0.031, calls: 12 },
    { model: 'claude-haiku-3.5', prompt_tokens: 24200, completion_tokens: 6940, cost_usd: 0.009, calls: 26 },
    { model: 'voyage-3-lite', prompt_tokens: 9200, completion_tokens: 4000, cost_usd: 0.007, calls: 8 },
  ],
};
