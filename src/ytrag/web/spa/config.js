// Runtime mode. "preview" = mock data, no auth (for design review);
// "production" = real API + Google login. Set via <meta name="cm-mode"> in index.html.
const meta = document.querySelector('meta[name="cm-mode"]');
export const MODE = meta && meta.content === 'production' ? 'production' : 'preview';
export const IS_PREVIEW = MODE === 'preview';

// No-login self-host mode. Set by the server (<meta name="cm-dev-login">) when
// YTRAG_DEV_LOGIN is enabled. When true, the SPA transparently establishes the
// local dev session on a 401 (via /auth/dev) so a fresh visitor gets in with no
// Google login. Empty/absent (a real OAuth deployment) leaves the Google flow intact.
const devMeta = document.querySelector('meta[name="cm-dev-login"]');
export const DEV_LOGIN = !!(devMeta && devMeta.content === '1');
