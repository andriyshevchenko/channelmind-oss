// Local no-op stub for @vue/devtools-api.
// vue-router's esm-browser build imports only `setupDevtoolsPlugin` to register a
// Vue Devtools plugin. Devtools integration is a development-only convenience and
// is irrelevant in production, so we ship a no-op instead of vendoring the whole
// @vue/devtools-api package tree (env/const/proxy/api/plugin/time modules). This
// keeps the buildless SPA fully self-hosted with zero third-party CDN requests.
export function setupDevtoolsPlugin() { /* no-op in production */ }
