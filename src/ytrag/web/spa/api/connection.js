// Shared, mode-independent connection state for the whole SPA.
//
// real.js flips `reconnecting` ON while the backend is briefly unreachable — a
// redeploy window where fetch() throws (upstream down) or Caddy answers 502/503/504
// from the dead app — and OFF again on the first request the app actually answers.
// The global chrome (app.js) renders a small non-blocking "Reconnecting…" pill off
// this flag.
//
// It lives in its own tiny module (not inside real.js) for two reasons:
//   1. Every in-flight request and every polling loop shares ONE indicator through
//      this single ref, instead of each surfacing its own error.
//   2. app.js can import it without pulling in the production adapter — the mock
//      adapter never touches it, so preview mode never shows the pill.
import { ref } from 'vue';

export const reconnecting = ref(false);
