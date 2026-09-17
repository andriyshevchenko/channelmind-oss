// API facade: picks the adapter by mode. In production the mock module is never imported.
import { MODE } from '../config.js';
const impl = MODE === 'production'
  ? await import('./real.js')
  : await import('../mock/api.js');
export const api = impl.api;
