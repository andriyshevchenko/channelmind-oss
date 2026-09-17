import { ref } from 'vue';
const KEY = 'cm-theme';
export const theme = ref(localStorage.getItem(KEY) || 'dark');
export function applyTheme() { document.documentElement.setAttribute('data-theme', theme.value); }
export function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark';
  localStorage.setItem(KEY, theme.value);
  applyTheme();
}
applyTheme();
