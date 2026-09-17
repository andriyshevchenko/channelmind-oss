import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { api } from '../api/index.js';
import { theme, toggleTheme } from '../theme.js';

export default {
  props: { user: Object },
  setup(props) {
    const menuOpen = ref(false);
    const logout = () => { menuOpen.value = false; api.logout(); };
    // Global in-flight work indicator (ingestion-UX brief §3): poll the ingest
    // jobs list across all bots and summarize running/queued into a pill.
    // Hidden entirely at rest. Clicking goes to My Bots, where per-card badges
    // point at the busy bot.
    const jobs = ref([]);
    let pollT = null;
    const poll = async () => {
      if (!props.user) { jobs.value = []; return; }
      try { jobs.value = await api.ingestJobs(); } catch (e) { /* non-fatal */ }
    };
    onMounted(() => { poll(); pollT = setInterval(poll, 4000); });
    onBeforeUnmount(() => clearInterval(pollT));
    const activity = computed(() => {
      const running = jobs.value.filter(j => j.status === 'running').length;
      const queued = jobs.value.filter(j => j.status === 'queued').length;
      if (!running && !queued) return null;
      const count = running + queued;
      if (running && queued) return { count, label: `${running} running · ${queued} queued` };
      if (running) return { count, label: running === 1 ? '1 import running' : running + ' imports running' };
      return { count, label: queued === 1 ? '1 import queued' : queued + ' imports queued' };
    });
    return { menuOpen, logout, theme, toggleTheme, activity };
  },
  template: `
  <header class="app-header">
    <div class="container bar">
      <div style="display: flex; align-items: center; gap: 24px;">
        <router-link to="/" class="logo">
          <div class="logo-mark">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"></path><rect x="4" y="6" width="16" height="12" rx="3"></rect><circle cx="9" cy="12" r="1" fill="#fff" stroke="none"></circle><circle cx="15" cy="12" r="1" fill="#fff" stroke="none"></circle></svg>
          </div>
          <span class="logo-name">Channelmind</span>
        </router-link>
        <nav class="nav">
          <router-link to="/" data-testid="nav-mybots" exact-active-class="active">My Bots</router-link>
          <router-link to="/settings" data-testid="nav-settings" active-class="active">Settings</router-link>
        </nav>
      </div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <router-link v-if="user && activity" to="/" class="hdr-activity" title="Background imports — click to see which bots are busy">
          <span class="badge-dot"></span><span class="hdr-activity-lbl">{{ activity.label }}</span><span class="hdr-activity-count">{{ activity.count }}</span>
        </router-link>
        <button class="icon-btn" style="padding: 7px;" :title="theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'" @click="toggleTheme">
          <svg v-if="theme === 'dark'" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path></svg>
          <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        </button>
        <div style="position: relative;" v-if="user">
        <button class="user-btn" data-testid="user-menu-btn" @click="menuOpen = !menuOpen">
          <div class="avatar">{{ user.name ? user.name[0] : '?' }}</div>
          <div class="hide-sm" style="display: flex; flex-direction: column; align-items: flex-start;">
            <span style="font-size: 12.5px; font-weight: 600; line-height: 1.25;">{{ user.name }}</span>
            <span style="font-size: 11px; color: var(--text-faint); line-height: 1.25;">{{ user.email }}</span>
          </div>
        </button>
        <div class="menu" v-if="menuOpen" @click="menuOpen = false">
          <router-link to="/account" data-testid="nav-account">Account</router-link>
          <a href="#" @click.prevent="logout" data-testid="nav-logout">Log out</a>
        </div>
        </div>
      </div>
    </div>
  </header>
  `,
};
