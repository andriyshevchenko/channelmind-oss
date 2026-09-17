import { ref, provide, onMounted, computed } from 'vue';
import { useRoute } from 'vue-router';
import { api } from './api/index.js';
import { reconnecting } from './api/connection.js';
import AppHeader from './components/AppHeader.js';

export default {
  components: { AppHeader },
  setup() {
    const user = ref(null);
    provide('user', user);
    const route = useRoute();
    const bare = computed(() => !!route.meta.bare);
    onMounted(async () => { try { user.value = await api.me(); } catch (e) { /* redirected to login in production */ } });
    // `reconnecting` (api/connection.js) is flipped by the real adapter while the
    // backend is briefly unreachable during a redeploy. Rendered here — not in
    // AppHeader — so the pill also covers the bare routes (login / guest chat) that
    // skip the header. It's position:fixed so appearing/disappearing never reflows.
    return { user, bare, reconnecting };
  },
  template: `
    <transition name="fade">
      <div v-if="reconnecting" class="reconnect-pill" data-testid="reconnect-pill" role="status" aria-live="polite">
        <span class="badge-dot"></span>Reconnecting…
      </div>
    </transition>
    <app-header v-if="!bare" :user="user"></app-header>
    <router-view></router-view>
  `,
};
