import { createRouter, createWebHashHistory } from 'vue-router';
import MyBots from './views/MyBots.js';
import BotDetail from './views/BotDetail.js';
import Settings from './views/Settings.js';
import Account from './views/Account.js';
import Login from './views/Login.js';
import PublicChat from './views/PublicChat.js';
import Landing from './views/Landing.js';
import LegalDoc from './views/LegalDoc.js';
import ErrorPage from './views/ErrorPage.js';

// meta.bare pages render without the app header (public / unauthenticated pages)
export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', component: MyBots },
    { path: '/bots/:id', component: BotDetail },
    { path: '/settings', component: Settings },
    { path: '/account', component: Account },
    { path: '/login', component: Login, meta: { bare: true } },
    { path: '/about', component: Landing, meta: { bare: true } },
    { path: '/s/:token', component: PublicChat, meta: { bare: true } },
    { path: '/terms', component: LegalDoc, props: { kind: 'terms' }, meta: { bare: true } },
    { path: '/privacy', component: LegalDoc, props: { kind: 'privacy' }, meta: { bare: true } },
    { path: '/error/:variant', component: ErrorPage, props: r => ({ variant: r.params.variant }), meta: { bare: true } },
    { path: '/:pathMatch(.*)*', component: ErrorPage, props: { variant: '404' }, meta: { bare: true } },
  ],
});
