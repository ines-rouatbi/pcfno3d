// Service worker d'ATOM Wind (application installable). Il ne garde QUE les fichiers statiques de l'application
// (scripts, styles, polices, icônes, traductions) pour qu'elle s'ouvre vite et hors ligne. Jamais l'API (/api/ :
// vent, comptes, sessions), jamais une autre origine, jamais le relief 3D (trop lourd) : les données restent
// celles du service, et le « dernier état connu » reste géré par la page elle-même.
const CACHE = 'atom-wind-statique-v1';
const STATIQUE = /^\/app\/(assets|vendor|atom-wind\/i18n)\/|^\/app\/(icone-[\w-]+\.png|manifest\.webmanifest)$|\.(woff2?|css|js)$/;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil((async () => {
  for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
  await self.clients.claim();
})()));

self.addEventListener('fetch', e => {
  const r = e.request, u = new URL(r.url);
  if (r.method !== 'GET' || u.origin !== location.origin || u.pathname.startsWith('/api/') || u.pathname.startsWith('/mcp')
      || u.pathname.startsWith('/app/terrain/')) return;
  // page de l'application : le réseau d'abord (toujours la dernière version), la copie si hors ligne
  if (r.mode === 'navigate') {
    e.respondWith(fetch(r).then(rep => { const c = rep.clone(); caches.open(CACHE).then(k => k.put(r, c)); return rep; })
      .catch(() => caches.match(r).then(m => m || caches.match('/app/atom-wind/'))));
    return;
  }
  if (!STATIQUE.test(u.pathname)) return;
  // fichiers statiques : la copie d'abord, rafraîchie en arrière-plan
  e.respondWith(caches.open(CACHE).then(async k => {
    const m = await k.match(r);
    const reseau = fetch(r).then(rep => { if (rep.ok) k.put(r, rep.clone()); return rep; }).catch(() => m);
    return m || reseau;
  }));
});
