// SpeakUp : réseau d'abord (toujours la dernière version), copie locale si pas d'internet
const CACHE = 'speakup-v7';
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then(c => c.addAll(['./', './apple-touch-icon.png', './manifest.webmanifest']))); });
// nouvelle version : on efface les anciennes copies gardées sur le téléphone
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const r = e.request;
  if (r.method !== 'GET' || new URL(r.url).origin !== location.origin) return;
  e.respondWith(fetch(r, r.mode === 'navigate' ? { cache: 'no-store' } : undefined).then(res => { const copy = res.clone(); caches.open(CACHE).then(c => c.put(r, copy)); return res; })
    .catch(() => caches.match(r).then(m => m || caches.match('./'))));
});
