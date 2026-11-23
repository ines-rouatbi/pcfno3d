// Finitions de l'interface (demande d'Ines, 20/11/2026), sans rien changer aux données ni aux rôles :
//   - apparition des cartes au défilement (IntersectionObserver) ;
//   - chiffres clés de la vue d'ensemble, qui « montent » jusqu'à leur valeur (valeurs réelles du service) ;
//   - carte « Mon espace » du compte connecté (raccourcis de SON rôle, gouvernorat favori) ;
//   - application installable (PWA : manifeste + service worker, fichiers statiques seulement) ;
//   - notifications de l'appareil pour les alertes fortes, si l'utilisateur les active.
import './styles/finitions.css';

const reduit = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = (t: string) => t.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
const A = () => (window as any).ATOM;

// ------------------------------------------------------------------ apparition au défilement
const vues = new IntersectionObserver(entries => entries.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add('is-visible'); vues.unobserve(e.target); }
}), { rootMargin: '0px 0px -40px 0px', threshold: 0.05 });
function reveler() {
  if (reduit()) return;
  document.querySelectorAll<HTMLElement>('section.view.on .card:not(.aw-revele)').forEach((c, i) => {
    c.classList.add('aw-revele'); c.style.transitionDelay = `${Math.min(i, 6) * 45}ms`; vues.observe(c);
  });
}

// ------------------------------------------------------------------ chiffres clés (vue d'ensemble)
function compter(el: HTMLElement, cible: number) {
  const f = (n: number) => n.toLocaleString('fr-FR');
  if (reduit()) { el.textContent = f(cible); return; }
  const debut = performance.now();
  const pas = (t: number) => { const k = Math.min(1, (t - debut) / 900); el.textContent = f(Math.round(cible * (1 - (1 - k) ** 3))); if (k < 1) requestAnimationFrame(pas); };
  requestAnimationFrame(pas);
}
function chiffresCles() {
  const vue = document.getElementById('v-tableau'), a = A();
  if (!vue || !a?.pret?.()) return;
  const national = !!a.domaine?.()?.national;
  const stations = (a.grille?.stations || []).length;
  const jours = 1 + ((a.direct?.previsions || []).length);
  const chiffres: [number, string][] = [
    ...(national ? [[24, 'gouvernorats couverts'] as [number, string]] : []),
    [stations, 'stations de mesure dans le calcul'],
    [jours, jours > 1 ? 'jours de prévision' : 'jour calculé'],
    [1000, 'mètres de hauteur lus au-dessus du sol'],
  ];
  let bloc = document.getElementById('aw-chiffres');
  const place = document.getElementById('aw-chiffres-place');
  if (!bloc) { bloc = document.createElement('div'); bloc.id = 'aw-chiffres'; bloc.className = 'aw-chiffres'; (place ?? vue).append(bloc); }
  else if (place && bloc.parentElement !== place) place.append(bloc);
  const cle = JSON.stringify(chiffres);
  if (bloc.dataset.cle === cle) return;
  bloc.dataset.cle = cle;
  bloc.innerHTML = chiffres.map(([, t]) => `<div><b>0</b><span>${esc(t)}</span></div>`).join('');
  bloc.querySelectorAll('b').forEach((b, i) => compter(b as HTMLElement, chiffres[i][0]));
}

// ------------------------------------------------------------------ « Mon espace » (compte connecté)
const RACCOURCIS: Record<string, [string, string][]> = {
  agri: [['#agri', 'Mes traitements et mon arrosage'], ['#gouvernorats', 'Le vent de mon gouvernorat'], ['#alertes', 'Mes alertes']],
  eolien: [['#eolien', 'La production de mon parc'], ['#carte3d', 'Le vent à 80 m en 3D'], ['#gouvernorats', 'Le vent des gouvernorats']],
  admin: [['#admin', 'Comptes et état du service'], ['#reseau', 'Réseau ANEO'], ['#alertes', 'Alertes du jour']],
  public: [['#gouvernorats', 'Le vent de mon gouvernorat'], ['#monvent', 'Mon vent'], ['#alertes', 'Les alertes']],
};
function monEspace() {
  const vue = document.getElementById('v-tableau'), w = A()?.workspace;
  if (!vue || !w) return;
  const role: string = w.role?.() || 'visiteur';
  let carte = document.getElementById('aw-mon-espace');
  if (role === 'visiteur') { carte?.remove(); return; }
  let fav = 'Tunis'; try { fav = localStorage.getItem('aw-gouvernorat') || fav; } catch { /* préférence */ }
  const titres: Record<string, string> = { agri: 'Votre espace agriculteur', eolien: 'Votre espace exploitant éolien', admin: 'Votre espace administrateur', public: 'Votre espace' };
  const html = `<div class="aw-me-tete"><span class="gv-sur">MON ESPACE</span><b>${esc(titres[role] || titres.public)}</b>`
    + `<span class="muted">Gouvernorat suivi : <a href="#gouvernorats">${esc(fav)}</a></span></div>`
    + `<div class="aw-me-liens">${(RACCOURCIS[role] || RACCOURCIS.public).map(([h, t]) => `<a class="btn" href="${h}">${esc(t)} →</a>`).join('')}</div>`;
  if (!carte) { carte = document.createElement('div'); carte.id = 'aw-mon-espace'; carte.className = 'card aw-mon-espace'; vue.prepend(carte); }
  if (carte.innerHTML !== html) carte.innerHTML = html;
}

// ------------------------------------------------------------------ notifications de l'appareil
const NOTIF_CLE = 'aw-notif-vues';
function notifier(alertes: { titre: string; lieu: string; quand: string; niveau: string }[]) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  let vues: string[] = []; try { vues = JSON.parse(localStorage.getItem(NOTIF_CLE) || '[]'); } catch { /* vide */ }
  const jour = new Date().toISOString().slice(0, 10);
  for (const a of alertes.filter(x => x.niveau === 'stop').slice(0, 3)) {
    const id = `${jour}|${a.titre}|${a.lieu}|${a.quand}`;
    if (vues.includes(id)) continue;
    vues.push(id);
    try { new Notification(`ATOM Wind · ${a.titre}`, { body: `${a.lieu} · ${a.quand}`, icon: `${import.meta.env.BASE_URL}icone-192.png`, tag: id }); } catch { /* navigateur sans notification directe */ }
  }
  try { localStorage.setItem(NOTIF_CLE, JSON.stringify(vues.slice(-60))); } catch { /* préférence */ }
}
function boutonNotifications() {
  const vue = document.getElementById('v-alertes');
  if (!vue || !('Notification' in window) || document.getElementById('aw-notif')) return;
  const b = document.createElement('div'); b.id = 'aw-notif'; b.className = 'card aw-notif';
  const maj = () => {
    const p = Notification.permission;
    b.innerHTML = p === 'granted' ? '<b>Notifications activées sur cet appareil</b><span class="muted">Vous serez prévenu ici des alertes « danger » calculées par le service (une seule fois chacune).</span>'
      : p === 'denied' ? '<b>Notifications bloquées</b><span class="muted">Autorisez-les dans les réglages du navigateur pour être prévenu sur cet appareil.</span>'
      : '<b>Être prévenu sur cet appareil</b><span class="muted">Une notification quand le service calcule une alerte « danger » (vent très fort).</span><button class="btn primary" type="button">Activer les notifications</button>';
    b.querySelector('button')?.addEventListener('click', () => { void Notification.requestPermission().then(maj); });
  };
  maj(); vue.prepend(b);
}

// ------------------------------------------------------------------ application installable (PWA)
let invite: any = null;
function installable() {
  const base = import.meta.env.BASE_URL;
  if (!document.querySelector('link[rel="manifest"]')) {
    const l = document.createElement('link'); l.rel = 'manifest'; l.href = `${base}manifest.webmanifest`; document.head.appendChild(l);
    const i = document.createElement('link'); i.rel = 'apple-touch-icon'; i.href = `${base}icone-192.png`; document.head.appendChild(i);
  }
  if ('serviceWorker' in navigator && location.protocol === 'https:') {
    void navigator.serviceWorker.register(`${base}sw.js`, { scope: base }).catch(() => { /* sans hors-ligne */ });
  }
  window.addEventListener('beforeinstallprompt', (e: Event) => { e.preventDefault(); invite = e; boutonInstaller(); });
  window.addEventListener('appinstalled', () => { invite = null; document.getElementById('aw-installer')?.remove(); });
}
function boutonInstaller() {
  if (!invite || document.getElementById('aw-installer')) return;
  const b = document.createElement('button'); b.id = 'aw-installer'; b.className = 'aw-installer'; b.type = 'button';
  b.innerHTML = '<span>⤓</span> Installer l’application';
  b.onclick = async () => { if (!invite) return; invite.prompt(); await invite.userChoice.catch(() => null); invite = null; b.remove(); };
  document.body.appendChild(b);
}

// ------------------------------------------------------------------ branchement
function surVue() { reveler(); chiffresCles(); monEspace(); boutonNotifications(); }
document.addEventListener('atom:vue', () => requestAnimationFrame(surVue));
window.addEventListener('atom:state', () => { chiffresCles(); monEspace(); });
window.addEventListener('atom:alertes', e => notifier((e as CustomEvent).detail || []));
installable();
