// Le curseur de l'assistant : il se déplace comme une main humaine, ouvre les pages, désigne les éléments
// et explique chaque étape dans une bulle. Les étapes viennent du guide partagé (assistant-guide.json,
// champ « demo ») : { vue } ouvre un espace, { cible, texte, action } désigne un élément (pointer ou cliquer).
// Échap ou « Arrêter » interrompt la démonstration ; le mouvement réduit est respecté.
import './curseur.css';

export type EtapeDemo = { vue?: string; cible?: string; texte?: string; action?: 'pointer' | 'cliquer' };
export type Demo = { id: string; titre: string; demo: EtapeDemo[] };
type Options = { nomsVues?: Record<string, string>; fin?: (raison: 'fini' | 'arrete' | 'bloque', message?: string) => void };

const attendre = (ms: number) => new Promise(r => setTimeout(r, ms));
const REDUIT = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const visible = (el: Element) => {
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== 'hidden' && !(el as HTMLElement).closest('[hidden]');
};
const premierVisible = (sel: string) => {
  try { return [...document.querySelectorAll(sel)].find(visible) as HTMLElement | undefined; } catch { return undefined; }
};

let enCours: { arreter: () => void } | null = null;
export const demoEnCours = () => !!enCours;
export function arreterDemo() { enCours?.arreter(); }

export function jouerDemo(d: Demo, opts: Options = {}) {
  arreterDemo();
  let stop = false;
  const racine = document.createElement('div');
  racine.className = 'aw-guide'; racine.setAttribute('role', 'status'); racine.setAttribute('aria-live', 'polite');
  racine.dir = document.documentElement.dir || 'ltr';
  racine.innerHTML = `
    <div class="aw-guide-halo" hidden></div>
    <div class="aw-guide-cursor" aria-hidden="true">
      <svg viewBox="0 0 28 32" width="28" height="32"><path d="M3 2l21 13.5-9.3 1.9 5.6 10.6-4.3 2.1-5.5-10.7L3 26z"/></svg>
      <span class="aw-guide-tag">ATOM</span>
      <span class="aw-guide-ripple"></span>
    </div>
    <div class="aw-guide-bubble" hidden><span class="aw-guide-text"></span></div>
    <div class="aw-guide-bar"><b class="aw-guide-title"></b><span class="aw-guide-count"></span>
      <button type="button" class="aw-guide-stop">Arrêter</button></div>`;
  document.body.appendChild(racine);
  const curseur = racine.querySelector<HTMLElement>('.aw-guide-cursor')!;
  const halo = racine.querySelector<HTMLElement>('.aw-guide-halo')!;
  const bulle = racine.querySelector<HTMLElement>('.aw-guide-bubble')!;
  const texte = racine.querySelector<HTMLElement>('.aw-guide-text')!;
  const compte = racine.querySelector<HTMLElement>('.aw-guide-count')!;
  racine.querySelector<HTMLElement>('.aw-guide-title')!.textContent = d.titre;
  const bouton = racine.querySelector<HTMLButtonElement>('.aw-guide-stop')!;

  // le curseur part du bouton de l'assistant (ou du coin bas droit)
  const depart = document.querySelector('.aw-assistant-fab, .assistant-entry')?.getBoundingClientRect();
  let x = depart ? depart.left + depart.width / 2 : innerWidth - 60, y = depart ? depart.top + depart.height / 2 : innerHeight - 60;
  const poser = () => { curseur.style.transform = `translate(${x}px, ${y}px)`; };
  poser();
  requestAnimationFrame(() => racine.classList.add('is-on'));

  const fermer = (raison: 'fini' | 'arrete' | 'bloque', message?: string) => {
    if (!enCours) return;
    stop = true; enCours = null;
    removeEventListener('keydown', clavier, true);
    racine.classList.remove('is-on');
    setTimeout(() => racine.remove(), 260);
    opts.fin?.(raison, message);
  };
  const clavier = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); fermer('arrete'); } };
  addEventListener('keydown', clavier, true);
  bouton.onclick = () => fermer('arrete');
  enCours = { arreter: () => fermer('arrete') };

  // déplacement le long d'une courbe, vitesse humaine : départ et arrivée en douceur
  const aller = (tx: number, ty: number) => new Promise<void>(res => {
    if (REDUIT()) { x = tx; y = ty; poser(); res(); return; }
    const x0 = x, y0 = y, dist = Math.hypot(tx - x0, ty - y0);
    const duree = Math.min(1200, 380 + dist * .75), t0 = performance.now();
    const cx = (x0 + tx) / 2 + (ty - y0) * .18, cy = (y0 + ty) / 2 - (tx - x0) * .18;   // léger arc
    const image = (t: number) => {
      if (stop) { res(); return; }
      const a = Math.min(1, (t - t0) / duree), e = a < .5 ? 4 * a * a * a : 1 - (-2 * a + 2) ** 3 / 2;
      x = (1 - e) ** 2 * x0 + 2 * (1 - e) * e * cx + e * e * tx;
      y = (1 - e) ** 2 * y0 + 2 * (1 - e) * e * cy + e * e * ty;
      poser();
      if (a < 1) requestAnimationFrame(image); else res();
    };
    requestAnimationFrame(image);
  });
  const cliquerAnim = async () => {
    curseur.classList.remove('is-click'); void curseur.offsetWidth; curseur.classList.add('is-click');
    await attendre(REDUIT() ? 60 : 260);
  };
  const montrerBulle = (t: string) => {
    texte.textContent = t; bulle.hidden = false;
    const w = bulle.offsetWidth, h = bulle.offsetHeight, m = 12;
    let bx = x + 34, by = y + 52;      // sous l'étiquette « ATOM », jamais dessus
    if (bx + w > innerWidth - m) bx = x - w - 16;
    if (by + h > innerHeight - 70) by = y - h - 18;
    bulle.style.transform = `translate(${Math.max(m, bx)}px, ${Math.max(m, by)}px)`;
  };
  const entourer = (el: HTMLElement | null) => {
    if (!el) { halo.hidden = true; return; }
    // cadre limité à la partie visible de l'élément (sous la barre du haut, au-dessus de la barre de guidage)
    const r = el.getBoundingClientRect(), p = 6, haut = Math.max(r.top, 76), bas = Math.min(r.bottom, innerHeight - 76);
    halo.hidden = false;
    Object.assign(halo.style, { left: `${r.left - p}px`, top: `${haut - p}px`, width: `${r.width + 2 * p}px`, height: `${Math.max(24, bas - haut) + 2 * p}px` });
  };
  const lecture = (t: string) => Math.min(6500, 1500 + t.length * 42);

  (async () => {
    const etapes = d.demo;
    for (let i = 0; i < etapes.length && !stop; i++) {
      const e = etapes[i];
      compte.textContent = `${i + 1} / ${etapes.length}`;
      if (e.vue) {
        const ws = window.ATOM?.workspace;
        if (ws?.allowed && !ws.allowed(e.vue)) {
          const nom = opts.nomsVues?.[e.vue] ?? e.vue;
          entourer(null); montrerBulle(`L’espace « ${nom} » demande un compte avec le bon rôle : connectez-vous pour le voir.`);
          await attendre(2600); fermer('bloque', `L’espace « ${nom} » demande un compte avec le bon rôle.`); return;
        }
        bulle.hidden = true; entourer(null);          // l'explication précédente ne suit pas sur la page suivante
        const lien = premierVisible(`nav.tabs a[data-v="${e.vue}"]`);
        if (lien) {
          const r = lien.getBoundingClientRect(); bulle.hidden = true; entourer(lien);
          await aller(r.left + Math.min(40, r.width / 2), r.top + r.height / 2); if (stop) return;
          await cliquerAnim();
        }
        ws?.navigate(e.vue);
        scrollTo({ top: 0, behavior: REDUIT() ? 'instant' as ScrollBehavior : 'smooth' });
        entourer(null);
        await attendre(950);
        continue;
      }
      const el = e.cible ? premierVisible(e.cible) : undefined;
      if (el) {
        // un élément plus haut que l'écran : on montre son début, sous la barre du haut
        const grand = el.getBoundingClientRect().height > innerHeight * .55;
        if (grand) scrollTo({ top: scrollY + el.getBoundingClientRect().top - 90, behavior: REDUIT() ? 'instant' as ScrollBehavior : 'smooth' });
        else el.scrollIntoView({ block: 'center', inline: 'nearest', behavior: REDUIT() ? 'instant' as ScrollBehavior : 'smooth' });
        await attendre(REDUIT() ? 80 : 560); if (stop) return;
        const r = el.getBoundingClientRect(), haut = Math.max(r.top, 76);
        bulle.hidden = true; entourer(el);
        await aller(r.left + Math.min(r.width * .5, 160), haut + Math.min((Math.min(r.bottom, innerHeight - 76) - haut) * .5, 120)); if (stop) return;
        if (e.action === 'cliquer') { await cliquerAnim(); el.click(); await attendre(400); entourer(el); }
      } else { entourer(null); }
      if (e.texte) { montrerBulle(e.texte); await attendre(lecture(e.texte)); }
    }
    if (!stop) { entourer(null); bulle.hidden = true; await attendre(250); fermer('fini'); }
  })();
}
