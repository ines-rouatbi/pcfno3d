import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, MapPin, ZoomIn } from 'lucide-react';
import { useEtatPage } from './api';
import { chargerContours, type Contours, type Gouvernorat } from './terrain/gouvernorats';
import { loadGfsForecast, windColor, type NationalForecast } from './terrain/national-forecast';
import './styles/monvent-tunisie.css';

// Carte « Tunisie » de Mon vent : les 24 gouvernorats colorés par le vent à la hauteur de l'activité ; un clic sur
// un gouvernorat l'agrandit (vent au point par couleur, flèches de direction, traînées), un clic dedans y place le
// point de Mon vent. Le vent vient du modèle PC-FNO3D servi dans sa zone (window.ATOM.workspace.terrainWind) ;
// ailleurs, de la prévision GFS (NOAA, via Open-Meteo), toujours nommée comme telle. Rien n'est inventé : sans
// valeur, la case reste grise. Rapide : tracés SVG calculés une fois, champ de couleur recalculé seulement quand
// l'heure, la hauteur ou le cadre changent, animation arrêtée hors écran.

type Cadre = [number, number, number, number];               // x, y, largeur, hauteur (unités SVG)
type Vent = { u: number; v: number; speed: number; modele: boolean };
const S = 100, KX = Math.cos(34 * Math.PI / 180), LON0 = 7.3, LAT1 = 37.7;
const X = (lon: number) => (lon - LON0) * KX * S, Y = (lat: number) => (LAT1 - lat) * S;
const LON = (x: number) => LON0 + x / S / KX, LAT = (y: number) => LAT1 - y / S;
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const versDeg = (u: number, v: number) => (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;   // là où va le vent
const DIRS = ['du nord', 'du nord-est', 'de l’est', 'du sud-est', 'du sud', 'du sud-ouest', 'de l’ouest', 'du nord-ouest'];
const dirNom = (u: number, v: number) => DIRS[Math.round(((versDeg(u, v) + 180) % 360) / 45) % 8];
const GFS = 'prévision GFS (NOAA, via Open-Meteo)';

function dansAnneau(x: number, y: number, r: [number, number][]) {
  let dedans = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    const xi = X(r[i][0]), yi = Y(r[i][1]), xj = X(r[j][0]), yj = Y(r[j][1]);
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) dedans = !dedans;
  }
  return dedans;
}
const dans = (g: Gouvernorat, x: number, y: number) => g.contours.some(r => dansAnneau(x, y, r));
const kmEntre = (la1: number, lo1: number, la2: number, lo2: number) => {
  const r = Math.PI / 180, a = Math.sin((la2 - la1) * r / 2) ** 2 + Math.cos(la1 * r) * Math.cos(la2 * r) * Math.sin((lo2 - lo1) * r / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
};

function boite(g: Gouvernorat | null, tous: Gouvernorat[]): [number, number, number, number] {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const x of g ? [g] : tous) for (const r of x.contours) for (const [lo, la] of r) {
    const px = X(lo), py = Y(la); x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  return [x0, y0, x1, y1];
}
/** Le cadre qui contient la boîte, au rapport largeur / hauteur de la zone d'affichage. */
function cadre([x0, y0, x1, y1]: number[], rapport: number, marge: number): Cadre {
  let w = (x1 - x0) * (1 + 2 * marge), h = (y1 - y0) * (1 + 2 * marge);
  if (w / h < rapport) w = h * rapport; else h = w / rapport;
  return [(x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2, w, h];
}

/** apercu : la version de la Vue d'ensemble (modèle national servi) : vent à 10 m, sans zoom ; un clic ouvre Mon vent
 *  sur le chef-lieu du gouvernorat, carte agrandie sur lui. */
export default memo(function CarteTunisie({ apercu = false }: { apercu?: boolean }) {
  const page = useEtatPage();
  const ouverte = page?.view === (apercu ? 'tableau' : 'monvent');
  const hauteur = !apercu && page?.altitude === 80 ? 80 : 10, heure = page?.hour ?? 15;
  const ms = !!page?.speed?.includes('m/s');
  const fmt = (v: number) => ms ? `${v.toFixed(1).replace('.', ',')} m/s` : `${Math.round(v * 3.6)} km/h`;

  const [geo, setGeo] = useState<Contours | null>(null);
  const [gfs, setGfs] = useState<NationalForecast | null>(null);
  const [gfsErreur, setGfsErreur] = useState(false);
  const [choisi, setChoisi] = useState<string | null>(null);
  const [anime, setAnime] = useState(false);
  const [rapport, setRapport] = useState(0.8);
  const [visible, setVisible] = useState(false);
  const [modelePret, setModelePret] = useState(false);
  const zone = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null);
  const fond = useRef<HTMLCanvasElement>(null), traits = useRef<HTMLCanvasElement>(null), bulle = useRef<HTMLDivElement>(null);
  const vue = useRef<Cadre>([0, 0, 1, 1]);

  useEffect(() => { if (ouverte && !geo) void chargerContours().then(setGeo); }, [ouverte, geo]);
  // en entrant dans Mon vent, la carte montre toute la Tunisie ; on agrandit ensuite un gouvernorat soi-même
  useEffect(() => { if (ouverte && !apercu) setChoisi(null); }, [ouverte, apercu]);
  // GFS seulement si le modèle servi ne couvre pas toute la Tunisie (sinon rien n'est demandé)
  const servie = typeof window !== 'undefined' ? window.ATOM?.workspace?.zoneServie?.() : undefined;
  useEffect(() => {
    if (!ouverte || gfs || gfsErreur || servie?.national) return;
    const ctl = new AbortController();
    loadGfsForecast(ctl.signal).then(setGfs, () => { if (!ctl.signal.aborted) setGfsErreur(true); });
    return () => ctl.abort();
  }, [ouverte, gfs, gfsErreur, servie?.national]);
  useEffect(() => { setModelePret(page?.source === 'modele' || page?.source === 'dernier_etat'); }, [page?.source]);
  useEffect(() => {
    const el = zone.current; if (!el) return;
    const ro = new ResizeObserver(() => { const b = el.getBoundingClientRect(); if (b.width && b.height) setRapport(b.width / b.height); });
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    ro.observe(el); io.observe(el); return () => { ro.disconnect(); io.disconnect(); };
  }, []);

  const indexGfs = useCallback((h: number) => gfs ? Math.min(gfs.data.times.length - 1, Math.floor(gfs.nearestHour / 24) * 24 + h) : -1, [gfs]);
  /** Le vent au point : le modèle dans sa zone, sinon GFS ; null s'il n'y a pas de valeur. */
  const vent = useCallback((lon: number, lat: number): Vent | null => {
    const m = window.ATOM?.workspace?.terrainWind?.(lon, lat, hauteur);
    if (m && Number.isFinite(m.speed)) return { u: m.u, v: m.v, speed: m.speed, modele: true };
    const s = gfs?.sample(lon, lat, hauteur, indexGfs(heure));
    return s ? { u: s.u, v: s.v, speed: s.speed, modele: false } : null;
  }, [gfs, hauteur, heure, indexGfs, modelePret]);   // eslint-disable-line react-hooks/exhaustive-deps

  const chemins = useMemo(() => geo?.gouvernorats.map(g => ({ g, d: g.contours.map(r => r.map(([lo, la], k) => `${k ? 'L' : 'M'}${X(lo).toFixed(2)} ${Y(la).toFixed(2)}`).join('') + 'Z').join('') })) ?? [], [geo]);
  const g = geo?.gouvernorats.find(x => x.nom === choisi) ?? null;
  const cible = useMemo<Cadre>(() => geo ? cadre(boite(g, geo.gouvernorats), rapport, g ? 0.08 : 0.03) : [0, 0, 1, 1], [geo, g, rapport]);
  const chefs = useMemo(() => geo?.gouvernorats.map(x => ({ x, w: vent(x.chef_lieu[1], x.chef_lieu[0]) })) ?? [], [geo, vent]);

  // zoom : le cadre glisse en ~450 ms (sans React à chaque image) ; les calques se redessinent à l'arrivée
  useEffect(() => {
    const el = svg.current; if (!el || !geo) return;
    const de = vue.current, reduit = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (de[2] === 1 || reduit) { vue.current = cible; el.setAttribute('viewBox', cible.join(' ')); setAnime(false); return; }
    const t0 = performance.now(); let id = 0; setAnime(true);
    const pas = (t: number) => { const k = Math.min(1, (t - t0) / 450), e = 1 - (1 - k) ** 3;
      const c = de.map((v, i) => v + (cible[i] - v) * e) as Cadre; vue.current = c; el.setAttribute('viewBox', c.join(' '));
      if (k < 1) id = requestAnimationFrame(pas); else setAnime(false); };
    id = requestAnimationFrame(pas); return () => cancelAnimationFrame(id);
  }, [cible, geo]);

  // champ de couleur du gouvernorat agrandi : une grille de ~70 colonnes, lissée, découpée au contour
  const grille = useMemo(() => {
    if (!g || anime) return null;
    const [vx, vy, vw, vh] = cible, nx = 70, ny = Math.max(2, Math.round(nx * vh / vw)), cel: (Vent | null)[] = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const x = vx + (i + .5) * vw / nx, y = vy + (j + .5) * vh / ny; cel.push(vent(LON(x), LAT(y))); }
    return { nx, ny, cel };
  }, [g, anime, cible, vent]);
  const chemin = useMemo(() => chemins.find(c => c.g === g)?.d ?? null, [chemins, g]);

  useEffect(() => {
    const cv = fond.current, el = zone.current; if (!cv || !el) return;
    const ctx = cv.getContext('2d')!, b = el.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = b.width * dpr; cv.height = b.height * dpr; ctx.clearRect(0, 0, cv.width, cv.height);
    if (!grille || !chemin) return;
    const img = new ImageData(grille.nx, grille.ny);
    grille.cel.forEach((w, k) => { if (!w) return; const c = windColor(w.speed); img.data.set([c[0], c[1], c[2], 255], k * 4); });
    const petit = document.createElement('canvas'); petit.width = grille.nx; petit.height = grille.ny; petit.getContext('2d')!.putImageData(img, 0, 0);
    const [vx, vy, vw] = cible, s = b.width / vw * dpr;
    ctx.save(); ctx.setTransform(s, 0, 0, s, -vx * s, -vy * s); ctx.clip(new Path2D(chemin));
    ctx.imageSmoothingEnabled = true; ctx.drawImage(petit, vx, vy, cible[2], cible[3]); ctx.restore();
  }, [grille, chemin, cible, rapport]);

  // traînées : particules poussées par la même grille, à l'intérieur du gouvernorat
  useEffect(() => {
    const cv = traits.current, el = zone.current;
    const reduit = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!cv || !el) return;
    const ctx = cv.getContext('2d')!, b = el.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = b.width * dpr; cv.height = b.height * dpr; ctx.clearRect(0, 0, cv.width, cv.height);
    if (!grille || !chemin || !ouverte || !visible || reduit) return;
    const [vx, vy, vw] = cible, s = b.width / vw, W = b.width, H = b.height, p2 = new Path2D(chemin);
    ctx.setTransform(s * dpr, 0, 0, s * dpr, -vx * s * dpr, -vy * s * dpr);
    const dedans = (x: number, y: number) => ctx.isPointInPath(p2, x * dpr, y * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const lire = (x: number, y: number) => { const i = Math.floor(x / W * grille.nx), j = Math.floor(y / H * grille.ny);
      return i < 0 || j < 0 || i >= grille.nx || j >= grille.ny ? null : grille.cel[j * grille.nx + i]; };
    const neuf = () => { for (let k = 0; k < 25; k++) { const x = Math.random() * W, y = Math.random() * H;
      ctx.setTransform(s * dpr, 0, 0, s * dpr, -vx * s * dpr, -vy * s * dpr); const ok = dedans(x, y); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (ok) return { x, y, age: Math.random() * 50 }; } return { x: -1, y: -1, age: 99 }; };
    const P = Array.from({ length: 450 }, neuf); let id = 0;
    ctx.save(); ctx.setTransform(s * dpr, 0, 0, s * dpr, -vx * s * dpr, -vy * s * dpr); ctx.clip(p2); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const image = () => {
      ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,.08)'; ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'source-over'; ctx.lineWidth = 1.2;
      for (let k = 0; k < P.length; k++) {
        const p = P[k], w = p.age < 80 ? lire(p.x, p.y) : null;
        if (!w) { P[k] = neuf(); continue; }
        const x2 = p.x + w.u * 0.45, y2 = p.y - w.v * 0.45;
        ctx.strokeStyle = `rgba(255,255,255,${Math.min(.9, .3 + w.speed / 16)})`;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(x2, y2); ctx.stroke(); p.x = x2; p.y = y2; p.age++;
      }
      id = requestAnimationFrame(image);
    };
    id = requestAnimationFrame(image);
    return () => { cancelAnimationFrame(id); ctx.restore(); };
  }, [grille, chemin, cible, ouverte, visible, rapport]);

  // flèches de direction sur le gouvernorat agrandi (~9 par largeur), à l'intérieur du contour
  const fleches = useMemo(() => {
    if (!g || anime) return [];
    const [vx, vy, vw, vh] = cible, pas = vw / 9, out: { x: number; y: number; a: number; c: string }[] = [];
    for (let y = vy + pas / 2; y < vy + vh; y += pas) for (let x = vx + pas / 2; x < vx + vw; x += pas) {
      if (!dans(g, x, y)) continue; const w = vent(LON(x), LAT(y)); if (!w) continue;
      out.push({ x, y, a: versDeg(w.u, w.v), c: rgb(windColor(w.speed)) });
    }
    return out;
  }, [g, anime, cible, vent]);

  const point = (e: { clientX: number; clientY: number }) => {
    const m = svg.current?.getScreenCTM(); if (!m) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse()); return [p.x, p.y] as [number, number];
  };
  const trouver = (x: number, y: number) => geo?.gouvernorats.find(o => dans(o, x, y)) ?? null;
  /** Place le point de Mon vent : le modèle s'il couvre le point, sinon la prévision GFS (nommée comme telle). */
  const placer = (lon: number, lat: number, gv: Gouvernorat) => {
    const ws = window.ATOM?.workspace; if (!ws) return;
    if (ws.placer?.(lat, lon) != null) return;
    if (!gfs) return;
    const d = kmEntre(lat, lon, gv.chef_lieu[0], gv.chef_lieu[1]);
    const nom = d < 3 ? gv.nom : `${gv.nom} · à ${Math.round(d)} km du chef-lieu`;
    const index = indexGfs;
    ws.placerExterne?.(lat, lon, nom, GFS, (z, h) => { const s = gfs.sample(lon, lat, z === 80 ? 80 : 10, index(h)); return s ? [s.u, s.v] : null; });
  };
  /** Vue d'ensemble -> Mon vent : point au chef-lieu, carte Tunisie agrandie sur le gouvernorat. */
  const ouvrirMonVent = (o: Gouvernorat) => {
    window.dispatchEvent(new CustomEvent('atom:zoom-gouvernorat', { detail: { nom: o.nom, placer: false } }));
    document.getElementById('bm-tunisie')?.click();
    placer(o.chef_lieu[1], o.chef_lieu[0], o);            // modèle s'il couvre le chef-lieu, sinon GFS (nommée)
    const ws = window.ATOM?.workspace;
    if (ws && ws.snapshot().view !== 'monvent') ws.navigate('monvent');
  };
  // ouvert depuis Gouvernorats ou la Vue d'ensemble : carte agrandie sur le gouvernorat ; avec placer, le point va au
  // chef-lieu s'il était ailleurs (hors du modèle, dès que la prévision GFS est chargée)
  const enAttente = useRef<string | null>(null);
  const pinDans = (o: Gouvernorat) => !!page?.pin && dans(o, X(page.pin.lon), Y(page.pin.lat));
  useEffect(() => { if (apercu) return; const f = (e: Event) => { const d = (e as CustomEvent).detail, nom = typeof d === 'string' ? d : d?.nom;
      if (!nom) return; setChoisi(nom); if (d?.placer) enAttente.current = nom; };
    addEventListener('atom:zoom-gouvernorat', f); return () => removeEventListener('atom:zoom-gouvernorat', f); }, [apercu]);
  useEffect(() => { const nom = enAttente.current, o = geo?.gouvernorats.find(x => x.nom === nom); if (!o) return;
    if (pinDans(o)) { enAttente.current = null; return; }
    const ws = window.ATOM?.workspace, dedans = ws?.zoneServie?.(), [la, lo] = o.chef_lieu;
    const couvert = !!dedans && la >= dedans.lat0 && la <= dedans.lat1 && lo >= dedans.lon0 && lo <= dedans.lon1;
    if (!couvert && !gfs) return;                        // attend la prévision GFS
    enAttente.current = null; placer(lo, la, o);
  });   // eslint-disable-line react-hooks/exhaustive-deps
  // bouton « Gouvernorats » : la page Gouvernorats s'ouvre sur le gouvernorat agrandi, sinon celui du point
  useEffect(() => { if (apercu) return; const f = () => { const nom = choisi ?? geo?.gouvernorats.find(o => pinDans(o))?.nom;
      if (nom) window.dispatchEvent(new CustomEvent('atom:gouvernorat-choisi', { detail: nom })); };
    addEventListener('atom:bascule-gouvernorats', f); return () => removeEventListener('atom:bascule-gouvernorats', f); });   // eslint-disable-line react-hooks/exhaustive-deps
  const clic = (e: React.MouseEvent) => {
    const p = point(e); if (!p) return; const o = trouver(p[0], p[1]); if (!o) return;
    if (apercu) { ouvrirMonVent(o); return; }
    if (!g || o.nom !== g.nom) { setChoisi(o.nom); return; }
    placer(LON(p[0]), LAT(p[1]), o);
  };
  const survol = (e: React.MouseEvent) => {
    const el = bulle.current, b = zone.current?.getBoundingClientRect(), p = point(e); if (!el || !b || !p) return;
    const o = trouver(p[0], p[1]); if (!o) { el.hidden = true; return; }
    const w = g && o.nom === g.nom ? vent(LON(p[0]), LAT(p[1])) : chefs.find(c => c.x === o)?.w ?? null;
    el.hidden = false; el.style.left = `${e.clientX - b.left}px`; el.style.top = `${e.clientY - b.top}px`;
    el.innerHTML = `<b>${o.nom}</b><span>${w ? `${fmt(w.speed)} · ${dirNom(w.u, w.v)}` : 'valeur indisponible'}</span>${g && o.nom === g.nom ? '<small>Touchez pour placer votre point</small>' : apercu ? '<small>Touchez pour l’ouvrir dans Mon vent</small>' : '<small>Touchez pour agrandir</small>'}`;
  };
  // gestes : écarter deux doigts (ou pincer sur le pavé tactile, Ctrl + molette) agrandit le gouvernorat sous les doigts ;
  // les rapprocher revient à toute la Tunisie. Un doigt fait toujours défiler la page (touch-action: pan-x pan-y).
  const doigts = useRef(new Map<number, [number, number]>()), geste = useRef<{ d0: number; fait: boolean } | null>(null);
  const zoomer = (entrer: boolean, cx: number, cy: number) => {
    if (apercu) return;
    if (!entrer) { if (choisi) setChoisi(null); return; }
    if (choisi) return;
    const p = point({ clientX: cx, clientY: cy }), o = p && trouver(p[0], p[1]); if (o) setChoisi(o.nom);
  };
  const ecart = () => { const [a, b] = [...doigts.current.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]); };
  const pDown = (e: React.PointerEvent) => { if (e.pointerType !== 'touch') return; doigts.current.set(e.pointerId, [e.clientX, e.clientY]);
    if (doigts.current.size === 2) geste.current = { d0: ecart(), fait: false }; };
  const pMove = (e: React.PointerEvent) => { if (!doigts.current.has(e.pointerId)) return; doigts.current.set(e.pointerId, [e.clientX, e.clientY]);
    const g0 = geste.current; if (!g0 || g0.fait || doigts.current.size !== 2) return;
    const k = ecart() / Math.max(1, g0.d0), [a, b] = [...doigts.current.values()];
    if (k < 0.75 || k > 1.3) { g0.fait = true; zoomer(k > 1, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); } };
  const pUp = (e: React.PointerEvent) => { doigts.current.delete(e.pointerId); if (doigts.current.size < 2) geste.current = null; };
  useEffect(() => {
    const el = zone.current; if (!el || apercu) return; let cumul = 0, t = 0;
    const roue = (e: WheelEvent) => { if (!e.ctrlKey) return;                       // pincement du pavé tactile / Ctrl + molette
      e.preventDefault(); if (performance.now() - t > 400) cumul = 0; t = performance.now(); cumul += e.deltaY;
      if (Math.abs(cumul) > 30) { zoomer(cumul < 0, e.clientX, e.clientY); cumul = 0; t = performance.now() + 500; } };
    el.addEventListener('wheel', roue, { passive: false }); return () => el.removeEventListener('wheel', roue);
  });
  useEffect(() => { const f = (e: KeyboardEvent) => { if (e.key === 'Escape' && choisi && ouverte) setChoisi(null); }; addEventListener('keydown', f); return () => removeEventListener('keydown', f); }, [choisi, ouverte]);

  const u = cible[2] / 100;                 // 1 % de la largeur du cadre, en unités SVG
  const pin = page?.pin, wChoisi = g ? chefs.find(c => c.x === g)?.w ?? null : null;
  const horsModele = chefs.some(c => c.w && !c.w.modele) || (!!g && fleches.length > 0 && !(wChoisi?.modele));
  const source = servie?.national ? 'ATOM Wind (toute la Tunisie)'
    : `ATOM Wind dans sa zone (${servie?.nom ?? 'Grand Tunis'}) · ailleurs : ${GFS}`;
  const z = servie && !servie.national ? [X(servie.lon0), Y(servie.lat1), X(servie.lon1) - X(servie.lon0), Y(servie.lat0) - Y(servie.lat1)] : null;

  return <div className={`tn-carte${apercu ? ' tn-apercu' : ''}`}>
    <div className="tn-tete">
      {g ? <button className="btn tn-retour" onClick={() => setChoisi(null)} title="Ou rapprochez deux doigts sur la carte"><ArrowLeft size={15}/>Toute la Tunisie</button>
        : <span className="tn-sur"><ZoomIn size={14}/>{apercu ? 'Touchez un gouvernorat pour l’ouvrir dans Mon vent' : 'Touchez ou écartez deux doigts sur un gouvernorat pour l’agrandir'}</span>}
      {g && <div className="tn-titre"><b>{g.nom}</b>{wChoisi ? <span>{fmt(wChoisi.speed)} · {dirNom(wChoisi.u, wChoisi.v)} au chef-lieu</span> : <span>valeur indisponible</span>}</div>}
      <span className="tn-haut">{hauteur} m · {heure} h</span>
    </div>
    <div className="tn-zone" ref={zone} onMouseLeave={() => { if (bulle.current) bulle.current.hidden = true; }}
      onPointerDown={pDown} onPointerMove={pMove} onPointerUp={pUp} onPointerCancel={pUp}>
      {!geo && <div className="tn-squelette" aria-hidden="true"/>}
      <canvas ref={fond} className="tn-fond" aria-hidden="true"/>
      <svg ref={svg} className={`tn-svg${g ? ' is-zoom' : ''}`} role="img" preserveAspectRatio="xMidYMid meet" onClick={clic} onMouseMove={survol}
        aria-label={g ? `${g.nom} agrandi : touchez la carte pour placer votre point` : 'Carte des 24 gouvernorats colorés par le vent'}>
        {chemins.map(({ g: x, d }) => { const w = chefs.find(c => c.x === x)?.w, sel = g?.nom === x.nom;
          return <path key={x.nom} d={d} vectorEffect="non-scaling-stroke" className={`tn-gouv${sel ? ' is-choisi' : ''}${g && !sel ? ' is-voisin' : ''}`}
            style={{ fill: sel && !anime ? 'transparent' : w ? rgb(windColor(w.speed)) : undefined }}
            tabIndex={0} role="button" aria-label={`${x.nom}${w ? ` : ${fmt(w.speed)}, vent ${dirNom(w.u, w.v)}` : ''}`}
            onKeyDown={e => { if (e.key !== 'Enter' && e.key !== ' ') return; e.preventDefault();
              if (apercu) ouvrirMonVent(x); else if (sel) placer(x.chef_lieu[1], x.chef_lieu[0], x); else setChoisi(x.nom); }}/>; })}
        {z && <rect x={z[0]} y={z[1]} width={z[2]} height={z[3]} className="tn-zone-modele" vectorEffect="non-scaling-stroke"/>}
        {!g && chefs.map(({ x, w }) => w && <g key={x.nom} transform={`translate(${X(x.chef_lieu[1])} ${Y(x.chef_lieu[0])}) rotate(${versDeg(w.u, w.v)}) scale(${u * .2})`} className="tn-mini"><path d="M0 -8 L4 6 L0 3 L-4 6 Z"/></g>)}
        {g && !anime && fleches.map((f, k) => <g key={k} transform={`translate(${f.x} ${f.y}) rotate(${f.a}) scale(${u * .4})`} className="tn-fleche"><path d="M0 -9 L5 7 L0 3.5 L-5 7 Z"/></g>)}
        {g && !anime && <g transform={`translate(${X(g.chef_lieu[1])} ${Y(g.chef_lieu[0])})`} className="tn-chef"><circle r={u * .9}/><text x={u * 1.6} y={u * .7} style={{ fontSize: u * 3.2 }}>{g.nom}</text></g>}
        {pin && <g transform={`translate(${X(pin.lon)} ${Y(pin.lat)})`} className="tn-pin"><circle r={u * 1.1}/><circle r={u * .45} className="tn-pin-c"/></g>}
      </svg>
      <canvas ref={traits} className="tn-traits" aria-hidden="true"/>
      <div ref={bulle} className="tn-bulle" hidden/>
    </div>
    {!apercu && <div className="tn-pied">
      <span className="tn-legende"><span>0</span><i/><span>{ms ? '17 m/s' : '62 km/h'}+</span></span>
      {g && <button className="btn tn-chef-btn" onClick={() => placer(g.chef_lieu[1], g.chef_lieu[0], g)}><MapPin size={14}/>Placer au chef-lieu</button>}
    </div>}
    <p className="tn-source">Vent : {source}.{horsModele && !gfs && !servie?.national ? (gfsErreur ? ' Prévision GFS indisponible : hors de la zone calculée par ATOM Wind, rien n’est affiché.' : ' Chargement de la prévision GFS…') : ''}{z ? ' Cadre pointillé : zone calculée par ATOM Wind.' : ''} Contours : geoBoundaries / © OpenStreetMap (ODbL).</p>
  </div>;
});
