import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Leaf, MapPinned, Navigation2, Wind, Zap } from 'lucide-react';
import { useEtatPage } from '../api';
import { chargerContours, type Contours, type Gouvernorat } from '../terrain/gouvernorats';
import { loadNationalForecast, MODEL_SOURCE, windColor, type NationalForecast } from '../terrain/national-forecast';
import '../styles/gouvernorats.css';

// Les 24 gouvernorats : carte interactive colorée par le vent, et la page de chaque gouvernorat (vent de l'heure,
// heure par heure, jours suivants, rose des vents, résumé selon le rôle). Les valeurs viennent du modèle national
// PC-FNO3D quand le service le sert, sinon de la prévision GFS (NOAA, via Open-Meteo) : la source est toujours
// affichée, et rien n'est inventé quand elle manque. Valeurs lues au chef-lieu de chaque gouvernorat.

type Echantillon = { u: number; v: number; speed: number; direction: string; formatted: string };
const W = 520, H = 760, LON0 = 7.35, LON1 = 11.75, LAT0 = 30.1, LAT1 = 37.6;
const KX = Math.cos(34 * Math.PI / 180);
const ECH = Math.min(W / ((LON1 - LON0) * KX), H / (LAT1 - LAT0));
const px = (lon: number, lat: number): [number, number] => [(lon - LON0) * KX * ECH, (LAT1 - lat) * ECH];
const deg = (lonlat: [number, number]) => [LON0 + lonlat[0] / ECH / KX, LAT1 - lonlat[1] / ECH] as [number, number];
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const versDeg = (s: Echantillon) => (Math.atan2(s.u, s.v) * 180 / Math.PI + 360) % 360;      // là où va le vent
const JOURS = new Intl.DateTimeFormat('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', timeZone: 'Africa/Tunis' });
const HEURE = new Intl.DateTimeFormat('en-GB', { hour: 'numeric', hourCycle: 'h23', timeZone: 'Africa/Tunis' });

function useMouvementReduit() {
  const [r, setR] = useState(() => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => { const m = matchMedia('(prefers-reduced-motion: reduce)'); const f = () => setR(m.matches); m.addEventListener('change', f); return () => m.removeEventListener('change', f); }, []);
  return r;
}

/** Un nombre qui « monte » jusqu'à sa valeur (sauf mouvement réduit). */
function Compteur({ valeur, unite }: { valeur: number; unite: string }) {
  const reduit = useMouvementReduit();
  const [affiche, setAffiche] = useState(valeur);
  const prec = useRef(valeur);
  useEffect(() => {
    if (reduit) { setAffiche(valeur); prec.current = valeur; return; }
    const debut = performance.now(), de = prec.current; let id = 0;
    const pas = (t: number) => { const k = Math.min(1, (t - debut) / 650), e = 1 - (1 - k) ** 3;
      setAffiche(de + (valeur - de) * e); if (k < 1) id = requestAnimationFrame(pas); else prec.current = valeur; };
    id = requestAnimationFrame(pas); return () => cancelAnimationFrame(id);
  }, [valeur, reduit]);
  return <span className="gv-compteur">{unite === 'm/s' ? affiche.toFixed(1).replace('.', ',') : Math.round(affiche)}<small>{unite}</small></span>;
}

function Fleche({ angle, taille = 18, couleur = 'currentColor' }: { angle: number; taille?: number; couleur?: string }) {
  return <svg width={taille} height={taille} viewBox="0 0 24 24" aria-hidden="true" className="gv-fleche" style={{ transform: `rotate(${angle}deg)` }}>
    <path d="M12 2 L18 20 L12 16 L6 20 Z" fill={couleur}/></svg>;
}

function Boussole({ s }: { s: Echantillon | null }) {
  const a = s ? versDeg(s) : 0;
  return <div className="gv-boussole" aria-label={s ? `Vent ${s.direction}` : 'Direction indisponible'}>
    <svg viewBox="0 0 120 120" aria-hidden="true">
      <circle cx="60" cy="60" r="54" className="gv-b-cercle"/>
      {Array.from({ length: 36 }, (_, i) => <line key={i} x1="60" y1="8" x2="60" y2={i % 9 === 0 ? 18 : 13} transform={`rotate(${i * 10} 60 60)`} className="gv-b-trait"/>)}
      {['N', 'E', 'S', 'O'].map((t, i) => <text key={t} x={60 + 40 * Math.sin(i * Math.PI / 2)} y={64 - 40 * Math.cos(i * Math.PI / 2)} textAnchor="middle" className="gv-b-lettre">{t}</text>)}
      <g className="gv-b-aiguille" style={{ transform: `rotate(${a}deg)` }}><path d="M60 22 L68 66 L60 58 L52 66 Z"/><circle cx="60" cy="60" r="4"/></g>
    </svg>
  </div>;
}

function Rose({ ech }: { ech: (Echantillon | null)[] }) {
  // fréquence pondérée par la vitesse, 8 secteurs de PROVENANCE
  const sect = Array(8).fill(0); let tot = 0;
  for (const s of ech) { if (!s) continue; const from = (versDeg(s) + 180) % 360; sect[Math.round(from / 45) % 8] += s.speed; tot += s.speed; }
  const mx = Math.max(...sect, 1e-9), noms = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
  return <svg viewBox="0 0 200 200" className="gv-rose" role="img" aria-label="Rose des vents de la journée">
    {[30, 55, 80].map(r => <circle key={r} cx="100" cy="100" r={r} className="gv-rose-cercle"/>)}
    {sect.map((v, i) => { const r = 14 + 66 * v / mx, a0 = (i * 45 - 20) * Math.PI / 180, a1 = (i * 45 + 20) * Math.PI / 180;
      return <path key={i} className="gv-rose-petale" style={{ animationDelay: `${i * 50}ms` }}
        d={`M100 100 L${100 + r * Math.sin(a0)} ${100 - r * Math.cos(a0)} A${r} ${r} 0 0 1 ${100 + r * Math.sin(a1)} ${100 - r * Math.cos(a1)} Z`}><title>{`${noms[i]} : ${tot ? Math.round(100 * v / tot) : 0} %`}</title></path>; })}
    {noms.map((t, i) => <text key={t} x={100 + 92 * Math.sin(i * Math.PI / 4)} y={104 - 92 * Math.cos(i * Math.PI / 4)} textAnchor="middle" className="gv-rose-lettre">{t}</text>)}
  </svg>;
}

/** Traînées de vent sur la carte : particules poussées par le champ de l'heure (le même que les couleurs). */
function Particules({ champ, chemin, actif }: { champ: ((lon: number, lat: number) => [number, number] | null) | null; chemin: Path2D | null; actif: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduit = useMouvementReduit();
  useEffect(() => {
    const cv = ref.current; if (!cv || !champ || !chemin || !actif || reduit) { cv?.getContext('2d')?.clearRect(0, 0, W, H); return; }
    const ctx = cv.getContext('2d')!; const dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // champ précalculé sur une grille de 0,1° : l'animation ne relit pas le modèle à chaque image
    const nx = 45, ny = 76, grille: ([number, number] | null)[] = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) grille.push(champ(LON0 + i * 0.1, LAT1 - j * 0.1));
    const vent = (x: number, y: number) => { const [lon, lat] = deg([x, y]); const i = Math.round((lon - LON0) / 0.1), j = Math.round((LAT1 - lat) / 0.1);
      return i < 0 || j < 0 || i >= nx || j >= ny ? null : grille[j * nx + i]; };
    const neuf = () => { for (let k = 0; k < 30; k++) { const x = Math.random() * W, y = Math.random() * H;
      if (ctx.isPointInPath(chemin, x * dpr, y * dpr)) return { x, y, age: Math.random() * 60 }; } return { x: W / 2, y: H / 3, age: 0 }; };
    const P = Array.from({ length: 700 }, neuf); let id = 0;
    const image = () => {
      ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,.09)'; ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'source-over'; ctx.lineWidth = 1.1;
      for (let k = 0; k < P.length; k++) {
        const p = P[k], w = vent(p.x, p.y);
        if (!w || p.age > 90) { P[k] = neuf(); continue; }
        const nx2 = p.x + w[0] * 0.55, ny2 = p.y - w[1] * 0.55, sp = Math.hypot(w[0], w[1]);
        ctx.strokeStyle = `rgba(255,255,255,${Math.min(.85, .25 + sp / 18)})`;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx2, ny2); ctx.stroke();
        p.x = nx2; p.y = ny2; p.age++;
      }
      id = requestAnimationFrame(image);
    };
    id = requestAnimationFrame(image);
    return () => cancelAnimationFrame(id);
  }, [champ, chemin, actif, reduit]);
  return <canvas ref={ref} className="gv-particules" aria-hidden="true"/>;
}

/** Le contenu de la page Gouvernorats (carte + fiche du gouvernorat) ; `vue` : l'espace où il est affiché
 *  (gouvernorats, ou tableau pour la Vue d'ensemble) — il ne charge et n'anime rien ailleurs. */
export function ContenuGouvernorats({ vue = 'gouvernorats' }: { vue?: string }) {
  const page = useEtatPage();
  const ouverte = page?.view === vue;
  const [fc, setFc] = useState<NationalForecast | null>(null);
  const [erreur, setErreur] = useState('');
  const [geo, setGeo] = useState<Contours | null>(null);
  const [nom, setNom] = useState(() => { try { return localStorage.getItem('aw-gouvernorat') || 'Tunis'; } catch { return 'Tunis'; } });
  const [index, setIndex] = useState(0);
  const [survol, setSurvol] = useState<{ g: Gouvernorat; x: number; y: number } | null>(null);
  const role = (typeof window !== 'undefined' && window.ATOM?.workspace?.role?.()) || 'visiteur';
  const ms = !!page?.speed?.includes('m/s');
  const unite = ms ? 'm/s' : 'km/h', val = (v: number) => ms ? v : v * 3.6;
  const fmt = (v: number) => ms ? `${v.toFixed(1).replace('.', ',')} m/s` : `${Math.round(v * 3.6)} km/h`;

  useEffect(() => { if (ouverte && !geo) void chargerContours().then(setGeo); }, [ouverte, geo]);
  useEffect(() => {
    if (!ouverte || fc) return;
    const ctl = new AbortController();
    loadNationalForecast(ctl.signal).then(f => { setFc(f); setIndex(f.nearestHour); }, e => { if (!ctl.signal.aborted) setErreur(String(e?.message || e)); });
    return () => ctl.abort();
  }, [ouverte, fc]);
  // curseur temporel commun (TimeScrubber) : il diffuse l'heure choisie à toutes les vues
  useEffect(() => { const f = (e: Event) => { const d = (e as CustomEvent).detail; if (fc && typeof d?.heure === 'number') setIndex(i => Math.floor(i / 24) * 24 + d.heure); };
    window.addEventListener('atom:heure', f); return () => window.removeEventListener('atom:heure', f); }, [fc]);
  // deux copies (page Gouvernorats, Vue d'ensemble) : seule la copie affichée enregistre son gouvernorat, et une
  // copie qui s'affiche reprend le dernier choisi ailleurs — le bouton « Mon vent » lit ce choix
  useEffect(() => { if (!ouverte) return; try { const n = localStorage.getItem('aw-gouvernorat'); if (n) setNom(n); } catch { /* préférence seulement */ } }, [ouverte]);
  useEffect(() => { if (!ouverte) return; try { localStorage.setItem('aw-gouvernorat', nom); } catch { /* préférence seulement */ } }, [nom, ouverte]);
  // bouton « Gouvernorats » depuis Mon vent : le gouvernorat agrandi là-bas (monvent-tunisie.tsx)
  useEffect(() => { const f = (e: Event) => { const n = (e as CustomEvent).detail; if (typeof n === 'string' && n) setNom(n); };
    window.addEventListener('atom:gouvernorat-choisi', f); return () => window.removeEventListener('atom:gouvernorat-choisi', f); }, []);

  const modele = fc?.data.source === MODEL_SOURCE;
  const nJours = fc ? Math.max(1, Math.floor(fc.data.times.length / 24)) : 0;
  const jour = Math.floor(index / 24), heure = index % 24;
  const lire = (g: { chef_lieu: [number, number] } | undefined, i: number, h = 10) => g && fc ? fc.sample(g.chef_lieu[1], g.chef_lieu[0], h, i) as Echantillon | null : null;
  const g = geo?.gouvernorats.find(x => x.nom === nom) ?? geo?.gouvernorats[0];
  const now = lire(g, index);
  const heures = useMemo(() => Array.from({ length: 24 }, (_, h) => lire(g, jour * 24 + h)), [fc, g, jour]);   // eslint-disable-line react-hooks/exhaustive-deps
  const chemins = useMemo(() => geo?.gouvernorats.map(x => ({ g: x, d: x.contours.map(r => r.map((p, k) => `${k ? 'L' : 'M'}${px(p[0], p[1]).map(n => n.toFixed(1)).join(' ')}`).join(' ') + 'Z').join(' ') })) ?? [], [geo]);
  const tunisie = useMemo(() => { if (!chemins.length) return null; const p = new Path2D(); chemins.forEach(c => p.addPath(new Path2D(c.d))); return p; }, [chemins]);
  const champ = useMemo(() => fc ? (lon: number, lat: number) => { const s = fc.sample(lon, lat, 10, index) as Echantillon | null; return s ? [s.u, s.v] as [number, number] : null; } : null, [fc, index]);
  const classement = useMemo(() => geo && fc ? geo.gouvernorats.map(x => ({ x, s: lire(x, index) })).filter(r => r.s).sort((a, b) => b.s!.speed - a.s!.speed) : [], [geo, fc, index]);   // eslint-disable-line react-hooks/exhaustive-deps
  const t = fc?.data.times[index];
  const quand = t ? `${JOURS.format(new Date(t * 1000))} · ${HEURE.format(new Date(t * 1000))} h` : '';
  const source = modele ? 'ATOM Wind' : fc ? `Prévision GFS (NOAA, via Open-Meteo)${fc.stale ? ' · copie en cache' : ''}` : '';

  // résumé selon le rôle, avec les seuils de l'application (« Mon vent ») ; 80 m pour l'éolien
  const kmhJ = heures.map(s => s ? s.speed * 3.6 : null);
  const plages = (ok: (k: number) => boolean) => kmhJ.flatMap((k, h) => k !== null && ok(k) ? [h] : []);
  const enTexte = (hs: number[]) => { if (!hs.length) return 'aucune'; const r: string[] = []; let a = hs[0], b = hs[0];
    for (const h of [...hs.slice(1), 99]) { if (h === b + 1) b = h; else { r.push(a === b ? `${a} h` : `${a} h–${b + 1} h`); a = b = h; } } return r.join(', '); };
  const traiter = plages(k => k >= 3 && k <= 15), fort = plages(k => k >= 50);
  const h80 = Array.from({ length: 24 }, (_, h) => lire(g, jour * 24 + h, 80));
  const produit = h80.filter(s => s && s.speed * 3.6 >= 11 && s.speed * 3.6 < 90).length;
  const moy80 = h80.filter(Boolean).reduce((a, s) => a + s!.speed, 0) / Math.max(1, h80.filter(Boolean).length);

  return <div className="gv-grille">
      <div className="card gv-carte">
        <div className="gv-carte-tete"><div><span className="gv-sur"><MapPinned size={14}/>24 GOUVERNORATS</span><b>Le vent sur la Tunisie</b><span className="muted">{quand || 'Chargement de la prévision…'}</span></div>
          <div className="gv-legende" aria-label="Échelle de vitesse"><span>0</span><i/><span>{ms ? '17 m/s' : '62 km/h'}+</span></div></div>
        <div className="gv-carte-zone" onMouseLeave={() => setSurvol(null)}>
          {!geo && <div className="gv-squelette gv-squelette-carte" aria-hidden="true"/>}
          <svg viewBox={`0 0 ${W} ${H}`} className="gv-svg" role="img" aria-label="Carte des gouvernorats colorés par la vitesse du vent">
            {chemins.map(({ g: x, d }) => { const s = lire(x, index);
              return <path key={x.nom} d={d} className={`gv-gouv${x.nom === nom ? ' is-choisi' : ''}`} style={{ fill: s ? rgb(windColor(s.speed)) : undefined }}
                tabIndex={0} role="button" aria-label={`${x.nom}${s ? ` : ${fmt(s.speed)} ${s.direction}` : ''}`}
                onClick={() => setNom(x.nom)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setNom(x.nom); } }}
                onMouseMove={e => { const b = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect(); setSurvol({ g: x, x: e.clientX - b.left, y: e.clientY - b.top }); }}/>; })}
            {geo?.gouvernorats.map(x => { const [cx, cy] = px(x.chef_lieu[1], x.chef_lieu[0]), s = lire(x, index);
              return <g key={x.nom} className="gv-point" transform={`translate(${cx} ${cy})`}>{s && <g style={{ transform: `rotate(${versDeg(s)}deg)` }} className="gv-mini-fleche"><path d="M0 -8 L4 6 L0 3 L-4 6 Z"/></g>}</g>; })}
          </svg>
          <Particules champ={champ} chemin={tunisie} actif={ouverte}/>
          {survol && (() => { const s = lire(survol.g, index); return <div className="gv-bulle" style={{ left: survol.x, top: survol.y }}>
            <b>{survol.g.nom}</b><span lang="ar">{survol.g.ar}</span>{s ? <span>{fmt(s.speed)} · {s.direction}</span> : <span>valeur indisponible</span>}</div>; })()}
        </div>
        <p className="gv-source">{source ? <>Source : <b>{source}</b>, au chef-lieu de chaque gouvernorat. </> : null}Contours : geoBoundaries / © OpenStreetMap (ODbL).</p>
      </div>

      <div className="gv-detail">
        <div className="card gv-tete">
          {!fc && !erreur && <div className="gv-squelette gv-squelette-tete" aria-hidden="true"/>}
          {erreur && <p className="tool-error" role="alert">Prévision indisponible : {erreur}. Aucune valeur de remplacement n’est affichée.</p>}
          {fc && <>
            <div className="gv-tete-haut"><div><span className="gv-sur">GOUVERNORAT</span><h2>{nom}</h2><span lang="ar" className="gv-ar">{g?.ar}</span></div>
              <select className="inp" value={nom} onChange={e => setNom(e.target.value)} aria-label="Choisir un gouvernorat">{geo?.gouvernorats.map(x => <option key={x.nom}>{x.nom}</option>)}</select></div>
            <div className="gv-maintenant">
              <div><span className="muted">{quand}</span>{now ? <Compteur valeur={val(now.speed)} unite={unite}/> : <span className="gv-compteur">—</span>}
                <span className="gv-dir">{now ? <><Navigation2 size={15}/>Vent {now.direction}, à 10 m du sol</> : 'Valeur indisponible à cette heure'}</span></div>
              <Boussole s={now}/>
            </div>
            <span className={`pill ${modele ? 'good' : 'mid'}`}>{modele ? 'ATOM Wind' : 'Prévision GFS'}</span>
          </>}
        </div>

        {fc && <div className="card gv-heures">
          <div className="gv-heures-tete"><b>Heure par heure</b>
            <div className="gv-jours" role="tablist" aria-label="Jour">{Array.from({ length: nJours }, (_, d) => { const tj = fc.data.times[d * 24];
              const vj = Array.from({ length: 24 }, (_, h) => lire(g, d * 24 + h)).filter(Boolean) as Echantillon[];
              const mx = vj.length ? Math.max(...vj.map(s => s.speed)) : null;
              return <button key={d} role="tab" aria-selected={d === jour} className={d === jour ? 'on' : ''} onClick={() => setIndex(d * 24 + heure)}>
                <span>{d === 0 ? 'Aujourd’hui' : d === 1 ? 'Demain' : new Intl.DateTimeFormat('fr-FR', { weekday: 'long', timeZone: 'Africa/Tunis' }).format(new Date(tj * 1000))}</span>
                <small>{mx !== null ? `max ${fmt(mx)}` : '—'}</small></button>; })}</div></div>
          <div className="gv-bande" role="list">{heures.map((s, h) => <button key={h} role="listitem" className={`gv-h${h === heure ? ' on' : ''}`} onClick={() => setIndex(jour * 24 + h)} aria-label={`${h} h : ${s ? fmt(s.speed) : 'indisponible'}`}>
            <span className="gv-h-heure">{h} h</span>{s ? <><Fleche angle={versDeg(s)} couleur={rgb(windColor(s.speed))}/><span className="gv-h-val">{ms ? s.speed.toFixed(1).replace('.', ',') : Math.round(s.speed * 3.6)}</span>
            <i className="gv-h-barre" style={{ height: `${Math.min(100, 8 + s.speed * 3.6 * 1.4)}%`, background: rgb(windColor(s.speed)) }}/></> : <span className="gv-h-val">—</span>}</button>)}</div>
          <span className="muted gv-unite">{unite}, à 10 m du sol · touchez une heure pour la lire</span>
        </div>}

        {fc && <div className="gv-bas">
          <div className="card gv-rose-carte"><b>Rose des vents de la journée</b><Rose ech={heures}/><span className="muted">D’où vient le vent, pondéré par sa force.</span></div>
          <div className="card gv-role">
            {role === 'agri' && <><span className="gv-sur"><Leaf size={14}/>VOS CULTURES</span><b>Traiter sans dérive</b>
              <p>Heures où le vent à 10 m reste entre 3 et 15 km/h : <b>{enTexte(traiter)}</b>.</p><a className="gv-lien" href="#agri">Préparer mes traitements<ArrowRight size={14}/></a></>}
            {role === 'eolien' && <><span className="gv-sur"><Zap size={14}/>PRODUCTION</span><b>Vent à 80 m</b>
              <p>Heures de production (11 à 90 km/h) : <b>{produit} h sur 24</b> · vent moyen <b>{fmt(moy80)}</b>.</p><a className="gv-lien" href="#eolien">Voir la production du parc<ArrowRight size={14}/></a></>}
            {role !== 'agri' && role !== 'eolien' && <><span className="gv-sur"><Wind size={14}/>À RETENIR</span><b>{fort.length ? 'Vent fort attendu' : 'Pas de vent fort attendu'}</b>
              <p>{fort.length ? <>Vent moyen d’au moins 50 km/h vers <b>{enTexte(fort)}</b> : arrimez ce qui peut s’envoler.</> : 'Le vent moyen reste sous 50 km/h toute la journée.'}</p>
              <a className="gv-lien" href="#alertes">Voir les alertes<ArrowRight size={14}/></a></>}
          </div>
          <div className="card gv-classement"><b>Les plus ventés à {heure} h</b><ol>{classement.slice(0, 5).map(({ x, s }) => <li key={x.nom}><button onClick={() => setNom(x.nom)}><span>{x.nom}</span><span style={{ color: rgb(windColor(s!.speed)) }}>{fmt(s!.speed)}</span></button></li>)}</ol></div>
        </div>}
      </div>
  </div>;
}

export default memo(function ViewGouvernorats() {
  return <section className="view" id="v-gouvernorats"><ContenuGouvernorats/></section>;
});
