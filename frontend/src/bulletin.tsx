import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Box } from 'lucide-react';
import { useEtatPage } from './api';
import './styles/bulletin.css';

// Le bulletin de la vue d'ensemble : une phrase écrite comme on la dirait, sur un ciel où passent les lignes
// du vent réel à 10 m (champ du modèle, heure affichée). Rien n'est inventé : sans données, pas de bulletin.
type Bulletin = {
  lieu: string; heure: number; vitesse: string; direction: string; versDeg: number; rafales: string;
  moyenne: string; ventMoyen: number; picRafale: string; picHeure: number; vigilance: number; beaufort: string;
  echantillon: (fx: number, fy: number) => [number, number];
};

const salut = (h: number) => (h < 5 || h >= 19 ? 'Bonsoir.' : 'Bonjour.');

function Ciel({ b }: { b: Bulletin }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const echantillon = useRef(b.echantillon); echantillon.current = b.echantillon;
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const ctx = cv.getContext('2d')!; let w = 0, h = 0, dpr = 1, id = 0, vivant = true;
    const reduit = matchMedia('(prefers-reduced-motion: reduce)').matches;
    type P = { x: number; y: number; age: number };
    let parts: P[] = [];
    const nouveau = (): P => ({ x: Math.random() * w, y: Math.random() * h, age: Math.random() * 120 | 0 });
    const taille = () => {
      dpr = Math.min(2, devicePixelRatio || 1); w = cv.clientWidth; h = cv.clientHeight;
      cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      parts = Array.from({ length: Math.round(w * h / 1400) }, nouveau);
    };
    const couleur = getComputedStyle(cv).color;
    const vent = (x: number, y: number) => { const [u, v] = echantillon.current(x / w, 1 - y / h); return [u, -v]; };
    const image = () => {
      if (!vivant) return;
      ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,.06)'; ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'source-over'; ctx.strokeStyle = couleur; ctx.lineCap = 'round';
      for (const p of parts) {
        const [u, v] = vent(p.x, p.y), s = Math.hypot(u, v);
        if (!isFinite(s)) { Object.assign(p, nouveau(), { age: 0 }); continue; }
        const nx = p.x + u * .28, ny = p.y + v * .28;
        ctx.globalAlpha = Math.min(.55, .12 + s / 22); ctx.lineWidth = 1 + Math.min(2.2, s / 7);
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny); ctx.stroke();
        p.x = nx; p.y = ny;
        if (++p.age > 140 || p.x < -10 || p.x > w + 10 || p.y < -10 || p.y > h + 10) Object.assign(p, nouveau(), { age: 0 });
      }
      ctx.globalAlpha = 1;
      if (!reduit && !document.hidden) id = requestAnimationFrame(image);
    };
    const statique = () => {               // mouvement réduit : des traits fixes dans le sens du vent
      ctx.clearRect(0, 0, w, h); ctx.strokeStyle = couleur; ctx.lineCap = 'round';
      for (let y = 18; y < h; y += 34) for (let x = 18; x < w; x += 46) {
        const [u, v] = vent(x, y), s = Math.hypot(u, v); if (!isFinite(s) || !s) continue;
        ctx.globalAlpha = Math.min(.5, .15 + s / 25); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(x - u / s * 9, y - v / s * 9); ctx.lineTo(x + u / s * 9, y + v / s * 9); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };
    taille();
    const ro = new ResizeObserver(() => { taille(); if (reduit) statique(); }); ro.observe(cv);
    if (reduit) statique(); else id = requestAnimationFrame(image);
    const vis = () => { cancelAnimationFrame(id); if (!document.hidden && !reduit) id = requestAnimationFrame(image); };
    document.addEventListener('visibilitychange', vis);
    return () => { vivant = false; cancelAnimationFrame(id); ro.disconnect(); document.removeEventListener('visibilitychange', vis); };
  }, []);
  return <canvas ref={ref} className="bulletin-sky" aria-hidden="true"/>;
}

export function BulletinDuJour() {
  const page = useEtatPage();
  const [b, setB] = useState<Bulletin | null>(null);
  useEffect(() => { setB(window.ATOM?.workspace?.bulletin?.() ?? null); }, [page]);
  if (!b) return null;
  const vigilance = b.vigilance === 0 ? 'Rien d’important à surveiller pour l’instant.'
    : b.vigilance === 1 ? 'Un point de vigilance est ouvert dans les alertes.' : `${b.vigilance} points de vigilance sont ouverts dans les alertes.`;
  return <section className="bulletin" aria-label="Le vent en ce moment">
    <Ciel b={b}/>
    <div className="bulletin-copy">
      <p className="bulletin-hello">{salut(b.heure)}</p>
      <h2 className="bulletin-sentence">{`À ${b.lieu}, à ${b.heure} h, le vent vient ${b.direction} à ${b.vitesse}.`}</h2>
      <p className="bulletin-detail"><span>{b.beaufort}</span> <span>{`Rafales autour de ${b.rafales}.`}</span> <span>{`Sur le Grand Tunis, ${b.moyenne} en moyenne, avec un pic de rafales de ${b.picRafale} vers ${b.picHeure} h.`}</span> <span>{vigilance}</span></p>
      <div className="bulletin-actions">
        <a className="btn primary" href="#monvent">Que puis-je faire ici ?<ArrowRight size={16}/></a>
        <a className="btn" href="#carte3d"><Box size={16}/>Voir le vent en 3D</a>
      </div>
    </div>
    <svg className="bulletin-compass" viewBox="-40 -40 80 80" role="img" aria-label={`Le vent va vers ${Math.round(b.versDeg)} degrés`}>
      <circle r="34"/>
      {['N', 'E', 'S', 'O'].map((n, i) => <text key={n} x={27 * Math.sin(i * Math.PI / 2)} y={-27 * Math.cos(i * Math.PI / 2) + 3.5} textAnchor="middle">{n}</text>)}
      <g transform={`rotate(${b.versDeg})`}><path d="M0 -20L6 6L0 2L-6 6Z"/></g>
    </svg>
  </section>;
}
