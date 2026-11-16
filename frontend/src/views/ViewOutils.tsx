import { memo, useEffect, useState, type ReactNode } from 'react';
import { Construction, Fence, Loader2, MapPin, RadioTower, Unplug } from 'lucide-react';
import { api, kmh, lieux, nomCourt, useEtatPage, versLaCarte } from '../api';
import '../styles/outils.css';

// Outils de terrain (cahier v12, espaces de perspective) : chantier, environnement, holding, exploitation.
// Chaque outil appelle sa route du service ; aucune valeur n'est calculée ici.
const JOURS = ["Aujourd'hui", 'Demain', 'Après-demain'];
type Outil = 'grue' | 'haies' | 'porteurs' | 'panne';

function useRequete<T>(cle: string | null, charger: () => Promise<T>) {
  const [etat, setEtat] = useState<{ cle: string | null; d?: T; erreur?: string; charge: boolean }>({ cle: null, charge: false });
  useEffect(() => {
    if (!cle) return;
    let vivant = true;
    setEtat({ cle, charge: true });
    charger().then(d => vivant && setEtat({ cle, d, charge: false }), (e: Error) => vivant && setEtat({ cle, erreur: e.message, charge: false }));
    return () => { vivant = false; };
  }, [cle]);   // eslint-disable-line react-hooks/exhaustive-deps
  return etat;
}

function Etat({ charge, erreur, children }: { charge: boolean; erreur?: string; children: ReactNode }) {
  if (charge) return <p className="tool-wait"><Loader2 size={16} className="spin"/>Calcul en cours…</p>;
  if (erreur) return <p className="tool-error" role="alert">{erreur}</p>;
  return <>{children}</>;
}

function ChoixLieu({ valeur, changer, id }: { valeur: string; changer: (v: string) => void; id: string }) {
  return <label htmlFor={id}>Lieu<select className="inp" id={id} value={valeur} onChange={e => changer(e.target.value)}>
    {lieux().map(l => <option key={l.nom} value={l.nom}>{l.nom}</option>)}</select></label>;
}

// ------------------------------------------------------------------ grue de chantier
function Grue({ actif }: { actif: boolean }) {
  const [lieu, setLieu] = useState('Ben Arous'), [j, setJ] = useState(0), [h, setH] = useState(40), [lim, setLim] = useState(20);
  const r = useRequete<any>(actif ? `${lieu}|${j}|${h}|${lim}` : null,
    () => api(`/atom-wind/grue?j=${j}&lieu=${encodeURIComponent(nomCourt(lieu))}&hauteur_m=${h}&limite_rafale_m_s=${lim}`));
  const d = r.d, mx = d ? Math.max(lim * 1.15, ...d.heures.map((x: any) => x.rafale_m_s)) : 1;
  return <div className="tool">
    <div className="tool-form">
      <ChoixLieu id="og-lieu" valeur={lieu} changer={setLieu}/>
      <label htmlFor="og-jour">Jour<select className="inp" id="og-jour" value={j} onChange={e => setJ(+e.target.value)}>{JOURS.map((n, i) => <option key={n} value={i}>{n}</option>)}</select></label>
      <label htmlFor="og-h">Hauteur de la flèche (m)<input className="inp" id="og-h" type="number" min={5} max={300} value={h} onChange={e => setH(Math.max(5, +e.target.value || 40))}/></label>
      <label htmlFor="og-lim">Limite de la grue en rafale (km/h)<input className="inp" id="og-lim" type="number" min={10} max={200} value={Math.round(lim * 3.6)} onChange={e => setLim(Math.max(3, (+e.target.value || 72) / 3.6))}/></label>
    </div>
    <Etat charge={r.charge} erreur={r.erreur}>{d && <>
      <div className="tool-verdict"><b>{d.creneaux_levage.length ? `Créneaux de levage : ${d.creneaux_levage.join(', ')}` : 'Aucun créneau de levage sous la limite'}</b>
        <span>Rafale la plus forte à {d.hauteur_fleche_m} m : {kmh(d.rafale_max_m_s)} km/h, pour une limite de {d.limite_rafale_km_h} km/h.</span></div>
      <div className="tool-bars" role="img" aria-label="Rafale à hauteur de flèche, heure par heure">
        <span className="tool-limit" style={{ bottom: `${lim / mx * 100}%` }}><i>limite {Math.round(lim * 3.6)} km/h</i></span>
        {d.heures.map((x: any) => <span key={x.heure} className={x.ok ? 'ok' : 'stop'} style={{ height: `${Math.max(4, x.rafale_m_s / mx * 100)}%` }} title={`${x.heure} h : rafale ${x.rafale_km_h} km/h`}/>)}
      </div>
      <div className="tool-axis">{d.heures.map((x: any) => <span key={x.heure}>{x.heure % 3 === 0 ? `${x.heure} h` : ''}</span>)}</div>
      <p className="tool-note">Vent moyen calculé par ATOM Wind ; rafale = facteur de rafale GFS réduit avec la hauteur. La décision de levage reste celle du chef de manœuvre, selon les abaques du constructeur.</p>
    </>}</Etat>
  </div>;
}

// ------------------------------------------------------------------ haies brise-vent
const SECTEURS16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSO', 'SO', 'OSO', 'O', 'ONO', 'NO', 'NNO'];
function Haies({ actif }: { actif: boolean }) {
  const [lieu, setLieu] = useState('Mornag (vignes)'), [h, setH] = useState(3);
  const r = useRequete<any>(actif ? `${lieu}|${h}` : null, () => api(`/atom-wind/haies?lieu=${encodeURIComponent(nomCourt(lieu))}&hauteur_haie_m=${h}`));
  const d = r.d, mx = d ? Math.max(1, ...d.rose.map((s: any) => s.part_vents_forts_pct)) : 1;
  return <div className="tool">
    <div className="tool-form">
      <ChoixLieu id="oh-lieu" valeur={lieu} changer={setLieu}/>
      <label htmlFor="oh-h">Hauteur adulte de la haie (m)<input className="inp" id="oh-h" type="number" min={1} max={30} step={0.5} value={h} onChange={e => setH(Math.max(1, +e.target.value || 3))}/></label>
    </div>
    <Etat charge={r.charge} erreur={r.erreur}>{d && <div className="tool-split">
      <svg viewBox="-110 -110 220 220" className="tool-rose" role="img" aria-label="Part des vents forts selon leur provenance">
        <circle r="96" className="rose-ring"/><circle r="48" className="rose-ring"/>
        {d.rose.map((s: any, i: number) => {
          const r0 = 10 + 84 * s.part_vents_forts_pct / mx, a0 = (i * 22.5 - 9) * Math.PI / 180, a1 = (i * 22.5 + 9) * Math.PI / 180;
          return s.part_vents_forts_pct > 0 && <path key={s.secteur} className="rose-petal" d={`M0 0L${r0 * Math.sin(a0)} ${-r0 * Math.cos(a0)}A${r0} ${r0} 0 0 1 ${r0 * Math.sin(a1)} ${-r0 * Math.cos(a1)}Z`}><title>{`${s.secteur} : ${s.part_vents_forts_pct} % des vents forts`}</title></path>;
        })}
        <line className="rose-hedge" x1={-90 * Math.sin(d.azimut_haie_deg * Math.PI / 180)} y1={90 * Math.cos(d.azimut_haie_deg * Math.PI / 180)} x2={90 * Math.sin(d.azimut_haie_deg * Math.PI / 180)} y2={-90 * Math.cos(d.azimut_haie_deg * Math.PI / 180)}/>
        {['N', 'E', 'S', 'O'].map((n, i) => <text key={n} x={104 * Math.sin(i * Math.PI / 2)} y={-104 * Math.cos(i * Math.PI / 2) + 4} textAnchor="middle">{n}</text>)}
      </svg>
      <div>
        <div className="tool-verdict"><b>Orientez la haie {SECTEURS16[Math.round(d.azimut_haie_deg / 22.5) % 16]}–{SECTEURS16[(Math.round(d.azimut_haie_deg / 22.5) + 8) % 16]}</b>
          <span>Les vents forts viennent surtout du {d.secteur_dominant} ({d.direction_dominante_deg}°), avec une constance de {Math.round(d.constance * 100)} %.</span></div>
        <dl className="tool-facts"><div><dt>Abri fort</dt><dd>{d.zone_protegee_m.abri_fort} m sous le vent</dd></div><div><dt>Abri sensible</dt><dd>{d.zone_protegee_m.abri} m</dd></div><div><dt>Côté exposé</dt><dd>{d.zone_protegee_m.au_vent} m</dd></div></dl>
        <p className="tool-note">Calcul sur {d.heures} heures de vent ({d.critere}). Le trait sur la rose montre l’orientation conseillée.</p>
      </div>
    </div>}</Etat>
  </div>;
}

// ------------------------------------------------------------------ mâts autonomes (Porteur)
function Porteurs({ actif }: { actif: boolean }) {
  const [h, setH] = useState(20), [conso, setConso] = useState(5);
  const r = useRequete<any>(actif ? `${h}|${conso}` : null, () => api(`/atom-wind/porteurs?hauteur_m=${h}&conso_w=${conso}&n=8`));
  return <div className="tool">
    <div className="tool-form">
      <label htmlFor="op-h">Hauteur du mât<select className="inp" id="op-h" value={h} onChange={e => setH(+e.target.value)}>{[15, 20, 25, 30].map(v => <option key={v} value={v}>{v} m</option>)}</select></label>
      <label htmlFor="op-c">Consommation des capteurs (W)<input className="inp" id="op-c" type="number" min={1} max={2000} value={conso} onChange={e => setConso(Math.max(1, +e.target.value || 5))}/></label>
    </div>
    <Etat charge={r.charge} erreur={r.erreur}>{r.d && <>
      <ol className="tool-sites">{r.d.sites.map((s: any) => <li key={s.rang}>
        <span className="tool-rank">{s.rang}</span>
        <div><b>{s.lat.toFixed(3)} N, {s.lon.toFixed(3)} E</b><span>sol à {s.altitude_m} m · vent moyen {kmh(s.vent_moyen_m_s)} km/h</span></div>
        <div className="tool-num"><b>{(s.production_wh_par_jour / 1000).toFixed(1).replace('.', ',')} kWh/j</b><span>{s.couverture_conso_pct} % des besoins</span></div>
        <button className="btn" onClick={() => versLaCarte(s)}><MapPin size={15}/>Voir</button>
      </li>)}</ol>
      <p className="tool-note">{r.d.limite}.</p>
    </>}</Etat>
  </div>;
}

// ------------------------------------------------------------------ panne d'une station (exploitation)
const STATIONS = ['Tunis-Carthage', 'Zaghouan', 'Nabeul'];
function Panne({ actif }: { actif: boolean }) {
  const [heure, setHeure] = useState(9), [retirees, setRetirees] = useState<string[]>(['Nabeul']);
  const cle = actif && retirees.length ? `${heure}|${retirees.join(',')}` : null;
  const r = useRequete<any>(cle, () => api(`/atom-wind/scenario?heure=${heure}&retirer=${encodeURIComponent(retirees.join(','))}`));
  const basculer = (s: string) => setRetirees(l => l.includes(s) ? l.filter(x => x !== s) : [...l, s]);
  return <div className="tool">
    <div className="tool-form">
      <label htmlFor="ox-h">Heure déjà mesurée d’aujourd’hui<input className="inp" id="ox-h" type="number" min={0} max={23} value={heure} onChange={e => setHeure(Math.min(23, Math.max(0, +e.target.value || 0)))}/></label>
      <fieldset className="tool-checks"><legend>Stations en panne</legend>{STATIONS.map(s => <label key={s} className="check"><input type="checkbox" checked={retirees.includes(s)} onChange={() => basculer(s)}/>{s}</label>)}</fieldset>
    </div>
    {!retirees.length ? <p className="tool-note">Choisissez au moins une station à retirer.</p> : <Etat charge={r.charge} erreur={r.erreur}>{r.d && <>
      <dl className="tool-facts">
        <div><dt>Écart moyen à 10 m</dt><dd>{kmh(r.d.domaine['10m'].ecart_moyen_m_s)} km/h</dd></div>
        <div><dt>Écart maximal à 10 m</dt><dd>{kmh(r.d.domaine['10m'].ecart_max_m_s)} km/h</dd></div>
        <div><dt>Zone changée de plus de 2 km/h</dt><dd>{Math.round(r.d.domaine['10m'].part_au_dessus_0_5_m_s * 100)} %</dd></div>
      </dl>
      <div className="tblwrap"><table className="t"><thead><tr><th>Station</th><th>État</th><th className="n">Erreur avec</th><th className="n">Erreur sans</th></tr></thead><tbody>
        {Object.entries(r.d.stations).map(([n, s]: [string, any]) => <tr key={n}><td>{n}</td><td>{s.retiree ? 'retirée' : s.mesuree ? 'mesurée' : 'sans mesure'}</td>
          <td className="n">{s.erreur_avec_m_s == null ? '—' : `${kmh(s.erreur_avec_m_s)} km/h`}</td><td className="n">{s.erreur_sans_m_s == null ? '—' : `${kmh(s.erreur_sans_m_s)} km/h`}</td></tr>)}
      </tbody></table></div>
      <p className="tool-note">L’heure est recalculée sans les stations choisies : l’écart montre ce que la panne change, l’erreur sans une station ce que le reste du réseau et GFS devinent à sa place.</p>
    </>}</Etat>}
  </div>;
}

const OUTILS: { id: Outil; nom: string; texte: string; icone: ReactNode; admin?: boolean }[] = [
  { id: 'grue', nom: 'Levage à la grue', texte: 'Rafale à hauteur de flèche et créneaux sûrs.', icone: <Construction size={20}/> },
  { id: 'haies', nom: 'Haies brise-vent', texte: 'Orientation et zone abritée.', icone: <Fence size={20}/> },
  { id: 'porteurs', nom: 'Mât autonome', texte: 'Où une petite éolienne alimente un mât.', icone: <RadioTower size={20}/> },
  { id: 'panne', nom: 'Panne d’une station', texte: 'Ce que le réseau perd sans elle.', icone: <Unplug size={20}/>, admin: true },
];

export default memo(function ViewOutils() {
  const page = useEtatPage();
  const [outil, setOutil] = useState<Outil>('grue');
  const ouverte = page?.view === 'outils' && page.source !== 'hors_ligne';
  const admin = typeof window !== 'undefined' && window.ATOM?.workspace?.role?.() === 'admin';
  return <section className="view" id="v-outils">
    <div className="tool-tabs" role="tablist" aria-label="Outils de terrain">
      {OUTILS.filter(o => !o.admin || admin).map(o => <button key={o.id} role="tab" aria-selected={outil === o.id} className={outil === o.id ? 'on' : ''} onClick={() => setOutil(o.id)}>
        <span className="tool-icon">{o.icone}</span><span><b>{o.nom}</b><small>{o.texte}</small></span></button>)}
    </div>
    <div className="card tool-card" role="tabpanel">
      {outil === 'grue' && <Grue actif={ouverte}/>}
      {outil === 'haies' && <Haies actif={ouverte}/>}
      {outil === 'porteurs' && <Porteurs actif={ouverte}/>}
      {outil === 'panne' && admin && <Panne actif={ouverte}/>}
    </div>
  </section>;
});
