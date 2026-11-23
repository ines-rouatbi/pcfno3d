import { memo, useEffect, useMemo, useState } from 'react';
import { Wind } from 'lucide-react';
import { useEtatPage } from './api';
import { chargerContours, type Contours } from './terrain/gouvernorats';
import './styles/vigilance.css';

// Ligne de vigilance de la page Alertes (à la place d'une carte, 22/11/2026) : pour les gouvernorats calculés, la
// rafale la plus forte à 10 m sur les 24 prochaines heures, lue au chef-lieu (le même point que les alertes).
// Seuils des alertes : 50 km/h (rafales fortes), 70 km/h (rafales très fortes). Rien n'est inventé hors zone.

type Lecture = { nom: string; kmh: number; heure: number };

export default memo(function ResumeVigilance() {
  const page = useEtatPage();
  const ouverte = page?.view === 'alertes';
  const [geo, setGeo] = useState<Contours | null>(null);
  useEffect(() => { if (ouverte && !geo) void chargerContours().then(setGeo); }, [ouverte, geo]);

  const lectures = useMemo(() => {
    if (!geo || !ouverte) return [] as Lecture[];
    return geo.gouvernorats.flatMap(g => { const r = window.ATOM?.workspace?.rafaleMax?.(g.chef_lieu[0], g.chef_lieu[1]);
      return r ? [{ nom: g.nom, kmh: r.kmh, heure: r.heure }] : []; });
  }, [geo, ouverte, page?.hour, page?.source]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (!lectures.length) return null;
  const max = lectures.reduce((a, b) => b.kmh > a.kmh ? b : a);
  const forts = lectures.filter(l => l.kmh >= 50).sort((a, b) => b.kmh - a.kmh);
  const niveau = max.kmh >= 70 ? 'rouge' : max.kmh >= 50 ? 'orange' : 'vert';
  const n = lectures.length, km = (v: number) => `${Math.round(v)} km/h`;
  const pic = <>rafale la plus forte : <b>{km(max.kmh)}</b> à <b>{max.nom}</b> vers {max.heure} h</>;
  return <div className={`vg-ligne is-${niveau}`} role="status">
    <Wind size={18} aria-hidden="true"/>
    <span className="vg-pastille">{niveau === 'vert' ? 'Aucun vent fort' : niveau === 'orange' ? 'Rafales fortes' : 'Rafales très fortes'}</span>
    <span className="vg-texte">
      {forts.length
        ? <>Vent fort attendu dans <b>{forts.length}</b> gouvernorat{forts.length > 1 ? 's' : ''} sur {n} ({forts.slice(0, 4).map(f => f.nom).join(', ')}{forts.length > 4 ? '…' : ''}) · {pic}.</>
        : <>{n} gouvernorat{n > 1 ? 's' : ''} calculé{n > 1 ? 's' : ''} : aucun vent fort sur les 24 prochaines heures · {pic}.</>}
    </span>
  </div>;
});
