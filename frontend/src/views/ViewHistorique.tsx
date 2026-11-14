import { memo, useEffect, useState } from 'react';
import { CalendarClock, Loader2, Radio, TrendingUp } from 'lucide-react';
import { api, useEtatPage } from '../api';

// Historique (cahier v12, BF40, HIS1) : les journées temps réel et les prévisions déjà calculées par le
// service. Ouvrir une journée la recharge dans tous les espaces, comme le sélecteur de journée.
type Journee = { nom: string; date: string; type: string; echeance_j: number | null; emise_le: string | null; taille_ko: number; ecrit_utc: string };
const dateFr = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });

export default memo(function ViewHistorique() {
  const page = useEtatPage();
  const [l, setL] = useState<Journee[] | null>(null), [err, setErr] = useState('');
  const ouverte = page?.view === 'historique' && page.source !== 'hors_ligne';
  useEffect(() => {
    if (!ouverte || l) return;
    api<{ journees: Journee[] }>('/atom-wind/historique').then(d => setL(d.journees), e => setErr(e.message));
  }, [ouverte, l]);
  const ouvrir = async (nom: string) => { await window.ATOM?.choisirJour?.(`hist:${nom}`); window.ATOM?.workspace?.navigate('tableau'); };
  const parJour = (l ?? []).reduce<Record<string, Journee[]>>((acc, j) => ((acc[j.date] ??= []).push(j), acc), {});
  return <section className="view" id="v-historique">
    <div className="card"><h2>Ce que le service a déjà calculé</h2><p className="muted" style={{ margin: 0 }}>Chaque journée de temps réel est gardée, avec les prévisions émises pour elle la veille et l’avant-veille : on peut ainsi rouvrir une journée passée et comparer une prévision à ce qui s’est réellement passé.</p></div>
    {err ? <div className="card"><p className="tool-error" role="alert">{err}</p></div>
      : !l ? <div className="card"><p className="tool-wait"><Loader2 size={16} className="spin"/>Lecture de l’historique…</p></div>
      : !l.length ? <div className="card empty-state"><b>Aucune journée archivée pour l’instant</b><p className="muted" style={{ margin: '6px 0 0' }}>Le service garde chaque journée calculée en temps réel ; elles apparaîtront ici.</p></div>
      : <div className="history">{Object.entries(parJour).sort(([a], [b]) => b.localeCompare(a)).map(([date, js]) => <article key={date} className="card history-day">
          <header><CalendarClock size={18}/><h3>{dateFr(date)}</h3></header>
          <ul>{js.sort((a, b) => (a.echeance_j ?? 0) - (b.echeance_j ?? 0)).map(j => <li key={j.nom}>
            <span className={`history-kind ${j.type === 'temps reel' ? 'live' : 'forecast'}`}>{j.type === 'temps reel' ? <Radio size={14}/> : <TrendingUp size={14}/>}
              {j.type === 'temps reel' ? 'Temps réel' : `Prévision à J+${j.echeance_j}`}</span>
            <span className="muted">{j.type === 'temps reel' ? `calculé le ${new Date(j.ecrit_utc).toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', timeZone: 'Africa/Tunis' })}` : `émise le ${new Date(`${j.emise_le}T12:00:00Z`).toLocaleDateString('fr-FR')}`}</span>
            <button className="btn" onClick={() => ouvrir(j.nom)}>Ouvrir</button>
          </li>)}</ul>
        </article>)}</div>}
  </section>;
});
