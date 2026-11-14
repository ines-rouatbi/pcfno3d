import { useEffect, useState } from 'react';
import { ArrowRight, Bell, Box, KeyRound, RefreshCw } from 'lucide-react';
import type { WorkspaceState } from './workspace';
import './styles/service.css';

/** Vues qui n'ont de sens qu'avec le champ de vent du modèle. */
export const VUES_DONNEES = new Set(['tableau', 'monvent', 'agri', 'eolien', 'fiabilite', 'outils', 'historique']);

const TEXTES = {
  modele_absent: {
    titre: 'Le modèle de vent arrive bientôt',
    texte: 'Le service ATOM Wind est en ligne, mais le modèle PC-FNO3D n’est pas encore installé. Dès qu’il le sera, les cartes, les conseils et les alertes de vent s’afficheront ici d’eux-mêmes.',
  },
  cle: {
    titre: 'Connectez-vous pour voir le vent',
    texte: 'Ce service est réservé aux comptes ATOM Wind. Connectez-vous, ou demandez un accès à l’administrateur de votre organisation.',
  },
  hors_ligne: {
    titre: 'Le service de calcul fait une pause',
    texte: 'Le calcul du vent ne répond pas pour le moment. Aucune valeur n’est affichée tant qu’il n’est pas revenu : vous ne verrez jamais de chiffres inventés.',
  },
} as const;

/** Un ciel calme : trois lignes de vent qui passent lentement au-dessus d'une colline. */
function Illustration() {
  return <svg className="service-illustration" viewBox="0 0 320 150" aria-hidden="true">
    <path className="service-hill" d="M0 132C52 96 98 92 148 110S240 82 320 104V150H0Z"/>
    <path className="service-hill far" d="M0 140C70 118 120 124 180 132S270 120 320 126V150H0Z"/>
    <g className="service-turbine" transform="translate(232 70)">
      <path d="M-1.6 0h3.2l1.4 44h-6z"/>
      <g className="service-rotor"><path d="M0 0C3 -6 3 -18 .8 -28L-.8 -28C-3 -18 -3 -6 0 0Z"/><path d="M0 0C3 -6 3 -18 .8 -28L-.8 -28C-3 -18 -3 -6 0 0Z" transform="rotate(120)"/><path d="M0 0C3 -6 3 -18 .8 -28L-.8 -28C-3 -18 -3 -6 0 0Z" transform="rotate(240)"/></g>
      <circle r="3"/>
    </g>
    <path className="service-gust g1" d="M24 44c34-14 62 8 96-4s52-6 70 2"/>
    <path className="service-gust g2" d="M60 66c30-10 54 6 84-2s44-4 58 2"/>
    <path className="service-gust g3" d="M14 86c26-8 44 4 70-2"/>
  </svg>;
}

export function ServiceState({ state }: { state: WorkspaceState }) {
  const etat = (state.service?.etat && state.service.etat in TEXTES ? state.service.etat : 'hors_ligne') as keyof typeof TEXTES;
  const t = TEXTES[etat];
  const [verifie, setVerifie] = useState<string>('');

  // revérifie le service toutes les minutes ; dès que le modèle répond, la page se recharge
  useEffect(() => {
    const base = window.ATOM?.apiBase ?? '';
    const tester = async () => {
      try {
        const r = await fetch(`${base}/api/health`, { cache: 'no-store' });
        const h = r.ok ? await r.json() : null;
        setVerifie(new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }));
        if (h && h.modele_installe !== false && h.status === 'ok' && etat !== 'cle') location.reload();
      } catch { setVerifie(new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })); }
    };
    const id = setInterval(tester, 60000);
    return () => clearInterval(id);
  }, [etat]);

  return <section className="service-state" role="status" aria-live="polite">
    <Illustration/>
    <div className="service-copy">
      <span className={`service-badge is-${etat}`}><span aria-hidden="true"/>{etat === 'modele_absent' ? 'Installation en cours' : etat === 'cle' ? 'Accès réservé' : 'Hors ligne'}</span>
      <h2>{t.titre}</h2>
      <p>{t.texte}</p>
      {etat === 'hors_ligne' && state.service?.detail && <p className="service-detail">Détail technique : {state.service.detail}</p>}
      <div className="service-actions">
        {etat === 'cle'
          ? <button className="btn primary" onClick={() => document.getElementById('b-login')?.click()}><KeyRound size={16}/>Se connecter</button>
          : <button className="btn primary" onClick={() => location.reload()}><RefreshCw size={16}/>Réessayer</button>}
        <a className="btn" href="#carte3d"><Box size={16}/>Relief et prévision nationale en 3D<ArrowRight size={15}/></a>
        <a className="btn" href="#alertes"><Bell size={16}/>Mes alertes</a>
      </div>
      <p className="service-note">{verifie ? `Dernière vérification à ${verifie}. ` : ''}La page vérifie le service toutes les minutes.</p>
    </div>
  </section>;
}
