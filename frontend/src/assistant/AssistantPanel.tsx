import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ArrowUp, BookOpen, Compass, Lightbulb, LocateFixed, MapPin, MousePointer2, RotateCcw, Sparkles, Wrench, X } from 'lucide-react';
import type { WorkspaceState } from '../workspace';
import {
  chargerGuide, demander, ErreurService, executer, localiser, parleDeSaPosition, repondreAvecLeGuide,
  serviceJoignable, type Action, type Contexte, type Guide, type Position,
} from './assistant-core';
import { jouerDemo } from './curseur';
import './assistant.css';

type Message = {
  id: number;
  role: 'user' | 'assistant';
  texte: string;
  etapes?: string[];
  actions?: Action[];        // boutons restants (les actions automatiques sont déjà faites)
  faits?: string[];          // ce que l'assistant a fait sur la page
  relances?: string[];
  sources?: string[];
  mode?: 'ia' | 'guide';
  erreur?: boolean;
};
type Etat = 'verification' | 'ia' | 'guide';

const CLE_FIL = 'aw-assistant-fil';
const NOMS_OUTILS: Record<string, string> = {
  vent_au_point: 'vent en un point', profil_vertical: 'profil vertical', decision_agricole: 'décision agricole',
  production_eolienne: 'production éolienne', alertes_vent: 'alertes de vent', fenetres_maintenance: 'créneaux de maintenance',
  variations_production: 'variations de production', vent_grue: 'vent à la grue', plan_haies: 'plan de haies',
  explication_point: 'stations influentes', guide_application: 'guide de l’application',
};
const INTERFACE = new Set(['ouvrir_vue', 'montrer_lieu', 'choisir_activite', 'demander_position']);

function lireFil(): Message[] {
  try { const v = sessionStorage.getItem(CLE_FIL); return v ? JSON.parse(v) : []; } catch { return []; }
}
function ecrireFil(m: Message[]) {
  try { sessionStorage.setItem(CLE_FIL, JSON.stringify(m.slice(-40))); } catch { /* stockage indisponible */ }
}
function langue(): string { try { return localStorage.getItem('aw-langue') || 'fr'; } catch { return 'fr'; } }

// ------------------------------------------------------------------ texte : gras, listes, paragraphes
function enLigne(t: string): ReactNode[] {
  return t.split(/(\*\*[^*]+\*\*)/g).map((p, i) => p.startsWith('**') && p.endsWith('**')
    ? <strong key={i}>{p.slice(2, -2)}</strong> : <Fragment key={i}>{p}</Fragment>);
}
function Texte({ texte }: { texte: string }) {
  const blocs: ReactNode[] = [];
  let liste: { ordonnee: boolean; items: string[] } | null = null;
  const fermer = () => {
    if (!liste) return;
    const L = liste; const items = L.items.map((x, i) => <li key={i}>{enLigne(x)}</li>);
    blocs.push(L.ordonnee ? <ol key={blocs.length}>{items}</ol> : <ul key={blocs.length}>{items}</ul>);
    liste = null;
  };
  for (const ligne of texte.split('\n')) {
    const puce = /^\s*[-•*]\s+(.*)$/.exec(ligne), num = /^\s*\d+[.)]\s+(.*)$/.exec(ligne);
    if (puce || num) {
      const ordonnee = !!num;
      if (!liste || liste.ordonnee !== ordonnee) { fermer(); liste = { ordonnee, items: [] }; }
      liste.items.push((puce || num)![1]);
    } else if (ligne.trim()) { fermer(); blocs.push(<p key={blocs.length}>{enLigne(ligne)}</p>); }
    else fermer();
  }
  fermer();
  return <>{blocs}</>;
}

// ------------------------------------------------------------------ panneau
export function AssistantPanel({ state, open, onClose }: { state: WorkspaceState; open: boolean; onClose: () => void }) {
  const [fil, setFil] = useState<Message[]>(lireFil);
  const [saisie, setSaisie] = useState('');
  const [attente, setAttente] = useState<string | null>(null);
  const [etat, setEtat] = useState<Etat>('verification');
  const [guide, setGuide] = useState<Guide>({ vues: {}, taches: [], projets: [] });
  const [position, setPosition] = useState<Position | null>(null);
  const champ = useRef<HTMLTextAreaElement>(null);
  const bas = useRef<HTMLDivElement>(null);
  const id = useRef(Date.now());
  const derniereVerif = useRef(0);
  const lang = langue();
  const rtl = lang === 'ar';

  useEffect(() => { chargerGuide().then(setGuide); }, []);
  useEffect(() => { ecrireFil(fil); bas.current?.scrollIntoView({ block: 'end' }); }, [fil, attente]);

  const verifier = useCallback(async () => {
    derniereVerif.current = Date.now();
    const ok = await serviceJoignable();
    setEtat(ok ? 'ia' : 'guide');
    return ok;
  }, []);
  useEffect(() => {
    if (!open) return;
    if (etat !== 'ia' || Date.now() - derniereVerif.current > 60000) verifier();   // hors ligne : on revérifie à chaque ouverture
    const t = setTimeout(() => champ.current?.focus(), 50);
    const echap = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', echap);
    return () => { clearTimeout(t); window.removeEventListener('keydown', echap); };
  }, [open, onClose, verifier]);   // eslint-disable-line react-hooks/exhaustive-deps  (etat lu à l'ouverture seulement)

  const ajouter = (m: Omit<Message, 'id'>) => setFil(f => [...f, { ...m, id: ++id.current }]);

  /** Exécute les actions automatiques, rend les phrases « fait » et les boutons qui restent. */
  const appliquer = (actions: Action[] = []) => {
    const faits: string[] = [], restent: Action[] = [];
    // le point et l'activité d'abord (ils passent par « Mon vent »), la page demandée en dernier
    const ordre = (a: Action) => (a.type === 'ouvrir_vue' ? 1 : 0);
    let demo: Extract<Action, { type: 'montrer' }> | null = null;
    for (const a of [...actions].sort((x, y) => ordre(x) - ordre(y))) {
      if (a.type === 'montrer') { if (a.auto && !demo) demo = a; restent.push({ ...a, auto: false }); continue; }
      if (a.auto) { const t = executer(a, guide); if (t) faits.push(t); }
      else restent.push(a);
    }
    if (demo) { const d = demo; setTimeout(() => montrer(d.tache), 450); faits.push(`Je vous montre : ${d.titre ?? 'la démarche'}.`); }
    return { faits, restent };
  };

  const obtenirPosition = async (): Promise<Position | null> => {
    setAttente('Recherche de votre position…');
    try {
      const p = await localiser();
      setPosition(p);
      return p;
    } catch (e) {
      ajouter({ role: 'assistant', texte: `${(e as Error).message} Vous pouvez aussi nommer un lieu (Mornag, Nabeul, La Marsa…) ou toucher la carte dans « Mon vent ».`, mode: 'guide', erreur: true });
      return null;
    }
  };

  const envoyer = async (brut: string, opts: { forcerPosition?: boolean } = {}) => {
    const question = brut.trim();
    if (!question || attente) return;
    setSaisie('');
    const historique = fil.filter(m => !m.erreur).map(m => ({ role: m.role, contenu: m.texte }));
    if (!opts.forcerPosition) ajouter({ role: 'user', texte: question });

    // 1. la question parle de « chez moi » : on demande la position au navigateur, puis on place le point
    let pos = position;
    const faitsAvant: string[] = [];
    if ((opts.forcerPosition || parleDeSaPosition(question)) && !pos) pos = await obtenirPosition();
    if (pos && (opts.forcerPosition || parleDeSaPosition(question))) {
      const nom = window.ATOM?.workspace?.placer?.(pos.lat, pos.lon);
      faitsAvant.push(nom ? `Position trouvée (à ${pos.precision_m} m près) : ${nom}.`
        : 'Votre position est en dehors du Grand Tunis : ATOM Wind ne calcule pas encore votre zone.');
    }

    const ws = window.ATOM?.workspace;
    const snap = ws?.snapshot?.() ?? state;
    const contexte: Contexte = {
      vue: snap.view, lieu: snap.location, heure: snap.hour, langue: lang, role: ws?.role?.(),
      ...(pos ? { lat_utilisateur: +pos.lat.toFixed(5), lon_utilisateur: +pos.lon.toFixed(5) } : {}),
    };

    // 2. le service (le modèle IA et les outils d'ATOM Wind) ; sinon le guide, sans aucun chiffre
    let enLigne = etat === 'ia' || (etat !== 'guide' || Date.now() - derniereVerif.current > 60000);
    if (enLigne && etat !== 'ia') enLigne = await verifier();
    if (enLigne) {
      setAttente('L’assistant consulte ATOM Wind…');
      try {
        const r = await demander(question, historique, contexte);
        const { faits, restent } = appliquer(r.actions);
        const sources = [...new Set((r.outils || []).filter(o => !INTERFACE.has(o.nom)).map(o => NOMS_OUTILS[o.nom] || o.nom))];
        ajouter({ role: 'assistant', texte: r.reponse, etapes: r.etapes?.liste, actions: restent, faits: [...faitsAvant, ...faits],
          relances: r.relances, sources, mode: 'ia' });
        setEtat('ia');
        setAttente(null);
        return;
      } catch (e) {
        const err = e as ErreurService;
        if (err.statut === 401 || err.statut === 403) {
          ajouter({ role: 'assistant', texte: 'Le service demande une clé ou une connexion. Connectez-vous (menu en haut à droite) pour utiliser l’assistant avec les données ; en attendant, je réponds avec le guide de l’application.', mode: 'guide', erreur: true });
        } else if (err.statut === 502 || err.statut === 503) {
          // le service répond, mais l'assistant IA n'est pas configuré ou son fournisseur est en panne
          ajouter({ role: 'assistant', texte: `L’assistant IA n’est pas disponible sur le service (${err.message}). En attendant, je réponds avec le guide de l’application.`, mode: 'guide', erreur: true });
        }
        setEtat('guide');
      }
    }
    const lieux = ws?.lieux?.() ?? [];
    const g = repondreAvecLeGuide(guide, question, snap.view, lieux);
    const { faits, restent } = appliquer(g.actions);
    ajouter({ role: 'assistant', texte: g.texte, etapes: g.etapes, actions: restent, faits: [...faitsAvant, ...faits],
      relances: g.relances, mode: 'guide' });
    setAttente(null);
  };

  /** Démonstration au curseur : le panneau se range, le curseur joue les étapes, puis le panneau revient. */
  const montrer = (id: string) => {
    const t = guide.taches.find(x => x.id === id);
    if (!t?.demo?.length) return;
    onClose();
    const noms = Object.fromEntries(Object.entries(guide.vues).map(([k, v]) => [k, v.nom]));
    jouerDemo({ id: t.id, titre: t.titre, demo: t.demo }, { nomsVues: noms, fin: (raison, message) => {
      window.dispatchEvent(new CustomEvent('atom:assistant-ouvrir'));
      if (raison === 'bloque' && message) ajouter({ role: 'assistant', texte: '', faits: [message], mode: 'guide' });
    } });
  };

  const cliquerAction = async (a: Action) => {
    if (a.type === 'montrer') { montrer(a.tache); return; }
    if (a.type === 'demander_position') {
      const derniere = [...fil].reverse().find(m => m.role === 'user');
      if (derniere) await envoyer(derniere.texte, { forcerPosition: true });
      else { const p = await obtenirPosition(); setAttente(null); if (p) window.ATOM?.workspace?.placer?.(p.lat, p.lon); }
      return;
    }
    const t = executer(a, guide);
    if (t) ajouter({ role: 'assistant', texte: '', faits: [t], mode: 'guide' });
  };

  const libelleAction = (a: Action) => a.type === 'montrer' ? 'Revoir la démonstration' : a.type === 'ouvrir_vue' ? (a.libelle || `Ouvrir « ${guide.vues[a.vue]?.nom ?? a.vue} »`)
    : a.type === 'demander_position' ? 'Partager ma position'
    : a.type === 'montrer_lieu' ? `Voir ${a.nom || 'le point'} sur la carte` : 'Choisir cette activité';

  const suggestions = useMemo(() => {
    const ici = guide.taches.filter(t => t.vue === state.view).flatMap(t => t.questions);
    return {
      faire: (ici.length ? ici : guide.taches.flatMap(t => t.questions)).slice(0, 3),
      apprendre: ['Fais-moi le tour de l’application', ...guide.taches.filter(t => (t.vue === state.view || !t.vue) && t.id !== 'visite').slice(0, 2)
        .map(t => `Montre-moi comment ${t.titre.charAt(0).toLowerCase()}${t.titre.slice(1)}`)],
      projets: guide.projets.map(p => p.question),
    };
  }, [guide, state.view]);

  if (!open) return null;
  const etatTexte = etat === 'ia' ? 'Connecté aux données' : etat === 'guide' ? 'Guide hors ligne' : 'Vérification…';

  return <aside className="aw-assistant card" role="dialog" aria-modal="false" aria-labelledby="aw-assistant-titre" dir={rtl ? 'rtl' : 'ltr'}>
    <header className="aw-assistant-head">
      <span className="aw-assistant-mark" aria-hidden="true"><Sparkles size={17}/></span>
      <div><b id="aw-assistant-titre">Assistant ATOM Wind</b>
        <span className={`aw-assistant-etat is-${etat}`}><span aria-hidden="true"/>{etatTexte}</span></div>
      <button className="icon-button" onClick={() => { setFil([]); setPosition(null); }} aria-label="Nouvelle conversation" title="Nouvelle conversation"><RotateCcw size={16}/></button>
      <button className="icon-button" onClick={onClose} aria-label="Fermer l’assistant"><X size={18}/></button>
    </header>

    <div className="aw-assistant-fil" aria-live="polite">
      {!fil.length && <div className="aw-assistant-accueil">
        <p>Bonjour. Je peux lire le vent pour vous, vous montrer où cliquer et vous aider à préparer un projet. Je peux aussi ouvrir la bonne page et vous localiser sur la carte.</p>
        <Groupe icone={<Wrench size={15}/>} titre="Faire une tâche" items={suggestions.faire} choisir={envoyer}/>
        <Groupe icone={<BookOpen size={15}/>} titre="Apprendre à faire" items={suggestions.apprendre} choisir={envoyer}/>
        <Groupe icone={<Lightbulb size={15}/>} titre="Réfléchir à un projet" items={suggestions.projets} choisir={envoyer}/>
      </div>}
      {fil.map(m => <div key={m.id} className={`aw-msg is-${m.role}${m.erreur ? ' is-erreur' : ''}`}>
        {m.texte && <div className="aw-msg-corps" data-aw-no-i18n="" dir="auto">{m.role === 'user' ? <p>{m.texte}</p> : <Texte texte={m.texte}/>}
          {m.etapes && <ol className="aw-etapes">{m.etapes.map((e, i) => <li key={i}>{e}</li>)}</ol>}</div>}
        {!!m.faits?.length && <ul className="aw-faits">{m.faits.map((f, i) => <li key={i}><Compass size={13} aria-hidden="true"/>{f}</li>)}</ul>}
        {!!m.actions?.length && <div className="aw-actions">{m.actions.map((a, i) =>
          <button key={i} className="btn" onClick={() => cliquerAction(a)}>{a.type === 'demander_position' ? <LocateFixed size={15}/> : a.type === 'montrer' ? <MousePointer2 size={15}/> : <MapPin size={15}/>}{libelleAction(a)}</button>)}</div>}
        {m.role === 'assistant' && m.texte && <div className="aw-msg-pied">
          {m.mode === 'guide' ? 'Guide de l’application, sans calcul' : m.sources?.length ? `Sources ATOM Wind : ${m.sources.join(', ')}` : 'Réponse de l’assistant IA'}
        </div>}
        {!!m.relances?.length && <div className="aw-relances">{m.relances.map((r, i) =>
          <button key={i} onClick={() => envoyer(r)} disabled={!!attente}>{r}</button>)}</div>}
      </div>)}
      {attente && <div className="aw-msg is-assistant is-attente" role="status"><span className="aw-points" aria-hidden="true"><i/><i/><i/></span>{attente}</div>}
      <div ref={bas}/>
    </div>

    <form className="aw-assistant-saisie" onSubmit={e => { e.preventDefault(); envoyer(saisie); }}>
      <button type="button" className="icon-button" onClick={async () => {
        const p = await obtenirPosition(); setAttente(null);
        if (p) { const nom = window.ATOM?.workspace?.placer?.(p.lat, p.lon); ajouter({ role: 'assistant', texte: '', faits: [nom ? `Position trouvée (à ${p.precision_m} m près) : ${nom}.` : 'Votre position est en dehors du Grand Tunis.'], mode: 'guide' }); }
      }} aria-label="Me localiser" title="Me localiser" disabled={!!attente}><LocateFixed size={18}/></button>
      <textarea ref={champ} value={saisie} rows={1} maxLength={2000} placeholder="Votre question, votre projet…"
        aria-label="Votre question" onChange={e => setSaisie(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); envoyer(saisie); } }}/>
      <button className="aw-envoyer" type="submit" aria-label="Envoyer" disabled={!saisie.trim() || !!attente}><ArrowUp size={18}/></button>
    </form>
    <p className="aw-assistant-note">{etat === 'ia'
      ? 'Réponses d’une IA à partir des données d’ATOM Wind. Vérifiez avant une décision importante.'
      : 'Le service ne répond pas : je vous guide dans l’application, sans donner de chiffres.'}</p>
  </aside>;
}

function Groupe({ icone, titre, items, choisir }: { icone: ReactNode; titre: string; items: string[]; choisir: (q: string) => void }) {
  if (!items.length) return null;
  return <div className="aw-groupe"><span className="aw-groupe-titre">{icone}{titre}</span>
    {items.map(q => <button key={q} onClick={() => choisir(q)}>{q}</button>)}</div>;
}
