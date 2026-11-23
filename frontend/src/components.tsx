import { useCallback, useEffect, useState } from 'react';
import { AssistantPanel } from './assistant/AssistantPanel';
import { Map as IconeCarte, ArrowDownToLine, ArrowUpRight, AudioLines, Box, ChevronDown, ChevronRight, Compass, Gauge, Layers3, Leaf, LifeBuoy, MapPin, Menu, Search, Settings2, ShieldCheck, Sparkles, Wind, X, Bell, Users, Activity, Command, Wrench, History } from 'lucide-react';
import type { WorkspaceState } from './workspace';

export function Mark({ small = false }: { small?: boolean }) {
  return <a className={`atom-brand ${small ? 'small' : ''}`} href="/app/atom-wind/" aria-label="ATOM Wind — accueil">
    <svg className="wind-mark" width="31" height="32" viewBox="0 0 36 32" fill="none" aria-hidden="true"><path d="M2 8H24C33 8 33 1 28 2M2 16H29C37 16 35 25 29 24M2 24H17C23 24 24 30 19 30" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
    <span className="brand-word">atom<span className="brand-wind">wind</span></span>
  </a>;
}

const navigation = [
  { label: 'Territoire', items: [
    { id: 'tableau', name: 'Vue d’ensemble', icon: Gauge },
    { id: 'alertes', name: 'Alertes', icon: Bell },
  ] },
  { label: 'Mes activités', items: [
    { id: 'agri', name: 'Agriculture', icon: Leaf },
    { id: 'eolien', name: 'Énergie éolienne', icon: Activity },
    { id: 'outils', name: 'Outils de terrain', icon: Wrench },
  ] },
  { label: 'Comprendre', items: [
    { id: 'carte3d', name: 'Atmosphère 3D', icon: Box },
    { id: 'historique', name: 'Historique', icon: History },
  ] },
  { label: 'Gestion', items: [{ id: 'admin', name: 'Administration', icon: Users }, { id: 'reseau', name: 'Réseau ANEO', icon: Layers3 }] },
];

export function Navigation({ open, close }: { open: boolean; close: () => void }) {
  return <>
    {open && <button className="rail-backdrop" onClick={close} aria-label="Fermer la navigation" />}
    <aside id="workspace-navigation" aria-label="Navigation principale" role={open ? 'dialog' : undefined} aria-modal={open || undefined} className={`workspace-rail ${open ? 'is-open' : ''}`}>
      <div className="rail-brand"><Mark/><button className="icon-button rail-close" onClick={close} aria-label="Fermer"><X size={20}/></button></div>
      <div className="rail-territory"><span className="territory-symbol"><MapPin size={17}/></span><div><b>Tunisie</b><span>24 gouvernorats</span></div></div>
      <nav className="tabs workspace-navigation" aria-label="Espaces">
        {navigation.map(group => <div className="nav-group" key={group.label}><span className="nav-label">{group.label}</span>{group.items.map(item => <a href={`#${item.id}`} data-v={item.id} key={item.id} onClick={close}><item.icon size={18} strokeWidth={1.7}/><span>{item.name}</span>{item.id === 'alertes' && <span className="count" id="alertcount">0</span>}</a>)}</div>)}
      </nav>
      <div className="rail-bottom"><button className="rail-help" id="retour"><LifeBuoy size={17}/>Prendre ses repères<ChevronRight size={14}/></button>
        
      </div>
    </aside>
  </>;
}

// barre de navigation du bas sur téléphone (comme une application) : mêmes espaces et mêmes droits que le menu
// (classe « tabs » : atom-runtime.js masque les espaces hors du rôle et marque l'espace ouvert)
const BAS = [
  { id: 'tableau', name: 'Accueil', icon: Gauge }, { id: 'monvent', name: 'Mon vent', icon: MapPin },
  { id: 'carte3d', name: '3D', icon: Box }, { id: 'alertes', name: 'Alertes', icon: Bell },
];
export function BarreMobile({ onMenu }: { onMenu: () => void }) {
  return <nav className="tabs aw-barre-mobile" aria-label="Navigation rapide">
    {BAS.map(i => <a key={i.id} href={`#${i.id}`} data-v={i.id}><i.icon size={20} strokeWidth={1.8}/><span>{i.name}</span></a>)}
    <button type="button" onClick={onMenu} aria-label="Tous les espaces"><Menu size={20} strokeWidth={1.8}/><span>Plus</span></button>
  </nav>;
}

export function Preferences() {
  return <details className="workspace-preferences"><summary className="icon-button" aria-label="Préférences d’affichage"><Settings2 size={19}/></summary><div className="preferences-panel">
    <b>Votre espace de travail</b>
    <label>Unités de vitesse</label><div className="seg" role="group" aria-label="Unité de vitesse"><button id="u-kmh" aria-pressed="true">km/h</button><button id="u-ms" aria-pressed="false">m/s</button></div>
    <label>Apparence</label><div className="seg" role="group" aria-label="Thème"><button id="th-auto" aria-pressed="true">Auto</button><button id="th-light" aria-pressed="false">Clair</button><button id="th-dark" aria-pressed="false">Sombre</button></div>
    <label>Langue</label><div className="seg" role="group" aria-label="Langue"><button data-lang="fr" aria-pressed="true">FR</button><button data-lang="ar" aria-pressed="false" lang="ar">عربي</button><button data-lang="en" aria-pressed="false" lang="en">EN</button></div>
    <button className="btn" id="b-sound" aria-pressed="false"><AudioLines size={16}/>Écouter le vent</button>
  </div></details>;
}

export function Topbar({ onMenu, menuOpen = false }: { onMenu: () => void; menuOpen?: boolean }) {
  const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);
  return <div id="topbar" className="top workspace-topbar">
    <button className="icon-button mobile-menu" onClick={onMenu} aria-label="Ouvrir la navigation" aria-controls="workspace-navigation" aria-expanded={menuOpen}><Menu size={21}/></button>
    <Mark/><div className="breadcrumb"><span>Tunisie</span><ChevronRight size={13}/><span>Le vent et vos activités</span></div>
    <div className="topbar-actions"><button className="search-trigger" id="b-search" aria-label="Rechercher"><Search size={17}/><span>Rechercher</span><kbd>{isMac ? <Command size={11}/> : "Ctrl"} K</kbd></button><Preferences/><div className="userbox umenu" id="userbox"/></div>
  </div>;
}

const pageInfo: Record<string, { eyebrow: string; title: string; description: string }> = {
  gouvernorats: { eyebrow: 'Tunisie', title: 'Les 24 gouvernorats', description: 'Le vent de chaque gouvernorat, heure par heure : touchez la carte pour ouvrir un gouvernorat.' },
  monvent: { eyebrow: '', title: 'Vue d’ensemble', description: 'Repérez les conditions locales, puis préparez votre activité.' },
  agri: { eyebrow: '', title: 'Mes cultures', description: 'Préparez vos traitements et votre arrosage à partir des conditions de la journée.' },
  tableau: { eyebrow: 'Tunisie', title: 'Vue d’ensemble', description: 'Repérez les conditions locales, puis préparez votre activité.' },
  carte3d: { eyebrow: '', title: 'Carte 3D', description: 'Explorez le vent à différentes hauteurs et inspectez un point.' },
  eolien: { eyebrow: '', title: 'Éolien', description: 'Production estimée, emplacement des sites et vent à hauteur de moyeu.' },
  alertes: { eyebrow: '', title: 'Les alertes à suivre', description: 'Où, quand et quelles précautions prendre.' },
  admin: { eyebrow: '', title: 'Administration', description: 'Les comptes, les accès et l’état du service.' },
  outils: { eyebrow: '', title: 'Outils de terrain', description: 'Levage, haies, mâts autonomes : le vent appliqué à vos chantiers.' },
  historique: { eyebrow: '', title: 'Historique', description: 'Les journées et les prévisions déjà calculées par le service.' },
  reseau: { eyebrow: 'Administration', title: 'Réseau ANEO', description: 'Planification nationale des capteurs, et installation zone par zone : cliquez une zone de la carte.' },
};

/** « Gouvernorats » et « Mon vent » forment une seule page : deux contenus, un bouton pour passer de l'un à l'autre.
 *  Le gouvernorat suit : ouvert dans Gouvernorats, Mon vent s'ouvre agrandi sur lui (point au chef-lieu s'il était
 *  ailleurs) ; agrandi dans Mon vent, Gouvernorats s'ouvre sur lui. Les vues restent séparées (#gouvernorats, #monvent). */
export function BasculeGouvernorats({ vue }: { vue: string }) {
  // Vue d'ensemble (22/11/2026) : « Gouvernorats » montre la carte et la fiche des gouvernorats (#tableau),
  // « Mon vent » montre Mon vent au même endroit (#monvent) ; la carte du bulletin, le bandeau et les chiffres restent
  const actif = vue === 'monvent' ? 'monvent' : 'gouvernorats';
  const aller = (vers: 'gouvernorats' | 'monvent') => {
    if (vers === actif) return;
    // Mon vent s'ouvre sur toute la Tunisie (demande du 22/11/2026) : pas de zoom sur un gouvernorat
    if (vers === 'monvent') document.getElementById('bm-tunisie')?.click();
    else window.dispatchEvent(new CustomEvent('atom:bascule-gouvernorats'));
    window.ATOM?.workspace?.navigate(vers === 'monvent' ? 'monvent' : 'tableau');
    setTimeout(() => document.querySelector('.overview-bascule')?.scrollIntoView({ block: 'start', behavior: 'smooth' }), 450);
  };
  return <div className="seg gv-bascule" role="group" aria-label="Contenu de la page">
    <button aria-pressed={actif === 'gouvernorats'} onClick={() => aller('gouvernorats')}><IconeCarte size={15}/>Gouvernorats</button>
    <button aria-pressed={actif === 'monvent'} onClick={() => aller('monvent')}><MapPin size={15}/>Mon vent</button>
  </div>;
}

export function PageHeading({ state }: { state: WorkspaceState }) {
  const info = pageInfo[state.view] || pageInfo.monvent;
  return <header className="page-heading"><div><div className="eyebrow"><span/>{info.eyebrow}</div><h1>{info.title}</h1><p>{info.description}</p></div><a href={state.view === 'carte3d' ? '#monvent' : '#carte3d'} className="explore-link">{state.view === 'carte3d' ? <Wind size={17}/> : <Box size={17}/>}<span>{state.view === 'carte3d' ? 'Mon vent' : 'Carte 3D'}</span><ArrowUpRight size={16}/></a></header>;
}

export function DataContext({ state }: { state: WorkspaceState }) {
  return <section className="data-context" aria-label="Source et contexte temporel"><div className="source-line"><span className={`source-indicator ${state.source}`}><span className="status-dot"/><span id="aw-fresh">Connexion au service…</span></span><span className="context-separator"/><span>Heure de Tunis · UTC+1</span><select id="aw-jour" hidden aria-label="Journée"/></div>
    <div className="timebar"><label htmlFor="hour">Journée</label><div className="skywrap"><span className="sun day" id="sun" aria-hidden="true"/><div className="sky" aria-hidden="true"/><input type="range" id="hour" min="0" max="23" step="1" defaultValue="15" aria-label="Heure de la journée"/></div><span className="hourbig" id="hourtxt">15 h</span><button className="btn" id="play" aria-label="Rejouer la journée">Rejouer la journée</button></div>
    <details className="provenance"><summary><ShieldCheck size={14}/>À propos des données<ChevronDown size={13}/></summary><div className="proto" id="aw-source">Connexion au service ATOM Wind…</div></details>
  </section>;
}

export function WindReadout({ state }: { state: WorkspaceState }) {
  if (!['monvent', 'carte3d'].includes(state.view)) return null;
  return <div className="wind-readout"><div className="readout-location"><span className="readout-kicker"><MapPin size={13}/>POINT SÉLECTIONNÉ</span><b>{state.location}</b><span>{state.source === 'dernier_etat' ? 'Dernier état réel connu' : 'Vent calculé par ATOM Wind'}</span></div><div className="readout-metric"><span>Vitesse du vent</span><strong>{state.speed}</strong></div><div className="readout-direction"><span>Provenance</span><b><Compass size={20}/>{state.direction}</b></div><div className="readout-height"><span>Hauteur de lecture</span><b>{state.altitude} <small>{state.heightReference === 'asl' ? 'm ASL' : 'm du sol'}</small></b></div><button className="inspect-button" onClick={() => window.ATOM?.workspace?.inspect()}><span>Inspecter le point</span><ArrowUpRight size={17}/></button></div>;
}

export function Assistant({ state }: { state: WorkspaceState }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  // après une démonstration au curseur, le panneau revient
  useEffect(() => { const ouvrir = () => setOpen(true); window.addEventListener('atom:assistant-ouvrir', ouvrir); return () => window.removeEventListener('atom:assistant-ouvrir', ouvrir); }, []);
  return <>
    <button className="assistant-entry" onClick={() => setOpen(!open)} aria-expanded={open}><Sparkles size={17}/><span>Assistant</span><span className="assistant-shortcut">Poser une question</span><ChevronRight size={15}/></button>
    {!open && <button className="aw-assistant-fab" onClick={() => setOpen(true)} aria-label="Ouvrir l’assistant"><Sparkles size={18}/><span>Assistant</span></button>}
    <AssistantPanel state={state} open={open} onClose={close}/>
  </>;
}
