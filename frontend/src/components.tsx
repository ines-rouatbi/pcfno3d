import { useState } from 'react';
import { ArrowDownToLine, ArrowUpRight, AudioLines, BookOpen, Box, ChevronDown, ChevronRight, Compass, Gauge, Layers3, Leaf, LifeBuoy, MapPin, Menu, Search, Settings2, ShieldCheck, Sparkles, Wind, X, Bell, Users, Activity, Command } from 'lucide-react';
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
    { id: 'monvent', name: 'Mon vent', icon: MapPin },
    { id: 'alertes', name: 'Alertes', icon: Bell },
  ] },
  { label: 'Mes activités', items: [
    { id: 'agri', name: 'Agriculture', icon: Leaf },
    { id: 'eolien', name: 'Énergie éolienne', icon: Activity },
  ] },
  { label: 'Comprendre', items: [
    { id: 'carte3d', name: 'Atmosphère 3D', icon: Box },
    { id: 'fiabilite', name: 'Données & fiabilité', icon: ShieldCheck },
  ] },
  { label: 'Gestion', items: [{ id: 'admin', name: 'Administration', icon: Users }] },
];

export function Navigation({ open, close, national = false }: { open: boolean; close: () => void; national?: boolean }) {
  return <>
    {open && <button className="rail-backdrop" onClick={close} aria-label="Fermer la navigation" />}
    <aside id="workspace-navigation" aria-label="Navigation principale" role={open ? 'dialog' : undefined} aria-modal={open || undefined} className={`workspace-rail ${open ? 'is-open' : ''}`}>
      <div className="rail-brand"><Mark/><button className="icon-button rail-close" onClick={close} aria-label="Fermer"><X size={20}/></button></div>
      <div className="rail-territory"><span className="territory-symbol"><MapPin size={17}/></span><div><b>{national ? 'Tunisie' : 'Grand Tunis'}</b><span>{national ? 'Prévisions nationales' : 'Territoire d’étude'}</span></div></div>
      <nav className="tabs workspace-navigation" aria-label="Espaces">
        {navigation.map(group => <div className="nav-group" key={group.label}><span className="nav-label">{group.label}</span>{group.items.map(item => <a href={`#${item.id}`} data-v={item.id} key={item.id} onClick={close}><item.icon size={18} strokeWidth={1.7}/><span>{item.name}</span>{item.id === 'alertes' && <span className="count" id="alertcount">0</span>}</a>)}</div>)}
      </nav>
      <div className="rail-bottom"><a className="lab-link" href="/app/"><Layers3 size={18}/><div><b>Espace scientifique</b><small>PC-FNO3D</small></div><ArrowUpRight size={16}/></a>
        <button className="rail-help" id="retour"><LifeBuoy size={17}/>Prendre ses repères<ChevronRight size={14}/></button>
        
      </div>
    </aside>
  </>;
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

export function Topbar({ onMenu, menuOpen = false, national = false }: { onMenu: () => void; menuOpen?: boolean; national?: boolean }) {
  const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);
  return <div id="topbar" className="top workspace-topbar">
    <button className="icon-button mobile-menu" onClick={onMenu} aria-label="Ouvrir la navigation" aria-controls="workspace-navigation" aria-expanded={menuOpen}><Menu size={21}/></button>
    <Mark/><div className="breadcrumb"><span>{national ? 'Tunisie' : 'Grand Tunis'}</span><ChevronRight size={13}/><span>Le vent et vos activités</span></div>
    <div className="topbar-actions"><button className="search-trigger" id="b-search" aria-label="Rechercher"><Search size={17}/><span>Rechercher</span><kbd>{isMac ? <Command size={11}/> : "Ctrl"} K</kbd></button><Preferences/><div className="userbox umenu" id="userbox"/></div>
  </div>;
}

const pageInfo: Record<string, { eyebrow: string; title: string; description: string }> = {
  monvent: { eyebrow: '', title: 'Mon vent', description: 'Choisissez un lieu et une activité pour lire les conditions qui vous concernent.' },
  agri: { eyebrow: '', title: 'Mes cultures', description: 'Préparez vos traitements et votre arrosage à partir des conditions de la journée.' },
  tableau: { eyebrow: 'Grand Tunis', title: 'Vue d’ensemble', description: 'Repérez les conditions locales, puis préparez votre activité.' },
  carte3d: { eyebrow: '', title: 'Carte 3D', description: 'Explorez le vent à différentes hauteurs et inspectez un point.' },
  eolien: { eyebrow: '', title: 'Éolien', description: 'Production estimée, emplacement des sites et vent à hauteur de moyeu.' },
  fiabilite: { eyebrow: '', title: 'Ce que l’on sait du vent', description: 'Les stations, la couverture et les écarts aux mesures.' },
  alertes: { eyebrow: '', title: 'Les alertes à suivre', description: 'Où, quand et quelles précautions prendre.' },
  admin: { eyebrow: '', title: 'Administration', description: 'Les comptes et les accès de votre organisation.' },
};

export function PageHeading({ state }: { state: WorkspaceState }) {
  const info = pageInfo[state.view] || pageInfo.monvent;
  return <header className="page-heading"><div><div className="eyebrow"><span/>{info.eyebrow}</div><h1>{info.title}</h1><p>{info.description}</p></div><a href={state.view === 'carte3d' ? '#monvent' : '#carte3d'} className="explore-link">{state.view === 'carte3d' ? <Wind size={17}/> : <Box size={17}/>}<span>{state.view === 'carte3d' ? 'Mon vent' : 'Carte 3D'}</span><ArrowUpRight size={16}/></a></header>;
}

export function DataContext({ state }: { state: WorkspaceState }) {
  return <section className="data-context" aria-label="Source et contexte temporel"><div className="source-line"><span className={`source-indicator ${state.source}`}><span className="status-dot"/><span id="aw-fresh">Démonstration : journée d’exemple</span></span><span className="context-separator"/><span>Heure de Tunis · UTC+1</span><select id="aw-jour" hidden aria-label="Journée"/></div>
    <div className="timebar"><label htmlFor="hour">Journée</label><div className="skywrap"><span className="sun day" id="sun" aria-hidden="true"/><div className="sky" aria-hidden="true"/><input type="range" id="hour" min="0" max="23" step="1" defaultValue="15" aria-label="Heure de la journée"/></div><span className="hourbig" id="hourtxt">15 h</span><button className="btn" id="play" aria-label="Rejouer la journée">Rejouer la journée</button></div>
    <details className="provenance"><summary><ShieldCheck size={14}/>À propos des données<ChevronDown size={13}/></summary><div className="proto" id="aw-source">Journée d’exemple d’été. Ce ne sont pas des mesures en direct.</div></details>
  </section>;
}

export function WindReadout({ state }: { state: WorkspaceState }) {
  if (!['monvent', 'carte3d'].includes(state.view)) return null;
  return <div className="wind-readout"><div className="readout-location"><span className="readout-kicker"><MapPin size={13}/>POINT SÉLECTIONNÉ</span><b>{state.location}</b><span>{state.source === 'demonstration' ? 'Valeurs de démonstration' : 'Vent calculé par le modèle'}</span></div><div className="readout-metric"><span>Vitesse du vent</span><strong>{state.speed}</strong></div><div className="readout-direction"><span>Provenance</span><b><Compass size={20}/>{state.direction}</b></div><div className="readout-height"><span>Hauteur de lecture</span><b>{state.altitude} <small>{state.heightReference === 'asl' ? 'm ASL' : 'm du sol'}</small></b></div><button className="inspect-button" onClick={() => window.ATOM?.workspace?.inspect()}><span>Inspecter le point</span><ArrowUpRight size={17}/></button></div>;
}

export function Assistant() {
  const [open, setOpen] = useState(false);
  const toggle = () => {
    const button = document.getElementById('aw-asst-btn');
    if (button) button.click(); else setOpen(!open);
  };
  return <><button className="assistant-entry" onClick={toggle}><Sparkles size={17}/><span>Assistant</span><span className="assistant-shortcut">Poser une question</span><ChevronRight size={15}/></button>{open && <aside className="assistant-unavailable card"><div className="row"><Sparkles size={18}/><b>Assistant ATOM Wind</b><button className="icon-button" onClick={() => setOpen(false)} aria-label="Fermer l’assistant"><X size={18}/></button></div><p>L’assistant nécessite une connexion au service PC-FNO3D. Les valeurs de cette journée sont des exemples.</p><a href="/app/" className="btn"><BookOpen size={15}/>Ouvrir le laboratoire</a></aside>}</>;
}
