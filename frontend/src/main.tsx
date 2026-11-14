import { useState, useEffect, Component, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/manrope';
import '@fontsource-variable/fraunces';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-serif/latin-400.css';
import '@fontsource/ibm-plex-serif/latin-500.css';
import './styles/atmos.css';
import './styles/workspace-layout.css';
import './styles/terrain.css';
import InstrumentViews from './views/InstrumentViews';
import { Navigation, Topbar, PageHeading, DataContext, WindReadout, Assistant } from './components';
import { useWorkspace } from './workspace';
import { ServiceState, VUES_DONNEES } from './service-state';
import { initializeTheme } from './theme';

class ErrorBoundary extends Component<{children: ReactNode}, {error: boolean}> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() { return this.state.error ? <div className="startup-error"><h1>L’espace n’a pas pu s’ouvrir.</h1><p>Rechargez la page pour rétablir la connexion.</p><button onClick={() => location.reload()}>Réessayer</button></div> : this.props.children; }
}

function App() {
  const { state, error } = useWorkspace();
  const [menu, setMenu] = useState(false);
  // sans données de vent, les vues qui en dépendent cèdent la place à l'état du service
  const gated = state.ready && state.source === 'hors_ligne' && VUES_DONNEES.has(state.view);
  const national = state.view === 'carte3d' && state.sceneMode !== 'analysis' && state.terrainSource !== 'local';
  useEffect(() => {
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setMenu(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);
  useEffect(() => {
    if (!menu) return;
    const narrow = window.matchMedia('(max-width: 900px)');
    if (!narrow.matches) { setMenu(false); return; }
    const previous = document.activeElement as HTMLElement | null;
    const drawer = document.getElementById('workspace-navigation');
    const background = [document.getElementById('workspace'), document.getElementById('topbar')];
    background.forEach(node => { if (node) node.inert = true; });
    drawer?.querySelector<HTMLButtonElement>('.rail-close')?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !drawer) return;
      const items = [...drawer.querySelectorAll<HTMLElement>('a[href],button')].filter(node => node.offsetParent && !node.hidden);
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    const resize = () => { if (!narrow.matches) setMenu(false); };
    document.addEventListener('keydown', trap);
    narrow.addEventListener('change', resize);
    return () => {
      document.removeEventListener('keydown', trap);
      narrow.removeEventListener('change', resize);
      background.forEach(node => { if (node) node.inert = false; });
      previous?.focus({ preventScroll: true });
    };
  }, [menu]);
  return <ErrorBoundary><a className="skip-link" href="#workspace">Aller à l’espace de travail</a><div className="workspace-main" data-view={state.view} data-gated={gated || undefined}><Topbar onMenu={() => setMenu(!menu)} menuOpen={menu} national={national}/><Navigation open={menu} close={() => setMenu(false)} national={national}/><main className="workspace-content" id="workspace" tabIndex={-1}><PageHeading state={state}/><DataContext state={state}/>{error && <div className="startup-error" role="alert">L’espace n’a pas pu démarrer : {error} <button onClick={() => location.reload()}>Réessayer</button></div>}<WindReadout state={state}/><div className="instruments" data-gated={gated || undefined}>{gated && <ServiceState state={state}/>}<InstrumentViews/></div><footer className="workspace-footer"><span>ATOM Wind <span> / </span> PC-FNO3D</span><span>{national ? 'Tunisie · Prévisions GFS · Outils locaux dans le Grand Tunis' : 'Grand Tunis · Altitudes et sources précisées dans chaque vue'}</span><Assistant state={state}/></footer></main></div><aside className="sheet card" id="sheet" hidden/><div className="tour" id="tour" hidden/><div className="tour" id="authm" hidden/><div className="toast" id="toast" role="status" hidden/><div className="gtip" id="gtip" hidden/><div className="palette" id="palette" hidden/></ErrorBoundary>;
}

initializeTheme();
createRoot(document.getElementById('root')!).render(<App/>);
