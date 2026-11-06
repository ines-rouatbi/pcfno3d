import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/manrope';
import '@fontsource/ibm-plex-mono/400.css';
import './styles/science.css';
import ScienceViews from './views/ScienceViews';
import '@fontsource/ibm-plex-serif/latin-400.css';
import { Mark } from './components';
import { initializeTheme } from './theme';

function ScienceApp() {
  useEffect(() => { import('./engine/science-runtime.js'); }, []);
  return <><div className="science-masthead"><Mark small/><a href="/app/atom-wind/">Retour à l’espace ATOM Wind ↗</a></div><div className="science-intro"><span className="eyebrow">OUTIL INTERNE · ÉQUIPE MODÈLE</span><h1>Laboratoire PC-FNO3D</h1><p>Champ de vent, profils verticaux et qualité de reconstruction.</p></div><ScienceViews/></>;
}
initializeTheme();
createRoot(document.getElementById('root')!).render(<ScienceApp/>);
