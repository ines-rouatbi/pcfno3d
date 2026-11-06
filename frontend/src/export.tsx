import { memo, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/manrope';
import '@fontsource/ibm-plex-mono/latin-400.css';
import './styles/science.css';
import markup from './views/export-instruments.html?raw';
import '@fontsource/ibm-plex-serif/latin-400.css';
import { Mark } from './components';
import { initializeTheme } from './theme';

// Binary export reading remains isolated from React reconciliation. This surface
// can also be opened at /web/ using the generated standalone entry point.
const ExportInstruments = memo(() => <div className="export-instruments" dangerouslySetInnerHTML={{__html: markup}}/>);
function ExportApp() {
  useEffect(() => { import('./engine/export-runtime.js'); }, []);
  return <><div className="science-masthead"><Mark small/><a href="/app/">Ouvrir le laboratoire ↗</a></div><div className="science-intro"><span className="eyebrow">VISUALISEUR D’EXPORT · PC-FNO3D</span><h1>Lire un export</h1><p>Les valeurs du fichier exporté, par niveau et par coupe. Altitudes en mètres au-dessus de la mer.</p></div><ExportInstruments/></>;
}
initializeTheme();
createRoot(document.getElementById('root')!).render(<ExportApp/>);
