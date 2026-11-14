import { useEffect, useState } from 'react';
import { installDialogFocus } from './accessibility';

export type WorkspaceState = {
  ready: boolean;
  view: string;
  hour: number;
  location: string;
  speed: string;
  direction: string;
  altitude: number;
  heightReference?: 'agl' | 'asl';
  source: 'demonstration' | 'modele';
  sceneMode?: 'terrain' | 'analysis';
  terrainSource?: 'national' | 'local';
};

export const initialWorkspace: WorkspaceState = {
  ready: false, view: 'monvent', hour: 15, location: 'Grand Tunis',
  speed: '—', direction: '—', altitude: 10, source: 'demonstration',
};

declare global {
  interface Window {
    ATOM: {
      [key: string]: any;
      workspace?: {
        navigate: (view: string) => void;
        inspect: () => void;
        snapshot: () => WorkspaceState;
        terrainWind?: (longitude: number, latitude: number, height: number) => {u:number;v:number;speed:number;formatted:string;direction:string;color:number[]}|null;
        terrainPoint?: (longitude: number, latitude: number) => void;
      };
    };
    THREE: any;
  }
}

let startup: Promise<void> | undefined;
export function startApplication() {
  return startup ??= (async () => {
    installDialogFocus();
    await import('./engine/atom-i18n.js');
    const { startWorkspace } = await import('./engine/atom-runtime.js');
    window.ATOM.startPage = startWorkspace;
    window.ATOM.loadThree = async () => {
      const [three, { OrbitControls }] = await Promise.all([
        import('three'), import('three/addons/controls/OrbitControls.js'),
      ]);
      window.THREE = { ...three, OrbitControls };
    };
    window.ATOM.loadTerrain = async () => {
      const { mountTerrain } = await import('./terrain/terrain-viewer');
      await mountTerrain(document.getElementById('terrain-host')!);
    };
    window.ATOM.loadTurbine = async (host: HTMLElement, opts: { label: string }) => {
      const { mountTurbine } = await import('./terrain/turbine-view');
      return mountTurbine(host, opts);
    };
    await import('./engine/atom-3d.js');
    // onglet d'administration « Réseau ANEO » : avant atom-data.js, qui lance la page
    await import('./engine/reseau-aneo.js');
    // fonctions du service sans écran jusque-là (sillage, maintenance, alertes du service, SMS, exports…)
    await import('./engine/services-atom.js');
    // installation des capteurs ANEO par zone (proposition d'emplacements dans une zone choisie sur la carte)
    await import('./engine/installation-aneo.js');
    await import('./engine/atom-data.js');
  })();
}

export function useWorkspace() {
  const [state, setState] = useState(initialWorkspace);
  const [error, setError] = useState('');
  useEffect(() => {
    const update = (event: Event) => setState((event as CustomEvent<WorkspaceState>).detail);
    window.addEventListener('atom:state', update);
    startApplication().catch(err => setError(String(err)));
    return () => window.removeEventListener('atom:state', update);
  }, []);
  return { state, error };
}
