// Appels de l'API PC-FNO3D depuis les vues React : mêmes en-têtes (session ou clé) et même adresse que la
// couche de données du moteur (atom-data.js). Une erreur porte le message du service, prêt à afficher.
import { useEffect, useState } from 'react';
import type { WorkspaceState } from './workspace';

export class ErreurApi extends Error {
  constructor(message: string, readonly statut: number) { super(message); }
}

const MESSAGES: Record<string, string> = {
  'modele non installe sur ce service': 'Le modèle de vent n’est pas encore installé sur ce service.',
};

export async function api<T = any>(chemin: string, init: RequestInit = {}, ms = 120000): Promise<T> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  const entetes: Record<string, string> = { ...(window.ATOM?.entetes?.() ?? {}) };
  if (init.body) entetes['Content-Type'] = 'application/json';
  try {
    const r = await fetch(`${window.ATOM?.apiBase ?? ''}/api${chemin}`, { ...init, headers: { ...entetes, ...(init.headers as any) }, signal: ctl.signal });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const m = typeof d.detail === 'string' ? d.detail : d.message || `erreur ${r.status}`;
      throw new ErreurApi(MESSAGES[m] ?? m, r.status);
    }
    return d as T;
  } catch (e) {
    if (e instanceof ErreurApi) throw e;
    throw new ErreurApi(ctl.signal.aborted ? 'Le service met trop de temps à répondre.' : 'Le service ATOM Wind ne répond pas.', 0);
  } finally { clearTimeout(t); }
}

/** L'état publié par le moteur (vue ouverte, source des données, rôle) : pour charger une vue quand elle s'ouvre. */
export function useEtatPage(): WorkspaceState | null {
  const [etat, setEtat] = useState<WorkspaceState | null>(() => typeof window === 'undefined' ? null : window.ATOM?.workspace?.snapshot?.() ?? null);
  useEffect(() => {
    const maj = (e: Event) => setEtat((e as CustomEvent<WorkspaceState>).detail);
    window.addEventListener('atom:state', maj);
    return () => window.removeEventListener('atom:state', maj);
  }, []);
  return etat;
}

export type Lieu = { nom: string; lat: number; lon: number };
export const lieux = (): Lieu[] => typeof window === 'undefined' ? [] : window.ATOM?.workspace?.lieux?.() ?? [];
export const nomCourt = (n: string) => n.split(' (')[0];
export const kmh = (ms: number) => Math.round(ms * 3.6);
export const versLaCarte = (l: { lat: number; lon: number }) => window.ATOM?.workspace?.placer?.(l.lat, l.lon);
