// Logique de l'assistant ATOM Wind, sans React : guide de l'application (mode hors ligne), appel de
// POST /api/assistant, géolocalisation et exécution des actions d'interface (liste blanche).
// Le guide JSON est le même fichier que celui de service/assistant.py (outil guide_application).

export type Vue = 'monvent' | 'agri' | 'tableau' | 'carte3d' | 'eolien' | 'fiabilite' | 'alertes' | 'admin' | 'outils' | 'historique';
export const VUES: Vue[] = ['monvent', 'agri', 'tableau', 'carte3d', 'eolien', 'fiabilite', 'alertes', 'admin', 'outils', 'historique'];

import type { EtapeDemo } from './curseur';
export type Tache = { id: string; titre: string; vue: Vue | null; mots_cles: string[]; etapes: string[]; questions: string[]; demo?: EtapeDemo[] };
export type Guide = {
  vues: Record<string, { nom: string; resume: string }>;
  taches: Tache[];
  projets: { titre: string; question: string }[];
};

export type Action =
  | { type: 'ouvrir_vue'; vue: Vue; libelle?: string; auto: boolean }
  | { type: 'montrer_lieu'; lat: number; lon: number; nom?: string | null; auto: boolean }
  | { type: 'choisir_activite'; activite: 'traiter' | 'arroser' | 'eolienne'; auto: boolean }
  | { type: 'demander_position'; raison?: string; auto: boolean }
  | { type: 'montrer'; tache: string; titre?: string; auto: boolean };

export type Reponse = {
  reponse: string;
  outils?: { nom: string }[];
  actions?: Action[];
  relances?: string[];
  etapes?: { titre: string; liste: string[] };   // étapes officielles du guide (question « comment »)
  modele?: string;
  arret?: string;
};

export type Contexte = {
  vue: string; lieu: string; heure: number; langue: string; role?: string;
  lat_utilisateur?: number; lon_utilisateur?: number;
};

const VIDE: Guide = { vues: {}, taches: [], projets: [] };
let guideCharge: Promise<Guide> | undefined;
export function chargerGuide(): Promise<Guide> {
  return guideCharge ??= fetch(`${import.meta.env.BASE_URL}atom-wind/assistant-guide.json`)
    .then(r => (r.ok ? r.json() : VIDE))
    .catch(() => VIDE);
}

export function normaliser(t: string): string {
  return t.toLowerCase().replace(/[’`]/g, "'").normalize('NFD').replace(/[̀-ͯ]/g, '');
}

/** Les tâches du guide les plus proches du texte (mêmes règles que service/assistant.chercher_guide). */
export function chercherGuide(guide: Guide, texte: string, n = 2): Tache[] {
  const q = ` ${normaliser(texte)} `;
  return guide.taches
    .map((t, k) => {
      let note = t.mots_cles.filter(m => q.includes(normaliser(m))).length * 3;
      note += (normaliser(t.titre).match(/\w{5,}/g) || []).filter(m => q.includes(m)).length;
      return { t, k, note };
    })
    .filter(x => x.note > 0)
    .sort((a, b) => b.note - a.note || a.k - b.k)
    .slice(0, n)
    .map(x => x.t);
}

// ------------------------------------------------------------------ intentions reconnues localement
const POSITION = /(chez moi|ma position|pr[eè]s de (moi|chez moi)|autour de moi|o[uù] je suis|\bici\b|ma parcelle|mon champ|mon terrain|my location|near me|around me|where i am|عندي|مكاني|قربي|حذايا|موقعي)/i;
export const parleDeSaPosition = (q: string) => POSITION.test(q);

const OUVRIR = /\b(ouvr|affich|va |aller|allons|emm[eè]ne|passe|open|go to|take me)/i;
const NOMS_VUES: [Vue, RegExp][] = [
  ['agri', /agricultur|parcelle|cultures?\b|زراع/i],
  ['eolien', /[eé]olien|[eé]nergie|production|gisement|wind energy|طاقة/i],
  ['carte3d', /\b3d\b|atmosph[eè]re|relief|carte 3d/i],
  ['alertes', /alerte|alert|تنبيه/i],
  ['fiabilite', /fiabilit|donn[eé]es|reliab/i],
  ['tableau', /vue d.ensemble|tableau de bord|dashboard|overview/i],
  ['admin', /administr|comptes|admin/i],
  ['monvent', /mon vent|my wind/i],
  ['outils', /outils|grue|haie|chantier|tools/i],
  ['historique', /historique|history/i],
];
/** « Ouvre les alertes », « montre-moi l'éolien » : la vue demandée, sinon null. */
export function vueDemandee(q: string): Vue | null {
  if (!OUVRIR.test(q)) return null;
  return NOMS_VUES.find(([, re]) => re.test(q))?.[0] ?? null;
}

export type Lieu = { nom: string; lat: number; lon: number };
/** Un lieu de l'application nommé dans la question (« Mornag », « La Marsa »...). */
export function lieuNomme(q: string, lieux: Lieu[]): Lieu | null {
  const n = normaliser(q);
  return lieux.find(l => n.includes(normaliser(l.nom.split(' (')[0]))) ?? null;
}

// ------------------------------------------------------------------ service
function base(): string { return window.ATOM?.apiBase ?? ''; }
function entetes(): Record<string, string> {
  try { return { ...(window.ATOM?.entetes?.() ?? {}) }; } catch { return {}; }
}

/** Le service répond-il ? (route publique /api/health) */
export async function serviceJoignable(ms = 6000): Promise<boolean> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  try { const r = await fetch(`${base()}/api/health`, { signal: ctl.signal }); return r.ok; }
  catch { return false; }
  finally { clearTimeout(t); }
}

export class ErreurService extends Error {
  constructor(message: string, readonly statut: number) { super(message); }
}

export async function demander(question: string, historique: { role: string; contenu: string }[],
                               contexte: Contexte, ms = 120000): Promise<Reponse> {
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(`${base()}/api/assistant`, {
      method: 'POST', signal: ctl.signal,
      headers: { 'Content-Type': 'application/json', ...entetes() },
      body: JSON.stringify({ question, historique: historique.slice(-10), contexte }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new ErreurService(d.detail || d.message || `erreur ${r.status}`, r.status);
    return d as Reponse;
  } catch (e) {
    if (e instanceof ErreurService) throw e;
    throw new ErreurService(ctl.signal.aborted ? 'délai dépassé' : 'service injoignable', 0);
  } finally { clearTimeout(t); }
}

// ------------------------------------------------------------------ position de l'utilisateur
export type Position = { lat: number; lon: number; precision_m: number };
export function localiser(ms = 10000): Promise<Position> {
  return new Promise((ok, ko) => {
    if (!('geolocation' in navigator)) { ko(new Error('Ce navigateur ne donne pas la position.')); return; }
    navigator.geolocation.getCurrentPosition(
      p => ok({ lat: p.coords.latitude, lon: p.coords.longitude, precision_m: Math.round(p.coords.accuracy) }),
      e => ko(new Error(e.code === 1 ? 'Vous avez refusé le partage de la position.' : 'Position introuvable pour le moment.')),
      { enableHighAccuracy: true, timeout: ms, maximumAge: 300000 });
  });
}

// ------------------------------------------------------------------ actions sur la page
/** Exécute une action de la liste blanche ; rend une phrase à afficher (ou null si rien à dire). */
export function executer(a: Action, guide: Guide): string | null {
  const ws = window.ATOM?.workspace;
  if (!ws) return null;
  if (a.type === 'ouvrir_vue') {
    if (!VUES.includes(a.vue)) return null;
    const nom = guide.vues[a.vue]?.nom ?? a.vue;
    if (ws.allowed && !ws.allowed(a.vue)) return `L’espace « ${nom} » demande un compte avec le bon rôle : connectez-vous pour l’ouvrir.`;
    ws.navigate(a.vue);
    return `Espace « ${nom} » ouvert.`;
  }
  if (a.type === 'montrer_lieu') {
    if (!Number.isFinite(a.lat) || !Number.isFinite(a.lon)) return null;
    const nom = ws.placer?.(a.lat, a.lon);
    return nom ? `Point placé : ${nom}.` : 'Ce point est en dehors du Grand Tunis, la zone calculée par ATOM Wind.';
  }
  if (a.type === 'choisir_activite') {
    return ws.activite?.(a.activite) ? 'Activité choisie dans « Mon vent ».' : null;
  }
  return null;
}

// ------------------------------------------------------------------ réponse du guide, hors ligne
export type ReponseGuide = { texte: string; etapes?: string[]; actions: Action[]; relances: string[] };

const CHIFFRES = /(combien|quel(le)? (vent|vitesse|production|rafale)|peut-on|puis-je|est-ce que je peux|faut-il|demain|aujourd|maintenant|cr[eé]neau|km\/h|m\/s|kwh|how much|can i)/i;

/** Réponse sans le service : seulement le guide (où cliquer), jamais un chiffre de vent. */
export function repondreAvecLeGuide(guide: Guide, q: string, vueCourante: string, lieux: Lieu[]): ReponseGuide {
  const actions: Action[] = [];
  const vue = vueDemandee(q);
  const lieu = lieuNomme(q, lieux);
  if (lieu) actions.push({ type: 'montrer_lieu', lat: lieu.lat, lon: lieu.lon, nom: lieu.nom, auto: true });
  // une page demandée explicitement : la fiche du guide de cette page, sinon son résumé
  const t = vue ? guide.taches.find(x => x.vue === vue && chercherGuide(guide, q, 5).includes(x)) ?? guide.taches.find(x => x.vue === vue)
    : chercherGuide(guide, q, 1)[0];
  const cible = vue ?? t?.vue ?? null;
  // une tâche avec démonstration : le curseur ouvre lui-même la page et montre les étapes
  if (t?.demo && !vue) actions.unshift({ type: 'montrer', tache: t.id, titre: t.titre, auto: true });
  else if (cible && cible !== vueCourante) actions.unshift({ type: 'ouvrir_vue', vue: cible, auto: true });
  const prefixe = CHIFFRES.test(q)
    ? 'Je ne peux pas calculer de chiffres en ce moment : le service ATOM Wind ne répond pas. Vous pouvez lire la réponse vous-même dans l’application.'
    : '';
  if (t) {
    return { texte: [prefixe, `**${t.titre}**`].filter(Boolean).join('\n\n'), etapes: t.etapes, actions,
      relances: t.questions.slice(0, 2) };
  }
  if (vue) {
    return { texte: `Voici l’espace « ${guide.vues[vue]?.nom ?? vue} » : ${guide.vues[vue]?.resume ?? ''}`, actions, relances: [] };
  }
  if (parleDeSaPosition(q)) {
    if (vueCourante !== 'monvent') actions.unshift({ type: 'ouvrir_vue', vue: 'monvent', auto: true });
    return {
      texte: [prefixe, 'Votre point est dans « Mon vent ». Lisez-y la vitesse et la provenance du vent, puis choisissez une activité dans « Que voulez-vous faire ? » pour le verdict heure par heure.'].filter(Boolean).join('\n\n'),
      actions, relances: [],
    };
  }
  const liste = Object.values(guide.vues).map(v => `- **${v.nom}** : ${v.resume}`).join('\n');
  return {
    texte: [prefixe || 'Je n’ai pas trouvé cette tâche dans le guide. Voici les espaces de l’application :', liste].join('\n\n'),
    actions, relances: guide.taches.slice(0, 3).map(x => `Comment ${x.titre.charAt(0).toLowerCase()}${x.titre.slice(1)} ?`),
  };
}
