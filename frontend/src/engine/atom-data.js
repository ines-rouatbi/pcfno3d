/* ATOM Wind — couche de données : branche l'interface sur l'API PC-FNO3D.
 *
 * Charge, avant le script de la page, le relief réel, les journées servies et le champ du
 * modèle (GET /api/atom-wind/grille, /jours, /jour?date=…, /reseau-aneo), puis lance le
 * moteur de la page. Si l'API est injoignable ou si le modèle n'est pas installé, aucune valeur
 * n'est affichée : l'interface montre l'état du service (jamais de données inventées).
 *
 * Adresse de l'API : même origine par défaut ; ?api=https://… pour une autre. La clé d'API
 * (si le service en demande une) est saisie dans le bandeau et gardée dans ce navigateur
 * (localStorage, même clé que l'application /app/) ; elle n'apparaît jamais dans l'URL.
 */
(() => {
"use strict";
const P = new URLSearchParams(location.search);
const BASE = (P.get("api") || "").replace(/\/+$/, "");
const KEY_STORAGE = "pcfno3d.apikey";
// OS5 : domaine servi. Par défaut la TUNISIE ENTIÈRE (modèle national) dès que le service la sert (demande d'Ines,
// 20/11/2026) ; le Grand Tunis reste un zoom (?domaine=<nom du domaine Grand Tunis>). Fixé au démarrage (demarrer).
let DOMAINE = P.get("domaine");
// toutes les routes ATOM Wind répondent pour le domaine de la page (agriculture, éolien, outils, alertes,
// historique, réseau ANEO…) ; /domaines liste les domaines, /point choisit le sien par la position
const SANS_DOMAINE = /^\/atom-wind\/(domaines|point)(\?|$)/;
function avecDomaine(path){
  if(!DOMAINE || !path.startsWith("/atom-wind/") || SANS_DOMAINE.test(path) || /[?&]domaine=/.test(path)) return path;
  return path + (path.includes("?") ? "&" : "?") + "domaine=" + encodeURIComponent(DOMAINE);
}
const NODATA = -32768;

function cle(){ try { return localStorage.getItem(KEY_STORAGE); } catch(e) { return null; } }
function poserCle(k){ try { if(k) localStorage.setItem(KEY_STORAGE, k); else localStorage.removeItem(KEY_STORAGE); } catch(e) {} }
// session d'un compte ATOM Wind (BF43) : prime sur la cle d'API ; derriere le Worker d'atom.zelqane.com elle
// voyage dans X-ATOM-Session (le Worker remplace sinon toute cle par sa cle de lecture)
// gardée pour CET onglet seulement (sessionStorage) : fermer le navigateur oblige à redonner son mot de passe ;
// le serveur fait de toute façon expirer la session au bout de 8 h
const SESSION_STORAGE = "aw-session-cle";
try { localStorage.removeItem(SESSION_STORAGE); } catch(e) {}         // ancienne session gardée sans fin : effacée
function session(){ try { return sessionStorage.getItem(SESSION_STORAGE); } catch(e) { return null; } }
function poserSession(k){ try { if(k) sessionStorage.setItem(SESSION_STORAGE, k); else sessionStorage.removeItem(SESSION_STORAGE); } catch(e) {} }
// 401 alors qu'une session était présentée : expirée, fermée par un changement de mot de passe ou un compte
// désactivé ; la page se déconnecte et redemande le mot de passe (atom-runtime.js, « atom:session-expiree »)
// Un 401 isolé n'est pas forcément une session expirée : sur le service Colab, le trousseau est sur Google Drive et
// une relecture périmée peut, quelques secondes, ne pas connaître une session toute neuve (25/11/2026). Avant de
// déconnecter, la page revérifie la session (/comptes/moi) jusqu'à 4 fois sur ~10 s ; elle ne se déconnecte que si
// le service la refuse à chaque fois.
let verifSession = null;
function sessionRefusee(statut){
  if(statut !== 401 || !session() || verifSession) return;
  const s0 = session();
  verifSession = (async () => {
    for(const attente of [1500, 2500, 3000, 3000]){
      await new Promise(r => setTimeout(r, attente));
      if(session() !== s0) return;                              // déconnexion ou nouvelle connexion entre-temps
      try {
        const r = await fetch(`${BASE}/api/comptes/moi`, { headers: { "X-API-Key": s0, "X-ATOM-Session": s0 } });
        if(r.ok){ const d = await r.json().catch(() => ({})); if(d && d.compte) return; }   // session reconnue
        else if(r.status !== 401) return;                        // service indisponible : ne pas déconnecter
      } catch(e) { return; }
    }
    if(session() !== s0) return;
    poserSession(null);
    try { document.dispatchEvent(new Event("atom:session-expiree")); } catch(e) {}
  })().finally(() => { verifSession = null; });
}
function entetes(){
  const s = session(), k = cle();
  return s ? { "X-API-Key": s, "X-ATOM-Session": s } : k ? { "X-API-Key": k } : {};
}

// Dernier état réel connu, gardé AUSSI par le navigateur (cahier v12, 17.1 ; demande d'Ines du 16/11) : chaque
// réponse réelle des routes ci-dessous est gardée (Cache Storage) avec sa date de réception ; si le service ne
// répond plus (Colab arrêté, tunnel coupé) et que la passerelle n'a rien servi, la page la ressert avec sa date
// au lieu de n'afficher aucune valeur. Sans rien de réel, la page dit l'état du service (aucune donnée d'exemple).
const ETAT_CACHE = "atom-dernier-etat-v1";
const ETAT_ROUTES = /^\/atom-wind\/(grille|jours|jour|direct|domaines|reseau-aneo)(\?|$)/;
async function garderEtat(url, texte){
  try {
    if(!("caches" in window)) return;
    const c = await caches.open(ETAT_CACHE);
    await c.put(url, new Response(texte, { headers: { "content-type": "application/json", "x-atom-recu-le": new Date().toISOString() } }));
  } catch(e) {}
}
async function lireEtat(url){
  try {
    if(!("caches" in window)) return null;
    const r = await (await caches.open(ETAT_CACHE)).match(url);
    return r ? { recu: r.headers.get("x-atom-recu-le"), json: await r.json() } : null;
  } catch(e) { return null; }
}

async function get(path, ms = 90000){
  const h = {}; Object.assign(h, entetes());
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  const debut = performance.now();
  const url = `${BASE}/api${avecDomaine(path)}`;
  try {
    const r = await fetch(url, { headers: h, signal: ctl.signal });
    if(A.mesureRequete) A.mesureRequete(path, performance.now() - debut);
    if(r.status === 401 || r.status === 403){ sessionRefusee(r.status); const e = new Error("clé d'API refusée ou absente"); e.auth = true; e.statut = r.status; throw e; }
    if(!r.ok){
      // 503 du Worker atom.zelqane.com quand le Colab dort : {error, message, detail}
      let d = ""; try { const j = await r.json(); d = j.message || j.detail || ""; } catch(e) {}
      const e = new Error(r.status === 503 && d ? d : `${path} : ${r.status} ${d}`);
      e.statut = r.status; e.modeleAbsent = r.status === 503 && /non install/.test(d); throw e;
    }
    // Worker atom.zelqane.com : service hors ligne, dernier état réel connu (cahier v12, 17.1)
    const de = r.headers.get("x-atom-dernier-etat");
    if(de !== null) A.dernierEtat = de || "date inconnue";
    const texte = await r.text();
    if(de === null && ETAT_ROUTES.test(path)) garderEtat(url, texte);
    return JSON.parse(texte);
  } catch(e){
    if(!e.auth && ETAT_ROUTES.test(path)){
      const s = await lireEtat(url);
      if(s){ A.dernierEtat = s.recu || "date inconnue"; A.dernierEtatNavigateur = true; return s.json; }
    }
    if(e.name === "AbortError") throw new Error("le service ATOM Wind ne répond pas");
    if(e instanceof TypeError){ const r = new Error("le service ATOM Wind est injoignable"); r.reseau = true; throw r; }
    throw e;
  } finally { clearTimeout(t); }
}

function decoder(b64){
  const bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
  for(let i = 0; i < n; i++){
    let v = bin.charCodeAt(2*i) | (bin.charCodeAt(2*i+1) << 8);
    if(v & 0x8000) v -= 0x10000;
    out[i] = v === NODATA ? NaN : v / 100;
  }
  return out;
}

// complète l'objet déjà posé par atom-i18n.js (langue, traduction) au lieu de le remplacer
const A = Object.assign(window.ATOM || {}, { grille: null, jours: [], jour: null, plan: null,
                                             erreur: null, rafraichir: null });
window.ATOM = A;
A.avecDomaine = avecDomaine;           // api.ts (vues React) : même domaine que la page

A.pret = () => !!(A.grille && A.jour && A.jour._vent);
// état lu par l'interface : « pret », « dernier_etat » (Worker, service endormi), « modele_absent »,
// « cle » (clé refusée), « hors_ligne » (rien ne répond) ; jamais de valeurs de remplacement
A.etatService = () => {
  if(A.pret()) return { etat: A.dernierEtat ? "dernier_etat" : "pret", depuis: A.dernierEtat || null };
  const e = A.erreur || {};
  return { etat: e.modeleAbsent ? "modele_absent" : e.auth ? "cle" : "hors_ligne", detail: e.message || null };
};
A.aSigma = () => !!(A.pret() && A.jour._sigma);
// mode (analyse, temps réel, prévision), date et statut (validé / expérimental) de l'heure affichée, tels que le
// service les donne (service.atom_outils.annoter) ; à 10 m, le statut de la couche 10 m quand elle existe
const MODES_FR = { "analyse": "Analyse", "temps reel": "Temps réel", "prevision": "Prévision" };
A.statutHeure = (h, z = 10) => {
  if(!A.pret() || !(A.jour.instants || []).length) return null;
  const I = A.jour.instants, i = I.reduce((b, x) => Math.abs(x.heure_locale - h) < Math.abs(b.heure_locale - h) ? x : b, I[0]);
  const mode = i.mode || (A.jour.temps_reel ? (i.source === "prevision" ? "prevision" : "temps reel") : "analyse");
  const c10 = z <= 10 && i.statut_couche_10m;
  return { mode, statut: c10 ? i.statut_couche_10m : (i.statut || null), date: i.date_donnee || i.timestamp_utc || A.jour.date,
           preuve: (c10 ? A.jour.statut_couche_10m_preuve : A.jour.statut_preuve) || "" };
};
// regle_service_zones_v5 (23/11/2026) : en temps réel et en prévision, le champ 3D national est « validé » à plus
// de 50 km de Tabarka et « expérimental (zone Tabarka) » dans ce rayon ; le service envoie la zone avec la journée
const LIGNE_ZONE = "Temps réel et prévision : validé hors d'un rayon de 50 km autour de Tabarka (règle de service du 23/11/2026, fondée sur le jugement 2026 : −10 % face à GFS, 11 aéroports sur 12) ; expérimental dans ce rayon.";
function distanceKm(lat1, lon1, lat2, lon2){
  const r = Math.PI / 180, p1 = lat1 * r, p2 = lat2 * r, dp = p2 - p1, dl = (lon2 - lon1) * r;
  const a = Math.sin(dp/2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl/2) ** 2;
  return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(a)));
}
// statut du point (x, y en km de la page) : celui de l'heure, et « expérimental (zone Tabarka) » dans la zone
A.statutPoint = (h, z = 10, x = null, y = null) => {
  const s = A.statutHeure(h, z);
  if(!s || !s.statut) return null;
  const zone = A.jour && A.jour.zone_experimentale, d = A.domaine();
  const c10 = z <= 10 && !!(A.jour.instants || []).some(i => i.statut_couche_10m);
  if(zone && !c10 && s.mode !== "analyse" && d && x !== null && y !== null){
    const lat = d.lat0 + y / d.ky, lon = d.lon0 + x / d.kx;
    if(distanceKm(lat, lon, zone.centre.lat, zone.centre.lon) <= zone.rayon_km)
      return { valide: false, libelle: "Expérimental (zone Tabarka)" };
  }
  return { valide: s.statut === "valide", libelle: s.statut === "valide" ? "Validé" : "Expérimental" };
};
A.donneeHtml = (h, z = 10, x = null, y = null) => {
  const s = A.statutHeure(h, z);
  if(!s) return "";                      // aucune donnée servie : la page montre l'état du service, jamais d'exemple
  const quand = /T/.test(String(s.date)) ? fmtInstant(s.date) : fmtDate(s.date);
  const hors = A.dernierEtat ? ` · <b>Service hors ligne</b> : dernier état réel connu, reçu le ${fmtInstant(A.dernierEtat)}.` : "";
  // badge du statut tel que le service le donne (regle_service_zones_v5 : suit la position du point) ; jamais la preuve
  const st = A.statutPoint(h, z, x, y);
  const badge = st ? ` <span class="aw-statut ${st.valide ? "is-valide" : "is-exp"}">${st.libelle}</span>` : "";
  // la vue « Données & fiabilité » est retirée : la ligne qui explique la zone suit la donnée
  const zone = A.jour.zone_experimentale && s.mode !== "analyse" ? `<p class="aw-zone muted">${LIGNE_ZONE}</p>` : "";
  return `<p class="aw-donnee">${MODES_FR[s.mode] || s.mode} · donnée du ${quand}${badge}${hors}</p>${zone}`;
};

// ------------------------------------------------------------------ domaine servi (phase 8 nationale)
// La page garde la géographie de la maquette (Grand Tunis, 68 × 56 km, blocs de 500 m) quand le modèle
// servi est celui du Grand Tunis. Sinon (domaine national : 388 × 828 km, blocs de 4 km), elle prend
// l'emprise, la projection et les stations de /atom-wind/grille : ATOM.domaine() est lu au démarrage de
// atom-runtime.js, avant tout calcul de la page.
const GT = { nx: 136, ny: 112, pas: .5, lat0: 36.395, lon0: 10.04 };
A.estGrandTunis = g => !!g && g.nx === GT.nx && g.ny === GT.ny && Math.abs(g.pas_km - GT.pas) < 1e-9
  && (g.lat0 === null || g.lat0 === undefined || (Math.abs(g.lat0 - GT.lat0) < .01 && Math.abs(g.lon0 - GT.lon0) < .01));
A.domaine = () => {
  const g = A.grille;
  if(!g || A.estGrandTunis(g) || !g.projection) return null;
  const p = g.projection, national = !!(g.domaine && g.domaine.etendue === "national");
  return { lat0: p.lat0, lon0: p.lon0, kx: p.km_par_deg_lon, ky: p.km_par_deg_lat, latc: p.lat_centrale,
           w: g.largeur_km, h: g.hauteur_km, g: 1 / g.pas_km, nx: g.nx, ny: g.ny, national,
           nom: national ? "Tunisie" : ((g.domaine && g.domaine.nom) || "zone servie"),
           stations: (g.stations || []).filter(s => s.lat !== null && s.lat !== undefined) };
};
A.nomDomaine = () => { const d = A.domaine(); return d ? d.nom : "Grand Tunis"; };
A.nomStation = c => {
  const s = A.grille && (A.grille.stations || []).find(x => x.code === c);
  return (s && s.nom) || NOMS_STATIONS[c] || c;
};

// ------------------------------------------------------------------ échantillonnage
// Une journée J = la réponse de /atom-wind/jour ou /atom-wind/direct, avec ses tableaux décodés
// (J._vent, J._sigma, J._t, J._hr). A.jour est la journée affichée ; A.trois[0..2] les journées
// aujourd'hui / demain / après-demain du temps réel (espace agriculteur, J0 à J+2).
function cellule(x, y){
  const g = A.grille, p = g.pas_km;
  return [Math.min(g.nx-1, Math.max(0, Math.floor(x/p))), Math.min(g.ny-1, Math.max(0, Math.floor(y/p)))];
}
function crochets(J, h){                   // instants encadrant l'heure locale h, et poids
  const I = J.instants, n = I.length;
  if(n === 1 || h <= I[0].heure_locale) return [0, 0, 0];
  if(h >= I[n-1].heure_locale) return [n-1, n-1, 0];
  let a = 0; while(a < n-2 && I[a+1].heure_locale <= h) a++;
  const t0 = I[a].heure_locale, t1 = I[a+1].heure_locale;
  return [a, a+1, (h - t0) / Math.max(t1 - t0, 1e-6)];
}
function lire(J, arr, nComp, it, ih, c, i, j){
  const g = A.grille, nH = J.hauteurs_agl_m.length;
  return arr[((((it*nH + ih)*nComp + c)*g.ny + j)*g.nx) + i];
}
function auNiveau(J, arr, nComp, it, z, i, j, c, z0){
  const H = J.hauteurs_agl_m, nH = H.length;
  if(z <= H[0]){
    // sous la première hauteur : profil logarithmique avec la rugosité de la maille
    let ih = 0, v = lire(J, arr, nComp, it, 0, c, i, j);
    while(!isFinite(v) && ih < nH-1){ ih++; v = lire(J, arr, nComp, it, ih, c, i, j); }
    if(!isFinite(v)) return 0;
    const zz = Math.max(z, z0 * 1.5), f = Math.log(zz / z0) / Math.log(H[ih] / z0);
    return v * Math.max(0, f);
  }
  let k = 0; while(k < nH-2 && H[k+1] < z) k++;
  const a = Math.min(1, (z - H[k]) / (H[k+1] - H[k]));
  const v0 = lire(J, arr, nComp, it, k, c, i, j), v1 = lire(J, arr, nComp, it, k+1, c, i, j);
  if(isFinite(v0) && isFinite(v1)) return (1-a)*v0 + a*v1;
  return isFinite(v1) ? v1 : isFinite(v0) ? v0 : 0;
}
function ventDe(J, x, y, z, h){
  const [i, j] = cellule(x, y), z0 = A.z0[j*A.grille.nx + i];
  const [a, b, w] = crochets(J, h);
  const ua = auNiveau(J, J._vent, 2, a, z, i, j, 0, z0), va = auNiveau(J, J._vent, 2, a, z, i, j, 1, z0);
  if(w === 0 || a === b) return [ua, va];
  const ub = auNiveau(J, J._vent, 2, b, z, i, j, 0, z0), vb = auNiveau(J, J._vent, 2, b, z, i, j, 1, z0);
  return [(1-w)*ua + w*ub, (1-w)*va + w*vb];
}
function surfaceDe(J, arr, x, y, h){         // champ 2D par instant (t2m, hr2m), interpolé en temps
  const g = A.grille, [i, j] = cellule(x, y), [a, b, w] = crochets(J, h), n = g.nx * g.ny;
  const va = arr[a*n + j*g.nx + i], vb = arr[b*n + j*g.nx + i];
  return isFinite(va) && isFinite(vb) ? (1-w)*va + w*vb : (isFinite(va) ? va : vb);
}
A.vent = (x, y, z, h) => ventDe(A.jour, x, y, z, h);
// vent du modele servi en latitude / longitude (carte nationale de la vue 3D) ; null hors du domaine
A.ventLatLon = (lat, lon, z, h) => {
  const d = A.domaine(); if(!A.pret() || !d) return null;
  const x = (lon - d.lon0) * d.kx, y = (lat - d.lat0) * d.ky;
  if(!(x >= 0 && y >= 0 && x <= d.w && y <= d.h)) return null;
  return ventDe(A.jour, x, y, z, h);
};
A.sigma = (x, y, z, h) => {
  const J = A.jour; if(!J || !J._sigma) return null;
  const [i, j] = cellule(x, y), [a, b, w] = crochets(J, h), z0 = A.z0[j*A.grille.nx + i];
  const zz = Math.max(z, J.hauteurs_agl_m[0]);
  const sa = auNiveau(J, J._sigma, 1, a, zz, i, j, 0, z0), sb = auNiveau(J, J._sigma, 1, b, zz, i, j, 0, z0);
  return (1-w)*sa + w*sb;
};
// espace agriculteur, jour d = 0 (aujourd'hui), 1 (demain), 2 (après-demain) : seulement en temps
// réel ; sinon ces jours ne sont pas disponibles (et l'interface le dit)
A.trois = {};
A.pretJour = d => !!(A.grille && A.jour && A.jour.temps_reel && A.trois[d] && A.trois[d]._vent);
A.ventJour = (d, x, y, z, h) => ventDe(A.trois[d], x, y, z, h);
A.aT = d => A.pretJour(d) && !!A.trois[d]._t;
A.tJour = (d, x, y, h) => surfaceDe(A.trois[d], A.trois[d]._t, x, y, h);
A.hrJour = (d, x, y, h) => surfaceDe(A.trois[d], A.trois[d]._hr, x, y, h);
A.rsJour = d => {                            // rayonnement solaire du jour (MJ/m²/j), GFS
  if(!A.pretJour(d)) return null;
  const r = A.trois[d].instants.map(i => i.rayonnement_w_m2).filter(v => v !== null && v !== undefined);
  return r.length >= 20 ? r.reduce((s, v) => s + v, 0) * 3600 / 1e6 * (24 / r.length) : null;
};
A.doyJour = d => {
  if(!A.pretJour(d)) return null;
  const t = new Date(A.trois[d].date + "T12:00:00Z"), d0 = Date.UTC(t.getUTCFullYear(), 0, 0);
  return Math.floor((t - d0) / 86400000);
};

// ------------------------------------------------------------------ relief réel
A.appliquerGrille = ({ NX, NY, G, elev, wat, z0, LAKES }) => {
  const g = A.grille; if(!g){ appliquerRelief(NX, NY, G, elev, wat, z0); return; }
  // la page est en mailles de 1/G km ; la grille du modèle peut avoir un autre pas : lecture par position
  for(let j = 0; j < NY; j++) for(let i = 0; i < NX; i++){
    const k = j*NX + i, gi = Math.min(g.nx-1, Math.floor((i+.5)/G/g.pas_km)), gj = Math.min(g.ny-1, Math.floor((j+.5)/G/g.pas_km));
    const eau = g.eau[gj][gi] === 1;
    if(eau){
      const x = (i+.5)/G, y = (j+.5)/G;
      let lac = false;
      for(const [cx, cy, rx, ry] of LAKES){ const dx = (x-cx)/rx, dy = (y-cy)/ry; if(dx*dx + dy*dy < 1.3) lac = true; }
      wat[k] = lac ? 2 : 1; elev[k] = 0;
    } else {
      wat[k] = 0; elev[k] = Math.max(1, g.terrain_m[gj][gi]);
    }
    z0[k] = g.z0_m[gj][gi];
  }
  // relief fin (DEM Copernicus 125 m) quand il est là : plus précis que la grille du modèle pour l'affichage
  // seulement sur le Grand Tunis : ce relief fin couvre ses 68 × 56 km, l'appliquer au domaine national effaçait le
  // relief de toute la Tunisie (mer partout sauf une bande au sud-ouest de la reconstruction 3D)
  if(A.relief && A.relief.pas_km < g.pas_km && A.estGrandTunis(g)){ const z = Float32Array.from(z0); appliquerRelief(NX, NY, G, elev, wat, z0);
    for(let k = 0; k < NX*NY; k++) if(!wat[k]) z0[k] = z[k]; }
};
// ------------------------------------------------------------------ relief réel local (sans API)
// Grille 125 m tirée du DEM Copernicus GLO-30 et des couches d'eau (scripts/build_grand_tunis_relief.mjs).
// Sert le relief de la page (plus fin que la grille du modèle) et le maillage de la vue 3D.
A.relief = null;
async function chargerRelief(){
  try {
    const base = new URL("relief/", location.href.replace(/[#?].*$/, "").replace(/[^/]*$/, ""));
    const h = await (await fetch(new URL("grand-tunis.json", base))).json();
    const r = await fetch(new URL(h.fichier, base)); if(!r.ok) throw new Error(`relief ${r.status}`);
    let buf = await r.arrayBuffer(); const b = new Uint8Array(buf);
    if(b[0] === 0x1f && b[1] === 0x8b) buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const n = h.nx * h.ny; if(buf.byteLength !== n * 3) throw new Error("relief : taille inattendue");
    const dm = new Int16Array(buf, 0, n), alt = new Float32Array(n); for(let k = 0; k < n; k++) alt[k] = dm[k] / 10;
    const eau = new Uint8Array(buf, n * 2, n), p = h.pas_km;
    const lire = (x, y) => [Math.min(h.nx - 1.001, Math.max(0, x / p - .5)), Math.min(h.ny - 1.001, Math.max(0, y / p - .5))];
    A.relief = { ...h, alt, eau,
      // altitude bilinéaire (m) et nature du sol au point (x, y) en km du domaine
      hauteur(x, y){ const [fx, fy] = lire(x, y), i = fx | 0, j = fy | 0, u = fx - i, v = fy - j, k = j * h.nx + i;
        return (alt[k] * (1 - u) + alt[k + 1] * u) * (1 - v) + (alt[k + h.nx] * (1 - u) + alt[k + h.nx + 1] * u) * v; },
      eauA(x, y){ const [fx, fy] = lire(x, y); return eau[Math.round(fy) * h.nx + Math.round(fx)]; } };
  } catch(e){ console.info("ATOM Wind : relief réel local absent (relief de la grille du modèle seulement)", e.message); }
}
function appliquerRelief(NX, NY, G, elev, wat, z0){
  const R = A.relief; if(!R) return;
  const f = Math.round(1 / (G * R.pas_km));               // cellules fines par cellule de la page
  for(let j = 0; j < NY; j++) for(let i = 0; i < NX; i++){
    let s = 0, c = 0, mer = 0, lac = 0;
    for(let b = 0; b < f; b++) for(let a = 0; a < f; a++){
      const kk = (j*f + b) * R.nx + i*f + a, e = R.eau[kk];
      if(e === 1) mer++; else if(e === 2) lac++; else { s += R.alt[kk]; c++; } }
    const k = j*NX + i, n = f*f;
    wat[k] = mer * 2 >= n ? 1 : (mer + lac) * 2 >= n ? 2 : 0;
    elev[k] = wat[k] ? 0 : Math.max(1, c ? s / c : 0);
    if(wat[k]) z0[k] = wat[k] === 1 ? .0002 : .001;         // rugosité de la page conservée sur terre
  }
}

// plus de note sur la fiabilité dans la page (décision d'Ines) : gardée pour les appelants, sans texte
A.noteFiabilite = () => "";
A.planXY = xy => (A.plan && A.plan.length) ? A.plan.map(p => xy(p.lat, p.lon)) : null;

// ------------------------------------------------------------------ bandeau
function fmtInstant(iso){
  const t = new Date(iso);
  return isNaN(t) ? iso : t.toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "Africa/Tunis" });
}
function fmtDate(d){ const [y, m, j] = d.split("-"); return `${j}/${m}/${y}`; }
function banniere(){
  const src = document.getElementById("aw-source"), fr = document.getElementById("aw-fresh");
  const sel = document.getElementById("aw-jour");
  if(!src) return;
  if(A.pret()){
    const horsLigne = A.dernierEtat ? `<b>Service hors ligne</b> : dernier état réel connu, reçu le ${fmtInstant(A.dernierEtat)}. ` : "";
    const I = A.jour.instants, hh = I.map(x => `${String(Math.floor(x.heure_locale)).padStart(2,"0")} h`).join(", ");
    if(A.jour.archive){
      const maj = fmtInstant(A.jour.mis_a_jour_utc);
      src.innerHTML = horsLigne + `<b>Historique</b> : vent du <b>${fmtDate(A.jour.date)}</b> tel que calculé par <b>ATOM Wind</b> `
        + (A.jour.prevision ? `en prévision (GFS), dernier calcul le ${maj}.` : `en temps réel (GFS et anémomètres), dernier calcul le ${maj}.`);
      if(fr) fr.textContent = `Historique · ${fmtDate(A.jour.date)}`;
    } else if(A.jour.temps_reel && A.jour.prevision){
      const maj = new Date(A.jour.mis_a_jour_utc), hm = maj.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", timeZone: "Africa/Tunis" });
      src.innerHTML = horsLigne + `<b>Prévision</b> du vent pour le <b>${fmtDate(A.jour.date)}</b>, calculée par <b>ATOM Wind</b> `
        + `à partir de la prévision GFS (pas encore de mesure des stations pour ce jour). Calculée à ${hm}.`
        + (A.jour.avertissement ? ` <span class="muted">(${A.jour.avertissement})</span>` : "");
      if(fr) fr.textContent = `Prévision · ${hm}`;
    } else if(A.jour.temps_reel){
      const maj = new Date(A.jour.mis_a_jour_utc), hm = maj.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", timeZone: "Africa/Tunis" });
      const passees = I.filter(x => x.source === "temps reel"), st = new Set(passees.flatMap(x => x.stations));
      src.innerHTML = horsLigne + `<b>Temps réel</b> : vent d'aujourd'hui (${fmtDate(A.jour.date)}) calculé par <b>ATOM Wind</b> `
        + `à partir de la grande échelle GFS et des anémomètres ${[...st].map(c => A.nomStation(c)).join(", ") || "(aucune mesure reçue)"}. `
        + `Heures passées : temps réel ; heures à venir : prévision. Mis à jour à ${hm}.`
        + (A.jour.avertissement ? ` <span class="muted">(${A.jour.avertissement})</span>` : "");
      if(fr) fr.textContent = `Temps réel · ${hm}`;
    } else {
      const zone = A.domaine() ? (A.domaine().national ? "la Tunisie" : A.domaine().nom) : "le Grand Tunis";
      src.innerHTML = horsLigne + `Vent calculé par <b>ATOM Wind</b> sur ${zone}, `
        + `journée du <b>${fmtDate(A.jour.date)}</b>. `
        + `Calculs à ${hh} (heure de Tunis), interpolés heure par heure.`;
      if(fr) fr.textContent = `Archive · ${fmtDate(A.jour.date)}`;
    }
    // couche_10m (nom du modèle, ERA5) et statut validé / expérimental : plus affichés (décision d'Ines)
    if((A.domaines || []).length > 1){
      const nat = !!(A.domaine() && A.domaine().national);
      src.insertAdjacentHTML("beforeend", ` <span class="aw-domaines">Zone : `
        + (nat ? `<b>Tunisie entière</b> · <a href="?domaine=${encodeURIComponent((A.domaines.find(d => d.nom !== "national") || {}).nom || "")}">zoom Grand Tunis</a>`
               : `<a href="?domaine=national">Tunisie entière</a> · <b>Grand Tunis</b>`)
        + `</span>`);
    }
    if(sel){
      const cle = A.jour.cle || A.jour.date;
      const opt = (v, t) => `<option value="${v}"${cle === v ? " selected" : ""}>${t}</option>`;
      const pv = (A.direct && A.direct.previsions) || [];
      sel.innerHTML = (A.direct ? opt("direct", "Aujourd'hui, temps réel")
          + (pv[0] ? opt("prev1", `Demain ${fmtDate(pv[0])} (prévision)`) : "")
          + (pv[1] ? opt("prev2", `Après-demain ${fmtDate(pv[1])} (prévision)`) : "") : "")
        + A.jours.map(d => `<option value="${d.date}"${d.date === cle ? " selected" : ""}>${fmtDate(d.date)} (${d.instants.length} instants)</option>`).join("")
        + ((A.historique || []).length ? `<optgroup label="Historique">` + A.historique.map(h => opt(`hist:${h.nom}`,
            h.type === "temps reel" ? `${fmtDate(h.date)} (temps réel archivé)` : `${fmtDate(h.date)} (prévision J+${h.echeance_j} émise le ${fmtDate(h.emise_le)})`)).join("")
            + `</optgroup>` : "");
      sel.hidden = false;
    }
  } else {
    const e = A.etatService();
    if(fr) fr.textContent = e.etat === "modele_absent" ? "Service en installation" : e.etat === "cle" ? "Accès requis" : "Service hors ligne";
    src.innerHTML = e.etat === "modele_absent"
      ? `<b>ATOM Wind en cours d'installation</b> : le service répond, mais le calcul du vent n'est pas encore installé. Aucune valeur de vent n'est affichée.`
      : e.etat === "cle" ? `<b>Accès requis</b> : ce service demande une connexion ou une clé d'accès.`
      : `<b>Service de calcul hors ligne</b> : aucune valeur de vent n'est affichée tant qu'il ne répond pas.${A.erreur ? ` <span class="muted">(${A.erreur.message})</span>` : ""}`;
    if(A.erreur && A.erreur.auth){
      src.insertAdjacentHTML("beforeend", ` <form id="aw-cle" style="display:inline-flex;gap:6px;align-items:center"><input type="password" placeholder="Clé d'API (pcf_…)" autocomplete="off" style="font-size:13px;padding:2px 6px"><button class="btn" style="padding:2px 10px;font-size:13px">Se connecter</button></form>`);
      document.getElementById("aw-cle").onsubmit = e => { e.preventDefault(); poserCle(e.target.querySelector("input").value.trim()); location.reload(); };
    }
  }
}
A.banniere = () => { banniere(); if(A.sourceEolien) A.sourceEolien(); };

// ------------------------------------------------------------------ chargement
const RELATIF = { direct: 0, prev1: 1, prev2: 2 };
async function telechargerJour(date){
  const j = RELATIF[date];
  // domaine national : premier calcul plus long (GFS par paquets de mailles, 24 passes sur 388 × 828 km)
  const d = j !== undefined ? await get(`/atom-wind/direct?j=${j}`, (A.domaine() && A.domaine().national) ? 600000 : 240000)
          : date.startsWith("hist:") ? await get(`/atom-wind/historique/${encodeURIComponent(date.slice(5))}`, 60000)
          : await get(`/atom-wind/jour?date=${encodeURIComponent(date)}`, 180000);
  d.cle = date;
  d._vent = decoder(d.vent);
  d._sigma = d.sigma_vitesse ? decoder(d.sigma_vitesse) : null;
  d._t = d.t2m ? decoder(d.t2m) : null;
  d._hr = d.hr2m ? decoder(d.hr2m) : null;
  d._raf = d.facteur_rafale ? decoder(d.facteur_rafale) : null;
  if(j !== undefined) A.trois[j] = d;
  return d;
}
async function chargerJour(date){ A.jour = await telechargerJour(date); }
// demain et après-demain pour l'espace agriculteur : en arrière-plan, l'un après l'autre
let troisEnCours = false;
async function chargerTrois(){
  if(troisEnCours || !A.direct) return;
  troisEnCours = true;
  try {
    for(const [cle, j] of [["prev1", 1], ["prev2", 2]]){
      try { await telechargerJour(cle); } catch(e){ console.warn("ATOM Wind : prévision J+" + j, e); }
      if(A.rafraichir) A.rafraichir();
    }
  } finally { troisEnCours = false; }
}
A.chargerTrois = chargerTrois;

// ------------------------------------------------------------------ ensemble GEFS (E19)
// Probabilité de pouvoir traiter et quantiles du vent, aux heures de traitement, J0 à J+2.
A.ensembles = {};
let ensEnCours = false;
async function chargerEnsembles(){
  if(ensEnCours || !A.direct) return;
  ensEnCours = true;
  try {
    for(const j of [0, 1, 2]){
      if(A.ensembles[j]) continue;
      try {
        const e = await get(`/atom-wind/ensemble?j=${j}`, 300000);
        e._p = decoder(e.proba_traitement); e._q = decoder(e.vitesse_quantiles);
        A.ensembles[j] = e;
      } catch(err){ console.warn("ATOM Wind : ensemble J+" + j, err); }
      if(A.rafraichir) A.rafraichir();
    }
  } finally { ensEnCours = false; }
}
A.chargerEnsembles = chargerEnsembles;

// ------------------------------------------------------------------ atlas éolien (E8, E15)
async function blob(path){
  const h = {}; Object.assign(h, entetes());
  const r = await fetch(`${BASE}/api${path}`, { headers: h });
  if(!r.ok){ sessionRefusee(r.status); throw new Error(`${path} : ${r.status}`); }
  return URL.createObjectURL(await r.blob());
}
async function chargerAtlas(){
  const el = document.getElementById("aw-atlas"); if(!el) return;
  let a;
  try { a = await get("/atom-wind/atlas", 30000); } catch(e){ return; }      // pas encore calculé
  A.atlas = a;
  const img = await blob("/atom-wind/atlas/atlas.png").catch(() => null);
  const sites = (a.sites || []).map(s => `<li><b>Site ${s.rang}</b> : ${s.lat !== undefined ? `${s.lat.toFixed(3)} N, ${s.lon.toFixed(3)} E` : `${(s.x_m/1000).toFixed(1)} km, ${(s.y_m/1000).toFixed(1)} km`}`
    + ` — vent moyen ${String(s.vitesse_moyenne_m_s).replace(".", ",")} m/s, facteur de charge ${Math.round(s.facteur_charge * 100)} %, ${Math.round(s.aep_mwh).toLocaleString("fr-FR")} MWh/an</li>`).join("");
  const tifs = (a.couches_geotiff || []).map(c => `<button class="btn" data-tif="${c}" style="padding:2px 10px;font-size:13px">${c}.tif</button>`).join(" ");
  el.innerHTML = `<h3>Atlas éolien annuel à ${a.hub_agl_m} m (historique des reconstructions)</h3>`
    + `<p class="muted" style="font-size:14px;margin:0 0 8px">${a.instants} reconstructions, courbe ${a.courbe} (${a.rated_kw / 1000} MW). `
    + `Vent moyen du domaine ${String(a.vitesse_moyenne_domaine_m_s).replace(".", ",")} m/s, facteur de charge moyen ${Math.round(a.facteur_charge_domaine * 100)} %.</p>`
    + (img ? `<img src="${img}" alt="Atlas éolien : vitesse moyenne et facteur de charge" style="width:100%;border-radius:8px">` : "")
    + `<ol style="margin:8px 0">${sites}</ol>`
    + (tifs ? `<p style="font-size:13px;margin:4px 0">Export SIG (GeoTIFF, WGS 84) : ${tifs}</p>` : "");
  el.querySelectorAll("button[data-tif]").forEach(bt => bt.onclick = async () => {
    const u = await blob(`/atom-wind/atlas/atlas_${bt.dataset.tif}.tif`);
    const lien = document.createElement("a"); lien.href = u; lien.download = `atlas_${bt.dataset.tif}.tif`; lien.click();
  });
  el.hidden = false;
}
A.chargerAtlas = chargerAtlas;

// ------------------------------------------------------------------ valeur des stations (E10, E11)
// Le modèle est relancé sans chaque station aux heures mesurées du jour : calcul lourd, à la demande.
function lectureStation(st){
  return st.erreur_sans_elle_m_s >= 2 ? ["bad", "précieuse"] : st.erreur_sans_elle_m_s < 1 ? ["good", "redondante"] : ["mid", "utile"];
}
function installerValeurStations(){
  const el = document.getElementById("aw-valeur"); if(!el || !A.pret() || !A.jour.temps_reel) return;
  // valeur de chaque station du réseau : outil d'exploitation du réseau, réservé à l'administrateur
  if(!(A.workspace && A.workspace.role && A.workspace.role() === "admin")){ el.hidden = true; el.innerHTML = ""; return; }
  el.innerHTML = `<h3>Valeur de chaque station aujourd'hui</h3>`
    + `<p class="muted" style="font-size:14px;margin:0 0 8px">ATOM Wind recalcule le vent sans chaque station, aux heures déjà mesurées : `
    + `on voit ce que les autres stations et GFS savent deviner sans elle, et jusqu'où elle change le vent.</p>`
    + `<button class="btn primary" id="aw-valeur-bt">Calculer (environ une minute)</button><div id="aw-valeur-res"></div>`;
  el.hidden = false;
  el.querySelector("#aw-valeur-bt").onclick = async ev => {
    const bt = ev.currentTarget, res = el.querySelector("#aw-valeur-res");
    bt.disabled = true; bt.textContent = "Calcul en cours…";
    try {
      const d = await get("/atom-wind/valeur-stations", 240000);
      d._influence = decoder(d.influence); A.valeurStations = d;
      const kmh = v => `${Math.round(v * 3.6)} km/h`;
      res.innerHTML = `<div class="tblwrap"><table class="t"><thead><tr><th>Station</th><th style="text-align:right">Erreur sans elle</th>`
        + `<th style="text-align:right">Avec elle</th><th style="text-align:right">Zone d'influence</th><th>Lecture</th></tr></thead><tbody>`
        + d.stations.map(st => { const [c, t] = lectureStation(st);
            return `<tr><td>${st.nom}</td><td class="n">${kmh(st.erreur_sans_elle_m_s)}</td><td class="n">${kmh(st.erreur_avec_elle_m_s)}</td>`
              + `<td class="n">${Math.round(st.aire_influence_km2).toLocaleString("fr-FR")} km²</td><td><span class="pill ${c}">${t}</span></td></tr>`; }).join("")
        + `</tbody></table></div><p class="muted" style="font-size:13px;margin:6px 0 0">Heures rejouées : ${d.heures_locales.map(h => `${h} h`).join(", ")}. `
        + `Zone d'influence : là où retirer la station change le vent à ${d.hauteur_carte_m} m de plus de ${kmh(d.seuil_influence_m_s)}. `
        + `Une station « précieuse » mesure ce que rien d'autre ne devine : à entretenir en priorité.</p>`;
      bt.textContent = "Recalculer";
    } catch(e){ res.innerHTML = `<p class="muted">Calcul impossible : ${e.message}</p>`; bt.textContent = "Réessayer"; }
    bt.disabled = false;
  };
}
A.installerValeurStations = installerValeurStations;

// ------------------------------------------------------------------ rapports PDF (E9)
async function telechargerRapport(bt, params, nom){
  const t = bt.textContent; bt.disabled = true; bt.textContent = "Préparation du PDF…";
  try {
    const u = await blob(`/atom-wind/rapport.pdf?${new URLSearchParams(params)}`);
    const a = document.createElement("a"); a.href = u; a.download = nom; a.click();
    setTimeout(() => URL.revokeObjectURL(u), 60000);
    bt.textContent = t;
  } catch(e){ bt.textContent = "Rapport indisponible"; setTimeout(() => { bt.textContent = t; }, 4000); }
  bt.disabled = false;
}
function installerRapports(){
  if(!A.pret() || !A.jour.temps_reel || document.getElementById("aw-pdf-agri")) return;
  const form = document.querySelector("#v-agri .agform"), lieu = document.getElementById("ag-place");
  if(form && lieu){
    const bt = document.createElement("button");
    bt.id = "aw-pdf-agri"; bt.className = "btn"; bt.type = "button"; bt.textContent = "Rapport PDF de la parcelle";
    bt.style.marginTop = "10px";
    bt.onclick = () => { const o = lieu.selectedOptions[0];
      const nom = (o && (A.texteFr ? A.texteFr(o) : o.textContent)) || "Mornag";      // nom français : l'API le reconnaît
      telechargerRapport(bt, { type: "parcelle", lieu: nom }, "atom_wind_parcelle.pdf"); };
    form.after(bt);
  }
  const src = document.getElementById("aw-eol-src");
  if(src){
    const z = document.createElement("div"); z.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;margin-top:8px";
    const b1 = document.createElement("button"), b2 = document.createElement("button");
    b1.className = b2.className = "btn"; b1.type = b2.type = "button";
    b1.textContent = "Rapport PDF du parc virtuel"; b2.textContent = "Rapport PDF du site (Djebel Ressas)";
    b1.onclick = () => telechargerRapport(b1, { type: "parc" }, "atom_wind_parc.pdf");
    b2.onclick = () => telechargerRapport(b2, { type: "site", lieu: "Djebel Ressas" }, "atom_wind_site.pdf");
    z.append(b1, b2); src.after(z);
  }
}
A.installerRapports = installerRapports;

// ------------------------------------------------------------------ qualité du modèle servi (TB5)
// Décision d'Ines : la page ne montre plus rien sur la qualité du calcul (DSO, écarts aux mesures, ballons-sondes,
// ERA5) ; ces résultats restent dans son mémoire. /atom-wind/qualite n'est plus lu et A.qualite reste null.
// Les conteneurs qui la montraient (#v-tableau .dsos, #aw-val-mesures, #aw-val-sondes) sont retirés de
// l'interface : s'il en reste un, il est vidé, et leur absence ne fait jamais d'erreur.
function rendreQualite(){
  for(const el of [document.querySelector("#v-tableau .dsos"), document.getElementById("aw-val-mesures"),
                   document.getElementById("aw-val-sondes")])
    if(el) el.innerHTML = "";
}
async function chargerQualite(){
  A.qualite = null;
  try { rendreQualite(); } catch(e){}
}
A.chargerQualite = chargerQualite;

// rafales : facteur rafale / vent moyen de GFS a 10 m, a l'heure affichee, decroissant avec la hauteur
// comme celui de la page (BF54, BF80) ; null hors temps reel (la page garde alors son facteur)
A.facteurRafale = (x, y, z, h) => {
  const J = A.jour; if(!A.pret() || !J._raf) return null;
  const f = surfaceDe(J, J._raf, x, y, h);
  return isFinite(f) && f >= 1 ? 1 + (f - 1) * Math.exp(-Math.max(z - 10, 0) / 300) : null;
};

// ------------------------------------------------------------------ clés d'API du service (BF42)
// Vue Administration : les vraies clés du service (création, quotas, révocation), seulement si la clé
// enregistrée dans ce navigateur a la portée admin (sinon la carte n'apparaît pas).
async function envoyer(methode, chemin, corps){
  const h = { "Content-Type": "application/json" }; Object.assign(h, entetes());
  const r = await fetch(`${BASE}/api${chemin}`, { method: methode, headers: h, body: corps ? JSON.stringify(corps) : undefined });
  const d = await r.json().catch(() => ({}));
  if(!r.ok){ sessionRefusee(r.status); throw new Error(d.detail ? (typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail)) : `erreur ${r.status}`); }
  return d;
}
async function installerClesService(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-cles")) return;
  try { await get("/keys", 15000); } catch(e){ return; }                       // pas admin : rien
  const carte = document.createElement("div");
  carte.className = "card"; carte.id = "aw-cles";
  vue.appendChild(carte);
  const rendre = async () => {
    // les sessions des comptes (clés « compte-… ») se gèrent avec les comptes : seules les clés de service ici
    const tout = (await get("/keys", 15000)).cles, sessions = tout.filter(c => c.name.startsWith("compte-") && !c.revoked).length;
    const l = tout.filter(c => !c.name.startsWith("compte-"));
    carte.innerHTML = `<h3>Clés d'API du service</h3>`
      + `<p class="muted" style="font-size:14px;margin:0 0 8px">Une clé permet à un logiciel partenaire de lire le vent. Seule une empreinte est gardée sur le serveur ; le secret ne s'affiche qu'à la création.</p>`
      + `<p class="muted" style="font-size:13px;margin:0 0 8px">${sessions} session(s) de compte ouverte(s), gérée(s) avec les comptes.</p>`
      + `<div class="tblwrap"><table class="t"><thead><tr><th>Nom</th><th>Portée</th><th>Quota / min</th><th>Quota / jour</th><th>Dernier usage</th><th></th></tr></thead><tbody>`
      + l.map(c => `<tr><td>${echapper(c.name)}</td><td>${c.scope}</td><td class="n">${c.quota_minute ?? "défaut"}</td><td class="n">${c.quota_jour ?? "défaut"}</td>`
        + `<td>${c.last_used_at ? fmtInstant(c.last_used_at) : "—"}</td><td>${c.revoked ? `<span class="pill mid">révoquée</span>`
        : `<button class="btn" data-rev="${echapper(c.name)}" style="padding:2px 10px;font-size:13px">Révoquer</button>`}</td></tr>`).join("")
      + `</tbody></table></div>`
      + `<form id="aw-cle-n" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px"><input class="inp" name="nom" placeholder="nom (ex. appli-agri)" required pattern="[A-Za-z0-9_.\\-]+">`
      + `<select class="inp" name="portee"><option value="read">lecture</option><option value="admin">admin</option></select>`
      + `<input class="inp" name="qm" type="number" min="1" placeholder="quota / min" style="width:120px">`
      + `<button class="btn primary">Créer la clé</button></form><div id="aw-cle-res" style="margin-top:6px"></div>`;
    carte.querySelectorAll("[data-rev]").forEach(b => b.onclick = async () => {
      if(b.dataset.ok !== "1"){ b.dataset.ok = "1"; b.textContent = "Confirmer la révocation"; return; }
      try { await envoyer("DELETE", `/keys/${encodeURIComponent(b.dataset.rev)}`); await rendre(); }
      catch(e){ carte.querySelector("#aw-cle-res").textContent = e.message; }
    });
    carte.querySelector("#aw-cle-n").onsubmit = async e => {
      e.preventDefault(); const f = e.target, res = carte.querySelector("#aw-cle-res");
      try {
        const d = await envoyer("POST", "/keys", { nom: f.nom.value.trim(), portee: f.portee.value,
                                                    quota_minute: f.qm.value ? Number(f.qm.value) : null });
        await rendre();
        carte.querySelector("#aw-cle-res").innerHTML = `<b>Clé créée, à noter maintenant (elle ne sera plus affichée)</b> : <code style="user-select:all">${echapper(d.cle)}</code>`;
      } catch(err){ res.textContent = err.message; }
    };
  };
  try { await rendre(); } catch(e){ carte.remove(); }
}
A.installerClesService = installerClesService;

// ------------------------------------------------------------------ comptes du service (BF43), admin
// Gestion des comptes : adresse confirmée ou non, rôle, validation des demandes d'accès éolien, renvoi du lien
// de confirmation, confirmation de l'adresse par l'administrateur, désactivation et réactivation.
const NOMS_ROLES = { public: "Grand public", agri: "Agriculteur", eolien: "Exploitant éolien", admin: "Administrateur" };
async function installerComptesService(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-comptes")) return;
  try { await get("/comptes", 15000); } catch(e){ return; }                     // pas admin : rien
  const carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-comptes";
  vue.appendChild(carte);
  let filtre = "";
  const petit = 'style="padding:2px 10px;font-size:13px"';
  const rendre = async () => {
    const tous = (await get("/comptes", 15000)).comptes;
    const l = tous.filter(c => !filtre || (c.email + " " + (c.nom || "")).toLowerCase().includes(filtre));
    const n = k => tous.filter(k).length;
    carte.innerHTML = `<h3>Gestion des comptes</h3>`
      + `<p class="muted" style="font-size:14px;margin:0 0 8px">${tous.length} compte(s) · ${n(c => c.email_verifie === false)} adresse(s) à confirmer · `
      + `${n(c => c.demande_role)} demande(s) d'accès · ${n(c => c.statut !== "actif")} désactivé(s). Chaque espace (Agriculture, Énergie éolienne, `
      + `Administration) n'est ouvert qu'à son rôle ; une connexion exige une adresse confirmée et le mot de passe, et dure 8 h.</p>`
      + `<input class="inp" id="aw-comptes-f" placeholder="Rechercher un nom ou une adresse" value="${echapper(filtre)}" style="margin-bottom:8px;max-width:340px">`
      + `<div class="tblwrap"><table class="t"><thead><tr><th>Compte</th><th>Adresse</th><th>Rôle</th><th>Demande</th><th>Statut</th><th>Dernière connexion</th><th></th></tr></thead><tbody>`
      + l.map(c => { const e = echapper(c.email), adm = c.role === "admin";
        return `<tr><td>${echapper(c.nom || "")}<br><span class="muted" style="font-size:12px">${e}</span></td>`
        + `<td>${c.email_verifie === false ? `<span class="pill mid">à confirmer</span>` : `<span class="pill good">confirmée</span>`}</td>`
        + `<td>${adm ? NOMS_ROLES.admin : `<select class="inp" data-role="${e}" aria-label="Rôle de ${e}" style="padding:2px 6px;font-size:13px;min-width:160px">`
          + ["public", "agri", "eolien"].map(r => `<option value="${r}"${r === c.role ? " selected" : ""}>${NOMS_ROLES[r]}</option>`).join("") + `</select>`}</td>`
        + `<td>${c.demande_role ? NOMS_ROLES[c.demande_role] || c.demande_role : "—"}</td>`
        + `<td>${c.statut === "actif" ? "actif" : `<span class="pill mid">désactivé</span>`}</td>`
        + `<td>${c.connecte_le ? fmtInstant(c.connecte_le) : "—"}</td><td style="white-space:nowrap">`
        + (c.demande_role ? `<button class="btn primary" data-val="${e}" ${petit}>Valider</button> <button class="btn" data-ref="${e}" ${petit}>Refuser</button> ` : "")
        + (c.email_verifie === false ? `<button class="btn" data-renv="${e}" ${petit}>Renvoyer le lien</button> <button class="btn" data-conf="${e}" ${petit}>Confirmer l'adresse</button> ` : "")
        + (adm ? "" : `<button class="btn" data-st="${e}" data-actif="${c.statut === "actif" ? "0" : "1"}" ${petit}>${c.statut === "actif" ? "Désactiver" : "Réactiver"}</button>`
          + ` <button class="btn" data-sup="${e}" style="padding:2px 10px;font-size:13px;color:var(--stop);border-color:var(--stop)">Supprimer</button>`)
        + `</td></tr>`; }).join("")
      + `</tbody></table></div><div id="aw-comptes-res" class="muted" role="status" style="font-size:13px;margin-top:6px"></div>`;
    const res = carte.querySelector("#aw-comptes-res");
    const agir = async (chemin, ok) => { try { await envoyer("POST", chemin); await rendre(); if(ok) carte.querySelector("#aw-comptes-res").textContent = ok; } catch(err){ res.textContent = err.message; } };
    const f = carte.querySelector("#aw-comptes-f");
    f.oninput = () => { filtre = f.value.trim().toLowerCase(); clearTimeout(f._t); f._t = setTimeout(async () => { await rendre(); const g = carte.querySelector("#aw-comptes-f"); g.focus(); g.setSelectionRange(g.value.length, g.value.length); }, 250); };
    const url = (e, suite) => `/comptes/${encodeURIComponent(e)}/${suite}`;
    carte.querySelectorAll("[data-val]").forEach(b => b.onclick = () => agir(url(b.dataset.val, "validation?accepter=true"), "Accès validé."));
    carte.querySelectorAll("[data-ref]").forEach(b => b.onclick = () => agir(url(b.dataset.ref, "validation?accepter=false"), "Demande refusée."));
    carte.querySelectorAll("[data-renv]").forEach(b => b.onclick = () => agir(url(b.dataset.renv, "verification/renvoi"), "Lien de confirmation renvoyé (au plus un par minute)."));
    carte.querySelectorAll("[data-conf]").forEach(b => b.onclick = () => {
      if(b.dataset.ok !== "1"){ b.dataset.ok = "1"; b.textContent = "Confirmer (adresse connue)"; return; }
      agir(url(b.dataset.conf, "email-verifie"), "Adresse confirmée par l'administrateur.");
    });
    carte.querySelectorAll("[data-role]").forEach(sel => sel.onchange = () => agir(url(sel.dataset.role, `role?role=${sel.value}`), `Rôle changé : ${NOMS_ROLES[sel.value]}.`));
    // suppression : irréversible, donc deux clics (le second confirme) ; la session et les alertes SMS du compte partent aussi
    carte.querySelectorAll("[data-sup]").forEach(b => b.onclick = async () => {
      if(b.dataset.ok !== "1"){ b.dataset.ok = "1"; b.textContent = "Confirmer la suppression définitive";
        setTimeout(() => { if(b.isConnected && b.dataset.ok === "1"){ b.dataset.ok = ""; b.textContent = "Supprimer"; } }, 6000); return; }
      try { const d = await envoyer("DELETE", `/comptes/${encodeURIComponent(b.dataset.sup)}`); await rendre();
        carte.querySelector("#aw-comptes-res").textContent = `Compte ${d.email} supprimé${d.alertes_sms_retirees ? ` (et ${d.alertes_sms_retirees} alerte(s) SMS)` : ""}. La personne devra se réinscrire.`; }
      catch(err){ res.textContent = err.message; }
    });
    carte.querySelectorAll("[data-st]").forEach(b => b.onclick = () => {
      if(b.dataset.actif === "0" && b.dataset.ok !== "1"){ b.dataset.ok = "1"; b.textContent = "Confirmer"; return; }
      agir(url(b.dataset.st, `statut?actif=${b.dataset.actif === "1"}`), b.dataset.actif === "1" ? "Compte réactivé." : "Compte désactivé : sa session est fermée.");
    });
  };
  try { await rendre(); } catch(e){ carte.remove(); }
}
A.installerComptesService = installerComptesService;

// ------------------------------------------------------------------ mode test UX1 / VIS1 (?test_ux=1)
// docs/PROTOCOLE_UX1_VIS1.md : chronometre des taches (UX1) et temps d'affichage des vues (VIS1), gardes dans
// ce navigateur et exportes en CSV ; rien n'est envoye au service.
const TACHES_UX = ["Peut-on traiter les vignes de Mornag demain matin ?", "D'où vient le vent à Nabeul maintenant, et à quelle vitesse ?",
  "Y a-t-il une alerte de vent fort aujourd'hui ?", "Combien d'eau faut-il pour 2 hectares de tomates aujourd'hui ?",
  "Le vent est-il plus fort à 100 m qu'au sol au-dessus de Djebel Ressas ?"];
const CLE_UX = "aw-test-ux";
function lignesUX(){ try { return JSON.parse(localStorage.getItem(CLE_UX) || "[]"); } catch(e){ return []; } }
function noterUX(l){ const t = lignesUX(); t.push({ t: new Date().toISOString(), ...l }); try { localStorage.setItem(CLE_UX, JSON.stringify(t)); } catch(e){} }
// des le chargement (le premier calcul de la journee arrive avant la page) : temps de /atom-wind/direct
A.mesureRequete = P.get("test_ux") === "1"
  ? (chemin, ms) => { if(/\/atom-wind\/direct/.test(chemin)) noterUX({ type: "VIS1", vue: "chargement direct", ms: Math.round(ms) }); }
  : null;
function installerTestUX(){
  if(P.get("test_ux") !== "1" || document.getElementById("aw-ux")) return;
  // VIS1 : clic sur une vue -> deux images plus tard (rendu fait)
  const VUES = { tableau: "tableau de bord", carte3d: "vent 3D" };
  window.addEventListener("hashchange", () => {
    const v = location.hash.slice(1); if(!VUES[v]) return;
    const t0 = performance.now();
    requestAnimationFrame(() => requestAnimationFrame(() => noterUX({ type: "VIS1", vue: VUES[v], ms: Math.round(performance.now() - t0) })));
  });
  const pan = document.createElement("aside");
  pan.id = "aw-ux"; pan.className = "card";
  pan.style.cssText = "position:fixed;left:12px;bottom:12px;z-index:70;width:min(360px,calc(100vw - 24px));max-height:60vh;overflow:auto;font-size:13px";
  let courante = null, t0 = 0;
  const rendre = () => {
    const L = lignesUX(), ux = L.filter(l => l.type === "UX1");
    pan.innerHTML = `<b>Test UX1 / VIS1</b> <span class="muted">(docs/PROTOCOLE_UX1_VIS1.md)</span>`
      + `<div style="display:flex;gap:6px;margin:6px 0"><input class="inp" id="aw-ux-p" placeholder="participant (P1 à P5)" style="flex:1" value="${echapper(pan.dataset.p || "")}"></div>`
      + TACHES_UX.map((q, i) => `<div style="margin:4px 0"><b>${i + 1}.</b> ${q}<br>`
        + (courante === i ? `<button class="btn primary" data-fin="ok">Trouvé</button> <button class="btn" data-fin="abandon">Abandon</button>`
                          : `<button class="btn" data-go="${i}">Démarrer</button>`) + `</div>`).join("")
      + `<p style="margin:6px 0">${ux.length} mesure(s) UX1, ${L.length - ux.length} mesure(s) VIS1.</p>`
      + `<button class="btn" id="aw-ux-csv">Exporter</button> <button class="btn" id="aw-ux-raz">Effacer</button>`;
    pan.querySelector("#aw-ux-p").oninput = e => { pan.dataset.p = e.target.value.trim(); };
    pan.querySelectorAll("[data-go]").forEach(b => b.onclick = () => { courante = +b.dataset.go; t0 = performance.now(); rendre(); });
    pan.querySelectorAll("[data-fin]").forEach(b => b.onclick = () => {
      const s = (performance.now() - t0) / 1000;
      noterUX({ type: "UX1", participant: pan.dataset.p || "?", tache: courante + 1, resultat: b.dataset.fin,
                secondes: Math.round(s * 10) / 10, sous_30_s: b.dataset.fin === "ok" && s < 30 });
      courante = null; rendre();
    });
    pan.querySelector("#aw-ux-csv").onclick = () => {
      const L2 = lignesUX(), cols = ["t", "type", "participant", "tache", "resultat", "secondes", "sous_30_s", "vue", "ms"];
      const csv = [cols.join(";")].concat(L2.map(l => cols.map(c => l[c] ?? "").join(";"))).join("\n");
      const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      a.download = `atom_wind_test_ux_${new Date().toISOString().slice(0, 10)}.csv`; a.click();
    };
    pan.querySelector("#aw-ux-raz").onclick = e => {
      if(e.target.dataset.ok !== "1"){ e.target.dataset.ok = "1"; e.target.textContent = "Confirmer l'effacement"; return; }
      try { localStorage.removeItem(CLE_UX); } catch(err){} rendre();
    };
  };
  document.body.appendChild(pan); rendre();
}
A.installerTestUX = installerTestUX;

// ------------------------------------------------------------------ supervision (BF41), admin seulement
async function installerSupervision(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-sup")) return;
  let d; try { d = await get("/supervision", 60000); } catch(e){ return; }
  const carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-sup";
  const tr = Object.entries(d.temps_reel).filter(([, v]) => typeof v === "object")
    .map(([date, v]) => `<li>${fmtDate(date)} : ${v.heures_mesurees} heures mesurées, stations ${v.stations_du_jour.join(", ") || "aucune"}, `
      + `calculé il y a ${v.calcule_il_y_a_min} min</li>`).join("");
  const a = d.acces_24h;
  carte.innerHTML = `<h3>Supervision du service</h3>`
    + `<p class="muted" style="font-size:13px;margin:0 0 6px">État du ${fmtInstant(d.genere_utc)}</p>`
    + `<ul style="margin:4px 0">${tr || "<li>aucune journée calculée depuis le démarrage</li>"}</ul>`
    + `<p style="margin:4px 0">Accès sur 24 h : ${a.requetes ?? 0} requêtes, ${a.refus ?? 0} refus, ${a.erreurs_serveur ?? 0} erreurs serveur, `
    + `${a.cles_actives ?? 0} clé(s) active(s) ; durée médiane ${a.duree_mediane_ms ?? "—"} ms.</p>`
    + `<p style="margin:4px 0">Historique : ${d.historique.journees} journée(s), ${String(d.historique.taille_mo).replace(".", ",")} Mo. `
    + `Notifications déjà envoyées : ${d.notifications.alertes_deja_envoyees}.</p>`;
  vue.appendChild(carte);
}
A.installerSupervision = installerSupervision;

// ------------------------------------------------------------------ abonnés aux alertes SMS (E12), admin
async function installerAbonnesService(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-abonnes")) return;
  let l; try { l = (await get("/sms/abonnes", 15000)).abonnes; } catch(e){ return; }
  const carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-abonnes";
  carte.innerHTML = `<h3>Abonnés aux alertes SMS</h3><p class="muted" style="font-size:14px;margin:0 0 8px">Numéros masqués. Chaque utilisateur gère ses alertes depuis l'espace Alertes.</p>`
    + (l.length ? `<div class="tblwrap"><table class="t"><thead><tr><th>Numéro</th><th>Lieu</th><th>Alerte</th><th>Langue</th><th>Depuis</th></tr></thead><tbody>`
      + l.map(a => `<tr><td>${echapper(a.numero)}</td><td>${echapper(a.lieu || `${a.lat}, ${a.lon}`)}</td><td>${echapper(a.activite || "vent du jour")}</td><td>${a.langue}</td><td>${a.depuis ? fmtInstant(a.depuis) : "—"}</td></tr>`).join("")
      + `</tbody></table></div>` : `<p class="muted" style="margin:0">Aucun abonné pour l'instant.</p>`);
  vue.appendChild(carte);
}
A.installerAbonnesService = installerAbonnesService;

// ------------------------------------------------------------------ comptes du service (BF19, BF43)
// Quand le service repond, la connexion et l'inscription de la page passent par ses comptes (mot de passe
// verifie et garde par le serveur, role relu a chaque requete) ; sinon la page garde ses comptes locaux.
A.comptesServeur = () => true;
async function requeteJSON(methode, chemin, corps){
  const h = { "Content-Type": "application/json", ...entetes() };
  let r;
  try { r = await fetch(`${BASE}/api${chemin}`, { method: methode, headers: h, body: corps ? JSON.stringify(corps) : undefined }); }
  catch(e){ return { erreur: "Le service ATOM Wind ne répond pas. Réessayez dans un instant." }; }
  const d = await r.json().catch(() => ({}));
  if(!r.ok) sessionRefusee(r.status);
  return r.ok ? d : { erreur: typeof d.detail === "string" ? d.detail : `erreur ${r.status}`, statut: r.status };
}
const postJSON = (chemin, corps) => requeteJSON("POST", chemin, corps || {});
// accès à l'API pour les onglets chargés à part (reseau-aneo-plan.js, services-atom.js) : même adresse, même clé ou session
A.api = { get: (chemin, ms) => get(chemin, ms), post: postJSON, blob };
const MESSAGES_COMPTE = {
  "adresse ou mot de passe incorrect, ou compte desactive": "E-mail ou mot de passe incorrect, ou compte désactivé.",
  "trop d'essais : reessayer dans 15 minutes": "Trop d'essais : réessayez dans 15 minutes.",
  "un compte existe deja pour cette adresse": "Un compte existe déjà avec cette adresse. Connectez-vous.",
  "adresse e-mail non verifiee : ouvrez le lien recu par e-mail": "Adresse e-mail pas encore confirmée : ouvrez le lien reçu par e-mail (pensez aux courriers indésirables).",
  "lien invalide ou deja utilise": "Ce lien n'est pas valable ou a déjà servi. Demandez-en un nouveau.",
  "lien expire : demandez un nouveau lien": "Ce lien a expiré. Demandez-en un nouveau.",
  "mot de passe : 8 caracteres au moins, dont un chiffre": "Le mot de passe doit faire 8 caractères au moins, dont un chiffre.",
};
const message = e => MESSAGES_COMPTE[e] || e;
A.connexionServeur = async (email, mdp) => {
  poserSession(null);
  const d = await postJSON("/comptes/connexion", { email, mot_de_passe: mdp });
  if(d.erreur) return { erreur: message(d.erreur), nonVerifie: d.statut === 403 };
  poserSession(d.cle);
  return d;
};
A.inscriptionServeur = async (nom, email, mdp, role) => {
  const d = await postJSON("/comptes", { email, mot_de_passe: mdp, nom, role: ["public", "agri", "eolien"].includes(role) ? role : "public" });
  return d.erreur ? { erreur: message(d.erreur) } : d;
};
A.verifierEmail = async jeton => { const d = await postJSON("/comptes/verification", { jeton }); return d.erreur ? { erreur: message(d.erreur) } : d; };
A.renvoyerVerification = async email => { const d = await postJSON("/comptes/verification/renvoi", { email }); return d.erreur ? { erreur: message(d.erreur) } : d; };
A.motDePasseOublie = async email => { const d = await postJSON("/comptes/mot-de-passe/oubli", { email }); return d.erreur ? { erreur: message(d.erreur) } : d; };
A.nouveauMotDePasse = async (jeton, mdp) => { const d = await postJSON("/comptes/mot-de-passe/reinitialisation", { jeton, mot_de_passe: mdp }); return d.erreur ? { erreur: message(d.erreur) } : d; };
A.deconnexionServeur = () => { if(session()){ postJSON("/comptes/deconnexion").catch(() => {}); poserSession(null); } };
// session gardée dans ce navigateur : le compte est relu sur le serveur au démarrage
A.compteCourant = async () => {
  if(!session()) return null;
  try { const d = await get("/comptes/moi", 15000); if(!d.compte) poserSession(null); return d.compte || null; }
  catch(e){ if(e.auth) poserSession(null); return null; }
};
// alertes SMS du compte connecté (/api/comptes/moi/abonnements)
const MESSAGES_ALERTE = {
  "numero au format international E.164 (ex. +21612345678)": "Numéro au format international, par exemple +216 12 345 678.",
  "ce numero recoit deja les alertes d'un autre compte": "Ce numéro reçoit déjà les alertes d'un autre compte.",
  "connectez-vous a un compte ATOM Wind pour gerer vos alertes": "Connectez-vous pour gérer vos alertes.",
};
A.mesAbonnements = async () => { const d = await requeteJSON("GET", "/comptes/moi/abonnements"); return d.erreur ? [] : d.abonnements; };
A.ajouterAbonnement = async o => { const d = await postJSON("/comptes/moi/abonnements", o); return d.erreur ? { erreur: MESSAGES_ALERTE[d.erreur] || d.erreur } : d; };
A.retirerAbonnement = async numero => { const d = await requeteJSON("DELETE", `/comptes/moi/abonnements/${encodeURIComponent(numero)}`); return d.erreur ? { erreur: MESSAGES_ALERTE[d.erreur] || d.erreur } : d; };
// le compte du serveur dans la liste locale de la page (sans mot de passe : seul le serveur le connait)
A.utilisateurLocal = (c, USERS) => {
  let u = USERS.find(x => x.email === c.email);
  if(!u){ u = { id: "s" + Date.now(), email: c.email, created: (c.cree_le || "").slice(0, 10) }; USERS.push(u); }
  Object.assign(u, { name: c.nom || c.email.split("@")[0], role: c.role, requested: c.demande_role || null,
                     status: c.statut === "desactive" ? "désactivé" : "actif", serveur: true });
  return u;
};

// ------------------------------------------------------------------ série au moyeu (TB4)
// Vent au moyeu du parc virtuel (E2, 80 m) heure par heure, avec l'intervalle à 80 % : sigma calibré du
// modèle s'il est servi, sinon quantiles 10-90 % de l'ensemble GEFS à 100 m (aux heures calculées), et
// les seuils de l'éolienne de référence (démarrage, puissance nominale, arrêt).
const SEUILS_EOL = [[3, "démarrage 3 m/s"], [12, "nominale 12 m/s"], [25, "arrêt 25 m/s"]];
function rendreSerieMoyeu(xy){
  const vue = document.getElementById("v-tableau"); if(!vue || !A.pret()) return;
  // vent au moyeu du parc virtuel : propre à l'exploitant éolien (espace réservé à son rôle)
  const roleCourant = A.workspace && A.workspace.role ? A.workspace.role() : "visiteur";
  if(roleCourant !== "eolien"){ const c = document.getElementById("aw-moyeu"); if(c) c.remove(); return; }
  let carte = document.getElementById("aw-moyeu");
  if(!carte){
    carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-moyeu";
    const ref = document.getElementById("tbday"); const apres = ref && ref.closest(".grid2");
    if(apres) apres.after(carte); else vue.appendChild(carte);
  }
  const pos = xy(36.612, 10.33);                                    // E2, au centre du parc virtuel
  const H = Array.from({ length: 24 }, (_, h) => h);
  const v = H.map(h => { const [u, w] = A.vent(pos[0], pos[1], 80, h); return Math.hypot(u, w); });
  // plus de bande d'incertitude (intervalle) sur la série : décision d'Ines, la page ne juge pas le calcul
  const W = 640, Hh = 220, pl = 52, pb = 24, mx = Math.max(27, ...v.map(z => z + 2));
  const X = h => pl + h * (W - pl - 10) / 23, Y = z => Hh - pb - z / mx * (Hh - pb - 12);
  const ligne = v.map((z, h) => `${h ? "L" : "M"}${X(h).toFixed(1)},${Y(z).toFixed(1)}`).join("");
  carte.hidden = false;
  carte.innerHTML = `<h3>Vent au moyeu du parc virtuel (80 m), heure par heure</h3>`
    + `<div class="tblwrap"><svg viewBox="0 0 ${W} ${Hh}" style="width:100%;min-width:320px;height:auto" role="img" aria-label="Vent au moyeu et seuils de l'éolienne">`
    + [0, 10, 20].map(z => `<line x1="${pl}" x2="${W - 10}" y1="${Y(z)}" y2="${Y(z)}" stroke="var(--line)"/><text x="${pl - 6}" y="${Y(z) + 4}" text-anchor="end">${z} m/s</text>`).join("")
    + [0, 6, 12, 18, 23].map(h => `<text x="${X(h)}" y="${Hh - 6}" text-anchor="middle">${h} h</text>`).join("")
    + SEUILS_EOL.map(([z, t]) => `<line x1="${pl}" x2="${W - 10}" y1="${Y(z)}" y2="${Y(z)}" stroke="var(--stop)" stroke-dasharray="4 4" stroke-opacity=".7"/>`
      + `<text x="${W - 12}" y="${Y(z) - 4}" text-anchor="end" font-size="11">${t}</text>`).join("")
    + `<path d="${ligne}" fill="none" stroke="var(--accent)" stroke-width="2.6"/></svg></div>`
    + `<p class="muted" style="font-size:13px;margin:6px 0 0">Seuils de l'éolienne de référence (générique 2 MW).</p>`;
}
A.rendreSerieMoyeu = rendreSerieMoyeu;

// ------------------------------------------------------------------ alertes du service (BF22, S4)
// Temps réel seulement : une station sans mesure depuis plusieurs heures (selon sa cadence : METAR
// horaire, SYNOP toutes les 3 h), et les heures dont le champ conserve mal la masse.
const POS_STATIONS = { DTTA: [36.8504, 10.2156], INM60729: [36.4333, 10.0833], INM60728: [36.45, 10.7333] };
const CADENCE_H = { DTTA: 1, INM60729: 3, INM60728: 3 };
// [code, [lat, lon], cadence en h] : les trois du Grand Tunis, ou les stations du domaine servi (aéroports)
function stationsSuivies(){
  const d = A.domaine();
  if(!d) return Object.entries(POS_STATIONS).map(([c, p]) => [c, p, CADENCE_H[c]]);
  return d.stations.filter(s => s.code).map(s => [s.code, [s.lat, s.lon], s.code.startsWith("INM") ? 3 : 1]);
}
A.alertesService = xy => {
  if(!A.pret() || !A.jour.temps_reel || A.jour.archive) return [];
  const out = [], I = A.jour.instants.filter(x => x.source === "temps reel" && Array.isArray(x.stations));
  for(const [code, [la, lo], c] of stationsSuivies()){
    if(!I.length) break;
    let n = 0; for(let k = I.length - 1; k >= 0 && !I[k].stations.includes(code); k--) n++;
    const nom = A.nomStation(code);
    if(n < 3 * c) continue;
    // station virtuelle (BF37) : le vent du modele a sa place, et l'erreur mesuree sans elle (REEL1)
    const h = heureLocale(), [pu, pv] = A.vent(...xy(la, lo), 10, h);
    const virt = isFinite(pu) ? ` Valeur de remplacement à ${h} h (station virtuelle) : ${Math.round(Math.hypot(pu, pv) * 3.6)} km/h.` : "";
    out.push({ cls: n >= 6 * c ? "warn" : "info", p: xy(la, lo), r: 1.5, t: `Station muette : ${nom}`, where: nom,
               when: n === I.length ? "aucune mesure reçue aujourd'hui" : `aucune mesure depuis ${n} h`,
               what: "Le vent est calculé sans cette station." + virt,
               todo: "Vérifier la transmission de la station (METAR ou SYNOP)." });
  }
  // l'alerte « résidu de masse » jugeait le calcul lui-même : retirée de la page (décision d'Ines)
  return out;
};

// bandeau du tableau de bord (TB1) : ce que le modèle servi a réellement fait à l'heure affichée
const NOMS_STATIONS = { DTTA: "Tunis-Carthage", INM60729: "Zaghouan", INM60728: "Nabeul" };
A.kpisReels = h => {
  if(!A.pret()) return [];
  const I = A.jour.instants || [], out = [];
  if(I.length){
    const x = I.reduce((b, y) => Math.abs(y.heure_locale - h) < Math.abs(b.heure_locale - h) ? y : b, I[0]);
    if(Array.isArray(x.stations)){
      const n = x.stations.length;
      const total = A.domaine() ? A.domaine().stations.length : 3;
      out.push(["Stations actives", `${n} / ${total}`, n ? x.stations.map(c => A.nomStation(c)).join(", ")
        : (x.source === "prevision" ? "prévision : pas encore de mesure" : "aucune mesure reçue")]);
    }
    if(typeof x.inference_s === "number")
      out.push(["Temps de reconstruction", `${x.inference_s.toFixed(2).replace(".", ",")} s`, "une carte 3D par ATOM Wind"]);
  }
  const g = A.grille; let s = 0, n = 0;
  for(let j = 0; j < g.ny; j += 3) for(let i = 0; i < g.nx; i += 3){
    const [u, v] = A.vent((i + .5) * g.pas_km, (j + .5) * g.pas_km, 80, h);
    if(isFinite(u) && isFinite(v)){ s += Math.hypot(u, v); n++; }
  }
  if(n) out.push(["Vent au moyeu (80 m)", `${(s / n).toFixed(1).replace(".", ",")} m/s`, "moyenne de la zone"]);
  // plus de « résidu de masse » (indicateur de qualité du calcul) dans le bandeau (décision d'Ines)
  return out;
};
A.htmlInfluence = (x, y) => {
  const d = A.valeurStations; if(!d || !A.grille) return "";
  const [i, j] = cellule(x, y), n = A.grille.nx * A.grille.ny;
  const L = d.codes.map((c, k) => [d.stations.find(s => s.code === c), d._influence[k*n + j*A.grille.nx + i]])
    .filter(q => q[0] && isFinite(q[1])).sort((a, b) => b[1] - a[1]);
  if(!L.length) return "";
  const t = L[0][1] >= .1 ? `La station qui compte le plus ici est <b>${L[0][0].nom}</b>.` : "Ici, aucune station ne change le vent de façon sensible : il vient surtout de GFS et du relief.";
  return `<p style="font-size:14px;margin:6px 0 0">${t} Sans chaque station, le vent à ${d.hauteur_carte_m} m changerait de : `
    + L.map(([s, v]) => `${s.nom} ${Math.round(v * 3.6)} km/h`).join(", ") + ".</p>";
};
function sourceEolien(){
  const el = document.getElementById("aw-eol-src"); if(!el || !A.pret()) return;
  el.textContent = A.jour.temps_reel
    ? "Aujourd'hui : vent en temps réel (GFS et anémomètres) ; demain et après-demain : prévision GFS affinée par ATOM Wind."
    : `Journée du ${fmtDate(A.jour.date)} (archive ATOM Wind) ; les prévisions de demain et d'après-demain viennent avec le temps réel.`;
}
A.sourceEolien = sourceEolien;

function echapper(t){ return String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// le panneau React de l'assistant (frontend/src/assistant) appelle l'API avec les mêmes en-têtes
A.entetes = entetes;
A.apiBase = BASE;
A.probaTraitement = (d, x, y) => {
  const e = A.ensembles[d]; if(!e || !A.grille) return null;
  const g = A.grille, [i, j] = cellule(x, y), n = g.nx * g.ny;
  return e.instants.map((it, k) => ({ h: it.heure_locale, p: e._p[k*n + j*g.nx + i] }));
};
A.htmlProba = (d, x, y) => {
  if(!(window.ATOM && A.jour && A.jour.temps_reel)) return "";
  const L = A.probaTraitement(d, x, y);
  if(!L) return `<p class="muted" style="font-size:13px;margin:4px 0">Probabilité (ensemble GEFS) : calcul en cours…</p>`;
  const e = A.ensembles[d];
  const cel = L.filter(r => isFinite(r.p)).map(r => {
    const pc = Math.round(r.p * 100), cls = pc >= 70 ? "good" : pc >= 40 ? "mid" : "bad";
    return `<span class="pill ${cls}">${String(r.h).padStart(2, "0")} h : ${pc} %</span>`;
  }).join(" ");
  return `<p style="margin:4px 0 8px;font-size:14px">Probabilité de pouvoir traiter (${e.instants[0].membres} scénarios GEFS) : ${cel}</p>`
    + `<p class="muted" style="font-size:12px;margin:0 0 8px">Ensemble partiel : vent à 10 m de chaque membre GEFS, profil vertical du GFS déterministe.</p>`;
};
A.choisirJour = async date => {
  const src = document.getElementById("aw-source");
  if(src) src.innerHTML = date === "direct" ? "Calcul du vent d'aujourd'hui par ATOM Wind…"
    : /^prev/.test(date) ? "Calcul de la prévision par ATOM Wind…"
    : date.startsWith("hist:") ? "Lecture de l'historique…"
    : `Calcul du vent du ${fmtDate(date)} par ATOM Wind…`;
  try { await chargerJour(date); A.erreur = null; }
  catch(e){ A.erreur = e; }
  A.banniere();
  if(A.rafraichir) A.rafraichir();
};
function journeeParDefaut(){
  const P2 = P.get("jour");
  if(P2 && A.jours.some(d => d.date === P2)) return P2;
  // la journée archivée la plus récente parmi les plus complètes
  const max = Math.max(...A.jours.map(d => d.instants.length));
  return A.jours.filter(d => d.instants.length === max).map(d => d.date).sort().pop();
}

function chargement(msg){
  let el = document.getElementById("aw-chargement");
  if(!el){
    el = document.createElement("div"); el.id = "aw-chargement"; el.setAttribute("role", "status");
    el.style.cssText = "position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;gap:14px;"
      + "align-items:center;justify-content:center;background:#0f1a22;color:#e8f1f5;"
      + "font:16px/1.4 system-ui,sans-serif;text-align:center;padding:24px";
    document.body.appendChild(el);
  }
  el.innerHTML = `<div style="font-size:22px;font-weight:700">ATOM Wind</div><div>${msg}</div><div class="aw-chargement-barre" aria-hidden="true"><i></i></div>`;
}
function lancerPage(){
  installerTestUX();
  const ov = document.getElementById("aw-chargement"); if(ov) ov.remove();
  A.startPage();
  banniere();
  A.workspace?.snapshot();
  const sel = document.getElementById("aw-jour");
  if(sel) sel.onchange = () => A.choisirJour(sel.value);
  if(A.pret() && A.jour.temps_reel && !A.jour.prevision && A.setHour) A.setHour(heureLocale());
  if(A.pret() && A.jour.temps_reel) chargerTrois().then(chargerEnsembles);
  if(A.pret()){ chargerAtlas(); sourceEolien(); installerValeurStations(); installerRapports(); }
  chargerQualite();
  window.addEventListener("atom:retry", () => location.reload());
  // temps réel : nouvelles mesures et nouvelle échéance toutes les 30 min
  setInterval(() => {
    if(!A.jour || !A.jour.temps_reel || A.jour.archive) return;
    A.trois = {}; A.ensembles = {};
    A.choisirJour(A.jour.cle || "direct").then(chargerTrois).then(chargerEnsembles);
  }, 30 * 60 * 1000);
}
function heureLocale(){
  // « fr-FR » formate « 15 h » (NaN une fois converti) : l'heure est lue sur ses parties, sans texte
  const h = new Intl.DateTimeFormat("en-GB", { hour: "numeric", hourCycle: "h23", timeZone: "Africa/Tunis" })
    .formatToParts(new Date()).find(x => x.type === "hour");
  return h ? Number(h.value) % 24 : new Date().getUTCHours();
}

async function demarrer(){
  chargement("Connexion au service ATOM Wind…");
  const relief = chargerRelief();
  try {
    try { A.domaines = (await get("/atom-wind/domaines", 15000)).domaines || []; } catch(e){ A.domaines = []; }
    if(DOMAINE === null) DOMAINE = A.domaines.some(d => d.nom === "national") ? "national" : "";
    A.grille = await get("/atom-wind/grille", 20000);
    // Grand Tunis à 500 m (géographie de la maquette) ou domaine géoréférencé (national) ; un mini-jeu
    // synthétique sans ancrage géographique reste servi au laboratoire seulement
    if(!A.estGrandTunis(A.grille) && !A.domaine()){
      throw new Error("la grille disponible n'est ni le Grand Tunis à 500 m ni un domaine géoréférencé ; ouvrez le laboratoire pour inspecter ce jeu de données");
    }
    A.z0 = Float32Array.from(A.grille.z0_m.flat());
    const J = await get("/atom-wind/jours", 20000);
    A.jours = J.jours || []; A.direct = J.direct || null;
    try { A.historique = (await get("/atom-wind/historique", 15000)).journees || []; } catch(e){ A.historique = []; }
    if(!A.jours.length && !A.direct) throw new Error("aucune journée servie");
    try { const p = await get("/atom-wind/reseau-aneo", 20000);
          A.plan = p.features.filter(f => f.properties.type === "capteur ANEO propose")
                             .map(f => ({ lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], ...f.properties })); }
    catch(e){ A.plan = null; }
    let fait = false;
    if(A.direct && P.get("jour") === null){
      chargement("Calcul du vent d'aujourd'hui en temps réel par ATOM Wind (jusqu'à une minute)…");
      try { await chargerJour("direct"); fait = true; }
      catch(e){ if(e.auth) throw e; console.warn("ATOM Wind : temps réel indisponible", e); A.avertDirect = e.message; }
    }
    if(!fait){
      const jd = journeeParDefaut();
      chargement(`Calcul du vent du ${fmtDate(jd)} par ATOM Wind (jusqu'à une minute au premier chargement)…`);
      await chargerJour(jd);
    }
  } catch(e){
    A.erreur = e; A.grille = null; A.jour = null;   // tout ou rien : l'interface montre l'état du service
    console.warn("ATOM Wind : données de vent indisponibles", e);
  }
  await relief;
  lancerPage();
}
if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", demarrer); else demarrer();
})();

export {};
