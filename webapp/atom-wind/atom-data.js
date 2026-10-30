/* ATOM Wind — couche de données : branche la maquette sur l'API PC-FNO3D.
 *
 * Charge, avant le script de la page, le relief réel, les journées servies et le champ du
 * modèle (GET /api/atom-wind/grille, /jours, /jour?date=…, /reseau-aneo), puis lance le
 * script de la page (<script type="text/plain" id="atom-main">). Si l'API est injoignable,
 * la page tourne sur sa journée d'exemple et le bandeau le dit.
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
const NODATA = -32768;

function cle(){ try { return localStorage.getItem(KEY_STORAGE); } catch(e) { return null; } }
function poserCle(k){ try { if(k) localStorage.setItem(KEY_STORAGE, k); else localStorage.removeItem(KEY_STORAGE); } catch(e) {} }
// session d'un compte ATOM Wind (BF43) : prime sur la cle d'API ; derriere le Worker d'atom.zelqane.com elle
// voyage dans X-ATOM-Session (le Worker remplace sinon toute cle par sa cle de lecture)
const SESSION_STORAGE = "aw-session-cle";
function session(){ try { return localStorage.getItem(SESSION_STORAGE); } catch(e) { return null; } }
function poserSession(k){ try { if(k) localStorage.setItem(SESSION_STORAGE, k); else localStorage.removeItem(SESSION_STORAGE); } catch(e) {} }
function entetes(){
  const s = session(), k = cle();
  return s ? { "X-API-Key": s, "X-ATOM-Session": s } : k ? { "X-API-Key": k } : {};
}

async function get(path, ms = 90000){
  const h = {}; Object.assign(h, entetes());
  const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(`${BASE}/api${path}`, { headers: h, signal: ctl.signal });
    if(r.status === 401 || r.status === 403){ const e = new Error("clé d'API refusée ou absente"); e.auth = true; throw e; }
    if(!r.ok){
      // 503 du Worker atom.zelqane.com quand le Colab dort : {error, message, detail}
      let d = ""; try { const j = await r.json(); d = j.message || j.detail || ""; } catch(e) {}
      throw new Error(r.status === 503 && d ? d : `${path} : ${r.status} ${d}`);
    }
    // Worker atom.zelqane.com : service hors ligne, dernier état réel connu (cahier v12, 17.1)
    const de = r.headers.get("x-atom-dernier-etat");
    if(de !== null) A.dernierEtat = de || "date inconnue";
    return await r.json();
  } catch(e){
    if(e.name === "AbortError") throw new Error("le service PC-FNO3D ne répond pas (Colab endormi ?)");
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

A.pret = () => !!(A.grille && A.jour && A.jour._vent);
A.aSigma = () => !!(A.pret() && A.jour._sigma);

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
A.sigma = (x, y, z, h) => {
  const J = A.jour; if(!J || !J._sigma) return null;
  const [i, j] = cellule(x, y), [a, b, w] = crochets(J, h), z0 = A.z0[j*A.grille.nx + i];
  const zz = Math.max(z, J.hauteurs_agl_m[0]);
  const sa = auNiveau(J, J._sigma, 1, a, zz, i, j, 0, z0), sb = auNiveau(J, J._sigma, 1, b, zz, i, j, 0, z0);
  return (1-w)*sa + w*sb;
};
// espace agriculteur, jour d = 0 (aujourd'hui), 1 (demain), 2 (après-demain) : seulement en temps
// réel ; sinon la page garde ses valeurs d'exemple (et le dit)
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
  const g = A.grille; if(!g) return;
  if(g.nx !== NX || g.ny !== NY){ console.warn("ATOM Wind : grille de l'API", g.nx, g.ny, "≠ page", NX, NY); }
  for(let j = 0; j < NY; j++) for(let i = 0; i < NX; i++){
    const k = j*NX + i, gi = Math.min(g.nx-1, i), gj = Math.min(g.ny-1, j);
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
};
A.noteFiabilite = () => {
  const f = A.aSigma() ? "Fiabilité du réseau actuel : incertitude calibrée du modèle (ensemble par masquage, DSO6)."
    : "Fiabilité indicative, selon la distance au capteur le plus proche (incertitude calibrée du modèle non servie).";
  const r = (A.plan && A.plan.length) ? ` Réseau ANEO : plan de déploiement OS6 (${A.plan.length} capteurs, placés là où l'erreur du modèle baissait le plus).`
    : " Réseau ANEO : disposition d'exemple (plan OS6 pas encore calculé).";
  return `<p class="muted" style="font-size:13px;margin-top:0">${f}${r}</p>`;
};
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
    const modele = A.jour.modele || A.grille.modele;
    if(A.jour.archive){
      const maj = fmtInstant(A.jour.mis_a_jour_utc);
      src.innerHTML = horsLigne + `<b>Historique</b> : vent du <b>${fmtDate(A.jour.date)}</b> tel que calculé par <b>PC-FNO3D</b> (${modele}) `
        + (A.jour.prevision ? `en prévision (GFS), dernier calcul le ${maj}.` : `en temps réel (GFS et anémomètres), dernier calcul le ${maj}.`);
      if(fr) fr.textContent = `Historique · ${fmtDate(A.jour.date)}`;
    } else if(A.jour.temps_reel && A.jour.prevision){
      const maj = new Date(A.jour.mis_a_jour_utc), hm = maj.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", timeZone: "Africa/Tunis" });
      src.innerHTML = horsLigne + `<b>Prévision</b> du vent pour le <b>${fmtDate(A.jour.date)}</b>, calculée par <b>PC-FNO3D</b> (${modele}) `
        + `à partir de la prévision GFS (pas encore de mesure des stations pour ce jour). Calculée à ${hm}.`
        + (A.jour.avertissement ? ` <span class="muted">(${A.jour.avertissement})</span>` : "");
      if(fr) fr.textContent = `Prévision · ${hm}`;
    } else if(A.jour.temps_reel){
      const maj = new Date(A.jour.mis_a_jour_utc), hm = maj.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", timeZone: "Africa/Tunis" });
      const passees = I.filter(x => x.source === "temps reel"), st = new Set(passees.flatMap(x => x.stations));
      const noms = { DTTA: "Tunis-Carthage", INM60729: "Zaghouan", INM60728: "Nabeul" };
      src.innerHTML = horsLigne + `<b>Temps réel</b> : vent d'aujourd'hui (${fmtDate(A.jour.date)}) calculé par <b>PC-FNO3D</b> (${modele}) `
        + `à partir de la grande échelle GFS et des anémomètres ${[...st].map(c => noms[c] || c).join(", ") || "(aucune mesure reçue)"}. `
        + `Heures passées : temps réel ; heures à venir : prévision. Mis à jour à ${hm}.`
        + (A.jour.avertissement ? ` <span class="muted">(${A.jour.avertissement})</span>` : "");
      if(fr) fr.textContent = `Temps réel · ${hm}`;
    } else {
      src.innerHTML = horsLigne + `Vent calculé par le modèle <b>PC-FNO3D</b> (${modele}) sur le Grand Tunis, `
        + `journée du <b>${fmtDate(A.jour.date)}</b> (jeu de test, jamais vu à l'entraînement). `
        + `Sorties du modèle à ${hh} (heure de Tunis), interpolées heure par heure. Ce ne sont pas des mesures en direct.`;
      if(fr) fr.textContent = `Modèle : ${modele}`;
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
    const raison = A.erreur ? ` <span class="muted">(API PC-FNO3D non jointe : ${A.erreur.message})</span>` : "";
    src.innerHTML = `Prototype : <b>journée d'exemple d'été</b>, calculée sur la zone réelle du Grand Tunis. Ce ne sont pas des mesures en direct.${raison}`;
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
  const d = j !== undefined ? await get(`/atom-wind/direct?j=${j}`, 240000)
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
  if(!r.ok) throw new Error(`${path} : ${r.status}`);
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
    + (tifs ? `<p style="font-size:13px;margin:4px 0">Export SIG (GeoTIFF, WGS 84) : ${tifs}</p>` : "")
    + `<p class="muted" style="font-size:12px;margin:0">${a.limites}</p>`;
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
  el.innerHTML = `<h3>Valeur de chaque station aujourd'hui</h3>`
    + `<p class="muted" style="font-size:14px;margin:0 0 8px">Le modèle est relancé sans chaque station, aux heures déjà mesurées : `
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
// Le tableau DSO du tableau de bord suit le modèle réellement servi (mis à jour à chaque version).
async function chargerQualite(){
  const bloc = document.querySelector("#v-tableau .dsos"); if(!bloc) return;
  let q; try { q = await get("/atom-wind/qualite", 20000); } catch(e){ return; }     // page : valeurs de la maquette
  A.qualite = q; if(A.rafraichir) A.rafraichir();               // le bandeau TB1 reprend le résidu de masse
  const nb = (v, n = 2) => v === null || v === undefined ? "—" : Number(v).toLocaleString("fr-FR", { maximumSignificantDigits: n });
  const pill = c => c === null ? `<span class="pill m">à mesurer</span>` : `<span class="pill ${c ? "good" : "mid"}">${c ? "atteint" : "non atteint"}</span>`;
  const titre = bloc.closest(".card") && bloc.closest(".card").querySelector("h3");
  if(titre) titre.innerHTML = `Qualité du modèle servi <span class="pill m">résultats réels</span>`;
  bloc.innerHTML = `<div><span>Écart moyen sur le champ 3D (u / v)</span><b>${nb(q.rmse_global_u_m_s)} / ${nb(q.rmse_global_v_m_s)} m/s</b></div>`
    + (q.r2_min_uv !== null ? `<div><span>Part du vent expliquée (R²)</span><b>≥ ${nb(q.r2_min_uv, 3)}</b></div>` : "")
    + (q.mae_direction_deg ? `<div><span>Erreur sur la direction</span><b>${nb(q.mae_direction_deg)}°</b></div>` : "")
    + q.dso.map(l => `<div><span>${l.code} ${l.libelle}${l.note ? ` <span class="muted" style="font-size:12px">(${l.note})</span>` : ""}</span>`
      + `<span>${l.valeur !== null ? `<b style="margin-right:6px">${nb(l.valeur, 3)}${l.unite ? " " + l.unite : ""}</b>` : ""}${pill(l.conforme)}</span></div>`).join("")
    + `<div class="muted" style="font-size:12px">Modèle ${q.run} (${q.modele}), ${q.instants_synoptiques} instants du test, évaluation ${q.evaluation}`
    + `${q.decision ? ` · ${q.decision.texte}` : ""}.</div>`;
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
  if(!r.ok) throw new Error(d.detail ? (typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail)) : `erreur ${r.status}`);
  return d;
}
async function installerClesService(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-cles")) return;
  try { await get("/keys", 15000); } catch(e){ return; }                       // pas admin : rien
  const carte = document.createElement("div");
  carte.className = "card"; carte.id = "aw-cles";
  vue.appendChild(carte);
  const rendre = async () => {
    const l = (await get("/keys", 15000)).cles;
    carte.innerHTML = `<h3>Clés d'API du service</h3>`
      + `<p class="muted" style="font-size:14px;margin:0 0 8px">Les vraies clés du service PC-FNO3D (le reste de cette page est une démonstration locale). Seule une empreinte est gardée ; un secret ne s'affiche qu'à la création.</p>`
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
// Les vrais comptes : validation des demandes d'accès éolien, désactivation et réactivation.
const NOMS_ROLES = { public: "Grand public", agri: "Agriculteur", eolien: "Exploitant éolien", admin: "Administrateur" };
async function installerComptesService(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-comptes")) return;
  try { await get("/comptes", 15000); } catch(e){ return; }                     // pas admin : rien
  const carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-comptes";
  vue.appendChild(carte);
  const rendre = async () => {
    const l = (await get("/comptes", 15000)).comptes;
    carte.innerHTML = `<h3>Comptes du service</h3>`
      + `<p class="muted" style="font-size:14px;margin:0 0 8px">Comptes enregistrés sur le serveur ; un accès éolien demandé attend votre validation.</p>`
      + `<div class="tblwrap"><table class="t"><thead><tr><th>Compte</th><th>Rôle</th><th>Demande</th><th>Statut</th><th></th></tr></thead><tbody>`
      + l.map(c => `<tr><td>${echapper(c.nom || "")}<br><span class="muted" style="font-size:12px">${echapper(c.email)}</span></td>`
        + `<td>${NOMS_ROLES[c.role] || c.role}</td><td>${c.demande_role ? NOMS_ROLES[c.demande_role] || c.demande_role : "—"}</td>`
        + `<td>${c.statut === "actif" ? "actif" : `<span class="pill mid">désactivé</span>`}</td><td style="white-space:nowrap">`
        + (c.demande_role ? `<button class="btn primary" data-val="${echapper(c.email)}" style="padding:2px 10px;font-size:13px">Valider</button> `
          + `<button class="btn" data-ref="${echapper(c.email)}" style="padding:2px 10px;font-size:13px">Refuser</button> ` : "")
        + (c.role === "admin" ? "" : `<button class="btn" data-st="${echapper(c.email)}" data-actif="${c.statut === "actif" ? "0" : "1"}" style="padding:2px 10px;font-size:13px">`
          + `${c.statut === "actif" ? "Désactiver" : "Réactiver"}</button>`) + `</td></tr>`).join("")
      + `</tbody></table></div><div id="aw-comptes-res" class="muted" style="font-size:13px;margin-top:6px"></div>`;
    const res = carte.querySelector("#aw-comptes-res");
    const agir = async (chemin) => { try { await envoyer("POST", chemin); await rendre(); } catch(err){ res.textContent = err.message; } };
    carte.querySelectorAll("[data-val]").forEach(b => b.onclick = () => agir(`/comptes/${encodeURIComponent(b.dataset.val)}/validation?accepter=true`));
    carte.querySelectorAll("[data-ref]").forEach(b => b.onclick = () => agir(`/comptes/${encodeURIComponent(b.dataset.ref)}/validation?accepter=false`));
    carte.querySelectorAll("[data-st]").forEach(b => b.onclick = () => {
      if(b.dataset.actif === "0" && b.dataset.ok !== "1"){ b.dataset.ok = "1"; b.textContent = "Confirmer"; return; }
      agir(`/comptes/${encodeURIComponent(b.dataset.st)}/statut?actif=${b.dataset.actif === "1"}`);
    });
  };
  try { await rendre(); } catch(e){ carte.remove(); }
}
A.installerComptesService = installerComptesService;

// ------------------------------------------------------------------ supervision (BF41), admin seulement
async function installerSupervision(){
  const vue = document.getElementById("v-admin"); if(!vue || document.getElementById("aw-sup")) return;
  let d; try { d = await get("/supervision", 60000); } catch(e){ return; }
  const carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-sup";
  const tr = Object.entries(d.temps_reel).filter(([, v]) => typeof v === "object")
    .map(([date, v]) => `<li>${fmtDate(date)} : ${v.heures_mesurees} heures mesurées, stations ${v.stations_du_jour.join(", ") || "aucune"}, `
      + `calculé il y a ${v.calcule_il_y_a_min} min ; ${v.heures_masse_hors_seuil} heure(s) au résidu de masse hors seuil</li>`).join("");
  const ec = d.ecart_modele_stations.stations ? Object.entries(d.ecart_modele_stations.stations)
    .map(([n, v]) => `${n} ${v.rmse_vect_m_s === null ? "—" : `${String(v.rmse_vect_m_s).replace(".", ",")} m/s`}`).join(", ") : "indisponible";
  const a = d.acces_24h;
  carte.innerHTML = `<h3>Supervision du service</h3>`
    + `<p class="muted" style="font-size:13px;margin:0 0 6px">Modèle ${echapper(d.modele)} · état du ${fmtInstant(d.genere_utc)}</p>`
    + `<ul style="margin:4px 0">${tr || "<li>aucune journée calculée depuis le démarrage</li>"}</ul>`
    + `<p style="margin:4px 0">Accès sur 24 h : ${a.requetes ?? 0} requêtes, ${a.refus ?? 0} refus, ${a.erreurs_serveur ?? 0} erreurs serveur, `
    + `${a.cles_actives ?? 0} clé(s) active(s) ; durée médiane ${a.duree_mediane_ms ?? "—"} ms.</p>`
    + `<p style="margin:4px 0">Historique : ${d.historique.journees} journée(s), ${String(d.historique.taille_mo).replace(".", ",")} Mo. `
    + `Notifications déjà envoyées : ${d.notifications.alertes_deja_envoyees}.</p>`
    + `<p style="margin:4px 0">Écart du modèle aux stations aujourd'hui (contrôle de cohérence) : ${ec}.</p>`;
  vue.appendChild(carte);
}
A.installerSupervision = installerSupervision;

// ------------------------------------------------------------------ comptes du service (BF19, BF43)
// Quand le service repond, la connexion et l'inscription de la page passent par ses comptes (mot de passe
// verifie et garde par le serveur, role relu a chaque requete) ; sinon la page garde ses comptes locaux.
A.comptesServeur = () => A.pret();
async function postJSON(chemin, corps){
  const h = { "Content-Type": "application/json", ...entetes() };
  const r = await fetch(`${BASE}/api${chemin}`, { method: "POST", headers: h, body: JSON.stringify(corps || {}) });
  const d = await r.json().catch(() => ({}));
  return r.ok ? d : { erreur: typeof d.detail === "string" ? d.detail : `erreur ${r.status}` };
}
const MESSAGES_COMPTE = {
  "adresse ou mot de passe incorrect, ou compte desactive": "E-mail ou mot de passe incorrect, ou compte désactivé.",
  "trop d'essais : reessayer dans 15 minutes": "Trop d'essais : réessayez dans 15 minutes.",
  "un compte existe deja pour cette adresse": "Un compte existe déjà avec cette adresse. Connectez-vous.",
};
const message = e => MESSAGES_COMPTE[e] || e;
A.connexionServeur = async (email, mdp) => {
  poserSession(null);
  const d = await postJSON("/comptes/connexion", { email, mot_de_passe: mdp });
  if(d.erreur) return { erreur: message(d.erreur) };
  poserSession(d.cle);
  return d;
};
A.inscriptionServeur = async (nom, email, mdp, role) => {
  const d = await postJSON("/comptes", { email, mot_de_passe: mdp, nom, role: ["public", "agri", "eolien"].includes(role) ? role : "public" });
  return d.erreur ? { erreur: message(d.erreur) } : d;
};
A.deconnexionServeur = () => { if(session()){ postJSON("/comptes/deconnexion").catch(() => {}); poserSession(null); } };
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
  let carte = document.getElementById("aw-moyeu");
  if(!carte){
    carte = document.createElement("div"); carte.className = "card"; carte.id = "aw-moyeu";
    const ref = document.getElementById("tbday"); const apres = ref && ref.closest(".grid2");
    if(apres) apres.after(carte); else vue.appendChild(carte);
  }
  const pos = xy(36.612, 10.33);                                    // E2, au centre du parc virtuel
  const H = Array.from({ length: 24 }, (_, h) => h);
  const v = H.map(h => { const [u, w] = A.vent(pos[0], pos[1], 80, h); return Math.hypot(u, w); });
  let bande = null, source = "";
  if(A.aSigma && A.aSigma()){
    bande = H.map(h => { const s = A.sigma(pos[0], pos[1], 80, h); return isFinite(s) ? [Math.max(0, v[h] - 1.28 * s), v[h] + 1.28 * s] : null; });
    source = "incertitude calibrée du modèle (ensemble par masquage)";
  } else if(A.jour.temps_reel && A.ensembles && A.ensembles[A.jour.cle === "prev1" ? 1 : A.jour.cle === "prev2" ? 2 : 0]){
    const e = A.ensembles[A.jour.cle === "prev1" ? 1 : A.jour.cle === "prev2" ? 2 : 0];
    const g = A.grille, [i, j] = cellule(pos[0], pos[1]), n = g.nx * g.ny, nq = 3;
    bande = H.map(() => null);
    e.instants.forEach((it, k) => {
      const base = ((k * 2 + 1) * nq) * n + j * g.nx + i;            // hauteur 100 m
      const q10 = e._q[base], q90 = e._q[base + 2 * n];
      if(isFinite(q10) && isFinite(q90)) bande[Math.round(it.heure_locale)] = [q10, q90];
    });
    source = "quantiles 10-90 % de l'ensemble GEFS à 100 m";
  }
  const W = 640, Hh = 220, pl = 52, pb = 24, mx = Math.max(27, ...v.map(z => z + 2));
  const X = h => pl + h * (W - pl - 10) / 23, Y = z => Hh - pb - z / mx * (Hh - pb - 12);
  const ligne = v.map((z, h) => `${h ? "L" : "M"}${X(h).toFixed(1)},${Y(z).toFixed(1)}`).join("");
  let poly = "";
  if(bande){
    const pts = bande.map((b, h) => b ? [h, b] : null).filter(Boolean);
    if(pts.length > 1) poly = `<path d="${pts.map(([h, b], k) => `${k ? "L" : "M"}${X(h)},${Y(b[1])}`).join("")}`
      + `${pts.slice().reverse().map(([h, b]) => `L${X(h)},${Y(b[0])}`).join("")}Z" fill="var(--accent)" opacity=".18"/>`
      + pts.map(([h, b]) => `<line x1="${X(h)}" x2="${X(h)}" y1="${Y(b[0])}" y2="${Y(b[1])}" stroke="var(--accent)" stroke-opacity=".5"/>`).join("");
  }
  carte.hidden = false;
  carte.innerHTML = `<h3>Vent au moyeu du parc virtuel (80 m), heure par heure</h3>`
    + `<div class="tblwrap"><svg viewBox="0 0 ${W} ${Hh}" style="width:100%;min-width:320px;height:auto" role="img" aria-label="Vent au moyeu avec intervalle et seuils">`
    + [0, 10, 20].map(z => `<line x1="${pl}" x2="${W - 10}" y1="${Y(z)}" y2="${Y(z)}" stroke="var(--line)"/><text x="${pl - 6}" y="${Y(z) + 4}" text-anchor="end">${z} m/s</text>`).join("")
    + [0, 6, 12, 18, 23].map(h => `<text x="${X(h)}" y="${Hh - 6}" text-anchor="middle">${h} h</text>`).join("")
    + SEUILS_EOL.map(([z, t]) => `<line x1="${pl}" x2="${W - 10}" y1="${Y(z)}" y2="${Y(z)}" stroke="var(--stop)" stroke-dasharray="4 4" stroke-opacity=".7"/>`
      + `<text x="${W - 12}" y="${Y(z) - 4}" text-anchor="end" font-size="11">${t}</text>`).join("")
    + poly + `<path d="${ligne}" fill="none" stroke="var(--accent)" stroke-width="2.6"/></svg></div>`
    + `<p class="muted" style="font-size:13px;margin:6px 0 0">${bande ? `Bande claire : intervalle à 80 % (${source}).` : "Intervalle non disponible : incertitude du modèle non servie et ensemble GEFS pas encore calculé."}`
    + ` Seuils de l'éolienne de référence (générique 2 MW).</p>`;
}
A.rendreSerieMoyeu = rendreSerieMoyeu;

// ------------------------------------------------------------------ alertes du service (BF22, S4)
// Temps réel seulement : une station sans mesure depuis plusieurs heures (selon sa cadence : METAR
// horaire, SYNOP toutes les 3 h), et les heures dont le champ conserve mal la masse.
const POS_STATIONS = { DTTA: [36.8504, 10.2156], INM60729: [36.4333, 10.0833], INM60728: [36.45, 10.7333] };
const CADENCE_H = { DTTA: 1, INM60729: 3, INM60728: 3 };
A.alertesService = xy => {
  if(!A.pret() || !A.jour.temps_reel || A.jour.archive) return [];
  const out = [], I = A.jour.instants.filter(x => x.source === "temps reel" && Array.isArray(x.stations));
  for(const [code, [la, lo]] of Object.entries(POS_STATIONS)){
    if(!I.length) break;
    let n = 0; for(let k = I.length - 1; k >= 0 && !I[k].stations.includes(code); k--) n++;
    const c = CADENCE_H[code], nom = NOMS_STATIONS[code];
    if(n < 3 * c) continue;
    // station virtuelle (BF37) : le vent du modele a sa place, et l'erreur mesuree sans elle (REEL1)
    const h = heureLocale(), [pu, pv] = A.vent(...xy(la, lo), 10, h);
    const reel = A.qualite && (A.qualite.stations || []).find(st => st.station === nom);
    const virt = isFinite(pu) ? ` Valeur de remplacement à ${h} h (station virtuelle) : ${Math.round(Math.hypot(pu, pv) * 3.6)} km/h`
      + (reel && reel.modele_m_s ? ` (erreur type sans elle : ± ${Math.round(reel.modele_m_s * 3.6)} km/h).` : ".") : "";
    out.push({ cls: n >= 6 * c ? "warn" : "info", p: xy(la, lo), r: 1.5, t: `Station muette : ${nom}`, where: nom,
               when: n === I.length ? "aucune mesure reçue aujourd'hui" : `aucune mesure depuis ${n} h`,
               what: "Le vent est calculé sans cette station : la fiabilité baisse autour d'elle." + virt,
               todo: "Vérifier la transmission de la station (METAR ou SYNOP)." });
  }
  const mauvais = A.jour.instants.filter(x => x.residu_masse_ok === false);
  if(mauvais.length)
    out.push({ cls: "warn", p: xy(36.65, 10.42), r: 6, at: mauvais[0].heure_locale, t: "Physique : résidu de masse au-dessus du seuil",
               where: "Grand Tunis", when: `${mauvais.length} heure(s), dès ${mauvais[0].heure_locale} h`,
               what: "Le champ calculé conserve mal la masse à ces heures : valeurs moins sûres.",
               todo: "Consulter la vue Fiabilité ; signaler à l'équipe si cela se répète." });
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
      out.push(["Stations actives", `${n} / 3`, n ? x.stations.map(c => NOMS_STATIONS[c] || c).join(", ")
        : (x.source === "prevision" ? "prévision : pas encore de mesure" : "aucune mesure reçue")]);
    }
    if(typeof x.inference_s === "number")
      out.push(["Temps de reconstruction", `${x.inference_s.toFixed(2).replace(".", ",")} s`, "une carte 3D par le modèle"]);
  }
  const g = A.grille; let s = 0, n = 0;
  for(let j = 0; j < g.ny; j += 3) for(let i = 0; i < g.nx; i += 3){
    const [u, v] = A.vent((i + .5) * g.pas_km, (j + .5) * g.pas_km, 80, h);
    if(isFinite(u) && isFinite(v)){ s += Math.hypot(u, v); n++; }
  }
  if(n) out.push(["Vent au moyeu (80 m)", `${(s / n).toFixed(1).replace(".", ",")} m/s`, "moyenne de la zone"]);
  const xi = I.length ? I.reduce((b, y) => Math.abs(y.heure_locale - h) < Math.abs(b.heure_locale - h) ? y : b, I[0]) : null;
  const d2 = A.qualite && A.qualite.dso.find(l => l.code === "DSO2");
  if(xi && typeof xi.residu_masse_rms === "number")
    out.push(["Résidu de masse", xi.residu_masse_rms.toExponential(1).replace(".", ","),
              xi.residu_masse_ok ? "cet instant, sous le seuil" : "cet instant, au-dessus du seuil"]);
  else if(d2 && d2.valeur !== null)
    out.push(["Résidu de masse", Number(d2.valeur).toExponential(1).replace(".", ","), d2.conforme ? "sous le seuil (test)" : "au-dessus du seuil (test)"]);
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
    : `Journée du ${fmtDate(A.jour.date)} (jeu de test) ; demain et après-demain restent des variantes d'exemple hors temps réel.`;
}
A.sourceEolien = sourceEolien;

// ------------------------------------------------------------------ assistant intégré (E6)
// Une question en langage naturel ; le service répond avec Claude et les outils d'ATOM Wind.
const NOMS_OUTILS = { vent_au_point: "vent en un point", profil_vertical: "profil vertical",
  decision_agricole: "décision agricole", production_eolienne: "production éolienne", alertes_vent: "alertes de vent" };
function echapper(t){ return String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function installerAssistant(){
  if(document.getElementById("aw-asst-btn")) return;
  const bt = document.createElement("button");
  bt.id = "aw-asst-btn"; bt.className = "btn primary"; bt.textContent = "Poser une question";
  bt.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:60;border-radius:999px;padding:10px 16px;box-shadow:0 4px 16px rgba(0,0,0,.2)";
  const pan = document.createElement("aside");
  pan.id = "aw-asst"; pan.className = "card"; pan.hidden = true;
  pan.setAttribute("aria-label", "Assistant ATOM Wind");
  pan.style.cssText = "position:fixed;right:16px;bottom:72px;z-index:61;width:min(420px,calc(100vw - 32px));max-height:min(70vh,560px);display:flex;flex-direction:column;gap:8px";
  pan.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center"><b>Assistant ATOM Wind</b>`
    + `<button class="x" aria-label="Fermer" id="aw-asst-x" style="font-size:20px;background:none;border:0;cursor:pointer">×</button></div>`
    + `<div id="aw-asst-fil" aria-live="polite" style="overflow:auto;flex:1;display:flex;flex-direction:column;gap:8px;font-size:14px">`
    + `<p class="muted" style="margin:0">Exemples : « Peut-on traiter les vignes de Mornag demain matin ? », « Quelle production pour une éolienne à Soliman demain ? »</p></div>`
    + `<form id="aw-asst-f" style="display:flex;gap:6px"><input id="aw-asst-q" class="inp" style="flex:1" maxlength="2000" placeholder="Votre question sur le vent" autocomplete="off">`
    + `<button class="btn primary" type="submit">Envoyer</button></form>`
    + `<p class="muted" style="font-size:11px;margin:0">Réponses générées par Claude à partir des données d'ATOM Wind ; vérifiez avant une décision importante.</p>`;
  document.body.append(bt, pan);
  const fil = pan.querySelector("#aw-asst-fil"), hist = [];
  const ajouter = (html, cls) => { const d = document.createElement("div"); d.className = cls;
    d.style.cssText = cls === "q" ? "align-self:flex-end;background:var(--accent,#1E6FB8);color:#fff;padding:6px 10px;border-radius:10px;max-width:85%"
      : "align-self:flex-start;background:var(--panel2,#f1f4f6);padding:6px 10px;border-radius:10px;max-width:92%;white-space:pre-wrap";
    d.innerHTML = html; fil.appendChild(d); fil.scrollTop = fil.scrollHeight; return d; };
  bt.onclick = () => { pan.hidden = !pan.hidden; if(!pan.hidden) pan.querySelector("#aw-asst-q").focus(); };
  pan.querySelector("#aw-asst-x").onclick = () => { pan.hidden = true; };
  pan.querySelector("#aw-asst-f").onsubmit = async e => {
    e.preventDefault();
    const inp = pan.querySelector("#aw-asst-q"), q = inp.value.trim(); if(!q) return;
    inp.value = ""; ajouter(echapper(q), "q");
    const att = ajouter("…", "r");
    try {
      const h = { "Content-Type": "application/json" }; Object.assign(h, entetes());
      const r = await fetch(`${BASE}/api/assistant`, { method: "POST", headers: h, body: JSON.stringify({ question: q, historique: hist.slice(-10) }) });
      const d = await r.json().catch(() => ({}));
      if(!r.ok) throw new Error(d.detail || d.message || `erreur ${r.status}`);
      const outils = [...new Set((d.outils || []).map(o => NOMS_OUTILS[o.nom] || o.nom))];
      att.innerHTML = echapper(d.reponse) + (outils.length ? `<div class="muted" style="font-size:11px;margin-top:4px">Sources : ${outils.join(", ")} (ATOM Wind)</div>` : "");
      hist.push({ role: "user", contenu: q }, { role: "assistant", contenu: d.reponse });
    } catch(err){ att.innerHTML = `<span class="muted">Assistant indisponible : ${echapper(err.message)}</span>`; }
  };
}
A.installerAssistant = installerAssistant;
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
  if(src) src.innerHTML = date === "direct" ? "Calcul du vent d'aujourd'hui par le modèle…"
    : /^prev/.test(date) ? "Calcul de la prévision par le modèle…"
    : date.startsWith("hist:") ? "Lecture de l'historique…"
    : `Calcul du vent du ${fmtDate(date)} par le modèle…`;
  try { await chargerJour(date); A.erreur = null; }
  catch(e){ A.erreur = e; }
  A.banniere();
  if(A.rafraichir) A.rafraichir();
};
function journeeParDefaut(){
  const P2 = P.get("jour");
  if(P2 && A.jours.some(d => d.date === P2)) return P2;
  // une journée d'été complète, comme la journée d'exemple de la maquette
  const ete = A.jours.filter(d => /-0[78]-/.test(d.date));
  const pool = ete.length ? ete : A.jours;
  return pool.slice().sort((a, b) => b.instants.length - a.instants.length)[0].date;
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
  el.innerHTML = `<div style="font-size:22px;font-weight:700">ATOM Wind</div><div>${msg}</div>`;
}
function lancerPage(){
  const ov = document.getElementById("aw-chargement"); if(ov) ov.remove();
  const s = document.getElementById("atom-main"), x = document.createElement("script");
  x.textContent = s.textContent; document.body.appendChild(x);
  banniere();
  const sel = document.getElementById("aw-jour");
  if(sel) sel.onchange = () => A.choisirJour(sel.value);
  if(A.pret() && A.jour.temps_reel && !A.jour.prevision && A.setHour) A.setHour(heureLocale());
  if(A.pret() && A.jour.temps_reel) chargerTrois().then(chargerEnsembles);
  if(A.pret()){ chargerAtlas(); sourceEolien(); installerAssistant(); installerValeurStations(); installerRapports(); chargerQualite(); installerClesService(); installerComptesService(); installerSupervision(); }
  // temps réel : nouvelles mesures et nouvelle échéance toutes les 30 min
  setInterval(() => {
    if(!A.jour || !A.jour.temps_reel || A.jour.archive) return;
    A.trois = {}; A.ensembles = {};
    A.choisirJour(A.jour.cle || "direct").then(chargerTrois).then(chargerEnsembles);
  }, 30 * 60 * 1000);
}
function heureLocale(){
  return Number(new Intl.DateTimeFormat("fr-FR", { hour: "numeric", hour12: false, timeZone: "Africa/Tunis" }).format(new Date())) % 24;
}

async function demarrer(){
  chargement("Connexion au modèle PC-FNO3D…");
  try {
    A.grille = await get("/atom-wind/grille", 20000);
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
      chargement("Calcul du vent d'aujourd'hui en temps réel par le modèle (jusqu'à une minute)…");
      try { await chargerJour("direct"); fait = true; }
      catch(e){ if(e.auth) throw e; console.warn("ATOM Wind : temps réel indisponible", e); A.avertDirect = e.message; }
    }
    if(!fait){
      const jd = journeeParDefaut();
      chargement(`Calcul du vent du ${fmtDate(jd)} par le modèle (jusqu'à une minute au premier chargement)…`);
      await chargerJour(jd);
    }
  } catch(e){
    A.erreur = e; A.grille = null; A.jour = null;   // tout ou rien : journée d'exemple
    console.warn("ATOM Wind : API PC-FNO3D indisponible, journée d'exemple", e);
  }
  lancerPage();
}
if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", demarrer); else demarrer();
})();
