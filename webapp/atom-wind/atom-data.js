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

async function get(path, ms = 90000){
  const h = {}; const k = cle(); if(k) h["X-API-Key"] = k;
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

const A = { grille: null, jours: [], jour: null, plan: null,
            erreur: null, rafraichir: null };
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
    if(A.jour.temps_reel && A.jour.prevision){
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
        + A.jours.map(d => `<option value="${d.date}"${d.date === cle ? " selected" : ""}>${fmtDate(d.date)} (${d.instants.length} instants)</option>`).join("");
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
                            : await get(`/atom-wind/jour?date=${encodeURIComponent(date)}`, 180000);
  d.cle = date;
  d._vent = decoder(d.vent);
  d._sigma = d.sigma_vitesse ? decoder(d.sigma_vitesse) : null;
  d._t = d.t2m ? decoder(d.t2m) : null;
  d._hr = d.hr2m ? decoder(d.hr2m) : null;
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
  const h = {}; const k = cle(); if(k) h["X-API-Key"] = k;
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
      const h = { "Content-Type": "application/json" }; const k = cle(); if(k) h["X-API-Key"] = k;
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
  if(A.pret()){ chargerAtlas(); sourceEolien(); installerAssistant(); }
  // temps réel : nouvelles mesures et nouvelle échéance toutes les 30 min
  setInterval(() => {
    if(!A.jour || !A.jour.temps_reel) return;
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
