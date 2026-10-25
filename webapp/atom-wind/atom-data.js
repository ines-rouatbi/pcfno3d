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
    return await r.json();
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

const A = { grille: null, jours: [], jour: null, ventV: null, sigmaV: null, plan: null,
            erreur: null, rafraichir: null };
window.ATOM = A;

A.pret = () => !!(A.grille && A.jour && A.ventV);
A.aSigma = () => !!(A.pret() && A.sigmaV);

// ------------------------------------------------------------------ échantillonnage
function cellule(x, y){
  const g = A.grille, p = g.pas_km;
  return [Math.min(g.nx-1, Math.max(0, Math.floor(x/p))), Math.min(g.ny-1, Math.max(0, Math.floor(y/p)))];
}
function crochets(h){                      // instants encadrant l'heure locale h, et poids
  const I = A.jour.instants, n = I.length;
  if(n === 1 || h <= I[0].heure_locale) return [0, 0, 0];
  if(h >= I[n-1].heure_locale) return [n-1, n-1, 0];
  let a = 0; while(a < n-2 && I[a+1].heure_locale <= h) a++;
  const t0 = I[a].heure_locale, t1 = I[a+1].heure_locale;
  return [a, a+1, (h - t0) / Math.max(t1 - t0, 1e-6)];
}
function lire(arr, nComp, it, ih, c, i, j){
  const g = A.grille, nH = A.jour.hauteurs_agl_m.length;
  return arr[((((it*nH + ih)*nComp + c)*g.ny + j)*g.nx) + i];
}
function auNiveau(arr, nComp, it, z, i, j, c, z0){
  const H = A.jour.hauteurs_agl_m, nH = H.length;
  if(z <= H[0]){
    // sous la première hauteur : profil logarithmique avec la rugosité de la maille
    let ih = 0, v = lire(arr, nComp, it, 0, c, i, j);
    while(!isFinite(v) && ih < nH-1){ ih++; v = lire(arr, nComp, it, ih, c, i, j); }
    if(!isFinite(v)) return 0;
    const zz = Math.max(z, z0 * 1.5), f = Math.log(zz / z0) / Math.log(H[ih] / z0);
    return v * Math.max(0, f);
  }
  let k = 0; while(k < nH-2 && H[k+1] < z) k++;
  const a = Math.min(1, (z - H[k]) / (H[k+1] - H[k]));
  const v0 = lire(arr, nComp, it, k, c, i, j), v1 = lire(arr, nComp, it, k+1, c, i, j);
  if(isFinite(v0) && isFinite(v1)) return (1-a)*v0 + a*v1;
  return isFinite(v1) ? v1 : isFinite(v0) ? v0 : 0;
}
A.vent = (x, y, z, h) => {
  const [i, j] = cellule(x, y), z0 = A.z0[j*A.grille.nx + i];
  const [a, b, w] = crochets(h);
  const ua = auNiveau(A.ventV, 2, a, z, i, j, 0, z0), va = auNiveau(A.ventV, 2, a, z, i, j, 1, z0);
  if(w === 0 || a === b) return [ua, va];
  const ub = auNiveau(A.ventV, 2, b, z, i, j, 0, z0), vb = auNiveau(A.ventV, 2, b, z, i, j, 1, z0);
  return [(1-w)*ua + w*ub, (1-w)*va + w*vb];
};
A.sigma = (x, y, z, h) => {
  if(!A.sigmaV) return null;
  const [i, j] = cellule(x, y), [a, b, w] = crochets(h), z0 = A.z0[j*A.grille.nx + i];
  const sa = auNiveau(A.sigmaV, 1, a, Math.max(z, A.jour.hauteurs_agl_m[0]), i, j, 0, z0);
  const sb = auNiveau(A.sigmaV, 1, b, Math.max(z, A.jour.hauteurs_agl_m[0]), i, j, 0, z0);
  return (1-w)*sa + w*sb;
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
function fmtDate(d){ const [y, m, j] = d.split("-"); return `${j}/${m}/${y}`; }
function banniere(){
  const src = document.getElementById("aw-source"), fr = document.getElementById("aw-fresh");
  const sel = document.getElementById("aw-jour");
  if(!src) return;
  if(A.pret()){
    const I = A.jour.instants, hh = I.map(x => `${String(Math.floor(x.heure_locale)).padStart(2,"0")} h`).join(", ");
    src.innerHTML = `Vent calculé par le modèle <b>PC-FNO3D</b> (${A.jour.modele || A.grille.modele}) sur le Grand Tunis, `
      + `journée du <b>${fmtDate(A.jour.date)}</b> (jeu de test, jamais vu à l'entraînement). `
      + `Sorties du modèle à ${hh} (heure de Tunis), interpolées heure par heure. Ce ne sont pas des mesures en direct.`;
    if(fr) fr.textContent = `Modèle : ${A.jour.modele || A.grille.modele}`;
    if(sel){
      sel.innerHTML = A.jours.map(d => `<option value="${d.date}"${d.date === A.jour.date ? " selected" : ""}>${fmtDate(d.date)} (${d.instants.length} instants)</option>`).join("");
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
A.banniere = banniere;

// ------------------------------------------------------------------ chargement
async function chargerJour(date){
  const d = await get(`/atom-wind/jour?date=${encodeURIComponent(date)}`, 180000);
  A.ventV = decoder(d.vent);
  A.sigmaV = d.sigma_vitesse ? decoder(d.sigma_vitesse) : null;
  A.jour = d;
}
A.choisirJour = async date => {
  const src = document.getElementById("aw-source");
  if(src) src.innerHTML = `Calcul du vent du ${fmtDate(date)} par le modèle…`;
  try { await chargerJour(date); A.erreur = null; }
  catch(e){ A.erreur = e; }
  banniere();
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

function lancerPage(){
  const s = document.getElementById("atom-main"), x = document.createElement("script");
  x.textContent = s.textContent; document.body.appendChild(x);
  banniere();
  const sel = document.getElementById("aw-jour");
  if(sel) sel.onchange = () => A.choisirJour(sel.value);
}

async function demarrer(){
  try {
    A.grille = await get("/atom-wind/grille");
    A.z0 = Float32Array.from(A.grille.z0_m.flat());
    A.jours = (await get("/atom-wind/jours")).jours;
    if(!A.jours.length) throw new Error("aucune journée servie");
    try { const p = await get("/atom-wind/reseau-aneo", 20000);
          A.plan = p.features.filter(f => f.properties.type === "capteur ANEO propose")
                             .map(f => ({ lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], ...f.properties })); }
    catch(e){ A.plan = null; }
    await chargerJour(journeeParDefaut());
  } catch(e){
    A.erreur = e; A.grille = null; A.jour = null; A.ventV = null;   // tout ou rien : journée d'exemple
    console.warn("ATOM Wind : API PC-FNO3D indisponible, journée d'exemple", e);
  }
  lancerPage();
}
if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", demarrer); else demarrer();
})();
