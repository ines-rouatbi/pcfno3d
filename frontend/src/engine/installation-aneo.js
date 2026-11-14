/* ATOM Wind — onglet d'administration « Installation ANEO » : où installer les capteurs dans une zone.
 *
 * L'administrateur clique une zone sur la carte (centre + rayon) et choisit le nombre de capteurs : l'application
 * propose les emplacements, du plus utile au moins utile, avec pour chacun ses coordonnées, le sol, la pente,
 * l'occupation du sol et le gain de fiabilité qu'il apporte à la zone.
 *
 * Méthode (indicative, calculée dans le navigateur, sans le modèle) :
 *  - fiabilité d'un point = la règle de l'écran « Données & fiabilité » : écart attendu selon la distance au capteur
 *    le plus proche (station existante : 0,45 + 1,9 (1 − e^(−d/11)) ; capteur ANEO : 0,35 + 0,75 (1 − e^(−d/3)) m/s) ;
 *  - emplacements possibles : terre ferme, sol ≤ 650 m, pente ≤ 20°, hors ville (rugosité < 0,5 m), à 1 km au moins
 *    du bord du domaine, des stations, des capteurs déjà prévus et des autres emplacements proposés
 *    (mêmes contraintes que le placement OS6, scripts/os6_reseau_aneo.py) ;
 *  - placement glouton : chaque nouveau capteur est mis là où il fait le plus baisser l'écart moyen de la zone.
 * Ce n'est PAS une mesure de l'erreur du modèle : le gain réel se mesure par le calcul OS6 (onglet Réseau ANEO),
 * qui relance PC-FNO3D avec les capteurs simulés. La page le dit à chaque proposition.
 */
(() => {
"use strict";
const A = window.ATOM = window.ATOM || {};
const MATS = [15, 20, 25, 30];                       // hauteurs des mâts Porteur
const SOL_MAX_M = 650, PENTE_MAX_DEG = 20, Z0_VILLE = .5, ECART_MIN_KM = 1;
const MAX_ZONE = 6000, MAX_CAND = 2500;
let zone = null, rayon = null, nombre = 5, mat = 20, avecPlan = true, fond = "fiabilite", res = null, message = "";
let fondCache = null;

const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? "—" : Number(v).toFixed(d).replace(".", ",");
const pct = v => `${Math.round(100 * v)} %`;

function grille(){ return A.grilleTerrain || null; }

// ------------------------------------------------------------------ calcul
function pente(g, i, j){
  const c = 1000 / g.G, k = (ii, jj) => g.elev[Math.min(g.NY - 1, Math.max(0, jj)) * g.NX + Math.min(g.NX - 1, Math.max(0, ii))];
  const dx = (k(i + 1, j) - k(i - 1, j)) / (2 * c), dy = (k(i, j + 1) - k(i, j - 1)) / (2 * c);
  return Math.atan(Math.hypot(dx, dy)) * 180 / Math.PI;
}
function occupation(z0, eau){
  if(eau) return "eau";
  if(z0 >= Z0_VILLE) return "ville";
  if(z0 >= .2) return "relief boisé ou accidenté";
  if(z0 >= .05) return "cultures, végétation basse";
  return "terrain dégagé";
}

function proposer(){
  const g = grille(); if(!g || !zone) { res = null; return; }
  const c = 1 / g.G, R = rayon, [cx, cy] = zone;
  const existants = g.stations.map(s => ({ p: s.p, f: g.sigmaStation }));
  if(avecPlan) for(const p of g.aneo) if(p) existants.push({ p, f: g.sigmaAneo });
  const i0 = Math.max(0, Math.floor((cx - R) * g.G)), i1 = Math.min(g.NX - 1, Math.ceil((cx + R) * g.G));
  const j0 = Math.max(0, Math.floor((cy - R) * g.G)), j1 = Math.min(g.NY - 1, Math.ceil((cy + R) * g.G));
  // cellules de terre de la zone (pas réduit si la zone est très grande ; chaque cellule gardée pèse pas²)
  let nTerre = 0;
  for(let j = j0; j <= j1; j++) for(let i = i0; i <= i1; i++){ const x = (i + .5) * c, y = (j + .5) * c;
    if((x - cx) ** 2 + (y - cy) ** 2 <= R * R && !g.wat[j * g.NX + i]) nTerre++; }
  if(!nTerre){ res = { vide: true }; return; }
  const pasZ = Math.max(1, Math.ceil(Math.sqrt(nTerre / MAX_ZONE))), pasC = Math.max(1, Math.ceil(Math.sqrt(nTerre / MAX_CAND)));
  const zx = [], zy = [], sg = [];
  for(let j = j0; j <= j1; j += pasZ) for(let i = i0; i <= i1; i += pasZ){ const x = (i + .5) * c, y = (j + .5) * c;
    if((x - cx) ** 2 + (y - cy) ** 2 > R * R || g.wat[j * g.NX + i]) continue;
    let s = 9; for(const e of existants) s = Math.min(s, e.f(Math.hypot(x - e.p[0], y - e.p[1])));
    zx.push(x); zy.push(y); sg.push(s); }
  const sigma0 = sg.slice();
  // emplacements possibles, et pourquoi les autres sont écartés
  const exclus = { eau: 0, altitude: 0, pente: 0, ville: 0, bord: 0, proche: 0 };
  const cand = [];
  for(let j = j0; j <= j1; j += pasC) for(let i = i0; i <= i1; i += pasC){ const x = (i + .5) * c, y = (j + .5) * c;
    if((x - cx) ** 2 + (y - cy) ** 2 > R * R) continue;
    const k = j * g.NX + i;
    if(g.wat[k]){ exclus.eau++; continue; }
    if(g.elev[k] > SOL_MAX_M){ exclus.altitude++; continue; }
    const pe = pente(g, i, j); if(pe > PENTE_MAX_DEG){ exclus.pente++; continue; }
    if(g.z0[k] >= Z0_VILLE){ exclus.ville++; continue; }
    if(x < ECART_MIN_KM || y < ECART_MIN_KM || x > g.W - ECART_MIN_KM || y > g.H - ECART_MIN_KM){ exclus.bord++; continue; }
    let dmin = 1e9; for(const e of existants) dmin = Math.min(dmin, Math.hypot(x - e.p[0], y - e.p[1]));
    if(dmin < ECART_MIN_KM){ exclus.proche++; continue; }
    cand.push({ x, y, k, pente: pe, dExist: dmin }); }
  const choisis = [], moy = a => a.reduce((s, v) => s + v, 0) / a.length;
  const bonne = a => a.filter(v => v < .9).length / a.length, faible = a => a.filter(v => v >= 1.5).length / a.length;
  for(let n = 0; n < nombre; n++){
    let best = null, bg = 0;
    for(const q of cand){
      if(choisis.some(p => Math.hypot(p.x - q.x, p.y - q.y) < ECART_MIN_KM)) continue;
      let gsum = 0;
      for(let z = 0; z < sg.length; z++){ const d = Math.hypot(zx[z] - q.x, zy[z] - q.y); if(d > 15) continue;
        const v = g.sigmaAneo(d); if(v < sg[z]) gsum += sg[z] - v; }
      if(gsum > bg){ bg = gsum; best = q; } }
    if(!best || bg < 1e-6) break;
    const avant = sg.slice();
    for(let z = 0; z < sg.length; z++) sg[z] = Math.min(sg[z], g.sigmaAneo(Math.hypot(zx[z] - best.x, zy[z] - best.y)));
    const [lat, lon] = g.latlon(best.x, best.y);
    let vm = null; try { vm = g.ventMoyen(best.x, best.y, mat); } catch(e) { vm = null; }
    choisis.push({ ...best, ordre: n + 1, lat, lon, sol: g.elev[best.k], z0: g.z0[best.k], occ: occupation(g.z0[best.k], false),
      sigmaPoint: avant.length ? (() => { let s = 9; for(const e of existants) s = Math.min(s, e.f(Math.hypot(best.x - e.p[0], best.y - e.p[1]))); return s; })() : null,
      gain: moy(avant) - moy(sg), bonneApres: bonne(sg), ventMoyen: vm }); }
  res = { R, n: choisis.length, demande: nombre, surface: nTerre * c * c, nCand: cand.length, exclus, choisis,
          avant: { moy: moy(sigma0), bonne: bonne(sigma0), faible: faible(sigma0) },
          apres: { moy: moy(sg), bonne: bonne(sg), faible: faible(sg) }, pasZ, pasC };
}

// ------------------------------------------------------------------ rendu
function rendre(){
  const R = document.getElementById("aneo-installation"); if(!R) return;
  const g = grille();
  if(!g){ R.innerHTML = `<div class="card"><p class="muted" style="margin:0">Chargement de la carte du domaine…</p></div>`; return; }
  if(rayon === null) rayon = Math.round(Math.max(3, Math.min(40, Math.min(g.W, g.H) / 6)));
  R.innerHTML = rendreBandeau(g)
    + `<div class="grid2 ia-haut">${rendreCarte(g)}${rendreReglages(g)}</div>`
    + rendreBilan() + rendreTable() + rendreSuite();
  dessiner();
  brancher(R);
}

function rendreBandeau(g){
  return `<div class="ia-bandeau" role="status"><b>Où installer les capteurs ANEO ?</b>`
    + `<span>Cliquez une zone sur la carte ${esc(g.ZONE)} : l'application propose les emplacements, du plus utile au moins utile. `
    + `Proposition indicative fondée sur la distance aux capteurs (même règle que l'écran « Données &amp; fiabilité ») ; `
    + `le gain réel en erreur du modèle se mesure par le calcul OS6 de l'onglet Réseau ANEO.</span></div>`;
}

function rendreCarte(g){
  const leg = fond === "fiabilite"
    ? `<span class="ia-leg"><i style="background:#3f9f6a"></i>fiabilité bonne</span><span class="ia-leg"><i style="background:#e0b23a"></i>moyenne</span><span class="ia-leg"><i style="background:#d0573f"></i>faible</span>`
    : `<span class="ia-leg"><i style="background:#b2cfde"></i>eau</span><span class="ia-leg"><i style="background:#c9c19c"></i>relief</span>`;
  return `<div class="card ia-carte"><div class="row ia-titre"><h3>Choisir la zone</h3>`
    + `<div class="seg" role="group" aria-label="Fond de carte"><button data-fond="fiabilite" aria-pressed="${fond === "fiabilite"}">Fiabilité actuelle</button>`
    + `<button data-fond="relief" aria-pressed="${fond === "relief"}">Relief</button></div></div>`
    + `<canvas id="ia-canvas" class="ia-canvas" role="img" tabindex="0" aria-label="Carte du domaine : cliquez pour choisir le centre de la zone d'installation"></canvas>`
    + `<div class="ia-legende">${leg}<span class="ia-leg"><i class="ia-tri"></i>station existante</span>`
    + `<span class="ia-leg"><i style="background:#7d8a91"></i>capteur ANEO déjà prévu</span><span class="ia-leg"><i style="background:#e8742a;border:2px solid #fff"></i>emplacement proposé</span></div>`
    + `<p class="muted ia-note">${zone ? `Centre de la zone : ${num(g.latlon(...zone)[0], 4)}° N, ${num(g.latlon(...zone)[1], 4)}° E. Cliquez ailleurs pour déplacer la zone.` : "Cliquez sur la carte pour choisir le centre de la zone."}</p></div>`;
}

function rendreReglages(g){
  const lieux = g.stations.map((s, i) => `<option value="${i}">${esc(s.n)}</option>`).join("");
  return `<div class="card ia-reglages"><h3>Réglages</h3>`
    + `<label class="ia-champ">Centrer sur<select id="ia-lieu"><option value="">— choisir —</option>${lieux}</select></label>`
    + `<label class="ia-champ">Rayon de la zone : <b>${rayon} km</b><input type="range" id="ia-rayon" min="2" max="${Math.max(10, Math.round(Math.min(g.W, g.H) / 2))}" step="1" value="${rayon}"></label>`
    + `<label class="ia-champ">Capteurs à installer : <b>${nombre}</b><input type="range" id="ia-nombre" min="1" max="20" step="1" value="${nombre}"></label>`
    + `<label class="ia-champ">Hauteur du mât<select id="ia-mat">${MATS.map(m => `<option value="${m}" ${m === mat ? "selected" : ""}>${m} m</option>`).join("")}</select></label>`
    + `<label class="ia-coche"><input type="checkbox" id="ia-plan" ${avecPlan ? "checked" : ""}> Tenir compte des capteurs ANEO déjà prévus (${g.aneo.length}${g.planOS6 ? ", plan OS6" : ", plan d'exemple"})</label>`
    + `<div class="ia-actions"><button class="btn" id="ia-csv" ${res && res.n ? "" : "disabled"}>Exporter en CSV</button>`
    + `<button class="btn" id="ia-geojson" ${res && res.n ? "" : "disabled"}>Exporter en GeoJSON</button></div>`
    + (message ? `<p class="ia-message" role="status">${esc(message)}</p>` : "")
    + `</div>`;
}

function rendreBilan(){
  if(!zone) return "";
  if(!res || res.vide) return `<div class="card"><p class="muted" style="margin:0">La zone choisie ne contient pas de terre ferme : choisissez une zone à terre.</p></div>`;
  const k = [["Surface de terre", `${Math.round(res.surface).toLocaleString("fr-FR")} km²`, `rayon ${res.R} km`],
             ["Emplacements possibles", res.nCand, `${Object.values(res.exclus).reduce((a, b) => a + b, 0)} écartés`],
             ["Fiabilité bonne", `${pct(res.avant.bonne)} → ${pct(res.apres.bonne)}`, "part de la zone, avant → après"],
             ["Écart attendu moyen", `${num(res.avant.moy)} → ${num(res.apres.moy)} m/s`, "indicatif, selon la distance aux capteurs"]];
  const ex = res.exclus, raisons = [["eau", "en mer ou en lac"], ["altitude", `sol au-dessus de ${SOL_MAX_M} m`], ["pente", `pente au-delà de ${PENTE_MAX_DEG}°`],
    ["ville", "en ville"], ["bord", "trop près du bord du domaine"], ["proche", "à moins de 1 km d'un capteur existant"]].filter(([c]) => ex[c]);
  return `<div class="kpis ia-kpis">${k.map(([t, v, s]) => `<div class="card kpi"><div class="k">${t}</div><div class="v">${v}</div><div class="s muted">${esc(s)}</div></div>`).join("")}</div>`
    + (res.n < res.demande ? `<div class="card"><p style="margin:0"><span class="pill mid">${res.n} sur ${res.demande}</span> Au-delà, un capteur de plus n'améliore plus la fiabilité de la zone, ou il ne reste plus d'emplacement possible.</p></div>` : "")
    + (raisons.length ? `<p class="muted ia-note">Emplacements écartés : ${raisons.map(([c, t]) => `${ex[c]} ${t}`).join(" ; ")}.</p>` : "");
}

function rendreTable(){
  if(!res || !res.n) return "";
  const lignes = res.choisis.map(p => `<tr><td><span class="ia-num">${p.ordre}</span></td><td>${num(p.lat, 5)}</td><td>${num(p.lon, 5)}</td>`
    + `<td>${num(p.sol, 0)} m</td><td>${num(p.pente, 1)}°</td><td>${esc(p.occ)}</td><td>${num(p.dExist, 1)} km</td>`
    + `<td>${num(p.sigmaPoint)} m/s</td><td>${num(p.gain, 3)} m/s</td><td>${pct(p.bonneApres)}</td>`
    + `<td>${p.ventMoyen === null || !isFinite(p.ventMoyen) ? "—" : `${num(p.ventMoyen, 1)} m/s`}</td></tr>`).join("");
  return `<div class="card"><h3>Emplacements proposés</h3><div class="tblwrap"><table class="t"><thead><tr><th>Ordre</th><th>Latitude</th><th>Longitude</th>`
    + `<th>Sol</th><th>Pente</th><th>Occupation du sol</th><th>Capteur le plus proche</th><th>Écart attendu aujourd'hui</th><th>Gain pour la zone</th>`
    + `<th>Fiabilité bonne après</th><th>Vent moyen à ${mat} m</th></tr></thead><tbody>${lignes}</tbody></table></div>`
    + `<p class="muted ia-note">Ordre = ordre d'installation conseillé : chaque capteur est placé là où il fait le plus baisser l'écart attendu moyen de la zone, `
    + `compte tenu des précédents. Vent moyen : moyenne des 24 heures de la journée affichée (${grille().reel() ? "service PC-FNO3D" : "journée d'exemple"}), à la hauteur du mât.</p></div>`;
}

function rendreSuite(){
  if(!res || !res.n) return "";
  return `<div class="card ia-suite"><h3>Avant d'installer</h3><ul class="ia-liste">`
    + `<li>Vérifier sur place l'accès, la propriété du terrain et l'absence d'obstacle proche (bâtiment, arbres) : la carte ne les connaît pas.</li>`
    + `<li>Mesurer le gain réel : ajouter ces emplacements au calcul OS6 (onglet Réseau ANEO), qui relance le modèle avec les capteurs simulés.</li>`
    + `<li>Une fois installés, cocher les capteurs dans l'onglet Réseau ANEO pour relancer le plan à partir du réseau réel.</li></ul></div>`;
}

// ------------------------------------------------------------------ carte
function couleurFiab(s){ return s < .9 ? [63, 159, 106] : s < 1.5 ? [224, 178, 58] : [208, 87, 63]; }
function fondImage(g){
  const cle = fond + "|" + avecPlan + "|" + g.NX + "|" + g.aneo.length;
  if(fondCache && fondCache.cle === cle) return fondCache.c;
  const c = document.createElement("canvas"); c.width = g.NX; c.height = g.NY;
  const x = c.getContext("2d"), img = x.createImageData(g.NX, g.NY), pas = 1 / g.G;
  const ex = g.stations.map(s => ({ p: s.p, f: g.sigmaStation }));
  if(avecPlan) for(const p of g.aneo) if(p) ex.push({ p, f: g.sigmaAneo });
  for(let j = 0; j < g.NY; j++) for(let i = 0; i < g.NX; i++){
    const k = j * g.NX + i, o = ((g.NY - 1 - j) * g.NX + i) * 4;
    let col;
    if(g.wat[k]) col = [178, 207, 222];
    else {
      const t = Math.min(1, (g.elev[k] || 0) / 900), relief = [222 - 50 * t, 226 - 70 * t, 205 - 80 * t];
      if(fond === "fiabilite"){
        const xx = (i + .5) * pas, yy = (j + .5) * pas; let s = 9;
        for(const e of ex){ const d = Math.hypot(xx - e.p[0], yy - e.p[1]); if(d < 60) s = Math.min(s, e.f(d)); }
        if(s === 9) s = g.sigmaStation(60);
        const f = couleurFiab(s); col = relief.map((v, n) => .55 * v + .45 * f[n]);
      } else col = relief;
    }
    img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  fondCache = { cle, c };
  return c;
}

function dessiner(){
  const g = grille(), cv = document.getElementById("ia-canvas"); if(!g || !cv) return;
  const larg = Math.min(640, cv.parentElement.clientWidth - 2 || 560), s = larg / g.W, haut = g.H * s;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(larg * dpr); cv.height = Math.round(haut * dpr); cv.style.width = `${larg}px`; cv.style.height = `${haut}px`;
  const x = cv.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.imageSmoothingEnabled = false;
  x.drawImage(fondImage(g), 0, 0, larg, haut);
  const P = (px, py) => [px * s, haut - py * s];
  if(avecPlan) for(const p of g.aneo){ if(!p) continue; const [a, b] = P(...p); x.beginPath(); x.arc(a, b, 3, 0, 7); x.fillStyle = "#7d8a91"; x.fill(); x.strokeStyle = "#fff"; x.lineWidth = 1; x.stroke(); }
  for(const st of g.stations){ const [a, b] = P(...st.p); x.beginPath(); x.moveTo(a, b - 7); x.lineTo(a + 6, b + 4); x.lineTo(a - 6, b + 4); x.closePath();
    x.fillStyle = "#3a6fd2"; x.fill(); x.strokeStyle = "#fff"; x.lineWidth = 1.2; x.stroke(); }
  if(zone){
    const [a, b] = P(...zone);
    x.beginPath(); x.arc(a, b, rayon * s, 0, 7); x.fillStyle = "rgba(232,116,42,.10)"; x.fill();
    x.setLineDash([6, 4]); x.strokeStyle = "#e8742a"; x.lineWidth = 2; x.stroke(); x.setLineDash([]);
    x.beginPath(); x.moveTo(a - 5, b); x.lineTo(a + 5, b); x.moveTo(a, b - 5); x.lineTo(a, b + 5); x.strokeStyle = "#e8742a"; x.lineWidth = 1.5; x.stroke();
  }
  if(res && res.choisis) for(const p of res.choisis){
    const [a, b] = P(p.x, p.y);
    x.beginPath(); x.arc(a, b, 9, 0, 7); x.fillStyle = "#e8742a"; x.fill(); x.strokeStyle = "#fff"; x.lineWidth = 2; x.stroke();
    x.fillStyle = "#fff"; x.font = "bold 10px system-ui, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText(String(p.ordre), a, b + .5);
  }
  cv._echelle = { s, haut };
}

// ------------------------------------------------------------------ actions
function telecharger(nom, texte, type){
  const u = URL.createObjectURL(new Blob([texte], { type }));
  const a = document.createElement("a"); a.href = u; a.download = nom; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(u), 5000);
}
function lignes(){
  return res.choisis.map(p => ({ ordre: p.ordre, lat: +p.lat.toFixed(6), lon: +p.lon.toFixed(6), altitude_sol_m: Math.round(p.sol),
    hauteur_mat_m: mat, pente_deg: +p.pente.toFixed(1), occupation: p.occ, distance_capteur_km: +p.dExist.toFixed(2),
    ecart_attendu_m_s: +p.sigmaPoint.toFixed(3), gain_zone_m_s: +p.gain.toFixed(4) }));
}
function exporterCSV(){
  const L = lignes(), cols = Object.keys(L[0]);
  const cell = v => { const t = String(v ?? ""); return (t.includes(",") || t.includes('"')) ? '"' + t.split('"').join('""') + '"' : t; };
  telecharger("installation_aneo_zone.csv", [cols.join(","), ...L.map(r => cols.map(k => cell(r[k])).join(","))].join("\n") + "\n", "text/csv");
}
function exporterGeoJSON(){
  const g = grille(), [la, lo] = g.latlon(...zone);
  const f = lignes().map(r => ({ type: "Feature", properties: { ...r, type: "emplacement ANEO propose" }, geometry: { type: "Point", coordinates: [r.lon, r.lat] } }));
  telecharger("installation_aneo_zone.geojson", JSON.stringify({ type: "FeatureCollection",
    metadata: { produit_par: "ATOM Wind, onglet Installation ANEO", methode: "placement glouton sur la fiabilité indicative (distance aux capteurs), à valider par OS6",
      centre_lat: +la.toFixed(5), centre_lon: +lo.toFixed(5), rayon_km: rayon, capteurs_prevus_pris_en_compte: avecPlan }, features: f }, null, 1), "application/geo+json");
}

function recalculer(){ message = ""; proposer(); rendre(); }

function brancher(R){
  const cv = R.querySelector("#ia-canvas");
  if(cv) cv.onclick = e => {
    const r = cv.getBoundingClientRect(), E = cv._echelle; if(!E) return;
    const g = grille(), px = (e.clientX - r.left) / E.s, py = (E.haut - (e.clientY - r.top)) / E.s;
    if(px < 0 || py < 0 || px > g.W || py > g.H) return;
    zone = [px, py]; recalculer();
  };
  R.querySelectorAll("[data-fond]").forEach(b => b.onclick = () => { fond = b.dataset.fond; rendre(); });
  const ry = R.querySelector("#ia-rayon"); if(ry) ry.onchange = () => { rayon = +ry.value; recalculer(); R.querySelector("#ia-rayon")?.focus(); };
  if(ry) ry.oninput = () => { const b = ry.previousElementSibling || ry.parentElement.querySelector("b"); if(b) b.textContent = `${ry.value} km`; };
  const nb = R.querySelector("#ia-nombre"); if(nb) nb.onchange = () => { nombre = +nb.value; recalculer(); R.querySelector("#ia-nombre")?.focus(); };
  if(nb) nb.oninput = () => { const b = nb.parentElement.querySelector("b"); if(b) b.textContent = nb.value; };
  const mt = R.querySelector("#ia-mat"); if(mt) mt.onchange = () => { mat = +mt.value; recalculer(); };
  const pl = R.querySelector("#ia-plan"); if(pl) pl.onchange = () => { avecPlan = pl.checked; recalculer(); };
  const li = R.querySelector("#ia-lieu"); if(li) li.onchange = () => { if(li.value === "") return; zone = grille().stations[+li.value].p.slice(); recalculer(); };
  const c1 = R.querySelector("#ia-csv"); if(c1) c1.onclick = exporterCSV;
  const c2 = R.querySelector("#ia-geojson"); if(c2) c2.onclick = exporterGeoJSON;
}

let redim = null;
window.addEventListener("resize", () => { clearTimeout(redim); redim = setTimeout(() => { if(document.querySelector("#v-installation.on")) dessiner(); }, 150); });
document.addEventListener("atom:vue", e => { if(e.detail && e.detail.vue === "installation") requestAnimationFrame(rendre); });

A.installationAneo = { afficher: rendre, proposer: (lat, lon, r, n) => {
  const g = grille(); if(!g) return null; zone = g.xy(lat, lon); if(r) rayon = r; if(n) nombre = n; proposer(); rendre(); return res; } };
})();

export {};
