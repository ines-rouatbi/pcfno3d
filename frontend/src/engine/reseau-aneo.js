/* ATOM Wind — onglet d'administration « Réseau ANEO » (feuille de route nationale, phase 8).
 *
 * PROVISOIRE. La maquette reseau_aneo.html de Tukhnanutha n'a pas été fournie : l'onglet suit la spécification
 * écrite d'Ines (carte des capteurs par priorité et par phase, curseur du nombre de capteurs avec erreur attendue,
 * gain et coût, stratégies optimisé / grille / aléatoire, courbe de dimensionnement avec le seuil DSO1, fiche de
 * capteur prévu / installé, tableau des phases, export CSV et GeoJSON) dans le style de l'administration
 * existante (cartes, tableaux, pastilles). Pour l'habiller d'après reseau_aneo.html quand elle arrivera :
 *   - l'habillage est ISOLÉ ici (fonctions rendre*) et dans styles/reseau-aneo.css (classes « ra- ») ;
 *   - le point de montage est <section id="v-reseau"><div id="aneo-reseau"> (views/ViewReseauAneo.tsx ;
 *     maquette historique : accroche 21 de scripts/build_atom_wind.py) ;
 *   - les données ne changent pas : GET /api/atom-wind/reseau-aneo?detail=true (portée admin) et
 *     POST /api/atom-wind/reseau-aneo/installes (écrit os6_installes.csv, renvoie la commande --installes que
 *     le service ne lance jamais).
 * Tant que les sorties OS6 manquent, le service renvoie un plan d'exemple : bandeau « Démonstration : plan
 * d'exemple », aucune erreur, aucun gain, aucun coût affichés (rien n'est inventé).
 */
(() => {
"use strict";
const A = window.ATOM = window.ATOM || {};
const STRAT = { optimise: ["Optimisé", "var(--accent)"], grille: ["Grille régulière", "#9a7b3f"], hasard: ["Aléatoire", "#7d8a91"] };
const PHASES = ["#0a6e7d", "#d98a00", "#6a4c93", "#1f9d55", "#c0392b", "#2c7fb8", "#8c564b", "#c2559a", "#5f6b73", "#9aa61f"];
const LIGNES_MAX = 40;
let V = null, enCours = null, erreur = null, choix = new Set(), sel = null, strategie = "optimise", couleur = "phase",
    iCurseur = null, tout = false, terrain = null, message = "";

const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? "—" : Number(v).toFixed(d).replace(".", ",");
const ms = v => (v === null || v === undefined || !isFinite(v)) ? "—" : `${num(v, 3)} m/s`;
const prix = v => (v === null || v === undefined) ? "—" : `${Math.round(v).toLocaleString("fr-FR")}${V && V.cout.devise ? ` ${esc(V.cout.devise)}` : ""}`;
const couleurPhase = p => PHASES[((p || 1) - 1) % PHASES.length];

// ------------------------------------------------------------------ données
async function charger(){
  const R = document.getElementById("aneo-reseau"); if(!R) return;
  if(!A.api || !(A.pret && A.pret())){
    erreur = "Service PC-FNO3D non joint : le plan OS6 et le statut des capteurs s'affichent quand le service répond.";
    V = null; rendre(); return;
  }
  R.innerHTML = `<div class="card"><p class="muted" style="margin:0">Lecture du plan de déploiement ANEO…</p></div>`;
  try {
    V = await A.api.get("/atom-wind/reseau-aneo?detail=true", 60000);
    erreur = null;
  } catch(e){
    V = null;
    erreur = e.auth ? "Onglet réservé aux administrateurs du service : connectez-vous avec un compte administrateur."
                    : `Plan de déploiement indisponible : ${e.message}`;
  }
  if(V){
    choix = new Set(V.capteurs.filter(c => c.statut === "installé").map(c => c.id));
    const et = etapes();
    const cible = V.nombre_recommande ?? (V.courbe ? et[et.length - 1] : Math.min(V.capteurs.length, 10));
    iCurseur = Math.max(0, et.findIndex(n => n >= cible));
    if(sel && !V.capteurs.some(c => c.id === sel)) sel = null;
  }
  rendre();
}
// paliers du curseur : nombres de capteurs où l'erreur a été mesurée (courbe OS6), sinon chaque capteur du plan
function etapes(){
  if(V.courbe && V.courbe.tailles.length) return [0, ...V.courbe.tailles];
  return Array.from({ length: V.capteurs.length + 1 }, (_, i) => i);
}
function mesure(strat, n){
  if(!V.courbe) return null;
  if(n === 0) return V.courbe.actuel || V.reseau_actuel;
  return (V.courbe.strategies[strat] || []).find(p => p.n === n) || null;
}

// ------------------------------------------------------------------ rendu
function rendre(){
  const R = document.getElementById("aneo-reseau"); if(!R) return;
  if(!V){
    R.innerHTML = `<div class="ra-bandeau demo"><b>Démonstration : plan d'exemple</b></div>`
      + `<div class="card"><p class="muted" style="margin:0">${esc(erreur || "")}</p></div>`;
    return;
  }
  R.innerHTML = rendreBandeau() + rendreIndicateurs()
    + `<div class="grid2 ra-haut">${rendreCarte()}${rendreFiche()}</div>`
    + rendreCurseur() + rendreCourbe() + rendrePhases() + rendreCapteurs() + rendreRelance() + rendreExports();
  brancher(R);
}

function rendreBandeau(){
  const demo = V.demonstration;
  return `<div class="ra-bandeau ${demo ? "demo" : "os6"}" role="status"><b>${esc(V.bandeau)}</b>`
    + (demo ? `<span>Emplacements d'exemple éloignés des stations, sans calcul OS6 : ni erreur, ni gain, ni coût. `
              + `Le plan réel s'affiche ici dès que scripts/os6_reseau_aneo.py a écrit ses sorties dans le dossier du modèle servi.</span>`
            : `<span>Sorties lues dans ${esc(V.dossier)}. ${esc(V.limites || "")}</span>`)
    + `</div>`;
}

function rendreIndicateurs(){
  const inst = V.capteurs.filter(c => c.statut === "installé").length;
  const rec = V.demonstration ? "—" : (V.nombre_recommande === null || V.nombre_recommande === undefined ? "non atteint" : V.nombre_recommande);
  const k = [["Capteurs prévus", V.capteurs.length, V.domaine.nom],
             ["Installés", inst, V.installes.hors_plan.length ? `${V.installes.hors_plan.length} hors du plan` : "statut du plan"],
             ["Nombre recommandé", rec, `seuil DSO1 : ${num(V.seuil_dso1_m_s, 2)} m/s au pire niveau`],
             ["Coût unitaire", V.cout.unitaire === null ? "à renseigner" : prix(V.cout.unitaire), V.cout.unitaire === null ? "os6_parametres.json" : "fourni par l'exploitant"]];
  return `<div class="kpis ra-kpis">${k.map(([t, v, s]) => `<div class="card kpi"><div class="k">${t}</div><div class="v">${v}</div><div class="s muted">${esc(s)}</div></div>`).join("")}</div>`;
}

function terrainURL(){
  const g = A.grille;
  if(terrain || !g || Math.abs(g.largeur_km - V.domaine.largeur_km) > 1e-6) return terrain;
  const c = document.createElement("canvas"); c.width = g.nx; c.height = g.ny;
  const x = c.getContext("2d"), img = x.createImageData(g.nx, g.ny);
  for(let j = 0; j < g.ny; j++) for(let i = 0; i < g.nx; i++){
    const o = ((g.ny - 1 - j) * g.nx + i) * 4, eau = g.eau[j][i] === 1, t = Math.min(1, (g.terrain_m[j][i] || 0) / 900);
    const col = eau ? [178, 207, 222] : [222 - 50 * t, 226 - 70 * t, 205 - 80 * t];
    img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  return terrain = c.toDataURL();
}

function rendreCarte(){
  const W = V.domaine.largeur_km, H = V.domaine.hauteur_km, s = Math.min(560 / W, 520 / H), pw = W * s, ph = H * s;
  const P = (x, y) => [x * s, (H - y) * s], nMax = etapes()[iCurseur] ?? 0, n = V.capteurs.length;
  const img = terrainURL();
  const pts = V.capteurs.filter(c => c.x_km !== null).map(c => {
    const [px, py] = P(c.x_km, c.y_km), dedans = c.ordre <= nMax, inst = choix.has(c.id);
    const fill = couleur === "phase" ? couleurPhase(c.phase) : "hsl(190,65%," + Math.round(22 + 50 * (c.ordre - 1) / Math.max(1, n - 1)) + "%)";
    const r = c.id === sel ? 7 : 4.5;
    return `<g class="ra-capteur" data-capteur="${esc(c.id)}" role="button" tabindex="0" aria-label="${esc(c.id)}, ordre ${c.ordre}, phase ${c.phase ?? "—"}, ${c.statut}">`
      + `<title>${esc(c.id)} · ordre ${c.ordre} · phase ${c.phase ?? "—"} · ${c.statut}</title>`
      + `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${r}" fill="${fill}" fill-opacity="${dedans ? 1 : .25}" stroke="${inst ? "var(--go)" : "#fff"}" stroke-width="${inst ? 2.6 : 1}"/></g>`;
  }).join("");
  const st = V.stations.map(t => { const [px, py] = P(t.x_km, t.y_km);
    return `<path d="M${px.toFixed(1)},${(py - 6).toFixed(1)}l5,9h-10z" fill="#3a6fd2" stroke="#fff" stroke-width="1"><title>Station ${esc(t.nom)}</title></path>`; }).join("");
  const leg = couleur === "phase"
    ? V.phases.map(p => `<span class="ra-leg"><i style="background:${couleurPhase(p.phase)}"></i>Phase ${p.phase} (${p.de}–${p.a})</span>`).join("")
    : `<span class="ra-leg"><i style="background:hsl(190,65%,22%)"></i>premiers capteurs</span><span class="ra-leg"><i style="background:hsl(190,65%,72%)"></i>derniers capteurs</span>`;
  return `<div class="card ra-carte"><div class="row ra-titre"><h3>Carte des capteurs</h3>`
    + `<div class="seg" role="group" aria-label="Couleur des capteurs"><button data-couleur="phase" aria-pressed="${couleur === "phase"}">Par phase</button>`
    + `<button data-couleur="priorite" aria-pressed="${couleur === "priorite"}">Par priorité</button></div></div>`
    + `<svg class="ra-svg" viewBox="0 0 ${pw.toFixed(1)} ${ph.toFixed(1)}" role="img" aria-label="Plan de déploiement ANEO sur ${esc(V.domaine.nom)}">`
    + (img ? `<image href="${img}" x="0" y="0" width="${pw.toFixed(1)}" height="${ph.toFixed(1)}" preserveAspectRatio="none" style="image-rendering:pixelated"/>`
           : `<rect width="${pw.toFixed(1)}" height="${ph.toFixed(1)}" fill="var(--map-land, #e8ece4)"/>`)
    + `<rect width="${pw.toFixed(1)}" height="${ph.toFixed(1)}" fill="none" stroke="var(--line)"/>${st}${pts}</svg>`
    + `<div class="ra-legende">${leg}<span class="ra-leg"><i class="ra-anneau"></i>installé</span><span class="ra-leg"><i class="ra-tri"></i>station existante</span></div>`
    + `<p class="muted ra-note">Points pâles : au-delà du nombre de capteurs choisi avec le curseur. Les dispositions « grille » et « aléatoire » `
    + `ne sont pas enregistrées par OS6 : la carte montre le plan optimisé.</p></div>`;
}

function rendreFiche(){
  const c = V.capteurs.find(x => x.id === sel);
  if(!c) return `<div class="card ra-fiche"><h3>Fiche du capteur</h3><p class="muted">Choisissez un capteur sur la carte ou dans le tableau.</p></div>`;
  const inst = choix.has(c.id);
  const lignes = [["Ordre d'installation", c.ordre], ["Phase", c.phase ?? "—"], ["Étape du placement", c.etape ?? "—"],
    ["Latitude", c.lat === null ? "—" : `${num(c.lat, 5)}° N`], ["Longitude", c.lon === null ? "—" : `${num(c.lon, 5)}° E`],
    ["Altitude du sol", c.altitude_sol_m === null ? "—" : `${num(c.altitude_sol_m, 0)} m`],
    ["Hauteur du mât", c.hauteur_mat_m === null ? "—" : `${num(c.hauteur_mat_m, 1)} m`],
    ["Gain de son étape (validation)", ms(c.gain_m_s)], ["Gain cumulé (validation)", ms(c.gain_cumule_m_s)],
    ["Erreur après son étape (validation)", ms(c.rmse_validation_apres_m_s)],
    ["Installé le", c.installe_le || "—"]];
  return `<div class="card ra-fiche"><h3>${esc(c.id)}</h3>`
    + `<p><span class="pill ${c.statut === "installé" ? "good" : "m"}">${c.statut}</span>${inst !== (c.statut === "installé") ? ` <span class="pill mid">modifié, à enregistrer</span>` : ""}</p>`
    + `<dl class="ra-dl">${lignes.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`
    + (V.demonstration ? `<p class="muted ra-note">Plan d'exemple : statut non modifiable.</p>`
       : `<label class="ra-coche"><input type="checkbox" data-installe="${esc(c.id)}" ${inst ? "checked" : ""}> Capteur installé sur le terrain</label>`)
    + `</div>`;
}

function rendreCurseur(){
  const et = etapes(), n = et[iCurseur] ?? 0, m = mesure(strategie, n), a = mesure(strategie, 0);
  const gain = (m && a && n > 0) ? a.pire_rmse_niveau_m_s - m.pire_rmse_niveau_m_s : null;
  const cout = V.cout.unitaire === null ? null : V.cout.unitaire * n;
  const ok = m ? m.pire_rmse_niveau_m_s <= V.seuil_dso1_m_s : null;
  return `<div class="card ra-curseur"><div class="row ra-titre"><h3>Nombre de capteurs</h3>`
    + `<div class="seg" role="group" aria-label="Stratégie">${Object.entries(STRAT).map(([k, [t]]) =>
        `<button data-strat="${k}" aria-pressed="${strategie === k}" ${V.courbe ? "" : "disabled"}>${t}</button>`).join("")}</div></div>`
    + `<input type="range" id="ra-n" min="0" max="${et.length - 1}" step="1" value="${iCurseur}" aria-label="Nombre de capteurs ANEO">`
    + `<div class="ra-lecture"><div><span>Capteurs</span><b>${n}</b></div>`
    + `<div><span>Erreur attendue (pire niveau)</span><b>${m ? ms(m.pire_rmse_niveau_m_s) : "—"}</b></div>`
    + `<div><span>Erreur vectorielle</span><b>${m ? ms(m.rmse_vect_m_s) : "—"}</b></div>`
    + `<div><span>Gain sur le réseau actuel</span><b>${gain === null ? "—" : ms(gain)}</b></div>`
    + `<div><span>Coût</span><b>${cout === null ? "à renseigner" : prix(cout)}</b></div>`
    + `<div><span>Seuil DSO1</span><b>${ok === null ? "—" : `<span class="pill ${ok ? "good" : "bad"}">${ok ? "respecté" : "non respecté"}</span>`}</b></div></div>`
    + `<p class="muted ra-note">${V.courbe ? `Erreur mesurée sur le test (${esc(V.courbe.jeu)}), aux seuls paliers de la courbe : rien n'est interpolé. `
        + `Stratégie « ${STRAT[strategie][0]} »${strategie === "hasard" ? " : moyenne de trois tirages" : ""}.`
        : "Résultats OS6 absents : aucune erreur ni aucun gain affichés."}`
    + `${V.cout.unitaire === null ? ` Coût : ${esc(V.cout.note)}.` : ""}</p></div>`;
}

function rendreCourbe(){
  if(!V.courbe) return `<div class="card"><h3>Courbe de dimensionnement</h3><p class="muted">Pas encore calculée : elle vient du test, une seule fois, par scripts/os6_reseau_aneo.py.</p></div>`;
  const W = 640, H = 260, pl = 56, pr = 14, pt = 14, pb = 34;
  const toutes = Object.values(V.courbe.strategies).flat(), a = V.courbe.actuel;
  const vals = [...toutes.map(p => p.pire_rmse_niveau_m_s), V.seuil_dso1_m_s, ...(a ? [a.pire_rmse_niveau_m_s] : [])];
  const lo = Math.min(...vals) * .97, hi = Math.max(...vals) * 1.03, nMax = Math.max(1, ...V.courbe.tailles);
  const X = n => pl + n / nMax * (W - pl - pr), Y = v => pt + (hi - v) / Math.max(hi - lo, 1e-9) * (H - pt - pb);
  const traits = Object.entries(STRAT).map(([k, [nom, col]]) => {
    const pts = [...(a ? [{ n: 0, pire_rmse_niveau_m_s: a.pire_rmse_niveau_m_s }] : []), ...(V.courbe.strategies[k] || [])];
    if(!pts.length) return "";
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.n).toFixed(1)},${Y(p.pire_rmse_niveau_m_s).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${k === strategie ? 3 : 1.6}" ${k === "hasard" ? 'stroke-dasharray="2 4"' : k === "grille" ? 'stroke-dasharray="6 4"' : ""}/>`
      + pts.map(p => `<circle cx="${X(p.n).toFixed(1)}" cy="${Y(p.pire_rmse_niveau_m_s).toFixed(1)}" r="${k === strategie ? 3.4 : 2.4}" fill="${col}"><title>${nom} · ${p.n} capteurs · ${ms(p.pire_rmse_niveau_m_s)}</title></circle>`).join("");
  }).join("");
  const n = etapes()[iCurseur] ?? 0, gradY = [lo, (lo + hi) / 2, hi];
  return `<div class="card ra-courbe"><h3>Courbe de dimensionnement</h3>`
    + `<div class="tblwrap"><svg viewBox="0 0 ${W} ${H}" class="ra-svg-courbe" role="img" aria-label="Erreur au pire niveau selon le nombre de capteurs">`
    + gradY.map(v => `<line x1="${pl}" x2="${W - pr}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="var(--line)"/><text x="${pl - 6}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end" font-size="11">${num(v, 2)}</text>`).join("")
    + [0, ...V.courbe.tailles].map(t => `<text x="${X(t).toFixed(1)}" y="${H - 14}" text-anchor="middle" font-size="11">${t}</text>`).join("")
    + `<text x="${(pl + W - pr) / 2}" y="${H - 1}" text-anchor="middle" font-size="11">capteurs ANEO ajoutés</text>`
    + `<line x1="${pl}" x2="${W - pr}" y1="${Y(V.seuil_dso1_m_s).toFixed(1)}" y2="${Y(V.seuil_dso1_m_s).toFixed(1)}" stroke="var(--stop)" stroke-dasharray="5 4"/>`
    + `<text x="${W - pr}" y="${(Y(V.seuil_dso1_m_s) - 4).toFixed(1)}" text-anchor="end" font-size="11" fill="var(--stop)">seuil DSO1 ${num(V.seuil_dso1_m_s, 2)} m/s</text>`
    + (a ? `<line x1="${pl}" x2="${W - pr}" y1="${Y(a.pire_rmse_niveau_m_s).toFixed(1)}" y2="${Y(a.pire_rmse_niveau_m_s).toFixed(1)}" stroke="var(--muted)" stroke-dasharray="1 3"/>`
         + `<text x="${pl + 4}" y="${(Y(a.pire_rmse_niveau_m_s) - 4).toFixed(1)}" font-size="11">réseau actuel</text>` : "")
    + `<line x1="${X(n).toFixed(1)}" x2="${X(n).toFixed(1)}" y1="${pt}" y2="${H - pb}" stroke="var(--accent)" stroke-opacity=".35" stroke-width="6"/>`
    + traits + `</svg></div>`
    + `<div class="ra-legende">${Object.entries(STRAT).map(([, [nom, col]]) => `<span class="ra-leg"><i style="background:${col}"></i>${nom}</span>`).join("")}</div>`
    + `<p class="muted ra-note">Pire RMSE de niveau (u ou v) sur le test, selon le nombre de capteurs ajoutés au réseau actuel.`
    + `${V.nombre_recommande === null || V.nombre_recommande === undefined ? " Le seuil DSO1 n'est atteint par aucun palier de la courbe." : ` Nombre recommandé : ${V.nombre_recommande} capteurs (plus petit palier optimisé sous le seuil).`}</p></div>`;
}

function rendrePhases(){
  const lignes = V.phases.map(p => `<tr><td><span class="ra-pastille" style="background:${couleurPhase(p.phase)}"></span> Phase ${p.phase}</td>`
    + `<td>${p.de} à ${p.a}</td><td>${p.capteurs}</td><td>${p.cumul}</td><td>${ms(p.pire_rmse_niveau_m_s)}</td><td>${ms(p.gain_pire_niveau_m_s)}</td>`
    + `<td>${p.seuil_dso1_respecte === null ? "—" : `<span class="pill ${p.seuil_dso1_respecte ? "good" : "bad"}">${p.seuil_dso1_respecte ? "oui" : "non"}</span>`}</td>`
    + `<td>${p.installes} / ${p.capteurs}</td><td>${p.cout === null ? "—" : prix(p.cout)}</td><td>${p.cout_cumule === null ? "—" : prix(p.cout_cumule)}</td></tr>`).join("");
  return `<div class="card"><h3>Phases de déploiement</h3><div class="tblwrap"><table class="t"><thead><tr><th>Phase</th><th>Capteurs (ordre)</th>`
    + `<th>Nombre</th><th>Cumul</th><th>Erreur attendue</th><th>Gain</th><th>DSO1</th><th>Installés</th><th>Coût</th><th>Coût cumulé</th></tr></thead>`
    + `<tbody>${lignes || `<tr><td colspan="10" class="muted">Aucun capteur dans le plan.</td></tr>`}</tbody></table></div>`
    + `<p class="muted ra-note">Une phase = un palier de la courbe de dimensionnement : la phase k pose les capteurs suivants jusqu'au palier, `
    + `son erreur attendue est celle de la courbe à ce palier, son gain l'écart au palier précédent (au réseau actuel pour la première).</p></div>`;
}

function rendreCapteurs(){
  const liste = tout ? V.capteurs : V.capteurs.slice(0, LIGNES_MAX);
  const lignes = liste.map(c => `<tr class="${c.id === sel ? "ra-choisi" : ""}"><td>${c.ordre}</td><td><button class="ra-lien" data-voir="${esc(c.id)}">${esc(c.id)}</button></td>`
    + `<td><span class="ra-pastille" style="background:${couleurPhase(c.phase)}"></span> ${c.phase ?? "—"}</td>`
    + `<td>${c.lat === null ? "—" : num(c.lat, 4)}</td><td>${c.lon === null ? "—" : num(c.lon, 4)}</td>`
    + `<td>${c.altitude_sol_m === null ? "—" : `${num(c.altitude_sol_m, 0)} m`}</td><td>${c.hauteur_mat_m === null ? "—" : `${num(c.hauteur_mat_m, 1)} m`}</td>`
    + `<td>${ms(c.gain_m_s)}</td><td>${ms(c.gain_cumule_m_s)}</td>`
    + `<td>${V.demonstration ? `<span class="pill m">${c.statut}</span>` : `<label class="ra-coche"><input type="checkbox" data-installe="${esc(c.id)}" ${choix.has(c.id) ? "checked" : ""} aria-label="${esc(c.id)} installé"> installé</label>`}</td></tr>`).join("");
  const modifie = !V.demonstration && V.capteurs.some(c => choix.has(c.id) !== (c.statut === "installé"));
  return `<div class="card"><div class="row ra-titre"><h3>Plan ordonné</h3>`
    + (V.demonstration ? "" : `<button class="btn primary" id="ra-enregistrer" ${modifie ? "" : "disabled"}>Enregistrer le statut installé</button>`) + `</div>`
    + `<div class="tblwrap"><table class="t"><thead><tr><th>Ordre</th><th>Capteur</th><th>Phase</th><th>Latitude</th><th>Longitude</th><th>Sol</th><th>Mât</th>`
    + `<th>Gain</th><th>Gain cumulé</th><th>Statut</th></tr></thead><tbody>${lignes}</tbody></table></div>`
    + (V.capteurs.length > LIGNES_MAX ? `<button class="btn" id="ra-tout">${tout ? "Afficher les premiers capteurs" : `Afficher les ${V.capteurs.length} capteurs`}</button>` : "")
    + (message ? `<p class="ra-message" role="status">${esc(message)}</p>` : "")
    + `<p class="muted ra-note">Gains mesurés sur la validation pendant le placement glouton (OS6), en m/s d'erreur vectorielle.</p></div>`;
}

function rendreRelance(){
  const r = V.relance;
  return `<div class="card ra-relance"><h3>${r.avec_installes ? "Relancer le plan à partir des capteurs installés" : "Calculer le plan OS6"}</h3>`
    + `<p class="muted ra-note">${esc(r.note)}</p>`
    + `<pre class="ra-commande" id="ra-commande">${esc(r.commande)}</pre>`
    + `<button class="btn" id="ra-copier">Copier la commande</button>`
    + (V.installes.fichier ? ` <span class="muted ra-note">Capteurs installés : ${esc(V.installes.fichier)}</span>` : "") + `</div>`;
}

function rendreExports(){
  const bruts = (V.fichiers || []).filter(f => /\.(csv|geojson|json|png)$/.test(f));
  return `<div class="card"><h3>Exports</h3><div class="row ra-exports">`
    + `<button class="btn" id="ra-csv">Plan en CSV</button><button class="btn" id="ra-geojson">Plan en GeoJSON</button>`
    + bruts.map(f => `<button class="btn" data-brut="${esc(f)}">${esc(f)}</button>`).join("") + `</div>`
    + `<p class="muted ra-note">CSV et GeoJSON : le plan affiché, avec sa phase et son statut. Les autres boutons rendent les sorties OS6 telles qu'écrites.</p></div>`;
}

// ------------------------------------------------------------------ actions
function telecharger(nom, texte, type){
  const u = URL.createObjectURL(new Blob([texte], { type }));
  const a = document.createElement("a"); a.href = u; a.download = nom; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(u), 5000);
}
function lignesExport(){
  return V.capteurs.map(c => ({ id: c.id, ordre: c.ordre, phase: c.phase, statut: choix.has(c.id) ? "installé" : "prévu",
    lat: c.lat, lon: c.lon, altitude_sol_m: c.altitude_sol_m, hauteur_mat_m: c.hauteur_mat_m,
    gain_m_s: c.gain_m_s, gain_cumule_m_s: c.gain_cumule_m_s, installe_le: c.installe_le }));
}
function exporterCSV(){
  const L = lignesExport(), cols = Object.keys(L[0] || { id: "" });
  // (sans expression reguliere : le lecteur de textes de scripts/extraire_textes_atom_wind.py s'y perdrait)
  const cell = v => {
    if(v === null || v === undefined) return "";
    const s = String(v);
    return (s.includes(",") || s.includes('"') || s.includes("\n")) ? '"' + s.split('"').join('""') + '"' : s;
  };
  telecharger(V.demonstration ? "plan_aneo_exemple.csv" : "plan_aneo_statut.csv",
              [cols.join(","), ...L.map(r => cols.map(k => cell(r[k])).join(","))].join("\n") + "\n", "text/csv");
}
function exporterGeoJSON(){
  const f = lignesExport().filter(r => r.lat !== null).map(r => ({ type: "Feature", properties: { ...r, type: "capteur ANEO propose" },
    geometry: { type: "Point", coordinates: [r.lon, r.lat] } }));
  const g = { type: "FeatureCollection", metadata: { produit_par: "ATOM Wind, onglet Réseau ANEO", demonstration: V.demonstration,
    domaine: V.domaine.nom, seuil_dso1_m_s: V.seuil_dso1_m_s, nombre_recommande: V.nombre_recommande }, features: f };
  telecharger(V.demonstration ? "plan_aneo_exemple.geojson" : "plan_aneo_statut.geojson", JSON.stringify(g, null, 1), "application/geo+json");
}
async function enregistrer(){
  const r = await A.api.post("/atom-wind/reseau-aneo/installes", { installes: [...choix] });
  if(r.erreur){ message = `Enregistrement refusé : ${r.erreur}`; rendre(); return; }
  message = `${r.installes} capteur(s) installé(s) enregistré(s) dans ${r.fichier}. Lancez la commande ci-dessous dans le carnet OS6 pour relancer le plan.`;
  await charger();
}

function brancher(R){
  R.querySelectorAll("[data-couleur]").forEach(b => b.onclick = () => { couleur = b.dataset.couleur; rendre(); });
  R.querySelectorAll("[data-strat]").forEach(b => b.onclick = () => { strategie = b.dataset.strat; rendre(); });
  const voir = id => { sel = id; rendre(); R.querySelector(".ra-fiche")?.scrollIntoView({ block: "nearest" }); };
  R.querySelectorAll("[data-capteur]").forEach(g => {
    g.onclick = () => voir(g.dataset.capteur);
    g.onkeydown = e => { if(e.key === "Enter" || e.key === " "){ e.preventDefault(); voir(g.dataset.capteur); } };
  });
  R.querySelectorAll("[data-voir]").forEach(b => b.onclick = () => voir(b.dataset.voir));
  R.querySelectorAll("[data-installe]").forEach(c => c.onchange = () => {
    if(c.checked) choix.add(c.dataset.installe); else choix.delete(c.dataset.installe);
    message = ""; rendre();
  });
  const n = R.querySelector("#ra-n");
  if(n) n.oninput = () => { iCurseur = +n.value; rendre(); R.querySelector("#ra-n")?.focus(); };
  const t = R.querySelector("#ra-tout"); if(t) t.onclick = () => { tout = !tout; rendre(); };
  const e = R.querySelector("#ra-enregistrer"); if(e) e.onclick = () => { e.disabled = true; enregistrer(); };
  const cp = R.querySelector("#ra-copier");
  if(cp) cp.onclick = async () => { try { await navigator.clipboard.writeText(V.relance.commande); cp.textContent = "Commande copiée"; } catch(err){ cp.textContent = "Copie impossible : sélectionnez le texte"; } };
  R.querySelector("#ra-csv").onclick = exporterCSV;
  R.querySelector("#ra-geojson").onclick = exporterGeoJSON;
  R.querySelectorAll("[data-brut]").forEach(b => b.onclick = async () => {
    try { const u = await A.api.blob("/atom-wind/reseau-aneo/fichier/" + encodeURIComponent(b.dataset.brut));
          const a = document.createElement("a"); a.href = u; a.download = b.dataset.brut; document.body.appendChild(a); a.click(); a.remove(); }
    catch(err){ message = `Export impossible : ${err.message}`; rendre(); }
  });
}

// point d'entrée : atom-runtime.js l'appelle à l'ouverture de la vue « reseau » (administration seulement)
A.reseauAneo = {
  afficher(){ if(V || erreur){ rendre(); return; } if(!enCours) enCours = charger().finally(() => { enCours = null; }); },
  recharger: () => charger(),
};
})();

export {};
