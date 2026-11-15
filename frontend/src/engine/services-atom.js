/* ATOM Wind — fonctions du service PC-FNO3D qu'aucun écran n'utilisait encore.
 *
 * Branche dans les vues existantes, à leur ouverture (événement « atom:vue » émis par atom-runtime.js) :
 *   - Éolien (#svc-eolien) : production du parc virtuel avec et sans sillage de Jensen et ses variations
 *     brusques (GET /api/atom-wind/variations), créneaux de travail en hauteur (/maintenance), vent et rafale
 *     pour une grue (/grue), dimensionnement d'un mât de mesure (/dimensionnement), emplacements des mâts
 *     Porteur (/porteurs) ;
 *   - Alertes (#svc-alertes) : alertes calculées par le service (/atom-wind/alertes) ;
 *   - Administration (#svc-admin) : abonnés SMS (/sms/abonnes, /sms/envoyer), notifications
 *     (/notifications/envoyer), exports d'un instant (/instants, /exports/{stem}.nc|csv), import d'un fichier
 *     de mesures avec le verdict du contrat de données (/reconstruct/upload) ;
 *   - Agriculteur (#svc-agri) : haies brise-vent (/atom-wind/haies) ;
 *   - Fiabilité (#svc-fiabilite) : scénario de panne (/atom-wind/scenario), journées archivées
 *     (/atom-wind/historique ; « Ouvrir » passe par ATOM.choisirJour d'atom-data.js).
 * Sans service (démonstration), chaque carte le dit et n'affiche aucun chiffre : rien n'est inventé.
 * Les appels passent par ATOM.api (atom-data.js : même adresse, même clé ou session) ; seuls l'envoi
 * multipart et la suppression d'un abonné font leur propre fetch, avec les mêmes en-têtes.
 */
(() => {
"use strict";
const A = window.ATOM = window.ATOM || {};
const BASE = (new URLSearchParams(location.search).get("api") || "").replace(/\/+$/, "");
const TTL = 3 * 60 * 1000;                      // cache court des réponses (par chemin)
const ROLE_EOLIEN = "Réservé aux rôles exploitant éolien et administrateur : connectez-vous avec un tel compte.";
const ROLE_ADMIN = "Réservé aux administrateurs du service : connectez-vous avec un compte administrateur.";
const ROLE_LECTURE = "Accès refusé par le service : connectez-vous ou enregistrez une clé d'API.";
const JOURS = [[0, "Aujourd'hui"], [1, "Demain"], [2, "Après-demain"]];
// lieux de la page (atom-runtime.js, PLACES) quand le domaine servi est le Grand Tunis
const LIEUX_GT = [["Tunis-Carthage (aéroport)", 36.8504, 10.2156], ["Tunis centre", 36.80, 10.18], ["La Marsa", 36.878, 10.325],
  ["Ben Arous", 36.75, 10.23], ["Hammam-Lif", 36.728, 10.34], ["Mornag (vignes)", 36.68, 10.29], ["Djebel Ressas", 36.61, 10.33],
  ["Soliman", 36.69, 10.49], ["Grombalia (agrumes)", 36.60, 10.50], ["Korbous", 36.82, 10.57], ["Nabeul", 36.455, 10.715],
  ["Zaghouan (station)", 36.4333, 10.0833], ["Mohamedia", 36.68, 10.16]];
const STATIONS_GT = ["Tunis-Carthage", "Zaghouan", "Nabeul"];

const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ok = v => v !== null && v !== undefined && isFinite(v);
const num = (v, d = 1) => ok(v) ? Number(v).toFixed(d).replace(".", ",") : "—";
const entier = v => ok(v) ? Math.round(v).toLocaleString("fr-FR") : "—";
const kmh = v => ok(v) ? `${Math.round(v * 3.6)} km/h` : "—";
const date = d => { if(!d) return "—"; const p = String(d).slice(0, 10).split("-"); return p.length === 3 ? `${p[2]}/${p[1]}/${p[0]}` : esc(d); };

// ------------------------------------------------------------------ accès au service
const enLigne = () => !!(A.api && A.pret && A.pret());
const cache = new Map();
function lire(chemin, ms, frais = false){
  const c = cache.get(chemin);
  if(!frais && c && Date.now() - c.t < TTL) return c.p;
  const p = A.api.get(chemin, ms);
  cache.set(chemin, { t: Date.now(), p });
  p.catch(() => cache.delete(chemin));
  return p;
}
// mêmes en-têtes qu'atom-data.js (fonction entetes) : la session d'un compte prime sur la clé d'API
function entetes(){
  let s = null, k = null;
  try { s = localStorage.getItem("aw-session-cle"); k = localStorage.getItem("pcfno3d.apikey"); } catch(e) {}
  return s ? { "X-API-Key": s, "X-ATOM-Session": s } : k ? { "X-API-Key": k } : {};
}
async function requete(methode, chemin, corps){
  const r = await fetch(`${BASE}/api${chemin}`, { method: methode, headers: entetes(), body: corps });
  let d = {}; try { d = await r.json(); } catch(e) {}
  return { statut: r.status, ok: r.ok, d };
}
const detail = d => typeof d.detail === "string" ? d.detail : d.message || (d.detail ? JSON.stringify(d.detail) : "");
// message d'une erreur d'ATOM.api.get : 401/403 -> le rôle, 503 -> le message du service
function texteErreur(e, role){
  if(e && e.auth) return role;
  let m = (e && e.message) || String(e);
  if(m.startsWith("/") && m.includes(" : ")) m = m.slice(m.indexOf(" : ") + 3);
  return `Réponse du service : ${m}`;
}

// ------------------------------------------------------------------ habillage commun
function style(){
  if(document.getElementById("sv-style")) return;
  const s = document.createElement("style"); s.id = "sv-style";
  s.textContent = ".sv-pile{display:grid;gap:16px}.sv-pile:empty{display:none}"
    + ".sv-form{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:flex-end;margin:6px 0 10px}"
    + ".sv-form label{display:grid;gap:3px;font-size:13px;font-weight:700}.sv-form .inp{min-width:0}"
    + ".sv-form input[type=number]{width:7em}.sv-note{font-size:13px;margin:6px 0 0}"
    + ".sv-bande{display:grid;grid-template-columns:repeat(24,minmax(22px,1fr));gap:2px;margin:8px 0;min-width:560px}"
    + ".sv-bande>span{font-size:11px;text-align:center;padding:4px 0;border-radius:3px;line-height:1.35;font-variant-numeric:tabular-nums}"
    + ".sv-bande .good{background:var(--go-soft);color:var(--go)}.sv-bande .mid{background:var(--warn-soft);color:var(--warn)}"
    + ".sv-bande .bad{background:var(--stop-soft);color:var(--stop)}.sv-svg{width:100%;height:auto;display:block;max-height:340px}"
    + ".sv-pills{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}.sv-res .kpis{margin:8px 0}"
    + ".sv-res .kpi .v{font-size:24px}.sv-liste{margin:6px 0 0;padding-left:20px;font-size:14px}";
  document.head.appendChild(s);
}
const carte = (id, titre, intro, formulaire = "") =>
  `<div class="card" id="${id}"><h3>${titre}</h3>${intro ? `<p class="muted sv-note" style="margin:0 0 8px">${intro}</p>` : ""}`
  + (enLigne() ? formulaire : "") + `<div class="sv-res" id="${id}-res" aria-live="polite"></div></div>`;
const absent = () => `<p class="muted" style="margin:6px 0 0"><span class="pill m">service absent</span> Disponible quand le service PC-FNO3D répond.</p>`;
const enCours = t => `<p class="muted" style="margin:6px 0 0">${esc(t || "Calcul par le service PC-FNO3D…")}</p>`;
const echec = t => `<p style="margin:6px 0 0"><span class="pill bad">indisponible</span> ${esc(t)}</p>`;
const note = t => t ? `<p class="muted sv-note">${esc(t)}</p>` : "";
const kpis = L => `<div class="kpis">${L.map(([k, v, s]) => `<div class="card kpi"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s muted">${s}</div>` : ""}</div>`).join("")}</div>`;
const tableau = (tetes, lignes) => `<div class="tblwrap"><table class="t"><thead><tr>${tetes.map(h => `<th>${h}</th>`).join("")}</tr></thead>`
  + `<tbody>${lignes.map(l => `<tr>${l.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const bande = (heures, f) => `<div class="tblwrap"><div class="sv-bande">${heures.map(r => { const [cls, txt, titre] = f(r);
  return `<span class="${cls}" title="${esc(titre)}">${r.heure} h<br>${txt}</span>`; }).join("")}</div></div>`;
const pastilles = (L, cls = "good") => L.length ? `<div class="sv-pills">${L.map(t => `<span class="pill ${cls}">${esc(t)}</span>`).join("")}</div>` : "";
const source = d => d.source === "prevision" ? "prévision" : d.source === "temps reel" ? "temps réel" : esc(d.source || "");
const poser = (id, html) => { const el = document.getElementById(id + "-res"); if(el) el.innerHTML = html; };
const val = (id, d) => { const el = document.getElementById(id); return el && el.value !== "" ? el.value : d; };
const champ = (id, libelle, v, min, max, pas = 1) =>
  `<label for="${id}">${libelle}<input class="inp" type="number" id="${id}" value="${v}" min="${min}" max="${max}" step="${pas}"></label>`;
const bouton = (id, texte) => `<button class="btn" id="${id}">${texte}</button>`;

// une carte chargée par chemin : ignore une réponse arrivée après une demande plus récente
const derniere = {};
async function charger(id, chemin, rendu, { role = ROLE_LECTURE, ms = 120000, frais = false, attente } = {}){
  if(!enLigne()){ poser(id, absent()); return; }
  derniere[id] = chemin;
  poser(id, enCours(attente));
  try {
    const d = await lire(chemin, ms, frais);
    if(derniere[id] !== chemin) return;
    poser(id, rendu(d));
  } catch(e){
    if(derniere[id] === chemin) poser(id, echec(texteErreur(e, role)));
  }
}

// lieux proposés : stations du domaine servi (national, aéroports), sinon les lieux de la page
function lieux(){
  const d = A.domaine && A.domaine();
  return d && d.stations.length ? d.stations.map(s => [s.nom, s.lat, s.lon]) : LIEUX_GT;
}
function choixLieu(id, libelle, defaut, premier){
  const L = lieux(), i0 = Math.max(0, L.findIndex(l => l[0] === defaut));
  return `<label for="${id}">${libelle}<select class="inp" id="${id}">`
    + (premier ? `<option value="">${premier}</option>` : "")
    + L.map((l, i) => `<option value="${i}"${!premier && i === i0 ? " selected" : ""}>${esc(l[0])}</option>`).join("") + `</select></label>`;
}
function lieuChoisi(id){
  const v = val(id, ""); if(v === "") return null;
  const l = lieux()[+v]; return l ? { nom: l[0], lat: l[1], lon: l[2] } : null;
}
const qLieu = l => l ? `&lat=${l.lat}&lon=${l.lon}` : "";
function heureTunis(){
  // en fr-FR l'heure s'écrit « 14 h » : parseInt, pas Number
  const h = parseInt(new Intl.DateTimeFormat("fr-FR", { hour: "numeric", hour12: false, timeZone: "Africa/Tunis" }).format(new Date()), 10);
  return isFinite(h) ? h % 24 : new Date().getUTCHours();
}
// monte le squelette d'une vue une fois, à nouveau si le service est apparu ou a disparu entretemps
const montees = {};
function monter(mount, html){
  const M = document.getElementById(mount); if(!M) return false;
  const etat = enLigne() ? "en ligne" : "absent";
  if(montees[mount] === etat && M.firstChild) return false;
  style(); M.innerHTML = html; montees[mount] = etat;
  return true;
}
function telecharger(url, nom){
  const a = document.createElement("a"); a.href = url; a.download = nom; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ================================================================== Éolien
let jourEol = 0;
function vueEolien(){
  const neuf = monter("svc-eolien",
    `<div class="card"><h3>Outils du service pour le parc et les travaux</h3>`
    + `<p class="muted sv-note" style="margin:0 0 8px">Calculés par le service PC-FNO3D sur le vent ATOM Wind de la journée choisie (aujourd'hui en temps réel, demain et après-demain en prévision).</p>`
    + (enLigne() ? `<div><b style="font-size:14px">Jour</b><div class="chips" id="sv-eo-jour" style="margin-top:4px">`
      + JOURS.map(([j, t]) => `<button class="chip" data-j="${j}" aria-pressed="${j === jourEol}">${t}</button>`).join("") + `</div></div>` : absent()) + `</div>`
    + carte("sv-sillage", "Production du parc avec et sans sillage",
        "Parc virtuel de trois éoliennes : la production heure par heure avec les pertes de sillage (modèle de Jensen) et sans elles, et les hausses ou chutes rapides utiles au gestionnaire du réseau.")
    + carte("sv-maint", "Créneaux sûrs pour le travail en hauteur",
        "Heures où le vent moyen à la nacelle ou au mât reste sous le seuil du site, en plages d'au moins la durée demandée.",
        `<div class="sv-form">${choixLieu("sv-maint-lieu", "Lieu", "", "Parc virtuel (Djebel Ressas)")}`
        + champ("sv-maint-h", "Hauteur (m)", 80, 2, 1000) + champ("sv-maint-seuil", "Seuil de vent (m/s)", 12, 1, 40, .5)
        + champ("sv-maint-duree", "Durée minimale (h)", 2, 1, 12) + bouton("sv-maint-go", "Calculer") + `</div>`)
    + carte("sv-grue", "Vent pour une grue",
        "Vent moyen et rafale à la hauteur de la flèche, heure par heure, contre la limite de service de la grue ; effort du vent sur la charge suspendue.",
        `<div class="sv-form">${choixLieu("sv-grue-lieu", "Chantier", "Tunis centre")}`
        + champ("sv-grue-h", "Hauteur de flèche (m)", 40, 5, 300) + champ("sv-grue-lim", "Limite en rafale (m/s)", 20, 1, 60, .5)
        + champ("sv-grue-surf", "Surface de la charge (m²)", 0, 0, 500, .5) + bouton("sv-grue-go", "Calculer") + `</div>`)
    + carte("sv-dim", "Dimensionner l'énergie d'un mât de mesure",
        "Production d'une micro-éolienne à la hauteur du mât, bilan avec la consommation de la station et taille de batterie, d'aujourd'hui à après-demain.",
        `<div class="sv-form">${choixLieu("sv-dim-lieu", "Lieu du mât", "Djebel Ressas")}`
        + champ("sv-dim-h", "Hauteur du mât (m)", 30, 2, 150) + champ("sv-dim-conso", "Consommation (W)", 5, .5, 2000, .5)
        + champ("sv-dim-auto", "Autonomie (jours)", 3, 0, 30, .5) + bouton("sv-dim-go", "Calculer") + `</div>`)
    + carte("sv-port", "Où installer les mâts Porteur ?",
        "Emplacements classés par la production de leur petite éolienne à la hauteur du mât (terre ferme, sites espacés), et capteurs du plan ANEO classés de la même façon.",
        `<div class="sv-form"><label for="sv-port-h">Hauteur du mât<select class="inp" id="sv-port-h">`
        + [15, 20, 25, 30].map(h => `<option value="${h}"${h === 20 ? " selected" : ""}>${h} m</option>`).join("") + `</select></label>`
        + champ("sv-port-n", "Nombre de sites", 10, 1, 50) + champ("sv-port-w", "Éolienne (W)", 400, 50, 5000, 50)
        + champ("sv-port-conso", "Consommation (W)", 5, .5, 2000, .5) + champ("sv-port-ecart", "Écart minimal (km)", 2, 0, 20, .5)
        + bouton("sv-port-go", "Calculer") + `</div>`));
  if(neuf && enLigne()){
    document.querySelectorAll("#sv-eo-jour [data-j]").forEach(b => b.onclick = () => {
      jourEol = +b.dataset.j;
      document.querySelectorAll("#sv-eo-jour [data-j]").forEach(x => x.setAttribute("aria-pressed", String(+x.dataset.j === jourEol)));
      sillage(); maintenance(); grue();
    });
    [["sv-maint-go", maintenance], ["sv-grue-go", grue], ["sv-dim-go", dimensionnement], ["sv-port-go", porteurs]]
      .forEach(([id, f]) => { document.getElementById(id).onclick = () => f(); });
  }
  sillage(); maintenance(); grue(); dimensionnement(); porteurs();
}

function courbes(series, ymax, titre){
  const W = 600, H = 190, g = 34, d = 8, b = 22, h = H - d - b, w = W - g - d;
  const X = i => g + w * i / 23, Y = v => d + h * (1 - Math.min(1, Math.max(0, v / (ymax || 1))));
  const axes = [0, .5, 1].map(f => `<line x1="${g}" x2="${W - d}" y1="${Y(ymax * f)}" y2="${Y(ymax * f)}" stroke="var(--line)"/>`
      + `<text x="${g - 4}" y="${Y(ymax * f) + 4}" font-size="10" text-anchor="end" fill="var(--muted)">${entier(ymax * f / 1000)}</text>`).join("")
    + [0, 6, 12, 18, 23].map(i => `<text x="${X(i)}" y="${H - 6}" font-size="10" text-anchor="middle" fill="var(--muted)">${i} h</text>`).join("");
  const lignes = series.map(([v, coul, tir]) => `<polyline fill="none" stroke="${coul}" stroke-width="2"${tir ? ` stroke-dasharray="5 4"` : ""} points="${v.map((y, i) => `${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}"/>`).join("");
  return `<svg class="sv-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(titre)}">${axes}${lignes}</svg>`;
}
function sillage(){
  charger("sv-sillage", `/atom-wind/variations?j=${jourEol}`, d => {
    const sans = d.puissance_sans_sillage_kw || [], avec = d.puissance_kw || [];
    const eSans = sans.reduce((s, x) => s + x, 0) / 1000;
    const R = d.rampes || [];
    return `<p style="margin:0 0 4px"><b>${esc(d.parc)}</b> · ${date(d.date)} · moyeu ${entier(d.moyeu_m)} m · ${entier(d.puissance_installee_kw / 1000)} MW installés</p>`
      + kpis([["Production du jour, avec sillage", `${num(d.energie_jour_mwh, 2)} MWh`, ""],
              ["Sans sillage", `${num(eSans, 2)} MWh`, ""],
              ["Perte due aux sillages", `${num(d.perte_sillage_pct, 1)} %`, esc(d.sillage)],
              ["Variations brusques", String(R.length), `au moins ${num(d.seuil_pct_par_heure, 0)} % de la puissance installée en une heure`]])
      + courbes([[sans, "var(--muted)", true], [avec, "var(--accent)", false]], d.puissance_installee_kw, "Puissance du parc heure par heure, avec et sans sillage")
      + `<div class="legend"><span><b style="color:var(--accent)">—</b> avec sillage</span><span><b style="color:var(--muted)">- -</b> sans sillage</span><span>puissance en MW</span></div>`
      + (R.length ? tableau(["Heures", "Sens", "Variation", "Niveau"], R.map(r => [`${r.de} h → ${r.a} h`,
            r.sens === "hausse" ? "hausse" : "chute", `${r.variation_kw > 0 ? "+" : ""}${entier(r.variation_kw)} kW (${num(r.variation_pct, 1)} %)`,
            `<span class="pill ${r.niveau === "important" ? "bad" : "mid"}">${r.niveau === "important" ? "important" : "attention"}</span>`]))
        : `<p class="sv-note"><span class="pill good">stable</span> Aucune variation brusque ce jour au pas horaire.</p>`)
      + note(d.note);
  }, { role: ROLE_EOLIEN });
}
function maintenance(){
  const l = lieuChoisi("sv-maint-lieu");
  const q = `/atom-wind/maintenance?j=${jourEol}&hauteur_m=${val("sv-maint-h", 80)}&seuil_m_s=${val("sv-maint-seuil", 12)}&duree_min_h=${val("sv-maint-duree", 2)}${qLieu(l)}`;
  charger("sv-maint", q, d => `<p style="margin:0 0 4px"><b>${esc(l ? l.nom : "Parc virtuel (Djebel Ressas)")}</b> · ${date(d.date)} (${source(d)}) · ${entier(d.hauteur_sol_m)} m · seuil ${num(d.seuil_m_s, 1)} m/s</p>`
    + (d.creneaux_surs.length ? `<p style="margin:6px 0 0">Créneaux sûrs :</p>${pastilles(d.creneaux_surs)}`
       : `<p style="margin:6px 0 0"><span class="pill bad">aucun créneau</span> Pas de plage d'au moins ${entier(d.duree_min_h)} h sous le seuil ce jour.</p>`)
    + bande(d.heures, r => [r.sur ? "good" : "bad", num(r.vent_m_s, 1), `${r.heure} h : ${num(r.vent_m_s, 1)} m/s, marge ${num(r.marge_m_s, 1)} m/s`])
    + `<p class="muted sv-note">Vent moyen en m/s à la hauteur demandée ; vert : sous le seuil.</p>` + note(d.note),
  { role: ROLE_EOLIEN });
}
function grue(){
  const l = lieuChoisi("sv-grue-lieu") || (() => { const L = lieux(), i = Math.max(0, L.findIndex(x => x[0] === "Tunis centre")); return { nom: L[i][0], lat: L[i][1], lon: L[i][2] }; })();
  const surf = +val("sv-grue-surf", 0);
  const q = `/atom-wind/grue?j=${jourEol}&hauteur_m=${val("sv-grue-h", 40)}&limite_rafale_m_s=${val("sv-grue-lim", 20)}&surface_charge_m2=${surf}${qLieu(l)}`;
  const ST = { ok: ["good", "levage possible"], attention: ["mid", "prudence"], arret: ["bad", "arrêt"] };
  charger("sv-grue", q, d => `<p style="margin:0 0 4px"><b>${esc(l.nom)}</b> · ${date(d.date)} (${source(d)}) · flèche à ${entier(d.hauteur_fleche_m)} m · limite ${entier(d.limite_rafale_km_h)} km/h en rafale</p>`
    + kpis([["Rafale maximale", kmh(d.rafale_max_m_s), `${num(d.rafale_max_m_s, 1)} m/s`],
            ["Créneaux de levage", String(d.creneaux_levage.length), "rafale sous 80 % de la limite"]])
    + pastilles(d.creneaux_levage)
    + bande(d.heures, r => [(ST[r.statut] || ST.ok)[0], String(r.rafale_km_h), `${r.heure} h : vent ${num(r.vent_m_s, 1)} m/s, rafale ${r.rafale_km_h} km/h, ${(ST[r.statut] || ST.ok)[1]}`])
    + `<p class="muted sv-note">Rafale en km/h à la hauteur de la flèche ; vert : levage possible, orange : prudence, rouge : arrêt. Rafale : ${esc(d.rafale_source)}.</p>`
    + (surf > 0 ? `<p class="sv-note">Effort du vent sur la charge (${num(d.surface_charge_m2, 1)} m²) à la rafale la plus forte : ${entier(Math.max(...d.heures.map(r => r.effort_charge_kgf)))} kgf.</p>` : "")
    + note(d.note));
}
function dimensionnement(){
  const l = lieuChoisi("sv-dim-lieu"); if(enLigne() && !l) return;
  const q = `/atom-wind/dimensionnement?hauteur_m=${val("sv-dim-h", 30)}&conso_w=${val("sv-dim-conso", 5)}&autonomie_j=${val("sv-dim-auto", 3)}${qLieu(l)}`;
  charger("sv-dim", q, d => `<p style="margin:0 0 4px"><b>${esc(l.nom)}</b> · mât de ${entier(d.hauteur_mat_m)} m · ${d.dates.map(date).join(", ")} (${d.heures} h)</p>`
    + kpis([["Vent moyen au mât", `${num(d.vent_moyen_m_s, 1)} m/s`, kmh(d.vent_moyen_m_s)],
            ["Production", `${entier(d.production_wh_par_jour)} Wh/j`, esc(d.eolienne)],
            ["Consommation", `${entier(d.consommation_wh_par_jour)} Wh/j`, ""],
            ["Couverture", ok(d.couverture_pct) ? `${entier(d.couverture_pct)} %` : "—", "production / consommation"],
            ["Batterie conseillée", `${entier(d.batterie_wh)} Wh`, `${entier(d.batterie_ah_12v)} Ah sous 12 V`],
            ["Plus grand déficit", `${entier(d.plus_grand_deficit_wh)} Wh`, "sur la période"]])
    + `<p class="muted sv-note">Hypothèses : ${num(d.hypotheses.conso_w, 1)} W en continu, ${num(d.hypotheses.autonomie_j, 1)} jour(s) d'autonomie, décharge à ${entier(d.hypotheses.profondeur_decharge * 100)} %, rendement ${entier(d.hypotheses.rendement * 100)} %.</p>`
    + note(d.limite));
}
function porteurs(){
  const q = `/atom-wind/porteurs?hauteur_m=${val("sv-port-h", 20)}&n=${val("sv-port-n", 10)}&eolienne_w=${val("sv-port-w", 400)}&conso_w=${val("sv-port-conso", 5)}&ecart_km=${val("sv-port-ecart", 2)}`;
  const ligne = s => [`${num(s.lat, 4)} N, ${num(s.lon, 4)} E`, `${entier(s.altitude_m)} m`, `${num(s.vent_moyen_m_s, 2)} m/s`,
    `${entier(s.production_wh_par_jour)} Wh/j`, ok(s.couverture_conso_pct) ? `${entier(s.couverture_conso_pct)} %` : "—"];
  charger("sv-port", q, d => {
    const avecPlan = d.sites.some(s => ok(s.capteur_aneo_prevu_km));
    return `<p style="margin:0 0 4px">Mât de ${entier(d.hauteur_mat_m)} m · ${esc(d.eolienne)} · ${d.dates.length} journée(s), ${d.heures} h de vent</p>`
      + (d.sites.length ? tableau(["Rang", "Position", "Altitude", "Vent moyen", "Production", "Couverture"].concat(avecPlan ? ["Capteur ANEO prévu"] : []),
          d.sites.map(s => [String(s.rang), ...ligne(s)].concat(avecPlan ? [ok(s.capteur_aneo_prevu_km) ? `à ${num(s.capteur_aneo_prevu_km, 1)} km` : "—"] : [])))
        : `<p class="sv-note">Aucun emplacement ne respecte les règles (terre ferme, altitude au plus ${entier(d.regles.altitude_max_m)} m).</p>`)
      + (d.plan_aneo ? `<h4 style="margin:12px 0 4px">Capteurs du plan ANEO, classés de la même façon</h4>`
          + tableau(["Ordre OS6", "Position", "Altitude", "Vent moyen", "Production", "Couverture"], d.plan_aneo.map(s => [ok(s.ordre_os6) ? String(s.ordre_os6) : "—", ...ligne(s)])) : "")
      + `<p class="muted sv-note">Règles : terre ferme, altitude au plus ${entier(d.regles.altitude_max_m)} m, ${num(d.regles.marge_bord_km, 1)} km du bord, sites espacés d'au moins ${num(d.regles.ecart_min_km, 1)} km ; couverture d'une consommation de ${num(d.regles.conso_w, 1)} W.</p>`
      + note(d.limite);
  }, { attente: "Classement des emplacements par le service (journées disponibles et historique)…" });
}

// ================================================================== Alertes
const GENRE = { station_muette: "Station muette", physique: "Résidu de masse", vent_fort: "Vent fort" };
function vueAlertes(){
  monter("svc-alertes", carte("sv-alertes", "Alertes du service en temps réel",
    "Calculées par le service PC-FNO3D sur le vent du jour : station muette, résidu de masse au-dessus du seuil, vent fort aujourd'hui et demain. Ce sont celles que le service notifie (une seule fois chacune)."));
  charger("sv-alertes", "/atom-wind/alertes", d => {
    const L = d.alertes || [];
    if(!L.length) return `<p style="margin:6px 0 0"><span class="pill good">rien à signaler</span> Aucune alerte du service en ce moment.</p>`;
    return tableau(["Niveau", "Alerte", "Pourquoi", "Que faire"], L.map(a => {
      const sv = a.station_virtuelle;
      const virt = sv && ok(sv.vitesse_m_s) ? `<br><span class="muted">Valeur de remplacement à ${entier(sv.heure_locale)} h (station virtuelle) : ${kmh(sv.vitesse_m_s)}`
        + (ok(sv.erreur_type_m_s) ? ` (erreur type sans elle : ± ${kmh(sv.erreur_type_m_s)}).` : ".") + `</span>` : "";
      return [`<span class="pill ${a.niveau === "important" ? "bad" : "m"}">${a.niveau === "important" ? "important" : "information"}</span>`,
              `<b>${esc(GENRE[a.kind] || a.kind)}</b><br>${esc(a.titre)}`, esc(a.cause) + virt, esc(a.recommandation)];
    }));
  }, { ms: 60000 });
}

// ================================================================== Administration
function vueAdmin(){
  const neuf = monter("svc-admin",
    carte("sv-sms", "Abonnés aux alertes SMS",
      "Le service envoie chaque jour le conseil de traitement et les alertes de vent du lieu de chaque abonné. Les numéros sont masqués à l'affichage.",
      `<div class="sv-form"><label for="sv-sms-num">Numéro (format international)<input class="inp" id="sv-sms-num" placeholder="+21612345678" autocomplete="off"></label>`
      + `<label for="sv-sms-langue">Langue<select class="inp" id="sv-sms-langue"><option value="fr">français</option><option value="ar">arabe</option></select></label>`
      + choixLieu("sv-sms-lieu", "Lieu", "Mornag (vignes)") + bouton("sv-sms-ajout", "Abonner") + bouton("sv-sms-retrait", "Désabonner ce numéro") + `</div>`
      + `<div class="sv-form"><label for="sv-sms-jour">SMS du jour<select class="inp" id="sv-sms-jour">${JOURS.map(([j, t]) => `<option value="${j}"${j === 1 ? " selected" : ""}>${t}</option>`).join("")}</select></label>`
      + `<label for="sv-sms-simu" style="display:flex;gap:6px;align-items:center"><input type="checkbox" id="sv-sms-simu" checked> Simulation (journal seulement)</label>`
      + bouton("sv-sms-envoi", "Envoyer à tous les abonnés") + `</div><div id="sv-sms-msg" role="status"></div>`)
    + carte("sv-notif", "Notifications des alertes",
      "Envoie maintenant les alertes importantes du service qui n'ont pas encore été notifiées (journal, e-mail ou WhatsApp selon la configuration du service).",
      `<div class="sv-form">${bouton("sv-notif-go", "Notifier maintenant")}</div>`)
    + carte("sv-exp", "Exports d'un instant",
      "Le champ 3D d'un instant du cache du modèle, en NetCDF (CF) ou en CSV des nœuds, avec la version du modèle et l'horodatage source.",
      `<div class="sv-form"><label for="sv-exp-stem">Instant<select class="inp" id="sv-exp-stem"></select></label>`
      + bouton("sv-exp-nc", "NetCDF (.nc)") + bouton("sv-exp-csv", "CSV des nœuds") + `</div>`)
    + carte("sv-imp", "Importer un fichier de mesures",
      "Un CSV de mesures d'un seul instant passe d'abord par le contrat de données ANEO : conforme ou en quarantaine, le modèle reconstruit le vent ; rejeté, le rapport dit pourquoi.",
      `<div class="sv-form"><label for="sv-imp-f">Fichier CSV (ou .csv.gz)<input class="inp" type="file" id="sv-imp-f" accept=".csv,.gz,text/csv"></label>`
      + bouton("sv-imp-go", "Contrôler et reconstruire") + `</div>`));
  if(!enLigne()){ ["sv-sms", "sv-notif", "sv-exp", "sv-imp"].forEach(id => poser(id, absent())); return; }
  if(neuf){
    document.getElementById("sv-sms-ajout").onclick = abonner;
    document.getElementById("sv-sms-retrait").onclick = desabonner;
    document.getElementById("sv-sms-envoi").onclick = envoyerSms;
    document.getElementById("sv-notif-go").onclick = notifier;
    document.getElementById("sv-exp-nc").onclick = () => exporter("nc");
    document.getElementById("sv-exp-csv").onclick = () => exporter("csv");
    document.getElementById("sv-imp-go").onclick = importer;
  }
  abonnes();
  instants();
}
const msgSms = html => { const el = document.getElementById("sv-sms-msg"); if(el) el.innerHTML = html; };
function abonnes(frais = false){
  charger("sv-sms", "/sms/abonnes", d => (d.abonnes || []).length
    ? tableau(["Numéro", "Langue", "Lieu", "Depuis", ""], d.abonnes.map(a => [`<span style="font-family:var(--f-data)">${esc(a.numero)}</span>`,
        a.langue === "ar" ? "arabe" : "français", esc(a.lieu || (ok(a.lat) ? `${num(a.lat, 4)} N, ${num(a.lon, 4)} E` : "—")), date(a.depuis),
        a.id ? `<button class="btn" data-sms-retirer="${esc(a.id)}" aria-label="Désabonner ${esc(a.numero)}">Désabonner</button>` : ""]))
    : `<p class="muted" style="margin:6px 0 0">Aucun abonné pour l'instant.</p>`, { role: ROLE_ADMIN, ms: 20000, frais });
}
async function abonner(){
  const numero = val("sv-sms-num", "").trim(), l = lieuChoisi("sv-sms-lieu");
  if(!numero){ msgSms(echec("Saisissez un numéro au format international, ex. +21612345678.")); return; }
  const r = await A.api.post("/sms/abonnes", { numero, langue: val("sv-sms-langue", "fr"), lieu: l ? l.nom : null, lat: l ? l.lat : null, lon: l ? l.lon : null });
  if(r.erreur){ msgSms(echec(`Abonnement refusé : ${r.erreur}`)); return; }
  msgSms(`<p class="sv-note"><span class="pill good">abonné</span> ${esc(r.numero)} recevra le SMS du jour pour ${esc(r.lieu || "son lieu")}.</p>`);
  document.getElementById("sv-sms-num").value = "";
  abonnes(true);
}
// désabonnement depuis la liste : le service désigne chaque abonné par un identifiant opaque (id), sans renvoyer
// le numéro complet au navigateur
document.addEventListener("click", e => {
  const b = e.target && e.target.closest ? e.target.closest("[data-sms-retirer]") : null;
  if(b){ b.disabled = true; desabonner(b.dataset.smsRetirer); }
});
async function desabonner(ident){
  const numero = typeof ident === "string" ? ident : val("sv-sms-num", "").trim();
  if(!numero){ msgSms(echec("Saisissez le numéro complet à désabonner, ou utilisez le bouton « Désabonner » de la liste.")); return; }
  try {
    const r = await requete("DELETE", "/sms/abonnes/" + encodeURIComponent(numero));
    if(r.statut === 401 || r.statut === 403){ msgSms(echec(ROLE_ADMIN)); return; }
    if(!r.ok){ msgSms(echec(r.statut === 404 ? "Ce numéro n'est pas abonné." : `Réponse du service : ${r.statut} ${detail(r.d)}`)); return; }
    msgSms(`<p class="sv-note"><span class="pill m">désabonné</span> ${esc(r.d.retire)}</p>`);
    document.getElementById("sv-sms-num").value = "";
    abonnes(true);
  } catch(e){ msgSms(echec(`Service injoignable : ${e.message}`)); }
}
async function envoyerSms(){
  const simulation = document.getElementById("sv-sms-simu").checked, jour = val("sv-sms-jour", 1);
  if(!simulation && !confirm("Envoyer de vrais SMS à tous les abonnés par le fournisseur configuré ?")) return;
  msgSms(enCours("Composition des SMS par le service…"));
  const r = await A.api.post(`/sms/envoyer?jour=${jour}&simulation=${simulation}`, {});
  if(r.erreur){ msgSms(echec(`Envoi impossible : ${r.erreur}`)); return; }
  msgSms(`<p class="sv-note">Fournisseur : <b>${esc(r.fournisseur)}</b> · ${r.abonnes} abonné(s)${simulation ? " · simulation : rien n'est parti" : ""}.</p>`
    + ((r.resultats || []).length ? tableau(["Numéro", "Message", "Résultat"], r.resultats.map(x => [esc(x.a), esc(x.texte || ""),
        x.erreur ? `<span class="pill bad">erreur</span> ${esc(x.erreur)}` : x.envoye ? `<span class="pill good">envoyé</span>` : `<span class="pill m">journal</span>`])) : ""));
}
async function notifier(){
  poser("sv-notif", enCours("Envoi des notifications par le service…"));
  const r = await A.api.post("/notifications/envoyer", {});
  if(r.erreur){ poser("sv-notif", echec(`Notification impossible : ${r.erreur}`)); return; }
  poser("sv-notif", `<p class="sv-note">Canaux : ${esc((r.canaux || []).join(", ") || "aucun")} · ${(r.envoyees || []).length} alerte(s) notifiée(s) · ${r.deja} déjà envoyée(s).</p>`
    + ((r.envoyees || []).length ? `<ul class="sv-liste">${r.envoyees.map(a => `<li>${esc(a.titre)} (${esc(a.niveau)})</li>`).join("")}</ul>` : "")
    + ((r.erreurs || []).length ? tableau(["Canal", "Alerte", "Erreur"], r.erreurs.map(x => [esc(x.canal), esc(x.id), esc(x.erreur)])) : ""));
}
function instants(){
  const sel = document.getElementById("sv-exp-stem");
  charger("sv-exp", "/instants?limit=1000", d => {
    const L = d.instants || [];
    if(sel) sel.innerHTML = L.map(x => `<option value="${esc(x.stem)}">${esc(x.timestamp)} (${esc(x.split)}) · ${esc(x.stem)}</option>`).join("");
    return `<p class="muted sv-note">${L.length} instant(s) proposés sur ${d.total} dans le cache.</p>`;
  }, { ms: 30000 });
}
async function exporter(fmt){
  const stem = val("sv-exp-stem", "");
  if(!stem){ poser("sv-exp", echec("Aucun instant à exporter.")); return; }
  poser("sv-exp", enCours("Préparation de l'export par le service…"));
  try {
    telecharger(await A.api.blob(`/exports/${encodeURIComponent(stem)}.${fmt}`), fmt === "nc" ? `${stem}.nc` : `${stem}_noeuds.csv`);
    poser("sv-exp", `<p class="sv-note"><span class="pill good">prêt</span> Export de ${esc(stem)} téléchargé.</p>`);
  } catch(e){ poser("sv-exp", echec(texteErreur(e, ROLE_LECTURE))); }
}
const VERDICT = { conforme: ["good", "conforme"], quarantaine: ["mid", "quarantaine"], rejet: ["bad", "rejet"] };
function rapportContrat(q){
  if(!q) return "";
  const r = q.resume || {}, E = q.erreurs || [];
  return `<p class="sv-note">Contrat ${esc(q.contrat || "")} · ${esc(q.fichier || "")} · ${entier(r.lignes)} ligne(s) dont ${entier(r.lignes_stations)} de stations`
    + `${ok(r.instants) ? `, ${r.instants} instant(s)` : ""}${r.premier_instant ? ` (${esc(r.premier_instant)})` : ""}`
    + `${ok(r.lignes_en_erreur_bloquante) ? ` · ${r.lignes_en_erreur_bloquante} en erreur bloquante` : ""}${ok(r.anomalies_de_mesure) ? ` · ${r.anomalies_de_mesure} anomalie(s) de mesure` : ""}.</p>`
    + ((q.motifs || []).length ? `<p style="margin:6px 0 0"><b>Motifs</b></p><ul class="sv-liste">${q.motifs.map(m => `<li>${esc(m)}</li>`).join("")}</ul>` : "")
    + (E.length ? tableau(["Règle", "Gravité", "Message", "Colonne", "Lignes"], E.map(e => [esc(e.regle), esc(e.gravite), esc(e.message), esc(e.colonne || "—"), entier(e.nombre)])) : "");
}
async function importer(){
  const f = document.getElementById("sv-imp-f").files[0];
  if(!f){ poser("sv-imp", echec("Choisissez d'abord un fichier CSV de mesures.")); return; }
  const corps = new FormData(); corps.append("file", f, f.name);
  poser("sv-imp", enCours("Contrôle du fichier et reconstruction par le modèle (jusqu'à quelques minutes)…"));
  let r;
  try { r = await requete("POST", "/reconstruct/upload", corps); }
  catch(e){ poser("sv-imp", echec(`Service injoignable : ${e.message}`)); return; }
  const d = r.d;
  if(r.statut === 401 || r.statut === 403){ poser("sv-imp", echec(ROLE_LECTURE)); return; }
  if(r.statut === 422 && d.qualite){
    poser("sv-imp", `<p style="margin:6px 0 0"><span class="pill bad">rejet</span> Fichier refusé par le contrat de données : aucune reconstruction.</p>` + rapportContrat(d.qualite));
    return;
  }
  if(!r.ok){ poser("sv-imp", echec(`Réponse du service : ${r.statut} ${detail(d)}`)); return; }
  const [cls, lib] = VERDICT[d.statut_qualite] || ["m", d.statut_qualite];
  poser("sv-imp", `<p style="margin:6px 0 0"><span class="pill ${cls}">${esc(lib)}</span> ${d.statut_qualite === "quarantaine"
      ? "Fichier accepté en quarantaine : les stations signalées ont été masquées avant le calcul."
      : "Fichier conforme au contrat de données."}</p>`
    + kpis([["Instant", esc(d.timestamp_utc || "—"), "UTC"], ["Capteurs utilisés", entier(d.capteurs_utilises), ""],
            ["Temps de reconstruction", `${num(d.inference_s, 2)} s`, ""], ["Vent maximal", `${num(d.vitesse_max_m_s, 1)} m/s`, kmh(d.vitesse_max_m_s)]])
    + ((d.avertissements || []).length ? `<p style="margin:6px 0 0"><b>Avertissements</b></p><ul class="sv-liste">${d.avertissements.map(m => `<li>${esc(m)}</li>`).join("")}</ul>` : "")
    + (d.stem ? `<p class="muted sv-note">Volume gardé dans le cache du service sous ${esc(d.stem)}.</p>` : "")
    + rapportContrat({ ...d.qualite, motifs: [] }));
}

// ================================================================== Agriculteur : haies brise-vent
function vueAgri(){
  const neuf = monter("svc-agri", carte("sv-haies", "Planter une haie brise-vent",
    "D'où viennent les vents qui dessèchent et couchent les cultures, comment orienter la haie, et quelle surface elle abrite.",
    `<div class="sv-form">${choixLieu("sv-haies-lieu", "Lieu", lieuAgri())}`
    + champ("sv-haies-h", "Hauteur de la haie (m)", 3, .5, 30, .5) + champ("sv-haies-l", "Longueur (m)", 200, 10, 5000, 10)
    + champ("sv-haies-seuil", "Vents comptés au-dessus de (m/s)", 5, 0, 30, .5) + bouton("sv-haies-go", "Calculer") + `</div>`));
  if(neuf && enLigne()) document.getElementById("sv-haies-go").onclick = haies;
  haies();
}
// lieu choisi dans le formulaire de la parcelle (nom français), sinon Mornag
function lieuAgri(){
  const s = document.getElementById("ag-place"), o = s && s.selectedOptions && s.selectedOptions[0];
  return (o && (A.texteFr ? A.texteFr(o) : o.textContent)) || "Mornag (vignes)";
}
let geoHaie = null;
function planHaie(d){
  const F = d.geojson.features, la0 = d.lat, lo0 = d.lon, kx = 111320 * Math.cos(la0 * Math.PI / 180), ky = 110540;
  const P = c => [(c[0] - lo0) * kx, (c[1] - la0) * ky];
  const pts = F.flatMap(f => f.geometry.type === "Polygon" ? f.geometry.coordinates[0] : f.geometry.coordinates).map(P);
  const r = Math.max(...pts.map(p => Math.hypot(p[0], p[1]))) * 1.25 || 1;
  const W = 320, s = W / (2 * r), X = p => (W / 2 + p[0] * s).toFixed(1), Y = p => (W / 2 - p[1] * s).toFixed(1);
  const COUL = { abri_fort: "var(--go)", abri: "var(--go-soft)", au_vent: "var(--info-soft)" };
  const zones = F.filter(f => f.geometry.type === "Polygon").map(f => `<polygon points="${f.geometry.coordinates[0].map(P).map(p => `${X(p)},${Y(p)}`).join(" ")}" fill="${COUL[f.properties.type] || "var(--surface-2)"}" fill-opacity="${f.properties.type === "abri_fort" ? .45 : .8}" stroke="var(--line)"/>`).join("");
  const haie = F.find(f => f.properties.type === "haie"), hp = haie ? haie.geometry.coordinates.map(P) : [];
  const t = d.direction_dominante_deg * Math.PI / 180, u = [Math.sin(t), Math.cos(t)];       // d'où vient le vent
  const a0 = [u[0] * r * .92, u[1] * r * .92], a1 = [u[0] * r * .45, u[1] * r * .45];
  return `<svg class="sv-svg" style="max-width:340px" viewBox="0 0 ${W} ${W}" role="img" aria-label="Haie conseillée et zone abritée, vue de dessus, nord en haut">`
    + `<defs><marker id="sv-fl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--accent)"/></marker></defs>`
    + `<rect width="${W}" height="${W}" fill="none" stroke="var(--line)"/>${zones}`
    + (hp.length ? `<line x1="${X(hp[0])}" y1="${Y(hp[0])}" x2="${X(hp[1])}" y2="${Y(hp[1])}" stroke="#2f6b2f" stroke-width="5" stroke-linecap="round"/>` : "")
    + `<line x1="${X(a0)}" y1="${Y(a0)}" x2="${X(a1)}" y2="${Y(a1)}" stroke="var(--accent)" stroke-width="3" marker-end="url(#sv-fl)"/>`
    + `<text x="${W - 14}" y="20" font-size="13" font-weight="700" text-anchor="middle" fill="var(--muted)">N</text>`
    + `<line x1="${W - 14}" y1="24" x2="${W - 14}" y2="40" stroke="var(--muted)" stroke-width="2"/></svg>`;
}
function haies(){
  const l = lieuChoisi("sv-haies-lieu"); if(enLigne() && !l) return;
  const q = `/atom-wind/haies?hauteur_haie_m=${val("sv-haies-h", 3)}&longueur_m=${val("sv-haies-l", 200)}&seuil_m_s=${val("sv-haies-seuil", 5)}${qLieu(l)}`;
  charger("sv-haies", q, d => {
    geoHaie = d.geojson;
    const rose = (d.rose || []).slice().sort((a, b) => b.part_vents_forts_pct - a.part_vents_forts_pct).slice(0, 5);
    const z = d.zone_protegee_m;
    setTimeout(() => { const b = document.getElementById("sv-haies-geo"); if(b) b.onclick = () => telecharger(URL.createObjectURL(new Blob([JSON.stringify(geoHaie, null, 1)], { type: "application/geo+json" })), "haie_brise_vent.geojson"); });
    return `<p style="margin:0 0 4px"><b>${esc(l.nom)}</b> · ${d.dates.length} journée(s), ${d.heures} heure(s) de vent</p>`
      + kpis([["Vents à craindre", `${esc(d.secteur_dominant)}`, `${entier(d.direction_dominante_deg)}° ; constance ${num(d.constance, 2)}`],
              ["Orientation de la haie", `${entier(d.azimut_haie_deg)}°`, "azimut de la ligne, perpendiculaire au vent"],
              ["Abri fort sous le vent", `${entier(z.abri_fort)} m`, "vent réduit d'environ 50 %"],
              ["Abri sous le vent", `${entier(z.abri)} m`, "vent réduit d'environ 20 à 50 %"],
              ["Côté au vent", `${entier(z.au_vent)} m`, "faible réduction"]])
      + `<div class="grid2" style="align-items:center">${planHaie(d)}<div>`
      + `<div class="legend"><span><b style="color:var(--accent)">→</b> vent dominant</span><span><b style="color:#2f6b2f">━</b> haie</span><span>vert : zone abritée</span></div>`
      + tableau(["Secteur", "Fréquence", "Part des vents forts"], rose.map(r => [esc(r.secteur), `${num(r.frequence_pct, 1)} %`, `${num(r.part_vents_forts_pct, 1)} %`]))
      + `<button class="btn" id="sv-haies-geo" style="margin-top:8px">Télécharger le plan en GeoJSON</button></div></div>`
      + note(d.critere) + note(d.limite);
  }, { attente: "Calcul du vent dominant par le service (journées disponibles et historique)…" });
}

// ================================================================== Fiabilité
function stationsDomaine(){
  const g = A.grille, L = g && (g.stations || []).map(s => s.nom).filter(Boolean);
  return L && L.length ? [...new Set(L)] : STATIONS_GT;
}
function vueFiabilite(){
  const h0 = Math.max(0, heureTunis() - 1);
  const neuf = monter("svc-fiabilite",
    carte("sv-panne", "Et si une station tombait en panne ?",
      "Une heure déjà mesurée aujourd'hui, recalculée par le modèle sans les stations choisies : ce que la panne change dans le champ, et ce que le reste du réseau devine sans elles.",
      `<div class="sv-form">${champ("sv-panne-h", "Heure (Tunis)", h0, 0, 23)}`
      + `<fieldset style="border:0;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:4px 12px"><legend style="font-size:13px;font-weight:700">Stations retirées</legend>`
      + stationsDomaine().map((n, i) => `<label style="display:flex;gap:6px;align-items:center;font-weight:400"><input type="checkbox" data-sv-station="${esc(n)}"${i === 0 ? " checked" : ""}> ${esc(n)}</label>`).join("")
      + `</fieldset>${bouton("sv-panne-go", "Recalculer sans ces stations")}</div>`)
    + carte("sv-hist", "Journées archivées par le service",
      "Chaque journée en temps réel et chaque prévision déjà calculées sont gardées par le service ; une prévision se compare ensuite au temps réel du même jour."));
  if(!enLigne()){ poser("sv-panne", absent()); poser("sv-hist", absent()); return; }
  if(neuf){
    document.getElementById("sv-panne-go").onclick = panne;
    poser("sv-panne", `<p class="muted sv-note">Choisissez une heure et les stations à retirer, puis lancez le calcul.</p>`);
  }
  historique();
}
function panne(){
  const st = [...document.querySelectorAll("[data-sv-station]")].filter(c => c.checked).map(c => c.dataset.svStation);
  if(!st.length){ poser("sv-panne", echec("Cochez au moins une station à retirer.")); return; }
  const q = `/atom-wind/scenario?heure=${val("sv-panne-h", 0)}&retirer=${encodeURIComponent(st.join(","))}`;
  charger("sv-panne", q, d => {
    const D = d.domaine || {}, S = Object.entries(d.stations || {});
    return `<p style="margin:0 0 4px">${date(d.date)}, ${entier(d.heure_locale)} h · retirées : <b>${esc(d.retirees.join(", "))}</b> · mesurées à cette heure : ${esc(d.mesurees.join(", ") || "aucune")}</p>`
      + tableau(["Hauteur", "Écart moyen", "Écart maximal", "Part du domaine au-delà de 0,5 m/s"],
          Object.entries(D).map(([h, x]) => [esc(h.replace("m", " m")), `${num(x.ecart_moyen_m_s, 2)} m/s`, `${num(x.ecart_max_m_s, 2)} m/s`, `${num(x.part_au_dessus_0_5_m_s * 100, 1)} %`]))
      + tableau(["Station", "État", "Écart du champ", "Erreur avec toutes", "Erreur sans les retirées"],
          S.map(([n, x]) => [esc(n), x.retiree ? `<span class="pill bad">retirée</span>` : x.mesuree ? `<span class="pill good">mesurée</span>` : `<span class="pill m">sans mesure</span>`,
            `${num(x.ecart_m_s, 2)} m/s`, ok(x.erreur_avec_m_s) ? `${num(x.erreur_avec_m_s, 2)} m/s` : "—", ok(x.erreur_sans_m_s) ? `${num(x.erreur_sans_m_s, 2)} m/s` : "—"]))
      + `<p class="muted sv-note">Écart : ce que la panne change dans le champ. Erreur sans les retirées, à une station retirée : ce que le reste du réseau et GFS devinent sans elle.</p>`;
  }, { attente: "Deux passes du modèle à cette heure, avec et sans les stations…" });
}
function historique(frais = false){
  charger("sv-hist", "/atom-wind/historique", d => {
    const L = d.journees || [];
    if(!L.length) return `<p class="muted" style="margin:6px 0 0">Aucune journée archivée pour l'instant.</p>`;
    const ouvrir = typeof A.choisirJour === "function";
    setTimeout(() => document.querySelectorAll("[data-sv-hist]").forEach(b => b.onclick = async () => {
      b.disabled = true; await A.choisirJour("hist:" + b.dataset.svHist);
      const sel = document.getElementById("aw-jour"); if(sel) sel.value = "hist:" + b.dataset.svHist;
      b.disabled = false; b.textContent = "Affichée";
    }));
    return tableau(["Journée", "Type", "Émise le", "Taille", "Écrite (UTC)"].concat(ouvrir ? [""] : []), L.map(h => [date(h.date),
        h.type === "temps reel" ? "temps réel" : `prévision J+${h.echeance_j}`, h.emise_le ? date(h.emise_le) : "—",
        `${entier(h.taille_ko)} ko`, esc((h.ecrit_utc || "").replace("T", " ").replace("Z", ""))]
      .concat(ouvrir ? [`<button class="btn" data-sv-hist="${esc(h.nom)}">Ouvrir</button>`] : [])))
      + (ouvrir ? `<p class="muted sv-note">« Ouvrir » affiche la journée archivée dans toutes les vues, comme le sélecteur de journée.</p>` : "");
  }, { ms: 20000, frais });
}

// ------------------------------------------------------------------ branchement sur l'ouverture des vues
const VUES = { eolien: vueEolien, alertes: vueAlertes, admin: vueAdmin, agri: vueAgri, fiabilite: vueFiabilite };
document.addEventListener("atom:vue", e => {
  const f = VUES[e.detail && e.detail.vue];
  if(f) try { f(); } catch(err){ console.warn("ATOM Wind : services de la vue " + e.detail.vue, err); }
});
A.servicesAtom = { afficher: v => VUES[v] && VUES[v](), viderCache: () => cache.clear() };
})();

export {};
