/* Application web PC-FNO3D : carte, profil vertical, verdict DSO, intégration.
 *
 * Tout passe par l'API JSON du même serveur. La clé d'API est conservée dans
 * localStorage (ou en mémoire si l'utilisateur refuse) et envoyée en en-tête X-API-Key.
 * Le serveur exige une clé par défaut. Seul le mode ouvert explicite du développement
 * local (`pcfno3d serve --insecure-open`, health.auth === "ouvert") saute l'écran de
 * connexion. Sans aucune clé configurée (health.auth === "aucune_cle"), le serveur refuse
 * tout : l'écran le dit au lieu d'attendre une clé qui ne peut pas exister.
 */
"use strict";

const KEY_STORAGE = "pcfno3d.apikey";
const el = (id) => document.getElementById(id);

const state = {
  key: null,
  meta: null,
  instant: null,
  field: null,     // dernière carte reçue
  point: null,
};

/* ------------------------------------------------------------------ transport */
function readStoredKey() {
  try { return localStorage.getItem(KEY_STORAGE); } catch { return null; }
}
function storeKey(key, remember) {
  state.key = key;
  if (!remember) return;
  try { localStorage.setItem(KEY_STORAGE, key); } catch { /* navigation privée */ }
}
function forgetKey() {
  state.key = null;
  try { localStorage.removeItem(KEY_STORAGE); } catch { /* ignore */ }
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.key) headers["X-API-Key"] = state.key;
  if (options.body) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, { ...options, headers });
  if (res.status === 429) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ||
      `quota dépassé, réessayer dans ${res.headers.get("Retry-After") || "?"} s`);
  }
  if (res.status === 401 || res.status === 403) {
    const err = new Error((await res.json().catch(() => ({}))).detail || "accès refusé");
    err.auth = true;
    throw err;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function setStatus(text, kind = "") {
  const node = el("status");
  node.textContent = text;
  node.className = `status ${kind}`;
}

/* ------------------------------------------------------------------ palette */
const STOPS = [[44, 46, 107], [46, 108, 168], [63, 165, 138], [183, 195, 74], [242, 224, 58]];
function color(t) {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(x), STOPS.length - 2);
  const f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/* ------------------------------------------------------------------ démarrage */
async function boot() {
  let health;
  try {
    health = await (await fetch("/api/health")).json();
  } catch {
    setStatus("serveur injoignable", "err");
    return;
  }
  /* fermé par défaut : seule une réponse explicite "ouvert" dispense de clé */
  const needsKey = health.auth !== "ouvert";
  state.key = readStoredKey();

  if (health.auth === "aucune_cle") {
    return showGate(health.message ||
      "aucune clé configurée sur ce serveur : toutes les requêtes sont refusées");
  }
  if (needsKey && !state.key) return showGate();
  try {
    await start(health);
  } catch (err) {
    if (err.auth && needsKey) return showGate(err.message);
    setStatus(err.message, "err");
  }
}

function showGate(message) {
  el("gate").hidden = false;
  el("app").hidden = true;
  setStatus("clé d'API requise");
  if (message) { el("keyError").textContent = message; el("keyError").hidden = false; }
  el("keyInput").focus();
}

async function start(health) {
  state.meta = await api("/meta");            // premier appel : valide aussi la clé
  el("gate").hidden = true;
  el("app").hidden = false;

  const levels = state.meta.z_levels_asl_m;
  const alt = el("alt");
  alt.min = levels[0];
  alt.max = levels[levels.length - 1];
  /* altitude de depart = niveau le plus haut, comme MAP_ALTITUDE du notebook v7 :
     sur le domaine reel le relief atteint 677 m, donc 120 m est souvent sous terre */
  alt.value = levels[levels.length - 1];
  el("levelTicks").innerHTML =
    `<span>${levels[0]} m</span><span>${levels[levels.length - 1]} m</span>`;
  el("pz").value = alt.value;

  const mcp = `${location.origin}/mcp`;
  el("mcpUrl").textContent =
    `${mcp}\n\n# en-tête requis\nX-API-Key: pcf_…\n\n` +
    `# test rapide (liste des outils)\ncurl -s ${mcp} \\\n` +
    `  -H "X-API-Key: pcf_…" \\\n` +
    `  -H "Content-Type: application/json" \\\n` +
    `  -H "Accept: application/json, text/event-stream" \\\n` +
    `  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'`;
  el("apiExample").textContent =
    `curl -H "X-API-Key: pcf_…" \\\n  "${location.origin}/api/point?x=5000&y=5000&z=120"`;

  /* en mode lien, le modèle tourne ailleurs : il faut que ça se voie */
  const lien = health.mode === "lien"
    ? ` · relayé vers ${(health.amont && health.amont.label) || "amont"}`
    : "";
  setStatus(`${state.meta.modele} · ${state.meta.device} · ` +
            `${state.meta.grille.nz}×${state.meta.grille.ny}×${state.meta.grille.nx} · ` +
            `${health.auth === "ouvert" ? "accès OUVERT (développement local)" : "clé d'API"}` +
            `${lien}`, health.auth === "ouvert" ? "" : "ok");

  await loadInstants();
  await refresh();
}

async function loadInstants() {
  const body = await api("/instants?split=test&limit=200");
  const list = body.instants.length ? body.instants
    : (await api("/instants?limit=200")).instants;
  const select = el("instant");
  select.innerHTML = list
    .map((r) => `<option value="${r.stem}">${r.timestamp} — ${r.split}</option>`).join("");
  state.instant = list.length ? list[0].stem : null;
  if (!list.length) setStatus("aucun instant en cache", "err");
}

/* ------------------------------------------------------------------ carte */
async function refresh() {
  if (!state.instant) return;
  const z = +el("alt").value;
  const stride = +el("stride").value;
  el("altVal").textContent = z;
  el("strideVal").textContent = stride;
  const variable = el("variable").value;

  setStatus("reconstruction…");
  try {
    const query = `?z=${z}&stem=${state.instant}&variable=${variable}&stride=${stride}`;
    const [field, summary] = await Promise.all([
      api(`/field${query}`),
      api("/reconstruct", { method: "POST", body: JSON.stringify({ stem: state.instant }) }),
    ]);
    state.field = field;
    drawMap(field, variable === "speed" || variable === "u" || variable === "v" ||
                   variable === "w");
    el("unit").textContent = variable === "direction" ? "degrés" : "m/s";
    el("vmin").textContent = field.min ?? "—";
    el("vmax").textContent = field.max ?? "—";
    el("instantInfo").innerHTML = `
      <dt>Date</dt><dd>${field.timestamp_utc ?? "—"}</dd>
      <dt>Capteurs</dt><dd>${summary.capteurs_utilises.toLocaleString("fr-FR")}</dd>
      <dt>Inférence</dt><dd>${(summary.inference_s * 1000).toFixed(0)} ms
        ${summary.dso4_conforme ? "✓" : "✗"} (DSO4)</dd>
      <dt>Nœuds 3D</dt><dd>${summary.noeuds_3d.toLocaleString("fr-FR")}</dd>
      <dt>Vitesse moy.</dt><dd>${summary.vitesse_moyenne_m_s} m/s</dd>
      <dt>Vitesse max</dt><dd>${summary.vitesse_max_m_s} m/s</dd>
      <dt>Relief</dt><dd>${field.terrain_min_m}–${field.terrain_max_m} m</dd>`;
    setStatus(`${state.meta.modele} · ${(summary.inference_s * 1000).toFixed(0)} ms`, "ok");
  } catch (err) {
    setStatus(err.message, "err");
  }
}

function drawMap(field, withArrows) {
  const { ny, nx, valeurs, min, max } = field;
  const canvas = el("map");
  canvas.width = nx; canvas.height = ny;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  const span = (max - min) || 1;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = valeurs[j][i];
      /* les lignes vont du sud au nord, le canvas du haut vers le bas */
      const p = ((ny - 1 - j) * nx + i) * 4;
      if (v === null) { img.data[p + 3] = 0; continue; }
      const c = color((v - min) / span);
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  if (withArrows && el("arrows").checked) drawArrows(ctx, field);
  if (state.point) markPoint(ctx, field);
}

async function drawArrows(ctx, field) {
  /* u et v sont demandés séparément : la carte affichée peut être une autre variable */
  const z = field.z_asl_m, stride = field.stride;
  const base = `?z=${z}&stem=${state.instant}&stride=${stride}`;
  try {
    const [fu, fv] = await Promise.all([api(`/field${base}&variable=u`),
                                        api(`/field${base}&variable=v`)]);
    const { ny, nx } = field;
    const step = Math.max(Math.round(nx / 26), 2);
    const len = step * 0.4;
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = Math.max(nx / 500, 0.5);
    for (let j = step; j < ny; j += step) {
      for (let i = step; i < nx; i += step) {
        const u = fu.valeurs[j][i], v = fv.valeurs[j][i];
        if (u === null || v === null) continue;
        const n = Math.hypot(u, v) || 1;
        const x = i + .5, y = ny - 1 - j + .5;
        ctx.beginPath();
        ctx.moveTo(x - (u / n) * len, y + (v / n) * len);
        ctx.lineTo(x + (u / n) * len, y - (v / n) * len);
        ctx.stroke();
      }
    }
  } catch { /* les flèches sont un bonus : pas de message d'erreur */ }
}

function markPoint(ctx, field) {
  const i = Math.round(state.point.x_m / field.dx_m);
  const j = Math.round(state.point.y_m / field.dy_m);
  const y = field.ny - 1 - j;
  ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(field.nx / 400, .8);
  ctx.beginPath(); ctx.arc(i + .5, y + .5, Math.max(field.nx / 90, 2), 0, Math.PI * 2);
  ctx.stroke();
}

/* ------------------------------------------------------------------ point et profil */
async function queryPoint(x, y, z) {
  try {
    const [point, prof] = await Promise.all([
      api(`/point?x=${x}&y=${y}&z=${z}&stem=${state.instant}`),
      api(`/profile?x=${x}&y=${y}&stem=${state.instant}`),
    ]);
    state.point = point;
    el("px").value = Math.round(point.x_m);
    el("py").value = Math.round(point.y_m);
    el("pointInfo").innerHTML = `
      <dt>Position</dt><dd>${Math.round(point.x_m)} ; ${Math.round(point.y_m)} m</dd>
      <dt>Relief</dt><dd>${point.z_terrain_m.toFixed(0)} m</dd>
      <dt>Altitude</dt><dd>${point.z_asl_m} m ASL (${fmt(point.z_agl_m, 0)} m sol)</dd>
      ${point.sous_le_relief ? `<dt>État</dt><dd class="err">sous le relief — pas de vent ici</dd>` : ""}
      <dt>Vitesse</dt><dd>${fmt(point.wind_speed_m_s)} m/s</dd>
      <dt>Direction</dt><dd>${fmt(point.wind_direction_deg, 1)}°</dd>
      <dt>u ; v</dt><dd>${fmt(point.wind_u_m_s)} ; ${fmt(point.wind_v_m_s)} m/s</dd>
      <dt>w</dt><dd>${fmt(point.wind_w_m_s, 3)} m/s</dd>`;
    drawProfile(prof);
    if (state.field) drawMap(state.field, true);
    el("mapHint").textContent = point.sous_le_relief
      ? `sous le relief à ${point.z_asl_m} m (sol à ${point.z_terrain_m.toFixed(0)} m)`
      : `${fmt(point.wind_speed_m_s)} m/s · ${fmt(point.wind_direction_deg, 0)}° ` +
        `à ${point.z_asl_m} m`;
  } catch (err) {
    setStatus(err.message, "err");
  }
}

function fmt(v, digits = 2) {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : (+v).toFixed(digits);
}

function drawProfile(prof) {
  const canvas = el("profile");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 560, H = 320;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const css = getComputedStyle(document.body);
  const ink = css.color;
  const muted = getComputedStyle(document.querySelector(".muted")).color;

  ctx.clearRect(0, 0, W, H);
  const pad = { l: 52, r: 14, t: 14, b: 34 };
  const usable = prof.niveaux.filter((n) => !n.sous_le_relief);
  if (!usable.length) {
    ctx.fillStyle = muted; ctx.font = "13px system-ui";
    ctx.fillText("Point entièrement sous le relief", pad.l, H / 2);
    return;
  }
  const zMax = prof.niveaux[prof.niveaux.length - 1].z_asl_m;
  const sMax = Math.max(...usable.map((n) => n.wind_speed_m_s)) * 1.15 || 1;
  const X = (s) => pad.l + (s / sMax) * (W - pad.l - pad.r);
  const Y = (z) => H - pad.b - (z / zMax) * (H - pad.t - pad.b);

  /* axes et graduations */
  ctx.strokeStyle = muted; ctx.globalAlpha = .35; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = muted; ctx.font = "11px system-ui";
  for (const n of prof.niveaux) {
    ctx.globalAlpha = .18;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(n.z_asl_m)); ctx.lineTo(W - pad.r, Y(n.z_asl_m));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(`${n.z_asl_m}`, 8, Y(n.z_asl_m) + 3);
  }
  ctx.fillText("vitesse (m/s)", W / 2 - 34, H - 8);
  ctx.save(); ctx.translate(12, pad.t + 8); ctx.fillText("m ASL", 0, 0); ctx.restore();
  for (let s = 0; s <= sMax; s += Math.max(1, Math.round(sMax / 5))) {
    ctx.fillText(`${s}`, X(s) - 4, H - pad.b + 15);
  }

  /* relief */
  ctx.fillStyle = "rgba(139,87,42,.35)";
  ctx.fillRect(pad.l, Y(prof.z_terrain_m), W - pad.l - pad.r, H - pad.b - Y(prof.z_terrain_m));

  /* profil de vitesse */
  ctx.strokeStyle = "#1b6ac9"; ctx.lineWidth = 2; ctx.beginPath();
  usable.forEach((n, k) => {
    const x = X(n.wind_speed_m_s), y = Y(n.z_asl_m);
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#1b6ac9";
  for (const n of usable) {
    ctx.beginPath(); ctx.arc(X(n.wind_speed_m_s), Y(n.z_asl_m), 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = ink; ctx.font = "12px system-ui";
  ctx.fillText(`x = ${Math.round(prof.x_m)} m, y = ${Math.round(prof.y_m)} m, ` +
               `relief ${prof.z_terrain_m.toFixed(0)} m`, pad.l + 6, pad.t + 12);
  el("profileNote").textContent =
    `${usable.length} niveaux au-dessus du relief sur ${prof.niveaux.length}.`;
}

/* ------------------------------------------------------------------ DSO */
async function loadDso() {
  const node = el("dso");
  node.innerHTML = `<p class="muted">chargement…</p>`;
  try {
    const body = await api("/dso");
    if (!body.disponible) {
      node.innerHTML = `<p class="err">${body.message}</p>
        <p class="muted small">Modèle chargé : <code>${body.modele ?? "—"}</code></p>`;
      return;
    }
    const s = body.resume;
    const src = s.source || null;          /* evaluation importee d'un notebook ? */
    const ou = src ? src.materiel : "ce serveur";
    const verdict = (ok) =>
      `<span class="verdict ${ok ? "ok" : "ko"}">${ok ? "CONFORME" : "NON CONFORME"}</span>`;
    const rows = [
      ["DSO1", "RMSE u, v ≤ 0,5 m/s par niveau", s.dso1_rmse],
      ["DSO1", "R² u, v ≥ 0,95 par niveau", s.dso1_r2],
      ["DSO1", `RMSE global u = ${fmt(s.dso1_rmse_global_u, 3)} · v = ` +
               `${fmt(s.dso1_rmse_global_v, 3)} m/s`, s.dso1_rmse_global_conforme],
      ["DSO1 révisé (proposé)", "RMSE u, v ≤ max(0,5 m/s ; 10 % de la vitesse moyenne " +
       "du niveau) — tolérance OMM sur la mesure du vent. Proposition à valider avec " +
       "Tukhnanutha : elle complète le DSO1 officiel ci-dessus, elle ne le remplace pas.",
       s.critere_omm],
      ["DSO2", `résidu de masse moyen = ${s.dso2_rms_moyen?.toExponential(2)}`,
       s.dso2_conforme],
      ["DSO3", `rugosité prédiction / vérité = ${fmt(s.dso3_rugosite, 3)}`, s.dso3_conforme],
      ["DSO4", `inférence = ${(s.dso4_inference_s * 1000).toFixed(0)} ms, mesurée sur ${ou}` +
               (src ? ` — sur ce poste sans GPU, compter ~20 s par instant` : ""),
       s.dso4_conforme],
      ["DSO5", `RMSE à 25 % de capteurs = ${fmt(s.dso5_rmse_25pct, 3)} m/s`, s.dso5_conforme],
    ];
    const levels = (s.par_niveau_synoptique || []).filter((r) => r.couverture >= 0.05);
    node.innerHTML = `
      <table><thead><tr><th>Objectif</th><th>Critère</th><th>Verdict</th></tr></thead><tbody>
      ${rows.map(([k, label, ok]) =>
        `<tr><td><b>${k}</b></td><td>${label}</td><td>${verdict(!!ok)}</td></tr>`).join("")}
      </tbody></table>
      <p class="muted small">Modèle <code>${s.modele}</code> · époque ${s.epoque ?? "—"} ·
        ${s.instants_synoptiques} instants synoptiques de test ·
        ${Math.round((s.capteurs_visibles ?? 0.5) * 100)} % des capteurs visibles ·
        MAE direction ${fmt(s.mae_direction_deg, 2)}°</p>
      ${src ? `<p class="small"><b>Source :</b> évaluation faite par le notebook
        <code>${src.notebook}</code> (BLOC ${src.bloc}) sur ${src.materiel}, importée telle
        quelle — rien n'est recalculé sur ce poste. ${src.precision}.</p>` : ""}
      ${levels.length ? `<div class="scroll-x"><table>
        <thead><tr><th>Niveau</th><th>Couverture</th><th>RMSE u</th><th>RMSE v</th>
        <th>RMSE w</th><th>R² u</th><th>R² v</th><th>R² w</th></tr></thead><tbody>
        ${levels.map((r) => `<tr><td>${r.z_m} m</td>
          <td>${(r.couverture * 100).toFixed(1)} %</td>
          <td>${fmt(r.rmse_u, 3)}</td><td>${fmt(r.rmse_v, 3)}</td><td>${fmt(r.rmse_w, 3)}</td>
          <td>${fmt(r.r2_u, 4)}</td><td>${fmt(r.r2_v, 4)}</td><td>${fmt(r.r2_w, 4)}</td>
          </tr>`).join("")}</tbody></table></div>` : ""}
      ${(s.robustesse || []).length ? `<h3>Robustesse au nombre de capteurs</h3>
        <div class="scroll-x"><table>
        <thead><tr><th>Capteurs visibles</th><th>RMSE max u</th><th>RMSE max v</th>
        <th>R² min u</th><th>R² min v</th></tr></thead><tbody>
        ${s.robustesse.map((r) => `<tr><td>${Math.round(r.capteurs_visibles * 100)} %</td>
          <td>${fmt(r.rmse_u_max, 3)}</td><td>${fmt(r.rmse_v_max, 3)}</td>
          <td>${fmt(r.r2_u_min, 4)}</td><td>${fmt(r.r2_v_min, 4)}</td></tr>`).join("")}
        </tbody></table></div>
        <p class="muted small">L'erreur devrait baisser quand on ajoute des capteurs. Si
        100 % fait moins bien que 50 %, le modèle n'a presque jamais vu tous les capteurs à
        l'entraînement — c'est ce que corrige <code>mask.p_full</code> en v8.</p>` : ""}`;
  } catch (err) {
    node.innerHTML = `<p class="err">${err.message}</p>`;
  }
}

/* ------------------------------------------------------------------ événements */
el("keySubmit").addEventListener("click", async () => {
  const key = el("keyInput").value.trim();
  if (!key) return;
  storeKey(key, el("keyRemember").checked);
  el("keyError").hidden = true;
  try {
    await start(await (await fetch("/api/health")).json());
  } catch (err) {
    forgetKey();
    el("keyError").textContent = err.message;
    el("keyError").hidden = false;
  }
});
el("keyInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") el("keySubmit").click();
});
el("logout").addEventListener("click", () => { forgetKey(); location.reload(); });

for (const [id, event] of [["alt", "change"], ["stride", "change"], ["variable", "change"],
                           ["arrows", "change"], ["instant", "change"]]) {
  el(id).addEventListener(event, () => {
    if (id === "instant") state.instant = el("instant").value;
    if (id === "alt") el("pz").value = el("alt").value;
    refresh();
  });
}
el("alt").addEventListener("input", () => { el("altVal").textContent = el("alt").value; });
el("stride").addEventListener("input", () => { el("strideVal").textContent = el("stride").value; });

el("map").addEventListener("click", (e) => {
  if (!state.field) return;
  const rect = el("map").getBoundingClientRect();
  const i = Math.floor(((e.clientX - rect.left) / rect.width) * state.field.nx);
  const j = state.field.ny - 1 - Math.floor(((e.clientY - rect.top) / rect.height) * state.field.ny);
  queryPoint(i * state.field.dx_m, j * state.field.dy_m, +el("alt").value);
});

el("pointQuery").addEventListener("click", () =>
  queryPoint(+el("px").value, +el("py").value, +el("pz").value));

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((s) => s.classList.remove("active"));
    tab.classList.add("active");
    el(`tab-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "qualite") loadDso();
  });
}

boot();
