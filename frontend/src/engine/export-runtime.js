
const NODATA = -9999;
const params = new URLSearchParams(location.search);
const srcInput = document.getElementById("src");
srcInput.value = params.get("src") || "../artifacts/mini_run/web/latest";

const el = (id) => document.getElementById(id);
const state = { manifest: null, fields: {}, terrain: null };

/* palette perceptuellement croissante, identique a la rampe CSS de la legende */
const STOPS = [[44,46,107],[46,108,168],[63,165,138],[183,195,74],[242,224,58]];
function color(t) {
  if (!isFinite(t)) return [0, 0, 0, 0];
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(x), STOPS.length - 2), f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, 255];
}

async function loadFloat32(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} : HTTP ${res.status}`);
  return new Float32Array(await res.arrayBuffer());
}

async function load(base) {
  el("status").textContent = "Chargement…";
  el("status").classList.remove("err");
  const res = await fetch(`${base}/manifest.json`);
  if (!res.ok) throw new Error(`manifest.json introuvable (HTTP ${res.status})`);
  const m = await res.json();
  state.manifest = m;
  state.terrain = await loadFloat32(`${base}/${m.terrain.file}`);
  state.fields = {};
  for (const [name, info] of Object.entries(m.champs)) {
    state.fields[name] = await loadFloat32(`${base}/${info.file}`);
  }
  const { nz, ny } = m.grille;
  el("alt").max = nz - 1; el("alt").value = Math.min(2, nz - 1);
  el("row").max = ny - 1; el("row").value = Math.floor(ny / 2);
  el("info").innerHTML = `
    <dt>Date</dt><dd>${m.timestamp_utc ?? "—"}</dd>
    <dt>Grille</dt><dd>${nz} × ${ny} × ${m.grille.nx}</dd>
    <dt>Pas</dt><dd>${m.grille.dx_m} m</dd>
    <dt>Capteurs</dt><dd>${(m.capteurs_utilises ?? 0).toLocaleString("fr-FR")}</dd>
    <dt>Inférence</dt><dd>${((m.inference_s ?? 0) * 1000).toFixed(0)} ms</dd>`;
  el("status").textContent =
    `${Object.keys(state.fields).join(", ")} — ${m.z_levels_asl_m.length} niveaux ` +
    `(${m.z_levels_asl_m[0]}–${m.z_levels_asl_m.at(-1)} m ASL)`;
  draw();
}

function levelSlice(name, k) {
  const { ny, nx } = state.manifest.grille;
  return state.fields[name].subarray(k * ny * nx, (k + 1) * ny * nx);
}

function extent(arr) {
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v === NODATA) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function draw() {
  const m = state.manifest;
  if (!m) return;
  const { ny, nx } = m.grille;
  const k = +el("alt").value, name = el("variable").value, row = +el("row").value;
  el("altVal").textContent = m.z_levels_asl_m[k];
  el("rowVal").textContent = row;

  /* carte horizontale */
  const plane = levelSlice(name, k);
  const [lo, hi] = extent(plane);
  el("vmin").textContent = isFinite(lo) ? lo.toFixed(1) : "—";
  el("vmax").textContent = isFinite(hi) ? hi.toFixed(1) : "—";
  const map = el("map");
  map.width = nx; map.height = ny;
  const ctx = map.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = plane[j * nx + i];
      /* l'origine du tableau est en bas a gauche, celle du canvas en haut a gauche */
      const p = ((ny - 1 - j) * nx + i) * 4;
      const c = v === NODATA ? [0, 0, 0, 0] : color((v - lo) / (hi - lo || 1));
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = c[3];
    }
  }
  ctx.putImageData(img, 0, 0);
  drawArrows(ctx, k, nx, ny);

  /* coupe verticale le long de la ligne choisie */
  const nz = m.grille.nz;
  const sec = el("section");
  sec.width = nx; sec.height = nz * 8;
  const sctx = sec.getContext("2d");
  sctx.clearRect(0, 0, sec.width, sec.height);
  const all = [];
  for (let kk = 0; kk < nz; kk++) {
    const s = levelSlice(name, kk);
    for (let i = 0; i < nx; i++) all.push(s[row * nx + i]);
  }
  const [slo, shi] = extent(all);
  for (let kk = 0; kk < nz; kk++) {
    const s = levelSlice(name, kk);
    for (let i = 0; i < nx; i++) {
      const v = s[row * nx + i];
      if (v === NODATA) continue;
      const c = color((v - slo) / (shi - slo || 1));
      sctx.fillStyle = `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
      sctx.fillRect(i, sec.height - (kk + 1) * 8, 1, 8);
    }
  }
  el("sectionLabel").textContent =
    `Coupe verticale, ligne ${row} — ${m.z_levels_asl_m[0]} m (bas) à ` +
    `${m.z_levels_asl_m.at(-1)} m ASL (haut)`;
}

function drawArrows(ctx, k, nx, ny) {
  if (!state.fields.u || !state.fields.v) return;
  const u = levelSlice("u", k), v = levelSlice("v", k);
  const step = Math.max(Math.round(nx / 28), 2), len = step * 0.42;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = Math.max(nx / 600, 0.6);
  for (let j = step; j < ny; j += step) {
    for (let i = step; i < nx; i += step) {
      const uu = u[j * nx + i], vv = v[j * nx + i];
      if (uu === NODATA || vv === NODATA) continue;
      const n = Math.hypot(uu, vv) || 1;
      const x = i + 0.5, y = ny - 1 - j + 0.5;
      ctx.beginPath();
      ctx.moveTo(x - (uu / n) * len, y + (vv / n) * len);
      ctx.lineTo(x + (uu / n) * len, y - (vv / n) * len);
      ctx.stroke();
    }
  }
}

function fail(err) {
  el("status").textContent = `${err.message} — vérifie le chemin de l'export`;
  el("status").classList.add("err");
}

for (const id of ["alt", "variable", "row"]) el(id).addEventListener("input", draw);
srcInput.addEventListener("change", () => load(srcInput.value.replace(/\/$/, "")).catch(fail));
load(srcInput.value.replace(/\/$/, "")).catch(fail);

export {};
