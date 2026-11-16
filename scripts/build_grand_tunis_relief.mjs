// Relief réel du domaine Grand Tunis (68 × 56 km) pour l'interface ATOM Wind, tiré du terrain
// Tunisie importé (scripts/import_tukai_terrain.mjs) : altitude moyenne par cellule depuis les
// grilles Copernicus GLO-30 (pas de 32 m), mer et lacs rasterisés depuis les couches d'eau
// (Natural Earth / geoBoundaries / ESA WorldCover 2021).
//
//   node scripts/build_grand_tunis_relief.mjs
//
// Sortie : frontend/public/atom-wind/relief/grand-tunis.json (en-tête) + grand-tunis.bin.gz
// (Int16 LE altitude en décimètres EGM2008, puis Uint8 eau : 0 terre, 1 mer, 2 lac ;
// rangées du sud au nord, comme la grille de la page). Sert le relief de démonstration quand
// le service PC-FNO3D ne fournit pas sa grille ; le vent de démonstration reste synthétique.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import { geodeticToUtm } from '../frontend/src/terrain/source/wgs84.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

// même projection linéaire que la page (atom-runtime.js : LAT0, LON0, KX, KY, W, H)
const LAT0 = 36.395, LON0 = 10.04, KX = 89.42, KY = 110.97, W = 68, H = 56;
const PAS_KM = .125, NX = Math.round(W / PAS_KM), NY = Math.round(H / PAS_KM);
const SRC = resolve('frontend/public/terrain/tunisia');
const OUT = resolve('frontend/public/atom-wind/relief');

const manifest = JSON.parse(await readFile(resolve(SRC, 'terrain-manifest.json'), 'utf8'));
const tiles = new Map(manifest.tiles.filter(t => t.lod === 2 && t.elevation).map(t => [t.id, t]));
const grids = new Map();
async function grid(t) {
  let v = grids.get(t.id);
  if (!v) {
    const raw = gunzipSync(await readFile(resolve(SRC, t.elevation.uri)));
    const m = t.elevation; v = new DataView(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    if (v.byteLength !== m.width * m.height * 2) throw new Error(`taille inattendue : ${t.id}`);
    if (m.predictor === 'horizontal-delta-int16-modulo')
      for (let r = 0; r < m.height; r++) { const s = r * m.width * 2; let p = v.getInt16(s, true);
        for (let c = 1; c < m.width; c++) { const a = s + c * 2; p = (p + v.getInt16(a, true)) << 16 >> 16; v.setInt16(a, p, true); } }
    else if (m.predictor) throw new Error(`prédicteur inconnu : ${m.predictor}`);
    grids.set(t.id, v);
  }
  return v;
}
// altitude DEM au plus proche pixel de 32 m ; null = pas de tuile ou noData (la mer)
async function dem(e, n) {
  const t = tiles.get(`7-${Math.floor(e / 8192)}-${Math.floor((n - 3276800) / 8192)}`);
  if (!t) return null;
  const v = await grid(t), m = t.elevation, [w, , , nn] = t.projectedBounds;
  const c = Math.min(m.width - 1, Math.max(0, Math.round((e - w) / 32))), r = Math.min(m.height - 1, Math.max(0, Math.round((nn - n) / 32)));
  const x = v.getInt16((r * m.width + c) * 2, true);
  return x === m.noData ? null : x * m.scale;
}
const utm = (lon, lat) => { const p = geodeticToUtm(lat, lon, 32); return [p.easting, p.northing]; };

// ---- couches d'eau : triangles en UTM (x scène = E − 500 000, z scène = 3 800 000 − N)
await MeshoptDecoder.ready;
async function triangles(uri) {
  const b = await readFile(resolve(SRC, uri)), jl = b.readUInt32LE(12), j = JSON.parse(b.subarray(20, 20 + jl).toString());
  const bin = b.subarray(20 + jl + 8);
  const view = i => {
    const bv = j.bufferViews[i], ext = bv.extensions?.EXT_meshopt_compression;
    if (!ext) return bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
    const out = new Uint8Array(ext.count * ext.byteStride);
    MeshoptDecoder.decodeGltfBuffer(out, ext.count, ext.byteStride, bin.subarray(ext.byteOffset || 0, (ext.byteOffset || 0) + ext.byteLength), ext.mode, ext.filter);
    return out;
  };
  const comp = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
  const read = i => {
    const a = j.accessors[i], bv = j.bufferViews[a.bufferView], T = comp[a.componentType], n = { SCALAR: 1, VEC3: 3, VEC4: 4 }[a.type];
    const bytes = view(a.bufferView), stride = (bv.extensions?.EXT_meshopt_compression?.byteStride || bv.byteStride || n * T.BYTES_PER_ELEMENT) / T.BYTES_PER_ELEMENT;
    const arr = new T(bytes.buffer, bytes.byteOffset + (a.byteOffset || 0), Math.floor((bytes.byteLength - (a.byteOffset || 0)) / T.BYTES_PER_ELEMENT));
    const out = new Float64Array(a.count * n), scale = a.normalized ? { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 }[a.componentType] : 1;
    for (let k = 0; k < a.count; k++) for (let c = 0; c < n; c++) out[k * n + c] = arr[k * stride + c] / scale;
    return out;
  };
  const tris = [];
  for (const node of j.nodes.filter(n => n.mesh !== undefined)) {
    const [tx, ty, tz] = node.translation || [0, 0, 0], [sx, sy, sz] = node.scale || [1, 1, 1];
    if (node.rotation && node.rotation.some((q, i) => Math.abs(q - (i === 3 ? 1 : 0)) > 1e-9)) throw new Error(`rotation non gérée : ${uri}`);
    for (const pr of j.meshes[node.mesh].primitives) {
      const P = read(pr.attributes.POSITION), I = pr.indices !== undefined ? read(pr.indices) : P.map((_, k) => k).filter(k => k < P.length / 3);
      for (let k = 0; k < I.length; k += 3) tris.push([0, 1, 2].map(q => { const v = I[k + q] * 3; return [P[v] * sx + tx + 500000, 3800000 - (P[v + 2] * sz + tz)]; }));
    }
  }
  return tris;
}
// index spatial des triangles qui touchent le domaine
const [e0, n0] = utm(LON0 - .02, LAT0 - .02), [e1, n1] = utm(LON0 + W / KX + .02, LAT0 + H / KY + .02);
function indexer(tris) {
  const B = 2000, bins = new Map();
  for (const t of tris) {
    const xs = t.map(p => p[0]), ys = t.map(p => p[1]);
    if (Math.max(...xs) < Math.min(e0, e1) - 5000 || Math.min(...xs) > Math.max(e0, e1) + 5000) continue;
    if (Math.max(...ys) < n0 - 5000 || Math.min(...ys) > n1 + 5000) continue;
    for (let bx = Math.floor(Math.min(...xs) / B); bx <= Math.floor(Math.max(...xs) / B); bx++)
      for (let by = Math.floor(Math.min(...ys) / B); by <= Math.floor(Math.max(...ys) / B); by++) {
        const k = `${bx},${by}`; if (!bins.has(k)) bins.set(k, []); bins.get(k).push(t);
      }
  }
  return (e, n) => {
    for (const [a, b, c] of bins.get(`${Math.floor(e / B)},${Math.floor(n / B)}`) || []) {
      const d1 = (e - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (n - b[1]), d2 = (e - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (n - c[1]), d3 = (e - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (n - a[1]);
      if (!((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))) return true;
    }
    return false;
  };
}
const dansLac = indexer(await triangles('water/natural-lakes.glb'));
const dansMer = indexer(await triangles('water/mediterranean.glb'));

// ---- grille : moyenne de 4 × 4 points DEM par cellule de 125 m
const alt = new Int16Array(NX * NY), eau = new Uint8Array(NX * NY);
let nMer = 0, nLac = 0, zmax = 0;
for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
  const k = j * NX + i, xc = (i + .5) * PAS_KM, yc = (j + .5) * PAS_KM;
  const [ec, nc] = utm(LON0 + xc / KX, LAT0 + yc / KY);
  let s = 0, c = 0, vides = 0;
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
    const [e, n] = utm(LON0 + (xc + (a - 1.5) * PAS_KM / 4) / KX, LAT0 + (yc + (b - 1.5) * PAS_KM / 4) / KY);
    const z = await dem(e, n); if (z === null) vides++; else { s += z; c++; }
  }
  const mer = dansMer(ec, nc) || vides > 8, lac = !mer && dansLac(ec, nc);
  eau[k] = mer ? 1 : lac ? 2 : 0; nMer += mer; nLac += lac;
  const z = c ? s / c : 0; alt[k] = Math.round((mer ? 0 : Math.max(lac ? -5 : 0, z)) * 10); zmax = Math.max(zmax, z);
}
const corps = Buffer.concat([Buffer.from(alt.buffer), Buffer.from(eau.buffer)]);
await mkdir(OUT, { recursive: true });
await writeFile(resolve(OUT, 'grand-tunis.bin.gz'), gzipSync(corps, { level: 9 }));
await writeFile(resolve(OUT, 'grand-tunis.json'), JSON.stringify({
  version: 1, nx: NX, ny: NY, pas_km: PAS_KM, origine: { lat: LAT0, lon: LON0 }, km_par_degre: { lon: KX, lat: KY },
  rangees: 'du sud au nord', altitude: { type: 'Int16LE', unite: 'dm', reference: 'EGM2008' }, eau: { type: 'Uint8', codes: { 0: 'terre', 1: 'mer', 2: 'lac' } },
  fichier: 'grand-tunis.bin.gz',
  sources: ['Copernicus DEM GLO-30 (moyenne par cellule de 125 m)', 'Natural Earth / geoBoundaries (trait de côte)', 'ESA WorldCover 2021 classe 80 (lacs)'],
  attribution: 'produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; © ESA WorldCover project 2021; Natural Earth; geoBoundaries',
}, null, 1) + '\n');
console.log(`grille ${NX} × ${NY} (${PAS_KM * 1000} m) : mer ${(nMer / NX / NY * 100).toFixed(1)} %, lacs ${(nLac / NX / NY * 100).toFixed(1)} %, sommet ${zmax.toFixed(0)} m`);
