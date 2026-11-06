import { readFile, mkdir, copyFile, writeFile, stat, readdir } from 'node:fs/promises';
import { resolve, dirname, sep } from 'node:path';

const supplied = process.argv[2];
if (!supplied) throw new Error('Usage: node scripts/import_tukai_terrain.mjs <ATOM_Wind_Tunisia_Terrain directory>');
const source = resolve(supplied, 'ATOM_Wind_Tunisia_Web');
const target = resolve('frontend/public/terrain/tunisia');
const manifest = JSON.parse(await readFile(resolve(source, 'terrain-manifest.json'), 'utf8'));
const geo = JSON.parse(await readFile(resolve(source, 'geo-manifest.json'), 'utf8'));
if (geo.sceneCRS.horizontal !== 'EPSG:32632' || geo.verticalExaggeration !== 1) throw new Error('Unsupported terrain coordinate contract');
const water = JSON.parse(await readFile(resolve(source, 'water/water-manifest.json'), 'utf8'));
const assets = new Set(['overview.glb', 'terrain-manifest.json', 'geo-manifest.json', 'source-inventory.json', 'LICENSES.md', 'README.md', 'water/water-manifest.json', ...water.assets.map(asset => asset.uri)]);
for (const tile of manifest.tiles) { assets.add(tile.uri); if (tile.elevation) assets.add(tile.elevation.uri); }
// The vector database and notices accompany the derived boundary.
for (const name of await readdir(resolve(source, 'boundaries'))) assets.add(`boundaries/${name}`);
let bytes = 0;
for (const name of assets) {
  const from = resolve(source, name), to = resolve(target, name);
  if (!from.startsWith(source + sep) || !to.startsWith(target + sep)) throw new Error(`Asset escapes dataset: ${name}`);
  await mkdir(dirname(to), { recursive: true });
  await copyFile(from, to);
  bytes += (await stat(from)).size;
}
const modules = resolve('frontend/src/terrain/source');
await mkdir(modules, { recursive: true });
await copyFile(resolve(source, 'integration', 'wgs84.js'), resolve(modules, 'wgs84.js'));
const originalGeo=await readFile(resolve(source,'integration','geo.js'),'utf8');
const compressedRead="const decoded=new Response(r.body.pipeThrough(new DecompressionStream('gzip')));const buffer=await decoded.arrayBuffer();";
if(!originalGeo.includes(compressedRead))throw new Error('Review the changed Tukai elevation loader before importing.');
// Keep the supplied projection and interpolation; adapt only HTTP decoding for the host server.
await writeFile(resolve(modules,'geo.js'),"import {elevationBytes} from '../elevation-codec';\n"+originalGeo.replace(compressedRead,'const buffer=await elevationBytes(r);'));
await writeFile(resolve(target, 'import.json'), JSON.stringify({ name: geo.name, imported: new Date().toISOString(), assets: assets.size, bytes, sourceCRS: geo.sourceCRS, sceneCRS: geo.sceneCRS }, null, 2));
console.log(`Imported ${assets.size} assets (${(bytes / 1024 / 1024).toFixed(1)} MiB), preserving the source coordinate contract.`);
