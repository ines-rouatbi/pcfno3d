// Valide des instantanés contre les schémas du contrat v1 (révision 2), et vérifie les invariants que
// JSON Schema ne sait pas exprimer : tailles des binaires, règle NaN ⇔ z − relief < min_agl_m, nœuds en
// x0 + i·dx, vitesse = ‖(u,v)‖, direction météo, sha256 des deux manifestes (instantané et modèle),
// roses des vents (somme des fréquences, secteurs égaux), nz niveaux par profil, open_alerts cohérent.
// Chaque scénario a son propre compteur : tous les contrôles s'exécutent pour tous les scénarios.
// Sort en code 1 s'il y a au moins un écart.
//
// Usage : cd contract && npm ci && node tools/validate.mjs [dossier_instantané ...]
//         (par défaut : tous les dossiers de contract/fixtures/ ; disposition décrite dans make-fixtures.mjs)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCHEMAS = path.join(HERE, '..', 'schemas');
const FIXTURES = path.join(HERE, '..', 'fixtures');

const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: false, strictTypes: false, allowUnionTypes: true });
addFormats(ajv);
for (const f of fs.readdirSync(SCHEMAS)) ajv.addSchema(JSON.parse(fs.readFileSync(path.join(SCHEMAS, f), 'utf8')), f);

// Fichiers JSON de l'instantané → schéma. Les profils et roses des vents sont ajoutés dynamiquement.
const FILE_SCHEMA = {
	'latest.json': 'latest', 'times.json': 'times', 'snapshot.json': 'snapshot', 'quality.json': 'quality',
	'alerts.json': 'alerts', 'stations.json': 'stations', 'wind-farm.json': 'wind-farm', 'data-quality.json': 'data-quality',
	'publish-manifest.json': 'publish-manifest',
};
// Fichiers présents dans le dossier mais jamais publiés dans l'instantané.
const NOT_PUBLISHED = new Set(['latest.json', 'times.json', 'publish-manifest.json']);

const sha256 = (b) => crypto.createHash('sha256').update(b).digest('hex');
const dirFrom = (u, v) => ((270 - Math.atan2(v, u) * 180 / Math.PI) % 360 + 360) % 360;
const angleDiff = (a, b) => Math.abs(((a - b) + 540) % 360 - 180);
const walk = (d, base = d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name), base) : [path.relative(base, path.join(d, e.name)).split(path.sep).join('/')]);

let total = 0;

function checkSnapshot(dir) {
	const name = path.basename(dir);
	let n = 0;
	const fail = (msg) => { n++; console.error(`✗ ${name} : ${msg}`); };
	const exists = (rel) => fs.existsSync(path.join(dir, rel));
	const read = (rel) => JSON.parse(fs.readFileSync(path.join(dir, rel), 'utf8'));
	const validate = (rel, schema) => {
		if (!exists(rel)) { fail(`${rel} absent`); return null; }
		let doc;
		try { doc = read(rel); } catch (e) { fail(`${rel} : JSON illisible (${e.message})`); return null; }
		if (!ajv.validate(`${schema}.schema.json`, doc)) for (const e of ajv.errors) fail(`${rel} ${e.instancePath || '/'} ${e.message}`);
		return doc;
	};
	const guard = (label, fn) => { try { fn(); } catch (e) { fail(`${label} : contrôle impossible (${e.message})`); } };

	// 1. Schémas.
	const doc = {};
	for (const [f, s] of Object.entries(FILE_SCHEMA)) doc[f] = validate(f, s);
	const profiles = exists('profiles') ? fs.readdirSync(path.join(dir, 'profiles')).filter((f) => f.endsWith('.json')) : [];
	const roses = exists('windrose') ? fs.readdirSync(path.join(dir, 'windrose')).filter((f) => f.endsWith('.json')) : [];
	for (const f of profiles) doc[`profiles/${f}`] = validate(`profiles/${f}`, 'profile');
	for (const f of roses) doc[`windrose/${f}`] = validate(`windrose/${f}`, 'windrose');

	const snap = doc['snapshot.json'], manifest = doc['publish-manifest.json'], latest = doc['latest.json'];
	if (!snap || !manifest) { console.error(`✗ ${name} : invariants non vérifiés (snapshot ou manifeste illisible)`); total += n || 1; return; }

	// 2. Modèle : sha8, versions, fichiers par modèle et leur manifeste.
	const sha8 = snap.snapshot_id.slice(-8);
	const mdir = `models/${sha8}`;
	if (snap.versions.model_sha256.slice(0, 8) !== sha8) fail(`snapshot_id ne finit pas par le sha8 de versions.model_sha256 (${snap.versions.model_sha256.slice(0, 8)})`);
	const meta = validate(`${mdir}/meta.json`, 'meta');
	const mman = validate(`${mdir}/model-manifest.json`, 'model-manifest');
	const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
	for (const [label, v] of [['publish-manifest', manifest.versions], ['latest', latest?.versions], ['meta', meta?.versions], ['model-manifest', mman?.versions]]) if (v && !same(v, snap.versions)) fail(`versions de ${label} ≠ versions de snapshot`);
	if (mman && mman.model_sha8 !== sha8) fail(`model-manifest.model_sha8 ${mman.model_sha8} ≠ ${sha8}`);
	if (latest && latest.snapshot_id !== snap.snapshot_id) fail('snapshot_id différent entre latest et snapshot');
	if (manifest.snapshot_id !== snap.snapshot_id) fail('snapshot_id différent entre publish-manifest et snapshot');
	if (latest && latest.data_status !== manifest.data_status) fail('data_status différent entre latest et publish-manifest');

	// 3. sha256 et tailles des deux manifestes ; tout fichier publié est listé ; rien du modèle dans l'instantané.
	const checkFiles = (label, base, files) => {
		for (const f of files) {
			const p = path.join(dir, base, f.path);
			if (!fs.existsSync(p)) { fail(`${label} : ${f.path} absent`); continue; }
			const b = fs.readFileSync(p);
			if (b.length !== f.bytes) fail(`${label} : ${f.path} : taille ${b.length} ≠ ${f.bytes}`);
			if (sha256(b) !== f.sha256) fail(`${label} : ${f.path} : sha256 différent`);
		}
	};
	checkFiles('publish-manifest', '', manifest.files);
	if (mman) checkFiles('model-manifest', mdir, mman.files);
	const listed = new Set(manifest.files.map((f) => f.path));
	for (const p of ['meta.json', 'terrain.f32']) if (listed.has(p)) fail(`publish-manifest contient ${p} (fichier du modèle)`);
	for (const p of walk(dir)) if (!p.startsWith('models/') && !NOT_PUBLISHED.has(p) && !listed.has(p)) fail(`${p} présent mais absent de publish-manifest`);
	if (mman) for (const p of ['meta.json', 'terrain.f32']) if (!mman.files.some((f) => f.path === p)) fail(`model-manifest ne liste pas ${p}`);

	if (!meta) { total += n; console.error(`✗ ${name} : invariants des champs non vérifiés (meta illisible)`); return; }

	// 4. Grille et relief.
	const g = meta.grid, { nx, ny, nz } = g, NN = nx * ny, levels = g.levels_asl_m;
	if (levels.length !== nz) fail(`levels_asl_m (${levels.length}) ≠ nz (${nz})`);
	if (levels.some((z, k) => k && z <= levels[k - 1])) fail('levels_asl_m non strictement croissants');
	if (Math.abs(meta.domain.x_max_m - (g.x0_m + (nx - 1) * g.dx_m)) > 1e-6) fail(`domain.x_max_m ${meta.domain.x_max_m} ≠ x0 + (nx−1)·dx`);
	if (Math.abs(meta.domain.y_max_m - (g.y0_m + (ny - 1) * g.dy_m)) > 1e-6) fail(`domain.y_max_m ${meta.domain.y_max_m} ≠ y0 + (ny−1)·dy`);
	if (!meta.terrain.href.endsWith(sha8)) fail('terrain.href ne porte pas le sha8 du modèle');
	if (Math.abs(meta.georef.m_per_deg_lon - 111320 * Math.cos(meta.georef.lat0 * Math.PI / 180)) > 0.01) fail('georef.m_per_deg_lon ≠ 111320·cos(lat0)');
	if (!same([...meta.fields].sort(), Object.keys(meta.encoding).sort())) fail('encoding ne décrit pas exactement meta.fields');
	const tb = fs.readFileSync(path.join(dir, mdir, 'terrain.f32'));
	if (tb.length !== NN * 4) fail(`terrain.f32 : ${tb.length} octets, attendu ${NN * 4}`);
	if (sha256(tb) !== meta.terrain.sha256 || tb.length !== meta.terrain.bytes) fail('terrain.f32 : sha256 ou taille ≠ meta.terrain');
	const terrain = new Float32Array(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.length));
	if (terrain.some(Number.isNaN)) fail('terrain.f32 contient des NaN');
	const below = (k, idx) => levels[k] - terrain[idx] < g.min_agl_m;
	let valid = 0; const byLevel = [];
	for (let k = 0; k < nz; k++) { let c = 0; for (let idx = 0; idx < NN; idx++) if (!below(k, idx)) c++; valid += c; byLevel.push(c / NN); }
	if (valid !== g.valid_nodes) fail(`grid.valid_nodes ${g.valid_nodes} ≠ ${valid} recomptés avec min_agl_m`);
	if (byLevel.some((f, k) => Math.abs(f - g.valid_fraction_by_level[k]) > 1e-3)) fail('valid_fraction_by_level ≠ recompte');

	// 5. Champs : un fichier par variable, tous niveaux ; règle NaN / 255 ; status ∈ {0, 1}.
	const F = {};
	for (const v of meta.fields) guard(`fields/${v}`, () => {
		const enc = meta.encoding[v], u8 = enc.dtype === 'uint8';
		const b = fs.readFileSync(path.join(dir, enc.file));
		const want = nz * NN * (u8 ? 1 : 4);
		if (b.length !== want) { fail(`${enc.file} : ${b.length} octets, attendu ${want} (nz·ny·nx·${u8 ? 1 : 4})`); return; }
		const a = u8 ? new Uint8Array(b.buffer, b.byteOffset, b.length) : new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
		F[v] = a;
		const absent = u8 ? (x) => x === 255 : Number.isNaN;
		outer: for (let k = 0; k < nz; k++) for (let idx = 0; idx < NN; idx++) {
			const x = a[k * NN + idx];
			if (below(k, idx) !== absent(x)) { fail(`${enc.file}[k=${k}, j=${Math.floor(idx / nx)}, i=${idx % nx}] : ${u8 ? 'valeur absente (255)' : 'NaN'} incohérent avec z − relief < min_agl_m`); break outer; }
			if (u8 && x !== 255 && (v === 'status' ? x > 1 : x > 254)) { fail(`${enc.file} : valeur ${x} hors codage`); break outer; }
		}
	});
	if (F.u && F.v && F.speed) for (let at = 0; at < F.speed.length; at++) if (!Number.isNaN(F.speed[at]) && Math.abs(Math.hypot(F.u[at], F.v[at]) - F.speed[at]) > 1e-3) { fail(`fields/speed[${at}] ≠ ‖(u,v)‖`); break; }

	// 6. Points déclarés et profils précalculés.
	const ids = snap.points.map((p) => p.id);
	if (new Set(ids).size !== ids.length) fail('snapshot.points : identifiants en double');
	for (const p of snap.points) {
		if (p.profile && !doc[`profiles/${p.id}.json`]) fail(`point ${p.id} : profile = true mais profiles/${p.id}.json absent`);
		if (p.windrose && !doc[`windrose/${p.id}.json`]) fail(`point ${p.id} : windrose = true mais windrose/${p.id}.json absent`);
	}
	for (const f of profiles) guard(`profiles/${f}`, () => {
		const pr = doc[`profiles/${f}`]; if (!pr) return;
		const where = `profiles/${f}`;
		if (`${pr.point_id}.json` !== f) fail(`${where} : point_id ${pr.point_id} ≠ nom du fichier`);
		if (pr.i >= nx || pr.j >= ny) { fail(`${where} : nœud (${pr.i}, ${pr.j}) hors grille`); return; }
		if (Math.abs(pr.x_m - (g.x0_m + pr.i * g.dx_m)) > 1e-6 || Math.abs(pr.y_m - (g.y0_m + pr.j * g.dy_m)) > 1e-6) fail(`${where} : (x_m, y_m) ≠ nœud (x0 + i·dx, y0 + j·dy)`);
		const idx = pr.j * nx + pr.i;
		if (Math.abs(pr.z_terrain_m - terrain[idx]) > 0.01) fail(`${where} : z_terrain_m ${pr.z_terrain_m} ≠ terrain[j·nx+i] ${terrain[idx].toFixed(3)}`);
		if (pr.levels.length !== nz) fail(`${where} : ${pr.levels.length} niveaux, attendu nz = ${nz}`);
		pr.levels.forEach((l, k) => {
			const at = k * NN + idx;
			if (l.z_asl_m !== levels[k]) fail(`${where} niveau ${k} : z_asl_m ${l.z_asl_m} ≠ ${levels[k]}`);
			if (l.below_terrain !== below(k, idx)) fail(`${where} z=${l.z_asl_m} : below_terrain incohérent avec min_agl_m`);
			if (l.below_terrain) { if (l.speed !== null || l.u !== null) fail(`${where} z=${l.z_asl_m} : valeurs non null sous min_agl_m`); return; }
			for (const v of ['u', 'v', 'w', 'speed', 'sigma_speed']) if (F[v] && Math.abs(l[v] - F[v][at]) > 2e-3) fail(`${where} z=${l.z_asl_m} : ${v} ${l[v]} ≠ champ publié ${F[v][at].toFixed(3)}`);
			const s = Math.hypot(l.u, l.v);
			if (s < 0.2) { if (l.dir_from_deg !== null) fail(`${where} z=${l.z_asl_m} : dir_from_deg doit être null (vitesse < 0,2)`); }
			else if (l.dir_from_deg === null || angleDiff(dirFrom(l.u, l.v), l.dir_from_deg) > 0.5) fail(`${where} z=${l.z_asl_m} : dir_from_deg ${l.dir_from_deg} ≠ ${dirFrom(l.u, l.v).toFixed(2)} recalculé depuis (u, v)`);
			const iv = l.speed_interval;
			if (iv.p50 !== l.speed) fail(`${where} z=${l.z_asl_m} : p50 ≠ speed`);
			if (Math.abs(iv.p10 - Math.max(0, l.speed - 1.2816 * l.sigma_speed)) > 0.01 || Math.abs(iv.p90 - (l.speed + 1.2816 * l.sigma_speed)) > 0.01) fail(`${where} z=${l.z_asl_m} : intervalle ≠ speed ± 1,2816·sigma_speed`);
		});
	});

	// 7. Roses des vents.
	for (const f of roses) {
		const w = doc[`windrose/${f}`]; if (!w) continue;
		const where = `windrose/${f}`, ns = w.sectors.length;
		if (`${w.site_id}.json` !== f) fail(`${where} : site_id ≠ nom du fichier`);
		const sum = w.sectors.reduce((a, s) => a + s.frequency, 0);
		if (Math.abs(sum - 1) > 0.01) fail(`${where} : somme des fréquences ${sum.toFixed(4)} ≠ 1 ± 0,01`);
		if (Math.abs(w.sector_width_deg * ns - 360) > 1e-6) fail(`${where} : ${ns} secteurs de ${w.sector_width_deg}° ≠ 360°`);
		w.sectors.forEach((s, k) => {
			if (Math.abs(s.dir_center_deg - k * w.sector_width_deg) > 1e-6) fail(`${where} secteur ${k} : centre ${s.dir_center_deg}° ≠ ${k * w.sector_width_deg}° (secteurs de largeur égale)`);
			if (s.speed_bins.length !== w.speed_bin_edges_ms.length - 1) fail(`${where} secteur ${k} : nombre de classes de vitesse incohérent`);
		});
	}

	// 8. Alertes et open_alerts.
	const alerts = doc['alerts.json']?.items ?? [];
	for (const a of alerts) if (sha256(`${a.kind}|${a.target}`).slice(0, 8) !== a.id) fail(`alerte ${a.id} : id ≠ sha8(kind|target)`);
	if (new Set(alerts.map((a) => a.id)).size !== alerts.length) fail('alertes : identifiants en double');
	const openAl = alerts.filter((a) => a.status === 'ouverte');
	if (snap.kpi.open_alerts) for (const lv of ['information', 'avertissement', 'important', 'critique']) {
		const c = openAl.filter((a) => a.level === lv).length;
		if (snap.kpi.open_alerts[lv] !== c) fail(`kpi.open_alerts.${lv} ${snap.kpi.open_alerts[lv]} ≠ ${c} alertes ouvertes`);
	}
	for (const it of doc['times.json']?.items ?? []) if (it.snapshot_id === snap.snapshot_id && it.open_alerts !== null && it.open_alerts !== undefined && it.open_alerts !== openAl.length) fail(`times.open_alerts ${it.open_alerts} ≠ ${openAl.length} alertes ouvertes`);

	// 9. Réseau.
	const st = doc['stations.json']?.items ?? [], net = snap.network;
	if (net.sensors_visible > net.sensors_total) fail('network.sensors_visible > sensors_total');
	if (net.kind === 'treillis_synthetique') {
		for (const s of st) {
			if (s.state === 'active') fail(`stations : ${s.id} est active (le treillis ne liste que les capteurs signalés)`);
			if (!/^c\d+_\d+$/.test(s.id)) fail(`stations : identifiant ${s.id} ≠ c<i>_<j>`);
		}
		const lost = st.filter((s) => s.state === 'muette' || s.state === 'exclue').length;
		if (net.sensors_total - net.sensors_visible !== lost) fail(`network : sensors_total − sensors_visible ≠ ${lost} capteurs muets ou exclus`);
	} else if (net.sensors_total !== st.length) fail(`network.sensors_total ${net.sensors_total} ≠ ${st.length} stations listées`);

	total += n;
	console.log(n ? `✗ ${name} : ${n} écart(s)` : `✓ ${name}`);
}

const args = process.argv.slice(2);
if (!args.length && !fs.existsSync(path.join(FIXTURES, 'normal', 'snapshot.json'))) {
	console.error('Instantanés de test absents : lance d’abord « node tools/make-fixtures.mjs ».');
	process.exit(1);
}
const dirs = args.length ? args : fs.readdirSync(FIXTURES).map((d) => path.join(FIXTURES, d)).filter((d) => fs.statSync(d).isDirectory());
for (const d of dirs) checkSnapshot(d);
if (total) { console.error(`${total} écart(s) au contrat`); process.exit(1); }
console.log('Contrat v1 : tous les instantanés sont conformes.');
