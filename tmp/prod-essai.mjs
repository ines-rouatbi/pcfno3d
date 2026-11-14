// Parcours de vérification production (non versionné : tmp/). OUT=dossier node tmp/prod-essai.mjs
import { chromium } from '@playwright/test';

const OUT = process.env.OUT || '.';
const b = await chromium.launch();
const DEMO = /démonstration|demonstration|d['’]exemple|prototype|simulé|maquette/i;
const resultats = [];
const note = (k, v) => { resultats.push([k, v]); console.log(k, '→', v); };

async function page(url, { w = 1440, h = 900 } = {}) {
  const c = await b.newContext({ viewport: { width: w, height: h } });
  await c.addInitScript(() => { localStorage.setItem('aw-tour', '1'); localStorage.setItem('aw-theme', JSON.stringify('light')); });
  const p = await c.newPage(); p.erreurs = [];
  p.on('pageerror', e => p.erreurs.push(String(e)));
  await p.goto(url);
  await p.waitForFunction(() => !document.getElementById('aw-chargement') && window.ATOM?.workspace, null, { timeout: 240000 });
  await p.waitForTimeout(1500);
  return p;
}
const texte = p => p.evaluate(() => document.body.innerText);

// 1. service local au format de production
let p = await page('http://127.0.0.1:8000/app/atom-wind/');
note('etat', JSON.stringify(await p.evaluate(() => window.ATOM.workspace.snapshot().source)));
for (const v of ['tableau', 'monvent', 'carte3d', 'alertes', 'fiabilite']) {
  await p.evaluate(v => window.ATOM.workspace.navigate(v), v); await p.waitForTimeout(1800);
  await p.screenshot({ path: `${OUT}/p_${v}.png`, fullPage: false });
  const t = await texte(p); const m = t.match(DEMO);
  note(`vue ${v}`, m ? `TEXTE DEMO : ${m[0]}` : 'ok');
}
// 2. compte agriculteur, alerte SMS, espace agriculture
await p.evaluate(() => window.ATOM.workspace.navigate('alertes'));
await p.click('#al-login-btn'); await p.click('#t-reg');
const mail = `essai${Date.now()}@exemple.tn`;
await p.fill('#r-name', 'Essai Agri'); await p.fill('#r-mail', mail); await p.fill('#r-pw', 'motdepasse1'); await p.fill('#r-pw2', 'motdepasse1');
await p.selectOption('#r-role', 'agri'); await p.check('#r-ok'); await p.click('#f-reg button[type=submit]');
await p.waitForTimeout(2500);
note('apres inscription', await p.evaluate(() => location.hash));
await p.evaluate(() => window.ATOM.workspace.navigate('agri')); await p.waitForTimeout(2000);
await p.screenshot({ path: `${OUT}/p_agri.png` });
note('vue agri', (await texte(p)).match(DEMO)?.[0] ?? 'ok');
await p.evaluate(() => window.ATOM.workspace.navigate('alertes')); await p.waitForTimeout(800);
await p.fill('#al-tel', '+216 12 345 678'); await p.click('#al-add'); await p.waitForTimeout(2000);
note('mes alertes', (await p.locator('#al-mine').innerText()).replace(/\s+/g, ' ').slice(0, 160));
await p.screenshot({ path: `${OUT}/p_alertes_compte.png` });
// session relue après rechargement
await p.reload(); await p.waitForFunction(() => !document.getElementById('aw-chargement') && window.ATOM?.workspace, null, { timeout: 240000 }); await p.waitForTimeout(2500);
note('session apres rechargement', await p.evaluate(() => window.ATOM.workspace.role()));
note('erreurs JS service local', JSON.stringify(p.erreurs));

// 3. service sans modèle
p = await page('http://127.0.0.1:8001/app/atom-wind/');
note('sans modele', (await p.locator('.service-state h2').innerText().catch(() => 'AUCUN ECRAN')));
await p.screenshot({ path: `${OUT}/p_sans_modele.png` });
note('erreurs JS sans modele', JSON.stringify(p.erreurs));

// 4. rien ne répond
p = await page('http://127.0.0.1:8000/app/atom-wind/?api=http://127.0.0.1:9');
note('hors ligne', (await p.locator('.service-state h2').innerText().catch(() => 'AUCUN ECRAN')));
await p.screenshot({ path: `${OUT}/p_hors_ligne.png` });
note('erreurs JS hors ligne', JSON.stringify(p.erreurs));
p = await page('http://127.0.0.1:8001/app/atom-wind/', { w: 390, h: 844 });
await p.screenshot({ path: `${OUT}/p_sans_modele_mobile.png` });
await b.close();
