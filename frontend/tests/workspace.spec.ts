import { test, expect, type Page } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { start as demarrer, admin, ADMIN_EMAIL, ADMIN_MDP } from './helpers';

const directory='output/redesign';
const views=['tableau','monvent','carte3d','agri','eolien','outils','fiabilite','historique','alertes','admin'];
const start=(page:Page)=>demarrer(page,'monvent');
async function route(page:Page,view:string,mobile=false) {
  if(mobile)await page.getByRole('button',{name:'Ouvrir la navigation',exact:true}).click();
  await page.locator(`nav.tabs a[data-v="${view}"]`).click();
  await expect(page.locator(`#v-${view}`)).toHaveClass(/on/);
  if(view==='carte3d')await expect(page.locator('#terrain-loading')).toBeHidden();
  await page.waitForTimeout(350);
  await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
}

for(const viewport of [{width:1440,height:1050},{width:390,height:844}]) {
  test(`all decision-support screens at ${viewport.width}px`,async({page})=>{
    await page.setViewportSize(viewport);
    const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
    await start(page);await admin(page);await mkdir(directory,{recursive:true});
    for(const view of views) {
      await route(page,view,viewport.width<700);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1);
      expect(overflow,`${view} must not overflow at ${viewport.width}px`).toBe(false);
      await page.screenshot({path:`${directory}/${viewport.width}-${view}.png`,fullPage:true});
    }
    expect(errors).toEqual([]);
  });
}

test('location, activities, units, timeline, inspection, and search stay connected',async({page})=>{
  await page.setViewportSize({width:1440,height:1000});await start(page);
  await page.locator('#town').selectOption('10');
  await expect(page.locator('.readout-location b')).toHaveText('Nabeul');
  await page.locator('.workspace-preferences summary').click();await page.locator('#u-ms').click();
  await expect(page.locator('.readout-metric strong')).toContainText('m/s');
  await page.locator('#th-dark').click();await expect(page.locator('html')).toHaveAttribute('data-theme','dark');
  await page.screenshot({path:`${directory}/1440-dark.png`,fullPage:true});
  await page.locator('#th-light').click();await page.locator('#u-kmh').click();await page.locator('.workspace-preferences summary').click();
  await page.locator('#hour').fill('8');await page.locator('#hour').dispatchEvent('input');
  await expect(page.locator('#hourtxt')).toHaveText('8 h');
  await page.locator('.act[data-a="irrig"]').count().then(async n=>{if(n)await page.locator('.act[data-a="irrig"]').click();else await page.locator('.act').nth(1).click();});
  await expect(page.locator('#answer')).not.toBeEmpty();
  await page.locator('.inspect-button').click();await expect(page.locator('#sheet')).toBeVisible();await expect(page.locator('#sheet svg[role="img"]')).toBeVisible();await page.locator('#sheet .x').click();
  await page.keyboard.press('Control+k');await page.locator('#pal-in').fill('La Marsa');await page.locator('#palette li[data-i]').first().click();
  await expect(page.locator('.readout-location b')).toHaveText('La Marsa');
  await page.locator('#c1').focus();await page.keyboard.press('ArrowRight');await expect(page.locator('#answer')).not.toBeEmpty();
});

test('roles, agriculture, energy, and local alerts retain their controls',async({page})=>{
  await start(page);
  await expect(page.locator('a[data-v="admin"]')).toBeHidden();
  await page.evaluate(()=>window.ATOM.workspace!.navigate('admin'));
  await expect(page.locator('#authm')).toBeVisible();await expect(page.locator('#v-admin')).not.toHaveClass(/on/);
  await page.locator('#authm .x, #a-close').first().click();await admin(page);
  await route(page,'agri');await page.locator('#ag-crop').selectOption('olivier');await page.locator('#ag-area').fill('3');await page.locator('#ag-area').dispatchEvent('change');
  await page.locator('#ag-day button').nth(1).click();await expect(page.locator('#ag-cards')).not.toBeEmpty();
  await route(page,'eolien');await page.locator('#tm').selectOption('s');await expect(page.locator('#rotor svg')).toBeVisible();await expect(page.locator('#prodchart svg')).toBeVisible();
  // alerte SMS du compte, gardée sur le serveur, puis supprimée
  await route(page,'alertes');await page.locator('#al-tel').fill('+21698765432');await page.locator('#al-add').click();
  await expect(page.locator('#al-mine .it')).toHaveCount(1);await page.locator('#al-mine [data-del]').first().click();await expect(page.locator('#al-mine .it')).toHaveCount(0);
  await route(page,'tableau');const h0=parseInt(await page.locator('#hourtxt').innerText());await page.locator('#tbday svg').focus();await page.keyboard.press(h0<23?'ArrowRight':'ArrowLeft');await expect(page.locator('#hourtxt')).toHaveText(`${h0<23?h0+1:h0-1} h`);
});

test('3D picking, cameras, layers, static rendering, and resource stability',async({page})=>{
  await page.setViewportSize({width:1440,height:1050});await start(page);await route(page,'carte3d');
  await page.locator('#terrain-mode-analysis').click();await page.waitForFunction(()=>window.ATOM?.sceneStats);
  const before=await page.evaluate(()=>({time:performance.now(),...window.ATOM.sceneStats()}));
  await page.waitForTimeout(3000);
  const after=await page.evaluate(()=>({time:performance.now(),...window.ATOM.sceneStats()}));
  const measuredFps=(after.frames-before.frames)*1000/(after.time-before.time);
  await writeFile(`${directory}/performance.json`,JSON.stringify({viewport:'1440×1050',browser:'Chrome headless, local machine',measuredFps,renderer:after},null,2));
  expect(after.frames).toBeGreaterThan(before.frames);
  await page.locator('[data-camera="top"]').click();await expect(page.locator('[data-camera="top"]')).toHaveAttribute('aria-pressed','true');
  const canvas=page.locator('#three>canvas');const rect=await canvas.boundingBox();
  await canvas.click({position:{x:rect!.width*.5,y:rect!.height*.5}});
  await expect(page.locator('#sheet')).toBeVisible();await expect(page.locator('#sheet')).toContainText('Point');await page.locator('#sheet .x').click();
  await page.locator('#aw3d summary').click();await page.locator('#aw3d-coul').selectOption('incertitude');
  await expect(page.locator('#v-carte3d .hslider .legend')).toContainText('distance aux stations');
  await page.locator('input[data-c="horizontale"]').check();await page.locator('#h3').fill('4');await page.locator('#h3').dispatchEvent('input');
  await page.waitForTimeout(500);await page.locator('#aw3d summary').click();
  await page.locator('#scene-pause').click();await page.mouse.move(10,10);await page.waitForTimeout(1000);
  const paused=await page.evaluate(()=>window.ATOM.sceneStats().frames);await page.waitForTimeout(600);expect(await page.evaluate(()=>window.ATOM.sceneStats().frames)).toBe(paused);
  await route(page,'monvent');const hidden=await page.evaluate(()=>window.ATOM.sceneStats().frames);await page.waitForTimeout(600);expect(await page.evaluate(()=>window.ATOM.sceneStats().frames)).toBe(hidden);
});

test('scientific field, vertical profile, DSO, integration, and export viewer',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  // outil interne : une vraie clé (session de l'administrateur de test), jamais d'accès anonyme
  const r=await page.request.post('/api/comptes/connexion',{data:{email:ADMIN_EMAIL,mot_de_passe:ADMIN_MDP}});expect(r.ok()).toBe(true);
  const cle=(await r.json()).cle;await page.addInitScript(k=>localStorage.setItem('pcfno3d.apikey',k),cle);
  await page.goto('/app/');await expect(page.locator('#app')).toBeVisible();await expect(page.locator('#instant option')).not.toHaveCount(0);
  await page.locator('#map').click({position:{x:200,y:100}});await expect(page.locator('#pointInfo')).not.toBeEmpty();
  await page.locator('[data-tab="point"]').click();await expect(page.locator('#profile')).toBeVisible();await page.screenshot({path:`${directory}/science-profile.png`,fullPage:true});
  await page.locator('[data-tab="qualite"]').click();await expect(page.locator('#dso')).not.toBeEmpty();await page.screenshot({path:`${directory}/science-quality.png`,fullPage:true});
  await page.locator('[data-tab="integration"]').click();await expect(page.locator('#mcpUrl')).toContainText('/mcp');
  // l'export est produit par le service (portée admin), puis lu par la visionneuse
  const ex=await page.request.post('/api/reconstruct',{data:{stem:'006_000000',export:true},headers:{'X-API-Key':cle}});expect(ex.ok()).toBe(true);
  await page.goto('/app/export/?src=/api/export/006_000000');await expect(page.locator('#status')).not.toContainText('introuvable');
  await expect(page.locator('#info')).toContainText('Grille');await page.locator('#variable').selectOption('u');await page.screenshot({path:`${directory}/export-viewer.png`,fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.goto('/app/');await expect(page.locator('#app')).toBeVisible();expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);await page.screenshot({path:`${directory}/science-mobile.png`,fullPage:true});
  expect(errors).toEqual([]);
});

test('reduced motion leaves a static scene until the user interacts',async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});await start(page);await route(page,'carte3d');await page.mouse.move(1,1);await page.waitForTimeout(1000);
  await page.locator('#terrain-mode-analysis').click();await page.waitForFunction(()=>window.ATOM?.sceneStats);await page.waitForTimeout(1000);
  const before=await page.evaluate(()=>window.ATOM.sceneStats().frames);await page.waitForTimeout(600);expect(await page.evaluate(()=>window.ATOM.sceneStats().frames)).toBe(before);
  await expect(page.locator('#scene-pause')).toHaveText('Animer le vent');
  await page.locator('[data-camera="profile"]').click();await expect(page.locator('[data-camera="profile"]')).toHaveAttribute('aria-pressed','true');
});

test('English and Arabic keep the dashboard connected and the mobile navigation contained',async({browser})=>{
  for(const language of ['en','ar']) {
    const context=await browser.newContext({viewport:{width:390,height:844}});
    await context.addInitScript(lang=>{localStorage.setItem('aw-langue',lang);localStorage.setItem('aw-tour','1');localStorage.setItem('aw-theme',JSON.stringify('light'));},language);
    const page=await context.newPage();const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto('/app/atom-wind/');await page.waitForFunction(()=>window.ATOM?.workspace&&!document.getElementById('aw-chargement'),null,{timeout:240000});
    await page.evaluate(()=>window.ATOM.workspace!.navigate('tableau'));
    await expect(page.locator('html')).toHaveAttribute('lang',language);
    await expect(page.locator('.page-heading h1')).toHaveText(language==='en'?'Overview':'نظرة عامة');
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
    await page.locator('.mobile-menu').click();await expect(page.locator('.workspace-rail')).toBeVisible();await page.keyboard.press('Escape');await expect(page.locator('.workspace-rail')).toBeHidden();
    expect(errors).toEqual([]);await context.close();
  }
});
