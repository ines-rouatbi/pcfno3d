import { test, expect, type Page } from '@playwright/test';

async function startTerrain(page:Page,source:'local'|'national'='local') {
  await page.addInitScript(()=>{localStorage.setItem('aw-tour','1');localStorage.setItem('aw-theme',JSON.stringify('light'));});
  await page.goto('/app/atom-wind/#carte3d');
  await page.waitForFunction(()=>window.ATOM?.terrainStats?.().ready,{timeout:45000});
  await expect(page.locator('#terrain-loading')).toBeHidden({timeout:45000});
  await expect(page.locator('#terrain-host canvas')).toBeVisible();
  if(source==='local')await page.locator('#terrain-source').selectOption('local');
  await page.waitForFunction(()=>window.ATOM?.terrainStats?.().windReady,{timeout:45000});
}

test('the supplied terrain streams locally, keeps LOD parents exclusive, and releases detail',async({page})=>{
  await page.setViewportSize({width:1440,height:1050});
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await startTerrain(page);
  const initial=await page.evaluate(()=>window.ATOM.terrainStats());
  expect(initial.windEnabled).toBe(true);expect(initial.windVectors).toBeGreaterThan(150);expect(initial.waterSamples).toBeGreaterThan(0);
  await page.locator('#terrain-country').click();
  await expect.poll(()=>page.evaluate(()=>window.ATOM.terrainStats().resident)).toBe(20);
  await page.screenshot({path:'output/redesign/terrain-tunisia.png',fullPage:true});
  await page.evaluate(()=>window.ATOM.terrainExplore.focus(10.29,36.68,14000));
  await expect(page.locator('#terrain-detail')).toContainText('32 m',{timeout:45000});
  const detailed=await page.evaluate(()=>window.ATOM.terrainStats());
  expect(detailed.resident).toBeLessThanOrEqual(80);expect(detailed.requests-initial.requests).toBeLessThanOrEqual(60);
  const ids=new Set<string>(detailed.activeIds);
  for(const id of ids){const [z,x,y]=id.split('-').map(Number);if(z===7)expect(ids.has(`5-${Math.floor(x/4)}-${Math.floor(y/4)}`)).toBe(false);if(z>3)expect(ids.has(`3-${Math.floor(x/2**(z-3))}-${Math.floor(y/2**(z-3))}`)).toBe(false);}
  await page.screenshot({path:'output/redesign/terrain-grand-tunis.png',fullPage:true});
  await page.locator('#terrain-country').click();
  await expect.poll(()=>page.evaluate(()=>window.ATOM.terrainStats().resident)).toBe(20);
  await page.waitForTimeout(500);const frames=await page.evaluate(()=>window.ATOM.terrainStats().frames);
  await page.waitForTimeout(500);expect(await page.evaluate(()=>window.ATOM.terrainStats().frames)).toBe(frames);
  expect(errors).toEqual([]);
});

test('national forecasts cover southern Tunisia and stay distinct from the local model',async({page})=>{
  await page.setViewportSize({width:1440,height:1050});await startTerrain(page,'national');
  await expect(page.locator('#terrain-source')).toHaveValue('national');
  await expect(page.locator('.rail-territory b')).toHaveText('Tunisie');
  await expect(page.locator('.breadcrumb > span').first()).toHaveText('Tunisie');
  await expect(page.locator('#terrain-live-meta')).toContainText('GFS');
  const stats=await page.evaluate(()=>window.ATOM.terrainStats());
  expect(stats.forecastLocations).toBe(170);expect(stats.windSource).toBe('national');expect(stats.windVectors).toBeGreaterThan(100);
  expect(Math.abs(Date.now()-stats.forecastTime*1000)).toBeLessThan(2*3600000);
  for(const [lon,lat] of [[10.76,34.74],[8.13,33.92],[10.45,32.93]])expect(await page.evaluate(([x,y])=>window.ATOM.terrainExplore.sample(x,y),[lon,lat])).not.toBeNull();
  await page.evaluate(()=>window.ATOM.terrainExplore.select(8.13,33.92));
  await expect(page.locator('#terrain-wind-reading')).toContainText('km/h');
  await expect(page.locator('#terrain-point-source')).toContainText('NOAA GFS');
  await expect(page.locator('#terrain-open-point')).toBeHidden();
  await page.locator('#terrain-close-point').click();
  await page.locator('#terrain-motion').click();
  await page.screenshot({path:'output/redesign/terrain-national-forecast.png',fullPage:true});
  await page.locator('#terrain-day').selectOption('1');
  await page.locator('#terrain-hour').fill('14');await page.locator('#terrain-hour').dispatchEvent('input');
  const changed=await page.evaluate(()=>window.ATOM.terrainStats());expect(changed.forecastTime).toBeGreaterThan(stats.forecastTime);
  await page.locator('#terrain-source').selectOption('local');
  await expect(page.locator('#terrain-live-meta')).toContainText('Démonstration locale');
  await expect(page.locator('.rail-territory b')).toHaveText('Grand Tunis');
  await page.waitForFunction(()=>window.ATOM.terrainStats().windReady);
  expect(await page.evaluate(()=>window.ATOM.terrainExplore.sample(8.13,33.92))).toBeNull();
  await page.locator('#terrain-source').selectOption('national');
  await expect(page.locator('.rail-territory b')).toHaveText('Tunisie');
  await page.locator('#terrain-mode-analysis').click();
  await expect(page.locator('.rail-territory b')).toHaveText('Grand Tunis');
  await page.locator('#terrain-mode-map').click();
  await expect(page.locator('.rail-territory b')).toHaveText('Tunisie');
});

test('wind remains within its domain and selection opens the same geographic point',async({page})=>{
  await page.setViewportSize({width:1440,height:1050});await startTerrain(page);
  expect(await page.evaluate(()=>window.ATOM.workspace!.terrainWind!(9,34,80))).toBeNull();
  await expect(page.locator('#terrain-wind')).toHaveAttribute('aria-pressed','true');
  await expect.poll(()=>page.evaluate(()=>window.ATOM.terrainStats().windVectors),{timeout:45000}).toBeGreaterThan(30);
  await expect(page.locator('#terrain-wind-source')).toContainText(/démonstration/i);
  await page.evaluate(()=>window.ATOM.terrainExplore.select(10.29,36.68));
  await expect(page.locator('#terrain-elevation')).toContainText('Sol DEM');
  await expect(page.locator('#terrain-wind-reading')).toContainText('80 m du sol');
  await expect(page.locator('#terrain-detail')).toContainText('128 m');
  const canvas=page.locator('#terrain-host canvas');
  const screenshotOptions={style:'.terrain-host > :not(canvas) { visibility: hidden !important; }'};
  const movingBefore=await canvas.screenshot(screenshotOptions);
  await page.waitForTimeout(350);
  const movingAfter=await canvas.screenshot(screenshotOptions);
  expect(movingBefore.equals(movingAfter),'Wind must visibly move in the rendered canvas').toBe(false);
  await page.locator('#terrain-height').selectOption('120');
  await expect(page.locator('#terrain-wind-reading')).toContainText('120 m du sol');
  const before=await page.evaluate(()=>window.ATOM.terrainStats().flowTime);
  await expect.poll(()=>page.evaluate(()=>window.ATOM.terrainStats().flowTime)).toBeGreaterThan(before);
  await page.locator('#terrain-motion').click();
  await expect(page.locator('#terrain-motion')).toHaveAttribute('aria-pressed','false');
  await page.waitForTimeout(200);const paused=await page.evaluate(()=>window.ATOM.terrainStats().flowTime);
  await page.waitForTimeout(350);expect(await page.evaluate(()=>window.ATOM.terrainStats().flowTime)).toBe(paused);
  const visibleWind=await canvas.screenshot(screenshotOptions);
  await page.locator('#terrain-wind').click();
  const noWind=await canvas.screenshot(screenshotOptions);
  expect(visibleWind.equals(noWind),'The enabled layer must change visible pixels, not only statistics').toBe(false);
  await page.locator('#terrain-wind').click();
  await page.locator('#terrain-hour').fill('7');await page.locator('#terrain-hour').dispatchEvent('input');
  await expect(page.locator('#hour')).toHaveValue('7');
  await expect(page.locator('#terrain-native-hour')).toHaveText('7 h');
  await page.screenshot({path:'output/redesign/terrain-wind-overlay.png',fullPage:true});
  await page.locator('#terrain-open-point').click();
  await expect(page.locator('#v-monvent')).toHaveClass(/on/);
  await expect(page.locator('.readout-location b')).toContainText('Mornag');
});

test('terrain works on a phone and stops drawing when hidden',async({page})=>{
  await page.setViewportSize({width:390,height:844});await startTerrain(page);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.locator('#terrain-host').focus();await page.keyboard.press('ArrowRight');
  const canvas=page.locator('#terrain-host canvas');const box=await canvas.boundingBox();
  await canvas.click({position:{x:box!.width*.5,y:box!.height*.5}});
  await expect(page.locator('#terrain-selection')).toBeVisible();
  await page.evaluate(()=>{(document.activeElement as HTMLElement)?.blur();window.scrollTo({top:0,behavior:'instant'});});
  await page.screenshot({path:'output/redesign/terrain-mobile.png',fullPage:true});
  await page.locator('.mobile-menu').click();await page.locator('a[data-v="monvent"]').click();
  await page.waitForTimeout(500);const frames=await page.evaluate(()=>window.ATOM.terrainStats().frames);
  await page.waitForTimeout(500);expect(await page.evaluate(()=>window.ATOM.terrainStats().frames)).toBe(frames);
});

test('national forecasts stay nationwide when a phone focuses on Grand Tunis',async({page})=>{
  await page.setViewportSize({width:390,height:844});await startTerrain(page,'national');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await expect(page.locator('#terrain-country')).toHaveAttribute('aria-pressed','true');
  await page.screenshot({path:'output/redesign/terrain-national-mobile.png',fullPage:true});
  await page.locator('#terrain-local').click();
  await expect(page.locator('#terrain-local')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#terrain-source')).toHaveValue('national');
  expect(await page.evaluate(()=>window.ATOM.terrainExplore.sample(10.45,32.93))).not.toBeNull();
  await page.locator('#terrain-country').click();
  await expect(page.locator('#terrain-country')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#terrain-return-wind')).toBeHidden();
});

test('a failed terrain download can be retried without leaving duplicate canvases',async({page})=>{
  await page.route('**/terrain/tunisia/overview.glb',route=>route.abort());
  await page.addInitScript(()=>localStorage.setItem('aw-tour','1'));
  await page.goto('/app/atom-wind/#carte3d');
  await expect(page.locator('#terrain-retry')).toBeVisible();
  await expect(page.locator('#terrain-status')).toContainText('n’a pas pu');
  await page.unroute('**/terrain/tunisia/overview.glb');
  await page.locator('#terrain-retry').click();
  await expect(page.locator('#terrain-loading')).toBeHidden();
  await expect(page.locator('#terrain-host canvas')).toHaveCount(1);
});

test('wind also loads when the server supplies HTTP gzip encoding, as Vite does',async({page})=>{
  await page.route('**/elevation/*.bin.gz',async route=>{
    const response=await route.fetch();
    await route.fulfill({response,headers:{...response.headers(),'content-encoding':'gzip'},body:await response.body()});
  });
  await startTerrain(page);
  await expect(page.locator('#terrain-flow-state')).toBeEmpty();
  expect(await page.evaluate(()=>window.ATOM.terrainStats().windVectors)).toBeGreaterThan(100);
});

test('reduced motion keeps the native wind visible and still until requested',async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});await startTerrain(page);
  await expect(page.locator('#terrain-motion')).toHaveAttribute('aria-pressed','false');
  expect(await page.evaluate(()=>window.ATOM.terrainStats().windVectors)).toBeGreaterThan(100);
  await page.waitForTimeout(250);const still=await page.evaluate(()=>window.ATOM.terrainStats().flowTime);
  await page.waitForTimeout(350);expect(await page.evaluate(()=>window.ATOM.terrainStats().flowTime)).toBe(still);
  await page.locator('#terrain-motion').click();
  await expect.poll(()=>page.evaluate(()=>window.ATOM.terrainStats().flowTime)).toBeGreaterThan(still);
});
