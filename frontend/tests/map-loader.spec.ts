import { test,expect,type Page } from '@playwright/test';

async function start(page:Page){
  await page.addInitScript(()=>{localStorage.setItem('aw-tour','1');localStorage.setItem('aw-basemap',JSON.stringify('relief'));});
  await page.goto('/app/atom-wind/#monvent');await page.waitForFunction(()=>window.ATOM?.workspace);
}

test('the real map loads its own local library and survives repeated clicks',async({page})=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  let vendor=false;page.on('request',request=>{if(request.url().includes('/vendor/leaflet/leaflet.js'))vendor=true;});
  await page.route('**/node_modules/.vite/deps/leaflet*',route=>route.abort());
  await page.route('**/*.tile.openstreetmap.org/**',route=>route.abort());
  await start(page);await page.locator('#bm-osm').dblclick();
  await expect(page.locator('#lmap')).toBeVisible();
  await expect(page.locator('#lmap')).toHaveClass(/leaflet-container/);
  await expect(page.locator('#lmap .leaflet-overlay-pane canvas')).toBeVisible();
  expect(vendor).toBe(true);
  await page.locator('#bm-relief').click();await expect(page.locator('#map1')).toBeVisible();
  await page.locator('#bm-osm').click();await expect(page.locator('#lmap .leaflet-map-pane')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('a library failure leaves the relief usable and can be retried',async({page})=>{
  await page.route('**/vendor/leaflet/leaflet.js',route=>route.abort());
  await start(page);await page.locator('#bm-osm').click();
  await expect(page.locator('#bm-relief')).toHaveAttribute('aria-pressed','true');
  await expect(page.locator('#map1')).toBeVisible();
  await page.unroute('**/vendor/leaflet/leaflet.js');
  await page.locator('#bm-osm').click();
  await expect(page.locator('#lmap')).toBeVisible();
  await expect(page.locator('#lmap')).toHaveClass(/leaflet-container/);
});
