import { chromium } from '@playwright/test';

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const suffix = process.argv[2] || '';
try {
  await page.addInitScript(() => {
    localStorage.setItem('aw-tour', '1');
    localStorage.setItem('aw-theme', JSON.stringify('light'));
  });
  await page.goto('http://127.0.0.1:8000/app/atom-wind/');
  await page.waitForFunction(() => window.ATOM?.workspace);
  await page.locator('#b-login').click();
  await page.locator('[data-demo="admin"]').click();
  await page.locator('a[data-v="tableau"]').click();
  await page.locator('#toast').waitFor({ state: 'hidden' });
  await page.evaluate(() => document.fonts.ready);
  for (const [width, height, file] of [
    [1440, 1050, 'dashboard-first-screen'],
    [1024, 1024, 'dashboard-1024'],
    [768, 1024, 'dashboard-tablet'],
    [390, 844, 'dashboard-phone-first-screen'],
  ]) {
    await page.setViewportSize({ width, height });
    await page.screenshot({ path: `output/redesign/${file}${suffix}.png` });
  }
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.locator('.workspace-preferences summary').click();
  await page.locator('#th-dark').click();
  await page.locator('.workspace-preferences summary').click();
  await page.waitForTimeout(250);
  await page.screenshot({ path: `output/redesign/dashboard-dark${suffix}.png` });
} finally {
  await browser.close();
}
