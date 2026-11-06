import { test, expect, type Page } from '@playwright/test';

async function dashboard(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem('aw-tour', '1');
    localStorage.setItem('aw-theme', JSON.stringify('light'));
  });
  await page.goto('/app/atom-wind/');
  await page.waitForFunction(() => window.ATOM?.workspace);
  await page.locator('#b-login').click();
  await page.locator('[data-demo="admin"]').click();
  await expect(page.locator('#toast')).toBeHidden();
  await page.evaluate(() => window.ATOM.workspace!.navigate('tableau'));
}

test('territory selection carries the place into local conditions and agriculture', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await dashboard(page);
  await page.locator('#tbmap [data-place="7"]').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#tb-place')).toHaveValue('7');
  await expect(page.locator('#tbmap [data-place="7"]')).toBeFocused();
  const speed = await page.locator('.point-reading strong').innerText();
  await page.locator('#tb-profile').click();
  await expect(page.locator('#sheet')).toContainText('Soliman');
  await expect(page.locator('#sheet h2')).toContainText(speed.replace(/\s+/g, ' ').trim());
  await page.locator('#sheet .x').click();
  await page.locator('.point-primary').click();
  await expect(page.locator('.readout-location b')).toHaveText('Soliman');
  await expect(page.locator('#town')).toHaveValue('7');
  await page.locator('#town').selectOption('0');
  await page.locator('a[data-v="tableau"]').click();
  await expect(page.locator('#tb-place')).toHaveValue('0');
  await page.locator('#tb-place').selectOption('10');
  await page.locator('[data-context="agri"]').click();
  await expect(page.locator('#v-agri')).toHaveClass(/on/);
  await expect(page.locator('#ag-place')).toHaveValue('10');
  await expect(page).toHaveURL(/#agri$/);
  await page.goBack();
  await expect(page.locator('#v-tableau')).toHaveClass(/on/);
  await expect(page.locator('#tb-place')).toHaveValue('10');
  await page.goForward();
  await expect(page.locator('#v-agri')).toHaveClass(/on/);
  await expect(page.locator('#ag-place')).toHaveValue('10');
  await page.locator('a[data-v="admin"]').click();
  await expect(page.locator('.data-context')).toBeHidden();
});

test('a dashboard alert opens its corresponding detail', async ({ page }) => {
  await dashboard(page);
  const alert = page.locator('[data-tb-alert="1"]');
  const title = await alert.locator('b').innerText();
  await alert.click();
  await expect(page.locator('#v-alertes')).toHaveClass(/on/);
  await expect(page.locator('#alertlist .alert.sel h3')).toHaveText(title);
});

test('tablet navigation contains focus and the overview fits intermediate widths', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await dashboard(page);
  await page.locator('.mobile-menu').click();
  await expect(page.locator('.rail-close')).toBeFocused();
  await page.locator('.rail-help').focus();
  await page.keyboard.press('Tab');
  await expect(page.locator('.rail-brand a')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.locator('.mobile-menu')).toBeFocused();
  for (const width of [768, 900, 1024, 1280]) {
    await page.setViewportSize({ width, height: 1024 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${width}px overflow`).toBe(true);
    await expect(page.locator('#tb-place')).toBeVisible();
  }
});
