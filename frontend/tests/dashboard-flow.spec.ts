import { test, expect, type Page } from '@playwright/test';
import { start, admin } from './helpers';

async function dashboard(page: Page) {
  await start(page);
  await admin(page);
  await page.evaluate(() => window.ATOM.workspace!.navigate('tableau'));
}

test('the overview shows the governorates and opens one', async ({ page }) => {
  // Vue d'ensemble (22/11/2026) : sous la carte du bulletin, le contenu des Gouvernorats remplace la carte et le point de lecture
  await page.setViewportSize({ width: 1440, height: 1000 });
  await dashboard(page);
  const bloc = page.locator('#v-tableau .overview-gouvernorats');
  await expect(bloc.locator('.gv-gouv')).toHaveCount(24);
  await bloc.locator('.gv-tete select').selectOption('Sfax');
  await expect(bloc.locator('.gv-tete h2')).toHaveText('Sfax');
  await page.locator('.bulletin .btn.primary').click();
  await expect(page.locator('#v-monvent')).toHaveClass(/on/);
});

test('a dashboard alert opens its corresponding detail', async ({ page }) => {
  await dashboard(page);
  const alert = page.locator('[data-tb-alert]').first();
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
    await expect(page.locator('#v-tableau .overview-gouvernorats')).toBeVisible();
  }
});
