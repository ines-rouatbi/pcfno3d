import { expect, type Page } from '@playwright/test';

// Les tests tournent contre un service qui sert des données (python scripts/dev_service.py, ou la
// production) : plus aucune connexion de démonstration. L'administrateur est un vrai compte du service,
// créé par `PCFNO3D_MOT_DE_PASSE_ADMIN=… python scripts/dev_service.py` (ou `pcfno3d compte-admin`).
export const ADMIN_EMAIL = process.env.ATOM_TEST_ADMIN_EMAIL || 'admin@atomwind.local';
export const ADMIN_MDP = process.env.ATOM_TEST_ADMIN_PASSWORD || process.env.PCFNO3D_MOT_DE_PASSE_ADMIN || '';

export async function start(page: Page, vue?: string) {
  await page.addInitScript(() => { localStorage.setItem('aw-tour', '1'); localStorage.setItem('aw-theme', JSON.stringify('light')); });
  await page.goto('/app/atom-wind/' + (vue ? `#${vue}` : ''));
  await page.waitForFunction(() => window.ATOM?.workspace && !document.getElementById('aw-chargement'), null, { timeout: 240000 });
}

export async function admin(page: Page) {
  if (!ADMIN_MDP) throw new Error('mot de passe administrateur de test absent : ATOM_TEST_ADMIN_PASSWORD');
  await page.locator('#b-login').click();
  await page.locator('#l-mail').fill(ADMIN_EMAIL);
  await page.locator('#l-pw').fill(ADMIN_MDP);
  await page.locator('#f-login button[type=submit]').click();
  await expect(page.locator('#authm')).toBeHidden();
  await expect.poll(() => page.evaluate(() => window.ATOM.workspace!.role!())).toBe('admin');
  await expect(page.locator('#toast')).toBeHidden({ timeout: 10000 });
}
