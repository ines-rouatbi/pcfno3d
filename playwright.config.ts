import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './frontend/tests',
  fullyParallel: false,
  workers: 1,
  timeout: 60000,
  expect: { timeout: 15000 },
  use: { baseURL: process.env.ATOM_TEST_URL || 'http://127.0.0.1:8000', channel: 'chrome', headless: true, trace: 'retain-on-failure' },
  reporter: [['list'], ['json',{outputFile:'output/redesign/browser-results.json'}]],
});
