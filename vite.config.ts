import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { resolve } from 'node:path';

export default defineConfig({
  root: 'frontend',
  base: '/app/',
  plugins: [react(), tailwindcss()],
  publicDir: 'public',
  build: {
    outDir: '../webapp',
    emptyOutDir: false,
    rollupOptions: {
      input: {
        science: resolve('frontend/index.html'),
        atom: resolve('frontend/atom-wind/index.html'),
        export: resolve('frontend/export/index.html'),
      },
      output: { manualChunks(id) {
        if (id.includes('/node_modules/three/')) return 'atmosphere';
        if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'react';
      } },
    },
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/mcp': 'http://127.0.0.1:8000' } },
});
