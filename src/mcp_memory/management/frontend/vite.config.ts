import path from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const daemonBaseUrl = process.env.VITE_DAEMON_BASE_URL ?? 'http://127.0.0.1:8765';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': daemonBaseUrl,
    },
  },
  build: {
    outDir: path.resolve(__dirname, '../static/dist'),
    emptyOutDir: true,
  },
});