import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Served under https://bmapbenoit-ui.github.io/stella-v7/ on GitHub Pages.
// Override via VITE_BASE when deploying elsewhere (e.g. Vercel: VITE_BASE=/).
const base = process.env.VITE_BASE ?? '/stella-v7/';

export default defineConfig({
  plugins: [react()],
  base,
  server: {
    host: true,
    port: 5173,
  },
});
