import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks: {
          // Plotly is ~3MB — split it so the app shell loads fast
          'vendor-plotly': ['react-plotly.js', 'plotly.js'],
          // React ecosystem — stable, long-lived cache
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
})

