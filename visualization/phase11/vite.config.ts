import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [
      react({
        // Enable Fast Refresh
        fastRefresh: true,
        // Include runtime in development
        include: "**/*.{jsx,tsx}",
      }),
      VitePWA({
        registerType: 'autoUpdate',
        workbox: {
          globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
          runtimeCaching: [
            {
              urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
              handler: 'CacheFirst',
              options: {
                cacheName: 'google-fonts-cache',
                expiration: {
                  maxEntries: 10,
                  maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
                }
              }
            }
          ]
        },
        manifest: {
          name: 'LLMKG Visualization Dashboard',
          short_name: 'LLMKG Viz',
          description: 'Brain-Inspired Cognitive Architecture Visualization System',
          theme_color: '#001529',
          background_color: '#ffffff',
          display: 'standalone',
          icons: [
            {
              src: '/icon-192x192.png',
              sizes: '192x192',
              type: 'image/png'
            },
            {
              src: '/icon-512x512.png',
              sizes: '512x512',
              type: 'image/png'
            }
          ]
        }
      })
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@types': path.resolve(__dirname, './src/types'),
        '@services': path.resolve(__dirname, './src/services'),
        '@stores': path.resolve(__dirname, './src/stores'),
        '@assets': path.resolve(__dirname, './src/assets'),
        '@styles': path.resolve(__dirname, './src/styles'),
        '@config': path.resolve(__dirname, './src/config'),
      },
    },
    define: {
      'process.env.NODE_ENV': JSON.stringify(mode),
      'process.env.VITE_APP_VERSION': JSON.stringify(process.env.npm_package_version),
      '__DEV__': mode === 'development',
    },
    server: {
      port: 3000,
      host: '0.0.0.0',
      open: true,
      cors: true,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8080',
          changeOrigin: true,
          secure: false,
        },
        '/ws': {
          target: env.VITE_WS_URL || 'ws://localhost:8080',
          ws: true,
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 4173,
      host: '0.0.0.0',
      open: true,
    },
    build: {
      outDir: 'dist',
      sourcemap: mode === 'development',
      minify: mode === 'production' ? 'esbuild' : false,
      target: 'esnext',
      rollupOptions: {
        output: {
          manualChunks: {
            // Core React libraries
            'react-vendor': ['react', 'react-dom', 'react-router-dom'],
            // State management
            'redux-vendor': ['react-redux', '@reduxjs/toolkit'],
            // UI libraries
            'ui-vendor': ['antd', '@ant-design/icons'],
            // Visualization libraries
            'd3-vendor': ['d3'],
            'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
            'chart-vendor': ['recharts'],
            // Utility libraries
            'utils-vendor': ['lodash', 'date-fns', 'uuid'],
            // Monaco editor (large dependency)
            'monaco-vendor': ['monaco-editor', '@monaco-editor/react'],
          },
        },
      },
      chunkSizeWarningLimit: 1000,
      reportCompressedSize: false,
    },
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        'react-redux',
        '@reduxjs/toolkit',
        'antd',
        '@ant-design/icons',
        'd3',
        'three',
        'recharts',
        'lodash',
        'date-fns',
        'framer-motion'
      ],
      exclude: ['@monaco-editor/react']
    },
    css: {
      preprocessorOptions: {
        less: {
          javascriptEnabled: true,
        },
      },
      modules: {
        localsConvention: 'camelCase',
      },
    },
    esbuild: {
      // Remove console logs in production
      drop: mode === 'production' ? ['console', 'debugger'] : [],
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      css: true,
    },
  };
});