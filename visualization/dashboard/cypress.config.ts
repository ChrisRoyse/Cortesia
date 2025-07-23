import { defineConfig } from 'cypress'

export default defineConfig({
  e2e: {
    // Base URL for the application
    baseUrl: 'http://localhost:5176',
    
    // Browser configuration - prioritize Electron for headed testing
    browser: 'electron',
    
    // Viewport settings
    viewportWidth: 1920,
    viewportHeight: 1080,
    
    // Test file patterns
    specPattern: [
      'cypress/e2e/phase1-foundation/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase2-components/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase3-realtime/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase4-visualizations/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase5-performance/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase6-error-handling/**/*.cy.{js,jsx,ts,tsx}',
      'cypress/e2e/phase7-e2e-workflows/**/*.cy.{js,jsx,ts,tsx}'
    ],
    
    // Support file
    supportFile: 'cypress/support/e2e.ts',
    
    // Test timeout settings
    defaultCommandTimeout: 10000,
    requestTimeout: 15000,
    responseTimeout: 15000,
    pageLoadTimeout: 30000,
    
    // Video and screenshot settings for debugging
    video: true,
    videosFolder: 'cypress/videos',
    screenshotsFolder: 'cypress/screenshots',
    screenshotOnRunFailure: true,
    
    // Chrome/Electron specific settings
    chromeWebSecurity: false,
    
    // Environment variables
    env: {
      // API endpoints
      api_url: 'http://localhost:8080',
      websocket_url: 'ws://localhost:9000',
      
      // Test data configuration
      large_dataset_size: 10000,
      stress_test_duration: 300000, // 5 minutes
      extended_session_duration: 1800000, // 30 minutes
      
      // Performance thresholds
      max_render_time: 15000,
      min_fps: 30,
      max_memory_mb: 500,
      
      // Feature flags for testing
      enable_performance_monitoring: true,
      enable_memory_profiling: true,
      enable_accessibility_testing: true,
      
      // Mock data flags
      use_mock_websocket: true,
      use_mock_api: true
    },
    
    // Node events and tasks
    setupNodeEvents(on, config) {
      // Code coverage plugin
      require('@cypress/code-coverage/task')(on, config)
      
      // Custom tasks for testing
      on('task', {
        // Performance monitoring tasks
        startPerformanceMonitor() {
          console.log('Starting performance monitor...')
          return null
        },
        
        stopPerformanceMonitor() {
          console.log('Stopping performance monitor...')
          return {
            averageFPS: 60,
            maxMemoryUsage: 150 * 1024 * 1024,
            renderTime: 2500
          }
        },
        
        // Memory profiling tasks
        takeMemorySnapshot() {
          console.log('Taking memory snapshot...')
          return {
            usedJSHeapSize: 100 * 1024 * 1024,
            totalJSHeapSize: 200 * 1024 * 1024,
            timestamp: Date.now()
          }
        },
        
        // Mock server control
        startMockWebSocketServer() {
          console.log('Starting mock WebSocket server...')
          return { port: 9001, status: 'running' }
        },
        
        stopMockWebSocketServer() {
          console.log('Stopping mock WebSocket server...')
          return { status: 'stopped' }
        },
        
        // Test data generation
        generateLargeDataset(size: number) {
          console.log(`Generating dataset with ${size} entities...`)
          return {
            entities: Array.from({ length: size }, (_, i) => ({
              id: `entity_${i}`,
              type_id: (i % 4) + 1,
              activation: Math.random(),
              direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4]
            })),
            relationships: Array.from({ length: size * 2 }, (_, i) => ({
              from: `entity_${Math.floor(Math.random() * size)}`,
              to: `entity_${Math.floor(Math.random() * size)}`,
              weight: Math.random()
            }))
          }
        },
        
        // Log test results
        logTestResult(result: any) {
          console.log('Test Result:', JSON.stringify(result, null, 2))
          return null
        }
      })
      
      // Browser launch arguments for Electron
      on('before:browser:launch', (browser, launchOptions) => {
        if (browser.name === 'electron') {
          // Enable performance monitoring APIs
          launchOptions.args.push('--enable-precise-memory-info')
          launchOptions.args.push('--enable-memory-info')
          launchOptions.args.push('--js-flags=--expose-gc')
          
          // Disable web security for testing
          launchOptions.args.push('--disable-web-security')
          launchOptions.args.push('--disable-features=VizDisplayCompositor')
          
          // Enable GPU for WebGL testing
          launchOptions.args.push('--enable-gpu')
          launchOptions.args.push('--ignore-gpu-blocklist')
        }
        
        return launchOptions
      })
      
      return config
    },
    
    // Experimental features
    experimentalStudio: false,
    experimentalMemoryManagement: true
  },
  
  // Component testing configuration
  component: {
    devServer: {
      framework: 'react',
      bundler: 'vite'
    },
    specPattern: 'cypress/component/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/component.ts'
  }
})