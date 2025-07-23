// Cypress E2E Support File
// This file is processed and loaded automatically before your test files.

import '@cypress/code-coverage/support'
import './commands'
import './performance-commands'
import './visualization-commands'
import './error-testing-commands'
import './accessibility-commands'
import './mock-websocket-server'

// Global configuration
Cypress.on('uncaught:exception', (err, runnable) => {
  // Prevent Cypress from failing on uncaught exceptions during testing
  // We want to test error handling, so we'll handle these in our tests
  console.log('Uncaught exception:', err.message)
  return false
})

// Performance monitoring setup
beforeEach(() => {
  // Reset performance counters
  cy.window().then((win) => {
    win.__performanceMetrics = {
      startTime: Date.now(),
      frameCount: 0,
      memoryUsage: [],
      errors: []
    }
  })
  
  // Set up mock WebSocket if enabled
  if (Cypress.env('use_mock_websocket')) {
    cy.setupMockWebSocketServer()
  }
})

afterEach(() => {
  // Clean up after each test
  cy.window().then((win) => {
    // Clear any test data
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.stop()
    }
    
    // Log performance metrics
    const metrics = win.__performanceMetrics
    if (metrics) {
      const duration = Date.now() - metrics.startTime
      cy.task('logTestResult', {
        testName: Cypress.currentTest.title,
        duration,
        frameCount: metrics.frameCount,
        memoryPeak: Math.max(...metrics.memoryUsage, 0),
        errorCount: metrics.errors.length
      })
    }
  })
  
  // Take screenshot on failure
  if (Cypress.currentTest.state === 'failed') {
    cy.screenshot(`failed-${Cypress.currentTest.title.replace(/\s+/g, '-')}`)
  }
})

// Global test data
declare global {
  namespace Cypress {
    interface Chainable {
      // Navigation commands
      switchTab(tabName: string): Chainable<void>
      waitForDashboardLoad(): Chainable<void>
      waitForWebSocketConnection(): Chainable<void>
      
      // Data mocking commands
      mockWebSocketMessage(data: any): Chainable<void>
      sendRawWebSocketMessage(message: string): Chainable<void>
      setupMockWebSocketServer(): Chainable<void>
      teardownMockWebSocketServer(): Chainable<void>
      
      // Performance commands
      startPerformanceMonitoring(): Chainable<void>
      stopPerformanceMonitoring(): Chainable<any>
      measureFrameRate(): Chainable<number>
      measureMemoryUsage(label?: string): Chainable<number>
      startFrameRateMonitoring(): Chainable<void>
      stopFrameRateMonitoring(): Chainable<number>
      
      // Visualization commands
      validateChartRendering(chartSelector: string): Chainable<void>
      checkDataVisualization(containerSelector: string, expectedDataPoints: number): Chainable<void>
      simulate3DInteraction(canvas: string, interaction: any): Chainable<void>
      
      // Error testing commands
      simulateNetworkFailure(errorType: string): Chainable<void>
      simulateWebSocketError(errorType: string, code?: number): Chainable<void>
      forceComponentError(componentId: string, error?: any): Chainable<void>
      validateErrorRecovery(componentSelector: string): Chainable<void>
      
      // Accessibility commands
      validateAccessibilityWorkflow(): Chainable<void>
      checkKeyboardNavigation(): Chainable<void>
      validateAriaLabels(): Chainable<void>
      
      // User workflow commands
      completeUserWorkflow(workflowType: string, options?: any): Chainable<void>
      simulateExtendedUsage(durationMinutes: number): Chainable<void>
      validateCollaborativeWorkflow(): Chainable<void>
    }
  }
}

// Test environment information
console.log('Cypress Environment:', {
  baseUrl: Cypress.config('baseUrl'),
  browser: Cypress.browser,
  viewport: {
    width: Cypress.config('viewportWidth'),
    height: Cypress.config('viewportHeight')
  },
  env: Cypress.env()
})