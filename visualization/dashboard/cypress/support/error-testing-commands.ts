// Error Testing Commands for LLMKG Dashboard

// Network Error Simulation
Cypress.Commands.add('simulateNetworkFailure', (errorType: string) => {
  switch (errorType) {
    case 'api_timeout':
      // Intercept API calls and delay them to cause timeout
      cy.intercept('GET', '**/api/**', { 
        delayMs: 30000 // 30 second delay to trigger timeout
      }).as('timeoutRequest')
      break
      
    case 'api_500':
      // Simulate server error
      cy.intercept('GET', '**/api/**', {
        statusCode: 500,
        body: { error: 'Internal Server Error', message: 'Database connection failed' }
      }).as('serverError')
      break
      
    case 'api_404':
      // Simulate not found error
      cy.intercept('GET', '**/api/**', {
        statusCode: 404,
        body: { error: 'Not Found', message: 'Endpoint does not exist' }
      }).as('notFoundError')
      break
      
    case 'network_offline':
      // Simulate network being offline
      cy.intercept('GET', '**/api/**', { forceNetworkError: true }).as('networkError')
      break
      
    case 'cors_error':
      // Simulate CORS error
      cy.intercept('GET', '**/api/**', {
        statusCode: 403,
        headers: { 'Access-Control-Allow-Origin': 'https://wrong-domain.com' },
        body: { error: 'CORS Error' }
      }).as('corsError')
      break
      
    case 'malformed_response':
      // Simulate malformed JSON response
      cy.intercept('GET', '**/api/**', {
        statusCode: 200,
        body: '{ "invalid": json, "missing": quote }'
      }).as('malformedResponse')
      break
      
    default:
      throw new Error(`Unknown error type: ${errorType}`)
  }
})

// WebSocket Error Simulation
Cypress.Commands.add('simulateWebSocketError', (errorType: string, code: number = 1000) => {
  cy.window().then((win) => {
    switch (errorType) {
      case 'connection_failed':
        // Simulate WebSocket connection failure
        win.postMessage({
          type: 'MOCK_WEBSOCKET_ERROR',
          error: { type: 'connection_failed', code: 1006, reason: 'Connection failed' }
        }, '*')
        break
        
      case 'connection_lost':
        // Simulate WebSocket connection lost
        win.postMessage({
          type: 'MOCK_WEBSOCKET_CLOSE',
          error: { code: 1001, reason: 'Going away' }
        }, '*')
        break
        
      case 'invalid_message':
        // Simulate invalid WebSocket message
        win.postMessage({
          type: 'MOCK_WEBSOCKET_MESSAGE',
          data: '{ invalid json message }'
        }, '*')
        break
        
      case 'message_too_large':
        // Simulate message size limit exceeded
        const largeMessage = 'x'.repeat(1024 * 1024) // 1MB message
        win.postMessage({
          type: 'MOCK_WEBSOCKET_ERROR',
          error: { type: 'message_too_large', data: largeMessage }
        }, '*')
        break
        
      case 'auth_failed':
        // Simulate authentication failure
        win.postMessage({
          type: 'MOCK_WEBSOCKET_ERROR',
          error: { type: 'auth_failed', code: 4001, reason: 'Authentication failed' }
        }, '*')
        break
        
      case 'rate_limited':
        // Simulate rate limiting
        win.postMessage({
          type: 'MOCK_WEBSOCKET_ERROR',
          error: { type: 'rate_limited', code: 4429, reason: 'Too many requests' }
        }, '*')
        break
        
      default:
        win.postMessage({
          type: 'MOCK_WEBSOCKET_ERROR',
          error: { type: errorType, code, reason: `Simulated error: ${errorType}` }
        }, '*')
    }
  })
})

// Component Error Injection
Cypress.Commands.add('forceComponentError', (componentId: string, error?: any) => {
  cy.window().then((win) => {
    // Inject error into React component
    const errorToInject = error || new Error('Simulated component error')
    
    win.postMessage({
      type: 'FORCE_COMPONENT_ERROR',
      componentId,
      error: {
        name: errorToInject.name,
        message: errorToInject.message,
        stack: errorToInject.stack
      }
    }, '*')
  })
})

// Memory Leak Simulation
Cypress.Commands.add('simulateMemoryLeak', (options: {
  leakType: 'event_listeners' | 'dom_nodes' | 'websocket_connections' | 'intervals'
  intensity: 'low' | 'medium' | 'high'
}) => {
  cy.window().then((win) => {
    const iterations = options.intensity === 'low' ? 100 : options.intensity === 'medium' ? 1000 : 10000
    
    switch (options.leakType) {
      case 'event_listeners':
        // Create event listeners that are never removed
        for (let i = 0; i < iterations; i++) {
          win.addEventListener('scroll', () => {
            // Leaked event listener
            console.log(`Leaked listener ${i}`)
          })
        }
        break
        
      case 'dom_nodes':
        // Create DOM nodes that are never cleaned up
        const container = win.document.createElement('div')
        container.style.display = 'none'
        win.document.body.appendChild(container)
        
        for (let i = 0; i < iterations; i++) {
          const node = win.document.createElement('div')
          node.innerHTML = `Leaked node ${i} with some content to consume memory`
          container.appendChild(node)
        }
        break
        
      case 'websocket_connections':
        // Create WebSocket connections that are never closed
        for (let i = 0; i < Math.min(iterations, 10); i++) { // Limit to 10 to avoid browser limits
          try {
            const ws = new WebSocket('ws://localhost:9999') // Non-existent endpoint
            ws.onopen = () => console.log(`Leaked WebSocket ${i}`)
          } catch (e) {
            // Expected to fail, but creates memory leak
          }
        }
        break
        
      case 'intervals':
        // Create intervals that are never cleared
        for (let i = 0; i < iterations; i++) {
          setInterval(() => {
            // Leaked interval
            const data = new Array(1000).fill(`leak-${i}`)
            console.log(data.length)
          }, 1000)
        }
        break
    }
  })
})

// Error Recovery Validation
Cypress.Commands.add('validateErrorRecovery', (componentSelector: string) => {
  // First, ensure component is in error state
  cy.get(componentSelector).within(() => {
    cy.get('.error-boundary, .error-message, [data-testid*="error"]').should('exist')
  })
  
  // Test retry functionality
  cy.get('[data-testid="retry-button"], .retry-button').click()
  
  // Wait for recovery attempt
  cy.wait(2000)
  
  // Validate recovery (component should either recover or show appropriate error)
  cy.get(componentSelector).should('be.visible')
})

// Console Error Monitoring
Cypress.Commands.add('startErrorMonitoring', () => {
  cy.window().then((win) => {
    // Capture console errors
    win.__consoleErrors = []
    win.__consoleWarnings = []
    
    const originalError = win.console.error
    const originalWarn = win.console.warn
    
    win.console.error = function(...args) {
      win.__consoleErrors.push({
        timestamp: Date.now(),
        args: args,
        stack: new Error().stack
      })
      originalError.apply(win.console, args)
    }
    
    win.console.warn = function(...args) {
      win.__consoleWarnings.push({
        timestamp: Date.now(),
        args: args
      })
      originalWarn.apply(win.console, args)
    }
    
    // Capture uncaught exceptions
    win.__uncaughtExceptions = []
    win.addEventListener('error', (event) => {
      win.__uncaughtExceptions.push({
        timestamp: Date.now(),
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
      })
    })
    
    // Capture unhandled promise rejections
    win.__unhandledRejections = []
    win.addEventListener('unhandledrejection', (event) => {
      win.__unhandledRejections.push({
        timestamp: Date.now(),
        reason: event.reason,
        promise: event.promise
      })
    })
  })
})

Cypress.Commands.add('stopErrorMonitoring', () => {
  cy.window().then((win) => {
    const errors = {
      consoleErrors: win.__consoleErrors || [],
      consoleWarnings: win.__consoleWarnings || [],
      uncaughtExceptions: win.__uncaughtExceptions || [],
      unhandledRejections: win.__unhandledRejections || []
    }
    
    // Clean up monitoring
    delete win.__consoleErrors
    delete win.__consoleWarnings
    delete win.__uncaughtExceptions
    delete win.__unhandledRejections
    
    return cy.wrap(errors)
  })
})

// Input Validation Error Testing
Cypress.Commands.add('testInputValidation', (formSelector: string, invalidInputs: {
  field: string
  value: string
  expectedError: string
}[]) => {
  cy.get(formSelector).should('be.visible')
  
  invalidInputs.forEach(({ field, value, expectedError }) => {
    // Clear and enter invalid value
    cy.get(`[data-testid="${field}"], [name="${field}"], #${field}`)
      .clear()
      .type(value)
      .blur()
    
    // Check for validation error
    cy.get(`[data-testid="${field}-error"], .error-message`).should('contain.text', expectedError)
  })
})

// Resource Loading Error Testing
Cypress.Commands.add('simulateResourceLoadFailure', (resourceType: 'image' | 'script' | 'stylesheet') => {
  cy.intercept('GET', '**/*', (req) => {
    const url = req.url.toLowerCase()
    
    switch (resourceType) {
      case 'image':
        if (url.includes('.png') || url.includes('.jpg') || url.includes('.svg')) {
          req.reply({ statusCode: 404 })
        }
        break
      case 'script':
        if (url.includes('.js')) {
          req.reply({ statusCode: 500 })
        }
        break
      case 'stylesheet':
        if (url.includes('.css')) {
          req.reply({ statusCode: 403 })
        }
        break
    }
  }).as('resourceFailure')
})

// Browser Compatibility Error Testing
Cypress.Commands.add('simulateBrowserIncompatibility', (feature: string) => {
  cy.window().then((win) => {
    switch (feature) {
      case 'webgl':
        // Disable WebGL
        const canvas = win.document.createElement('canvas')
        const getContext = canvas.getContext.bind(canvas)
        canvas.getContext = function(contextType) {
          if (contextType === 'webgl' || contextType === 'experimental-webgl') {
            return null
          }
          return getContext(contextType)
        }
        break
        
      case 'websocket':
        // Disable WebSocket
        win.WebSocket = undefined as any
        break
        
      case 'local_storage':
        // Disable localStorage
        win.localStorage = undefined as any
        break
        
      case 'geolocation':
        // Disable geolocation
        win.navigator.geolocation = undefined as any
        break
    }
  })
})

// Error Boundary Testing
Cypress.Commands.add('testErrorBoundary', (componentSelector: string) => {
  // Force an error in the component
  cy.forceComponentError(componentSelector, new Error('Test error boundary'))
  
  // Verify error boundary catches the error
  cy.get('.error-boundary, [data-testid="error-boundary"]').should('be.visible')
  cy.get('.error-boundary').should('contain.text', 'Something went wrong')
  
  // Test error boundary reset
  cy.get('[data-testid="reset-error"], .error-reset-button').click()
  
  // Verify component recovers
  cy.get(componentSelector).should('be.visible')
})

declare global {
  namespace Cypress {
    interface Chainable {
      // Network errors
      simulateNetworkFailure(errorType: string): Chainable<void>
      
      // WebSocket errors
      simulateWebSocketError(errorType: string, code?: number): Chainable<void>
      
      // Component errors
      forceComponentError(componentId: string, error?: any): Chainable<void>
      validateErrorRecovery(componentSelector: string): Chainable<void>
      
      // Memory and resource errors
      simulateMemoryLeak(options: {
        leakType: 'event_listeners' | 'dom_nodes' | 'websocket_connections' | 'intervals'
        intensity: 'low' | 'medium' | 'high'
      }): Chainable<void>
      
      // Error monitoring
      startErrorMonitoring(): Chainable<void>
      stopErrorMonitoring(): Chainable<any>
      
      // Input validation
      testInputValidation(formSelector: string, invalidInputs: {
        field: string
        value: string
        expectedError: string
      }[]): Chainable<void>
      
      // Resource loading
      simulateResourceLoadFailure(resourceType: 'image' | 'script' | 'stylesheet'): Chainable<void>
      
      // Browser compatibility
      simulateBrowserIncompatibility(feature: string): Chainable<void>
      
      // Error boundaries
      testErrorBoundary(componentSelector: string): Chainable<void>
    }
  }
}