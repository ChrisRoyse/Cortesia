# Phase 6: Error Handling and Edge Cases

**Duration**: 3-4 days  
**Priority**: Critical - Production reliability validation  
**Focus**: Network failures, invalid data, error boundaries  
**Prerequisites**: Phases 1-5 completed successfully

## Objectives
- Test comprehensive error handling scenarios
- Validate graceful degradation under failure conditions
- Ensure proper error logging and user feedback
- Test recovery mechanisms and fallback strategies
- Validate data validation and sanitization
- Test edge cases and boundary conditions

## Test Categories

### 6.1 Network and Connectivity Error Testing

#### 6.1.1 WebSocket Connection Failures
```javascript
describe('WebSocket Connection Error Handling', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should handle initial connection failure gracefully', () => {
    cy.mockWebSocketConnectionFailure('ECONNREFUSED')
    cy.reload()
    
    cy.get('[data-testid="connection-error-message"]')
      .should('be.visible')
      .and('contain.text', 'Unable to connect to server')
    
    cy.get('[data-testid="retry-connection-button"]')
      .should('be.visible')
    
    cy.get('[data-testid="offline-mode-indicator"]')
      .should('be.visible')
      .and('contain.text', 'Working Offline')
  })

  it('should attempt automatic reconnection with backoff', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    // Simulate sudden disconnection
    cy.simulateNetworkFailure('CONNECTION_LOST')
    
    cy.get('[data-testid="connection-status"]')
      .should('contain.text', 'Reconnecting')
    
    // Should show increasing retry attempts
    cy.get('[data-testid="retry-attempt"]')
      .should('contain.text', 'Attempt 1')
    
    cy.wait(2000)
    cy.get('[data-testid="retry-attempt"]')
      .should('contain.text', 'Attempt 2')
    
    cy.wait(4000)
    cy.get('[data-testid="retry-attempt"]')
      .should('contain.text', 'Attempt 3')
    
    // Verify exponential backoff
    cy.get('[data-testid="next-retry-in"]')
      .should('be.visible')
  })

  it('should handle intermittent connection drops', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    // Send some data successfully
    cy.mockWebSocketMessage({
      type: 'brain_metrics_update',
      data: { entityCount: 100, avgActivation: 0.5 }
    })
    
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', '100')
    
    // Simulate brief connection drop
    cy.simulateIntermittentConnection(500) // 500ms outage
    
    cy.get('[data-testid="connection-unstable-warning"]')
      .should('be.visible')
    
    // Should reconnect and continue working
    cy.mockWebSocketMessage({
      type: 'brain_metrics_update', 
      data: { entityCount: 150, avgActivation: 0.7 }
    })
    
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', '150')
  })

  it('should handle WebSocket protocol errors', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    // Simulate protocol error
    cy.simulateWebSocketError('PROTOCOL_ERROR', 1002)
    
    cy.get('[data-testid="protocol-error-message"]')
      .should('be.visible')
      .and('contain.text', 'Connection protocol error')
    
    cy.get('[data-testid="diagnostic-info"]')
      .should('contain.text', 'Error Code: 1002')
    
    // Should attempt to reconnect with fresh connection
    cy.allowWebSocketConnection()
    cy.get('[data-testid="connection-status"]', { timeout: 10000 })
      .should('contain.text', 'Connected')
  })

  it('should handle server unavailable scenarios', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    // Simulate server shutdown
    cy.simulateServerUnavailable()
    
    cy.get('[data-testid="server-unavailable-message"]')
      .should('be.visible')
      .and('contain.text', 'Server is currently unavailable')
    
    cy.get('[data-testid="offline-mode-button"]')
      .should('be.visible')
      .click()
    
    cy.get('[data-testid="demo-mode-indicator"]')
      .should('be.visible')
      .and('contain.text', 'Demo Mode')
    
    // Should work with demo data
    cy.get('[data-testid="dashboard-container"]')
      .should('be.visible')
  })
})
```

#### 6.1.2 API Endpoint Failures
```javascript
describe('API Endpoint Error Handling', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should handle 404 errors for missing endpoints', () => {
    cy.intercept('GET', '/api/brain/metrics', { statusCode: 404 }).as('notFound')
    
    cy.triggerAPICall('/api/brain/metrics')
    
    cy.get('[data-testid="api-error-404"]')
      .should('be.visible')
      .and('contain.text', 'Endpoint not found')
    
    cy.get('[data-testid="fallback-data-notice"]')
      .should('be.visible')
      .and('contain.text', 'Using cached data')
  })

  it('should handle 500 server errors with retry logic', () => {
    cy.intercept('GET', '/api/brain/status', { statusCode: 500 }).as('serverError')
    
    cy.triggerAPICall('/api/brain/status')
    
    cy.get('[data-testid="server-error-message"]')
      .should('be.visible')
      .and('contain.text', 'Server error occurred')
    
    cy.get('[data-testid="auto-retry-indicator"]')
      .should('be.visible')
    
    // Mock successful retry
    cy.intercept('GET', '/api/brain/status', { statusCode: 200, body: { status: 'healthy' } })
    
    cy.get('[data-testid="api-call-success"]', { timeout: 10000 })
      .should('be.visible')
  })

  it('should handle timeout errors gracefully', () => {
    cy.intercept('GET', '/api/brain/large-dataset', {
      delay: 30000 // 30 second delay to trigger timeout
    }).as('timeoutRequest')
    
    cy.triggerAPICall('/api/brain/large-dataset')
    
    cy.get('[data-testid="request-timeout-message"]', { timeout: 35000 })
      .should('be.visible')
      .and('contain.text', 'Request timed out')
    
    cy.get('[data-testid="cancel-request-button"]')
      .should('be.visible')
    
    cy.get('[data-testid="retry-with-smaller-dataset"]')
      .should('be.visible')
  })

  it('should handle rate limiting gracefully', () => {
    cy.intercept('GET', '/api/**', { 
      statusCode: 429,
      headers: { 'Retry-After': '60' },
      body: { error: 'Rate limit exceeded' }
    }).as('rateLimited')
    
    cy.triggerAPICall('/api/brain/metrics')
    
    cy.get('[data-testid="rate-limit-message"]')
      .should('be.visible')
      .and('contain.text', 'Rate limit exceeded')
    
    cy.get('[data-testid="retry-countdown"]')
      .should('be.visible')
      .and('contain.text', '60')
  })
})
```

### 6.2 Data Validation and Sanitization Testing

#### 6.2.1 Invalid Data Structure Testing
```javascript
describe('Invalid Data Structure Handling', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should handle malformed JSON messages', () => {
    cy.sendRawWebSocketMessage('{ invalid json }')
    
    cy.get('[data-testid="json-parse-error"]')
      .should('be.visible')
      .and('contain.text', 'Invalid message format')
    
    cy.get('[data-testid="error-log-entry"]')
      .should('contain.text', 'JSON parse error')
    
    // Should continue working with valid messages
    cy.mockWebSocketMessage({
      type: 'brain_metrics_update',
      data: { entityCount: 100 }
    })
    
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', '100')
  })

  it('should validate entity data types and ranges', () => {
    const invalidEntityData = {
      type: 'entity_update',
      data: {
        entities: [
          { id: null, activation: 'invalid' }, // Invalid types
          { id: 'valid', activation: -0.5 },   // Invalid range
          { id: 'valid2', activation: 1.5 },   // Invalid range
          { id: 'valid3', activation: 0.75 }   // Valid
        ]
      }
    }
    
    cy.mockWebSocketMessage(invalidEntityData)
    
    cy.get('[data-testid="validation-errors"]')
      .should('be.visible')
      .and('contain.text', '3 invalid entities filtered')
    
    cy.get('[data-testid="valid-entities-processed"]')
      .should('contain.text', '1')
    
    // Only valid entity should be processed
    cy.get('[data-testid="entity-valid3"]')
      .should('be.visible')
  })

  it('should handle missing required fields', () => {
    const incompleteData = {
      type: 'brain_metrics_update',
      data: {
        // Missing entityCount and other required fields
        avgActivation: 0.5
      }
    }
    
    cy.mockWebSocketMessage(incompleteData)
    
    cy.get('[data-testid="incomplete-data-warning"]')
      .should('be.visible')
      .and('contain.text', 'Incomplete data received')
    
    cy.get('[data-testid="missing-fields-list"]')
      .should('contain.text', 'entityCount')
    
    // Should use default values or maintain previous state
    cy.get('[data-testid="using-default-values"]')
      .should('be.visible')
  })

  it('should sanitize potentially dangerous data', () => {
    const dangerousData = {
      type: 'entity_update',
      data: {
        entities: [
          {
            id: '<script>alert("xss")</script>',
            activation: 0.5,
            properties: {
              name: '<img src=x onerror=alert("xss")>',
              description: 'javascript:alert("xss")'
            }
          }
        ]
      }
    }
    
    cy.mockWebSocketMessage(dangerousData)
    
    // Should sanitize the data
    cy.get('[data-testid="entity-details"]')
      .should('not.contain.html', '<script>')
      .and('not.contain.html', '<img')
      .and('not.contain.text', 'javascript:')
    
    cy.get('[data-testid="sanitization-log"]')
      .should('contain.text', 'Potentially dangerous content sanitized')
  })
})
```

#### 6.2.2 Edge Case Data Values Testing
```javascript
describe('Edge Case Data Values', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should handle extreme numeric values', () => {
    const extremeData = {
      type: 'brain_metrics_update',
      data: {
        entityCount: Number.MAX_SAFE_INTEGER,
        avgActivation: Number.POSITIVE_INFINITY,
        minActivation: Number.NEGATIVE_INFINITY,
        maxActivation: NaN,
        timestamp: 0
      }
    }
    
    cy.mockWebSocketMessage(extremeData)
    
    cy.get('[data-testid="extreme-values-warning"]')
      .should('be.visible')
    
    // Should clamp values to reasonable ranges
    cy.get('[data-testid="entity-count"]')
      .should('not.contain.text', 'Infinity')
    
    cy.get('[data-testid="avg-activation"]')
      .should('not.contain.text', 'NaN')
  })

  it('should handle empty and null datasets', () => {
    const emptyData = {
      type: 'brain_graph_update',
      data: {
        entities: [],
        relationships: [],
        statistics: null
      }
    }
    
    cy.switchTab('brain-graph')
    cy.mockWebSocketMessage(emptyData)
    
    cy.get('[data-testid="empty-dataset-message"]')
      .should('be.visible')
      .and('contain.text', 'No data to display')
    
    cy.get('[data-testid="empty-state-illustration"]')
      .should('be.visible')
    
    cy.get('[data-testid="sample-data-button"]')
      .should('be.visible')
  })

  it('should handle very large string values', () => {
    const largeStringData = {
      type: 'entity_update',
      data: {
        entities: [
          {
            id: 'test',
            activation: 0.5,
            properties: {
              description: 'A'.repeat(10000), // 10KB string
              metadata: 'B'.repeat(100000)    // 100KB string
            }
          }
        ]
      }
    }
    
    cy.mockWebSocketMessage(largeStringData)
    
    cy.get('[data-testid="large-string-truncated"]')
      .should('be.visible')
    
    cy.get('[data-testid="entity-description"]')
      .should('not.contain.text', 'A'.repeat(10000))
    
    cy.get('[data-testid="show-full-text-button"]')
      .should('be.visible')
  })

  it('should handle circular reference data structures', () => {
    // Simulate circular reference in serialized data
    const circularData = {
      type: 'complex_update',
      data: {
        entity: {
          id: 'test',
          parent: '[Circular Reference]',
          children: ['[Circular Reference]']
        }
      }
    }
    
    cy.mockWebSocketMessage(circularData)
    
    cy.get('[data-testid="circular-reference-warning"]')
      .should('be.visible')
    
    cy.get('[data-testid="entity-test"]')
      .should('be.visible')
      .and('not.contain.text', '[Circular Reference]')
  })
})
```

### 6.3 Component Error Boundary Testing

#### 6.3.1 Visualization Component Failures
```javascript
describe('Visualization Component Error Boundaries', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should catch and recover from Three.js rendering errors', () => {
    cy.switchTab('brain-graph')
    
    // Force Three.js error by sending invalid geometry data
    cy.window().then((win) => {
      win.postMessage({
        type: 'FORCE_THREEJS_ERROR',
        error: 'Invalid geometry'
      }, '*')
    })
    
    cy.get('[data-testid="threejs-error-boundary"]')
      .should('be.visible')
      .and('contain.text', 'Visualization Error')
    
    cy.get('[data-testid="error-details"]')
      .should('contain.text', 'Invalid geometry')
    
    cy.get('[data-testid="restart-visualization-button"]')
      .should('be.visible')
      .click()
    
    // Should recover and work normally
    cy.get('[data-testid="three-canvas"]')
      .should('be.visible')
  })

  it('should handle D3.js SVG rendering failures', () => {
    cy.switchTab('neural-activity')
    
    // Force D3 error by corrupting SVG container
    cy.window().then((win) => {
      win.postMessage({
        type: 'FORCE_D3_ERROR',
        error: 'SVG container not found'
      }, '*')
    })
    
    cy.get('[data-testid="d3-error-boundary"]')
      .should('be.visible')
      .and('contain.text', 'Chart Error')
    
    cy.get('[data-testid="fallback-table"]')
      .should('be.visible')
      .and('contain.text', 'Data Table View')
    
    cy.get('[data-testid="retry-chart-button"]')
      .click()
    
    cy.get('[data-testid="heatmap-svg"]')
      .should('be.visible')
  })

  it('should isolate component failures to prevent cascade', () => {
    cy.switchTab('overview')
    
    // Force error in one metric card
    cy.get('[data-testid="cpu-metric-card"]').then(($card) => {
      cy.window().then((win) => {
        win.postMessage({
          type: 'FORCE_COMPONENT_ERROR',
          componentId: 'cpu-metric-card'
        }, '*')
      })
    })
    
    cy.get('[data-testid="cpu-metric-error"]')
      .should('be.visible')
    
    // Other metric cards should continue working
    cy.get('[data-testid="memory-metric-card"]')
      .should('be.visible')
      .and('not.have.class', 'error')
    
    cy.get('[data-testid="network-metric-card"]')
      .should('be.visible')
      .and('not.have.class', 'error')
  })

  it('should provide diagnostic information for debugging', () => {
    cy.switchTab('brain-graph')
    
    cy.window().then((win) => {
      win.postMessage({
        type: 'FORCE_COMPONENT_ERROR',
        error: new Error('Test error for debugging'),
        componentStack: 'BrainGraph > ThreeScene > EntityRenderer'
      }, '*')
    })
    
    cy.get('[data-testid="error-boundary"]')
      .should('be.visible')
    
    cy.get('[data-testid="error-diagnostic-button"]')
      .click()
    
    cy.get('[data-testid="error-stack-trace"]')
      .should('be.visible')
      .and('contain.text', 'BrainGraph')
    
    cy.get('[data-testid="browser-info"]')
      .should('be.visible')
    
    cy.get('[data-testid="memory-usage"]')
      .should('be.visible')
    
    cy.get('[data-testid="send-error-report-button"]')
      .should('be.visible')
  })
})
```

### 6.4 Resource Exhaustion and Browser Limits

#### 6.4.1 Browser Resource Limits Testing
```javascript
describe('Browser Resource Limits', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should handle canvas size limitations', () => {
    cy.switchTab('brain-graph')
    
    // Try to create extremely large canvas
    cy.window().then((win) => {
      win.postMessage({
        type: 'SET_CANVAS_SIZE',
        width: 32768,  // Exceeds many browser limits
        height: 32768
      }, '*')
    })
    
    cy.get('[data-testid="canvas-size-warning"]')
      .should('be.visible')
      .and('contain.text', 'Canvas size too large')
    
    cy.get('[data-testid="canvas-size-clamped"]')
      .should('be.visible')
    
    // Should fallback to maximum supported size
    cy.get('[data-testid="three-canvas"]')
      .should('have.attr', 'width')
      .and('not.equal', '32768')
  })

  it('should handle DOM node count limits', () => {
    cy.switchTab('neural-activity')
    
    // Try to create excessive DOM nodes
    const extremeDataset = {
      entities: Array.from({ length: 200000 }, (_, i) => ({
        id: `entity_${i}`,
        activation: Math.random()
      }))
    }
    
    cy.mockWebSocketMessage(extremeDataset)
    
    cy.get('[data-testid="dom-limit-warning"]', { timeout: 10000 })
      .should('be.visible')
      .and('contain.text', 'Dataset too large for direct rendering')
    
    cy.get('[data-testid="virtualization-enabled"]')
      .should('be.visible')
    
    // Should use virtualization
    cy.get('[data-testid="virtual-row-count"]')
      .should('be.visible')
      .and('not.contain.text', '200000')
  })

  it('should handle local storage quota exceeded', () => {
    // Fill up localStorage
    cy.window().then((win) => {
      let i = 0
      try {
        while (true) {
          win.localStorage.setItem(`test_${i}`, 'x'.repeat(1024 * 1024)) // 1MB chunks
          i++
        }
      } catch (e) {
        // Expected when quota is reached
      }
    })
    
    // Try to save dashboard state
    cy.mockWebSocketMessage({
      type: 'brain_metrics_update',
      data: { entityCount: 1000, avgActivation: 0.5 }
    })
    
    cy.get('[data-testid="storage-quota-exceeded"]')
      .should('be.visible')
    
    cy.get('[data-testid="clear-storage-button"]')
      .click()
    
    cy.get('[data-testid="storage-cleared-confirmation"]')
      .should('be.visible')
    
    // Should now be able to save state
    cy.get('[data-testid="state-saved-indicator"]')
      .should('be.visible')
  })

  it('should handle CPU throttling scenarios', () => {
    cy.switchTab('brain-graph')
    
    // Simulate CPU throttling by creating expensive operations
    cy.window().then((win) => {
      win.postMessage({
        type: 'SIMULATE_CPU_THROTTLING',
        intensity: 'high'
      }, '*')
    })
    
    const largeDataset = generateLargeDataset(5000, 10000)
    
    cy.startFrameRateMonitoring()
    cy.mockWebSocketMessage(largeDataset)
    
    cy.wait(5000)
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      if (avgFps < 15) { // If severely throttled
        cy.get('[data-testid="performance-degradation-warning"]')
          .should('be.visible')
        
        cy.get('[data-testid="reduced-quality-mode"]')
          .should('be.visible')
      }
    })
  })
})
```

### 6.5 Cross-Browser Compatibility Edge Cases

#### 6.5.1 Browser-Specific Error Handling
```javascript
describe('Browser-Specific Error Handling', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should handle WebGL context creation failures', () => {
    cy.switchTab('brain-graph')
    
    // Mock WebGL not available
    cy.window().then((win) => {
      const originalGetContext = HTMLCanvasElement.prototype.getContext
      HTMLCanvasElement.prototype.getContext = function(type) {
        if (type === 'webgl' || type === 'experimental-webgl') {
          return null
        }
        return originalGetContext.call(this, type)
      }
    })
    
    cy.reload()
    cy.switchTab('brain-graph')
    
    cy.get('[data-testid="webgl-not-supported"]')
      .should('be.visible')
      .and('contain.text', 'WebGL not supported')
    
    cy.get('[data-testid="fallback-2d-renderer"]')
      .should('be.visible')
    
    cy.get('[data-testid="canvas-2d-fallback"]')
      .should('be.visible')
  })

  it('should handle Web Workers not available', () => {
    cy.window().then((win) => {
      // Mock Web Workers not available
      delete win.Worker
    })
    
    // Try to trigger heavy computation
    const largeDataset = generateLargeDataset(10000, 20000)
    cy.mockWebSocketMessage(largeDataset)
    
    cy.get('[data-testid="web-workers-unavailable"]')
      .should('be.visible')
    
    cy.get('[data-testid="main-thread-processing"]')
      .should('be.visible')
      .and('contain.text', 'Processing on main thread')
    
    // Should still work but with performance warning
    cy.get('[data-testid="performance-warning"]')
      .should('contain.text', 'Reduced performance')
  })

  it('should handle IndexedDB unavailable', () => {
    cy.window().then((win) => {
      // Mock IndexedDB not available
      delete win.indexedDB
    })
    
    cy.reload()
    
    cy.get('[data-testid="indexeddb-unavailable"]')
      .should('be.visible')
    
    cy.get('[data-testid="fallback-storage-warning"]')
      .should('contain.text', 'Limited offline capabilities')
    
    // Should fallback to localStorage
    cy.get('[data-testid="using-localstorage-fallback"]')
      .should('be.visible')
  })
})
```

## Custom Commands for Error Testing

```javascript
// cypress/support/error-testing-commands.js

Cypress.Commands.add('simulateNetworkFailure', (errorType) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_NETWORK_FAILURE',
      errorType: errorType
    }, '*')
  })
})

Cypress.Commands.add('simulateIntermittentConnection', (outageMs) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_INTERMITTENT_CONNECTION',
      outageDuration: outageMs
    }, '*')
  })
})

Cypress.Commands.add('simulateWebSocketError', (errorType, code) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_WEBSOCKET_ERROR',
      errorType: errorType,
      code: code
    }, '*')
  })
})

Cypress.Commands.add('simulateServerUnavailable', () => {
  cy.intercept('**', { statusCode: 503 }).as('serverUnavailable')
})

Cypress.Commands.add('sendRawWebSocketMessage', (message) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SEND_RAW_WEBSOCKET_MESSAGE',
      message: message
    }, '*')
  })
})

Cypress.Commands.add('triggerAPICall', (endpoint) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'TRIGGER_API_CALL',
      endpoint: endpoint
    }, '*')
  })
})

Cypress.Commands.add('mockWebSocketConnectionFailure', (errorType) => {
  cy.window().then((win) => {
    win.__mockWebSocketShouldFail = true
    win.__mockWebSocketError = errorType
  })
})

Cypress.Commands.add('forceComponentError', (componentId, error) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'FORCE_COMPONENT_ERROR',
      componentId: componentId,
      error: error
    }, '*')
  })
})

Cypress.Commands.add('simulateBrowserLimits', (limitType, value) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_BROWSER_LIMITS',
      limitType: limitType,
      value: value
    }, '*')
  })
})

Cypress.Commands.add('validateErrorRecovery', (componentSelector) => {
  // Check if component has error boundary
  cy.get(`${componentSelector} [data-testid*="error"]`)
    .should('exist')
  
  // Check if retry mechanism exists
  cy.get(`${componentSelector} [data-testid*="retry"]`)
    .should('be.visible')
    .click()
  
  // Verify recovery
  cy.get(`${componentSelector} [data-testid*="error"]`)
    .should('not.exist')
})

Cypress.Commands.add('checkErrorLogging', (expectedErrorType) => {
  cy.window().then((win) => {
    const errorLog = win.__errorLog || []
    const hasExpectedError = errorLog.some(error => 
      error.type === expectedErrorType
    )
    expect(hasExpectedError).to.be.true
  })
})

Cypress.Commands.add('verifyGracefulDegradation', (feature, fallback) => {
  cy.get(`[data-testid="${feature}-unavailable"]`)
    .should('be.visible')
  
  cy.get(`[data-testid="${fallback}-active"]`)
    .should('be.visible')
  
  // Verify functionality still works
  cy.get('[data-testid="dashboard-container"]')
    .should('be.visible')
})
```

## Error Test Fixtures

```json
// cypress/fixtures/error-scenarios.json
{
  "networkErrors": [
    {
      "type": "CONNECTION_REFUSED",
      "message": "ECONNREFUSED",
      "expectedBehavior": "Show offline mode"
    },
    {
      "type": "TIMEOUT",
      "message": "ETIMEDOUT", 
      "expectedBehavior": "Retry with backoff"
    },
    {
      "type": "DNS_FAILURE",
      "message": "ENOTFOUND",
      "expectedBehavior": "Show connection help"
    }
  ],
  "invalidData": [
    {
      "name": "null_entities",
      "data": { "entities": null },
      "expectedValidation": "entities_required"
    },
    {
      "name": "invalid_activation",
      "data": { "entities": [{ "activation": "invalid" }] },
      "expectedValidation": "activation_numeric"
    },
    {
      "name": "missing_id",
      "data": { "entities": [{ "activation": 0.5 }] },
      "expectedValidation": "id_required"
    }
  ]
}
```

## Success Criteria

### Phase 6 Completion Requirements
- [ ] All network failure scenarios handled gracefully
- [ ] Data validation and sanitization working
- [ ] Error boundaries catching component failures
- [ ] Browser resource limits handled
- [ ] Cross-browser compatibility verified
- [ ] Error logging and reporting functional
- [ ] Recovery mechanisms working

### Error Handling Quality Gates
- No unhandled errors reach the browser console
- All error states provide clear user feedback
- Recovery mechanisms restore full functionality
- Error logging captures sufficient diagnostic info
- Graceful degradation maintains core functionality
- User data is never corrupted during errors

### Recovery Time Requirements
- WebSocket reconnection: < 10 seconds
- Component error recovery: < 5 seconds
- Resource limit handling: Immediate
- API failure fallback: < 2 seconds

## Dependencies for Next Phase
- Error handling mechanisms validated
- Recovery procedures working
- Graceful degradation confirmed
- Error logging functional
- User feedback systems operational

This phase ensures the dashboard handles all error conditions gracefully and maintains reliability in production environments before final end-to-end workflow testing in Phase 7.