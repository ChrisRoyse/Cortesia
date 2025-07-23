# Phase 7: End-to-End User Workflows

**Duration**: 3-4 days  
**Priority**: Critical - User experience validation  
**Focus**: Complete user journeys and workflow validation  
**Prerequisites**: Phases 1-6 completed successfully

## Objectives
- Test complete user workflows from start to finish
- Validate user experience across all dashboard features
- Ensure smooth transitions between different use cases
- Test realistic user scenarios and edge cases
- Validate accessibility throughout complete workflows
- Ensure consistent performance across extended usage

## Test Categories

### 7.1 Data Scientist/Researcher Workflows

#### 7.1.1 Brain Analysis Discovery Workflow
```javascript
describe('Brain Analysis Discovery Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support complete brain analysis discovery session', () => {
    // Step 1: User arrives and reviews system overview
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
    cy.get('[data-testid="connection-status"]').should('contain.text', 'Connected')
    
    // Check overall system health
    cy.get('[data-testid="system-health-indicator"]').should('be.visible')
    cy.get('[data-testid="entity-count"]').should('be.visible')
    
    // Step 2: Explore brain structure in 3D
    cy.switchTab('brain-graph')
    
    // Load research dataset
    cy.fixture('research-brain-dataset.json').then((dataset) => {
      cy.mockWebSocketMessage(dataset)
    })
    
    cy.get('[data-testid="scene-ready"]', { timeout: 10000 }).should('be.visible')
    
    // Explore the 3D structure
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousedown', { button: 0, clientX: 400, clientY: 300 })
      .trigger('mousemove', { clientX: 500, clientY: 200 })
      .trigger('mouseup')
    
    // Zoom in for detailed inspection
    cy.get('[data-testid="zoom-in-button"]').click().click()
    
    // Search for specific entities of interest
    cy.get('[data-testid="entity-search-input"]').type('cortex')
    cy.get('[data-testid="search-result-item"]').first().click()
    
    // Step 3: Analyze activation patterns
    cy.switchTab('neural-activity')
    
    // Examine heatmap patterns
    cy.get('[data-testid="heatmap-container"]').should('be.visible')
    
    // Create brush selection for detailed analysis
    cy.get('[data-testid="heatmap-svg"]')
      .trigger('mousedown', { clientX: 100, clientY: 100 })
      .trigger('mousemove', { clientX: 300, clientY: 250 })
      .trigger('mouseup')
    
    cy.get('[data-testid="selected-cells-panel"]').should('be.visible')
    cy.get('[data-testid="selection-statistics"]').should('be.visible')
    
    // Filter by activation threshold
    cy.get('[data-testid="activation-filter-slider"]')
      .invoke('val', 0.7)
      .trigger('input')
    
    // Step 4: Investigate cognitive patterns
    cy.switchTab('cognitive-systems')
    
    // Analyze cognitive pattern radar chart
    cy.get('[data-testid="radar-chart-svg"]').should('be.visible')
    
    // Click on convergent thinking pattern for details
    cy.get('[data-testid="radar-point-convergent"]').click()
    cy.get('[data-testid="pattern-detail-panel"]').should('be.visible')
    
    // Enable comparison mode to see historical patterns
    cy.get('[data-testid="comparison-mode-toggle"]').click()
    cy.get('[data-testid="historical-radar-polygon"]').should('be.visible')
    
    // Step 5: Examine memory systems
    cy.switchTab('memory')
    
    // Investigate working memory utilization
    cy.get('[data-testid="buffer-visualization-svg"]').should('be.visible')
    
    // Hover over visual buffer for details
    cy.get('[data-testid="buffer-bar-visual"]').trigger('mouseover')
    cy.get('[data-testid="buffer-tooltip"]').should('be.visible')
    
    // Explore SDR patterns
    cy.get('[data-testid="sdr-pattern-grid"]').should('be.visible')
    cy.get('[data-testid="sdr-bit"]').first().click()
    cy.get('[data-testid="bit-detail-panel"]').should('be.visible')
    
    // Step 6: Generate insights summary
    cy.get('[data-testid="generate-report-button"]').click()
    cy.get('[data-testid="analysis-report"]', { timeout: 5000 }).should('be.visible')
    
    // Verify complete workflow took reasonable time
    cy.then(() => {
      const workflowEnd = Date.now()
      const workflowStart = Cypress.env('workflowStart') || workflowEnd
      const duration = workflowEnd - workflowStart
      expect(duration).to.be.lessThan(300000) // 5 minutes max
    })
  })

  it('should maintain context across tab switches during analysis', () => {
    // Load dataset and establish baseline
    cy.fixture('analysis-dataset.json').then((dataset) => {
      cy.mockWebSocketMessage(dataset)
    })
    
    // Select entity in brain graph
    cy.switchTab('brain-graph')
    cy.get('[data-testid="scene-ready"]', { timeout: 10000 }).should('be.visible')
    cy.get('[data-testid="three-canvas"]').click(400, 300)
    
    cy.get('[data-testid="selected-entity-id"]').then(($el) => {
      const selectedEntityId = $el.text()
      
      // Switch to neural activity - should maintain selection context
      cy.switchTab('neural-activity')
      cy.get(`[data-testid="heatmap-cell-${selectedEntityId}"]`)
        .should('have.class', 'selected')
      
      // Switch to cognitive systems - should show related patterns
      cy.switchTab('cognitive-systems')
      cy.get('[data-testid="entity-related-patterns"]')
        .should('contain.text', selectedEntityId)
      
      // Return to brain graph - selection should persist
      cy.switchTab('brain-graph')
      cy.get('[data-testid="selected-entity-panel"]').should('be.visible')
      cy.get('[data-testid="selected-entity-id"]').should('contain.text', selectedEntityId)
    })
  })
})
```

#### 7.1.2 Comparative Analysis Workflow
```javascript
describe('Comparative Analysis Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support before/after analysis workflow', () => {
    // Step 1: Load initial baseline dataset
    cy.fixture('baseline-brain-data.json').then((baselineData) => {
      cy.mockWebSocketMessage(baselineData)
    })
    
    // Capture baseline metrics
    cy.switchTab('overview')
    cy.get('[data-testid="entity-count"]').then(($el) => {
      const baselineEntityCount = $el.text()
      cy.wrap(baselineEntityCount).as('baselineEntities')
    })
    
    cy.get('[data-testid="avg-activation"]').then(($el) => {
      const baselineActivation = $el.text()
      cy.wrap(baselineActivation).as('baselineActivation')
    })
    
    // Step 2: Take snapshot for comparison
    cy.get('[data-testid="take-snapshot-button"]').click()
    cy.get('[data-testid="snapshot-saved"]').should('be.visible')
    
    // Step 3: Load modified dataset (after intervention)
    cy.fixture('modified-brain-data.json').then((modifiedData) => {
      cy.mockWebSocketMessage(modifiedData)
    })
    
    // Step 4: Enable comparison mode
    cy.get('[data-testid="comparison-mode-toggle"]').click()
    cy.get('[data-testid="comparison-active"]').should('be.visible')
    
    // Step 5: Analyze differences in brain graph
    cy.switchTab('brain-graph')
    cy.get('[data-testid="comparison-overlay"]').should('be.visible')
    cy.get('[data-testid="changed-entities-highlight"]').should('be.visible')
    
    // Step 6: Examine activation pattern changes
    cy.switchTab('neural-activity')
    cy.get('[data-testid="heatmap-comparison-mode"]').should('be.visible')
    cy.get('[data-testid="activation-diff-legend"]').should('be.visible')
    
    // Step 7: Review cognitive pattern changes
    cy.switchTab('cognitive-systems')
    cy.get('[data-testid="pattern-comparison-chart"]').should('be.visible')
    
    // Step 8: Generate comparison report
    cy.get('[data-testid="generate-comparison-report"]').click()
    cy.get('[data-testid="comparison-report"]', { timeout: 5000 }).should('be.visible')
    
    // Verify significant changes are highlighted
    cy.get('[data-testid="significant-changes"]').should('be.visible')
    cy.get('[data-testid="statistical-significance"]').should('be.visible')
  })
})
```

### 7.2 System Administrator Workflows

#### 7.2.1 System Monitoring and Troubleshooting Workflow
```javascript
describe('System Administrator Monitoring Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support complete system health monitoring workflow', () => {
    // Step 1: Initial system health check
    cy.get('[data-testid="system-health-indicator"]').should('be.visible')
    
    // Verify all services are online
    cy.get('[data-testid="api-status-indicator"]')
      .should('have.class', 'status-online')
    cy.get('[data-testid="websocket-status-indicator"]')
      .should('have.class', 'status-online')
    
    // Step 2: Review performance metrics
    cy.switchTab('overview')
    
    cy.get('[data-testid="cpu-usage-card"]').should('be.visible')
    cy.get('[data-testid="memory-usage-card"]').should('be.visible')
    cy.get('[data-testid="network-latency-card"]').should('be.visible')
    
    // Check for any warnings or critical alerts
    cy.get('[data-testid="performance-alerts"]').should('be.visible')
    
    // Step 3: Simulate system stress and monitor response
    cy.mockWebSocketMessage({
      type: 'system_stress_simulation',
      data: {
        cpu: 85,
        memory: 90,
        networkLatency: 150
      }
    })
    
    // Verify warning states appear
    cy.get('[data-testid="cpu-usage-card"]')
      .should('have.class', 'status-warning')
    cy.get('[data-testid="memory-usage-card"]')
      .should('have.class', 'status-critical')
    
    // Step 4: Check error monitoring
    cy.switchTab('errors')
    
    // Simulate some errors
    cy.mockWebSocketMessage({
      type: 'error_event',
      data: {
        type: 'WebSocketConnectionError',
        message: 'Temporary connection issue',
        timestamp: Date.now(),
        severity: 'warning'
      }
    })
    
    cy.get('[data-testid="error-log-entry"]').should('be.visible')
    cy.get('[data-testid="error-severity-warning"]').should('be.visible')
    
    // Step 5: Investigate API flow issues
    cy.switchTab('api-flow')
    
    cy.get('[data-testid="api-endpoint-status"]').should('be.visible')
    cy.get('[data-testid="request-rate-chart"]').should('be.visible')
    
    // Check for any failed requests
    cy.get('[data-testid="failed-requests-indicator"]').should('be.visible')
    
    // Step 6: Generate system health report
    cy.get('[data-testid="generate-health-report"]').click()
    cy.get('[data-testid="health-report"]', { timeout: 5000 }).should('be.visible')
    
    cy.get('[data-testid="recommendations-section"]').should('be.visible')
    cy.get('[data-testid="action-items"]').should('be.visible')
  })

  it('should handle system recovery workflow', () => {
    // Step 1: Simulate system failure
    cy.simulateServerUnavailable()
    
    cy.get('[data-testid="server-unavailable-message"]')
      .should('be.visible')
    
    // Step 2: Enter diagnostic mode
    cy.get('[data-testid="diagnostic-mode-button"]').click()
    cy.get('[data-testid="diagnostic-panel"]').should('be.visible')
    
    // Step 3: Run connection tests
    cy.get('[data-testid="test-connectivity-button"]').click()
    cy.get('[data-testid="connectivity-test-results"]', { timeout: 10000 })
      .should('be.visible')
    
    // Step 4: Check service status
    cy.get('[data-testid="check-services-button"]').click()
    cy.get('[data-testid="service-status-list"]').should('be.visible')
    
    // Step 5: Attempt recovery
    cy.allowWebSocketConnection()
    cy.get('[data-testid="retry-connection-button"]').click()
    
    cy.get('[data-testid="connection-status"]', { timeout: 15000 })
      .should('contain.text', 'Connected')
    
    // Step 6: Verify system recovery
    cy.get('[data-testid="system-recovery-confirmation"]')
      .should('be.visible')
    
    cy.get('[data-testid="post-recovery-health-check"]').click()
    cy.get('[data-testid="health-check-passed"]').should('be.visible')
  })
})
```

### 7.3 Developer/Debugging Workflows

#### 7.3.1 Performance Analysis Workflow
```javascript
describe('Developer Performance Analysis Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support complete performance debugging workflow', () => {
    // Step 1: Enable performance monitoring
    cy.get('[data-testid="developer-tools-button"]').click()
    cy.get('[data-testid="enable-performance-monitoring"]').click()
    
    cy.get('[data-testid="performance-monitoring-active"]')
      .should('be.visible')
    
    // Step 2: Load performance test dataset
    cy.fixture('performance-test-dataset.json').then((dataset) => {
      cy.mockWebSocketMessage(dataset)
    })
    
    // Step 3: Monitor rendering performance
    cy.switchTab('brain-graph')
    cy.startFrameRateMonitoring()
    
    cy.get('[data-testid="scene-ready"]', { timeout: 15000 }).should('be.visible')
    
    // Perform interaction stress test
    for (let i = 0; i < 10; i++) {
      cy.get('[data-testid="three-canvas"]')
        .trigger('mousedown', { button: 0, clientX: 300 + i * 10, clientY: 300 })
        .trigger('mousemove', { clientX: 400 + i * 10, clientY: 200 })
        .trigger('mouseup')
      cy.wait(100)
    }
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      cy.log(`Average FPS during stress test: ${avgFps}`)
    })
    
    // Step 4: Analyze memory usage
    cy.get('[data-testid="memory-profiler"]').click()
    cy.get('[data-testid="memory-usage-chart"]').should('be.visible')
    
    // Take memory snapshot
    cy.get('[data-testid="take-memory-snapshot"]').click()
    cy.get('[data-testid="memory-snapshot-analysis"]').should('be.visible')
    
    // Step 5: Check for performance bottlenecks
    cy.switchTab('analytics')
    
    cy.get('[data-testid="performance-bottlenecks"]').should('be.visible')
    cy.get('[data-testid="render-time-analysis"]').should('be.visible')
    cy.get('[data-testid="memory-leak-detection"]').should('be.visible')
    
    // Step 6: Generate performance report
    cy.get('[data-testid="generate-performance-report"]').click()
    cy.get('[data-testid="performance-report"]', { timeout: 5000 })
      .should('be.visible')
    
    cy.get('[data-testid="optimization-suggestions"]').should('be.visible')
    cy.get('[data-testid="performance-metrics-summary"]').should('be.visible')
  })

  it('should support real-time debugging workflow', () => {
    // Step 1: Enable debug mode
    cy.get('[data-testid="debug-mode-toggle"]').click()
    cy.get('[data-testid="debug-panel"]').should('be.visible')
    
    // Step 2: Monitor WebSocket messages
    cy.get('[data-testid="websocket-debugger"]').click()
    cy.get('[data-testid="message-log"]').should('be.visible')
    
    // Send test message and verify logging
    cy.mockWebSocketMessage({
      type: 'debug_test_message',
      data: { test: 'debugging' }
    })
    
    cy.get('[data-testid="message-log-entry"]').should('be.visible')
    cy.get('[data-testid="message-type-debug_test_message"]').should('be.visible')
    
    // Step 3: Test error injection
    cy.get('[data-testid="error-injection-panel"]').click()
    
    cy.get('[data-testid="inject-websocket-error"]').click()
    cy.get('[data-testid="error-boundary-triggered"]').should('be.visible')
    
    // Step 4: Verify error recovery
    cy.get('[data-testid="clear-errors-button"]').click()
    cy.get('[data-testid="error-boundary-triggered"]').should('not.exist')
    
    // Step 5: Export debug session
    cy.get('[data-testid="export-debug-session"]').click()
    cy.get('[data-testid="debug-export-complete"]').should('be.visible')
  })
})
```

### 7.4 Multi-User Collaboration Workflows

#### 7.4.1 Shared Analysis Session Workflow
```javascript
describe('Shared Analysis Session Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support collaborative analysis workflow', () => {
    // Step 1: Start collaboration session
    cy.get('[data-testid="collaboration-button"]').click()
    cy.get('[data-testid="start-session-button"]').click()
    
    cy.get('[data-testid="session-id"]').then(($el) => {
      const sessionId = $el.text()
      cy.wrap(sessionId).as('sessionId')
    })
    
    // Step 2: Share session link
    cy.get('[data-testid="copy-session-link"]').click()
    cy.get('[data-testid="link-copied-confirmation"]').should('be.visible')
    
    // Step 3: Simulate second user joining
    cy.window().then((win) => {
      win.postMessage({
        type: 'SIMULATE_USER_JOIN',
        userId: 'user_2',
        username: 'Collaborator'
      }, '*')
    })
    
    cy.get('[data-testid="user-joined-notification"]')
      .should('contain.text', 'Collaborator joined')
    
    // Step 4: Perform collaborative selection
    cy.switchTab('brain-graph')
    cy.fixture('collaborative-dataset.json').then((dataset) => {
      cy.mockWebSocketMessage(dataset)
    })
    
    cy.get('[data-testid="scene-ready"]', { timeout: 10000 }).should('be.visible')
    
    // Select entity as user 1
    cy.get('[data-testid="three-canvas"]').click(400, 300)
    
    // Simulate user 2 making selection
    cy.window().then((win) => {
      win.postMessage({
        type: 'SIMULATE_COLLABORATIVE_SELECTION',
        userId: 'user_2',
        entityId: 'entity_5',
        coordinates: { x: 450, y: 350 }
      }, '*')
    })
    
    // Verify both selections are visible
    cy.get('[data-testid="selection-user-1"]').should('be.visible')
    cy.get('[data-testid="selection-user-2"]').should('be.visible')
    
    // Step 5: Share annotations
    cy.get('[data-testid="add-annotation-button"]').click()
    cy.get('[data-testid="annotation-text"]').type('Interesting activation pattern here')
    cy.get('[data-testid="save-annotation"]').click()
    
    cy.get('[data-testid="annotation-marker"]').should('be.visible')
    
    // Step 6: Synchronize views
    cy.get('[data-testid="sync-views-button"]').click()
    cy.get('[data-testid="views-synchronized"]').should('be.visible')
    
    // Step 7: Export collaborative session
    cy.get('[data-testid="export-session-button"]').click()
    cy.get('[data-testid="session-export-options"]').should('be.visible')
    
    cy.get('[data-testid="include-annotations"]').check()
    cy.get('[data-testid="include-selections"]').check()
    cy.get('[data-testid="confirm-export"]').click()
    
    cy.get('[data-testid="export-complete"]').should('be.visible')
  })
})
```

### 7.5 Accessibility and Assistive Technology Workflows

#### 7.5.1 Screen Reader User Workflow
```javascript
describe('Screen Reader User Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should support complete screen reader navigation workflow', () => {
    // Step 1: Enable screen reader mode
    cy.get('[data-testid="accessibility-menu"]').click()
    cy.get('[data-testid="screen-reader-mode"]').click()
    
    cy.get('[data-testid="screen-reader-active"]').should('be.visible')
    
    // Step 2: Navigate using keyboard only
    cy.get('body').tab() // Focus first element
    
    // Navigate through main navigation
    cy.focused().should('have.attr', 'data-testid', 'tab-overview')
    cy.focused().should('have.attr', 'aria-label')
    
    // Use arrow keys to navigate tabs
    cy.focused().type('{rightarrow}')
    cy.focused().should('have.attr', 'data-testid', 'tab-brain-graph')
    
    cy.focused().type('{enter}') // Activate brain graph tab
    
    // Step 3: Access graph description
    cy.get('[data-testid="graph-description"]')
      .should('be.visible')
      .and('have.attr', 'role', 'region')
      .and('have.attr', 'aria-label', 'Brain graph description')
    
    // Step 4: Navigate data table alternative
    cy.get('[data-testid="data-table-view"]').click()
    cy.get('[data-testid="entities-table"]')
      .should('be.visible')
      .and('have.attr', 'role', 'table')
    
    // Navigate table with keyboard
    cy.get('[data-testid="entities-table"] tbody tr').first().focus()
    cy.focused().type('{downarrow}') // Move to next row
    
    // Step 5: Access detailed entity information
    cy.focused().type('{enter}') // Select entity
    cy.get('[data-testid="entity-details-modal"]')
      .should('be.visible')
      .and('have.attr', 'role', 'dialog')
    
    // Verify modal is properly announced
    cy.get('[data-testid="modal-title"]')
      .should('have.attr', 'id')
    cy.get('[data-testid="entity-details-modal"]')
      .should('have.attr', 'aria-labelledby')
    
    // Step 6: Navigate to neural activity with keyboard
    cy.get('[data-testid="entity-details-modal"]').type('{esc}') // Close modal
    
    cy.get('[data-testid="tab-neural-activity"]').focus().type('{enter}')
    
    // Access heatmap alternative
    cy.get('[data-testid="heatmap-data-table"]')
      .should('be.visible')
      .and('have.attr', 'aria-label', 'Neural activation data')
    
    // Step 7: Use filtering with keyboard
    cy.get('[data-testid="activation-filter"]')
      .focus()
      .type('{rightarrow}'.repeat(5)) // Adjust filter
    
    cy.get('[data-testid="filter-applied-announcement"]')
      .should('have.attr', 'aria-live', 'polite')
  })

  it('should provide comprehensive audio descriptions', () => {
    // Enable audio descriptions
    cy.get('[data-testid="accessibility-menu"]').click()
    cy.get('[data-testid="audio-descriptions"]').click()
    
    cy.get('[data-testid="audio-descriptions-active"]').should('be.visible')
    
    // Navigate to brain graph
    cy.switchTab('brain-graph')
    cy.fixture('accessibility-dataset.json').then((dataset) => {
      cy.mockWebSocketMessage(dataset)
    })
    
    // Verify audio description is provided
    cy.get('[data-testid="audio-description"]')
      .should('be.visible')
      .and('contain.text', 'Brain graph with')
      .and('contain.text', 'entities and')
      .and('contain.text', 'relationships')
    
    // Test dynamic updates
    cy.mockWebSocketMessage({
      type: 'entity_update',
      data: { entity_id: 'test', activation: 0.9 }
    })
    
    cy.get('[data-testid="update-announcement"]')
      .should('have.attr', 'aria-live', 'assertive')
      .and('contain.text', 'Entity activation updated')
  })
})
```

### 7.6 Performance Under Extended Usage

#### 7.6.1 Long Session Stability Workflow
```javascript
describe('Extended Usage Stability Workflow', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should maintain performance during 30-minute session', () => {
    cy.startPerformanceMonitoring()
    
    const sessionDuration = 5 * 60 * 1000 // 5 minutes for testing (30 min in production)
    const updateInterval = 1000 // 1 second
    const totalUpdates = sessionDuration / updateInterval
    
    // Simulate extended session with regular data updates
    for (let i = 0; i < totalUpdates; i++) {
      cy.then(() => {
        setTimeout(() => {
          cy.mockWebSocketMessage({
            type: 'brain_metrics_update',
            data: {
              entityCount: 1000 + Math.floor(Math.random() * 100),
              avgActivation: 0.3 + Math.random() * 0.4,
              timestamp: Date.now()
            }
          })
        }, i * updateInterval)
      })
      
      // Periodically switch tabs to simulate user behavior
      if (i % 30 === 0) {
        const tabs = ['overview', 'brain-graph', 'neural-activity', 'cognitive-systems', 'memory']
        const randomTab = tabs[Math.floor(Math.random() * tabs.length)]
        cy.switchTab(randomTab)
      }
      
      // Periodically interact with visualizations
      if (i % 60 === 0) {
        cy.get('[data-testid="three-canvas"]')
          .trigger('mousemove', { 
            clientX: 300 + Math.random() * 200, 
            clientY: 200 + Math.random() * 200 
          })
      }
    }
    
    cy.wait(sessionDuration + 5000) // Wait for all updates plus buffer
    
    cy.stopPerformanceMonitoring().then((metrics) => {
      expect(metrics.averageFPS).to.be.greaterThan(25) // Acceptable degradation
      expect(metrics.maxMemoryUsage).to.be.lessThan(500 * 1024 * 1024) // 500MB max
    })
    
    // Verify UI is still responsive
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
    cy.get('[data-testid="connection-status"]').should('contain.text', 'Connected')
  })

  it('should handle memory cleanup during extended usage', () => {
    cy.measureMemoryUsage('initial').then((initialMemory) => {
      
      // Simulate memory-intensive operations
      for (let cycle = 0; cycle < 20; cycle++) {
        // Load large dataset
        const largeDataset = generateLargeDataset(2000, 4000)
        cy.mockWebSocketMessage(largeDataset)
        
        // Switch between visualizations
        cy.switchTab('brain-graph')
        cy.wait(500)
        cy.switchTab('neural-activity')
        cy.wait(500)
        cy.switchTab('overview')
        cy.wait(500)
        
        // Force garbage collection if available
        if (cycle % 5 === 0) {
          cy.window().then((win) => {
            if (win.gc && typeof win.gc === 'function') {
              win.gc()
            }
          })
        }
      }
      
      cy.measureMemoryUsage('final').then((finalMemory) => {
        const memoryIncrease = finalMemory - initialMemory
        expect(memoryIncrease).to.be.lessThan(100 * 1024 * 1024) // 100MB max increase
      })
    })
  })
})
```

## Custom Commands for E2E Testing

```javascript
// cypress/support/e2e-commands.js

Cypress.Commands.add('completeUserWorkflow', (workflowType, options = {}) => {
  const workflows = {
    'data-scientist': () => {
      cy.switchTab('overview')
      cy.get('[data-testid="system-health-indicator"]').should('be.visible')
      
      cy.switchTab('brain-graph')
      cy.get('[data-testid="scene-ready"]', { timeout: 10000 }).should('be.visible')
      
      cy.get('[data-testid="entity-search-input"]').type('cortex')
      cy.get('[data-testid="search-result-item"]').first().click()
      
      cy.switchTab('neural-activity')
      cy.get('[data-testid="heatmap-container"]').should('be.visible')
      
      cy.switchTab('cognitive-systems')
      cy.get('[data-testid="radar-chart-svg"]').should('be.visible')
    },
    
    'system-admin': () => {
      cy.switchTab('overview')
      cy.get('[data-testid="system-health-indicator"]').should('be.visible')
      
      cy.switchTab('errors')
      cy.get('[data-testid="error-log"]').should('be.visible')
      
      cy.switchTab('api-flow')
      cy.get('[data-testid="api-endpoint-status"]').should('be.visible')
    },
    
    'developer': () => {
      cy.get('[data-testid="developer-tools-button"]').click()
      cy.get('[data-testid="enable-performance-monitoring"]').click()
      
      cy.switchTab('analytics')
      cy.get('[data-testid="performance-bottlenecks"]').should('be.visible')
    }
  }
  
  if (workflows[workflowType]) {
    workflows[workflowType]()
  }
})

Cypress.Commands.add('simulateExtendedUsage', (durationMinutes) => {
  const updates = durationMinutes * 60 // One update per second
  const tabs = ['overview', 'brain-graph', 'neural-activity', 'cognitive-systems', 'memory']
  
  for (let i = 0; i < updates; i++) {
    cy.then(() => {
      setTimeout(() => {
        // Send data update
        cy.mockWebSocketMessage({
          type: 'brain_metrics_update',
          data: {
            entityCount: 1000 + Math.floor(Math.random() * 100),
            avgActivation: Math.random(),
            timestamp: Date.now()
          }
        })
        
        // Occasionally switch tabs
        if (i % 60 === 0) {
          const randomTab = tabs[Math.floor(Math.random() * tabs.length)]
          cy.switchTab(randomTab)
        }
      }, i * 1000)
    })
  }
})

Cypress.Commands.add('validateAccessibilityWorkflow', () => {
  // Enable accessibility features
  cy.get('[data-testid="accessibility-menu"]').click()
  cy.get('[data-testid="screen-reader-mode"]').click()
  
  // Test keyboard navigation
  cy.get('body').tab()
  cy.focused().type('{rightarrow}')
  cy.focused().type('{enter}')
  
  // Verify ARIA labels and roles
  cy.get('[role="main"]').should('exist')
  cy.get('[role="navigation"]').should('exist')
  cy.get('[aria-live]').should('exist')
})

Cypress.Commands.add('validateCollaborativeWorkflow', () => {
  // Start collaboration
  cy.get('[data-testid="collaboration-button"]').click()
  cy.get('[data-testid="start-session-button"]').click()
  
  // Simulate second user
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_USER_JOIN',
      userId: 'test_user_2'
    }, '*')
  })
  
  cy.get('[data-testid="user-joined-notification"]').should('be.visible')
  
  // Test collaborative selection
  cy.switchTab('brain-graph')
  cy.get('[data-testid="three-canvas"]').click(400, 300)
  
  cy.get('[data-testid="collaborative-selection"]').should('be.visible')
})

// Helper function for generating test datasets
function generateLargeDataset(entityCount, relationshipCount) {
  return {
    entities: Array.from({ length: entityCount }, (_, i) => ({
      id: `entity_${i}`,
      type_id: (i % 4) + 1,
      activation: Math.random(),
      direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4]
    })),
    relationships: Array.from({ length: relationshipCount }, (_, i) => ({
      from: `entity_${Math.floor(Math.random() * entityCount)}`,
      to: `entity_${Math.floor(Math.random() * entityCount)}`,
      weight: Math.random()
    }))
  }
}
```

## E2E Test Configuration

```javascript
// cypress/support/e2e-config.js

export const WORKFLOW_TIMEOUTS = {
  short: 30000,    // 30 seconds
  medium: 120000,  // 2 minutes
  long: 300000,    // 5 minutes
  extended: 1800000 // 30 minutes
}

export const USER_PERSONAS = {
  dataScientist: {
    name: 'Data Scientist',
    primaryTabs: ['brain-graph', 'neural-activity', 'cognitive-systems'],
    workflows: ['analysis', 'comparison', 'discovery']
  },
  systemAdmin: {
    name: 'System Administrator', 
    primaryTabs: ['overview', 'errors', 'api-flow'],
    workflows: ['monitoring', 'troubleshooting', 'maintenance']
  },
  developer: {
    name: 'Developer',
    primaryTabs: ['analytics', 'architecture', 'errors'],
    workflows: ['debugging', 'performance', 'optimization']
  }
}

export const ACCESSIBILITY_REQUIREMENTS = {
  keyboardNavigation: true,
  screenReaderSupport: true,
  colorContrastCompliant: true,
  ariaLabelsComplete: true,
  focusManagement: true
}
```

## Success Criteria

### Phase 7 Completion Requirements
- [ ] All user persona workflows completed successfully
- [ ] Extended usage stability verified
- [ ] Accessibility workflows functional
- [ ] Collaborative features working
- [ ] Performance maintained during realistic usage
- [ ] Error recovery in real-world scenarios
- [ ] Complete user journeys documented

### User Experience Quality Gates
- Workflow completion time within reasonable limits
- No blocking errors during normal usage
- Consistent performance across extended sessions
- Accessible to users with disabilities
- Collaborative features enable team workflows
- Recovery from failures is intuitive

### Performance Requirements During E2E
- Memory usage stable over 30-minute sessions
- Frame rate maintained above 30 FPS
- Response times under 100ms for interactions
- No memory leaks detected
- Graceful degradation under load

## Final Validation

This phase represents the culmination of the testing strategy, ensuring that:

1. **Real Users Can Accomplish Their Goals**: All major user personas can complete their intended workflows
2. **System Reliability**: The dashboard remains stable and performant under realistic usage patterns
3. **Accessibility**: The system is usable by people with various abilities and assistive technologies
4. **Collaboration**: Multiple users can work together effectively
5. **Production Readiness**: The system is ready for deployment and real-world usage

The completion of Phase 7 indicates that the LLMKG dashboard has been thoroughly tested and validated for production deployment.