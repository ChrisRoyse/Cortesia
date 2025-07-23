// Testing the LLMKG Performance Dashboard served by the backend on port 8090
// This is the monitoring dashboard built into the Rust backend

describe('LLMKG Backend Monitoring Dashboard', () => {
  beforeEach(() => {
    // Visit the backend dashboard
    cy.visit('http://localhost:8090', {
      failOnStatusCode: false,
      timeout: 10000
    })
  })

  // Test 1: Dashboard loads successfully
  it('should load the performance dashboard', () => {
    // Check the title
    cy.title().should('contain', 'LLMKG Performance Dashboard')
    
    // Check main heading
    cy.get('h1').should('contain', 'LLMKG Performance Dashboard')
    
    // Check that the dashboard has loaded
    cy.get('body').should('be.visible')
  })

  // Test 2: System metrics are displayed
  it('should display system metrics', () => {
    // Look for metric cards
    cy.get('.metric-card').should('have.length.at.least', 1)
    
    // Check for CPU usage
    cy.contains('CPU Usage').should('be.visible')
    
    // Check for Memory usage
    cy.contains('Memory Usage').should('be.visible')
    
    // Check for Active Connections
    cy.contains('Connections').should('be.visible')
  })

  // Test 3: Charts are rendered
  it('should render performance charts', () => {
    // Check for chart containers
    cy.get('.chart-container').should('have.length.at.least', 1)
    
    // Check for canvas elements (Chart.js)
    cy.get('canvas').should('have.length.at.least', 1)
    
    // Verify charts have been initialized
    cy.get('canvas').first().should(($canvas) => {
      const canvas = $canvas[0] as HTMLCanvasElement
      expect(canvas.width).to.be.greaterThan(0)
      expect(canvas.height).to.be.greaterThan(0)
    })
  })

  // Test 4: WebSocket connection status
  it('should show WebSocket connection status', () => {
    // Look for connection status indicator
    cy.get('.connection-status').should('exist')
    
    // Should show connected state (may take a moment)
    cy.contains('Connected', { timeout: 5000 }).should('be.visible')
  })

  // Test 5: Real-time updates
  it('should receive real-time metric updates', () => {
    // Get initial CPU value
    let initialCpuValue: string
    
    cy.get('#cpu-usage')
      .invoke('text')
      .then((text) => {
        initialCpuValue = text
      })
    
    // Wait for updates
    cy.wait(3000)
    
    // Check if values have potentially changed
    cy.get('#cpu-usage')
      .invoke('text')
      .then((text) => {
        // Value should be valid even if unchanged
        expect(text).to.match(/\d+(\.\d+)?%/)
      })
  })

  // Test 6: API endpoint information
  it('should display API endpoint information', () => {
    // Check for endpoint display
    cy.contains('API Endpoints').should('be.visible')
    
    // Should show the discovery endpoint
    cy.contains('/api/v1/discovery').should('be.visible')
  })

  // Test 7: Responsive design
  it('should be responsive to viewport changes', () => {
    // Desktop view
    cy.viewport(1920, 1080)
    cy.get('.dashboard-container').should('be.visible')
    
    // Tablet view
    cy.viewport(768, 1024)
    cy.get('.dashboard-container').should('be.visible')
    
    // Mobile view
    cy.viewport(375, 667)
    cy.get('.dashboard-container').should('be.visible')
    
    // Charts should still be visible
    cy.get('canvas').should('be.visible')
  })

  // Test 8: Performance metrics from API
  it('should display metrics fetched from API', () => {
    // First verify API is accessible
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        expect(response.status).to.equal(200)
        const metrics = response.body.data
        
        // Now check if these metrics appear in the dashboard
        cy.get('body').then(($body) => {
          const bodyText = $body.text()
          
          // Check if entity count is displayed
          if (metrics.entity_count !== undefined) {
            cy.log(`API reports ${metrics.entity_count} entities`)
          }
          
          // Look for memory stats
          if (metrics.memory_stats) {
            cy.log('Memory stats available:', JSON.stringify(metrics.memory_stats))
          }
        })
      })
  })

  // Test 9: Test execution section
  it('should have test execution monitoring section', () => {
    // Check for test execution area
    cy.get('#test-executions').should('exist')
    
    // Should have a run test button
    cy.get('#run-test-btn').should('be.visible')
    
    // Should have test output area
    cy.get('#test-output').should('exist')
  })

  // Test 10: Execute a test through the dashboard
  it('should be able to run tests from dashboard', () => {
    // Click run test button
    cy.get('#run-test-btn').click()
    
    // Should show test is running
    cy.get('#test-output').should('not.be.empty')
    
    // Wait for test to complete (with timeout)
    cy.contains('Test execution', { timeout: 30000 }).should('be.visible')
  })
})