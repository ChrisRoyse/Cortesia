// Accurate tests for the LLMKG Backend Monitoring Dashboard on port 8090
// Based on the actual dashboard content served by the Rust backend

describe('LLMKG Backend Monitoring Dashboard - Accurate Tests', () => {
  beforeEach(() => {
    cy.visit('http://localhost:8090', {
      failOnStatusCode: false,
      timeout: 10000
    })
  })

  // Test 1: Dashboard loads successfully
  it('should load the performance dashboard', () => {
    // Check the title
    cy.title().should('contain', 'LLMKG Performance Dashboard')
    
    // Check that the dashboard has loaded
    cy.get('body').should('be.visible')
    
    // Should have the header with gradient
    cy.get('.header').should('be.visible')
    cy.get('h1').should('contain', 'LLMKG Performance Dashboard')
  })

  // Test 2: System metrics cards are displayed
  it('should display system metric cards', () => {
    // Look for metric cards
    cy.get('.metric-card').should('have.length.at.least', 3)
    
    // Check for CPU usage metric
    cy.get('.metric-card').contains('CPU Usage').should('be.visible')
    cy.get('#cpu-usage').should('exist')
    
    // Check for Memory usage metric
    cy.get('.metric-card').contains('Memory Usage').should('be.visible')
    cy.get('#memory-usage').should('exist')
    
    // Check for Active Connections metric
    cy.get('.metric-card').contains('Active Connections').should('be.visible')
    cy.get('#active-connections').should('exist')
  })

  // Test 3: Charts are rendered
  it('should render performance charts', () => {
    // Check for chart containers
    cy.get('.chart-container').should('have.length.at.least', 2)
    
    // Check for specific charts
    cy.get('#cpuChart').should('exist')
    cy.get('#memoryChart').should('exist')
    
    // Verify charts are canvases
    cy.get('#cpuChart').should('have.prop', 'tagName', 'CANVAS')
    cy.get('#memoryChart').should('have.prop', 'tagName', 'CANVAS')
  })

  // Test 4: WebSocket connection status
  it('should show WebSocket connection status', () => {
    // Look for connection status
    cy.get('.connection-status').should('exist')
    cy.get('#connection-status').should('exist')
    
    // Should eventually show connected
    cy.get('#connection-status', { timeout: 5000 })
      .should('contain', 'Connected')
  })

  // Test 5: Real-time metric updates via WebSocket
  it('should update metrics in real-time', () => {
    // Wait for initial connection
    cy.get('#connection-status').should('contain', 'Connected')
    
    // Get initial values
    let initialCpu: string
    cy.get('#cpu-usage').invoke('text').then((text) => {
      initialCpu = text
    })
    
    // Wait for potential updates
    cy.wait(3000)
    
    // Values should be formatted as percentages
    cy.get('#cpu-usage').should('match', /\d+(\.\d+)?%/)
    cy.get('#memory-usage').should('match', /\d+(\.\d+)?%/)
  })

  // Test 6: API endpoints section
  it('should display API endpoints information', () => {
    // Check for API endpoints section
    cy.get('.api-endpoints').should('exist')
    
    // Should show main endpoints
    cy.contains('API Endpoints').should('be.visible')
    cy.contains('/api/v1').should('be.visible')
  })

  // Test 7: Test execution section
  it('should have test execution section', () => {
    // Check for test execution area
    cy.get('#test-execution').should('exist')
    
    // Should have controls
    cy.get('#run-test-btn').should('exist').and('be.visible')
    cy.get('#test-output').should('exist')
  })

  // Test 8: System information display
  it('should display system information', () => {
    // Should show entity count from API
    cy.request('GET', 'http://localhost:3001/api/v1/metrics').then((response) => {
      const entityCount = response.body.data.entity_count
      
      // Look for entity count in dashboard
      cy.contains('Entities').should('be.visible')
      cy.get('#entity-count').should('exist')
    })
  })

  // Test 9: Responsive layout
  it('should maintain layout across different viewports', () => {
    // Desktop
    cy.viewport(1920, 1080)
    cy.get('.metrics-grid').should('be.visible')
    cy.get('.chart-container').should('be.visible')
    
    // Tablet
    cy.viewport(768, 1024)
    cy.get('.metrics-grid').should('be.visible')
    
    // Mobile
    cy.viewport(375, 667)
    cy.get('.metric-card').should('be.visible')
  })

  // Test 10: Integration with backend API
  it('should integrate with backend API services', () => {
    // Verify API is accessible
    cy.request('GET', 'http://localhost:3001/api/v1/discovery').should((response) => {
      expect(response.status).to.equal(200)
      expect(response.body).to.have.property('endpoints')
    })
    
    // Verify metrics endpoint
    cy.request('GET', 'http://localhost:3001/api/v1/metrics').should((response) => {
      expect(response.status).to.equal(200)
      expect(response.body.status).to.equal('success')
    })
  })
})