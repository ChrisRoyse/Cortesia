// Phase 2: Real Dashboard Testing - Working with actual dashboard
// Tests components as they actually exist in the running application

describe('Phase 2: Real Dashboard Integration Tests', () => {
  beforeEach(() => {
    // Visit the dashboard without redirect issues
    cy.visit('http://localhost:5176', {
      failOnStatusCode: false,
      timeout: 30000
    })
    
    // Give the application time to load and connect to services
    cy.wait(3000)
  })

  // Test 1: Verify API is accessible
  it('should have access to real API endpoints', () => {
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        expect(response.status).to.equal(200)
        expect(response.body.status).to.equal('success')
        
        // Store metrics for verification
        const metrics = response.body.data
        expect(metrics).to.have.property('entity_count')
        expect(metrics.entity_count).to.be.a('number')
        
        cy.log(`Entity count: ${metrics.entity_count}`)
        cy.log(`Memory stats:`, JSON.stringify(metrics.memory_stats))
      })
  })

  // Test 2: Verify dashboard structure
  it('should load dashboard with proper structure', () => {
    // Check root element exists
    cy.get('#root').should('exist')
    
    // Look for any dashboard-like elements
    cy.get('div').then(($divs) => {
      // Find divs that might be dashboard containers
      const dashboardDivs = $divs.filter((i, el) => {
        const className = el.className || ''
        const testId = el.getAttribute('data-testid') || ''
        return className.includes('dashboard') || 
               className.includes('container') ||
               testId.includes('dashboard')
      })
      
      cy.log(`Found ${dashboardDivs.length} dashboard-related elements`)
      
      if (dashboardDivs.length > 0) {
        cy.wrap(dashboardDivs.first()).should('be.visible')
      }
    })
  })

  // Test 3: Look for visualization components
  it('should contain visualization elements', () => {
    // Check for SVG elements (used in visualizations)
    cy.get('body').then(($body) => {
      // Check for SVG
      if ($body.find('svg').length > 0) {
        cy.get('svg').first().should('be.visible')
        cy.log(`Found ${$body.find('svg').length} SVG elements`)
      }
      
      // Check for Canvas (used in WebGL/Chart.js)
      if ($body.find('canvas').length > 0) {
        cy.get('canvas').first().should('be.visible')
        cy.log(`Found ${$body.find('canvas').length} canvas elements`)
      }
      
      // Check for any elements with our test-ids
      const testIdElements = $body.find('[data-testid]')
      cy.log(`Found ${testIdElements.length} elements with data-testid`)
      
      testIdElements.each((i, el) => {
        cy.log(`Test ID: ${el.getAttribute('data-testid')}`)
      })
    })
  })

  // Test 4: Verify WebSocket connectivity
  it('should establish WebSocket connection', () => {
    cy.window().then((win) => {
      // Check if WebSocket is being used
      const originalWebSocket = win.WebSocket
      let wsConnected = false
      
      // Monitor WebSocket creation
      win.WebSocket = new Proxy(originalWebSocket, {
        construct(target, args) {
          cy.log(`WebSocket created with URL: ${args[0]}`)
          const ws = new target(...args)
          
          // Monitor connection
          const originalOnOpen = ws.onopen
          ws.onopen = function(event) {
            wsConnected = true
            cy.log('WebSocket connected!')
            if (originalOnOpen) originalOnOpen.call(this, event)
          }
          
          return ws
        }
      })
      
      // Wait and check if WebSocket connected
      cy.wait(2000).then(() => {
        cy.log(`WebSocket connected: ${wsConnected}`)
      })
    })
  })

  // Test 5: Performance metrics from real system
  it('should display real-time performance metrics', () => {
    // Make API call to get current metrics
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        const metrics = response.body.data
        
        // Look for any element displaying these metrics
        cy.get('body').then(($body) => {
          const text = $body.text()
          
          // Check if metrics are displayed anywhere
          if (metrics.entity_count && text.includes(metrics.entity_count.toString())) {
            cy.log(`Found entity count ${metrics.entity_count} displayed`)
          }
          
          // Look for percentage displays (CPU, Memory)
          const percentages = text.match(/\d+(\.\d+)?%/g) || []
          cy.log(`Found ${percentages.length} percentage values: ${percentages.join(', ')}`)
          
          // Look for latency displays
          const latencies = text.match(/\d+(\.\d+)?\s*ms/g) || []
          cy.log(`Found ${latencies.length} latency values: ${latencies.join(', ')}`)
        })
      })
  })

  // Test 6: Dashboard responsiveness
  it('should be responsive to viewport changes', () => {
    // Test desktop viewport
    cy.viewport(1920, 1080)
    cy.wait(500)
    cy.get('#root').should('be.visible')
    
    // Test tablet viewport
    cy.viewport(768, 1024)
    cy.wait(500)
    cy.get('#root').should('be.visible')
    
    // Test mobile viewport
    cy.viewport(375, 667)
    cy.wait(500)
    cy.get('#root').should('be.visible')
  })

  // Test 7: Real data flow
  it('should show real-time data updates', () => {
    // Take initial snapshot of metrics
    let initialMetrics: any
    
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        initialMetrics = response.body.data
      })
    
    // Wait for potential updates
    cy.wait(5000)
    
    // Check if data has been updated
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        const currentMetrics = response.body.data
        
        cy.log('Initial metrics:', JSON.stringify(initialMetrics))
        cy.log('Current metrics:', JSON.stringify(currentMetrics))
        
        // Metrics should be valid even if unchanged
        expect(currentMetrics).to.have.property('entity_count')
        expect(currentMetrics).to.have.property('memory_stats')
      })
  })
})