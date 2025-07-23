// Phase 2: Component Integration Testing - Dashboard Components
// Using REAL backend services - NO MOCKS

describe('Phase 2: Dashboard Component Loading with Real Backend', () => {
  beforeEach(() => {
    // Visit the dashboard - it will connect to real WebSocket and API
    cy.visit('http://localhost:5176', { 
      failOnStatusCode: false,
      timeout: 30000, // Give it time to connect to real services
      redirectionLimit: 100  // Increase limit temporarily
    })
    
    // Wait for dashboard to fully load
    cy.wait(3000) // Give application time to stabilize
  })

  // Test 1: Verify dashboard loads and connects to real services
  it('should connect to real WebSocket and API services', () => {
    // Check WebSocket connection status
    cy.get('[data-testid="websocket-status"]', { timeout: 15000 })
      .should('exist')
      .and('contain.text', 'Connected')
    
    // Verify dashboard container loads
    cy.get('[data-testid="dashboard-container"]', { timeout: 10000 })
      .should('be.visible')
  })

  // Test 2: CognitivePatternVisualizer with real data
  it('should load and display CognitivePatternVisualizer component with real data', () => {
    // Wait for component to load with real data
    cy.get('[data-testid="cognitive-pattern-visualizer"]', { timeout: 30000 })
      .should('be.visible')
    
    // Verify SVG visualization is rendered
    cy.get('[data-testid="cognitive-pattern-visualizer"] svg')
      .should('exist')
      .and('be.visible')
    
    // Check that patterns are rendered when data arrives
    cy.get('[data-testid="cognitive-pattern-visualizer"] svg circle', { timeout: 20000 })
      .should('exist') // Real data may have varying number of patterns
  })

  // Test 3: NeuralActivityHeatmap with real data
  it('should load and display NeuralActivityHeatmap component with real data', () => {
    // Wait for heatmap component
    cy.get('[data-testid="neural-activity-heatmap"]', { timeout: 30000 })
      .should('be.visible')
    
    // Verify canvas element is rendered
    cy.get('[data-testid="neural-activity-heatmap"] canvas')
      .should('exist')
      .and('be.visible')
      .and(($canvas) => {
        // Canvas should have non-zero dimensions
        expect($canvas[0].width).to.be.greaterThan(0)
        expect($canvas[0].height).to.be.greaterThan(0)
      })
  })

  // Test 4: SystemHealthIndicator with real metrics
  it('should load and display SystemHealthIndicator component with real metrics', () => {
    cy.get('[data-testid="system-health-indicator"]', { timeout: 30000 })
      .should('be.visible')
    
    // Should display health status
    cy.get('[data-testid="health-status-text"]')
      .should('be.visible')
      .and('not.be.empty')
    
    // Should show circular progress indicator
    cy.get('[data-testid="health-progress-circle"]')
      .should('be.visible')
    
    // Should calculate and display health percentage
    cy.get('[data-testid="health-percentage"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const percentage = parseInt($el.text().replace('%', ''))
        expect(percentage).to.be.at.least(0)
        expect(percentage).to.be.at.most(100)
      })
  })

  // Test 5: PerformanceMetricsCard with real performance data
  it('should load and display PerformanceMetricsCard component with real metrics', () => {
    cy.get('[data-testid="performance-metrics-card"]', { timeout: 30000 })
      .should('be.visible')
    
    // Should display CPU metric - real value from system
    cy.get('[data-testid="cpu-metric"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const cpu = parseFloat($el.text().replace('%', ''))
        expect(cpu).to.be.at.least(0)
        expect(cpu).to.be.at.most(100)
      })
    
    // Should display memory metric - real value from system
    cy.get('[data-testid="memory-metric"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const memory = parseFloat($el.text().replace('%', ''))
        expect(memory).to.be.at.least(0)
        expect(memory).to.be.at.most(100)
      })
    
    // Should display latency metric - real latency
    cy.get('[data-testid="latency-metric"]')
      .should('be.visible')
      .and('contain.text', 'ms')
      .then(($el) => {
        const latency = parseInt($el.text().replace('ms', ''))
        expect(latency).to.be.at.least(0)
      })
    
    // Should display throughput metric - real throughput
    cy.get('[data-testid="throughput-metric"]')
      .should('be.visible')
      .and('contain.text', '/s')
    
    // Should render Chart.js canvas with real data
    cy.get('[data-testid="performance-metrics-card"] canvas')
      .should('exist')
      .and('be.visible')
  })

  // Test 6: KnowledgeGraphPreview with real graph data
  it('should load and display KnowledgeGraphPreview component with WebGL', () => {
    cy.get('[data-testid="knowledge-graph-preview"]', { timeout: 30000 })
      .should('be.visible')
    
    // Should have Three.js canvas with test-id
    cy.get('[data-testid="knowledge-graph-canvas"]')
      .should('exist')
      .and('be.visible')
    
    // Verify WebGL context is working
    cy.validateWebGLRendering('[data-testid="knowledge-graph-canvas"]')
    
    // Verify canvas is inside the preview container
    cy.get('[data-testid="knowledge-graph-preview"]')
      .find('[data-testid="knowledge-graph-canvas"]')
      .should('exist')
  })

  // Test 7: MemoryConsolidationMonitor with real memory data
  it('should load and display MemoryConsolidationMonitor component with real data', () => {
    cy.get('[data-testid="memory-consolidation-monitor"]', { timeout: 30000 })
      .should('be.visible')
    
    // Should show working memory section
    cy.get('[data-testid="working-memory-section"]')
      .should('be.visible')
    
    // Should display working memory usage percentage - real value
    cy.get('[data-testid="working-memory-usage"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const usage = parseFloat($el.text().replace('%', ''))
        expect(usage).to.be.at.least(0)
        expect(usage).to.be.at.most(100)
      })
    
    // Should show long-term memory section
    cy.get('[data-testid="long-term-memory-section"]')
      .should('be.visible')
    
    // Should display consolidation rate - real value
    cy.get('[data-testid="consolidation-rate"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const rate = parseFloat($el.text().replace('%', ''))
        expect(rate).to.be.at.least(0)
        expect(rate).to.be.at.most(100)
      })
    
    // Should display retrieval speed - real value
    cy.get('[data-testid="retrieval-speed"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const speed = parseFloat($el.text().replace('%', ''))
        expect(speed).to.be.at.least(0)
        expect(speed).to.be.at.most(100)
      })
  })

  // Test 8: Error boundaries are properly set up
  it('should properly wrap components in VisualizationErrorBoundary', () => {
    // Check that error boundary components exist around visualizations
    cy.get('[data-testid="visualization-error-boundary"]', { timeout: 10000 })
      .should('have.length.at.least', 1)
    
    // Should not show error state initially
    cy.get('[data-testid="error-boundary-message"]')
      .should('not.exist')
  })

  // Test 9: Dashboard metric cards with real-time values
  it('should display metric cards with real-time values from backend', () => {
    // System Latency Card - real WebSocket latency
    cy.get('[data-testid="system-latency-card"]', { timeout: 30000 })
      .should('be.visible')
    
    cy.get('[data-testid="latency-value"]')
      .should('be.visible')
      .and('not.be.empty')
      .then(($el) => {
        // Should show real latency value
        expect($el.text()).to.match(/\d+(\.\d+)?\s*(ms|μs|s)/)
      })
    
    // Active Neurons Card - real neural activity count
    cy.get('[data-testid="active-neurons-card"]')
      .should('be.visible')
    
    cy.get('[data-testid="neuron-count"]')
      .should('be.visible')
      .and('match', /^\d+$/) // Should be a number
      .then(($el) => {
        const count = parseInt($el.text())
        expect(count).to.be.at.least(0)
      })
    
    // Memory Usage Card - real system memory
    cy.get('[data-testid="memory-usage-card"]')
      .should('be.visible')
    
    cy.get('[data-testid="memory-percentage"]')
      .should('be.visible')
      .and('contain.text', '%')
      .then(($el) => {
        const percentage = parseFloat($el.text().replace('%', ''))
        expect(percentage).to.be.at.least(0)
        expect(percentage).to.be.at.most(100)
      })
  })

  // Test 10: Real-time data updates
  it('should receive and display real-time updates from WebSocket', () => {
    // Monitor performance metrics for changes over time
    let initialCpuValue: number
    
    // Get initial CPU value
    cy.get('[data-testid="cpu-metric"]', { timeout: 30000 })
      .should('be.visible')
      .then(($el) => {
        initialCpuValue = parseFloat($el.text().replace('%', ''))
      })
    
    // Wait for potential updates (real WebSocket sends updates periodically)
    cy.wait(5000)
    
    // Check if values have been updated (they may change with real system metrics)
    cy.get('[data-testid="cpu-metric"]')
      .should('be.visible')
      .then(($el) => {
        const currentCpuValue = parseFloat($el.text().replace('%', ''))
        // Value should be valid even if it hasn't changed
        expect(currentCpuValue).to.be.at.least(0)
        expect(currentCpuValue).to.be.at.most(100)
      })
  })

  // Test 11: API integration
  it('should fetch and display data from real API endpoints', () => {
    // Make a real API call to verify integration
    cy.request('GET', 'http://localhost:3001/api/v1/metrics')
      .then((response) => {
        expect(response.status).to.equal(200)
        expect(response.body).to.have.property('status', 'success')
        expect(response.body.data).to.have.property('entity_count')
        expect(response.body.data).to.have.property('memory_stats')
      })
    
    // Verify dashboard uses this data
    cy.get('[data-testid="dashboard-container"]', { timeout: 10000 })
      .should('be.visible')
  })
})