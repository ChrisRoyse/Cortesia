// Visualization Testing Commands for LLMKG Dashboard

// Chart and Graph Validation
Cypress.Commands.add('validateChartRendering', (chartSelector: string) => {
  cy.get(chartSelector, { timeout: 10000 }).should('be.visible')
  
  // Check if chart has rendered content
  cy.get(chartSelector).within(() => {
    // Look for common chart elements
    cy.get('svg, canvas, .recharts-wrapper, .chart-container').should('exist')
  })
  
  // Validate chart data points
  cy.get(chartSelector).then(($chart) => {
    if ($chart.find('svg').length > 0) {
      // SVG-based chart (D3, Recharts)
      cy.get(`${chartSelector} svg`).should('have.attr', 'width').and('not.be.empty')
      cy.get(`${chartSelector} svg`).should('have.attr', 'height').and('not.be.empty')
    } else if ($chart.find('canvas').length > 0) {
      // Canvas-based chart (Chart.js, Three.js)
      cy.get(`${chartSelector} canvas`).should('have.attr', 'width').and('not.be.empty')
      cy.get(`${chartSelector} canvas`).should('have.attr', 'height').and('not.be.empty')
    }
  })
})

Cypress.Commands.add('checkDataVisualization', (containerSelector: string, expectedDataPoints: number) => {
  cy.get(containerSelector).should('be.visible')
  
  // Wait for data to load
  cy.get(containerSelector).should('not.contain', 'Loading...')
  cy.get(containerSelector).should('not.contain', 'No data available')
  
  // Check for data elements based on visualization type
  cy.get(containerSelector).then(($container) => {
    // For SVG visualizations
    if ($container.find('svg').length > 0) {
      cy.get(`${containerSelector} svg`).within(() => {
        // Count data elements (circles, bars, paths, etc.)
        cy.get('circle, rect, path, line').should('have.length.at.least', 1)
      })
    }
    
    // For canvas visualizations
    if ($container.find('canvas').length > 0) {
      cy.get(`${containerSelector} canvas`).should('exist')
      
      // Validate canvas has content by checking if it's not blank
      cy.get(`${containerSelector} canvas`).then(($canvas) => {
        const canvas = $canvas[0] as HTMLCanvasElement
        const ctx = canvas.getContext('2d')
        if (ctx) {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
          const data = imageData.data
          
          // Check if canvas has non-transparent pixels
          let hasContent = false
          for (let i = 3; i < data.length; i += 4) {
            if (data[i] > 0) {
              hasContent = true
              break
            }
          }
          expect(hasContent).to.be.true
        }
      })
    }
  })
})

// 3D Visualization Commands
Cypress.Commands.add('simulate3DInteraction', (canvasSelector: string, interaction: {
  type: 'click' | 'drag' | 'zoom' | 'rotate'
  coordinates?: { x: number, y: number }
  deltaX?: number
  deltaY?: number
  zoomLevel?: number
}) => {
  cy.get(canvasSelector).should('be.visible')
  
  switch (interaction.type) {
    case 'click':
      if (interaction.coordinates) {
        cy.get(canvasSelector).click(interaction.coordinates.x, interaction.coordinates.y)
      } else {
        cy.get(canvasSelector).click()
      }
      break
      
    case 'drag':
      if (interaction.coordinates && interaction.deltaX && interaction.deltaY) {
        cy.get(canvasSelector)
          .trigger('mousedown', interaction.coordinates.x, interaction.coordinates.y)
          .trigger('mousemove', interaction.coordinates.x + interaction.deltaX, interaction.coordinates.y + interaction.deltaY)
          .trigger('mouseup')
      }
      break
      
    case 'zoom':
      if (interaction.zoomLevel) {
        cy.get(canvasSelector)
          .trigger('wheel', { deltaY: interaction.zoomLevel > 1 ? -100 : 100 })
      }
      break
      
    case 'rotate':
      if (interaction.deltaX && interaction.deltaY) {
        cy.get(canvasSelector)
          .trigger('mousedown', { which: 1 })
          .trigger('mousemove', { clientX: interaction.deltaX, clientY: interaction.deltaY })
          .trigger('mouseup')
      }
      break
  }
  
  // Wait for interaction to take effect
  cy.wait(500)
})

// Knowledge Graph Specific Commands
Cypress.Commands.add('validateKnowledgeGraph', (graphSelector: string) => {
  cy.get(graphSelector).should('be.visible')
  
  // Check for Three.js canvas
  cy.get(`${graphSelector} canvas`).should('exist')
  
  // Validate graph has rendered entities and relationships
  cy.window().then((win) => {
    // Access the Three.js scene if available
    if (win.__threeJSScene) {
      const scene = win.__threeJSScene
      expect(scene.children).to.have.length.greaterThan(0)
    }
  })
  
  // Check for graph controls
  cy.get(graphSelector).within(() => {
    cy.get('[data-testid*="graph-controls"], .graph-controls').should('exist')
  })
})

Cypress.Commands.add('testGraphFiltering', (graphSelector: string, filterOptions: {
  activationThreshold?: number
  nodeType?: string
  searchTerm?: string
}) => {
  cy.get(graphSelector).should('be.visible')
  
  if (filterOptions.activationThreshold !== undefined) {
    cy.get('[data-testid="activation-threshold-slider"]')
      .should('be.visible')
      .invoke('val', filterOptions.activationThreshold)
      .trigger('input')
    
    cy.wait(1000) // Wait for filter to apply
  }
  
  if (filterOptions.nodeType) {
    cy.get('[data-testid="node-type-filter"]')
      .should('be.visible')
      .select(filterOptions.nodeType)
    
    cy.wait(1000)
  }
  
  if (filterOptions.searchTerm) {
    cy.get('[data-testid="graph-search-input"]')
      .should('be.visible')
      .clear()
      .type(filterOptions.searchTerm)
    
    cy.wait(1000)
  }
})

// Heatmap Validation
Cypress.Commands.add('validateHeatmap', (heatmapSelector: string) => {
  cy.get(heatmapSelector).should('be.visible')
  
  // Check for heatmap grid
  cy.get(heatmapSelector).within(() => {
    cy.get('svg, canvas, .heatmap-grid').should('exist')
    
    // Validate color scale/legend
    cy.get('.color-scale, .legend, [data-testid*="legend"]').should('exist')
  })
  
  // Test heatmap tooltips
  cy.get(heatmapSelector).within(() => {
    cy.get('rect, .heatmap-cell').first().trigger('mouseover')
    cy.get('.tooltip, [data-testid="tooltip"]').should('be.visible')
    cy.get('rect, .heatmap-cell').first().trigger('mouseout')
  })
})

// Time Series Chart Validation
Cypress.Commands.add('validateTimeSeriesChart', (chartSelector: string) => {
  cy.get(chartSelector).should('be.visible')
  
  // Check for time axis
  cy.get(chartSelector).within(() => {
    cy.get('.x-axis, .time-axis, [data-testid*="time-axis"]').should('exist')
    cy.get('.y-axis, [data-testid*="y-axis"]').should('exist')
  })
  
  // Validate data lines/points
  cy.get(chartSelector).within(() => {
    cy.get('path, line, circle, .data-point').should('have.length.at.least', 1)
  })
  
  // Test chart interactions
  cy.get(chartSelector).trigger('mousemove', 100, 100)
  cy.get('.tooltip, .crosshair, [data-testid="chart-tooltip"]').should('be.visible')
})

// Dashboard Layout Validation
Cypress.Commands.add('validateDashboardLayout', () => {
  // Check responsive grid layout
  cy.get('[data-testid="dashboard-grid"]').should('be.visible')
  
  // Validate all dashboard cards are visible
  cy.get('[data-testid*="dashboard-card"]').should('have.length.at.least', 1)
  cy.get('[data-testid*="dashboard-card"]').each(($card) => {
    cy.wrap($card).should('be.visible')
  })
  
  // Test responsive behavior
  cy.viewport(768, 1024) // Tablet
  cy.get('[data-testid="dashboard-grid"]').should('be.visible')
  
  cy.viewport(375, 667) // Mobile
  cy.get('[data-testid="dashboard-grid"]').should('be.visible')
  
  cy.viewport(1920, 1080) // Desktop
})

// Performance Metrics Visualization
Cypress.Commands.add('validatePerformanceMetrics', (metricsSelector: string) => {
  cy.get(metricsSelector).should('be.visible')
  
  // Check for metric cards
  cy.get(metricsSelector).within(() => {
    cy.get('[data-testid*="metric-card"]').should('have.length.at.least', 1)
    
    // Validate each metric has a value
    cy.get('[data-testid*="metric-value"]').each(($value) => {
      cy.wrap($value).should('not.be.empty')
    })
  })
  
  // Check for trend indicators
  cy.get(metricsSelector).within(() => {
    cy.get('.trend-up, .trend-down, .trend-neutral, [data-testid*="trend"]').should('exist')
  })
})

// WebGL Rendering Validation
Cypress.Commands.add('validateWebGLRendering', (canvasSelector: string) => {
  cy.get(canvasSelector).should('be.visible')
  
  cy.get(canvasSelector).then(($canvas) => {
    const canvas = $canvas[0] as HTMLCanvasElement
    
    // Check WebGL context
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
    expect(gl).to.not.be.null
    
    if (gl) {
      // Validate WebGL capabilities
      const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE)
      expect(maxTextureSize).to.be.greaterThan(0)
      
      const maxVertexAttribs = gl.getParameter(gl.MAX_VERTEX_ATTRIBS)
      expect(maxVertexAttribs).to.be.greaterThan(0)
    }
  })
})

// Animation Testing
Cypress.Commands.add('validateAnimations', (containerSelector: string) => {
  cy.get(containerSelector).should('be.visible')
  
  // Check for CSS animations
  cy.get(containerSelector).within(() => {
    cy.get('[class*="animate"], [class*="transition"]').should('exist')
  })
  
  // Test animation performance
  cy.window().then((win) => {
    const startTime = performance.now()
    
    // Trigger animations
    cy.get(containerSelector).trigger('mouseenter')
    cy.wait(500)
    
    const endTime = performance.now()
    const duration = endTime - startTime
    
    // Ensure animations don't cause significant performance issues
    expect(duration).to.be.lessThan(1000) // Less than 1 second
  })
})

// Data Export Validation
Cypress.Commands.add('testDataExport', (exportButtonSelector: string, expectedFormat: 'json' | 'csv' | 'png' | 'svg') => {
  cy.get(exportButtonSelector).should('be.visible').click()
  
  // Depending on format, validate different behaviors
  switch (expectedFormat) {
    case 'json':
    case 'csv':
      // Should trigger file download
      cy.readFile('cypress/downloads/*.' + expectedFormat, { timeout: 10000 })
        .should('exist')
      break
      
    case 'png':
    case 'svg':
      // Should trigger image download or show in new tab
      cy.get('a[download], img[src*="blob:"]').should('exist')
      break
  }
})

// Real-time Data Update Validation
Cypress.Commands.add('validateRealTimeUpdates', (containerSelector: string) => {
  cy.get(containerSelector).should('be.visible')
  
  // Get initial state
  cy.get(containerSelector).invoke('text').as('initialText')
  
  // Simulate real-time data update
  cy.mockWebSocketMessage({
    type: 'brain_update',
    data: {
      entities: [
        { id: 'test_entity', activation: Math.random() }
      ]
    }
  })
  
  // Verify data has updated
  cy.get(containerSelector).should(($container) => {
    const currentText = $container.text()
    // The text should have changed (activation values should be different)
    expect(currentText).to.not.equal(cy.get('@initialText'))
  })
})

declare global {
  namespace Cypress {
    interface Chainable {
      // Chart validation
      validateChartRendering(chartSelector: string): Chainable<void>
      checkDataVisualization(containerSelector: string, expectedDataPoints: number): Chainable<void>
      
      // 3D interactions
      simulate3DInteraction(canvasSelector: string, interaction: {
        type: 'click' | 'drag' | 'zoom' | 'rotate'
        coordinates?: { x: number, y: number }
        deltaX?: number
        deltaY?: number
        zoomLevel?: number
      }): Chainable<void>
      
      // Knowledge graph
      validateKnowledgeGraph(graphSelector: string): Chainable<void>
      testGraphFiltering(graphSelector: string, filterOptions: {
        activationThreshold?: number
        nodeType?: string
        searchTerm?: string
      }): Chainable<void>
      
      // Specialized visualizations
      validateHeatmap(heatmapSelector: string): Chainable<void>
      validateTimeSeriesChart(chartSelector: string): Chainable<void>
      validatePerformanceMetrics(metricsSelector: string): Chainable<void>
      
      // Layout and responsiveness
      validateDashboardLayout(): Chainable<void>
      
      // WebGL and animations
      validateWebGLRendering(canvasSelector: string): Chainable<void>
      validateAnimations(containerSelector: string): Chainable<void>
      
      // Data operations
      testDataExport(exportButtonSelector: string, expectedFormat: 'json' | 'csv' | 'png' | 'svg'): Chainable<void>
      validateRealTimeUpdates(containerSelector: string): Chainable<void>
    }
  }
}