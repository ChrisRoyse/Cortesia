# Phase 4: Interactive Visualization Testing

**Duration**: 4-5 days  
**Priority**: High - User experience validation  
**Focus**: 3D brain graphs, D3.js visualizations, user interactions  
**Prerequisites**: Phases 1-3 completed successfully

## Objectives
- Test Three.js 3D brain graph interactions and performance
- Validate D3.js chart interactions and responsiveness
- Ensure proper user interaction feedback and visual cues
- Test complex visualization state management
- Verify accessibility and usability of interactive elements
- Validate performance under various interaction scenarios

## Test Categories

### 4.1 3D Brain Graph Visualization Testing

#### 4.1.1 Three.js Scene Rendering
```javascript
describe('3D Brain Graph Rendering', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
    cy.fixture('large-brain-dataset.json').as('brainData')
  })

  it('should initialize Three.js scene correctly', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="three-canvas"]')
      .should('be.visible')
      .and('have.attr', 'width')
      .and('have.attr', 'height')
    
    // Verify WebGL context
    cy.get('[data-testid="three-canvas"]').then(($canvas) => {
      const canvas = $canvas[0]
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      expect(gl).to.not.be.null
    })
  })

  it('should render entities as 3D objects with correct positioning', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    // Wait for 3D scene to initialize
    cy.wait(1000)
    
    cy.get('[data-testid="entity-count-rendered"]')
      .should('contain.text', this.brainData.entities.length.toString())
    
    // Verify entities are distributed in 3D space
    cy.window().then((win) => {
      const scene = win.__threeJsScene
      expect(scene).to.exist
      expect(scene.children.length).to.be.greaterThan(0)
    })
  })

  it('should apply correct colors based on entity direction', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // Test color legend
    cy.get('[data-testid="legend-input"]')
      .should('contain.text', 'Input')
      .and('have.css', 'color', 'rgb(33, 150, 243)') // Blue
    
    cy.get('[data-testid="legend-output"]')
      .should('contain.text', 'Output')
      .and('have.css', 'color', 'rgb(76, 175, 80)') // Green
    
    cy.get('[data-testid="legend-hidden"]')
      .should('contain.text', 'Hidden')
      .and('have.css', 'color', 'rgb(156, 39, 176)') // Purple
    
    cy.get('[data-testid="legend-gate"]')
      .should('contain.text', 'Gate')
      .and('have.css', 'color', 'rgb(255, 152, 0)') // Orange
  })

  it('should render relationships as connections between entities', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    cy.get('[data-testid="relationship-count-rendered"]')
      .should('contain.text', this.brainData.relationships.length.toString())
    
    // Verify inhibitory relationships are colored differently
    cy.get('[data-testid="inhibitory-connections-count"]')
      .should('be.visible')
  })

  it('should handle large datasets without performance degradation', function() {
    const largeBrainData = {
      entities: Array.from({ length: 1000 }, (_, i) => ({
        id: `entity_${i}`,
        type_id: (i % 4) + 1,
        activation: Math.random(),
        direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
        properties: { name: `Entity ${i}` }
      })),
      relationships: Array.from({ length: 2000 }, (_, i) => ({
        from: `entity_${Math.floor(i / 2)}`,
        to: `entity_${Math.floor(i / 2) + 1}`,
        weight: Math.random(),
        inhibitory: i % 10 === 0
      }))
    }
    
    const startTime = Date.now()
    cy.mockWebSocketMessage(largeBrainData)
    
    cy.get('[data-testid="scene-ready"]', { timeout: 10000 })
      .should('be.visible')
    
    cy.then(() => {
      const renderTime = Date.now() - startTime
      expect(renderTime).to.be.lessThan(5000) // Should render within 5 seconds
    })
    
    // Test frame rate with large dataset
    cy.measureFrameRate().then((fps) => {
      expect(fps).to.be.greaterThan(30) // Minimum 30 FPS
    })
  })
})
```

#### 4.1.2 3D Interaction Testing
```javascript
describe('3D Brain Graph Interactions', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').as('brainData')
  })

  it('should support mouse camera controls', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    const canvas = cy.get('[data-testid="three-canvas"]')
    
    // Test camera rotation (left mouse drag)
    canvas
      .trigger('mousedown', { button: 0, clientX: 400, clientY: 300 })
      .trigger('mousemove', { clientX: 500, clientY: 200 })
      .trigger('mouseup')
    
    cy.get('[data-testid="camera-position"]')
      .should('not.contain.text', 'Default')
    
    // Test camera zoom (mouse wheel)
    canvas.trigger('wheel', { deltaY: -100 })
    
    cy.get('[data-testid="zoom-level"]')
      .should('not.contain.text', '1.0x')
  })

  it('should support touch controls for mobile interaction', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    const canvas = cy.get('[data-testid="three-canvas"]')
    
    // Test pinch zoom
    canvas
      .trigger('touchstart', { 
        touches: [
          { clientX: 350, clientY: 300 },
          { clientX: 450, clientY: 300 }
        ]
      })
      .trigger('touchmove', {
        touches: [
          { clientX: 300, clientY: 300 },
          { clientX: 500, clientY: 300 }
        ]
      })
      .trigger('touchend')
    
    cy.get('[data-testid="zoom-level"]')
      .should('not.contain.text', '1.0x')
  })

  it('should highlight entities on hover', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // Simulate hover over entity (approximate screen coordinates)
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousemove', { clientX: 400, clientY: 300 })
    
    cy.get('[data-testid="entity-tooltip"]', { timeout: 2000 })
      .should('be.visible')
      .and('contain.text', 'Entity')
    
    cy.get('[data-testid="tooltip-activation"]')
      .should('be.visible')
    cy.get('[data-testid="tooltip-direction"]')
      .should('be.visible')
  })

  it('should select entities on click', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // Click on entity
    cy.get('[data-testid="three-canvas"]')
      .click(400, 300)
    
    cy.get('[data-testid="selected-entity-panel"]')
      .should('be.visible')
    
    cy.get('[data-testid="selected-entity-id"]')
      .should('not.be.empty')
    
    cy.get('[data-testid="selected-entity-properties"]')
      .should('be.visible')
  })

  it('should support multi-selection with Ctrl+click', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // First selection
    cy.get('[data-testid="three-canvas"]')
      .click(400, 300)
    
    // Multi-selection
    cy.get('[data-testid="three-canvas"]')
      .click(450, 350, { ctrlKey: true })
    
    cy.get('[data-testid="selected-entities-count"]')
      .should('contain.text', '2')
    
    cy.get('[data-testid="multi-selection-panel"]')
      .should('be.visible')
  })

  it('should support selection box drag', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // Drag selection box
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousedown', { button: 0, shiftKey: true, clientX: 350, clientY: 250 })
      .trigger('mousemove', { clientX: 450, clientY: 350 })
      .trigger('mouseup')
    
    cy.get('[data-testid="selection-box"]')
      .should('have.been.visible') // Should have appeared during drag
    
    cy.get('[data-testid="selected-entities-count"]')
      .should('not.contain.text', '0')
  })
})
```

#### 4.1.3 3D Visualization Controls Testing
```javascript
describe('3D Visualization Controls', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').as('brainData')
  })

  it('should filter entities by activation threshold', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    const initialCount = this.brainData.entities.length
    
    cy.get('[data-testid="activation-filter-slider"]')
      .invoke('val', 0.5)
      .trigger('input')
    
    cy.get('[data-testid="visible-entities-count"]')
      .should('not.contain.text', initialCount.toString())
    
    // Verify 3D scene updates
    cy.get('[data-testid="filtered-entities-indicator"]')
      .should('contain.text', 'Filtered')
  })

  it('should search and highlight entities', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    cy.get('[data-testid="entity-search-input"]')
      .type('entity_1')
    
    cy.get('[data-testid="search-results-list"]')
      .should('be.visible')
    
    cy.get('[data-testid="search-result-item"]')
      .first()
      .click()
    
    // Should highlight and focus on searched entity
    cy.get('[data-testid="highlighted-entity"]')
      .should('be.visible')
    
    cy.get('[data-testid="camera-focused"]')
      .should('contain.text', 'true')
  })

  it('should toggle relationship visibility', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    const initialRelationshipCount = this.brainData.relationships.length
    
    cy.get('[data-testid="toggle-relationships"]')
      .click()
    
    cy.get('[data-testid="visible-relationships-count"]')
      .should('contain.text', '0')
    
    cy.get('[data-testid="toggle-relationships"]')
      .click()
    
    cy.get('[data-testid="visible-relationships-count"]')
      .should('contain.text', initialRelationshipCount.toString())
  })

  it('should adjust entity size based on activation', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    cy.get('[data-testid="size-by-activation-toggle"]')
      .click()
    
    // Verify size adjustment is enabled
    cy.get('[data-testid="size-mode-indicator"]')
      .should('contain.text', 'Activation')
    
    // High activation entities should be larger
    cy.window().then((win) => {
      const scene = win.__threeJsScene
      const entities = scene.children.filter(child => child.userData?.type === 'entity')
      
      // Find entities with different activations
      const highActivationEntity = entities.find(e => e.userData.activation > 0.8)
      const lowActivationEntity = entities.find(e => e.userData.activation < 0.3)
      
      if (highActivationEntity && lowActivationEntity) {
        expect(highActivationEntity.scale.x).to.be.greaterThan(lowActivationEntity.scale.x)
      }
    })
  })

  it('should reset camera to default position', function() {
    cy.mockWebSocketMessage(this.brainData)
    cy.wait(1000)
    
    // Move camera away from default
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousedown', { button: 0, clientX: 400, clientY: 300 })
      .trigger('mousemove', { clientX: 500, clientY: 200 })
      .trigger('mouseup')
    
    cy.get('[data-testid="reset-camera-button"]')
      .click()
    
    cy.get('[data-testid="camera-position"]')
      .should('contain.text', 'Center')
    
    cy.get('[data-testid="zoom-level"]')
      .should('contain.text', '1.0x')
  })
})
```

### 4.2 D3.js Visualization Testing

#### 4.2.1 Neural Activation Heatmap Interactions
```javascript
describe('D3.js Neural Activation Heatmap', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('neural-activity')
    cy.fixture('neural-activity-data.json').as('neuralData')
  })

  it('should render interactive heatmap grid', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="heatmap-svg"]')
      .should('be.visible')
      .and('have.attr', 'width')
      .and('have.attr', 'height')
    
    cy.get('[data-testid="heatmap-cell"]')
      .should('have.length.greaterThan', 0)
  })

  it('should show cell details on hover', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .trigger('mouseover')
    
    cy.get('[data-testid="heatmap-tooltip"]')
      .should('be.visible')
      .and('contain.text', 'Entity:')
      .and('contain.text', 'Activation:')
      .and('contain.text', 'Layer:')
  })

  it('should highlight connected cells on hover', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .trigger('mouseover')
    
    // Should highlight related cells
    cy.get('[data-testid="heatmap-cell"].highlighted')
      .should('have.length.greaterThan', 1)
    
    cy.get('[data-testid="connection-lines"]')
      .should('be.visible')
  })

  it('should filter layers interactively', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="layer-filter-input"]')
      .click()
    
    cy.get('[data-testid="layer-option-hidden"]')
      .click()
    
    // Should only show hidden layer cells
    cy.get('[data-testid="heatmap-cell"][data-layer="Input"]')
      .should('not.be.visible')
    
    cy.get('[data-testid="heatmap-cell"][data-layer="Hidden"]')
      .should('be.visible')
  })

  it('should support brush selection for detailed analysis', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    // Create brush selection
    cy.get('[data-testid="heatmap-svg"]')
      .trigger('mousedown', { clientX: 100, clientY: 100 })
      .trigger('mousemove', { clientX: 200, clientY: 200 })
      .trigger('mouseup')
    
    cy.get('[data-testid="brush-selection"]')
      .should('be.visible')
    
    cy.get('[data-testid="selected-cells-panel"]')
      .should('be.visible')
    
    cy.get('[data-testid="selection-statistics"]')
      .should('contain.text', 'Selected')
  })

  it('should update in real-time with animation', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    // Send updated activation data
    const updatedData = { ...this.neuralData }
    updatedData.entities[0].activation = 0.95
    
    cy.mockWebSocketMessage(updatedData)
    
    // Should see color transition animation
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .should('have.class', 'updating')
    
    cy.wait(500) // Animation duration
    
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .should('have.css', 'fill')
      .and('not.equal', 'rgb(0, 0, 0)') // Should have updated color
  })
})
```

#### 4.2.2 Cognitive Pattern Radar Chart Testing
```javascript
describe('Cognitive Pattern Radar Chart', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('cognitive-systems')
    cy.fixture('cognitive-data.json').as('cognitiveData')
  })

  it('should render interactive radar chart', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="radar-chart-svg"]')
      .should('be.visible')
    
    cy.get('[data-testid="radar-axis"]')
      .should('have.length', 5) // 5 cognitive patterns
    
    cy.get('[data-testid="radar-polygon"]')
      .should('be.visible')
  })

  it('should show pattern details on axis hover', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="radar-axis-convergent"]')
      .trigger('mouseover')
    
    cy.get('[data-testid="pattern-tooltip"]')
      .should('be.visible')
      .and('contain.text', 'Convergent Thinking')
      .and('contain.text', 'Strength:')
      .and('contain.text', 'Confidence:')
  })

  it('should animate pattern strength changes', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    // Update pattern strength
    const updatedData = { ...this.cognitiveData }
    updatedData.patterns[0].strength = 0.9
    
    cy.mockWebSocketMessage(updatedData)
    
    // Should see smooth transition
    cy.get('[data-testid="radar-polygon"]')
      .should('have.class', 'transitioning')
    
    cy.wait(1000) // Animation duration
    
    cy.get('[data-testid="pattern-convergent-value"]')
      .should('contain.text', '0.9')
  })

  it('should support pattern comparison mode', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="comparison-mode-toggle"]')
      .click()
    
    // Should show historical pattern overlay
    cy.get('[data-testid="historical-radar-polygon"]')
      .should('be.visible')
    
    cy.get('[data-testid="comparison-legend"]')
      .should('contain.text', 'Current')
      .and('contain.text', 'Previous')
  })

  it('should allow pattern drill-down', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="radar-point-convergent"]')
      .click()
    
    cy.get('[data-testid="pattern-detail-panel"]')
      .should('be.visible')
    
    cy.get('[data-testid="pattern-history-chart"]')
      .should('be.visible')
    
    cy.get('[data-testid="pattern-resources-chart"]')
      .should('be.visible')
  })
})
```

#### 4.2.3 Memory System Visualizations Testing
```javascript
describe('Memory System D3 Visualizations', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('memory')
    cy.fixture('memory-data.json').as('memoryData')
  })

  it('should render working memory buffer visualization', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="buffer-visualization-svg"]')
      .should('be.visible')
    
    cy.get('[data-testid="buffer-visual"]')
      .should('be.visible')
    cy.get('[data-testid="buffer-verbal"]')
      .should('be.visible')
    cy.get('[data-testid="buffer-executive"]')
      .should('be.visible')
  })

  it('should show buffer utilization with interactive bars', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="buffer-bar-visual"]')
      .trigger('mouseover')
    
    cy.get('[data-testid="buffer-tooltip"]')
      .should('be.visible')
      .and('contain.text', 'Visual Buffer')
      .and('contain.text', 'Usage: 45%')
      .and('contain.text', 'Capacity: 100')
  })

  it('should visualize memory consolidation process', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="consolidation-flow-diagram"]')
      .should('be.visible')
    
    cy.get('[data-testid="consolidation-queue"]')
      .should('be.visible')
    
    cy.get('[data-testid="consolidation-progress"]')
      .should('have.attr', 'data-progress')
  })

  it('should render SDR pattern bit visualization', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="sdr-pattern-grid"]')
      .should('be.visible')
    
    cy.get('[data-testid="sdr-bit"]')
      .should('have.length.greaterThan', 0)
    
    // Test sparsity visualization
    cy.get('[data-testid="sdr-bit"].active')
      .then($activeBits => {
        const totalBits = this.memoryData.sdr.totalBits
        const activeBits = $activeBits.length
        const sparsity = activeBits / totalBits
        
        expect(sparsity).to.be.approximately(
          this.memoryData.sdr.averageSparsity, 0.01
        )
      })
  })

  it('should support interactive SDR pattern exploration', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="sdr-bit"]')
      .first()
      .click()
    
    cy.get('[data-testid="bit-detail-panel"]')
      .should('be.visible')
    
    cy.get('[data-testid="related-patterns"]')
      .should('be.visible')
    
    // Should highlight related bits
    cy.get('[data-testid="sdr-bit"].related')
      .should('have.length.greaterThan', 0)
  })
})
```

### 4.3 Chart Interaction and Performance Testing

#### 4.3.1 Chart Responsiveness Testing
```javascript
describe('Chart Responsiveness', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.fixture('performance-test-data.json').as('perfData')
  })

  it('should maintain 60fps during chart interactions', function() {
    cy.switchTab('neural-activity')
    cy.mockWebSocketMessage(this.perfData)
    
    cy.startFrameRateMonitoring()
    
    // Perform rapid interactions
    for (let i = 0; i < 10; i++) {
      cy.get('[data-testid="heatmap-cell"]')
        .eq(i % 5)
        .trigger('mouseover')
      cy.wait(50)
    }
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      expect(avgFps).to.be.greaterThan(55) // Allow some tolerance
    })
  })

  it('should handle rapid data updates without lag', function() {
    cy.switchTab('overview')
    
    cy.measureRenderTime(() => {
      // Send rapid data updates
      for (let i = 0; i < 20; i++) {
        const data = { ...this.perfData }
        data.performance.cpu = 30 + (i * 2)
        cy.mockWebSocketMessage(data)
      }
    }).then((renderTime) => {
      expect(renderTime).to.be.lessThan(200) // Should update within 200ms
    })
  })

  it('should maintain responsiveness with large datasets', function() {
    const largeDataset = {
      entities: Array.from({ length: 2000 }, (_, i) => ({
        id: `entity_${i}`,
        activation: Math.random(),
        direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4]
      }))
    }
    
    cy.switchTab('neural-activity')
    
    const startTime = Date.now()
    cy.mockWebSocketMessage(largeDataset)
    
    cy.get('[data-testid="heatmap-cell"]', { timeout: 10000 })
      .should('have.length', 2000)
    
    cy.then(() => {
      const renderTime = Date.now() - startTime
      expect(renderTime).to.be.lessThan(3000) // Should render within 3 seconds
    })
    
    // Test interaction responsiveness with large dataset
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .trigger('mouseover')
    
    cy.get('[data-testid="heatmap-tooltip"]', { timeout: 100 })
      .should('be.visible')
  })
})
```

#### 4.3.2 Memory and Resource Management
```javascript
describe('Visualization Memory Management', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should clean up Three.js resources when switching tabs', () => {
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    cy.wait(2000) // Allow 3D scene to fully load
    
    cy.window().then((win) => {
      const initialMemory = win.performance.memory?.usedJSHeapSize || 0
      
      // Switch away and back multiple times
      cy.switchTab('overview')
      cy.wait(500)
      cy.switchTab('brain-graph')
      cy.wait(500)
      cy.switchTab('neural-activity')
      cy.wait(500)
      cy.switchTab('brain-graph')
      cy.wait(2000)
      
      cy.window().then((win2) => {
        const finalMemory = win2.performance.memory?.usedJSHeapSize || 0
        const memoryIncrease = finalMemory - initialMemory
        
        // Memory increase should be reasonable (less than 50MB)
        expect(memoryIncrease).to.be.lessThan(50 * 1024 * 1024)
      })
    })
  })

  it('should dispose D3.js event listeners properly', () => {
    cy.switchTab('neural-activity')
    cy.fixture('neural-activity-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    // Switch tabs multiple times to test cleanup
    for (let i = 0; i < 5; i++) {
      cy.switchTab('overview')
      cy.wait(200)
      cy.switchTab('neural-activity')
      cy.wait(200)
    }
    
    // Should not have memory leaks from event listeners
    cy.window().then((win) => {
      const listenerCount = win.__d3EventListenerCount || 0
      expect(listenerCount).to.be.lessThan(100) // Reasonable threshold
    })
  })

  it('should handle browser tab visibility changes gracefully', () => {
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    // Simulate tab becoming hidden
    cy.window().then((win) => {
      Object.defineProperty(win.document, 'hidden', {
        value: true,
        writable: true
      })
      win.document.dispatchEvent(new Event('visibilitychange'))
    })
    
    cy.wait(1000)
    
    // Simulate tab becoming visible again
    cy.window().then((win) => {
      Object.defineProperty(win.document, 'hidden', {
        value: false,
        writable: true
      })
      win.document.dispatchEvent(new Event('visibilitychange'))
    })
    
    // Visualization should resume normal operation
    cy.get('[data-testid="three-canvas"]')
      .should('be.visible')
    
    cy.measureFrameRate().then((fps) => {
      expect(fps).to.be.greaterThan(30)
    })
  })
})
```

### 4.4 Accessibility and Usability Testing

#### 4.4.1 Keyboard Navigation Testing
```javascript
describe('Visualization Accessibility', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should support keyboard navigation for 3D controls', () => {
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    // Focus on 3D canvas
    cy.get('[data-testid="three-canvas"]')
      .focus()
    
    // Test keyboard controls
    cy.get('[data-testid="three-canvas"]')
      .type('{uparrow}') // Rotate up
      .type('{leftarrow}') // Rotate left
      .type('{+}') // Zoom in
      .type('{-}') // Zoom out
    
    cy.get('[data-testid="camera-position"]')
      .should('not.contain.text', 'Default')
  })

  it('should provide screen reader descriptions for visualizations', () => {
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    cy.get('[data-testid="three-canvas"]')
      .should('have.attr', 'aria-label')
      .and('contain', '3D Brain Graph')
    
    cy.get('[data-testid="graph-description"]')
      .should('be.visible')
      .and('contain.text', 'entities')
      .and('contain.text', 'relationships')
  })

  it('should support high contrast mode', () => {
    cy.switchTab('neural-activity')
    cy.fixture('neural-activity-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    // Enable high contrast mode
    cy.get('[data-testid="accessibility-menu"]').click()
    cy.get('[data-testid="high-contrast-toggle"]').click()
    
    // Verify color changes
    cy.get('[data-testid="heatmap-cell"]')
      .first()
      .should('have.css', 'stroke-width', '2px')
    
    cy.get('body')
      .should('have.class', 'high-contrast-mode')
  })

  it('should provide alternative text descriptions for charts', () => {
    cy.switchTab('cognitive-systems')
    cy.fixture('cognitive-data.json').then((data) => {
      cy.mockWebSocketMessage(data)
    })
    
    cy.get('[data-testid="radar-chart-description"]')
      .should('be.visible')
      .and('contain.text', 'Cognitive pattern strengths')
    
    cy.get('[data-testid="chart-data-table"]')
      .should('be.visible')
    
    cy.get('[data-testid="chart-data-table"] tr')
      .should('have.length.greaterThan', 1)
  })
})
```

## Custom Commands for Visualization Testing

```javascript
// cypress/support/visualization-commands.js

Cypress.Commands.add('measureFrameRate', () => {
  cy.window().then((win) => {
    return new Promise((resolve) => {
      let frameCount = 0
      const startTime = performance.now()
      const duration = 1000 // 1 second
      
      const countFrame = () => {
        frameCount++
        if (performance.now() - startTime < duration) {
          win.requestAnimationFrame(countFrame)
        } else {
          resolve(frameCount)
        }
      }
      
      win.requestAnimationFrame(countFrame)
    })
  })
})

Cypress.Commands.add('startFrameRateMonitoring', () => {
  cy.window().then((win) => {
    win.__frameRateMonitor = {
      frames: [],
      startTime: performance.now()
    }
    
    const monitor = () => {
      win.__frameRateMonitor.frames.push(performance.now())
      if (win.__frameRateMonitor.active) {
        win.requestAnimationFrame(monitor)
      }
    }
    
    win.__frameRateMonitor.active = true
    win.requestAnimationFrame(monitor)
  })
})

Cypress.Commands.add('stopFrameRateMonitoring', () => {
  cy.window().then((win) => {
    if (win.__frameRateMonitor) {
      win.__frameRateMonitor.active = false
      
      const frames = win.__frameRateMonitor.frames
      const duration = frames[frames.length - 1] - frames[0]
      const avgFps = (frames.length / duration) * 1000
      
      return cy.wrap(avgFps)
    }
    return cy.wrap(0)
  })
})

Cypress.Commands.add('measureRenderTime', (callback) => {
  const startTime = performance.now()
  callback()
  
  cy.then(() => {
    const endTime = performance.now()
    return cy.wrap(endTime - startTime)
  })
})

Cypress.Commands.add('simulateTouch', (element, touches) => {
  cy.get(element).trigger('touchstart', { touches })
})

Cypress.Commands.add('simulate3DInteraction', (canvas, interaction) => {
  const interactions = {
    rotate: () => {
      cy.get(canvas)
        .trigger('mousedown', { button: 0, clientX: 400, clientY: 300 })
        .trigger('mousemove', { clientX: 500, clientY: 200 })
        .trigger('mouseup')
    },
    zoom: (delta) => {
      cy.get(canvas).trigger('wheel', { deltaY: delta })
    },
    pan: () => {
      cy.get(canvas)
        .trigger('mousedown', { button: 2, clientX: 400, clientY: 300 })
        .trigger('mousemove', { clientX: 350, clientY: 250 })
        .trigger('mouseup')
    }
  }
  
  interactions[interaction.type]()
})
```

## Test Fixtures for Visualization Testing

```json
// cypress/fixtures/large-brain-dataset.json
{
  "entities": [
    // 500 entities with varied properties
  ],
  "relationships": [
    // 1000 relationships
  ],
  "statistics": {
    "entityCount": 500,
    "relationshipCount": 1000,
    "avgActivation": 0.65
  }
}
```

```json
// cypress/fixtures/performance-test-data.json
{
  "performance": {
    "cpu": 45,
    "memory": 67,
    "networkLatency": 23
  },
  "entities": [
    // Large dataset for performance testing
  ]
}
```

## Success Criteria

### Phase 4 Completion Requirements
- [ ] 3D visualization rendering and interactions working
- [ ] D3.js charts interactive and responsive
- [ ] Performance within acceptable limits
- [ ] Accessibility standards met
- [ ] Memory management proper
- [ ] Touch and keyboard support functional
- [ ] Real-time updates smooth and animated

### Performance Requirements
- Frame rate: 30+ FPS during interactions
- Chart update time: < 50ms
- Large dataset rendering: < 3 seconds
- Memory usage: < 200MB for visualizations
- Interaction response: < 16ms

### Quality Gates
- No visual artifacts during interactions
- Smooth animations and transitions
- Proper cleanup of resources
- Accessible to screen readers
- Cross-platform compatibility
- Responsive design working

## Dependencies for Next Phase
- Interactive visualizations functional
- Performance baseline established
- Accessibility compliance verified
- Memory management validated
- User interaction patterns confirmed

This phase ensures all interactive visualizations work correctly and performantly before testing edge cases and error scenarios in Phase 5.