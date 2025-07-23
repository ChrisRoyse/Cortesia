# Phase 2: Component Integration Testing

**Duration**: 3-4 days  
**Priority**: High - Core functionality validation  
**Focus**: Individual component functionality and integration points  
**Prerequisites**: Phase 1 completed successfully

## Objectives
- Test each dashboard component in isolation
- Validate component prop handling and state management
- Ensure proper data flow between parent and child components
- Verify component lifecycle and cleanup
- Test component-specific user interactions

## Test Categories

### 2.1 System Metrics Overview Component

#### 2.1.1 Metrics Display Testing
```javascript
describe('System Metrics Overview', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('overview')
    cy.fixture('system-metrics.json').as('metricsData')
  })

  it('should display CPU usage metrics correctly', function() {
    cy.mockWebSocketMessage(this.metricsData)
    
    cy.get('[data-testid="cpu-usage-card"]').should('be.visible')
    cy.get('[data-testid="cpu-usage-value"]').should('contain.text', '45%')
    cy.get('[data-testid="cpu-usage-trend"]').should('have.class', 'trend-stable')
  })

  it('should display memory usage with correct color coding', function() {
    const highMemoryData = { ...this.metricsData }
    highMemoryData.performance.memory = 85
    
    cy.mockWebSocketMessage(highMemoryData)
    
    cy.get('[data-testid="memory-usage-card"]').should('be.visible')
    cy.get('[data-testid="memory-usage-value"]').should('contain.text', '85%')
    cy.get('[data-testid="memory-usage-card"]').should('have.class', 'status-warning')
  })

  it('should show critical status for high resource usage', function() {
    const criticalData = { ...this.metricsData }
    criticalData.performance.cpu = 95
    criticalData.performance.memory = 95
    
    cy.mockWebSocketMessage(criticalData)
    
    cy.get('[data-testid="cpu-usage-card"]').should('have.class', 'status-critical')
    cy.get('[data-testid="memory-usage-card"]').should('have.class', 'status-critical')
  })

  it('should update metrics in real-time', function() {
    cy.mockWebSocketMessage(this.metricsData)
    cy.get('[data-testid="cpu-usage-value"]').should('contain.text', '45%')
    
    const updatedData = { ...this.metricsData }
    updatedData.performance.cpu = 67
    cy.mockWebSocketMessage(updatedData)
    
    cy.get('[data-testid="cpu-usage-value"]').should('contain.text', '67%')
  })
})
```

#### 2.1.2 Status Indicators Testing
```javascript
describe('Status Indicators', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('overview')
  })

  it('should display overall health status', () => {
    cy.get('[data-testid="health-status-indicator"]').should('be.visible')
    cy.get('[data-testid="health-status-label"]').should('contain.text', 'Overall Health')
  })

  it('should show API status correctly', () => {
    cy.mockApiHealth('healthy')
    cy.get('[data-testid="api-status-indicator"]')
      .should('be.visible')
      .and('have.class', 'status-online')
  })

  it('should handle WebSocket disconnection', () => {
    cy.simulateWebSocketDisconnect()
    cy.get('[data-testid="websocket-status-indicator"]')
      .should('have.class', 'status-error')
    cy.get('[data-testid="connection-warning"]').should('be.visible')
  })
})
```

### 2.2 Brain Knowledge Graph Component

#### 2.2.1 Graph Rendering Testing
```javascript
describe('Brain Knowledge Graph', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').as('brainData')
  })

  it('should render 3D scene container', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="brain-graph-container"]').should('be.visible')
    cy.get('[data-testid="three-canvas"]').should('be.visible')
    cy.get('[data-testid="graph-controls"]').should('be.visible')
  })

  it('should display correct number of entities', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', this.brainData.entities.length)
  })

  it('should render entities with correct colors by direction', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    // Test entity color coding
    cy.get('[data-testid="three-canvas"]').then(($canvas) => {
      const canvas = $canvas[0]
      const context = canvas.getContext('webgl')
      
      // Verify WebGL context is active
      expect(context).to.not.be.null
    })
    
    cy.get('[data-testid="legend-input"]').should('be.visible')
    cy.get('[data-testid="legend-output"]').should('be.visible')
    cy.get('[data-testid="legend-hidden"]').should('be.visible')
    cy.get('[data-testid="legend-gate"]').should('be.visible')
  })

  it('should handle entity hover interactions', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    // Simulate mouse hover on canvas (Three.js interaction)
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousemove', { clientX: 400, clientY: 300 })
    
    cy.get('[data-testid="entity-tooltip"]', { timeout: 1000 })
      .should('be.visible')
  })
})
```

#### 2.2.2 Graph Controls Testing
```javascript
describe('Brain Graph Controls', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
    cy.fixture('brain-graph-data.json').as('brainData')
  })

  it('should filter entities by activation threshold', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="activation-filter-slider"]')
      .invoke('val', 0.7)
      .trigger('input')
    
    cy.get('[data-testid="filtered-entity-count"]')
      .should('contain.text', 'Showing')
    
    // Verify fewer entities are displayed
    cy.get('[data-testid="visible-entities-count"]')
      .should('not.contain.text', this.brainData.entities.length)
  })

  it('should search entities by name', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="entity-search-input"]')
      .type('entity_1')
    
    cy.get('[data-testid="search-results"]').should('be.visible')
    cy.get('[data-testid="search-result-item"]').should('have.length.at.least', 1)
  })

  it('should zoom in and out of the graph', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    cy.get('[data-testid="zoom-in-button"]').click()
    cy.get('[data-testid="zoom-level"]').should('contain.text', '1.2x')
    
    cy.get('[data-testid="zoom-out-button"]').click()
    cy.get('[data-testid="zoom-level"]').should('contain.text', '1.0x')
  })

  it('should reset camera position', function() {
    cy.mockWebSocketMessage(this.brainData)
    
    // Interact with canvas to change camera
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousedown', { clientX: 300, clientY: 200 })
      .trigger('mousemove', { clientX: 400, clientY: 300 })
      .trigger('mouseup')
    
    cy.get('[data-testid="reset-camera-button"]').click()
    
    // Verify camera reset (position should return to default)
    cy.get('[data-testid="camera-position"]')
      .should('contain.text', 'Center')
  })
})
```

### 2.3 Neural Activation Heatmap Component

#### 2.3.1 Heatmap Rendering Testing
```javascript
describe('Neural Activation Heatmap', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('neural-activity')
    cy.fixture('neural-activity-data.json').as('neuralData')
  })

  it('should render D3 heatmap visualization', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="heatmap-container"]').should('be.visible')
    cy.get('[data-testid="heatmap-svg"]').should('be.visible')
    cy.get('[data-testid="heatmap-cells"]').should('have.length.greaterThan', 0)
  })

  it('should display activation layers correctly', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="layer-input"]').should('be.visible')
    cy.get('[data-testid="layer-hidden"]').should('be.visible')
    cy.get('[data-testid="layer-output"]').should('be.visible')
  })

  it('should show activation intensity with color scale', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="color-scale-legend"]').should('be.visible')
    cy.get('[data-testid="scale-min"]').should('contain.text', '0.0')
    cy.get('[data-testid="scale-max"]').should('contain.text', '1.0')
  })

  it('should highlight spike activations', function() {
    const spikeData = { ...this.neuralData }
    spikeData.entities[0].activation = 0.95
    
    cy.mockWebSocketMessage(spikeData)
    
    cy.get('[data-testid="spike-indicator"]').should('be.visible')
    cy.get('[data-testid="spike-count"]').should('contain.text', '1')
  })
})
```

#### 2.3.2 Activation Distribution Testing
```javascript
describe('Activation Distribution Charts', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('neural-activity')
    cy.fixture('neural-activity-data.json').as('neuralData')
  })

  it('should display activation distribution histogram', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="distribution-chart"]').should('be.visible')
    cy.get('[data-testid="histogram-bars"]').should('have.length', 5)
  })

  it('should show time series of activation history', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    cy.get('[data-testid="time-series-chart"]').should('be.visible')
    cy.get('[data-testid="time-series-line"]').should('be.visible')
  })

  it('should update charts in real-time', function() {
    cy.mockWebSocketMessage(this.neuralData)
    
    const initialValue = '45'
    cy.get('[data-testid="avg-activation-value"]')
      .should('contain.text', initialValue)
    
    const updatedData = { ...this.neuralData }
    updatedData.statistics.avgActivation = 0.67
    cy.mockWebSocketMessage(updatedData)
    
    cy.get('[data-testid="avg-activation-value"]')
      .should('contain.text', '67')
  })
})
```

### 2.4 Cognitive Systems Dashboard Component

#### 2.4.1 Cognitive Patterns Testing
```javascript
describe('Cognitive Systems Dashboard', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('cognitive-systems')
    cy.fixture('cognitive-data.json').as('cognitiveData')
  })

  it('should render radar chart for cognitive patterns', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="cognitive-radar-chart"]').should('be.visible')
    cy.get('[data-testid="radar-chart-svg"]').should('be.visible')
  })

  it('should display all cognitive pattern types', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    const expectedPatterns = [
      'convergent', 'divergent', 'lateral', 'systems', 'critical'
    ]
    
    expectedPatterns.forEach(pattern => {
      cy.get(`[data-testid="pattern-${pattern}"]`).should('be.visible')
    })
  })

  it('should show pattern strength values', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="pattern-convergent-strength"]')
      .should('contain.text', '0.7')
    cy.get('[data-testid="pattern-divergent-strength"]')
      .should('contain.text', '0.6')
  })

  it('should highlight active patterns', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="pattern-convergent"]')
      .should('have.class', 'pattern-active')
    cy.get('[data-testid="pattern-systems"]')
      .should('have.class', 'pattern-active')
  })
})
```

#### 2.4.2 Attention Allocation Testing
```javascript
describe('Attention Allocation', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('cognitive-systems')
    cy.fixture('cognitive-data.json').as('cognitiveData')
  })

  it('should display attention doughnut chart', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="attention-doughnut-chart"]').should('be.visible')
    cy.get('[data-testid="attention-chart-segments"]')
      .should('have.length', 4)
  })

  it('should show attention targets with priorities', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="attention-target-pattern-recognition"]')
      .should('contain.text', 'Pattern Recognition')
      .and('contain.text', '30%')
    
    cy.get('[data-testid="attention-target-memory-consolidation"]')
      .should('contain.text', 'Memory Consolidation')
      .and('contain.text', '25%')
  })

  it('should show total capacity utilization', function() {
    cy.mockWebSocketMessage(this.cognitiveData)
    
    cy.get('[data-testid="attention-capacity"]')
      .should('contain.text', '100%')
    cy.get('[data-testid="capacity-status"]')
      .should('have.class', 'capacity-full')
  })
})
```

### 2.5 Memory Systems Monitor Component

#### 2.5.1 Working Memory Testing
```javascript
describe('Memory Systems Monitor', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('memory')
    cy.fixture('memory-data.json').as('memoryData')
  })

  it('should display working memory buffers', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="working-memory-section"]').should('be.visible')
    cy.get('[data-testid="buffer-visual"]').should('be.visible')
    cy.get('[data-testid="buffer-verbal"]').should('be.visible')
    cy.get('[data-testid="buffer-executive"]').should('be.visible')
  })

  it('should show buffer usage percentages', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="buffer-visual-usage"]')
      .should('contain.text', '45%')
    cy.get('[data-testid="buffer-verbal-usage"]')
      .should('contain.text', '30%')
    cy.get('[data-testid="buffer-executive-usage"]')
      .should('contain.text', '20%')
  })

  it('should display buffer capacity limits', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="buffer-visual-capacity"]')
      .should('contain.text', '100')
    cy.get('[data-testid="total-working-memory"]')
      .should('contain.text', '250')
  })

  it('should show decay rates for buffers', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="buffer-visual-decay"]')
      .should('contain.text', '0.5')
    cy.get('[data-testid="buffer-verbal-decay"]')
      .should('contain.text', '0.3')
  })
})
```

#### 2.5.2 Long-term Memory Testing
```javascript
describe('Long-term Memory Systems', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('memory')
    cy.fixture('memory-data.json').as('memoryData')
  })

  it('should display consolidation metrics', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="consolidation-rate"]')
      .should('contain.text', '0.85')
    cy.get('[data-testid="retrieval-speed"]')
      .should('contain.text', '50ms')
  })

  it('should show SDR pattern visualization', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="sdr-patterns-section"]').should('be.visible')
    cy.get('[data-testid="sdr-sparsity"]').should('contain.text', '0.02')
    cy.get('[data-testid="sdr-total-bits"]').should('contain.text', '2048')
  })

  it('should display zero-copy engine status', function() {
    cy.mockWebSocketMessage(this.memoryData)
    
    cy.get('[data-testid="zero-copy-enabled"]')
      .should('contain.text', 'Enabled')
    cy.get('[data-testid="zero-copy-regions"]')
      .should('contain.text', '5')
    cy.get('[data-testid="zero-copy-latency"]')
      .should('contain.text', '5ms')
  })
})
```

## Custom Commands for Component Testing

```javascript
// cypress/support/component-commands.js

Cypress.Commands.add('mockWebSocketMessage', (data) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'MOCK_WEBSOCKET_MESSAGE',
      data: data
    }, '*')
  })
})

Cypress.Commands.add('mockApiHealth', (status) => {
  cy.intercept('GET', '/api/health', {
    statusCode: status === 'healthy' ? 200 : 500,
    body: { status: status }
  }).as('healthCheck')
})

Cypress.Commands.add('simulateWebSocketDisconnect', () => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SIMULATE_WEBSOCKET_DISCONNECT'
    }, '*')
  })
})

Cypress.Commands.add('validateChartRendering', (chartSelector) => {
  cy.get(chartSelector).should('be.visible')
  cy.get(`${chartSelector} svg`).should('exist')
  cy.get(`${chartSelector} svg`).should('have.attr', 'width')
  cy.get(`${chartSelector} svg`).should('have.attr', 'height')
})

Cypress.Commands.add('checkDataVisualization', (containerSelector, expectedDataPoints) => {
  cy.get(containerSelector).should('be.visible')
  cy.get(`${containerSelector} [data-testid*="data-point"]`)
    .should('have.length', expectedDataPoints)
})
```

## Test Fixtures

```json
// cypress/fixtures/system-metrics.json
{
  "timestamp": 1690123456789,
  "performance": {
    "cpu": 45,
    "memory": 67,
    "networkLatency": 23,
    "throughput": 456
  },
  "systemStatus": {
    "overall": "healthy",
    "api": "healthy",
    "websocket": "connected",
    "backend": "running"
  }
}
```

```json
// cypress/fixtures/brain-graph-data.json
{
  "entities": [
    {
      "id": "entity_1",
      "type_id": 1,
      "activation": 0.75,
      "direction": "Input",
      "properties": { "name": "Input Entity 1" }
    },
    {
      "id": "entity_2", 
      "type_id": 2,
      "activation": 0.45,
      "direction": "Hidden",
      "properties": { "name": "Hidden Entity 1" }
    }
  ],
  "relationships": [
    {
      "from": "entity_1",
      "to": "entity_2",
      "weight": 0.8,
      "inhibitory": false
    }
  ],
  "statistics": {
    "entityCount": 2,
    "relationshipCount": 1,
    "avgActivation": 0.6
  }
}
```

## Success Criteria

### Phase 2 Completion Requirements
- [ ] All component rendering tests pass
- [ ] Data prop handling validated
- [ ] User interactions work correctly  
- [ ] Real-time updates function properly
- [ ] Chart and visualization rendering confirmed
- [ ] Component state management verified
- [ ] Error handling within components tested

### Quality Gates
- No component crashes or errors
- All visualizations render correctly
- Data updates reflect in UI immediately
- User interactions provide proper feedback
- Components handle empty/invalid data gracefully

### Performance Requirements
- Component render time: < 100ms
- Chart update time: < 50ms
- User interaction response: < 16ms (60 FPS)
- Memory usage per component: < 10MB

## Dependencies for Next Phase
- All components render reliably
- Data flow mechanisms validated
- User interaction patterns established
- Visualization libraries functioning
- Component error boundaries working

This phase ensures each individual component works correctly before testing complex data flows and real-time scenarios in Phase 3.