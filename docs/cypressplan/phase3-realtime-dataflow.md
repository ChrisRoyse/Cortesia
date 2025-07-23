# Phase 3: Real-time Data Flow Testing

**Duration**: 3-4 days  
**Priority**: Critical - Core functionality  
**Focus**: WebSocket connections, data streaming, and state management  
**Prerequisites**: Phases 1-2 completed successfully

## Objectives
- Validate WebSocket connection establishment and maintenance
- Test real-time data streaming and message handling
- Verify state management and data persistence
- Ensure proper error handling for connection issues
- Test data transformation and processing pipelines
- Validate concurrent data streams and message ordering

## Test Categories

### 3.1 WebSocket Connection Management

#### 3.1.1 Connection Establishment Testing
```javascript
describe('WebSocket Connection', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
  })

  afterEach(() => {
    cy.teardownMockWebSocketServer()
  })

  it('should establish WebSocket connection on app load', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    cy.get('[data-testid="websocket-status"]')
      .should('contain.text', 'Connected')
    cy.get('[data-testid="connection-indicator"]')
      .should('have.class', 'connected')
  })

  it('should retry connection on initial failure', () => {
    cy.mockWebSocketConnectionFailure()
    cy.visit('/')
    
    cy.get('[data-testid="websocket-status"]')
      .should('contain.text', 'Connecting')
    
    cy.allowWebSocketConnection()
    cy.get('[data-testid="websocket-status"]', { timeout: 10000 })
      .should('contain.text', 'Connected')
  })

  it('should handle connection timeout gracefully', () => {
    cy.mockWebSocketTimeout()
    cy.visit('/')
    
    cy.get('[data-testid="connection-timeout-message"]', { timeout: 15000 })
      .should('be.visible')
    cy.get('[data-testid="retry-connection-button"]')
      .should('be.visible')
  })

  it('should reconnect automatically after disconnection', () => {
    cy.visit('/')
    cy.waitForWebSocketConnection()
    
    cy.simulateWebSocketDisconnection()
    cy.get('[data-testid="websocket-status"]')
      .should('contain.text', 'Reconnecting')
    
    cy.allowWebSocketReconnection()
    cy.get('[data-testid="websocket-status"]', { timeout: 10000 })
      .should('contain.text', 'Connected')
  })
})
```

#### 3.1.2 Connection State Management Testing
```javascript
describe('Connection State Management', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should maintain connection state in Redux store', () => {
    cy.window().its('store').invoke('getState').then((state) => {
      expect(state.websocket.connectionState).to.equal('connected')
      expect(state.websocket.lastConnected).to.be.a('number')
    })
  })

  it('should update connection metrics', () => {
    cy.get('[data-testid="connection-uptime"]')
      .should('be.visible')
    cy.get('[data-testid="messages-received"]')
      .should('contain.text', '0')
    
    cy.sendMockWebSocketMessage({ type: 'heartbeat' })
    cy.get('[data-testid="messages-received"]')
      .should('contain.text', '1')
  })

  it('should track connection quality metrics', () => {
    // Send messages with varying delays to test quality tracking
    for (let i = 0; i < 5; i++) {
      cy.sendMockWebSocketMessage({ 
        type: 'test_message',
        timestamp: Date.now(),
        sequence: i 
      })
      cy.wait(100)
    }
    
    cy.get('[data-testid="connection-quality"]')
      .should('be.visible')
    cy.get('[data-testid="latency-avg"]')
      .should('contain.text', 'ms')
  })
})
```

### 3.2 Real-time Data Streaming

#### 3.2.1 Brain Metrics Streaming Testing
```javascript
describe('Brain Metrics Streaming', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
    cy.fixture('streaming-brain-data.json').as('streamingData')
  })

  it('should receive and process brain metrics updates', function() {
    cy.switchTab('overview')
    
    cy.sendMockWebSocketMessage({
      type: 'brain_metrics_update',
      data: this.streamingData.brain_metrics
    })
    
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', this.streamingData.brain_metrics.entityCount)
    cy.get('[data-testid="avg-activation"]')
      .should('contain.text', (this.streamingData.brain_metrics.avgActivation * 100).toFixed(0))
  })

  it('should handle rapid brain metrics updates', function() {
    cy.switchTab('overview')
    
    // Send rapid updates to test debouncing and performance
    for (let i = 0; i < 10; i++) {
      const data = { ...this.streamingData.brain_metrics }
      data.avgActivation = 0.1 + (i * 0.08)
      data.timestamp = Date.now() + i
      
      cy.sendMockWebSocketMessage({
        type: 'brain_metrics_update',
        data: data
      })
    }
    
    // Should show the latest value
    cy.get('[data-testid="avg-activation"]')
      .should('contain.text', '82') // 0.82 * 100
  })

  it('should maintain metrics history for trend analysis', function() {
    cy.switchTab('overview')
    
    // Send historical data points
    const timestamps = [Date.now() - 3000, Date.now() - 2000, Date.now() - 1000, Date.now()]
    const activations = [0.3, 0.5, 0.7, 0.6]
    
    timestamps.forEach((timestamp, index) => {
      cy.sendMockWebSocketMessage({
        type: 'brain_metrics_update',
        data: {
          ...this.streamingData.brain_metrics,
          avgActivation: activations[index],
          timestamp: timestamp
        }
      })
    })
    
    cy.get('[data-testid="activation-trend-chart"]').should('be.visible')
    cy.get('[data-testid="trend-data-points"]')
      .should('have.length', 4)
  })
})
```

#### 3.2.2 Entity Updates Streaming Testing
```javascript
describe('Entity Updates Streaming', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
    cy.fixture('entity-updates.json').as('entityUpdates')
  })

  it('should update individual entity activations in real-time', function() {
    cy.switchTab('brain-graph')
    
    // Initial entity data
    cy.sendMockWebSocketMessage({
      type: 'entity_update',
      data: this.entityUpdates.initial_entities
    })
    
    cy.get('[data-testid="entity-entity_1-activation"]')
      .should('contain.text', '0.45')
    
    // Update specific entity
    cy.sendMockWebSocketMessage({
      type: 'entity_update',
      data: {
        entity_id: 'entity_1',
        activation: 0.85,
        timestamp: Date.now()
      }
    })
    
    cy.get('[data-testid="entity-entity_1-activation"]')
      .should('contain.text', '0.85')
  })

  it('should handle batch entity updates efficiently', function() {
    cy.switchTab('brain-graph')
    
    const batchUpdate = {
      type: 'batch_entity_update',
      data: this.entityUpdates.batch_updates
    }
    
    const startTime = Date.now()
    cy.sendMockWebSocketMessage(batchUpdate)
    
    // Verify all entities updated
    this.entityUpdates.batch_updates.forEach(entity => {
      cy.get(`[data-testid="entity-${entity.id}-activation"]`)
        .should('contain.text', entity.activation.toString())
    })
    
    // Verify update completed quickly
    cy.then(() => {
      const updateTime = Date.now() - startTime
      expect(updateTime).to.be.lessThan(100) // Should update within 100ms
    })
  })

  it('should visualize entity activation changes in heatmap', function() {
    cy.switchTab('neural-activity')
    
    // Send entity updates with varying activations
    const activationLevels = [0.1, 0.3, 0.6, 0.9]
    const entityIds = ['entity_1', 'entity_2', 'entity_3', 'entity_4']
    
    entityIds.forEach((entityId, index) => {
      cy.sendMockWebSocketMessage({
        type: 'entity_update',
        data: {
          entity_id: entityId,
          activation: activationLevels[index],
          timestamp: Date.now()
        }
      })
    })
    
    // Verify heatmap updates
    cy.get('[data-testid="heatmap-cell-entity_1"]')
      .should('have.class', 'activation-low')
    cy.get('[data-testid="heatmap-cell-entity_4"]')
      .should('have.class', 'activation-very-high')
  })
})
```

### 3.3 Cognitive System Updates

#### 3.3.1 Pattern State Changes Testing
```javascript
describe('Cognitive Pattern Updates', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
    cy.fixture('cognitive-updates.json').as('cognitiveUpdates')
  })

  it('should update cognitive pattern strengths in real-time', function() {
    cy.switchTab('cognitive-systems')
    
    cy.sendMockWebSocketMessage({
      type: 'cognitive_update',
      data: this.cognitiveUpdates.pattern_strength_update
    })
    
    cy.get('[data-testid="pattern-convergent-strength"]')
      .should('contain.text', '0.85')
    cy.get('[data-testid="pattern-divergent-strength"]')
      .should('contain.text', '0.32')
    
    // Verify radar chart updates
    cy.get('[data-testid="radar-chart-convergent-point"]')
      .should('have.attr', 'data-value', '0.85')
  })

  it('should handle pattern switching events', function() {
    cy.switchTab('cognitive-systems')
    
    // Initial state - convergent active
    cy.sendMockWebSocketMessage({
      type: 'pattern_switch',
      data: {
        from_pattern: 'convergent',
        to_pattern: 'lateral',
        timestamp: Date.now()
      }
    })
    
    cy.get('[data-testid="pattern-convergent"]')
      .should('not.have.class', 'pattern-active')
    cy.get('[data-testid="pattern-lateral"]')
      .should('have.class', 'pattern-active')
    
    // Verify switch is logged in timeline
    cy.get('[data-testid="pattern-switch-timeline"]')
      .should('contain.text', 'Switched to Lateral')
  })

  it('should update attention allocation in real-time', function() {
    cy.switchTab('cognitive-systems')
    
    cy.sendMockWebSocketMessage({
      type: 'attention_update',
      data: this.cognitiveUpdates.attention_reallocation
    })
    
    cy.get('[data-testid="attention-target-pattern-recognition"]')
      .should('contain.text', '40%') // Updated from 30%
    cy.get('[data-testid="attention-target-memory-consolidation"]')
      .should('contain.text', '20%') // Updated from 25%
    
    // Verify doughnut chart updates
    cy.get('[data-testid="attention-chart-segment-pattern-recognition"]')
      .should('have.attr', 'data-percentage', '40')
  })
})
```

#### 3.3.2 Memory System Updates Testing
```javascript
describe('Memory System Updates', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
    cy.fixture('memory-updates.json').as('memoryUpdates')
  })

  it('should update working memory buffer usage', function() {
    cy.switchTab('memory')
    
    cy.sendMockWebSocketMessage({
      type: 'memory_update',
      data: this.memoryUpdates.working_memory_change
    })
    
    cy.get('[data-testid="buffer-visual-usage"]')
      .should('contain.text', '67%') // Updated from 45%
    cy.get('[data-testid="buffer-verbal-usage"]')
      .should('contain.text', '23%') // Updated from 30%
    
    // Verify visual progress bars update
    cy.get('[data-testid="buffer-visual-progress"]')
      .should('have.attr', 'aria-valuenow', '67')
  })

  it('should track consolidation process updates', function() {
    cy.switchTab('memory')
    
    cy.sendMockWebSocketMessage({
      type: 'consolidation_update',
      data: this.memoryUpdates.consolidation_progress
    })
    
    cy.get('[data-testid="consolidation-rate"]')
      .should('contain.text', '0.92')
    cy.get('[data-testid="consolidation-queue-size"]')
      .should('contain.text', '15')
    
    // Verify consolidation process visualization
    cy.get('[data-testid="consolidation-progress-bar"]')
      .should('have.attr', 'aria-valuenow', '92')
  })

  it('should update SDR pattern metrics', function() {
    cy.switchTab('memory')
    
    cy.sendMockWebSocketMessage({
      type: 'sdr_update',
      data: this.memoryUpdates.sdr_metrics
    })
    
    cy.get('[data-testid="sdr-active-patterns"]')
      .should('contain.text', '1247')
    cy.get('[data-testid="sdr-sparsity"]')
      .should('contain.text', '0.018')
    
    // Verify SDR visualization updates
    cy.get('[data-testid="sdr-pattern-visualization"]')
      .should('be.visible')
  })
})
```

### 3.4 Data Pipeline and State Management

#### 3.4.1 Data Transformation Testing
```javascript
describe('Data Pipeline Processing', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should transform raw WebSocket data to component format', () => {
    const rawData = {
      type: 'brain_metrics_raw',
      entities: [
        { id: 1, activation: 0.75, type: 'input' },
        { id: 2, activation: 0.45, type: 'hidden' }
      ],
      timestamp: Date.now()
    }
    
    cy.sendMockWebSocketMessage(rawData)
    
    // Verify transformation in Redux store
    cy.window().its('store').invoke('getState').then((state) => {
      expect(state.data.current.entities).to.have.length(2)
      expect(state.data.current.entities[0]).to.have.property('direction')
      expect(state.data.current.statistics).to.exist
    })
  })

  it('should handle data validation and sanitization', () => {
    const invalidData = {
      type: 'brain_metrics_update',
      entities: [
        { id: 'invalid', activation: 'not_a_number' },
        { id: 'valid', activation: 0.75 }
      ]
    }
    
    cy.sendMockWebSocketMessage(invalidData)
    
    // Should filter out invalid entities
    cy.window().its('store').invoke('getState').then((state) => {
      const entities = state.data.current.entities || []
      expect(entities).to.have.length(1)
      expect(entities[0].id).to.equal('valid')
    })
    
    // Should log validation errors
    cy.get('[data-testid="error-log"]')
      .should('contain.text', 'Invalid data received')
  })

  it('should maintain data history and perform aggregations', () => {
    const dataPoints = [
      { timestamp: Date.now() - 3000, avgActivation: 0.3 },
      { timestamp: Date.now() - 2000, avgActivation: 0.5 },
      { timestamp: Date.now() - 1000, avgActivation: 0.7 },
      { timestamp: Date.now(), avgActivation: 0.6 }
    ]
    
    dataPoints.forEach(point => {
      cy.sendMockWebSocketMessage({
        type: 'brain_metrics_update',
        data: { ...point, entityCount: 50 }
      })
    })
    
    cy.window().its('store').invoke('getState').then((state) => {
      expect(state.data.history).to.have.length(4)
      expect(state.data.aggregations.avgActivationTrend).to.exist
    })
  })
})
```

#### 3.4.2 State Persistence Testing
```javascript
describe('State Persistence', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should persist critical state to localStorage', () => {
    cy.sendMockWebSocketMessage({
      type: 'brain_metrics_update',
      data: { entityCount: 100, avgActivation: 0.75 }
    })
    
    cy.wait(1000) // Allow for debounced persistence
    
    cy.window().its('localStorage').then((localStorage) => {
      const persistedData = JSON.parse(localStorage.getItem('llmkg_dashboard_state'))
      expect(persistedData.lastUpdate).to.exist
      expect(persistedData.entityCount).to.equal(100)
    })
  })

  it('should restore state from localStorage on app reload', () => {
    // Set initial state
    cy.window().then((win) => {
      win.localStorage.setItem('llmkg_dashboard_state', JSON.stringify({
        entityCount: 150,
        avgActivation: 0.65,
        lastUpdate: Date.now()
      }))
    })
    
    cy.reload()
    cy.waitForWebSocketConnection()
    
    // Should show restored values until new data arrives
    cy.get('[data-testid="entity-count"]')
      .should('contain.text', '150')
  })

  it('should handle localStorage quota exceeded gracefully', () => {
    // Fill localStorage to near capacity
    cy.window().then((win) => {
      try {
        const largeData = 'x'.repeat(5 * 1024 * 1024) // 5MB string
        win.localStorage.setItem('test_large_data', largeData)
      } catch (e) {
        // Expected to fail
      }
    })
    
    // Send data that would trigger persistence
    cy.sendMockWebSocketMessage({
      type: 'brain_metrics_update',
      data: { entityCount: 200, avgActivation: 0.85 }
    })
    
    // Should not crash the app
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
    cy.get('[data-testid="storage-warning"]')
      .should('contain.text', 'Storage limit reached')
  })
})
```

### 3.5 Message Ordering and Concurrency

#### 3.5.1 Message Ordering Testing
```javascript
describe('Message Ordering', () => {
  beforeEach(() => {
    cy.setupMockWebSocketServer()
    cy.visit('/')
    cy.waitForWebSocketConnection()
  })

  it('should process messages in correct order', () => {
    const messages = [
      { type: 'entity_update', sequence: 1, entity_id: 'test', activation: 0.3 },
      { type: 'entity_update', sequence: 2, entity_id: 'test', activation: 0.6 },
      { type: 'entity_update', sequence: 3, entity_id: 'test', activation: 0.9 }
    ]
    
    // Send messages rapidly
    messages.forEach(msg => {
      cy.sendMockWebSocketMessage(msg)
    })
    
    // Should show final value
    cy.get('[data-testid="entity-test-activation"]')
      .should('contain.text', '0.9')
    
    // Verify processing order in logs
    cy.window().its('store').invoke('getState').then((state) => {
      const processedSequences = state.data.messageLog.map(log => log.sequence)
      expect(processedSequences).to.deep.equal([1, 2, 3])
    })
  })

  it('should handle out-of-order messages correctly', () => {
    const messages = [
      { type: 'entity_update', sequence: 3, timestamp: Date.now() + 2000, entity_id: 'test', activation: 0.9 },
      { type: 'entity_update', sequence: 1, timestamp: Date.now(), entity_id: 'test', activation: 0.3 },
      { type: 'entity_update', sequence: 2, timestamp: Date.now() + 1000, entity_id: 'test', activation: 0.6 }
    ]
    
    messages.forEach(msg => {
      cy.sendMockWebSocketMessage(msg)
    })
    
    // Should show value based on timestamp ordering
    cy.get('[data-testid="entity-test-activation"]')
      .should('contain.text', '0.9') // Latest timestamp
  })

  it('should buffer and process concurrent updates efficiently', () => {
    const concurrentUpdates = Array.from({ length: 100 }, (_, i) => ({
      type: 'entity_update',
      entity_id: `entity_${i}`,
      activation: Math.random(),
      timestamp: Date.now()
    }))
    
    const startTime = Date.now()
    
    // Send all updates rapidly
    concurrentUpdates.forEach(update => {
      cy.sendMockWebSocketMessage(update)
    })
    
    // Verify all updates processed
    cy.get('[data-testid="entities-updated-count"]')
      .should('contain.text', '100')
    
    cy.then(() => {
      const processingTime = Date.now() - startTime
      expect(processingTime).to.be.lessThan(500) // Should process within 500ms
    })
  })
})
```

## Custom Commands for Real-time Testing

```javascript
// cypress/support/realtime-commands.js

Cypress.Commands.add('setupMockWebSocketServer', () => {
  cy.window().then((win) => {
    win.__mockWebSocketServer = new MockWebSocketServer()
    win.__mockWebSocketServer.start()
  })
})

Cypress.Commands.add('teardownMockWebSocketServer', () => {
  cy.window().then((win) => {
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.stop()
    }
  })
})

Cypress.Commands.add('sendMockWebSocketMessage', (message) => {
  cy.window().then((win) => {
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.sendMessage(message)
    }
  })
})

Cypress.Commands.add('waitForWebSocketConnection', () => {
  cy.get('[data-testid="websocket-status"]', { timeout: 10000 })
    .should('contain.text', 'Connected')
})

Cypress.Commands.add('simulateWebSocketDisconnection', () => {
  cy.window().then((win) => {
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.disconnect()
    }
  })
})

Cypress.Commands.add('allowWebSocketReconnection', () => {
  cy.window().then((win) => {
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.reconnect()
    }
  })
})

Cypress.Commands.add('mockWebSocketConnectionFailure', () => {
  cy.window().then((win) => {
    win.__mockWebSocketConnectionShouldFail = true
  })
})

Cypress.Commands.add('allowWebSocketConnection', () => {
  cy.window().then((win) => {
    win.__mockWebSocketConnectionShouldFail = false
  })
})

Cypress.Commands.add('verifyDataPipeline', (inputData, expectedOutput) => {
  cy.sendMockWebSocketMessage(inputData)
  
  cy.window().its('store').invoke('getState').then((state) => {
    Object.keys(expectedOutput).forEach(key => {
      expect(state.data.current[key]).to.deep.equal(expectedOutput[key])
    })
  })
})
```

## Test Fixtures for Real-time Data

```json
// cypress/fixtures/streaming-brain-data.json
{
  "brain_metrics": {
    "entityCount": 150,
    "relationshipCount": 300,
    "avgActivation": 0.67,
    "activeEntities": 89,
    "timestamp": 1690123456789
  },
  "performance_metrics": {
    "processingSpeed": 1250,
    "memoryUsage": 67,
    "networkLatency": 23
  }
}
```

```json
// cypress/fixtures/entity-updates.json
{
  "initial_entities": [
    {
      "id": "entity_1",
      "activation": 0.45,
      "direction": "Input"
    },
    {
      "id": "entity_2", 
      "activation": 0.67,
      "direction": "Hidden"
    }
  ],
  "batch_updates": [
    {
      "id": "entity_1",
      "activation": 0.85
    },
    {
      "id": "entity_2",
      "activation": 0.92
    },
    {
      "id": "entity_3",
      "activation": 0.34
    }
  ]
}
```

## Performance Monitoring

```javascript
// cypress/support/performance-monitoring.js

Cypress.Commands.add('monitorMessageProcessingTime', () => {
  cy.window().then((win) => {
    win.__messageProcessingTimes = []
    
    const originalProcessMessage = win.processWebSocketMessage
    win.processWebSocketMessage = function(message) {
      const startTime = performance.now()
      const result = originalProcessMessage.call(this, message)
      const endTime = performance.now()
      
      win.__messageProcessingTimes.push(endTime - startTime)
      return result
    }
  })
})

Cypress.Commands.add('getAverageProcessingTime', () => {
  cy.window().then((win) => {
    const times = win.__messageProcessingTimes || []
    const average = times.reduce((sum, time) => sum + time, 0) / times.length
    return cy.wrap(average)
  })
})
```

## Success Criteria

### Phase 3 Completion Requirements
- [ ] WebSocket connection established reliably
- [ ] Real-time data streaming validated
- [ ] Message ordering and concurrency handled
- [ ] State management working correctly
- [ ] Data transformation pipeline functional
- [ ] Error handling for connection issues
- [ ] Performance within acceptable limits

### Performance Requirements
- Message processing time: < 10ms average
- UI update time after data received: < 50ms
- Connection establishment: < 2 seconds
- Reconnection time: < 5 seconds
- Concurrent message handling: 100+ messages/second

### Quality Gates
- No data corruption during streaming
- No memory leaks during long sessions
- Graceful degradation during connection issues
- Consistent state across all components
- Proper error logging and user feedback

## Dependencies for Next Phase
- Stable real-time data flow
- Validated state management
- Working WebSocket infrastructure
- Data transformation pipeline
- Performance baseline established

This phase ensures the real-time capabilities of the dashboard work correctly before testing complex interactive scenarios in Phase 4.