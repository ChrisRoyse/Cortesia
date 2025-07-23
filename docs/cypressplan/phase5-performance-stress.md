# Phase 5: Performance and Stress Testing

**Duration**: 2-3 days  
**Priority**: High - Production readiness validation  
**Focus**: Large datasets, memory usage, rendering performance  
**Prerequisites**: Phases 1-4 completed successfully

## Objectives
- Test dashboard performance under extreme data loads
- Validate memory usage and leak detection
- Ensure graceful degradation under stress
- Test concurrent user simulation scenarios
- Benchmark rendering performance with large datasets
- Validate resource management and cleanup

## Test Categories

### 5.1 Large Dataset Stress Testing

#### 5.1.1 Massive Brain Graph Testing
```javascript
describe('Massive Brain Graph Performance', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('brain-graph')
  })

  it('should handle 10,000 entities without crashing', () => {
    const massiveDataset = {
      entities: Array.from({ length: 10000 }, (_, i) => ({
        id: `entity_${i}`,
        type_id: (i % 4) + 1,
        activation: Math.random(),
        direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
        properties: { name: `Entity ${i}` },
        embedding: Array.from({ length: 128 }, () => Math.random())
      })),
      relationships: Array.from({ length: 25000 }, (_, i) => ({
        from: `entity_${Math.floor(Math.random() * 10000)}`,
        to: `entity_${Math.floor(Math.random() * 10000)}`,
        weight: Math.random(),
        inhibitory: Math.random() > 0.9
      })),
      statistics: {
        entityCount: 10000,
        relationshipCount: 25000,
        avgActivation: 0.5
      }
    }

    cy.startPerformanceMonitoring()
    
    const startTime = Date.now()
    cy.mockWebSocketMessage(massiveDataset)
    
    // Should not crash and render within reasonable time
    cy.get('[data-testid="scene-ready"]', { timeout: 30000 })
      .should('be.visible')
    
    cy.then(() => {
      const renderTime = Date.now() - startTime
      expect(renderTime).to.be.lessThan(15000) // 15 second max for 10k entities
    })
    
    cy.stopPerformanceMonitoring().then((metrics) => {
      expect(metrics.averageFPS).to.be.greaterThan(15) // Minimum playable FPS
      expect(metrics.maxMemoryUsage).to.be.lessThan(500 * 1024 * 1024) // 500MB max
    })
  })

  it('should maintain interactivity with large datasets', () => {
    const largeDataset = generateLargeDataset(5000, 12000)
    
    cy.mockWebSocketMessage(largeDataset)
    cy.get('[data-testid="scene-ready"]', { timeout: 20000 }).should('be.visible')
    
    // Test interaction responsiveness
    const interactionStartTime = Date.now()
    
    cy.get('[data-testid="three-canvas"]')
      .trigger('mousedown', { button: 0, clientX: 400, clientY: 300 })
      .trigger('mousemove', { clientX: 500, clientY: 200 })
      .trigger('mouseup')
    
    cy.get('[data-testid="camera-position"]')
      .should('not.contain.text', 'Default')
    
    cy.then(() => {
      const interactionTime = Date.now() - interactionStartTime
      expect(interactionTime).to.be.lessThan(100) // Responsive interaction
    })
  })

  it('should handle rapid entity updates at scale', () => {
    const baseDataset = generateLargeDataset(3000, 7000)
    cy.mockWebSocketMessage(baseDataset)
    cy.get('[data-testid="scene-ready"]', { timeout: 15000 }).should('be.visible')
    
    cy.startFrameRateMonitoring()
    
    // Send 100 rapid entity updates
    for (let i = 0; i < 100; i++) {
      const updateMessage = {
        type: 'batch_entity_update',
        data: Array.from({ length: 50 }, (_, j) => ({
          entity_id: `entity_${(i * 50 + j) % 3000}`,
          activation: Math.random(),
          timestamp: Date.now() + i
        }))
      }
      cy.mockWebSocketMessage(updateMessage)
    }
    
    cy.wait(2000) // Allow updates to process
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      expect(avgFps).to.be.greaterThan(20) // Should maintain reasonable FPS
    })
  })
})
```

#### 5.1.2 Extreme Heatmap Stress Testing
```javascript
describe('Neural Heatmap Extreme Scale', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.switchTab('neural-activity')
  })

  it('should render 50,000 heatmap cells efficiently', () => {
    const extremeHeatmapData = {
      entities: Array.from({ length: 50000 }, (_, i) => ({
        id: `entity_${i}`,
        activation: Math.random(),
        direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
        layer: Math.floor(i / 1000) // 50 layers of 1000 entities each
      })),
      activationDistribution: {
        veryLow: 10000,
        low: 12000,
        medium: 15000,
        high: 8000,
        veryHigh: 5000
      }
    }

    cy.startMemoryMonitoring()
    const startTime = Date.now()
    
    cy.mockWebSocketMessage(extremeHeatmapData)
    
    // Should render without crashing
    cy.get('[data-testid="heatmap-cells-rendered"]', { timeout: 20000 })
      .should('contain.text', '50000')
    
    cy.then(() => {
      const renderTime = Date.now() - startTime
      expect(renderTime).to.be.lessThan(10000) // 10 second max render time
    })
    
    cy.stopMemoryMonitoring().then((memoryUsage) => {
      expect(memoryUsage.peak).to.be.lessThan(300 * 1024 * 1024) // 300MB max
      expect(memoryUsage.growth).to.be.lessThan(100 * 1024 * 1024) // 100MB growth
    })
  })

  it('should handle intensive hover interactions on large heatmap', () => {
    const largeHeatmapData = generateLargeHeatmapData(20000)
    cy.mockWebSocketMessage(largeHeatmapData)
    
    cy.get('[data-testid="heatmap-cells-rendered"]', { timeout: 15000 })
      .should('contain.text', '20000')
    
    cy.startFrameRateMonitoring()
    
    // Perform rapid hover interactions
    for (let i = 0; i < 20; i++) {
      const x = 100 + (i * 10)
      const y = 100 + (i * 5)
      
      cy.get('[data-testid="heatmap-svg"]')
        .trigger('mousemove', { clientX: x, clientY: y })
      
      cy.wait(25) // Rapid hover simulation
    }
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      expect(avgFps).to.be.greaterThan(30) // Should maintain smooth interaction
    })
  })

  it('should efficiently handle heatmap brush selection on large dataset', () => {
    const largeHeatmapData = generateLargeHeatmapData(15000)
    cy.mockWebSocketMessage(largeHeatmapData)
    
    cy.get('[data-testid="heatmap-cells-rendered"]', { timeout: 15000 })
      .should('contain.text', '15000')
    
    const selectionStartTime = Date.now()
    
    // Create large brush selection
    cy.get('[data-testid="heatmap-svg"]')
      .trigger('mousedown', { clientX: 50, clientY: 50 })
      .trigger('mousemove', { clientX: 400, clientY: 300 })
      .trigger('mouseup')
    
    cy.get('[data-testid="selected-cells-count"]')
      .should('be.visible')
      .and('not.contain.text', '0')
    
    cy.then(() => {
      const selectionTime = Date.now() - selectionStartTime
      expect(selectionTime).to.be.lessThan(200) // Quick selection processing
    })
  })
})
```

### 5.2 Memory Stress and Leak Testing

#### 5.2.1 Memory Leak Detection
```javascript
describe('Memory Leak Detection', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should not leak memory during repeated tab switching', () => {
    const tabs = ['overview', 'brain-graph', 'neural-activity', 'cognitive-systems', 'memory']
    
    cy.measureMemoryUsage('initial').then((initialMemory) => {
      // Perform 50 tab switches
      for (let cycle = 0; cycle < 10; cycle++) {
        tabs.forEach(tab => {
          cy.switchTab(tab)
          cy.wait(200)
          
          // Send some data to force rendering
          cy.mockWebSocketMessage({
            type: 'test_data',
            entities: Array.from({ length: 100 }, (_, i) => ({
              id: `entity_${i}`,
              activation: Math.random()
            }))
          })
          cy.wait(100)
        })
      }
      
      // Force garbage collection if possible
      cy.window().then((win) => {
        if (win.gc && typeof win.gc === 'function') {
          win.gc()
        }
      })
      
      cy.wait(2000) // Allow GC to run
      
      cy.measureMemoryUsage('final').then((finalMemory) => {
        const memoryIncrease = finalMemory - initialMemory
        expect(memoryIncrease).to.be.lessThan(50 * 1024 * 1024) // 50MB max increase
      })
    })
  })

  it('should properly cleanup Three.js resources', () => {
    cy.switchTab('brain-graph')
    
    cy.measureGPUMemory('initial').then((initialGPU) => {
      // Load and unload 3D scenes multiple times
      for (let i = 0; i < 10; i++) {
        const dataset = generateLargeDataset(1000, 2000)
        cy.mockWebSocketMessage(dataset)
        cy.get('[data-testid="scene-ready"]', { timeout: 10000 }).should('be.visible')
        
        // Switch away to trigger cleanup
        cy.switchTab('overview')
        cy.wait(500)
        cy.switchTab('brain-graph')
        cy.wait(500)
      }
      
      cy.measureGPUMemory('final').then((finalGPU) => {
        const gpuIncrease = finalGPU - initialGPU
        expect(gpuIncrease).to.be.lessThan(100 * 1024 * 1024) // 100MB max GPU memory increase
      })
    })
  })

  it('should handle D3.js event listener cleanup', () => {
    cy.switchTab('neural-activity')
    
    cy.window().its('__d3EventListeners').then((initialListeners) => {
      const initial = initialListeners ? initialListeners.length : 0
      
      // Create and destroy heatmaps multiple times
      for (let i = 0; i < 20; i++) {
        const heatmapData = generateLargeHeatmapData(1000)
        cy.mockWebSocketMessage(heatmapData)
        cy.wait(200)
        
        // Trigger interactions to create listeners
        cy.get('[data-testid="heatmap-svg"]')
          .trigger('mousemove', { clientX: 100 + i * 5, clientY: 100 + i * 3 })
        
        cy.switchTab('overview')
        cy.wait(100)
        cy.switchTab('neural-activity')
        cy.wait(100)
      }
      
      cy.window().its('__d3EventListeners').then((finalListeners) => {
        const final = finalListeners ? finalListeners.length : 0
        const listenerIncrease = final - initial
        
        expect(listenerIncrease).to.be.lessThan(50) // Reasonable listener growth
      })
    })
  })
})
```

#### 5.2.2 Resource Exhaustion Testing
```javascript
describe('Resource Exhaustion Handling', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should gracefully handle WebGL context loss', () => {
    cy.switchTab('brain-graph')
    
    const dataset = generateLargeDataset(2000, 4000)
    cy.mockWebSocketMessage(dataset)
    cy.get('[data-testid="scene-ready"]', { timeout: 15000 }).should('be.visible')
    
    // Simulate WebGL context loss
    cy.get('[data-testid="three-canvas"]').then(($canvas) => {
      const canvas = $canvas[0]
      const gl = canvas.getContext('webgl')
      
      if (gl && gl.getExtension('WEBGL_lose_context')) {
        gl.getExtension('WEBGL_lose_context').loseContext()
      }
    })
    
    // Should show context loss message and recovery option
    cy.get('[data-testid="webgl-context-lost"]', { timeout: 5000 })
      .should('be.visible')
    
    cy.get('[data-testid="restore-context-button"]')
      .should('be.visible')
      .click()
    
    // Should recover and continue working
    cy.get('[data-testid="scene-ready"]', { timeout: 10000 })
      .should('be.visible')
  })

  it('should handle DOM node limit gracefully', () => {
    cy.switchTab('neural-activity')
    
    // Try to create excessive DOM nodes
    const extremeDataset = {
      entities: Array.from({ length: 100000 }, (_, i) => ({
        id: `entity_${i}`,
        activation: Math.random()
      }))
    }
    
    cy.mockWebSocketMessage(extremeDataset)
    
    // Should either render with virtualization or show warning
    cy.get('[data-testid="virtualization-active"]', { timeout: 10000 })
      .should('be.visible')
      .or(cy.get('[data-testid="dataset-too-large-warning"]').should('be.visible'))
  })

  it('should handle localStorage quota exceeded', () => {
    // Fill localStorage to capacity
    cy.window().then((win) => {
      try {
        for (let i = 0; i < 100; i++) {
          const largeData = 'x'.repeat(100 * 1024) // 100KB chunks
          win.localStorage.setItem(`test_data_${i}`, largeData)
        }
      } catch (e) {
        // Expected to fail when quota is reached
      }
    })
    
    // Try to save dashboard state
    const largeDataset = generateLargeDataset(1000, 2000)
    cy.mockWebSocketMessage(largeDataset)
    
    // Should handle quota exceeded gracefully
    cy.get('[data-testid="storage-quota-warning"]', { timeout: 5000 })
      .should('be.visible')
    
    cy.get('[data-testid="clear-storage-button"]')
      .should('be.visible')
  })
})
```

### 5.3 Concurrent Load Testing

#### 5.3.1 Simulated Multi-User Scenarios
```javascript
describe('Concurrent Load Simulation', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should handle multiple rapid WebSocket connections', () => {
    // Simulate multiple concurrent WebSocket connections
    cy.window().then((win) => {
      const connections = []
      
      for (let i = 0; i < 10; i++) {
        const mockConnection = {
          id: i,
          sendMessage: (msg) => {
            win.postMessage({
              type: 'MOCK_WEBSOCKET_MESSAGE',
              connectionId: i,
              data: msg
            }, '*')
          }
        }
        connections.push(mockConnection)
      }
      
      // Send concurrent messages from all connections
      connections.forEach((conn, index) => {
        const message = {
          type: 'brain_metrics_update',
          data: {
            entityCount: 100 + index * 10,
            avgActivation: 0.5 + (index * 0.05),
            timestamp: Date.now() + index
          }
        }
        conn.sendMessage(message)
      })
      
      // Should handle all messages without errors
      cy.get('[data-testid="connection-errors"]')
        .should('contain.text', '0')
      
      cy.get('[data-testid="messages-processed"]')
        .should('contain.text', '10')
    })
  })

  it('should maintain performance under concurrent data streams', () => {
    cy.startPerformanceMonitoring()
    
    // Simulate multiple data streams
    const streamCount = 5
    const messagesPerStream = 20
    
    for (let stream = 0; stream < streamCount; stream++) {
      cy.window().then((win) => {
        for (let msg = 0; msg < messagesPerStream; msg++) {
          setTimeout(() => {
            win.postMessage({
              type: 'MOCK_WEBSOCKET_MESSAGE',
              data: {
                type: 'entity_update',
                stream_id: stream,
                entity_id: `stream_${stream}_entity_${msg}`,
                activation: Math.random(),
                timestamp: Date.now()
              }
            }, '*')
          }, msg * 50) // 50ms intervals
        }
      })
    }
    
    cy.wait(3000) // Allow all messages to process
    
    cy.stopPerformanceMonitoring().then((metrics) => {
      expect(metrics.averageFPS).to.be.greaterThan(25)
      expect(metrics.messageProcessingTime).to.be.lessThan(10) // 10ms average
    })
  })

  it('should handle concurrent visualization updates', () => {
    const tabs = ['brain-graph', 'neural-activity', 'cognitive-systems', 'memory']
    
    // Open multiple visualizations
    tabs.forEach(tab => {
      cy.switchTab(tab)
      cy.wait(500)
    })
    
    cy.startFrameRateMonitoring()
    
    // Send updates to all visualizations simultaneously
    const concurrentUpdates = tabs.map(tab => ({
      type: `${tab}_update`,
      data: generateTestDataForTab(tab)
    }))
    
    concurrentUpdates.forEach(update => {
      cy.mockWebSocketMessage(update)
    })
    
    cy.wait(2000)
    
    cy.stopFrameRateMonitoring().then((avgFps) => {
      expect(avgFps).to.be.greaterThan(20) // Acceptable performance
    })
  })
})
```

### 5.4 Performance Benchmarking

#### 5.4.1 Rendering Performance Benchmarks
```javascript
describe('Rendering Performance Benchmarks', () => {
  const benchmarkSizes = [
    { entities: 100, relationships: 200, name: 'Small' },
    { entities: 500, relationships: 1000, name: 'Medium' },
    { entities: 1000, relationships: 2500, name: 'Large' },
    { entities: 2500, relationships: 6000, name: 'Very Large' },
    { entities: 5000, relationships: 12000, name: 'Extreme' }
  ]

  benchmarkSizes.forEach(benchmark => {
    it(`should render ${benchmark.name} dataset (${benchmark.entities} entities) within performance budget`, () => {
      cy.visit('/')
      cy.switchTab('brain-graph')
      
      const dataset = generateLargeDataset(benchmark.entities, benchmark.relationships)
      
      const startTime = Date.now()
      cy.mockWebSocketMessage(dataset)
      
      cy.get('[data-testid="scene-ready"]', { timeout: 30000 })
        .should('be.visible')
      
      cy.then(() => {
        const renderTime = Date.now() - startTime
        
        // Performance budgets based on dataset size
        const budgets = {
          'Small': 1000,    // 1 second
          'Medium': 3000,   // 3 seconds
          'Large': 5000,    // 5 seconds
          'Very Large': 10000, // 10 seconds
          'Extreme': 15000  // 15 seconds
        }
        
        expect(renderTime).to.be.lessThan(budgets[benchmark.name])
        
        // Log performance metrics
        cy.log(`${benchmark.name} dataset rendered in ${renderTime}ms`)
      })
      
      // Test interaction responsiveness
      cy.measureFrameRate().then((fps) => {
        expect(fps).to.be.greaterThan(30)
      })
    })
  })
})
```

#### 5.4.2 Memory Usage Benchmarks
```javascript
describe('Memory Usage Benchmarks', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should stay within memory budgets for various dataset sizes', () => {
    const memoryBenchmarks = [
      { size: 1000, budget: 50 * 1024 * 1024 },   // 50MB
      { size: 2500, budget: 100 * 1024 * 1024 },  // 100MB
      { size: 5000, budget: 200 * 1024 * 1024 },  // 200MB
      { size: 10000, budget: 400 * 1024 * 1024 }  // 400MB
    ]

    memoryBenchmarks.forEach(benchmark => {
      cy.measureMemoryUsage('baseline').then((baseline) => {
        const dataset = generateLargeDataset(benchmark.size, benchmark.size * 2)
        
        cy.switchTab('brain-graph')
        cy.mockWebSocketMessage(dataset)
        cy.get('[data-testid="scene-ready"]', { timeout: 20000 }).should('be.visible')
        
        cy.measureMemoryUsage('loaded').then((loaded) => {
          const memoryUsed = loaded - baseline
          expect(memoryUsed).to.be.lessThan(benchmark.budget)
          
          cy.log(`${benchmark.size} entities used ${Math.round(memoryUsed / 1024 / 1024)}MB`)
        })
      })
    })
  })
})
```

## Custom Commands for Performance Testing

```javascript
// cypress/support/performance-commands.js

Cypress.Commands.add('startPerformanceMonitoring', () => {
  cy.window().then((win) => {
    win.__performanceMonitor = {
      startTime: performance.now(),
      frameCount: 0,
      messageProcessingTimes: [],
      memorySnapshots: []
    }
    
    // Monitor frame rate
    const countFrames = () => {
      win.__performanceMonitor.frameCount++
      if (win.__performanceMonitor.active) {
        win.requestAnimationFrame(countFrames)
      }
    }
    
    win.__performanceMonitor.active = true
    win.requestAnimationFrame(countFrames)
    
    // Monitor memory usage
    setInterval(() => {
      if (win.performance.memory && win.__performanceMonitor.active) {
        win.__performanceMonitor.memorySnapshots.push({
          used: win.performance.memory.usedJSHeapSize,
          total: win.performance.memory.totalJSHeapSize,
          timestamp: performance.now()
        })
      }
    }, 1000)
  })
})

Cypress.Commands.add('stopPerformanceMonitoring', () => {
  cy.window().then((win) => {
    if (win.__performanceMonitor) {
      win.__performanceMonitor.active = false
      
      const monitor = win.__performanceMonitor
      const duration = performance.now() - monitor.startTime
      const averageFPS = (monitor.frameCount / duration) * 1000
      
      const maxMemory = Math.max(...monitor.memorySnapshots.map(s => s.used))
      const avgProcessingTime = monitor.messageProcessingTimes.reduce((a, b) => a + b, 0) / monitor.messageProcessingTimes.length || 0
      
      return cy.wrap({
        averageFPS,
        maxMemoryUsage: maxMemory,
        messageProcessingTime: avgProcessingTime,
        duration
      })
    }
    return cy.wrap({})
  })
})

Cypress.Commands.add('measureMemoryUsage', (label) => {
  cy.window().then((win) => {
    const memory = win.performance.memory
    if (memory) {
      const used = memory.usedJSHeapSize
      cy.log(`Memory usage (${label}): ${Math.round(used / 1024 / 1024)}MB`)
      return cy.wrap(used)
    }
    return cy.wrap(0)
  })
})

Cypress.Commands.add('measureGPUMemory', (label) => {
  cy.window().then((win) => {
    // Attempt to measure GPU memory (limited browser support)
    const canvas = win.document.querySelector('[data-testid="three-canvas"]')
    if (canvas) {
      const gl = canvas.getContext('webgl')
      if (gl) {
        const info = gl.getExtension('WEBGL_debug_renderer_info')
        if (info) {
          const renderer = gl.getParameter(info.UNMASKED_RENDERER_WEBGL)
          cy.log(`GPU Info (${label}): ${renderer}`)
        }
      }
    }
    return cy.wrap(0) // Placeholder since actual GPU memory is hard to measure
  })
})

Cypress.Commands.add('startMemoryMonitoring', () => {
  cy.window().then((win) => {
    win.__memoryMonitor = {
      initial: win.performance.memory?.usedJSHeapSize || 0,
      peak: 0,
      samples: []
    }
    
    const monitor = setInterval(() => {
      if (win.performance.memory && win.__memoryMonitor) {
        const current = win.performance.memory.usedJSHeapSize
        win.__memoryMonitor.peak = Math.max(win.__memoryMonitor.peak, current)
        win.__memoryMonitor.samples.push(current)
      }
    }, 100)
    
    win.__memoryMonitor.intervalId = monitor
  })
})

Cypress.Commands.add('stopMemoryMonitoring', () => {
  cy.window().then((win) => {
    if (win.__memoryMonitor) {
      clearInterval(win.__memoryMonitor.intervalId)
      
      const monitor = win.__memoryMonitor
      const final = win.performance.memory?.usedJSHeapSize || 0
      
      return cy.wrap({
        initial: monitor.initial,
        final: final,
        peak: monitor.peak,
        growth: final - monitor.initial,
        samples: monitor.samples
      })
    }
    return cy.wrap({})
  })
})

// Helper functions
function generateLargeDataset(entityCount, relationshipCount) {
  return {
    entities: Array.from({ length: entityCount }, (_, i) => ({
      id: `entity_${i}`,
      type_id: (i % 4) + 1,
      activation: Math.random(),
      direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
      properties: { name: `Entity ${i}` }
    })),
    relationships: Array.from({ length: relationshipCount }, (_, i) => ({
      from: `entity_${Math.floor(Math.random() * entityCount)}`,
      to: `entity_${Math.floor(Math.random() * entityCount)}`,
      weight: Math.random(),
      inhibitory: Math.random() > 0.9
    })),
    statistics: {
      entityCount,
      relationshipCount,
      avgActivation: 0.5
    }
  }
}

function generateLargeHeatmapData(cellCount) {
  return {
    entities: Array.from({ length: cellCount }, (_, i) => ({
      id: `entity_${i}`,
      activation: Math.random(),
      direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
      layer: Math.floor(i / 100)
    }))
  }
}

function generateTestDataForTab(tab) {
  const generators = {
    'brain-graph': () => generateLargeDataset(500, 1000),
    'neural-activity': () => generateLargeHeatmapData(2000),
    'cognitive-systems': () => ({
      patterns: Array.from({ length: 5 }, (_, i) => ({
        id: `pattern_${i}`,
        strength: Math.random(),
        active: Math.random() > 0.5
      }))
    }),
    'memory': () => ({
      workingMemory: {
        buffers: Array.from({ length: 3 }, (_, i) => ({
          id: `buffer_${i}`,
          usage: Math.random() * 100
        }))
      }
    })
  }
  
  return generators[tab] ? generators[tab]() : {}
}
```

## Performance Test Configuration

```javascript
// cypress/support/performance-config.js

export const PERFORMANCE_BUDGETS = {
  rendering: {
    small: 1000,    // 1s for < 500 entities
    medium: 3000,   // 3s for < 1500 entities  
    large: 5000,    // 5s for < 3000 entities
    extreme: 15000  // 15s for > 3000 entities
  },
  memory: {
    small: 50 * 1024 * 1024,   // 50MB
    medium: 100 * 1024 * 1024, // 100MB
    large: 200 * 1024 * 1024,  // 200MB
    extreme: 400 * 1024 * 1024 // 400MB
  },
  frameRate: {
    minimum: 30,    // Minimum acceptable FPS
    target: 60,     // Target FPS
    interaction: 45 // Minimum during interactions
  },
  interaction: {
    response: 16,   // 16ms for 60fps
    hover: 50,      // 50ms for hover feedback
    click: 100      // 100ms for click feedback
  }
}

export const STRESS_TEST_SCENARIOS = {
  massive_dataset: {
    entities: 10000,
    relationships: 25000,
    timeout: 30000
  },
  extreme_heatmap: {
    cells: 50000,
    timeout: 20000
  },
  concurrent_streams: {
    streamCount: 10,
    messagesPerStream: 50,
    interval: 25
  }
}
```

## Success Criteria

### Phase 5 Completion Requirements
- [ ] Large dataset rendering within performance budgets
- [ ] Memory usage stays within acceptable limits
- [ ] No memory leaks detected during stress testing
- [ ] Graceful degradation under extreme load
- [ ] Resource exhaustion handled properly
- [ ] Concurrent load scenarios working
- [ ] Performance benchmarks established

### Performance Benchmarks
- 10,000 entities: < 15 seconds render time
- 50,000 heatmap cells: < 10 seconds render time
- Memory usage: < 400MB for largest datasets
- Frame rate: > 30 FPS under normal load
- Interaction response: < 100ms

### Quality Gates
- Zero memory leaks over 10-minute stress test
- Graceful error handling for resource exhaustion
- Performance degradation is predictable and documented
- System remains stable under extreme conditions
- Recovery mechanisms work correctly

## Dependencies for Next Phase
- Performance baseline established
- Stress testing limits identified
- Resource management validated
- Error handling under load confirmed
- Benchmark suite functional

This phase ensures the dashboard can handle production-scale loads and maintains acceptable performance under stress before testing error scenarios in Phase 6.