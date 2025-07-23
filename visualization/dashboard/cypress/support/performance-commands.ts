// Performance Testing Commands for LLMKG Dashboard

// Frame Rate Monitoring
Cypress.Commands.add('measureFrameRate', () => {
  cy.window().then((win) => {
    return new Promise<number>((resolve) => {
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
      startTime: performance.now(),
      active: true
    }
    
    const monitor = () => {
      if (win.__frameRateMonitor?.active) {
        win.__frameRateMonitor.frames.push(performance.now())
        win.requestAnimationFrame(monitor)
      }
    }
    
    win.requestAnimationFrame(monitor)
  })
})

Cypress.Commands.add('stopFrameRateMonitoring', () => {
  cy.window().then((win) => {
    if (win.__frameRateMonitor) {
      win.__frameRateMonitor.active = false
      
      const frames = win.__frameRateMonitor.frames
      if (frames.length > 1) {
        const duration = frames[frames.length - 1] - frames[0]
        const avgFps = (frames.length / duration) * 1000
        return cy.wrap(avgFps)
      }
    }
    return cy.wrap(0)
  })
})

// Memory Usage Monitoring
Cypress.Commands.add('measureMemoryUsage', (label: string = 'current') => {
  cy.window().then((win) => {
    if (win.performance && win.performance.memory) {
      const used = win.performance.memory.usedJSHeapSize
      cy.log(`Memory usage (${label}): ${Math.round(used / 1024 / 1024)}MB`)
      return cy.wrap(used)
    }
    return cy.wrap(0)
  })
})

Cypress.Commands.add('startMemoryMonitoring', () => {
  cy.window().then((win) => {
    win.__memoryMonitor = {
      initial: win.performance.memory?.usedJSHeapSize || 0,
      peak: 0,
      samples: [],
      intervalId: null
    }
    
    const monitor = setInterval(() => {
      if (win.performance.memory && win.__memoryMonitor) {
        const current = win.performance.memory.usedJSHeapSize
        win.__memoryMonitor.peak = Math.max(win.__memoryMonitor.peak, current)
        win.__memoryMonitor.samples.push({
          timestamp: Date.now(),
          usage: current
        })
      }
    }, 100)
    
    win.__memoryMonitor.intervalId = monitor
  })
})

Cypress.Commands.add('stopMemoryMonitoring', () => {
  cy.window().then((win) => {
    if (win.__memoryMonitor) {
      if (win.__memoryMonitor.intervalId) {
        clearInterval(win.__memoryMonitor.intervalId)
      }
      
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

// Performance Benchmarking
Cypress.Commands.add('startPerformanceMonitoring', () => {
  cy.window().then((win) => {
    win.__performanceMonitor = {
      startTime: performance.now(),
      frameCount: 0,
      messageProcessingTimes: [],
      memorySnapshots: [],
      renderTimes: [],
      active: true
    }
    
    // Monitor frame rate
    const countFrames = () => {
      if (win.__performanceMonitor?.active) {
        win.__performanceMonitor.frameCount++
        win.requestAnimationFrame(countFrames)
      }
    }
    
    win.requestAnimationFrame(countFrames)
    
    // Monitor memory usage
    const memoryMonitor = setInterval(() => {
      if (win.performance.memory && win.__performanceMonitor?.active) {
        win.__performanceMonitor.memorySnapshots.push({
          used: win.performance.memory.usedJSHeapSize,
          total: win.performance.memory.totalJSHeapSize,
          timestamp: performance.now()
        })
      }
    }, 1000)
    
    win.__performanceMonitor.memoryIntervalId = memoryMonitor
    
    // Hook into message processing
    const originalProcessMessage = win.processWebSocketMessage
    if (originalProcessMessage) {
      win.processWebSocketMessage = function(message: any) {
        const startTime = performance.now()
        const result = originalProcessMessage.call(this, message)
        const endTime = performance.now()
        
        if (win.__performanceMonitor?.active) {
          win.__performanceMonitor.messageProcessingTimes.push(endTime - startTime)
        }
        
        return result
      }
    }
  })
})

Cypress.Commands.add('stopPerformanceMonitoring', () => {
  cy.window().then((win) => {
    if (win.__performanceMonitor) {
      win.__performanceMonitor.active = false
      
      if (win.__performanceMonitor.memoryIntervalId) {
        clearInterval(win.__performanceMonitor.memoryIntervalId)
      }
      
      const monitor = win.__performanceMonitor
      const duration = performance.now() - monitor.startTime
      const averageFPS = (monitor.frameCount / duration) * 1000
      
      const maxMemory = Math.max(...monitor.memorySnapshots.map(s => s.used), 0)
      const avgProcessingTime = monitor.messageProcessingTimes.length > 0
        ? monitor.messageProcessingTimes.reduce((a, b) => a + b, 0) / monitor.messageProcessingTimes.length
        : 0
      
      return cy.wrap({
        duration,
        averageFPS,
        maxMemoryUsage: maxMemory,
        messageProcessingTime: avgProcessingTime,
        totalMessages: monitor.messageProcessingTimes.length,
        memoryGrowth: maxMemory - (monitor.memorySnapshots[0]?.used || 0)
      })
    }
    return cy.wrap({})
  })
})

// Render Time Measurement
Cypress.Commands.add('measureRenderTime', (callback: () => void) => {
  const startTime = performance.now()
  callback()
  
  cy.then(() => {
    const endTime = performance.now()
    return cy.wrap(endTime - startTime)
  })
})

// GPU Memory Estimation (limited browser support)
Cypress.Commands.add('measureGPUMemory', (label: string = 'current') => {
  cy.window().then((win) => {
    // Attempt to get WebGL info for GPU memory estimation
    const canvas = win.document.querySelector('[data-testid="three-canvas"]') as HTMLCanvasElement
    if (canvas) {
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info')
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
          cy.log(`GPU Info (${label}): ${renderer}`)
        }
        
        // Estimate GPU memory usage based on texture and buffer counts
        const textureUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
        const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE)
        
        cy.log(`GPU Capabilities (${label}): ${textureUnits} texture units, ${maxTexSize} max texture size`)
      }
    }
    return cy.wrap(0) // Placeholder since actual GPU memory is hard to measure
  })
})

// Network Performance
Cypress.Commands.add('measureNetworkLatency', () => {
  cy.window().then((win) => {
    const startTime = performance.now()
    
    // Use a simple image request to measure latency
    const img = new Image()
    img.onload = () => {
      const endTime = performance.now()
      const latency = endTime - startTime
      win.__networkLatency = latency
    }
    img.src = `${Cypress.config('baseUrl')}/favicon.ico?t=${Date.now()}`
    
    cy.waitForCondition(() => win.__networkLatency !== undefined, 5000)
    cy.window().its('__networkLatency')
  })
})

// WebGL Performance
Cypress.Commands.add('measureWebGLPerformance', () => {
  cy.window().then((win) => {
    const canvas = win.document.querySelector('[data-testid="three-canvas"]') as HTMLCanvasElement
    if (canvas) {
      const gl = canvas.getContext('webgl')
      if (gl) {
        // Test WebGL capabilities
        const performance = {
          maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
          maxVertexUniformVectors: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
          maxFragmentUniformVectors: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
          maxTextureImageUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
          maxRenderBufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
          maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS)
        }
        
        return cy.wrap(performance)
      }
    }
    return cy.wrap(null)
  })
})

// Performance Budget Validation
Cypress.Commands.add('validatePerformanceBudget', (budget: {
  maxRenderTime?: number
  minFPS?: number
  maxMemoryMB?: number
}) => {
  cy.window().then((win) => {
    const violations = []
    
    // Check render time
    if (budget.maxRenderTime && win.__lastRenderTime > budget.maxRenderTime) {
      violations.push(`Render time ${win.__lastRenderTime}ms exceeds budget ${budget.maxRenderTime}ms`)
    }
    
    // Check FPS
    if (budget.minFPS && win.__currentFPS < budget.minFPS) {
      violations.push(`FPS ${win.__currentFPS} below budget ${budget.minFPS}`)
    }
    
    // Check memory
    if (budget.maxMemoryMB && win.performance.memory) {
      const memoryMB = win.performance.memory.usedJSHeapSize / 1024 / 1024
      if (memoryMB > budget.maxMemoryMB) {
        violations.push(`Memory usage ${memoryMB.toFixed(1)}MB exceeds budget ${budget.maxMemoryMB}MB`)
      }
    }
    
    if (violations.length > 0) {
      throw new Error(`Performance budget violations: ${violations.join(', ')}`)
    }
  })
})

// Stress Testing
Cypress.Commands.add('runStressTest', (options: {
  duration: number
  operations: (() => void)[]
  memoryBudget?: number
}) => {
  cy.startPerformanceMonitoring()
  cy.startMemoryMonitoring()
  
  const startTime = Date.now()
  const { duration, operations, memoryBudget } = options
  
  const runOperations = () => {
    if (Date.now() - startTime < duration) {
      // Run random operations
      const randomOp = operations[Math.floor(Math.random() * operations.length)]
      randomOp()
      
      // Schedule next operation
      setTimeout(runOperations, 100)
    }
  }
  
  runOperations()
  
  cy.wait(duration + 1000) // Wait for stress test to complete
  
  cy.stopPerformanceMonitoring().then((perfMetrics) => {
    cy.stopMemoryMonitoring().then((memMetrics) => {
      // Validate results
      if (memoryBudget && memMetrics.peak > memoryBudget) {
        throw new Error(`Memory usage ${Math.round(memMetrics.peak / 1024 / 1024)}MB exceeded budget ${Math.round(memoryBudget / 1024 / 1024)}MB`)
      }
      
      return cy.wrap({
        performance: perfMetrics,
        memory: memMetrics
      })
    })
  })
})

declare global {
  namespace Cypress {
    interface Chainable {
      // Frame rate monitoring
      measureFrameRate(): Chainable<number>
      startFrameRateMonitoring(): Chainable<void>
      stopFrameRateMonitoring(): Chainable<number>
      
      // Memory monitoring
      measureMemoryUsage(label?: string): Chainable<number>
      startMemoryMonitoring(): Chainable<void>
      stopMemoryMonitoring(): Chainable<any>
      
      // Performance monitoring
      startPerformanceMonitoring(): Chainable<void>
      stopPerformanceMonitoring(): Chainable<any>
      measureRenderTime(callback: () => void): Chainable<number>
      
      // GPU and WebGL
      measureGPUMemory(label?: string): Chainable<number>
      measureWebGLPerformance(): Chainable<any>
      
      // Network performance
      measureNetworkLatency(): Chainable<number>
      
      // Performance validation
      validatePerformanceBudget(budget: {
        maxRenderTime?: number
        minFPS?: number
        maxMemoryMB?: number
      }): Chainable<void>
      
      // Stress testing
      runStressTest(options: {
        duration: number
        operations: (() => void)[]
        memoryBudget?: number
      }): Chainable<any>
    }
  }
}