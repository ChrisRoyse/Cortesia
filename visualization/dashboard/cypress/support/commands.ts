// Basic Cypress Commands for LLMKG Dashboard Testing

// Navigation Commands
Cypress.Commands.add('switchTab', (tabName: string) => {
  cy.get(`[data-testid="tab-${tabName}"]`).click()
  cy.get(`[data-testid="${tabName}-container"]`).should('be.visible')
})

Cypress.Commands.add('waitForDashboardLoad', () => {
  cy.get('[data-testid="dashboard-container"]', { timeout: 10000 }).should('be.visible')
  cy.get('[data-testid="loading-spinner"]').should('not.exist')
})

Cypress.Commands.add('waitForWebSocketConnection', () => {
  cy.get('[data-testid="websocket-status"]', { timeout: 10000 })
    .should('contain.text', 'Connected')
})

// Mock WebSocket Commands
Cypress.Commands.add('mockWebSocketMessage', (data: any) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'MOCK_WEBSOCKET_MESSAGE',
      data: data
    }, '*')
  })
})

Cypress.Commands.add('sendRawWebSocketMessage', (message: string) => {
  cy.window().then((win) => {
    win.postMessage({
      type: 'SEND_RAW_WEBSOCKET_MESSAGE',
      message: message
    }, '*')
  })
})

Cypress.Commands.add('setupMockWebSocketServer', () => {
  cy.window().then((win) => {
    // Create mock WebSocket server
    win.__mockWebSocketServer = {
      start: () => console.log('Mock WebSocket server started'),
      stop: () => console.log('Mock WebSocket server stopped'),
      sendMessage: (msg: any) => {
        win.postMessage({
          type: 'MOCK_WEBSOCKET_MESSAGE',
          data: msg
        }, '*')
      }
    }
    
    win.__mockWebSocketServer.start()
  })
})

Cypress.Commands.add('teardownMockWebSocketServer', () => {
  cy.window().then((win) => {
    if (win.__mockWebSocketServer) {
      win.__mockWebSocketServer.stop()
      delete win.__mockWebSocketServer
    }
  })
})

// Utility Commands for Test Data Generation
Cypress.Commands.add('generateTestBrainData', (entityCount: number = 100) => {
  return cy.wrap({
    entities: Array.from({ length: entityCount }, (_, i) => ({
      id: `entity_${i}`,
      type_id: (i % 4) + 1,
      activation: Math.random(),
      direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
      properties: { name: `Entity ${i}` },
      embedding: Array.from({ length: 128 }, () => Math.random())
    })),
    relationships: Array.from({ length: entityCount * 2 }, (_, i) => ({
      from: `entity_${Math.floor(Math.random() * entityCount)}`,
      to: `entity_${Math.floor(Math.random() * entityCount)}`,
      weight: Math.random(),
      inhibitory: Math.random() > 0.9
    })),
    statistics: {
      entityCount,
      relationshipCount: entityCount * 2,
      avgActivation: 0.5,
      minActivation: 0,
      maxActivation: 1
    }
  })
})

// Assertion helpers
Cypress.Commands.add('shouldBeWithinRange', { prevSubject: true }, (subject, min: number, max: number) => {
  const value = parseFloat(subject as string)
  expect(value).to.be.at.least(min)
  expect(value).to.be.at.most(max)
  return cy.wrap(subject)
})

// Debugging commands
Cypress.Commands.add('logTestContext', (message: string) => {
  cy.window().then((win) => {
    console.log(`[${Cypress.currentTest.title}] ${message}`)
    
    // Log current state
    console.log('Current URL:', win.location.href)
    console.log('Viewport:', {
      width: win.innerWidth,
      height: win.innerHeight
    })
    
    if (win.performance && win.performance.memory) {
      console.log('Memory Usage:', {
        used: Math.round(win.performance.memory.usedJSHeapSize / 1024 / 1024) + 'MB',
        total: Math.round(win.performance.memory.totalJSHeapSize / 1024 / 1024) + 'MB'
      })
    }
  })
})

// Wait for specific conditions
Cypress.Commands.add('waitForCondition', (conditionFn: () => boolean, timeout: number = 5000) => {
  const startTime = Date.now()
  
  const checkCondition = () => {
    if (conditionFn()) {
      return
    }
    
    if (Date.now() - startTime > timeout) {
      throw new Error(`Condition not met within ${timeout}ms`)
    }
    
    cy.wait(100).then(checkCondition)
  }
  
  checkCondition()
})

// Viewport management
Cypress.Commands.add('setMobileViewport', () => {
  cy.viewport(375, 667) // iPhone viewport
})

Cypress.Commands.add('setTabletViewport', () => {
  cy.viewport(768, 1024) // iPad viewport
})

Cypress.Commands.add('setDesktopViewport', () => {
  cy.viewport(1920, 1080) // Desktop viewport
})

// Data attribute helpers
Cypress.Commands.add('getByTestId', (testId: string) => {
  return cy.get(`[data-testid="${testId}"]`)
})

Cypress.Commands.add('findByTestId', { prevSubject: true }, (subject, testId: string) => {
  return cy.wrap(subject).find(`[data-testid="${testId}"]`)
})

// Custom matchers
Cypress.Commands.add('shouldHaveTestId', { prevSubject: true }, (subject, testId: string) => {
  return cy.wrap(subject).should('have.attr', 'data-testid', testId)
})

// Network request helpers
Cypress.Commands.add('mockApiResponse', (method: string, url: string, response: any) => {
  cy.intercept(method, url, response).as('mockedRequest')
})

Cypress.Commands.add('waitForApiCall', (alias: string) => {
  cy.wait(`@${alias}`)
})

// Local storage helpers
Cypress.Commands.add('clearDashboardStorage', () => {
  cy.window().then((win) => {
    // Clear all dashboard-related localStorage
    Object.keys(win.localStorage).forEach(key => {
      if (key.startsWith('llmkg_') || key.startsWith('dashboard_')) {
        win.localStorage.removeItem(key)
      }
    })
  })
})

Cypress.Commands.add('setDashboardState', (state: any) => {
  cy.window().then((win) => {
    win.localStorage.setItem('llmkg_dashboard_state', JSON.stringify(state))
  })
})

// Error handling
Cypress.Commands.add('expectNoConsoleErrors', () => {
  cy.window().then((win) => {
    const errors = win.__consoleErrors || []
    expect(errors).to.have.length(0)
  })
})

// Performance helpers
Cypress.Commands.add('measureLoadTime', () => {
  cy.window().then((win) => {
    const navigation = win.performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    const loadTime = navigation.loadEventEnd - navigation.loadEventStart
    cy.wrap(loadTime)
  })
})

declare global {
  namespace Cypress {
    interface Chainable {
      // Test data generation
      generateTestBrainData(entityCount?: number): Chainable<any>
      
      // Assertions
      shouldBeWithinRange(min: number, max: number): Chainable<any>
      shouldHaveTestId(testId: string): Chainable<any>
      
      // Debugging
      logTestContext(message: string): Chainable<void>
      expectNoConsoleErrors(): Chainable<void>
      
      // Conditions
      waitForCondition(conditionFn: () => boolean, timeout?: number): Chainable<void>
      
      // Viewport
      setMobileViewport(): Chainable<void>
      setTabletViewport(): Chainable<void>
      setDesktopViewport(): Chainable<void>
      
      // Data attributes
      getByTestId(testId: string): Chainable<JQuery<HTMLElement>>
      findByTestId(testId: string): Chainable<JQuery<HTMLElement>>
      
      // Network
      mockApiResponse(method: string, url: string, response: any): Chainable<void>
      waitForApiCall(alias: string): Chainable<void>
      
      // Storage
      clearDashboardStorage(): Chainable<void>
      setDashboardState(state: any): Chainable<void>
      
      // Performance
      measureLoadTime(): Chainable<number>
    }
  }
}