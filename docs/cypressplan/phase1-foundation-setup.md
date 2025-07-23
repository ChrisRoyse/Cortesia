# Phase 1: Foundation and Setup Testing

**Duration**: 2-3 days  
**Priority**: Critical - Must complete before other phases  
**Focus**: Basic infrastructure, routing, and core component rendering

## Objectives
- Validate basic application startup and initialization
- Test routing and navigation functionality
- Ensure core components render without errors
- Verify theme and styling systems
- Establish baseline performance metrics

## Test Categories

### 1.1 Application Bootstrap Testing

#### 1.1.1 Initial Load Tests
```javascript
describe('Application Bootstrap', () => {
  it('should load the application without errors', () => {
    cy.visit('/')
    cy.get('[data-testid="dashboard-root"]').should('be.visible')
    cy.get('.error-boundary').should('not.exist')
  })

  it('should display loading states appropriately', () => {
    cy.intercept('GET', '/api/**', { delay: 2000 }).as('slowApi')
    cy.visit('/')
    cy.get('[data-testid="loading-spinner"]').should('be.visible')
    cy.wait('@slowApi')
    cy.get('[data-testid="loading-spinner"]').should('not.exist')
  })

  it('should initialize with correct viewport dimensions', () => {
    cy.visit('/')
    cy.window().then((win) => {
      expect(win.innerWidth).to.be.greaterThan(1024)
      expect(win.innerHeight).to.be.greaterThan(768)
    })
  })
})
```

#### 1.1.2 Environment Configuration Tests
```javascript
describe('Environment Setup', () => {
  it('should load with correct development configuration', () => {
    cy.visit('/')
    cy.window().its('__REDUX_DEVTOOLS_EXTENSION__').should('exist')
    cy.get('[data-testid="dev-tools-indicator"]').should('be.visible')
  })

  it('should establish WebSocket connection on startup', () => {
    cy.visit('/')
    cy.get('[data-testid="websocket-status"]', { timeout: 10000 })
      .should('contain.text', 'Connected')
  })
})
```

### 1.2 Routing and Navigation Testing

#### 1.2.1 Basic Route Navigation
```javascript
describe('Routing System', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should navigate to dashboard root route', () => {
    cy.url().should('eq', `${Cypress.config().baseUrl}/`)
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
  })

  it('should handle invalid routes with redirect', () => {
    cy.visit('/invalid-route')
    cy.url().should('eq', `${Cypress.config().baseUrl}/`)
  })

  it('should maintain route state on page refresh', () => {
    cy.get('[data-testid="tab-brain-graph"]').click()
    cy.url().should('include', '#brain-graph')
    cy.reload()
    cy.get('[data-testid="brain-graph-container"]').should('be.visible')
  })
})
```

#### 1.2.2 Tab Navigation Testing
```javascript
describe('Tab Navigation', () => {
  const tabs = [
    { id: 'overview', testid: 'tab-overview', container: 'overview-container' },
    { id: 'brain-graph', testid: 'tab-brain-graph', container: 'brain-graph-container' },
    { id: 'neural-activity', testid: 'tab-neural-activity', container: 'neural-activity-container' },
    { id: 'cognitive-systems', testid: 'tab-cognitive-systems', container: 'cognitive-systems-container' },
    { id: 'memory', testid: 'tab-memory', container: 'memory-container' },
    { id: 'learning', testid: 'tab-learning', container: 'learning-container' },
    { id: 'api-flow', testid: 'tab-api-flow', container: 'api-flow-container' },
    { id: 'errors', testid: 'tab-errors', container: 'errors-container' },
    { id: 'analytics', testid: 'tab-analytics', container: 'analytics-container' },
    { id: 'architecture', testid: 'tab-architecture', container: 'architecture-container' }
  ]

  beforeEach(() => {
    cy.visit('/')
  })

  tabs.forEach((tab) => {
    it(`should navigate to ${tab.id} tab`, () => {
      cy.get(`[data-testid="${tab.testid}"]`).click()
      cy.get(`[data-testid="${tab.container}"]`).should('be.visible')
      cy.get(`[data-testid="${tab.testid}"]`).should('have.class', 'active')
    })
  })

  it('should navigate between all tabs sequentially', () => {
    tabs.forEach((tab) => {
      cy.get(`[data-testid="${tab.testid}"]`).click()
      cy.get(`[data-testid="${tab.container}"]`).should('be.visible')
      cy.wait(500) // Allow for tab transition
    })
  })
})
```

### 1.3 Component Rendering Testing

#### 1.3.1 Core Layout Components
```javascript
describe('Layout Components', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should render main dashboard layout', () => {
    cy.get('[data-testid="dashboard-layout"]').should('be.visible')
    cy.get('[data-testid="sidebar"]').should('be.visible')
    cy.get('[data-testid="main-content"]').should('be.visible')
    cy.get('[data-testid="header"]').should('be.visible')
  })

  it('should render tab navigation correctly', () => {
    cy.get('[data-testid="tab-navigation"]').should('be.visible')
    cy.get('[data-testid="tab-navigation"] .tab').should('have.length', 10)
  })

  it('should render connection status indicator', () => {
    cy.get('[data-testid="connection-status"]').should('be.visible')
    cy.get('[data-testid="websocket-indicator"]').should('be.visible')
  })
})
```

#### 1.3.2 Error Boundary Testing
```javascript
describe('Error Boundaries', () => {
  it('should catch and display component errors gracefully', () => {
    // Force a component error
    cy.window().then((win) => {
      win.postMessage({ type: 'FORCE_COMPONENT_ERROR' }, '*')
    })
    
    cy.get('[data-testid="error-boundary"]').should('be.visible')
    cy.get('[data-testid="error-message"]').should('contain.text', 'Something went wrong')
    cy.get('[data-testid="error-retry-button"]').should('be.visible')
  })

  it('should recover from errors when retry is clicked', () => {
    // Force error and recovery
    cy.window().then((win) => {
      win.postMessage({ type: 'FORCE_COMPONENT_ERROR' }, '*')
    })
    
    cy.get('[data-testid="error-retry-button"]').click()
    cy.get('[data-testid="error-boundary"]').should('not.exist')
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
  })
})
```

### 1.4 Theme and Styling Testing

#### 1.4.1 Theme System Validation
```javascript
describe('Theme System', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should load with default light theme', () => {
    cy.get('body').should('have.attr', 'data-theme', 'light')
    cy.get('[data-testid="theme-toggle"]').should('contain.text', 'Dark Mode')
  })

  it('should toggle to dark theme', () => {
    cy.get('[data-testid="theme-toggle"]').click()
    cy.get('body').should('have.attr', 'data-theme', 'dark')
    cy.get('[data-testid="theme-toggle"]').should('contain.text', 'Light Mode')
  })

  it('should persist theme preference across sessions', () => {
    cy.get('[data-testid="theme-toggle"]').click()
    cy.reload()
    cy.get('body').should('have.attr', 'data-theme', 'dark')
  })
})
```

#### 1.4.2 Responsive Design Testing
```javascript
describe('Responsive Design', () => {
  const viewports = [
    { name: 'Mobile', width: 375, height: 667 },
    { name: 'Tablet', width: 768, height: 1024 },
    { name: 'Desktop', width: 1920, height: 1080 },
    { name: 'Large Desktop', width: 2560, height: 1440 }
  ]

  viewports.forEach((viewport) => {
    it(`should render correctly on ${viewport.name}`, () => {
      cy.viewport(viewport.width, viewport.height)
      cy.visit('/')
      
      cy.get('[data-testid="dashboard-layout"]').should('be.visible')
      
      if (viewport.width < 768) {
        cy.get('[data-testid="mobile-menu-toggle"]').should('be.visible')
        cy.get('[data-testid="sidebar"]').should('not.be.visible')
      } else {
        cy.get('[data-testid="sidebar"]').should('be.visible')
      }
    })
  })
})
```

### 1.5 Performance Baseline Testing

#### 1.5.1 Initial Load Performance
```javascript
describe('Performance Baselines', () => {
  it('should load within acceptable time limits', () => {
    const start = Date.now()
    cy.visit('/')
    cy.get('[data-testid="dashboard-container"]').should('be.visible')
    
    cy.then(() => {
      const loadTime = Date.now() - start
      expect(loadTime).to.be.lessThan(3000) // 3 second max load time
    })
  })

  it('should have acceptable First Contentful Paint', () => {
    cy.visit('/')
    cy.window().its('performance').then((perf) => {
      cy.wrap(null).should(() => {
        const fcpEntry = perf.getEntriesByType('paint')
          .find(entry => entry.name === 'first-contentful-paint')
        if (fcpEntry) {
          expect(fcpEntry.startTime).to.be.lessThan(1500)
        }
      })
    })
  })

  it('should maintain smooth frame rates during interactions', () => {
    cy.visit('/')
    
    // Monitor frame rate during tab switching
    let frameCount = 0
    const startTime = Date.now()
    
    const countFrames = () => {
      frameCount++
      if (Date.now() - startTime < 1000) {
        requestAnimationFrame(countFrames)
      }
    }
    
    cy.window().then((win) => {
      win.requestAnimationFrame(countFrames)
    })
    
    // Perform rapid tab switching
    cy.get('[data-testid="tab-brain-graph"]').click()
    cy.get('[data-testid="tab-neural-activity"]').click()
    cy.get('[data-testid="tab-cognitive-systems"]').click()
    
    cy.then(() => {
      const fps = frameCount
      expect(fps).to.be.greaterThan(30) // Minimum 30 FPS
    })
  })
})
```

## Test Data Setup

### Mock Data Configuration
```javascript
// cypress/fixtures/foundation-test-data.json
{
  "basicSystemStatus": {
    "websocket": "connected",
    "api": "healthy",
    "backend": "running"
  },
  "minimalBrainData": {
    "entities": [
      {
        "id": "test-entity-1",
        "type_id": 1,
        "activation": 0.5,
        "direction": "Input"
      }
    ],
    "relationships": [],
    "statistics": {
      "entityCount": 1,
      "relationshipCount": 0
    }
  }
}
```

## Custom Cypress Commands

```javascript
// cypress/support/commands.js
Cypress.Commands.add('waitForDashboardLoad', () => {
  cy.get('[data-testid="dashboard-container"]', { timeout: 10000 }).should('be.visible')
  cy.get('[data-testid="loading-spinner"]').should('not.exist')
})

Cypress.Commands.add('checkWebSocketConnection', () => {
  cy.get('[data-testid="websocket-status"]', { timeout: 5000 })
    .should('contain.text', 'Connected')
})

Cypress.Commands.add('switchTab', (tabName) => {
  cy.get(`[data-testid="tab-${tabName}"]`).click()
  cy.get(`[data-testid="${tabName}-container"]`).should('be.visible')
})
```

## Success Criteria

### Phase 1 Completion Requirements
- [ ] All basic routing tests pass
- [ ] Component rendering tests complete successfully
- [ ] Theme system functions correctly
- [ ] Error boundaries catch and handle errors
- [ ] Performance baselines established
- [ ] WebSocket connection established
- [ ] All tabs accessible and render

### Performance Benchmarks
- Load time: < 3 seconds
- First Contentful Paint: < 1.5 seconds
- Frame rate: > 30 FPS during interactions
- Memory usage: < 100MB initial load

### Quality Gates
- Test coverage: 100% of foundation components
- No console errors during normal operation
- All accessibility standards met
- Cross-viewport compatibility verified

## Dependencies for Next Phase
- Stable application bootstrap
- Working WebSocket connection
- Functional tab navigation
- Error handling mechanisms
- Performance monitoring setup

This foundation phase ensures the basic infrastructure is solid before testing complex interactions and visualizations in subsequent phases.