// Cypress Component Testing Support File
// This file is processed and loaded automatically before your component test files.

import { mount } from 'cypress/react18'
import '@cypress/code-coverage/support'

// Import global styles if needed
import '../../src/styles/globals.css'

// Augment the Cypress namespace to include type definitions for custom commands
declare global {
  namespace Cypress {
    interface Chainable {
      mount: typeof mount
    }
  }
}

// Add mount command
Cypress.Commands.add('mount', mount)

// Configure component testing
beforeEach(() => {
  // Reset any global state before each test
  cy.window().then((win) => {
    // Clear local storage
    win.localStorage.clear()
    
    // Reset any global variables
    delete win.__mockWebSocketServer
    delete win.__performanceMetrics
  })
})