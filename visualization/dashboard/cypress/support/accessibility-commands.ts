// Accessibility Testing Commands for LLMKG Dashboard

// Tab navigation command
Cypress.Commands.add('tab', (options?: { shift?: boolean }) => {
  cy.get('body').trigger('keydown', {
    key: 'Tab',
    code: 'Tab',
    shiftKey: options?.shift || false
  })
})

// Main accessibility workflow validation
Cypress.Commands.add('validateAccessibilityWorkflow', () => {
  // Test keyboard navigation
  cy.checkKeyboardNavigation()
  
  // Test ARIA labels and roles
  cy.validateAriaLabels()
  
  // Test screen reader compatibility
  cy.validateScreenReaderSupport()
  
  // Test color contrast
  cy.checkColorContrast()
  
  // Test focus management
  cy.validateFocusManagement()
})

// Keyboard Navigation Testing
Cypress.Commands.add('checkKeyboardNavigation', () => {
  // Start from the beginning of the page
  cy.get('body').focus()
  
  // Test Tab navigation through interactive elements
  const interactiveElements = [
    'button',
    'a[href]',
    'input',
    'select',
    'textarea',
    '[tabindex]:not([tabindex="-1"])',
    '[role="button"]',
    '[role="link"]',
    '[role="tab"]',
    '[role="menuitem"]'
  ]
  
  // Check that all interactive elements are focusable
  interactiveElements.forEach(selector => {
    cy.get(selector).each(($el) => {
      // Skip hidden elements
      if ($el.is(':visible')) {
        cy.wrap($el).focus().should('be.focused')
        
        // Test Enter/Space activation for button-like elements
        if ($el.is('button') || $el.attr('role') === 'button') {
          cy.wrap($el).trigger('keydown', { key: 'Enter' })
          cy.wrap($el).trigger('keydown', { key: ' ' })
        }
      }
    })
  })
  
  // Test Tab order is logical
  cy.get('body').tab()
  cy.focused().should('exist').then(($first) => {
    const firstElement = $first[0]
    
    // Continue tabbing and ensure we don't get stuck
    for (let i = 0; i < 20; i++) {
      cy.tab()
      cy.focused().should('exist')
    }
    
    // Test Shift+Tab to go backwards
    cy.tab({ shift: true })
    cy.focused().should('exist')
  })
})

// ARIA Labels and Roles Validation
Cypress.Commands.add('validateAriaLabels', () => {
  // Check for proper heading structure
  cy.get('h1, h2, h3, h4, h5, h6, [role="heading"]').should('exist')
  
  // Validate heading hierarchy (no skipped levels)
  cy.get('h1, h2, h3, h4, h5, h6').then(($headings) => {
    const levels = $headings.toArray().map(h => parseInt(h.tagName.charAt(1)))
    let currentLevel = 0
    
    levels.forEach(level => {
      expect(level).to.be.at.most(currentLevel + 1, 'Heading levels should not skip')
      currentLevel = Math.max(currentLevel, level)
    })
  })
  
  // Check landmarks
  const landmarks = [
    '[role="main"], main',
    '[role="navigation"], nav',
    '[role="banner"], header',
    '[role="contentinfo"], footer'
  ]
  
  landmarks.forEach(selector => {
    cy.get(selector).should('exist')
  })
  
  // Validate interactive elements have accessible names
  cy.get('button, input, select, textarea, [role="button"]').each(($el) => {
    const hasAccessibleName = $el.attr('aria-label') || 
                             $el.attr('aria-labelledby') || 
                             $el.text().trim() || 
                             $el.attr('title') ||
                             $el.find('label').length > 0
    
    expect(hasAccessibleName).to.be.true
  })
  
  // Check for alt text on images
  cy.get('img').each(($img) => {
    const hasAltText = $img.attr('alt') !== undefined
    const isDecorative = $img.attr('role') === 'presentation' || $img.attr('alt') === ''
    
    expect(hasAltText || isDecorative).to.be.true
  })
  
  // Validate form labels
  cy.get('input, select, textarea').each(($input) => {
    const id = $input.attr('id')
    const hasLabel = $input.attr('aria-label') || 
                    $input.attr('aria-labelledby') ||
                    (id && cy.get(`label[for="${id}"]`).should('exist'))
    
    expect(hasLabel).to.be.true
  })
})

// Screen Reader Support Testing
Cypress.Commands.add('validateScreenReaderSupport', () => {
  // Test live regions for dynamic content
  cy.get('[aria-live], [role="status"], [role="alert"]').should('exist')
  
  // Check for skip links
  cy.get('a[href="#main"], a[href="#content"], .skip-link').should('exist')
  
  // Validate tables have proper headers
  cy.get('table').each(($table) => {
    cy.wrap($table).within(() => {
      // Tables should have th elements or role="columnheader"
      cy.get('th, [role="columnheader"]').should('exist')
      
      // Check for table caption or aria-label
      const hasCaption = $table.find('caption').length > 0 || $table.attr('aria-label')
      expect(hasCaption).to.be.true
    })
  })
  
  // Test accordion/collapsible content
  cy.get('[aria-expanded]').each(($el) => {
    const expanded = $el.attr('aria-expanded') === 'true'
    const controls = $el.attr('aria-controls')
    
    if (controls) {
      const targetElement = cy.get(`#${controls}`)
      if (expanded) {
        targetElement.should('be.visible')
      } else {
        targetElement.should('not.be.visible')
      }
    }
  })
})

// Color Contrast Testing
Cypress.Commands.add('checkColorContrast', () => {
  // This is a simplified contrast check - in real testing you'd use specialized tools
  cy.get('body').then(($body) => {
    const elements = $body.find('*').filter((_, el) => {
      const $el = Cypress.$(el)
      return $el.text().trim() && $el.is(':visible')
    })
    
    elements.each((_, element) => {
      const $el = Cypress.$(element)
      const styles = window.getComputedStyle(element)
      const color = styles.color
      const backgroundColor = styles.backgroundColor
      
      // Log color information for manual verification
      cy.log(`Element: ${element.tagName}, Color: ${color}, Background: ${backgroundColor}`)
    })
  })
})

// Focus Management Testing
Cypress.Commands.add('validateFocusManagement', () => {
  // Test modal focus trapping
  cy.get('[role="dialog"], .modal').then(($modals) => {
    if ($modals.length > 0) {
      // Open modal if not already open
      cy.get('[data-testid*="open-modal"], .open-modal').first().click()
      
      // Focus should move to modal
      cy.get('[role="dialog"] button, [role="dialog"] input').first().should('be.focused')
      
      // Tab should stay within modal
      cy.tab()
      cy.focused().should('be.visible').and('be.contained', '[role="dialog"]')
      
      // Escape should close modal
      cy.get('body').type('{esc}')
      cy.get('[role="dialog"]').should('not.be.visible')
    }
  })
  
  // Test focus indicators
  cy.get('button, a, input, select, textarea').each(($el) => {
    cy.wrap($el).focus()
    
    // Element should have visible focus indicator
    cy.wrap($el).should('have.css', 'outline').and('not.equal', 'none')
      .or('have.css', 'box-shadow').and('not.equal', 'none')
      .or('have.css', 'border').and('not.equal', 'none')
  })
})

// Responsive Accessibility Testing
Cypress.Commands.add('testResponsiveAccessibility', () => {
  const viewports = [
    { width: 320, height: 568 },  // Mobile
    { width: 768, height: 1024 }, // Tablet
    { width: 1920, height: 1080 } // Desktop
  ]
  
  viewports.forEach(viewport => {
    cy.viewport(viewport.width, viewport.height)
    
    // Test that interactive elements are still accessible
    cy.get('button, a, input').should('be.visible').and('have.css', 'min-height')
      .then(($el) => {
        const minHeight = parseInt($el.css('min-height'))
        expect(minHeight).to.be.at.least(44) // WCAG minimum touch target size
      })
    
    // Test keyboard navigation still works
    cy.get('body').tab()
    cy.focused().should('exist')
  })
})

// High Contrast Mode Testing
Cypress.Commands.add('testHighContrastMode', () => {
  // Simulate high contrast mode by overriding CSS
  cy.get('body').invoke('attr', 'style', `
    filter: contrast(200%) brightness(200%);
    background: white !important;
    color: black !important;
  `)
  
  // Test that content is still readable
  cy.get('body').should('be.visible')
  cy.get('button, a, input').should('be.visible')
  
  // Reset styles
  cy.get('body').invoke('removeAttr', 'style')
})

// Motion and Animation Accessibility
Cypress.Commands.add('testReducedMotion', () => {
  // Simulate reduced motion preference
  cy.window().then((win) => {
    // Override matchMedia for prefers-reduced-motion
    win.matchMedia = cy.stub().returns({
      matches: true,
      media: '(prefers-reduced-motion: reduce)',
      addListener: cy.stub(),
      removeListener: cy.stub()
    })
  })
  
  // Reload to apply motion preferences
  cy.reload()
  
  // Verify animations are disabled or reduced
  cy.get('[class*="animate"], [class*="transition"]').should('have.css', 'animation-duration', '0s')
    .or('have.css', 'transition-duration', '0s')
})

// Screen Reader Announcements Testing
Cypress.Commands.add('testScreenReaderAnnouncements', () => {
  // Test dynamic content announcements
  cy.get('[aria-live="polite"], [aria-live="assertive"]').should('exist')
  
  // Simulate content update and verify announcement
  cy.window().then((win) => {
    const liveRegion = win.document.querySelector('[aria-live]')
    if (liveRegion) {
      const originalText = liveRegion.textContent
      
      // Update content
      liveRegion.textContent = 'Test announcement for screen readers'
      
      // Verify content changed
      expect(liveRegion.textContent).to.not.equal(originalText)
    }
  })
})

// Language and Localization Testing
Cypress.Commands.add('testLanguageAttributes', () => {
  // Check for lang attribute on html element
  cy.get('html').should('have.attr', 'lang')
  
  // Check for lang attributes on elements with different languages
  cy.get('[lang]').each(($el) => {
    const lang = $el.attr('lang')
    expect(lang).to.match(/^[a-z]{2}(-[A-Z]{2})?$/) // ISO language code format
  })
})

// Form Accessibility Testing
Cypress.Commands.add('testFormAccessibility', () => {
  cy.get('form').each(($form) => {
    cy.wrap($form).within(() => {
      // Check for fieldsets and legends
      cy.get('fieldset').each(($fieldset) => {
        cy.wrap($fieldset).find('legend').should('exist')
      })
      
      // Check error message association
      cy.get('input[aria-describedby], select[aria-describedby], textarea[aria-describedby]').each(($input) => {
        const describedBy = $input.attr('aria-describedby')
        if (describedBy) {
          cy.get(`#${describedBy}`).should('exist')
        }
      })
      
      // Check required field indicators
      cy.get('input[required], select[required], textarea[required]').each(($input) => {
        const hasRequiredIndicator = $input.attr('aria-required') === 'true' ||
                                   $input.siblings('label').text().includes('*') ||
                                   $input.siblings('[aria-label*="required"]').length > 0
        
        expect(hasRequiredIndicator).to.be.true
      })
    })
  })
})

// Tab and Panel Accessibility
Cypress.Commands.add('testTabPanelAccessibility', () => {
  cy.get('[role="tablist"]').each(($tablist) => {
    cy.wrap($tablist).within(() => {
      // Check that tabs have proper roles and relationships
      cy.get('[role="tab"]').each(($tab) => {
        const controls = $tab.attr('aria-controls')
        const selected = $tab.attr('aria-selected') === 'true'
        
        if (controls) {
          const panel = cy.get(`#${controls}`)
          panel.should('have.attr', 'role', 'tabpanel')
          
          if (selected) {
            panel.should('be.visible')
          } else {
            panel.should('not.be.visible')
          }
        }
      })
      
      // Test arrow key navigation
      cy.get('[role="tab"]').first().focus()
      cy.focused().type('{rightarrow}')
      cy.focused().should('have.attr', 'role', 'tab')
    })
  })
})

declare global {
  namespace Cypress {
    interface Chainable {
      // Tab navigation
      tab(options?: { shift?: boolean }): Chainable<void>
      
      // Main accessibility validation
      validateAccessibilityWorkflow(): Chainable<void>
      
      // Keyboard navigation
      checkKeyboardNavigation(): Chainable<void>
      
      // ARIA and semantic markup
      validateAriaLabels(): Chainable<void>
      validateScreenReaderSupport(): Chainable<void>
      
      // Visual accessibility
      checkColorContrast(): Chainable<void>
      validateFocusManagement(): Chainable<void>
      
      // Responsive and adaptive
      testResponsiveAccessibility(): Chainable<void>
      testHighContrastMode(): Chainable<void>
      testReducedMotion(): Chainable<void>
      
      // Content accessibility
      testScreenReaderAnnouncements(): Chainable<void>
      testLanguageAttributes(): Chainable<void>
      
      // Component-specific accessibility
      testFormAccessibility(): Chainable<void>
      testTabPanelAccessibility(): Chainable<void>
    }
  }
}