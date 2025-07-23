// Test to discover what's actually in the backend dashboard
// This will help us write accurate tests

describe('LLMKG Backend Dashboard - Content Discovery', () => {
  it('should discover dashboard structure and content', () => {
    cy.visit('http://localhost:8090', {
      failOnStatusCode: false,
      timeout: 10000
    })
    
    // Log the page title
    cy.title().then((title) => {
      cy.log('Page Title:', title)
    })
    
    // Log all headings
    cy.get('h1, h2, h3, h4, h5, h6').each(($el) => {
      cy.log(`${$el.prop('tagName')}: ${$el.text()}`)
    })
    
    // Log all divs with classes
    cy.get('div[class]').then(($divs) => {
      const classes = new Set()
      $divs.each((i, el) => {
        const classList = el.className.split(' ').filter(c => c)
        classList.forEach(c => classes.add(c))
      })
      cy.log('Div classes found:', Array.from(classes).join(', '))
    })
    
    // Log all IDs
    cy.get('[id]').then(($elements) => {
      const ids = []
      $elements.each((i, el) => {
        ids.push(el.id)
      })
      cy.log('Element IDs found:', ids.join(', '))
    })
    
    // Check for specific text content
    const searchTerms = [
      'CPU', 'Memory', 'Connection', 'API', 'WebSocket', 
      'Performance', 'Metrics', 'Entity', 'Test', 'Execution'
    ]
    
    searchTerms.forEach(term => {
      cy.get('body').then(($body) => {
        if ($body.text().includes(term)) {
          cy.log(`Found text: "${term}"`)
        }
      })
    })
    
    // Log canvas elements
    cy.get('canvas').then(($canvases) => {
      cy.log(`Found ${$canvases.length} canvas elements`)
      $canvases.each((i, el) => {
        cy.log(`Canvas ${i}: ${el.width}x${el.height}, ID: ${el.id || 'none'}`)
      })
    })
    
    // Log any error messages
    cy.get('body').then(($body) => {
      const text = $body.text()
      if (text.includes('error') || text.includes('Error')) {
        cy.log('Possible error message found')
      }
    })
  })
})