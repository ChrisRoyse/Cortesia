# Micro-Phase 9.43: Integration Tests

## Objective
Implement comprehensive integration testing framework to validate end-to-end workflows, WASM-JavaScript communication, and complete system functionality.

## Prerequisites
- Completed micro-phase 9.42 (JavaScript Unit Tests)
- WASM and JavaScript components fully implemented (phases 9.01-9.40)
- Storage and visualization systems operational (phases 9.11-9.35)

## Task Description
Create robust integration testing infrastructure covering complete user workflows, cross-component communication, data persistence, and system reliability. Implement automated test scenarios that validate the entire cortical column allocation system from input to visualization.

## Specific Actions

1. **Configure integration test environment**
```javascript
// playwright.config.js
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/integration',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/integration-results.json' }],
    ['junit', { outputFile: 'test-results/integration-junit.xml' }]
  ],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 12'] },
    }
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  }
});
```

```javascript
// tests/integration/setup/global-setup.js
import { chromium } from '@playwright/test';

async function globalSetup() {
  // Start browser for WASM compilation check
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Pre-warm WASM compilation
  await page.goto('http://localhost:3000');
  await page.waitForFunction(() => window.wasmLoaded === true, {
    timeout: 30000
  });
  
  await browser.close();
  
  // Setup test database
  await setupTestDatabase();
  
  console.log('Global integration test setup completed');
}

async function setupTestDatabase() {
  // Initialize clean test environment
  const testData = {
    concepts: [
      {
        id: 'test-concept-1',
        name: 'Neural Network',
        data: generateTestConceptData(100),
        relationships: ['test-concept-2']
      },
      {
        id: 'test-concept-2', 
        name: 'Machine Learning',
        data: generateTestConceptData(150),
        relationships: ['test-concept-1']
      }
    ],
    allocations: [
      {
        conceptId: 'test-concept-1',
        columnIndex: 42,
        strength: 0.85
      }
    ]
  };
  
  // Store test data
  await storeTestData(testData);
}

export default globalSetup;
```

2. **Create end-to-end workflow tests**
```javascript
// tests/integration/workflows/complete-allocation-workflow.spec.js
import { test, expect } from '@playwright/test';

test.describe('Complete Allocation Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="cortical-visualizer"]');
    await page.waitForFunction(() => window.wasmLoaded === true);
  });

  test('should complete full concept allocation and visualization cycle', async ({ page }) => {
    // Step 1: Enter new concept
    await page.fill('[data-testid="concept-input"]', 'Deep Learning Architecture');
    await page.fill('[data-testid="concept-description"]', 'Multi-layer neural network with backpropagation');
    await page.click('[data-testid="add-concept-button"]');

    // Step 2: Wait for WASM processing
    await page.waitForSelector('[data-testid="processing-indicator"]', { state: 'visible' });
    await page.waitForSelector('[data-testid="processing-indicator"]', { state: 'hidden', timeout: 10000 });

    // Step 3: Verify allocation in visualization
    const allocationResult = await page.waitForSelector('[data-testid="allocation-result"]');
    const allocationData = await allocationResult.textContent();
    
    expect(allocationData).toContain('Column:');
    expect(allocationData).toContain('Strength:');
    
    // Step 4: Check visual representation
    const canvas = page.locator('[data-testid="cortical-canvas"]');
    await expect(canvas).toBeVisible();
    
    // Verify canvas has been updated with new allocation
    const canvasData = await page.evaluate(() => {
      const canvas = document.querySelector('[data-testid="cortical-canvas"]');
      const ctx = canvas.getContext('2d');
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      return Array.from(imageData.data).some(pixel => pixel > 0);
    });
    
    expect(canvasData).toBe(true);

    // Step 5: Verify persistence
    await page.reload();
    await page.waitForFunction(() => window.wasmLoaded === true);
    
    const persistedConcept = await page.locator('[data-testid="concept-list"]')
      .locator('text=Deep Learning Architecture').first();
    await expect(persistedConcept).toBeVisible();
  });

  test('should handle concept search and retrieval', async ({ page }) => {
    // Add test concept first
    await addTestConcept(page, 'Reinforcement Learning', 'Agent-based learning system');

    // Search for concept
    await page.fill('[data-testid="search-input"]', 'Reinforcement');
    await page.press('[data-testid="search-input"]', 'Enter');

    // Wait for search results
    await page.waitForSelector('[data-testid="search-results"]');
    
    const searchResults = await page.locator('[data-testid="search-result-item"]').all();
    expect(searchResults.length).toBeGreaterThan(0);

    // Click on search result
    await searchResults[0].click();

    // Verify concept details loaded
    await expect(page.locator('[data-testid="concept-details"]')).toBeVisible();
    
    const conceptName = await page.locator('[data-testid="concept-name"]').textContent();
    expect(conceptName).toContain('Reinforcement Learning');

    // Verify allocation visualization updated
    const highlightedColumns = await page.evaluate(() => {
      return window.cortexVisualizer?.getHighlightedColumns() || [];
    });
    
    expect(highlightedColumns.length).toBeGreaterThan(0);
  });

  test('should handle complex query operations', async ({ page }) => {
    // Setup multiple related concepts
    await addTestConcept(page, 'Artificial Intelligence', 'Computational intelligence systems');
    await addTestConcept(page, 'Machine Learning', 'Algorithms that learn from data');
    await addTestConcept(page, 'Neural Networks', 'Interconnected processing nodes');

    // Perform relationship query
    await page.fill('[data-testid="query-input"]', 'RELATED_TO("Artificial Intelligence")');
    await page.click('[data-testid="execute-query-button"]');

    // Wait for query results
    await page.waitForSelector('[data-testid="query-results"]');
    
    const queryResults = await page.locator('[data-testid="query-result-item"]').all();
    expect(queryResults.length).toBeGreaterThanOrEqual(2);

    // Verify relationship visualization
    const relationshipLines = await page.evaluate(() => {
      const canvas = document.querySelector('[data-testid="cortical-canvas"]');
      const ctx = canvas.getContext('2d');
      return window.cortexVisualizer?.getActiveRelationships() || [];
    });
    
    expect(relationshipLines.length).toBeGreaterThan(0);
  });
});

async function addTestConcept(page, name, description) {
  await page.fill('[data-testid="concept-input"]', name);
  await page.fill('[data-testid="concept-description"]', description);
  await page.click('[data-testid="add-concept-button"]');
  await page.waitForSelector('[data-testid="processing-indicator"]', { state: 'hidden' });
}
```

3. **Create WASM-JavaScript communication tests**
```javascript
// tests/integration/communication/wasm-js-bridge.spec.js
import { test, expect } from '@playwright/test';

test.describe('WASM-JavaScript Communication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => window.wasmLoaded === true);
  });

  test('should handle large data transfers between WASM and JS', async ({ page }) => {
    // Create large concept data
    const largeConceptData = await page.evaluate(() => {
      const size = 10000; // 10K elements
      const data = new Array(size).fill(0).map((_, i) => Math.sin(i / 100));
      return window.cortexWrapper.allocateLargeConcept('large-concept', data);
    });

    expect(largeConceptData.success).toBe(true);
    expect(largeConceptData.size).toBe(10000);
    expect(largeConceptData.memoryUsed).toBeGreaterThan(0);

    // Verify data integrity
    const retrievedData = await page.evaluate(() => {
      return window.cortexWrapper.getConcept('large-concept');
    });

    expect(retrievedData.data.length).toBe(10000);
    
    // Spot check data integrity
    const firstTen = retrievedData.data.slice(0, 10);
    const expectedFirstTen = Array.from({ length: 10 }, (_, i) => Math.sin(i / 100));
    
    for (let i = 0; i < 10; i++) {
      expect(Math.abs(firstTen[i] - expectedFirstTen[i])).toBeLessThan(0.0001);
    }
  });

  test('should handle concurrent WASM operations', async ({ page }) => {
    // Start multiple concurrent operations
    const operations = await page.evaluate(async () => {
      const promises = [];
      
      // Spatial pooling operations
      for (let i = 0; i < 5; i++) {
        const pattern = new Array(100).fill(0).map(() => Math.random() > 0.8 ? 1 : 0);
        promises.push(
          window.cortexWrapper.spatialPooling(pattern)
            .then(result => ({ type: 'spatial', id: i, success: true, result }))
            .catch(error => ({ type: 'spatial', id: i, success: false, error: error.message }))
        );
      }
      
      // Allocation operations
      for (let i = 0; i < 3; i++) {
        promises.push(
          window.cortexWrapper.allocateConcept(`concurrent-${i}`, 100 + i * 50)
            .then(result => ({ type: 'allocation', id: i, success: true, result }))
            .catch(error => ({ type: 'allocation', id: i, success: false, error: error.message }))
        );
      }
      
      return Promise.all(promises);
    });

    // Verify all operations completed successfully
    const spatialOps = operations.filter(op => op.type === 'spatial');
    const allocationOps = operations.filter(op => op.type === 'allocation');
    
    expect(spatialOps.every(op => op.success)).toBe(true);
    expect(allocationOps.every(op => op.success)).toBe(true);
    
    expect(spatialOps.length).toBe(5);
    expect(allocationOps.length).toBe(3);
  });

  test('should maintain memory consistency across operations', async ({ page }) => {
    const memoryTests = await page.evaluate(async () => {
      const results = [];
      
      // Initial memory state
      const initialMemory = window.cortexWrapper.getMemoryUsage();
      results.push({ stage: 'initial', memory: initialMemory });
      
      // Allocate concepts
      const concepts = [];
      for (let i = 0; i < 10; i++) {
        const concept = await window.cortexWrapper.allocateConcept(`memory-test-${i}`, 100);
        concepts.push(concept);
      }
      
      const afterAllocation = window.cortexWrapper.getMemoryUsage();
      results.push({ stage: 'after_allocation', memory: afterAllocation });
      
      // Deallocate half
      for (let i = 0; i < 5; i++) {
        await window.cortexWrapper.deallocateConcept(concepts[i].id);
      }
      
      const afterPartialDeallocation = window.cortexWrapper.getMemoryUsage();
      results.push({ stage: 'after_partial_deallocation', memory: afterPartialDeallocation });
      
      // Deallocate remaining
      for (let i = 5; i < 10; i++) {
        await window.cortexWrapper.deallocateConcept(concepts[i].id);
      }
      
      const afterFullDeallocation = window.cortexWrapper.getMemoryUsage();
      results.push({ stage: 'after_full_deallocation', memory: afterFullDeallocation });
      
      return results;
    });

    // Verify memory progression
    const initial = memoryTests.find(r => r.stage === 'initial').memory;
    const afterAlloc = memoryTests.find(r => r.stage === 'after_allocation').memory;
    const afterPartial = memoryTests.find(r => r.stage === 'after_partial_deallocation').memory;
    const afterFull = memoryTests.find(r => r.stage === 'after_full_deallocation').memory;
    
    // Memory should increase after allocation
    expect(afterAlloc.allocated).toBeGreaterThan(initial.allocated);
    
    // Memory should decrease after partial deallocation
    expect(afterPartial.allocated).toBeLessThan(afterAlloc.allocated);
    
    // Memory should return to initial state after full deallocation
    expect(afterFull.allocated).toBe(initial.allocated);
    
    // No memory leaks
    expect(afterFull.leaked).toBe(0);
  });
});
```

4. **Create storage integration tests**
```javascript
// tests/integration/storage/persistence.spec.js
import { test, expect } from '@playwright/test';

test.describe('Storage Integration', () => {
  test.beforeEach(async ({ page, context }) => {
    // Clear storage before each test
    await context.clearCookies();
    await context.clearPermissions();
    await page.goto('/');
    await page.waitForFunction(() => window.wasmLoaded === true);
  });

  test('should persist data across browser sessions', async ({ page, context }) => {
    // Add concept in first session
    const conceptData = {
      name: 'Persistent Concept',
      description: 'Should survive browser restart',
      data: 'test-data-for-persistence'
    };

    await addConcept(page, conceptData);
    
    // Verify concept is stored
    const initialConcepts = await page.evaluate(() => {
      return window.storageManager.getAllConcepts();
    });
    
    expect(initialConcepts.length).toBe(1);
    expect(initialConcepts[0].name).toBe(conceptData.name);

    // Simulate browser restart by creating new context
    const newContext = await page.context().browser().newContext();
    const newPage = await newContext.newPage();
    
    await newPage.goto('/');
    await newPage.waitForFunction(() => window.wasmLoaded === true);
    
    // Verify data persisted
    const persistedConcepts = await newPage.evaluate(() => {
      return window.storageManager.getAllConcepts();
    });
    
    expect(persistedConcepts.length).toBe(1);
    expect(persistedConcepts[0].name).toBe(conceptData.name);
    expect(persistedConcepts[0].description).toBe(conceptData.description);
    
    await newContext.close();
  });

  test('should handle storage quota limits gracefully', async ({ page }) => {
    // Mock storage quota
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'storage', {
        value: {
          estimate: () => Promise.resolve({
            quota: 1024 * 1024, // 1MB limit
            usage: 0
          })
        }
      });
    });

    // Add concepts until near quota limit
    const results = await page.evaluate(async () => {
      const concepts = [];
      const results = { success: 0, failed: 0, quotaWarning: false };
      
      for (let i = 0; i < 100; i++) {
        try {
          const largeData = new Array(10000).fill(i); // ~40KB per concept
          const concept = await window.storageManager.storeConcept({
            id: `quota-test-${i}`,
            name: `Concept ${i}`,
            data: largeData
          });
          
          concepts.push(concept);
          results.success++;
          
          // Check quota after each addition
          const quota = await window.storageManager.checkQuota();
          if (quota.usagePercentage > 80) {
            results.quotaWarning = true;
            break;
          }
        } catch (error) {
          results.failed++;
          if (error.message.includes('quota')) {
            break;
          }
        }
      }
      
      return results;
    });

    expect(results.success).toBeGreaterThan(0);
    expect(results.quotaWarning).toBe(true);
    
    // Verify storage manager handled quota properly
    const quotaStatus = await page.evaluate(() => {
      return window.storageManager.getQuotaStatus();
    });
    
    expect(quotaStatus.nearLimit).toBe(true);
    expect(quotaStatus.usagePercentage).toBeGreaterThan(80);
  });

  test('should sync data across multiple tabs', async ({ page, context }) => {
    // Add concept in first tab
    await addConcept(page, {
      name: 'Multi-tab Concept',
      description: 'Should sync across tabs'
    });

    // Open second tab
    const secondTab = await context.newPage();
    await secondTab.goto('/');
    await secondTab.waitForFunction(() => window.wasmLoaded === true);

    // Verify concept appears in second tab
    const conceptsInSecondTab = await secondTab.evaluate(() => {
      return window.storageManager.getAllConcepts();
    });
    
    expect(conceptsInSecondTab.length).toBe(1);
    expect(conceptsInSecondTab[0].name).toBe('Multi-tab Concept');

    // Add concept in second tab
    await addConcept(secondTab, {
      name: 'Second Tab Concept',
      description: 'Added from second tab'
    });

    // Wait for sync event
    await page.waitForFunction(() => {
      return window.storageManager.getAllConcepts().length === 2;
    }, { timeout: 5000 });

    // Verify sync to first tab
    const conceptsInFirstTab = await page.evaluate(() => {
      return window.storageManager.getAllConcepts();
    });
    
    expect(conceptsInFirstTab.length).toBe(2);
    const conceptNames = conceptsInFirstTab.map(c => c.name);
    expect(conceptNames).toContain('Multi-tab Concept');
    expect(conceptNames).toContain('Second Tab Concept');

    await secondTab.close();
  });
});

async function addConcept(page, concept) {
  await page.fill('[data-testid="concept-input"]', concept.name);
  await page.fill('[data-testid="concept-description"]', concept.description);
  await page.click('[data-testid="add-concept-button"]');
  await page.waitForSelector('[data-testid="processing-indicator"]', { state: 'hidden' });
}
```

5. **Create performance integration tests**
```javascript
// tests/integration/performance/system-performance.spec.js
import { test, expect } from '@playwright/test';

test.describe('System Performance Integration', () => {
  test('should maintain performance under load', async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => window.wasmLoaded === true);

    // Start performance monitoring
    await page.evaluate(() => {
      window.performanceMonitor.startMonitoring();
    });

    // Simulate heavy workload
    const performanceResults = await page.evaluate(async () => {
      const startTime = performance.now();
      const operations = [];
      
      // Create 50 concepts concurrently
      for (let i = 0; i < 50; i++) {
        operations.push(
          window.cortexWrapper.allocateConcept(`perf-test-${i}`, 100 + i)
        );
      }
      
      await Promise.all(operations);
      
      // Perform 100 spatial pooling operations
      const spatialOps = [];
      for (let i = 0; i < 100; i++) {
        const pattern = new Array(100).fill(0).map(() => Math.random() > 0.8 ? 1 : 0);
        spatialOps.push(window.cortexWrapper.spatialPooling(pattern));
      }
      
      await Promise.all(spatialOps);
      
      const endTime = performance.now();
      const metrics = window.performanceMonitor.getMetrics();
      
      return {
        totalTime: endTime - startTime,
        memoryUsage: metrics.memory,
        wasmPerformance: metrics.wasm,
        frameRate: metrics.rendering.fps
      };
    });

    // Verify performance benchmarks
    expect(performanceResults.totalTime).toBeLessThan(10000); // < 10 seconds
    expect(performanceResults.frameRate).toBeGreaterThan(30); // > 30 FPS
    expect(performanceResults.memoryUsage.fragmentation).toBeLessThan(0.2); // < 20% fragmentation
    expect(performanceResults.wasmPerformance.avgExecutionTime).toBeLessThan(10); // < 10ms average
  });

  test('should handle memory pressure gracefully', async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => window.wasmLoaded === true);

    const memoryStressTest = await page.evaluate(async () => {
      const results = { phases: [], crashed: false };
      
      try {
        // Phase 1: Normal allocation
        for (let i = 0; i < 100; i++) {
          await window.cortexWrapper.allocateConcept(`normal-${i}`, 100);
        }
        results.phases.push({ phase: 'normal', memory: window.cortexWrapper.getMemoryUsage() });
        
        // Phase 2: Large allocations
        for (let i = 0; i < 50; i++) {
          await window.cortexWrapper.allocateConcept(`large-${i}`, 10000);
        }
        results.phases.push({ phase: 'large', memory: window.cortexWrapper.getMemoryUsage() });
        
        // Phase 3: Stress test with very large allocations
        for (let i = 0; i < 10; i++) {
          try {
            await window.cortexWrapper.allocateConcept(`stress-${i}`, 100000);
          } catch (error) {
            if (error.message.includes('memory')) {
              results.phases.push({ phase: 'stress_limit_reached', error: error.message });
              break;
            }
          }
        }
        
        // Phase 4: Recovery - cleanup and verify system still functional
        await window.cortexWrapper.cleanup();
        const postCleanupMemory = window.cortexWrapper.getMemoryUsage();
        results.phases.push({ phase: 'recovery', memory: postCleanupMemory });
        
        // Verify system still functional
        const testAllocation = await window.cortexWrapper.allocateConcept('recovery-test', 100);
        results.phases.push({ phase: 'functional_test', success: !!testAllocation });
        
      } catch (error) {
        results.crashed = true;
        results.error = error.message;
      }
      
      return results;
    });

    expect(memoryStressTest.crashed).toBe(false);
    expect(memoryStressTest.phases.length).toBeGreaterThan(3);
    
    const recoveryPhase = memoryStressTest.phases.find(p => p.phase === 'recovery');
    const functionalTest = memoryStressTest.phases.find(p => p.phase === 'functional_test');
    
    expect(recoveryPhase).toBeDefined();
    expect(functionalTest.success).toBe(true);
  });
});
```

## Expected Outputs
- Comprehensive integration test suite covering all system workflows
- Cross-browser compatibility validation across Chrome, Firefox, Safari
- Performance benchmarks for complete user scenarios and system stress testing
- Data persistence verification across browser sessions and storage limits
- Automated test execution pipeline with detailed reporting and failure analysis

## Validation
1. Verify end-to-end workflows complete successfully in under 30 seconds
2. Confirm WASM-JavaScript communication handles concurrent operations without data corruption
3. Test storage systems maintain data integrity across browser restarts and tab synchronization
4. Validate system performance remains stable under heavy load and memory pressure
5. Ensure integration tests catch regressions in cross-component interactions

## Next Steps
- Proceed to micro-phase 9.44 (Browser Compatibility Tests)
- Configure continuous integration with automated test execution
- Set up performance monitoring and regression detection