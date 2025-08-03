# Micro-Phase 9.42: JavaScript Unit Tests

## Objective
Implement comprehensive JavaScript unit testing framework using Jest/Vitest to validate frontend components, WASM integration, and user interface functionality.

## Prerequisites
- Completed micro-phase 9.41 (WASM Unit Tests)
- JavaScript frontend components implemented (phases 9.16-9.35)
- WASM loader and API wrapper configured (phases 9.17-9.20)

## Task Description
Create robust JavaScript testing infrastructure covering React components, WASM bindings, storage systems, and user interactions. Implement test automation with mock services, coverage reporting, and accessibility testing.

## Specific Actions

1. **Configure JavaScript test environment**
```json
// package.json test configuration
{
  "scripts": {
    "test": "vitest",
    "test:watch": "vitest --watch",
    "test:coverage": "vitest --coverage",
    "test:ui": "vitest --ui",
    "test:e2e": "playwright test"
  },
  "devDependencies": {
    "vitest": "^1.0.0",
    "@vitest/ui": "^1.0.0",
    "@vitest/coverage-v8": "^1.0.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/jest-dom": "^6.0.0",
    "@testing-library/user-event": "^14.5.0",
    "jsdom": "^22.0.0",
    "msw": "^2.0.0",
    "fake-indexeddb": "^4.0.0"
  }
}
```

```javascript
// vitest.config.js
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.js'
      ]
    }
  }
});
```

2. **Create WASM integration tests**
```javascript
// src/test/wasm-integration.test.js
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { WASMLoader } from '../wasm/WASMLoader';
import { CortexKGWrapper } from '../wasm/CortexKGWrapper';

describe('WASM Integration', () => {
  let wasmLoader;
  let cortexWrapper;

  beforeEach(async () => {
    wasmLoader = new WASMLoader();
    await wasmLoader.initialize();
    cortexWrapper = new CortexKGWrapper(wasmLoader);
  });

  it('should load WASM module successfully', async () => {
    expect(wasmLoader.isLoaded()).toBe(true);
    expect(wasmLoader.getExports()).toBeDefined();
    expect(typeof wasmLoader.getExports().memory).toBe('object');
  });

  it('should create CortexKG instance with valid configuration', async () => {
    const config = {
      columns: 100,
      minicolumns_per_column: 32,
      cells_per_minicolumn: 8,
      proximal_threshold: 0.5,
      distal_threshold: 0.3
    };

    const cortex = await cortexWrapper.createInstance(config);
    
    expect(cortex).toBeDefined();
    expect(cortex.getColumnCount()).toBe(100);
    expect(cortex.getTotalCellCount()).toBe(100 * 32 * 8);
  });

  it('should handle WASM memory allocation properly', async () => {
    const initialMemory = wasmLoader.getMemoryUsage();
    
    // Allocate large concept
    const conceptId = await cortexWrapper.allocateConcept('test_concept', 1000);
    const afterAllocation = wasmLoader.getMemoryUsage();
    
    expect(afterAllocation.allocated).toBeGreaterThan(initialMemory.allocated);
    expect(conceptId).toBeTypeOf('number');
    
    // Cleanup
    await cortexWrapper.deallocateConcept(conceptId);
    const afterDeallocation = wasmLoader.getMemoryUsage();
    
    expect(afterDeallocation.allocated).toBe(initialMemory.allocated);
  });

  it('should process spatial pooling correctly', async () => {
    const cortex = await cortexWrapper.createInstance({
      columns: 10,
      minicolumns_per_column: 4,
      cells_per_minicolumn: 4
    });

    const inputPattern = [1, 0, 1, 1, 0, 0, 1, 0];
    const result = await cortex.spatialPooling(inputPattern);

    expect(result.activeColumns).toBeInstanceOf(Array);
    expect(result.activeColumns.length).toBeGreaterThan(0);
    expect(result.overlapScores).toBeInstanceOf(Array);
    expect(result.boostFactors.every(f => f >= 1.0)).toBe(true);
  });

  it('should handle errors gracefully', async () => {
    const cortex = await cortexWrapper.createInstance({});
    
    // Test invalid input
    await expect(cortex.spatialPooling(null)).rejects.toThrow();
    await expect(cortex.allocateConcept('', -1)).rejects.toThrow();
    
    // Test memory overflow protection
    const largeSize = 1024 * 1024 * 100; // 100MB
    await expect(cortex.allocateConcept('huge', largeSize))
      .rejects.toThrow(/memory limit/i);
  });
});
```

3. **Create React component tests**
```javascript
// src/test/components/CorticalVisualizer.test.jsx
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CorticalVisualizer } from '../../components/CorticalVisualizer';
import { WASMContext } from '../../context/WASMContext';

const mockWASMContext = {
  wasmLoader: {
    isLoaded: () => true,
    getExports: () => ({ memory: new WebAssembly.Memory({ initial: 1 }) })
  },
  cortexWrapper: {
    createInstance: vi.fn(),
    allocateConcept: vi.fn(),
    spatialPooling: vi.fn()
  }
};

describe('CorticalVisualizer Component', () => {
  const user = userEvent.setup();

  const renderWithContext = (props = {}) => {
    return render(
      <WASMContext.Provider value={mockWASMContext}>
        <CorticalVisualizer {...props} />
      </WASMContext.Provider>
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render canvas element', () => {
    renderWithContext();
    
    const canvas = screen.getByRole('img', { name: /cortical column visualization/i });
    expect(canvas).toBeInTheDocument();
    expect(canvas.tagName).toBe('CANVAS');
  });

  it('should initialize with default configuration', () => {
    renderWithContext();
    
    expect(mockWASMContext.cortexWrapper.createInstance).toHaveBeenCalledWith({
      columns: 100,
      minicolumns_per_column: 32,
      cells_per_minicolumn: 8,
      proximal_threshold: 0.5,
      distal_threshold: 0.3
    });
  });

  it('should handle column click interactions', async () => {
    const onColumnClick = vi.fn();
    renderWithContext({ onColumnClick });
    
    const canvas = screen.getByRole('img');
    
    // Simulate click on canvas
    fireEvent.click(canvas, { clientX: 100, clientY: 100 });
    
    await waitFor(() => {
      expect(onColumnClick).toHaveBeenCalledWith(
        expect.objectContaining({
          columnIndex: expect.any(Number),
          position: expect.objectContaining({
            x: expect.any(Number),
            y: expect.any(Number)
          })
        })
      );
    });
  });

  it('should update visualization when props change', async () => {
    const { rerender } = renderWithContext({ columns: 50 });
    
    // Change configuration
    rerender(
      <WASMContext.Provider value={mockWASMContext}>
        <CorticalVisualizer columns={200} />
      </WASMContext.Provider>
    );

    await waitFor(() => {
      expect(mockWASMContext.cortexWrapper.createInstance).toHaveBeenCalledWith(
        expect.objectContaining({ columns: 200 })
      );
    });
  });

  it('should handle touch gestures on mobile', async () => {
    const onGesture = vi.fn();
    renderWithContext({ onGesture });
    
    const canvas = screen.getByRole('img');
    
    // Simulate pinch gesture
    fireEvent.touchStart(canvas, {
      touches: [
        { clientX: 100, clientY: 100 },
        { clientX: 200, clientY: 200 }
      ]
    });
    
    fireEvent.touchMove(canvas, {
      touches: [
        { clientX: 90, clientY: 90 },
        { clientX: 210, clientY: 210 }
      ]
    });
    
    fireEvent.touchEnd(canvas);
    
    await waitFor(() => {
      expect(onGesture).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'pinch',
          scale: expect.any(Number)
        })
      );
    });
  });
});
```

4. **Create storage system tests**
```javascript
// src/test/storage/IndexedDBWrapper.test.js
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import FDBFactory from 'fake-indexeddb/lib/FDBFactory';
import FDBKeyRange from 'fake-indexeddb/lib/FDBKeyRange';
import { IndexedDBWrapper } from '../../storage/IndexedDBWrapper';

// Setup fake IndexedDB
global.indexedDB = new FDBFactory();
global.IDBKeyRange = FDBKeyRange;

describe('IndexedDBWrapper', () => {
  let dbWrapper;

  beforeEach(async () => {
    dbWrapper = new IndexedDBWrapper('test-cortex-db', 1);
    await dbWrapper.initialize();
  });

  afterEach(async () => {
    if (dbWrapper) {
      await dbWrapper.close();
    }
    // Clean up test database
    const deleteReq = indexedDB.deleteDatabase('test-cortex-db');
    await new Promise((resolve) => {
      deleteReq.onsuccess = resolve;
      deleteReq.onerror = resolve;
    });
  });

  it('should create database with correct schema', async () => {
    const stores = await dbWrapper.getStoreNames();
    
    expect(stores).toContain('concepts');
    expect(stores).toContain('allocations');
    expect(stores).toContain('metadata');
    expect(stores).toContain('cache');
  });

  it('should store and retrieve concepts', async () => {
    const concept = {
      id: 'test-concept-1',
      name: 'Test Concept',
      data: new Uint8Array([1, 2, 3, 4]),
      timestamp: Date.now(),
      size: 4
    };

    await dbWrapper.storeConcept(concept);
    const retrieved = await dbWrapper.getConcept('test-concept-1');

    expect(retrieved).toEqual(concept);
    expect(retrieved.data).toBeInstanceOf(Uint8Array);
    expect(Array.from(retrieved.data)).toEqual([1, 2, 3, 4]);
  });

  it('should handle allocation metadata', async () => {
    const allocation = {
      id: 'alloc-1',
      conceptId: 'concept-1',
      columnIndex: 42,
      minicolumnIndex: 15,
      cellIndices: [1, 5, 12],
      strength: 0.85,
      timestamp: Date.now()
    };

    await dbWrapper.storeAllocation(allocation);
    const retrieved = await dbWrapper.getAllocation('alloc-1');

    expect(retrieved).toEqual(allocation);
  });

  it('should implement caching with TTL', async () => {
    const cacheEntry = {
      key: 'cache-test',
      value: { data: 'test-data' },
      ttl: 1000 // 1 second
    };

    await dbWrapper.setCache(cacheEntry.key, cacheEntry.value, cacheEntry.ttl);
    
    // Should retrieve immediately
    let cached = await dbWrapper.getCache(cacheEntry.key);
    expect(cached).toEqual(cacheEntry.value);
    
    // Wait for expiration
    await new Promise(resolve => setTimeout(resolve, 1100));
    
    // Should be expired
    cached = await dbWrapper.getCache(cacheEntry.key);
    expect(cached).toBeNull();
  });

  it('should handle storage quota limits', async () => {
    // Mock storage quota
    const originalEstimate = navigator.storage?.estimate;
    navigator.storage = {
      estimate: vi.fn().mockResolvedValue({
        quota: 1024 * 1024, // 1MB
        usage: 900 * 1024   // 900KB used
      })
    };

    const quota = await dbWrapper.checkStorageQuota();
    
    expect(quota.available).toBe(1024 * 1024);
    expect(quota.used).toBe(900 * 1024);
    expect(quota.remaining).toBe(124 * 1024);
    expect(quota.usagePercentage).toBeCloseTo(87.89, 1);

    // Restore original
    if (originalEstimate) {
      navigator.storage.estimate = originalEstimate;
    }
  });

  it('should perform batch operations efficiently', async () => {
    const concepts = Array.from({ length: 100 }, (_, i) => ({
      id: `concept-${i}`,
      name: `Concept ${i}`,
      data: new Uint8Array([i, i + 1, i + 2]),
      timestamp: Date.now() + i,
      size: 3
    }));

    const startTime = performance.now();
    await dbWrapper.batchStoreConcepts(concepts);
    const endTime = performance.now();

    // Should complete batch operation quickly
    expect(endTime - startTime).toBeLessThan(100);

    // Verify all concepts stored
    const count = await dbWrapper.getConceptCount();
    expect(count).toBe(100);
  });
});
```

5. **Create performance and accessibility tests**
```javascript
// src/test/performance/Performance.test.js
import { describe, it, expect, vi } from 'vitest';
import { PerformanceMonitor } from '../../monitoring/PerformanceMonitor';
import { renderHook, act } from '@testing-library/react';
import { usePerformanceMetrics } from '../../hooks/usePerformanceMetrics';

describe('Performance Testing', () => {
  it('should track component render times', async () => {
    const monitor = new PerformanceMonitor();
    monitor.initialize();

    const { result } = renderHook(() => usePerformanceMetrics());

    act(() => {
      result.current.startMeasurement('component-render');
    });

    // Simulate component work
    await new Promise(resolve => setTimeout(resolve, 10));

    act(() => {
      result.current.endMeasurement('component-render');
    });

    const metrics = result.current.getMetrics();
    expect(metrics['component-render']).toBeGreaterThan(0);
    expect(metrics['component-render']).toBeLessThan(100);
  });

  it('should detect memory leaks in component lifecycle', async () => {
    const initialMemory = performance.memory?.usedJSHeapSize || 0;
    
    // Create many components
    const components = [];
    for (let i = 0; i < 1000; i++) {
      const { unmount } = render(<CorticalVisualizer key={i} />);
      components.push(unmount);
    }

    // Unmount all components
    components.forEach(unmount => unmount());

    // Force garbage collection (if available)
    if (global.gc) {
      global.gc();
    }

    // Wait for cleanup
    await new Promise(resolve => setTimeout(resolve, 100));

    const finalMemory = performance.memory?.usedJSHeapSize || 0;
    const memoryIncrease = finalMemory - initialMemory;

    // Memory increase should be minimal after cleanup
    expect(memoryIncrease).toBeLessThan(1024 * 1024); // Less than 1MB
  });
});

// src/test/accessibility/Accessibility.test.jsx
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { CorticalVisualizer } from '../../components/CorticalVisualizer';
import { QueryInterface } from '../../components/QueryInterface';

expect.extend(toHaveNoViolations);

describe('Accessibility Testing', () => {
  it('should have no accessibility violations in CorticalVisualizer', async () => {
    const { container } = render(<CorticalVisualizer />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('should have proper ARIA labels and roles', () => {
    render(<QueryInterface />);
    
    const searchInput = screen.getByRole('textbox', { name: /search concepts/i });
    const submitButton = screen.getByRole('button', { name: /search/i });
    
    expect(searchInput).toHaveAttribute('aria-label');
    expect(submitButton).toHaveAttribute('aria-describedby');
  });

  it('should support keyboard navigation', async () => {
    const user = userEvent.setup();
    render(<QueryInterface />);
    
    const searchInput = screen.getByRole('textbox');
    const submitButton = screen.getByRole('button');
    
    // Tab navigation
    await user.tab();
    expect(searchInput).toHaveFocus();
    
    await user.tab();
    expect(submitButton).toHaveFocus();
    
    // Enter key submission
    await user.type(searchInput, 'test query');
    await user.keyboard('{Enter}');
    
    // Should trigger search
    expect(searchInput.value).toBe('test query');
  });
});
```

## Expected Outputs
- Complete JavaScript test suite with 90%+ code coverage
- Automated testing pipeline with CI/CD integration
- Performance benchmarks for critical UI components and WASM interactions
- Accessibility compliance verification across all user interfaces
- Mock services for isolated testing of complex integrations

## Validation
1. Verify all React components render correctly with various prop combinations
2. Confirm WASM integration tests cover all binding scenarios and error cases
3. Test storage operations handle edge cases and quota limits appropriately
4. Validate performance tests detect regression in component render times
5. Ensure accessibility tests identify and prevent WCAG compliance violations

## Next Steps
- Proceed to micro-phase 9.43 (Integration Tests)
- Configure automated testing in CI/CD pipeline
- Set up test coverage reporting and quality gates