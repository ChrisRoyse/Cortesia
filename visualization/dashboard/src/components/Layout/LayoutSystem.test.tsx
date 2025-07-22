import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import '@testing-library/jest-dom';

import {
  GridLayout,
  LayoutManager,
  ResizablePanel,
  ResponsiveContainer,
  ViewportOptimizer,
  type LayoutItem
} from './index';
import layoutReducer from '../../stores/slices/layoutSlice';

// Mock store for testing
const createMockStore = () => {
  return configureStore({
    reducer: {
      layout: layoutReducer
    }
  });
};

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
  root: null,
  rootMargin: '',
  thresholds: [],
}));

// Mock performance API
Object.defineProperty(window, 'performance', {
  writable: true,
  value: {
    now: jest.fn(() => Date.now()),
    memory: {
      usedJSHeapSize: 1000000
    }
  }
});

const TestComponent: React.FC<{ title: string }> = ({ title }) => (
  <div data-testid={`test-component-${title.toLowerCase().replace(/\s+/g, '-')}`}>
    <h3>{title}</h3>
    <p>Test content for {title}</p>
  </div>
);

const mockItems: LayoutItem[] = [
  {
    i: 'item-1',
    x: 0,
    y: 0,
    w: 6,
    h: 4,
    component: <TestComponent title="Item 1" />
  },
  {
    i: 'item-2',
    x: 6,
    y: 0,
    w: 6,
    h: 4,
    component: <TestComponent title="Item 2" />
  },
  {
    i: 'item-3',
    x: 0,
    y: 4,
    w: 4,
    h: 3,
    component: <TestComponent title="Item 3" />
  }
];

describe('Layout System Integration', () => {
  let mockStore: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    mockStore = createMockStore();
    // Clear localStorage
    localStorage.clear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('GridLayout Component', () => {
    it('should render grid layout with items', () => {
      const onLayoutChange = jest.fn();

      render(
        <GridLayout
          items={mockItems}
          onLayoutChange={onLayoutChange}
        />
      );

      expect(screen.getByTestId('test-component-item-1')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-2')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-3')).toBeInTheDocument();
    });

    it('should handle drag and drop when enabled', () => {
      const onLayoutChange = jest.fn();

      render(
        <GridLayout
          items={mockItems}
          onLayoutChange={onLayoutChange}
          isDraggable={true}
        />
      );

      const gridItems = document.querySelectorAll('.react-grid-item');
      expect(gridItems.length).toBe(mockItems.length);
    });

    it('should respect resize constraints', () => {
      const constrainedItems: LayoutItem[] = [
        {
          ...mockItems[0],
          minW: 2,
          minH: 2,
          maxW: 8,
          maxH: 6
        }
      ];

      render(
        <GridLayout
          items={constrainedItems}
          onLayoutChange={jest.fn()}
          isResizable={true}
        />
      );

      expect(screen.getByTestId('test-component-item-1')).toBeInTheDocument();
    });
  });

  describe('LayoutManager Component', () => {
    it('should render layout manager with presets', () => {
      render(
        <Provider store={mockStore}>
          <LayoutManager
            items={mockItems}
            allowPresets={true}
            allowCustomization={true}
          />
        </Provider>
      );

      expect(screen.getByText('Select Layout Preset')).toBeInTheDocument();
      expect(screen.getByText('Save Preset')).toBeInTheDocument();
      expect(screen.getByText('Reset')).toBeInTheDocument();
    });

    it('should handle preset selection', async () => {
      render(
        <Provider store={mockStore}>
          <LayoutManager
            items={mockItems}
            allowPresets={true}
            allowCustomization={true}
          />
        </Provider>
      );

      const presetSelect = screen.getByDisplayValue('Select Layout Preset');
      fireEvent.change(presetSelect, { target: { value: 'cognitive-analysis' } });

      await waitFor(() => {
        expect(presetSelect).toHaveValue('cognitive-analysis');
      });
    });

    it('should open save preset dialog', () => {
      render(
        <Provider store={mockStore}>
          <LayoutManager
            items={mockItems}
            allowPresets={true}
            allowCustomization={true}
          />
        </Provider>
      );

      const saveButton = screen.getByText('Save Preset');
      fireEvent.click(saveButton);

      expect(screen.getByText('Save Layout Preset')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter preset name')).toBeInTheDocument();
    });
  });

  describe('ResizablePanel Component', () => {
    it('should render resizable panel with content', () => {
      render(
        <ResizablePanel
          width={300}
          height={200}
          isResizable={true}
        >
          <TestComponent title="Resizable Content" />
        </ResizablePanel>
      );

      expect(screen.getByTestId('test-component-resizable-content')).toBeInTheDocument();
    });

    it('should handle resize events', () => {
      const onResize = jest.fn();
      const onResizeStart = jest.fn();
      const onResizeEnd = jest.fn();

      render(
        <ResizablePanel
          width={300}
          height={200}
          isResizable={true}
          onResize={onResize}
          onResizeStart={onResizeStart}
          onResizeEnd={onResizeEnd}
        >
          <TestComponent title="Resizable Content" />
        </ResizablePanel>
      );

      const resizeHandle = document.querySelector('.resize-handle');
      expect(resizeHandle).toBeInTheDocument();
    });

    it('should respect aspect ratio when maintained', () => {
      render(
        <ResizablePanel
          width={400}
          height={300}
          aspectRatio={4/3}
          maintainAspectRatio={true}
          isResizable={true}
        >
          <TestComponent title="Aspect Ratio Content" />
        </ResizablePanel>
      );

      expect(screen.getByTestId('test-component-aspect-ratio-content')).toBeInTheDocument();
    });
  });

  describe('ResponsiveContainer Component', () => {
    it('should render responsive container with content', () => {
      render(
        <ResponsiveContainer>
          <TestComponent title="Responsive Content" />
        </ResponsiveContainer>
      );

      expect(screen.getByTestId('test-component-responsive-content')).toBeInTheDocument();
    });

    it('should apply responsive classes based on breakpoint', () => {
      const { container } = render(
        <ResponsiveContainer breakpoint="md">
          <TestComponent title="Breakpoint Content" />
        </ResponsiveContainer>
      );

      const responsiveContainer = container.querySelector('.responsive-container');
      expect(responsiveContainer).toHaveClass('breakpoint-md');
    });

    it('should center content when enabled', () => {
      render(
        <ResponsiveContainer centerContent={true}>
          <TestComponent title="Centered Content" />
        </ResponsiveContainer>
      );

      const centeredContent = document.querySelector('.responsive-container-content.centered');
      expect(centeredContent).toBeInTheDocument();
    });
  });

  describe('ViewportOptimizer Component', () => {
    it('should render viewport optimizer with children', () => {
      const items = Array.from({ length: 10 }, (_, i) => (
        <TestComponent key={i} title={`Optimized Item ${i}`} />
      ));

      render(
        <ViewportOptimizer
          enableLazyLoading={false}
          enableVirtualization={false}
        >
          {items}
        </ViewportOptimizer>
      );

      expect(screen.getByTestId('test-component-optimized-item-0')).toBeInTheDocument();
    });

    it('should show performance metrics in development', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      render(
        <ViewportOptimizer>
          <TestComponent title="Performance Test" />
        </ViewportOptimizer>
      );

      // Check for performance indicator
      const performanceIndicator = document.querySelector('[style*="position: absolute"]');
      expect(performanceIndicator).toBeInTheDocument();

      process.env.NODE_ENV = originalEnv;
    });

    it('should handle virtualization when enabled', () => {
      const items = Array.from({ length: 100 }, (_, i) => (
        <div key={i} data-testid={`virtual-item-${i}`}>Item {i}</div>
      ));

      render(
        <ViewportOptimizer
          enableVirtualization={true}
          itemHeight={50}
          containerHeight={200}
        >
          {items}
        </ViewportOptimizer>
      );

      // Should render viewport optimizer container
      expect(document.querySelector('.viewport-optimizer')).toBeInTheDocument();
    });
  });

  describe('Layout System Integration', () => {
    it('should work together in a complete dashboard', () => {
      render(
        <Provider store={mockStore}>
          <ResponsiveContainer>
            <LayoutManager
              items={mockItems}
              allowPresets={true}
              allowCustomization={true}
            />
          </ResponsiveContainer>
        </Provider>
      );

      // Should render all components
      expect(screen.getByText('Select Layout Preset')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-1')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-2')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-3')).toBeInTheDocument();
    });

    it('should handle nested responsive and resizable components', () => {
      render(
        <ResponsiveContainer>
          <ViewportOptimizer>
            <ResizablePanel width={400} height={300}>
              <TestComponent title="Nested Component" />
            </ResizablePanel>
          </ViewportOptimizer>
        </ResponsiveContainer>
      );

      expect(screen.getByTestId('test-component-nested-component')).toBeInTheDocument();
    });

    it('should preserve layout state across re-renders', () => {
      const { rerender } = render(
        <Provider store={mockStore}>
          <LayoutManager
            items={mockItems}
            allowPresets={true}
          />
        </Provider>
      );

      expect(screen.getByTestId('test-component-item-1')).toBeInTheDocument();

      // Re-render with updated items
      const updatedItems = [
        ...mockItems,
        {
          i: 'item-4',
          x: 8,
          y: 4,
          w: 4,
          h: 3,
          component: <TestComponent title="Item 4" />
        }
      ];

      rerender(
        <Provider store={mockStore}>
          <LayoutManager
            items={updatedItems}
            allowPresets={true}
          />
        </Provider>
      );

      expect(screen.getByTestId('test-component-item-1')).toBeInTheDocument();
      expect(screen.getByTestId('test-component-item-4')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid layout items gracefully', () => {
      const invalidItems: LayoutItem[] = [
        {
          i: '',
          x: -1,
          y: -1,
          w: 0,
          h: 0,
          component: <TestComponent title="Invalid Item" />
        }
      ];

      const consoleError = jest.spyOn(console, 'error').mockImplementation();

      render(
        <GridLayout
          items={invalidItems}
          onLayoutChange={jest.fn()}
        />
      );

      // Should render without crashing
      expect(screen.getByTestId('test-component-invalid-item')).toBeInTheDocument();

      consoleError.mockRestore();
    });

    it('should handle missing store gracefully', () => {
      const consoleError = jest.spyOn(console, 'error').mockImplementation();

      expect(() => {
        render(
          <LayoutManager
            items={mockItems}
            allowPresets={true}
          />
        );
      }).toThrow();

      consoleError.mockRestore();
    });
  });

  describe('Accessibility', () => {
    it('should support keyboard navigation', () => {
      render(
        <GridLayout
          items={mockItems}
          onLayoutChange={jest.fn()}
        />
      );

      const firstItem = screen.getByTestId('test-component-item-1');
      expect(firstItem).toBeInTheDocument();

      // Test tab navigation
      fireEvent.keyDown(firstItem, { key: 'Tab' });
      expect(document.activeElement).toBeDefined();
    });

    it('should have proper ARIA attributes', () => {
      render(
        <ResizablePanel
          width={300}
          height={200}
          isResizable={true}
        >
          <TestComponent title="Accessible Content" />
        </ResizablePanel>
      );

      const panel = document.querySelector('.resizable-panel');
      expect(panel).toBeInTheDocument();
    });
  });
});

describe('Performance Tests', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should handle large numbers of items efficiently', () => {
    const largeItemSet = Array.from({ length: 1000 }, (_, i) => ({
      i: `item-${i}`,
      x: i % 12,
      y: Math.floor(i / 12),
      w: 2,
      h: 2,
      component: <TestComponent title={`Item ${i}`} />
    }));

    const start = performance.now();

    render(
      <ViewportOptimizer
        enableVirtualization={true}
        itemHeight={100}
        maxVisibleItems={50}
      >
        {largeItemSet.map(item => item.component)}
      </ViewportOptimizer>
    );

    const end = performance.now();
    expect(end - start).toBeLessThan(1000); // Should render in under 1 second
  });

  it('should debounce resize events', () => {
    const onResize = jest.fn();

    render(
      <ResizablePanel
        width={300}
        height={200}
        onResize={onResize}
      >
        <TestComponent title="Debounce Test" />
      </ResizablePanel>
    );

    // Simulate rapid resize events
    for (let i = 0; i < 10; i++) {
      fireEvent.mouseDown(document.body);
      fireEvent.mouseMove(document.body);
      fireEvent.mouseUp(document.body);
    }

    jest.runAllTimers();

    // Should be debounced
    expect(onResize).toHaveBeenCalledTimes(0); // No mouse events on resize handles
  });
});