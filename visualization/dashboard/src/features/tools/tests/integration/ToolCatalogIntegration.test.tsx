import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { configureStore } from '@reduxjs/toolkit';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { ThemeProvider } from '../../../../components/ThemeProvider/ThemeProvider';
import toolsSlice from '../../stores/toolsSlice';
import { ToolRegistry } from '../../services/ToolRegistry';
import { ToolStatusMonitor } from '../../services/ToolStatusMonitor';
import { ToolAnalytics } from '../../services/ToolAnalytics';

// Mock components that will be tested
import { ToolCatalog } from '../../components/catalog/ToolCatalog';
import { StatusDashboard } from '../../components/monitoring/StatusDashboard';
import { ToolTester } from '../../components/testing/ToolTester';
import { PerformanceDashboard } from '../../components/analytics/PerformanceDashboard';
import { ToolDocViewer } from '../../components/documentation/ToolDocViewer';

// Mock WebSocket for real-time features
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  readyState: WebSocket.OPEN,
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
};

// Mock tool data for testing
const mockTools = [
  {
    id: 'tool-1',
    name: 'Knowledge Graph Query',
    version: '1.0.0',
    description: 'Query the knowledge graph using natural language',
    category: 'knowledge-graph' as const,
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural language query' }
      },
      required: ['query']
    },
    outputSchema: {
      type: 'object',
      properties: {
        results: { type: 'array', items: { type: 'object' } }
      }
    },
    status: {
      available: true,
      health: 'healthy' as const,
      lastChecked: new Date(),
      responseTime: 150,
      errorRate: 0.02,
    },
    metrics: {
      totalExecutions: 1250,
      successRate: 0.98,
      averageResponseTime: 145,
      p95ResponseTime: 280,
      p99ResponseTime: 450,
      lastExecutionTime: new Date(),
      errorCount: 25,
      errorTypes: {
        'timeout': 15,
        'validation': 10
      }
    },
    tags: ['nlp', 'query', 'graph'],
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date(),
  },
  {
    id: 'tool-2', 
    name: 'Neural Network Analyzer',
    version: '2.1.0',
    description: 'Analyze neural network patterns and performance',
    category: 'neural' as const,
    inputSchema: {
      type: 'object',
      properties: {
        networkData: { type: 'object', description: 'Neural network data' }
      },
      required: ['networkData']
    },
    status: {
      available: true,
      health: 'degraded' as const,
      lastChecked: new Date(),
      responseTime: 320,
      errorRate: 0.08,
      message: 'High response time detected'
    },
    metrics: {
      totalExecutions: 890,
      successRate: 0.92,
      averageResponseTime: 310,
      p95ResponseTime: 580,
      p99ResponseTime: 850,
      lastExecutionTime: new Date(Date.now() - 3600000), // 1 hour ago
      errorCount: 71,
      errorTypes: {
        'memory': 45,
        'timeout': 26
      }
    },
    tags: ['neural', 'analysis', 'performance'],
    createdAt: new Date('2024-02-01'),
    updatedAt: new Date(),
  },
  {
    id: 'tool-3',
    name: 'Memory Store Manager',
    version: '1.5.2',
    description: 'Manage persistent memory storage and retrieval',
    category: 'memory' as const,
    inputSchema: {
      type: 'object',
      properties: {
        operation: { type: 'string', enum: ['store', 'retrieve', 'delete'] },
        data: { type: 'any' }
      },
      required: ['operation']
    },
    status: {
      available: false,
      health: 'unavailable' as const,
      lastChecked: new Date(),
      responseTime: 0,
      errorRate: 1.0,
      message: 'Service temporarily unavailable'
    },
    metrics: {
      totalExecutions: 2340,
      successRate: 0.95,
      averageResponseTime: 85,
      p95ResponseTime: 150,
      p99ResponseTime: 220,
      lastExecutionTime: new Date(Date.now() - 7200000), // 2 hours ago
      errorCount: 117,
      errorTypes: {
        'connection': 78,
        'authorization': 39
      }
    },
    tags: ['memory', 'storage', 'persistence'],
    createdAt: new Date('2023-12-10'),
    updatedAt: new Date(),
  }
];

// Test store setup
const createTestStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      tools: toolsSlice
    },
    preloadedState: {
      tools: {
        tools: mockTools,
        loading: false,
        error: null,
        executions: {},
        executionHistory: [],
        selectedToolId: null,
        filters: {
          searchTerm: '',
          categories: [],
          status: [],
          tags: []
        },
        view: 'grid',
        sortBy: 'name',
        sortOrder: 'asc',
        ...initialState
      }
    }
  });
};

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode; store?: any }> = ({ 
  children, 
  store = createTestStore() 
}) => (
  <Provider store={store}>
    <BrowserRouter>
      <ThemeProvider defaultMode="dark">
        {children}
      </ThemeProvider>
    </BrowserRouter>
  </Provider>
);

describe('Tool Catalog Integration Tests', () => {
  let mockRegistry: ToolRegistry;
  let mockStatusMonitor: ToolStatusMonitor;
  let mockAnalytics: ToolAnalytics;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Mock WebSocket globally
    global.WebSocket = jest.fn(() => mockWebSocket) as any;
    
    // Initialize services
    mockRegistry = ToolRegistry.getInstance();
    mockTools.forEach(tool => mockRegistry.registerTool(tool));
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Component Integration', () => {
    test('should render tool catalog with all tools', async () => {
      render(
        <TestWrapper>
          <ToolCatalog />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Knowledge Graph Query')).toBeInTheDocument();
        expect(screen.getByText('Neural Network Analyzer')).toBeInTheDocument();
        expect(screen.getByText('Memory Store Manager')).toBeInTheDocument();
      });
    });

    test('should filter tools by category', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <ToolCatalog />
        </TestWrapper>
      );

      // Find and click the category filter
      const categoryFilter = screen.getByRole('button', { name: /neural/i });
      await user.click(categoryFilter);

      await waitFor(() => {
        expect(screen.getByText('Neural Network Analyzer')).toBeInTheDocument();
        expect(screen.queryByText('Knowledge Graph Query')).not.toBeInTheDocument();
      });
    });

    test('should search tools by name and description', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <ToolCatalog />
        </TestWrapper>
      );

      const searchInput = screen.getByPlaceholderText(/search tools/i);
      await user.type(searchInput, 'knowledge graph');

      await waitFor(() => {
        expect(screen.getByText('Knowledge Graph Query')).toBeInTheDocument();
        expect(screen.queryByText('Neural Network Analyzer')).not.toBeInTheDocument();
        expect(screen.queryByText('Memory Store Manager')).not.toBeInTheDocument();
      });
    });
  });

  describe('Status Monitoring Integration', () => {
    test('should display tool health status correctly', async () => {
      render(
        <TestWrapper>
          <StatusDashboard />
        </TestWrapper>
      );

      await waitFor(() => {
        // Check healthy tool
        expect(screen.getByText(/healthy/i)).toBeInTheDocument();
        
        // Check degraded tool
        expect(screen.getByText(/degraded/i)).toBeInTheDocument();
        
        // Check unavailable tool
        expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
      });
    });

    test('should update status in real-time', async () => {
      const store = createTestStore();
      
      render(
        <TestWrapper store={store}>
          <StatusDashboard />
        </TestWrapper>
      );

      // Simulate real-time status update
      const updatedStatus = {
        available: true,
        health: 'healthy' as const,
        lastChecked: new Date(),
        responseTime: 180,
        errorRate: 0.01,
      };

      // Mock WebSocket message
      const statusUpdateEvent = new MessageEvent('message', {
        data: JSON.stringify({
          type: 'tool-status-update',
          toolId: 'tool-2',
          status: updatedStatus
        })
      });

      // Simulate receiving status update
      if (mockWebSocket.addEventListener.mock.calls.length > 0) {
        const messageHandler = mockWebSocket.addEventListener.mock.calls
          .find(call => call[0] === 'message')[1];
        messageHandler(statusUpdateEvent);
      }

      await waitFor(() => {
        // Status should be updated in the UI
        expect(screen.queryByText(/high response time detected/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Tool Testing Integration', () => {
    test('should execute tool and display results', async () => {
      const user = userEvent.setup();
      const mockTool = mockTools[0];
      
      render(
        <TestWrapper>
          <ToolTester toolId={mockTool.id} />
        </TestWrapper>
      );

      // Fill in input parameters
      const queryInput = screen.getByLabelText(/query/i);
      await user.type(queryInput, 'test query');

      // Execute tool
      const executeButton = screen.getByRole('button', { name: /execute/i });
      await user.click(executeButton);

      await waitFor(() => {
        expect(screen.getByText(/executing/i)).toBeInTheDocument();
      });

      // Mock successful execution
      await waitFor(() => {
        expect(screen.getByText(/success/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    test('should handle tool execution errors gracefully', async () => {
      const user = userEvent.setup();
      const mockTool = mockTools[2]; // Unavailable tool
      
      render(
        <TestWrapper>
          <ToolTester toolId={mockTool.id} />
        </TestWrapper>
      );

      // Try to execute unavailable tool
      const executeButton = screen.getByRole('button', { name: /execute/i });
      await user.click(executeButton);

      await waitFor(() => {
        expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
      });
    });
  });

  describe('Analytics Integration', () => {
    test('should display performance metrics', async () => {
      render(
        <TestWrapper>
          <PerformanceDashboard />
        </TestWrapper>
      );

      await waitFor(() => {
        // Check for metric displays
        expect(screen.getByText(/total executions/i)).toBeInTheDocument();
        expect(screen.getByText(/success rate/i)).toBeInTheDocument();
        expect(screen.getByText(/average response time/i)).toBeInTheDocument();
      });
    });

    test('should update metrics after tool execution', async () => {
      const store = createTestStore();
      
      render(
        <TestWrapper store={store}>
          <PerformanceDashboard />
        </TestWrapper>
      );

      // Simulate tool execution completion
      const executionData = {
        toolId: 'tool-1',
        responseTime: 120,
        success: true,
        timestamp: Date.now()
      };

      // Mock execution completion event
      const metricsUpdateEvent = new MessageEvent('message', {
        data: JSON.stringify({
          type: 'execution-completed',
          data: executionData
        })
      });

      if (mockWebSocket.addEventListener.mock.calls.length > 0) {
        const messageHandler = mockWebSocket.addEventListener.mock.calls
          .find(call => call[0] === 'message')[1];
        messageHandler(metricsUpdateEvent);
      }

      await waitFor(() => {
        // Metrics should be updated
        expect(screen.getByText(/1251/)).toBeInTheDocument(); // Updated total executions
      });
    });
  });

  describe('Documentation Integration', () => {
    test('should display tool documentation', async () => {
      const mockTool = mockTools[0];
      
      render(
        <TestWrapper>
          <ToolDocViewer toolId={mockTool.id} />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText(mockTool.name)).toBeInTheDocument();
        expect(screen.getByText(mockTool.description)).toBeInTheDocument();
        expect(screen.getByText(/parameters/i)).toBeInTheDocument();
      });
    });

    test('should generate code examples for different languages', async () => {
      const user = userEvent.setup();
      const mockTool = mockTools[0];
      
      render(
        <TestWrapper>
          <ToolDocViewer toolId={mockTool.id} />
        </TestWrapper>
      );

      // Check for code examples section
      await waitFor(() => {
        expect(screen.getByText(/examples/i)).toBeInTheDocument();
      });

      // Check for language tabs
      const jsTab = screen.getByRole('tab', { name: /javascript/i });
      const pythonTab = screen.getByRole('tab', { name: /python/i });
      
      expect(jsTab).toBeInTheDocument();
      expect(pythonTab).toBeInTheDocument();

      // Switch to Python tab
      await user.click(pythonTab);
      
      await waitFor(() => {
        expect(screen.getByText(/import/)).toBeInTheDocument();
      });
    });
  });

  describe('Data Flow Integration', () => {
    test('should maintain consistent state across all components', async () => {
      const store = createTestStore();
      
      const { rerender } = render(
        <TestWrapper store={store}>
          <div>
            <ToolCatalog />
            <StatusDashboard />
            <PerformanceDashboard />
          </div>
        </TestWrapper>
      );

      // Verify initial state
      await waitFor(() => {
        expect(screen.getByText('Knowledge Graph Query')).toBeInTheDocument();
        expect(screen.getByText(/healthy/i)).toBeInTheDocument();
        expect(screen.getByText(/1250/)).toBeInTheDocument(); // Total executions
      });

      // Update a tool's status
      store.dispatch({
        type: 'tools/updateTool',
        payload: {
          id: 'tool-1',
          updates: {
            status: {
              available: true,
              health: 'degraded',
              lastChecked: new Date(),
              responseTime: 250,
              errorRate: 0.05,
              message: 'Increased response time'
            }
          }
        }
      });

      // All components should reflect the update
      await waitFor(() => {
        expect(screen.getByText(/degraded/i)).toBeInTheDocument();
        expect(screen.getByText(/increased response time/i)).toBeInTheDocument();
      });
    });
  });

  describe('Performance Integration', () => {
    test('should handle large number of tools efficiently', async () => {
      // Create a store with many tools
      const largeMockTools = Array.from({ length: 100 }, (_, i) => ({
        ...mockTools[0],
        id: `tool-${i}`,
        name: `Test Tool ${i}`,
        description: `Test tool number ${i}`
      }));

      const largeStore = createTestStore({
        tools: largeMockTools
      });

      const startTime = performance.now();
      
      render(
        <TestWrapper store={largeStore}>
          <ToolCatalog />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Tool 0')).toBeInTheDocument();
      });

      const endTime = performance.now();
      const renderTime = endTime - startTime;

      // Should render within reasonable time (< 1000ms)
      expect(renderTime).toBeLessThan(1000);
    });

    test('should debounce search input for performance', async () => {
      const user = userEvent.setup();
      jest.useFakeTimers();
      
      render(
        <TestWrapper>
          <ToolCatalog />
        </TestWrapper>
      );

      const searchInput = screen.getByPlaceholderText(/search tools/i);
      
      // Type multiple characters quickly
      await user.type(searchInput, 'test');
      
      // Fast-forward timers to trigger debounce
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        // Should only trigger search once after debounce delay
        expect(searchInput).toHaveValue('test');
      });

      jest.useRealTimers();
    });
  });

  describe('Error Handling Integration', () => {
    test('should handle service errors gracefully', async () => {
      // Mock service error
      const errorStore = createTestStore({
        error: 'Failed to load tools'
      });
      
      render(
        <TestWrapper store={errorStore}>
          <ToolCatalog />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText(/failed to load tools/i)).toBeInTheDocument();
      });
    });

    test('should recover from WebSocket connection errors', async () => {
      const store = createTestStore();
      
      render(
        <TestWrapper store={store}>
          <StatusDashboard />
        </TestWrapper>
      );

      // Simulate WebSocket error
      const errorEvent = new Event('error');
      if (mockWebSocket.addEventListener.mock.calls.length > 0) {
        const errorHandler = mockWebSocket.addEventListener.mock.calls
          .find(call => call[0] === 'error')[1];
        errorHandler(errorEvent);
      }

      // Should handle error gracefully without crashing
      await waitFor(() => {
        expect(screen.getByText(/connection/i)).toBeInTheDocument();
      });
    });
  });

  describe('Theme Integration', () => {
    test('should apply theme styles consistently across all components', async () => {
      render(
        <TestWrapper>
          <div>
            <ToolCatalog />
            <StatusDashboard />
            <ToolTester toolId="tool-1" />
          </div>
        </TestWrapper>
      );

      await waitFor(() => {
        // Check for dark theme CSS variables
        const catalogElement = screen.getByText('Knowledge Graph Query').closest('.tool-card');
        const styles = window.getComputedStyle(catalogElement!);
        
        // Should use CSS variables for theming
        expect(styles.getPropertyValue('background')).toContain('var(');
      });
    });
  });

  describe('Responsive Design Integration', () => {
    test('should adapt layout for different screen sizes', async () => {
      // Mock different viewport sizes
      const originalInnerWidth = window.innerWidth;
      
      // Test mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
      
      render(
        <TestWrapper>
          <ToolCatalog />
        </TestWrapper>
      );

      // Trigger resize event
      window.dispatchEvent(new Event('resize'));

      await waitFor(() => {
        const catalogGrid = screen.getByText('Knowledge Graph Query').closest('.tools-grid');
        const styles = window.getComputedStyle(catalogGrid!);
        
        // Should adapt grid for mobile
        expect(styles.gridTemplateColumns).toContain('1fr');
      });

      // Restore original width
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: originalInnerWidth,
      });
    });
  });
});

// Helper function to wait for async operations
export const waitForAsync = async (callback: () => void | Promise<void>, timeout = 1000) => {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(`Async operation timed out after ${timeout}ms`));
    }, timeout);

    Promise.resolve(callback())
      .then(() => {
        clearTimeout(timeoutId);
        resolve(undefined);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
};

// Performance testing utilities
export const measureRenderTime = async (component: React.ReactElement): Promise<number> => {
  const start = performance.now();
  render(component);
  const end = performance.now();
  return end - start;
};

// Mock data generators for large-scale testing
export const generateMockTools = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    ...mockTools[0],
    id: `generated-tool-${i}`,
    name: `Generated Tool ${i}`,
    description: `Auto-generated tool ${i} for testing purposes`,
    category: (['knowledge-graph', 'neural', 'memory', 'analysis'] as const)[i % 4],
  }));
};