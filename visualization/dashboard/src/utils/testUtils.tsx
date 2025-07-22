import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../components/ThemeProvider/ThemeProvider';
import { WebSocketProvider } from '../providers/WebSocketProvider';
import { MCPProvider } from '../providers/MCPProvider';
import { store } from '../stores';
import { WebSocketStatus } from '../types';

// Mock WebSocket for testing
export class MockWebSocket {
  static instances: MockWebSocket[] = [];
  
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    (global as any).WebSocket.instances = MockWebSocket.instances;
    
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }

  send(data: string) {
    // Simulate echo for testing
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', { data }));
      }
    }, 5);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

// Mock data generators
export const mockKnowledgeGraphData = () => ({
  nodes: [
    { id: '1', label: 'Entity A', type: 'concept', properties: { confidence: 0.9 } },
    { id: '2', label: 'Entity B', type: 'fact', properties: { confidence: 0.85 } },
    { id: '3', label: 'Entity C', type: 'relationship', properties: { confidence: 0.95 } },
  ],
  edges: [
    { source: '1', target: '2', type: 'related_to', properties: { weight: 0.8 } },
    { source: '2', target: '3', type: 'implies', properties: { weight: 0.7 } },
  ],
});

export const mockCognitivePatterns = () => ({
  patterns: [
    {
      id: 'pattern1',
      name: 'Pattern Alpha',
      activation: 0.75,
      connections: 42,
      lastUpdated: new Date().toISOString(),
    },
    {
      id: 'pattern2',
      name: 'Pattern Beta',
      activation: 0.62,
      connections: 38,
      lastUpdated: new Date().toISOString(),
    },
  ],
  timestamp: Date.now(),
});

export const mockMemoryMetrics = () => ({
  workingMemory: {
    capacity: 7,
    used: 5,
    efficiency: 0.85,
  },
  longTermMemory: {
    totalItems: 1523,
    recentAccess: 42,
    consolidationRate: 0.92,
  },
  timestamp: Date.now(),
});

export const mockNeuralActivity = () => {
  const size = 10;
  const data: number[][] = [];
  for (let i = 0; i < size; i++) {
    data[i] = [];
    for (let j = 0; j < size; j++) {
      data[i][j] = Math.random();
    }
  }
  return { data, timestamp: Date.now() };
};

// Custom render function with all providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialState?: any;
  wsUrl?: string;
}

export function renderWithProviders(
  ui: ReactElement,
  {
    initialState,
    wsUrl = 'ws://localhost:8080',
    ...renderOptions
  }: CustomRenderOptions = {}
): ReturnType<typeof render> {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <Provider store={store}>
        <BrowserRouter>
          <ThemeProvider>
            <WebSocketProvider url={wsUrl}>
              <MCPProvider>
                {children}
              </MCPProvider>
            </WebSocketProvider>
          </ThemeProvider>
        </BrowserRouter>
      </Provider>
    );
  }

  return render(ui, { wrapper: Wrapper, ...renderOptions });
}

// Test utilities for responsive testing
export const setViewport = (width: number, height: number) => {
  Object.defineProperty(window, 'innerWidth', {
    writable: true,
    configurable: true,
    value: width,
  });
  Object.defineProperty(window, 'innerHeight', {
    writable: true,
    configurable: true,
    value: height,
  });
  window.dispatchEvent(new Event('resize'));
};

export const viewportSizes = {
  mobile: { width: 375, height: 667 },
  tablet: { width: 768, height: 1024 },
  desktop: { width: 1920, height: 1080 },
};

// Async utilities
export const waitForWebSocketConnection = async (
  getStatus: () => WebSocketStatus,
  timeout = 5000
): Promise<void> => {
  const startTime = Date.now();
  while (getStatus() !== 'connected' && Date.now() - startTime < timeout) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  if (getStatus() !== 'connected') {
    throw new Error('WebSocket connection timeout');
  }
};

// Performance testing utilities
export const measureRenderTime = async (
  component: ReactElement,
  iterations = 10
): Promise<{ mean: number; min: number; max: number }> => {
  const times: number[] = [];
  
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    const { unmount } = renderWithProviders(component);
    const end = performance.now();
    times.push(end - start);
    unmount();
  }
  
  return {
    mean: times.reduce((a, b) => a + b) / times.length,
    min: Math.min(...times),
    max: Math.max(...times),
  };
};

// Mock server messages
export const mockServerMessages = {
  knowledgeGraph: {
    type: 'knowledge_graph_update',
    data: mockKnowledgeGraphData(),
  },
  cognitivePatterns: {
    type: 'cognitive_patterns_update',
    data: mockCognitivePatterns(),
  },
  memoryMetrics: {
    type: 'memory_metrics_update',
    data: mockMemoryMetrics(),
  },
  neuralActivity: {
    type: 'neural_activity_update',
    data: mockNeuralActivity(),
  },
};

// Test assertion helpers
export const expectNoConsoleErrors = () => {
  const originalError = console.error;
  const errors: any[] = [];
  
  beforeEach(() => {
    console.error = jest.fn((...args) => {
      errors.push(args);
      originalError(...args);
    });
  });
  
  afterEach(() => {
    console.error = originalError;
    if (errors.length > 0) {
      throw new Error(`Console errors detected: ${JSON.stringify(errors)}`);
    }
    errors.length = 0;
  });
};

export * from '@testing-library/react';