import React, { ReactElement } from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '../components/ThemeProvider/ThemeProvider';
import { WebSocketProvider } from '../providers/WebSocketProvider';
import { MCPProvider } from '../providers/MCPProvider';
import { store } from '../stores';

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
    
    // Simulate connection after next tick
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    console.log('MockWebSocket send:', data);
  }

  close(code?: number, reason?: string): void {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      const event = new CloseEvent('close', { code, reason });
      this.onclose(event);
    }
  }

  static reset(): void {
    MockWebSocket.instances = [];
  }

  // Utility method to simulate receiving messages
  simulateMessage(data: any): void {
    if (this.onmessage && this.readyState === WebSocket.OPEN) {
      const event = new MessageEvent('message', { data: JSON.stringify(data) });
      this.onmessage(event);
    }
  }

  // Utility method to simulate errors
  simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
}

// Test wrapper component
interface TestProviderProps {
  children: ReactElement;
  initialState?: any;
  mockWebSocket?: boolean;
}

export const TestProvider: React.FC<TestProviderProps> = ({ 
  children, 
  initialState,
  mockWebSocket = true 
}) => {
  return (
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider>
          <WebSocketProvider>
            <MCPProvider>
              {children}
            </MCPProvider>
          </WebSocketProvider>
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  );
};

// Test data generators
export const generateMockBrainData = () => ({
  entities: Array.from({ length: 50 }, (_, i) => ({
    id: `entity_${i}`,
    type_id: Math.floor(Math.random() * 3) + 1,
    properties: { name: `Entity ${i}` },
    embedding: Array.from({ length: 128 }, () => Math.random()),
    activation: Math.random(),
    direction: ['Input', 'Output', 'Hidden', 'Gate'][Math.floor(Math.random() * 4)],
    lastActivation: Date.now(),
    lastUpdate: Date.now(),
    conceptIds: []
  })),
  relationships: Array.from({ length: 100 }, (_, i) => ({
    from: `entity_${Math.floor(Math.random() * 50)}`,
    to: `entity_${Math.floor(Math.random() * 50)}`,
    relType: 1,
    weight: Math.random(),
    inhibitory: Math.random() > 0.8,
    temporalDecay: 0.1,
    lastActivation: Date.now(),
    usageCount: Math.floor(Math.random() * 100)
  })),
  concepts: [],
  logicGates: [],
  statistics: {
    entityCount: 50,
    relationshipCount: 100,
    avgActivation: 0.5,
    minActivation: 0,
    maxActivation: 1,
    totalActivation: 25,
    graphDensity: 0.04,
    clusteringCoefficient: 0.3,
    betweennessCentrality: 0.2,
    learningEfficiency: 0.85,
    conceptCoherence: 0.7,
    activeEntities: 30,
    avgRelationshipsPerEntity: 2,
    uniqueEntityTypes: 3
  },
  activationDistribution: {
    veryLow: 10,
    low: 10,
    medium: 15,
    high: 10,
    veryHigh: 5
  },
  metrics: {
    brain_clustering_coefficient: 0.3,
    brain_learning_efficiency: 0.85,
    brain_concept_coherence: 0.7,
    brain_active_entities: 30
  }
});

export const generateMockSystemMetrics = () => ({
  timestamp: Date.now(),
  performance: {
    cpu: Math.random() * 100,
    memory: Math.random() * 100,
    networkLatency: Math.random() * 100 + 10,
    throughput: Math.random() * 1000 + 100
  },
  knowledgeGraph: {
    nodes: Array.from({ length: 25 }, (_, i) => ({
      id: `node_${i}`,
      type: ['concept', 'entity', 'relation'][Math.floor(Math.random() * 3)],
      weight: Math.random(),
      metadata: { name: `Node ${i}` },
      embedding: Array.from({ length: 64 }, () => Math.random())
    })),
    edges: Array.from({ length: 50 }, (_, i) => ({
      source: `node_${Math.floor(Math.random() * 25)}`,
      target: `node_${Math.floor(Math.random() * 25)}`,
      weight: Math.random(),
      type: Math.random() > 0.9 ? 'inhibitory' : 'normal'
    })),
    metrics: {
      density: Math.random(),
      clustering: Math.random(),
      avgPathLength: Math.random() * 5 + 1
    }
  },
  memory: {
    workingMemory: {
      usage: Math.random() * 100
    },
    longTermMemory: {
      consolidationRate: Math.random(),
      retrievalSpeed: Math.random() * 100 + 10
    }
  },
  metrics: {
    brain_clustering_coefficient: Math.random(),
    brain_learning_efficiency: Math.random(),
    brain_concept_coherence: Math.random(),
    brain_active_entities: Math.floor(Math.random() * 50) + 10
  }
});

// Mock setup and teardown helpers
export const setupMockWebSocket = () => {
  (global as any).WebSocket = MockWebSocket;
};

export const teardownMockWebSocket = () => {
  MockWebSocket.reset();
  delete (global as any).WebSocket;
};

// Utility functions for testing
export const waitForNextTick = () => new Promise(resolve => setTimeout(resolve, 0));

export const createMockFunction = (): any => {
  const fn = (...args: any[]) => fn.mock.results[fn.mock.calls.length - 1]?.value;
  fn.mock = {
    calls: [],
    results: []
  };
  return fn;
};