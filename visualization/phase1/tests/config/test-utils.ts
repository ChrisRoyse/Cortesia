/**
 * Test utilities and helpers for LLMKG Visualization Phase 1
 * 
 * Common testing utilities, data generators, and helper functions
 */

import { MCPClient } from '../../src/mcp/client';
import { BaseCollector, CollectedData, CollectorConfig } from '../../src/collectors/base';
import { WebSocketServer, ClientConnection } from '../../src/websocket/server';
import { MessageType, WebSocketMessage } from '../../src/websocket/protocol';

/**
 * Performance measurement utilities
 */
export class PerformanceTracker {
  private measurements: Map<string, number[]> = new Map();

  start(operation: string): void {
    const key = `${operation}_start`;
    this.measurements.set(key, [performance.now()]);
  }

  end(operation: string): number {
    const startKey = `${operation}_start`;
    const endTime = performance.now();
    
    const startTimes = this.measurements.get(startKey);
    if (!startTimes || startTimes.length === 0) {
      throw new Error(`No start time recorded for operation: ${operation}`);
    }

    const duration = endTime - startTimes[0];
    
    const durationsKey = `${operation}_durations`;
    const existing = this.measurements.get(durationsKey) || [];
    existing.push(duration);
    this.measurements.set(durationsKey, existing);

    return duration;
  }

  getDurations(operation: string): number[] {
    return this.measurements.get(`${operation}_durations`) || [];
  }

  getAverageDuration(operation: string): number {
    const durations = this.getDurations(operation);
    return durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;
  }

  getStats(operation: string) {
    const durations = this.getDurations(operation);
    if (durations.length === 0) {
      return {
        count: 0,
        average: 0,
        min: 0,
        max: 0,
        p95: 0,
        p99: 0
      };
    }

    const sorted = [...durations].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);
    const p99Index = Math.floor(sorted.length * 0.99);

    return {
      count: durations.length,
      average: durations.reduce((a, b) => a + b, 0) / durations.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p95: sorted[p95Index],
      p99: sorted[p99Index]
    };
  }

  clear(): void {
    this.measurements.clear();
  }
}

/**
 * LLMKG-specific test data generators
 */
export class LLMKGDataGenerator {
  /**
   * Generate mock cognitive pattern data
   */
  static generateCognitivePattern(overrides: Partial<any> = {}) {
    return {
      timestamp: Date.now(),
      type: 'cognitive_pattern',
      pattern_id: `pattern_${Math.random().toString(36).substr(2, 9)}`,
      cortical_region: Math.random() > 0.5 ? 'prefrontal' : 'temporal',
      activation_level: Math.random(),
      attention_weight: Math.random(),
      pattern_strength: Math.random() * 100,
      connections: Array.from({ length: Math.floor(Math.random() * 10) + 1 }, () => ({
        target_pattern: `pattern_${Math.random().toString(36).substr(2, 9)}`,
        weight: Math.random(),
        type: Math.random() > 0.5 ? 'excitatory' : 'inhibitory'
      })),
      metadata: {
        processing_time: Math.random() * 50,
        confidence: Math.random(),
        source: 'test_generator'
      },
      ...overrides
    };
  }

  /**
   * Generate mock SDR (Sparse Distributed Representation) data
   */
  static generateSDRData(overrides: Partial<any> = {}) {
    const size = 2048; // Standard SDR size
    const sparsity = 0.02; // 2% sparsity
    const activeCount = Math.floor(size * sparsity);
    
    const activeBits = new Set<number>();
    while (activeBits.size < activeCount) {
      activeBits.add(Math.floor(Math.random() * size));
    }

    return {
      timestamp: Date.now(),
      type: 'sdr_data',
      sdr_id: `sdr_${Math.random().toString(36).substr(2, 9)}`,
      size,
      sparsity,
      active_bits: Array.from(activeBits),
      semantic_meaning: `concept_${Math.random().toString(36).substr(2, 9)}`,
      encoding_time: Math.random() * 10,
      overlap_score: Math.random(),
      stability: Math.random(),
      ...overrides
    };
  }

  /**
   * Generate mock neural activity data
   */
  static generateNeuralActivity(overrides: Partial<any> = {}) {
    return {
      timestamp: Date.now(),
      type: 'neural_activity',
      neuron_id: `neuron_${Math.random().toString(36).substr(2, 9)}`,
      layer_id: `layer_${Math.floor(Math.random() * 10)}`,
      activation_value: Math.random(),
      firing_rate: Math.random() * 100,
      membrane_potential: -70 + Math.random() * 50,
      synaptic_inputs: Array.from({ length: Math.floor(Math.random() * 20) + 1 }, () => ({
        source_neuron: `neuron_${Math.random().toString(36).substr(2, 9)}`,
        weight: Math.random() * 2 - 1, // -1 to 1
        delay: Math.random() * 10
      })),
      spike_train: Array.from({ length: Math.floor(Math.random() * 100) }, () => Math.random() * 1000),
      ...overrides
    };
  }

  /**
   * Generate mock memory system data
   */
  static generateMemoryData(overrides: Partial<any> = {}) {
    return {
      timestamp: Date.now(),
      type: 'memory_data',
      memory_id: `mem_${Math.random().toString(36).substr(2, 9)}`,
      memory_type: Math.random() > 0.5 ? 'episodic' : 'semantic',
      consolidation_level: Math.random(),
      retrieval_strength: Math.random(),
      decay_rate: Math.random() * 0.1,
      associations: Array.from({ length: Math.floor(Math.random() * 5) + 1 }, () => ({
        target_memory: `mem_${Math.random().toString(36).substr(2, 9)}`,
        association_strength: Math.random(),
        association_type: 'semantic'
      })),
      encoding_context: {
        spatial_context: `location_${Math.random().toString(36).substr(2, 9)}`,
        temporal_context: Date.now() - Math.random() * 86400000,
        emotional_valence: Math.random() * 2 - 1
      },
      ...overrides
    };
  }

  /**
   * Generate mock attention mechanism data
   */
  static generateAttentionData(overrides: Partial<any> = {}) {
    return {
      timestamp: Date.now(),
      type: 'attention_data',
      focus_id: `focus_${Math.random().toString(36).substr(2, 9)}`,
      attention_type: Math.random() > 0.5 ? 'spatial' : 'feature',
      focus_strength: Math.random(),
      focus_duration: Math.random() * 1000,
      distractor_count: Math.floor(Math.random() * 10),
      salience_map: Array.from({ length: 100 }, () => Math.random()),
      top_down_signals: Array.from({ length: 5 }, () => ({
        source: `region_${Math.random().toString(36).substr(2, 9)}`,
        strength: Math.random(),
        priority: Math.random()
      })),
      bottom_up_signals: Array.from({ length: 10 }, () => ({
        stimulus_id: `stimulus_${Math.random().toString(36).substr(2, 9)}`,
        intensity: Math.random(),
        novelty: Math.random()
      })),
      ...overrides
    };
  }

  /**
   * Generate mock knowledge graph data
   */
  static generateKnowledgeGraphData(overrides: Partial<any> = {}) {
    return {
      timestamp: Date.now(),
      type: 'knowledge_graph_data',
      graph_id: `graph_${Math.random().toString(36).substr(2, 9)}`,
      nodes: Array.from({ length: Math.floor(Math.random() * 20) + 5 }, () => ({
        id: `node_${Math.random().toString(36).substr(2, 9)}`,
        type: Math.random() > 0.5 ? 'concept' : 'relation',
        properties: {
          label: `label_${Math.random().toString(36).substr(2, 9)}`,
          weight: Math.random(),
          centrality: Math.random(),
          cluster: Math.floor(Math.random() * 5)
        }
      })),
      edges: Array.from({ length: Math.floor(Math.random() * 50) + 10 }, () => ({
        source: `node_${Math.random().toString(36).substr(2, 9)}`,
        target: `node_${Math.random().toString(36).substr(2, 9)}`,
        type: 'relationship',
        weight: Math.random(),
        directed: Math.random() > 0.5
      })),
      metrics: {
        node_count: 0,
        edge_count: 0,
        density: Math.random(),
        clustering_coefficient: Math.random(),
        average_path_length: Math.random() * 10
      },
      ...overrides
    };
  }

  /**
   * Generate batch of mixed LLMKG data
   */
  static generateMixedLLMKGBatch(count: number) {
    const generators = [
      this.generateCognitivePattern,
      this.generateSDRData,
      this.generateNeuralActivity,
      this.generateMemoryData,
      this.generateAttentionData,
      this.generateKnowledgeGraphData
    ];

    return Array.from({ length: count }, () => {
      const generator = generators[Math.floor(Math.random() * generators.length)];
      return generator();
    });
  }
}

/**
 * Mock MCP Client for testing
 */
export class MockMCPClient extends MCPClient {
  private mockDataGenerator: () => any;

  constructor(config: any = {}) {
    super(config);
    this.mockDataGenerator = () => LLMKGDataGenerator.generateMixedLLMKGBatch(1)[0];
  }

  setMockDataGenerator(generator: () => any): void {
    this.mockDataGenerator = generator;
  }

  connect(): void {
    super.connect();
    this.emit('connected', { timestamp: Date.now() });
  }

  disconnect(): void {
    super.disconnect();
    this.emit('disconnected', { timestamp: Date.now() });
  }

  // Override the mock data generation with more realistic data
  private generateMockMCPData(): any {
    return this.mockDataGenerator();
  }
}

/**
 * Mock WebSocket for testing
 */
export class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  protocol: string;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  private messageQueue: any[] = [];

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocol = Array.isArray(protocols) ? protocols[0] : protocols || '';
    
    setTimeout(() => this.simulateOpen(), 10);
  }

  private simulateOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }

  send(data: string | ArrayBuffer | Blob | ArrayBufferView): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    
    this.messageQueue.push(data);
    
    // Simulate message echo for testing
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', { data }));
      }
    }, 1);
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSING;
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose(new CloseEvent('close', { code, reason }));
      }
    }, 10);
  }

  getMessageQueue(): any[] {
    return [...this.messageQueue];
  }

  clearMessageQueue(): void {
    this.messageQueue = [];
  }
}

/**
 * Test collector for unit testing
 */
export class TestCollector extends BaseCollector {
  private collectData: (() => Promise<CollectedData[]>) | null = null;

  async initialize(): Promise<void> {
    // Mock initialization
  }

  async cleanup(): Promise<void> {
    // Mock cleanup
  }

  async collect(): Promise<CollectedData[]> {
    if (this.collectData) {
      return this.collectData();
    }

    return [{
      id: this.generateId(),
      timestamp: Date.now(),
      source: 'test',
      type: 'test_data',
      data: LLMKGDataGenerator.generateCognitivePattern(),
      metadata: this.createMetadata('test_collect', 5)
    }];
  }

  setCollectFunction(fn: () => Promise<CollectedData[]>): void {
    this.collectData = fn;
  }
}

/**
 * Load testing utilities
 */
export class LoadTestRunner {
  private running = false;
  private metrics = new PerformanceTracker();

  async runLoadTest(options: {
    duration: number;
    concurrentRequests: number;
    requestGenerator: () => Promise<any>;
    name: string;
  }): Promise<{
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    averageLatency: number;
    throughput: number;
    stats: ReturnType<PerformanceTracker['getStats']>;
  }> {
    const { duration, concurrentRequests, requestGenerator, name } = options;
    
    this.running = true;
    this.metrics.clear();

    const results = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0
    };

    const startTime = Date.now();
    const endTime = startTime + duration;

    const workers: Promise<void>[] = [];

    // Start concurrent workers
    for (let i = 0; i < concurrentRequests; i++) {
      const worker = async () => {
        while (this.running && Date.now() < endTime) {
          try {
            this.metrics.start(`${name}_request`);
            await requestGenerator();
            this.metrics.end(`${name}_request`);
            results.successfulRequests++;
          } catch (error) {
            results.failedRequests++;
          }
          results.totalRequests++;
        }
      };

      workers.push(worker());
    }

    // Wait for test duration
    await new Promise(resolve => setTimeout(resolve, duration));
    this.running = false;

    // Wait for all workers to finish
    await Promise.all(workers);

    const actualDuration = Date.now() - startTime;
    const stats = this.metrics.getStats(`${name}_request`);

    return {
      ...results,
      averageLatency: stats.average,
      throughput: (results.successfulRequests / actualDuration) * 1000, // requests per second
      stats
    };
  }

  stop(): void {
    this.running = false;
  }
}

/**
 * Test helpers for common assertions
 */
export const TestHelpers = {
  /**
   * Wait for a condition to be true within a timeout
   */
  waitFor: async (
    condition: () => boolean | Promise<boolean>,
    timeout: number = 5000,
    interval: number = 100
  ): Promise<void> => {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const result = await condition();
      if (result) {
        return;
      }
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    throw new Error(`Condition not met within ${timeout}ms timeout`);
  },

  /**
   * Create a deferred promise
   */
  createDeferred: <T>() => {
    let resolve: (value: T) => void;
    let reject: (error: Error) => void;
    
    const promise = new Promise<T>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    
    return { promise, resolve: resolve!, reject: reject! };
  },

  /**
   * Measure execution time
   */
  measureTime: async <T>(fn: () => Promise<T>): Promise<{ result: T; duration: number }> => {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;
    return { result, duration };
  },

  /**
   * Generate test data with specific properties
   */
  generateTestData: (type: string, count: number = 1) => {
    const generators: Record<string, () => any> = {
      'cognitive': LLMKGDataGenerator.generateCognitivePattern,
      'sdr': LLMKGDataGenerator.generateSDRData,
      'neural': LLMKGDataGenerator.generateNeuralActivity,
      'memory': LLMKGDataGenerator.generateMemoryData,
      'attention': LLMKGDataGenerator.generateAttentionData,
      'graph': LLMKGDataGenerator.generateKnowledgeGraphData
    };

    const generator = generators[type];
    if (!generator) {
      throw new Error(`Unknown test data type: ${type}`);
    }

    return Array.from({ length: count }, generator);
  }
};

export { PerformanceTracker, LoadTestRunner };