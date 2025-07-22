/**
 * LLMKG Operation Instrumentation System
 * 
 * Provides ultra-low overhead instrumentation for LLMKG-specific operations
 * including SDR processing, cognitive patterns, neural activity, memory systems,
 * attention mechanisms, and graph queries.
 */

import { telemetryRecorder, TelemetryEvent } from './recorder.js';
import { telemetryConfig } from './config.js';

export interface InstrumentationPoint {
  /** Unique identifier for this instrumentation point */
  id: string;
  
  /** Human-readable name */
  name: string;
  
  /** Category of operation */
  category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph';
  
  /** Operation type */
  type: 'query' | 'update' | 'process' | 'analyze' | 'store' | 'retrieve';
  
  /** Sampling rate (0-1) */
  samplingRate: number;
  
  /** Performance thresholds */
  thresholds: {
    warning: number;  // milliseconds
    error: number;    // milliseconds
  };
  
  /** Custom metadata */
  metadata?: Record<string, any>;
}

export interface OperationMetrics {
  /** Total number of calls */
  totalCalls: number;
  
  /** Total execution time */
  totalDuration: number;
  
  /** Average execution time */
  averageDuration: number;
  
  /** Minimum execution time */
  minDuration: number;
  
  /** Maximum execution time */
  maxDuration: number;
  
  /** Number of errors */
  errorCount: number;
  
  /** Success rate */
  successRate: number;
  
  /** Memory usage statistics */
  memory: {
    totalAllocated: number;
    averageAllocated: number;
    peakAllocated: number;
  };
  
  /** Last updated timestamp */
  lastUpdated: number;
}

export interface PerformanceProfile {
  /** Operation identifier */
  operation: string;
  
  /** Performance percentiles */
  percentiles: {
    p50: number;
    p95: number;
    p99: number;
  };
  
  /** Throughput metrics */
  throughput: {
    operationsPerSecond: number;
    peakOperationsPerSecond: number;
  };
  
  /** Resource utilization */
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    ioOperations: number;
  };
}

/**
 * Ultra-low overhead operation instrumentation
 */
export class LLMKGInstrumentation {
  private instrumentationPoints: Map<string, InstrumentationPoint> = new Map();
  private operationMetrics: Map<string, OperationMetrics> = new Map();
  private performanceBuffer: Map<string, number[]> = new Map();
  private instrumentationOverhead = 0;
  private isEnabled = true;

  constructor() {
    this.setupDefaultInstrumentationPoints();
    
    // Monitor our own overhead
    this.monitorInstrumentationOverhead();
  }

  /**
   * Register an instrumentation point
   */
  registerInstrumentationPoint(point: InstrumentationPoint): void {
    this.instrumentationPoints.set(point.id, point);
    
    // Initialize metrics
    this.operationMetrics.set(point.id, {
      totalCalls: 0,
      totalDuration: 0,
      averageDuration: 0,
      minDuration: Infinity,
      maxDuration: 0,
      errorCount: 0,
      successRate: 1.0,
      memory: {
        totalAllocated: 0,
        averageAllocated: 0,
        peakAllocated: 0,
      },
      lastUpdated: Date.now(),
    });
    
    this.performanceBuffer.set(point.id, []);
  }

  /**
   * Instrument an SDR operation
   */
  instrumentSDROperation<T>(
    operationName: string,
    operation: () => T,
    metadata?: Record<string, any>
  ): T {
    if (!this.shouldInstrument('sdr', operationName)) {
      return operation();
    }

    const startTime = performance.now();
    const startMemory = process.memoryUsage().heapUsed;
    let result: T;
    let error: Error | undefined;

    try {
      result = operation();
      return result;
    } catch (err) {
      error = err as Error;
      throw error;
    } finally {
      const endTime = performance.now();
      const endMemory = process.memoryUsage().heapUsed;
      const duration = endTime - startTime;
      const memoryDelta = endMemory - startMemory;

      this.recordOperation(
        'sdr',
        operationName,
        duration,
        memoryDelta,
        !error,
        { ...metadata, error: error?.message }
      );
    }
  }

  /**
   * Instrument a cognitive pattern operation
   */
  instrumentCognitiveOperation<T>(
    operationName: string,
    operation: () => T,
    patternData?: {
      patternType: string;
      complexity: number;
      activationLevel: number;
    }
  ): T {
    if (!this.shouldInstrument('cognitive', operationName)) {
      return operation();
    }

    return this.withInstrumentation('cognitive', operationName, operation, patternData);
  }

  /**
   * Instrument a neural processing operation
   */
  instrumentNeuralOperation<T>(
    operationName: string,
    operation: () => T,
    neuralData?: {
      nodeCount: number;
      connectionCount: number;
      activationFunction: string;
    }
  ): T {
    if (!this.shouldInstrument('neural', operationName)) {
      return operation();
    }

    return this.withInstrumentation('neural', operationName, operation, neuralData);
  }

  /**
   * Instrument a memory system operation
   */
  instrumentMemoryOperation<T>(
    operationName: string,
    operation: () => T,
    memoryData?: {
      operationType: 'read' | 'write' | 'delete' | 'search';
      dataSize: number;
      cacheHit?: boolean;
    }
  ): T {
    if (!this.shouldInstrument('memory', operationName)) {
      return operation();
    }

    return this.withInstrumentation('memory', operationName, operation, memoryData);
  }

  /**
   * Instrument an attention mechanism operation
   */
  instrumentAttentionOperation<T>(
    operationName: string,
    operation: () => T,
    attentionData?: {
      focusTargets: number;
      attentionWeights: number[];
      contextSize: number;
    }
  ): T {
    if (!this.shouldInstrument('attention', operationName)) {
      return operation();
    }

    return this.withInstrumentation('attention', operationName, operation, attentionData);
  }

  /**
   * Instrument a graph query operation
   */
  instrumentGraphOperation<T>(
    operationName: string,
    operation: () => T,
    graphData?: {
      queryType: string;
      nodeCount: number;
      edgeCount: number;
      traversalDepth: number;
    }
  ): T {
    if (!this.shouldInstrument('graph', operationName)) {
      return operation();
    }

    return this.withInstrumentation('graph', operationName, operation, graphData);
  }

  /**
   * Instrument an async operation
   */
  async instrumentAsync<T>(
    category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph',
    operationName: string,
    operation: () => Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> {
    if (!this.shouldInstrument(category, operationName)) {
      return operation();
    }

    const startTime = performance.now();
    const startMemory = process.memoryUsage().heapUsed;
    let result: T;
    let error: Error | undefined;

    try {
      result = await operation();
      return result;
    } catch (err) {
      error = err as Error;
      throw error;
    } finally {
      const endTime = performance.now();
      const endMemory = process.memoryUsage().heapUsed;
      const duration = endTime - startTime;
      const memoryDelta = endMemory - startMemory;

      this.recordOperation(
        category,
        operationName,
        duration,
        memoryDelta,
        !error,
        { ...metadata, error: error?.message, async: true }
      );
    }
  }

  /**
   * Get operation metrics
   */
  getOperationMetrics(operationId?: string): Map<string, OperationMetrics> | OperationMetrics | undefined {
    if (operationId) {
      return this.operationMetrics.get(operationId);
    }
    return this.operationMetrics;
  }

  /**
   * Get performance profile for an operation
   */
  getPerformanceProfile(operationId: string): PerformanceProfile | undefined {
    const metrics = this.operationMetrics.get(operationId);
    const durations = this.performanceBuffer.get(operationId);
    
    if (!metrics || !durations || durations.length === 0) {
      return undefined;
    }

    const sortedDurations = [...durations].sort((a, b) => a - b);
    const length = sortedDurations.length;

    return {
      operation: operationId,
      percentiles: {
        p50: sortedDurations[Math.floor(length * 0.5)],
        p95: sortedDurations[Math.floor(length * 0.95)],
        p99: sortedDurations[Math.floor(length * 0.99)],
      },
      throughput: {
        operationsPerSecond: metrics.totalCalls / ((Date.now() - (metrics.lastUpdated - metrics.totalDuration)) / 1000),
        peakOperationsPerSecond: 1 / (metrics.minDuration / 1000),
      },
      resources: {
        cpuUsage: metrics.averageDuration, // Approximation
        memoryUsage: metrics.memory.averageAllocated,
        ioOperations: 0, // Would need additional instrumentation
      },
    };
  }

  /**
   * Get instrumentation overhead
   */
  getInstrumentationOverhead(): number {
    return this.instrumentationOverhead;
  }

  /**
   * Enable/disable instrumentation
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
  }

  /**
   * Clear all metrics and reset
   */
  reset(): void {
    this.operationMetrics.clear();
    this.performanceBuffer.clear();
    
    // Reinitialize with current instrumentation points
    for (const point of this.instrumentationPoints.values()) {
      this.operationMetrics.set(point.id, {
        totalCalls: 0,
        totalDuration: 0,
        averageDuration: 0,
        minDuration: Infinity,
        maxDuration: 0,
        errorCount: 0,
        successRate: 1.0,
        memory: {
          totalAllocated: 0,
          averageAllocated: 0,
          peakAllocated: 0,
        },
        lastUpdated: Date.now(),
      });
      
      this.performanceBuffer.set(point.id, []);
    }
  }

  /**
   * Export metrics as telemetry events
   */
  exportMetrics(): TelemetryEvent[] {
    const events: TelemetryEvent[] = [];

    for (const [operationId, metrics] of this.operationMetrics) {
      const point = this.instrumentationPoints.get(operationId);
      if (!point) continue;

      // Main performance metrics
      events.push({
        id: `metrics_${operationId}_${Date.now()}`,
        timestamp: Date.now(),
        type: 'metric',
        category: point.category,
        name: `${point.category}.${operationId}.performance`,
        data: {
          totalCalls: metrics.totalCalls,
          averageDuration: metrics.averageDuration,
          successRate: metrics.successRate,
          errorCount: metrics.errorCount,
        },
        tags: {
          operation: operationId,
          category: point.category,
          type: point.type,
        },
      });

      // Memory metrics
      events.push({
        id: `memory_${operationId}_${Date.now()}`,
        timestamp: Date.now(),
        type: 'metric',
        category: point.category,
        name: `${point.category}.${operationId}.memory`,
        data: {
          averageAllocated: metrics.memory.averageAllocated,
          peakAllocated: metrics.memory.peakAllocated,
          totalAllocated: metrics.memory.totalAllocated,
        },
        tags: {
          operation: operationId,
          category: point.category,
        },
      });
    }

    return events;
  }

  /**
   * Setup default instrumentation points for LLMKG operations
   */
  private setupDefaultInstrumentationPoints(): void {
    const defaultPoints: InstrumentationPoint[] = [
      // SDR Operations
      {
        id: 'sdr_encode',
        name: 'SDR Encoding',
        category: 'sdr',
        type: 'process',
        samplingRate: 1.0,
        thresholds: { warning: 10, error: 50 },
      },
      {
        id: 'sdr_decode',
        name: 'SDR Decoding',
        category: 'sdr',
        type: 'process',
        samplingRate: 1.0,
        thresholds: { warning: 5, error: 25 },
      },
      {
        id: 'sdr_union',
        name: 'SDR Union Operation',
        category: 'sdr',
        type: 'process',
        samplingRate: 0.5,
        thresholds: { warning: 1, error: 10 },
      },
      {
        id: 'sdr_intersection',
        name: 'SDR Intersection Operation',
        category: 'sdr',
        type: 'process',
        samplingRate: 0.5,
        thresholds: { warning: 1, error: 10 },
      },

      // Cognitive Operations
      {
        id: 'cognitive_pattern_recognition',
        name: 'Cognitive Pattern Recognition',
        category: 'cognitive',
        type: 'analyze',
        samplingRate: 1.0,
        thresholds: { warning: 20, error: 100 },
      },
      {
        id: 'cognitive_inhibition',
        name: 'Cognitive Inhibition Processing',
        category: 'cognitive',
        type: 'process',
        samplingRate: 0.8,
        thresholds: { warning: 15, error: 75 },
      },

      // Neural Operations
      {
        id: 'neural_activation',
        name: 'Neural Activation Processing',
        category: 'neural',
        type: 'process',
        samplingRate: 0.3,
        thresholds: { warning: 5, error: 30 },
      },
      {
        id: 'neural_propagation',
        name: 'Neural Signal Propagation',
        category: 'neural',
        type: 'process',
        samplingRate: 0.1,
        thresholds: { warning: 1, error: 10 },
      },

      // Memory Operations
      {
        id: 'memory_store',
        name: 'Memory Storage Operation',
        category: 'memory',
        type: 'store',
        samplingRate: 1.0,
        thresholds: { warning: 10, error: 50 },
      },
      {
        id: 'memory_retrieve',
        name: 'Memory Retrieval Operation',
        category: 'memory',
        type: 'retrieve',
        samplingRate: 1.0,
        thresholds: { warning: 5, error: 25 },
      },
      {
        id: 'zero_copy_operation',
        name: 'Zero Copy Memory Operation',
        category: 'memory',
        type: 'process',
        samplingRate: 0.5,
        thresholds: { warning: 1, error: 5 },
      },

      // Attention Operations
      {
        id: 'attention_focus',
        name: 'Attention Focus Calculation',
        category: 'attention',
        type: 'analyze',
        samplingRate: 0.8,
        thresholds: { warning: 8, error: 40 },
      },
      {
        id: 'attention_shift',
        name: 'Attention Shifting',
        category: 'attention',
        type: 'update',
        samplingRate: 0.6,
        thresholds: { warning: 3, error: 15 },
      },

      // Graph Operations
      {
        id: 'graph_traverse',
        name: 'Graph Traversal',
        category: 'graph',
        type: 'query',
        samplingRate: 1.0,
        thresholds: { warning: 25, error: 100 },
      },
      {
        id: 'graph_update',
        name: 'Graph Update',
        category: 'graph',
        type: 'update',
        samplingRate: 1.0,
        thresholds: { warning: 15, error: 75 },
      },
    ];

    defaultPoints.forEach(point => this.registerInstrumentationPoint(point));
  }

  /**
   * Check if an operation should be instrumented
   */
  private shouldInstrument(category: string, operationName: string): boolean {
    if (!this.isEnabled || !telemetryConfig.isInstrumentationEnabled(category as any)) {
      return false;
    }

    const point = this.instrumentationPoints.get(operationName) || 
                  Array.from(this.instrumentationPoints.values())
                    .find(p => p.category === category && p.name.includes(operationName));

    if (!point) {
      return false;
    }

    return Math.random() < point.samplingRate;
  }

  /**
   * Generic instrumentation wrapper
   */
  private withInstrumentation<T>(
    category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph',
    operationName: string,
    operation: () => T,
    metadata?: Record<string, any>
  ): T {
    const overheadStart = performance.now();
    const startTime = performance.now();
    const startMemory = process.memoryUsage().heapUsed;
    let result: T;
    let error: Error | undefined;

    try {
      const overheadEnd = performance.now();
      this.instrumentationOverhead += overheadEnd - overheadStart;
      
      result = operation();
      return result;
    } catch (err) {
      error = err as Error;
      throw error;
    } finally {
      const endTime = performance.now();
      const endMemory = process.memoryUsage().heapUsed;
      const duration = endTime - startTime;
      const memoryDelta = endMemory - startMemory;

      this.recordOperation(
        category,
        operationName,
        duration,
        memoryDelta,
        !error,
        { ...metadata, error: error?.message }
      );
    }
  }

  /**
   * Record operation metrics
   */
  private recordOperation(
    category: string,
    operationName: string,
    duration: number,
    memoryDelta: number,
    success: boolean,
    metadata?: Record<string, any>
  ): void {
    const point = this.instrumentationPoints.get(operationName);
    const pointId = point?.id || `${category}_${operationName}`;
    
    let metrics = this.operationMetrics.get(pointId);
    if (!metrics) {
      // Create metrics on-the-fly for dynamic operations
      metrics = {
        totalCalls: 0,
        totalDuration: 0,
        averageDuration: 0,
        minDuration: Infinity,
        maxDuration: 0,
        errorCount: 0,
        successRate: 1.0,
        memory: {
          totalAllocated: 0,
          averageAllocated: 0,
          peakAllocated: 0,
        },
        lastUpdated: Date.now(),
      };
      this.operationMetrics.set(pointId, metrics);
      this.performanceBuffer.set(pointId, []);
    }

    // Update metrics
    metrics.totalCalls++;
    metrics.totalDuration += duration;
    metrics.averageDuration = metrics.totalDuration / metrics.totalCalls;
    metrics.minDuration = Math.min(metrics.minDuration, duration);
    metrics.maxDuration = Math.max(metrics.maxDuration, duration);
    metrics.lastUpdated = Date.now();

    if (!success) {
      metrics.errorCount++;
    }
    metrics.successRate = (metrics.totalCalls - metrics.errorCount) / metrics.totalCalls;

    // Update memory metrics
    if (memoryDelta > 0) {
      metrics.memory.totalAllocated += memoryDelta;
      metrics.memory.averageAllocated = metrics.memory.totalAllocated / metrics.totalCalls;
      metrics.memory.peakAllocated = Math.max(metrics.memory.peakAllocated, memoryDelta);
    }

    // Update performance buffer (keep last 1000 samples)
    const buffer = this.performanceBuffer.get(pointId)!;
    buffer.push(duration);
    if (buffer.length > 1000) {
      buffer.shift();
    }

    // Check thresholds and emit warnings
    if (point) {
      if (duration > point.thresholds.error) {
        telemetryRecorder.recordLog(
          point.category,
          `instrumentation.threshold.error.${pointId}`,
          `Operation ${operationName} exceeded error threshold: ${duration}ms > ${point.thresholds.error}ms`,
          'error',
          { duration, threshold: point.thresholds.error, ...metadata }
        );
      } else if (duration > point.thresholds.warning) {
        telemetryRecorder.recordLog(
          point.category,
          `instrumentation.threshold.warning.${pointId}`,
          `Operation ${operationName} exceeded warning threshold: ${duration}ms > ${point.thresholds.warning}ms`,
          'warn',
          { duration, threshold: point.thresholds.warning, ...metadata }
        );
      }
    }

    // Record to telemetry system
    telemetryRecorder.recordTrace(
      category as any,
      `instrumentation.${category}.${operationName}`,
      {
        operationName,
        success,
        ...metadata,
      },
      {
        duration,
        memoryUsage: memoryDelta,
      },
      {
        category,
        operation: operationName,
        success: success.toString(),
      }
    );
  }

  /**
   * Monitor instrumentation overhead
   */
  private monitorInstrumentationOverhead(): void {
    setInterval(() => {
      if (this.instrumentationOverhead > 0) {
        telemetryRecorder.recordMetric(
          'system',
          'instrumentation.overhead',
          this.instrumentationOverhead,
          {
            period: '1min',
          }
        );
        
        // Reset overhead counter
        this.instrumentationOverhead = 0;
      }
    }, 60000); // Every minute
  }
}

// Global instrumentation instance
export const llmkgInstrumentation = new LLMKGInstrumentation();