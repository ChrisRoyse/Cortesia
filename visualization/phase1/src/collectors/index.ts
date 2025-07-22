/**
 * @fileoverview Data Collectors Module for LLMKG Visualization
 * 
 * This module provides a comprehensive data collection system for LLMKG visualization
 * dashboard. It includes specialized collectors for different LLMKG components and
 * a centralized manager for orchestrating all collection activities.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

// Export base collector
export { 
  BaseCollector, 
  CircularBuffer, 
  DataAggregator,
  type CollectorConfig,
  type CollectedData,
  type CollectionMetadata,
  type CollectorStats
} from './base.js';

// Export knowledge graph collector
export { 
  KnowledgeGraphCollector,
  type KnowledgeGraphCollectorConfig,
  type EntityMetrics,
  type TopologyData,
  type TripleMetrics,
  type QueryMetrics
} from './knowledge-graph.js';

// Export cognitive patterns collector
export { 
  CognitivePatternCollector,
  type CognitivePatternCollectorConfig,
  type AttentionMetrics,
  type ReasoningPatterns,
  type DecisionMetrics,
  type CognitiveLoadMetrics,
  type MetacognitiveMetrics
} from './cognitive-patterns.js';

// Export neural activity collector
export { 
  NeuralActivityCollector,
  type NeuralActivityCollectorConfig,
  type SDRMetrics,
  type NeuralActivationMetrics,
  type BrainProcessingMetrics,
  type SynapticMetrics
} from './neural-activity.js';

// Export memory systems collector
export { 
  MemorySystemsCollector,
  type MemorySystemsCollectorConfig,
  type WorkingMemoryMetrics,
  type LongTermMemoryMetrics,
  type EpisodicMemoryMetrics,
  type SemanticMemoryMetrics,
  type ConsolidationMetrics,
  type ZeroCopyMetrics
} from './memory-systems.js';

// Export collector manager
export { 
  CollectorManager,
  type CollectorManagerConfig,
  type CollectorConfigs,
  type ManagerStats,
  type HealthCheckResults,
  type AggregatedData,
  type LoadBalancingStrategy
} from './manager.js';

/**
 * Default collector configurations
 */
export const DEFAULT_COLLECTOR_CONFIGS = {
  knowledgeGraph: {
    name: 'knowledge-graph-collector',
    collectionInterval: 100,
    monitorEntities: true,
    monitorTopology: true,
    monitorTriples: true,
    monitorQueries: true
  },
  
  cognitivePatterns: {
    name: 'cognitive-pattern-collector',
    collectionInterval: 50,
    monitorAttention: true,
    monitorReasoning: true,
    monitorDecisions: true,
    monitorCognitiveLoad: true,
    monitorMetacognition: true
  },
  
  neuralActivity: {
    name: 'neural-activity-collector',
    collectionInterval: 25,
    monitorSDR: true,
    monitorActivations: true,
    monitorBrainProcessing: true,
    monitorSynaptic: true,
    realTimeAnalysis: true
  },
  
  memorySystems: {
    name: 'memory-systems-collector',
    collectionInterval: 100,
    monitorWorkingMemory: true,
    monitorLongTermMemory: true,
    monitorEpisodicMemory: true,
    monitorSemanticMemory: true,
    monitorConsolidation: true,
    monitorZeroCopy: true
  }
} as const;

/**
 * High-performance collector preset for >1000 events/sec
 */
export const HIGH_PERFORMANCE_CONFIGS = {
  knowledgeGraph: {
    ...DEFAULT_COLLECTOR_CONFIGS.knowledgeGraph,
    collectionInterval: 50, // 20 Hz
    bufferSize: 20000,
    sampleRate: 0.8 // Sample 80% of events
  },
  
  cognitivePatterns: {
    ...DEFAULT_COLLECTOR_CONFIGS.cognitivePatterns,
    collectionInterval: 25, // 40 Hz
    attentionSamplingRate: 100, // 100 Hz
    bufferSize: 15000
  },
  
  neuralActivity: {
    ...DEFAULT_COLLECTOR_CONFIGS.neuralActivity,
    collectionInterval: 20, // 50 Hz
    neuralSamplingRate: 200, // 200 Hz
    bufferSize: 25000
  },
  
  memorySystems: {
    ...DEFAULT_COLLECTOR_CONFIGS.memorySystems,
    collectionInterval: 50, // 20 Hz
    memorySamplingRate: 20, // 20 Hz
    bufferSize: 12000
  }
} as const;

/**
 * Low-latency collector preset for real-time monitoring
 */
export const LOW_LATENCY_CONFIGS = {
  knowledgeGraph: {
    ...DEFAULT_COLLECTOR_CONFIGS.knowledgeGraph,
    collectionInterval: 20, // 50 Hz
    autoFlush: true,
    flushInterval: 1000 // 1 second
  },
  
  cognitivePatterns: {
    ...DEFAULT_COLLECTOR_CONFIGS.cognitivePatterns,
    collectionInterval: 10, // 100 Hz
    attentionSamplingRate: 200, // 200 Hz
    autoFlush: true,
    flushInterval: 500 // 0.5 seconds
  },
  
  neuralActivity: {
    ...DEFAULT_COLLECTOR_CONFIGS.neuralActivity,
    collectionInterval: 10, // 100 Hz
    neuralSamplingRate: 500, // 500 Hz
    realTimeAnalysis: true,
    autoFlush: true,
    flushInterval: 200 // 0.2 seconds
  },
  
  memorySystems: {
    ...DEFAULT_COLLECTOR_CONFIGS.memorySystems,
    collectionInterval: 20, // 50 Hz
    memorySamplingRate: 50, // 50 Hz
    autoFlush: true,
    flushInterval: 1000 // 1 second
  }
} as const;

/**
 * Memory-optimized collector preset for resource-constrained environments
 */
export const MEMORY_OPTIMIZED_CONFIGS = {
  knowledgeGraph: {
    ...DEFAULT_COLLECTOR_CONFIGS.knowledgeGraph,
    bufferSize: 1000,
    maxMemoryUsage: 32, // 32 MB
    enableCompression: true
  },
  
  cognitivePatterns: {
    ...DEFAULT_COLLECTOR_CONFIGS.cognitivePatterns,
    bufferSize: 800,
    maxMemoryUsage: 24, // 24 MB
    enableCompression: true
  },
  
  neuralActivity: {
    ...DEFAULT_COLLECTOR_CONFIGS.neuralActivity,
    bufferSize: 1200,
    maxMemoryUsage: 40, // 40 MB
    enableCompression: true,
    sampleRate: 0.5 // Sample 50% of events
  },
  
  memorySystems: {
    ...DEFAULT_COLLECTOR_CONFIGS.memorySystems,
    bufferSize: 600,
    maxMemoryUsage: 20, // 20 MB
    enableCompression: true
  }
} as const;

/**
 * Collector type enumeration for easy reference
 */
export enum CollectorType {
  KNOWLEDGE_GRAPH = 'knowledge-graph',
  COGNITIVE_PATTERNS = 'cognitive-patterns',
  NEURAL_ACTIVITY = 'neural-activity',
  MEMORY_SYSTEMS = 'memory-systems'
}

/**
 * Collector factory for creating instances with preset configurations
 */
export class CollectorFactory {
  /**
   * Creates a collector with default configuration
   */
  static createDefault(type: CollectorType, mcpClient: any): BaseCollector {
    switch (type) {
      case CollectorType.KNOWLEDGE_GRAPH:
        return new KnowledgeGraphCollector(mcpClient, DEFAULT_COLLECTOR_CONFIGS.knowledgeGraph);
      case CollectorType.COGNITIVE_PATTERNS:
        return new CognitivePatternCollector(mcpClient, DEFAULT_COLLECTOR_CONFIGS.cognitivePatterns);
      case CollectorType.NEURAL_ACTIVITY:
        return new NeuralActivityCollector(mcpClient, DEFAULT_COLLECTOR_CONFIGS.neuralActivity);
      case CollectorType.MEMORY_SYSTEMS:
        return new MemorySystemsCollector(mcpClient, DEFAULT_COLLECTOR_CONFIGS.memorySystems);
      default:
        throw new Error(`Unknown collector type: ${type}`);
    }
  }

  /**
   * Creates a collector with high-performance configuration
   */
  static createHighPerformance(type: CollectorType, mcpClient: any): BaseCollector {
    switch (type) {
      case CollectorType.KNOWLEDGE_GRAPH:
        return new KnowledgeGraphCollector(mcpClient, HIGH_PERFORMANCE_CONFIGS.knowledgeGraph);
      case CollectorType.COGNITIVE_PATTERNS:
        return new CognitivePatternCollector(mcpClient, HIGH_PERFORMANCE_CONFIGS.cognitivePatterns);
      case CollectorType.NEURAL_ACTIVITY:
        return new NeuralActivityCollector(mcpClient, HIGH_PERFORMANCE_CONFIGS.neuralActivity);
      case CollectorType.MEMORY_SYSTEMS:
        return new MemorySystemsCollector(mcpClient, HIGH_PERFORMANCE_CONFIGS.memorySystems);
      default:
        throw new Error(`Unknown collector type: ${type}`);
    }
  }

  /**
   * Creates a collector with low-latency configuration
   */
  static createLowLatency(type: CollectorType, mcpClient: any): BaseCollector {
    switch (type) {
      case CollectorType.KNOWLEDGE_GRAPH:
        return new KnowledgeGraphCollector(mcpClient, LOW_LATENCY_CONFIGS.knowledgeGraph);
      case CollectorType.COGNITIVE_PATTERNS:
        return new CognitivePatternCollector(mcpClient, LOW_LATENCY_CONFIGS.cognitivePatterns);
      case CollectorType.NEURAL_ACTIVITY:
        return new NeuralActivityCollector(mcpClient, LOW_LATENCY_CONFIGS.neuralActivity);
      case CollectorType.MEMORY_SYSTEMS:
        return new MemorySystemsCollector(mcpClient, LOW_LATENCY_CONFIGS.memorySystems);
      default:
        throw new Error(`Unknown collector type: ${type}`);
    }
  }

  /**
   * Creates a collector with memory-optimized configuration
   */
  static createMemoryOptimized(type: CollectorType, mcpClient: any): BaseCollector {
    switch (type) {
      case CollectorType.KNOWLEDGE_GRAPH:
        return new KnowledgeGraphCollector(mcpClient, MEMORY_OPTIMIZED_CONFIGS.knowledgeGraph);
      case CollectorType.COGNITIVE_PATTERNS:
        return new CognitivePatternCollector(mcpClient, MEMORY_OPTIMIZED_CONFIGS.cognitivePatterns);
      case CollectorType.NEURAL_ACTIVITY:
        return new NeuralActivityCollector(mcpClient, MEMORY_OPTIMIZED_CONFIGS.neuralActivity);
      case CollectorType.MEMORY_SYSTEMS:
        return new MemorySystemsCollector(mcpClient, MEMORY_OPTIMIZED_CONFIGS.memorySystems);
      default:
        throw new Error(`Unknown collector type: ${type}`);
    }
  }

  /**
   * Creates a complete collector manager with all collectors
   */
  static createManager(
    mcpClient: any, 
    preset: 'default' | 'high-performance' | 'low-latency' | 'memory-optimized' = 'default'
  ): CollectorManager {
    const configs = preset === 'high-performance' ? HIGH_PERFORMANCE_CONFIGS :
                   preset === 'low-latency' ? LOW_LATENCY_CONFIGS :
                   preset === 'memory-optimized' ? MEMORY_OPTIMIZED_CONFIGS :
                   DEFAULT_COLLECTOR_CONFIGS;

    const managerConfig: Partial<CollectorManagerConfig> = {
      autoStart: true,
      loadBalancingStrategy: preset === 'high-performance' ? 'load-aware' : 'adaptive',
      performanceMonitoring: true
    };

    return new CollectorManager(mcpClient, managerConfig);
  }
}

/**
 * Utility functions for collector management
 */
export const CollectorUtils = {
  /**
   * Validates collector configuration
   */
  validateConfig(config: Partial<CollectorConfig>): boolean {
    if (config.collectionInterval && config.collectionInterval < 10) {
      console.warn('Collection interval below 10ms may cause performance issues');
    }

    if (config.bufferSize && config.bufferSize > 50000) {
      console.warn('Buffer size above 50,000 may cause memory issues');
    }

    if (config.sampleRate && (config.sampleRate < 0 || config.sampleRate > 1)) {
      console.error('Sample rate must be between 0.0 and 1.0');
      return false;
    }

    return true;
  },

  /**
   * Calculates optimal collection interval based on target rate
   */
  calculateOptimalInterval(targetEventsPerSecond: number): number {
    return Math.max(10, 1000 / targetEventsPerSecond);
  },

  /**
   * Estimates memory usage for given configuration
   */
  estimateMemoryUsage(config: Partial<CollectorConfig>): number {
    const bufferSize = config.bufferSize || 1000;
    const avgDataSize = 1024; // Assume 1KB per data point
    return (bufferSize * avgDataSize) / 1024 / 1024; // Convert to MB
  },

  /**
   * Creates a health monitoring configuration
   */
  createHealthMonitoringConfig(): Partial<CollectorManagerConfig> {
    return {
      healthCheckInterval: 15000, // 15 seconds
      performanceMonitoring: true,
      errorRecoveryAttempts: 5,
      resourceLimits: {
        maxMemoryPerCollector: 128,
        maxTotalMemory: 512,
        maxCpuPerCollector: 30,
        maxEventsPerSecond: 2000,
        maxBufferSize: 15000
      }
    };
  }
};

/**
 * Export everything from this module
 */
export * from './base.js';
export * from './knowledge-graph.js';
export * from './cognitive-patterns.js';
export * from './neural-activity.js';
export * from './memory-systems.js';
export * from './manager.js';