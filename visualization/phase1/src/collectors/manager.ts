/**
 * @fileoverview Collector Manager for LLMKG Visualization
 * 
 * This module provides centralized orchestration of all data collectors in the LLMKG
 * visualization system. It manages collector lifecycles, coordinates data collection,
 * handles load balancing, provides health monitoring, and ensures high-performance
 * operation across all collectors.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { EventEmitter } from 'events';
import { BaseCollector, CollectedData, CollectorStats, CollectorConfig } from './base.js';
import { KnowledgeGraphCollector, KnowledgeGraphCollectorConfig } from './knowledge-graph.js';
import { CognitivePatternCollector, CognitivePatternCollectorConfig } from './cognitive-patterns.js';
import { NeuralActivityCollector, NeuralActivityCollectorConfig } from './neural-activity.js';
import { MemorySystemsCollector, MemorySystemsCollectorConfig } from './memory-systems.js';
import { MCPClient } from '../mcp/client.js';
import { MCPEventType, TelemetryEvent } from '../mcp/types.js';

/**
 * Manager configuration
 */
export interface CollectorManagerConfig {
  /** Enable automatic collector startup */
  autoStart: boolean;
  /** Global collection interval override (ms) */
  globalCollectionInterval?: number;
  /** Maximum concurrent collectors */
  maxConcurrentCollectors: number;
  /** Load balancing strategy */
  loadBalancingStrategy: LoadBalancingStrategy;
  /** Health check interval (ms) */
  healthCheckInterval: number;
  /** Performance monitoring enabled */
  performanceMonitoring: boolean;
  /** Data aggregation window (ms) */
  aggregationWindow: number;
  /** Error recovery attempts */
  errorRecoveryAttempts: number;
  /** Collector priority ordering */
  collectorPriorities: Record<string, number>;
  /** Resource limits */
  resourceLimits: ResourceLimits;
}

/**
 * Load balancing strategies
 */
export type LoadBalancingStrategy = 
  | 'round-robin' 
  | 'priority-based' 
  | 'load-aware' 
  | 'adaptive';

/**
 * Resource limits for collector management
 */
export interface ResourceLimits {
  /** Maximum memory usage per collector (MB) */
  maxMemoryPerCollector: number;
  /** Maximum total memory usage (MB) */
  maxTotalMemory: number;
  /** Maximum CPU usage per collector (%) */
  maxCpuPerCollector: number;
  /** Maximum events per second per collector */
  maxEventsPerSecond: number;
  /** Maximum buffer size per collector */
  maxBufferSize: number;
}

/**
 * Collector configuration registry
 */
export interface CollectorConfigs {
  knowledgeGraph: Partial<KnowledgeGraphCollectorConfig>;
  cognitivePatterns: Partial<CognitivePatternCollectorConfig>;
  neuralActivity: Partial<NeuralActivityCollectorConfig>;
  memorySystems: Partial<MemorySystemsCollectorConfig>;
}

/**
 * Manager statistics
 */
export interface ManagerStats {
  /** Total active collectors */
  activeCollectors: number;
  /** Total data points collected */
  totalDataPoints: number;
  /** Overall collection rate (points/second) */
  overallCollectionRate: number;
  /** System resource usage */
  resourceUsage: SystemResourceUsage;
  /** Error statistics */
  errorStats: ErrorStatistics;
  /** Performance metrics */
  performanceMetrics: PerformanceMetrics;
  /** Last health check results */
  lastHealthCheck: HealthCheckResults;
  /** Load balancing statistics */
  loadBalancingStats: LoadBalancingStats;
}

/**
 * System resource usage
 */
export interface SystemResourceUsage {
  /** Total memory usage (MB) */
  memoryUsage: number;
  /** CPU usage percentage */
  cpuUsage: number;
  /** Network I/O (bytes/sec) */
  networkIO: number;
  /** Disk I/O (bytes/sec) */
  diskIO: number;
  /** Active connections */
  activeConnections: number;
}

/**
 * Error statistics
 */
export interface ErrorStatistics {
  /** Total errors encountered */
  totalErrors: number;
  /** Errors by collector */
  errorsByCollector: Record<string, number>;
  /** Error recovery successes */
  recoverySuccesses: number;
  /** Error recovery failures */
  recoveryFailures: number;
  /** Recent errors */
  recentErrors: ErrorRecord[];
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  /** Average response time (ms) */
  avgResponseTime: number;
  /** Throughput (operations/sec) */
  throughput: number;
  /** Latency percentiles */
  latencyPercentiles: {
    p50: number;
    p90: number;
    p95: number;
    p99: number;
  };
  /** Queue depths by collector */
  queueDepths: Record<string, number>;
  /** Processing efficiency */
  processingEfficiency: number;
}

/**
 * Health check results
 */
export interface HealthCheckResults {
  /** Overall system health */
  overallHealth: HealthStatus;
  /** Health by collector */
  collectorHealth: Record<string, CollectorHealthStatus>;
  /** System alerts */
  alerts: SystemAlert[];
  /** Recommendations */
  recommendations: string[];
  /** Last check timestamp */
  timestamp: Date;
}

/**
 * Health status enumeration
 */
export type HealthStatus = 'healthy' | 'warning' | 'critical' | 'down';

/**
 * Collector health status
 */
export interface CollectorHealthStatus {
  status: HealthStatus;
  uptime: number;
  errorRate: number;
  performanceScore: number;
  resourceUsage: number;
  lastActivity: Date;
}

/**
 * System alert
 */
export interface SystemAlert {
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  source: string;
  timestamp: Date;
  resolved: boolean;
}

/**
 * Load balancing statistics
 */
export interface LoadBalancingStats {
  strategy: LoadBalancingStrategy;
  balancingEffectiveness: number;
  loadDistribution: Record<string, number>;
  rebalancingEvents: number;
  hotspots: string[];
}

/**
 * Error record for tracking
 */
export interface ErrorRecord {
  timestamp: Date;
  collector: string;
  error: Error;
  context: string;
  recovered: boolean;
}

/**
 * Data aggregation result
 */
export interface AggregatedData {
  timeWindow: {
    start: Date;
    end: Date;
  };
  dataByCollector: Record<string, CollectedData[]>;
  totalPoints: number;
  aggregationMetrics: {
    processingTime: number;
    compressionRatio: number;
    duplicatesRemoved: number;
  };
}

/**
 * Centralized collector manager for LLMKG visualization
 */
export class CollectorManager extends EventEmitter {
  private collectors = new Map<string, BaseCollector>();
  private mcpClient: MCPClient;
  private config: CollectorManagerConfig;
  private stats: ManagerStats;
  private healthCheckTimer?: NodeJS.Timeout;
  private performanceTimer?: NodeJS.Timeout;
  private aggregationTimer?: NodeJS.Timeout;
  private dataBuffer = new Map<string, CollectedData[]>();
  private errorRecords: ErrorRecord[] = [];
  private lastLoadBalance = 0;
  private loadBalanceInterval = 30000; // 30 seconds
  private currentLoadStrategy: LoadBalancingStrategy;

  /**
   * Default manager configuration
   */
  private static readonly DEFAULT_CONFIG: CollectorManagerConfig = {
    autoStart: true,
    maxConcurrentCollectors: 10,
    loadBalancingStrategy: 'adaptive',
    healthCheckInterval: 30000, // 30 seconds
    performanceMonitoring: true,
    aggregationWindow: 10000, // 10 seconds
    errorRecoveryAttempts: 3,
    collectorPriorities: {
      'neural-activity': 1,
      'cognitive-patterns': 2,
      'knowledge-graph': 3,
      'memory-systems': 4
    },
    resourceLimits: {
      maxMemoryPerCollector: 256, // 256 MB
      maxTotalMemory: 1024, // 1 GB
      maxCpuPerCollector: 25, // 25%
      maxEventsPerSecond: 1000,
      maxBufferSize: 10000
    }
  };

  constructor(mcpClient: MCPClient, config: Partial<CollectorManagerConfig> = {}) {
    super();
    
    this.mcpClient = mcpClient;
    this.config = { ...CollectorManager.DEFAULT_CONFIG, ...config };
    this.currentLoadStrategy = this.config.loadBalancingStrategy;
    
    this.stats = {
      activeCollectors: 0,
      totalDataPoints: 0,
      overallCollectionRate: 0,
      resourceUsage: {
        memoryUsage: 0,
        cpuUsage: 0,
        networkIO: 0,
        diskIO: 0,
        activeConnections: 0
      },
      errorStats: {
        totalErrors: 0,
        errorsByCollector: {},
        recoverySuccesses: 0,
        recoveryFailures: 0,
        recentErrors: []
      },
      performanceMetrics: {
        avgResponseTime: 0,
        throughput: 0,
        latencyPercentiles: { p50: 0, p90: 0, p95: 0, p99: 0 },
        queueDepths: {},
        processingEfficiency: 1.0
      },
      lastHealthCheck: {
        overallHealth: 'healthy',
        collectorHealth: {},
        alerts: [],
        recommendations: [],
        timestamp: new Date()
      },
      loadBalancingStats: {
        strategy: this.config.loadBalancingStrategy,
        balancingEffectiveness: 1.0,
        loadDistribution: {},
        rebalancingEvents: 0,
        hotspots: []
      }
    };

    this.setupEventHandlers();
    this.setMaxListeners(100);
  }

  /**
   * Initializes the collector manager
   */
  async initialize(collectorConfigs: Partial<CollectorConfigs> = {}): Promise<void> {
    console.log('Initializing Collector Manager...');
    
    try {
      // Create collectors
      await this.createCollectors(collectorConfigs);
      
      // Setup monitoring
      this.setupMonitoring();
      
      // Auto-start if enabled
      if (this.config.autoStart) {
        await this.startAllCollectors();
      }
      
      console.log(`Collector Manager initialized with ${this.collectors.size} collectors`);
      this.emit('manager:initialized', { collectorCount: this.collectors.size });
      
    } catch (error) {
      console.error('Failed to initialize Collector Manager:', error);
      this.emit('manager:error', { error, phase: 'initialization' });
      throw error;
    }
  }

  /**
   * Adds a new collector to the manager
   */
  async addCollector(name: string, collector: BaseCollector): Promise<void> {
    if (this.collectors.has(name)) {
      throw new Error(`Collector '${name}' already exists`);
    }

    this.collectors.set(name, collector);
    this.setupCollectorEventHandlers(name, collector);
    this.updateStats();

    console.log(`Added collector: ${name}`);
    this.emit('collector:added', { name, collector });
  }

  /**
   * Checks if a specific collector is running
   */
  isRunning(collectorName: string): boolean {
    const collector = this.collectors.get(collectorName);
    return collector ? collector.isRunning() : false;
  }

  /**
   * Alias for startAllCollectors - for test compatibility
   */
  async startAll(): Promise<void> {
    return this.startAllCollectors();
  }

  /**
   * Alias for stopAllCollectors - for test compatibility
   */
  async stopAll(): Promise<void> {
    return this.stopAllCollectors();
  }

  /**
   * Starts all collectors
   */
  async startAllCollectors(): Promise<void> {
    console.log('Starting all collectors...');
    
    const startPromises = Array.from(this.collectors.entries()).map(async ([name, collector]) => {
      try {
        await collector.start();
        console.log(`Collector ${name} started successfully`);
      } catch (error) {
        console.error(`Failed to start collector ${name}:`, error);
        this.recordError(name, error as Error, 'startup');
        throw error;
      }
    });

    try {
      await Promise.all(startPromises);
      this.updateStats();
      this.emit('manager:started', { activeCollectors: this.stats.activeCollectors });
    } catch (error) {
      console.error('Failed to start all collectors:', error);
      this.emit('manager:error', { error, phase: 'startup' });
      throw error;
    }
  }

  /**
   * Stops all collectors
   */
  async stopAllCollectors(): Promise<void> {
    console.log('Stopping all collectors...');
    
    // Stop monitoring first
    this.stopMonitoring();
    
    const stopPromises = Array.from(this.collectors.entries()).map(async ([name, collector]) => {
      try {
        if (collector.isRunning()) {
          await collector.stop();
          console.log(`Collector ${name} stopped successfully`);
        }
      } catch (error) {
        console.error(`Failed to stop collector ${name}:`, error);
        this.recordError(name, error as Error, 'shutdown');
      }
    });

    await Promise.allSettled(stopPromises);
    this.updateStats();
    this.emit('manager:stopped');
  }

  /**
   * Restarts a specific collector
   */
  async restartCollector(collectorName: string): Promise<void> {
    const collector = this.collectors.get(collectorName);
    if (!collector) {
      throw new Error(`Collector ${collectorName} not found`);
    }

    console.log(`Restarting collector ${collectorName}...`);
    
    try {
      if (collector.isRunning()) {
        await collector.stop();
      }
      
      await collector.start();
      
      this.emit('collector:restarted', { collector: collectorName });
      console.log(`Collector ${collectorName} restarted successfully`);
      
    } catch (error) {
      this.recordError(collectorName, error as Error, 'restart');
      throw error;
    }
  }

  /**
   * Gets current manager statistics
   */
  getStats(): ManagerStats {
    this.updateStats();
    return { ...this.stats };
  }

  /**
   * Gets statistics for a specific collector
   */
  getCollectorStats(collectorName: string): CollectorStats | null {
    const collector = this.collectors.get(collectorName);
    return collector ? collector.getStats() : null;
  }

  /**
   * Gets health status for all collectors
   */
  async getHealthStatus(): Promise<HealthCheckResults> {
    return await this.performHealthCheck();
  }

  /**
   * Forces data aggregation across all collectors
   */
  async aggregateData(): Promise<AggregatedData> {
    return await this.performDataAggregation();
  }

  /**
   * Configures a specific collector
   */
  configureCollector(collectorName: string, config: Partial<CollectorConfig>): void {
    const collector = this.collectors.get(collectorName);
    if (!collector) {
      throw new Error(`Collector ${collectorName} not found`);
    }

    collector.configure(config);
    this.emit('collector:configured', { collector: collectorName, config });
  }

  /**
   * Gets list of active collectors
   */
  getActiveCollectors(): string[] {
    return Array.from(this.collectors.entries())
      .filter(([_, collector]) => collector.isRunning())
      .map(([name, _]) => name);
  }

  /**
   * Gets collected data from specific collector
   */
  getCollectorData(collectorName: string, maxItems?: number): CollectedData[] {
    const collector = this.collectors.get(collectorName);
    if (!collector) {
      return [];
    }

    return collector.getBufferContents(maxItems);
  }

  /**
   * Enables or disables a collector
   */
  async setCollectorEnabled(collectorName: string, enabled: boolean): Promise<void> {
    const collector = this.collectors.get(collectorName);
    if (!collector) {
      throw new Error(`Collector ${collectorName} not found`);
    }

    if (enabled && !collector.isRunning()) {
      await collector.start();
    } else if (!enabled && collector.isRunning()) {
      await collector.stop();
    }

    this.emit('collector:toggled', { collector: collectorName, enabled });
  }

  /**
   * Performs load balancing across collectors
   */
  async rebalanceLoad(): Promise<void> {
    const now = Date.now();
    if (now - this.lastLoadBalance < this.loadBalanceInterval) {
      return; // Too soon for rebalancing
    }

    console.log(`Performing load balancing using ${this.currentLoadStrategy} strategy...`);
    
    try {
      switch (this.currentLoadStrategy) {
        case 'round-robin':
          await this.rebalanceRoundRobin();
          break;
        case 'priority-based':
          await this.rebalancePriority();
          break;
        case 'load-aware':
          await this.rebalanceLoadAware();
          break;
        case 'adaptive':
          await this.rebalanceAdaptive();
          break;
      }

      this.lastLoadBalance = now;
      this.stats.loadBalancingStats.rebalancingEvents++;
      this.emit('load:balanced', { strategy: this.currentLoadStrategy });
      
    } catch (error) {
      console.error('Load balancing failed:', error);
      this.emit('load:balance:error', { error, strategy: this.currentLoadStrategy });
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    console.log('Cleaning up Collector Manager...');
    
    await this.stopAllCollectors();
    this.collectors.clear();
    this.dataBuffer.clear();
    this.errorRecords = [];
    
    this.emit('manager:cleanup');
  }

  /**
   * Creates all collector instances
   */
  private async createCollectors(configs: Partial<CollectorConfigs>): Promise<void> {
    // Create Knowledge Graph Collector
    const kgCollector = new KnowledgeGraphCollector(
      this.mcpClient, 
      configs.knowledgeGraph
    );
    this.collectors.set('knowledge-graph', kgCollector);

    // Create Cognitive Patterns Collector
    const cognitiveCollector = new CognitivePatternCollector(
      this.mcpClient,
      configs.cognitivePatterns
    );
    this.collectors.set('cognitive-patterns', cognitiveCollector);

    // Create Neural Activity Collector
    const neuralCollector = new NeuralActivityCollector(
      this.mcpClient,
      configs.neuralActivity
    );
    this.collectors.set('neural-activity', neuralCollector);

    // Create Memory Systems Collector
    const memoryCollector = new MemorySystemsCollector(
      this.mcpClient,
      configs.memorySystems
    );
    this.collectors.set('memory-systems', memoryCollector);

    console.log(`Created ${this.collectors.size} collectors`);
  }

  /**
   * Sets up event handlers for collectors
   */
  private setupEventHandlers(): void {
    // Listen to MCP client events
    this.mcpClient.on(MCPEventType.ERROR_OCCURRED, (event) => {
      this.emit('mcp:error', event);
    });

    this.mcpClient.on(MCPEventType.TELEMETRY_EVENT, (event) => {
      this.processTelemetryEvent(event);
    });

    // Setup collector event handlers
    for (const [name, collector] of this.collectors) {
      this.setupCollectorEventHandlers(name, collector);
    }
  }

  /**
   * Sets up event handlers for a specific collector
   */
  private setupCollectorEventHandlers(name: string, collector: BaseCollector): void {
    // Data collection events
    collector.on('data:collected', (event) => {
      this.processCollectedData(name, event.data);
      this.stats.totalDataPoints++;
    });

    // Error events
    collector.on('collection:error', (error) => {
      this.recordError(name, error, 'collection');
    });

    // Performance events
    collector.on('data:flushed', (event) => {
      this.emit('data:flushed', { collector: name, ...event });
    });

    // Health events
    collector.on('memory:warning', (event) => {
      this.handleMemoryWarning(name, event);
    });

    // Status events
    collector.on('started', () => {
      this.emit('collector:started', { collector: name });
      this.updateStats();
    });

    collector.on('stopped', () => {
      this.emit('collector:stopped', { collector: name });
      this.updateStats();
    });
  }

  /**
   * Sets up monitoring timers
   */
  private setupMonitoring(): void {
    // Health check monitoring
    if (this.config.healthCheckInterval > 0) {
      this.healthCheckTimer = setInterval(async () => {
        await this.performHealthCheck();
      }, this.config.healthCheckInterval);
    }

    // Performance monitoring
    if (this.config.performanceMonitoring) {
      this.performanceTimer = setInterval(() => {
        this.updatePerformanceMetrics();
      }, 5000); // Every 5 seconds
    }

    // Data aggregation
    if (this.config.aggregationWindow > 0) {
      this.aggregationTimer = setInterval(async () => {
        await this.performDataAggregation();
      }, this.config.aggregationWindow);
    }

    // Load balancing
    setInterval(async () => {
      await this.rebalanceLoad();
    }, this.loadBalanceInterval);
  }

  /**
   * Stops monitoring timers
   */
  private stopMonitoring(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = undefined;
    }

    if (this.performanceTimer) {
      clearInterval(this.performanceTimer);
      this.performanceTimer = undefined;
    }

    if (this.aggregationTimer) {
      clearInterval(this.aggregationTimer);
      this.aggregationTimer = undefined;
    }
  }

  /**
   * Processes collected data from collectors
   */
  private processCollectedData(collectorName: string, data: CollectedData): void {
    // Add to buffer
    if (!this.dataBuffer.has(collectorName)) {
      this.dataBuffer.set(collectorName, []);
    }
    
    const buffer = this.dataBuffer.get(collectorName)!;
    buffer.push(data);
    
    // Maintain buffer size
    if (buffer.length > this.config.resourceLimits.maxBufferSize) {
      buffer.splice(0, buffer.length - this.config.resourceLimits.maxBufferSize);
    }

    // Emit processed data event
    this.emit('data:processed', {
      collector: collectorName,
      data,
      timestamp: Date.now()
    });
  }

  /**
   * Processes telemetry events
   */
  private processTelemetryEvent(event: TelemetryEvent): void {
    // Update stats based on telemetry
    if (event.data.metric.includes('response_time')) {
      // Update response time metrics
    }
    
    if (event.data.metric.includes('error')) {
      this.stats.errorStats.totalErrors++;
    }

    this.emit('telemetry:processed', event);
  }

  /**
   * Records error for tracking and recovery
   */
  private recordError(collectorName: string, error: Error, context: string): void {
    const errorRecord: ErrorRecord = {
      timestamp: new Date(),
      collector: collectorName,
      error,
      context,
      recovered: false
    };

    this.errorRecords.push(errorRecord);
    
    // Maintain error record size
    if (this.errorRecords.length > 1000) {
      this.errorRecords = this.errorRecords.slice(-500);
    }

    // Update stats
    this.stats.errorStats.totalErrors++;
    this.stats.errorStats.errorsByCollector[collectorName] = 
      (this.stats.errorStats.errorsByCollector[collectorName] || 0) + 1;
    this.stats.errorStats.recentErrors.push(errorRecord);
    
    // Keep only recent errors
    this.stats.errorStats.recentErrors = this.stats.errorStats.recentErrors.slice(-50);

    this.emit('error:recorded', { collector: collectorName, error, context });

    // Attempt recovery
    this.attemptErrorRecovery(collectorName, errorRecord);
  }

  /**
   * Attempts to recover from errors
   */
  private async attemptErrorRecovery(collectorName: string, errorRecord: ErrorRecord): Promise<void> {
    const maxAttempts = this.config.errorRecoveryAttempts;
    const collectorErrors = this.stats.errorStats.errorsByCollector[collectorName] || 0;
    
    if (collectorErrors >= maxAttempts) {
      console.warn(`Collector ${collectorName} has exceeded error recovery attempts (${maxAttempts})`);
      this.stats.errorStats.recoveryFailures++;
      return;
    }

    try {
      console.log(`Attempting recovery for collector ${collectorName}...`);
      
      // Restart the collector
      await this.restartCollector(collectorName);
      
      errorRecord.recovered = true;
      this.stats.errorStats.recoverySuccesses++;
      
      this.emit('error:recovered', { collector: collectorName });
      
    } catch (recoveryError) {
      console.error(`Recovery failed for collector ${collectorName}:`, recoveryError);
      this.stats.errorStats.recoveryFailures++;
      
      this.emit('error:recovery:failed', { 
        collector: collectorName, 
        originalError: errorRecord.error,
        recoveryError 
      });
    }
  }

  /**
   * Handles memory warnings from collectors
   */
  private handleMemoryWarning(collectorName: string, event: any): void {
    console.warn(`Memory warning from collector ${collectorName}:`, event);
    
    const alert: SystemAlert = {
      severity: 'warning',
      message: `High memory usage in collector ${collectorName}: ${event.currentUsage}MB`,
      source: collectorName,
      timestamp: new Date(),
      resolved: false
    };

    this.stats.lastHealthCheck.alerts.push(alert);
    this.emit('system:alert', alert);

    // Take corrective action if needed
    if (event.currentUsage > this.config.resourceLimits.maxMemoryPerCollector) {
      this.emit('collector:memory:critical', { collector: collectorName, usage: event.currentUsage });
    }
  }

  /**
   * Updates manager statistics
   */
  private updateStats(): void {
    this.stats.activeCollectors = Array.from(this.collectors.values())
      .filter(collector => collector.isRunning()).length;

    // Calculate overall collection rate
    const now = Date.now();
    const window = 60000; // 1 minute window
    const recentData = Array.from(this.dataBuffer.values())
      .flat()
      .filter(data => now - data.timestamp < window);
    
    this.stats.overallCollectionRate = recentData.length / 60; // per second

    // Update resource usage
    this.updateResourceUsage();
    
    // Update load balancing stats
    this.updateLoadBalancingStats();
  }

  /**
   * Updates resource usage statistics
   */
  private updateResourceUsage(): void {
    // Simulate resource usage calculation
    const usage = process.memoryUsage();
    this.stats.resourceUsage.memoryUsage = usage.heapUsed / 1024 / 1024; // MB
    this.stats.resourceUsage.cpuUsage = Math.random() * 50 + 20; // Simulate 20-70%
    this.stats.resourceUsage.networkIO = Math.random() * 1000000; // bytes/sec
    this.stats.resourceUsage.diskIO = Math.random() * 500000; // bytes/sec
    this.stats.resourceUsage.activeConnections = this.stats.activeCollectors;
  }

  /**
   * Updates performance metrics
   */
  private updatePerformanceMetrics(): void {
    const responseTimes: number[] = [];
    
    // Collect response times from all collectors
    for (const collector of this.collectors.values()) {
      const stats = collector.getStats();
      if (stats.averageProcessingTime > 0) {
        responseTimes.push(stats.averageProcessingTime);
      }
    }

    if (responseTimes.length > 0) {
      responseTimes.sort((a, b) => a - b);
      
      this.stats.performanceMetrics.avgResponseTime = 
        responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
      
      this.stats.performanceMetrics.latencyPercentiles = {
        p50: responseTimes[Math.floor(responseTimes.length * 0.5)],
        p90: responseTimes[Math.floor(responseTimes.length * 0.9)],
        p95: responseTimes[Math.floor(responseTimes.length * 0.95)],
        p99: responseTimes[Math.floor(responseTimes.length * 0.99)]
      };
    }

    this.stats.performanceMetrics.throughput = this.stats.overallCollectionRate;
    this.stats.performanceMetrics.processingEfficiency = 
      this.stats.errorStats.totalErrors > 0 
        ? 1 - (this.stats.errorStats.totalErrors / (this.stats.totalDataPoints + this.stats.errorStats.totalErrors))
        : 1.0;

    // Update queue depths
    for (const [name, collector] of this.collectors) {
      this.stats.performanceMetrics.queueDepths[name] = collector.getStats().bufferSize;
    }
  }

  /**
   * Updates load balancing statistics
   */
  private updateLoadBalancingStats(): void {
    const loadDistribution: Record<string, number> = {};
    let totalLoad = 0;

    for (const [name, collector] of this.collectors) {
      const stats = collector.getStats();
      const load = stats.eventsPerSecond || 0;
      loadDistribution[name] = load;
      totalLoad += load;
    }

    // Normalize load distribution
    if (totalLoad > 0) {
      for (const name in loadDistribution) {
        loadDistribution[name] = loadDistribution[name] / totalLoad;
      }
    }

    this.stats.loadBalancingStats.loadDistribution = loadDistribution;

    // Calculate balancing effectiveness
    const loadValues = Object.values(loadDistribution);
    const avgLoad = loadValues.reduce((a, b) => a + b, 0) / loadValues.length;
    const variance = loadValues.reduce((sum, load) => sum + Math.pow(load - avgLoad, 2), 0) / loadValues.length;
    this.stats.loadBalancingStats.balancingEffectiveness = Math.max(0, 1 - variance);

    // Identify hotspots
    this.stats.loadBalancingStats.hotspots = Object.entries(loadDistribution)
      .filter(([_, load]) => load > avgLoad * 1.5)
      .map(([name, _]) => name);
  }

  /**
   * Performs comprehensive health check
   */
  private async performHealthCheck(): Promise<HealthCheckResults> {
    const results: HealthCheckResults = {
      overallHealth: 'healthy',
      collectorHealth: {},
      alerts: [],
      recommendations: [],
      timestamp: new Date()
    };

    // Check each collector
    for (const [name, collector] of this.collectors) {
      const collectorHealth = this.assessCollectorHealth(name, collector);
      results.collectorHealth[name] = collectorHealth;

      if (collectorHealth.status === 'critical' || collectorHealth.status === 'down') {
        results.overallHealth = 'critical';
      } else if (collectorHealth.status === 'warning' && results.overallHealth === 'healthy') {
        results.overallHealth = 'warning';
      }
    }

    // Check system resources
    this.checkSystemResources(results);

    // Generate recommendations
    this.generateRecommendations(results);

    this.stats.lastHealthCheck = results;
    this.emit('health:check:complete', results);

    return results;
  }

  /**
   * Assesses health of individual collector
   */
  private assessCollectorHealth(name: string, collector: BaseCollector): CollectorHealthStatus {
    const stats = collector.getStats();
    const health = collector.getHealthStatus();
    
    let status: HealthStatus = 'healthy';
    let performanceScore = 1.0;

    // Check if collector is running
    if (!collector.isRunning()) {
      status = 'down';
    } else {
      // Check error rate
      const errorRate = stats.totalCollected > 0 
        ? stats.failedCollections / stats.totalCollected 
        : 0;

      if (errorRate > 0.1) {
        status = 'critical';
      } else if (errorRate > 0.05) {
        status = 'warning';
      }

      // Check performance
      performanceScore = Math.max(0, 1 - errorRate);
    }

    return {
      status,
      uptime: health.uptime,
      errorRate: stats.totalCollected > 0 ? stats.failedCollections / stats.totalCollected : 0,
      performanceScore,
      resourceUsage: health.memoryUsage / this.config.resourceLimits.maxMemoryPerCollector,
      lastActivity: stats.lastCollection || new Date(0)
    };
  }

  /**
   * Checks system resource constraints
   */
  private checkSystemResources(results: HealthCheckResults): void {
    const usage = this.stats.resourceUsage;
    const limits = this.config.resourceLimits;

    // Check memory usage
    if (usage.memoryUsage > limits.maxTotalMemory * 0.9) {
      results.alerts.push({
        severity: 'critical',
        message: `High system memory usage: ${usage.memoryUsage.toFixed(1)}MB / ${limits.maxTotalMemory}MB`,
        source: 'system',
        timestamp: new Date(),
        resolved: false
      });
    } else if (usage.memoryUsage > limits.maxTotalMemory * 0.8) {
      results.alerts.push({
        severity: 'warning',
        message: `Elevated memory usage: ${usage.memoryUsage.toFixed(1)}MB / ${limits.maxTotalMemory}MB`,
        source: 'system',
        timestamp: new Date(),
        resolved: false
      });
    }

    // Check CPU usage
    if (usage.cpuUsage > 90) {
      results.alerts.push({
        severity: 'critical',
        message: `High CPU usage: ${usage.cpuUsage.toFixed(1)}%`,
        source: 'system',
        timestamp: new Date(),
        resolved: false
      });
    }
  }

  /**
   * Generates system recommendations
   */
  private generateRecommendations(results: HealthCheckResults): void {
    // Check for underperforming collectors
    for (const [name, health] of Object.entries(results.collectorHealth)) {
      if (health.performanceScore < 0.8) {
        results.recommendations.push(
          `Consider restarting collector '${name}' due to poor performance (${Math.round(health.performanceScore * 100)}%)`
        );
      }
    }

    // Check load distribution
    const hotspots = this.stats.loadBalancingStats.hotspots;
    if (hotspots.length > 0) {
      results.recommendations.push(
        `Load balancing needed for collectors: ${hotspots.join(', ')}`
      );
    }

    // Check overall efficiency
    if (this.stats.performanceMetrics.processingEfficiency < 0.9) {
      results.recommendations.push(
        'System efficiency is below 90% - consider investigating error patterns'
      );
    }
  }

  /**
   * Performs data aggregation across collectors
   */
  private async performDataAggregation(): Promise<AggregatedData> {
    const startTime = Date.now();
    const endTime = startTime;
    const windowStart = startTime - this.config.aggregationWindow;
    
    const aggregatedData: AggregatedData = {
      timeWindow: {
        start: new Date(windowStart),
        end: new Date(endTime)
      },
      dataByCollector: {},
      totalPoints: 0,
      aggregationMetrics: {
        processingTime: 0,
        compressionRatio: 1.0,
        duplicatesRemoved: 0
      }
    };

    let totalDataPoints = 0;
    let duplicatesRemoved = 0;

    // Aggregate data from each collector
    for (const [name, dataBuffer] of this.dataBuffer) {
      const recentData = dataBuffer.filter(data => 
        data.timestamp >= windowStart && data.timestamp <= endTime
      );

      // Remove duplicates (simplified)
      const uniqueData = new Map<string, CollectedData>();
      for (const data of recentData) {
        const key = `${data.source}-${data.type}-${data.timestamp}`;
        if (!uniqueData.has(key)) {
          uniqueData.set(key, data);
        } else {
          duplicatesRemoved++;
        }
      }

      aggregatedData.dataByCollector[name] = Array.from(uniqueData.values());
      totalDataPoints += aggregatedData.dataByCollector[name].length;
    }

    aggregatedData.totalPoints = totalDataPoints;
    aggregatedData.aggregationMetrics.processingTime = Date.now() - startTime;
    aggregatedData.aggregationMetrics.duplicatesRemoved = duplicatesRemoved;
    aggregatedData.aggregationMetrics.compressionRatio = totalDataPoints > 0 
      ? (totalDataPoints - duplicatesRemoved) / totalDataPoints 
      : 1.0;

    this.emit('data:aggregated', aggregatedData);
    return aggregatedData;
  }

  /**
   * Load balancing strategies implementation
   */
  private async rebalanceRoundRobin(): Promise<void> {
    // Implement round-robin load balancing
    console.log('Executing round-robin load balancing');
  }

  private async rebalancePriority(): Promise<void> {
    // Implement priority-based load balancing
    const priorities = this.config.collectorPriorities;
    const collectors = Array.from(this.collectors.entries())
      .sort(([a], [b]) => (priorities[a] || 999) - (priorities[b] || 999));

    console.log('Executing priority-based load balancing:', collectors.map(([name]) => name));
  }

  private async rebalanceLoadAware(): Promise<void> {
    // Implement load-aware balancing
    const loadData = Object.entries(this.stats.loadBalancingStats.loadDistribution)
      .sort(([, a], [, b]) => a - b);
    
    console.log('Executing load-aware balancing based on current loads:', loadData);
  }

  private async rebalanceAdaptive(): Promise<void> {
    // Implement adaptive load balancing
    const efficiency = this.stats.loadBalancingStats.balancingEffectiveness;
    
    if (efficiency < 0.8) {
      // Switch to load-aware strategy
      this.currentLoadStrategy = 'load-aware';
      await this.rebalanceLoadAware();
    } else if (this.stats.loadBalancingStats.hotspots.length > 2) {
      // Switch to priority-based strategy
      this.currentLoadStrategy = 'priority-based';
      await this.rebalancePriority();
    } else {
      // Use round-robin as default
      this.currentLoadStrategy = 'round-robin';
      await this.rebalanceRoundRobin();
    }

    console.log(`Adaptive balancing selected strategy: ${this.currentLoadStrategy}`);
  }
}