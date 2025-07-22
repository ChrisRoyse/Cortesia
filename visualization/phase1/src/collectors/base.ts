/**
 * @fileoverview Base Data Collector for LLMKG Visualization
 * 
 * This module provides the abstract base class for all LLMKG data collectors.
 * It includes common functionality such as buffering, aggregation, event handling,
 * and high-frequency data processing capabilities optimized for >1000 events/sec.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { EventEmitter } from 'events';
import { MCPClient } from '../mcp/client.js';
import { MCPEventType, TelemetryEvent } from '../mcp/types.js';

/**
 * Configuration for data collector instances
 */
export interface CollectorConfig {
  /** Collector name/identifier */
  name: string;
  /** Collection interval in milliseconds */
  collectionInterval: number;
  /** Buffer size for batching events */
  bufferSize: number;
  /** Maximum age of data in buffer (ms) */
  maxBufferAge: number;
  /** Enable automatic flushing */
  autoFlush: boolean;
  /** Flush interval in milliseconds */
  flushInterval: number;
  /** Enable compression for large data */
  enableCompression: boolean;
  /** Maximum memory usage in MB */
  maxMemoryUsage: number;
  /** Sample rate for high-frequency data (0.0-1.0) */
  sampleRate: number;
}

/**
 * Base data structure for collected data
 */
export interface CollectedData {
  /** Unique identifier for this data point */
  id: string;
  /** Timestamp when data was collected */
  timestamp: number;
  /** Source of the data (tool name, component, etc.) */
  source: string;
  /** Data type identifier */
  type: string;
  /** The actual collected data */
  data: any;
  /** Metadata about the collection */
  metadata: CollectionMetadata;
}

/**
 * Metadata about the data collection process
 */
export interface CollectionMetadata {
  /** Collector instance that generated this data */
  collector: string;
  /** Collection method used */
  method: string;
  /** Processing duration in milliseconds */
  processingTime?: number;
  /** Data quality score (0.0-1.0) */
  quality?: number;
  /** Tags for categorization */
  tags: Record<string, string>;
  /** Additional context */
  context?: Record<string, any>;
}

/**
 * Statistics about collector performance
 */
export interface CollectorStats {
  /** Total number of data points collected */
  totalCollected: number;
  /** Number of successful collections */
  successfulCollections: number;
  /** Number of failed collections */
  failedCollections: number;
  /** Current buffer size */
  bufferSize: number;
  /** Average processing time in milliseconds */
  averageProcessingTime: number;
  /** Current memory usage in MB */
  memoryUsage: number;
  /** Events per second rate */
  eventsPerSecond: number;
  /** Last collection timestamp */
  lastCollection?: Date;
  /** Collection success rate (0.0-1.0) */
  successRate: number;
}

/**
 * Circular buffer for high-performance data storage
 */
export class CircularBuffer<T> {
  private buffer: T[] = [];
  private writeIndex: number = 0;
  private readIndex: number = 0;
  private size: number = 0;
  private totalAdded: number = 0;

  constructor(private capacity: number) {}

  /**
   * Adds an item to the buffer
   */
  add(item: T): boolean {
    if (this.size < this.capacity) {
      this.buffer[this.writeIndex] = item;
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      this.size++;
    } else {
      // Overwrite oldest item
      this.buffer[this.writeIndex] = item;
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
      this.readIndex = (this.readIndex + 1) % this.capacity;
    }
    
    this.totalAdded++;
    return true;
  }

  /**
   * Retrieves items from buffer without removing them
   */
  peek(count?: number): T[] {
    const items: T[] = [];
    const actualCount = Math.min(count || this.size, this.size);
    
    for (let i = 0; i < actualCount; i++) {
      const index = (this.readIndex + i) % this.capacity;
      items.push(this.buffer[index]);
    }
    
    return items;
  }

  /**
   * Removes and returns items from buffer
   */
  drain(count?: number): T[] {
    const items: T[] = [];
    const actualCount = Math.min(count || this.size, this.size);
    
    for (let i = 0; i < actualCount; i++) {
      const index = this.readIndex;
      items.push(this.buffer[index]);
      this.readIndex = (this.readIndex + 1) % this.capacity;
      this.size--;
    }
    
    return items;
  }

  /**
   * Clears all items from buffer
   */
  clear(): void {
    this.buffer = [];
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
  }

  /**
   * Gets current buffer statistics
   */
  getStats() {
    return {
      size: this.size,
      capacity: this.capacity,
      utilization: this.size / this.capacity,
      totalAdded: this.totalAdded
    };
  }

  /**
   * Checks if buffer is full
   */
  isFull(): boolean {
    return this.size >= this.capacity;
  }

  /**
   * Checks if buffer is empty
   */
  isEmpty(): boolean {
    return this.size === 0;
  }
}

/**
 * Data aggregator for statistical analysis
 */
export class DataAggregator {
  private values: number[] = [];
  private timestamps: number[] = [];
  private maxSize: number = 10000;

  /**
   * Adds a data point for aggregation
   */
  addValue(value: number, timestamp: number = Date.now()): void {
    this.values.push(value);
    this.timestamps.push(timestamp);

    // Maintain size limit
    if (this.values.length > this.maxSize) {
      this.values = this.values.slice(-this.maxSize * 0.8);
      this.timestamps = this.timestamps.slice(-this.maxSize * 0.8);
    }
  }

  /**
   * Calculates statistical measures
   */
  getStatistics() {
    if (this.values.length === 0) {
      return {
        count: 0,
        mean: 0,
        median: 0,
        stdDev: 0,
        min: 0,
        max: 0,
        sum: 0
      };
    }

    const sorted = [...this.values].sort((a, b) => a - b);
    const sum = this.values.reduce((a, b) => a + b, 0);
    const mean = sum / this.values.length;
    
    const variance = this.values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / this.values.length;
    const stdDev = Math.sqrt(variance);

    return {
      count: this.values.length,
      mean,
      median: sorted[Math.floor(sorted.length / 2)],
      stdDev,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      sum
    };
  }

  /**
   * Gets rate of change over time
   */
  getRate(windowMs: number = 60000): number {
    const now = Date.now();
    const cutoff = now - windowMs;
    
    let count = 0;
    for (const timestamp of this.timestamps) {
      if (timestamp >= cutoff) count++;
    }
    
    return count / (windowMs / 1000); // Events per second
  }

  /**
   * Clears all aggregated data
   */
  clear(): void {
    this.values = [];
    this.timestamps = [];
  }
}

/**
 * Base abstract class for all LLMKG data collectors
 */
export abstract class BaseCollector extends EventEmitter {
  protected buffer: CircularBuffer<CollectedData>;
  protected aggregator: DataAggregator;
  protected config: CollectorConfig;
  protected stats: CollectorStats;
  protected isActive: boolean = false;
  protected collectionTimer?: NodeJS.Timeout;
  protected flushTimer?: NodeJS.Timeout;
  protected startTime: number = Date.now();
  protected processingTimes: number[] = [];

  /**
   * Default configuration for collectors
   */
  protected static readonly DEFAULT_CONFIG: CollectorConfig = {
    name: 'base-collector',
    collectionInterval: 100, // 10 Hz for high-frequency collection
    bufferSize: 10000,
    maxBufferAge: 60000, // 1 minute
    autoFlush: true,
    flushInterval: 5000, // 5 seconds
    enableCompression: false,
    maxMemoryUsage: 100, // 100 MB
    sampleRate: 1.0 // Collect all data by default
  };

  constructor(protected mcpClient: MCPClient, config: Partial<CollectorConfig> = {}) {
    super();
    
    this.config = { ...BaseCollector.DEFAULT_CONFIG, ...config };
    this.buffer = new CircularBuffer<CollectedData>(this.config.bufferSize);
    this.aggregator = new DataAggregator();
    
    this.stats = {
      totalCollected: 0,
      successfulCollections: 0,
      failedCollections: 0,
      bufferSize: 0,
      averageProcessingTime: 0,
      memoryUsage: 0,
      eventsPerSecond: 0,
      successRate: 0
    };

    this.setupEventHandlers();
    this.setMaxListeners(1000);
  }

  /**
   * Abstract method to be implemented by specific collectors
   */
  abstract collect(): Promise<CollectedData[]>;

  /**
   * Abstract method for collector-specific initialization
   */
  abstract initialize(): Promise<void>;

  /**
   * Abstract method for collector-specific cleanup
   */
  abstract cleanup(): Promise<void>;

  /**
   * Starts the data collection process
   */
  async start(): Promise<void> {
    if (this.isActive) {
      console.warn(`Collector ${this.config.name} is already active`);
      return;
    }

    try {
      await this.initialize();
      this.isActive = true;
      
      // Start collection timer
      this.collectionTimer = setInterval(async () => {
        await this.performCollection();
      }, this.config.collectionInterval);

      // Start flush timer if auto-flush is enabled
      if (this.config.autoFlush) {
        this.flushTimer = setInterval(async () => {
          await this.flush();
        }, this.config.flushInterval);
      }

      this.emit('started', { collector: this.config.name });
      console.log(`Collector ${this.config.name} started successfully`);
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Stops the data collection process
   */
  async stop(): Promise<void> {
    if (!this.isActive) {
      return;
    }

    this.isActive = false;

    // Clear timers
    if (this.collectionTimer) {
      clearInterval(this.collectionTimer);
      this.collectionTimer = undefined;
    }

    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = undefined;
    }

    // Flush remaining data
    await this.flush();

    // Cleanup resources
    await this.cleanup();

    this.emit('stopped', { collector: this.config.name });
    console.log(`Collector ${this.config.name} stopped successfully`);
  }

  /**
   * Performs a single collection cycle
   */
  private async performCollection(): Promise<void> {
    if (!this.isActive) return;

    const startTime = Date.now();
    
    try {
      // Apply sampling if configured
      if (this.config.sampleRate < 1.0 && Math.random() > this.config.sampleRate) {
        return;
      }

      const collectedData = await this.collect();
      
      // Process each collected data point
      for (const data of collectedData) {
        this.processCollectedData(data);
      }

      const processingTime = Date.now() - startTime;
      this.recordProcessingTime(processingTime);
      
      this.stats.successfulCollections++;
      this.stats.lastCollection = new Date();
      
    } catch (error) {
      this.stats.failedCollections++;
      this.emit('collection:error', {
        collector: this.config.name,
        error,
        timestamp: Date.now()
      });
    }

    this.updateStats();
  }

  /**
   * Processes and stores collected data
   */
  private processCollectedData(data: CollectedData): void {
    // Validate data
    if (!this.validateData(data)) {
      console.warn(`Invalid data rejected by ${this.config.name}:`, data);
      return;
    }

    // Add to buffer
    this.buffer.add(data);
    
    // Add to aggregator if data contains numeric value
    if (typeof data.data === 'number') {
      this.aggregator.addValue(data.data, data.timestamp);
    }

    this.stats.totalCollected++;
    this.stats.bufferSize = this.buffer.getStats().size;

    this.emit('data:collected', {
      collector: this.config.name,
      data,
      bufferSize: this.stats.bufferSize
    });

    // Check memory usage
    this.checkMemoryUsage();

    // Auto-flush if buffer is full
    if (this.buffer.isFull()) {
      setImmediate(() => this.flush());
    }
  }

  /**
   * Validates collected data
   */
  protected validateData(data: CollectedData): boolean {
    return (
      data &&
      typeof data.id === 'string' &&
      typeof data.timestamp === 'number' &&
      typeof data.source === 'string' &&
      typeof data.type === 'string' &&
      data.data !== undefined &&
      data.metadata &&
      typeof data.metadata.collector === 'string'
    );
  }

  /**
   * Flushes buffered data
   */
  async flush(): Promise<void> {
    if (this.buffer.isEmpty()) return;

    const data = this.buffer.drain();
    
    try {
      await this.processFlush(data);
      
      this.emit('data:flushed', {
        collector: this.config.name,
        count: data.length,
        timestamp: Date.now()
      });
      
    } catch (error) {
      // Re-add data to buffer if flush fails
      for (const item of data.reverse()) {
        this.buffer.add(item);
      }
      
      this.emit('flush:error', {
        collector: this.config.name,
        error,
        dataCount: data.length
      });
    }
  }

  /**
   * Processes flushed data (override in subclasses)
   */
  protected async processFlush(data: CollectedData[]): Promise<void> {
    // Default implementation: emit telemetry events
    for (const item of data) {
      this.emit('telemetry', {
        type: MCPEventType.TELEMETRY_EVENT,
        timestamp: new Date(),
        data: {
          metric: `${this.config.name}.${item.type}`,
          value: item.data,
          tags: {
            collector: this.config.name,
            source: item.source,
            ...item.metadata.tags
          },
          metadata: item.metadata
        }
      } as TelemetryEvent);
    }
  }

  /**
   * Gets current collector statistics
   */
  getStats(): CollectorStats {
    return { ...this.stats };
  }

  /**
   * Gets current buffer contents
   */
  getBufferContents(maxItems?: number): CollectedData[] {
    return this.buffer.peek(maxItems);
  }

  /**
   * Gets aggregated statistics
   */
  getAggregatedStats() {
    return this.aggregator.getStatistics();
  }

  /**
   * Configures the collector
   */
  configure(config: Partial<CollectorConfig>): void {
    this.config = { ...this.config, ...config };
    
    // Recreate buffer if size changed
    if (config.bufferSize && config.bufferSize !== this.buffer.getStats().capacity) {
      const oldData = this.buffer.drain();
      this.buffer = new CircularBuffer<CollectedData>(config.bufferSize);
      
      // Re-add data up to new capacity
      for (const item of oldData.slice(-config.bufferSize)) {
        this.buffer.add(item);
      }
    }

    this.emit('configured', {
      collector: this.config.name,
      config: this.config
    });
  }

  /**
   * Records processing time for performance monitoring
   */
  private recordProcessingTime(duration: number): void {
    this.processingTimes.push(duration);
    
    // Keep only last 1000 measurements
    if (this.processingTimes.length > 1000) {
      this.processingTimes = this.processingTimes.slice(-1000);
    }

    this.stats.averageProcessingTime = 
      this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
  }

  /**
   * Updates collector statistics
   */
  private updateStats(): void {
    const now = Date.now();
    const runtime = (now - this.startTime) / 1000; // seconds
    
    this.stats.successRate = this.stats.totalCollected > 0 
      ? this.stats.successfulCollections / this.stats.totalCollected
      : 0;
    
    this.stats.eventsPerSecond = this.stats.totalCollected / runtime;
    this.stats.bufferSize = this.buffer.getStats().size;
  }

  /**
   * Checks memory usage and takes action if needed
   */
  private checkMemoryUsage(): void {
    const used = process.memoryUsage();
    const usedMB = used.heapUsed / 1024 / 1024;
    
    this.stats.memoryUsage = usedMB;

    if (usedMB > this.config.maxMemoryUsage) {
      console.warn(`Collector ${this.config.name} memory usage (${usedMB.toFixed(1)}MB) exceeds limit (${this.config.maxMemoryUsage}MB)`);
      
      // Emergency flush
      setImmediate(() => this.flush());
      
      this.emit('memory:warning', {
        collector: this.config.name,
        currentUsage: usedMB,
        maxUsage: this.config.maxMemoryUsage
      });
    }
  }

  /**
   * Sets up event handlers for MCP client
   */
  private setupEventHandlers(): void {
    this.mcpClient.on(MCPEventType.TOOL_RESPONSE, (event) => {
      this.emit('mcp:tool:response', event);
    });

    this.mcpClient.on(MCPEventType.ERROR_OCCURRED, (event) => {
      this.emit('mcp:error', event);
    });

    this.mcpClient.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
      this.emit('mcp:connection', event);
    });
  }

  /**
   * Generates a unique ID for collected data
   */
  protected generateId(): string {
    return `${this.config.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Creates standard metadata for collected data
   */
  protected createMetadata(method: string, processingTime?: number, context?: Record<string, any>): CollectionMetadata {
    return {
      collector: this.config.name,
      method,
      processingTime,
      tags: {
        collector: this.config.name,
        method,
        timestamp: Date.now().toString()
      },
      context
    };
  }

  /**
   * Checks if collector is currently active
   */
  isRunning(): boolean {
    return this.isActive;
  }

  /**
   * Gets collector health status
   */
  getHealthStatus() {
    const bufferStats = this.buffer.getStats();
    const aggregateStats = this.aggregator.getStatistics();
    
    return {
      isActive: this.isActive,
      health: this.stats.successRate > 0.95 ? 'healthy' : 
              this.stats.successRate > 0.80 ? 'warning' : 'critical',
      collector: this.config.name,
      stats: this.stats,
      buffer: bufferStats,
      aggregates: aggregateStats,
      memoryUsage: this.stats.memoryUsage,
      uptime: (Date.now() - this.startTime) / 1000
    };
  }
}