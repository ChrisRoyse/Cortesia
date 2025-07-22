/**
 * Telemetry Data Recording and Buffering System
 * 
 * Provides efficient, low-overhead telemetry data collection and buffering
 * with automatic performance impact monitoring.
 */

import { telemetryConfig, TelemetryConfig } from './config.js';

export interface TelemetryEvent {
  /** Unique identifier for the event */
  id: string;
  
  /** Event timestamp in milliseconds */
  timestamp: number;
  
  /** Event type categorization */
  type: 'metric' | 'trace' | 'log' | 'performance';
  
  /** Event category for grouping */
  category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph' | 'mcp' | 'system';
  
  /** Event name/identifier */
  name: string;
  
  /** Event data payload */
  data: Record<string, any>;
  
  /** Performance metrics */
  performance?: {
    duration?: number;
    memoryUsage?: number;
    cpuUsage?: number;
  };
  
  /** Severity level for logs */
  severity?: 'debug' | 'info' | 'warn' | 'error';
  
  /** Tags for filtering and grouping */
  tags?: Record<string, string>;
}

export interface TelemetryBatch {
  /** Batch identifier */
  batchId: string;
  
  /** Batch timestamp */
  timestamp: number;
  
  /** Events in this batch */
  events: TelemetryEvent[];
  
  /** Batch metadata */
  metadata: {
    totalEvents: number;
    totalSize: number;
    processingTime: number;
  };
}

export interface PerformanceImpact {
  /** Current overhead percentage */
  currentOverhead: number;
  
  /** Average overhead over time */
  averageOverhead: number;
  
  /** Peak overhead recorded */
  peakOverhead: number;
  
  /** Number of samples used */
  sampleCount: number;
  
  /** Last measurement timestamp */
  lastMeasurement: number;
  
  /** Whether overhead is within limits */
  withinLimits: boolean;
}

/**
 * High-performance telemetry data recorder with automatic buffering
 */
export class TelemetryRecorder {
  private buffer: TelemetryEvent[] = [];
  private batchCounter = 0;
  private eventCounter = 0;
  private flushTimer?: NodeJS.Timeout;
  private performanceMonitor?: PerformanceMonitor;
  private flushCallbacks: Array<(batch: TelemetryBatch) => Promise<void>> = [];
  private isRecording = false;
  private config: TelemetryConfig;

  constructor() {
    this.config = telemetryConfig.getConfig();
    
    // Subscribe to configuration changes
    telemetryConfig.onConfigChange((newConfig) => {
      this.updateConfiguration(newConfig);
    });

    // Initialize performance monitoring if enabled
    if (this.config.performance.enableImpactMonitoring) {
      this.performanceMonitor = new PerformanceMonitor(this.config);
    }

    // Start recording if enabled
    if (this.config.enabled) {
      this.startRecording();
    }
  }

  /**
   * Start recording telemetry data
   */
  startRecording(): void {
    if (this.isRecording) return;

    this.isRecording = true;
    this.scheduleFlush();
    
    if (this.performanceMonitor) {
      this.performanceMonitor.start();
    }
  }

  /**
   * Stop recording telemetry data
   */
  stopRecording(): void {
    if (!this.isRecording) return;

    this.isRecording = false;
    this.clearFlushTimer();
    
    if (this.performanceMonitor) {
      this.performanceMonitor.stop();
    }

    // Flush remaining data
    if (this.buffer.length > 0) {
      this.flushBuffer();
    }
  }

  /**
   * Record a telemetry event
   */
  recordEvent(event: Omit<TelemetryEvent, 'id' | 'timestamp'>): void {
    if (!this.isRecording || !this.config.enabled) return;

    // Check performance impact if monitoring is enabled
    if (this.performanceMonitor) {
      const impact = this.performanceMonitor.measureImpact(() => {
        this.doRecordEvent(event);
      });

      // If overhead is too high, reduce recording
      if (impact.currentOverhead > this.config.maxOverhead) {
        this.handleHighOverhead(impact);
        return;
      }
    } else {
      this.doRecordEvent(event);
    }
  }

  /**
   * Internal event recording implementation
   */
  private doRecordEvent(event: Omit<TelemetryEvent, 'id' | 'timestamp'>): void {
    const fullEvent: TelemetryEvent = {
      ...event,
      id: this.generateEventId(),
      timestamp: Date.now(),
    };

    // Filter based on configuration
    if (!this.shouldRecordEvent(fullEvent)) {
      return;
    }

    this.buffer.push(fullEvent);

    // Flush if buffer is full
    if (this.buffer.length >= this.config.bufferSize) {
      this.flushBuffer();
    }
  }

  /**
   * Check if event should be recorded based on configuration
   */
  private shouldRecordEvent(event: TelemetryEvent): boolean {
    // Check if event type is enabled
    switch (event.type) {
      case 'metric':
        return this.config.collection.enableMetrics;
      case 'trace':
        return this.config.collection.enableTraces;
      case 'log':
        return this.config.collection.enableLogs;
      default:
        return true;
    }
  }

  /**
   * Record a metric event
   */
  recordMetric(
    category: TelemetryEvent['category'],
    name: string,
    value: number,
    tags?: Record<string, string>
  ): void {
    this.recordEvent({
      type: 'metric',
      category,
      name,
      data: { value },
      tags,
    });
  }

  /**
   * Record a trace event
   */
  recordTrace(
    category: TelemetryEvent['category'],
    name: string,
    data: Record<string, any>,
    performance?: TelemetryEvent['performance'],
    tags?: Record<string, string>
  ): void {
    this.recordEvent({
      type: 'trace',
      category,
      name,
      data,
      performance,
      tags,
    });
  }

  /**
   * Record a log event
   */
  recordLog(
    category: TelemetryEvent['category'],
    name: string,
    message: string,
    severity: TelemetryEvent['severity'] = 'info',
    data?: Record<string, any>,
    tags?: Record<string, string>
  ): void {
    this.recordEvent({
      type: 'log',
      category,
      name,
      data: { message, ...(data || {}) },
      severity,
      tags,
    });
  }

  /**
   * Register a callback for when batches are flushed
   */
  onFlush(callback: (batch: TelemetryBatch) => Promise<void>): void {
    this.flushCallbacks.push(callback);
  }

  /**
   * Unregister a flush callback
   */
  offFlush(callback: (batch: TelemetryBatch) => Promise<void>): void {
    const index = this.flushCallbacks.indexOf(callback);
    if (index !== -1) {
      this.flushCallbacks.splice(index, 1);
    }
  }

  /**
   * Force flush the buffer
   */
  flush(): Promise<void> {
    return this.flushBuffer();
  }

  /**
   * Get current buffer statistics
   */
  getBufferStats(): {
    eventCount: number;
    bufferUsage: number;
    oldestEvent?: number;
    newestEvent?: number;
  } {
    return {
      eventCount: this.buffer.length,
      bufferUsage: this.buffer.length / this.config.bufferSize,
      oldestEvent: this.buffer.length > 0 ? this.buffer[0].timestamp : undefined,
      newestEvent: this.buffer.length > 0 ? this.buffer[this.buffer.length - 1].timestamp : undefined,
    };
  }

  /**
   * Get performance impact statistics
   */
  getPerformanceImpact(): PerformanceImpact | null {
    return this.performanceMonitor?.getImpact() || null;
  }

  /**
   * Clear the buffer without flushing
   */
  clearBuffer(): void {
    this.buffer = [];
  }

  /**
   * Update configuration
   */
  private updateConfiguration(newConfig: TelemetryConfig): void {
    const wasRecording = this.isRecording;
    
    if (wasRecording) {
      this.stopRecording();
    }

    this.config = newConfig;

    // Update performance monitor
    if (this.config.performance.enableImpactMonitoring && !this.performanceMonitor) {
      this.performanceMonitor = new PerformanceMonitor(this.config);
    } else if (!this.config.performance.enableImpactMonitoring && this.performanceMonitor) {
      this.performanceMonitor.stop();
      this.performanceMonitor = undefined;
    } else if (this.performanceMonitor) {
      this.performanceMonitor.updateConfig(this.config);
    }

    if (this.config.enabled && wasRecording) {
      this.startRecording();
    }
  }

  /**
   * Schedule the next buffer flush
   */
  private scheduleFlush(): void {
    this.clearFlushTimer();
    this.flushTimer = setTimeout(() => {
      this.flushBuffer();
      if (this.isRecording) {
        this.scheduleFlush();
      }
    }, this.config.flushInterval);
  }

  /**
   * Clear the flush timer
   */
  private clearFlushTimer(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = undefined;
    }
  }

  /**
   * Flush the buffer to all registered callbacks
   */
  private async flushBuffer(): Promise<void> {
    if (this.buffer.length === 0) return;

    const startTime = performance.now();
    const events = [...this.buffer];
    this.buffer = [];

    const batch: TelemetryBatch = {
      batchId: this.generateBatchId(),
      timestamp: Date.now(),
      events,
      metadata: {
        totalEvents: events.length,
        totalSize: this.calculateBatchSize(events),
        processingTime: 0, // Will be updated after processing
      },
    };

    // Execute flush callbacks
    const flushPromises = this.flushCallbacks.map(async (callback) => {
      try {
        await callback(batch);
      } catch (error) {
        console.warn('Error in telemetry flush callback:', error);
      }
    });

    await Promise.all(flushPromises);

    batch.metadata.processingTime = performance.now() - startTime;
  }

  /**
   * Generate unique event ID
   */
  private generateEventId(): string {
    return `event_${Date.now()}_${++this.eventCounter}`;
  }

  /**
   * Generate unique batch ID
   */
  private generateBatchId(): string {
    return `batch_${Date.now()}_${++this.batchCounter}`;
  }

  /**
   * Calculate approximate batch size in bytes
   */
  private calculateBatchSize(events: TelemetryEvent[]): number {
    return JSON.stringify(events).length;
  }

  /**
   * Handle high performance overhead
   */
  private handleHighOverhead(impact: PerformanceImpact): void {
    console.warn(`Telemetry overhead too high: ${impact.currentOverhead.toFixed(2)}%`);
    
    // Temporarily reduce recording based on severity
    if (impact.currentOverhead > this.config.maxOverhead * 2) {
      // Stop recording temporarily
      this.stopRecording();
      setTimeout(() => {
        if (this.config.enabled) {
          this.startRecording();
        }
      }, 5000);
    }
  }
}

/**
 * Performance impact monitoring system
 */
class PerformanceMonitor {
  private samples: number[] = [];
  private isMonitoring = false;
  private sampleInterval?: NodeJS.Timeout;
  private config: TelemetryConfig;

  constructor(config: TelemetryConfig) {
    this.config = config;
  }

  /**
   * Start performance monitoring
   */
  start(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    this.sampleInterval = setInterval(() => {
      this.takeSample();
    }, 1000); // Sample every second
  }

  /**
   * Stop performance monitoring
   */
  stop(): void {
    if (!this.isMonitoring) return;

    this.isMonitoring = false;
    if (this.sampleInterval) {
      clearInterval(this.sampleInterval);
      this.sampleInterval = undefined;
    }
  }

  /**
   * Measure performance impact of a function
   */
  measureImpact<T>(fn: () => T): { result: T; impact: PerformanceImpact } {
    const startTime = performance.now();
    const startMemory = process.memoryUsage();

    const result = fn();

    const endTime = performance.now();
    const endMemory = process.memoryUsage();

    const duration = endTime - startTime;
    const memoryDelta = endMemory.heapUsed - startMemory.heapUsed;

    // Estimate overhead as percentage of execution time
    const estimatedOverhead = (duration / 100) * 100; // Rough estimate
    this.addSample(Math.min(estimatedOverhead, 100));

    return {
      result,
      impact: this.getImpact(),
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: TelemetryConfig): void {
    this.config = config;
  }

  /**
   * Get current performance impact statistics
   */
  getImpact(): PerformanceImpact {
    const currentOverhead = this.samples.length > 0 ? this.samples[this.samples.length - 1] : 0;
    const averageOverhead = this.samples.length > 0 ? 
      this.samples.reduce((sum, sample) => sum + sample, 0) / this.samples.length : 0;
    const peakOverhead = this.samples.length > 0 ? Math.max(...this.samples) : 0;

    return {
      currentOverhead,
      averageOverhead,
      peakOverhead,
      sampleCount: this.samples.length,
      lastMeasurement: Date.now(),
      withinLimits: currentOverhead <= this.config.maxOverhead,
    };
  }

  /**
   * Take a performance sample
   */
  private takeSample(): void {
    if (Math.random() > this.config.performance.samplingRate) {
      return; // Skip this sample based on sampling rate
    }

    // Simple overhead estimation based on current system metrics
    const memUsage = process.memoryUsage();
    const overhead = (memUsage.heapUsed / (1024 * 1024 * 100)) * 100; // Rough estimate
    
    this.addSample(Math.min(overhead, 100));
  }

  /**
   * Add a performance sample
   */
  private addSample(overhead: number): void {
    this.samples.push(overhead);
    
    // Keep only last 1000 samples
    if (this.samples.length > 1000) {
      this.samples.shift();
    }
  }
}

// Global recorder instance
export const telemetryRecorder = new TelemetryRecorder();