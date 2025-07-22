/**
 * Telemetry Orchestration Manager
 * 
 * Central orchestration system that manages all telemetry components,
 * provides unified APIs, and ensures seamless integration with LLMKG
 * without requiring any core code modifications.
 */

import { EventEmitter } from 'events';
import { telemetryConfig, TelemetryConfig, TelemetryConfigManager } from './config.js';
import { telemetryRecorder, TelemetryRecorder, TelemetryBatch, PerformanceImpact } from './recorder.js';
import { mcpTelemetryProxy, MCPTelemetryProxy, TelemetryData } from './proxy.js';
import { telemetryInjector, TelemetryInjector } from './injector.js';
import { llmkgInstrumentation, LLMKGInstrumentation, OperationMetrics, PerformanceProfile } from './instrumentation.js';
import { MCPClient } from '../mcp/client.js';

export interface TelemetryManagerConfig {
  /** Enable auto-initialization */
  autoInitialize: boolean;
  
  /** Data export configuration */
  export: {
    enabled: boolean;
    format: 'json' | 'csv' | 'parquet';
    destination: 'file' | 'http' | 'websocket' | 'mcp';
    interval: number;
  };
  
  /** Real-time monitoring */
  realtime: {
    enabled: boolean;
    websocketPort?: number;
    httpPort?: number;
  };
  
  /** Performance alerting */
  alerts: {
    enabled: boolean;
    thresholds: {
      overheadPercentage: number;
      errorRate: number;
      responseTime: number;
    };
    webhooks?: string[];
  };
  
  /** Data retention */
  retention: {
    enabled: boolean;
    maxAge: number; // milliseconds
    maxSize: number; // bytes
    compressionEnabled: boolean;
  };
}

export const DEFAULT_MANAGER_CONFIG: TelemetryManagerConfig = {
  autoInitialize: true,
  export: {
    enabled: false,
    format: 'json',
    destination: 'file',
    interval: 60000, // 1 minute
  },
  realtime: {
    enabled: false,
  },
  alerts: {
    enabled: true,
    thresholds: {
      overheadPercentage: 1.0,
      errorRate: 0.05, // 5%
      responseTime: 1000, // 1 second
    },
  },
  retention: {
    enabled: true,
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    maxSize: 100 * 1024 * 1024, // 100MB
    compressionEnabled: true,
  },
};

export interface TelemetrySnapshot {
  timestamp: number;
  performance: {
    overallOverhead: number;
    componentOverheads: {
      recorder: number;
      proxy: number;
      injector: number;
      instrumentation: number;
    };
    impactMetrics: PerformanceImpact | null;
  };
  operations: {
    totalOperations: number;
    operationsByCategory: Record<string, number>;
    averageResponseTime: number;
    errorRate: number;
    topOperations: Array<{
      name: string;
      calls: number;
      averageTime: number;
    }>;
  };
  system: {
    memoryUsage: NodeJS.MemoryUsage;
    uptime: number;
    isEnabled: boolean;
    configurationHash: string;
  };
  buffers: {
    recorderBuffer: {
      size: number;
      usage: number;
    };
    instrumentationBuffer: {
      operations: number;
      avgLatency: number;
    };
  };
}

export interface TelemetryAlert {
  id: string;
  timestamp: number;
  severity: 'info' | 'warning' | 'error' | 'critical';
  category: 'performance' | 'error' | 'system' | 'configuration';
  message: string;
  details: Record<string, any>;
  resolved?: boolean;
  resolvedAt?: number;
}

/**
 * Central telemetry orchestration manager
 */
export class TelemetryManager extends EventEmitter {
  private config: TelemetryManagerConfig;
  private isInitialized = false;
  private isRunning = false;
  private components: {
    config: TelemetryConfigManager;
    recorder: TelemetryRecorder;
    proxy: MCPTelemetryProxy;
    injector: TelemetryInjector;
    instrumentation: LLMKGInstrumentation;
  };
  private exportTimer?: NodeJS.Timeout;
  private monitoringTimer?: NodeJS.Timeout;
  private retentionTimer?: NodeJS.Timeout;
  private activeAlerts: Map<string, TelemetryAlert> = new Map();
  private startTime: number;
  private lastSnapshot?: TelemetrySnapshot;

  constructor(config: Partial<TelemetryManagerConfig> = {}) {
    super();
    
    this.config = { ...DEFAULT_MANAGER_CONFIG, ...config };
    this.startTime = Date.now();
    
    // Initialize components
    this.components = {
      config: telemetryConfig,
      recorder: telemetryRecorder,
      proxy: mcpTelemetryProxy,
      injector: telemetryInjector,
      instrumentation: llmkgInstrumentation,
    };

    // Auto-initialize if configured
    if (this.config.autoInitialize) {
      process.nextTick(() => this.initialize());
    }
  }

  /**
   * Initialize the telemetry system
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      this.emit('initializing');

      // Initialize injector first (sets up environment)
      await this.components.injector.initialize();

      // Start recorder if telemetry is enabled
      if (telemetryConfig.isEnabled()) {
        this.components.recorder.startRecording();
      }

      // Setup component integrations
      this.setupComponentIntegrations();

      // Setup monitoring and export timers
      this.setupTimers();

      // Setup alert system
      this.setupAlertSystem();

      this.isInitialized = true;
      this.isRunning = true;

      this.emit('initialized');
      
      // Record initialization event
      this.components.recorder.recordLog(
        'system',
        'telemetry.manager.initialized',
        'Telemetry manager initialized successfully',
        'info',
        {
          config: this.config,
          componentsActive: this.getComponentStatus(),
        }
      );

    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Shutdown the telemetry system
   */
  async shutdown(): Promise<void> {
    if (!this.isRunning) return;

    this.emit('shutting_down');

    // Stop all timers
    this.clearTimers();

    // Flush remaining data
    await this.components.recorder.flush();

    // Stop recording
    this.components.recorder.stopRecording();

    // Cleanup injector
    this.components.injector.cleanup();

    this.isRunning = false;
    
    this.emit('shutdown');

    // Final log entry
    this.components.recorder.recordLog(
      'system',
      'telemetry.manager.shutdown',
      'Telemetry manager shut down',
      'info'
    );
  }

  /**
   * Wrap an MCP client with telemetry
   */
  wrapMCPClient(client: MCPClient): MCPClient {
    return this.components.injector.injectMCPClient(client);
  }

  /**
   * Wrap a tool function with telemetry
   */
  wrapTool(
    toolName: string,
    toolFunction: Function,
    category?: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph'
  ): Function {
    return this.components.injector.injectTool(toolName, toolFunction, category);
  }

  /**
   * Instrument an operation with LLMKG-specific monitoring
   */
  instrument<T>(
    category: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph',
    operationName: string,
    operation: () => T,
    metadata?: Record<string, any>
  ): T {
    switch (category) {
      case 'sdr':
        return this.components.instrumentation.instrumentSDROperation(operationName, operation, metadata);
      case 'cognitive':
        return this.components.instrumentation.instrumentCognitiveOperation(operationName, operation, metadata as any);
      case 'neural':
        return this.components.instrumentation.instrumentNeuralOperation(operationName, operation, metadata as any);
      case 'memory':
        return this.components.instrumentation.instrumentMemoryOperation(operationName, operation, metadata as any);
      case 'attention':
        return this.components.instrumentation.instrumentAttentionOperation(operationName, operation, metadata as any);
      case 'graph':
        return this.components.instrumentation.instrumentGraphOperation(operationName, operation, metadata as any);
    }
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
    return this.components.instrumentation.instrumentAsync(category, operationName, operation, metadata);
  }

  /**
   * Get current telemetry snapshot
   */
  getSnapshot(): TelemetrySnapshot {
    const bufferStats = this.components.recorder.getBufferStats();
    const performanceImpact = this.components.recorder.getPerformanceImpact();
    const instrumentationOverhead = this.components.instrumentation.getInstrumentationOverhead();
    const operationMetrics = this.components.instrumentation.getOperationMetrics() as Map<string, OperationMetrics>;
    
    // Calculate operation statistics
    const operations = Array.from(operationMetrics.values());
    const totalOperations = operations.reduce((sum, op) => sum + op.totalCalls, 0);
    const totalDuration = operations.reduce((sum, op) => sum + op.totalDuration, 0);
    const totalErrors = operations.reduce((sum, op) => sum + op.errorCount, 0);

    // Group operations by category
    const operationsByCategory: Record<string, number> = {};
    for (const [operationId, metrics] of operationMetrics) {
      const category = operationId.split('_')[0] || 'unknown';
      operationsByCategory[category] = (operationsByCategory[category] || 0) + metrics.totalCalls;
    }

    // Get top operations
    const topOperations = operations
      .map(op => ({
        name: Array.from(operationMetrics.entries()).find(([_, m]) => m === op)?.[0] || 'unknown',
        calls: op.totalCalls,
        averageTime: op.averageDuration,
      }))
      .sort((a, b) => b.calls - a.calls)
      .slice(0, 10);

    const snapshot: TelemetrySnapshot = {
      timestamp: Date.now(),
      performance: {
        overallOverhead: performanceImpact?.currentOverhead || 0,
        componentOverheads: {
          recorder: 0, // Would need to instrument the recorder itself
          proxy: 0,    // Would need to instrument the proxy itself
          injector: 0, // Would need to instrument the injector itself
          instrumentation: instrumentationOverhead,
        },
        impactMetrics: performanceImpact,
      },
      operations: {
        totalOperations,
        operationsByCategory,
        averageResponseTime: totalOperations > 0 ? totalDuration / totalOperations : 0,
        errorRate: totalOperations > 0 ? totalErrors / totalOperations : 0,
        topOperations,
      },
      system: {
        memoryUsage: process.memoryUsage(),
        uptime: Date.now() - this.startTime,
        isEnabled: telemetryConfig.isEnabled(),
        configurationHash: this.calculateConfigHash(),
      },
      buffers: {
        recorderBuffer: {
          size: bufferStats.eventCount,
          usage: bufferStats.bufferUsage,
        },
        instrumentationBuffer: {
          operations: operationMetrics.size,
          avgLatency: totalOperations > 0 ? totalDuration / totalOperations : 0,
        },
      },
    };

    this.lastSnapshot = snapshot;
    return snapshot;
  }

  /**
   * Get performance profile for an operation
   */
  getPerformanceProfile(operationId: string): PerformanceProfile | undefined {
    return this.components.instrumentation.getPerformanceProfile(operationId);
  }

  /**
   * Get all active alerts
   */
  getActiveAlerts(): TelemetryAlert[] {
    return Array.from(this.activeAlerts.values()).filter(alert => !alert.resolved);
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = Date.now();
      this.emit('alert_resolved', alert);
      return true;
    }
    return false;
  }

  /**
   * Export telemetry data
   */
  async exportData(): Promise<void> {
    if (!this.config.export.enabled) return;

    const snapshot = this.getSnapshot();
    const exportData = {
      snapshot,
      metrics: this.components.instrumentation.exportMetrics(),
      timestamp: Date.now(),
    };

    switch (this.config.export.destination) {
      case 'file':
        await this.exportToFile(exportData);
        break;
      case 'http':
        await this.exportToHTTP(exportData);
        break;
      case 'websocket':
        await this.exportToWebSocket(exportData);
        break;
      case 'mcp':
        await this.exportToMCP(exportData);
        break;
    }

    this.emit('data_exported', exportData);
  }

  /**
   * Update manager configuration
   */
  updateConfig(newConfig: Partial<TelemetryManagerConfig>): void {
    const oldConfig = this.config;
    this.config = { ...this.config, ...newConfig };

    // Restart timers if intervals changed
    if (oldConfig.export.interval !== this.config.export.interval ||
        oldConfig.export.enabled !== this.config.export.enabled) {
      this.setupExportTimer();
    }

    this.emit('config_updated', { oldConfig, newConfig: this.config });
  }

  /**
   * Get component status
   */
  getComponentStatus(): Record<string, boolean> {
    return {
      config: true, // Always available
      recorder: this.components.recorder !== undefined,
      proxy: this.components.proxy !== undefined,
      injector: this.components.injector !== undefined,
      instrumentation: this.components.instrumentation !== undefined,
    };
  }

  /**
   * Get system health status
   */
  getHealthStatus(): {
    healthy: boolean;
    issues: string[];
    recommendations: string[];
  } {
    const issues: string[] = [];
    const recommendations: string[] = [];
    const snapshot = this.lastSnapshot || this.getSnapshot();

    // Check performance overhead
    if (snapshot.performance.overallOverhead > telemetryConfig.getConfig().maxOverhead) {
      issues.push(`Performance overhead too high: ${snapshot.performance.overallOverhead.toFixed(2)}%`);
      recommendations.push('Reduce telemetry sampling rates or disable non-essential instrumentation');
    }

    // Check error rate
    if (snapshot.operations.errorRate > this.config.alerts.thresholds.errorRate) {
      issues.push(`High error rate: ${(snapshot.operations.errorRate * 100).toFixed(2)}%`);
      recommendations.push('Investigate failing operations and improve error handling');
    }

    // Check buffer usage
    if (snapshot.buffers.recorderBuffer.usage > 0.8) {
      issues.push('Telemetry buffer near capacity');
      recommendations.push('Increase flush frequency or buffer size');
    }

    // Check memory usage
    const memoryUsageMB = snapshot.system.memoryUsage.heapUsed / (1024 * 1024);
    if (memoryUsageMB > 500) {
      recommendations.push('Monitor memory usage - telemetry system using significant memory');
    }

    return {
      healthy: issues.length === 0,
      issues,
      recommendations,
    };
  }

  /**
   * Setup component integrations
   */
  private setupComponentIntegrations(): void {
    // Setup recorder flush callback
    this.components.recorder.onFlush(async (batch: TelemetryBatch) => {
      this.emit('batch_flushed', batch);
      
      // Check for performance issues in the batch
      this.analyzePerformanceBatch(batch);
    });

    // Setup configuration change handling
    this.components.config.onConfigChange((newConfig: TelemetryConfig) => {
      this.emit('telemetry_config_changed', newConfig);
      
      // Adjust manager behavior based on telemetry config
      if (!newConfig.enabled && this.isRunning) {
        this.components.recorder.stopRecording();
      } else if (newConfig.enabled && this.isRunning) {
        this.components.recorder.startRecording();
      }
    });
  }

  /**
   * Setup timers for periodic operations
   */
  private setupTimers(): void {
    this.setupExportTimer();
    this.setupMonitoringTimer();
    this.setupRetentionTimer();
  }

  /**
   * Setup export timer
   */
  private setupExportTimer(): void {
    if (this.exportTimer) {
      clearInterval(this.exportTimer);
    }

    if (this.config.export.enabled) {
      this.exportTimer = setInterval(() => {
        this.exportData().catch(error => {
          this.emit('error', new Error(`Export failed: ${error.message}`));
        });
      }, this.config.export.interval);
    }
  }

  /**
   * Setup monitoring timer
   */
  private setupMonitoringTimer(): void {
    this.monitoringTimer = setInterval(() => {
      const snapshot = this.getSnapshot();
      this.checkForAlerts(snapshot);
      this.emit('monitoring_tick', snapshot);
    }, 30000); // Monitor every 30 seconds
  }

  /**
   * Setup retention timer
   */
  private setupRetentionTimer(): void {
    if (this.config.retention.enabled) {
      this.retentionTimer = setInterval(() => {
        this.performRetentionCleanup();
      }, 300000); // Clean up every 5 minutes
    }
  }

  /**
   * Clear all timers
   */
  private clearTimers(): void {
    if (this.exportTimer) {
      clearInterval(this.exportTimer);
      this.exportTimer = undefined;
    }

    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = undefined;
    }

    if (this.retentionTimer) {
      clearInterval(this.retentionTimer);
      this.retentionTimer = undefined;
    }
  }

  /**
   * Setup alert system
   */
  private setupAlertSystem(): void {
    if (!this.config.alerts.enabled) return;

    // Listen for performance issues
    this.on('performance_issue', (issue: any) => {
      this.createAlert('performance', 'warning', 'Performance threshold exceeded', issue);
    });

    // Listen for errors
    this.on('error', (error: Error) => {
      this.createAlert('error', 'error', `System error: ${error.message}`, { error: error.stack });
    });
  }

  /**
   * Create and manage an alert
   */
  private createAlert(
    category: TelemetryAlert['category'],
    severity: TelemetryAlert['severity'],
    message: string,
    details: Record<string, any>
  ): void {
    const alert: TelemetryAlert = {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      severity,
      category,
      message,
      details,
    };

    this.activeAlerts.set(alert.id, alert);
    this.emit('alert_created', alert);

    // Send to webhooks if configured
    if (this.config.alerts.webhooks) {
      this.sendAlertToWebhooks(alert);
    }
  }

  /**
   * Analyze performance batch for issues
   */
  private analyzePerformanceBatch(batch: TelemetryBatch): void {
    // Look for performance anomalies in the batch
    const performanceEvents = batch.events.filter(e => e.performance && e.performance.duration);
    
    if (performanceEvents.length > 0) {
      const avgDuration = performanceEvents.reduce((sum, e) => sum + (e.performance?.duration || 0), 0) / performanceEvents.length;
      
      if (avgDuration > this.config.alerts.thresholds.responseTime) {
        this.emit('performance_issue', {
          batchId: batch.batchId,
          averageDuration: avgDuration,
          threshold: this.config.alerts.thresholds.responseTime,
        });
      }
    }
  }

  /**
   * Check snapshot for alerts
   */
  private checkForAlerts(snapshot: TelemetrySnapshot): void {
    // Check overhead threshold
    if (snapshot.performance.overallOverhead > this.config.alerts.thresholds.overheadPercentage) {
      this.createAlert(
        'performance',
        'warning',
        `Overhead threshold exceeded: ${snapshot.performance.overallOverhead.toFixed(2)}%`,
        { overhead: snapshot.performance.overallOverhead, threshold: this.config.alerts.thresholds.overheadPercentage }
      );
    }

    // Check error rate threshold
    if (snapshot.operations.errorRate > this.config.alerts.thresholds.errorRate) {
      this.createAlert(
        'error',
        'warning',
        `Error rate threshold exceeded: ${(snapshot.operations.errorRate * 100).toFixed(2)}%`,
        { errorRate: snapshot.operations.errorRate, threshold: this.config.alerts.thresholds.errorRate }
      );
    }
  }

  /**
   * Send alert to configured webhooks
   */
  private async sendAlertToWebhooks(alert: TelemetryAlert): Promise<void> {
    if (!this.config.alerts.webhooks) return;

    const promises = this.config.alerts.webhooks.map(async (webhook) => {
      try {
        const response = await fetch(webhook, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(alert),
        });

        if (!response.ok) {
          console.warn(`Failed to send alert to webhook ${webhook}: ${response.status}`);
        }
      } catch (error) {
        console.warn(`Error sending alert to webhook ${webhook}:`, error);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Perform retention cleanup
   */
  private performRetentionCleanup(): void {
    // This would clean up old data based on retention policy
    // Implementation depends on how data is stored
    this.emit('retention_cleanup');
  }

  /**
   * Calculate configuration hash for change detection
   */
  private calculateConfigHash(): string {
    const configString = JSON.stringify({
      telemetryConfig: telemetryConfig.getConfig(),
      managerConfig: this.config,
    });
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < configString.length; i++) {
      const char = configString.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return hash.toString(36);
  }

  /**
   * Export data to file
   */
  private async exportToFile(data: any): Promise<void> {
    // Implementation would depend on Node.js fs module
    console.log('Exporting to file:', data);
  }

  /**
   * Export data via HTTP
   */
  private async exportToHTTP(data: any): Promise<void> {
    // Implementation would use fetch or http client
    console.log('Exporting via HTTP:', data);
  }

  /**
   * Export data via WebSocket
   */
  private async exportToWebSocket(data: any): Promise<void> {
    // Implementation would use WebSocket
    console.log('Exporting via WebSocket:', data);
  }

  /**
   * Export data to MCP server
   */
  private async exportToMCP(data: any): Promise<void> {
    // Implementation would use MCP client to send data
    console.log('Exporting to MCP:', data);
  }
}

// Global manager instance
export const telemetryManager = new TelemetryManager();