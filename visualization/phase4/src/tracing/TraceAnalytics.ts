/**
 * TraceAnalytics.ts
 * 
 * Advanced analytics system for MCP request tracing and performance monitoring.
 * Provides comprehensive insights into request flow, performance patterns, and system health.
 */

import { MCPRequest, TraceEvent } from './MCPRequestTracer';

export interface PerformanceMetric {
  id: string;
  requestId: string;
  timestamp: number;
  metric: 'latency' | 'throughput' | 'error_rate' | 'cognitive_load' | 'path_efficiency';
  value: number;
  unit: string;
  context?: any;
}

export interface PathAnalysis {
  requestId: string;
  path: string[];
  totalTime: number;
  segmentTimes: { [segment: string]: number };
  efficiency: number; // 0-1 score
  bottlenecks: string[];
  optimizations: string[];
}

export interface SystemHealth {
  timestamp: number;
  overallHealth: number; // 0-1 score
  metrics: {
    avgLatency: number;
    errorRate: number;
    throughput: number;
    cognitiveLoad: number;
    pathEfficiency: number;
  };
  alerts: Alert[];
}

export interface Alert {
  id: string;
  timestamp: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: 'performance' | 'error' | 'bottleneck' | 'anomaly';
  message: string;
  context: any;
  acknowledged: boolean;
}

export interface AnalyticsConfig {
  metricsRetentionPeriod: number; // milliseconds
  healthCheckInterval: number; // milliseconds
  performanceThresholds: {
    latencyWarning: number;
    latencyCritical: number;
    errorRateWarning: number;
    errorRateCritical: number;
    throughputMin: number;
  };
  pathOptimization: {
    enabled: boolean;
    maxPathLength: number;
    efficiencyThreshold: number;
  };
}

export class TraceAnalytics {
  private metrics: Map<string, PerformanceMetric[]> = new Map();
  private pathAnalyses: Map<string, PathAnalysis> = new Map();
  private systemHealthHistory: SystemHealth[] = [];
  private alerts: Map<string, Alert> = new Map();
  private requestStartTimes: Map<string, number> = new Map();
  private cognitiveActivations: Map<string, TraceEvent[]> = new Map();
  private healthCheckInterval: number | null = null;
  private config: AnalyticsConfig;

  constructor(config?: Partial<AnalyticsConfig>) {
    this.config = {
      metricsRetentionPeriod: 24 * 60 * 60 * 1000, // 24 hours
      healthCheckInterval: 30000, // 30 seconds
      performanceThresholds: {
        latencyWarning: 1000, // 1 second
        latencyCritical: 5000, // 5 seconds
        errorRateWarning: 0.05, // 5%
        errorRateCritical: 0.15, // 15%
        throughputMin: 1 // requests per second
      },
      pathOptimization: {
        enabled: true,
        maxPathLength: 10,
        efficiencyThreshold: 0.7
      },
      ...config
    };

    this.startHealthChecks();
  }

  /**
   * Process trace event for analytics
   */
  public processTraceEvent(event: TraceEvent): void {
    switch (event.type) {
      case 'request':
        this.handleRequestEvent(event);
        break;
      case 'cognitive_activation':
        this.handleCognitiveEvent(event);
        break;
      case 'response':
        this.handleResponseEvent(event);
        break;
      case 'error':
        this.handleErrorEvent(event);
        break;
      case 'performance':
        this.handlePerformanceEvent(event);
        break;
    }

    this.cleanupOldMetrics();
  }

  /**
   * Handle request initiation analytics
   */
  private handleRequestEvent(event: TraceEvent): void {
    const request = event.data as MCPRequest;
    this.requestStartTimes.set(request.id, event.timestamp);

    // Record throughput metric
    this.addMetric({
      id: `throughput_${event.id}`,
      requestId: request.id,
      timestamp: event.timestamp,
      metric: 'throughput',
      value: 1,
      unit: 'requests',
      context: { method: request.method, source: request.source }
    });

    // Initialize cognitive activations tracking
    this.cognitiveActivations.set(request.id, []);
  }

  /**
   * Handle cognitive activation analytics
   */
  private handleCognitiveEvent(event: TraceEvent): void {
    const activations = this.cognitiveActivations.get(event.requestId) || [];
    activations.push(event);
    this.cognitiveActivations.set(event.requestId, activations);

    // Calculate cognitive load
    const cognitiveLoad = this.calculateCognitiveLoad(event.requestId);
    this.addMetric({
      id: `cognitive_load_${event.id}`,
      requestId: event.requestId,
      timestamp: event.timestamp,
      metric: 'cognitive_load',
      value: cognitiveLoad,
      unit: 'normalized',
      context: { pattern: event.data.pattern, totalActivations: activations.length }
    });
  }

  /**
   * Handle response completion analytics
   */
  private handleResponseEvent(event: TraceEvent): void {
    const startTime = this.requestStartTimes.get(event.requestId);
    if (!startTime) return;

    const latency = event.timestamp - startTime;
    const isError = !!event.data.error;

    // Record latency metric
    this.addMetric({
      id: `latency_${event.id}`,
      requestId: event.requestId,
      timestamp: event.timestamp,
      metric: 'latency',
      value: latency,
      unit: 'milliseconds',
      context: { success: !isError, method: event.data.method }
    });

    // Record error rate metric
    this.addMetric({
      id: `error_rate_${event.id}`,
      requestId: event.requestId,
      timestamp: event.timestamp,
      metric: 'error_rate',
      value: isError ? 1 : 0,
      unit: 'boolean',
      context: { error: event.data.error }
    });

    // Perform path analysis
    this.performPathAnalysis(event.requestId, event.timestamp);

    // Check performance thresholds
    this.checkPerformanceThresholds(latency, isError);

    // Cleanup
    this.requestStartTimes.delete(event.requestId);
  }

  /**
   * Handle error event analytics
   */
  private handleErrorEvent(event: TraceEvent): void {
    this.addMetric({
      id: `error_${event.id}`,
      requestId: event.requestId,
      timestamp: event.timestamp,
      metric: 'error_rate',
      value: 1,
      unit: 'count',
      context: { error: event.data.error, phase: event.phase }
    });

    // Create alert for error
    this.createAlert({
      id: `error_alert_${event.id}`,
      timestamp: event.timestamp,
      severity: 'medium',
      type: 'error',
      message: `Request ${event.requestId} failed: ${event.data.error}`,
      context: event.data,
      acknowledged: false
    });
  }

  /**
   * Handle performance event analytics
   */
  private handlePerformanceEvent(event: TraceEvent): void {
    const perfData = event.data;
    
    Object.entries(perfData).forEach(([key, value]) => {
      if (typeof value === 'number') {
        this.addMetric({
          id: `perf_${key}_${event.id}`,
          requestId: event.requestId,
          timestamp: event.timestamp,
          metric: 'performance' as any,
          value,
          unit: this.getMetricUnit(key),
          context: { metric: key, ...perfData }
        });
      }
    });
  }

  /**
   * Add performance metric
   */
  private addMetric(metric: PerformanceMetric): void {
    const requestMetrics = this.metrics.get(metric.requestId) || [];
    requestMetrics.push(metric);
    this.metrics.set(metric.requestId, requestMetrics);
  }

  /**
   * Calculate cognitive load for a request
   */
  private calculateCognitiveLoad(requestId: string): number {
    const activations = this.cognitiveActivations.get(requestId) || [];
    const uniquePatterns = new Set(activations.map(a => a.data.pattern)).size;
    const totalActivations = activations.length;
    
    // Normalize cognitive load (0-1 scale)
    const rawLoad = (uniquePatterns * 0.3 + totalActivations * 0.1);
    return Math.min(rawLoad / 5, 1); // Cap at 1.0
  }

  /**
   * Perform path analysis for a completed request
   */
  private performPathAnalysis(requestId: string, completionTime: number): void {
    const activations = this.cognitiveActivations.get(requestId) || [];
    const startTime = this.requestStartTimes.get(requestId) || completionTime;
    
    if (activations.length === 0) return;

    const path = activations.map(a => a.data.pattern);
    const totalTime = completionTime - startTime;
    const segmentTimes: { [segment: string]: number } = {};

    // Calculate segment times
    for (let i = 0; i < activations.length; i++) {
      const current = activations[i];
      const next = activations[i + 1];
      const segmentTime = next ? next.timestamp - current.timestamp : completionTime - current.timestamp;
      segmentTimes[current.data.pattern] = (segmentTimes[current.data.pattern] || 0) + segmentTime;
    }

    // Calculate efficiency (inverse of path length and time)
    const efficiency = this.calculatePathEfficiency(path, totalTime);
    
    // Identify bottlenecks (segments taking > 30% of total time)
    const bottlenecks = Object.entries(segmentTimes)
      .filter(([_, time]) => time > totalTime * 0.3)
      .map(([segment, _]) => segment);

    // Generate optimization suggestions
    const optimizations = this.generateOptimizations(path, segmentTimes, totalTime);

    const analysis: PathAnalysis = {
      requestId,
      path,
      totalTime,
      segmentTimes,
      efficiency,
      bottlenecks,
      optimizations
    };

    this.pathAnalyses.set(requestId, analysis);

    // Record path efficiency metric
    this.addMetric({
      id: `path_efficiency_${requestId}`,
      requestId,
      timestamp: completionTime,
      metric: 'path_efficiency',
      value: efficiency,
      unit: 'normalized',
      context: { pathLength: path.length, bottlenecks: bottlenecks.length }
    });
  }

  /**
   * Calculate path efficiency score
   */
  private calculatePathEfficiency(path: string[], totalTime: number): number {
    const optimalPathLength = 3; // Assumed optimal path length
    const optimalTime = 500; // Assumed optimal time (ms)
    
    const lengthPenalty = Math.min(path.length / optimalPathLength, 2) - 1;
    const timePenalty = Math.min(totalTime / optimalTime, 3) - 1;
    
    const efficiency = 1 - (lengthPenalty * 0.3 + timePenalty * 0.7);
    return Math.max(0, Math.min(1, efficiency));
  }

  /**
   * Generate path optimization suggestions
   */
  private generateOptimizations(path: string[], segmentTimes: { [segment: string]: number }, totalTime: number): string[] {
    const optimizations: string[] = [];

    // Check for redundant patterns
    const pathCounts = path.reduce((acc, pattern) => {
      acc[pattern] = (acc[pattern] || 0) + 1;
      return acc;
    }, {} as { [pattern: string]: number });

    Object.entries(pathCounts).forEach(([pattern, count]) => {
      if (count > 2) {
        optimizations.push(`Reduce redundant ${pattern} activations (${count} occurrences)`);
      }
    });

    // Check for slow segments
    Object.entries(segmentTimes).forEach(([segment, time]) => {
      if (time > totalTime * 0.4) {
        optimizations.push(`Optimize ${segment} processing (${time}ms, ${Math.round(time/totalTime*100)}% of total)`);
      }
    });

    // Check path length
    if (path.length > this.config.pathOptimization.maxPathLength) {
      optimizations.push(`Simplify cognitive path (${path.length} steps, recommended: ${this.config.pathOptimization.maxPathLength})`);
    }

    return optimizations;
  }

  /**
   * Check performance thresholds and create alerts
   */
  private checkPerformanceThresholds(latency: number, isError: boolean): void {
    // Latency alerts
    if (latency > this.config.performanceThresholds.latencyCritical) {
      this.createAlert({
        id: `latency_critical_${Date.now()}`,
        timestamp: Date.now(),
        severity: 'critical',
        type: 'performance',
        message: `Critical latency detected: ${latency}ms`,
        context: { latency, threshold: this.config.performanceThresholds.latencyCritical },
        acknowledged: false
      });
    } else if (latency > this.config.performanceThresholds.latencyWarning) {
      this.createAlert({
        id: `latency_warning_${Date.now()}`,
        timestamp: Date.now(),
        severity: 'medium',
        type: 'performance',
        message: `High latency detected: ${latency}ms`,
        context: { latency, threshold: this.config.performanceThresholds.latencyWarning },
        acknowledged: false
      });
    }
  }

  /**
   * Create system alert
   */
  private createAlert(alert: Alert): void {
    this.alerts.set(alert.id, alert);
    console.warn(`TraceAnalytics Alert [${alert.severity}]: ${alert.message}`, alert.context);
  }

  /**
   * Start periodic health checks
   */
  private startHealthChecks(): void {
    this.healthCheckInterval = window.setInterval(() => {
      this.performHealthCheck();
    }, this.config.healthCheckInterval);
  }

  /**
   * Perform system health analysis
   */
  private performHealthCheck(): void {
    const now = Date.now();
    const windowStart = now - this.config.healthCheckInterval * 2; // 2x interval window

    const recentMetrics = this.getMetricsInWindow(windowStart, now);
    const health = this.calculateSystemHealth(recentMetrics, now);

    this.systemHealthHistory.push(health);

    // Keep only last 100 health checks
    if (this.systemHealthHistory.length > 100) {
      this.systemHealthHistory = this.systemHealthHistory.slice(-100);
    }

    // Check for health-based alerts
    this.checkHealthAlerts(health);
  }

  /**
   * Get metrics within time window
   */
  private getMetricsInWindow(start: number, end: number): PerformanceMetric[] {
    const metrics: PerformanceMetric[] = [];
    
    this.metrics.forEach(requestMetrics => {
      requestMetrics.forEach(metric => {
        if (metric.timestamp >= start && metric.timestamp <= end) {
          metrics.push(metric);
        }
      });
    });

    return metrics;
  }

  /**
   * Calculate overall system health
   */
  private calculateSystemHealth(metrics: PerformanceMetric[], timestamp: number): SystemHealth {
    const latencyMetrics = metrics.filter(m => m.metric === 'latency');
    const errorMetrics = metrics.filter(m => m.metric === 'error_rate');
    const throughputMetrics = metrics.filter(m => m.metric === 'throughput');
    const cognitiveLoadMetrics = metrics.filter(m => m.metric === 'cognitive_load');
    const efficiencyMetrics = metrics.filter(m => m.metric === 'path_efficiency');

    const avgLatency = latencyMetrics.length > 0 ? 
      latencyMetrics.reduce((sum, m) => sum + m.value, 0) / latencyMetrics.length : 0;
    
    const errorRate = errorMetrics.length > 0 ?
      errorMetrics.reduce((sum, m) => sum + m.value, 0) / errorMetrics.length : 0;
    
    const throughput = throughputMetrics.length;
    
    const avgCognitiveLoad = cognitiveLoadMetrics.length > 0 ?
      cognitiveLoadMetrics.reduce((sum, m) => sum + m.value, 0) / cognitiveLoadMetrics.length : 0;
    
    const avgPathEfficiency = efficiencyMetrics.length > 0 ?
      efficiencyMetrics.reduce((sum, m) => sum + m.value, 0) / efficiencyMetrics.length : 1;

    // Calculate overall health score (0-1)
    const latencyScore = Math.max(0, 1 - avgLatency / this.config.performanceThresholds.latencyCritical);
    const errorScore = Math.max(0, 1 - errorRate / this.config.performanceThresholds.errorRateCritical);
    const throughputScore = Math.min(1, throughput / this.config.performanceThresholds.throughputMin);
    const cognitiveScore = Math.max(0, 1 - avgCognitiveLoad);
    const efficiencyScore = avgPathEfficiency;

    const overallHealth = (latencyScore + errorScore + throughputScore + cognitiveScore + efficiencyScore) / 5;

    return {
      timestamp,
      overallHealth,
      metrics: {
        avgLatency,
        errorRate,
        throughput,
        cognitiveLoad: avgCognitiveLoad,
        pathEfficiency: avgPathEfficiency
      },
      alerts: Array.from(this.alerts.values()).filter(a => !a.acknowledged)
    };
  }

  /**
   * Check for health-based alerts
   */
  private checkHealthAlerts(health: SystemHealth): void {
    if (health.overallHealth < 0.3) {
      this.createAlert({
        id: `system_health_critical_${Date.now()}`,
        timestamp: health.timestamp,
        severity: 'critical',
        type: 'performance',
        message: `System health critical: ${Math.round(health.overallHealth * 100)}%`,
        context: health.metrics,
        acknowledged: false
      });
    } else if (health.overallHealth < 0.6) {
      this.createAlert({
        id: `system_health_warning_${Date.now()}`,
        timestamp: health.timestamp,
        severity: 'medium',
        type: 'performance',
        message: `System health degraded: ${Math.round(health.overallHealth * 100)}%`,
        context: health.metrics,
        acknowledged: false
      });
    }
  }

  /**
   * Get metric unit based on key
   */
  private getMetricUnit(key: string): string {
    const unitMap: { [key: string]: string } = {
      'latency': 'ms',
      'processingTime': 'ms',
      'throughput': 'requests/sec',
      'errorRate': 'percentage',
      'cognitiveLoad': 'normalized',
      'pathEfficiency': 'normalized',
      'memoryUsage': 'bytes',
      'cpuUsage': 'percentage'
    };

    return unitMap[key] || 'value';
  }

  /**
   * Clean up old metrics
   */
  private cleanupOldMetrics(): void {
    const cutoffTime = Date.now() - this.config.metricsRetentionPeriod;
    
    this.metrics.forEach((requestMetrics, requestId) => {
      const filteredMetrics = requestMetrics.filter(metric => metric.timestamp > cutoffTime);
      if (filteredMetrics.length === 0) {
        this.metrics.delete(requestId);
      } else {
        this.metrics.set(requestId, filteredMetrics);
      }
    });

    // Clean up old path analyses
    Array.from(this.pathAnalyses.keys()).forEach(requestId => {
      const analysis = this.pathAnalyses.get(requestId);
      if (analysis && analysis.totalTime < cutoffTime) {
        this.pathAnalyses.delete(requestId);
      }
    });

    // Clean up old alerts
    Array.from(this.alerts.keys()).forEach(alertId => {
      const alert = this.alerts.get(alertId);
      if (alert && alert.timestamp < cutoffTime) {
        this.alerts.delete(alertId);
      }
    });
  }

  /**
   * Public API methods
   */

  /**
   * Get current system health
   */
  public getCurrentHealth(): SystemHealth | null {
    return this.systemHealthHistory[this.systemHealthHistory.length - 1] || null;
  }

  /**
   * Get system health history
   */
  public getHealthHistory(): SystemHealth[] {
    return [...this.systemHealthHistory];
  }

  /**
   * Get performance metrics for a request
   */
  public getRequestMetrics(requestId: string): PerformanceMetric[] {
    return this.metrics.get(requestId) || [];
  }

  /**
   * Get all performance metrics
   */
  public getAllMetrics(): PerformanceMetric[] {
    const allMetrics: PerformanceMetric[] = [];
    this.metrics.forEach(requestMetrics => {
      allMetrics.push(...requestMetrics);
    });
    return allMetrics;
  }

  /**
   * Get path analysis for a request
   */
  public getPathAnalysis(requestId: string): PathAnalysis | undefined {
    return this.pathAnalyses.get(requestId);
  }

  /**
   * Get all path analyses
   */
  public getAllPathAnalyses(): PathAnalysis[] {
    return Array.from(this.pathAnalyses.values());
  }

  /**
   * Get active alerts
   */
  public getActiveAlerts(): Alert[] {
    return Array.from(this.alerts.values()).filter(a => !a.acknowledged);
  }

  /**
   * Get all alerts
   */
  public getAllAlerts(): Alert[] {
    return Array.from(this.alerts.values());
  }

  /**
   * Acknowledge alert
   */
  public acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.get(alertId);
    if (alert) {
      alert.acknowledged = true;
      return true;
    }
    return false;
  }

  /**
   * Get aggregated metrics by type
   */
  public getAggregatedMetrics(metricType: PerformanceMetric['metric'], timeWindow?: number): {
    avg: number;
    min: number;
    max: number;
    count: number;
  } {
    const cutoff = timeWindow ? Date.now() - timeWindow : 0;
    const metrics = this.getAllMetrics()
      .filter(m => m.metric === metricType && m.timestamp > cutoff);

    if (metrics.length === 0) {
      return { avg: 0, min: 0, max: 0, count: 0 };
    }

    const values = metrics.map(m => m.value);
    return {
      avg: values.reduce((sum, val) => sum + val, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    };
  }

  /**
   * Get top bottlenecks
   */
  public getTopBottlenecks(limit = 5): Array<{ segment: string; occurrences: number; avgTime: number }> {
    const bottleneckCounts: { [segment: string]: { count: number; totalTime: number } } = {};

    this.pathAnalyses.forEach(analysis => {
      analysis.bottlenecks.forEach(bottleneck => {
        if (!bottleneckCounts[bottleneck]) {
          bottleneckCounts[bottleneck] = { count: 0, totalTime: 0 };
        }
        bottleneckCounts[bottleneck].count++;
        bottleneckCounts[bottleneck].totalTime += analysis.segmentTimes[bottleneck] || 0;
      });
    });

    return Object.entries(bottleneckCounts)
      .map(([segment, data]) => ({
        segment,
        occurrences: data.count,
        avgTime: data.totalTime / data.count
      }))
      .sort((a, b) => b.occurrences - a.occurrences)
      .slice(0, limit);
  }

  /**
   * Export analytics data
   */
  public exportAnalyticsData(): {
    metrics: PerformanceMetric[];
    pathAnalyses: PathAnalysis[];
    healthHistory: SystemHealth[];
    alerts: Alert[];
    summary: any;
  } {
    return {
      metrics: this.getAllMetrics(),
      pathAnalyses: this.getAllPathAnalyses(),
      healthHistory: this.getHealthHistory(),
      alerts: this.getAllAlerts(),
      summary: {
        totalRequests: this.metrics.size,
        totalAlerts: this.alerts.size,
        currentHealth: this.getCurrentHealth()?.overallHealth || 0,
        topBottlenecks: this.getTopBottlenecks()
      }
    };
  }

  /**
   * Dispose and cleanup
   */
  public dispose(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    this.metrics.clear();
    this.pathAnalyses.clear();
    this.systemHealthHistory.length = 0;
    this.alerts.clear();
    this.requestStartTimes.clear();
    this.cognitiveActivations.clear();
  }
}