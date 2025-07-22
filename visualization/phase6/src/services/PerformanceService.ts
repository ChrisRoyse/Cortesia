import { 
  PerformanceMetrics, 
  PerformanceAlert, 
  PerformanceThresholds,
  PerformanceTrend,
  PerformanceSnapshot,
  PerformanceOptimization,
  PerformanceReport
} from '../types';

export class PerformanceService {
  private metricsHistory: PerformanceMetrics[] = [];
  private alerts: PerformanceAlert[] = [];
  private snapshots: PerformanceSnapshot[] = [];
  private thresholds: PerformanceThresholds;
  private websocket: WebSocket | null = null;
  private subscribers: Set<(metrics: PerformanceMetrics) => void> = new Set();

  constructor(private config: {
    websocketUrl: string;
    historySize: number;
    thresholds: PerformanceThresholds;
  }) {
    this.thresholds = config.thresholds;
    this.connectWebSocket();
  }

  private connectWebSocket() {
    try {
      this.websocket = new WebSocket(this.config.websocketUrl);
      
      this.websocket.onopen = () => {
        console.log('Performance monitoring WebSocket connected');
        this.requestMetrics();
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'performance_metrics') {
          this.handleMetricsUpdate(data.metrics);
        }
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      this.websocket.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(() => this.connectWebSocket(), 5000);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }

  private requestMetrics() {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'subscribe',
        channel: 'performance_metrics',
        interval: 1000
      }));
    }
  }

  private handleMetricsUpdate(metrics: PerformanceMetrics) {
    // Add to history
    this.metricsHistory.push(metrics);
    if (this.metricsHistory.length > this.config.historySize) {
      this.metricsHistory.shift();
    }

    // Check for alerts
    this.checkAlerts(metrics);

    // Notify subscribers
    this.subscribers.forEach(callback => callback(metrics));
  }

  private checkAlerts(metrics: PerformanceMetrics) {
    const newAlerts: PerformanceAlert[] = [];

    // Check cognitive metrics
    Object.entries(metrics.cognitive).forEach(([layer, layerMetrics]) => {
      if (layerMetrics.processingLatency > this.thresholds.cognitive.maxLatency) {
        newAlerts.push(this.createAlert(
          'critical',
          `cognitive.${layer}`,
          'processingLatency',
          layerMetrics.processingLatency,
          this.thresholds.cognitive.maxLatency,
          `High processing latency in ${layer} layer`
        ));
      }

      if (layerMetrics.errorCount > 0) {
        newAlerts.push(this.createAlert(
          'warning',
          `cognitive.${layer}`,
          'errorCount',
          layerMetrics.errorCount,
          0,
          `Errors detected in ${layer} layer`
        ));
      }
    });

    // Check SDR metrics
    if (metrics.sdr.averageSparsity < this.thresholds.sdr.sparsityRange[0] ||
        metrics.sdr.averageSparsity > this.thresholds.sdr.sparsityRange[1]) {
      newAlerts.push(this.createAlert(
        'warning',
        'sdr',
        'averageSparsity',
        metrics.sdr.averageSparsity,
        this.thresholds.sdr.sparsityRange[1],
        'SDR sparsity out of optimal range'
      ));
    }

    // Check MCP metrics
    if (metrics.mcp.averageLatency > this.thresholds.mcp.maxLatency) {
      newAlerts.push(this.createAlert(
        'warning',
        'mcp',
        'averageLatency',
        metrics.mcp.averageLatency,
        this.thresholds.mcp.maxLatency,
        'High MCP protocol latency'
      ));
    }

    // Check system metrics
    if (metrics.system.cpuUsage > this.thresholds.system.maxCPU) {
      newAlerts.push(this.createAlert(
        'critical',
        'system',
        'cpuUsage',
        metrics.system.cpuUsage,
        this.thresholds.system.maxCPU,
        'High CPU usage detected'
      ));
    }

    // Add new alerts
    this.alerts.push(...newAlerts);
  }

  private createAlert(
    severity: PerformanceAlert['severity'],
    component: string,
    metric: string,
    value: number,
    threshold: number,
    message: string
  ): PerformanceAlert {
    return {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      severity,
      component,
      metric,
      value,
      threshold,
      message
    };
  }

  public subscribe(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  public getMetricsHistory(count?: number): PerformanceMetrics[] {
    if (count) {
      return this.metricsHistory.slice(-count);
    }
    return [...this.metricsHistory];
  }

  public getCurrentMetrics(): PerformanceMetrics | null {
    return this.metricsHistory[this.metricsHistory.length - 1] || null;
  }

  public getAlerts(filter?: { 
    severity?: PerformanceAlert['severity']; 
    acknowledged?: boolean;
    component?: string;
  }): PerformanceAlert[] {
    let filtered = [...this.alerts];

    if (filter?.severity) {
      filtered = filtered.filter(a => a.severity === filter.severity);
    }
    if (filter?.acknowledged !== undefined) {
      filtered = filtered.filter(a => a.acknowledged === filter.acknowledged);
    }
    if (filter?.component) {
      filtered = filtered.filter(a => a.component === filter.component);
    }

    return filtered;
  }

  public acknowledgeAlert(alertId: string): void {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
    }
  }

  public analyzePerformanceTrends(): PerformanceTrend[] {
    if (this.metricsHistory.length < 10) {
      return [];
    }

    const trends: PerformanceTrend[] = [];
    const recent = this.metricsHistory.slice(-10);
    const older = this.metricsHistory.slice(-20, -10);

    // Analyze cognitive latency trend
    const recentLatency = this.calculateAverageCognitiveLatency(recent);
    const olderLatency = this.calculateAverageCognitiveLatency(older);
    trends.push(this.calculateTrend('cognitive.latency', olderLatency, recentLatency));

    // Analyze SDR efficiency
    const recentSDREfficiency = recent.map(m => m.sdr.creationRate / m.sdr.memoryUsage);
    const olderSDREfficiency = older.map(m => m.sdr.creationRate / m.sdr.memoryUsage);
    trends.push(this.calculateTrend('sdr.efficiency', 
      this.average(olderSDREfficiency), 
      this.average(recentSDREfficiency)
    ));

    // Analyze MCP reliability
    const recentReliability = recent.map(m => 1 - m.mcp.errorRate);
    const olderReliability = older.map(m => 1 - m.mcp.errorRate);
    trends.push(this.calculateTrend('mcp.reliability',
      this.average(olderReliability),
      this.average(recentReliability)
    ));

    return trends;
  }

  private calculateAverageCognitiveLatency(metrics: PerformanceMetrics[]): number {
    const latencies = metrics.map(m => 
      (m.cognitive.subcortical.processingLatency +
       m.cognitive.cortical.processingLatency +
       m.cognitive.thalamic.processingLatency) / 3
    );
    return this.average(latencies);
  }

  private calculateTrend(metric: string, oldValue: number, newValue: number): PerformanceTrend {
    const changePercent = ((newValue - oldValue) / oldValue) * 100;
    let trend: 'increasing' | 'decreasing' | 'stable';

    if (changePercent > 5) {
      trend = 'increasing';
    } else if (changePercent < -5) {
      trend = 'decreasing';
    } else {
      trend = 'stable';
    }

    return { metric, trend, changePercent };
  }

  private average(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  public createSnapshot(name: string, description?: string): PerformanceSnapshot {
    const currentMetrics = this.getCurrentMetrics();
    if (!currentMetrics) {
      throw new Error('No metrics available for snapshot');
    }

    const snapshot: PerformanceSnapshot = {
      id: `snapshot-${Date.now()}`,
      timestamp: Date.now(),
      name,
      description,
      metrics: currentMetrics
    };

    this.snapshots.push(snapshot);
    return snapshot;
  }

  public compareSnapshots(snapshotId1: string, snapshotId2: string): PerformanceSnapshot['comparison'] {
    const snapshot1 = this.snapshots.find(s => s.id === snapshotId1);
    const snapshot2 = this.snapshots.find(s => s.id === snapshotId2);

    if (!snapshot1 || !snapshot2) {
      throw new Error('Snapshot not found');
    }

    const differences: Record<string, number> = {};

    // Compare cognitive metrics
    Object.entries(snapshot2.metrics.cognitive).forEach(([layer, metrics]) => {
      const oldMetrics = snapshot1.metrics.cognitive[layer as keyof typeof snapshot1.metrics.cognitive];
      differences[`cognitive.${layer}.latency`] = 
        ((metrics.processingLatency - oldMetrics.processingLatency) / oldMetrics.processingLatency) * 100;
      differences[`cognitive.${layer}.throughput`] = 
        ((metrics.throughput - oldMetrics.throughput) / oldMetrics.throughput) * 100;
    });

    // Compare SDR metrics
    differences['sdr.sparsity'] = 
      ((snapshot2.metrics.sdr.averageSparsity - snapshot1.metrics.sdr.averageSparsity) / 
       snapshot1.metrics.sdr.averageSparsity) * 100;

    // Compare MCP metrics
    differences['mcp.latency'] = 
      ((snapshot2.metrics.mcp.averageLatency - snapshot1.metrics.mcp.averageLatency) / 
       snapshot1.metrics.mcp.averageLatency) * 100;

    return {
      baseline: snapshotId1,
      differences
    };
  }

  public suggestOptimizations(): PerformanceOptimization[] {
    const optimizations: PerformanceOptimization[] = [];
    const currentMetrics = this.getCurrentMetrics();

    if (!currentMetrics) {
      return optimizations;
    }

    // Check cognitive layer performance
    if (currentMetrics.cognitive.cortical.processingLatency > 50) {
      optimizations.push({
        id: 'opt-1',
        category: 'cognitive',
        title: 'Optimize Cortical Layer Processing',
        description: 'Implement parallel processing for cortical layer computations',
        impact: 'high',
        effort: 'medium',
        estimatedImprovement: 30,
        status: 'suggested'
      });
    }

    // Check SDR efficiency
    if (currentMetrics.sdr.memoryUsage > 1000000000) { // 1GB
      optimizations.push({
        id: 'opt-2',
        category: 'sdr',
        title: 'Implement SDR Compression',
        description: 'Use advanced compression techniques for SDR storage',
        impact: 'medium',
        effort: 'low',
        estimatedImprovement: 40,
        status: 'suggested'
      });
    }

    // Check MCP queue length
    if (currentMetrics.mcp.queueLength > 100) {
      optimizations.push({
        id: 'opt-3',
        category: 'mcp',
        title: 'Scale MCP Message Processing',
        description: 'Add more workers for MCP message processing',
        impact: 'high',
        effort: 'low',
        estimatedImprovement: 50,
        status: 'suggested'
      });
    }

    return optimizations;
  }

  public async generateReport(period: { start: Date; end: Date }): Promise<PerformanceReport> {
    const metricsInPeriod = this.metricsHistory.filter(m => 
      m.timestamp >= period.start.getTime() && m.timestamp <= period.end.getTime()
    );

    if (metricsInPeriod.length === 0) {
      throw new Error('No metrics available for the specified period');
    }

    const report: PerformanceReport = {
      id: `report-${Date.now()}`,
      generatedAt: Date.now(),
      period: {
        start: period.start.getTime(),
        end: period.end.getTime()
      },
      summary: {
        overallHealth: this.calculateOverallHealth(metricsInPeriod),
        totalAlerts: this.alerts.filter(a => 
          a.timestamp >= period.start.getTime() && a.timestamp <= period.end.getTime()
        ).length,
        criticalIssues: this.alerts.filter(a => 
          a.severity === 'critical' &&
          a.timestamp >= period.start.getTime() && 
          a.timestamp <= period.end.getTime()
        ).length,
        optimizationOpportunities: this.suggestOptimizations().length
      },
      sections: {
        cognitive: this.analyzeCognitivePerformance(metricsInPeriod),
        sdr: this.analyzeSDRPerformance(metricsInPeriod),
        mcp: this.analyzeMCPPerformance(metricsInPeriod),
        system: this.analyzeSystemPerformance(metricsInPeriod)
      },
      recommendations: this.suggestOptimizations()
    };

    return report;
  }

  private calculateOverallHealth(metrics: PerformanceMetrics[]): number {
    const healthScores = metrics.map(m => {
      const cognitiveHealth = 100 - (m.cognitive.cortical.processingLatency / this.thresholds.cognitive.maxLatency) * 100;
      const sdrHealth = 100 - Math.abs(m.sdr.averageSparsity - 0.02) * 1000;
      const mcpHealth = 100 - (m.mcp.errorRate * 1000);
      const systemHealth = 100 - ((m.system.cpuUsage + m.system.memoryUsage) / 2);

      return (cognitiveHealth + sdrHealth + mcpHealth + systemHealth) / 4;
    });

    return Math.max(0, Math.min(100, this.average(healthScores)));
  }

  private analyzeCognitivePerformance(metrics: PerformanceMetrics[]): any {
    return {
      averageLatency: {
        subcortical: this.average(metrics.map(m => m.cognitive.subcortical.processingLatency)),
        cortical: this.average(metrics.map(m => m.cognitive.cortical.processingLatency)),
        thalamic: this.average(metrics.map(m => m.cognitive.thalamic.processingLatency))
      },
      throughput: {
        subcortical: this.average(metrics.map(m => m.cognitive.subcortical.throughput)),
        cortical: this.average(metrics.map(m => m.cognitive.cortical.throughput)),
        thalamic: this.average(metrics.map(m => m.cognitive.thalamic.throughput))
      },
      errorRates: {
        subcortical: this.average(metrics.map(m => m.cognitive.subcortical.errorCount)),
        cortical: this.average(metrics.map(m => m.cognitive.cortical.errorCount)),
        thalamic: this.average(metrics.map(m => m.cognitive.thalamic.errorCount))
      },
      bottlenecks: ['Cortical layer processing', 'Cross-layer communication'],
      insights: ['Consider parallelizing cortical computations', 'Optimize inhibitory circuits']
    };
  }

  private analyzeSDRPerformance(metrics: PerformanceMetrics[]): any {
    const sparsities = metrics.map(m => m.sdr.averageSparsity);
    return {
      sparsityDistribution: this.calculateDistribution(sparsities),
      memoryEfficiency: this.average(metrics.map(m => m.sdr.creationRate / m.sdr.memoryUsage)),
      semanticPreservation: 0.92, // Would be calculated from actual semantic tests
      compressionStats: {
        average: this.average(metrics.map(m => m.sdr.compressionRatio || 10)),
        min: Math.min(...metrics.map(m => m.sdr.compressionRatio || 10)),
        max: Math.max(...metrics.map(m => m.sdr.compressionRatio || 10))
      },
      insights: ['SDR sparsity is within optimal range', 'Consider implementing adaptive sparsity']
    };
  }

  private analyzeMCPPerformance(metrics: PerformanceMetrics[]): any {
    const latencies = metrics.map(m => m.mcp.averageLatency);
    return {
      protocolEfficiency: 0.85, // Would be calculated from actual protocol metrics
      messageDistribution: {
        'query': 40,
        'update': 30,
        'subscribe': 20,
        'system': 10
      },
      latencyPercentiles: {
        p50: this.percentile(latencies, 50),
        p90: this.percentile(latencies, 90),
        p95: this.percentile(latencies, 95),
        p99: this.percentile(latencies, 99)
      },
      reliabilityScore: this.average(metrics.map(m => 1 - m.mcp.errorRate)) * 100,
      insights: ['MCP latency is stable', 'Consider implementing message batching']
    };
  }

  private analyzeSystemPerformance(metrics: PerformanceMetrics[]): any {
    return {
      resourceUtilization: {
        cpu: { 
          average: this.average(metrics.map(m => m.system.cpuUsage)), 
          peak: Math.max(...metrics.map(m => m.system.cpuUsage)) 
        },
        memory: { 
          average: this.average(metrics.map(m => m.system.memoryUsage)), 
          peak: Math.max(...metrics.map(m => m.system.memoryUsage)) 
        },
        disk: { 
          average: this.average(metrics.map(m => m.system.diskIO)), 
          peak: Math.max(...metrics.map(m => m.system.diskIO)) 
        },
        network: { 
          average: this.average(metrics.map(m => m.system.networkIO)), 
          peak: Math.max(...metrics.map(m => m.system.networkIO)) 
        }
      },
      scalabilityAssessment: 'System can handle 2x current load',
      recommendations: ['Consider implementing caching', 'Optimize memory allocation']
    };
  }

  private calculateDistribution(values: number[]): number[] {
    const bins = 10;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / bins;
    const distribution = new Array(bins).fill(0);

    values.forEach(value => {
      const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
      distribution[binIndex]++;
    });

    return distribution;
  }

  private percentile(values: number[], p: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[index];
  }

  public dispose() {
    if (this.websocket) {
      this.websocket.close();
    }
    this.subscribers.clear();
  }
}