import { 
  MCPTool, 
  ToolExecution, 
  ToolMetrics, 
  ToolUsageAnalytics,
  UsagePattern,
  ToolCategory 
} from '../types';

export interface TimeRange {
  start: Date;
  end: Date;
  period: 'hour' | 'day' | 'week' | 'month' | 'custom';
}

export interface TrendAnalysis {
  toolId: string;
  period: TimeRange;
  metrics: {
    responseTime: TrendMetric;
    successRate: TrendMetric;
    throughput: TrendMetric;
    errorRate: TrendMetric;
  };
  predictions: {
    nextHourLoad: number;
    performanceTrend: 'improving' | 'stable' | 'degrading';
    riskScore: number;
  };
}

export interface TrendMetric {
  current: number;
  previous: number;
  change: number;
  changePercent: number;
  trend: 'up' | 'down' | 'stable';
  dataPoints: Array<{ time: Date; value: number }>;
}

export interface Anomaly {
  toolId: string;
  type: 'performance' | 'error_rate' | 'usage_spike' | 'latency';
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  value: number;
  threshold: number;
  description: string;
  suggestedAction?: string;
}

export interface Insight {
  id: string;
  type: 'optimization' | 'pattern' | 'anomaly' | 'recommendation';
  category: 'performance' | 'usage' | 'reliability' | 'cost';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  affectedTools: string[];
  metrics: Record<string, number>;
  actions: string[];
  priority: number;
}

export interface PerformanceReport {
  period: TimeRange;
  summary: {
    totalExecutions: number;
    totalTools: number;
    overallSuccessRate: number;
    averageResponseTime: number;
    peakConcurrency: number;
  };
  topPerformers: Array<{ tool: MCPTool; metrics: ToolMetrics }>;
  bottomPerformers: Array<{ tool: MCPTool; metrics: ToolMetrics }>;
  insights: Insight[];
  recommendations: string[];
}

export interface ComparativeAnalysis {
  tools: string[];
  period: TimeRange;
  metrics: {
    [toolId: string]: {
      responseTime: ComparisonMetric;
      successRate: ComparisonMetric;
      throughput: ComparisonMetric;
      reliability: ComparisonMetric;
    };
  };
  winner: string;
  summary: string;
}

export interface ComparisonMetric {
  value: number;
  rank: number;
  percentile: number;
  vsAverage: number;
}

export interface ResourceConsumption {
  toolId: string;
  cpu: number;
  memory: number;
  networkBandwidth: number;
  storageIO: number;
  timestamp: Date;
}

export interface CognitiveMetrics {
  toolId: string;
  neuralProcessingSpeed: number;
  memoryConsolidationRate: number;
  knowledgeGraphQueryTime: number;
  federationLatency: number;
  patternRecognitionAccuracy: number;
}

class ToolAnalytics {
  private executionCache: Map<string, ToolExecution[]> = new Map();
  private metricsCache: Map<string, ToolMetrics> = new Map();
  private anomalyThresholds = {
    responseTime: { p95: 2000, p99: 5000 }, // milliseconds
    errorRate: 0.05, // 5%
    usageSpike: 3, // 3x normal usage
    latency: 1000, // milliseconds
  };

  /**
   * Calculate comprehensive metrics for tool executions
   */
  calculateMetrics(executions: ToolExecution[]): ToolMetrics {
    if (executions.length === 0) {
      return this.getEmptyMetrics();
    }

    const successfulExecutions = executions.filter(e => e.status === 'success');
    const errorExecutions = executions.filter(e => e.status === 'error');
    
    const responseTimes = successfulExecutions
      .map(e => e.endTime ? e.endTime - e.startTime : 0)
      .filter(t => t > 0)
      .sort((a, b) => a - b);

    const errorTypes: Record<string, number> = {};
    errorExecutions.forEach(e => {
      if (e.error) {
        const errorType = e.error.split(':')[0] || 'Unknown';
        errorTypes[errorType] = (errorTypes[errorType] || 0) + 1;
      }
    });

    return {
      totalExecutions: executions.length,
      successRate: successfulExecutions.length / executions.length,
      averageResponseTime: this.calculateAverage(responseTimes),
      p50ResponseTime: this.calculatePercentile(responseTimes, 50),
      p95ResponseTime: this.calculatePercentile(responseTimes, 95),
      p99ResponseTime: this.calculatePercentile(responseTimes, 99),
      lastExecutionTime: executions[executions.length - 1]?.startTime ? 
        new Date(executions[executions.length - 1].startTime) : undefined,
      errorCount: errorExecutions.length,
      errorTypes,
    };
  }

  /**
   * Analyze trends over a specific time period
   */
  analyzeTrends(toolId: string, period: TimeRange): TrendAnalysis {
    const executions = this.getExecutionsInRange(toolId, period);
    const previousPeriod = this.getPreviousPeriod(period);
    const previousExecutions = this.getExecutionsInRange(toolId, previousPeriod);

    const currentMetrics = this.calculateMetrics(executions);
    const previousMetrics = this.calculateMetrics(previousExecutions);

    const hourlyBuckets = this.bucketizeExecutions(executions, 'hour');
    const throughputData = Object.entries(hourlyBuckets).map(([time, execs]) => ({
      time: new Date(time),
      value: execs.length,
    }));

    return {
      toolId,
      period,
      metrics: {
        responseTime: this.createTrendMetric(
          currentMetrics.averageResponseTime,
          previousMetrics.averageResponseTime,
          this.getResponseTimeDataPoints(executions)
        ),
        successRate: this.createTrendMetric(
          currentMetrics.successRate,
          previousMetrics.successRate,
          this.getSuccessRateDataPoints(executions)
        ),
        throughput: this.createTrendMetric(
          executions.length,
          previousExecutions.length,
          throughputData
        ),
        errorRate: this.createTrendMetric(
          1 - currentMetrics.successRate,
          1 - previousMetrics.successRate,
          this.getErrorRateDataPoints(executions)
        ),
      },
      predictions: this.generatePredictions(executions, currentMetrics),
    };
  }

  /**
   * Detect anomalies in tool performance
   */
  detectAnomalies(metrics: ToolMetrics[], toolId?: string): Anomaly[] {
    const anomalies: Anomaly[] = [];

    metrics.forEach((metric, index) => {
      // Response time anomalies
      if (metric.p95ResponseTime > this.anomalyThresholds.responseTime.p95) {
        anomalies.push({
          toolId: toolId || 'unknown',
          type: 'performance',
          severity: metric.p99ResponseTime > this.anomalyThresholds.responseTime.p99 ? 'high' : 'medium',
          timestamp: new Date(),
          value: metric.p95ResponseTime,
          threshold: this.anomalyThresholds.responseTime.p95,
          description: `Response time P95 (${metric.p95ResponseTime}ms) exceeds threshold`,
          suggestedAction: 'Consider optimizing query patterns or increasing resources',
        });
      }

      // Error rate anomalies
      const errorRate = 1 - metric.successRate;
      if (errorRate > this.anomalyThresholds.errorRate) {
        anomalies.push({
          toolId: toolId || 'unknown',
          type: 'error_rate',
          severity: errorRate > 0.1 ? 'critical' : 'high',
          timestamp: new Date(),
          value: errorRate,
          threshold: this.anomalyThresholds.errorRate,
          description: `Error rate (${(errorRate * 100).toFixed(1)}%) exceeds threshold`,
          suggestedAction: 'Investigate error patterns and implement retry logic',
        });
      }

      // Usage spike detection
      if (index > 0) {
        const previousMetric = metrics[index - 1];
        const usageRatio = metric.totalExecutions / (previousMetric.totalExecutions || 1);
        
        if (usageRatio > this.anomalyThresholds.usageSpike) {
          anomalies.push({
            toolId: toolId || 'unknown',
            type: 'usage_spike',
            severity: 'medium',
            timestamp: new Date(),
            value: usageRatio,
            threshold: this.anomalyThresholds.usageSpike,
            description: `Usage spike detected (${usageRatio.toFixed(1)}x normal)`,
            suggestedAction: 'Monitor for sustained high usage and scale if needed',
          });
        }
      }
    });

    return anomalies;
  }

  /**
   * Generate insights from usage analytics
   */
  generateInsights(analytics: ToolUsageAnalytics): Insight[] {
    const insights: Insight[] = [];

    // Performance optimization opportunities
    if (analytics.averageResponseTime > 1000) {
      insights.push({
        id: `perf-opt-${analytics.toolId}`,
        type: 'optimization',
        category: 'performance',
        title: 'Performance Optimization Opportunity',
        description: `Average response time of ${analytics.averageResponseTime}ms could be improved`,
        impact: analytics.averageResponseTime > 2000 ? 'high' : 'medium',
        affectedTools: [analytics.toolId],
        metrics: { responseTime: analytics.averageResponseTime },
        actions: [
          'Implement caching for frequently accessed data',
          'Optimize database queries',
          'Consider asynchronous processing for heavy operations',
        ],
        priority: analytics.averageResponseTime > 2000 ? 1 : 2,
      });
    }

    // Pattern detection insights
    if (analytics.patterns && analytics.patterns.length > 0) {
      analytics.patterns.forEach(pattern => {
        if (pattern.type === 'sequence' && pattern.frequency > 10) {
          insights.push({
            id: `pattern-${analytics.toolId}-${pattern.type}`,
            type: 'pattern',
            category: 'usage',
            title: 'Frequent Usage Pattern Detected',
            description: pattern.description,
            impact: 'medium',
            affectedTools: pattern.tools || [analytics.toolId],
            metrics: { frequency: pattern.frequency },
            actions: [
              'Consider creating a composite tool for this workflow',
              'Optimize the sequence for better performance',
            ],
            priority: 3,
          });
        }
      });
    }

    // Reliability insights
    const errorRate = (analytics.errorCount / analytics.executions) * 100;
    if (errorRate > 1) {
      insights.push({
        id: `reliability-${analytics.toolId}`,
        type: 'recommendation',
        category: 'reliability',
        title: 'Reliability Improvement Needed',
        description: `Error rate of ${errorRate.toFixed(1)}% affects user experience`,
        impact: errorRate > 5 ? 'high' : 'medium',
        affectedTools: [analytics.toolId],
        metrics: { errorRate },
        actions: [
          'Implement comprehensive error handling',
          'Add retry logic with exponential backoff',
          'Monitor and alert on error spikes',
        ],
        priority: errorRate > 5 ? 1 : 2,
      });
    }

    return insights;
  }

  /**
   * Export analytics report in specified format
   */
  async exportReport(
    toolIds: string[], 
    period: TimeRange,
    format: 'csv' | 'pdf' | 'json'
  ): Promise<Blob> {
    const report = await this.generateReport(toolIds, period);

    switch (format) {
      case 'csv':
        return this.exportAsCSV(report);
      case 'pdf':
        return this.exportAsPDF(report);
      case 'json':
        return this.exportAsJSON(report);
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  /**
   * Compare multiple tools' performance
   */
  compareTools(toolIds: string[], period: TimeRange): ComparativeAnalysis {
    const toolMetrics: Record<string, any> = {};
    const allMetrics: number[] = [];

    toolIds.forEach(toolId => {
      const executions = this.getExecutionsInRange(toolId, period);
      const metrics = this.calculateMetrics(executions);
      
      toolMetrics[toolId] = {
        responseTime: { value: metrics.averageResponseTime },
        successRate: { value: metrics.successRate },
        throughput: { value: executions.length },
        reliability: { value: metrics.successRate * (1 - (metrics.errorCount / executions.length)) },
      };

      allMetrics.push(metrics.averageResponseTime);
    });

    // Calculate ranks and percentiles
    toolIds.forEach(toolId => {
      const toolData = toolMetrics[toolId];
      
      // Response time (lower is better)
      toolData.responseTime.rank = this.calculateRank(
        toolData.responseTime.value,
        toolIds.map(id => toolMetrics[id].responseTime.value),
        'asc'
      );
      
      // Success rate (higher is better)
      toolData.successRate.rank = this.calculateRank(
        toolData.successRate.value,
        toolIds.map(id => toolMetrics[id].successRate.value),
        'desc'
      );
      
      // Throughput (higher is better)
      toolData.throughput.rank = this.calculateRank(
        toolData.throughput.value,
        toolIds.map(id => toolMetrics[id].throughput.value),
        'desc'
      );
      
      // Reliability (higher is better)
      toolData.reliability.rank = this.calculateRank(
        toolData.reliability.value,
        toolIds.map(id => toolMetrics[id].reliability.value),
        'desc'
      );
    });

    // Determine winner based on composite score
    const scores = toolIds.map(id => {
      const metrics = toolMetrics[id];
      return {
        id,
        score: (
          (1 / metrics.responseTime.rank) * 0.3 +
          (1 / metrics.successRate.rank) * 0.3 +
          (1 / metrics.throughput.rank) * 0.2 +
          (1 / metrics.reliability.rank) * 0.2
        ),
      };
    });

    const winner = scores.reduce((a, b) => a.score > b.score ? a : b).id;

    return {
      tools: toolIds,
      period,
      metrics: toolMetrics,
      winner,
      summary: this.generateComparisonSummary(toolMetrics, winner),
    };
  }

  /**
   * Get cognitive-specific metrics for LLMKG tools
   */
  getCognitiveMetrics(toolId: string, executions: ToolExecution[]): CognitiveMetrics {
    // Extract cognitive-specific metrics from execution metadata
    const cognitiveExecutions = executions.filter(e => e.metadata?.cognitive);
    
    const neuralSpeeds = cognitiveExecutions
      .map(e => e.metadata?.cognitive?.neuralProcessingTime)
      .filter(Boolean) as number[];
    
    const consolidationRates = cognitiveExecutions
      .map(e => e.metadata?.cognitive?.memoryConsolidationRate)
      .filter(Boolean) as number[];
    
    const graphQueryTimes = cognitiveExecutions
      .map(e => e.metadata?.cognitive?.knowledgeGraphQueryTime)
      .filter(Boolean) as number[];
    
    const federationLatencies = cognitiveExecutions
      .map(e => e.metadata?.cognitive?.federationLatency)
      .filter(Boolean) as number[];
    
    const patternAccuracies = cognitiveExecutions
      .map(e => e.metadata?.cognitive?.patternRecognitionAccuracy)
      .filter(Boolean) as number[];

    return {
      toolId,
      neuralProcessingSpeed: this.calculateAverage(neuralSpeeds),
      memoryConsolidationRate: this.calculateAverage(consolidationRates),
      knowledgeGraphQueryTime: this.calculateAverage(graphQueryTimes),
      federationLatency: this.calculateAverage(federationLatencies),
      patternRecognitionAccuracy: this.calculateAverage(patternAccuracies),
    };
  }

  // Helper methods
  private getEmptyMetrics(): ToolMetrics {
    return {
      totalExecutions: 0,
      successRate: 0,
      averageResponseTime: 0,
      p50ResponseTime: 0,
      p95ResponseTime: 0,
      p99ResponseTime: 0,
      errorCount: 0,
      errorTypes: {},
    };
  }

  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculatePercentile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 0) return 0;
    const index = Math.ceil((percentile / 100) * sortedValues.length) - 1;
    return sortedValues[Math.max(0, Math.min(index, sortedValues.length - 1))];
  }

  private getExecutionsInRange(toolId: string, range: TimeRange): ToolExecution[] {
    // In a real implementation, this would fetch from a database or API
    const cached = this.executionCache.get(toolId) || [];
    return cached.filter(e => {
      const execTime = new Date(e.startTime);
      return execTime >= range.start && execTime <= range.end;
    });
  }

  private getPreviousPeriod(period: TimeRange): TimeRange {
    const duration = period.end.getTime() - period.start.getTime();
    return {
      start: new Date(period.start.getTime() - duration),
      end: new Date(period.start.getTime()),
      period: period.period,
    };
  }

  private bucketizeExecutions(
    executions: ToolExecution[], 
    bucketSize: 'hour' | 'day'
  ): Record<string, ToolExecution[]> {
    const buckets: Record<string, ToolExecution[]> = {};
    
    executions.forEach(exec => {
      const date = new Date(exec.startTime);
      const key = bucketSize === 'hour' 
        ? `${date.toISOString().slice(0, 13)}:00:00`
        : date.toISOString().slice(0, 10);
      
      if (!buckets[key]) {
        buckets[key] = [];
      }
      buckets[key].push(exec);
    });

    return buckets;
  }

  private createTrendMetric(
    current: number, 
    previous: number,
    dataPoints: Array<{ time: Date; value: number }>
  ): TrendMetric {
    const change = current - previous;
    const changePercent = previous !== 0 ? (change / previous) * 100 : 0;
    
    return {
      current,
      previous,
      change,
      changePercent,
      trend: change > 0.05 ? 'up' : change < -0.05 ? 'down' : 'stable',
      dataPoints,
    };
  }

  private getResponseTimeDataPoints(executions: ToolExecution[]): Array<{ time: Date; value: number }> {
    return executions
      .filter(e => e.status === 'success' && e.endTime)
      .map(e => ({
        time: new Date(e.startTime),
        value: e.endTime! - e.startTime,
      }));
  }

  private getSuccessRateDataPoints(executions: ToolExecution[]): Array<{ time: Date; value: number }> {
    const hourlyBuckets = this.bucketizeExecutions(executions, 'hour');
    
    return Object.entries(hourlyBuckets).map(([time, execs]) => {
      const successful = execs.filter(e => e.status === 'success').length;
      return {
        time: new Date(time),
        value: execs.length > 0 ? successful / execs.length : 0,
      };
    });
  }

  private getErrorRateDataPoints(executions: ToolExecution[]): Array<{ time: Date; value: number }> {
    const hourlyBuckets = this.bucketizeExecutions(executions, 'hour');
    
    return Object.entries(hourlyBuckets).map(([time, execs]) => {
      const errors = execs.filter(e => e.status === 'error').length;
      return {
        time: new Date(time),
        value: execs.length > 0 ? errors / execs.length : 0,
      };
    });
  }

  private generatePredictions(
    executions: ToolExecution[], 
    currentMetrics: ToolMetrics
  ): TrendAnalysis['predictions'] {
    // Simple prediction logic - in production, use ML models
    const recentExecutions = executions.slice(-100);
    const recentResponseTimes = recentExecutions
      .filter(e => e.status === 'success' && e.endTime)
      .map(e => e.endTime! - e.startTime);

    const trend = this.calculateTrend(recentResponseTimes);
    const avgLoad = executions.length / 24; // Average hourly load

    return {
      nextHourLoad: Math.round(avgLoad * (1 + Math.random() * 0.2 - 0.1)),
      performanceTrend: trend > 0.1 ? 'degrading' : trend < -0.1 ? 'improving' : 'stable',
      riskScore: this.calculateRiskScore(currentMetrics),
    };
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    // Simple linear regression slope
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = values.reduce((sum, val, i) => sum + val * i, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  }

  private calculateRiskScore(metrics: ToolMetrics): number {
    let score = 0;
    
    // Response time risk
    if (metrics.p95ResponseTime > 2000) score += 0.3;
    if (metrics.p99ResponseTime > 5000) score += 0.2;
    
    // Error rate risk
    const errorRate = 1 - metrics.successRate;
    if (errorRate > 0.05) score += 0.3;
    if (errorRate > 0.1) score += 0.2;
    
    return Math.min(score, 1);
  }

  private calculateRank(value: number, allValues: number[], order: 'asc' | 'desc'): number {
    const sorted = [...allValues].sort((a, b) => order === 'asc' ? a - b : b - a);
    return sorted.indexOf(value) + 1;
  }

  private generateComparisonSummary(metrics: Record<string, any>, winner: string): string {
    const winnerMetrics = metrics[winner];
    const advantages: string[] = [];
    
    if (winnerMetrics.responseTime.rank === 1) {
      advantages.push('fastest response time');
    }
    if (winnerMetrics.successRate.rank === 1) {
      advantages.push('highest success rate');
    }
    if (winnerMetrics.throughput.rank === 1) {
      advantages.push('highest throughput');
    }
    if (winnerMetrics.reliability.rank === 1) {
      advantages.push('most reliable');
    }
    
    return `${winner} performs best overall with ${advantages.join(', ')}`;
  }

  private async generateReport(toolIds: string[], period: TimeRange): Promise<PerformanceReport> {
    const allExecutions: ToolExecution[] = [];
    const toolMetrics: Array<{ tool: MCPTool; metrics: ToolMetrics }> = [];

    // Collect data for all tools
    for (const toolId of toolIds) {
      const executions = this.getExecutionsInRange(toolId, period);
      allExecutions.push(...executions);
      
      const metrics = this.calculateMetrics(executions);
      // Note: In production, fetch actual tool data from store
      toolMetrics.push({
        tool: { id: toolId } as MCPTool,
        metrics,
      });
    }

    // Sort by performance
    const topPerformers = [...toolMetrics]
      .sort((a, b) => a.metrics.averageResponseTime - b.metrics.averageResponseTime)
      .slice(0, 5);
    
    const bottomPerformers = [...toolMetrics]
      .sort((a, b) => b.metrics.averageResponseTime - a.metrics.averageResponseTime)
      .slice(0, 5);

    // Generate insights
    const insights: Insight[] = [];
    toolMetrics.forEach(({ tool, metrics }) => {
      const analytics: ToolUsageAnalytics = {
        toolId: tool.id,
        period: period.period,
        executions: metrics.totalExecutions,
        successCount: Math.round(metrics.totalExecutions * metrics.successRate),
        errorCount: metrics.errorCount,
        averageResponseTime: metrics.averageResponseTime,
      };
      insights.push(...this.generateInsights(analytics));
    });

    return {
      period,
      summary: {
        totalExecutions: allExecutions.length,
        totalTools: toolIds.length,
        overallSuccessRate: this.calculateAverage(toolMetrics.map(t => t.metrics.successRate)),
        averageResponseTime: this.calculateAverage(toolMetrics.map(t => t.metrics.averageResponseTime)),
        peakConcurrency: this.calculatePeakConcurrency(allExecutions),
      },
      topPerformers,
      bottomPerformers,
      insights,
      recommendations: this.generateRecommendations(insights),
    };
  }

  private calculatePeakConcurrency(executions: ToolExecution[]): number {
    if (executions.length === 0) return 0;
    
    const events: Array<{ time: number; type: 'start' | 'end' }> = [];
    
    executions.forEach(exec => {
      events.push({ time: exec.startTime, type: 'start' });
      if (exec.endTime) {
        events.push({ time: exec.endTime, type: 'end' });
      }
    });
    
    events.sort((a, b) => a.time - b.time);
    
    let current = 0;
    let peak = 0;
    
    events.forEach(event => {
      if (event.type === 'start') {
        current++;
        peak = Math.max(peak, current);
      } else {
        current--;
      }
    });
    
    return peak;
  }

  private generateRecommendations(insights: Insight[]): string[] {
    const recommendations: string[] = [];
    
    const highPriorityInsights = insights.filter(i => i.priority === 1);
    if (highPriorityInsights.length > 0) {
      recommendations.push('Address high-priority performance issues immediately');
    }
    
    const performanceInsights = insights.filter(i => i.category === 'performance');
    if (performanceInsights.length > 3) {
      recommendations.push('Consider implementing a performance monitoring dashboard');
    }
    
    const reliabilityInsights = insights.filter(i => i.category === 'reliability');
    if (reliabilityInsights.length > 0) {
      recommendations.push('Implement comprehensive error tracking and alerting');
    }
    
    return recommendations;
  }

  private async exportAsCSV(report: PerformanceReport): Promise<Blob> {
    const csvContent = this.convertReportToCSV(report);
    return new Blob([csvContent], { type: 'text/csv' });
  }

  private async exportAsPDF(report: PerformanceReport): Promise<Blob> {
    // In production, use a PDF generation library
    const pdfContent = JSON.stringify(report, null, 2);
    return new Blob([pdfContent], { type: 'application/pdf' });
  }

  private async exportAsJSON(report: PerformanceReport): Promise<Blob> {
    const jsonContent = JSON.stringify(report, null, 2);
    return new Blob([jsonContent], { type: 'application/json' });
  }

  private convertReportToCSV(report: PerformanceReport): string {
    const rows: string[] = [];
    
    // Header
    rows.push('Metric,Value');
    
    // Summary metrics
    rows.push(`Total Executions,${report.summary.totalExecutions}`);
    rows.push(`Total Tools,${report.summary.totalTools}`);
    rows.push(`Overall Success Rate,${(report.summary.overallSuccessRate * 100).toFixed(2)}%`);
    rows.push(`Average Response Time,${report.summary.averageResponseTime.toFixed(2)}ms`);
    rows.push(`Peak Concurrency,${report.summary.peakConcurrency}`);
    
    return rows.join('\n');
  }
}

export default new ToolAnalytics();