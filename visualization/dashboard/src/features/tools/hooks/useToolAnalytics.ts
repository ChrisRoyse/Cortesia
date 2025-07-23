import { useState, useEffect, useCallback, useMemo } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { selectExecutionHistory, selectToolById } from '../stores/toolsSlice';
import ToolAnalytics, {
  TimeRange,
  TrendAnalysis,
  Anomaly,
  Insight,
  PerformanceReport,
  ComparativeAnalysis,
  CognitiveMetrics,
} from '../services/ToolAnalytics';
import { ToolExecution } from '../types';

interface UseToolAnalyticsResult {
  loading: boolean;
  error: string | null;
  report: PerformanceReport | null;
  trends: TrendAnalysis[];
  anomalies: Anomaly[];
  insights: Insight[];
  comparativeAnalysis: ComparativeAnalysis | null;
  cognitiveMetrics: CognitiveMetrics[];
  refresh: () => Promise<void>;
  exportReport: (format: 'csv' | 'pdf' | 'json') => Promise<void>;
  compareTools: (toolIds: string[]) => Promise<void>;
}

/**
 * Hook for accessing tool analytics data and functionality
 */
const useToolAnalytics = (
  toolIds: string[],
  timeRange: TimeRange,
  options?: {
    autoRefresh?: boolean;
    refreshInterval?: number;
    enableCognitiveMetrics?: boolean;
  }
): UseToolAnalyticsResult => {
  const dispatch = useDispatch();
  const executionHistory = useSelector(selectExecutionHistory);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<PerformanceReport | null>(null);
  const [trends, setTrends] = useState<TrendAnalysis[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [comparativeAnalysis, setComparativeAnalysis] = useState<ComparativeAnalysis | null>(null);
  const [cognitiveMetrics, setCognitiveMetrics] = useState<CognitiveMetrics[]>([]);

  const {
    autoRefresh = false,
    refreshInterval = 60000, // 1 minute
    enableCognitiveMetrics = true,
  } = options || {};

  // Filter executions for selected tools and time range
  const filteredExecutions = useMemo(() => {
    return executionHistory.filter(exec => {
      const execTime = new Date(exec.startTime);
      const inTimeRange = execTime >= timeRange.start && execTime <= timeRange.end;
      const inSelectedTools = toolIds.length === 0 || toolIds.includes(exec.toolId);
      return inTimeRange && inSelectedTools;
    });
  }, [executionHistory, toolIds, timeRange]);

  // Group executions by tool
  const executionsByTool = useMemo(() => {
    const grouped = new Map<string, ToolExecution[]>();
    
    filteredExecutions.forEach(exec => {
      if (!grouped.has(exec.toolId)) {
        grouped.set(exec.toolId, []);
      }
      grouped.get(exec.toolId)!.push(exec);
    });

    return grouped;
  }, [filteredExecutions]);

  // Calculate analytics data
  const calculateAnalytics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Calculate trends for each tool
      const trendPromises = toolIds.map(toolId => 
        ToolAnalytics.analyzeTrends(toolId, timeRange)
      );
      const trendResults = await Promise.all(trendPromises);
      setTrends(trendResults);

      // Detect anomalies
      const allAnomalies: Anomaly[] = [];
      executionsByTool.forEach((executions, toolId) => {
        const metrics = ToolAnalytics.calculateMetrics(executions);
        const toolAnomalies = ToolAnalytics.detectAnomalies([metrics], toolId);
        allAnomalies.push(...toolAnomalies);
      });
      setAnomalies(allAnomalies);

      // Generate insights
      const allInsights: Insight[] = [];
      executionsByTool.forEach((executions, toolId) => {
        const metrics = ToolAnalytics.calculateMetrics(executions);
        const analytics = {
          toolId,
          period: timeRange.period,
          executions: executions.length,
          successCount: executions.filter(e => e.status === 'success').length,
          errorCount: executions.filter(e => e.status === 'error').length,
          averageResponseTime: metrics.averageResponseTime,
          patterns: detectUsagePatterns(executions),
        };
        const toolInsights = ToolAnalytics.generateInsights(analytics);
        allInsights.push(...toolInsights);
      });
      setInsights(allInsights);

      // Generate performance report
      const reportData = await generateReport(toolIds, timeRange, executionsByTool);
      setReport(reportData);

      // Calculate cognitive metrics if enabled
      if (enableCognitiveMetrics) {
        const cogMetrics: CognitiveMetrics[] = [];
        executionsByTool.forEach((executions, toolId) => {
          const metrics = ToolAnalytics.getCognitiveMetrics(toolId, executions);
          if (metrics.neuralProcessingSpeed > 0) { // Only add if we have cognitive data
            cogMetrics.push(metrics);
          }
        });
        setCognitiveMetrics(cogMetrics);
      }

    } catch (err) {
      console.error('Error calculating analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to calculate analytics');
    } finally {
      setLoading(false);
    }
  }, [toolIds, timeRange, executionsByTool, enableCognitiveMetrics]);

  // Refresh analytics data
  const refresh = useCallback(async () => {
    await calculateAnalytics();
  }, [calculateAnalytics]);

  // Export report
  const exportReport = useCallback(async (format: 'csv' | 'pdf' | 'json') => {
    try {
      const blob = await ToolAnalytics.exportReport(toolIds, timeRange, format);
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `tool-analytics-${format}-${Date.now()}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting report:', err);
      setError(err instanceof Error ? err.message : 'Failed to export report');
    }
  }, [toolIds, timeRange]);

  // Compare tools
  const compareTools = useCallback(async (compareToolIds: string[]) => {
    try {
      const comparison = ToolAnalytics.compareTools(compareToolIds, timeRange);
      setComparativeAnalysis(comparison);
    } catch (err) {
      console.error('Error comparing tools:', err);
      setError(err instanceof Error ? err.message : 'Failed to compare tools');
    }
  }, [timeRange]);

  // Effect to calculate analytics on mount and when dependencies change
  useEffect(() => {
    calculateAnalytics();
  }, [calculateAnalytics]);

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      calculateAnalytics();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, calculateAnalytics]);

  return {
    loading,
    error,
    report,
    trends,
    anomalies,
    insights,
    comparativeAnalysis,
    cognitiveMetrics,
    refresh,
    exportReport,
    compareTools,
  };
};

// Helper function to detect usage patterns
function detectUsagePatterns(executions: ToolExecution[]): any[] {
  const patterns: any[] = [];

  // Sort executions by time
  const sortedExecutions = [...executions].sort((a, b) => a.startTime - b.startTime);

  // Detect temporal patterns (peak hours)
  const hourlyDistribution = new Map<number, number>();
  sortedExecutions.forEach(exec => {
    const hour = new Date(exec.startTime).getHours();
    hourlyDistribution.set(hour, (hourlyDistribution.get(hour) || 0) + 1);
  });

  const avgHourlyCount = Array.from(hourlyDistribution.values()).reduce((a, b) => a + b, 0) / 24;
  const peakHours = Array.from(hourlyDistribution.entries())
    .filter(([_, count]) => count > avgHourlyCount * 1.5)
    .map(([hour]) => hour);

  if (peakHours.length > 0) {
    patterns.push({
      type: 'temporal',
      description: `Peak usage hours: ${peakHours.join(', ')}`,
      frequency: peakHours.length,
    });
  }

  // Detect error patterns
  const errorExecutions = sortedExecutions.filter(e => e.status === 'error');
  if (errorExecutions.length > 5) {
    const errorTypes = new Map<string, number>();
    errorExecutions.forEach(exec => {
      if (exec.error) {
        const errorType = exec.error.split(':')[0];
        errorTypes.set(errorType, (errorTypes.get(errorType) || 0) + 1);
      }
    });

    const commonErrors = Array.from(errorTypes.entries())
      .filter(([_, count]) => count > 2)
      .sort((a, b) => b[1] - a[1]);

    if (commonErrors.length > 0) {
      patterns.push({
        type: 'error',
        description: `Common error: ${commonErrors[0][0]} (${commonErrors[0][1]} occurrences)`,
        frequency: commonErrors[0][1],
      });
    }
  }

  // Detect burst patterns
  const timeDiffs: number[] = [];
  for (let i = 1; i < sortedExecutions.length; i++) {
    timeDiffs.push(sortedExecutions[i].startTime - sortedExecutions[i - 1].startTime);
  }

  const avgTimeDiff = timeDiffs.reduce((a, b) => a + b, 0) / timeDiffs.length;
  const burstThreshold = avgTimeDiff / 3;
  let burstCount = 0;

  timeDiffs.forEach(diff => {
    if (diff < burstThreshold) burstCount++;
  });

  if (burstCount > timeDiffs.length * 0.2) {
    patterns.push({
      type: 'burst',
      description: 'Frequent burst usage pattern detected',
      frequency: burstCount,
    });
  }

  return patterns;
}

// Helper function to generate performance report
async function generateReport(
  toolIds: string[],
  timeRange: TimeRange,
  executionsByTool: Map<string, ToolExecution[]>
): Promise<PerformanceReport> {
  const allExecutions: ToolExecution[] = [];
  const toolMetrics: Array<{ tool: any; metrics: any }> = [];

  executionsByTool.forEach((executions, toolId) => {
    allExecutions.push(...executions);
    const metrics = ToolAnalytics.calculateMetrics(executions);
    toolMetrics.push({
      tool: { id: toolId, name: toolId }, // In production, fetch actual tool data
      metrics,
    });
  });

  // Sort by performance
  const topPerformers = [...toolMetrics]
    .sort((a, b) => a.metrics.averageResponseTime - b.metrics.averageResponseTime)
    .slice(0, 5);

  const bottomPerformers = [...toolMetrics]
    .sort((a, b) => b.metrics.averageResponseTime - a.metrics.averageResponseTime)
    .slice(0, 5);

  // Calculate summary statistics
  const totalExecutions = allExecutions.length;
  const successfulExecutions = allExecutions.filter(e => e.status === 'success');
  const overallSuccessRate = totalExecutions > 0 ? successfulExecutions.length / totalExecutions : 0;

  const responseTimes = successfulExecutions
    .filter(e => e.endTime)
    .map(e => e.endTime! - e.startTime);
  const averageResponseTime = responseTimes.length > 0
    ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length
    : 0;

  // Calculate peak concurrency
  const peakConcurrency = calculatePeakConcurrency(allExecutions);

  return {
    period: timeRange,
    summary: {
      totalExecutions,
      totalTools: toolIds.length,
      overallSuccessRate,
      averageResponseTime,
      peakConcurrency,
    },
    topPerformers,
    bottomPerformers,
    insights: [],
    recommendations: generateRecommendations(toolMetrics),
  };
}

// Helper function to calculate peak concurrency
function calculatePeakConcurrency(executions: ToolExecution[]): number {
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

// Helper function to generate recommendations
function generateRecommendations(toolMetrics: Array<{ tool: any; metrics: any }>): string[] {
  const recommendations: string[] = [];

  // Check for slow tools
  const slowTools = toolMetrics.filter(t => t.metrics.averageResponseTime > 3000);
  if (slowTools.length > 0) {
    recommendations.push(`${slowTools.length} tool(s) have slow response times. Consider optimization or caching.`);
  }

  // Check for unreliable tools
  const unreliableTools = toolMetrics.filter(t => t.metrics.successRate < 0.9);
  if (unreliableTools.length > 0) {
    recommendations.push(`${unreliableTools.length} tool(s) have low success rates. Investigate error patterns.`);
  }

  // Check for high variance tools
  const highVarianceTools = toolMetrics.filter(t => {
    const variance = t.metrics.p95ResponseTime > 0
      ? (t.metrics.p95ResponseTime - t.metrics.p50ResponseTime) / t.metrics.p95ResponseTime
      : 0;
    return variance > 0.5;
  });
  if (highVarianceTools.length > 0) {
    recommendations.push(`${highVarianceTools.length} tool(s) show high response time variance. Consider load balancing.`);
  }

  if (recommendations.length === 0) {
    recommendations.push('All tools are performing within acceptable parameters.');
  }

  return recommendations;
}

export { useToolAnalytics };
export default useToolAnalytics;