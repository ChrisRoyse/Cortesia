import { useState, useEffect, useMemo } from 'react';
import { ArchitectureData, ComponentMetrics, SystemHealth } from '../types';

export interface ArchitectureMetrics {
  totalComponents: number;
  activeComponents: number;
  healthyComponents: number;
  warningComponents: number;
  criticalComponents: number;
  offlineComponents: number;
  averageCPU: number;
  averageMemory: number;
  averageLatency: number;
  totalThroughput: number;
  averageErrorRate: number;
  connectionCount: number;
  activeConnections: number;
  lastUpdated: number;
}

export function useArchitectureMetrics(data: ArchitectureData) {
  const [metrics, setMetrics] = useState<ArchitectureMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Calculate metrics from architecture data
  const calculatedMetrics = useMemo(() => {
    if (!data || !data.nodes) {
      return null;
    }

    const nodes = data.nodes;
    const connections = data.connections || [];

    // Component status counts
    const healthyComponents = nodes.filter(n => n.status === 'healthy').length;
    const warningComponents = nodes.filter(n => n.status === 'warning').length;
    const criticalComponents = nodes.filter(n => n.status === 'critical').length;
    const offlineComponents = nodes.filter(n => n.status === 'offline').length;
    const processingComponents = nodes.filter(n => n.status === 'processing').length;
    const activeComponents = nodes.length - offlineComponents;

    // Calculate average metrics from nodes with metrics
    const nodesWithMetrics = nodes.filter(n => n.metrics);
    const metricsCount = nodesWithMetrics.length;

    let averageCPU = 0;
    let averageMemory = 0;
    let averageLatency = 0;
    let totalThroughput = 0;
    let averageErrorRate = 0;

    if (metricsCount > 0) {
      const totals = nodesWithMetrics.reduce((acc, node) => {
        const metrics = node.metrics!;
        return {
          cpu: acc.cpu + metrics.cpu.current,
          memory: acc.memory + metrics.memory.current,
          latency: acc.latency + metrics.latency.current,
          throughput: acc.throughput + metrics.throughput.current,
          errorRate: acc.errorRate + metrics.errorRate.current,
        };
      }, { cpu: 0, memory: 0, latency: 0, throughput: 0, errorRate: 0 });

      averageCPU = totals.cpu / metricsCount;
      averageMemory = totals.memory / metricsCount;
      averageLatency = totals.latency / metricsCount;
      totalThroughput = totals.throughput; // Total, not average
      averageErrorRate = totals.errorRate / metricsCount;
    }

    // Connection metrics
    const activeConnections = connections.filter(c => c.active).length;

    return {
      totalComponents: nodes.length,
      activeComponents,
      healthyComponents,
      warningComponents,
      criticalComponents,
      offlineComponents,
      averageCPU,
      averageMemory,
      averageLatency,
      totalThroughput,
      averageErrorRate,
      connectionCount: connections.length,
      activeConnections,
      lastUpdated: Date.now(),
    };
  }, [data]);

  // Update metrics when data changes
  useEffect(() => {
    if (calculatedMetrics) {
      setMetrics(calculatedMetrics);
      setIsLoading(false);
    }
  }, [calculatedMetrics]);

  // Calculate performance trends
  const trends = useMemo(() => {
    if (!metrics) return null;

    // This would typically store historical data
    // For now, we'll return mock trend data
    return {
      cpuTrend: 'stable' as 'improving' | 'stable' | 'degrading',
      memoryTrend: 'stable' as 'improving' | 'stable' | 'degrading',
      latencyTrend: 'improving' as 'improving' | 'stable' | 'degrading',
      throughputTrend: 'stable' as 'improving' | 'stable' | 'degrading',
      errorRateTrend: 'improving' as 'improving' | 'stable' | 'degrading',
    };
  }, [metrics]);

  // Calculate system health score
  const healthScore = useMemo(() => {
    if (!metrics) return 0;

    const weights = {
      healthy: 1.0,
      warning: 0.7,
      critical: 0.3,
      offline: 0.0,
      processing: 0.8,
    };

    const totalNodes = metrics.totalComponents;
    if (totalNodes === 0) return 1.0;

    const weightedScore = (
      metrics.healthyComponents * weights.healthy +
      metrics.warningComponents * weights.warning +
      metrics.criticalComponents * weights.critical +
      metrics.offlineComponents * weights.offline
    ) / totalNodes;

    return Math.max(0, Math.min(1, weightedScore));
  }, [metrics]);

  // Get performance alerts
  const alerts = useMemo(() => {
    if (!metrics) return [];

    const alerts = [];

    if (metrics.averageCPU > 80) {
      alerts.push({
        type: 'warning',
        message: `High average CPU usage: ${metrics.averageCPU.toFixed(1)}%`,
        severity: metrics.averageCPU > 90 ? 'critical' : 'warning',
      });
    }

    if (metrics.averageMemory > 85) {
      alerts.push({
        type: 'warning',
        message: `High average memory usage: ${metrics.averageMemory.toFixed(1)}%`,
        severity: metrics.averageMemory > 95 ? 'critical' : 'warning',
      });
    }

    if (metrics.averageLatency > 200) {
      alerts.push({
        type: 'warning',
        message: `High average latency: ${metrics.averageLatency.toFixed(0)}ms`,
        severity: metrics.averageLatency > 500 ? 'critical' : 'warning',
      });
    }

    if (metrics.averageErrorRate > 5) {
      alerts.push({
        type: 'error',
        message: `High error rate: ${metrics.averageErrorRate.toFixed(1)}%`,
        severity: 'critical',
      });
    }

    if (metrics.criticalComponents > 0) {
      alerts.push({
        type: 'error',
        message: `${metrics.criticalComponents} component${metrics.criticalComponents > 1 ? 's' : ''} in critical state`,
        severity: 'critical',
      });
    }

    if (metrics.offlineComponents > 0) {
      alerts.push({
        type: 'warning',
        message: `${metrics.offlineComponents} component${metrics.offlineComponents > 1 ? 's' : ''} offline`,
        severity: 'warning',
      });
    }

    return alerts;
  }, [metrics]);

  // Get component distribution by type
  const componentDistribution = useMemo(() => {
    if (!data || !data.nodes) return {};

    return data.nodes.reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
  }, [data]);

  // Get layer distribution
  const layerDistribution = useMemo(() => {
    if (!data || !data.nodes) return {};

    return data.nodes.reduce((acc, node) => {
      acc[node.layer] = (acc[node.layer] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
  }, [data]);

  return {
    architectureMetrics: metrics,
    isLoading,
    trends,
    healthScore,
    alerts,
    componentDistribution,
    layerDistribution,
  };
}