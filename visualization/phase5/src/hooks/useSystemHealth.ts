import { useState, useEffect, useMemo } from 'react';
import { ArchitectureData, SystemHealth, ComponentStatus, ArchitectureNode } from '../types';

export interface HealthTrends {
  hourly: { timestamp: number; score: number }[];
  daily: { timestamp: number; score: number }[];
  weekly: { timestamp: number; score: number }[];
}

export interface HealthRecommendations {
  priority: 'high' | 'medium' | 'low';
  category: 'performance' | 'reliability' | 'capacity' | 'configuration';
  title: string;
  description: string;
  action: string;
  affectedComponents?: string[];
}

export function useSystemHealth(data: ArchitectureData) {
  const [healthHistory, setHealthHistory] = useState<HealthTrends>({
    hourly: [],
    daily: [],
    weekly: []
  });
  const [isMonitoring, setIsMonitoring] = useState(true);

  // Calculate current system health
  const systemHealth = useMemo((): SystemHealth => {
    if (!data || !data.nodes || data.nodes.length === 0) {
      return {
        overall: 'offline',
        score: 0,
        totalComponents: 0,
        healthyComponents: 0,
        warningComponents: 0,
        criticalComponents: 0,
        offlineComponents: 0,
        recommendations: ['No components detected'],
        lastUpdated: Date.now()
      };
    }

    const nodes = data.nodes;
    
    // Count components by status
    const statusCounts = nodes.reduce((acc, node) => {
      acc[node.status] = (acc[node.status] || 0) + 1;
      return acc;
    }, {} as Record<ComponentStatus, number>);

    const totalComponents = nodes.length;
    const healthyComponents = statusCounts.healthy || 0;
    const warningComponents = statusCounts.warning || 0;
    const criticalComponents = statusCounts.critical || 0;
    const offlineComponents = statusCounts.offline || 0;
    const processingComponents = statusCounts.processing || 0;

    // Calculate health score (0-1)
    const weights = {
      healthy: 1.0,
      processing: 0.9,
      warning: 0.6,
      critical: 0.2,
      offline: 0.0
    };

    const weightedScore = (
      healthyComponents * weights.healthy +
      processingComponents * weights.processing +
      warningComponents * weights.warning +
      criticalComponents * weights.critical +
      offlineComponents * weights.offline
    ) / totalComponents;

    // Determine overall status
    let overall: ComponentStatus = 'healthy';
    if (criticalComponents > 0) {
      overall = 'critical';
    } else if (offlineComponents > totalComponents * 0.1) {
      overall = 'critical';
    } else if (warningComponents > 0 || offlineComponents > 0) {
      overall = 'warning';
    } else if (processingComponents > totalComponents * 0.5) {
      overall = 'processing';
    }

    const recommendations = generateRecommendations(data, {
      totalComponents,
      healthyComponents,
      warningComponents,
      criticalComponents,
      offlineComponents,
      score: weightedScore
    });

    return {
      overall,
      score: weightedScore,
      totalComponents,
      healthyComponents,
      warningComponents,
      criticalComponents,
      offlineComponents,
      recommendations: recommendations.map(r => r.title),
      lastUpdated: Date.now()
    };
  }, [data]);

  // Generate detailed health recommendations
  const detailedRecommendations = useMemo((): HealthRecommendations[] => {
    return generateRecommendations(data, systemHealth);
  }, [data, systemHealth]);

  // Calculate component health by layer
  const layerHealth = useMemo(() => {
    if (!data || !data.nodes) return {};

    const layerStats = data.nodes.reduce((acc, node) => {
      if (!acc[node.layer]) {
        acc[node.layer] = {
          total: 0,
          healthy: 0,
          warning: 0,
          critical: 0,
          offline: 0,
          processing: 0
        };
      }

      acc[node.layer].total++;
      acc[node.layer][node.status]++;

      return acc;
    }, {} as Record<string, Record<ComponentStatus | 'total', number>>);

    // Calculate health score for each layer
    return Object.entries(layerStats).reduce((acc, [layer, stats]) => {
      const total = stats.total;
      const score = (
        stats.healthy * 1.0 +
        stats.processing * 0.9 +
        stats.warning * 0.6 +
        stats.critical * 0.2 +
        stats.offline * 0.0
      ) / total;

      let status: ComponentStatus = 'healthy';
      if (stats.critical > 0) {
        status = 'critical';
      } else if (stats.warning > 0 || stats.offline > 0) {
        status = 'warning';
      } else if (stats.processing > total * 0.5) {
        status = 'processing';
      }

      acc[layer] = {
        status,
        score,
        components: stats
      };

      return acc;
    }, {} as Record<string, { 
      status: ComponentStatus; 
      score: number; 
      components: Record<ComponentStatus | 'total', number> 
    }>);
  }, [data]);

  // Calculate component health by type
  const typeHealth = useMemo(() => {
    if (!data || !data.nodes) return {};

    const typeStats = data.nodes.reduce((acc, node) => {
      if (!acc[node.type]) {
        acc[node.type] = {
          total: 0,
          healthy: 0,
          warning: 0,
          critical: 0,
          offline: 0,
          processing: 0
        };
      }

      acc[node.type].total++;
      acc[node.type][node.status]++;

      return acc;
    }, {} as Record<string, Record<ComponentStatus | 'total', number>>);

    return Object.entries(typeStats).reduce((acc, [type, stats]) => {
      const total = stats.total;
      const score = (
        stats.healthy * 1.0 +
        stats.processing * 0.9 +
        stats.warning * 0.6 +
        stats.critical * 0.2 +
        stats.offline * 0.0
      ) / total;

      let status: ComponentStatus = 'healthy';
      if (stats.critical > 0) {
        status = 'critical';
      } else if (stats.warning > 0 || stats.offline > 0) {
        status = 'warning';
      } else if (stats.processing > total * 0.5) {
        status = 'processing';
      }

      acc[type] = {
        status,
        score,
        components: stats
      };

      return acc;
    }, {} as Record<string, {
      status: ComponentStatus;
      score: number;
      components: Record<ComponentStatus | 'total', number>
    }>);
  }, [data]);

  // Update health history periodically
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      const now = Date.now();
      const newEntry = { timestamp: now, score: systemHealth.score };

      setHealthHistory(prev => ({
        hourly: [...prev.hourly.slice(-59), newEntry], // Keep last 60 entries (1 hour)
        daily: prev.daily.length === 0 || now - prev.daily[prev.daily.length - 1].timestamp > 60000 
          ? [...prev.daily.slice(-1439), newEntry] // Keep last 1440 entries (24 hours)
          : prev.daily,
        weekly: prev.weekly.length === 0 || now - prev.weekly[prev.weekly.length - 1].timestamp > 3600000
          ? [...prev.weekly.slice(-167), newEntry] // Keep last 168 entries (1 week)
          : prev.weekly
      }));
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [systemHealth.score, isMonitoring]);

  // Get critical components that need immediate attention
  const criticalComponents = useMemo(() => {
    if (!data || !data.nodes) return [];

    return data.nodes
      .filter(node => node.status === 'critical')
      .map(node => ({
        id: node.id,
        label: node.label,
        type: node.type,
        layer: node.layer,
        status: node.status,
        metrics: node.metrics,
        lastSeen: node.metadata?.lastSeen || Date.now()
      }))
      .sort((a, b) => {
        // Sort by severity (based on metrics) and recency
        const aMetrics = a.metrics;
        const bMetrics = b.metrics;
        
        if (aMetrics && bMetrics) {
          const aSeverity = calculateComponentSeverity(aMetrics);
          const bSeverity = calculateComponentSeverity(bMetrics);
          if (aSeverity !== bSeverity) {
            return bSeverity - aSeverity;
          }
        }
        
        return b.lastSeen - a.lastSeen;
      });
  }, [data]);

  // Calculate health trend
  const healthTrend = useMemo(() => {
    if (healthHistory.hourly.length < 2) return 'stable';

    const recent = healthHistory.hourly.slice(-10);
    const older = healthHistory.hourly.slice(-20, -10);
    
    if (recent.length < 5 || older.length < 5) return 'stable';

    const recentAvg = recent.reduce((sum, entry) => sum + entry.score, 0) / recent.length;
    const olderAvg = older.reduce((sum, entry) => sum + entry.score, 0) / older.length;

    const change = (recentAvg - olderAvg) / olderAvg;

    if (change > 0.05) return 'improving';
    if (change < -0.05) return 'degrading';
    return 'stable';
  }, [healthHistory.hourly]);

  return {
    systemHealth,
    detailedRecommendations,
    layerHealth,
    typeHealth,
    criticalComponents,
    healthHistory,
    healthTrend,
    isMonitoring,
    setIsMonitoring
  };
}

// Helper function to generate health recommendations
function generateRecommendations(
  data: ArchitectureData,
  health: Partial<SystemHealth>
): HealthRecommendations[] {
  const recommendations: HealthRecommendations[] = [];

  if (!data || !data.nodes) return recommendations;

  // Critical components recommendation
  if (health.criticalComponents && health.criticalComponents > 0) {
    recommendations.push({
      priority: 'high',
      category: 'reliability',
      title: 'Critical Components Detected',
      description: `${health.criticalComponents} component(s) are in critical state and require immediate attention.`,
      action: 'Investigate and resolve critical component issues immediately.',
      affectedComponents: data.nodes
        .filter(n => n.status === 'critical')
        .map(n => n.id)
    });
  }

  // High resource usage recommendation
  const highCPUNodes = data.nodes.filter(n => 
    n.metrics && n.metrics.cpu.current > 80
  );
  
  if (highCPUNodes.length > 0) {
    recommendations.push({
      priority: 'medium',
      category: 'performance',
      title: 'High CPU Usage Detected',
      description: `${highCPUNodes.length} component(s) are experiencing high CPU usage (>80%).`,
      action: 'Monitor CPU usage and consider scaling or optimization.',
      affectedComponents: highCPUNodes.map(n => n.id)
    });
  }

  // Memory usage recommendation
  const highMemoryNodes = data.nodes.filter(n =>
    n.metrics && n.metrics.memory.current > 85
  );

  if (highMemoryNodes.length > 0) {
    recommendations.push({
      priority: 'medium',
      category: 'capacity',
      title: 'High Memory Usage Detected',
      description: `${highMemoryNodes.length} component(s) are experiencing high memory usage (>85%).`,
      action: 'Monitor memory usage and consider increasing capacity.',
      affectedComponents: highMemoryNodes.map(n => n.id)
    });
  }

  // Network connectivity issues
  const offlineNodes = data.nodes.filter(n => n.status === 'offline');
  if (offlineNodes.length > 0) {
    recommendations.push({
      priority: 'high',
      category: 'reliability',
      title: 'Offline Components',
      description: `${offlineNodes.length} component(s) are offline and not responding.`,
      action: 'Check network connectivity and restart components if necessary.',
      affectedComponents: offlineNodes.map(n => n.id)
    });
  }

  // Connection issues
  const activeConnections = data.connections?.filter(c => c.active).length || 0;
  const totalConnections = data.connections?.length || 0;
  
  if (totalConnections > 0 && activeConnections / totalConnections < 0.8) {
    recommendations.push({
      priority: 'medium',
      category: 'reliability',
      title: 'Connection Issues',
      description: `Only ${Math.round(activeConnections / totalConnections * 100)}% of connections are active.`,
      action: 'Investigate connection failures and network issues.',
    });
  }

  // Performance degradation
  const highLatencyNodes = data.nodes.filter(n =>
    n.metrics && n.metrics.latency.current > 200
  );

  if (highLatencyNodes.length > 0) {
    recommendations.push({
      priority: 'medium',
      category: 'performance',
      title: 'High Latency Detected',
      description: `${highLatencyNodes.length} component(s) are experiencing high latency (>200ms).`,
      action: 'Investigate performance bottlenecks and optimize processing.',
      affectedComponents: highLatencyNodes.map(n => n.id)
    });
  }

  // Error rate issues
  const highErrorNodes = data.nodes.filter(n =>
    n.metrics && n.metrics.errorRate.current > 5
  );

  if (highErrorNodes.length > 0) {
    recommendations.push({
      priority: 'high',
      category: 'reliability',
      title: 'High Error Rate Detected',
      description: `${highErrorNodes.length} component(s) are experiencing high error rates (>5%).`,
      action: 'Investigate error causes and implement fixes.',
      affectedComponents: highErrorNodes.map(n => n.id)
    });
  }

  // Sort recommendations by priority
  return recommendations.sort((a, b) => {
    const priorityOrder = { high: 3, medium: 2, low: 1 };
    return priorityOrder[b.priority] - priorityOrder[a.priority];
  });
}

// Helper function to calculate component severity based on metrics
function calculateComponentSeverity(metrics: any): number {
  let severity = 0;
  
  // CPU contribution
  if (metrics.cpu?.current > 90) severity += 3;
  else if (metrics.cpu?.current > 80) severity += 2;
  else if (metrics.cpu?.current > 70) severity += 1;

  // Memory contribution
  if (metrics.memory?.current > 95) severity += 3;
  else if (metrics.memory?.current > 85) severity += 2;
  else if (metrics.memory?.current > 75) severity += 1;

  // Error rate contribution
  if (metrics.errorRate?.current > 10) severity += 4;
  else if (metrics.errorRate?.current > 5) severity += 2;
  else if (metrics.errorRate?.current > 1) severity += 1;

  // Latency contribution
  if (metrics.latency?.current > 1000) severity += 2;
  else if (metrics.latency?.current > 500) severity += 1;

  return severity;
}