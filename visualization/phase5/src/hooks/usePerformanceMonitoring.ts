import { useState, useEffect, useMemo, useCallback } from 'react';
import { PerformanceMetrics, Bottleneck } from '../types';

export interface PerformanceConfig {
  sampleInterval: number;
  historySize: number;
  memoryThreshold: number;
  fpsThreshold: number;
  renderTimeThreshold: number;
}

export interface PerformanceHistory {
  timestamp: number;
  fps: number;
  renderTime: number;
  memoryUsage: number;
  nodeCount: number;
  connectionCount: number;
}

const defaultConfig: PerformanceConfig = {
  sampleInterval: 1000, // 1 second
  historySize: 300, // 5 minutes of history
  memoryThreshold: 100 * 1024 * 1024, // 100MB
  fpsThreshold: 30,
  renderTimeThreshold: 100, // 100ms
};

export function usePerformanceMonitoring(config: Partial<PerformanceConfig> = {}) {
  const finalConfig = { ...defaultConfig, ...config };

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    animationFPS: 60,
    memoryUsage: 0,
    nodeCount: 0,
    connectionCount: 0,
    lastMeasurement: Date.now()
  });

  const [performanceHistory, setPerformanceHistory] = useState<PerformanceHistory[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [fpsCounter, setFpsCounter] = useState({ frames: 0, lastTime: Date.now() });

  // Frame rate monitoring
  const updateFPS = useCallback(() => {
    if (!isMonitoring) return;

    const now = Date.now();
    const deltaTime = now - fpsCounter.lastTime;

    if (deltaTime >= 1000) { // Update every second
      const fps = Math.round((fpsCounter.frames * 1000) / deltaTime);
      
      setPerformanceMetrics(prev => ({
        ...prev,
        animationFPS: fps,
        lastMeasurement: now
      }));

      setFpsCounter({ frames: 0, lastTime: now });
    } else {
      setFpsCounter(prev => ({ ...prev, frames: prev.frames + 1 }));
    }

    requestAnimationFrame(updateFPS);
  }, [isMonitoring, fpsCounter.lastTime, fpsCounter.frames]);

  // Memory monitoring
  const updateMemoryUsage = useCallback(() => {
    if (!isMonitoring) return;

    try {
      const memory = (performance as any).memory;
      if (memory) {
        setPerformanceMetrics(prev => ({
          ...prev,
          memoryUsage: memory.usedJSHeapSize,
          lastMeasurement: Date.now()
        }));
      }
    } catch (error) {
      console.warn('Memory monitoring not available:', error);
    }
  }, [isMonitoring]);

  // Render time monitoring
  const measureRenderTime = useCallback((renderTime: number) => {
    if (!isMonitoring) return;

    setPerformanceMetrics(prev => ({
      ...prev,
      renderTime,
      lastMeasurement: Date.now()
    }));
  }, [isMonitoring]);

  // Update node and connection counts
  const updateCounts = useCallback((nodeCount: number, connectionCount: number) => {
    setPerformanceMetrics(prev => ({
      ...prev,
      nodeCount,
      connectionCount,
      lastMeasurement: Date.now()
    }));
  }, []);

  // Start monitoring
  useEffect(() => {
    if (!isMonitoring) return;

    // Start FPS monitoring
    requestAnimationFrame(updateFPS);

    // Start memory monitoring
    const memoryInterval = setInterval(updateMemoryUsage, finalConfig.sampleInterval);

    return () => {
      clearInterval(memoryInterval);
    };
  }, [isMonitoring, updateFPS, updateMemoryUsage, finalConfig.sampleInterval]);

  // Update performance history
  useEffect(() => {
    if (!isMonitoring) return;

    const interval = setInterval(() => {
      const historyEntry: PerformanceHistory = {
        timestamp: Date.now(),
        fps: performanceMetrics.animationFPS,
        renderTime: performanceMetrics.renderTime,
        memoryUsage: performanceMetrics.memoryUsage,
        nodeCount: performanceMetrics.nodeCount,
        connectionCount: performanceMetrics.connectionCount
      };

      setPerformanceHistory(prev => {
        const newHistory = [...prev, historyEntry];
        // Keep only recent history
        return newHistory.slice(-finalConfig.historySize);
      });
    }, finalConfig.sampleInterval);

    return () => clearInterval(interval);
  }, [isMonitoring, performanceMetrics, finalConfig.sampleInterval, finalConfig.historySize]);

  // Calculate performance statistics
  const performanceStats = useMemo(() => {
    if (performanceHistory.length === 0) {
      return {
        averageFPS: 60,
        minFPS: 60,
        maxFPS: 60,
        averageRenderTime: 0,
        maxRenderTime: 0,
        averageMemory: 0,
        maxMemory: 0,
        memoryTrend: 'stable' as const,
        fpsTrend: 'stable' as const,
        renderTimeTrend: 'stable' as const
      };
    }

    const recent = performanceHistory.slice(-30); // Last 30 samples
    const older = performanceHistory.slice(-60, -30); // Previous 30 samples

    // FPS statistics
    const fpsValues = performanceHistory.map(h => h.fps);
    const averageFPS = fpsValues.reduce((a, b) => a + b, 0) / fpsValues.length;
    const minFPS = Math.min(...fpsValues);
    const maxFPS = Math.max(...fpsValues);

    // Render time statistics
    const renderTimes = performanceHistory.map(h => h.renderTime);
    const averageRenderTime = renderTimes.reduce((a, b) => a + b, 0) / renderTimes.length;
    const maxRenderTime = Math.max(...renderTimes);

    // Memory statistics
    const memoryValues = performanceHistory.map(h => h.memoryUsage);
    const averageMemory = memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length;
    const maxMemory = Math.max(...memoryValues);

    // Calculate trends
    const memoryTrend = calculateTrend(
      recent.map(h => h.memoryUsage),
      older.map(h => h.memoryUsage)
    );

    const fpsTrend = calculateTrend(
      recent.map(h => h.fps),
      older.map(h => h.fps)
    );

    const renderTimeTrend = calculateTrend(
      recent.map(h => h.renderTime),
      older.map(h => h.renderTime)
    );

    return {
      averageFPS: Math.round(averageFPS),
      minFPS: Math.round(minFPS),
      maxFPS: Math.round(maxFPS),
      averageRenderTime: Math.round(averageRenderTime),
      maxRenderTime: Math.round(maxRenderTime),
      averageMemory: Math.round(averageMemory),
      maxMemory: Math.round(maxMemory),
      memoryTrend,
      fpsTrend,
      renderTimeTrend
    };
  }, [performanceHistory]);

  // Detect performance bottlenecks
  const bottlenecks = useMemo((): Bottleneck[] => {
    const issues: Bottleneck[] = [];
    const now = Date.now();

    // Low FPS
    if (performanceMetrics.animationFPS < finalConfig.fpsThreshold) {
      issues.push({
        id: 'low-fps',
        component: 'Animation Engine',
        type: 'rendering',
        severity: performanceMetrics.animationFPS < 15 ? 'high' : 'medium',
        description: `Frame rate is ${performanceMetrics.animationFPS} FPS, below threshold of ${finalConfig.fpsThreshold} FPS`,
        recommendation: 'Reduce animation complexity or disable animations for better performance',
        timestamp: now
      });
    }

    // High render time
    if (performanceMetrics.renderTime > finalConfig.renderTimeThreshold) {
      issues.push({
        id: 'high-render-time',
        component: 'Rendering Engine',
        type: 'rendering',
        severity: performanceMetrics.renderTime > 200 ? 'high' : 'medium',
        description: `Render time is ${performanceMetrics.renderTime}ms, above threshold of ${finalConfig.renderTimeThreshold}ms`,
        recommendation: 'Optimize rendering logic or reduce the number of rendered elements',
        timestamp: now
      });
    }

    // High memory usage
    if (performanceMetrics.memoryUsage > finalConfig.memoryThreshold) {
      issues.push({
        id: 'high-memory',
        component: 'Memory Management',
        type: 'memory',
        severity: performanceMetrics.memoryUsage > finalConfig.memoryThreshold * 2 ? 'high' : 'medium',
        description: `Memory usage is ${Math.round(performanceMetrics.memoryUsage / 1024 / 1024)}MB, above threshold of ${Math.round(finalConfig.memoryThreshold / 1024 / 1024)}MB`,
        recommendation: 'Check for memory leaks and optimize data structures',
        timestamp: now
      });
    }

    // Too many nodes/connections
    if (performanceMetrics.nodeCount > 200) {
      issues.push({
        id: 'too-many-nodes',
        component: 'Layout Engine',
        type: 'cpu',
        severity: performanceMetrics.nodeCount > 500 ? 'high' : 'medium',
        description: `${performanceMetrics.nodeCount} nodes are being rendered, which may impact performance`,
        recommendation: 'Consider filtering or pagination for large datasets',
        timestamp: now
      });
    }

    if (performanceMetrics.connectionCount > 1000) {
      issues.push({
        id: 'too-many-connections',
        component: 'Connection Renderer',
        type: 'rendering',
        severity: performanceMetrics.connectionCount > 2000 ? 'high' : 'medium',
        description: `${performanceMetrics.connectionCount} connections are being rendered, which may impact performance`,
        recommendation: 'Consider hiding connections at high zoom levels or implementing level-of-detail rendering',
        timestamp: now
      });
    }

    // Performance trends
    if (performanceStats.memoryTrend === 'degrading') {
      issues.push({
        id: 'memory-leak',
        component: 'Memory Management',
        type: 'memory',
        severity: 'medium',
        description: 'Memory usage is steadily increasing, indicating a possible memory leak',
        recommendation: 'Investigate memory usage patterns and ensure proper cleanup',
        timestamp: now
      });
    }

    if (performanceStats.fpsTrend === 'degrading') {
      issues.push({
        id: 'performance-degradation',
        component: 'Overall Performance',
        type: 'rendering',
        severity: 'medium',
        description: 'Frame rate is steadily decreasing over time',
        recommendation: 'Monitor system resources and optimize performance-critical code paths',
        timestamp: now
      });
    }

    return issues.sort((a, b) => {
      const severityOrder = { high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }, [performanceMetrics, finalConfig, performanceStats]);

  // Performance optimization suggestions
  const optimizationSuggestions = useMemo(() => {
    const suggestions = [];

    if (performanceMetrics.nodeCount > 100) {
      suggestions.push('Enable node culling for off-screen elements');
      suggestions.push('Implement level-of-detail rendering based on zoom level');
    }

    if (performanceMetrics.connectionCount > 500) {
      suggestions.push('Hide connections when zoomed out');
      suggestions.push('Use simplified connection rendering at low zoom levels');
    }

    if (performanceMetrics.animationFPS < 45) {
      suggestions.push('Reduce animation complexity');
      suggestions.push('Use CSS transforms instead of JavaScript animations when possible');
      suggestions.push('Implement animation priority system');
    }

    if (performanceMetrics.renderTime > 50) {
      suggestions.push('Optimize SVG rendering with fewer DOM elements');
      suggestions.push('Use Canvas rendering for better performance');
      suggestions.push('Implement virtual scrolling for large datasets');
    }

    return suggestions;
  }, [performanceMetrics]);

  // Performance monitoring controls
  const startMonitoring = useCallback(() => {
    setIsMonitoring(true);
  }, []);

  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
  }, []);

  const clearHistory = useCallback(() => {
    setPerformanceHistory([]);
  }, []);

  return {
    performanceMetrics,
    performanceHistory,
    performanceStats,
    bottlenecks,
    optimizationSuggestions,
    isMonitoring,
    measureRenderTime,
    updateCounts,
    startMonitoring,
    stopMonitoring,
    clearHistory,
  };
}

// Helper function to calculate trend
function calculateTrend(recent: number[], older: number[]): 'improving' | 'stable' | 'degrading' {
  if (recent.length === 0 || older.length === 0) return 'stable';

  const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
  const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;

  const change = (recentAvg - olderAvg) / olderAvg;

  if (Math.abs(change) < 0.05) return 'stable'; // Less than 5% change
  return change > 0 ? 'degrading' : 'improving'; // Assuming higher values are worse
}