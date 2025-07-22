/**
 * Phase 5 Monitoring System - Entry Point
 * 
 * Exports all monitoring components and utilities for LLMKG real-time system monitoring.
 * This is the main entry point for the Phase 5 monitoring dashboard.
 */

// Core monitoring engine and system
export { RealTimeMonitor } from './RealTimeMonitor';
export { ComponentMonitor } from './ComponentMonitor';
export { AlertSystem } from './AlertSystem';

// Health visualization components
export {
  HealthVisualization,
  SystemHealthOverview,
  ComponentHealthGrid,
  CognitivePatternVisualization,
  MemorySystemVisualization,
  HealthStatusIcon,
  HealthScoreGauge,
  HealthTrendIndicator
} from './HealthVisualization';

// Performance metrics components
export {
  PerformanceMetricsDashboard,
  PerformanceChart,
  PerformanceHeatmap,
  MetricsCard,
  CognitivePerformanceAnalysis
} from './PerformanceMetrics';

// Main dashboard component
export {
  SystemDashboard,
  AlertPanel,
  DashboardLayout
} from './SystemDashboard';

// Type definitions
export * from '../types/MonitoringTypes';

// Default configuration for monitoring system
export const DEFAULT_MONITORING_CONFIG = {
  updateInterval: 100, // Sub-100ms updates for critical metrics
  websocketEndpoint: 'ws://localhost:8080/monitoring',
  enableCognitivePatterns: true,
  enableBrainComponents: true,
  enableMCPToolMonitoring: true,
  enableMemorySystemMonitoring: true,
  enableFederationMonitoring: true,
  maxHistorySize: 1000,
  alertThresholds: {
    cpu: { warning: 70, critical: 90 },
    memory: { warning: 80, critical: 95 },
    latency: { warning: 1000, critical: 5000 },
    errorRate: { warning: 0.05, critical: 0.1 },
    healthScore: { warning: 60, critical: 30 },
    cognitivePatterns: {
      maxActivationTime: 30000, // 30 seconds
      minActivationLevel: 0.1,
      maxInhibitionImbalance: 0.8
    },
    memorySystem: {
      maxUtilization: 0.9,
      minHitRate: 0.7,
      maxFragmentation: 0.5
    },
    federation: {
      minConnectionQuality: 0.8,
      maxSyncLatency: 1000,
      minTrustScore: 0.6
    }
  }
};

// Utility functions for monitoring system integration
export const createMonitoringSystem = async (config?: any) => {
  const realTimeMonitor = new RealTimeMonitor();
  const alertSystem = new AlertSystem(config?.alertThresholds || DEFAULT_MONITORING_CONFIG.alertThresholds);
  
  await realTimeMonitor.startMonitoring({
    ...DEFAULT_MONITORING_CONFIG,
    ...config
  });
  
  return {
    realTimeMonitor,
    alertSystem
  };
};

// Export monitoring system factory
export const MonitoringSystemFactory = {
  createRealTimeMonitor: (config: any) => new RealTimeMonitor(),
  createAlertSystem: (thresholds: any) => new AlertSystem(thresholds),
  createDefaultConfig: () => DEFAULT_MONITORING_CONFIG
};