// Phase 5 Custom React Hooks
// All hooks for system architecture visualization

export { useArchitectureMetrics } from './useArchitectureMetrics';
export { useSystemHealth } from './useSystemHealth';
export { useRealTimeUpdates } from './useRealTimeUpdates';
export { usePerformanceMonitoring } from './usePerformanceMonitoring';

// Re-export hook return types
export type {
  ArchitectureMetrics,
} from './useArchitectureMetrics';

export type {
  HealthTrends,
  HealthRecommendations,
} from './useSystemHealth';

export type {
  RealTimeConfig,
} from './useRealTimeUpdates';

export type {
  PerformanceConfig,
  PerformanceHistory,
} from './usePerformanceMonitoring';