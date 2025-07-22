/**
 * LLMKG Phase 4 Data Flow Visualization Engine
 * Entry point for the complete visualization system
 */

// Core engine exports
export { 
  LLMKGDataFlowVisualizer,
  type VisualizationConfig,
  type DataFlowNode,
  type DataFlowConnection,
  type CognitivePattern
} from './core/LLMKGDataFlowVisualizer';

export {
  ParticleSystem,
  type ParticleConfig,
  type DataFlowParticle
} from './core/ParticleSystem';

export {
  ShaderLibrary,
  type ShaderConfig
} from './core/ShaderLibrary';

// React components and hooks
export {
  DataFlowCanvas,
  type DataFlowCanvasProps,
  createDemoNode,
  createDemoConnection,
  createDemoCognitivePattern
} from './components/DataFlowCanvas';

export {
  useVisualizationEngine,
  type VisualizationState,
  type WebSocketData,
  type UseVisualizationEngineOptions
} from './hooks/useVisualizationEngine';

// Memory visualization exports
export {
  SDRVisualizer,
  MemoryOperationVisualizer,
  StorageEfficiency,
  MemoryAnalytics,
  MemoryVisualizationSystem,
  type SDRConfig,
  type SDRPattern,
  type SDRVisualizationConfig,
  type SDRComparisonResult,
  type MemoryOperation,
  type MemoryBlock,
  type MemoryVisualizationConfig,
  type MemoryStats,
  type StorageMetrics,
  type StorageBlock,
  type StorageEfficiencyConfig,
  type EfficiencyTrend,
  type MemoryPerformanceMetrics,
  type MemoryInsight,
  type MemoryPattern,
  type AnalyticsConfig,
  type MemoryVisualizationSystemConfig
} from './memory';

// Tracing and request visualization exports
export {
  MCPRequestTracer,
  RequestPathVisualizer,
  TraceAnalytics,
  ParticleEffects,
  type TraceConfig,
  type RequestTrace,
  type TraceNode,
  type TraceConnection,
  type PathVisualizationConfig,
  type AnalyticsData
} from './tracing';

// Cognitive pattern visualization exports
export {
  CognitivePatternVisualizer,
  PatternEffects,
  PatternInteractions,
  type CognitiveVisualizationConfig,
  type PatternVisualizationData
} from './cognitive';

// Re-export Three.js types that users might need
export { Vector3, Color, Matrix4 } from 'three';

// Version and metadata
export const PHASE4_VERSION = '1.0.0';
export const SUPPORTED_WEBGL_VERSION = '2.0';
export const MIN_FPS_TARGET = 30;
export const OPTIMAL_FPS_TARGET = 60;