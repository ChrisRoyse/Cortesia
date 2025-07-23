// Main entry point for LLMKG Unified Visualization System - Phase 10
export * from './components/UnifiedDashboard';
export * from './components/SystemOverview';
export * from './components/ConnectionStatus';
export * from './components/ComponentRegistry';
export * from './integration/VisualizationCore';
export * from './types';
export * from './stores';

// Re-export components from other phases for easy access
export { MemoryDashboard } from '@phase7/components/MemoryDashboard';
export { SDRStorageVisualization } from '@phase7/components/SDRStorageVisualization';
export { KnowledgeGraphTreemap } from '@phase7/components/KnowledgeGraphTreemap';
export { ZeroCopyMonitor } from '@phase7/components/ZeroCopyMonitor';
export { MemoryFlowVisualization } from '@phase7/components/MemoryFlowVisualization';
export { CognitiveLayerMemory } from '@phase7/components/CognitiveLayerMemory';

export { CognitivePatternDashboard } from '@phase8/components/CognitivePatternDashboard';
export { PatternActivation3D } from '@phase8/components/PatternActivation3D';
export { PatternClassification } from '@phase8/components/PatternClassification';
export { InhibitionExcitationBalance } from '@phase8/components/InhibitionExcitationBalance';
export { TemporalPatternAnalysis } from '@phase8/components/TemporalPatternAnalysis';

export { DebuggingDashboard } from '@phase9/components/DebuggingDashboard';
export { DistributedTracing } from '@phase9/components/DistributedTracing';
export { TimeTravelDebugger } from '@phase9/components/TimeTravelDebugger';
export { QueryAnalyzer } from '@phase9/components/QueryAnalyzer';
export { ErrorLoggingDashboard } from '@phase9/components/ErrorLoggingDashboard';

// Default configuration
export const defaultConfig = {
  mcp: {
    endpoint: 'ws://localhost:8080',
    protocol: 'ws' as const,
    reconnect: {
      enabled: true,
      maxAttempts: 5,
      delay: 5000,
    },
  },
  visualization: {
    theme: 'dark' as const,
    updateInterval: 1000,
    maxDataPoints: 1000,
    enableAnimations: true,
    enableDebugMode: false,
  },
  performance: {
    enableProfiling: false,
    sampleRate: 1,
    maxMemoryUsage: 512 * 1024 * 1024,
    enableLazyLoading: true,
  },
  features: {
    enabledPhases: ['phase7', 'phase8', 'phase9', 'phase10'],
    experimentalFeatures: [],
  },
};

// Version information
export const VERSION = '1.0.0';
export const BUILD_DATE = new Date().toISOString();
export const SUPPORTED_PHASES = ['phase7', 'phase8', 'phase9', 'phase10', 'phase11'];

// Utility functions
export const createVisualizationConfig = (overrides = {}) => ({
  ...defaultConfig,
  ...overrides,
});

export const isPhaseEnabled = (phase: string, config: any) => 
  config.features?.enabledPhases?.includes(phase) ?? false;

export const getEnabledComponents = (phase: string) => {
  // This would typically query the component registry
  // For now, return mock data
  return [];
};