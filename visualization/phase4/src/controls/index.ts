/**
 * LLMKG Phase 4 Controls - Main exports
 * Interactive controls and filtering system for comprehensive visualization management
 */

// Main control components
export { default as VisualizationControls } from './VisualizationControls';
export { default as PerformanceMonitor } from './PerformanceMonitor';
export { default as DebugConsole } from './DebugConsole';

// Core systems
export { filteringSystem, FilteringSystem } from './FilteringSystem';
export { exportTools } from './ExportTools';

// Type definitions
export type {
  FilterCondition,
  FilterGroup,
  FilterPreset,
  TimeWindow,
  FilteringState
} from './FilteringSystem';

export type {
  ScreenshotOptions,
  VideoOptions,
  DataExportOptions,
  ReportOptions,
  AnnotationData
} from './ExportTools';

// Utility functions
export const createFilterGroup = (name: string) => filteringSystem.addFilterGroup(name);
export const applyFilters = (data: any[]) => filteringSystem.applyFilters(data);
export const captureScreenshot = (element: HTMLElement, options?: any) => 
  exportTools.captureScreenshot(element, options);
export const exportData = (data: any[], options?: any) => 
  exportTools.exportData(data, options);

// Default configurations
export const defaultVisualizationSettings = {
  layers: {
    mcpRequests: true,
    cognitivePatterns: true,
    memoryOperations: true,
    performanceMetrics: false,
    connections: true,
    particles: true,
    heatmaps: false,
    timelines: false
  },
  quality: {
    particleCount: 5000,
    renderQuality: 'high' as const,
    antiAliasing: true,
    shadows: true,
    postProcessing: true
  },
  camera: {
    smoothTransitions: true,
    followMouse: false,
    autoRotate: false
  },
  theme: {
    current: 'dark',
    accessibility: {
      highContrast: false,
      reducedMotion: false,
      colorBlindSafe: false
    }
  },
  playback: {
    isPlaying: false,
    speed: 1.0,
    currentTime: new Date(),
    startTime: new Date(Date.now() - 3600000),
    endTime: new Date()
  }
};

export const defaultOptimizationSettings = {
  autoOptimize: true,
  targetFPS: 60,
  maxMemoryUsage: 512,
  enableAdaptiveQuality: true,
  enableGPUOptimization: true,
  enableNetworkOptimization: true,
  alertThresholds: {
    lowFPS: 30,
    highMemory: 80,
    highGPU: 85,
    slowRender: 33.33
  }
};