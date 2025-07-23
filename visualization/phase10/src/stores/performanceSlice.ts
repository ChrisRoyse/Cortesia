import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { PerformanceMetrics, PerformanceAlert } from '@/types';

interface PerformanceState {
  current: PerformanceMetrics;
  history: {
    timestamp: number;
    metrics: PerformanceMetrics;
  }[];
  alerts: PerformanceAlert[];
  profiling: {
    enabled: boolean;
    sampleRate: number;
    startTime: number | null;
    endTime: number | null;
  };
  thresholds: {
    renderTime: number;
    memoryUsage: number;
    networkLatency: number;
    dataProcessingTime: number;
  };
  benchmarks: {
    name: string;
    duration: number;
    results: Record<string, number>;
    timestamp: number;
  }[];
}

const initialState: PerformanceState = {
  current: {
    renderTime: 0,
    componentCount: 0,
    memoryUsage: 0,
    updateFrequency: 0,
    dataProcessingTime: 0,
    networkLatency: 0,
  },
  history: [],
  alerts: [],
  profiling: {
    enabled: false,
    sampleRate: 1,
    startTime: null,
    endTime: null,
  },
  thresholds: {
    renderTime: 16, // 60fps
    memoryUsage: 512 * 1024 * 1024, // 512MB
    networkLatency: 200, // 200ms
    dataProcessingTime: 100, // 100ms
  },
  benchmarks: [],
};

const performanceSlice = createSlice({
  name: 'performance',
  initialState,
  reducers: {
    updateMetrics: (state, action: PayloadAction<Partial<PerformanceMetrics>>) => {
      const newMetrics = { ...state.current, ...action.payload };
      state.current = newMetrics;
      
      // Add to history
      state.history.push({
        timestamp: Date.now(),
        metrics: newMetrics,
      });
      
      // Keep only last 1000 entries
      if (state.history.length > 1000) {
        state.history = state.history.slice(-1000);
      }
      
      // Check thresholds and create alerts
      if (action.payload.renderTime && action.payload.renderTime > state.thresholds.renderTime) {
        state.alerts.push({
          id: `render-${Date.now()}`,
          type: 'render',
          threshold: state.thresholds.renderTime,
          currentValue: action.payload.renderTime,
          message: `Render time exceeded threshold: ${action.payload.renderTime}ms > ${state.thresholds.renderTime}ms`,
          timestamp: Date.now(),
        });
      }
      
      if (action.payload.memoryUsage && action.payload.memoryUsage > state.thresholds.memoryUsage) {
        state.alerts.push({
          id: `memory-${Date.now()}`,
          type: 'memory',
          threshold: state.thresholds.memoryUsage,
          currentValue: action.payload.memoryUsage,
          message: `Memory usage exceeded threshold: ${Math.round(action.payload.memoryUsage / 1024 / 1024)}MB > ${Math.round(state.thresholds.memoryUsage / 1024 / 1024)}MB`,
          timestamp: Date.now(),
        });
      }
      
      if (action.payload.networkLatency && action.payload.networkLatency > state.thresholds.networkLatency) {
        state.alerts.push({
          id: `network-${Date.now()}`,
          type: 'network',
          threshold: state.thresholds.networkLatency,
          currentValue: action.payload.networkLatency,
          message: `Network latency exceeded threshold: ${action.payload.networkLatency}ms > ${state.thresholds.networkLatency}ms`,
          timestamp: Date.now(),
        });
      }
      
      if (action.payload.dataProcessingTime && action.payload.dataProcessingTime > state.thresholds.dataProcessingTime) {
        state.alerts.push({
          id: `data-${Date.now()}`,
          type: 'data',
          threshold: state.thresholds.dataProcessingTime,
          currentValue: action.payload.dataProcessingTime,
          message: `Data processing time exceeded threshold: ${action.payload.dataProcessingTime}ms > ${state.thresholds.dataProcessingTime}ms`,
          timestamp: Date.now(),
        });
      }
      
      // Keep only last 100 alerts
      if (state.alerts.length > 100) {
        state.alerts = state.alerts.slice(-100);
      }
    },
    
    addAlert: (state, action: PayloadAction<Omit<PerformanceAlert, 'id' | 'timestamp'>>) => {
      state.alerts.push({
        ...action.payload,
        id: `alert-${Date.now()}`,
        timestamp: Date.now(),
      });
    },
    
    removeAlert: (state, action: PayloadAction<string>) => {
      state.alerts = state.alerts.filter(alert => alert.id !== action.payload);
    },
    
    clearAlerts: (state) => {
      state.alerts = [];
    },
    
    updateThresholds: (state, action: PayloadAction<Partial<PerformanceState['thresholds']>>) => {
      state.thresholds = { ...state.thresholds, ...action.payload };
    },
    
    startProfiling: (state, action: PayloadAction<{ sampleRate?: number }>) => {
      state.profiling.enabled = true;
      state.profiling.sampleRate = action.payload.sampleRate || 1;
      state.profiling.startTime = Date.now();
      state.profiling.endTime = null;
    },
    
    stopProfiling: (state) => {
      state.profiling.enabled = false;
      state.profiling.endTime = Date.now();
    },
    
    addBenchmark: (state, action: PayloadAction<{
      name: string;
      results: Record<string, number>;
    }>) => {
      const benchmark = {
        name: action.payload.name,
        duration: 0,
        results: action.payload.results,
        timestamp: Date.now(),
      };
      
      if (state.profiling.startTime && state.profiling.endTime) {
        benchmark.duration = state.profiling.endTime - state.profiling.startTime;
      }
      
      state.benchmarks.push(benchmark);
      
      // Keep only last 50 benchmarks
      if (state.benchmarks.length > 50) {
        state.benchmarks = state.benchmarks.slice(-50);
      }
    },
    
    clearBenchmarks: (state) => {
      state.benchmarks = [];
    },
    
    clearHistory: (state) => {
      state.history = [];
    },
    
    resetPerformance: (state) => {
      state.current = {
        renderTime: 0,
        componentCount: 0,
        memoryUsage: 0,
        updateFrequency: 0,
        dataProcessingTime: 0,
        networkLatency: 0,
      };
      state.history = [];
      state.alerts = [];
      state.benchmarks = [];
      state.profiling = {
        enabled: false,
        sampleRate: 1,
        startTime: null,
        endTime: null,
      };
    },
    
    optimizePerformance: (state) => {
      // Auto-optimization actions
      if (state.current.memoryUsage > state.thresholds.memoryUsage * 0.8) {
        // Trigger memory cleanup
        state.alerts.push({
          id: `optimize-${Date.now()}`,
          type: 'memory',
          threshold: state.thresholds.memoryUsage,
          currentValue: state.current.memoryUsage,
          message: 'Automatic memory optimization triggered',
          timestamp: Date.now(),
        });
      }
      
      if (state.current.renderTime > state.thresholds.renderTime * 0.8) {
        // Suggest render optimizations
        state.alerts.push({
          id: `optimize-render-${Date.now()}`,
          type: 'render',
          threshold: state.thresholds.renderTime,
          currentValue: state.current.renderTime,
          message: 'Consider reducing animation complexity or enabling lazy loading',
          timestamp: Date.now(),
        });
      }
    },
  },
});

export const {
  updateMetrics,
  addAlert,
  removeAlert,
  clearAlerts,
  updateThresholds,
  startProfiling,
  stopProfiling,
  addBenchmark,
  clearBenchmarks,
  clearHistory,
  resetPerformance,
  optimizePerformance,
} = performanceSlice.actions;

export default performanceSlice.reducer;

// Selectors
export const selectCurrentMetrics = (state: { performance: PerformanceState }) => 
  state.performance.current;

export const selectPerformanceHistory = (state: { performance: PerformanceState }) => 
  state.performance.history;

export const selectRecentHistory = (minutes: number = 10) => 
  (state: { performance: PerformanceState }) => {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    return state.performance.history.filter(entry => entry.timestamp > cutoff);
  };

export const selectPerformanceAlerts = (state: { performance: PerformanceState }) => 
  state.performance.alerts;

export const selectRecentAlerts = (hours: number = 1) => 
  (state: { performance: PerformanceState }) => {
    const cutoff = Date.now() - (hours * 60 * 60 * 1000);
    return state.performance.alerts.filter(alert => alert.timestamp > cutoff);
  };

export const selectAlertsByType = (type: string) => 
  (state: { performance: PerformanceState }) =>
    state.performance.alerts.filter(alert => alert.type === type);

export const selectProfiling = (state: { performance: PerformanceState }) => 
  state.performance.profiling;

export const selectThresholds = (state: { performance: PerformanceState }) => 
  state.performance.thresholds;

export const selectBenchmarks = (state: { performance: PerformanceState }) => 
  state.performance.benchmarks;

export const selectRecentBenchmarks = (count: number = 10) => 
  (state: { performance: PerformanceState }) =>
    state.performance.benchmarks.slice(-count);

export const selectAverageMetrics = (minutes: number = 5) => 
  (state: { performance: PerformanceState }) => {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentEntries = state.performance.history.filter(entry => entry.timestamp > cutoff);
    
    if (recentEntries.length === 0) return state.performance.current;
    
    const avg = recentEntries.reduce(
      (acc, entry) => ({
        renderTime: acc.renderTime + entry.metrics.renderTime,
        componentCount: acc.componentCount + entry.metrics.componentCount,
        memoryUsage: acc.memoryUsage + entry.metrics.memoryUsage,
        updateFrequency: acc.updateFrequency + entry.metrics.updateFrequency,
        dataProcessingTime: acc.dataProcessingTime + entry.metrics.dataProcessingTime,
        networkLatency: acc.networkLatency + entry.metrics.networkLatency,
      }),
      {
        renderTime: 0,
        componentCount: 0,
        memoryUsage: 0,
        updateFrequency: 0,
        dataProcessingTime: 0,
        networkLatency: 0,
      }
    );
    
    const count = recentEntries.length;
    return {
      renderTime: avg.renderTime / count,
      componentCount: Math.round(avg.componentCount / count),
      memoryUsage: avg.memoryUsage / count,
      updateFrequency: avg.updateFrequency / count,
      dataProcessingTime: avg.dataProcessingTime / count,
      networkLatency: avg.networkLatency / count,
    };
  };