import { useCallback, useEffect, useRef } from 'react';
import { performanceMonitor, getOptimizationSuggestions } from '../utils/performance';

export function usePerformanceMonitoring() {
  const isInitialized = useRef(false);

  const startMonitoring = useCallback(() => {
    if (!isInitialized.current) {
      performanceMonitor.init();
      isInitialized.current = true;
    }
  }, []);

  const stopMonitoring = useCallback(() => {
    if (isInitialized.current) {
      performanceMonitor.cleanup();
      isInitialized.current = false;
    }
  }, []);

  const getPerformanceData = useCallback(() => {
    return {
      entries: performanceMonitor.getEntries(),
      vitals: performanceMonitor.getVitals(),
      summary: performanceMonitor.getPerformanceSummary(),
      suggestions: getOptimizationSuggestions(),
    };
  }, []);

  const measureFunction = useCallback((name: string, fn: () => void) => {
    return performanceMonitor.measure(name, fn);
  }, []);

  const measureAsyncFunction = useCallback(async (name: string, fn: () => Promise<void>) => {
    return performanceMonitor.measureAsync(name, fn);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMonitoring();
    };
  }, [stopMonitoring]);

  return {
    startMonitoring,
    stopMonitoring,
    getPerformanceData,
    measureFunction,
    measureAsyncFunction,
  };
}