import { useState, useEffect, useRef, useCallback } from 'react';
import { PerformanceService } from '../services/PerformanceService';
import type { 
  PerformanceMetrics, 
  PerformanceAlert, 
  PerformanceTrend,
  PerformanceThresholds,
  UsePerformanceMetrics 
} from '../types';

const DEFAULT_THRESHOLDS: PerformanceThresholds = {
  cognitive: {
    maxLatency: 100,
    minThroughput: 100,
    maxErrorRate: 0.01,
    hebbianLearningRange: [0.1, 0.9]
  },
  sdr: {
    sparsityRange: [0.02, 0.05],
    maxMemoryUsage: 1000000000, // 1GB
    minCompressionRatio: 5
  },
  mcp: {
    maxLatency: 50,
    maxErrorRate: 0.01,
    maxQueueLength: 100
  },
  system: {
    maxCPU: 80,
    maxMemory: 80,
    maxDiskIO: 80,
    maxNetworkIO: 80
  }
};

export interface UsePerformanceMetricsOptions {
  websocketUrl: string;
  refreshInterval?: number;
  historySize?: number;
  thresholds?: PerformanceThresholds;
}

export function usePerformanceMetrics({
  websocketUrl,
  refreshInterval = 1000,
  historySize = 100,
  thresholds = DEFAULT_THRESHOLDS
}: UsePerformanceMetricsOptions): UsePerformanceMetrics {
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<PerformanceMetrics | null>(null);
  const [trends, setTrends] = useState<PerformanceTrend[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const serviceRef = useRef<PerformanceService | null>(null);

  useEffect(() => {
    // Initialize performance service
    try {
      const service = new PerformanceService({
        websocketUrl,
        historySize,
        thresholds
      });

      serviceRef.current = service;

      // Subscribe to metrics updates
      const unsubscribe = service.subscribe((newMetrics) => {
        setCurrentMetrics(newMetrics);
        setMetrics(prevMetrics => {
          const updated = [...prevMetrics, newMetrics];
          return updated.slice(-historySize);
        });
      });

      // Start periodic updates
      const updateInterval = setInterval(() => {
        // Update trends
        const newTrends = service.analyzePerformanceTrends();
        setTrends(newTrends);

        // Update alerts
        const newAlerts = service.getAlerts({ acknowledged: false });
        setAlerts(newAlerts);
      }, refreshInterval);

      // Initial data fetch
      setTimeout(() => {
        const history = service.getMetricsHistory();
        if (history.length > 0) {
          setMetrics(history);
          setCurrentMetrics(history[history.length - 1]);
        }
        setIsLoading(false);
      }, 1000);

      return () => {
        unsubscribe();
        clearInterval(updateInterval);
        service.dispose();
      };
    } catch (err) {
      setError(err as Error);
      setIsLoading(false);
    }
  }, [websocketUrl, refreshInterval, historySize]);

  const refresh = useCallback(() => {
    if (serviceRef.current) {
      const history = serviceRef.current.getMetricsHistory();
      setMetrics(history);
      if (history.length > 0) {
        setCurrentMetrics(history[history.length - 1]);
      }
      setTrends(serviceRef.current.analyzePerformanceTrends());
      setAlerts(serviceRef.current.getAlerts({ acknowledged: false }));
    }
  }, []);

  return {
    metrics,
    currentMetrics,
    trends,
    alerts,
    isLoading,
    error,
    refresh
  };
}

// Additional hooks for specific functionality

export function usePerformanceAlerts(severity?: PerformanceAlert['severity']) {
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const serviceRef = useRef<PerformanceService | null>(null);

  const acknowledgeAlert = useCallback((alertId: string) => {
    if (serviceRef.current) {
      serviceRef.current.acknowledgeAlert(alertId);
      setAlerts(prevAlerts => prevAlerts.filter(a => a.id !== alertId));
    }
  }, []);

  return {
    alerts,
    acknowledgeAlert
  };
}

export function usePerformanceSnapshots() {
  const serviceRef = useRef<PerformanceService | null>(null);

  const createSnapshot = useCallback((name: string, description?: string) => {
    if (serviceRef.current) {
      return serviceRef.current.createSnapshot(name, description);
    }
    throw new Error('Performance service not initialized');
  }, []);

  const compareSnapshots = useCallback((id1: string, id2: string) => {
    if (serviceRef.current) {
      return serviceRef.current.compareSnapshots(id1, id2);
    }
    throw new Error('Performance service not initialized');
  }, []);

  return {
    createSnapshot,
    compareSnapshots
  };
}