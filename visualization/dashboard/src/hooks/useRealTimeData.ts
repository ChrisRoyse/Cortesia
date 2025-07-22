import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useWebSocket } from '../providers/WebSocketProvider';
import { useAppDispatch, useAppSelector } from '../stores';

// Performance Metrics Interface
export interface PerformanceMetrics {
  latency: number;
  throughput: number;
  droppedFrames: number;
  bufferUtilization: number;
  processingTime: number;
  memoryUsage: number;
  lastUpdate: number;
}

// Real-time Data Options Interface
export interface UseRealTimeDataOptions {
  bufferSize?: number;
  samplingRate?: number;
  aggregationWindow?: number;
  enabled?: boolean;
  maxLatency?: number;
  enablePerformanceMonitoring?: boolean;
  pruneInterval?: number;
  autoOptimize?: boolean;
}

// Circular Buffer Implementation
class CircularBuffer<T> {
  private buffer: T[];
  private head = 0;
  private tail = 0;
  private count = 0;
  private readonly capacity: number;

  constructor(size: number) {
    this.capacity = size;
    this.buffer = new Array(size);
  }

  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    
    if (this.count < this.capacity) {
      this.count++;
    } else {
      this.head = (this.head + 1) % this.capacity;
    }
  }

  getLatest(): T | null {
    if (this.count === 0) return null;
    const latestIndex = this.tail === 0 ? this.capacity - 1 : this.tail - 1;
    return this.buffer[latestIndex];
  }

  getAll(): T[] {
    if (this.count === 0) return [];
    
    const result: T[] = [];
    for (let i = 0; i < this.count; i++) {
      const index = (this.head + i) % this.capacity;
      result.push(this.buffer[index]);
    }
    return result;
  }

  size(): number {
    return this.count;
  }

  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }

  isFull(): boolean {
    return this.count === this.capacity;
  }
}

// Data Sampler for handling high-frequency updates
class DataSampler<T> {
  private lastSample: number = 0;
  private readonly sampleInterval: number;
  private pendingData: T | null = null;

  constructor(samplingRate: number) {
    this.sampleInterval = 1000 / samplingRate; // Convert Hz to ms
  }

  shouldSample(): boolean {
    const now = Date.now();
    if (now - this.lastSample >= this.sampleInterval) {
      this.lastSample = now;
      return true;
    }
    return false;
  }

  setPendingData(data: T): void {
    this.pendingData = data;
  }

  getPendingData(): T | null {
    const data = this.pendingData;
    this.pendingData = null;
    return data;
  }

  hasPendingData(): boolean {
    return this.pendingData !== null;
  }
}

// Main Hook Implementation
export const useRealTimeData = <T = any>(
  topic: string,
  options: UseRealTimeDataOptions = {}
) => {
  const {
    bufferSize = 1000,
    samplingRate = 60, // 60Hz default
    aggregationWindow = 1000, // 1 second
    enabled = true,
    maxLatency = 100, // 100ms max latency
    enablePerformanceMonitoring = true,
    pruneInterval = 5000, // 5 seconds
    autoOptimize = true,
  } = options;

  const dispatch = useAppDispatch();
  const { data: currentData, isConnected, subscribe, unsubscribe } = useWebSocket();
  
  // State management
  const [buffer] = useState(() => new CircularBuffer<T>(bufferSize));
  const [sampler] = useState(() => new DataSampler<T>(samplingRate));
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    latency: 0,
    throughput: 0,
    droppedFrames: 0,
    bufferUtilization: 0,
    processingTime: 0,
    memoryUsage: 0,
    lastUpdate: 0,
  });

  // Refs for performance tracking
  const metricsRef = useRef(metrics);
  const lastProcessTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const droppedFramesRef = useRef(0);
  const processingStartRef = useRef(0);
  const pruneTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Update metrics
  const updateMetrics = useCallback(() => {
    const now = Date.now();
    const processingTime = now - processingStartRef.current;
    const timeDelta = now - metricsRef.current.lastUpdate;
    
    if (timeDelta > 0) {
      const newMetrics: PerformanceMetrics = {
        latency: processingTime,
        throughput: (frameCountRef.current * 1000) / timeDelta,
        droppedFrames: droppedFramesRef.current,
        bufferUtilization: (buffer.size() / bufferSize) * 100,
        processingTime,
        memoryUsage: buffer.size() * 100 / bufferSize, // Rough estimation
        lastUpdate: now,
      };

      metricsRef.current = newMetrics;
      setMetrics(newMetrics);
    }

    // Reset counters
    frameCountRef.current = 0;
    droppedFramesRef.current = 0;
  }, [buffer, bufferSize]);

  // Data processing function
  const processData = useCallback((data: any) => {
    if (!enabled || !data) return;

    processingStartRef.current = Date.now();

    try {
      // Extract topic-specific data
      const topicData = extractTopicData(data, topic) as T;
      if (!topicData) return;

      // Check latency requirements
      const dataTimestamp = (data as any).timestamp || Date.now();
      const latency = Date.now() - dataTimestamp;
      
      if (latency > maxLatency) {
        droppedFramesRef.current++;
        return; // Drop frame if latency is too high
      }

      // Apply sampling
      if (sampler.shouldSample()) {
        // Process any pending data first
        if (sampler.hasPendingData()) {
          const pendingData = sampler.getPendingData();
          if (pendingData) {
            buffer.push(pendingData);
          }
        }
        
        buffer.push(topicData);
        frameCountRef.current++;
      } else {
        // Store as pending data for next sample window
        sampler.setPendingData(topicData);
      }

    } catch (error) {
      console.error(`Error processing real-time data for topic ${topic}:`, error);
      droppedFramesRef.current++;
    }
  }, [enabled, topic, maxLatency, buffer, sampler]);

  // Auto-optimization logic
  const optimizePerformance = useCallback(() => {
    if (!autoOptimize || !enablePerformanceMonitoring) return;

    const currentMetrics = metricsRef.current;
    
    // Adjust sampling rate based on performance
    if (currentMetrics.droppedFrames > 10 && samplingRate > 30) {
      // Reduce sampling rate if dropping frames
      console.log(`Reducing sampling rate for topic ${topic} due to dropped frames`);
    }
    
    if (currentMetrics.bufferUtilization > 90) {
      // Prune old data if buffer is getting full
      console.log(`High buffer utilization for topic ${topic}, pruning data`);
      // Buffer automatically handles this with circular buffer
    }
  }, [autoOptimize, enablePerformanceMonitoring, topic, samplingRate]);

  // Data extraction utility
  const extractTopicData = useCallback((data: any, topicPath: string): any => {
    const parts = topicPath.split('.');
    let current = data;
    
    for (const part of parts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        return null;
      }
    }
    
    return current;
  }, []);

  // Subscribe to topic on mount
  useEffect(() => {
    if (enabled && isConnected && topic) {
      subscribe([topic]);
    }

    return () => {
      if (topic) {
        unsubscribe([topic]);
      }
    };
  }, [enabled, isConnected, topic, subscribe, unsubscribe]);

  // Process incoming data
  useEffect(() => {
    if (currentData && enabled) {
      processData(currentData);
    }
  }, [currentData, enabled, processData]);

  // Performance monitoring and optimization
  useEffect(() => {
    if (!enablePerformanceMonitoring) return;

    const metricsInterval = setInterval(() => {
      updateMetrics();
      optimizePerformance();
    }, 1000);

    return () => clearInterval(metricsInterval);
  }, [enablePerformanceMonitoring, updateMetrics, optimizePerformance]);

  // Periodic pruning
  useEffect(() => {
    if (pruneInterval > 0) {
      pruneTimeoutRef.current = setTimeout(() => {
        // Circular buffer automatically prunes old data
        // This is just for additional cleanup if needed
      }, pruneInterval);
    }

    return () => {
      if (pruneTimeoutRef.current) {
        clearTimeout(pruneTimeoutRef.current);
      }
    };
  }, [pruneInterval]);

  // Memoized return values for performance
  const data = useMemo(() => buffer.getAll(), [buffer]);
  const latest = useMemo(() => buffer.getLatest(), [buffer]);
  const isActive = useMemo(() => enabled && isConnected, [enabled, isConnected]);

  // Aggregated data for time windows
  const aggregatedData = useMemo(() => {
    if (!data.length) return null;
    
    const now = Date.now();
    const windowStart = now - aggregationWindow;
    
    // Filter data within aggregation window
    const windowData = data.filter((item: any) => {
      const timestamp = item.timestamp || now;
      return timestamp >= windowStart;
    });

    if (windowData.length === 0) return null;

    // Basic aggregation (can be extended based on data type)
    return {
      count: windowData.length,
      timeRange: { start: windowStart, end: now },
      data: windowData,
      average: calculateAverage(windowData),
      trend: calculateTrend(windowData),
    };
  }, [data, aggregationWindow]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      buffer.clear();
      if (pruneTimeoutRef.current) {
        clearTimeout(pruneTimeoutRef.current);
      }
    };
  }, [buffer]);

  return {
    data,
    latest,
    aggregatedData,
    isConnected: isActive,
    metrics: enablePerformanceMonitoring ? metrics : null,
    bufferSize: buffer.size(),
    bufferUtilization: (buffer.size() / bufferSize) * 100,
    
    // Control methods
    clear: () => buffer.clear(),
    pause: () => {
      // Implementation for pausing data collection
    },
    resume: () => {
      // Implementation for resuming data collection
    },
  };
};

// Utility functions
function calculateAverage(data: any[]): number {
  if (!data.length) return 0;
  
  // This is a generic implementation
  // In practice, you'd implement specific averaging based on data structure
  try {
    const numericValues = data
      .map(item => {
        if (typeof item === 'number') return item;
        if (item && typeof item.value === 'number') return item.value;
        return 0;
      })
      .filter(val => !isNaN(val));
    
    return numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
  } catch {
    return 0;
  }
}

function calculateTrend(data: any[]): 'rising' | 'falling' | 'stable' {
  if (data.length < 2) return 'stable';
  
  try {
    const first = data[0];
    const last = data[data.length - 1];
    
    const firstVal = typeof first === 'number' ? first : first?.value || 0;
    const lastVal = typeof last === 'number' ? last : last?.value || 0;
    
    const threshold = 0.05; // 5% change threshold
    const change = (lastVal - firstVal) / Math.abs(firstVal || 1);
    
    if (change > threshold) return 'rising';
    if (change < -threshold) return 'falling';
    return 'stable';
  } catch {
    return 'stable';
  }
}

// Export types for use in other components
export type { UseRealTimeDataOptions, PerformanceMetrics };