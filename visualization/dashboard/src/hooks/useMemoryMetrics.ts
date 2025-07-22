import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useRealTimeData } from './useRealTimeData';
import { MemoryData, MemoryUsage, PerformanceMetrics, MemoryStore } from '../types';

// Memory-specific interfaces
export interface MemoryMetricsOptions {
  historySize?: number;
  alertThresholds?: AlertThresholds;
  enableUsagePatterns?: boolean;
  enableConsolidationTracking?: boolean;
  trackAccessPatterns?: boolean;
  samplingRate?: number;
  memoryCacheSize?: number;
}

export interface AlertThresholds {
  usage: number;         // Usage percentage threshold (e.g., 0.8 for 80%)
  latency: number;       // Latency threshold in ms
  errorRate: number;     // Error rate threshold (e.g., 0.05 for 5%)
  fragmentationLevel: number; // Fragmentation threshold
}

export interface MemoryAlert {
  id: string;
  timestamp: number;
  type: 'usage' | 'performance' | 'fragmentation' | 'leak' | 'consolidation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  storeId?: string;
  value: number;
  threshold: number;
  recommendations: string[];
}

export interface UsagePattern {
  pattern: 'steady' | 'spike' | 'gradual_increase' | 'sawtooth' | 'oscillating';
  confidence: number;
  duration: number;
  peakUsage: number;
  averageUsage: number;
  predictedNext: number;
  cycleDetected: boolean;
  cycleLength?: number;
}

export interface ConsolidationEvent {
  id: string;
  timestamp: number;
  type: 'gc' | 'defrag' | 'merge' | 'cleanup';
  storeId: string;
  beforeSize: number;
  afterSize: number;
  duration: number;
  efficiencyGain: number;
  triggeredBy: 'threshold' | 'schedule' | 'manual' | 'system';
  success: boolean;
}

export interface AccessPattern {
  storeId: string;
  accessType: 'read' | 'write' | 'delete' | 'update';
  frequency: number;
  hotspots: Array<{
    region: string;
    accessCount: number;
    lastAccessed: number;
  }>;
  temporalDistribution: Record<string, number>; // Hour -> access count
  spatialDistribution: Record<string, number>;  // Memory region -> access count
}

export interface MemoryFragmentation {
  storeId: string;
  fragmentationLevel: number; // 0-1, where 1 is completely fragmented
  largestFreeBlock: number;
  totalFreeSpace: number;
  freeBlockCount: number;
  averageBlockSize: number;
  recommendedDefragmentation: boolean;
}

export interface SDROperations {
  unionOperations: number;
  intersectionOperations: number;
  differenceOperations: number;
  complementOperations: number;
  averageOperationTime: number;
  sparsityLevel: number;
  compressionRatio: number;
  collisionRate: number;
}

export interface MemoryStoreAnalysis {
  id: string;
  name: string;
  type: 'sdr' | 'zce' | 'cache';
  
  // Basic metrics
  currentSize: number;
  utilization: number;
  accessCount: number;
  hitRate: number;
  missRate: number;
  
  // Performance metrics
  avgAccessTime: number;
  avgWriteTime: number;
  throughput: number;
  
  // Health metrics
  fragmentation: MemoryFragmentation;
  errorRate: number;
  uptime: number;
  
  // Advanced metrics
  accessPattern: AccessPattern;
  usagePattern: UsagePattern;
  consolidationHistory: ConsolidationEvent[];
  sdrOperations?: SDROperations; // Only for SDR stores
  
  // Predictions
  predictedGrowth: number;
  recommendedActions: string[];
  healthScore: number;
}

export interface SystemMemoryMetrics {
  totalMemory: number;
  usedMemory: number;
  freeMemory: number;
  utilization: number;
  
  // Performance
  overallLatency: number;
  overallThroughput: number;
  systemEfficiency: number;
  
  // Health
  errorRate: number;
  fragmentationLevel: number;
  consolidationRate: number;
  
  // Trends
  growthRate: number;
  stabilityScore: number;
  performanceTrend: 'improving' | 'stable' | 'degrading';
  
  // Alerts
  activeAlerts: MemoryAlert[];
  alertHistory: MemoryAlert[];
}

// Memory Analysis Engine
class MemoryAnalysisEngine {
  private usageHistory: Map<string, Array<{ timestamp: number; usage: number }>> = new Map();
  private accessHistory: Map<string, Array<{ timestamp: number; type: string; duration: number }>> = new Map();
  private consolidationHistory: Map<string, ConsolidationEvent[]> = new Map();
  private alertHistory: MemoryAlert[] = [];
  private readonly historySize: number;
  private readonly alertThresholds: AlertThresholds;

  constructor(historySize: number = 200, alertThresholds: AlertThresholds) {
    this.historySize = historySize;
    this.alertThresholds = alertThresholds;
  }

  analyzeMemoryData(memoryData: MemoryData): {
    storeAnalyses: MemoryStoreAnalysis[];
    systemMetrics: SystemMemoryMetrics;
    newAlerts: MemoryAlert[];
  } {
    const timestamp = Date.now();
    const storeAnalyses: MemoryStoreAnalysis[] = [];
    const newAlerts: MemoryAlert[] = [];

    // Analyze each memory store
    for (const store of memoryData.stores) {
      const analysis = this.analyzeMemoryStore(store, memoryData, timestamp);
      storeAnalyses.push(analysis);

      // Check for alerts
      const storeAlerts = this.checkStoreAlerts(analysis, timestamp);
      newAlerts.push(...storeAlerts);
    }

    // Calculate system-wide metrics
    const systemMetrics = this.calculateSystemMetrics(memoryData, storeAnalyses, timestamp);
    
    // Check for system-wide alerts
    const systemAlerts = this.checkSystemAlerts(systemMetrics, timestamp);
    newAlerts.push(...systemAlerts);

    // Update alert history
    this.alertHistory.push(...newAlerts);
    if (this.alertHistory.length > this.historySize * 2) {
      this.alertHistory = this.alertHistory.slice(-this.historySize);
    }

    return { storeAnalyses, systemMetrics, newAlerts };
  }

  private analyzeMemoryStore(store: MemoryStore, memoryData: MemoryData, timestamp: number): MemoryStoreAnalysis {
    // Update usage history
    const usageHistory = this.usageHistory.get(store.id) || [];
    usageHistory.push({ timestamp, usage: store.utilization });
    if (usageHistory.length > this.historySize) {
      usageHistory.shift();
    }
    this.usageHistory.set(store.id, usageHistory);

    // Calculate basic metrics
    const hitRate = this.calculateHitRate(store);
    const missRate = 1 - hitRate;
    const avgAccessTime = this.calculateAverageAccessTime(store);
    const avgWriteTime = avgAccessTime * 1.2; // Estimate write time
    const throughput = store.accessCount > 0 ? 1000 / avgAccessTime : 0;

    // Analyze fragmentation
    const fragmentation = this.analyzeFragmentation(store);

    // Analyze usage patterns
    const usagePattern = this.analyzeUsagePattern(store.id, usageHistory);

    // Analyze access patterns
    const accessPattern = this.analyzeAccessPattern(store);

    // Get consolidation history
    const consolidationHistory = this.consolidationHistory.get(store.id) || [];

    // Calculate SDR operations (if applicable)
    let sdrOperations: SDROperations | undefined;
    if (store.type === 'sdr') {
      sdrOperations = this.analyzeSDROperations(store);
    }

    // Calculate health score
    const healthScore = this.calculateHealthScore(store, fragmentation, usagePattern);

    // Generate recommendations
    const recommendedActions = this.generateRecommendations(store, fragmentation, usagePattern, healthScore);

    return {
      id: store.id,
      name: store.name,
      type: store.type,
      currentSize: store.size,
      utilization: store.utilization,
      accessCount: store.accessCount,
      hitRate,
      missRate,
      avgAccessTime,
      avgWriteTime,
      throughput,
      fragmentation,
      errorRate: this.calculateErrorRate(store),
      uptime: this.calculateUptime(store),
      accessPattern,
      usagePattern,
      consolidationHistory,
      sdrOperations,
      predictedGrowth: this.predictGrowth(usageHistory),
      recommendedActions,
      healthScore,
    };
  }

  private calculateHitRate(store: MemoryStore): number {
    // Simplified hit rate calculation
    // In a real implementation, this would track actual hits/misses
    const baseHitRate = store.type === 'cache' ? 0.85 : 0.95;
    const utilizationFactor = Math.max(0.5, 1 - store.utilization * 0.3);
    return baseHitRate * utilizationFactor;
  }

  private calculateAverageAccessTime(store: MemoryStore): number {
    // Base access times by store type (in ms)
    const baseTimes = {
      'sdr': 0.1,   // Very fast sparse operations
      'zce': 0.5,   // Fast zero-copy operations
      'cache': 1.0, // Standard cache access
    };

    const baseTime = baseTimes[store.type] || 1.0;
    
    // Access time increases with utilization and decreases with hit rate
    const utilizationFactor = 1 + store.utilization * 0.5;
    const sizeFactor = 1 + Math.log10(store.size / 1024) * 0.1; // Log scale for size impact
    
    return baseTime * utilizationFactor * sizeFactor;
  }

  private analyzeFragmentation(store: MemoryStore): MemoryFragmentation {
    // Simplified fragmentation analysis
    // In practice, this would require detailed memory layout information
    
    const baseFragmentation = store.utilization > 0.8 ? 0.3 : 0.1;
    const utilizationFactor = Math.pow(store.utilization, 2);
    const fragmentationLevel = Math.min(0.95, baseFragmentation + utilizationFactor * 0.4);
    
    const totalFreeSpace = store.size * (1 - store.utilization);
    const estimatedBlocks = Math.ceil(fragmentationLevel * 100);
    const averageBlockSize = totalFreeSpace / Math.max(estimatedBlocks, 1);
    const largestFreeBlock = averageBlockSize * (2 - fragmentationLevel);

    return {
      storeId: store.id,
      fragmentationLevel,
      largestFreeBlock,
      totalFreeSpace,
      freeBlockCount: estimatedBlocks,
      averageBlockSize,
      recommendedDefragmentation: fragmentationLevel > 0.7,
    };
  }

  private analyzeUsagePattern(storeId: string, history: Array<{ timestamp: number; usage: number }>): UsagePattern {
    if (history.length < 10) {
      return {
        pattern: 'steady',
        confidence: 0.5,
        duration: 0,
        peakUsage: 0,
        averageUsage: 0,
        predictedNext: 0,
        cycleDetected: false,
      };
    }

    const values = history.map(h => h.usage);
    const timespan = history[history.length - 1].timestamp - history[0].timestamp;
    const averageUsage = values.reduce((sum, v) => sum + v, 0) / values.length;
    const peakUsage = Math.max(...values);
    
    // Calculate variance and trend
    const variance = values.reduce((sum, v) => sum + Math.pow(v - averageUsage, 2), 0) / values.length;
    const cv = averageUsage > 0 ? Math.sqrt(variance) / averageUsage : 0;

    // Detect pattern type
    let pattern: UsagePattern['pattern'];
    let confidence = 0.7;

    if (cv < 0.1) {
      pattern = 'steady';
      confidence = 0.9;
    } else if (cv > 0.8) {
      // Check for spikes vs sawtooth
      const spikes = this.detectSpikes(values);
      if (spikes.length > values.length * 0.1) {
        pattern = 'spike';
      } else {
        pattern = 'sawtooth';
      }
    } else {
      // Check for gradual increase or oscillation
      const trend = this.calculateTrend(values);
      if (Math.abs(trend) > 0.1) {
        pattern = 'gradual_increase';
      } else {
        pattern = 'oscillating';
      }
    }

    // Detect cycles
    const { cycleDetected, cycleLength } = this.detectCycle(values);
    
    // Predict next value
    const predictedNext = this.predictNextUsage(values, pattern);

    return {
      pattern,
      confidence,
      duration: timespan,
      peakUsage,
      averageUsage,
      predictedNext,
      cycleDetected,
      cycleLength,
    };
  }

  private detectSpikes(values: number[]): number[] {
    const spikes: number[] = [];
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
    const threshold = mean + 2 * std;

    values.forEach((value, index) => {
      if (value > threshold) {
        spikes.push(index);
      }
    });

    return spikes;
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;

    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, v) => sum + v, 0);
    const sumXY = values.reduce((sum, v, i) => sum + i * v, 0);
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6;

    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }

  private detectCycle(values: number[]): { cycleDetected: boolean; cycleLength?: number } {
    if (values.length < 20) return { cycleDetected: false };

    // Simple autocorrelation-based cycle detection
    const maxLag = Math.floor(values.length / 4);
    let bestLag = 0;
    let bestCorrelation = 0;

    for (let lag = 2; lag <= maxLag; lag++) {
      const correlation = this.calculateAutocorrelation(values, lag);
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestLag = lag;
      }
    }

    return {
      cycleDetected: bestCorrelation > 0.7,
      cycleLength: bestCorrelation > 0.7 ? bestLag : undefined,
    };
  }

  private calculateAutocorrelation(values: number[], lag: number): number {
    if (lag >= values.length) return 0;

    const n = values.length - lag;
    const mean1 = values.slice(0, n).reduce((sum, v) => sum + v, 0) / n;
    const mean2 = values.slice(lag).reduce((sum, v) => sum + v, 0) / n;

    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;

    for (let i = 0; i < n; i++) {
      const x1 = values[i] - mean1;
      const x2 = values[i + lag] - mean2;
      numerator += x1 * x2;
      denominator1 += x1 * x1;
      denominator2 += x2 * x2;
    }

    const denominator = Math.sqrt(denominator1 * denominator2);
    return denominator > 0 ? numerator / denominator : 0;
  }

  private predictNextUsage(values: number[], pattern: UsagePattern['pattern']): number {
    if (values.length < 3) return values[values.length - 1] || 0;

    const recent = values.slice(-5);
    const lastValue = recent[recent.length - 1];

    switch (pattern) {
      case 'steady':
        return recent.reduce((sum, v) => sum + v, 0) / recent.length;
        
      case 'gradual_increase':
        const trend = this.calculateTrend(recent);
        return Math.max(0, Math.min(1, lastValue + trend));
        
      case 'oscillating':
        // Simple oscillation prediction
        const prev = recent[recent.length - 2];
        const delta = lastValue - prev;
        return Math.max(0, Math.min(1, lastValue - delta * 0.8));
        
      case 'spike':
        // Assume spikes are temporary
        return recent.slice(0, -1).reduce((sum, v) => sum + v, 0) / (recent.length - 1);
        
      case 'sawtooth':
        // Assume continued sawtooth pattern
        const maxRecent = Math.max(...recent);
        return lastValue < maxRecent * 0.5 ? lastValue * 1.5 : lastValue * 0.7;
        
      default:
        return lastValue;
    }
  }

  private analyzeAccessPattern(store: MemoryStore): AccessPattern {
    // Simplified access pattern analysis
    // In practice, this would track actual access events
    
    const now = new Date();
    const currentHour = now.getHours();
    
    // Simulate temporal distribution
    const temporalDistribution: Record<string, number> = {};
    for (let hour = 0; hour < 24; hour++) {
      const baseAccess = 100;
      const variance = hour === currentHour ? 1.5 : Math.random() * 0.5 + 0.5;
      temporalDistribution[hour.toString()] = Math.floor(baseAccess * variance);
    }

    // Simulate spatial distribution
    const regions = ['region_0', 'region_1', 'region_2', 'region_3'];
    const spatialDistribution: Record<string, number> = {};
    regions.forEach(region => {
      spatialDistribution[region] = Math.floor(Math.random() * 1000) + 100;
    });

    // Simulate hotspots
    const hotspots = regions.slice(0, 2).map(region => ({
      region,
      accessCount: spatialDistribution[region],
      lastAccessed: Date.now() - Math.random() * 3600000, // Within last hour
    }));

    return {
      storeId: store.id,
      accessType: 'read', // Simplified
      frequency: store.accessCount / 3600, // Accesses per hour
      hotspots,
      temporalDistribution,
      spatialDistribution,
    };
  }

  private analyzeSDROperations(store: MemoryStore): SDROperations {
    // Simulate SDR operations analysis
    const baseOperations = store.accessCount;
    
    return {
      unionOperations: Math.floor(baseOperations * 0.3),
      intersectionOperations: Math.floor(baseOperations * 0.4),
      differenceOperations: Math.floor(baseOperations * 0.2),
      complementOperations: Math.floor(baseOperations * 0.1),
      averageOperationTime: 0.05, // 0.05ms average for SDR operations
      sparsityLevel: 0.95, // 95% sparse
      compressionRatio: 0.02, // 50:1 compression
      collisionRate: 0.001, // 0.1% collision rate
    };
  }

  private calculateErrorRate(store: MemoryStore): number {
    // Simplified error rate calculation
    const baseErrorRate = {
      'sdr': 0.0001,  // Very low error rate
      'zce': 0.0005,  // Low error rate
      'cache': 0.001, // Standard cache error rate
    };
    
    const base = baseErrorRate[store.type] || 0.001;
    const stressFactor = store.utilization > 0.9 ? 2.0 : 1.0;
    
    return base * stressFactor;
  }

  private calculateUptime(store: MemoryStore): number {
    // Simplified uptime calculation (percentage)
    const baseUptime = 0.999; // 99.9% base uptime
    const stressFactor = store.utilization > 0.95 ? 0.995 : 1.0;
    
    return baseUptime * stressFactor;
  }

  private predictGrowth(history: Array<{ timestamp: number; usage: number }>): number {
    if (history.length < 5) return 0;

    const recent = history.slice(-10);
    const values = recent.map(h => h.usage);
    const trend = this.calculateTrend(values);
    
    // Predict growth over next hour (in percentage points)
    return trend * 60; // Scale to per-hour growth
  }

  private calculateHealthScore(
    store: MemoryStore, 
    fragmentation: MemoryFragmentation, 
    usagePattern: UsagePattern
  ): number {
    // Calculate health score (0-1, where 1 is perfect health)
    let score = 1.0;
    
    // Utilization impact (penalty for very high utilization)
    if (store.utilization > 0.9) {
      score -= 0.3;
    } else if (store.utilization > 0.8) {
      score -= 0.1;
    }
    
    // Fragmentation impact
    score -= fragmentation.fragmentationLevel * 0.3;
    
    // Access pattern stability
    if (usagePattern.pattern === 'spike') {
      score -= 0.2;
    } else if (usagePattern.pattern === 'steady') {
      score += 0.1;
    }
    
    // Error rate impact
    const errorRate = this.calculateErrorRate(store);
    score -= errorRate * 100; // Scale error rate impact
    
    return Math.max(0, Math.min(1, score));
  }

  private generateRecommendations(
    store: MemoryStore, 
    fragmentation: MemoryFragmentation, 
    usagePattern: UsagePattern,
    healthScore: number
  ): string[] {
    const recommendations: string[] = [];
    
    if (store.utilization > 0.9) {
      recommendations.push('Consider increasing memory allocation or implementing compression');
    }
    
    if (fragmentation.recommendedDefragmentation) {
      recommendations.push('Schedule defragmentation to improve memory efficiency');
    }
    
    if (usagePattern.pattern === 'spike') {
      recommendations.push('Implement spike detection and auto-scaling to handle usage spikes');
    }
    
    if (usagePattern.pattern === 'gradual_increase') {
      recommendations.push('Monitor for potential memory leaks or implement growth limits');
    }
    
    if (healthScore < 0.7) {
      recommendations.push('System health is degraded - consider immediate maintenance');
    }
    
    if (store.type === 'sdr' && usagePattern.averageUsage > 0.8) {
      recommendations.push('Consider optimizing SDR sparsity levels for better performance');
    }
    
    return recommendations;
  }

  private calculateSystemMetrics(
    memoryData: MemoryData,
    storeAnalyses: MemoryStoreAnalysis[],
    timestamp: number
  ): SystemMemoryMetrics {
    // System-wide calculations
    const totalMemory = storeAnalyses.reduce((sum, store) => sum + store.currentSize, 0);
    const usedMemory = storeAnalyses.reduce((sum, store) => sum + store.currentSize * store.utilization, 0);
    const freeMemory = totalMemory - usedMemory;
    const utilization = totalMemory > 0 ? usedMemory / totalMemory : 0;
    
    // Performance metrics
    const overallLatency = storeAnalyses.reduce((sum, store) => sum + store.avgAccessTime, 0) / storeAnalyses.length;
    const overallThroughput = storeAnalyses.reduce((sum, store) => sum + store.throughput, 0);
    const systemEfficiency = storeAnalyses.reduce((sum, store) => sum + store.healthScore, 0) / storeAnalyses.length;
    
    // Health metrics
    const errorRate = storeAnalyses.reduce((sum, store) => sum + store.errorRate, 0) / storeAnalyses.length;
    const fragmentationLevel = storeAnalyses.reduce((sum, store) => sum + store.fragmentation.fragmentationLevel, 0) / storeAnalyses.length;
    const consolidationRate = this.calculateConsolidationRate(storeAnalyses);
    
    // Trends
    const growthRate = storeAnalyses.reduce((sum, store) => sum + store.predictedGrowth, 0) / storeAnalyses.length;
    const stabilityScore = this.calculateSystemStability(storeAnalyses);
    const performanceTrend = this.calculatePerformanceTrend(storeAnalyses);
    
    return {
      totalMemory,
      usedMemory,
      freeMemory,
      utilization,
      overallLatency,
      overallThroughput,
      systemEfficiency,
      errorRate,
      fragmentationLevel,
      consolidationRate,
      growthRate,
      stabilityScore,
      performanceTrend,
      activeAlerts: [],
      alertHistory: this.alertHistory.slice(-50),
    };
  }

  private calculateConsolidationRate(storeAnalyses: MemoryStoreAnalysis[]): number {
    // Calculate consolidation events per hour
    const recentConsolidations = storeAnalyses.reduce((count, store) => {
      const recentEvents = store.consolidationHistory.filter(
        event => Date.now() - event.timestamp <= 3600000 // Last hour
      );
      return count + recentEvents.length;
    }, 0);
    
    return recentConsolidations;
  }

  private calculateSystemStability(storeAnalyses: MemoryStoreAnalysis[]): number {
    // System stability based on individual store stability
    const stabilityScores = storeAnalyses.map(store => {
      const patternStability = store.usagePattern.pattern === 'steady' ? 1.0 : 0.7;
      const healthStability = store.healthScore;
      const fragmentationStability = 1 - store.fragmentation.fragmentationLevel;
      
      return (patternStability + healthStability + fragmentationStability) / 3;
    });
    
    return stabilityScores.reduce((sum, score) => sum + score, 0) / stabilityScores.length;
  }

  private calculatePerformanceTrend(storeAnalyses: MemoryStoreAnalysis[]): 'improving' | 'stable' | 'degrading' {
    const avgHealthScore = storeAnalyses.reduce((sum, store) => sum + store.healthScore, 0) / storeAnalyses.length;
    const avgFragmentation = storeAnalyses.reduce((sum, store) => sum + store.fragmentation.fragmentationLevel, 0) / storeAnalyses.length;
    
    if (avgHealthScore > 0.8 && avgFragmentation < 0.3) {
      return 'improving';
    } else if (avgHealthScore < 0.6 || avgFragmentation > 0.7) {
      return 'degrading';
    }
    
    return 'stable';
  }

  private checkStoreAlerts(store: MemoryStoreAnalysis, timestamp: number): MemoryAlert[] {
    const alerts: MemoryAlert[] = [];

    // Usage alerts
    if (store.utilization > this.alertThresholds.usage) {
      alerts.push({
        id: `usage_${store.id}_${timestamp}`,
        timestamp,
        type: 'usage',
        severity: store.utilization > 0.95 ? 'critical' : 'high',
        message: `High memory usage detected in ${store.name}`,
        storeId: store.id,
        value: store.utilization,
        threshold: this.alertThresholds.usage,
        recommendations: store.recommendedActions.slice(0, 3),
      });
    }

    // Performance alerts
    if (store.avgAccessTime > this.alertThresholds.latency) {
      alerts.push({
        id: `performance_${store.id}_${timestamp}`,
        timestamp,
        type: 'performance',
        severity: store.avgAccessTime > this.alertThresholds.latency * 2 ? 'high' : 'medium',
        message: `High latency detected in ${store.name}`,
        storeId: store.id,
        value: store.avgAccessTime,
        threshold: this.alertThresholds.latency,
        recommendations: ['Investigate access patterns', 'Consider caching optimizations'],
      });
    }

    // Fragmentation alerts
    if (store.fragmentation.fragmentationLevel > this.alertThresholds.fragmentationLevel) {
      alerts.push({
        id: `fragmentation_${store.id}_${timestamp}`,
        timestamp,
        type: 'fragmentation',
        severity: store.fragmentation.recommendedDefragmentation ? 'high' : 'medium',
        message: `High fragmentation detected in ${store.name}`,
        storeId: store.id,
        value: store.fragmentation.fragmentationLevel,
        threshold: this.alertThresholds.fragmentationLevel,
        recommendations: ['Schedule defragmentation', 'Review allocation patterns'],
      });
    }

    return alerts;
  }

  private checkSystemAlerts(systemMetrics: SystemMemoryMetrics, timestamp: number): MemoryAlert[] {
    const alerts: MemoryAlert[] = [];

    // System-wide usage alert
    if (systemMetrics.utilization > this.alertThresholds.usage) {
      alerts.push({
        id: `system_usage_${timestamp}`,
        timestamp,
        type: 'usage',
        severity: systemMetrics.utilization > 0.95 ? 'critical' : 'high',
        message: 'High system memory usage detected',
        value: systemMetrics.utilization,
        threshold: this.alertThresholds.usage,
        recommendations: ['Review individual store usage', 'Consider memory optimization'],
      });
    }

    return alerts;
  }
}

// Main Hook Implementation
export const useMemoryMetrics = (options: MemoryMetricsOptions = {}) => {
  const {
    historySize = 200,
    alertThresholds = {
      usage: 0.8,
      latency: 10, // 10ms
      errorRate: 0.05, // 5%
      fragmentationLevel: 0.7,
    },
    enableUsagePatterns = true,
    enableConsolidationTracking = true,
    trackAccessPatterns = true,
    samplingRate = 10, // 10Hz for memory metrics
    memoryCacheSize = 100,
  } = options;

  // Use base real-time data hook
  const {
    data: rawMemoryData,
    latest: latestMemory,
    isConnected,
    metrics: baseMetrics,
    aggregatedData,
  } = useRealTimeData<MemoryData>('memory', {
    bufferSize: historySize,
    samplingRate,
    aggregationWindow: 30000, // 30 second aggregation window
    enablePerformanceMonitoring: true,
  });

  // Analysis engine
  const [memoryEngine] = useState(() => new MemoryAnalysisEngine(historySize, alertThresholds));

  // State management
  const [storeAnalyses, setStoreAnalyses] = useState<MemoryStoreAnalysis[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMemoryMetrics>({
    totalMemory: 0,
    usedMemory: 0,
    freeMemory: 0,
    utilization: 0,
    overallLatency: 0,
    overallThroughput: 0,
    systemEfficiency: 0,
    errorRate: 0,
    fragmentationLevel: 0,
    consolidationRate: 0,
    growthRate: 0,
    stabilityScore: 0,
    performanceTrend: 'stable',
    activeAlerts: [],
    alertHistory: [],
  });
  const [memoryAlerts, setMemoryAlerts] = useState<MemoryAlert[]>([]);

  // Process memory data updates
  useEffect(() => {
    if (!latestMemory || !isConnected) return;

    try {
      const analysis = memoryEngine.analyzeMemoryData(latestMemory);
      
      setStoreAnalyses(analysis.storeAnalyses);
      setSystemMetrics(analysis.systemMetrics);
      
      // Update alerts
      if (analysis.newAlerts.length > 0) {
        setMemoryAlerts(prev => {
          const updated = [...prev, ...analysis.newAlerts];
          return updated.slice(-memoryCacheSize);
        });
      }

    } catch (error) {
      console.error('Error analyzing memory metrics:', error);
    }
  }, [latestMemory, isConnected, memoryEngine, memoryCacheSize]);

  // Calculate derived metrics
  const derivedMetrics = useMemo(() => {
    if (!storeAnalyses.length) return null;

    return {
      // Store type breakdown
      storeTypeBreakdown: storeAnalyses.reduce((breakdown, store) => {
        breakdown[store.type] = (breakdown[store.type] || 0) + 1;
        return breakdown;
      }, {} as Record<string, number>),

      // Top performers and underperformers
      topPerformers: storeAnalyses
        .sort((a, b) => b.healthScore - a.healthScore)
        .slice(0, 3)
        .map(s => ({ id: s.id, name: s.name, score: s.healthScore })),
      
      underperformers: storeAnalyses
        .sort((a, b) => a.healthScore - b.healthScore)
        .slice(0, 3)
        .map(s => ({ id: s.id, name: s.name, score: s.healthScore })),

      // Usage pattern distribution
      usagePatternDistribution: storeAnalyses.reduce((dist, store) => {
        dist[store.usagePattern.pattern] = (dist[store.usagePattern.pattern] || 0) + 1;
        return dist;
      }, {} as Record<string, number>),

      // Critical stores (high usage or low health)
      criticalStores: storeAnalyses.filter(
        store => store.utilization > 0.9 || store.healthScore < 0.5
      ),
    };
  }, [storeAnalyses]);

  return {
    // Core data
    stores: storeAnalyses,
    systemMetrics,
    usage: latestMemory?.usage,
    performance: latestMemory?.performance,
    
    // Analysis results
    alerts: memoryAlerts,
    derivedMetrics,
    isConnected,
    
    // Aggregated and historical data
    historicalData: rawMemoryData,
    aggregatedData,
    
    // Control methods
    clearHistory: useCallback(() => {
      setMemoryAlerts([]);
    }, []),
    
    getStoreAnalysis: useCallback((storeId: string) => {
      return storeAnalyses.find(store => store.id === storeId) || null;
    }, [storeAnalyses]),
    
    getActiveAlerts: useCallback((severity?: MemoryAlert['severity']) => {
      return severity 
        ? memoryAlerts.filter(alert => alert.severity === severity)
        : memoryAlerts.filter(alert => Date.now() - alert.timestamp <= 300000); // Active = last 5 minutes
    }, [memoryAlerts]),
    
    getUsagePattern: useCallback((storeId: string) => {
      const store = storeAnalyses.find(s => s.id === storeId);
      return store?.usagePattern || null;
    }, [storeAnalyses]),
    
    getRecommendations: useCallback((storeId?: string) => {
      if (storeId) {
        const store = storeAnalyses.find(s => s.id === storeId);
        return store?.recommendedActions || [];
      }
      
      // Return system-wide recommendations
      const allRecommendations = storeAnalyses.flatMap(store => store.recommendedActions);
      return [...new Set(allRecommendations)]; // Remove duplicates
    }, [storeAnalyses]),
  };
};

// Export types
export type {
  MemoryMetricsOptions,
  AlertThresholds,
  MemoryAlert,
  UsagePattern,
  ConsolidationEvent,
  AccessPattern,
  MemoryFragmentation,
  SDROperations,
  MemoryStoreAnalysis,
  SystemMemoryMetrics,
};