import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useRealTimeData, PerformanceMetrics } from './useRealTimeData';
import { CognitiveData, CognitivePattern } from '../types';

// Cognitive-specific interfaces
export interface CognitiveOptions {
  patternHistorySize?: number;
  inhibitoryThreshold?: number;
  confidenceThreshold?: number;
  enablePatternRecognition?: boolean;
  activationWindowMs?: number;
  hierarchicalDepthTracking?: boolean;
  samplingRate?: number;
}

export interface InhibitoryData {
  current: number;
  average: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  peaks: Array<{ timestamp: number; level: number }>;
  valleys: Array<{ timestamp: number; level: number }>;
  oscillationFrequency: number;
}

export interface PatternStrengthMap {
  [patternId: string]: {
    current: number;
    peak: number;
    average: number;
    stability: number;
    lastActive: number;
    duration: number;
  };
}

export interface ActivationEvent {
  timestamp: number;
  patternId: string;
  type: 'activation' | 'deactivation' | 'strength_change';
  value: number;
  threshold: number;
  metadata: Record<string, any>;
}

export interface PatternCluster {
  id: string;
  patterns: string[];
  strength: number;
  coherence: number;
  dominantType: 'hierarchical' | 'lateral' | 'feedback';
  activationSynchrony: number;
}

export interface CognitiveMetrics {
  totalPatterns: number;
  activePatterns: number;
  averageStrength: number;
  inhibitoryEffectiveness: number;
  hierarchicalBalance: number;
  patternDiversity: number;
  activationRate: number; // patterns per second
  coherenceScore: number;
}

// Pattern Recognition Engine
class PatternRecognitionEngine {
  private patternHistory: Map<string, Array<{ timestamp: number; strength: number }>> = new Map();
  private activationThresholds: Map<string, number> = new Map();
  private readonly historySize: number;

  constructor(historySize: number = 100) {
    this.historySize = historySize;
  }

  updatePattern(pattern: CognitivePattern): void {
    const history = this.patternHistory.get(pattern.id) || [];
    history.push({ timestamp: pattern.timestamp, strength: pattern.strength });
    
    // Keep only recent history
    if (history.length > this.historySize) {
      history.shift();
    }
    
    this.patternHistory.set(pattern.id, history);
    
    // Auto-adjust activation threshold based on historical data
    this.updateActivationThreshold(pattern.id, history);
  }

  private updateActivationThreshold(patternId: string, history: Array<{ timestamp: number; strength: number }>): void {
    if (history.length < 5) return;
    
    const strengths = history.map(h => h.strength);
    const mean = strengths.reduce((sum, s) => sum + s, 0) / strengths.length;
    const stdDev = Math.sqrt(strengths.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / strengths.length);
    
    // Set threshold at mean + 0.5 * standard deviation
    this.activationThresholds.set(patternId, mean + 0.5 * stdDev);
  }

  getPatternTrend(patternId: string, windowMs: number = 5000): 'rising' | 'falling' | 'stable' {
    const history = this.patternHistory.get(patternId);
    if (!history || history.length < 3) return 'stable';
    
    const now = Date.now();
    const windowData = history.filter(h => now - h.timestamp <= windowMs);
    
    if (windowData.length < 3) return 'stable';
    
    // Calculate linear regression slope
    const n = windowData.length;
    const sumX = windowData.reduce((sum, h, i) => sum + i, 0);
    const sumY = windowData.reduce((sum, h) => sum + h.strength, 0);
    const sumXY = windowData.reduce((sum, h, i) => sum + i * h.strength, 0);
    const sumXX = windowData.reduce((sum, h, i) => sum + i * i, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    const threshold = 0.01; // Sensitivity threshold
    if (slope > threshold) return 'rising';
    if (slope < -threshold) return 'falling';
    return 'stable';
  }

  getPatternStability(patternId: string): number {
    const history = this.patternHistory.get(patternId);
    if (!history || history.length < 3) return 0;
    
    const strengths = history.map(h => h.strength);
    const mean = strengths.reduce((sum, s) => sum + s, 0) / strengths.length;
    const variance = strengths.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / strengths.length;
    
    // Return stability as inverse of coefficient of variation
    return mean > 0 ? 1 / (Math.sqrt(variance) / mean + 0.001) : 0;
  }

  detectPatternClusters(patterns: CognitivePattern[]): PatternCluster[] {
    const clusters: PatternCluster[] = [];
    const processed = new Set<string>();
    
    for (const pattern of patterns) {
      if (processed.has(pattern.id) || pattern.strength < 0.3) continue;
      
      const cluster: PatternCluster = {
        id: `cluster_${Date.now()}_${pattern.id}`,
        patterns: [pattern.id],
        strength: pattern.strength,
        coherence: 1.0,
        dominantType: pattern.type,
        activationSynchrony: 1.0,
      };
      
      // Find related patterns (those with overlapping active nodes)
      for (const otherPattern of patterns) {
        if (otherPattern.id === pattern.id || processed.has(otherPattern.id)) continue;
        
        const overlap = this.calculateNodeOverlap(pattern.activeNodes, otherPattern.activeNodes);
        if (overlap > 0.3) { // 30% overlap threshold
          cluster.patterns.push(otherPattern.id);
          cluster.strength = Math.max(cluster.strength, otherPattern.strength);
          processed.add(otherPattern.id);
        }
      }
      
      // Calculate cluster coherence and synchrony
      if (cluster.patterns.length > 1) {
        cluster.coherence = this.calculateClusterCoherence(cluster.patterns, patterns);
        cluster.activationSynchrony = this.calculateActivationSynchrony(cluster.patterns);
      }
      
      clusters.push(cluster);
      processed.add(pattern.id);
    }
    
    return clusters.sort((a, b) => b.strength - a.strength);
  }

  private calculateNodeOverlap(nodes1: string[], nodes2: string[]): number {
    const set1 = new Set(nodes1);
    const overlap = nodes2.filter(node => set1.has(node)).length;
    return overlap / Math.max(nodes1.length, nodes2.length);
  }

  private calculateClusterCoherence(patternIds: string[], allPatterns: CognitivePattern[]): number {
    const clusterPatterns = allPatterns.filter(p => patternIds.includes(p.id));
    if (clusterPatterns.length < 2) return 1.0;
    
    let totalCoherence = 0;
    let comparisons = 0;
    
    for (let i = 0; i < clusterPatterns.length; i++) {
      for (let j = i + 1; j < clusterPatterns.length; j++) {
        const overlap = this.calculateNodeOverlap(
          clusterPatterns[i].activeNodes,
          clusterPatterns[j].activeNodes
        );
        totalCoherence += overlap;
        comparisons++;
      }
    }
    
    return comparisons > 0 ? totalCoherence / comparisons : 0;
  }

  private calculateActivationSynchrony(patternIds: string[]): number {
    // Simplified synchrony calculation based on activation timing
    const histories = patternIds
      .map(id => this.patternHistory.get(id))
      .filter(h => h && h.length > 0);
    
    if (histories.length < 2) return 1.0;
    
    // Calculate correlation between activation patterns
    // This is a simplified version - in practice you'd use more sophisticated synchrony measures
    let correlationSum = 0;
    let comparisons = 0;
    
    for (let i = 0; i < histories.length; i++) {
      for (let j = i + 1; j < histories.length; j++) {
        const correlation = this.calculateTemporalCorrelation(histories[i]!, histories[j]!);
        correlationSum += correlation;
        comparisons++;
      }
    }
    
    return comparisons > 0 ? correlationSum / comparisons : 0;
  }

  private calculateTemporalCorrelation(
    hist1: Array<{ timestamp: number; strength: number }>,
    hist2: Array<{ timestamp: number; strength: number }>
  ): number {
    // Simplified correlation calculation
    if (hist1.length < 2 || hist2.length < 2) return 0;
    
    const timeWindow = 1000; // 1 second window
    let correlations: number[] = [];
    
    for (const point1 of hist1) {
      const synchronousPoints = hist2.filter(
        point2 => Math.abs(point1.timestamp - point2.timestamp) <= timeWindow
      );
      
      if (synchronousPoints.length > 0) {
        const avgStrength2 = synchronousPoints.reduce((sum, p) => sum + p.strength, 0) / synchronousPoints.length;
        correlations.push(point1.strength * avgStrength2);
      }
    }
    
    return correlations.length > 0 
      ? correlations.reduce((sum, c) => sum + c, 0) / correlations.length 
      : 0;
  }
}

// Main Hook Implementation
export const useCognitivePatterns = (options: CognitiveOptions = {}) => {
  const {
    patternHistorySize = 200,
    inhibitoryThreshold = 0.7,
    confidenceThreshold = 0.5,
    enablePatternRecognition = true,
    activationWindowMs = 5000,
    hierarchicalDepthTracking = true,
    samplingRate = 30, // 30Hz for cognitive data
  } = options;

  // Use base real-time data hook
  const {
    data: rawCognitiveData,
    latest: latestCognitive,
    isConnected,
    metrics: baseMetrics,
    aggregatedData,
  } = useRealTimeData<CognitiveData>('cognitive', {
    bufferSize: patternHistorySize,
    samplingRate,
    aggregationWindow: activationWindowMs,
    enablePerformanceMonitoring: true,
  });

  // Pattern recognition engine
  const [patternEngine] = useState(() => new PatternRecognitionEngine(patternHistorySize));
  const [activationHistory, setActivationHistory] = useState<ActivationEvent[]>([]);
  const [patternClusters, setPatternClusters] = useState<PatternCluster[]>([]);
  
  // Refs for tracking
  const previousPatternsRef = useRef<CognitivePattern[]>([]);
  const inhibitoryHistoryRef = useRef<Array<{ timestamp: number; level: number }>>([]);

  // Process cognitive patterns
  useEffect(() => {
    if (!latestCognitive || !enablePatternRecognition) return;

    const { patterns, inhibitoryLevel } = latestCognitive;
    const timestamp = Date.now();
    
    // Update pattern engine with new patterns
    patterns.forEach(pattern => {
      patternEngine.updatePattern(pattern);
    });

    // Track inhibitory level
    inhibitoryHistoryRef.current.push({ timestamp, level: inhibitoryLevel });
    if (inhibitoryHistoryRef.current.length > patternHistorySize) {
      inhibitoryHistoryRef.current.shift();
    }

    // Detect pattern activation/deactivation events
    const previousPatterns = previousPatternsRef.current;
    const newActivationEvents: ActivationEvent[] = [];

    // Check for new activations
    patterns.forEach(currentPattern => {
      const previousPattern = previousPatterns.find(p => p.id === currentPattern.id);
      
      if (!previousPattern) {
        // New pattern activation
        newActivationEvents.push({
          timestamp,
          patternId: currentPattern.id,
          type: 'activation',
          value: currentPattern.strength,
          threshold: confidenceThreshold,
          metadata: {
            patternType: currentPattern.type,
            nodeCount: currentPattern.activeNodes.length,
          },
        });
      } else if (Math.abs(currentPattern.strength - previousPattern.strength) > 0.1) {
        // Significant strength change
        newActivationEvents.push({
          timestamp,
          patternId: currentPattern.id,
          type: 'strength_change',
          value: currentPattern.strength,
          threshold: previousPattern.strength,
          metadata: {
            delta: currentPattern.strength - previousPattern.strength,
            patternType: currentPattern.type,
          },
        });
      }
    });

    // Check for deactivations
    previousPatterns.forEach(previousPattern => {
      const stillActive = patterns.some(p => p.id === previousPattern.id);
      if (!stillActive) {
        newActivationEvents.push({
          timestamp,
          patternId: previousPattern.id,
          type: 'deactivation',
          value: 0,
          threshold: confidenceThreshold,
          metadata: {
            lastStrength: previousPattern.strength,
            patternType: previousPattern.type,
          },
        });
      }
    });

    // Update activation history
    setActivationHistory(prev => {
      const updated = [...prev, ...newActivationEvents];
      return updated.slice(-patternHistorySize); // Keep only recent events
    });

    // Update pattern clusters
    if (patterns.length > 0) {
      const clusters = patternEngine.detectPatternClusters(patterns);
      setPatternClusters(clusters);
    }

    // Update previous patterns
    previousPatternsRef.current = patterns;
  }, [latestCognitive, enablePatternRecognition, patternEngine, confidenceThreshold, patternHistorySize]);

  // Calculate inhibitory data
  const inhibitoryData = useMemo((): InhibitoryData => {
    const history = inhibitoryHistoryRef.current;
    if (history.length === 0) {
      return {
        current: 0,
        average: 0,
        trend: 'stable',
        peaks: [],
        valleys: [],
        oscillationFrequency: 0,
      };
    }

    const levels = history.map(h => h.level);
    const average = levels.reduce((sum, l) => sum + l, 0) / levels.length;
    const current = levels[levels.length - 1];

    // Calculate trend
    const recentLevels = levels.slice(-10); // Last 10 samples
    const trend = recentLevels.length >= 3 ? calculateTrend(recentLevels) : 'stable';

    // Find peaks and valleys
    const peaks: Array<{ timestamp: number; level: number }> = [];
    const valleys: Array<{ timestamp: number; level: number }> = [];

    for (let i = 1; i < history.length - 1; i++) {
      const prev = history[i - 1];
      const curr = history[i];
      const next = history[i + 1];

      if (curr.level > prev.level && curr.level > next.level && curr.level > average) {
        peaks.push({ timestamp: curr.timestamp, level: curr.level });
      } else if (curr.level < prev.level && curr.level < next.level && curr.level < average) {
        valleys.push({ timestamp: curr.timestamp, level: curr.level });
      }
    }

    // Calculate oscillation frequency (simplified)
    const oscillationFrequency = peaks.length > 1 
      ? 1000 / ((peaks[peaks.length - 1].timestamp - peaks[0].timestamp) / (peaks.length - 1))
      : 0;

    return {
      current,
      average,
      trend,
      peaks: peaks.slice(-10), // Keep only recent peaks
      valleys: valleys.slice(-10), // Keep only recent valleys
      oscillationFrequency,
    };
  }, [rawCognitiveData]);

  // Calculate pattern strength map
  const patternStrengths = useMemo((): PatternStrengthMap => {
    if (!latestCognitive) return {};

    const strengthMap: PatternStrengthMap = {};

    latestCognitive.patterns.forEach(pattern => {
      const history = rawCognitiveData
        .filter(d => d.patterns.some(p => p.id === pattern.id))
        .map(d => d.patterns.find(p => p.id === pattern.id)!)
        .filter(Boolean);

      const strengths = history.map(p => p.strength);
      const peak = Math.max(...strengths);
      const average = strengths.reduce((sum, s) => sum + s, 0) / strengths.length || 0;
      const stability = patternEngine.getPatternStability(pattern.id);

      // Calculate duration (time since first activation)
      const firstActivation = history[0]?.timestamp || Date.now();
      const duration = Date.now() - firstActivation;

      strengthMap[pattern.id] = {
        current: pattern.strength,
        peak,
        average,
        stability,
        lastActive: pattern.timestamp,
        duration,
      };
    });

    return strengthMap;
  }, [latestCognitive, rawCognitiveData, patternEngine]);

  // Calculate cognitive metrics
  const cognitiveMetrics = useMemo((): CognitiveMetrics => {
    if (!latestCognitive || rawCognitiveData.length === 0) {
      return {
        totalPatterns: 0,
        activePatterns: 0,
        averageStrength: 0,
        inhibitoryEffectiveness: 0,
        hierarchicalBalance: 0,
        patternDiversity: 0,
        activationRate: 0,
        coherenceScore: 0,
      };
    }

    const patterns = latestCognitive.patterns;
    const activePatterns = patterns.filter(p => p.strength > confidenceThreshold);
    
    const averageStrength = patterns.length > 0 
      ? patterns.reduce((sum, p) => sum + p.strength, 0) / patterns.length 
      : 0;

    // Calculate inhibitory effectiveness
    const inhibitoryEffectiveness = inhibitoryData.current > inhibitoryThreshold 
      ? Math.min(activePatterns.length / Math.max(patterns.length, 1), 1) 
      : 1;

    // Calculate hierarchical balance
    const typeCount = patterns.reduce((count, p) => {
      count[p.type] = (count[p.type] || 0) + 1;
      return count;
    }, {} as Record<string, number>);

    const totalTypes = Object.keys(typeCount).length;
    const hierarchicalBalance = totalTypes > 1 
      ? 1 - Math.max(...Object.values(typeCount)) / patterns.length 
      : 0;

    // Calculate pattern diversity (Shannon entropy)
    const patternDiversity = patterns.length > 1 
      ? -Object.values(typeCount).reduce((entropy, count) => {
          const p = count / patterns.length;
          return entropy + p * Math.log2(p);
        }, 0) / Math.log2(totalTypes || 1)
      : 0;

    // Calculate activation rate
    const recentActivations = activationHistory.filter(
      event => Date.now() - event.timestamp <= activationWindowMs
    );
    const activationRate = (recentActivations.length * 1000) / activationWindowMs;

    // Calculate coherence score
    const coherenceScore = patternClusters.length > 0
      ? patternClusters.reduce((sum, cluster) => sum + cluster.coherence, 0) / patternClusters.length
      : 0;

    return {
      totalPatterns: patterns.length,
      activePatterns: activePatterns.length,
      averageStrength,
      inhibitoryEffectiveness,
      hierarchicalBalance,
      patternDiversity,
      activationRate,
      coherenceScore,
    };
  }, [latestCognitive, rawCognitiveData, confidenceThreshold, inhibitoryData, inhibitoryThreshold, activationHistory, activationWindowMs, patternClusters]);

  // Cleanup
  useEffect(() => {
    return () => {
      // Cleanup if needed
    };
  }, []);

  return {
    // Core data
    patterns: latestCognitive?.patterns || [],
    inhibitoryLevels: inhibitoryData,
    patternStrengths,
    activationHistory,
    patternClusters,
    
    // Metrics and analysis
    metrics: cognitiveMetrics,
    isConnected,
    
    // Aggregated and historical data
    historicalData: rawCognitiveData,
    aggregatedData,
    
    // Control methods
    clearHistory: useCallback(() => {
      setActivationHistory([]);
      inhibitoryHistoryRef.current = [];
    }, []),
    
    getPatternTrend: useCallback((patternId: string) => {
      return patternEngine.getPatternTrend(patternId, activationWindowMs);
    }, [patternEngine, activationWindowMs]),
    
    getPatternStability: useCallback((patternId: string) => {
      return patternEngine.getPatternStability(patternId);
    }, [patternEngine]),
  };
};

// Utility functions
function calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
  if (values.length < 2) return 'stable';
  
  const first = values.slice(0, Math.ceil(values.length / 2));
  const second = values.slice(Math.floor(values.length / 2));
  
  const firstAvg = first.reduce((sum, v) => sum + v, 0) / first.length;
  const secondAvg = second.reduce((sum, v) => sum + v, 0) / second.length;
  
  const threshold = 0.05; // 5% change threshold
  const change = (secondAvg - firstAvg) / Math.abs(firstAvg || 1);
  
  if (change > threshold) return 'increasing';
  if (change < -threshold) return 'decreasing';
  return 'stable';
}

// Export types
export type {
  CognitiveOptions,
  InhibitoryData,
  PatternStrengthMap,
  ActivationEvent,
  PatternCluster,
  CognitiveMetrics,
};