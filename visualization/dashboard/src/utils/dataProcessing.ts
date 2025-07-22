/**
 * Advanced Data Processing Utilities for LLMKG Real-time Dashboard
 * 
 * This module provides sophisticated data processing capabilities including:
 * - Time-series aggregation and windowing
 * - Pattern detection and anomaly detection
 * - Statistical analysis and trend prediction
 * - Data compression and sampling
 * - Multi-dimensional data correlation
 */

// Core interfaces
export interface TimeSeriesPoint<T = number> {
  timestamp: number;
  value: T;
  metadata?: Record<string, any>;
}

export interface AggregationWindow<T = number> {
  start: number;
  end: number;
  duration: number;
  aggregationType: AggregationType;
  value: T;
  count: number;
  quality: number;
  confidence: number;
}

export interface PatternDetectionResult {
  pattern: DetectedPattern;
  confidence: number;
  strength: number;
  frequency: number;
  duration: number;
  nextOccurrence?: number;
  metadata: Record<string, any>;
}

export interface AnomalyDetectionResult {
  timestamp: number;
  value: number;
  expectedValue: number;
  deviation: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  type: AnomalyType;
  context: Record<string, any>;
}

export interface TrendAnalysisResult {
  trend: 'increasing' | 'decreasing' | 'stable' | 'oscillating';
  strength: number;
  direction: number; // -1 to 1
  velocity: number;
  acceleration: number;
  confidence: number;
  r2: number; // Coefficient of determination
  forecast: ForecastResult[];
}

export interface ForecastResult {
  timestamp: number;
  predictedValue: number;
  confidence: number;
  lowerBound: number;
  upperBound: number;
}

export interface CorrelationMatrix {
  variables: string[];
  matrix: number[][];
  pValues: number[][];
  significantPairs: Array<{
    variable1: string;
    variable2: string;
    correlation: number;
    pValue: number;
    significance: 'low' | 'medium' | 'high';
  }>;
}

export interface CompressionResult<T> {
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  data: T[];
  metadata: {
    algorithm: string;
    parameters: Record<string, any>;
    quality: number;
    lossType: 'lossless' | 'lossy';
  };
}

// Enums
export enum AggregationType {
  SUM = 'sum',
  AVERAGE = 'average',
  MIN = 'min',
  MAX = 'max',
  COUNT = 'count',
  MEDIAN = 'median',
  PERCENTILE = 'percentile',
  STANDARD_DEVIATION = 'std',
  VARIANCE = 'variance',
  RANGE = 'range',
  FIRST = 'first',
  LAST = 'last'
}

export enum DetectedPattern {
  PERIODIC = 'periodic',
  TRENDING = 'trending',
  SEASONAL = 'seasonal',
  CYCLICAL = 'cyclical',
  SPIKE = 'spike',
  PLATEAU = 'plateau',
  SAWTOOTH = 'sawtooth',
  EXPONENTIAL = 'exponential',
  LOGARITHMIC = 'logarithmic',
  STEP = 'step'
}

export enum AnomalyType {
  STATISTICAL = 'statistical',
  SEASONAL = 'seasonal',
  TREND = 'trend',
  CONTEXTUAL = 'contextual',
  COLLECTIVE = 'collective',
  POINT = 'point'
}

// Configuration interfaces
export interface AggregationConfig {
  windowSize: number;
  stepSize?: number;
  aggregationType: AggregationType;
  percentile?: number; // For percentile aggregation
  weightingFunction?: (timestamp: number, currentTime: number) => number;
  qualityThreshold?: number;
}

export interface PatternDetectionConfig {
  minPatternLength: number;
  maxPatternLength: number;
  similarityThreshold: number;
  frequencyThreshold: number;
  enabledPatterns: DetectedPattern[];
  windowSize?: number;
}

export interface AnomalyDetectionConfig {
  method: 'statistical' | 'isolation_forest' | 'local_outlier_factor' | 'ensemble';
  sensitivityThreshold: number;
  windowSize: number;
  seasonalityPeriod?: number;
  contextVariables?: string[];
}

export interface TrendAnalysisConfig {
  windowSize: number;
  degreeOfPolynomial: number;
  forecastHorizon: number;
  confidenceInterval: number; // e.g., 0.95 for 95% CI
  seasonalityDetection: boolean;
}

/**
 * Time Series Aggregation Engine
 */
export class TimeSeriesAggregator<T = number> {
  private config: AggregationConfig;

  constructor(config: AggregationConfig) {
    this.config = config;
  }

  /**
   * Aggregate time series data into windows
   */
  aggregate(data: TimeSeriesPoint<T>[]): AggregationWindow<T>[] {
    if (data.length === 0) return [];

    const windows: AggregationWindow<T>[] = [];
    const sortedData = data.sort((a, b) => a.timestamp - b.timestamp);
    
    const firstTimestamp = sortedData[0].timestamp;
    const lastTimestamp = sortedData[sortedData.length - 1].timestamp;
    const stepSize = this.config.stepSize || this.config.windowSize;

    for (let start = firstTimestamp; start < lastTimestamp; start += stepSize) {
      const end = start + this.config.windowSize;
      const windowData = sortedData.filter(point => 
        point.timestamp >= start && point.timestamp < end
      );

      if (windowData.length === 0) continue;

      const aggregatedValue = this.aggregateWindow(windowData);
      const quality = this.calculateWindowQuality(windowData, this.config.windowSize);
      const confidence = this.calculateWindowConfidence(windowData);

      windows.push({
        start,
        end,
        duration: this.config.windowSize,
        aggregationType: this.config.aggregationType,
        value: aggregatedValue,
        count: windowData.length,
        quality,
        confidence,
      });
    }

    return windows;
  }

  private aggregateWindow(data: TimeSeriesPoint<T>[]): T {
    if (data.length === 0) return 0 as T;

    const values = data.map(point => point.value);

    switch (this.config.aggregationType) {
      case AggregationType.SUM:
        return values.reduce((sum, val) => (sum as number) + (val as number), 0) as T;

      case AggregationType.AVERAGE:
        const numericValues = values as number[];
        return (numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length) as T;

      case AggregationType.MIN:
        return values.reduce((min, val) => (val as number) < (min as number) ? val : min);

      case AggregationType.MAX:
        return values.reduce((max, val) => (val as number) > (max as number) ? val : max);

      case AggregationType.COUNT:
        return values.length as T;

      case AggregationType.MEDIAN:
        const sorted = (values as number[]).sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return (sorted.length % 2 === 0 
          ? (sorted[mid - 1] + sorted[mid]) / 2 
          : sorted[mid]) as T;

      case AggregationType.PERCENTILE:
        if (!this.config.percentile) return values[0];
        const sortedVals = (values as number[]).sort((a, b) => a - b);
        const index = Math.floor((this.config.percentile / 100) * sortedVals.length);
        return sortedVals[Math.min(index, sortedVals.length - 1)] as T;

      case AggregationType.STANDARD_DEVIATION:
        const nums = values as number[];
        const mean = nums.reduce((sum, val) => sum + val, 0) / nums.length;
        const variance = nums.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / nums.length;
        return Math.sqrt(variance) as T;

      case AggregationType.VARIANCE:
        const numVals = values as number[];
        const avg = numVals.reduce((sum, val) => sum + val, 0) / numVals.length;
        return (numVals.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / numVals.length) as T;

      case AggregationType.RANGE:
        const minVal = Math.min(...(values as number[]));
        const maxVal = Math.max(...(values as number[]));
        return (maxVal - minVal) as T;

      case AggregationType.FIRST:
        return values[0];

      case AggregationType.LAST:
        return values[values.length - 1];

      default:
        return values[0];
    }
  }

  private calculateWindowQuality(data: TimeSeriesPoint<T>[], expectedDuration: number): number {
    if (data.length === 0) return 0;

    const actualDuration = data[data.length - 1].timestamp - data[0].timestamp;
    const coverageRatio = Math.min(actualDuration / expectedDuration, 1);
    
    // Quality based on data density and coverage
    const expectedDataPoints = expectedDuration / 1000; // Assume 1 point per second
    const densityRatio = Math.min(data.length / expectedDataPoints, 1);
    
    return (coverageRatio + densityRatio) / 2;
  }

  private calculateWindowConfidence(data: TimeSeriesPoint<T>[]): number {
    // Confidence based on data consistency and variance
    if (data.length < 2) return 0.5;

    const values = data.map(point => point.value as number);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const cv = mean !== 0 ? Math.sqrt(variance) / Math.abs(mean) : 0;

    // Lower coefficient of variation = higher confidence
    return Math.max(0, Math.min(1, 1 - cv));
  }
}

/**
 * Pattern Detection Engine
 */
export class PatternDetector {
  private config: PatternDetectionConfig;

  constructor(config: PatternDetectionConfig) {
    this.config = config;
  }

  /**
   * Detect patterns in time series data
   */
  detectPatterns(data: TimeSeriesPoint[]): PatternDetectionResult[] {
    const patterns: PatternDetectionResult[] = [];

    if (data.length < this.config.minPatternLength) return patterns;

    const values = data.map(point => point.value);
    const timestamps = data.map(point => point.timestamp);

    // Detect each enabled pattern type
    for (const patternType of this.config.enabledPatterns) {
      const result = this.detectSpecificPattern(values, timestamps, patternType);
      if (result && result.confidence >= this.config.similarityThreshold) {
        patterns.push(result);
      }
    }

    return patterns.sort((a, b) => b.confidence - a.confidence);
  }

  private detectSpecificPattern(
    values: number[], 
    timestamps: number[], 
    patternType: DetectedPattern
  ): PatternDetectionResult | null {
    switch (patternType) {
      case DetectedPattern.PERIODIC:
        return this.detectPeriodicPattern(values, timestamps);
      
      case DetectedPattern.TRENDING:
        return this.detectTrendingPattern(values, timestamps);
      
      case DetectedPattern.SEASONAL:
        return this.detectSeasonalPattern(values, timestamps);
      
      case DetectedPattern.CYCLICAL:
        return this.detectCyclicalPattern(values, timestamps);
      
      case DetectedPattern.SPIKE:
        return this.detectSpikePattern(values, timestamps);
      
      case DetectedPattern.PLATEAU:
        return this.detectPlateauPattern(values, timestamps);
      
      case DetectedPattern.SAWTOOTH:
        return this.detectSawtoothPattern(values, timestamps);
      
      case DetectedPattern.EXPONENTIAL:
        return this.detectExponentialPattern(values, timestamps);
      
      case DetectedPattern.LOGARITHMIC:
        return this.detectLogarithmicPattern(values, timestamps);
      
      case DetectedPattern.STEP:
        return this.detectStepPattern(values, timestamps);
      
      default:
        return null;
    }
  }

  private detectPeriodicPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 6) return null;

    // Use autocorrelation to detect periodicity
    const autocorrelations = this.calculateAutocorrelation(values);
    const peaks = this.findPeaks(autocorrelations);
    
    if (peaks.length === 0) return null;

    const dominantPeak = peaks.reduce((max, peak) => 
      autocorrelations[peak] > autocorrelations[max] ? peak : max
    );

    const period = dominantPeak;
    const strength = autocorrelations[dominantPeak];
    const frequency = timestamps.length > 1 
      ? 1 / ((timestamps[timestamps.length - 1] - timestamps[0]) / timestamps.length * period)
      : 0;

    if (strength < 0.3) return null;

    return {
      pattern: DetectedPattern.PERIODIC,
      confidence: Math.min(strength * 1.5, 1.0),
      strength,
      frequency,
      duration: period * (timestamps.length > 1 ? (timestamps[timestamps.length - 1] - timestamps[0]) / timestamps.length : 1000),
      nextOccurrence: timestamps[timestamps.length - 1] + (period * 1000),
      metadata: { period, autocorrelation: strength }
    };
  }

  private detectTrendingPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 3) return null;

    const { slope, r2 } = this.linearRegression(values);
    
    if (r2 < 0.5) return null; // Weak linear relationship

    const strength = Math.abs(slope) / (Math.max(...values) - Math.min(...values) + 0.001);
    const confidence = r2;

    return {
      pattern: DetectedPattern.TRENDING,
      confidence,
      strength,
      frequency: 0, // Not applicable for trends
      duration: timestamps[timestamps.length - 1] - timestamps[0],
      metadata: { slope, r2, direction: slope > 0 ? 'increasing' : 'decreasing' }
    };
  }

  private detectSeasonalPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    // Simplified seasonal detection - would need more sophisticated analysis for production
    if (values.length < 24) return null; // Need at least 24 data points for basic seasonality

    const seasonLength = Math.floor(values.length / 4);
    const seasons: number[][] = [];
    
    for (let i = 0; i < 4; i++) {
      const start = i * seasonLength;
      const end = (i + 1) * seasonLength;
      seasons.push(values.slice(start, end));
    }

    // Calculate similarity between seasons
    const similarities: number[] = [];
    for (let i = 0; i < seasons.length - 1; i++) {
      const correlation = this.pearsonCorrelation(seasons[i], seasons[i + 1]);
      similarities.push(correlation);
    }

    const avgSimilarity = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
    
    if (avgSimilarity < 0.6) return null; // Not seasonal enough

    return {
      pattern: DetectedPattern.SEASONAL,
      confidence: avgSimilarity,
      strength: avgSimilarity,
      frequency: 4 / (timestamps[timestamps.length - 1] - timestamps[0]),
      duration: seasonLength * (timestamps.length > 1 ? (timestamps[timestamps.length - 1] - timestamps[0]) / timestamps.length : 1000),
      metadata: { seasonLength, avgSimilarity, seasons: seasons.length }
    };
  }

  private detectCyclicalPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    // Similar to periodic but looks for longer-term cycles
    if (values.length < 10) return null;

    const cycles = this.findCycles(values);
    if (cycles.length === 0) return null;

    const averageCycleLength = cycles.reduce((sum, cycle) => sum + cycle.length, 0) / cycles.length;
    const cycleStrength = cycles.reduce((sum, cycle) => sum + cycle.strength, 0) / cycles.length;

    if (cycleStrength < 0.4) return null;

    return {
      pattern: DetectedPattern.CYCLICAL,
      confidence: cycleStrength,
      strength: cycleStrength,
      frequency: 1 / averageCycleLength,
      duration: averageCycleLength * 1000,
      metadata: { averageCycleLength, cycleCount: cycles.length }
    };
  }

  private detectSpikePattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 3) return null;

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const std = Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length);
    
    const spikes = values.filter(val => Math.abs(val - mean) > 2 * std);
    const spikeRatio = spikes.length / values.length;
    
    if (spikeRatio < 0.05) return null; // Less than 5% spikes

    const avgSpikeIntensity = spikes.reduce((sum, spike) => sum + Math.abs(spike - mean), 0) / spikes.length;
    const strength = Math.min(avgSpikeIntensity / (3 * std), 1);

    return {
      pattern: DetectedPattern.SPIKE,
      confidence: Math.min(spikeRatio * 10, 1), // Scale spike ratio
      strength,
      frequency: spikes.length / (timestamps[timestamps.length - 1] - timestamps[0]) * 1000,
      duration: 0, // Spikes are instantaneous
      metadata: { spikeCount: spikes.length, spikeRatio, avgIntensity: avgSpikeIntensity }
    };
  }

  private detectPlateauPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 5) return null;

    // Find regions of low variance (plateaus)
    const windowSize = Math.max(3, Math.floor(values.length / 10));
    const plateaus: Array<{start: number, end: number, value: number, variance: number}> = [];

    for (let i = 0; i <= values.length - windowSize; i++) {
      const window = values.slice(i, i + windowSize);
      const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
      const variance = window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length;
      
      if (variance < 0.01 * Math.pow(Math.max(...values) - Math.min(...values), 2)) { // Low variance threshold
        plateaus.push({
          start: i,
          end: i + windowSize,
          value: mean,
          variance
        });
      }
    }

    if (plateaus.length === 0) return null;

    // Merge adjacent plateaus
    const mergedPlateaus = this.mergePlateaus(plateaus);
    const longestPlateau = mergedPlateaus.reduce((max, plateau) => 
      (plateau.end - plateau.start) > (max.end - max.start) ? plateau : max
    );

    const plateauRatio = (longestPlateau.end - longestPlateau.start) / values.length;
    
    if (plateauRatio < 0.2) return null; // At least 20% of data should be plateau

    return {
      pattern: DetectedPattern.PLATEAU,
      confidence: Math.min(plateauRatio * 2, 1),
      strength: 1 - longestPlateau.variance,
      frequency: 0,
      duration: (longestPlateau.end - longestPlateau.start) * 1000,
      metadata: { plateauCount: mergedPlateaus.length, longestDuration: longestPlateau.end - longestPlateau.start }
    };
  }

  private detectSawtoothPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 6) return null;

    // Look for repeated patterns of gradual increase/decrease followed by sharp change
    const peaks = this.findPeaks(values);
    const troughs = this.findPeaks(values.map(v => -v)).map(i => ({ index: i, value: -values[i] }));

    if (peaks.length < 2 || troughs.length < 2) return null;

    // Check for alternating peaks and troughs
    const alternatingScore = this.calculateAlternatingScore(peaks, troughs);
    
    if (alternatingScore < 0.6) return null;

    const avgPeriod = this.calculateAveragePeriod(peaks, troughs);
    const strength = this.calculateSawtoothStrength(values, peaks, troughs);

    return {
      pattern: DetectedPattern.SAWTOOTH,
      confidence: alternatingScore,
      strength,
      frequency: 1 / avgPeriod,
      duration: avgPeriod * 1000,
      metadata: { peakCount: peaks.length, troughCount: troughs.length, avgPeriod }
    };
  }

  private detectExponentialPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 4) return null;

    // Fit exponential curve: y = a * e^(b*x)
    // Transform to linear: ln(y) = ln(a) + b*x
    const positiveValues = values.filter(v => v > 0);
    if (positiveValues.length < values.length * 0.8) return null; // Need mostly positive values

    const logValues = positiveValues.map(v => Math.log(v));
    const indices = positiveValues.map((_, i) => i);

    const { slope, r2 } = this.linearRegression(logValues, indices);
    
    if (r2 < 0.7) return null; // Not a good exponential fit

    const strength = Math.abs(slope);
    const confidence = r2;

    return {
      pattern: DetectedPattern.EXPONENTIAL,
      confidence,
      strength,
      frequency: 0,
      duration: timestamps[timestamps.length - 1] - timestamps[0],
      metadata: { exponent: slope, r2, direction: slope > 0 ? 'growth' : 'decay' }
    };
  }

  private detectLogarithmicPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 4) return null;

    // Fit logarithmic curve: y = a * ln(x) + b
    const indices = values.map((_, i) => i + 1); // Avoid ln(0)
    const logIndices = indices.map(i => Math.log(i));

    const { slope, r2 } = this.linearRegression(values, logIndices);
    
    if (r2 < 0.7) return null; // Not a good logarithmic fit

    const strength = Math.abs(slope);
    const confidence = r2;

    return {
      pattern: DetectedPattern.LOGARITHMIC,
      confidence,
      strength,
      frequency: 0,
      duration: timestamps[timestamps.length - 1] - timestamps[0],
      metadata: { coefficient: slope, r2 }
    };
  }

  private detectStepPattern(values: number[], timestamps: number[]): PatternDetectionResult | null {
    if (values.length < 4) return null;

    const steps = this.findStepChanges(values);
    if (steps.length === 0) return null;

    const avgStepSize = steps.reduce((sum, step) => sum + Math.abs(step.magnitude), 0) / steps.length;
    const totalRange = Math.max(...values) - Math.min(...values);
    const stepRatio = avgStepSize / totalRange;

    if (stepRatio < 0.2) return null; // Steps should be significant

    return {
      pattern: DetectedPattern.STEP,
      confidence: Math.min(stepRatio * 2, 1),
      strength: stepRatio,
      frequency: steps.length / (timestamps[timestamps.length - 1] - timestamps[0]) * 1000,
      duration: 0, // Steps are instantaneous
      metadata: { stepCount: steps.length, avgStepSize, totalRange }
    };
  }

  // Helper methods for pattern detection
  private calculateAutocorrelation(values: number[]): number[] {
    const n = values.length;
    const mean = values.reduce((sum, val) => sum + val, 0) / n;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    
    const autocorr: number[] = [];
    
    for (let lag = 1; lag < Math.floor(n / 2); lag++) {
      let covariance = 0;
      for (let i = 0; i < n - lag; i++) {
        covariance += (values[i] - mean) * (values[i + lag] - mean);
      }
      covariance /= (n - lag);
      
      autocorr[lag] = covariance / variance;
    }
    
    return autocorr;
  }

  private findPeaks(values: number[]): number[] {
    const peaks: number[] = [];
    
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
        peaks.push(i);
      }
    }
    
    return peaks;
  }

  private linearRegression(y: number[], x?: number[]): { slope: number; intercept: number; r2: number } {
    const xValues = x || y.map((_, i) => i);
    const n = y.length;
    
    const sumX = xValues.reduce((sum, val) => sum + val, 0);
    const sumY = y.reduce((sum, val) => sum + val, 0);
    const sumXY = xValues.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumXX = xValues.reduce((sum, val) => sum + val * val, 0);
    const sumYY = y.reduce((sum, val) => sum + val * val, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R²
    const yMean = sumY / n;
    const ssRes = y.reduce((sum, val, i) => {
      const predicted = slope * xValues[i] + intercept;
      return sum + Math.pow(val - predicted, 2);
    }, 0);
    const ssTot = y.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
    const r2 = 1 - (ssRes / ssTot);
    
    return { slope, intercept, r2 };
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;
    
    const sumX = x.slice(0, n).reduce((sum, val) => sum + val, 0);
    const sumY = y.slice(0, n).reduce((sum, val) => sum + val, 0);
    const sumXY = x.slice(0, n).reduce((sum, val, i) => sum + val * y[i], 0);
    const sumXX = x.slice(0, n).reduce((sum, val) => sum + val * val, 0);
    const sumYY = y.slice(0, n).reduce((sum, val) => sum + val * val, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private findCycles(values: number[]): Array<{ length: number; strength: number }> {
    // Simplified cycle detection
    const cycles: Array<{ length: number; strength: number }> = [];
    const minCycleLength = 4;
    const maxCycleLength = Math.floor(values.length / 2);

    for (let length = minCycleLength; length <= maxCycleLength; length++) {
      const segments: number[][] = [];
      for (let i = 0; i <= values.length - length; i += length) {
        segments.push(values.slice(i, i + length));
      }

      if (segments.length < 2) continue;

      let totalCorrelation = 0;
      let comparisons = 0;

      for (let i = 0; i < segments.length - 1; i++) {
        const correlation = this.pearsonCorrelation(segments[i], segments[i + 1]);
        totalCorrelation += correlation;
        comparisons++;
      }

      const avgCorrelation = comparisons > 0 ? totalCorrelation / comparisons : 0;
      if (avgCorrelation > 0.6) {
        cycles.push({ length, strength: avgCorrelation });
      }
    }

    return cycles;
  }

  private mergePlateaus(plateaus: Array<{start: number, end: number, value: number, variance: number}>): typeof plateaus {
    if (plateaus.length === 0) return [];

    const merged = [plateaus[0]];
    
    for (let i = 1; i < plateaus.length; i++) {
      const current = plateaus[i];
      const last = merged[merged.length - 1];
      
      if (current.start <= last.end + 1 && Math.abs(current.value - last.value) < 0.1) {
        // Merge adjacent plateaus with similar values
        last.end = current.end;
        last.variance = Math.max(last.variance, current.variance);
      } else {
        merged.push(current);
      }
    }
    
    return merged;
  }

  private calculateAlternatingScore(peaks: number[], troughs: Array<{index: number, value: number}>): number {
    // Check how well peaks and troughs alternate
    const allExtrema = [...peaks.map(p => ({index: p, type: 'peak'})), 
                       ...troughs.map(t => ({index: t.index, type: 'trough'}))];
    allExtrema.sort((a, b) => a.index - b.index);
    
    if (allExtrema.length < 4) return 0;
    
    let alternatingCount = 0;
    for (let i = 1; i < allExtrema.length; i++) {
      if (allExtrema[i].type !== allExtrema[i - 1].type) {
        alternatingCount++;
      }
    }
    
    return alternatingCount / (allExtrema.length - 1);
  }

  private calculateAveragePeriod(peaks: number[], troughs: Array<{index: number, value: number}>): number {
    const allIndices = [...peaks, ...troughs.map(t => t.index)].sort((a, b) => a - b);
    
    if (allIndices.length < 2) return 1;
    
    const periods: number[] = [];
    for (let i = 1; i < allIndices.length; i++) {
      periods.push(allIndices[i] - allIndices[i - 1]);
    }
    
    return periods.reduce((sum, period) => sum + period, 0) / periods.length;
  }

  private calculateSawtoothStrength(values: number[], peaks: number[], troughs: Array<{index: number, value: number}>): number {
    if (peaks.length === 0 || troughs.length === 0) return 0;
    
    const peakValues = peaks.map(i => values[i]);
    const troughValues = troughs.map(t => t.value);
    
    const avgPeak = peakValues.reduce((sum, val) => sum + val, 0) / peakValues.length;
    const avgTrough = troughValues.reduce((sum, val) => sum + val, 0) / troughValues.length;
    
    const amplitude = Math.abs(avgPeak - avgTrough);
    const range = Math.max(...values) - Math.min(...values);
    
    return range > 0 ? amplitude / range : 0;
  }

  private findStepChanges(values: number[]): Array<{ index: number; magnitude: number; direction: 'up' | 'down' }> {
    const steps: Array<{ index: number; magnitude: number; direction: 'up' | 'down' }> = [];
    const threshold = 0.1 * (Math.max(...values) - Math.min(...values)); // 10% of range
    
    for (let i = 1; i < values.length; i++) {
      const change = values[i] - values[i - 1];
      if (Math.abs(change) > threshold) {
        steps.push({
          index: i,
          magnitude: Math.abs(change),
          direction: change > 0 ? 'up' : 'down'
        });
      }
    }
    
    return steps;
  }
}

/**
 * Anomaly Detection Engine
 */
export class AnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: AnomalyDetectionConfig) {
    this.config = config;
  }

  /**
   * Detect anomalies in time series data
   */
  detectAnomalies(data: TimeSeriesPoint[]): AnomalyDetectionResult[] {
    if (data.length < this.config.windowSize) return [];

    switch (this.config.method) {
      case 'statistical':
        return this.statisticalAnomalyDetection(data);
      
      case 'isolation_forest':
        return this.isolationForestDetection(data);
      
      case 'local_outlier_factor':
        return this.localOutlierFactorDetection(data);
      
      case 'ensemble':
        return this.ensembleAnomalyDetection(data);
      
      default:
        return this.statisticalAnomalyDetection(data);
    }
  }

  private statisticalAnomalyDetection(data: TimeSeriesPoint[]): AnomalyDetectionResult[] {
    const anomalies: AnomalyDetectionResult[] = [];
    const values = data.map(point => point.value);
    
    // Moving window statistics
    for (let i = this.config.windowSize; i < data.length; i++) {
      const window = values.slice(i - this.config.windowSize, i);
      const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
      const std = Math.sqrt(window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length);
      
      const currentValue = values[i];
      const zScore = std > 0 ? Math.abs(currentValue - mean) / std : 0;
      
      if (zScore > this.config.sensitivityThreshold) {
        const deviation = Math.abs(currentValue - mean);
        const severity = this.calculateSeverity(zScore);
        
        anomalies.push({
          timestamp: data[i].timestamp,
          value: currentValue,
          expectedValue: mean,
          deviation,
          severity,
          confidence: Math.min(zScore / 4, 1), // Normalize z-score to confidence
          type: AnomalyType.STATISTICAL,
          context: {
            zScore,
            windowMean: mean,
            windowStd: std,
            windowSize: this.config.windowSize
          }
        });
      }
    }
    
    return anomalies;
  }

  private isolationForestDetection(data: TimeSeriesPoint[]): AnomalyDetectionResult[] {
    // Simplified isolation forest implementation
    // In production, you'd use a proper isolation forest algorithm
    const anomalies: AnomalyDetectionResult[] = [];
    const values = data.map(point => point.value);
    
    // Create features: value, local trend, local variance
    const features: number[][] = [];
    for (let i = 2; i < values.length; i++) {
      const value = values[i];
      const trend = values[i] - values[i - 2]; // 2-point trend
      const localValues = values.slice(Math.max(0, i - 5), i + 1);
      const variance = localValues.reduce((sum, val) => {
        const mean = localValues.reduce((s, v) => s + v, 0) / localValues.length;
        return sum + Math.pow(val - mean, 2);
      }, 0) / localValues.length;
      
      features.push([value, trend, variance]);
    }
    
    // Simple anomaly scoring based on feature deviation
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const scores = feature.map((f, idx) => {
        const allFeaturesAtIdx = features.map(feat => feat[idx]);
        const mean = allFeaturesAtIdx.reduce((sum, val) => sum + val, 0) / allFeaturesAtIdx.length;
        const std = Math.sqrt(allFeaturesAtIdx.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / allFeaturesAtIdx.length);
        return std > 0 ? Math.abs(f - mean) / std : 0;
      });
      
      const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      
      if (avgScore > this.config.sensitivityThreshold) {
        const dataIndex = i + 2; // Adjust for feature window
        anomalies.push({
          timestamp: data[dataIndex].timestamp,
          value: values[dataIndex],
          expectedValue: features.map(f => f[0]).reduce((sum, val) => sum + val, 0) / features.length,
          deviation: avgScore,
          severity: this.calculateSeverity(avgScore),
          confidence: Math.min(avgScore / 3, 1),
          type: AnomalyType.CONTEXTUAL,
          context: {
            isolationScore: avgScore,
            featureScores: scores
          }
        });
      }
    }
    
    return anomalies;
  }

  private localOutlierFactorDetection(data: TimeSeriesPoint[]): AnomalyDetectionResult[] {
    // Simplified LOF implementation
    const anomalies: AnomalyDetectionResult[] = [];
    const values = data.map(point => point.value);
    const k = Math.min(20, Math.floor(values.length / 10)); // Neighborhood size
    
    for (let i = k; i < values.length - k; i++) {
      const point = values[i];
      const neighbors = [];
      
      // Find k nearest neighbors
      for (let j = Math.max(0, i - k); j <= Math.min(values.length - 1, i + k); j++) {
        if (j !== i) {
          neighbors.push({
            index: j,
            value: values[j],
            distance: Math.abs(values[j] - point)
          });
        }
      }
      
      neighbors.sort((a, b) => a.distance - b.distance);
      const kNeighbors = neighbors.slice(0, k);
      
      // Calculate local density
      const avgDistance = kNeighbors.reduce((sum, n) => sum + n.distance, 0) / kNeighbors.length;
      const localDensity = avgDistance > 0 ? 1 / avgDistance : Infinity;
      
      // Calculate neighbor densities
      const neighborDensities = kNeighbors.map(neighbor => {
        const neighborNeighbors = [];
        for (let l = Math.max(0, neighbor.index - k); l <= Math.min(values.length - 1, neighbor.index + k); l++) {
          if (l !== neighbor.index) {
            neighborNeighbors.push(Math.abs(values[l] - neighbor.value));
          }
        }
        neighborNeighbors.sort((a, b) => a - b);
        const avgNeighborDistance = neighborNeighbors.slice(0, k).reduce((sum, d) => sum + d, 0) / k;
        return avgNeighborDistance > 0 ? 1 / avgNeighborDistance : Infinity;
      });
      
      // Calculate LOF score
      const avgNeighborDensity = neighborDensities.reduce((sum, d) => sum + d, 0) / neighborDensities.length;
      const lofScore = localDensity > 0 ? avgNeighborDensity / localDensity : 1;
      
      if (lofScore > this.config.sensitivityThreshold) {
        anomalies.push({
          timestamp: data[i].timestamp,
          value: point,
          expectedValue: kNeighbors.reduce((sum, n) => sum + n.value, 0) / kNeighbors.length,
          deviation: lofScore - 1,
          severity: this.calculateSeverity(lofScore),
          confidence: Math.min((lofScore - 1) / 2, 1),
          type: AnomalyType.POINT,
          context: {
            lofScore,
            localDensity,
            avgNeighborDensity,
            kNeighbors: k
          }
        });
      }
    }
    
    return anomalies;
  }

  private ensembleAnomalyDetection(data: TimeSeriesPoint[]): AnomalyDetectionResult[] {
    // Combine multiple methods
    const statisticalAnomalies = this.statisticalAnomalyDetection(data);
    const isolationAnomalies = this.isolationForestDetection(data);
    const lofAnomalies = this.localOutlierFactorDetection(data);
    
    // Merge and weight results
    const allAnomalies = new Map<number, AnomalyDetectionResult>();
    
    // Add statistical anomalies with weight 1.0
    statisticalAnomalies.forEach(anomaly => {
      allAnomalies.set(anomaly.timestamp, anomaly);
    });
    
    // Add or enhance with isolation forest results (weight 0.8)
    isolationAnomalies.forEach(anomaly => {
      const existing = allAnomalies.get(anomaly.timestamp);
      if (existing) {
        existing.confidence = Math.min(1, existing.confidence + anomaly.confidence * 0.8);
        existing.deviation = Math.max(existing.deviation, anomaly.deviation);
      } else {
        anomaly.confidence *= 0.8;
        allAnomalies.set(anomaly.timestamp, anomaly);
      }
    });
    
    // Add or enhance with LOF results (weight 0.6)
    lofAnomalies.forEach(anomaly => {
      const existing = allAnomalies.get(anomaly.timestamp);
      if (existing) {
        existing.confidence = Math.min(1, existing.confidence + anomaly.confidence * 0.6);
        existing.deviation = Math.max(existing.deviation, anomaly.deviation);
      } else {
        anomaly.confidence *= 0.6;
        allAnomalies.set(anomaly.timestamp, anomaly);
      }
    });
    
    return Array.from(allAnomalies.values())
      .sort((a, b) => b.confidence - a.confidence);
  }

  private calculateSeverity(score: number): 'low' | 'medium' | 'high' | 'critical' {
    if (score < 2) return 'low';
    if (score < 3) return 'medium';
    if (score < 4) return 'high';
    return 'critical';
  }
}

/**
 * Trend Analysis Engine
 */
export class TrendAnalyzer {
  private config: TrendAnalysisConfig;

  constructor(config: TrendAnalysisConfig) {
    this.config = config;
  }

  /**
   * Analyze trends and generate forecasts
   */
  analyzeTrend(data: TimeSeriesPoint[]): TrendAnalysisResult {
    if (data.length < this.config.windowSize) {
      return this.getEmptyTrendResult();
    }

    const values = data.map(point => point.value);
    const timestamps = data.map(point => point.timestamp);
    
    // Perform polynomial regression
    const regression = this.polynomialRegression(values, this.config.degreeOfPolynomial);
    
    // Calculate trend characteristics
    const trend = this.determineTrend(regression.coefficients);
    const strength = Math.abs(regression.coefficients[1] || 0); // Linear coefficient
    const direction = Math.sign(regression.coefficients[1] || 0);
    
    // Calculate velocity (first derivative) and acceleration (second derivative)
    const velocity = this.calculateVelocity(regression.coefficients);
    const acceleration = this.calculateAcceleration(regression.coefficients);
    
    // Generate forecast
    const forecast = this.generateForecast(regression.coefficients, timestamps, this.config.forecastHorizon);
    
    return {
      trend,
      strength,
      direction,
      velocity,
      acceleration,
      confidence: regression.r2,
      r2: regression.r2,
      forecast
    };
  }

  private polynomialRegression(y: number[], degree: number): { coefficients: number[]; r2: number } {
    const n = y.length;
    const x = Array.from({ length: n }, (_, i) => i);
    
    // Create design matrix
    const X: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j <= degree; j++) {
        row.push(Math.pow(x[i], j));
      }
      X.push(row);
    }
    
    // Solve using normal equation: (X^T * X)^-1 * X^T * y
    const XT = this.transpose(X);
    const XTX = this.matrixMultiply(XT, X);
    const XTXInv = this.matrixInverse(XTX);
    const XTy = this.vectorMatrixMultiply(XT, y);
    const coefficients = this.vectorMatrixMultiply(XTXInv, XTy);
    
    // Calculate R²
    const yMean = y.reduce((sum, val) => sum + val, 0) / n;
    let ssRes = 0;
    let ssTot = 0;
    
    for (let i = 0; i < n; i++) {
      const predicted = coefficients.reduce((sum, coef, j) => sum + coef * Math.pow(x[i], j), 0);
      ssRes += Math.pow(y[i] - predicted, 2);
      ssTot += Math.pow(y[i] - yMean, 2);
    }
    
    const r2 = 1 - (ssRes / ssTot);
    
    return { coefficients, r2 };
  }

  private determineTrend(coefficients: number[]): 'increasing' | 'decreasing' | 'stable' | 'oscillating' {
    if (coefficients.length < 2) return 'stable';
    
    const linearCoef = coefficients[1];
    const quadraticCoef = coefficients[2] || 0;
    
    if (Math.abs(linearCoef) < 0.01) return 'stable';
    
    // Check for oscillation (higher-order terms with alternating signs)
    if (coefficients.length > 3) {
      const hasOscillation = coefficients.slice(2).some((coef, idx) => 
        idx > 0 && Math.sign(coef) !== Math.sign(coefficients[idx + 1])
      );
      if (hasOscillation && Math.abs(quadraticCoef) > Math.abs(linearCoef)) {
        return 'oscillating';
      }
    }
    
    return linearCoef > 0 ? 'increasing' : 'decreasing';
  }

  private calculateVelocity(coefficients: number[]): number {
    // First derivative at the end point
    if (coefficients.length < 2) return 0;
    
    const n = coefficients.length - 1; // Last point index
    let velocity = 0;
    
    for (let i = 1; i < coefficients.length; i++) {
      velocity += i * coefficients[i] * Math.pow(n, i - 1);
    }
    
    return velocity;
  }

  private calculateAcceleration(coefficients: number[]): number {
    // Second derivative at the end point
    if (coefficients.length < 3) return 0;
    
    const n = coefficients.length - 1; // Last point index
    let acceleration = 0;
    
    for (let i = 2; i < coefficients.length; i++) {
      acceleration += i * (i - 1) * coefficients[i] * Math.pow(n, i - 2);
    }
    
    return acceleration;
  }

  private generateForecast(
    coefficients: number[], 
    timestamps: number[], 
    horizonMs: number
  ): ForecastResult[] {
    const forecast: ForecastResult[] = [];
    const lastTimestamp = timestamps[timestamps.length - 1];
    const timeStep = timestamps.length > 1 ? (timestamps[timestamps.length - 1] - timestamps[0]) / (timestamps.length - 1) : 1000;
    
    const forecastPoints = Math.floor(horizonMs / timeStep);
    const dataLength = timestamps.length;
    
    for (let i = 1; i <= forecastPoints; i++) {
      const x = dataLength + i - 1; // Continue the sequence
      let predictedValue = 0;
      
      // Calculate polynomial value
      for (let j = 0; j < coefficients.length; j++) {
        predictedValue += coefficients[j] * Math.pow(x, j);
      }
      
      // Calculate confidence bounds (simplified)
      const confidence = Math.max(0.1, 0.95 - (i / forecastPoints) * 0.4); // Decreasing confidence
      const errorMargin = Math.abs(predictedValue) * (1 - confidence);
      
      forecast.push({
        timestamp: lastTimestamp + i * timeStep,
        predictedValue,
        confidence,
        lowerBound: predictedValue - errorMargin,
        upperBound: predictedValue + errorMargin
      });
    }
    
    return forecast;
  }

  private getEmptyTrendResult(): TrendAnalysisResult {
    return {
      trend: 'stable',
      strength: 0,
      direction: 0,
      velocity: 0,
      acceleration: 0,
      confidence: 0,
      r2: 0,
      forecast: []
    };
  }

  // Matrix operations helpers
  private transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  private matrixMultiply(a: number[][], b: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < b.length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  private vectorMatrixMultiply(matrix: number[][], vector: number[]): number[] {
    return matrix.map(row => 
      row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
    );
  }

  private matrixInverse(matrix: number[][]): number[][] {
    // Simplified matrix inversion for small matrices (Gauss-Jordan elimination)
    const n = matrix.length;
    const augmented: number[][] = matrix.map((row, i) => 
      [...row, ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]
    );
    
    // Forward elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      
      // Swap rows
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
      
      // Make diagonal element 1
      const pivot = augmented[i][i];
      if (Math.abs(pivot) < 1e-10) continue; // Skip near-zero pivots
      
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }
      
      // Eliminate column
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = augmented[k][i];
          for (let j = 0; j < 2 * n; j++) {
            augmented[k][j] -= factor * augmented[i][j];
          }
        }
      }
    }
    
    // Extract inverse matrix
    return augmented.map(row => row.slice(n));
  }
}

/**
 * Data Compression Engine
 */
export class DataCompressor {
  /**
   * Compress time series data using various algorithms
   */
  static compressTimeSeries<T>(
    data: TimeSeriesPoint<T>[],
    algorithm: 'douglas_peucker' | 'uniform_sampling' | 'adaptive_sampling' | 'perceptually_important_points',
    parameters: Record<string, any> = {}
  ): CompressionResult<TimeSeriesPoint<T>> {
    switch (algorithm) {
      case 'douglas_peucker':
        return this.douglasPeuckerCompression(data, parameters.tolerance || 1.0);
      
      case 'uniform_sampling':
        return this.uniformSamplingCompression(data, parameters.sampleRate || 0.5);
      
      case 'adaptive_sampling':
        return this.adaptiveSamplingCompression(data, parameters);
      
      case 'perceptually_important_points':
        return this.perceptuallyImportantPointsCompression(data, parameters.threshold || 0.1);
      
      default:
        return {
          originalSize: data.length,
          compressedSize: data.length,
          compressionRatio: 1.0,
          data,
          metadata: {
            algorithm: 'none',
            parameters: {},
            quality: 1.0,
            lossType: 'lossless'
          }
        };
    }
  }

  private static douglasPeuckerCompression<T>(
    data: TimeSeriesPoint<T>[],
    tolerance: number
  ): CompressionResult<TimeSeriesPoint<T>> {
    if (data.length <= 2) {
      return {
        originalSize: data.length,
        compressedSize: data.length,
        compressionRatio: 1.0,
        data,
        metadata: {
          algorithm: 'douglas_peucker',
          parameters: { tolerance },
          quality: 1.0,
          lossType: 'lossy'
        }
      };
    }

    const compressed = this.douglasPeuckerRecursive(data, tolerance);
    
    return {
      originalSize: data.length,
      compressedSize: compressed.length,
      compressionRatio: compressed.length / data.length,
      data: compressed,
      metadata: {
        algorithm: 'douglas_peucker',
        parameters: { tolerance },
        quality: this.calculateCompressionQuality(data, compressed),
        lossType: 'lossy'
      }
    };
  }

  private static douglasPeuckerRecursive<T>(
    data: TimeSeriesPoint<T>[],
    tolerance: number
  ): TimeSeriesPoint<T>[] {
    if (data.length <= 2) return data;

    const first = data[0];
    const last = data[data.length - 1];
    
    // Find point with maximum distance from line connecting first and last points
    let maxDistance = 0;
    let maxIndex = 0;
    
    for (let i = 1; i < data.length - 1; i++) {
      const distance = this.perpendicularDistance(
        data[i].timestamp, data[i].value as number,
        first.timestamp, first.value as number,
        last.timestamp, last.value as number
      );
      
      if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = i;
      }
    }
    
    // If max distance is greater than tolerance, recursively simplify
    if (maxDistance > tolerance) {
      const left = this.douglasPeuckerRecursive(data.slice(0, maxIndex + 1), tolerance);
      const right = this.douglasPeuckerRecursive(data.slice(maxIndex), tolerance);
      
      return [...left.slice(0, -1), ...right];
    } else {
      return [first, last];
    }
  }

  private static perpendicularDistance(
    px: number, py: number,
    x1: number, y1: number,
    x2: number, y2: number
  ): number {
    const A = x2 - x1;
    const B = y2 - y1;
    const C = x1 - px;
    const D = y1 - py;
    
    const dot = A * C + B * D;
    const lenSq = A * A + B * B;
    
    if (lenSq === 0) return Math.sqrt(C * C + D * D);
    
    const param = -dot / lenSq;
    
    let xx: number, yy: number;
    
    if (param < 0) {
      xx = x1;
      yy = y1;
    } else if (param > 1) {
      xx = x2;
      yy = y2;
    } else {
      xx = x1 + param * A;
      yy = y1 + param * B;
    }
    
    const dx = px - xx;
    const dy = py - yy;
    
    return Math.sqrt(dx * dx + dy * dy);
  }

  private static uniformSamplingCompression<T>(
    data: TimeSeriesPoint<T>[],
    sampleRate: number
  ): CompressionResult<TimeSeriesPoint<T>> {
    const step = Math.max(1, Math.floor(1 / sampleRate));
    const compressed = data.filter((_, index) => index % step === 0);
    
    return {
      originalSize: data.length,
      compressedSize: compressed.length,
      compressionRatio: compressed.length / data.length,
      data: compressed,
      metadata: {
        algorithm: 'uniform_sampling',
        parameters: { sampleRate, step },
        quality: this.calculateCompressionQuality(data, compressed),
        lossType: 'lossy'
      }
    };
  }

  private static adaptiveSamplingCompression<T>(
    data: TimeSeriesPoint<T>[],
    parameters: Record<string, any>
  ): CompressionResult<TimeSeriesPoint<T>> {
    const threshold = parameters.threshold || 0.1;
    const minSamples = parameters.minSamples || 10;
    
    if (data.length <= minSamples) {
      return {
        originalSize: data.length,
        compressedSize: data.length,
        compressionRatio: 1.0,
        data,
        metadata: {
          algorithm: 'adaptive_sampling',
          parameters,
          quality: 1.0,
          lossType: 'lossy'
        }
      };
    }

    const compressed: TimeSeriesPoint<T>[] = [data[0]]; // Always keep first point
    const values = data.map(point => point.value as number);
    
    // Calculate local variance for each point
    for (let i = 1; i < data.length - 1; i++) {
      const windowSize = Math.min(5, Math.floor(data.length / 10));
      const start = Math.max(0, i - windowSize);
      const end = Math.min(data.length, i + windowSize + 1);
      const window = values.slice(start, end);
      
      const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
      const variance = window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length;
      const normalizedVariance = variance / (Math.max(...values) - Math.min(...values) + 0.001);
      
      // Keep points with high local variance
      if (normalizedVariance > threshold) {
        compressed.push(data[i]);
      }
    }
    
    compressed.push(data[data.length - 1]); // Always keep last point
    
    return {
      originalSize: data.length,
      compressedSize: compressed.length,
      compressionRatio: compressed.length / data.length,
      data: compressed,
      metadata: {
        algorithm: 'adaptive_sampling',
        parameters,
        quality: this.calculateCompressionQuality(data, compressed),
        lossType: 'lossy'
      }
    };
  }

  private static perceptuallyImportantPointsCompression<T>(
    data: TimeSeriesPoint<T>[],
    threshold: number
  ): CompressionResult<TimeSeriesPoint<T>> {
    if (data.length <= 3) {
      return {
        originalSize: data.length,
        compressedSize: data.length,
        compressionRatio: 1.0,
        data,
        metadata: {
          algorithm: 'perceptually_important_points',
          parameters: { threshold },
          quality: 1.0,
          lossType: 'lossy'
        }
      };
    }

    const compressed: TimeSeriesPoint<T>[] = [data[0]]; // Always keep first point
    const values = data.map(point => point.value as number);
    
    for (let i = 1; i < data.length - 1; i++) {
      const prev = values[i - 1];
      const curr = values[i];
      const next = values[i + 1];
      
      // Calculate perceptual importance based on local curvature
      const curvature = Math.abs(2 * curr - prev - next);
      const normalizedCurvature = curvature / (Math.max(...values) - Math.min(...values) + 0.001);
      
      // Keep points with high curvature (turning points)
      if (normalizedCurvature > threshold) {
        compressed.push(data[i]);
      }
    }
    
    compressed.push(data[data.length - 1]); // Always keep last point
    
    return {
      originalSize: data.length,
      compressedSize: compressed.length,
      compressionRatio: compressed.length / data.length,
      data: compressed,
      metadata: {
        algorithm: 'perceptually_important_points',
        parameters: { threshold },
        quality: this.calculateCompressionQuality(data, compressed),
        lossType: 'lossy'
      }
    };
  }

  private static calculateCompressionQuality<T>(
    original: TimeSeriesPoint<T>[],
    compressed: TimeSeriesPoint<T>[]
  ): number {
    if (original.length === 0 || compressed.length === 0) return 0;
    
    // Simple quality metric based on coverage
    const coverageRatio = compressed.length / original.length;
    
    // TODO: Implement more sophisticated quality metrics like RMSE, etc.
    return Math.min(1.0, coverageRatio + 0.3); // Boost for reasonable compression
  }
}

/**
 * Utility functions for data processing
 */
export class DataProcessingUtils {
  /**
   * Calculate correlation matrix for multiple time series
   */
  static calculateCorrelationMatrix(
    dataSeries: Record<string, TimeSeriesPoint[]>,
    method: 'pearson' | 'spearman' | 'kendall' = 'pearson'
  ): CorrelationMatrix {
    const variables = Object.keys(dataSeries);
    const n = variables.length;
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
    const pValues: number[][] = Array(n).fill(null).map(() => Array(n).fill(1));
    const significantPairs: CorrelationMatrix['significantPairs'] = [];
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
          pValues[i][j] = 0.0;
        } else {
          const series1 = dataSeries[variables[i]].map(point => point.value as number);
          const series2 = dataSeries[variables[j]].map(point => point.value as number);
          
          const { correlation, pValue } = this.calculateCorrelation(series1, series2, method);
          matrix[i][j] = correlation;
          pValues[i][j] = pValue;
          
          if (i < j && Math.abs(correlation) > 0.3) { // Only add upper triangle pairs
            significantPairs.push({
              variable1: variables[i],
              variable2: variables[j],
              correlation,
              pValue,
              significance: this.classifySignificance(pValue)
            });
          }
        }
      }
    }
    
    return {
      variables,
      matrix,
      pValues,
      significantPairs: significantPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
    };
  }

  private static calculateCorrelation(
    x: number[], 
    y: number[], 
    method: 'pearson' | 'spearman' | 'kendall'
  ): { correlation: number; pValue: number } {
    const n = Math.min(x.length, y.length);
    if (n < 3) return { correlation: 0, pValue: 1 };
    
    switch (method) {
      case 'pearson':
        return this.pearsonCorrelation(x.slice(0, n), y.slice(0, n));
      
      case 'spearman':
        return this.spearmanCorrelation(x.slice(0, n), y.slice(0, n));
      
      case 'kendall':
        return this.kendallCorrelation(x.slice(0, n), y.slice(0, n));
      
      default:
        return this.pearsonCorrelation(x.slice(0, n), y.slice(0, n));
    }
  }

  private static pearsonCorrelation(x: number[], y: number[]): { correlation: number; pValue: number } {
    const n = x.length;
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = y.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumXX = x.reduce((sum, val) => sum + val * val, 0);
    const sumYY = y.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    const correlation = denominator === 0 ? 0 : numerator / denominator;
    
    // Calculate t-statistic for p-value
    const t = correlation * Math.sqrt((n - 2) / (1 - correlation * correlation));
    const pValue = this.tTestPValue(t, n - 2);
    
    return { correlation, pValue };
  }

  private static spearmanCorrelation(x: number[], y: number[]): { correlation: number; pValue: number } {
    // Convert to ranks
    const xRanks = this.getRanks(x);
    const yRanks = this.getRanks(y);
    
    return this.pearsonCorrelation(xRanks, yRanks);
  }

  private static kendallCorrelation(x: number[], y: number[]): { correlation: number; pValue: number } {
    const n = x.length;
    let concordant = 0;
    let discordant = 0;
    
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const xDiff = x[j] - x[i];
        const yDiff = y[j] - y[i];
        
        if ((xDiff > 0 && yDiff > 0) || (xDiff < 0 && yDiff < 0)) {
          concordant++;
        } else if ((xDiff > 0 && yDiff < 0) || (xDiff < 0 && yDiff > 0)) {
          discordant++;
        }
      }
    }
    
    const totalPairs = n * (n - 1) / 2;
    const correlation = totalPairs > 0 ? (concordant - discordant) / totalPairs : 0;
    
    // Simplified p-value calculation for Kendall's tau
    const variance = (4 * n + 10) / (9 * n * (n - 1));
    const z = correlation / Math.sqrt(variance);
    const pValue = 2 * (1 - this.standardNormalCDF(Math.abs(z)));
    
    return { correlation, pValue };
  }

  private static getRanks(values: number[]): number[] {
    const indexed = values.map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => a.val - b.val);
    
    const ranks = new Array(values.length);
    
    for (let i = 0; i < indexed.length; i++) {
      ranks[indexed[i].idx] = i + 1;
    }
    
    return ranks;
  }

  private static tTestPValue(t: number, df: number): number {
    // Simplified t-test p-value calculation
    // In production, use a proper statistical library
    const x = df / (df + t * t);
    return this.betaIncomplete(0.5 * df, 0.5, x);
  }

  private static betaIncomplete(a: number, b: number, x: number): number {
    // Simplified incomplete beta function
    // This is a very basic approximation
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    
    // Use approximation for simplicity
    return Math.pow(x, a) * Math.pow(1 - x, b) / (a + b);
  }

  private static standardNormalCDF(z: number): number {
    // Approximation of standard normal CDF
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  private static erf(x: number): number {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }

  private static classifySignificance(pValue: number): 'low' | 'medium' | 'high' {
    if (pValue < 0.01) return 'high';
    if (pValue < 0.05) return 'medium';
    return 'low';
  }

  /**
   * Smooth time series data using various algorithms
   */
  static smoothTimeSeries(
    data: TimeSeriesPoint[],
    algorithm: 'moving_average' | 'exponential_smoothing' | 'savitzky_golay' | 'lowess',
    parameters: Record<string, any> = {}
  ): TimeSeriesPoint[] {
    switch (algorithm) {
      case 'moving_average':
        return this.movingAverageSmooth(data, parameters.window || 5);
      
      case 'exponential_smoothing':
        return this.exponentialSmooth(data, parameters.alpha || 0.3);
      
      case 'savitzky_golay':
        return this.savitzkyGolaySmooth(data, parameters.window || 5, parameters.degree || 2);
      
      case 'lowess':
        return this.lowessSmooth(data, parameters.bandwidth || 0.3);
      
      default:
        return data;
    }
  }

  private static movingAverageSmooth(data: TimeSeriesPoint[], window: number): TimeSeriesPoint[] {
    if (data.length === 0 || window <= 1) return data;
    
    const smoothed: TimeSeriesPoint[] = [];
    const halfWindow = Math.floor(window / 2);
    
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(data.length, i + halfWindow + 1);
      const windowData = data.slice(start, end);
      
      const avgValue = windowData.reduce((sum, point) => sum + (point.value as number), 0) / windowData.length;
      
      smoothed.push({
        timestamp: data[i].timestamp,
        value: avgValue,
        metadata: { ...data[i].metadata, smoothed: true }
      });
    }
    
    return smoothed;
  }

  private static exponentialSmooth(data: TimeSeriesPoint[], alpha: number): TimeSeriesPoint[] {
    if (data.length === 0) return data;
    
    const smoothed: TimeSeriesPoint[] = [];
    let smoothedValue = data[0].value as number;
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        smoothedValue = data[i].value as number;
      } else {
        smoothedValue = alpha * (data[i].value as number) + (1 - alpha) * smoothedValue;
      }
      
      smoothed.push({
        timestamp: data[i].timestamp,
        value: smoothedValue,
        metadata: { ...data[i].metadata, smoothed: true }
      });
    }
    
    return smoothed;
  }

  private static savitzkyGolaySmooth(data: TimeSeriesPoint[], window: number, degree: number): TimeSeriesPoint[] {
    // Simplified Savitzky-Golay filter
    if (data.length === 0 || window <= degree) return data;
    
    const smoothed: TimeSeriesPoint[] = [];
    const halfWindow = Math.floor(window / 2);
    
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(data.length, i + halfWindow + 1);
      const windowData = data.slice(start, end);
      
      // For simplicity, use weighted average (not true Savitzky-Golay)
      const weights = this.generateSavitzkyGolayWeights(windowData.length, degree);
      let smoothedValue = 0;
      
      for (let j = 0; j < windowData.length; j++) {
        smoothedValue += (windowData[j].value as number) * weights[j];
      }
      
      smoothed.push({
        timestamp: data[i].timestamp,
        value: smoothedValue,
        metadata: { ...data[i].metadata, smoothed: true }
      });
    }
    
    return smoothed;
  }

  private static generateSavitzkyGolayWeights(windowSize: number, degree: number): number[] {
    // Simplified weight generation - in practice, use proper Savitzky-Golay coefficients
    const weights = new Array(windowSize);
    const center = Math.floor(windowSize / 2);
    
    for (let i = 0; i < windowSize; i++) {
      const distance = Math.abs(i - center);
      weights[i] = Math.exp(-distance * distance / (2 * Math.pow(windowSize / 4, 2)));
    }
    
    // Normalize weights
    const sum = weights.reduce((s, w) => s + w, 0);
    return weights.map(w => w / sum);
  }

  private static lowessSmooth(data: TimeSeriesPoint[], bandwidth: number): TimeSeriesPoint[] {
    // Simplified LOWESS (LOcally WEighted Scatterplot Smoothing)
    if (data.length === 0) return data;
    
    const smoothed: TimeSeriesPoint[] = [];
    const windowSize = Math.max(3, Math.floor(data.length * bandwidth));
    
    for (let i = 0; i < data.length; i++) {
      // Find nearest neighbors
      const distances = data.map((point, idx) => ({
        idx,
        distance: Math.abs(point.timestamp - data[i].timestamp)
      }));
      
      distances.sort((a, b) => a.distance - b.distance);
      const neighbors = distances.slice(0, windowSize);
      
      // Calculate weighted regression
      const maxDistance = neighbors[neighbors.length - 1].distance;
      let weightedSum = 0;
      let weightSum = 0;
      
      for (const neighbor of neighbors) {
        const weight = maxDistance > 0 ? Math.pow(1 - Math.pow(neighbor.distance / maxDistance, 3), 3) : 1;
        weightedSum += (data[neighbor.idx].value as number) * weight;
        weightSum += weight;
      }
      
      const smoothedValue = weightSum > 0 ? weightedSum / weightSum : data[i].value as number;
      
      smoothed.push({
        timestamp: data[i].timestamp,
        value: smoothedValue,
        metadata: { ...data[i].metadata, smoothed: true }
      });
    }
    
    return smoothed;
  }
}

// Export all classes and utilities
export {
  TimeSeriesAggregator,
  PatternDetector,
  AnomalyDetector,
  TrendAnalyzer,
  DataCompressor,
  DataProcessingUtils
};