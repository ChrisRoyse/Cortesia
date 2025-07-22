/**
 * HealthMetrics - Health metrics calculation and trending analysis
 * Provides statistical analysis and trending capabilities for health data
 */

import { ComponentHealth, HealthSnapshot, HealthDimension, HealthStatus } from './SystemHealthEngine';

export interface MetricTrendPoint {
  timestamp: Date;
  value: number;
  smoothed?: number;
  prediction?: number;
}

export interface TrendAnalysis {
  metric: string;
  trend: 'improving' | 'declining' | 'stable' | 'volatile';
  slope: number;
  correlation: number;
  confidence: number;
  seasonality?: SeasonalPattern;
  changePoints?: ChangePoint[];
}

export interface SeasonalPattern {
  period: number; // hours
  amplitude: number;
  phase: number;
  detected: boolean;
  confidence: number;
}

export interface ChangePoint {
  timestamp: Date;
  magnitude: number;
  direction: 'increase' | 'decrease';
  significance: number;
  cause?: string;
}

export interface HealthMetricSummary {
  metric: string;
  current: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
  percentiles: { p25: number; p50: number; p75: number; p90: number; p95: number; p99: number };
  trend: TrendAnalysis;
  sla: SLAMetrics;
}

export interface SLAMetrics {
  target: number;
  current: number;
  compliance: number; // 0-1
  breachCount: number;
  mtbf: number; // Mean time between failures (hours)
  mttr: number; // Mean time to recovery (hours)
  availability: number; // 0-1
}

export interface HealthCorrelation {
  metric1: string;
  metric2: string;
  correlation: number;
  pValue: number;
  significant: boolean;
  lagHours?: number;
}

export interface AnomalyDetection {
  timestamp: Date;
  metric: string;
  value: number;
  expectedValue: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  type: 'spike' | 'drop' | 'drift' | 'outlier';
  description: string;
}

export interface PerformanceBenchmark {
  metric: string;
  period: 'daily' | 'weekly' | 'monthly';
  baseline: number;
  current: number;
  change: number;
  changePercent: number;
  significance: 'improved' | 'degraded' | 'stable';
}

export class HealthMetrics {
  private readonly movingAverageWindow = 12; // 12-point moving average
  private readonly anomalyThreshold = 2.5; // Standard deviations for anomaly detection
  private readonly minDataPoints = 10;

  /**
   * Calculate comprehensive health metrics summary
   */
  calculateMetricsSummary(
    componentHealth: ComponentHealth, 
    dimensionName?: string
  ): HealthMetricSummary[] {
    const summaries: HealthMetricSummary[] = [];

    if (dimensionName) {
      const dimension = componentHealth.dimensions.find(d => d.name === dimensionName);
      if (dimension) {
        summaries.push(this.calculateDimensionSummary(componentHealth, dimension));
      }
    } else {
      // Calculate summary for overall health
      summaries.push(this.calculateOverallHealthSummary(componentHealth));
      
      // Calculate summary for each dimension
      for (const dimension of componentHealth.dimensions) {
        summaries.push(this.calculateDimensionSummary(componentHealth, dimension));
      }
    }

    return summaries;
  }

  /**
   * Analyze health trends over time
   */
  analyzeTrends(data: MetricTrendPoint[], metric: string): TrendAnalysis {
    if (data.length < this.minDataPoints) {
      return {
        metric,
        trend: 'stable',
        slope: 0,
        correlation: 0,
        confidence: 0
      };
    }

    // Sort data by timestamp
    const sortedData = data.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    
    // Calculate trend using linear regression
    const { slope, correlation } = this.calculateLinearRegression(sortedData);
    
    // Classify trend
    const trend = this.classifyTrend(slope, this.calculateVolatility(sortedData.map(d => d.value)));
    
    // Detect seasonality
    const seasonality = this.detectSeasonality(sortedData);
    
    // Detect change points
    const changePoints = this.detectChangePoints(sortedData);
    
    // Calculate confidence based on data quality and consistency
    const confidence = this.calculateTrendConfidence(sortedData, slope, correlation);

    return {
      metric,
      trend,
      slope,
      correlation,
      confidence,
      seasonality,
      changePoints
    };
  }

  /**
   * Apply smoothing to noisy health data
   */
  applySmoothingFilter(
    data: MetricTrendPoint[], 
    method: 'exponential' | 'moving_average' | 'kalman' = 'exponential',
    alpha: number = 0.3
  ): MetricTrendPoint[] {
    if (data.length === 0) return [];

    const smoothed = [...data];

    switch (method) {
      case 'exponential':
        this.applyExponentialSmoothing(smoothed, alpha);
        break;
      case 'moving_average':
        this.applyMovingAverage(smoothed);
        break;
      case 'kalman':
        this.applyKalmanFilter(smoothed);
        break;
    }

    return smoothed;
  }

  /**
   * Detect anomalies in health metrics
   */
  detectAnomalies(data: MetricTrendPoint[], metric: string): AnomalyDetection[] {
    if (data.length < this.minDataPoints) return [];

    const anomalies: AnomalyDetection[] = [];
    const values = data.map(d => d.value);
    const mean = this.calculateMean(values);
    const stdDev = this.calculateStandardDeviation(values, mean);

    // Z-score based anomaly detection
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      const zScore = Math.abs((point.value - mean) / stdDev);
      
      if (zScore > this.anomalyThreshold) {
        const severity = zScore > 4 ? 'high' : zScore > 3 ? 'medium' : 'low';
        const type = this.classifyAnomalyType(data, i, mean, stdDev);
        
        anomalies.push({
          timestamp: point.timestamp,
          metric,
          value: point.value,
          expectedValue: mean,
          severity,
          confidence: Math.min(0.95, (zScore - this.anomalyThreshold) / 2),
          type,
          description: this.generateAnomalyDescription(type, point.value, mean, zScore)
        });
      }
    }

    return anomalies;
  }

  /**
   * Calculate correlations between different health metrics
   */
  calculateCorrelations(
    metrics: Map<string, MetricTrendPoint[]>,
    maxLag: number = 3
  ): HealthCorrelation[] {
    const correlations: HealthCorrelation[] = [];
    const metricNames = Array.from(metrics.keys());

    for (let i = 0; i < metricNames.length; i++) {
      for (let j = i + 1; j < metricNames.length; j++) {
        const metric1 = metricNames[i];
        const metric2 = metricNames[j];
        const data1 = metrics.get(metric1) || [];
        const data2 = metrics.get(metric2) || [];

        // Calculate correlation with different lags
        let bestCorrelation = 0;
        let bestLag = 0;
        
        for (let lag = 0; lag <= maxLag; lag++) {
          const correlation = this.calculateCrossCorrelation(data1, data2, lag);
          if (Math.abs(correlation) > Math.abs(bestCorrelation)) {
            bestCorrelation = correlation;
            bestLag = lag;
          }
        }

        // Calculate statistical significance
        const n = Math.min(data1.length, data2.length);
        const pValue = this.calculatePValue(bestCorrelation, n);
        const significant = pValue < 0.05;

        correlations.push({
          metric1,
          metric2,
          correlation: bestCorrelation,
          pValue,
          significant,
          lagHours: bestLag > 0 ? bestLag : undefined
        });
      }
    }

    return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  }

  /**
   * Generate performance benchmarks
   */
  generateBenchmarks(
    currentData: MetricTrendPoint[],
    historicalData: MetricTrendPoint[],
    metric: string
  ): PerformanceBenchmark[] {
    const benchmarks: PerformanceBenchmark[] = [];
    const periods: Array<{ period: 'daily' | 'weekly' | 'monthly', hours: number }> = [
      { period: 'daily', hours: 24 },
      { period: 'weekly', hours: 168 },
      { period: 'monthly', hours: 720 }
    ];

    for (const { period, hours } of periods) {
      const currentPeriodData = this.getDataForPeriod(currentData, hours);
      const baselinePeriodData = this.getDataForPeriod(historicalData, hours);

      if (currentPeriodData.length > 0 && baselinePeriodData.length > 0) {
        const current = this.calculateMean(currentPeriodData.map(d => d.value));
        const baseline = this.calculateMean(baselinePeriodData.map(d => d.value));
        const change = current - baseline;
        const changePercent = baseline !== 0 ? (change / baseline) * 100 : 0;

        // Determine significance based on statistical test
        const significance = this.determineSignificance(
          currentPeriodData.map(d => d.value),
          baselinePeriodData.map(d => d.value)
        );

        benchmarks.push({
          metric,
          period,
          baseline,
          current,
          change,
          changePercent,
          significance
        });
      }
    }

    return benchmarks;
  }

  /**
   * Calculate SLA metrics
   */
  calculateSLAMetrics(
    data: MetricTrendPoint[],
    target: number,
    isHigherBetter: boolean = true
  ): SLAMetrics {
    if (data.length === 0) {
      return {
        target,
        current: 0,
        compliance: 0,
        breachCount: 0,
        mtbf: 0,
        mttr: 0,
        availability: 0
      };
    }

    const current = data[data.length - 1].value;
    const values = data.map(d => d.value);
    
    // Calculate compliance
    const compliantValues = values.filter(v => 
      isHigherBetter ? v >= target : v <= target
    );
    const compliance = compliantValues.length / values.length;

    // Calculate breach periods
    const breaches = this.identifyBreachPeriods(data, target, isHigherBetter);
    const breachCount = breaches.length;

    // Calculate MTBF (Mean Time Between Failures)
    const mtbf = breachCount > 1 ? 
      (data[data.length - 1].timestamp.getTime() - data[0].timestamp.getTime()) / 
      (1000 * 60 * 60 * (breachCount - 1)) : 0;

    // Calculate MTTR (Mean Time To Recovery)
    const mttr = breaches.length > 0 ?
      breaches.reduce((sum, breach) => sum + breach.duration, 0) / breaches.length : 0;

    // Calculate availability (time in compliant state)
    const totalDuration = data.length > 1 ?
      (data[data.length - 1].timestamp.getTime() - data[0].timestamp.getTime()) / (1000 * 60 * 60) : 1;
    const breachDuration = breaches.reduce((sum, breach) => sum + breach.duration, 0);
    const availability = Math.max(0, (totalDuration - breachDuration) / totalDuration);

    return {
      target,
      current,
      compliance,
      breachCount,
      mtbf,
      mttr,
      availability
    };
  }

  private calculateOverallHealthSummary(componentHealth: ComponentHealth): HealthMetricSummary {
    const history = componentHealth.healthHistory;
    const values = history.map(h => h.overallHealth);
    
    return {
      metric: 'overall_health',
      current: componentHealth.overallHealth,
      min: values.length > 0 ? Math.min(...values) : componentHealth.overallHealth,
      max: values.length > 0 ? Math.max(...values) : componentHealth.overallHealth,
      mean: this.calculateMean(values),
      median: this.calculateMedian(values),
      stdDev: this.calculateStandardDeviation(values),
      percentiles: this.calculatePercentiles(values),
      trend: this.analyzeTrends(
        history.map(h => ({ timestamp: h.timestamp, value: h.overallHealth })),
        'overall_health'
      ),
      sla: this.calculateSLAMetrics(
        history.map(h => ({ timestamp: h.timestamp, value: h.overallHealth })),
        0.95, // Target 95% health
        true
      )
    };
  }

  private calculateDimensionSummary(
    componentHealth: ComponentHealth, 
    dimension: HealthDimension
  ): HealthMetricSummary {
    const history = componentHealth.healthHistory;
    const values = history
      .map(h => h.dimensions[dimension.name])
      .filter(v => v !== undefined);
    
    return {
      metric: dimension.name,
      current: dimension.score,
      min: values.length > 0 ? Math.min(...values) : dimension.score,
      max: values.length > 0 ? Math.max(...values) : dimension.score,
      mean: this.calculateMean(values),
      median: this.calculateMedian(values),
      stdDev: this.calculateStandardDeviation(values),
      percentiles: this.calculatePercentiles(values),
      trend: this.analyzeTrends(
        history
          .filter(h => h.dimensions[dimension.name] !== undefined)
          .map(h => ({ 
            timestamp: h.timestamp, 
            value: h.dimensions[dimension.name] 
          })),
        dimension.name
      ),
      sla: this.calculateSLAMetrics(
        history
          .filter(h => h.dimensions[dimension.name] !== undefined)
          .map(h => ({ 
            timestamp: h.timestamp, 
            value: h.dimensions[dimension.name] 
          })),
        0.8, // Target 80% for individual dimensions
        true
      )
    };
  }

  private calculateLinearRegression(data: MetricTrendPoint[]): { slope: number; correlation: number } {
    const n = data.length;
    if (n < 2) return { slope: 0, correlation: 0 };

    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    
    for (let i = 0; i < n; i++) {
      const x = i;
      const y = data[i].value;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
      sumY2 += y * y;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const correlation = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return { 
      slope: isNaN(slope) ? 0 : slope, 
      correlation: isNaN(correlation) ? 0 : correlation 
    };
  }

  private classifyTrend(slope: number, volatility: number): TrendAnalysis['trend'] {
    const absSlope = Math.abs(slope);
    
    if (volatility > 0.1) return 'volatile';
    if (absSlope < 0.001) return 'stable';
    return slope > 0 ? 'improving' : 'declining';
  }

  private detectSeasonality(data: MetricTrendPoint[]): SeasonalPattern {
    // Simple autocorrelation-based seasonality detection
    const periods = [24, 168, 720]; // Daily, weekly, monthly in hours
    let bestPeriod = 24;
    let bestAmplitude = 0;
    let bestCorrelation = 0;

    for (const period of periods) {
      if (data.length >= period * 2) {
        const correlation = this.calculateAutocorrelation(data, period);
        if (Math.abs(correlation) > Math.abs(bestCorrelation)) {
          bestCorrelation = correlation;
          bestPeriod = period;
          bestAmplitude = this.calculateSeasonalAmplitude(data, period);
        }
      }
    }

    return {
      period: bestPeriod,
      amplitude: bestAmplitude,
      phase: 0, // Simplified - would need FFT for accurate phase detection
      detected: Math.abs(bestCorrelation) > 0.3,
      confidence: Math.abs(bestCorrelation)
    };
  }

  private detectChangePoints(data: MetricTrendPoint[]): ChangePoint[] {
    const changePoints: ChangePoint[] = [];
    const windowSize = Math.min(10, Math.floor(data.length / 4));
    
    for (let i = windowSize; i < data.length - windowSize; i++) {
      const beforeWindow = data.slice(i - windowSize, i);
      const afterWindow = data.slice(i, i + windowSize);
      
      const beforeMean = this.calculateMean(beforeWindow.map(d => d.value));
      const afterMean = this.calculateMean(afterWindow.map(d => d.value));
      
      const magnitude = Math.abs(afterMean - beforeMean);
      const pooledStdDev = this.calculatePooledStandardDeviation(
        beforeWindow.map(d => d.value),
        afterWindow.map(d => d.value)
      );
      
      const significance = pooledStdDev > 0 ? magnitude / pooledStdDev : 0;
      
      if (significance > 2.0) { // Significant change
        changePoints.push({
          timestamp: data[i].timestamp,
          magnitude,
          direction: afterMean > beforeMean ? 'increase' : 'decrease',
          significance
        });
      }
    }

    return changePoints;
  }

  private applyExponentialSmoothing(data: MetricTrendPoint[], alpha: number): void {
    if (data.length === 0) return;
    
    data[0].smoothed = data[0].value;
    
    for (let i = 1; i < data.length; i++) {
      data[i].smoothed = alpha * data[i].value + (1 - alpha) * (data[i - 1].smoothed || data[i - 1].value);
    }
  }

  private applyMovingAverage(data: MetricTrendPoint[]): void {
    const window = Math.min(this.movingAverageWindow, data.length);
    
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - Math.floor(window / 2));
      const end = Math.min(data.length, start + window);
      const values = data.slice(start, end).map(d => d.value);
      data[i].smoothed = this.calculateMean(values);
    }
  }

  private applyKalmanFilter(data: MetricTrendPoint[]): void {
    // Simplified Kalman filter implementation
    if (data.length === 0) return;
    
    let estimate = data[0].value;
    let errorEstimate = 1.0;
    const processNoise = 0.01;
    const measurementNoise = 0.1;
    
    for (let i = 0; i < data.length; i++) {
      // Prediction step
      const predictedEstimate = estimate;
      const predictedError = errorEstimate + processNoise;
      
      // Update step
      const kalmanGain = predictedError / (predictedError + measurementNoise);
      estimate = predictedEstimate + kalmanGain * (data[i].value - predictedEstimate);
      errorEstimate = (1 - kalmanGain) * predictedError;
      
      data[i].smoothed = estimate;
    }
  }

  private classifyAnomalyType(
    data: MetricTrendPoint[], 
    index: number, 
    mean: number, 
    stdDev: number
  ): AnomalyDetection['type'] {
    const point = data[index];
    const value = point.value;
    
    // Check for spike or drop
    if (Math.abs(value - mean) > 3 * stdDev) {
      return value > mean ? 'spike' : 'drop';
    }
    
    // Check for drift (persistent change from baseline)
    const windowSize = 5;
    const start = Math.max(0, index - windowSize);
    const end = Math.min(data.length, index + windowSize);
    const window = data.slice(start, end);
    const windowMean = this.calculateMean(window.map(d => d.value));
    
    if (Math.abs(windowMean - mean) > 1.5 * stdDev) {
      return 'drift';
    }
    
    return 'outlier';
  }

  private generateAnomalyDescription(
    type: AnomalyDetection['type'],
    value: number,
    expected: number,
    zScore: number
  ): string {
    const deviation = ((value - expected) / expected * 100).toFixed(1);
    
    switch (type) {
      case 'spike':
        return `Sudden spike: ${deviation}% above expected (${zScore.toFixed(1)}σ)`;
      case 'drop':
        return `Sudden drop: ${Math.abs(parseFloat(deviation))}% below expected (${zScore.toFixed(1)}σ)`;
      case 'drift':
        return `Persistent drift: ${deviation}% from baseline (${zScore.toFixed(1)}σ)`;
      default:
        return `Outlier detected: ${deviation}% deviation (${zScore.toFixed(1)}σ)`;
    }
  }

  private calculateCrossCorrelation(
    data1: MetricTrendPoint[], 
    data2: MetricTrendPoint[], 
    lag: number
  ): number {
    const n = Math.min(data1.length, data2.length - lag);
    if (n < 2) return 0;
    
    const values1 = data1.slice(0, n).map(d => d.value);
    const values2 = data2.slice(lag, lag + n).map(d => d.value);
    
    const mean1 = this.calculateMean(values1);
    const mean2 = this.calculateMean(values2);
    
    let numerator = 0;
    let sum1Sq = 0;
    let sum2Sq = 0;
    
    for (let i = 0; i < n; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;
      numerator += diff1 * diff2;
      sum1Sq += diff1 * diff1;
      sum2Sq += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(sum1Sq * sum2Sq);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private calculateAutocorrelation(data: MetricTrendPoint[], lag: number): number {
    return this.calculateCrossCorrelation(data, data, lag);
  }

  private calculateSeasonalAmplitude(data: MetricTrendPoint[], period: number): number {
    const seasonalMeans: number[] = [];
    
    for (let phase = 0; phase < period; phase++) {
      const phaseValues: number[] = [];
      for (let i = phase; i < data.length; i += period) {
        phaseValues.push(data[i].value);
      }
      if (phaseValues.length > 0) {
        seasonalMeans.push(this.calculateMean(phaseValues));
      }
    }
    
    if (seasonalMeans.length === 0) return 0;
    
    const overallMean = this.calculateMean(seasonalMeans);
    const maxDeviation = Math.max(...seasonalMeans.map(m => Math.abs(m - overallMean)));
    
    return maxDeviation;
  }

  // Utility methods
  private calculateMean(values: number[]): number {
    return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
  }

  private calculateMedian(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  private calculateStandardDeviation(values: number[], mean?: number): number {
    if (values.length <= 1) return 0;
    const avg = mean ?? this.calculateMean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / (values.length - 1);
    return Math.sqrt(variance);
  }

  private calculatePercentiles(values: number[]): HealthMetricSummary['percentiles'] {
    if (values.length === 0) {
      return { p25: 0, p50: 0, p75: 0, p90: 0, p95: 0, p99: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const getPercentile = (p: number) => {
      const index = (p / 100) * (sorted.length - 1);
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
    };
    
    return {
      p25: getPercentile(25),
      p50: getPercentile(50),
      p75: getPercentile(75),
      p90: getPercentile(90),
      p95: getPercentile(95),
      p99: getPercentile(99)
    };
  }

  private calculateVolatility(values: number[]): number {
    if (values.length <= 1) return 0;
    
    const returns = [];
    for (let i = 1; i < values.length; i++) {
      if (values[i - 1] !== 0) {
        returns.push((values[i] - values[i - 1]) / values[i - 1]);
      }
    }
    
    return this.calculateStandardDeviation(returns);
  }

  private calculateTrendConfidence(
    data: MetricTrendPoint[], 
    slope: number, 
    correlation: number
  ): number {
    const dataQuality = Math.min(1, data.length / 50); // More data = higher confidence
    const trendStrength = Math.abs(correlation);
    const consistency = 1 - this.calculateVolatility(data.map(d => d.value));
    
    return Math.max(0.1, Math.min(0.95, (dataQuality + trendStrength + consistency) / 3));
  }

  private calculatePValue(correlation: number, n: number): number {
    if (n <= 2) return 1.0;
    
    const t = Math.abs(correlation) * Math.sqrt((n - 2) / (1 - correlation * correlation));
    
    // Simplified p-value calculation (would use t-distribution in practice)
    return Math.max(0.001, Math.min(0.999, 2 * (1 - this.normalCDF(t))));
  }

  private normalCDF(x: number): number {
    // Approximation of the normal cumulative distribution function
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Approximation of the error function
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

  private getDataForPeriod(data: MetricTrendPoint[], hours: number): MetricTrendPoint[] {
    if (data.length === 0) return [];
    
    const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);
    return data.filter(d => d.timestamp >= cutoffTime);
  }

  private determineSignificance(
    current: number[], 
    baseline: number[]
  ): PerformanceBenchmark['significance'] {
    if (current.length < 2 || baseline.length < 2) return 'stable';
    
    // Simple t-test approximation
    const currentMean = this.calculateMean(current);
    const baselineMean = this.calculateMean(baseline);
    const difference = currentMean - baselineMean;
    
    const currentStdDev = this.calculateStandardDeviation(current);
    const baselineStdDev = this.calculateStandardDeviation(baseline);
    const pooledStdDev = this.calculatePooledStandardDeviation(current, baseline);
    
    if (pooledStdDev === 0) return 'stable';
    
    const tStat = Math.abs(difference) / pooledStdDev;
    
    if (tStat > 2.0) { // Significant at ~95% confidence
      return difference > 0 ? 'improved' : 'degraded';
    }
    
    return 'stable';
  }

  private calculatePooledStandardDeviation(sample1: number[], sample2: number[]): number {
    const n1 = sample1.length;
    const n2 = sample2.length;
    
    if (n1 <= 1 && n2 <= 1) return 0;
    
    const var1 = Math.pow(this.calculateStandardDeviation(sample1), 2);
    const var2 = Math.pow(this.calculateStandardDeviation(sample2), 2);
    
    const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    const standardError = Math.sqrt(pooledVar * (1/n1 + 1/n2));
    
    return standardError;
  }

  private identifyBreachPeriods(
    data: MetricTrendPoint[], 
    target: number, 
    isHigherBetter: boolean
  ): Array<{ start: Date; end: Date; duration: number }> {
    const breaches: Array<{ start: Date; end: Date; duration: number }> = [];
    let inBreach = false;
    let breachStart: Date | null = null;
    
    for (const point of data) {
      const isBreach = isHigherBetter ? point.value < target : point.value > target;
      
      if (isBreach && !inBreach) {
        inBreach = true;
        breachStart = point.timestamp;
      } else if (!isBreach && inBreach) {
        inBreach = false;
        if (breachStart) {
          const duration = (point.timestamp.getTime() - breachStart.getTime()) / (1000 * 60 * 60);
          breaches.push({
            start: breachStart,
            end: point.timestamp,
            duration
          });
        }
        breachStart = null;
      }
    }
    
    // Handle ongoing breach
    if (inBreach && breachStart) {
      const duration = (Date.now() - breachStart.getTime()) / (1000 * 60 * 60);
      breaches.push({
        start: breachStart,
        end: new Date(),
        duration
      });
    }
    
    return breaches;
  }
}

export { HealthMetrics };