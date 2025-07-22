/**
 * Memory Analytics for Phase 4
 * Track memory performance metrics and provide optimization insights
 */

import * as THREE from 'three';
import { ShaderLibrary } from '../core/ShaderLibrary';
import { StorageMetrics, EfficiencyTrend } from './StorageEfficiency';
import { MemoryOperation, MemoryStats } from './MemoryOperationVisualizer';
import { SDRPattern, SDRComparisonResult } from './SDRVisualizer';

export interface MemoryPerformanceMetrics {
  timestamp: number;
  allocatedMemory: number;
  freeMemory: number;
  fragmentationLevel: number;
  allocationRate: number;
  deallocationRate: number;
  gcPressure: number;
  cacheHitRate: number;
  ioThroughput: number;
  compressionRatio: number;
  sdrSparsity: number;
  queryLatency: number;
}

export interface MemoryInsight {
  id: string;
  type: 'warning' | 'optimization' | 'anomaly' | 'trend';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  recommendation: string;
  metrics: Record<string, number>;
  timestamp: number;
}

export interface MemoryPattern {
  patternId: string;
  patternType: 'allocation' | 'access' | 'fragmentation' | 'compression';
  frequency: number;
  impact: number;
  predictedTrend: 'improving' | 'stable' | 'degrading';
  timeWindow: number;
  confidence: number;
}

export interface AnalyticsConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  historySize: number;
  analysisWindow: number;
  alertThresholds: {
    fragmentation: number;
    memoryUsage: number;
    cacheHitRate: number;
    compressionRatio: number;
    queryLatency: number;
  };
}

export class MemoryAnalytics {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private config: AnalyticsConfig;
  private shaderLibrary: ShaderLibrary;

  // Analytics data
  private performanceHistory: MemoryPerformanceMetrics[] = [];
  private insights: MemoryInsight[] = [];
  private patterns: Map<string, MemoryPattern> = new Map();
  private alerts: MemoryInsight[] = [];
  
  // Visualization components
  private metricsChart: THREE.Group;
  private insightPanels: THREE.Group;
  private patternVisualization: THREE.Group;
  private trendPrediction: THREE.Group;
  
  // Chart elements
  private chartGeometry: THREE.PlaneGeometry;
  private chartMaterial: THREE.ShaderMaterial;
  private chartLines: Map<string, THREE.Line> = new Map();
  
  // Insight visualization
  private insightGeometry: THREE.PlaneGeometry;
  private insightMaterials: Map<string, THREE.MeshBasicMaterial> = new Map();
  private insightMeshes: Map<string, THREE.Mesh> = new Map();
  
  // Animation and interaction
  private animationClock: THREE.Clock;
  private lastAnalysisTime: number = 0;
  private analysisInterval: number = 1000; // 1 second
  
  // Statistical analysis
  private movingAverages: Map<string, number[]> = new Map();
  private correlationMatrix: Map<string, Map<string, number>> = new Map();
  private anomalyThresholds: Map<string, { mean: number; std: number }> = new Map();

  constructor(config: AnalyticsConfig) {
    this.config = { ...config };
    this.shaderLibrary = ShaderLibrary.getInstance();
    this.animationClock = new THREE.Clock();

    this.initializeRenderer();
    this.initializeScene();
    this.initializeCamera();
    this.initializeChartVisualization();
    this.initializeInsightVisualization();
    this.initializePatternVisualization();
    
    // Initialize statistical tracking
    this.initializeStatisticalAnalysis();
  }

  private initializeRenderer(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.config.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    
    this.renderer.setSize(this.config.width, this.config.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0c0c0c, 1.0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  private initializeScene(): void {
    this.scene = new THREE.Scene();
    
    // Add subtle lighting
    const ambientLight = new THREE.AmbientLight(0x333333, 0.4);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0x8888ff, 0.6);
    directionalLight.position.set(10, 10, 5);
    this.scene.add(directionalLight);
  }

  private initializeCamera(): void {
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.config.width / this.config.height,
      0.1,
      1000
    );
    
    this.camera.position.set(0, 0, 50);
    this.camera.lookAt(0, 0, 0);
  }

  private initializeChartVisualization(): void {
    this.metricsChart = new THREE.Group();
    this.scene.add(this.metricsChart);
    
    // Create chart background
    this.chartGeometry = new THREE.PlaneGeometry(30, 20);
    this.chartMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        gridColor: { value: new THREE.Color(0x333333) },
        backgroundColor: { value: new THREE.Color(0x1a1a1a) }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 gridColor;
        uniform vec3 backgroundColor;
        varying vec2 vUv;
        
        void main() {
          // Create grid pattern
          vec2 grid = abs(fract(vUv * 20.0) - 0.5);
          float line = smoothstep(0.0, 0.1, min(grid.x, grid.y));
          
          vec3 color = mix(gridColor, backgroundColor, line);
          gl_FragColor = vec4(color, 0.8);
        }
      `,
      transparent: true
    });
    
    const chartBackground = new THREE.Mesh(this.chartGeometry, this.chartMaterial);
    chartBackground.position.set(0, 5, -1);
    this.metricsChart.add(chartBackground);
    
    // Initialize chart lines for different metrics
    this.initializeChartLines();
  }

  private initializeChartLines(): void {
    const metrics = [
      'memoryUsage', 'fragmentation', 'cacheHitRate', 
      'ioThroughput', 'compressionRatio', 'queryLatency'
    ];
    
    const colors = [
      0xff4444, 0x44ff44, 0x4444ff, 
      0xffff44, 0xff44ff, 0x44ffff
    ];
    
    metrics.forEach((metric, index) => {
      const lineGeometry = new THREE.BufferGeometry();
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: colors[index],
        linewidth: 2
      });
      
      const line = new THREE.Line(lineGeometry, lineMaterial);
      this.metricsChart.add(line);
      this.chartLines.set(metric, line);
    });
  }

  private initializeInsightVisualization(): void {
    this.insightPanels = new THREE.Group();
    this.scene.add(this.insightPanels);
    
    this.insightGeometry = new THREE.PlaneGeometry(8, 2);
    
    // Create materials for different insight types
    const insightTypes = ['warning', 'optimization', 'anomaly', 'trend'];
    const colors = [0xff6b35, 0x35ff6b, 0xff3535, 0x3535ff];
    
    insightTypes.forEach((type, index) => {
      const material = new THREE.MeshBasicMaterial({
        color: colors[index],
        transparent: true,
        opacity: 0.7
      });
      this.insightMaterials.set(type, material);
    });
  }

  private initializePatternVisualization(): void {
    this.patternVisualization = new THREE.Group();
    this.scene.add(this.patternVisualization);
    
    // Pattern visualization will show recurring patterns in memory behavior
    this.patternVisualization.position.set(20, 0, 0);
  }

  private initializeStatisticalAnalysis(): void {
    const metrics = [
      'memoryUsage', 'fragmentation', 'cacheHitRate', 
      'ioThroughput', 'compressionRatio', 'queryLatency'
    ];
    
    metrics.forEach(metric => {
      this.movingAverages.set(metric, []);
      this.anomalyThresholds.set(metric, { mean: 0, std: 1 });
      
      // Initialize correlation matrix
      const correlations = new Map<string, number>();
      metrics.forEach(otherMetric => {
        correlations.set(otherMetric, 0);
      });
      this.correlationMatrix.set(metric, correlations);
    });
  }

  public recordMetrics(
    storageMetrics: StorageMetrics,
    memoryStats: MemoryStats,
    sdrPatterns: SDRPattern[],
    recentOperations: MemoryOperation[]
  ): void {
    const timestamp = Date.now();
    
    // Calculate derived metrics
    const gcPressure = this.calculateGCPressure(recentOperations);
    const allocationRate = this.calculateAllocationRate(recentOperations);
    const deallocationRate = this.calculateDeallocationRate(recentOperations);
    const sdrSparsity = this.calculateAverageSparsity(sdrPatterns);
    const queryLatency = this.calculateQueryLatency(recentOperations);
    
    const metrics: MemoryPerformanceMetrics = {
      timestamp,
      allocatedMemory: storageMetrics.usedStorage,
      freeMemory: storageMetrics.freeStorage,
      fragmentationLevel: storageMetrics.fragmentation,
      allocationRate,
      deallocationRate,
      gcPressure,
      cacheHitRate: storageMetrics.cacheHitRate,
      ioThroughput: storageMetrics.ioOperations,
      compressionRatio: storageMetrics.compressionRatio,
      sdrSparsity,
      queryLatency
    };
    
    // Add to history
    this.performanceHistory.push(metrics);
    
    // Limit history size
    if (this.performanceHistory.length > this.config.historySize) {
      this.performanceHistory.shift();
    }
    
    // Update statistical analysis
    this.updateStatisticalAnalysis(metrics);
    
    // Generate insights
    this.generateInsights(metrics);
    
    // Update visualization
    this.updateChartVisualization();
  }

  private calculateGCPressure(operations: MemoryOperation[]): number {
    const recentOperations = operations.filter(op => 
      Date.now() - op.timestamp < 10000 // Last 10 seconds
    );
    
    const deallocations = recentOperations.filter(op => op.type === 'delete');
    return deallocations.length / Math.max(recentOperations.length, 1);
  }

  private calculateAllocationRate(operations: MemoryOperation[]): number {
    const recentOperations = operations.filter(op => 
      Date.now() - op.timestamp < 1000 // Last second
    );
    
    return recentOperations.filter(op => op.type === 'write').length;
  }

  private calculateDeallocationRate(operations: MemoryOperation[]): number {
    const recentOperations = operations.filter(op => 
      Date.now() - op.timestamp < 1000 // Last second
    );
    
    return recentOperations.filter(op => op.type === 'delete').length;
  }

  private calculateAverageSparsity(patterns: SDRPattern[]): number {
    if (patterns.length === 0) return 0;
    
    const totalSparsity = patterns.reduce((sum, pattern) => 
      sum + (pattern.activeBits.size / pattern.totalBits), 0
    );
    
    return totalSparsity / patterns.length;
  }

  private calculateQueryLatency(operations: MemoryOperation[]): number {
    const readOperations = operations.filter(op => 
      op.type === 'read' && op.duration !== undefined
    );
    
    if (readOperations.length === 0) return 0;
    
    const totalLatency = readOperations.reduce((sum, op) => sum + (op.duration || 0), 0);
    return totalLatency / readOperations.length;
  }

  private updateStatisticalAnalysis(metrics: MemoryPerformanceMetrics): void {
    const metricMap: Record<string, number> = {
      memoryUsage: metrics.allocatedMemory / (metrics.allocatedMemory + metrics.freeMemory),
      fragmentation: metrics.fragmentationLevel,
      cacheHitRate: metrics.cacheHitRate,
      ioThroughput: metrics.ioThroughput,
      compressionRatio: metrics.compressionRatio,
      queryLatency: metrics.queryLatency
    };
    
    // Update moving averages and anomaly detection
    for (const [metricName, value] of Object.entries(metricMap)) {
      const averages = this.movingAverages.get(metricName) || [];
      averages.push(value);
      
      // Keep only last N values for moving average
      if (averages.length > 20) {
        averages.shift();
      }
      
      this.movingAverages.set(metricName, averages);
      
      // Update anomaly thresholds
      if (averages.length > 5) {
        const mean = averages.reduce((sum, val) => sum + val, 0) / averages.length;
        const variance = averages.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / averages.length;
        const std = Math.sqrt(variance);
        
        this.anomalyThresholds.set(metricName, { mean, std });
      }
    }
    
    // Update correlations
    this.updateCorrelationMatrix(metricMap);
  }

  private updateCorrelationMatrix(metrics: Record<string, number>): void {
    const metricNames = Object.keys(metrics);
    
    for (const metric1 of metricNames) {
      const correlations = this.correlationMatrix.get(metric1);
      if (!correlations) continue;
      
      for (const metric2 of metricNames) {
        if (metric1 === metric2) continue;
        
        // Simple correlation calculation (could be improved with proper statistical methods)
        const correlation = this.calculateCorrelation(metric1, metric2);
        correlations.set(metric2, correlation);
      }
    }
  }

  private calculateCorrelation(metric1: string, metric2: string): number {
    if (this.performanceHistory.length < 10) return 0;
    
    const recent = this.performanceHistory.slice(-20); // Last 20 data points
    
    const values1 = recent.map(m => this.getMetricValue(m, metric1));
    const values2 = recent.map(m => this.getMetricValue(m, metric2));
    
    const mean1 = values1.reduce((sum, val) => sum + val, 0) / values1.length;
    const mean2 = values2.reduce((sum, val) => sum + val, 0) / values2.length;
    
    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;
    
    for (let i = 0; i < values1.length; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;
      
      numerator += diff1 * diff2;
      denominator1 += diff1 * diff1;
      denominator2 += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(denominator1 * denominator2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private getMetricValue(metrics: MemoryPerformanceMetrics, metricName: string): number {
    switch (metricName) {
      case 'memoryUsage': return metrics.allocatedMemory / (metrics.allocatedMemory + metrics.freeMemory);
      case 'fragmentation': return metrics.fragmentationLevel;
      case 'cacheHitRate': return metrics.cacheHitRate;
      case 'ioThroughput': return metrics.ioThroughput;
      case 'compressionRatio': return metrics.compressionRatio;
      case 'queryLatency': return metrics.queryLatency;
      default: return 0;
    }
  }

  private generateInsights(metrics: MemoryPerformanceMetrics): void {
    const newInsights: MemoryInsight[] = [];
    
    // Check for memory pressure
    const memoryUsage = metrics.allocatedMemory / (metrics.allocatedMemory + metrics.freeMemory);
    if (memoryUsage > this.config.alertThresholds.memoryUsage) {
      newInsights.push({
        id: `memory-pressure-${Date.now()}`,
        type: 'warning',
        severity: memoryUsage > 0.9 ? 'critical' : 'high',
        title: 'High Memory Usage',
        description: `Memory usage is at ${(memoryUsage * 100).toFixed(1)}%`,
        recommendation: 'Consider running garbage collection or freeing unused allocations',
        metrics: { memoryUsage, fragmentation: metrics.fragmentationLevel },
        timestamp: metrics.timestamp
      });
    }
    
    // Check for fragmentation
    if (metrics.fragmentationLevel > this.config.alertThresholds.fragmentation) {
      newInsights.push({
        id: `fragmentation-${Date.now()}`,
        type: 'optimization',
        severity: 'medium',
        title: 'Memory Fragmentation Detected',
        description: `Fragmentation level is ${(metrics.fragmentationLevel * 100).toFixed(1)}%`,
        recommendation: 'Consider memory compaction or allocation strategy optimization',
        metrics: { fragmentation: metrics.fragmentationLevel },
        timestamp: metrics.timestamp
      });
    }
    
    // Check for cache efficiency
    if (metrics.cacheHitRate < this.config.alertThresholds.cacheHitRate) {
      newInsights.push({
        id: `cache-miss-${Date.now()}`,
        type: 'optimization',
        severity: 'medium',
        title: 'Low Cache Hit Rate',
        description: `Cache hit rate is ${(metrics.cacheHitRate * 100).toFixed(1)}%`,
        recommendation: 'Review cache sizing and eviction policies',
        metrics: { cacheHitRate: metrics.cacheHitRate },
        timestamp: metrics.timestamp
      });
    }
    
    // Check for compression efficiency
    if (metrics.compressionRatio < this.config.alertThresholds.compressionRatio) {
      newInsights.push({
        id: `compression-${Date.now()}`,
        type: 'optimization',
        severity: 'low',
        title: 'Low Compression Ratio',
        description: `Compression ratio is ${metrics.compressionRatio.toFixed(2)}:1`,
        recommendation: 'Consider alternative compression algorithms or data structures',
        metrics: { compressionRatio: metrics.compressionRatio },
        timestamp: metrics.timestamp
      });
    }
    
    // Detect anomalies
    this.detectAnomalies(metrics, newInsights);
    
    // Pattern detection
    this.detectPatterns(metrics, newInsights);
    
    // Add insights to collection
    this.insights.push(...newInsights);
    
    // Limit insights history
    if (this.insights.length > 100) {
      this.insights.splice(0, this.insights.length - 100);
    }
    
    // Update visualization
    this.updateInsightVisualization();
  }

  private detectAnomalies(metrics: MemoryPerformanceMetrics, insights: MemoryInsight[]): void {
    const metricMap: Record<string, number> = {
      memoryUsage: metrics.allocatedMemory / (metrics.allocatedMemory + metrics.freeMemory),
      fragmentation: metrics.fragmentationLevel,
      cacheHitRate: metrics.cacheHitRate,
      ioThroughput: metrics.ioThroughput,
      compressionRatio: metrics.compressionRatio,
      queryLatency: metrics.queryLatency
    };
    
    for (const [metricName, value] of Object.entries(metricMap)) {
      const threshold = this.anomalyThresholds.get(metricName);
      if (!threshold) continue;
      
      const zScore = Math.abs((value - threshold.mean) / threshold.std);
      
      if (zScore > 3) { // 3-sigma anomaly detection
        insights.push({
          id: `anomaly-${metricName}-${Date.now()}`,
          type: 'anomaly',
          severity: zScore > 4 ? 'high' : 'medium',
          title: `${metricName} Anomaly Detected`,
          description: `${metricName} value (${value.toFixed(3)}) is ${zScore.toFixed(1)} standard deviations from normal`,
          recommendation: `Investigate recent changes that might affect ${metricName}`,
          metrics: { [metricName]: value, zScore },
          timestamp: metrics.timestamp
        });
      }
    }
  }

  private detectPatterns(metrics: MemoryPerformanceMetrics, insights: MemoryInsight[]): void {
    if (this.performanceHistory.length < 10) return;
    
    // Simple trend detection for fragmentation
    const recentFragmentation = this.performanceHistory.slice(-5).map(m => m.fragmentationLevel);
    const fragmentationTrend = this.calculateTrend(recentFragmentation);
    
    if (Math.abs(fragmentationTrend) > 0.01) { // Significant trend
      const pattern: MemoryPattern = {
        patternId: `fragmentation-trend-${Date.now()}`,
        patternType: 'fragmentation',
        frequency: 1.0,
        impact: Math.abs(fragmentationTrend),
        predictedTrend: fragmentationTrend > 0 ? 'degrading' : 'improving',
        timeWindow: 5000, // 5 seconds
        confidence: 0.7
      };
      
      this.patterns.set(pattern.patternId, pattern);
      
      insights.push({
        id: `pattern-${pattern.patternId}`,
        type: 'trend',
        severity: pattern.impact > 0.02 ? 'medium' : 'low',
        title: `Fragmentation ${pattern.predictedTrend} Trend`,
        description: `Memory fragmentation is ${pattern.predictedTrend} at a rate of ${(fragmentationTrend * 100).toFixed(2)}% per measurement`,
        recommendation: pattern.predictedTrend === 'degrading' ? 
          'Consider proactive memory compaction' : 
          'Current memory management strategy is effective',
        metrics: { trend: fragmentationTrend, impact: pattern.impact },
        timestamp: metrics.timestamp
      });
    }
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    // Simple linear regression slope
    const n = values.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  private updateChartVisualization(): void {
    if (this.performanceHistory.length < 2) return;
    
    const maxPoints = Math.min(this.performanceHistory.length, 50);
    const recentHistory = this.performanceHistory.slice(-maxPoints);
    
    // Update each chart line
    this.chartLines.forEach((line, metricName) => {
      const points: THREE.Vector3[] = [];
      
      for (let i = 0; i < recentHistory.length; i++) {
        const x = (i / (recentHistory.length - 1)) * 25 - 12.5; // Chart width
        const value = this.getMetricValue(recentHistory[i], metricName);
        const y = (value * 15) - 5; // Scale to chart height
        
        points.push(new THREE.Vector3(x, y, 0));
      }
      
      line.geometry.setFromPoints(points);
    });
  }

  private updateInsightVisualization(): void {
    // Clear existing insight meshes
    this.insightMeshes.forEach(mesh => {
      this.insightPanels.remove(mesh);
      mesh.geometry.dispose();
    });
    this.insightMeshes.clear();
    
    // Create meshes for recent insights
    const recentInsights = this.insights.slice(-10);
    
    recentInsights.forEach((insight, index) => {
      const material = this.insightMaterials.get(insight.type);
      if (!material) return;
      
      const mesh = new THREE.Mesh(this.insightGeometry.clone(), material);
      mesh.position.set(-20, 15 - index * 3, 0);
      
      this.insightPanels.add(mesh);
      this.insightMeshes.set(insight.id, mesh);
    });
  }

  public animate(): void {
    const deltaTime = this.animationClock.getDelta();
    const elapsedTime = this.animationClock.getElapsedTime();
    
    // Update shader time
    if (this.chartMaterial.uniforms.time) {
      this.chartMaterial.uniforms.time.value = elapsedTime;
    }
    
    // Periodic analysis
    if (elapsedTime - this.lastAnalysisTime > this.analysisInterval / 1000) {
      this.runPeriodicAnalysis();
      this.lastAnalysisTime = elapsedTime;
    }
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }

  private runPeriodicAnalysis(): void {
    // Clean up old insights
    const cutoffTime = Date.now() - 30000; // 30 seconds
    this.insights = this.insights.filter(insight => insight.timestamp > cutoffTime);
    
    // Clean up old patterns
    const currentTime = Date.now();
    for (const [patternId, pattern] of this.patterns) {
      if (currentTime - pattern.timeWindow > 60000) { // 1 minute
        this.patterns.delete(patternId);
      }
    }
    
    // Update alerts (high severity insights)
    this.alerts = this.insights.filter(insight => 
      insight.severity === 'high' || insight.severity === 'critical'
    );
  }

  public getInsights(type?: MemoryInsight['type']): MemoryInsight[] {
    if (type) {
      return this.insights.filter(insight => insight.type === type);
    }
    return [...this.insights];
  }

  public getPatterns(): MemoryPattern[] {
    return Array.from(this.patterns.values());
  }

  public getAlerts(): MemoryInsight[] {
    return [...this.alerts];
  }

  public getCorrelationMatrix(): Map<string, Map<string, number>> {
    return new Map(this.correlationMatrix);
  }

  public getOptimizationRecommendations(): string[] {
    const recommendations: string[] = [];
    
    // Based on recent insights
    const optimizationInsights = this.insights.filter(insight => 
      insight.type === 'optimization' && 
      Date.now() - insight.timestamp < 60000 // Last minute
    );
    
    optimizationInsights.forEach(insight => {
      recommendations.push(insight.recommendation);
    });
    
    // Based on patterns
    for (const pattern of this.patterns.values()) {
      if (pattern.predictedTrend === 'degrading' && pattern.confidence > 0.6) {
        recommendations.push(`Address ${pattern.patternType} degradation pattern`);
      }
    }
    
    return recommendations;
  }

  public dispose(): void {
    // Dispose of all Three.js resources
    this.chartGeometry.dispose();
    this.chartMaterial.dispose();
    this.insightGeometry.dispose();
    
    this.chartLines.forEach(line => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    });
    
    this.insightMaterials.forEach(material => {
      material.dispose();
    });
    
    this.insightMeshes.forEach(mesh => {
      mesh.geometry.dispose();
    });
    
    this.renderer.dispose();
    
    // Clear collections
    this.performanceHistory.length = 0;
    this.insights.length = 0;
    this.patterns.clear();
    this.alerts.length = 0;
    this.chartLines.clear();
    this.insightMaterials.clear();
    this.insightMeshes.clear();
    this.movingAverages.clear();
    this.correlationMatrix.clear();
    this.anomalyThresholds.clear();
  }
}

export default MemoryAnalytics;