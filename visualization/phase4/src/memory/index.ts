/**
 * Memory Visualization Module Index
 * Exports all SDR and memory visualization components for Phase 4
 */

// Core SDR Visualization
export { 
  default as SDRVisualizer,
  SDRConfig,
  SDRPattern,
  SDRVisualizationConfig,
  SDRComparisonResult
} from './SDRVisualizer';

// Memory Operations Visualization
export {
  default as MemoryOperationVisualizer,
  MemoryOperation,
  MemoryBlock,
  MemoryVisualizationConfig,
  MemoryStats
} from './MemoryOperationVisualizer';

// Storage Efficiency Monitoring
export {
  default as StorageEfficiency,
  StorageMetrics,
  StorageBlock,
  StorageEfficiencyConfig,
  EfficiencyTrend
} from './StorageEfficiency';

// Memory Analytics and Insights
export {
  default as MemoryAnalytics,
  MemoryPerformanceMetrics,
  MemoryInsight,
  MemoryPattern,
  AnalyticsConfig
} from './MemoryAnalytics';

// Type re-exports for convenience
export type {
  MemoryPerformanceMetrics as PerformanceMetrics,
  MemoryInsight as Insight,
  MemoryPattern as Pattern
} from './MemoryAnalytics';

/**
 * Combined Memory Visualization System
 * Orchestrates all memory visualization components
 */
import SDRVisualizer, { SDRVisualizationConfig, SDRPattern } from './SDRVisualizer';
import MemoryOperationVisualizer, { MemoryVisualizationConfig, MemoryOperation } from './MemoryOperationVisualizer';
import StorageEfficiency, { StorageEfficiencyConfig, StorageBlock } from './StorageEfficiency';
import MemoryAnalytics, { AnalyticsConfig } from './MemoryAnalytics';

export interface MemoryVisualizationSystemConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  
  // Component-specific configs
  sdrConfig: Partial<SDRVisualizationConfig>;
  operationConfig: Partial<MemoryVisualizationConfig>;
  efficiencyConfig: Partial<StorageEfficiencyConfig>;
  analyticsConfig: Partial<AnalyticsConfig>;
  
  // Layout configuration
  layout: 'grid' | 'tabs' | 'overlay' | 'split';
  updateInterval: number;
  
  // Data integration settings
  enableRealTimeUpdates: boolean;
  enableCrossComponentAnalysis: boolean;
  maxHistorySize: number;
}

export class MemoryVisualizationSystem {
  private sdrVisualizer: SDRVisualizer;
  private operationVisualizer: MemoryOperationVisualizer;
  private storageEfficiency: StorageEfficiency;
  private memoryAnalytics: MemoryAnalytics;
  
  private config: MemoryVisualizationSystemConfig;
  private updateTimer: number | null = null;
  
  constructor(config: MemoryVisualizationSystemConfig) {
    this.config = { ...config };
    
    // Initialize components with merged configurations
    this.sdrVisualizer = new SDRVisualizer({
      canvas: config.canvas,
      width: config.width,
      height: config.height,
      maxPatterns: 1000,
      cellSize: 2,
      gridDimensions: { rows: 32, cols: 64 },
      colorScheme: {
        active: new THREE.Color(0x00ff88),
        inactive: new THREE.Color(0x333333),
        highlight: new THREE.Color(0xff8800)
      },
      ...config.sdrConfig
    } as SDRVisualizationConfig);
    
    this.operationVisualizer = new MemoryOperationVisualizer({
      canvas: config.canvas,
      width: config.width,
      height: config.height,
      memorySize: 1024 * 1024 * 1024, // 1GB
      blockHeight: 4,
      animationDuration: 2.0,
      colorScheme: {
        read: new THREE.Color(0x4488ff),
        write: new THREE.Color(0x44ff44),
        update: new THREE.Color(0xffff44),
        delete: new THREE.Color(0xff4444),
        allocated: new THREE.Color(0x88ff88),
        free: new THREE.Color(0x333333),
        reserved: new THREE.Color(0xff8888)
      },
      ...config.operationConfig
    } as MemoryVisualizationConfig);
    
    this.storageEfficiency = new StorageEfficiency({
      canvas: config.canvas,
      width: config.width,
      height: config.height,
      maxBlocks: 2000,
      updateInterval: 1000,
      colorScheme: {
        data: new THREE.Color(0x4488ff),
        index: new THREE.Color(0xff8844),
        cache: new THREE.Color(0x44ff88),
        free: new THREE.Color(0x333333),
        fragmented: new THREE.Color(0xff4444),
        compressed: new THREE.Color(0x8844ff)
      },
      ...config.efficiencyConfig
    } as StorageEfficiencyConfig);
    
    this.memoryAnalytics = new MemoryAnalytics({
      canvas: config.canvas,
      width: config.width,
      height: config.height,
      historySize: config.maxHistorySize || 1000,
      analysisWindow: 30000, // 30 seconds
      alertThresholds: {
        fragmentation: 0.3,
        memoryUsage: 0.8,
        cacheHitRate: 0.6,
        compressionRatio: 1.5,
        queryLatency: 100
      },
      ...config.analyticsConfig
    } as AnalyticsConfig);
    
    // Start real-time updates if enabled
    if (config.enableRealTimeUpdates) {
      this.startRealTimeUpdates();
    }
  }
  
  // SDR Operations
  public addSDRPattern(pattern: SDRPattern): void {
    this.sdrVisualizer.addSDRPattern(pattern);
  }
  
  public removeSDRPattern(patternId: string): void {
    this.sdrVisualizer.removeSDRPattern(patternId);
  }
  
  public compareSDRPatterns(patternIdA: string, patternIdB: string) {
    return this.sdrVisualizer.comparePatterns(patternIdA, patternIdB);
  }
  
  // Memory Operations
  public startMemoryOperation(operation: MemoryOperation): void {
    this.operationVisualizer.startOperation(operation);
  }
  
  public completeMemoryOperation(operationId: string, success: boolean): void {
    this.operationVisualizer.completeOperation(operationId, success);
  }
  
  public addMemoryBlock(block: MemoryBlock): void {
    this.operationVisualizer.addMemoryBlock(block);
  }
  
  // Storage Efficiency
  public updateStorageBlock(block: StorageBlock): void {
    this.storageEfficiency.updateStorageBlock(block);
  }
  
  public recordIOOperation(): void {
    this.storageEfficiency.recordIOOperation();
  }
  
  // Analytics Integration
  public recordMetrics(): void {
    const storageMetrics = this.storageEfficiency.getMetrics();
    const memoryStats = this.operationVisualizer.getStats();
    const operationHistory = this.operationVisualizer.getOperationHistory();
    
    // Get SDR patterns (would need to be implemented in SDRVisualizer)
    const sdrPatterns: SDRPattern[] = []; // Placeholder
    
    this.memoryAnalytics.recordMetrics(
      storageMetrics,
      memoryStats,
      sdrPatterns,
      operationHistory
    );
  }
  
  // System Control
  public animate(): void {
    // Animate all components
    this.sdrVisualizer.animate();
    this.operationVisualizer.animate();
    this.storageEfficiency.animate();
    this.memoryAnalytics.animate();
    
    // Cross-component analysis if enabled
    if (this.config.enableCrossComponentAnalysis) {
      this.performCrossComponentAnalysis();
    }
  }
  
  private startRealTimeUpdates(): void {
    this.updateTimer = window.setInterval(() => {
      this.recordMetrics();
    }, this.config.updateInterval);
  }
  
  private stopRealTimeUpdates(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
  }
  
  private performCrossComponentAnalysis(): void {
    // Analyze relationships between different memory subsystems
    const insights = this.memoryAnalytics.getInsights();
    const patterns = this.memoryAnalytics.getPatterns();
    
    // Example: Correlate SDR sparsity with memory fragmentation
    // This would involve more sophisticated analysis
  }
  
  // Performance and Diagnostics
  public getPerformanceMetrics() {
    return {
      sdr: this.sdrVisualizer.getPerformanceMetrics(),
      operations: this.operationVisualizer.getStats(),
      efficiency: this.storageEfficiency.getMetrics(),
      analytics: this.memoryAnalytics.getInsights('warning'),
      system: {
        totalMemory: this.getTotalMemoryUsage(),
        activeComponents: 4,
        updateRate: 1000 / this.config.updateInterval
      }
    };
  }
  
  private getTotalMemoryUsage(): number {
    // Estimate total memory usage across all visualization components
    return 50 * 1024 * 1024; // 50MB estimate
  }
  
  public getOptimizationRecommendations(): string[] {
    const recommendations = this.memoryAnalytics.getOptimizationRecommendations();
    
    // Add system-level recommendations
    const systemRecommendations: string[] = [];
    
    const metrics = this.getPerformanceMetrics();
    if (metrics.system.totalMemory > 100 * 1024 * 1024) {
      systemRecommendations.push('Consider reducing visualization detail for better performance');
    }
    
    return [...recommendations, ...systemRecommendations];
  }
  
  public dispose(): void {
    this.stopRealTimeUpdates();
    
    this.sdrVisualizer.dispose();
    this.operationVisualizer.dispose();
    this.storageEfficiency.dispose();
    this.memoryAnalytics.dispose();
  }
}

export default MemoryVisualizationSystem;