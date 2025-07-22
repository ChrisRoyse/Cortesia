// Phase 6 Performance Monitoring Types

export interface PerformanceMetrics {
  timestamp: number;
  cognitive: CognitiveMetrics;
  sdr: SDRMetrics;
  mcp: MCPMetrics;
  system: SystemMetrics;
}

export interface CognitiveMetrics {
  subcortical: LayerMetrics;
  cortical: LayerMetrics;
  thalamic: LayerMetrics;
}

export interface LayerMetrics {
  activationRate: number;
  inhibitionRate: number;
  processingLatency: number;
  throughput: number;
  errorCount: number;
  hebbianLearningRate?: number;
  attentionFocus?: number;
  cognitiveLoad?: number;
}

export interface SDRMetrics {
  creationRate: number;
  averageSparsity: number;
  overlapRatio: number;
  memoryUsage: number;
  compressionRatio?: number;
  semanticAccuracy?: number;
  storageEfficiency?: number;
}

export interface MCPMetrics {
  messageRate: number;
  averageLatency: number;
  errorRate: number;
  queueLength: number;
  protocolOverhead?: number;
  throughput?: number;
  reliability?: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskIO: number;
  networkIO: number;
  gpuUsage?: number;
  cacheHitRate?: number;
}

export interface PerformanceTrend {
  metric: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  changePercent: number;
  forecast?: number[];
}

export interface PerformanceAlert {
  id: string;
  timestamp: number;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  component: string;
  metric: string;
  value: number;
  threshold: number;
  message: string;
  acknowledged?: boolean;
  resolvedAt?: number;
}

export interface PerformanceThresholds {
  cognitive: {
    maxLatency: number;
    minThroughput: number;
    maxErrorRate: number;
    hebbianLearningRange: [number, number];
  };
  sdr: {
    sparsityRange: [number, number];
    maxMemoryUsage: number;
    minCompressionRatio: number;
  };
  mcp: {
    maxLatency: number;
    maxErrorRate: number;
    maxQueueLength: number;
  };
  system: {
    maxCPU: number;
    maxMemory: number;
    maxDiskIO: number;
    maxNetworkIO: number;
  };
}

export interface PerformanceSnapshot {
  id: string;
  timestamp: number;
  name: string;
  description?: string;
  metrics: PerformanceMetrics;
  comparison?: {
    baseline: string;
    differences: Record<string, number>;
  };
}

export interface PerformanceOptimization {
  id: string;
  category: 'cognitive' | 'sdr' | 'mcp' | 'system';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  effort: 'low' | 'medium' | 'high';
  estimatedImprovement: number;
  status: 'suggested' | 'in_progress' | 'completed' | 'rejected';
}

export interface ChartConfiguration {
  timeWindow: number; // seconds
  refreshRate: number; // milliseconds
  metrics: string[];
  chartType: 'line' | 'bar' | 'area' | 'scatter' | 'heatmap';
  aggregation?: 'none' | 'average' | 'sum' | 'min' | 'max';
  showPredictions?: boolean;
  enableZoom?: boolean;
  showAnomalies?: boolean;
}

export interface MetricDefinition {
  key: string;
  name: string;
  unit: string;
  category: string;
  description: string;
  goodRange?: [number, number];
  warningThreshold?: number;
  criticalThreshold?: number;
  aggregationType?: 'average' | 'sum' | 'min' | 'max' | 'latest';
}

export interface PerformanceReport {
  id: string;
  generatedAt: number;
  period: {
    start: number;
    end: number;
  };
  summary: {
    overallHealth: number; // 0-100
    totalAlerts: number;
    criticalIssues: number;
    optimizationOpportunities: number;
  };
  sections: {
    cognitive: CognitivePerformanceAnalysis;
    sdr: SDRPerformanceAnalysis;
    mcp: MCPPerformanceAnalysis;
    system: SystemPerformanceAnalysis;
  };
  recommendations: PerformanceOptimization[];
}

export interface CognitivePerformanceAnalysis {
  averageLatency: Record<string, number>;
  throughput: Record<string, number>;
  errorRates: Record<string, number>;
  bottlenecks: string[];
  insights: string[];
}

export interface SDRPerformanceAnalysis {
  sparsityDistribution: number[];
  memoryEfficiency: number;
  semanticPreservation: number;
  compressionStats: {
    average: number;
    min: number;
    max: number;
  };
  insights: string[];
}

export interface MCPPerformanceAnalysis {
  protocolEfficiency: number;
  messageDistribution: Record<string, number>;
  latencyPercentiles: {
    p50: number;
    p90: number;
    p95: number;
    p99: number;
  };
  reliabilityScore: number;
  insights: string[];
}

export interface SystemPerformanceAnalysis {
  resourceUtilization: {
    cpu: { average: number; peak: number };
    memory: { average: number; peak: number };
    disk: { average: number; peak: number };
    network: { average: number; peak: number };
  };
  scalabilityAssessment: string;
  recommendations: string[];
}

// Hook return types
export interface UsePerformanceMetrics {
  metrics: PerformanceMetrics[];
  currentMetrics: PerformanceMetrics | null;
  trends: PerformanceTrend[];
  alerts: PerformanceAlert[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

export interface UsePerformanceOptimization {
  optimizations: PerformanceOptimization[];
  applyOptimization: (id: string) => Promise<void>;
  rejectOptimization: (id: string) => void;
  getOptimizationDetails: (id: string) => PerformanceOptimization | null;
}

export interface UsePerformanceReport {
  generateReport: (period: { start: Date; end: Date }) => Promise<PerformanceReport>;
  exportReport: (reportId: string, format: 'pdf' | 'csv' | 'json') => Promise<Blob>;
  scheduleReport: (config: ReportSchedule) => void;
  getReportHistory: () => PerformanceReport[];
}

export interface ReportSchedule {
  frequency: 'daily' | 'weekly' | 'monthly';
  recipients?: string[];
  format: 'pdf' | 'csv' | 'json';
  includeCharts: boolean;
}