import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';
import { LLMKGData, CognitiveData, NeuralData, KnowledgeGraphData, MemoryData } from '../../types';

// Enhanced interfaces for real-time state management
export interface RealtimeState {
  // Data streams
  cognitive: CognitiveStreamState;
  neural: NeuralStreamState;
  knowledgeGraph: KnowledgeGraphStreamState;
  memory: MemoryStreamState;
  
  // Performance monitoring
  performance: PerformanceState;
  
  // Stream management
  subscriptions: SubscriptionState;
  
  // Error handling
  errors: ErrorState;
  
  // Advanced features
  analytics: AnalyticsState;
  filters: FilterState;
}

export interface CognitiveStreamState {
  current: CognitiveData | null;
  history: CognitiveData[];
  patterns: PatternBuffer<CognitiveData>;
  aggregations: TimeWindowAggregation[];
  lastUpdate: number;
  isActive: boolean;
  quality: DataQualityMetrics;
}

export interface NeuralStreamState {
  current: NeuralData | null;
  history: NeuralData[];
  patterns: PatternBuffer<NeuralData>;
  aggregations: TimeWindowAggregation[];
  layerAnalysis: LayerAnalysisState;
  spikeEvents: SpikeEventBuffer;
  lastUpdate: number;
  isActive: boolean;
  quality: DataQualityMetrics;
}

export interface KnowledgeGraphStreamState {
  current: KnowledgeGraphData | null;
  history: KnowledgeGraphData[];
  patterns: PatternBuffer<KnowledgeGraphData>;
  aggregations: TimeWindowAggregation[];
  nodeChanges: NodeChangeBuffer;
  edgeChanges: EdgeChangeBuffer;
  communities: CommunityBuffer;
  lastUpdate: number;
  isActive: boolean;
  quality: DataQualityMetrics;
}

export interface MemoryStreamState {
  current: MemoryData | null;
  history: MemoryData[];
  patterns: PatternBuffer<MemoryData>;
  aggregations: TimeWindowAggregation[];
  alerts: AlertBuffer;
  consolidationEvents: ConsolidationBuffer;
  lastUpdate: number;
  isActive: boolean;
  quality: DataQualityMetrics;
}

export interface PerformanceState {
  latency: LatencyMetrics;
  throughput: ThroughputMetrics;
  memory: MemoryMetrics;
  cpu: CPUMetrics;
  network: NetworkMetrics;
  errors: ErrorMetrics;
  alerts: PerformanceAlert[];
  trends: PerformanceTrend[];
}

export interface SubscriptionState {
  active: ActiveSubscription[];
  pending: PendingSubscription[];
  failed: FailedSubscription[];
  totalBandwidth: number;
  lastActivityTimestamp: number;
}

export interface AnalyticsState {
  anomalies: AnomalyDetection[];
  predictions: PredictionResult[];
  correlations: CorrelationAnalysis[];
  trends: TrendAnalysis[];
  patterns: PatternAnalysis[];
  insights: DataInsight[];
}

export interface FilterState {
  temporal: TemporalFilter;
  spatial: SpatialFilter;
  thematic: ThematicFilter;
  quality: QualityFilter;
  active: boolean;
}

// Supporting interfaces
export interface PatternBuffer<T> {
  data: T[];
  maxSize: number;
  currentSize: number;
  oldestTimestamp: number;
  newestTimestamp: number;
  compressionRatio: number;
}

export interface TimeWindowAggregation {
  windowStart: number;
  windowEnd: number;
  windowSize: number;
  aggregationType: 'sum' | 'avg' | 'min' | 'max' | 'count' | 'std';
  value: number;
  dataPoints: number;
  quality: number;
}

export interface DataQualityMetrics {
  completeness: number;     // 0-1: Percentage of expected data received
  consistency: number;      // 0-1: Data consistency score
  accuracy: number;         // 0-1: Data accuracy score
  timeliness: number;       // 0-1: Data freshness score
  validity: number;         // 0-1: Data validation score
  lastAssessment: number;
}

export interface LayerAnalysisState {
  layerMetrics: Record<string, LayerMetric>;
  connectivityMatrix: number[][];
  activationWaves: ActivationWave[];
  spatialClusters: SpatialCluster[];
}

export interface LayerMetric {
  id: string;
  averageActivation: number;
  peakActivation: number;
  nodeCount: number;
  activeNodeCount: number;
  utilization: number;
  trend: 'rising' | 'falling' | 'stable';
}

export interface ActivationWave {
  id: string;
  origin: { x: number; y: number };
  radius: number;
  strength: number;
  propagationSpeed: number;
  timestamp: number;
  affectedLayers: string[];
}

export interface SpatialCluster {
  id: string;
  centroid: { x: number; y: number };
  radius: number;
  nodeCount: number;
  averageActivation: number;
  density: number;
  stability: number;
}

export interface SpikeEventBuffer {
  events: SpikeEvent[];
  maxSize: number;
  totalCount: number;
  averageRate: number;
  peakRate: number;
  lastSpikeTimestamp: number;
}

export interface SpikeEvent {
  timestamp: number;
  nodeId: string;
  amplitude: number;
  duration: number;
  layer: string;
  position: { x: number; y: number };
}

export interface NodeChangeBuffer {
  changes: NodeChange[];
  maxSize: number;
  changeRate: number;
  significantChanges: number;
}

export interface EdgeChangeBuffer {
  changes: EdgeChange[];
  maxSize: number;
  changeRate: number;
  significantChanges: number;
}

export interface CommunityBuffer {
  communities: Community[];
  maxSize: number;
  stabilityScore: number;
  formationRate: number;
}

export interface NodeChange {
  timestamp: number;
  nodeId: string;
  changeType: 'added' | 'removed' | 'modified';
  oldValue?: any;
  newValue?: any;
  impact: number;
}

export interface EdgeChange {
  timestamp: number;
  edgeId: string;
  changeType: 'added' | 'removed' | 'modified';
  source: string;
  target: string;
  oldValue?: any;
  newValue?: any;
  impact: number;
}

export interface Community {
  id: string;
  nodes: string[];
  density: number;
  modularity: number;
  timestamp: number;
  stability: number;
}

export interface AlertBuffer {
  alerts: Alert[];
  maxSize: number;
  activeCount: number;
  criticalCount: number;
  lastAlertTimestamp: number;
}

export interface ConsolidationBuffer {
  events: ConsolidationEvent[];
  maxSize: number;
  successRate: number;
  averageEfficiencyGain: number;
  lastEventTimestamp: number;
}

export interface Alert {
  id: string;
  timestamp: number;
  type: 'usage' | 'performance' | 'error' | 'anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  message: string;
  value: number;
  threshold: number;
  acknowledged: boolean;
}

export interface ConsolidationEvent {
  id: string;
  timestamp: number;
  storeId: string;
  type: 'gc' | 'defrag' | 'merge';
  beforeSize: number;
  afterSize: number;
  duration: number;
  efficiencyGain: number;
  success: boolean;
}

// Performance metrics interfaces
export interface LatencyMetrics {
  current: number;
  average: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  trend: 'improving' | 'stable' | 'degrading';
}

export interface ThroughputMetrics {
  current: number;
  average: number;
  peak: number;
  messagesPerSecond: number;
  bytesPerSecond: number;
  trend: 'increasing' | 'stable' | 'decreasing';
}

export interface MemoryMetrics {
  heapUsed: number;
  heapTotal: number;
  bufferUsage: number;
  gcFrequency: number;
  leakDetected: boolean;
}

export interface CPUMetrics {
  usage: number;
  loadAverage: number;
  processingTime: number;
  queueLength: number;
}

export interface NetworkMetrics {
  bytesReceived: number;
  bytesSent: number;
  packetsLost: number;
  connectionQuality: number;
  bandwidth: number;
}

export interface ErrorMetrics {
  total: number;
  rate: number;
  types: Record<string, number>;
  recent: Error[];
}

export interface PerformanceAlert {
  id: string;
  timestamp: number;
  metric: string;
  value: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  acknowledged: boolean;
}

export interface PerformanceTrend {
  metric: string;
  direction: 'up' | 'down' | 'stable';
  rate: number;
  confidence: number;
  duration: number;
}

// Subscription interfaces
export interface ActiveSubscription {
  id: string;
  topic: string;
  startTime: number;
  messagesReceived: number;
  bytesReceived: number;
  lastMessageTime: number;
  quality: number;
}

export interface PendingSubscription {
  id: string;
  topic: string;
  requestTime: number;
  retryCount: number;
  reason: string;
}

export interface FailedSubscription {
  id: string;
  topic: string;
  failureTime: number;
  reason: string;
  retryCount: number;
  willRetry: boolean;
}

// Analytics interfaces
export interface AnomalyDetection {
  id: string;
  timestamp: number;
  type: 'statistical' | 'pattern' | 'behavioral';
  severity: number;
  confidence: number;
  source: string;
  description: string;
  affectedData: string[];
}

export interface PredictionResult {
  id: string;
  timestamp: number;
  predictionHorizon: number;
  targetMetric: string;
  predictedValue: number;
  confidence: number;
  method: string;
}

export interface CorrelationAnalysis {
  id: string;
  timestamp: number;
  metrics: string[];
  correlation: number;
  significance: number;
  duration: number;
}

export interface TrendAnalysis {
  id: string;
  timestamp: number;
  metric: string;
  trend: 'increasing' | 'decreasing' | 'stable' | 'cyclic';
  strength: number;
  duration: number;
  forecastAccuracy: number;
}

export interface PatternAnalysis {
  id: string;
  timestamp: number;
  pattern: string;
  frequency: number;
  confidence: number;
  duration: number;
  nextOccurrence?: number;
}

export interface DataInsight {
  id: string;
  timestamp: number;
  type: 'anomaly' | 'trend' | 'pattern' | 'correlation' | 'performance';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  description: string;
  recommendations: string[];
  confidence: number;
}

// Filter interfaces
export interface TemporalFilter {
  enabled: boolean;
  startTime?: number;
  endTime?: number;
  timeWindow?: number;
  samplingRate?: number;
}

export interface SpatialFilter {
  enabled: boolean;
  boundingBox?: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  };
  layers?: string[];
  regions?: string[];
}

export interface ThematicFilter {
  enabled: boolean;
  dataTypes?: string[];
  categories?: string[];
  tags?: string[];
  severity?: string[];
}

export interface QualityFilter {
  enabled: boolean;
  minCompleteness?: number;
  minAccuracy?: number;
  minTimeliness?: number;
  maxLatency?: number;
}

export interface ErrorState {
  hasError: boolean;
  lastError: Error | null;
  errorCount: number;
  errorRate: number;
  recentErrors: Array<{
    timestamp: number;
    error: Error;
    source: string;
    resolved: boolean;
  }>;
}

// Initial state
const initialRealtimeState: RealtimeState = {
  cognitive: {
    current: null,
    history: [],
    patterns: { data: [], maxSize: 1000, currentSize: 0, oldestTimestamp: 0, newestTimestamp: 0, compressionRatio: 1.0 },
    aggregations: [],
    lastUpdate: 0,
    isActive: false,
    quality: { completeness: 1, consistency: 1, accuracy: 1, timeliness: 1, validity: 1, lastAssessment: 0 },
  },
  neural: {
    current: null,
    history: [],
    patterns: { data: [], maxSize: 1000, currentSize: 0, oldestTimestamp: 0, newestTimestamp: 0, compressionRatio: 1.0 },
    aggregations: [],
    layerAnalysis: {
      layerMetrics: {},
      connectivityMatrix: [],
      activationWaves: [],
      spatialClusters: [],
    },
    spikeEvents: { events: [], maxSize: 1000, totalCount: 0, averageRate: 0, peakRate: 0, lastSpikeTimestamp: 0 },
    lastUpdate: 0,
    isActive: false,
    quality: { completeness: 1, consistency: 1, accuracy: 1, timeliness: 1, validity: 1, lastAssessment: 0 },
  },
  knowledgeGraph: {
    current: null,
    history: [],
    patterns: { data: [], maxSize: 1000, currentSize: 0, oldestTimestamp: 0, newestTimestamp: 0, compressionRatio: 1.0 },
    aggregations: [],
    nodeChanges: { changes: [], maxSize: 1000, changeRate: 0, significantChanges: 0 },
    edgeChanges: { changes: [], maxSize: 1000, changeRate: 0, significantChanges: 0 },
    communities: { communities: [], maxSize: 100, stabilityScore: 1.0, formationRate: 0 },
    lastUpdate: 0,
    isActive: false,
    quality: { completeness: 1, consistency: 1, accuracy: 1, timeliness: 1, validity: 1, lastAssessment: 0 },
  },
  memory: {
    current: null,
    history: [],
    patterns: { data: [], maxSize: 1000, currentSize: 0, oldestTimestamp: 0, newestTimestamp: 0, compressionRatio: 1.0 },
    aggregations: [],
    alerts: { alerts: [], maxSize: 500, activeCount: 0, criticalCount: 0, lastAlertTimestamp: 0 },
    consolidationEvents: { events: [], maxSize: 200, successRate: 1.0, averageEfficiencyGain: 0, lastEventTimestamp: 0 },
    lastUpdate: 0,
    isActive: false,
    quality: { completeness: 1, consistency: 1, accuracy: 1, timeliness: 1, validity: 1, lastAssessment: 0 },
  },
  performance: {
    latency: { current: 0, average: 0, p95: 0, p99: 0, min: 0, max: 0, trend: 'stable' },
    throughput: { current: 0, average: 0, peak: 0, messagesPerSecond: 0, bytesPerSecond: 0, trend: 'stable' },
    memory: { heapUsed: 0, heapTotal: 0, bufferUsage: 0, gcFrequency: 0, leakDetected: false },
    cpu: { usage: 0, loadAverage: 0, processingTime: 0, queueLength: 0 },
    network: { bytesReceived: 0, bytesSent: 0, packetsLost: 0, connectionQuality: 1.0, bandwidth: 0 },
    errors: { total: 0, rate: 0, types: {}, recent: [] },
    alerts: [],
    trends: [],
  },
  subscriptions: {
    active: [],
    pending: [],
    failed: [],
    totalBandwidth: 0,
    lastActivityTimestamp: 0,
  },
  errors: {
    hasError: false,
    lastError: null,
    errorCount: 0,
    errorRate: 0,
    recentErrors: [],
  },
  analytics: {
    anomalies: [],
    predictions: [],
    correlations: [],
    trends: [],
    patterns: [],
    insights: [],
  },
  filters: {
    temporal: { enabled: false },
    spatial: { enabled: false },
    thematic: { enabled: false },
    quality: { enabled: false },
    active: false,
  },
};

// Async thunks for advanced operations
export const analyzeAnomalies = createAsyncThunk(
  'realtime/analyzeAnomalies',
  async (data: LLMKGData[]) => {
    // Simulate anomaly detection
    const anomalies: AnomalyDetection[] = [];
    
    if (data.length > 10) {
      const recent = data.slice(-10);
      const cognitive = recent.map(d => d.cognitive.inhibitoryLevel);
      const avgInhibition = cognitive.reduce((sum, val) => sum + val, 0) / cognitive.length;
      const stdDev = Math.sqrt(cognitive.reduce((sum, val) => sum + Math.pow(val - avgInhibition, 2), 0) / cognitive.length);
      
      const latest = cognitive[cognitive.length - 1];
      if (Math.abs(latest - avgInhibition) > 2 * stdDev) {
        anomalies.push({
          id: `anomaly_${Date.now()}`,
          timestamp: Date.now(),
          type: 'statistical',
          severity: Math.abs(latest - avgInhibition) / stdDev,
          confidence: 0.8,
          source: 'cognitive.inhibitoryLevel',
          description: `Unusual inhibitory level detected: ${latest.toFixed(3)} (avg: ${avgInhibition.toFixed(3)}, std: ${stdDev.toFixed(3)})`,
          affectedData: ['cognitive'],
        });
      }
    }
    
    return anomalies;
  }
);

export const generatePredictions = createAsyncThunk(
  'realtime/generatePredictions',
  async (data: LLMKGData[]) => {
    // Simulate prediction generation
    const predictions: PredictionResult[] = [];
    
    if (data.length > 5) {
      const recent = data.slice(-5);
      const memoryUsage = recent.map(d => d.memory.usage.percentage);
      const trend = (memoryUsage[memoryUsage.length - 1] - memoryUsage[0]) / memoryUsage.length;
      
      predictions.push({
        id: `prediction_${Date.now()}`,
        timestamp: Date.now(),
        predictionHorizon: 60000, // 1 minute
        targetMetric: 'memory.usage.percentage',
        predictedValue: Math.max(0, Math.min(100, memoryUsage[memoryUsage.length - 1] + trend * 12)), // 12 = 60s / 5 data points
        confidence: 0.7,
        method: 'linear_trend',
      });
    }
    
    return predictions;
  }
);

// Slice definition
const realtimeSlice = createSlice({
  name: 'realtime',
  initialState: initialRealtimeState,
  reducers: {
    // Cognitive data actions
    updateCognitiveData: (state, action: PayloadAction<CognitiveData>) => {
      const timestamp = Date.now();
      state.cognitive.current = action.payload;
      state.cognitive.history.push(action.payload);
      state.cognitive.lastUpdate = timestamp;
      state.cognitive.isActive = true;
      
      // Maintain history size
      if (state.cognitive.history.length > state.cognitive.patterns.maxSize) {
        state.cognitive.history.shift();
      }
      
      // Update pattern buffer
      state.cognitive.patterns.data.push(action.payload);
      state.cognitive.patterns.currentSize = state.cognitive.patterns.data.length;
      state.cognitive.patterns.newestTimestamp = timestamp;
      
      if (state.cognitive.patterns.data.length > state.cognitive.patterns.maxSize) {
        state.cognitive.patterns.data.shift();
        state.cognitive.patterns.oldestTimestamp = timestamp - (state.cognitive.patterns.maxSize * 100); // Estimate
      }
    },

    // Neural data actions
    updateNeuralData: (state, action: PayloadAction<NeuralData>) => {
      const timestamp = Date.now();
      state.neural.current = action.payload;
      state.neural.history.push(action.payload);
      state.neural.lastUpdate = timestamp;
      state.neural.isActive = true;
      
      // Maintain history size
      if (state.neural.history.length > state.neural.patterns.maxSize) {
        state.neural.history.shift();
      }
      
      // Update pattern buffer
      state.neural.patterns.data.push(action.payload);
      state.neural.patterns.currentSize = state.neural.patterns.data.length;
      state.neural.patterns.newestTimestamp = timestamp;
      
      // Update layer analysis
      action.payload.layers.forEach(layer => {
        state.neural.layerAnalysis.layerMetrics[layer.id] = {
          id: layer.id,
          averageActivation: layer.averageActivation,
          peakActivation: layer.averageActivation * 1.5, // Estimate
          nodeCount: layer.nodeCount,
          activeNodeCount: Math.floor(layer.nodeCount * layer.averageActivation),
          utilization: layer.averageActivation,
          trend: 'stable',
        };
      });
      
      // Detect spikes
      const spikes = action.payload.activity.filter(activity => activity.activation > 0.8);
      spikes.forEach(activity => {
        if (state.neural.spikeEvents.events.length >= state.neural.spikeEvents.maxSize) {
          state.neural.spikeEvents.events.shift();
        }
        
        state.neural.spikeEvents.events.push({
          timestamp,
          nodeId: activity.nodeId,
          amplitude: activity.activation,
          duration: 10, // Estimate 10ms
          layer: activity.layer.toString(),
          position: activity.position,
        });
        
        state.neural.spikeEvents.totalCount++;
        state.neural.spikeEvents.lastSpikeTimestamp = timestamp;
      });
    },

    // Knowledge Graph data actions
    updateKnowledgeGraphData: (state, action: PayloadAction<KnowledgeGraphData>) => {
      const timestamp = Date.now();
      const previousData = state.knowledgeGraph.current;
      
      state.knowledgeGraph.current = action.payload;
      state.knowledgeGraph.history.push(action.payload);
      state.knowledgeGraph.lastUpdate = timestamp;
      state.knowledgeGraph.isActive = true;
      
      // Maintain history size
      if (state.knowledgeGraph.history.length > state.knowledgeGraph.patterns.maxSize) {
        state.knowledgeGraph.history.shift();
      }
      
      // Track node changes
      if (previousData) {
        const nodeChanges: NodeChange[] = [];
        const oldNodeIds = new Set(previousData.nodes.map(n => n.id));
        const newNodeIds = new Set(action.payload.nodes.map(n => n.id));
        
        // Detect added nodes
        action.payload.nodes.forEach(node => {
          if (!oldNodeIds.has(node.id)) {
            nodeChanges.push({
              timestamp,
              nodeId: node.id,
              changeType: 'added',
              newValue: node,
              impact: node.weight,
            });
          }
        });
        
        // Detect removed nodes
        previousData.nodes.forEach(node => {
          if (!newNodeIds.has(node.id)) {
            nodeChanges.push({
              timestamp,
              nodeId: node.id,
              changeType: 'removed',
              oldValue: node,
              impact: node.weight,
            });
          }
        });
        
        // Add to buffer
        state.knowledgeGraph.nodeChanges.changes.push(...nodeChanges);
        if (state.knowledgeGraph.nodeChanges.changes.length > state.knowledgeGraph.nodeChanges.maxSize) {
          state.knowledgeGraph.nodeChanges.changes = state.knowledgeGraph.nodeChanges.changes.slice(-state.knowledgeGraph.nodeChanges.maxSize);
        }
        
        state.knowledgeGraph.nodeChanges.significantChanges += nodeChanges.filter(c => c.impact > 0.5).length;
      }
      
      // Update communities
      action.payload.clusters.forEach(cluster => {
        const community: Community = {
          id: cluster.id,
          nodes: cluster.nodes,
          density: cluster.density,
          modularity: 0.5, // Estimate
          timestamp,
          stability: cluster.density,
        };
        
        // Check if community already exists
        const existingIndex = state.knowledgeGraph.communities.communities.findIndex(c => c.id === cluster.id);
        if (existingIndex >= 0) {
          state.knowledgeGraph.communities.communities[existingIndex] = community;
        } else {
          state.knowledgeGraph.communities.communities.push(community);
          if (state.knowledgeGraph.communities.communities.length > state.knowledgeGraph.communities.maxSize) {
            state.knowledgeGraph.communities.communities.shift();
          }
        }
      });
    },

    // Memory data actions
    updateMemoryData: (state, action: PayloadAction<MemoryData>) => {
      const timestamp = Date.now();
      state.memory.current = action.payload;
      state.memory.history.push(action.payload);
      state.memory.lastUpdate = timestamp;
      state.memory.isActive = true;
      
      // Maintain history size
      if (state.memory.history.length > state.memory.patterns.maxSize) {
        state.memory.history.shift();
      }
      
      // Check for alerts
      if (action.payload.usage.percentage > 0.9) {
        const alert: Alert = {
          id: `alert_${timestamp}`,
          timestamp,
          type: 'usage',
          severity: action.payload.usage.percentage > 0.95 ? 'critical' : 'high',
          source: 'memory.usage',
          message: `High memory usage: ${(action.payload.usage.percentage * 100).toFixed(1)}%`,
          value: action.payload.usage.percentage,
          threshold: 0.9,
          acknowledged: false,
        };
        
        state.memory.alerts.alerts.push(alert);
        state.memory.alerts.activeCount++;
        if (alert.severity === 'critical') {
          state.memory.alerts.criticalCount++;
        }
        state.memory.alerts.lastAlertTimestamp = timestamp;
        
        if (state.memory.alerts.alerts.length > state.memory.alerts.maxSize) {
          const removed = state.memory.alerts.alerts.shift()!;
          if (removed.severity === 'critical') {
            state.memory.alerts.criticalCount = Math.max(0, state.memory.alerts.criticalCount - 1);
          }
        }
      }
    },

    // Performance actions
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<PerformanceState>>) => {
      Object.assign(state.performance, action.payload);
    },

    // Subscription management
    addSubscription: (state, action: PayloadAction<{ topic: string; id: string }>) => {
      const subscription: ActiveSubscription = {
        id: action.payload.id,
        topic: action.payload.topic,
        startTime: Date.now(),
        messagesReceived: 0,
        bytesReceived: 0,
        lastMessageTime: 0,
        quality: 1.0,
      };
      
      state.subscriptions.active.push(subscription);
      state.subscriptions.lastActivityTimestamp = Date.now();
    },

    removeSubscription: (state, action: PayloadAction<string>) => {
      state.subscriptions.active = state.subscriptions.active.filter(sub => sub.id !== action.payload);
    },

    // Analytics actions
    addAnomaly: (state, action: PayloadAction<AnomalyDetection>) => {
      state.analytics.anomalies.push(action.payload);
      if (state.analytics.anomalies.length > 100) {
        state.analytics.anomalies.shift();
      }
    },

    addPrediction: (state, action: PayloadAction<PredictionResult>) => {
      state.analytics.predictions.push(action.payload);
      if (state.analytics.predictions.length > 50) {
        state.analytics.predictions.shift();
      }
    },

    // Filter actions
    updateFilters: (state, action: PayloadAction<Partial<FilterState>>) => {
      Object.assign(state.filters, action.payload);
      state.filters.active = Object.values(state.filters).some(filter => 
        typeof filter === 'object' && filter && 'enabled' in filter && filter.enabled
      );
    },

    // Data quality actions
    updateDataQuality: (state, action: PayloadAction<{ stream: keyof RealtimeState; quality: Partial<DataQualityMetrics> }>) => {
      const { stream, quality } = action.payload;
      if (stream in state && 'quality' in state[stream]) {
        Object.assign((state[stream] as any).quality, { ...quality, lastAssessment: Date.now() });
      }
    },

    // Error handling
    addError: (state, action: PayloadAction<{ error: Error; source: string }>) => {
      state.errors.hasError = true;
      state.errors.lastError = action.payload.error;
      state.errors.errorCount++;
      state.errors.recentErrors.push({
        timestamp: Date.now(),
        error: action.payload.error,
        source: action.payload.source,
        resolved: false,
      });
      
      if (state.errors.recentErrors.length > 100) {
        state.errors.recentErrors.shift();
      }
      
      // Calculate error rate
      const recentWindow = Date.now() - 60000; // Last minute
      const recentErrorCount = state.errors.recentErrors.filter(e => e.timestamp > recentWindow).length;
      state.errors.errorRate = recentErrorCount / 60; // Errors per second
    },

    clearErrors: (state) => {
      state.errors.hasError = false;
      state.errors.lastError = null;
      state.errors.recentErrors = [];
      state.errors.errorRate = 0;
    },

    // Bulk operations
    resetStream: (state, action: PayloadAction<keyof RealtimeState>) => {
      const streamName = action.payload;
      if (streamName in initialRealtimeState) {
        (state as any)[streamName] = { ...(initialRealtimeState as any)[streamName] };
      }
    },

    // Aggregation actions
    updateAggregations: (state, action: PayloadAction<{ 
      stream: 'cognitive' | 'neural' | 'knowledgeGraph' | 'memory'; 
      aggregations: TimeWindowAggregation[] 
    }>) => {
      const { stream, aggregations } = action.payload;
      state[stream].aggregations = aggregations;
    },

    // Maintenance actions
    pruneHistory: (state, action: PayloadAction<{ maxAge: number }>) => {
      const cutoffTime = Date.now() - action.payload.maxAge;
      
      // Prune cognitive history
      state.cognitive.history = state.cognitive.history.filter(data => (data as any).timestamp > cutoffTime);
      state.cognitive.patterns.data = state.cognitive.patterns.data.filter(data => (data as any).timestamp > cutoffTime);
      
      // Prune neural history
      state.neural.history = state.neural.history.filter(data => (data as any).timestamp > cutoffTime);
      state.neural.patterns.data = state.neural.patterns.data.filter(data => (data as any).timestamp > cutoffTime);
      state.neural.spikeEvents.events = state.neural.spikeEvents.events.filter(event => event.timestamp > cutoffTime);
      
      // Prune knowledge graph history
      state.knowledgeGraph.history = state.knowledgeGraph.history.filter(data => (data as any).timestamp > cutoffTime);
      state.knowledgeGraph.nodeChanges.changes = state.knowledgeGraph.nodeChanges.changes.filter(change => change.timestamp > cutoffTime);
      state.knowledgeGraph.edgeChanges.changes = state.knowledgeGraph.edgeChanges.changes.filter(change => change.timestamp > cutoffTime);
      
      // Prune memory history
      state.memory.history = state.memory.history.filter(data => (data as any).timestamp > cutoffTime);
      state.memory.alerts.alerts = state.memory.alerts.alerts.filter(alert => alert.timestamp > cutoffTime);
      state.memory.consolidationEvents.events = state.memory.consolidationEvents.events.filter(event => event.timestamp > cutoffTime);
      
      // Prune analytics
      state.analytics.anomalies = state.analytics.anomalies.filter(anomaly => anomaly.timestamp > cutoffTime);
      state.analytics.predictions = state.analytics.predictions.filter(prediction => prediction.timestamp > cutoffTime);
      state.analytics.insights = state.analytics.insights.filter(insight => insight.timestamp > cutoffTime);
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(analyzeAnomalies.fulfilled, (state, action) => {
        action.payload.forEach(anomaly => {
          state.analytics.anomalies.push(anomaly);
        });
        
        // Keep only recent anomalies
        if (state.analytics.anomalies.length > 100) {
          state.analytics.anomalies = state.analytics.anomalies.slice(-100);
        }
      })
      .addCase(generatePredictions.fulfilled, (state, action) => {
        action.payload.forEach(prediction => {
          state.analytics.predictions.push(prediction);
        });
        
        // Keep only recent predictions
        if (state.analytics.predictions.length > 50) {
          state.analytics.predictions = state.analytics.predictions.slice(-50);
        }
      });
  },
});

// Export actions
export const {
  updateCognitiveData,
  updateNeuralData,
  updateKnowledgeGraphData,
  updateMemoryData,
  updatePerformanceMetrics,
  addSubscription,
  removeSubscription,
  addAnomaly,
  addPrediction,
  updateFilters,
  updateDataQuality,
  addError,
  clearErrors,
  resetStream,
  updateAggregations,
  pruneHistory,
} = realtimeSlice.actions;

// Selectors
export const selectCognitiveStream = (state: { realtime: RealtimeState }) => state.realtime.cognitive;
export const selectNeuralStream = (state: { realtime: RealtimeState }) => state.realtime.neural;
export const selectKnowledgeGraphStream = (state: { realtime: RealtimeState }) => state.realtime.knowledgeGraph;
export const selectMemoryStream = (state: { realtime: RealtimeState }) => state.realtime.memory;
export const selectPerformanceMetrics = (state: { realtime: RealtimeState }) => state.realtime.performance;
export const selectActiveSubscriptions = (state: { realtime: RealtimeState }) => state.realtime.subscriptions.active;
export const selectAnalytics = (state: { realtime: RealtimeState }) => state.realtime.analytics;
export const selectFilters = (state: { realtime: RealtimeState }) => state.realtime.filters;
export const selectErrors = (state: { realtime: RealtimeState }) => state.realtime.errors;

// Advanced selectors
export const selectStreamHealth = (state: { realtime: RealtimeState }) => {
  const streams = ['cognitive', 'neural', 'knowledgeGraph', 'memory'] as const;
  return streams.reduce((health, streamName) => {
    const stream = state.realtime[streamName];
    health[streamName] = {
      isActive: stream.isActive,
      lastUpdate: stream.lastUpdate,
      dataAge: Date.now() - stream.lastUpdate,
      quality: stream.quality,
      healthScore: Object.values(stream.quality).reduce((sum, val) => sum + val, 0) / Object.keys(stream.quality).length,
    };
    return health;
  }, {} as Record<string, any>);
};

export const selectRecentAnomalies = (state: { realtime: RealtimeState }, timeWindow: number = 300000) => {
  const cutoff = Date.now() - timeWindow;
  return state.realtime.analytics.anomalies.filter(anomaly => anomaly.timestamp > cutoff);
};

export const selectCriticalAlerts = (state: { realtime: RealtimeState }) => {
  return state.realtime.memory.alerts.alerts.filter(alert => 
    alert.severity === 'critical' && !alert.acknowledged
  );
};

export default realtimeSlice.reducer;