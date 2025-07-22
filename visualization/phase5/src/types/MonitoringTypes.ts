/**
 * Phase 5 Real-Time System Monitoring Types
 * 
 * Comprehensive type definitions for LLMKG real-time monitoring system
 * including LLMKG-specific cognitive pattern monitoring and brain-inspired components.
 */

// Core monitoring interfaces
export interface MonitoringConfig {
  updateInterval: number; // Sub-100ms for critical metrics
  websocketEndpoint: string;
  enableCognitivePatterns: boolean;
  enableBrainComponents: boolean;
  enableMCPToolMonitoring: boolean;
  enableMemorySystemMonitoring: boolean;
  enableFederationMonitoring: boolean;
  maxHistorySize: number;
  alertThresholds: AlertThresholds;
}

// Component status and health definitions
export type ComponentStatus = 'active' | 'idle' | 'processing' | 'error' | 'degraded' | 'offline';

export interface ComponentHealth {
  componentId: string;
  status: ComponentStatus;
  healthScore: number; // 0-100
  lastUpdated: number;
  uptime: number;
  errorRate: number;
  responseTime: number;
  throughput: number;
  cpuUsage: number;
  memoryUsage: number;
  trend: 'improving' | 'stable' | 'degrading';
}

// LLMKG-specific component types
export type LLMKGComponentType = 
  | 'activation_engine'
  | 'inhibitory_circuit' 
  | 'cognitive_pattern'
  | 'working_memory'
  | 'sdr_storage'
  | 'knowledge_engine'
  | 'mcp_tool'
  | 'federation_node'
  | 'neural_bridge'
  | 'attention_manager'
  | 'pattern_detector';

// Cognitive pattern activation monitoring
export interface CognitivePatternActivation {
  patternId: string;
  patternType: 'abstract' | 'convergent' | 'divergent' | 'critical' | 'lateral' | 'systems' | 'adaptive';
  activationLevel: number; // 0-1
  timestamp: number;
  duration: number;
  affectedComponents: string[];
  confidence: number;
  inhibitionLevel: number;
  excitationLevel: number;
}

// Brain-inspired component health
export interface BrainComponentHealth {
  componentType: LLMKGComponentType;
  neuralActivityLevel: number; // 0-1
  synapticStrength: number; // 0-1
  plasticityScore: number; // 0-1
  inhibitionBalance: number; // -1 to 1
  excitationBalance: number; // -1 to 1
  adaptationRate: number; // 0-1
  connectivityDensity: number; // 0-1
  informationFlow: number; // bits/sec
}

// MCP tool monitoring
export interface MCPToolHealth {
  toolId: string;
  toolName: string;
  toolType: string;
  isAvailable: boolean;
  responseTime: number;
  successRate: number;
  errorCount: number;
  lastUsed: number;
  usageFrequency: number;
  performanceScore: number;
  dependencies: string[];
  version: string;
}

// Memory system utilization
export interface MemorySystemMetrics {
  sdrUtilization: number; // 0-1
  workingMemoryLoad: number; // 0-1
  longTermStorageUsage: number; // 0-1
  compressionRatio: number;
  accessLatency: number;
  hitRate: number;
  evictionRate: number;
  fragmentationLevel: number;
  indexEfficiency: number;
}

// Federation node health
export interface FederationNodeHealth {
  nodeId: string;
  isOnline: boolean;
  connectionQuality: number; // 0-1
  syncLatency: number;
  messageQueue: number;
  conflictResolutionRate: number;
  dataConsistency: number; // 0-1
  bandwidthUsage: number;
  peerCount: number;
  trustScore: number; // 0-1
}

// Real-time performance metrics
export interface PerformanceMetrics {
  componentId: string;
  timestamp: number;
  cpu: number;
  memory: number;
  latency: number;
  throughput: number;
  errorRate: number;
  customMetrics: Record<string, number>;
}

// Historical data points
export interface HistoricalDataPoint {
  timestamp: number;
  value: number;
  metric: string;
  componentId: string;
}

// Alert system definitions
export interface AlertThresholds {
  cpu: { warning: number; critical: number };
  memory: { warning: number; critical: number };
  latency: { warning: number; critical: number };
  errorRate: { warning: number; critical: number };
  healthScore: { warning: number; critical: number };
  cognitivePatterns: {
    maxActivationTime: number;
    minActivationLevel: number;
    maxInhibitionImbalance: number;
  };
  memorySystem: {
    maxUtilization: number;
    minHitRate: number;
    maxFragmentation: number;
  };
  federation: {
    minConnectionQuality: number;
    maxSyncLatency: number;
    minTrustScore: number;
  };
}

export interface SystemAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  title: string;
  message: string;
  componentId: string;
  componentType: LLMKGComponentType;
  timestamp: number;
  acknowledged: boolean;
  resolvedAt?: number;
  metadata: Record<string, any>;
}

// System dashboard data structures
export interface SystemHealthSummary {
  overall: ComponentStatus;
  healthScore: number;
  totalComponents: number;
  activeComponents: number;
  degradedComponents: number;
  offlineComponents: number;
  activeAlerts: number;
  cognitivePatternActivity: number;
  memorySystemHealth: number;
  federationHealth: number;
  lastUpdated: number;
}

// WebSocket message types for real-time updates
export interface WebSocketMessage {
  type: 'health_update' | 'performance_metrics' | 'cognitive_activation' | 'alert' | 'system_status';
  timestamp: number;
  data: any;
}

export interface HealthUpdateMessage extends WebSocketMessage {
  type: 'health_update';
  data: ComponentHealth;
}

export interface PerformanceMetricsMessage extends WebSocketMessage {
  type: 'performance_metrics';
  data: PerformanceMetrics;
}

export interface CognitiveActivationMessage extends WebSocketMessage {
  type: 'cognitive_activation';
  data: CognitivePatternActivation;
}

export interface AlertMessage extends WebSocketMessage {
  type: 'alert';
  data: SystemAlert;
}

export interface SystemStatusMessage extends WebSocketMessage {
  type: 'system_status';
  data: SystemHealthSummary;
}

// Visualization-specific types
export interface HealthVisualizationConfig {
  showTrends: boolean;
  animateChanges: boolean;
  colorScheme: 'default' | 'colorblind' | 'dark';
  updateFrequency: number;
  showDetails: boolean;
}

export interface PerformanceChartConfig {
  timeWindow: number; // seconds
  refreshRate: number; // ms
  metrics: string[];
  showPredictions: boolean;
  enableZoom: boolean;
  chartType: 'line' | 'bar' | 'area';
}

// Component monitoring subscription
export interface MonitoringSubscription {
  componentId: string;
  metrics: string[];
  callback: (data: PerformanceMetrics) => void;
  throttleMs: number;
}

// Integration with Phase 1 telemetry
export interface TelemetryIntegration {
  endpoint: string;
  authenticationType: 'none' | 'bearer' | 'api_key';
  credentials?: string;
  dataFormat: 'json' | 'msgpack' | 'protobuf';
  compression: boolean;
}

// Export interfaces for monitoring system
export interface MonitoringSystemInterface {
  startMonitoring(config: MonitoringConfig): Promise<void>;
  stopMonitoring(): Promise<void>;
  subscribeToComponent(subscription: MonitoringSubscription): () => void;
  getSystemHealth(): Promise<SystemHealthSummary>;
  getComponentHealth(componentId: string): Promise<ComponentHealth>;
  getCognitivePatternActivity(): Promise<CognitivePatternActivation[]>;
  getMemorySystemMetrics(): Promise<MemorySystemMetrics>;
  getFederationHealth(): Promise<FederationNodeHealth[]>;
  configureAlerts(thresholds: AlertThresholds): void;
  acknowledgeAlert(alertId: string): void;
  exportHealthData(format: 'json' | 'csv' | 'xml'): Promise<Blob>;
}