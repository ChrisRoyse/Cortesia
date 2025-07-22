// Core LLMKG Dashboard Types
export interface LLMKGData {
  cognitive: CognitiveData;
  neural: NeuralData;
  knowledgeGraph: KnowledgeGraphData;
  memory: MemoryData;
  timestamp: number;
}

export interface CognitiveData {
  patterns: CognitivePattern[];
  inhibitoryLevel: number;
  activationThreshold: number;
  hierarchicalDepth: number;
}

export interface CognitivePattern {
  id: string;
  type: 'hierarchical' | 'lateral' | 'feedback';
  strength: number;
  activeNodes: string[];
  timestamp: number;
}

export interface NeuralData {
  activity: NeuralActivity[];
  layers: NeuralLayer[];
  connections: NeuralConnection[];
  overallActivity: number;
}

export interface NeuralActivity {
  nodeId: string;
  activation: number;
  position: { x: number; y: number; z?: number };
  layer: number;
}

export interface NeuralLayer {
  id: string;
  name: string;
  nodeCount: number;
  averageActivation: number;
}

export interface NeuralConnection {
  from: string;
  to: string;
  weight: number;
  active: boolean;
}

export interface KnowledgeGraphData {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  clusters: KnowledgeCluster[];
  metrics: GraphMetrics;
}

export interface KnowledgeNode {
  id: string;
  label: string;
  type: 'concept' | 'entity' | 'relation' | 'property';
  weight: number;
  position: { x: number; y: number };
  metadata: Record<string, any>;
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  confidence: number;
}

export interface KnowledgeCluster {
  id: string;
  nodes: string[];
  centroid: { x: number; y: number };
  density: number;
  topic: string;
}

export interface GraphMetrics {
  nodeCount: number;
  edgeCount: number;
  clusterCount: number;
  density: number;
  avgDegree: number;
}

export interface MemoryData {
  usage: MemoryUsage;
  performance: PerformanceMetrics;
  stores: MemoryStore[];
}

export interface MemoryUsage {
  total: number;
  used: number;
  available: number;
  percentage: number;
}

export interface PerformanceMetrics {
  latency: number;
  throughput: number;
  errorRate: number;
  uptime: number;
}

export interface MemoryStore {
  id: string;
  name: string;
  type: 'sdr' | 'zce' | 'cache';
  size: number;
  utilization: number;
  accessCount: number;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'data' | 'subscribe' | 'unsubscribe' | 'error' | 'ping' | 'pong';
  topic?: string;
  topics?: string[];
  data?: any;
  error?: string;
  timestamp: number;
}

export interface WebSocketContextType {
  isConnected: boolean;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  data: LLMKGData | null;
  lastMessage: WebSocketMessage | null;
  send: (message: WebSocketMessage) => void;
  subscribe: (topics: string[]) => void;
  unsubscribe: (topics: string[]) => void;
  error: string | null;
}

// MCP Types
export interface MCPTool {
  name: string;
  description: string;
  parameters: Record<string, any>;
  category: 'cognitive' | 'neural' | 'knowledge' | 'memory' | 'utility';
}

export interface MCPContextType {
  tools: MCPTool[];
  loading: boolean;
  error: string | null;
  executeTool: (toolName: string, parameters: Record<string, any>) => Promise<any>;
  refreshTools: () => Promise<void>;
}

// Dashboard Types
export interface DashboardConfig {
  theme: 'light' | 'dark' | 'auto';
  refreshRate: number;
  maxDataPoints: number;
  enableAnimations: boolean;
}

export interface DashboardState {
  config: DashboardConfig;
  activeView: string;
  isFullscreen: boolean;
  sidebarCollapsed: boolean;
}

// Component Types
export interface ComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingState {
  isLoading: boolean;
  progress?: number;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  error: Error | null;
  errorBoundary?: boolean;
}

// Event Types
export interface DashboardEvent {
  type: string;
  payload: any;
  timestamp: number;
  source: string;
}

export interface UserInteraction {
  action: 'click' | 'hover' | 'select' | 'zoom' | 'pan';
  target: string;
  data: any;
  timestamp: number;
}

// Theme Types
export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  error: string;
  warning: string;
  success: string;
}

export interface Theme {
  colors: ThemeColors;
  spacing: Record<string, string>;
  typography: Record<string, any>;
  breakpoints: Record<string, string>;
  animations: Record<string, any>;
}