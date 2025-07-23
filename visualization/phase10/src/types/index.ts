// Unified type definitions for LLMKG Visualization System
export * from '@phase7/types';
export * from '@phase8/types';
export * from '@phase9/types';

// Core integration types
export interface LLMKGVisualizationConfig {
  mcp: {
    endpoint: string;
    protocol: 'ws' | 'http';
    authentication?: {
      type: 'bearer' | 'api-key';
      token: string;
    };
    reconnect?: {
      enabled: boolean;
      maxAttempts: number;
      delay: number;
    };
  };
  visualization: {
    theme: 'light' | 'dark' | 'auto';
    updateInterval: number;
    maxDataPoints: number;
    enableAnimations: boolean;
    enableDebugMode: boolean;
  };
  performance: {
    enableProfiling: boolean;
    sampleRate: number;
    maxMemoryUsage: number;
    enableLazyLoading: boolean;
  };
  features: {
    enabledPhases: string[];
    experimentalFeatures: string[];
  };
}

export interface LLMKGContext {
  config: LLMKGVisualizationConfig;
  mcpClient: any | null;
  sdrProcessor: any | null;
  cognitiveEngine: any | null;
  knowledgeGraph: any | null;
  connected: boolean;
  connectionStatus: ConnectionStatus;
  error: Error | null;
  lastUpdate: number;
}

export type ConnectionStatus = 
  | 'disconnected' 
  | 'connecting' 
  | 'connected' 
  | 'reconnecting' 
  | 'error';

export interface VisualizationUpdate {
  type: UpdateType;
  timestamp: number;
  data: any;
  source: string;
}

export type UpdateType = 
  | 'cognitive-state'
  | 'sdr-update'
  | 'knowledge-graph-update'
  | 'performance-metrics'
  | 'system-event'
  | 'memory-update'
  | 'pattern-update'
  | 'debug-event';

export interface ComponentRegistration {
  id: string;
  name: string;
  phase: string;
  component: React.ComponentType<any>;
  props?: Record<string, any>;
  dependencies?: string[];
  enabled: boolean;
}

export interface NavigationItem {
  key: string;
  label: string;
  icon?: string;
  path: string;
  phase: string;
  children?: NavigationItem[];
}

export interface DashboardLayout {
  id: string;
  name: string;
  components: LayoutComponent[];
  gridLayout: GridLayoutItem[];
}

export interface LayoutComponent {
  id: string;
  componentId: string;
  title: string;
  props: Record<string, any>;
}

export interface GridLayoutItem {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
  static?: boolean;
}

export interface ThemeConfig {
  primaryColor: string;
  backgroundColor: string;
  textColor: string;
  borderColor: string;
  accentColor: string;
  chartColors: string[];
  darkMode: boolean;
}

// Git and version control types
export interface GitInfo {
  branch: string;
  commit: string;
  shortCommit: string;
  author: string;
  email: string;
  message: string;
  timestamp: string;
  isDirty: boolean;
  tags: string[];
}

export interface BuildInfo {
  version: string;
  buildNumber: string;
  buildDate: string;
  environment: 'development' | 'staging' | 'production';
  gitInfo: GitInfo;
}

export interface ChangelogEntry {
  version: string;
  date: string;
  author: string;
  changes: {
    type: 'feature' | 'fix' | 'enhancement' | 'breaking';
    description: string;
    component?: string;
  }[];
}

export interface DeploymentRecord {
  id: string;
  version: string;
  environment: string;
  timestamp: string;
  author: string;
  status: 'success' | 'failed' | 'in-progress';
  duration?: number;
  rollbackFrom?: string;
}

// Documentation types
export interface DocSection {
  id: string;
  title: string;
  content: string;
  examples: CodeExample[];
  api?: APIEndpoint[];
}

export interface CodeExample {
  title: string;
  language: string;
  code: string;
  runnable: boolean;
}

export interface APIEndpoint {
  method: string;
  path: string;
  description: string;
  parameters: Parameter[];
  response: ResponseSchema;
}

export interface Parameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

export interface ResponseSchema {
  type: string;
  properties: Record<string, any>;
}

// Error handling types
export interface ErrorInfo {
  id: string;
  type: 'connection' | 'component' | 'data' | 'performance';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  details?: any;
  timestamp: number;
  component?: string;
  stack?: string;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

// Performance monitoring types
export interface PerformanceMetrics {
  renderTime: number;
  componentCount: number;
  memoryUsage: number;
  updateFrequency: number;
  dataProcessingTime: number;
  networkLatency: number;
}

export interface PerformanceAlert {
  id: string;
  type: 'memory' | 'render' | 'network' | 'data';
  threshold: number;
  currentValue: number;
  message: string;
  timestamp: number;
}

// Event system types
export interface EventBusMessage {
  type: string;
  data: any;
  source: string;
  timestamp: number;
  id: string;
}

export type EventHandler = (message: EventBusMessage) => void;

export interface EventSubscription {
  id: string;
  type: string;
  handler: EventHandler;
  options?: {
    once?: boolean;
    priority?: number;
  };
}