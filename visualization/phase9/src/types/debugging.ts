export interface TraceSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  operationName: string;
  serviceName: string;
  startTime: number;
  endTime: number;
  duration: number;
  status: 'success' | 'error' | 'warning';
  tags: Record<string, any>;
  logs: TraceLog[];
  references: SpanReference[];
}

export interface TraceLog {
  timestamp: number;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  fields?: Record<string, any>;
}

export interface SpanReference {
  type: 'child_of' | 'follows_from';
  spanId: string;
  traceId: string;
}

export interface DistributedTrace {
  traceId: string;
  spans: TraceSpan[];
  rootSpan: TraceSpan;
  services: string[];
  startTime: number;
  endTime: number;
  duration: number;
  spanCount: number;
  errorCount: number;
}

export interface TimeTravelSnapshot {
  id: string;
  timestamp: number;
  label: string;
  state: {
    patterns: any[];
    connections: any[];
    memory: any;
    activations: any;
  };
  metadata: {
    trigger: string;
    changes: string[];
    performance: {
      cpu: number;
      memory: number;
    };
  };
}

export interface TimeTravelSession {
  sessionId: string;
  snapshots: TimeTravelSnapshot[];
  currentIndex: number;
  playbackSpeed: number;
  isPlaying: boolean;
  comparison?: {
    baseSnapshot: string;
    compareSnapshot: string;
  };
}

export interface QueryAnalysis {
  queryId: string;
  query: string;
  timestamp: number;
  executionTime: number;
  plan: QueryPlan;
  profile: QueryProfile;
  suggestions: OptimizationSuggestion[];
  bottlenecks: Bottleneck[];
}

export interface QueryPlan {
  nodes: PlanNode[];
  estimatedCost: number;
  estimatedRows: number;
}

export interface PlanNode {
  id: string;
  type: string;
  operation: string;
  cost: number;
  rows: number;
  width: number;
  children: PlanNode[];
  details: Record<string, any>;
}

export interface QueryProfile {
  actualTime: number;
  planningTime: number;
  executionTime: number;
  rowsProcessed: number;
  bytesProcessed: number;
  memoryUsed: number;
  cacheHits: number;
  cacheMisses: number;
}

export interface OptimizationSuggestion {
  type: 'index' | 'query' | 'schema' | 'cache';
  priority: 'low' | 'medium' | 'high';
  description: string;
  impact: string;
  implementation: string;
}

export interface Bottleneck {
  component: string;
  operation: string;
  duration: number;
  percentage: number;
  cause: string;
}

export interface ErrorLog {
  id: string;
  timestamp: number;
  level: 'warning' | 'error' | 'critical';
  category: string;
  message: string;
  stack?: string;
  context: {
    service: string;
    operation: string;
    userId?: string;
    requestId?: string;
    metadata?: Record<string, any>;
  };
  frequency: number;
  firstSeen: number;
  lastSeen: number;
  resolved: boolean;
}

export interface ErrorStats {
  total: number;
  byLevel: Record<string, number>;
  byCategory: Record<string, number>;
  byService: Record<string, number>;
  trend: {
    timestamp: number;
    count: number;
  }[];
  topErrors: ErrorLog[];
}

export interface DebuggerState {
  isActive: boolean;
  breakpoints: Breakpoint[];
  currentBreakpoint?: Breakpoint;
  callStack: CallFrame[];
  variables: Variable[];
  watchExpressions: WatchExpression[];
}

export interface Breakpoint {
  id: string;
  location: {
    file: string;
    line: number;
    column?: number;
  };
  condition?: string;
  hitCount: number;
  enabled: boolean;
}

export interface CallFrame {
  id: string;
  functionName: string;
  location: {
    file: string;
    line: number;
    column: number;
  };
  arguments: Variable[];
  locals: Variable[];
}

export interface Variable {
  name: string;
  value: any;
  type: string;
  scope: 'local' | 'global' | 'closure';
  mutable: boolean;
}

export interface WatchExpression {
  id: string;
  expression: string;
  value: any;
  error?: string;
}