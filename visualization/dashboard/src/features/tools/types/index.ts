// MCP Tool Catalog Type Definitions

export type ToolCategory = 
  | 'knowledge-graph'
  | 'cognitive'
  | 'neural'
  | 'memory'
  | 'analysis'
  | 'federation'
  | 'utility';

export type ToolStatus = 
  | 'healthy'
  | 'degraded'
  | 'unavailable'
  | 'unknown';

export interface JSONSchema {
  type: string;
  properties?: Record<string, any>;
  required?: string[];
  additionalProperties?: boolean;
  items?: any;
  enum?: any[];
  description?: string;
  examples?: any[];
}

export interface ToolExample {
  name: string;
  description: string;
  input: Record<string, any>;
  output?: Record<string, any>;
  tags?: string[];
}

export interface ToolSchema {
  inputSchema: JSONSchema;
  outputSchema: JSONSchema;
  errorSchema?: JSONSchema;
  examples?: ToolExample[];
}

export interface ToolMetrics {
  totalExecutions: number;
  successRate: number;
  averageResponseTime: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  lastExecutionTime?: Date;
  errorCount: number;
  errorTypes: Record<string, number>;
}

export interface ToolDocumentation {
  summary: string;
  description: string;
  parameters: ParameterDoc[];
  returns: ReturnDoc;
  examples: CodeExample[];
  relatedTools?: string[];
  tags?: string[];
}

export interface ParameterDoc {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default?: any;
  examples?: any[];
}

export interface ReturnDoc {
  type: string;
  description: string;
  schema?: JSONSchema;
}

export interface CodeExample {
  language: 'javascript' | 'python' | 'curl' | 'rust';
  code: string;
  description?: string;
}

export interface MCPTool {
  id: string;
  name: string;
  version: string;
  description: string;
  category: ToolCategory;
  inputSchema: JSONSchema;
  outputSchema?: JSONSchema;
  examples?: ToolExample[];
  status: ToolStatusInfo;
  responseTime?: number;
  endpoint?: string;
  tags?: string[];
  metrics: ToolMetrics;
  createdAt: Date;
  updatedAt: Date;
}

export interface ToolStatusInfo {
  available: boolean;
  health: ToolStatus;
  lastChecked: Date;
  responseTime: number;
  errorRate: number;
  message?: string;
  details?: Record<string, any>;
}

export type ExecutionStatus = 'pending' | 'running' | 'success' | 'error' | 'cancelled';

export interface ToolExecution {
  id: string;
  toolId: string;
  toolName?: string;
  input: any;
  output?: any;
  error?: string | ToolError;
  status: ExecutionStatus;
  startTime: number | Date;
  endTime?: number | Date;
  duration?: number;
  metadata?: Record<string, any>;
}

export interface ToolError {
  code: string;
  message: string;
  details?: any;
  stack?: string;
}

export interface ToolFilter {
  categories?: ToolCategory[];
  status?: ToolStatus[];
  searchTerm?: string;
  tags?: string[];
  sortBy?: 'name' | 'category' | 'status' | 'performance';
  sortOrder?: 'asc' | 'desc';
}

export interface ToolUsageAnalytics {
  toolId: string;
  period: 'hour' | 'day' | 'week' | 'month';
  executions: number;
  successCount: number;
  errorCount: number;
  averageResponseTime: number;
  peakUsageTime?: Date;
  userCount?: number;
  patterns?: UsagePattern[];
}

export interface UsagePattern {
  type: 'sequence' | 'temporal' | 'correlation';
  description: string;
  frequency: number;
  tools?: string[];
  timeRange?: { start: Date; end: Date };
}