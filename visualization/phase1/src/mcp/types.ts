/**
 * @fileoverview MCP Protocol Type Definitions for LLMKG Visualization
 * 
 * This module defines all TypeScript interfaces and types for the Model Context Protocol (MCP)
 * used in LLMKG visualization system. It provides complete type safety for MCP communication
 * between the visualization dashboard and LLMKG's MCP servers.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

/**
 * Base MCP message structure that all protocol messages extend
 */
export interface MCPMessage {
  /** Unique identifier for the message */
  id: string;
  /** The method name being called */
  method: string;
  /** Optional parameters for the method call */
  params?: any;
  /** JSON-RPC version (always "2.0" for MCP) */
  jsonrpc: "2.0";
}

/**
 * MCP request message structure
 */
export interface MCPRequest extends MCPMessage {
  /** Request parameters specific to the method */
  params?: Record<string, any>;
}

/**
 * MCP response message structure
 */
export interface MCPResponse extends Omit<MCPMessage, 'method'> {
  /** Response result data */
  result?: any;
  /** Error information if the request failed */
  error?: MCPError;
}

/**
 * MCP notification message (no response expected)
 */
export interface MCPNotification extends Omit<MCPMessage, 'id'> {
  /** Notification parameters */
  params?: Record<string, any>;
}

/**
 * MCP error structure
 */
export interface MCPError {
  /** Error code following JSON-RPC error codes */
  code: number;
  /** Human-readable error message */
  message: string;
  /** Additional error data */
  data?: any;
}

/**
 * MCP tool definition with complete schema information
 */
export interface MCPTool {
  /** Tool name (unique identifier) */
  name: string;
  /** Human-readable description of what the tool does */
  description: string;
  /** JSON Schema for tool input parameters */
  inputSchema: {
    type: "object";
    properties?: Record<string, any>;
    required?: string[];
    additionalProperties?: boolean;
  };
}

/**
 * MCP server capabilities
 */
export interface MCPCapabilities {
  /** List of supported tools */
  tools?: {
    /** Whether the server supports listing tools */
    listChanged?: boolean;
  };
  /** Supported resources */
  resources?: {
    /** Whether the server supports subscribing to resource changes */
    subscribe?: boolean;
    /** Whether the server supports listing resources */
    listChanged?: boolean;
  };
  /** Supported prompts */
  prompts?: {
    /** Whether the server supports listing prompts */
    listChanged?: boolean;
  };
  /** Experimental features */
  experimental?: Record<string, any>;
}

/**
 * Client information for MCP initialization
 */
export interface MCPClientInfo {
  /** Client name */
  name: string;
  /** Client version */
  version: string;
  /** Client capabilities */
  capabilities: MCPCapabilities;
}

/**
 * Server information received during MCP initialization
 */
export interface MCPServerInfo {
  /** Server name */
  name: string;
  /** Server version */
  version: string;
  /** Server capabilities */
  capabilities: MCPCapabilities;
  /** Protocol version supported */
  protocolVersion: string;
  /** Additional server metadata */
  metadata?: Record<string, any>;
}

/**
 * Connection state enumeration
 */
export const enum ConnectionState {
  DISCONNECTED = "disconnected",
  CONNECTING = "connecting", 
  CONNECTED = "connected",
  RECONNECTING = "reconnecting",
  ERROR = "error"
}

/**
 * Connection configuration options
 */
export interface ConnectionConfig {
  /** Server endpoint URL */
  endpoint: string;
  /** Connection timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retry attempts */
  maxRetries?: number;
  /** Base delay for exponential backoff (ms) */
  baseDelay?: number;
  /** Maximum delay between retries (ms) */
  maxDelay?: number;
  /** Custom headers for the connection */
  headers?: Record<string, string>;
  /** Authentication token if required */
  authToken?: string;
}

/**
 * Retry configuration for failed operations
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  maxAttempts: number;
  /** Base delay in milliseconds */
  baseDelay: number;
  /** Maximum delay in milliseconds */
  maxDelay: number;
  /** Multiplier for exponential backoff */
  backoffMultiplier: number;
  /** Jitter to add randomness to delays */
  jitter: boolean;
}

/**
 * Event types emitted by the MCP client
 */
export const enum MCPEventType {
  CONNECTION_STATE_CHANGED = "connectionStateChanged",
  CONNECTION_ERROR = "connectionError",
  TOOLS_DISCOVERED = "toolsDiscovered",
  MESSAGE_RECEIVED = "messageReceived",
  MESSAGE_SENT = "messageSent",
  ERROR_OCCURRED = "errorOccurred",
  TOOL_CALLED = "toolCalled",
  TOOL_RESPONSE = "toolResponse",
  TELEMETRY_EVENT = "telemetryEvent"
}

/**
 * Base event structure
 */
export interface MCPEvent {
  /** Event type */
  type: MCPEventType;
  /** Event timestamp */
  timestamp: Date;
  /** Event data */
  data: any;
}

/**
 * Connection state change event
 */
export interface ConnectionStateChangeEvent extends MCPEvent {
  type: MCPEventType.CONNECTION_STATE_CHANGED;
  data: {
    oldState: ConnectionState;
    newState: ConnectionState;
    endpoint: string;
    error?: Error;
  };
}

/**
 * Message event for sent/received messages
 */
export interface MessageEvent extends MCPEvent {
  type: MCPEventType.MESSAGE_RECEIVED | MCPEventType.MESSAGE_SENT;
  data: {
    message: MCPMessage;
    endpoint: string;
  };
}

/**
 * Error event
 */
export interface ErrorEvent extends MCPEvent {
  type: MCPEventType.ERROR_OCCURRED;
  data: {
    error: Error;
    context: string;
    endpoint?: string;
  };
}

/**
 * Tool call event
 */
export interface ToolEvent extends MCPEvent {
  type: MCPEventType.TOOL_CALLED | MCPEventType.TOOL_RESPONSE;
  data: {
    toolName: string;
    params?: any;
    result?: any;
    error?: Error;
    duration?: number;
  };
}

/**
 * Telemetry event for metrics collection
 */
export interface TelemetryEvent extends MCPEvent {
  type: MCPEventType.TELEMETRY_EVENT;
  data: {
    metric: string;
    value: number | string;
    tags?: Record<string, string>;
    metadata?: Record<string, any>;
  };
}

/**
 * LLMKG-specific tool schemas for type safety
 */
export namespace LLMKGTools {
  /**
   * Brain visualization tool parameters
   */
  export interface BrainVisualizationParams {
    /** Region of interest to visualize */
    region?: string;
    /** Visualization type */
    type: "activation" | "connectivity" | "sdr";
    /** Time range for the visualization */
    timeRange?: {
      start: Date;
      end: Date;
    };
    /** Resolution level */
    resolution?: "low" | "medium" | "high";
  }

  /**
   * Federated learning metrics parameters
   */
  export interface FederatedMetricsParams {
    /** Client ID to get metrics for */
    clientId?: string;
    /** Metric types to retrieve */
    metrics: string[];
    /** Aggregation period */
    period?: "1m" | "5m" | "15m" | "1h" | "1d";
  }

  /**
   * Knowledge graph query parameters
   */
  export interface KnowledgeGraphParams {
    /** Query string */
    query: string;
    /** Maximum number of results */
    limit?: number;
    /** Include relationship strength */
    includeWeights?: boolean;
    /** Filter by entity types */
    entityTypes?: string[];
  }
}

/**
 * LLMKG-specific tools interface for type-safe tool calls
 */
export interface LLMKGTools {
  /** Generate brain visualization data */
  brainVisualization(params: LLMKGTools.BrainVisualizationParams): Promise<any>;
  
  /** Analyze connectivity patterns */
  analyzeConnectivity(startNode: string, maxDepth?: number): Promise<any>;
  
  /** Get federated metrics */
  federatedMetrics(params: Record<string, any>): Promise<any>;
}

/**
 * Type guard to check if an object is an MCP message
 */
export function isMCPMessage(obj: any): obj is MCPMessage {
  return (
    typeof obj === "object" &&
    obj !== null &&
    typeof obj.id === "string" &&
    typeof obj.method === "string" &&
    obj.jsonrpc === "2.0"
  );
}

/**
 * Type guard to check if an object is an MCP response
 */
export function isMCPResponse(obj: any): obj is MCPResponse {
  return (
    typeof obj === "object" &&
    obj !== null &&
    typeof obj.id === "string" &&
    obj.jsonrpc === "2.0" &&
    (obj.result !== undefined || obj.error !== undefined)
  );
}

/**
 * Type guard to check if an object is an MCP error
 */
export function isMCPError(obj: any): obj is MCPError {
  return (
    typeof obj === "object" &&
    obj !== null &&
    typeof obj.code === "number" &&
    typeof obj.message === "string"
  );
}

/**
 * Standard MCP error codes
 */
export const enum MCPErrorCode {
  PARSE_ERROR = -32700,
  INVALID_REQUEST = -32600,
  METHOD_NOT_FOUND = -32601,
  INVALID_PARAMS = -32602,
  INTERNAL_ERROR = -32603,
  SERVER_ERROR_START = -32099,
  SERVER_ERROR_END = -32000,
  CONNECTION_ERROR = -32001,
  TIMEOUT_ERROR = -32002,
  AUTHENTICATION_ERROR = -32003
}