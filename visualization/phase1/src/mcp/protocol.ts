/**
 * @fileoverview MCP Protocol Message Handling for LLMKG Visualization
 * 
 * This module provides comprehensive protocol message handling for MCP communication,
 * including message validation, serialization, and protocol-level operations.
 * It implements the full MCP specification with LLMKG-specific extensions.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { v4 as uuidv4 } from 'uuid';
import {
  MCPMessage,
  MCPRequest,
  MCPResponse,
  MCPNotification,
  MCPError,
  MCPTool,
  MCPClientInfo,
  MCPServerInfo,
  MCPCapabilities,
  MCPErrorCode,
  isMCPMessage,
  isMCPResponse,
  isMCPError,
  LLMKGTools
} from './types.js';

/**
 * Protocol version supported by this implementation
 */
export const MCP_PROTOCOL_VERSION = "2024-11-05";

/**
 * MCP protocol message builder and validator
 */
export class MCPProtocol {
  private static instance: MCPProtocol;
  private messageId = 0;

  /**
   * Gets the singleton instance of the protocol handler
   */
  public static getInstance(): MCPProtocol {
    if (!MCPProtocol.instance) {
      MCPProtocol.instance = new MCPProtocol();
    }
    return MCPProtocol.instance;
  }

  /**
   * Generates a unique message ID
   */
  public generateMessageId(): string {
    return `msg_${++this.messageId}_${Date.now()}`;
  }

  /**
   * Creates an MCP initialize request
   * 
   * @param clientInfo - Client information
   * @returns MCP initialize request
   */
  public createInitializeRequest(clientInfo: MCPClientInfo): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "initialize",
      params: {
        protocolVersion: MCP_PROTOCOL_VERSION,
        clientInfo,
        capabilities: clientInfo.capabilities
      }
    };
  }

  /**
   * Creates an MCP initialized notification
   * 
   * @returns MCP initialized notification
   */
  public createInitializedNotification(): MCPNotification {
    return {
      jsonrpc: "2.0",
      method: "initialized",
      params: {}
    };
  }

  /**
   * Creates a tools/list request
   * 
   * @param cursor - Optional cursor for pagination
   * @returns MCP tools list request
   */
  public createToolsListRequest(cursor?: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "tools/list",
      params: cursor ? { cursor } : {}
    };
  }

  /**
   * Creates a tools/call request
   * 
   * @param name - Tool name to call
   * @param arguments_ - Tool arguments
   * @returns MCP tool call request
   */
  public createToolCallRequest(name: string, arguments_: any = {}): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "tools/call",
      params: {
        name,
        arguments: arguments_
      }
    };
  }

  /**
   * Creates a resources/list request
   * 
   * @param cursor - Optional cursor for pagination
   * @returns MCP resources list request
   */
  public createResourcesListRequest(cursor?: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "resources/list",
      params: cursor ? { cursor } : {}
    };
  }

  /**
   * Creates a resources/read request
   * 
   * @param uri - Resource URI to read
   * @returns MCP resource read request
   */
  public createResourceReadRequest(uri: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "resources/read",
      params: { uri }
    };
  }

  /**
   * Creates a resources/subscribe request
   * 
   * @param uri - Resource URI to subscribe to
   * @returns MCP resource subscribe request
   */
  public createResourceSubscribeRequest(uri: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "resources/subscribe",
      params: { uri }
    };
  }

  /**
   * Creates a resources/unsubscribe request
   * 
   * @param uri - Resource URI to unsubscribe from
   * @returns MCP resource unsubscribe request
   */
  public createResourceUnsubscribeRequest(uri: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "resources/unsubscribe",
      params: { uri }
    };
  }

  /**
   * Creates a prompts/list request
   * 
   * @param cursor - Optional cursor for pagination
   * @returns MCP prompts list request
   */
  public createPromptsListRequest(cursor?: string): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "prompts/list",
      params: cursor ? { cursor } : {}
    };
  }

  /**
   * Creates a prompts/get request
   * 
   * @param name - Prompt name
   * @param arguments_ - Prompt arguments
   * @returns MCP prompt get request
   */
  public createPromptGetRequest(name: string, arguments_: any = {}): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "prompts/get",
      params: {
        name,
        arguments: arguments_
      }
    };
  }

  /**
   * Creates a ping request for connection health checks
   * 
   * @returns MCP ping request
   */
  public createPingRequest(): MCPRequest {
    return {
      jsonrpc: "2.0",
      id: this.generateMessageId(),
      method: "ping",
      params: {}
    };
  }

  /**
   * Creates a successful response message
   * 
   * @param id - Request ID to respond to
   * @param result - Response result
   * @returns MCP success response
   */
  public createSuccessResponse(id: string, result: any): MCPResponse {
    return {
      jsonrpc: "2.0",
      id,
      result
    };
  }

  /**
   * Creates an error response message
   * 
   * @param id - Request ID to respond to
   * @param code - Error code
   * @param message - Error message
   * @param data - Optional error data
   * @returns MCP error response
   */
  public createErrorResponse(
    id: string,
    code: number,
    message: string,
    data?: any
  ): MCPResponse {
    return {
      jsonrpc: "2.0",
      id,
      error: { code, message, data }
    };
  }

  /**
   * Creates a notification message
   * 
   * @param method - Notification method
   * @param params - Notification parameters
   * @returns MCP notification
   */
  public createNotification(method: string, params: any = {}): MCPNotification {
    return {
      jsonrpc: "2.0",
      method,
      params
    };
  }

  /**
   * Validates an MCP message structure
   * 
   * @param message - Message to validate
   * @returns Validation result with errors if any
   */
  public validateMessage(message: any): {
    valid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!message || typeof message !== 'object') {
      errors.push("Message must be an object");
      return { valid: false, errors };
    }

    if (message.jsonrpc !== "2.0") {
      errors.push("Invalid jsonrpc version, must be '2.0'");
    }

    if (typeof message.method !== 'string') {
      errors.push("Method must be a string");
    }

    // Validate request/response specific fields
    if (message.id !== undefined) {
      if (typeof message.id !== 'string' && typeof message.id !== 'number') {
        errors.push("ID must be a string or number");
      }

      // This is a request or response
      if (message.result === undefined && message.error === undefined && message.method) {
        // This is a request - validate params
        if (message.params !== undefined && typeof message.params !== 'object') {
          errors.push("Params must be an object");
        }
      } else if (message.result !== undefined || message.error !== undefined) {
        // This is a response
        if (message.error && !isMCPError(message.error)) {
          errors.push("Error must have code and message properties");
        }
      }
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Validates a tool definition
   * 
   * @param tool - Tool to validate
   * @returns Validation result
   */
  public validateTool(tool: any): {
    valid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!tool || typeof tool !== 'object') {
      errors.push("Tool must be an object");
      return { valid: false, errors };
    }

    if (typeof tool.name !== 'string' || tool.name.length === 0) {
      errors.push("Tool name must be a non-empty string");
    }

    if (typeof tool.description !== 'string') {
      errors.push("Tool description must be a string");
    }

    if (!tool.inputSchema || typeof tool.inputSchema !== 'object') {
      errors.push("Tool must have an inputSchema object");
    } else {
      if (tool.inputSchema.type !== 'object') {
        errors.push("Tool inputSchema type must be 'object'");
      }
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Parses and validates a JSON message
   * 
   * @param data - Raw message data
   * @returns Parsed and validated message
   * @throws Error if message is invalid
   */
  public parseMessage(data: string | Buffer): MCPMessage {
    let parsed: any;

    try {
      parsed = JSON.parse(data.toString());
    } catch (error) {
      throw new Error(`Failed to parse JSON: ${error}`);
    }

    const validation = this.validateMessage(parsed);
    if (!validation.valid) {
      throw new Error(`Invalid MCP message: ${validation.errors.join(', ')}`);
    }

    return parsed as MCPMessage;
  }

  /**
   * Serializes a message to JSON string
   * 
   * @param message - Message to serialize
   * @returns Serialized JSON string
   */
  public serializeMessage(message: MCPMessage | MCPNotification): string {
    try {
      return JSON.stringify(message);
    } catch (error) {
      throw new Error(`Failed to serialize message: ${error}`);
    }
  }

  /**
   * Creates LLMKG-specific tool calls
   */
  public createLLMKGToolCalls = {
    /**
     * Creates a brain visualization tool call
     */
    brainVisualization: (params: LLMKGTools.BrainVisualizationParams): MCPRequest => {
      return this.createToolCallRequest("brain_visualization", params);
    },

    /**
     * Creates a federated metrics tool call
     */
    federatedMetrics: (params: LLMKGTools.FederatedMetricsParams): MCPRequest => {
      return this.createToolCallRequest("federated_metrics", params);
    },

    /**
     * Creates a knowledge graph query tool call
     */
    knowledgeGraphQuery: (params: LLMKGTools.KnowledgeGraphParams): MCPRequest => {
      return this.createToolCallRequest("knowledge_graph_query", params);
    },

    /**
     * Creates an activation patterns tool call
     */
    activationPatterns: (params: {
      region: string;
      timeWindow: number;
      threshold?: number;
    }): MCPRequest => {
      return this.createToolCallRequest("activation_patterns", params);
    },

    /**
     * Creates a connectivity analysis tool call
     */
    connectivityAnalysis: (params: {
      sourceRegion: string;
      targetRegion?: string;
      analysisType: "functional" | "structural" | "effective";
    }): MCPRequest => {
      return this.createToolCallRequest("connectivity_analysis", params);
    },

    /**
     * Creates an SDR analysis tool call
     */
    sdrAnalysis: (params: {
      sdrId: string;
      includeOverlap?: boolean;
      includeSparsity?: boolean;
    }): MCPRequest => {
      return this.createToolCallRequest("sdr_analysis", params);
    }
  };
}

/**
 * Message handler for processing different MCP message types
 */
export class MCPMessageHandler {
  private handlers = new Map<string, (message: MCPMessage) => Promise<any>>();

  /**
   * Registers a handler for a specific method
   * 
   * @param method - Method name to handle
   * @param handler - Handler function
   */
  public registerHandler(
    method: string,
    handler: (message: MCPMessage) => Promise<any>
  ): void {
    this.handlers.set(method, handler);
  }

  /**
   * Handles an incoming message
   * 
   * @param message - Message to handle
   * @returns Response or null for notifications
   */
  public async handleMessage(message: MCPMessage): Promise<MCPResponse | null> {
    const handler = this.handlers.get(message.method);
    
    if (!handler) {
      if ('id' in message) {
        return {
          jsonrpc: "2.0",
          id: message.id,
          error: {
            code: MCPErrorCode.METHOD_NOT_FOUND,
            message: `Method not found: ${message.method}`
          }
        };
      }
      return null; // Ignore unknown notifications
    }

    try {
      const result = await handler(message);
      
      if ('id' in message) {
        return {
          jsonrpc: "2.0",
          id: message.id,
          result
        };
      }
      
      return null; // Notifications don't need responses
    } catch (error) {
      if ('id' in message) {
        return {
          jsonrpc: "2.0",
          id: message.id,
          error: {
            code: MCPErrorCode.INTERNAL_ERROR,
            message: error instanceof Error ? error.message : "Internal error",
            data: error instanceof Error ? error.stack : undefined
          }
        };
      }
      
      throw error; // Re-throw for notifications
    }
  }
}

/**
 * Utility functions for working with MCP protocol messages
 */
export class MCPProtocolUtils {
  /**
   * Checks if a message is a request
   */
  public static isRequest(message: MCPMessage): message is MCPRequest {
    return 'id' in message && 'method' in message && !('result' in message) && !('error' in message);
  }

  /**
   * Checks if a message is a response
   */
  public static isResponse(message: any): message is MCPResponse {
    return 'id' in message && (('result' in message) || ('error' in message));
  }

  /**
   * Checks if a message is a notification
   */
  public static isNotification(message: any): message is MCPNotification {
    return !('id' in message) && 'method' in message && message.jsonrpc === "2.0";
  }

  /**
   * Extracts error information from a response
   */
  public static getErrorFromResponse(response: MCPResponse): MCPError | null {
    return response.error || null;
  }

  /**
   * Creates a standard error object
   */
  public static createError(
    code: MCPErrorCode,
    message: string,
    data?: any
  ): MCPError {
    return { code, message, data };
  }

  /**
   * Validates method names according to MCP specification
   */
  public static isValidMethodName(method: string): boolean {
    // Method names should follow the pattern: category/action or just action
    const methodPattern = /^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)?$/;
    return methodPattern.test(method);
  }

  /**
   * Gets the category from a method name
   */
  public static getMethodCategory(method: string): string | null {
    const parts = method.split('/');
    return parts.length > 1 ? parts[0]! : null;
  }

  /**
   * Gets the action from a method name
   */
  public static getMethodAction(method: string): string {
    const parts = method.split('/');
    return parts[parts.length - 1]!;
  }
}

/**
 * Default client capabilities for LLMKG visualization
 */
export const LLMKG_CLIENT_CAPABILITIES: MCPCapabilities = {
  tools: {
    listChanged: true
  },
  resources: {
    subscribe: true,
    listChanged: true
  },
  prompts: {
    listChanged: true
  },
  experimental: {
    llmkg_brain_visualization: true,
    llmkg_federated_learning: true,
    llmkg_real_time_updates: true
  }
};

/**
 * Default client info for LLMKG visualization
 */
export const LLMKG_CLIENT_INFO: MCPClientInfo = {
  name: "LLMKG Visualization Dashboard",
  version: "1.0.0",
  capabilities: LLMKG_CLIENT_CAPABILITIES
};