/**
 * MCP Telemetry Proxy System
 * 
 * Non-intrusive proxy interceptor for MCP calls that provides comprehensive
 * telemetry collection without modifying LLMKG's core codebase.
 */

import { 
  MCPMessage, 
  MCPRequest, 
  MCPResponse, 
  MCPError, 
  MCPTool,
  MCPEventType,
  isMCPMessage,
  isMCPResponse 
} from '../mcp/types.js';
import { telemetryRecorder } from './recorder.js';
import { telemetryConfig } from './config.js';

export interface MCPCall {
  /** Unique identifier for this call */
  id: string;
  
  /** MCP method being called */
  method: string;
  
  /** Call parameters */
  params?: any;
  
  /** Call timestamp */
  timestamp: number;
  
  /** Source server/endpoint */
  source: string;
  
  /** Call context information */
  context?: {
    toolName?: string;
    operation?: string;
    category?: 'sdr' | 'cognitive' | 'neural' | 'memory' | 'attention' | 'graph';
  };
}

export interface TelemetryData {
  /** Call information */
  call: MCPCall;
  
  /** Response data */
  response?: MCPResponse;
  
  /** Error information if call failed */
  error?: MCPError | Error;
  
  /** Performance metrics */
  performance: {
    duration: number;
    startTime: number;
    endTime: number;
    memoryBefore?: number;
    memoryAfter?: number;
    cpuBefore?: number;
    cpuAfter?: number;
  };
  
  /** Success status */
  success: boolean;
  
  /** Additional metadata */
  metadata?: Record<string, any>;
}

export interface ProxyConfig {
  /** Enable/disable proxy interception */
  enabled: boolean;
  
  /** Sampling rate for call interception (0-1) */
  samplingRate: number;
  
  /** Maximum overhead threshold */
  maxOverheadMs: number;
  
  /** Methods to intercept (empty = all) */
  interceptMethods: string[];
  
  /** Methods to exclude from interception */
  excludeMethods: string[];
  
  /** Enable detailed performance monitoring */
  enablePerformanceMonitoring: boolean;
  
  /** Enable error tracking */
  enableErrorTracking: boolean;
}

const DEFAULT_PROXY_CONFIG: ProxyConfig = {
  enabled: true,
  samplingRate: 1.0,
  maxOverheadMs: 10, // 10ms maximum overhead per call
  interceptMethods: [],
  excludeMethods: ['ping', 'heartbeat'],
  enablePerformanceMonitoring: true,
  enableErrorTracking: true,
};

/**
 * Non-intrusive MCP telemetry proxy
 */
export class MCPTelemetryProxy {
  private config: ProxyConfig;
  private callCounter = 0;
  private activeInstrumentations = new Map<string, ToolInstrumentation>();

  constructor(config: Partial<ProxyConfig> = {}) {
    this.config = { ...DEFAULT_PROXY_CONFIG, ...config };
  }

  /**
   * Intercept an MCP call and collect telemetry data
   */
  async intercept(
    originalCall: () => Promise<MCPResponse>,
    callInfo: Omit<MCPCall, 'id' | 'timestamp'>
  ): Promise<{ response: MCPResponse; telemetryData: TelemetryData }> {
    // Check if telemetry is enabled and should sample this call
    if (!this.shouldIntercept(callInfo.method)) {
      const response = await originalCall();
      return { 
        response, 
        telemetryData: this.createMinimalTelemetryData(callInfo, response) 
      };
    }

    const call: MCPCall = {
      ...callInfo,
      id: this.generateCallId(),
      timestamp: Date.now(),
    };

    let telemetryData: TelemetryData;
    let response: MCPResponse;
    let error: MCPError | Error | undefined;

    const startTime = performance.now();
    const memoryBefore = this.config.enablePerformanceMonitoring ? 
      process.memoryUsage().heapUsed : undefined;

    try {
      // Execute the original call
      response = await originalCall();
      
      const endTime = performance.now();
      const memoryAfter = this.config.enablePerformanceMonitoring ? 
        process.memoryUsage().heapUsed : undefined;

      // Create telemetry data
      telemetryData = {
        call,
        response,
        performance: {
          duration: endTime - startTime,
          startTime,
          endTime,
          memoryBefore,
          memoryAfter,
        },
        success: !response.error,
        metadata: this.extractMetadata(call, response),
      };

      if (response.error) {
        telemetryData.error = response.error;
      }

    } catch (err) {
      const endTime = performance.now();
      const memoryAfter = this.config.enablePerformanceMonitoring ? 
        process.memoryUsage().heapUsed : undefined;

      error = err as Error;
      
      // Create error response
      response = {
        id: call.id,
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: error.message,
          data: { 
            name: error.name,
            stack: error.stack,
          },
        },
      };

      telemetryData = {
        call,
        response,
        error,
        performance: {
          duration: endTime - startTime,
          startTime,
          endTime,
          memoryBefore,
          memoryAfter,
        },
        success: false,
        metadata: this.extractMetadata(call, response),
      };
    }

    // Record telemetry data
    this.recordTelemetryData(telemetryData);

    return { response, telemetryData };
  }

  /**
   * Wrap a tool function with telemetry collection
   */
  wrapTool(toolName: string, originalFunction: Function, category?: MCPCall['context']['category']): Function {
    if (!telemetryConfig.isEnabled() || !this.config.enabled) {
      return originalFunction;
    }

    return async (...args: any[]) => {
      const callInfo: Omit<MCPCall, 'id' | 'timestamp'> = {
        method: 'tools/call',
        params: { name: toolName, arguments: args },
        source: 'tool_wrapper',
        context: {
          toolName,
          operation: 'tool_execution',
          category: category || 'system',
        },
      };

      const originalCall = async (): Promise<MCPResponse> => {
        try {
          const result = await originalFunction(...args);
          return {
            id: this.generateCallId(),
            jsonrpc: "2.0",
            result,
          };
        } catch (error) {
          throw error;
        }
      };

      const { response } = await this.intercept(originalCall, callInfo);
      
      if (response.error) {
        throw new Error(response.error.message);
      }
      
      return response.result;
    };
  }

  /**
   * Create tool-specific instrumentation
   */
  createToolInstrumentation(toolName: string, tool: MCPTool): ToolInstrumentation {
    const instrumentation = new ToolInstrumentation(toolName, tool, this);
    this.activeInstrumentations.set(toolName, instrumentation);
    return instrumentation;
  }

  /**
   * Remove tool instrumentation
   */
  removeToolInstrumentation(toolName: string): void {
    const instrumentation = this.activeInstrumentations.get(toolName);
    if (instrumentation) {
      instrumentation.cleanup();
      this.activeInstrumentations.delete(toolName);
    }
  }

  /**
   * Get instrumentation for a specific tool
   */
  getToolInstrumentation(toolName: string): ToolInstrumentation | undefined {
    return this.activeInstrumentations.get(toolName);
  }

  /**
   * Intercept MCP client connection methods
   */
  wrapMCPClient(client: any): any {
    if (!telemetryConfig.isEnabled() || !this.config.enabled) {
      return client;
    }

    const proxy = this;

    return new Proxy(client, {
      get(target, prop, receiver) {
        const originalValue = Reflect.get(target, prop, receiver);

        // Wrap methods that send MCP messages
        if (typeof originalValue === 'function' && proxy.shouldWrapMethod(String(prop))) {
          return function (...args: any[]) {
            return proxy.wrapClientMethod(String(prop), originalValue, target, args);
          };
        }

        return originalValue;
      },
    });
  }

  /**
   * Check if a method should be wrapped
   */
  private shouldWrapMethod(methodName: string): boolean {
    const wrapMethods = [
      'callTool',
      'listTools',
      'sendRequest',
      'sendNotification',
      'connect',
      'disconnect',
    ];

    return wrapMethods.includes(methodName);
  }

  /**
   * Wrap a client method call
   */
  private async wrapClientMethod(
    methodName: string,
    originalMethod: Function,
    target: any,
    args: any[]
  ): Promise<any> {
    const callInfo: Omit<MCPCall, 'id' | 'timestamp'> = {
      method: methodName,
      params: args.length > 0 ? args[0] : undefined,
      source: 'mcp_client',
      context: {
        operation: methodName,
        category: this.inferCategoryFromMethod(methodName),
      },
    };

    const originalCall = async (): Promise<MCPResponse> => {
      try {
        const result = await originalMethod.apply(target, args);
        return {
          id: this.generateCallId(),
          jsonrpc: "2.0",
          result,
        };
      } catch (error) {
        throw error;
      }
    };

    const { response } = await this.intercept(originalCall, callInfo);
    
    if (response.error) {
      throw new Error(response.error.message);
    }
    
    return response.result;
  }

  /**
   * Infer category from method name
   */
  private inferCategoryFromMethod(methodName: string): MCPCall['context']['category'] {
    const categoryMap: Record<string, MCPCall['context']['category']> = {
      'brain_visualization': 'cognitive',
      'sdr_query': 'sdr',
      'neural_activity': 'neural',
      'memory_access': 'memory',
      'attention_focus': 'attention',
      'graph_query': 'graph',
    };

    for (const [key, category] of Object.entries(categoryMap)) {
      if (methodName.toLowerCase().includes(key)) {
        return category;
      }
    }

    return 'system';
  }

  /**
   * Check if a call should be intercepted
   */
  private shouldIntercept(method: string): boolean {
    if (!this.config.enabled || !telemetryConfig.isEnabled()) {
      return false;
    }

    // Check exclude list
    if (this.config.excludeMethods.includes(method)) {
      return false;
    }

    // Check include list (if specified)
    if (this.config.interceptMethods.length > 0 && 
        !this.config.interceptMethods.includes(method)) {
      return false;
    }

    // Apply sampling rate
    if (Math.random() > this.config.samplingRate) {
      return false;
    }

    return true;
  }

  /**
   * Create minimal telemetry data for non-intercepted calls
   */
  private createMinimalTelemetryData(
    callInfo: Omit<MCPCall, 'id' | 'timestamp'>,
    response: MCPResponse
  ): TelemetryData {
    return {
      call: {
        ...callInfo,
        id: this.generateCallId(),
        timestamp: Date.now(),
      },
      response,
      performance: {
        duration: 0,
        startTime: 0,
        endTime: 0,
      },
      success: !response.error,
    };
  }

  /**
   * Record telemetry data
   */
  private recordTelemetryData(data: TelemetryData): void {
    // Record performance metrics
    if (data.performance.duration > 0) {
      telemetryRecorder.recordMetric(
        data.call.context?.category || 'system',
        `mcp.call.duration.${data.call.method}`,
        data.performance.duration,
        {
          method: data.call.method,
          source: data.call.source,
          success: data.success.toString(),
        }
      );

      if (data.performance.memoryBefore && data.performance.memoryAfter) {
        const memoryDelta = data.performance.memoryAfter - data.performance.memoryBefore;
        telemetryRecorder.recordMetric(
          data.call.context?.category || 'system',
          `mcp.call.memory.${data.call.method}`,
          memoryDelta,
          {
            method: data.call.method,
            source: data.call.source,
          }
        );
      }
    }

    // Record trace data
    if (telemetryConfig.getConfig().collection.enableTraces) {
      telemetryRecorder.recordTrace(
        data.call.context?.category || 'system',
        `mcp.call.${data.call.method}`,
        {
          callId: data.call.id,
          method: data.call.method,
          params: data.call.params,
          result: data.response?.result,
          error: data.error,
          source: data.call.source,
        },
        {
          duration: data.performance.duration,
          memoryUsage: data.performance.memoryAfter ? 
            data.performance.memoryAfter - (data.performance.memoryBefore || 0) : undefined,
        },
        {
          method: data.call.method,
          source: data.call.source,
          success: data.success.toString(),
          toolName: data.call.context?.toolName || 'unknown',
        }
      );
    }

    // Record errors
    if (!data.success && this.config.enableErrorTracking) {
      telemetryRecorder.recordLog(
        data.call.context?.category || 'system',
        `mcp.call.error.${data.call.method}`,
        data.error?.message || 'Unknown error',
        'error',
        {
          callId: data.call.id,
          method: data.call.method,
          params: data.call.params,
          error: data.error,
          source: data.call.source,
        },
        {
          method: data.call.method,
          source: data.call.source,
          errorType: data.error?.name || 'unknown',
        }
      );
    }
  }

  /**
   * Extract metadata from call and response
   */
  private extractMetadata(call: MCPCall, response: MCPResponse): Record<string, any> {
    const metadata: Record<string, any> = {
      method: call.method,
      source: call.source,
      hasParams: !!call.params,
      hasResult: !!response.result,
      hasError: !!response.error,
    };

    if (call.context) {
      metadata.context = call.context;
    }

    // Extract LLMKG-specific metadata
    if (call.context?.toolName) {
      metadata.toolName = call.context.toolName;
      metadata.toolCategory = call.context.category;
    }

    return metadata;
  }

  /**
   * Generate unique call ID
   */
  private generateCallId(): string {
    return `call_${Date.now()}_${++this.callCounter}`;
  }

  /**
   * Update proxy configuration
   */
  updateConfig(newConfig: Partial<ProxyConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current proxy configuration
   */
  getConfig(): ProxyConfig {
    return { ...this.config };
  }

  /**
   * Get proxy statistics
   */
  getStats(): {
    totalCalls: number;
    activeInstrumentations: number;
  } {
    return {
      totalCalls: this.callCounter,
      activeInstrumentations: this.activeInstrumentations.size,
    };
  }
}

/**
 * Tool-specific instrumentation wrapper
 */
export class ToolInstrumentation {
  private toolName: string;
  private tool: MCPTool;
  private proxy: MCPTelemetryProxy;
  private callStats = new Map<string, { count: number; totalDuration: number; errors: number }>();

  constructor(toolName: string, tool: MCPTool, proxy: MCPTelemetryProxy) {
    this.toolName = toolName;
    this.tool = tool;
    this.proxy = proxy;
  }

  /**
   * Instrument a tool call with LLMKG-specific monitoring
   */
  instrument<T extends Function>(originalFunction: T, category?: MCPCall['context']['category']): T {
    const instrumentation = this;
    
    return ((...args: any[]) => {
      return instrumentation.proxy.wrapTool(
        instrumentation.toolName, 
        originalFunction, 
        category
      )(...args);
    }) as unknown as T;
  }

  /**
   * Get tool statistics
   */
  getStats(): Record<string, { count: number; totalDuration: number; errors: number; avgDuration: number }> {
    const stats: Record<string, any> = {};
    
    for (const [operation, data] of this.callStats) {
      stats[operation] = {
        ...data,
        avgDuration: data.count > 0 ? data.totalDuration / data.count : 0,
      };
    }
    
    return stats;
  }

  /**
   * Clean up instrumentation
   */
  cleanup(): void {
    this.callStats.clear();
  }

  /**
   * Record call statistics
   */
  recordCall(operation: string, duration: number, success: boolean): void {
    const stats = this.callStats.get(operation) || { count: 0, totalDuration: 0, errors: 0 };
    
    stats.count++;
    stats.totalDuration += duration;
    
    if (!success) {
      stats.errors++;
    }
    
    this.callStats.set(operation, stats);
  }
}

// Global proxy instance
export const mcpTelemetryProxy = new MCPTelemetryProxy();