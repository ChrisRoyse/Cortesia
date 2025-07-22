# Phase 1: MCP Integration & Data Collection

## Overview

Phase 1 focuses on building a comprehensive MCP (Model Context Protocol) client implementation in JavaScript/TypeScript that can interface with the LLMKG system, collect telemetry data from MCP tool executions, and stream real-time data to the visualization dashboard without modifying the core Rust codebase.

## Table of Contents

1. [Objectives](#objectives)
2. [MCP Protocol Understanding](#mcp-protocol-understanding)
3. [JavaScript/TypeScript MCP Client Implementation](#javascripttypescript-mcp-client-implementation)
4. [Data Collection Agents](#data-collection-agents)
5. [Telemetry Injection Strategies](#telemetry-injection-strategies)
6. [WebSocket Communication](#websocket-communication)
7. [Testing Procedures](#testing-procedures)
8. [Deliverables Checklist](#deliverables-checklist)

## Objectives

### Primary Goals

1. **MCP Client Implementation**
   - Build a fully functional MCP client in JavaScript/TypeScript
   - Support all MCP protocol features including tool discovery, execution, and response handling
   - Implement proper error handling and retry mechanisms

2. **Data Collection Infrastructure**
   - Create agents that intercept and collect data from LLMKG's MCP tools
   - Build a telemetry pipeline that captures tool execution metrics
   - Implement data transformation layers for visualization

3. **Real-time Streaming**
   - Establish WebSocket connections for live data streaming
   - Create event-based architecture for tool execution monitoring
   - Build buffering and batching mechanisms for efficient data transfer

4. **Non-intrusive Integration**
   - Implement telemetry without modifying LLMKG's Rust core
   - Use proxy patterns and middleware for data interception
   - Maintain system performance and reliability

## MCP Protocol Understanding

### Protocol Specification

```typescript
// MCP Protocol Types
interface MCPMessage {
  jsonrpc: "2.0";
  id?: string | number;
  method?: string;
  params?: any;
  result?: any;
  error?: MCPError;
}

interface MCPError {
  code: number;
  message: string;
  data?: any;
}

interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, any>;
    required?: string[];
  };
}

interface MCPToolExecution {
  tool: string;
  arguments: Record<string, any>;
  timestamp: number;
  duration?: number;
  result?: any;
  error?: MCPError;
}
```

### MCP Communication Flow

1. **Connection Establishment**
   - Client connects to MCP server via stdio or HTTP
   - Performs capability negotiation
   - Discovers available tools

2. **Tool Discovery**
   - Client sends `tools/list` request
   - Server responds with available tools and schemas
   - Client caches tool metadata

3. **Tool Execution**
   - Client sends `tools/call` request with tool name and arguments
   - Server executes tool and returns result
   - Client handles response or error

## JavaScript/TypeScript MCP Client Implementation

### Core MCP Client

```typescript
// mcp-client.ts
import { EventEmitter } from 'events';
import { spawn, ChildProcess } from 'child_process';
import { v4 as uuidv4 } from 'uuid';

export class MCPClient extends EventEmitter {
  private process: ChildProcess | null = null;
  private pendingRequests: Map<string, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
    timestamp: number;
  }> = new Map();
  private tools: Map<string, MCPTool> = new Map();
  private buffer: string = '';

  constructor(private config: MCPClientConfig) {
    super();
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const args = this.config.args || [];
      this.process = spawn(this.config.command, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, ...this.config.env }
      });

      this.process.stdout?.on('data', (data) => this.handleData(data));
      this.process.stderr?.on('data', (data) => this.handleError(data));
      
      this.process.on('exit', (code) => {
        this.emit('disconnected', code);
        this.cleanup();
      });

      // Initialize connection
      this.sendRequest('initialize', {
        protocolVersion: '1.0',
        capabilities: {
          tools: true,
          resources: true
        }
      }).then(() => {
        this.emit('connected');
        resolve();
      }).catch(reject);
    });
  }

  private handleData(data: Buffer): void {
    this.buffer += data.toString();
    
    // Parse complete JSON-RPC messages
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line) as MCPMessage;
          this.handleMessage(message);
        } catch (error) {
          this.emit('error', new Error(`Failed to parse message: ${line}`));
        }
      }
    }
  }

  private handleMessage(message: MCPMessage): void {
    // Emit raw message for telemetry
    this.emit('message', message);

    if (message.id && this.pendingRequests.has(String(message.id))) {
      const request = this.pendingRequests.get(String(message.id))!;
      this.pendingRequests.delete(String(message.id));
      
      const duration = Date.now() - request.timestamp;
      
      if (message.error) {
        request.reject(message.error);
        this.emit('tool:error', {
          id: message.id,
          error: message.error,
          duration
        });
      } else {
        request.resolve(message.result);
        this.emit('tool:success', {
          id: message.id,
          result: message.result,
          duration
        });
      }
    } else if (message.method) {
      // Handle server-initiated messages
      this.handleNotification(message);
    }
  }

  private handleNotification(message: MCPMessage): void {
    this.emit('notification', message);
  }

  private handleError(data: Buffer): void {
    this.emit('error', new Error(`MCP stderr: ${data.toString()}`));
  }

  async sendRequest(method: string, params?: any): Promise<any> {
    const id = uuidv4();
    const message: MCPMessage = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve,
        reject,
        timestamp: Date.now()
      });

      const messageStr = JSON.stringify(message) + '\n';
      this.process?.stdin?.write(messageStr, (error) => {
        if (error) {
          this.pendingRequests.delete(id);
          reject(error);
        }
      });

      // Timeout handling
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request timeout: ${method}`));
        }
      }, this.config.timeout || 30000);
    });
  }

  async discoverTools(): Promise<MCPTool[]> {
    const response = await this.sendRequest('tools/list');
    const tools = response.tools || [];
    
    // Cache tools
    this.tools.clear();
    for (const tool of tools) {
      this.tools.set(tool.name, tool);
    }
    
    this.emit('tools:discovered', tools);
    return tools;
  }

  async executeTool(name: string, args: Record<string, any>): Promise<any> {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new Error(`Unknown tool: ${name}`);
    }

    const startTime = Date.now();
    this.emit('tool:start', { tool: name, arguments: args, timestamp: startTime });

    try {
      const result = await this.sendRequest('tools/call', {
        name,
        arguments: args
      });

      const execution: MCPToolExecution = {
        tool: name,
        arguments: args,
        timestamp: startTime,
        duration: Date.now() - startTime,
        result
      };

      this.emit('tool:complete', execution);
      return result;
    } catch (error) {
      const execution: MCPToolExecution = {
        tool: name,
        arguments: args,
        timestamp: startTime,
        duration: Date.now() - startTime,
        error: error as MCPError
      };

      this.emit('tool:complete', execution);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.process) {
      this.process.kill();
      this.cleanup();
    }
  }

  private cleanup(): void {
    this.process = null;
    this.pendingRequests.clear();
    this.tools.clear();
    this.buffer = '';
  }
}

interface MCPClientConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  timeout?: number;
}
```

### Enhanced MCP Client with Telemetry

```typescript
// telemetry-mcp-client.ts
import { MCPClient } from './mcp-client';
import { TelemetryCollector } from './telemetry-collector';

export class TelemetryMCPClient extends MCPClient {
  private telemetry: TelemetryCollector;
  private sessionId: string;

  constructor(config: MCPClientConfig, telemetryConfig: TelemetryConfig) {
    super(config);
    this.sessionId = uuidv4();
    this.telemetry = new TelemetryCollector(telemetryConfig);
    this.setupTelemetryHooks();
  }

  private setupTelemetryHooks(): void {
    // Connection telemetry
    this.on('connected', () => {
      this.telemetry.recordEvent({
        type: 'connection',
        status: 'connected',
        sessionId: this.sessionId,
        timestamp: Date.now()
      });
    });

    this.on('disconnected', (code) => {
      this.telemetry.recordEvent({
        type: 'connection',
        status: 'disconnected',
        sessionId: this.sessionId,
        exitCode: code,
        timestamp: Date.now()
      });
    });

    // Tool execution telemetry
    this.on('tool:start', (data) => {
      this.telemetry.recordToolStart({
        sessionId: this.sessionId,
        ...data
      });
    });

    this.on('tool:complete', (execution) => {
      this.telemetry.recordToolComplete({
        sessionId: this.sessionId,
        ...execution
      });
    });

    // Message telemetry
    this.on('message', (message) => {
      this.telemetry.recordMessage({
        sessionId: this.sessionId,
        message,
        timestamp: Date.now()
      });
    });

    // Error telemetry
    this.on('error', (error) => {
      this.telemetry.recordError({
        sessionId: this.sessionId,
        error: error.message,
        stack: error.stack,
        timestamp: Date.now()
      });
    });
  }

  async executeTool(name: string, args: Record<string, any>): Promise<any> {
    // Add context information
    const context = {
      sessionId: this.sessionId,
      tool: name,
      timestamp: Date.now(),
      metadata: this.extractMetadata(name, args)
    };

    this.telemetry.beginTransaction(context);
    
    try {
      const result = await super.executeTool(name, args);
      this.telemetry.endTransaction(context, { success: true, result });
      return result;
    } catch (error) {
      this.telemetry.endTransaction(context, { success: false, error });
      throw error;
    }
  }

  private extractMetadata(tool: string, args: Record<string, any>): any {
    // Extract relevant metadata based on tool type
    const metadata: any = {};

    // LLMKG-specific tool metadata extraction
    switch (tool) {
      case 'knowledge_graph_query':
        metadata.queryType = args.query_type;
        metadata.entityCount = args.entities?.length || 0;
        break;
      
      case 'neural_pattern_detection':
        metadata.patternType = args.pattern_type;
        metadata.threshold = args.threshold;
        break;
      
      case 'cognitive_reasoning':
        metadata.reasoningType = args.type;
        metadata.depth = args.depth;
        break;
      
      // Add more LLMKG tool-specific metadata extraction
    }

    return metadata;
  }

  getTelemetryStats(): TelemetryStats {
    return this.telemetry.getStats();
  }
}

interface TelemetryConfig {
  endpoint: string;
  batchSize: number;
  flushInterval: number;
}

interface TelemetryStats {
  totalTools: number;
  successfulTools: number;
  failedTools: number;
  avgDuration: number;
  toolBreakdown: Record<string, ToolStats>;
}

interface ToolStats {
  count: number;
  successRate: number;
  avgDuration: number;
  errors: number;
}
```

## Data Collection Agents

### Telemetry Collector

```typescript
// telemetry-collector.ts
import { EventEmitter } from 'events';
import axios from 'axios';

export class TelemetryCollector extends EventEmitter {
  private buffer: TelemetryEvent[] = [];
  private stats: TelemetryStats;
  private flushTimer: NodeJS.Timer | null = null;

  constructor(private config: TelemetryConfig) {
    super();
    this.stats = this.initializeStats();
    this.startFlushTimer();
  }

  private initializeStats(): TelemetryStats {
    return {
      totalTools: 0,
      successfulTools: 0,
      failedTools: 0,
      avgDuration: 0,
      toolBreakdown: {}
    };
  }

  recordEvent(event: TelemetryEvent): void {
    this.buffer.push(event);
    this.emit('event', event);

    if (this.buffer.length >= this.config.batchSize) {
      this.flush();
    }
  }

  recordToolStart(data: ToolStartEvent): void {
    this.recordEvent({
      type: 'tool:start',
      ...data
    });
  }

  recordToolComplete(execution: MCPToolExecution & { sessionId: string }): void {
    this.stats.totalTools++;
    
    if (execution.error) {
      this.stats.failedTools++;
    } else {
      this.stats.successfulTools++;
    }

    // Update tool-specific stats
    if (!this.stats.toolBreakdown[execution.tool]) {
      this.stats.toolBreakdown[execution.tool] = {
        count: 0,
        successRate: 0,
        avgDuration: 0,
        errors: 0
      };
    }

    const toolStats = this.stats.toolBreakdown[execution.tool];
    toolStats.count++;
    
    if (execution.error) {
      toolStats.errors++;
    }
    
    if (execution.duration) {
      toolStats.avgDuration = 
        (toolStats.avgDuration * (toolStats.count - 1) + execution.duration) / 
        toolStats.count;
    }
    
    toolStats.successRate = 
      (toolStats.count - toolStats.errors) / toolStats.count;

    // Update overall average duration
    if (execution.duration) {
      this.stats.avgDuration = 
        (this.stats.avgDuration * (this.stats.totalTools - 1) + execution.duration) / 
        this.stats.totalTools;
    }

    this.recordEvent({
      type: 'tool:complete',
      ...execution
    });

    this.emit('stats:updated', this.stats);
  }

  recordMessage(data: MessageEvent): void {
    this.recordEvent({
      type: 'message',
      ...data
    });
  }

  recordError(data: ErrorEvent): void {
    this.recordEvent({
      type: 'error',
      ...data
    });
  }

  beginTransaction(context: TransactionContext): void {
    this.recordEvent({
      type: 'transaction:begin',
      ...context
    });
  }

  endTransaction(context: TransactionContext, result: TransactionResult): void {
    this.recordEvent({
      type: 'transaction:end',
      ...context,
      result
    });
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      if (this.buffer.length > 0) {
        this.flush();
      }
    }, this.config.flushInterval);
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const events = [...this.buffer];
    this.buffer = [];

    try {
      await axios.post(this.config.endpoint, {
        events,
        timestamp: Date.now()
      });
      
      this.emit('flush:success', events.length);
    } catch (error) {
      // Re-add events to buffer on failure
      this.buffer.unshift(...events);
      this.emit('flush:error', error);
    }
  }

  getStats(): TelemetryStats {
    return { ...this.stats };
  }

  stop(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush();
  }
}

interface TelemetryEvent {
  type: string;
  timestamp: number;
  [key: string]: any;
}

interface ToolStartEvent {
  tool: string;
  arguments: Record<string, any>;
  timestamp: number;
  sessionId: string;
}

interface MessageEvent {
  message: MCPMessage;
  timestamp: number;
  sessionId: string;
}

interface ErrorEvent {
  error: string;
  stack?: string;
  timestamp: number;
  sessionId: string;
}

interface TransactionContext {
  sessionId: string;
  tool: string;
  timestamp: number;
  metadata: any;
}

interface TransactionResult {
  success: boolean;
  result?: any;
  error?: any;
}
```

### LLMKG-Specific Data Collectors

```typescript
// llmkg-collectors.ts
import { TelemetryCollector } from './telemetry-collector';

export class LLMKGDataCollector {
  private collectors: Map<string, SpecializedCollector> = new Map();

  constructor(private telemetry: TelemetryCollector) {
    this.initializeCollectors();
  }

  private initializeCollectors(): void {
    // Knowledge Graph Collector
    this.collectors.set('knowledge_graph', new KnowledgeGraphCollector(this.telemetry));
    
    // Neural Pattern Collector
    this.collectors.set('neural_pattern', new NeuralPatternCollector(this.telemetry));
    
    // Cognitive Reasoning Collector
    this.collectors.set('cognitive', new CognitiveReasoningCollector(this.telemetry));
    
    // Memory System Collector
    this.collectors.set('memory', new MemorySystemCollector(this.telemetry));
  }

  processToolExecution(execution: MCPToolExecution): void {
    // Route to appropriate collector based on tool name
    const collectorKey = this.getCollectorKey(execution.tool);
    const collector = this.collectors.get(collectorKey);
    
    if (collector) {
      collector.process(execution);
    }
    
    // Always process with base telemetry
    this.processBaseTelemetry(execution);
  }

  private getCollectorKey(toolName: string): string {
    if (toolName.includes('knowledge') || toolName.includes('graph')) {
      return 'knowledge_graph';
    } else if (toolName.includes('neural') || toolName.includes('pattern')) {
      return 'neural_pattern';
    } else if (toolName.includes('cognitive') || toolName.includes('reasoning')) {
      return 'cognitive';
    } else if (toolName.includes('memory')) {
      return 'memory';
    }
    return 'default';
  }

  private processBaseTelemetry(execution: MCPToolExecution): void {
    // Extract common metrics
    const metrics = {
      tool: execution.tool,
      duration: execution.duration,
      success: !execution.error,
      timestamp: execution.timestamp,
      
      // LLMKG-specific metrics
      entityCount: this.extractEntityCount(execution),
      tripleCount: this.extractTripleCount(execution),
      activationStrength: this.extractActivationStrength(execution),
      confidenceScore: this.extractConfidenceScore(execution)
    };

    this.telemetry.recordEvent({
      type: 'llmkg:metrics',
      ...metrics
    });
  }

  private extractEntityCount(execution: MCPToolExecution): number {
    if (execution.result?.entities) {
      return execution.result.entities.length;
    }
    return 0;
  }

  private extractTripleCount(execution: MCPToolExecution): number {
    if (execution.result?.triples) {
      return execution.result.triples.length;
    }
    return 0;
  }

  private extractActivationStrength(execution: MCPToolExecution): number {
    return execution.result?.activation_strength || 0;
  }

  private extractConfidenceScore(execution: MCPToolExecution): number {
    return execution.result?.confidence || 0;
  }
}

abstract class SpecializedCollector {
  constructor(protected telemetry: TelemetryCollector) {}
  
  abstract process(execution: MCPToolExecution): void;
}

class KnowledgeGraphCollector extends SpecializedCollector {
  process(execution: MCPToolExecution): void {
    const graphMetrics = {
      nodeCount: execution.result?.graph?.nodes?.length || 0,
      edgeCount: execution.result?.graph?.edges?.length || 0,
      connectedComponents: execution.result?.graph?.components || 0,
      averageDegree: this.calculateAverageDegree(execution.result?.graph),
      queryType: execution.arguments?.query_type,
      queryDepth: execution.arguments?.depth || 1
    };

    this.telemetry.recordEvent({
      type: 'llmkg:knowledge_graph',
      tool: execution.tool,
      metrics: graphMetrics,
      timestamp: execution.timestamp
    });
  }

  private calculateAverageDegree(graph: any): number {
    if (!graph?.nodes || !graph?.edges) return 0;
    return (graph.edges.length * 2) / graph.nodes.length;
  }
}

class NeuralPatternCollector extends SpecializedCollector {
  process(execution: MCPToolExecution): void {
    const patternMetrics = {
      patternType: execution.arguments?.pattern_type,
      patternCount: execution.result?.patterns?.length || 0,
      activationLevels: this.extractActivationLevels(execution.result),
      sdrDensity: execution.result?.sdr_density || 0,
      neuralComplexity: execution.result?.complexity || 0
    };

    this.telemetry.recordEvent({
      type: 'llmkg:neural_pattern',
      tool: execution.tool,
      metrics: patternMetrics,
      timestamp: execution.timestamp
    });
  }

  private extractActivationLevels(result: any): number[] {
    return result?.activation_levels || [];
  }
}

class CognitiveReasoningCollector extends SpecializedCollector {
  process(execution: MCPToolExecution): void {
    const reasoningMetrics = {
      reasoningType: execution.arguments?.type,
      reasoningDepth: execution.arguments?.depth || 1,
      inferenceSteps: execution.result?.steps?.length || 0,
      confidenceScores: this.extractConfidenceScores(execution.result),
      branchingFactor: execution.result?.branching_factor || 0
    };

    this.telemetry.recordEvent({
      type: 'llmkg:cognitive_reasoning',
      tool: execution.tool,
      metrics: reasoningMetrics,
      timestamp: execution.timestamp
    });
  }

  private extractConfidenceScores(result: any): number[] {
    return result?.confidence_scores || [];
  }
}

class MemorySystemCollector extends SpecializedCollector {
  process(execution: MCPToolExecution): void {
    const memoryMetrics = {
      memoryType: execution.arguments?.memory_type,
      memorySize: execution.result?.size || 0,
      hitRate: execution.result?.hit_rate || 0,
      evictionCount: execution.result?.evictions || 0,
      consolidationLevel: execution.result?.consolidation || 0
    };

    this.telemetry.recordEvent({
      type: 'llmkg:memory_system',
      tool: execution.tool,
      metrics: memoryMetrics,
      timestamp: execution.timestamp
    });
  }
}
```

## Telemetry Injection Strategies

### Proxy Pattern Implementation

```typescript
// mcp-proxy.ts
import { MCPClient } from './mcp-client';
import { LLMKGDataCollector } from './llmkg-collectors';
import { WebSocketRelay } from './websocket-relay';

export class MCPProxy {
  private originalClients: Map<string, MCPClient> = new Map();
  private proxiedClients: Map<string, ProxiedMCPClient> = new Map();
  private collector: LLMKGDataCollector;
  private relay: WebSocketRelay;

  constructor(
    private telemetryConfig: TelemetryConfig,
    private wsConfig: WebSocketConfig
  ) {
    const telemetry = new TelemetryCollector(telemetryConfig);
    this.collector = new LLMKGDataCollector(telemetry);
    this.relay = new WebSocketRelay(wsConfig);
  }

  createProxy(clientId: string, originalClient: MCPClient): ProxiedMCPClient {
    this.originalClients.set(clientId, originalClient);
    
    const proxied = new ProxiedMCPClient(
      originalClient,
      this.collector,
      this.relay,
      clientId
    );
    
    this.proxiedClients.set(clientId, proxied);
    return proxied;
  }

  getStats(): Record<string, any> {
    const stats: Record<string, any> = {};
    
    for (const [id, client] of this.proxiedClients) {
      stats[id] = client.getStats();
    }
    
    return stats;
  }
}

class ProxiedMCPClient {
  private stats = {
    totalCalls: 0,
    interceptedCalls: 0,
    errors: 0
  };

  constructor(
    private originalClient: MCPClient,
    private collector: LLMKGDataCollector,
    private relay: WebSocketRelay,
    private clientId: string
  ) {
    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Intercept executeTool
    const originalExecuteTool = this.originalClient.executeTool.bind(this.originalClient);
    
    this.originalClient.executeTool = async (name: string, args: Record<string, any>) => {
      this.stats.totalCalls++;
      
      const execution: Partial<MCPToolExecution> = {
        tool: name,
        arguments: args,
        timestamp: Date.now()
      };

      // Notify relay of tool start
      this.relay.broadcast({
        type: 'tool:start',
        clientId: this.clientId,
        execution
      });

      try {
        const result = await originalExecuteTool(name, args);
        execution.result = result;
        execution.duration = Date.now() - execution.timestamp;
        
        // Process with collector
        this.collector.processToolExecution(execution as MCPToolExecution);
        
        // Broadcast completion
        this.relay.broadcast({
          type: 'tool:complete',
          clientId: this.clientId,
          execution
        });
        
        this.stats.interceptedCalls++;
        return result;
      } catch (error) {
        execution.error = error as MCPError;
        execution.duration = Date.now() - execution.timestamp;
        
        // Process error
        this.collector.processToolExecution(execution as MCPToolExecution);
        
        // Broadcast error
        this.relay.broadcast({
          type: 'tool:error',
          clientId: this.clientId,
          execution,
          error
        });
        
        this.stats.errors++;
        throw error;
      }
    };

    // Intercept other methods as needed
    this.interceptMethod('connect');
    this.interceptMethod('disconnect');
    this.interceptMethod('discoverTools');
  }

  private interceptMethod(methodName: string): void {
    const original = (this.originalClient as any)[methodName];
    if (typeof original === 'function') {
      (this.originalClient as any)[methodName] = async (...args: any[]) => {
        this.relay.broadcast({
          type: `method:${methodName}`,
          clientId: this.clientId,
          args,
          timestamp: Date.now()
        });
        
        return original.apply(this.originalClient, args);
      };
    }
  }

  getStats(): any {
    return { ...this.stats };
  }
}
```

### Environment Variable Injection

```typescript
// env-injector.ts
export class TelemetryEnvInjector {
  private originalEnv: Record<string, string> = {};

  inject(config: TelemetryConfig): void {
    // Save original environment
    this.originalEnv = { ...process.env };

    // Inject telemetry configuration
    process.env.LLMKG_TELEMETRY_ENABLED = 'true';
    process.env.LLMKG_TELEMETRY_ENDPOINT = config.endpoint;
    process.env.LLMKG_TELEMETRY_BATCH_SIZE = String(config.batchSize);
    process.env.LLMKG_TELEMETRY_FLUSH_INTERVAL = String(config.flushInterval);

    // Add proxy configuration if needed
    if (config.proxy) {
      process.env.LLMKG_MCP_PROXY = config.proxy.address;
      process.env.LLMKG_MCP_PROXY_PORT = String(config.proxy.port);
    }
  }

  restore(): void {
    // Restore original environment
    Object.keys(process.env).forEach(key => {
      if (key.startsWith('LLMKG_TELEMETRY_') || key.startsWith('LLMKG_MCP_')) {
        delete process.env[key];
      }
    });

    Object.assign(process.env, this.originalEnv);
  }
}
```

## WebSocket Communication

### WebSocket Relay Server

```typescript
// websocket-relay.ts
import { WebSocket, WebSocketServer } from 'ws';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

export class WebSocketRelay extends EventEmitter {
  private wss: WebSocketServer;
  private clients: Map<string, WebSocketClient> = new Map();
  private messageBuffer: CircularBuffer<RelayMessage>;

  constructor(private config: WebSocketConfig) {
    super();
    this.messageBuffer = new CircularBuffer(config.bufferSize || 1000);
    this.createServer();
  }

  private createServer(): void {
    this.wss = new WebSocketServer({
      port: this.config.port,
      perMessageDeflate: {
        zlibDeflateOptions: {
          chunkSize: 1024,
          memLevel: 7,
          level: 3
        },
        zlibInflateOptions: {
          chunkSize: 10 * 1024
        },
        clientNoContextTakeover: true,
        serverNoContextTakeover: true,
        serverMaxWindowBits: 10,
        concurrencyLimit: 10,
        threshold: 1024
      }
    });

    this.wss.on('connection', (ws, req) => {
      const clientId = uuidv4();
      const client = new WebSocketClient(clientId, ws, req);
      
      this.clients.set(clientId, client);
      this.emit('client:connected', client);

      // Send buffered messages to new client
      this.sendBufferedMessages(client);

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleClientMessage(client, message);
        } catch (error) {
          client.send({
            type: 'error',
            error: 'Invalid message format'
          });
        }
      });

      ws.on('close', () => {
        this.clients.delete(clientId);
        this.emit('client:disconnected', client);
      });

      ws.on('error', (error) => {
        this.emit('client:error', { client, error });
      });

      // Send initial handshake
      client.send({
        type: 'handshake',
        clientId,
        timestamp: Date.now()
      });
    });

    this.wss.on('error', (error) => {
      this.emit('server:error', error);
    });
  }

  broadcast(message: RelayMessage): void {
    const timestampedMessage = {
      ...message,
      timestamp: message.timestamp || Date.now(),
      id: uuidv4()
    };

    // Add to buffer
    this.messageBuffer.add(timestampedMessage);

    // Broadcast to all connected clients
    for (const client of this.clients.values()) {
      if (client.isAlive()) {
        client.send(timestampedMessage);
      }
    }

    this.emit('message:broadcast', timestampedMessage);
  }

  private handleClientMessage(client: WebSocketClient, message: any): void {
    switch (message.type) {
      case 'ping':
        client.send({ type: 'pong', timestamp: Date.now() });
        break;

      case 'subscribe':
        this.handleSubscription(client, message);
        break;

      case 'unsubscribe':
        this.handleUnsubscription(client, message);
        break;

      case 'query':
        this.handleQuery(client, message);
        break;

      default:
        this.emit('client:message', { client, message });
    }
  }

  private handleSubscription(client: WebSocketClient, message: any): void {
    const { topics = [] } = message;
    client.subscribe(topics);
    
    client.send({
      type: 'subscribed',
      topics,
      timestamp: Date.now()
    });
  }

  private handleUnsubscription(client: WebSocketClient, message: any): void {
    const { topics = [] } = message;
    client.unsubscribe(topics);
    
    client.send({
      type: 'unsubscribed',
      topics,
      timestamp: Date.now()
    });
  }

  private handleQuery(client: WebSocketClient, message: any): void {
    const { query, limit = 100 } = message;
    
    // Simple query implementation
    const results = this.messageBuffer
      .getAll()
      .filter(msg => this.matchesQuery(msg, query))
      .slice(-limit);

    client.send({
      type: 'query:result',
      queryId: message.id,
      results,
      timestamp: Date.now()
    });
  }

  private matchesQuery(message: RelayMessage, query: any): boolean {
    // Implement query matching logic
    if (query.type && message.type !== query.type) return false;
    if (query.clientId && message.clientId !== query.clientId) return false;
    if (query.after && message.timestamp < query.after) return false;
    if (query.before && message.timestamp > query.before) return false;
    
    return true;
  }

  private sendBufferedMessages(client: WebSocketClient): void {
    const messages = this.messageBuffer.getAll();
    
    client.send({
      type: 'buffer:sync',
      messages,
      timestamp: Date.now()
    });
  }

  getStats(): RelayStats {
    return {
      connectedClients: this.clients.size,
      bufferedMessages: this.messageBuffer.size(),
      totalBroadcasts: this.messageBuffer.totalAdded()
    };
  }

  close(): Promise<void> {
    return new Promise((resolve) => {
      // Close all client connections
      for (const client of this.clients.values()) {
        client.close();
      }

      // Close server
      this.wss.close(() => {
        this.emit('server:closed');
        resolve();
      });
    });
  }
}

class WebSocketClient {
  private subscriptions: Set<string> = new Set();
  private alive: boolean = true;

  constructor(
    public id: string,
    private ws: WebSocket,
    private req: any
  ) {
    // Setup heartbeat
    this.ws.on('pong', () => {
      this.alive = true;
    });
  }

  send(data: any): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  subscribe(topics: string[]): void {
    topics.forEach(topic => this.subscriptions.add(topic));
  }

  unsubscribe(topics: string[]): void {
    topics.forEach(topic => this.subscriptions.delete(topic));
  }

  isSubscribed(topic: string): boolean {
    return this.subscriptions.has(topic);
  }

  isAlive(): boolean {
    return this.alive && this.ws.readyState === WebSocket.OPEN;
  }

  close(): void {
    this.ws.close();
  }

  getInfo(): ClientInfo {
    return {
      id: this.id,
      remoteAddress: this.req.socket.remoteAddress,
      subscriptions: Array.from(this.subscriptions),
      connected: this.isAlive()
    };
  }
}

class CircularBuffer<T> {
  private buffer: T[] = [];
  private index: number = 0;
  private total: number = 0;

  constructor(private capacity: number) {}

  add(item: T): void {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(item);
    } else {
      this.buffer[this.index] = item;
      this.index = (this.index + 1) % this.capacity;
    }
    this.total++;
  }

  getAll(): T[] {
    if (this.buffer.length < this.capacity) {
      return [...this.buffer];
    }
    
    // Return in chronological order
    return [
      ...this.buffer.slice(this.index),
      ...this.buffer.slice(0, this.index)
    ];
  }

  size(): number {
    return this.buffer.length;
  }

  totalAdded(): number {
    return this.total;
  }
}

interface WebSocketConfig {
  port: number;
  bufferSize?: number;
}

interface RelayMessage {
  type: string;
  timestamp?: number;
  [key: string]: any;
}

interface RelayStats {
  connectedClients: number;
  bufferedMessages: number;
  totalBroadcasts: number;
}

interface ClientInfo {
  id: string;
  remoteAddress: string;
  subscriptions: string[];
  connected: boolean;
}
```

### Dashboard WebSocket Client

```typescript
// dashboard-ws-client.ts
export class DashboardWSClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timer | null = null;
  private eventHandlers: Map<string, Set<Function>> = new Map();
  private messageQueue: any[] = [];
  private connected: boolean = false;

  constructor(
    private url: string,
    private options: DashboardWSOptions = {}
  ) {
    this.connect();
  }

  private connect(): void {
    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        this.connected = true;
        this.emit('connected');
        this.flushMessageQueue();
        
        // Clear reconnect timer
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          this.emit('error', error);
        }
      };

      this.ws.onclose = () => {
        this.connected = false;
        this.emit('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        this.emit('error', error);
      };
    } catch (error) {
      this.emit('error', error);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    
    const delay = this.options.reconnectDelay || 5000;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private handleMessage(message: any): void {
    // Route message to appropriate handler
    switch (message.type) {
      case 'handshake':
        this.handleHandshake(message);
        break;
        
      case 'tool:start':
      case 'tool:complete':
      case 'tool:error':
        this.emit('tool', message);
        break;
        
      case 'llmkg:metrics':
      case 'llmkg:knowledge_graph':
      case 'llmkg:neural_pattern':
      case 'llmkg:cognitive_reasoning':
      case 'llmkg:memory_system':
        this.emit('telemetry', message);
        break;
        
      case 'buffer:sync':
        this.handleBufferSync(message);
        break;
        
      default:
        this.emit('message', message);
    }
  }

  private handleHandshake(message: any): void {
    this.emit('handshake', message);
    
    // Subscribe to relevant topics
    if (this.options.topics) {
      this.send({
        type: 'subscribe',
        topics: this.options.topics
      });
    }
  }

  private handleBufferSync(message: any): void {
    // Process buffered messages
    const { messages = [] } = message;
    messages.forEach((msg: any) => this.handleMessage(msg));
  }

  send(data: any): void {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // Queue message for later
      this.messageQueue.push(data);
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  query(params: QueryParams): Promise<any> {
    return new Promise((resolve, reject) => {
      const queryId = uuidv4();
      
      const handler = (message: any) => {
        if (message.type === 'query:result' && message.queryId === queryId) {
          this.off('message', handler);
          resolve(message.results);
        }
      };
      
      this.on('message', handler);
      
      this.send({
        type: 'query',
        id: queryId,
        ...params
      });
      
      // Timeout
      setTimeout(() => {
        this.off('message', handler);
        reject(new Error('Query timeout'));
      }, params.timeout || 10000);
    });
  }

  on(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  off(event: string, handler: Function): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  private emit(event: string, ...args: any[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => handler(...args));
    }
  }

  close(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

interface DashboardWSOptions {
  reconnectDelay?: number;
  topics?: string[];
}

interface QueryParams {
  type?: string;
  clientId?: string;
  after?: number;
  before?: number;
  limit?: number;
  timeout?: number;
}
```

## Testing Procedures

### MCP Client Test Suite

```typescript
// tests/mcp-client.test.ts
import { MCPClient } from '../src/mcp-client';
import { TelemetryMCPClient } from '../src/telemetry-mcp-client';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';

describe('MCP Client Tests', () => {
  let client: MCPClient;
  let mockProcess: MockProcess;

  beforeEach(() => {
    mockProcess = new MockProcess();
    jest.spyOn(require('child_process'), 'spawn').mockReturnValue(mockProcess);
    
    client = new MCPClient({
      command: 'llmkg-mcp',
      args: ['--mode', 'test']
    });
  });

  afterEach(() => {
    client.disconnect();
    jest.restoreAllMocks();
  });

  describe('Connection', () => {
    test('should connect successfully', async () => {
      const connectPromise = client.connect();
      
      // Simulate successful initialization
      mockProcess.simulateMessage({
        jsonrpc: '2.0',
        id: expect.any(String),
        result: { initialized: true }
      });

      await expect(connectPromise).resolves.toBeUndefined();
    });

    test('should handle connection errors', async () => {
      const connectPromise = client.connect();
      
      // Simulate error
      mockProcess.simulateError('Connection failed');

      await expect(connectPromise).rejects.toThrow();
    });
  });

  describe('Tool Discovery', () => {
    test('should discover available tools', async () => {
      await client.connect();
      
      const discoverPromise = client.discoverTools();
      
      // Simulate tools response
      mockProcess.simulateMessage({
        jsonrpc: '2.0',
        id: expect.any(String),
        result: {
          tools: [
            {
              name: 'knowledge_graph_query',
              description: 'Query the knowledge graph',
              inputSchema: {
                type: 'object',
                properties: {
                  query: { type: 'string' }
                }
              }
            }
          ]
        }
      });

      const tools = await discoverPromise;
      expect(tools).toHaveLength(1);
      expect(tools[0].name).toBe('knowledge_graph_query');
    });
  });

  describe('Tool Execution', () => {
    test('should execute tool successfully', async () => {
      await client.connect();
      
      // Setup tool
      client['tools'].set('test_tool', {
        name: 'test_tool',
        description: 'Test tool',
        inputSchema: { type: 'object' }
      });

      const executePromise = client.executeTool('test_tool', { param: 'value' });
      
      // Simulate successful execution
      mockProcess.simulateMessage({
        jsonrpc: '2.0',
        id: expect.any(String),
        result: { success: true, data: 'test' }
      });

      const result = await executePromise;
      expect(result).toEqual({ success: true, data: 'test' });
    });

    test('should handle tool execution errors', async () => {
      await client.connect();
      
      client['tools'].set('test_tool', {
        name: 'test_tool',
        description: 'Test tool',
        inputSchema: { type: 'object' }
      });

      const executePromise = client.executeTool('test_tool', {});
      
      // Simulate error
      mockProcess.simulateMessage({
        jsonrpc: '2.0',
        id: expect.any(String),
        error: {
          code: -32000,
          message: 'Tool execution failed'
        }
      });

      await expect(executePromise).rejects.toEqual({
        code: -32000,
        message: 'Tool execution failed'
      });
    });
  });
});

class MockProcess extends EventEmitter {
  stdin = { write: jest.fn() };
  stdout = new EventEmitter();
  stderr = new EventEmitter();

  kill = jest.fn();

  simulateMessage(message: any): void {
    const data = JSON.stringify(message) + '\n';
    this.stdout.emit('data', Buffer.from(data));
  }

  simulateError(error: string): void {
    this.stderr.emit('data', Buffer.from(error));
  }
}
```

### Integration Tests

```typescript
// tests/integration.test.ts
import { TelemetryMCPClient } from '../src/telemetry-mcp-client';
import { WebSocketRelay } from '../src/websocket-relay';
import { DashboardWSClient } from '../src/dashboard-ws-client';

describe('Integration Tests', () => {
  let mcpClient: TelemetryMCPClient;
  let wsRelay: WebSocketRelay;
  let dashboardClient: DashboardWSClient;

  beforeAll(async () => {
    // Start WebSocket relay
    wsRelay = new WebSocketRelay({ port: 8080 });
    
    // Create MCP client with telemetry
    mcpClient = new TelemetryMCPClient(
      {
        command: 'llmkg-mcp',
        args: ['--test']
      },
      {
        endpoint: 'http://localhost:3000/telemetry',
        batchSize: 10,
        flushInterval: 1000
      }
    );

    // Create dashboard client
    dashboardClient = new DashboardWSClient('ws://localhost:8080');
    
    await mcpClient.connect();
  });

  afterAll(async () => {
    await mcpClient.disconnect();
    await wsRelay.close();
    dashboardClient.close();
  });

  test('should stream tool executions to dashboard', async () => {
    const toolEvents: any[] = [];
    
    dashboardClient.on('tool', (event) => {
      toolEvents.push(event);
    });

    // Execute a tool
    await mcpClient.executeTool('test_tool', { param: 'value' });

    // Wait for events
    await new Promise(resolve => setTimeout(resolve, 100));

    expect(toolEvents).toHaveLength(2); // start and complete
    expect(toolEvents[0].type).toBe('tool:start');
    expect(toolEvents[1].type).toBe('tool:complete');
  });

  test('should collect telemetry data', async () => {
    const telemetryEvents: any[] = [];
    
    dashboardClient.on('telemetry', (event) => {
      telemetryEvents.push(event);
    });

    // Execute LLMKG-specific tool
    await mcpClient.executeTool('knowledge_graph_query', {
      query: 'test query',
      depth: 2
    });

    await new Promise(resolve => setTimeout(resolve, 100));

    const graphEvent = telemetryEvents.find(e => e.type === 'llmkg:knowledge_graph');
    expect(graphEvent).toBeDefined();
    expect(graphEvent.metrics.queryDepth).toBe(2);
  });
});
```

### Example MCP Tool Interactions

```typescript
// examples/llmkg-tools.ts
import { TelemetryMCPClient } from '../src/telemetry-mcp-client';

async function demonstrateLLMKGTools() {
  const client = new TelemetryMCPClient(
    {
      command: 'llmkg-mcp',
      args: ['--port', '3456']
    },
    {
      endpoint: 'http://localhost:3000/telemetry',
      batchSize: 50,
      flushInterval: 5000
    }
  );

  try {
    await client.connect();
    const tools = await client.discoverTools();
    
    console.log('Available tools:', tools.map(t => t.name));

    // Example 1: Knowledge Graph Query
    const graphResult = await client.executeTool('knowledge_graph_query', {
      query: 'machine learning concepts',
      query_type: 'semantic',
      depth: 3,
      max_results: 50
    });
    
    console.log('Graph query result:', {
      nodes: graphResult.graph?.nodes?.length,
      edges: graphResult.graph?.edges?.length
    });

    // Example 2: Neural Pattern Detection
    const patternResult = await client.executeTool('neural_pattern_detection', {
      input_text: 'Artificial intelligence is transforming healthcare',
      pattern_type: 'conceptual',
      threshold: 0.7
    });
    
    console.log('Pattern detection result:', {
      patterns: patternResult.patterns?.length,
      avgActivation: patternResult.activation_levels?.reduce((a: number, b: number) => a + b, 0) / patternResult.activation_levels?.length
    });

    // Example 3: Cognitive Reasoning
    const reasoningResult = await client.executeTool('cognitive_reasoning', {
      premise: 'All mammals are warm-blooded',
      query: 'Are dolphins warm-blooded?',
      type: 'deductive',
      depth: 2
    });
    
    console.log('Reasoning result:', {
      conclusion: reasoningResult.conclusion,
      confidence: reasoningResult.confidence,
      steps: reasoningResult.steps?.length
    });

    // Example 4: Memory Consolidation
    const memoryResult = await client.executeTool('memory_consolidation', {
      entries: [
        { content: 'Python is a programming language', timestamp: Date.now() - 3600000 },
        { content: 'Python is interpreted', timestamp: Date.now() - 1800000 },
        { content: 'Python supports OOP', timestamp: Date.now() }
      ],
      consolidation_type: 'semantic'
    });
    
    console.log('Memory consolidation result:', {
      consolidated: memoryResult.consolidated_entries?.length,
      clusters: memoryResult.clusters?.length
    });

    // Get telemetry stats
    const stats = client.getTelemetryStats();
    console.log('Telemetry stats:', stats);

  } catch (error) {
    console.error('Error:', error);
  } finally {
    await client.disconnect();
  }
}

// Run demonstration
demonstrateLLMKGTools();
```

## Deliverables Checklist

### Core Components
- [ ] **MCP Client Implementation**
  - [ ] Basic MCP client with full protocol support
  - [ ] Tool discovery and caching
  - [ ] Tool execution with proper error handling
  - [ ] Connection management and reconnection logic
  - [ ] Message parsing and routing

- [ ] **Telemetry System**
  - [ ] Telemetry collector with buffering
  - [ ] Event recording and aggregation
  - [ ] Statistics calculation and tracking
  - [ ] Batch upload with retry logic
  - [ ] Error handling and recovery

- [ ] **Data Collection Agents**
  - [ ] Base collector framework
  - [ ] Knowledge graph metrics collector
  - [ ] Neural pattern metrics collector
  - [ ] Cognitive reasoning metrics collector
  - [ ] Memory system metrics collector

- [ ] **WebSocket Infrastructure**
  - [ ] WebSocket relay server
  - [ ] Client connection management
  - [ ] Message broadcasting and routing
  - [ ] Subscription management
  - [ ] Query handling
  - [ ] Circular buffer for message history

- [ ] **Dashboard Integration**
  - [ ] Dashboard WebSocket client
  - [ ] Auto-reconnection logic
  - [ ] Event handling system
  - [ ] Query interface
  - [ ] Real-time data streaming

### Testing & Documentation
- [ ] **Test Suites**
  - [ ] Unit tests for MCP client
  - [ ] Unit tests for telemetry system
  - [ ] Integration tests for full pipeline
  - [ ] Performance tests
  - [ ] Error scenario tests

- [ ] **Examples & Demos**
  - [ ] Basic MCP client usage
  - [ ] LLMKG tool demonstrations
  - [ ] Telemetry visualization examples
  - [ ] WebSocket communication demos

- [ ] **Documentation**
  - [ ] API documentation
  - [ ] Integration guide
  - [ ] Configuration reference
  - [ ] Troubleshooting guide

### Deployment Artifacts
- [ ] **NPM Package**
  - [ ] Package configuration
  - [ ] TypeScript definitions
  - [ ] Build scripts
  - [ ] Publishing workflow

- [ ] **Docker Support**
  - [ ] Dockerfile for relay server
  - [ ] Docker compose configuration
  - [ ] Environment variable documentation

- [ ] **Configuration Files**
  - [ ] Default telemetry configuration
  - [ ] WebSocket server configuration
  - [ ] Example client configurations

### Performance & Monitoring
- [ ] **Metrics**
  - [ ] Client performance metrics
  - [ ] WebSocket throughput metrics
  - [ ] Telemetry processing metrics
  - [ ] Memory usage tracking

- [ ] **Optimization**
  - [ ] Message batching optimization
  - [ ] Compression implementation
  - [ ] Connection pooling
  - [ ] Resource cleanup

This completes Phase 1 of the LLMKG visualization system, providing a robust foundation for MCP integration and data collection without modifying the core Rust codebase.