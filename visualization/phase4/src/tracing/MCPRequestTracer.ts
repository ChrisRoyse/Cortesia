/**
 * MCPRequestTracer.ts
 * 
 * Core tracing system for capturing and processing MCP requests from Phase 1 WebSocket.
 * Provides real-time request monitoring and data collection for visualization.
 */

export interface MCPRequest {
  id: string;
  method: string;
  params: any;
  timestamp: number;
  source: 'client' | 'server';
  phase: 'incoming' | 'processing' | 'complete' | 'error';
  path?: string[];
  cognitivePattern?: string;
  processingTime?: number;
  error?: string;
}

export interface MCPResponse {
  id: string;
  requestId: string;
  result?: any;
  error?: any;
  timestamp: number;
  processingTime: number;
}

export interface TraceEvent {
  id: string;
  requestId: string;
  type: 'request' | 'response' | 'cognitive_activation' | 'error' | 'performance';
  timestamp: number;
  data: any;
  phase: string;
  coordinates?: { x: number; y: number; z?: number };
}

export class MCPRequestTracer {
  private websocket: WebSocket | null = null;
  private requests: Map<string, MCPRequest> = new Map();
  private traces: TraceEvent[] = [];
  private listeners: Map<string, ((event: TraceEvent) => void)[]> = new Map();
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 2000;

  constructor(private websocketUrl: string = 'ws://localhost:8080/mcp') {
    this.connect();
  }

  /**
   * Connect to Phase 1 WebSocket for MCP request data
   */
  private connect(): void {
    try {
      this.websocket = new WebSocket(this.websocketUrl);
      
      this.websocket.onopen = () => {
        console.log('MCPRequestTracer: Connected to Phase 1 WebSocket');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emitEvent('connection', { status: 'connected' });
      };

      this.websocket.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.websocket.onclose = () => {
        console.log('MCPRequestTracer: Connection closed');
        this.isConnected = false;
        this.attemptReconnect();
      };

      this.websocket.onerror = (error) => {
        console.error('MCPRequestTracer: WebSocket error', error);
        this.emitEvent('error', { error: 'WebSocket connection error' });
      };

    } catch (error) {
      console.error('MCPRequestTracer: Failed to connect', error);
      this.attemptReconnect();
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      switch (message.type) {
        case 'mcp_request':
          this.handleMCPRequest(message.payload);
          break;
        case 'mcp_response':
          this.handleMCPResponse(message.payload);
          break;
        case 'cognitive_activation':
          this.handleCognitiveActivation(message.payload);
          break;
        case 'performance_metric':
          this.handlePerformanceMetric(message.payload);
          break;
        case 'error':
          this.handleError(message.payload);
          break;
        default:
          console.warn('MCPRequestTracer: Unknown message type', message.type);
      }
    } catch (error) {
      console.error('MCPRequestTracer: Failed to parse message', error);
    }
  }

  /**
   * Process MCP request data
   */
  private handleMCPRequest(payload: any): void {
    const request: MCPRequest = {
      id: payload.id || this.generateId(),
      method: payload.method,
      params: payload.params,
      timestamp: payload.timestamp || Date.now(),
      source: payload.source || 'client',
      phase: 'incoming',
      path: payload.path || [],
      cognitivePattern: payload.cognitivePattern
    };

    this.requests.set(request.id, request);

    const traceEvent: TraceEvent = {
      id: this.generateId(),
      requestId: request.id,
      type: 'request',
      timestamp: request.timestamp,
      data: request,
      phase: 'incoming',
      coordinates: this.calculateEntryCoordinates(request)
    };

    this.addTrace(traceEvent);
    this.emitEvent('request', traceEvent);
  }

  /**
   * Process MCP response data
   */
  private handleMCPResponse(payload: any): void {
    const response: MCPResponse = {
      id: payload.id || this.generateId(),
      requestId: payload.requestId,
      result: payload.result,
      error: payload.error,
      timestamp: payload.timestamp || Date.now(),
      processingTime: payload.processingTime || 0
    };

    // Update request status
    const request = this.requests.get(response.requestId);
    if (request) {
      request.phase = response.error ? 'error' : 'complete';
      request.processingTime = response.processingTime;
      request.error = response.error?.message;
    }

    const traceEvent: TraceEvent = {
      id: this.generateId(),
      requestId: response.requestId,
      type: 'response',
      timestamp: response.timestamp,
      data: response,
      phase: response.error ? 'error' : 'complete',
      coordinates: this.calculateExitCoordinates(response)
    };

    this.addTrace(traceEvent);
    this.emitEvent('response', traceEvent);
  }

  /**
   * Process cognitive activation events
   */
  private handleCognitiveActivation(payload: any): void {
    const request = this.requests.get(payload.requestId);
    if (request) {
      request.cognitivePattern = payload.pattern;
      request.phase = 'processing';
      if (payload.path) {
        request.path = [...(request.path || []), ...payload.path];
      }
    }

    const traceEvent: TraceEvent = {
      id: this.generateId(),
      requestId: payload.requestId,
      type: 'cognitive_activation',
      timestamp: payload.timestamp || Date.now(),
      data: payload,
      phase: 'processing',
      coordinates: this.calculateCognitiveCoordinates(payload)
    };

    this.addTrace(traceEvent);
    this.emitEvent('cognitive_activation', traceEvent);
  }

  /**
   * Process performance metrics
   */
  private handlePerformanceMetric(payload: any): void {
    const traceEvent: TraceEvent = {
      id: this.generateId(),
      requestId: payload.requestId,
      type: 'performance',
      timestamp: payload.timestamp || Date.now(),
      data: payload,
      phase: 'processing'
    };

    this.addTrace(traceEvent);
    this.emitEvent('performance', traceEvent);
  }

  /**
   * Process error events
   */
  private handleError(payload: any): void {
    const request = this.requests.get(payload.requestId);
    if (request) {
      request.phase = 'error';
      request.error = payload.error;
    }

    const traceEvent: TraceEvent = {
      id: this.generateId(),
      requestId: payload.requestId,
      type: 'error',
      timestamp: payload.timestamp || Date.now(),
      data: payload,
      phase: 'error',
      coordinates: this.calculateErrorCoordinates(payload)
    };

    this.addTrace(traceEvent);
    this.emitEvent('error', traceEvent);
  }

  /**
   * Calculate entry coordinates for request visualization
   */
  private calculateEntryCoordinates(request: MCPRequest): { x: number; y: number; z?: number } {
    // Entry points based on request method
    const methodCoordinates = {
      'tools/list': { x: -100, y: 0 },
      'tools/call': { x: -100, y: 50 },
      'resources/list': { x: -100, y: -50 },
      'prompts/list': { x: -100, y: 25 },
      'default': { x: -100, y: 0 }
    };

    return methodCoordinates[request.method] || methodCoordinates.default;
  }

  /**
   * Calculate exit coordinates for response visualization
   */
  private calculateExitCoordinates(response: MCPResponse): { x: number; y: number; z?: number } {
    return response.error ? { x: 100, y: -25 } : { x: 100, y: 25 };
  }

  /**
   * Calculate cognitive system coordinates
   */
  private calculateCognitiveCoordinates(payload: any): { x: number; y: number; z?: number } {
    const patternCoordinates = {
      'hierarchical_inhibitory': { x: 0, y: 50 },
      'working_memory': { x: 25, y: 25 },
      'knowledge_engine': { x: 0, y: 0 },
      'activation_engine': { x: -25, y: 25 },
      'default': { x: 0, y: 0 }
    };

    return patternCoordinates[payload.pattern] || patternCoordinates.default;
  }

  /**
   * Calculate error coordinates
   */
  private calculateErrorCoordinates(payload: any): { x: number; y: number; z?: number } {
    return { x: 0, y: -50 };
  }

  /**
   * Attempt to reconnect to WebSocket
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('MCPRequestTracer: Max reconnection attempts reached');
      this.emitEvent('connection', { status: 'failed', attempts: this.reconnectAttempts });
      return;
    }

    this.reconnectAttempts++;
    console.log(`MCPRequestTracer: Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

    setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * this.reconnectAttempts);
  }

  /**
   * Add trace event to collection
   */
  private addTrace(event: TraceEvent): void {
    this.traces.push(event);
    
    // Keep only last 1000 traces for memory management
    if (this.traces.length > 1000) {
      this.traces = this.traces.slice(-1000);
    }
  }

  /**
   * Emit event to listeners
   */
  private emitEvent(eventType: string, data: any): void {
    const listeners = this.listeners.get(eventType) || [];
    listeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        console.error('MCPRequestTracer: Listener error', error);
      }
    });
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Public API methods
   */

  /**
   * Add event listener
   */
  public addEventListener(eventType: string, listener: (event: TraceEvent) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)!.push(listener);
  }

  /**
   * Remove event listener
   */
  public removeEventListener(eventType: string, listener: (event: TraceEvent) => void): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  /**
   * Get all requests
   */
  public getRequests(): MCPRequest[] {
    return Array.from(this.requests.values());
  }

  /**
   * Get request by ID
   */
  public getRequest(id: string): MCPRequest | undefined {
    return this.requests.get(id);
  }

  /**
   * Get all trace events
   */
  public getTraces(): TraceEvent[] {
    return [...this.traces];
  }

  /**
   * Get traces for specific request
   */
  public getRequestTraces(requestId: string): TraceEvent[] {
    return this.traces.filter(trace => trace.requestId === requestId);
  }

  /**
   * Get connection status
   */
  public isConnectedToPhase1(): boolean {
    return this.isConnected;
  }

  /**
   * Manually send request (for testing)
   */
  public simulateRequest(request: Partial<MCPRequest>): void {
    const fullRequest: MCPRequest = {
      id: request.id || this.generateId(),
      method: request.method || 'test/simulate',
      params: request.params || {},
      timestamp: request.timestamp || Date.now(),
      source: request.source || 'client',
      phase: request.phase || 'incoming',
      path: request.path || [],
      cognitivePattern: request.cognitivePattern
    };

    this.handleMCPRequest(fullRequest);
  }

  /**
   * Clear all traces and requests
   */
  public clear(): void {
    this.requests.clear();
    this.traces.length = 0;
    this.emitEvent('clear', {});
  }

  /**
   * Disconnect and cleanup
   */
  public disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.isConnected = false;
    this.listeners.clear();
  }
}