/**
 * @fileoverview MCP Connection Management for LLMKG Visualization
 * 
 * This module provides robust connection management for MCP clients, including
 * auto-reconnection, retry logic with exponential backoff, and connection pooling.
 * It handles WebSocket connections to LLMKG MCP servers with comprehensive error
 * handling and telemetry collection.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import { EventEmitter } from 'events';
import {
  ConnectionConfig,
  ConnectionState,
  RetryConfig,
  MCPEventType,
  ConnectionStateChangeEvent,
  MessageEvent,
  ErrorEvent,
  TelemetryEvent,
  MCPMessage,
  MCPResponse,
  MCPErrorCode
} from './types.js';

/**
 * WebSocket connection manager with auto-reconnection and retry logic
 */
export class MCPConnection extends EventEmitter {
  private ws: WebSocket | null = null;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private config: ConnectionConfig & Required<Omit<ConnectionConfig, 'authToken'>>;
  private retryConfig: RetryConfig;
  private retryCount = 0;
  private retryTimeout: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private messageQueue: MCPMessage[] = [];
  private pendingRequests = new Map<string, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>();

  /**
   * Default connection configuration
   */
  private static readonly DEFAULT_CONFIG: Required<Omit<ConnectionConfig, 'endpoint' | 'authToken'>> & { authToken?: string } = {
    timeout: 30000,
    maxRetries: 5,
    baseDelay: 1000,
    maxDelay: 30000,
    headers: {},
    authToken: undefined
  };

  /**
   * Default retry configuration
   */
  private static readonly DEFAULT_RETRY_CONFIG: RetryConfig = {
    maxAttempts: 5,
    baseDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
    jitter: true
  };

  /**
   * Creates a new MCP connection manager
   * 
   * @param config - Connection configuration
   * @param retryConfig - Retry behavior configuration
   */
  constructor(
    config: ConnectionConfig,
    retryConfig: Partial<RetryConfig> = {}
  ) {
    super();
    this.config = { ...MCPConnection.DEFAULT_CONFIG, ...config };
    this.retryConfig = { ...MCPConnection.DEFAULT_RETRY_CONFIG, ...retryConfig };
    
    // Set max listeners to handle multiple subscribers
    this.setMaxListeners(100);
  }

  /**
   * Gets the current connection state
   */
  public get connectionState(): ConnectionState {
    return this.state;
  }

  /**
   * Gets the endpoint URL
   */
  public get endpoint(): string {
    return this.config.endpoint;
  }

  /**
   * Checks if the connection is currently active
   */
  public get isConnected(): boolean {
    return this.state === ConnectionState.CONNECTED;
  }

  /**
   * Establishes a connection to the MCP server
   * 
   * @returns Promise that resolves when connection is established
   * @throws Error if connection fails after all retries
   */
  public async connect(): Promise<void> {
    if (this.state === ConnectionState.CONNECTING || this.state === ConnectionState.CONNECTED) {
      return Promise.resolve();
    }

    this.setState(ConnectionState.CONNECTING);
    this.retryCount = 0;

    return this.attemptConnection();
  }

  /**
   * Disconnects from the MCP server
   * 
   * @param code - WebSocket close code
   * @param reason - Close reason
   */
  public async disconnect(code = 1000, reason = "Client disconnect"): Promise<void> {
    this.clearRetryTimeout();
    this.clearPingInterval();
    this.clearPendingRequests();
    this.messageQueue.length = 0;

    if (this.ws) {
      this.ws.close(code, reason);
      this.ws = null;
    }

    this.setState(ConnectionState.DISCONNECTED);
  }

  /**
   * Sends a message to the server and waits for a response
   * 
   * @param message - MCP message to send
   * @param timeout - Request timeout in milliseconds
   * @returns Promise that resolves with the server response
   */
  public async sendRequest<T = any>(
    message: MCPMessage,
    timeout = this.config.timeout
  ): Promise<T> {
    if (!this.isConnected) {
      throw new Error("Connection not established");
    }

    return new Promise<T>((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        this.pendingRequests.delete(message.id);
        const error = new Error(`Request timeout after ${timeout}ms`);
        this.emitError(error, `sendRequest:${message.method}`);
        reject(error);
      }, timeout);

      this.pendingRequests.set(message.id, {
        resolve,
        reject,
        timeout: timeoutHandle
      });

      this.sendMessage(message);
    });
  }

  /**
   * Sends a message without waiting for a response
   * 
   * @param message - MCP message to send
   */
  public sendNotification(message: Omit<MCPMessage, 'id'>): void {
    const notification = { ...message, jsonrpc: "2.0" as const };
    this.sendMessage(notification);
  }

  /**
   * Internal method to send a message through the WebSocket
   */
  private sendMessage(message: MCPMessage | Omit<MCPMessage, 'id'>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      if ('id' in message) {
        this.messageQueue.push(message);
      }
      return;
    }

    try {
      const messageStr = JSON.stringify(message);
      this.ws.send(messageStr);
      
      this.emit(MCPEventType.MESSAGE_SENT, {
        type: MCPEventType.MESSAGE_SENT,
        timestamp: new Date(),
        data: { message, endpoint: this.config.endpoint }
      } as MessageEvent);

      this.emitTelemetry("mcp.message.sent", 1, {
        method: message.method,
        endpoint: this.config.endpoint
      });
    } catch (error) {
      this.emitError(error as Error, "sendMessage");
    }
  }

  /**
   * Attempts to establish a connection with retry logic
   */
  private async attemptConnection(): Promise<void> {
    try {
      await this.createConnection();
      this.retryCount = 0;
      this.setState(ConnectionState.CONNECTED);
      this.startPingInterval();
      this.flushMessageQueue();
    } catch (error) {
      this.handleConnectionError(error as Error);
    }
  }

  /**
   * Creates the actual WebSocket connection
   */
  private createConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = this.buildWebSocketUrl();
        this.ws = new WebSocket(wsUrl);
        
        const connectTimeout = setTimeout(() => {
          reject(new Error("Connection timeout"));
        }, this.config.timeout);

        this.ws.onopen = () => {
          clearTimeout(connectTimeout);
          this.setupEventHandlers();
          resolve();
        };

        this.ws.onerror = (event) => {
          clearTimeout(connectTimeout);
          reject(new Error(`WebSocket error: ${event}`));
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectTimeout);
          if (event.code !== 1000) {
            reject(new Error(`Connection closed: ${event.code} ${event.reason}`));
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Builds the WebSocket URL with authentication if needed
   */
  private buildWebSocketUrl(): string {
    const url = new URL(this.config.endpoint);
    
    if (this.config.authToken) {
      url.searchParams.set('token', this.config.authToken);
    }

    return url.toString();
  }

  /**
   * Sets up WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        this.handleIncomingMessage(message);
      } catch (error) {
        this.emitError(error as Error, "parseMessage");
      }
    };

    this.ws.onclose = (event) => {
      this.handleConnectionClose(event);
    };

    this.ws.onerror = (event) => {
      this.emitError(new Error(`WebSocket error: ${event}`), "websocket");
    };
  }

  /**
   * Handles incoming messages from the server
   */
  private handleIncomingMessage(message: any): void {
    this.emit(MCPEventType.MESSAGE_RECEIVED, {
      type: MCPEventType.MESSAGE_RECEIVED,
      timestamp: new Date(),
      data: { message, endpoint: this.config.endpoint }
    } as MessageEvent);

    this.emitTelemetry("mcp.message.received", 1, {
      method: message.method || "response",
      endpoint: this.config.endpoint
    });

    // Handle responses to pending requests
    if (message.id && this.pendingRequests.has(message.id)) {
      const pending = this.pendingRequests.get(message.id)!;
      this.pendingRequests.delete(message.id);
      clearTimeout(pending.timeout);

      if (message.error) {
        pending.reject(new Error(message.error.message));
      } else {
        pending.resolve(message.result);
      }
    }
  }

  /**
   * Handles connection close events
   */
  private handleConnectionClose(event: CloseEvent): void {
    this.clearPingInterval();
    this.ws = null;

    if (event.code === 1000) {
      // Normal closure
      this.setState(ConnectionState.DISCONNECTED);
    } else {
      // Abnormal closure - attempt reconnection
      this.handleConnectionError(new Error(`Connection closed: ${event.code} ${event.reason}`));
    }
  }

  /**
   * Handles connection errors and implements retry logic
   */
  private handleConnectionError(error: Error): void {
    this.emitError(error, "connection");
    this.emitTelemetry("mcp.connection.error", 1, {
      error: error.message,
      endpoint: this.config.endpoint
    });

    if (this.retryCount < this.retryConfig.maxAttempts) {
      this.setState(ConnectionState.RECONNECTING);
      const delay = this.calculateRetryDelay();
      
      this.retryTimeout = setTimeout(() => {
        this.retryCount++;
        this.attemptConnection();
      }, delay);
    } else {
      this.setState(ConnectionState.ERROR, error);
    }
  }

  /**
   * Calculates the delay for the next retry attempt using exponential backoff
   */
  private calculateRetryDelay(): number {
    const baseDelay = Math.min(
      this.retryConfig.baseDelay * Math.pow(this.retryConfig.backoffMultiplier, this.retryCount),
      this.retryConfig.maxDelay
    );

    if (this.retryConfig.jitter) {
      // Add up to 25% jitter
      const jitter = baseDelay * 0.25 * Math.random();
      return baseDelay + jitter;
    }

    return baseDelay;
  }

  /**
   * Sets the connection state and emits state change events
   */
  private setState(newState: ConnectionState, error?: Error): void {
    const oldState = this.state;
    this.state = newState;

    this.emit(MCPEventType.CONNECTION_STATE_CHANGED, {
      type: MCPEventType.CONNECTION_STATE_CHANGED,
      timestamp: new Date(),
      data: { oldState, newState, endpoint: this.config.endpoint, error }
    } as ConnectionStateChangeEvent);

    this.emitTelemetry("mcp.connection.state", newState, {
      endpoint: this.config.endpoint
    });
  }

  /**
   * Starts the ping interval to keep the connection alive
   */
  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (this.isConnected) {
        this.sendNotification({
          method: "ping",
          jsonrpc: "2.0"
        });
      }
    }, 30000); // Ping every 30 seconds
  }

  /**
   * Flushes any queued messages when connection is established
   */
  private flushMessageQueue(): void {
    const messages = [...this.messageQueue];
    this.messageQueue.length = 0;

    for (const message of messages) {
      this.sendMessage(message);
    }
  }

  /**
   * Clears the retry timeout
   */
  private clearRetryTimeout(): void {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout);
      this.retryTimeout = null;
    }
  }

  /**
   * Clears the ping interval
   */
  private clearPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Clears all pending requests with timeout errors
   */
  private clearPendingRequests(): void {
    for (const [id, pending] of this.pendingRequests) {
      clearTimeout(pending.timeout);
      pending.reject(new Error("Connection closed"));
    }
    this.pendingRequests.clear();
  }

  /**
   * Emits an error event
   */
  private emitError(error: Error, context: string): void {
    this.emit(MCPEventType.ERROR_OCCURRED, {
      type: MCPEventType.ERROR_OCCURRED,
      timestamp: new Date(),
      data: { error, context, endpoint: this.config.endpoint }
    } as ErrorEvent);
  }

  /**
   * Emits a telemetry event
   */
  private emitTelemetry(
    metric: string,
    value: number | string,
    tags: Record<string, string> = {}
  ): void {
    this.emit(MCPEventType.TELEMETRY_EVENT, {
      type: MCPEventType.TELEMETRY_EVENT,
      timestamp: new Date(),
      data: { metric, value, tags }
    } as TelemetryEvent);
  }
}

/**
 * Connection pool manager for handling multiple MCP server connections
 */
export class MCPConnectionPool extends EventEmitter {
  private connections = new Map<string, MCPConnection>();
  private defaultConfig: Partial<ConnectionConfig>;

  /**
   * Creates a new connection pool
   * 
   * @param defaultConfig - Default configuration for new connections
   */
  constructor(defaultConfig: Partial<ConnectionConfig> = {}) {
    super();
    this.defaultConfig = defaultConfig;
    this.setMaxListeners(100);
  }

  /**
   * Creates or retrieves a connection for the given endpoint
   * 
   * @param endpoint - Server endpoint URL
   * @param config - Optional connection-specific configuration
   * @returns MCP connection instance
   */
  public getConnection(
    endpoint: string,
    config: Partial<ConnectionConfig> = {}
  ): MCPConnection {
    if (this.connections.has(endpoint)) {
      return this.connections.get(endpoint)!;
    }

    const connectionConfig = { ...this.defaultConfig, ...config, endpoint };
    const connection = new MCPConnection(connectionConfig as ConnectionConfig);

    // Forward all events from individual connections
    connection.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
      this.emit(MCPEventType.CONNECTION_STATE_CHANGED, event);
    });

    connection.on(MCPEventType.MESSAGE_RECEIVED, (event) => {
      this.emit(MCPEventType.MESSAGE_RECEIVED, event);
    });

    connection.on(MCPEventType.MESSAGE_SENT, (event) => {
      this.emit(MCPEventType.MESSAGE_SENT, event);
    });

    connection.on(MCPEventType.ERROR_OCCURRED, (event) => {
      this.emit(MCPEventType.ERROR_OCCURRED, event);
    });

    connection.on(MCPEventType.TELEMETRY_EVENT, (event) => {
      this.emit(MCPEventType.TELEMETRY_EVENT, event);
    });

    this.connections.set(endpoint, connection);
    return connection;
  }

  /**
   * Connects to all configured endpoints
   * 
   * @param endpoints - Array of endpoint URLs
   * @returns Promise that resolves when all connections are established
   */
  public async connectAll(endpoints: string[]): Promise<void> {
    const connections = endpoints.map(endpoint => this.getConnection(endpoint));
    await Promise.all(connections.map(conn => conn.connect()));
  }

  /**
   * Disconnects from a specific endpoint
   * 
   * @param endpoint - Endpoint URL to disconnect from
   */
  public async disconnect(endpoint: string): Promise<void> {
    const connection = this.connections.get(endpoint);
    if (connection) {
      await connection.disconnect();
      this.connections.delete(endpoint);
    }
  }

  /**
   * Disconnects from all endpoints
   */
  public async disconnectAll(): Promise<void> {
    const disconnectPromises = Array.from(this.connections.values())
      .map(conn => conn.disconnect());
    
    await Promise.all(disconnectPromises);
    this.connections.clear();
  }

  /**
   * Gets all active connections
   */
  public getActiveConnections(): MCPConnection[] {
    return Array.from(this.connections.values())
      .filter(conn => conn.isConnected);
  }

  /**
   * Gets connection statistics
   */
  public getStats(): {
    total: number;
    connected: number;
    connecting: number;
    disconnected: number;
    error: number;
  } {
    const connections = Array.from(this.connections.values());
    return {
      total: connections.length,
      connected: connections.filter(c => c.connectionState === ConnectionState.CONNECTED).length,
      connecting: connections.filter(c => c.connectionState === ConnectionState.CONNECTING).length,
      disconnected: connections.filter(c => c.connectionState === ConnectionState.DISCONNECTED).length,
      error: connections.filter(c => c.connectionState === ConnectionState.ERROR).length
    };
  }
}