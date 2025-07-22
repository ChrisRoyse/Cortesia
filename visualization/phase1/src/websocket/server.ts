/**
 * WebSocket Server Implementation for LLMKG Visualization
 * 
 * High-performance WebSocket server with real-time data streaming,
 * connection management, and LLMKG-specific protocol support.
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { IncomingMessage } from 'http';
import { 
  WebSocketMessage, 
  MessageType, 
  ConnectMessage,
  HeartbeatMessage,
  ErrorMessage,
  SubscribeMessage,
  UnsubscribeMessage,
  ProtocolValidator,
  protocolLogger 
} from './protocol';
import { MessageRouter, routerLogger } from './router';
import { MessageBuffer, MessagePriority, bufferLogger } from './buffer';
import { Logger } from '../utils/logger';

const logger = new Logger('WebSocketServer');

export interface ClientConnection {
  id: string;
  socket: WebSocket;
  isAlive: boolean;
  connectedAt: number;
  lastActivity: number;
  messageCount: number;
  subscriptions: string[];
  metadata: {
    userAgent?: string;
    ipAddress?: string;
    version?: string;
    capabilities?: string[];
  };
}

export interface ServerConfig {
  port: number;
  host?: string;
  heartbeatInterval: number;
  connectionTimeout: number;
  maxConnections: number;
  maxMessageSize: number;
  enableCompression: boolean;
  enableBuffering: boolean;
  corsOrigins?: string[];
}

export const DEFAULT_SERVER_CONFIG: ServerConfig = {
  port: 8080,
  host: '0.0.0.0',
  heartbeatInterval: 30000, // 30 seconds
  connectionTimeout: 60000, // 60 seconds
  maxConnections: 1000,
  maxMessageSize: 1024 * 1024, // 1MB
  enableCompression: true,
  enableBuffering: true,
  corsOrigins: ['*']
};

export class WebSocketServer extends EventEmitter {
  private server: WebSocket.Server | null = null;
  private clients = new Map<string, ClientConnection>();
  private messageRouter: MessageRouter;
  private messageBuffer: MessageBuffer | null = null;
  private config: ServerConfig;
  private isRunning = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private serverStats = {
    totalConnections: 0,
    currentConnections: 0,
    totalMessages: 0,
    messagesSent: 0,
    messagesReceived: 0,
    errors: 0
  };

  constructor(config: Partial<ServerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_SERVER_CONFIG, ...config };
    this.messageRouter = new MessageRouter();
    
    if (this.config.enableBuffering) {
      this.messageBuffer = new MessageBuffer({
        flushInterval: 100,
        maxBufferSize: 10000
      });
    }

    this.setupEventHandlers();
  }

  /**
   * Setup event handlers for components
   */
  private setupEventHandlers(): void {
    // Router event handlers
    this.messageRouter.on('subscription', (subscription) => {
      logger.debug('New subscription created', { subscriptionId: subscription.id });
    });

    this.messageRouter.on('unsubscription', (subscriptionId) => {
      logger.debug('Subscription removed', { subscriptionId });
    });

    // Buffer event handlers (if enabled)
    if (this.messageBuffer) {
      this.messageBuffer.on('messageReady', ({ message, topic, clientId }) => {
        if (clientId) {
          this.sendToClient(clientId, message);
        } else if (topic) {
          this.broadcast(topic, message);
        } else {
          this.broadcastToAll(message);
        }
      });

      this.messageBuffer.on('error', (error) => {
        logger.error('Buffer error:', error);
        this.serverStats.errors++;
      });
    }
  }

  /**
   * Start the WebSocket server
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Server is already running');
    }

    try {
      // Create WebSocket server
      this.server = new WebSocket.Server({
        port: this.config.port,
        host: this.config.host,
        perMessageDeflate: this.config.enableCompression,
        maxPayload: this.config.maxMessageSize,
        verifyClient: this.verifyClient.bind(this)
      });

      // Setup server event handlers
      this.server.on('connection', this.handleConnection.bind(this));
      this.server.on('error', this.handleServerError.bind(this));
      this.server.on('listening', () => {
        logger.info('WebSocket server started', {
          host: this.config.host,
          port: this.config.port,
          compression: this.config.enableCompression,
          buffering: this.config.enableBuffering
        });
      });

      // Start message buffer if enabled
      if (this.messageBuffer) {
        this.messageBuffer.start();
      }

      // Start heartbeat interval
      this.startHeartbeat();

      this.isRunning = true;
      this.emit('started', { port: this.config.port });

    } catch (error) {
      logger.error('Failed to start WebSocket server:', error);
      throw error;
    }
  }

  /**
   * Stop the WebSocket server
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    try {
      this.isRunning = false;

      // Stop heartbeat
      if (this.heartbeatInterval) {
        clearInterval(this.heartbeatInterval);
        this.heartbeatInterval = null;
      }

      // Stop message buffer
      if (this.messageBuffer) {
        await this.messageBuffer.stop();
      }

      // Close all client connections
      for (const [clientId, client] of this.clients.entries()) {
        this.disconnectClient(clientId, 'Server shutting down');
      }

      // Close server
      if (this.server) {
        await new Promise<void>((resolve, reject) => {
          this.server!.close((error) => {
            if (error) {
              reject(error);
            } else {
              resolve();
            }
          });
        });
        this.server = null;
      }

      logger.info('WebSocket server stopped', {
        totalConnections: this.serverStats.totalConnections,
        totalMessages: this.serverStats.totalMessages
      });

      this.emit('stopped');

    } catch (error) {
      logger.error('Error stopping WebSocket server:', error);
      throw error;
    }
  }

  /**
   * Verify client connections (CORS and connection limits)
   */
  private verifyClient(info: { origin: string; secure: boolean; req: IncomingMessage }): boolean {
    // Check connection limit
    if (this.clients.size >= this.config.maxConnections) {
      logger.warn('Connection rejected - max connections reached', {
        current: this.clients.size,
        max: this.config.maxConnections
      });
      return false;
    }

    // Check CORS origins
    if (this.config.corsOrigins && !this.config.corsOrigins.includes('*')) {
      const origin = info.origin;
      if (!this.config.corsOrigins.includes(origin)) {
        logger.warn('Connection rejected - CORS origin not allowed', { origin });
        return false;
      }
    }

    return true;
  }

  /**
   * Handle new client connections
   */
  private handleConnection(socket: WebSocket, request: IncomingMessage): void {
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const ipAddress = request.socket.remoteAddress;
    const userAgent = request.headers['user-agent'];

    const client: ClientConnection = {
      id: clientId,
      socket,
      isAlive: true,
      connectedAt: Date.now(),
      lastActivity: Date.now(),
      messageCount: 0,
      subscriptions: [],
      metadata: {
        ipAddress,
        userAgent
      }
    };

    // Store client connection
    this.clients.set(clientId, client);
    this.serverStats.totalConnections++;
    this.serverStats.currentConnections++;

    // Setup client event handlers
    socket.on('message', (data) => this.handleClientMessage(clientId, data));
    socket.on('close', (code, reason) => this.handleClientDisconnect(clientId, code, reason));
    socket.on('error', (error) => this.handleClientError(clientId, error));
    socket.on('pong', () => this.handleClientPong(clientId));

    logger.info('Client connected', {
      clientId,
      ipAddress,
      totalClients: this.clients.size
    });

    this.emit('clientConnected', client);

    // Send welcome message
    this.sendWelcomeMessage(client);
  }

  /**
   * Handle messages from clients
   */
  private handleClientMessage(clientId: string, data: WebSocket.Data): void {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    try {
      const rawMessage = data.toString();
      const message = JSON.parse(rawMessage);

      // Validate message
      if (!ProtocolValidator.validateMessage(message)) {
        this.sendError(clientId, 'INVALID_MESSAGE', 'Message validation failed');
        return;
      }

      // Update client activity
      client.lastActivity = Date.now();
      client.messageCount++;
      this.serverStats.messagesReceived++;

      // Handle different message types
      this.handleMessageByType(clientId, message);

    } catch (error) {
      logger.error('Error processing client message:', error);
      this.sendError(clientId, 'MESSAGE_PROCESSING_ERROR', 'Failed to process message');
      this.serverStats.errors++;
    }
  }

  /**
   * Handle messages based on their type
   */
  private handleMessageByType(clientId: string, message: WebSocketMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    switch (message.type) {
      case MessageType.CONNECT:
        this.handleConnectMessage(clientId, message as ConnectMessage);
        break;

      case MessageType.HEARTBEAT:
        this.handleHeartbeatMessage(clientId, message as HeartbeatMessage);
        break;

      case MessageType.SUBSCRIBE:
        this.handleSubscribeMessage(clientId, message as SubscribeMessage);
        break;

      case MessageType.UNSUBSCRIBE:
        this.handleUnsubscribeMessage(clientId, message as UnsubscribeMessage);
        break;

      default:
        logger.warn('Unhandled message type from client', {
          clientId,
          messageType: message.type
        });
    }
  }

  /**
   * Handle client connect messages
   */
  private handleConnectMessage(clientId: string, message: ConnectMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Update client metadata
    client.metadata.version = message.version;
    client.metadata.capabilities = message.capabilities;

    logger.info('Client connect message received', {
      clientId,
      version: message.version,
      capabilities: message.capabilities
    });

    // Send connection acknowledgment
    const ackMessage: ConnectMessage = ProtocolValidator.createMessage(
      MessageType.CONNECT,
      {
        clientId,
        version: '1.0.0',
        capabilities: ['compression', 'batching', 'subscriptions']
      }
    );

    this.sendToClient(clientId, ackMessage);
  }

  /**
   * Handle client heartbeat messages
   */
  private handleHeartbeatMessage(clientId: string, message: HeartbeatMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.isAlive = true;
    client.lastActivity = Date.now();

    // Send heartbeat response
    const response: HeartbeatMessage = ProtocolValidator.createMessage(
      MessageType.HEARTBEAT,
      {
        clientId
      }
    );

    this.sendToClient(clientId, response);
  }

  /**
   * Handle client subscription messages
   */
  private handleSubscribeMessage(clientId: string, message: SubscribeMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Process subscription through router
    const ack = this.messageRouter.subscribe(message);
    
    // Update client subscriptions
    client.subscriptions = this.messageRouter.getClientSubscriptions(clientId)
      .map(sub => sub.id);

    // Send acknowledgment
    this.sendToClient(clientId, ack);

    logger.info('Client subscription processed', {
      clientId,
      topics: message.topics,
      totalSubscriptions: client.subscriptions.length
    });
  }

  /**
   * Handle client unsubscription messages
   */
  private handleUnsubscribeMessage(clientId: string, message: UnsubscribeMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Process unsubscription through router
    const ack = this.messageRouter.unsubscribe(message);
    
    // Update client subscriptions
    client.subscriptions = this.messageRouter.getClientSubscriptions(clientId)
      .map(sub => sub.id);

    // Send acknowledgment
    this.sendToClient(clientId, ack);

    logger.info('Client unsubscription processed', {
      clientId,
      topics: message.topics,
      totalSubscriptions: client.subscriptions.length
    });
  }

  /**
   * Handle client disconnections
   */
  private handleClientDisconnect(clientId: string, code: number, reason: Buffer): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Remove client subscriptions
    this.messageRouter.removeClientSubscriptions(clientId);

    // Remove client connection
    this.clients.delete(clientId);
    this.serverStats.currentConnections--;

    logger.info('Client disconnected', {
      clientId,
      code,
      reason: reason.toString(),
      connectedTime: Date.now() - client.connectedAt,
      messageCount: client.messageCount
    });

    this.emit('clientDisconnected', { clientId, code, reason: reason.toString() });
  }

  /**
   * Handle client errors
   */
  private handleClientError(clientId: string, error: Error): void {
    logger.error('Client error:', { clientId, error });
    this.serverStats.errors++;
    this.emit('clientError', { clientId, error });
  }

  /**
   * Handle client pong responses
   */
  private handleClientPong(clientId: string): void {
    const client = this.clients.get(clientId);
    if (client) {
      client.isAlive = true;
    }
  }

  /**
   * Handle server errors
   */
  private handleServerError(error: Error): void {
    logger.error('Server error:', error);
    this.serverStats.errors++;
    this.emit('error', error);
  }

  /**
   * Send welcome message to new client
   */
  private sendWelcomeMessage(client: ClientConnection): void {
    const welcomeMessage: ConnectMessage = ProtocolValidator.createMessage(
      MessageType.CONNECT,
      {
        clientId: client.id,
        version: '1.0.0',
        capabilities: ['compression', 'batching', 'subscriptions']
      }
    );

    this.sendToClient(client.id, welcomeMessage);
  }

  /**
   * Send error message to client
   */
  private sendError(clientId: string, code: string, message: string, details?: any): void {
    const errorMessage: ErrorMessage = ProtocolValidator.createMessage(
      MessageType.ERROR,
      {
        error: { code, message, details }
      }
    );

    this.sendToClient(clientId, errorMessage);
  }

  /**
   * Send message to a specific client
   */
  sendToClient(clientId: string, message: WebSocketMessage): boolean {
    const client = this.clients.get(clientId);
    if (!client || client.socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      const serialized = JSON.stringify(message);
      client.socket.send(serialized);
      this.serverStats.messagesSent++;
      return true;
    } catch (error) {
      logger.error('Error sending message to client:', { clientId, error });
      this.serverStats.errors++;
      return false;
    }
  }

  /**
   * Broadcast message to all subscribers of a topic
   */
  broadcast(topic: string, data: any): void {
    const message: WebSocketMessage = ProtocolValidator.createMessage(
      MessageType.TELEMETRY_DATA,
      { data },
      'server'
    );

    if (this.messageBuffer) {
      // Use buffering system
      const priority = MessageBuffer.getMessagePriority(message);
      this.messageBuffer.addMessage(message, priority, topic);
    } else {
      // Direct routing
      const matches = this.messageRouter.routeMessage(message, topic);
      
      for (const match of matches) {
        this.sendToClient(match.subscription.clientId, message);
      }
    }
  }

  /**
   * Broadcast message to all connected clients
   */
  broadcastToAll(message: WebSocketMessage): void {
    for (const [clientId, client] of this.clients.entries()) {
      if (client.socket.readyState === WebSocket.OPEN) {
        this.sendToClient(clientId, message);
      }
    }
  }

  /**
   * Disconnect a specific client
   */
  disconnectClient(clientId: string, reason: string): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Send disconnect notification
    const disconnectMessage: ErrorMessage = ProtocolValidator.createMessage(
      MessageType.DISCONNECT,
      {
        error: { code: 'DISCONNECT', message: reason }
      }
    );

    this.sendToClient(clientId, disconnectMessage);

    // Close connection
    client.socket.close(1000, reason);
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      for (const [clientId, client] of this.clients.entries()) {
        if (!client.isAlive) {
          // Client failed to respond to previous ping
          client.socket.terminate();
          continue;
        }

        // Check for inactive clients
        const inactiveTime = Date.now() - client.lastActivity;
        if (inactiveTime > this.config.connectionTimeout) {
          logger.warn('Client timeout - disconnecting', { clientId, inactiveTime });
          this.disconnectClient(clientId, 'Connection timeout');
          continue;
        }

        // Send ping
        client.isAlive = false;
        client.socket.ping();
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Get server statistics
   */
  getStats(): {
    server: typeof this.serverStats;
    clients: number;
    router: ReturnType<MessageRouter['getStats']>;
    buffer?: ReturnType<MessageBuffer['getStats']>;
  } {
    const stats = {
      server: this.serverStats,
      clients: this.clients.size,
      router: this.messageRouter.getStats()
    };

    if (this.messageBuffer) {
      (stats as any).buffer = this.messageBuffer.getStats();
    }

    return stats;
  }

  /**
   * Get connected clients info
   */
  getClients(): ClientConnection[] {
    return Array.from(this.clients.values());
  }

  /**
   * Check if server is running
   */
  isActive(): boolean {
    return this.isRunning;
  }
}

export { logger as serverLogger };