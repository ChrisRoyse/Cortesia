/**
 * Client-side WebSocket Management for LLMKG Dashboard
 * 
 * Provides robust WebSocket client with auto-reconnection, subscription management,
 * and LLMKG-specific data handling for dashboard visualization.
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { 
  WebSocketMessage, 
  MessageType,
  DataTopic,
  ConnectMessage,
  HeartbeatMessage,
  SubscribeMessage,
  UnsubscribeMessage,
  SubscriptionAckMessage,
  CompressedDataMessage,
  ProtocolValidator,
  TopicManager
} from './protocol';
import { MessageBuffer } from './buffer';
import { Logger } from '../utils/logger';

const logger = new Logger('WebSocketClient');

export interface ClientConfig {
  url: string;
  autoReconnect: boolean;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  connectionTimeout: number;
  enableCompression: boolean;
  subscriptionTimeout: number;
}

export const DEFAULT_CLIENT_CONFIG: ClientConfig = {
  url: 'ws://localhost:8080',
  autoReconnect: true,
  reconnectInterval: 5000, // 5 seconds
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000, // 30 seconds
  connectionTimeout: 10000, // 10 seconds
  enableCompression: true,
  subscriptionTimeout: 5000 // 5 seconds
};

export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  FAILED = 'failed'
}

export interface Subscription {
  topics: string[];
  filters?: Record<string, any>;
  callback: (message: WebSocketMessage, topic: string) => void;
  subscriptionId?: string;
  active: boolean;
}

export class DashboardWebSocketClient extends EventEmitter {
  private socket: WebSocket | null = null;
  private config: ClientConfig;
  private state = ConnectionState.DISCONNECTED;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  
  private subscriptions = new Map<string, Subscription>();
  private pendingSubscriptions = new Map<string, Subscription>();
  private clientId: string | null = null;
  private serverCapabilities: string[] = [];
  
  private clientStats = {
    totalConnections: 0,
    totalReconnections: 0,
    messagesReceived: 0,
    messagesSent: 0,
    subscriptionCount: 0,
    errors: 0,
    lastConnectedAt: 0,
    totalUptime: 0
  };

  constructor(config: Partial<ClientConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CLIENT_CONFIG, ...config };
    this.setupEventHandlers();
  }

  /**
   * Setup internal event handlers
   */
  private setupEventHandlers(): void {
    this.on('connected', () => {
      this.clientStats.lastConnectedAt = Date.now();
      this.startHeartbeat();
      this.resubscribeAll();
    });

    this.on('disconnected', () => {
      this.stopHeartbeat();
      this.updateUptime();
    });

    this.on('message', (message: WebSocketMessage) => {
      this.handleReceivedMessage(message);
    });
  }

  /**
   * Connect to the WebSocket server
   */
  async connect(url?: string): Promise<void> {
    if (this.state === ConnectionState.CONNECTING || this.state === ConnectionState.CONNECTED) {
      return;
    }

    const connectUrl = url || this.config.url;
    this.setState(ConnectionState.CONNECTING);

    return new Promise((resolve, reject) => {
      try {
        // Create WebSocket connection
        this.socket = new WebSocket(connectUrl, {
          perMessageDeflate: this.config.enableCompression
        });

        // Set connection timeout
        this.connectionTimer = setTimeout(() => {
          if (this.state === ConnectionState.CONNECTING) {
            this.socket?.close();
            reject(new Error('Connection timeout'));
          }
        }, this.config.connectionTimeout);

        // Setup socket event handlers
        this.socket.on('open', () => {
          this.handleConnectionOpen();
          resolve();
        });

        this.socket.on('message', (data) => {
          this.handleSocketMessage(data);
        });

        this.socket.on('close', (code, reason) => {
          this.handleConnectionClose(code, reason);
        });

        this.socket.on('error', (error) => {
          this.handleConnectionError(error);
          if (this.state === ConnectionState.CONNECTING) {
            reject(error);
          }
        });

      } catch (error) {
        this.setState(ConnectionState.FAILED);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.config.autoReconnect = false; // Disable auto-reconnect
    
    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
    }
    
    this.cleanup();
    this.setState(ConnectionState.DISCONNECTED);
  }

  /**
   * Subscribe to topics with optional filters
   */
  async subscribe(
    topics: string[], 
    callback: (message: WebSocketMessage, topic: string) => void,
    filters?: Record<string, any>
  ): Promise<void> {
    // Validate topics
    const validTopics = topics.filter(topic => 
      TopicManager.isValidTopic(topic) || topic.includes('*')
    );

    if (validTopics.length === 0) {
      throw new Error('No valid topics provided');
    }

    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const subscription: Subscription = {
      topics: validTopics,
      filters,
      callback,
      subscriptionId,
      active: false
    };

    // Store subscription
    this.subscriptions.set(subscriptionId, subscription);

    // If connected, send subscription immediately
    if (this.state === ConnectionState.CONNECTED && this.clientId) {
      await this.sendSubscription(subscription);
    }

    logger.info('Subscription created', { 
      subscriptionId, 
      topics: validTopics,
      connected: this.state === ConnectionState.CONNECTED
    });
  }

  /**
   * Unsubscribe from topics
   */
  async unsubscribe(topics: string[]): Promise<void> {
    const subscriptionsToRemove: string[] = [];

    // Find subscriptions that match the topics
    for (const [subscriptionId, subscription] of this.subscriptions.entries()) {
      const matchingTopics = subscription.topics.filter(topic => topics.includes(topic));
      
      if (matchingTopics.length > 0) {
        if (matchingTopics.length === subscription.topics.length) {
          // Remove entire subscription
          subscriptionsToRemove.push(subscriptionId);
        } else {
          // Update subscription to remove only matching topics
          subscription.topics = subscription.topics.filter(topic => !topics.includes(topic));
        }
      }
    }

    // Remove complete subscriptions
    for (const subscriptionId of subscriptionsToRemove) {
      this.subscriptions.delete(subscriptionId);
    }

    // Send unsubscription to server if connected
    if (this.state === ConnectionState.CONNECTED && this.clientId) {
      const unsubMessage: UnsubscribeMessage = ProtocolValidator.createMessage(
        MessageType.UNSUBSCRIBE,
        {
          topics,
          clientId: this.clientId
        }
      );

      await this.sendMessage(unsubMessage);
    }

    logger.info('Unsubscribed from topics', { 
      topics, 
      removedSubscriptions: subscriptionsToRemove.length
    });
  }

  /**
   * Send a message to the server
   */
  private async sendMessage(message: WebSocketMessage): Promise<void> {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }

    try {
      const serialized = JSON.stringify(message);
      this.socket.send(serialized);
      this.clientStats.messagesSent++;
    } catch (error) {
      this.clientStats.errors++;
      logger.error('Failed to send message:', error);
      throw error;
    }
  }

  /**
   * Handle WebSocket connection open
   */
  private handleConnectionOpen(): void {
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }

    this.setState(ConnectionState.CONNECTED);
    this.reconnectAttempts = 0;
    this.clientStats.totalConnections++;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    logger.info('WebSocket connected', { url: this.config.url });

    // Send connect message
    this.sendConnectMessage();
    
    this.emit('connected');
  }

  /**
   * Handle WebSocket messages
   */
  private handleSocketMessage(data: WebSocket.Data): void {
    try {
      const rawMessage = data.toString();
      const message = JSON.parse(rawMessage);

      if (!ProtocolValidator.validateMessage(message)) {
        logger.warn('Received invalid message from server');
        return;
      }

      this.clientStats.messagesReceived++;
      this.emit('message', message);

    } catch (error) {
      this.clientStats.errors++;
      logger.error('Error processing message from server:', error);
    }
  }

  /**
   * Handle received messages by type
   */
  private handleReceivedMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case MessageType.CONNECT:
        this.handleConnectMessage(message as ConnectMessage);
        break;

      case MessageType.SUBSCRIPTION_ACK:
        this.handleSubscriptionAck(message as SubscriptionAckMessage);
        break;

      case MessageType.COMPRESSED_DATA:
        this.handleCompressedMessage(message as CompressedDataMessage);
        break;

      case MessageType.HEARTBEAT:
        this.handleHeartbeat(message as HeartbeatMessage);
        break;

      case MessageType.ERROR:
        this.handleError(message);
        break;

      default:
        // Route data messages to subscriptions
        this.routeMessageToSubscriptions(message);
        break;
    }
  }

  /**
   * Handle connect acknowledgment from server
   */
  private handleConnectMessage(message: ConnectMessage): void {
    this.clientId = message.clientId;
    this.serverCapabilities = message.capabilities || [];

    logger.info('Connection acknowledged by server', {
      clientId: this.clientId,
      serverCapabilities: this.serverCapabilities
    });
  }

  /**
   * Handle subscription acknowledgment
   */
  private handleSubscriptionAck(message: SubscriptionAckMessage): void {
    const { topics, status } = message;

    if (status === 'success') {
      // Mark subscriptions as active
      for (const subscription of this.subscriptions.values()) {
        const hasMatchingTopic = subscription.topics.some(topic => topics.includes(topic));
        if (hasMatchingTopic) {
          subscription.active = true;
        }
      }

      this.clientStats.subscriptionCount = Array.from(this.subscriptions.values())
        .filter(sub => sub.active).length;

      logger.info('Subscription acknowledged', { topics, status });
    } else {
      logger.error('Subscription failed', { topics, message: message.message });
    }
  }

  /**
   * Handle compressed messages
   */
  private async handleCompressedMessage(message: CompressedDataMessage): Promise<void> {
    try {
      const decompressed = await MessageBuffer.decompressMessage(message);
      this.handleReceivedMessage(decompressed);
    } catch (error) {
      logger.error('Failed to decompress message:', error);
      this.clientStats.errors++;
    }
  }

  /**
   * Handle heartbeat from server
   */
  private handleHeartbeat(message: HeartbeatMessage): void {
    // Respond to server heartbeat
    const response: HeartbeatMessage = ProtocolValidator.createMessage(
      MessageType.HEARTBEAT,
      {
        clientId: this.clientId || 'unknown'
      }
    );

    this.sendMessage(response).catch(error => {
      logger.error('Failed to respond to heartbeat:', error);
    });
  }

  /**
   * Handle error messages
   */
  private handleError(message: WebSocketMessage): void {
    if ('error' in message) {
      const error = (message as any).error;
      logger.error('Server error:', error);
      this.emit('error', error);
    }
  }

  /**
   * Route data messages to appropriate subscriptions
   */
  private routeMessageToSubscriptions(message: WebSocketMessage): void {
    const messageTopic = this.determineMessageTopic(message);
    if (!messageTopic) return;

    for (const subscription of this.subscriptions.values()) {
      if (!subscription.active) continue;

      // Check if any subscription topic matches the message topic
      const matchingTopic = subscription.topics.find(topic => 
        topic === messageTopic || TopicManager.matchesTopic(messageTopic, topic)
      );

      if (matchingTopic && this.messagePassesFilters(message, subscription.filters)) {
        try {
          subscription.callback(message, messageTopic);
        } catch (error) {
          logger.error('Error in subscription callback:', error);
        }
      }
    }
  }

  /**
   * Determine message topic from message type
   */
  private determineMessageTopic(message: WebSocketMessage): string | null {
    switch (message.type) {
      case MessageType.COGNITIVE_PATTERN:
        return DataTopic.COGNITIVE_PATTERNS;
      case MessageType.NEURAL_ACTIVITY:
        return DataTopic.NEURAL_ACTIVITY;
      case MessageType.KNOWLEDGE_GRAPH_UPDATE:
        return DataTopic.KNOWLEDGE_GRAPH;
      case MessageType.SDR_OPERATION:
        return DataTopic.SDR_OPERATIONS;
      case MessageType.MEMORY_METRICS:
        return DataTopic.MEMORY_SYSTEM;
      case MessageType.ATTENTION_FOCUS:
        return DataTopic.ATTENTION_MECHANISM;
      case MessageType.TELEMETRY_DATA:
        return DataTopic.TELEMETRY;
      case MessageType.PERFORMANCE_METRICS:
        return DataTopic.PERFORMANCE;
      default:
        return null;
    }
  }

  /**
   * Check if message passes subscription filters
   */
  private messagePassesFilters(message: WebSocketMessage, filters?: Record<string, any>): boolean {
    if (!filters) return true;

    // Apply the same filter logic as in the router
    for (const [filterKey, filterValue] of Object.entries(filters)) {
      switch (filterKey) {
        case 'messageTypes':
          if (Array.isArray(filterValue) && !filterValue.includes(message.type)) {
            return false;
          }
          break;
        // Add more filter types as needed
      }
    }

    return true;
  }

  /**
   * Handle connection close
   */
  private handleConnectionClose(code: number, reason: Buffer): void {
    this.updateUptime();
    
    logger.info('WebSocket connection closed', { 
      code, 
      reason: reason.toString(),
      reconnectAttempts: this.reconnectAttempts
    });

    this.cleanup();
    
    if (this.config.autoReconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else {
      this.setState(ConnectionState.FAILED);
    }

    this.emit('disconnected', { code, reason: reason.toString() });
  }

  /**
   * Handle connection errors
   */
  private handleConnectionError(error: Error): void {
    this.clientStats.errors++;
    logger.error('WebSocket connection error:', error);
    this.emit('connectionError', error);
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.setState(ConnectionState.RECONNECTING);
    this.reconnectAttempts++;

    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    logger.info('Scheduling reconnection', { 
      attempt: this.reconnectAttempts, 
      delay,
      maxAttempts: this.config.maxReconnectAttempts
    });

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
        this.clientStats.totalReconnections++;
      } catch (error) {
        logger.error('Reconnection failed:', error);
        
        if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else {
          this.setState(ConnectionState.FAILED);
        }
      }
    }, delay);
  }

  /**
   * Send connect message to server
   */
  private async sendConnectMessage(): Promise<void> {
    const connectMessage: ConnectMessage = ProtocolValidator.createMessage(
      MessageType.CONNECT,
      {
        clientId: this.clientId || 'unknown',
        version: '1.0.0',
        capabilities: ['compression', 'subscriptions', 'auto-reconnect']
      }
    );

    await this.sendMessage(connectMessage);
  }

  /**
   * Send subscription to server
   */
  private async sendSubscription(subscription: Subscription): Promise<void> {
    if (!this.clientId) return;

    const subscribeMessage: SubscribeMessage = ProtocolValidator.createMessage(
      MessageType.SUBSCRIBE,
      {
        topics: subscription.topics,
        clientId: this.clientId,
        filters: subscription.filters
      }
    );

    await this.sendMessage(subscribeMessage);
    
    // Add to pending subscriptions
    if (subscription.subscriptionId) {
      this.pendingSubscriptions.set(subscription.subscriptionId, subscription);
    }
  }

  /**
   * Resubscribe to all subscriptions after reconnection
   */
  private async resubscribeAll(): Promise<void> {
    for (const subscription of this.subscriptions.values()) {
      subscription.active = false; // Reset active state
      try {
        await this.sendSubscription(subscription);
      } catch (error) {
        logger.error('Failed to resubscribe:', error);
      }
    }
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    if (this.heartbeatTimer) return;

    this.heartbeatTimer = setInterval(() => {
      if (this.state === ConnectionState.CONNECTED && this.clientId) {
        const heartbeat: HeartbeatMessage = ProtocolValidator.createMessage(
          MessageType.HEARTBEAT,
          {
            clientId: this.clientId
          }
        );

        this.sendMessage(heartbeat).catch(error => {
          logger.error('Failed to send heartbeat:', error);
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Update uptime statistics
   */
  private updateUptime(): void {
    if (this.clientStats.lastConnectedAt > 0) {
      this.clientStats.totalUptime += Date.now() - this.clientStats.lastConnectedAt;
    }
  }

  /**
   * Clean up resources
   */
  private cleanup(): void {
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }

    this.stopHeartbeat();
    this.socket = null;
    this.clientId = null;
    this.serverCapabilities = [];
    
    // Mark all subscriptions as inactive
    for (const subscription of this.subscriptions.values()) {
      subscription.active = false;
    }
  }

  /**
   * Set connection state and emit event
   */
  private setState(newState: ConnectionState): void {
    const oldState = this.state;
    this.state = newState;
    
    if (oldState !== newState) {
      this.emit('stateChange', { oldState, newState });
    }
  }

  /**
   * Get client statistics
   */
  getStats(): typeof this.clientStats & { 
    state: ConnectionState;
    activeSubscriptions: number;
    serverCapabilities: string[];
  } {
    return {
      ...this.clientStats,
      state: this.state,
      activeSubscriptions: Array.from(this.subscriptions.values()).filter(sub => sub.active).length,
      serverCapabilities: this.serverCapabilities
    };
  }

  /**
   * Get connection state
   */
  getState(): ConnectionState {
    return this.state;
  }

  /**
   * Check if client is connected
   */
  isConnected(): boolean {
    return this.state === ConnectionState.CONNECTED;
  }

  /**
   * Get active subscriptions
   */
  getSubscriptions(): Subscription[] {
    return Array.from(this.subscriptions.values());
  }
}

export { logger as clientLogger };