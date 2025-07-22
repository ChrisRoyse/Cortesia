/**
 * WebSocket Connection Orchestration Manager
 * 
 * Provides high-level orchestration of WebSocket server and client connections,
 * integrating with telemetry, data collection, and MCP systems.
 */

import { EventEmitter } from 'events';
import { WebSocketServer, ServerConfig, serverLogger } from './server';
import { DashboardWebSocketClient, ClientConfig, ConnectionState, clientLogger } from './client';
import { MessageRouter, routerLogger } from './router';
import { MessageBuffer, MessagePriority, bufferLogger } from './buffer';
import { 
  WebSocketMessage, 
  MessageType, 
  DataTopic,
  CognitivePatternMessage,
  NeuralActivityMessage,
  KnowledgeGraphUpdateMessage,
  SDROperationMessage,
  MemoryMetricsMessage,
  AttentionFocusMessage,
  ProtocolValidator
} from './protocol';
import { TelemetryCollector } from '../telemetry/collector';
import { DataCollectionAgent } from '../collectors/agent';
import { MCPClient } from '../mcp/client';
import { Logger } from '../utils/logger';

const logger = new Logger('WebSocketManager');

export interface ManagerConfig {
  server: Partial<ServerConfig>;
  client?: Partial<ClientConfig>;
  enableServer: boolean;
  enableClient: boolean;
  enableTelemetryIntegration: boolean;
  enableDataCollectionIntegration: boolean;
  enableMCPIntegration: boolean;
  dataStreamingInterval: number;
  batchingEnabled: boolean;
  compressionEnabled: boolean;
}

export const DEFAULT_MANAGER_CONFIG: ManagerConfig = {
  server: {
    port: 8080,
    enableCompression: true,
    enableBuffering: true
  },
  enableServer: true,
  enableClient: false,
  enableTelemetryIntegration: true,
  enableDataCollectionIntegration: true,
  enableMCPIntegration: true,
  dataStreamingInterval: 100, // 100ms
  batchingEnabled: true,
  compressionEnabled: true
};

export interface DataStreamConfig {
  cognitivePatterns: {
    enabled: boolean;
    updateInterval: number;
    minConfidence: number;
  };
  neuralActivity: {
    enabled: boolean;
    updateInterval: number;
    minIntensity: number;
  };
  knowledgeGraph: {
    enabled: boolean;
    batchUpdates: boolean;
    maxBatchSize: number;
  };
  sdrOperations: {
    enabled: boolean;
    trackTransformations: boolean;
  };
  memoryMetrics: {
    enabled: boolean;
    updateInterval: number;
  };
  attentionMechanisms: {
    enabled: boolean;
    trackFocusChanges: boolean;
    minIntensity: number;
  };
}

export const DEFAULT_STREAM_CONFIG: DataStreamConfig = {
  cognitivePatterns: {
    enabled: true,
    updateInterval: 50, // 50ms - high frequency for real-time visualization
    minConfidence: 0.3
  },
  neuralActivity: {
    enabled: true,
    updateInterval: 25, // 25ms - very high frequency for neural visualization
    minIntensity: 0.1
  },
  knowledgeGraph: {
    enabled: true,
    batchUpdates: true,
    maxBatchSize: 50
  },
  sdrOperations: {
    enabled: true,
    trackTransformations: true
  },
  memoryMetrics: {
    enabled: true,
    updateInterval: 1000 // 1 second - less frequent for performance metrics
  },
  attentionMechanisms: {
    enabled: true,
    trackFocusChanges: true,
    minIntensity: 0.2
  }
};

export class WebSocketManager extends EventEmitter {
  private server: WebSocketServer | null = null;
  private client: DashboardWebSocketClient | null = null;
  private config: ManagerConfig;
  private streamConfig: DataStreamConfig;
  
  // External system integrations
  private telemetryCollector: TelemetryCollector | null = null;
  private dataCollectionAgent: DataCollectionAgent | null = null;
  private mcpClient: MCPClient | null = null;
  
  // Data streaming timers
  private streamingTimers = new Map<string, NodeJS.Timeout>();
  private isStreaming = false;
  
  // Statistics and monitoring
  private managerStats = {
    totalDataStreamed: 0,
    totalClientsServed: 0,
    streamingUptime: 0,
    errors: 0,
    lastStreamStart: 0,
    dataTypeStats: {
      cognitivePatterns: 0,
      neuralActivity: 0,
      knowledgeGraph: 0,
      sdrOperations: 0,
      memoryMetrics: 0,
      attentionFocus: 0
    }
  };

  constructor(
    config: Partial<ManagerConfig> = {},
    streamConfig: Partial<DataStreamConfig> = {}
  ) {
    super();
    this.config = { ...DEFAULT_MANAGER_CONFIG, ...config };
    this.streamConfig = { ...DEFAULT_STREAM_CONFIG, ...streamConfig };
    
    this.setupEventHandlers();
  }

  /**
   * Setup event handlers for the manager
   */
  private setupEventHandlers(): void {
    this.on('serverStarted', () => {
      logger.info('WebSocket server started successfully');
      if (this.config.enableTelemetryIntegration) {
        this.initializeTelemetryIntegration();
      }
      if (this.config.enableDataCollectionIntegration) {
        this.initializeDataCollectionIntegration();
      }
      if (this.config.enableMCPIntegration) {
        this.initializeMCPIntegration();
      }
    });

    this.on('clientConnected', () => {
      logger.info('WebSocket client connected successfully');
      this.startDataStreaming();
    });

    this.on('clientDisconnected', () => {
      logger.info('WebSocket client disconnected');
      this.stopDataStreaming();
    });

    this.on('error', (error) => {
      this.managerStats.errors++;
      logger.error('WebSocket manager error:', error);
    });
  }

  /**
   * Initialize the WebSocket manager
   */
  async initialize(): Promise<void> {
    try {
      logger.info('Initializing WebSocket manager', { config: this.config });

      // Initialize server if enabled
      if (this.config.enableServer) {
        await this.initializeServer();
      }

      // Initialize client if enabled
      if (this.config.enableClient && this.config.client) {
        await this.initializeClient();
      }

      logger.info('WebSocket manager initialized successfully');
      this.emit('initialized');

    } catch (error) {
      logger.error('Failed to initialize WebSocket manager:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Initialize WebSocket server
   */
  private async initializeServer(): Promise<void> {
    if (this.server) {
      logger.warn('WebSocket server already initialized');
      return;
    }

    this.server = new WebSocketServer(this.config.server);

    // Setup server event handlers
    this.server.on('started', () => {
      this.emit('serverStarted');
    });

    this.server.on('clientConnected', (client) => {
      this.managerStats.totalClientsServed++;
      logger.info('Client connected to server', { clientId: client.id });
    });

    this.server.on('clientDisconnected', ({ clientId }) => {
      logger.info('Client disconnected from server', { clientId });
    });

    this.server.on('error', (error) => {
      this.emit('error', error);
    });

    // Start the server
    await this.server.start();
  }

  /**
   * Initialize WebSocket client
   */
  private async initializeClient(): Promise<void> {
    if (this.client) {
      logger.warn('WebSocket client already initialized');
      return;
    }

    this.client = new DashboardWebSocketClient(this.config.client);

    // Setup client event handlers
    this.client.on('connected', () => {
      this.emit('clientConnected');
    });

    this.client.on('disconnected', () => {
      this.emit('clientDisconnected');
    });

    this.client.on('stateChange', ({ newState }) => {
      logger.info('Client state changed', { newState });
    });

    this.client.on('error', (error) => {
      this.emit('error', error);
    });

    // Connect the client
    await this.client.connect();
  }

  /**
   * Initialize telemetry system integration
   */
  private initializeTelemetryIntegration(): void {
    try {
      this.telemetryCollector = new TelemetryCollector({
        enableRealTimeStreaming: true,
        streamingInterval: this.config.dataStreamingInterval
      });

      // Setup telemetry data streaming
      this.telemetryCollector.on('dataCollected', (data) => {
        this.streamTelemetryData(data);
      });

      this.telemetryCollector.start();
      logger.info('Telemetry integration initialized');

    } catch (error) {
      logger.error('Failed to initialize telemetry integration:', error);
    }
  }

  /**
   * Initialize data collection integration
   */
  private initializeDataCollectionIntegration(): void {
    try {
      this.dataCollectionAgent = new DataCollectionAgent({
        enableRealTimeCollection: true,
        collectionInterval: this.config.dataStreamingInterval
      });

      // Setup data collection streaming
      this.dataCollectionAgent.on('cognitivePatternDetected', (pattern) => {
        this.streamCognitivePattern(pattern);
      });

      this.dataCollectionAgent.on('neuralActivityDetected', (activity) => {
        this.streamNeuralActivity(activity);
      });

      this.dataCollectionAgent.on('knowledgeGraphUpdate', (update) => {
        this.streamKnowledgeGraphUpdate(update);
      });

      this.dataCollectionAgent.on('sdrOperationExecuted', (operation) => {
        this.streamSDROperation(operation);
      });

      this.dataCollectionAgent.on('memoryMetricsUpdated', (metrics) => {
        this.streamMemoryMetrics(metrics);
      });

      this.dataCollectionAgent.on('attentionFocusChanged', (focus) => {
        this.streamAttentionFocus(focus);
      });

      this.dataCollectionAgent.start();
      logger.info('Data collection integration initialized');

    } catch (error) {
      logger.error('Failed to initialize data collection integration:', error);
    }
  }

  /**
   * Initialize MCP client integration
   */
  private initializeMCPIntegration(): void {
    try {
      this.mcpClient = new MCPClient({
        enableRealtimeUpdates: true
      });

      // Setup MCP data streaming
      this.mcpClient.on('dataUpdate', (data) => {
        this.streamMCPData(data);
      });

      this.mcpClient.connect();
      logger.info('MCP integration initialized');

    } catch (error) {
      logger.error('Failed to initialize MCP integration:', error);
    }
  }

  /**
   * Start data streaming to connected clients
   */
  private startDataStreaming(): void {
    if (this.isStreaming) {
      return;
    }

    this.isStreaming = true;
    this.managerStats.lastStreamStart = Date.now();

    logger.info('Starting data streaming', { streamConfig: this.streamConfig });

    // Start streaming for each enabled data type
    if (this.streamConfig.cognitivePatterns.enabled) {
      this.startCognitivePatternStreaming();
    }

    if (this.streamConfig.neuralActivity.enabled) {
      this.startNeuralActivityStreaming();
    }

    if (this.streamConfig.memoryMetrics.enabled) {
      this.startMemoryMetricsStreaming();
    }

    this.emit('streamingStarted');
  }

  /**
   * Stop data streaming
   */
  private stopDataStreaming(): void {
    if (!this.isStreaming) {
      return;
    }

    this.isStreaming = false;

    // Clear all streaming timers
    for (const [name, timer] of this.streamingTimers.entries()) {
      clearInterval(timer);
      logger.debug('Stopped streaming timer', { name });
    }
    this.streamingTimers.clear();

    // Update uptime stats
    if (this.managerStats.lastStreamStart > 0) {
      this.managerStats.streamingUptime += Date.now() - this.managerStats.lastStreamStart;
    }

    logger.info('Data streaming stopped');
    this.emit('streamingStopped');
  }

  /**
   * Start cognitive pattern streaming
   */
  private startCognitivePatternStreaming(): void {
    const config = this.streamConfig.cognitivePatterns;
    const timer = setInterval(() => {
      if (this.dataCollectionAgent) {
        // Request latest cognitive patterns from data collection agent
        this.dataCollectionAgent.getCognitivePatterns({
          minConfidence: config.minConfidence
        }).then(patterns => {
          patterns.forEach(pattern => this.streamCognitivePattern(pattern));
        }).catch(error => {
          logger.error('Error streaming cognitive patterns:', error);
        });
      }
    }, config.updateInterval);

    this.streamingTimers.set('cognitivePatterns', timer);
  }

  /**
   * Start neural activity streaming
   */
  private startNeuralActivityStreaming(): void {
    const config = this.streamConfig.neuralActivity;
    const timer = setInterval(() => {
      if (this.dataCollectionAgent) {
        // Request latest neural activity from data collection agent
        this.dataCollectionAgent.getNeuralActivity({
          minIntensity: config.minIntensity
        }).then(activities => {
          activities.forEach(activity => this.streamNeuralActivity(activity));
        }).catch(error => {
          logger.error('Error streaming neural activity:', error);
        });
      }
    }, config.updateInterval);

    this.streamingTimers.set('neuralActivity', timer);
  }

  /**
   * Start memory metrics streaming
   */
  private startMemoryMetricsStreaming(): void {
    const config = this.streamConfig.memoryMetrics;
    const timer = setInterval(() => {
      if (this.dataCollectionAgent) {
        // Request latest memory metrics from data collection agent
        this.dataCollectionAgent.getMemoryMetrics().then(metrics => {
          metrics.forEach(metric => this.streamMemoryMetrics(metric));
        }).catch(error => {
          logger.error('Error streaming memory metrics:', error);
        });
      }
    }, config.updateInterval);

    this.streamingTimers.set('memoryMetrics', timer);
  }

  /**
   * Stream cognitive pattern data
   */
  private streamCognitivePattern(pattern: any): void {
    if (!this.server) return;

    const message: CognitivePatternMessage = ProtocolValidator.createMessage(
      MessageType.COGNITIVE_PATTERN,
      {
        data: {
          patternId: pattern.id,
          patternType: pattern.type,
          activation: pattern.activation,
          confidence: pattern.confidence,
          context: pattern.context,
          hierarchy: pattern.hierarchy
        }
      }
    );

    this.server.broadcast(DataTopic.COGNITIVE_PATTERNS, message);
    this.managerStats.dataTypeStats.cognitivePatterns++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream neural activity data
   */
  private streamNeuralActivity(activity: any): void {
    if (!this.server) return;

    const message: NeuralActivityMessage = ProtocolValidator.createMessage(
      MessageType.NEURAL_ACTIVITY,
      {
        data: {
          nodeId: activity.nodeId,
          activityType: activity.type,
          intensity: activity.intensity,
          connections: activity.connections,
          spatialLocation: activity.location
        }
      }
    );

    this.server.broadcast(DataTopic.NEURAL_ACTIVITY, message);
    this.managerStats.dataTypeStats.neuralActivity++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream knowledge graph updates
   */
  private streamKnowledgeGraphUpdate(update: any): void {
    if (!this.server) return;

    const message: KnowledgeGraphUpdateMessage = ProtocolValidator.createMessage(
      MessageType.KNOWLEDGE_GRAPH_UPDATE,
      {
        data: {
          updateType: update.type,
          entityType: update.entityType,
          entity: update.entity
        }
      }
    );

    this.server.broadcast(DataTopic.KNOWLEDGE_GRAPH, message);
    this.managerStats.dataTypeStats.knowledgeGraph++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream SDR operation data
   */
  private streamSDROperation(operation: any): void {
    if (!this.server) return;

    const message: SDROperationMessage = ProtocolValidator.createMessage(
      MessageType.SDR_OPERATION,
      {
        data: {
          operationId: operation.id,
          operationType: operation.type,
          sdrData: operation.sdrData,
          transformation: operation.transformation
        }
      }
    );

    this.server.broadcast(DataTopic.SDR_OPERATIONS, message);
    this.managerStats.dataTypeStats.sdrOperations++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream memory metrics
   */
  private streamMemoryMetrics(metrics: any): void {
    if (!this.server) return;

    const message: MemoryMetricsMessage = ProtocolValidator.createMessage(
      MessageType.MEMORY_METRICS,
      {
        data: {
          memoryType: metrics.type,
          metrics: metrics.metrics,
          operations: metrics.operations
        }
      }
    );

    this.server.broadcast(DataTopic.MEMORY_SYSTEM, message);
    this.managerStats.dataTypeStats.memoryMetrics++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream attention focus changes
   */
  private streamAttentionFocus(focus: any): void {
    if (!this.server) return;

    const message: AttentionFocusMessage = ProtocolValidator.createMessage(
      MessageType.ATTENTION_FOCUS,
      {
        data: {
          focusType: focus.type,
          targets: focus.targets,
          intensity: focus.intensity,
          duration: focus.duration,
          context: focus.context
        }
      }
    );

    this.server.broadcast(DataTopic.ATTENTION_MECHANISM, message);
    this.managerStats.dataTypeStats.attentionFocus++;
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream telemetry data
   */
  private streamTelemetryData(data: any): void {
    if (!this.server) return;

    this.server.broadcast(DataTopic.TELEMETRY, data);
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Stream MCP data
   */
  private streamMCPData(data: any): void {
    if (!this.server) return;

    this.server.broadcast(DataTopic.SYSTEM, data);
    this.managerStats.totalDataStreamed++;
  }

  /**
   * Subscribe client to specific topics
   */
  async subscribeToTopics(
    topics: string[], 
    callback: (message: WebSocketMessage, topic: string) => void
  ): Promise<void> {
    if (!this.client) {
      throw new Error('Client not initialized');
    }

    await this.client.subscribe(topics, callback);
  }

  /**
   * Unsubscribe client from topics
   */
  async unsubscribeFromTopics(topics: string[]): Promise<void> {
    if (!this.client) {
      throw new Error('Client not initialized');
    }

    await this.client.unsubscribe(topics);
  }

  /**
   * Get comprehensive statistics
   */
  getStats(): {
    manager: typeof this.managerStats;
    server?: ReturnType<WebSocketServer['getStats']>;
    client?: ReturnType<DashboardWebSocketClient['getStats']>;
    streaming: {
      isActive: boolean;
      activeTimers: string[];
      uptime: number;
    };
  } {
    const stats = {
      manager: this.managerStats,
      streaming: {
        isActive: this.isStreaming,
        activeTimers: Array.from(this.streamingTimers.keys()),
        uptime: this.managerStats.streamingUptime + (
          this.isStreaming && this.managerStats.lastStreamStart > 0
            ? Date.now() - this.managerStats.lastStreamStart
            : 0
        )
      }
    };

    if (this.server) {
      (stats as any).server = this.server.getStats();
    }

    if (this.client) {
      (stats as any).client = this.client.getStats();
    }

    return stats;
  }

  /**
   * Update streaming configuration
   */
  updateStreamConfig(config: Partial<DataStreamConfig>): void {
    this.streamConfig = { ...this.streamConfig, ...config };
    
    if (this.isStreaming) {
      this.stopDataStreaming();
      this.startDataStreaming();
    }

    logger.info('Stream configuration updated', { newConfig: this.streamConfig });
  }

  /**
   * Shutdown the WebSocket manager
   */
  async shutdown(): Promise<void> {
    try {
      logger.info('Shutting down WebSocket manager');

      // Stop data streaming
      this.stopDataStreaming();

      // Stop external system integrations
      if (this.telemetryCollector) {
        this.telemetryCollector.stop();
      }

      if (this.dataCollectionAgent) {
        this.dataCollectionAgent.stop();
      }

      if (this.mcpClient) {
        this.mcpClient.disconnect();
      }

      // Shutdown server
      if (this.server) {
        await this.server.stop();
      }

      // Disconnect client
      if (this.client) {
        this.client.disconnect();
      }

      logger.info('WebSocket manager shutdown completed');
      this.emit('shutdown');

    } catch (error) {
      logger.error('Error during WebSocket manager shutdown:', error);
      throw error;
    }
  }

  /**
   * Check if manager is active
   */
  isActive(): boolean {
    return (this.server?.isActive() || false) || (this.client?.isConnected() || false);
  }

  /**
   * Get current configuration
   */
  getConfig(): { manager: ManagerConfig; streaming: DataStreamConfig } {
    return {
      manager: this.config,
      streaming: this.streamConfig
    };
  }
}

export { logger as managerLogger };