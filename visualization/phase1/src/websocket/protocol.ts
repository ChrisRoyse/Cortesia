/**
 * WebSocket Communication Protocol Definitions for LLMKG Visualization
 * 
 * Defines message types, data structures, and communication patterns
 * for real-time streaming of LLMKG cognitive and neural data.
 */

import { Logger } from '../utils/logger';

const logger = new Logger('WebSocketProtocol');

// Message Types for LLMKG-specific data streaming
export enum MessageType {
  // Connection management
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  HEARTBEAT = 'heartbeat',
  ERROR = 'error',
  
  // Subscription management
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  SUBSCRIPTION_ACK = 'subscription_ack',
  
  // LLMKG data types
  COGNITIVE_PATTERN = 'cognitive_pattern',
  NEURAL_ACTIVITY = 'neural_activity',
  KNOWLEDGE_GRAPH_UPDATE = 'knowledge_graph_update',
  SDR_OPERATION = 'sdr_operation',
  MEMORY_METRICS = 'memory_metrics',
  ATTENTION_FOCUS = 'attention_focus',
  
  // System events
  TELEMETRY_DATA = 'telemetry_data',
  PERFORMANCE_METRICS = 'performance_metrics',
  SYSTEM_STATUS = 'system_status',
  
  // Batch operations
  BATCH_DATA = 'batch_data',
  COMPRESSED_DATA = 'compressed_data'
}

// Topic definitions for subscription system
export enum DataTopic {
  COGNITIVE_PATTERNS = 'cognitive.patterns',
  NEURAL_ACTIVITY = 'neural.activity',
  KNOWLEDGE_GRAPH = 'knowledge.graph',
  SDR_OPERATIONS = 'sdr.operations',
  MEMORY_SYSTEM = 'memory.system',
  ATTENTION_MECHANISM = 'attention.mechanism',
  TELEMETRY = 'telemetry.all',
  PERFORMANCE = 'performance.all',
  SYSTEM = 'system.all'
}

// Base message interface
export interface BaseMessage {
  id: string;
  type: MessageType;
  timestamp: number;
  source?: string;
  compressed?: boolean;
}

// Connection messages
export interface ConnectMessage extends BaseMessage {
  type: MessageType.CONNECT;
  clientId: string;
  version: string;
  capabilities: string[];
}

export interface HeartbeatMessage extends BaseMessage {
  type: MessageType.HEARTBEAT;
  clientId: string;
}

export interface ErrorMessage extends BaseMessage {
  type: MessageType.ERROR;
  error: {
    code: string;
    message: string;
    details?: any;
  };
}

// Subscription messages
export interface SubscribeMessage extends BaseMessage {
  type: MessageType.SUBSCRIBE;
  topics: string[];
  clientId: string;
  filters?: Record<string, any>;
}

export interface UnsubscribeMessage extends BaseMessage {
  type: MessageType.UNSUBSCRIBE;
  topics: string[];
  clientId: string;
}

export interface SubscriptionAckMessage extends BaseMessage {
  type: MessageType.SUBSCRIPTION_ACK;
  topics: string[];
  status: 'success' | 'error';
  message?: string;
}

// LLMKG-specific data messages
export interface CognitivePatternMessage extends BaseMessage {
  type: MessageType.COGNITIVE_PATTERN;
  data: {
    patternId: string;
    patternType: string;
    activation: number;
    confidence: number;
    context: Record<string, any>;
    hierarchy?: {
      level: number;
      parent?: string;
      children?: string[];
    };
  };
}

export interface NeuralActivityMessage extends BaseMessage {
  type: MessageType.NEURAL_ACTIVITY;
  data: {
    nodeId: string;
    activityType: 'activation' | 'inhibition' | 'propagation';
    intensity: number;
    connections: {
      incoming: string[];
      outgoing: string[];
    };
    spatialLocation?: {
      x: number;
      y: number;
      z?: number;
    };
  };
}

export interface KnowledgeGraphUpdateMessage extends BaseMessage {
  type: MessageType.KNOWLEDGE_GRAPH_UPDATE;
  data: {
    updateType: 'add' | 'update' | 'remove';
    entityType: 'node' | 'edge';
    entity: {
      id: string;
      type: string;
      properties: Record<string, any>;
      relationships?: {
        source: string;
        target: string;
        type: string;
        weight?: number;
      }[];
    };
  };
}

export interface SDROperationMessage extends BaseMessage {
  type: MessageType.SDR_OPERATION;
  data: {
    operationId: string;
    operationType: 'encode' | 'decode' | 'transform' | 'merge';
    sdrData: {
      dimensions: number;
      sparsity: number;
      activeBits: number[];
      semanticWeight: number;
    };
    transformation?: {
      input: number[];
      output: number[];
      algorithm: string;
    };
  };
}

export interface MemoryMetricsMessage extends BaseMessage {
  type: MessageType.MEMORY_METRICS;
  data: {
    memoryType: 'working' | 'episodic' | 'semantic' | 'procedural';
    metrics: {
      capacity: number;
      utilization: number;
      retrievalLatency: number;
      storageEfficiency: number;
      compressionRatio?: number;
    };
    operations: {
      reads: number;
      writes: number;
      evictions: number;
    };
  };
}

export interface AttentionFocusMessage extends BaseMessage {
  type: MessageType.ATTENTION_FOCUS;
  data: {
    focusType: 'selective' | 'divided' | 'sustained';
    targets: {
      id: string;
      type: string;
      weight: number;
      coordinates?: {
        x: number;
        y: number;
        z?: number;
      };
    }[];
    intensity: number;
    duration: number;
    context: Record<string, any>;
  };
}

// Batch and compressed message types
export interface BatchDataMessage extends BaseMessage {
  type: MessageType.BATCH_DATA;
  data: {
    batchId: string;
    messageCount: number;
    messages: BaseMessage[];
    timeRange: {
      start: number;
      end: number;
    };
  };
}

export interface CompressedDataMessage extends BaseMessage {
  type: MessageType.COMPRESSED_DATA;
  compressed: true;
  data: {
    algorithm: 'gzip' | 'lz4' | 'snappy';
    originalSize: number;
    compressedSize: number;
    payload: string; // Base64 encoded compressed data
  };
}

// Union type for all possible messages
export type WebSocketMessage = 
  | ConnectMessage
  | HeartbeatMessage
  | ErrorMessage
  | SubscribeMessage
  | UnsubscribeMessage
  | SubscriptionAckMessage
  | CognitivePatternMessage
  | NeuralActivityMessage
  | KnowledgeGraphUpdateMessage
  | SDROperationMessage
  | MemoryMetricsMessage
  | AttentionFocusMessage
  | BatchDataMessage
  | CompressedDataMessage;

// Message validation
export class ProtocolValidator {
  static validateMessage(message: any): message is WebSocketMessage {
    try {
      if (!message || typeof message !== 'object') {
        return false;
      }

      const required = ['id', 'type', 'timestamp'];
      for (const field of required) {
        if (!(field in message)) {
          logger.warn(`Missing required field: ${field}`);
          return false;
        }
      }

      if (!Object.values(MessageType).includes(message.type)) {
        logger.warn(`Invalid message type: ${message.type}`);
        return false;
      }

      return true;
    } catch (error) {
      logger.error('Message validation error:', error);
      return false;
    }
  }

  static createMessage<T extends WebSocketMessage>(
    type: MessageType,
    data: Omit<T, 'id' | 'type' | 'timestamp'>,
    source?: string
  ): T {
    return {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      timestamp: Date.now(),
      source,
      ...data
    } as T;
  }
}

// Protocol configuration
export interface ProtocolConfig {
  version: string;
  heartbeatInterval: number;
  messageTimeout: number;
  maxMessageSize: number;
  compressionThreshold: number;
  batchSize: number;
  supportedTopics: string[];
}

export const DEFAULT_PROTOCOL_CONFIG: ProtocolConfig = {
  version: '1.0.0',
  heartbeatInterval: 30000, // 30 seconds
  messageTimeout: 5000, // 5 seconds
  maxMessageSize: 1024 * 1024, // 1MB
  compressionThreshold: 1024, // 1KB
  batchSize: 100,
  supportedTopics: Object.values(DataTopic)
};

// Topic subscription utilities
export class TopicManager {
  static isValidTopic(topic: string): boolean {
    return Object.values(DataTopic).includes(topic as DataTopic);
  }

  static matchesTopic(topic: string, pattern: string): boolean {
    // Support wildcard matching (e.g., 'cognitive.*' matches 'cognitive.patterns')
    const regex = new RegExp(pattern.replace(/\*/g, '.*'));
    return regex.test(topic);
  }

  static getTopicHierarchy(topic: string): string[] {
    const parts = topic.split('.');
    const hierarchy: string[] = [];
    
    for (let i = 0; i < parts.length; i++) {
      hierarchy.push(parts.slice(0, i + 1).join('.'));
    }
    
    return hierarchy;
  }
}

export { logger as protocolLogger };