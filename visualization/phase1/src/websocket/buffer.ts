/**
 * Data Buffering and Compression System for WebSocket Communication
 * 
 * Handles high-frequency data buffering, compression, and efficient
 * message batching for optimal WebSocket performance.
 */

import * as zlib from 'zlib';
import { promisify } from 'util';
import { EventEmitter } from 'events';
import { 
  WebSocketMessage, 
  MessageType, 
  BatchDataMessage, 
  CompressedDataMessage,
  ProtocolValidator,
  DEFAULT_PROTOCOL_CONFIG 
} from './protocol';
import { Logger } from '../utils/logger';

const logger = new Logger('WebSocketBuffer');

// Compression utilities
const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

export interface BufferConfig {
  maxBufferSize: number;
  flushInterval: number;
  compressionThreshold: number;
  compressionAlgorithm: 'gzip' | 'none';
  enableBatching: boolean;
  maxBatchSize: number;
  priorityQueues: boolean;
}

export const DEFAULT_BUFFER_CONFIG: BufferConfig = {
  maxBufferSize: 10000, // 10k messages
  flushInterval: 100, // 100ms
  compressionThreshold: 1024, // 1KB
  compressionAlgorithm: 'gzip',
  enableBatching: true,
  maxBatchSize: 100,
  priorityQueues: true
};

export enum MessagePriority {
  CRITICAL = 0,    // System errors, connection issues
  HIGH = 1,        // Real-time cognitive patterns, neural activity
  MEDIUM = 2,      // Knowledge graph updates, SDR operations  
  LOW = 3,         // Memory metrics, performance data
  BATCH = 4        // Bulk data, historical information
}

interface BufferedMessage {
  message: WebSocketMessage;
  priority: MessagePriority;
  timestamp: number;
  topic?: string;
  clientId?: string;
}

export class MessageBuffer extends EventEmitter {
  private buffer: Map<MessagePriority, BufferedMessage[]> = new Map();
  private flushTimer: NodeJS.Timeout | null = null;
  private config: BufferConfig;
  private isActive = false;
  private totalBufferedMessages = 0;
  private compressionStats = {
    messagesCompressed: 0,
    originalSize: 0,
    compressedSize: 0
  };

  constructor(config: Partial<BufferConfig> = {}) {
    super();
    this.config = { ...DEFAULT_BUFFER_CONFIG, ...config };
    this.initializePriorityQueues();
  }

  private initializePriorityQueues(): void {
    for (const priority of Object.values(MessagePriority)) {
      if (typeof priority === 'number') {
        this.buffer.set(priority, []);
      }
    }
  }

  /**
   * Start the buffer system
   */
  start(): void {
    if (this.isActive) {
      return;
    }

    this.isActive = true;
    this.startFlushTimer();
    logger.info('Message buffer started', { config: this.config });
  }

  /**
   * Stop the buffer system and flush remaining messages
   */
  stop(): Promise<void> {
    return new Promise((resolve) => {
      this.isActive = false;
      
      if (this.flushTimer) {
        clearInterval(this.flushTimer);
        this.flushTimer = null;
      }

      // Flush remaining messages
      this.flushMessages().then(() => {
        logger.info('Message buffer stopped', {
          totalProcessed: this.compressionStats.messagesCompressed,
          compressionRatio: this.getCompressionRatio()
        });
        resolve();
      });
    });
  }

  /**
   * Add a message to the buffer
   */
  addMessage(
    message: WebSocketMessage, 
    priority: MessagePriority = MessagePriority.MEDIUM,
    topic?: string,
    clientId?: string
  ): void {
    if (!this.isActive) {
      logger.warn('Buffer is not active, message discarded');
      return;
    }

    const bufferedMessage: BufferedMessage = {
      message,
      priority,
      timestamp: Date.now(),
      topic,
      clientId
    };

    const queue = this.buffer.get(priority);
    if (queue) {
      queue.push(bufferedMessage);
      this.totalBufferedMessages++;

      // Check for buffer overflow
      if (this.totalBufferedMessages >= this.config.maxBufferSize) {
        logger.warn('Buffer overflow detected, flushing immediately');
        this.flushMessages();
      }

      this.emit('messageAdded', { priority, queueSize: queue.length });
    }
  }

  /**
   * Get messages for a specific topic
   */
  getMessagesForTopic(topic: string): BufferedMessage[] {
    const messages: BufferedMessage[] = [];
    
    for (const queue of this.buffer.values()) {
      messages.push(...queue.filter(msg => msg.topic === topic));
    }

    return messages.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Get messages for a specific client
   */
  getMessagesForClient(clientId: string): BufferedMessage[] {
    const messages: BufferedMessage[] = [];
    
    for (const queue of this.buffer.values()) {
      messages.push(...queue.filter(msg => msg.clientId === clientId));
    }

    return messages.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Start the periodic flush timer
   */
  private startFlushTimer(): void {
    if (this.flushTimer) {
      return;
    }

    this.flushTimer = setInterval(() => {
      if (this.totalBufferedMessages > 0) {
        this.flushMessages();
      }
    }, this.config.flushInterval);
  }

  /**
   * Flush buffered messages
   */
  private async flushMessages(): Promise<void> {
    if (this.totalBufferedMessages === 0) {
      return;
    }

    try {
      // Process messages by priority
      for (const [priority, queue] of this.buffer.entries()) {
        if (queue.length === 0) continue;

        // Extract messages to process
        const messages = queue.splice(0, this.config.maxBatchSize);
        this.totalBufferedMessages -= messages.length;

        // Process messages based on priority
        if (priority === MessagePriority.CRITICAL) {
          // Send critical messages immediately, one by one
          for (const bufferedMsg of messages) {
            await this.emitProcessedMessage(bufferedMsg.message, bufferedMsg.topic, bufferedMsg.clientId);
          }
        } else if (this.config.enableBatching && messages.length > 1) {
          // Batch non-critical messages
          const batchedMessage = await this.createBatchMessage(messages);
          await this.emitProcessedMessage(batchedMessage);
        } else {
          // Send individual messages
          for (const bufferedMsg of messages) {
            await this.emitProcessedMessage(bufferedMsg.message, bufferedMsg.topic, bufferedMsg.clientId);
          }
        }
      }
    } catch (error) {
      logger.error('Error flushing messages:', error);
      this.emit('error', error);
    }
  }

  /**
   * Create a batch message from multiple buffered messages
   */
  private async createBatchMessage(messages: BufferedMessage[]): Promise<WebSocketMessage> {
    const timeRange = {
      start: Math.min(...messages.map(m => m.timestamp)),
      end: Math.max(...messages.map(m => m.timestamp))
    };

    const batchMessage: BatchDataMessage = ProtocolValidator.createMessage(
      MessageType.BATCH_DATA,
      {
        data: {
          batchId: `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          messageCount: messages.length,
          messages: messages.map(m => m.message),
          timeRange
        }
      }
    );

    // Check if compression is beneficial
    const serialized = JSON.stringify(batchMessage);
    if (serialized.length >= this.config.compressionThreshold && 
        this.config.compressionAlgorithm === 'gzip') {
      return await this.compressMessage(batchMessage);
    }

    return batchMessage;
  }

  /**
   * Compress a message using the configured algorithm
   */
  private async compressMessage(message: WebSocketMessage): Promise<CompressedDataMessage> {
    try {
      const originalData = JSON.stringify(message);
      const originalSize = Buffer.byteLength(originalData, 'utf8');

      let compressedData: Buffer;
      let algorithm: 'gzip' | 'lz4' | 'snappy' = 'gzip';

      switch (this.config.compressionAlgorithm) {
        case 'gzip':
          compressedData = await gzip(originalData);
          break;
        default:
          throw new Error(`Unsupported compression algorithm: ${this.config.compressionAlgorithm}`);
      }

      const compressedSize = compressedData.length;
      const payload = compressedData.toString('base64');

      // Update compression stats
      this.compressionStats.messagesCompressed++;
      this.compressionStats.originalSize += originalSize;
      this.compressionStats.compressedSize += compressedSize;

      const compressedMessage: CompressedDataMessage = ProtocolValidator.createMessage(
        MessageType.COMPRESSED_DATA,
        {
          compressed: true,
          data: {
            algorithm,
            originalSize,
            compressedSize,
            payload
          }
        }
      );

      logger.debug('Message compressed', {
        originalSize,
        compressedSize,
        ratio: (compressedSize / originalSize * 100).toFixed(2) + '%'
      });

      return compressedMessage;
    } catch (error) {
      logger.error('Compression failed:', error);
      throw error;
    }
  }

  /**
   * Decompress a compressed message
   */
  static async decompressMessage(compressedMessage: CompressedDataMessage): Promise<WebSocketMessage> {
    try {
      const { algorithm, payload } = compressedMessage.data;
      const compressedData = Buffer.from(payload, 'base64');

      let decompressedData: Buffer;

      switch (algorithm) {
        case 'gzip':
          decompressedData = await gunzip(compressedData);
          break;
        default:
          throw new Error(`Unsupported compression algorithm: ${algorithm}`);
      }

      const originalMessage = JSON.parse(decompressedData.toString('utf8'));
      
      if (!ProtocolValidator.validateMessage(originalMessage)) {
        throw new Error('Decompressed message validation failed');
      }

      return originalMessage;
    } catch (error) {
      logger.error('Decompression failed:', error);
      throw error;
    }
  }

  /**
   * Emit a processed message
   */
  private async emitProcessedMessage(
    message: WebSocketMessage, 
    topic?: string, 
    clientId?: string
  ): Promise<void> {
    this.emit('messageReady', { message, topic, clientId });
  }

  /**
   * Get current buffer statistics
   */
  getStats(): {
    totalBuffered: number;
    queueSizes: Record<MessagePriority, number>;
    compressionStats: typeof this.compressionStats & { ratio: string };
    config: BufferConfig;
  } {
    const queueSizes: Record<MessagePriority, number> = {};
    
    for (const [priority, queue] of this.buffer.entries()) {
      queueSizes[priority as MessagePriority] = queue.length;
    }

    return {
      totalBuffered: this.totalBufferedMessages,
      queueSizes,
      compressionStats: {
        ...this.compressionStats,
        ratio: this.getCompressionRatio()
      },
      config: this.config
    };
  }

  /**
   * Get compression ratio as percentage
   */
  private getCompressionRatio(): string {
    if (this.compressionStats.originalSize === 0) {
      return '0%';
    }
    
    const ratio = (this.compressionStats.compressedSize / this.compressionStats.originalSize) * 100;
    return ratio.toFixed(2) + '%';
  }

  /**
   * Clear all buffered messages
   */
  clear(): void {
    for (const queue of this.buffer.values()) {
      queue.length = 0;
    }
    this.totalBufferedMessages = 0;
    logger.info('Message buffer cleared');
  }

  /**
   * Determine message priority based on message type
   */
  static getMessagePriority(message: WebSocketMessage): MessagePriority {
    switch (message.type) {
      case MessageType.ERROR:
      case MessageType.DISCONNECT:
        return MessagePriority.CRITICAL;
      
      case MessageType.COGNITIVE_PATTERN:
      case MessageType.NEURAL_ACTIVITY:
      case MessageType.ATTENTION_FOCUS:
        return MessagePriority.HIGH;
      
      case MessageType.KNOWLEDGE_GRAPH_UPDATE:
      case MessageType.SDR_OPERATION:
        return MessagePriority.MEDIUM;
      
      case MessageType.MEMORY_METRICS:
      case MessageType.PERFORMANCE_METRICS:
      case MessageType.TELEMETRY_DATA:
        return MessagePriority.LOW;
      
      default:
        return MessagePriority.MEDIUM;
    }
  }
}

export { logger as bufferLogger };