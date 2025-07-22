/**
 * WebSocket Communication System - Main Export
 * 
 * This module provides a complete WebSocket communication system for
 * LLMKG visualization with real-time data streaming capabilities.
 */

// Core WebSocket components
export { WebSocketServer, ServerConfig, DEFAULT_SERVER_CONFIG, serverLogger } from './server';
export { DashboardWebSocketClient, ClientConfig, DEFAULT_CLIENT_CONFIG, ConnectionState, clientLogger } from './client';
export { MessageRouter, Subscription, RouteMatch, RouteFilter, routerLogger } from './router';
export { MessageBuffer, MessagePriority, BufferConfig, DEFAULT_BUFFER_CONFIG, bufferLogger } from './buffer';
export { WebSocketManager, ManagerConfig, DataStreamConfig, DEFAULT_MANAGER_CONFIG, DEFAULT_STREAM_CONFIG, managerLogger } from './manager';

// Protocol definitions
export {
  // Message types and enums
  MessageType,
  DataTopic,
  
  // Core interfaces
  BaseMessage,
  WebSocketMessage,
  
  // Connection messages
  ConnectMessage,
  HeartbeatMessage,
  ErrorMessage,
  
  // Subscription messages
  SubscribeMessage,
  UnsubscribeMessage,
  SubscriptionAckMessage,
  
  // LLMKG data messages
  CognitivePatternMessage,
  NeuralActivityMessage,
  KnowledgeGraphUpdateMessage,
  SDROperationMessage,
  MemoryMetricsMessage,
  AttentionFocusMessage,
  
  // Batch and compression
  BatchDataMessage,
  CompressedDataMessage,
  
  // Utilities
  ProtocolValidator,
  ProtocolConfig,
  DEFAULT_PROTOCOL_CONFIG,
  TopicManager,
  protocolLogger
} from './protocol';

// Utilities
export { Logger, LogLevel, LogEntry, defaultLogger } from '../utils/logger';

/**
 * Quick setup function for WebSocket server
 */
export async function createWebSocketServer(config?: Partial<ServerConfig>): Promise<WebSocketServer> {
  const server = new WebSocketServer(config);
  await server.start();
  return server;
}

/**
 * Quick setup function for WebSocket client
 */
export async function createWebSocketClient(config?: Partial<ClientConfig>): Promise<DashboardWebSocketClient> {
  const client = new DashboardWebSocketClient(config);
  await client.connect();
  return client;
}

/**
 * Quick setup function for complete WebSocket manager
 */
export async function createWebSocketManager(
  managerConfig?: Partial<ManagerConfig>,
  streamConfig?: Partial<DataStreamConfig>
): Promise<WebSocketManager> {
  const manager = new WebSocketManager(managerConfig, streamConfig);
  await manager.initialize();
  return manager;
}