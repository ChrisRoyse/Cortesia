/**
 * @fileoverview MCP Client Library for LLMKG Visualization
 * 
 * This is the main entry point for the LLMKG MCP client library, providing
 * a comprehensive set of tools for communicating with LLMKG MCP servers
 * from the visualization dashboard.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

// Export all types
export * from './types.js';

// Export connection management
export { MCPConnection, MCPConnectionPool } from './connection.js';

// Export protocol handling
export { 
  MCPProtocol, 
  MCPMessageHandler, 
  MCPProtocolUtils,
  LLMKG_CLIENT_CAPABILITIES,
  LLMKG_CLIENT_INFO,
  MCP_PROTOCOL_VERSION
} from './protocol.js';

// Export main client
export { MCPClient, type MCPClientConfig, type ClientStats } from './client.js';

// Re-export commonly used types for convenience
export type {
  MCPMessage,
  MCPRequest,
  MCPResponse,
  MCPNotification,
  MCPError,
  MCPTool,
  MCPClientInfo,
  MCPServerInfo,
  MCPCapabilities,
  ConnectionConfig,
  LLMKGTools
} from './types.js';

// Re-export enums as values
export {
  ConnectionState,
  MCPEventType,
  MCPErrorCode
} from './types.js';