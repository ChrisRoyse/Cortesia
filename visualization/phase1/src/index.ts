/**
 * @fileoverview LLMKG Visualization Phase 1 - Main Entry Point
 * 
 * This is the main entry point for the LLMKG Visualization Phase 1 implementation,
 * providing the complete MCP client system for communicating with LLMKG servers.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

// Export the complete MCP client system
export * from './mcp/index.js';

// Export the telemetry injection system
export * from './telemetry/index.js';

// Export data collection agents
export * from './collectors/index.js';

// Export examples for reference
export * as Examples from './examples/basic-usage.js';

/**
 * Library version information
 */
export const VERSION = "1.0.0";
export const PROTOCOL_VERSION = "2024-11-05";

/**
 * Quick start function for common LLMKG visualization use cases with telemetry
 * 
 * @param endpoints - Array of LLMKG server endpoints to connect to
 * @param telemetryConfig - Optional telemetry configuration
 * @returns Configured and connected MCP client with telemetry
 */
export async function createLLMKGClient(endpoints: string[] = [
  "ws://localhost:8001/mcp", // Brain-inspired MCP server
  "ws://localhost:8002/mcp", // Federated MCP server  
  "ws://localhost:8003/mcp"  // Knowledge graph MCP server
], telemetryConfig?: any) {
  // Initialize telemetry system first
  const { initializeTelemetry, wrapMCPClient } = await import('./telemetry/index.js');
  await initializeTelemetry(telemetryConfig);
  
  const { MCPClient } = await import('./mcp/index.js');
  
  const client = new MCPClient({
    enableTelemetry: true,
    autoDiscoverTools: true,
    connectionConfig: {
      timeout: 30000,
      maxRetries: 5,
      baseDelay: 1000
    }
  });

  // Wrap client with telemetry
  const instrumentedClient = wrapMCPClient(client);
  
  // Connect to all endpoints
  await instrumentedClient.connectMultiple(endpoints);
  
  return instrumentedClient;
}

/**
 * Initialize LLMKG visualization with full telemetry stack
 * 
 * @param config - Optional configuration for telemetry and visualization
 * @returns Promise that resolves when initialization is complete
 */
export async function initializeLLMKGVisualization(config?: {
  telemetry?: any;
  endpoints?: string[];
  collectors?: boolean;
}) {
  // Initialize telemetry system
  const { initializeTelemetry, telemetryManager } = await import('./telemetry/index.js');
  await initializeTelemetry(config?.telemetry);

  // Initialize data collection agents if enabled
  if (config?.collectors !== false) {
    const { CollectorManager } = await import('./collectors/index.js');
    const collectorManager = new CollectorManager();
    await collectorManager.initialize();
  }

  // Create and return instrumented client
  const client = await createLLMKGClient(config?.endpoints, config?.telemetry);
  
  return {
    client,
    telemetryManager,
    getTelemetryStatus: () => {
      const { getTelemetryStatus } = require('./telemetry/index.js');
      return getTelemetryStatus();
    },
    shutdown: async () => {
      const { shutdownTelemetry } = await import('./telemetry/index.js');
      await shutdownTelemetry();
    }
  };
}