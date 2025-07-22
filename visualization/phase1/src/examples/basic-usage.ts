/**
 * @fileoverview Basic Usage Example for LLMKG MCP Client
 * 
 * This example demonstrates how to use the MCP client to connect to LLMKG servers
 * and perform common operations like tool discovery and calling LLMKG-specific tools.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import {
  MCPClient,
  MCPClientConfig,
  ConnectionState,
  MCPEventType,
  LLMKGTools
} from '../mcp/index.js';

/**
 * Example configuration for LLMKG MCP servers
 */
const LLMKG_SERVERS = {
  BRAIN_INSPIRED: "ws://localhost:8001/mcp",
  FEDERATED: "ws://localhost:8002/mcp", 
  KNOWLEDGE_GRAPH: "ws://localhost:8003/mcp"
};

/**
 * Basic usage example
 */
async function basicUsageExample() {
  console.log("=== LLMKG MCP Client Basic Usage Example ===\n");

  // Create client with configuration
  const config: MCPClientConfig = {
    enableTelemetry: true,
    autoDiscoverTools: true,
    requestTimeout: 30000,
    connectionConfig: {
      maxRetries: 3,
      baseDelay: 1000
    }
  };

  const client = new MCPClient(config);

  // Set up event listeners
  client.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
    console.log(`Connection state changed: ${event.data.endpoint} -> ${event.data.newState}`);
  });

  client.on(MCPEventType.ERROR_OCCURRED, (event) => {
    console.error(`Error occurred: ${event.data.context} - ${event.data.error.message}`);
  });

  client.on(MCPEventType.TELEMETRY_EVENT, (event) => {
    console.log(`Telemetry: ${event.data.metric} = ${event.data.value}`);
  });

  try {
    // Connect to LLMKG servers
    console.log("Connecting to LLMKG servers...");
    const servers = await client.connectMultiple([
      LLMKG_SERVERS.BRAIN_INSPIRED,
      LLMKG_SERVERS.FEDERATED,
      LLMKG_SERVERS.KNOWLEDGE_GRAPH
    ]);

    console.log(`Connected to ${servers.length} servers:`);
    servers.forEach(server => {
      console.log(`  - ${server.name} v${server.version}`);
    });

    // List available tools
    console.log("\nDiscovering available tools...");
    const tools = await client.listTools();
    console.log(`Found ${tools.length} tools:`);
    tools.forEach(tool => {
      console.log(`  - ${tool.name}: ${tool.description}`);
    });

    // Example: Get brain activation patterns
    console.log("\nGetting brain activation patterns...");
    const activationResult = await client.llmkg.getActivationPatterns(
      "hippocampus",
      5000, // 5 second window
      0.3   // 30% threshold
    );
    console.log("Activation patterns:", activationResult);

    // Example: Query knowledge graph
    console.log("\nQuerying knowledge graph...");
    const kgResult = await client.llmkg.knowledgeGraphQuery({
      query: "SELECT * FROM entities WHERE type = 'concept' LIMIT 10",
      includeWeights: true
    });
    console.log("Knowledge graph results:", kgResult);

    // Example: Get federated learning metrics
    console.log("\nGetting federated learning metrics...");
    const fedResult = await client.llmkg.federatedMetrics({
      metrics: ["accuracy", "loss", "communication_rounds"],
      period: "1h"
    });
    console.log("Federated metrics:", fedResult);

    // Health check
    console.log("\nPerforming health check...");
    const health = await client.healthCheck();
    console.log("Server health:", health);

    // Show client statistics
    console.log("\nClient statistics:");
    console.log(JSON.stringify(client.statistics, null, 2));

  } catch (error) {
    console.error("Example failed:", error);
  } finally {
    // Clean up
    console.log("\nDisconnecting from servers...");
    await client.disconnectAll();
    console.log("Example completed.");
  }
}

/**
 * Advanced usage example with error handling and reconnection
 */
async function advancedUsageExample() {
  console.log("\n=== LLMKG MCP Client Advanced Usage Example ===\n");

  const client = new MCPClient({
    enableTelemetry: true,
    connectionConfig: {
      maxRetries: 5,
      baseDelay: 2000,
      maxDelay: 30000
    },
    retryConfig: {
      maxAttempts: 3,
      baseDelay: 1000,
      maxDelay: 10000,
      backoffMultiplier: 2,
      jitter: true
    }
  });

  // Advanced event handling
  client.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
    switch (event.data.newState) {
      case ConnectionState.CONNECTED:
        console.log(`✅ Connected to ${event.data.endpoint}`);
        break;
      case ConnectionState.RECONNECTING:
        console.log(`🔄 Reconnecting to ${event.data.endpoint}`);
        break;
      case ConnectionState.ERROR:
        console.log(`❌ Connection error for ${event.data.endpoint}: ${event.data.error?.message}`);
        break;
    }
  });

  try {
    // Connect with individual server configurations
    const brainServer = await client.connect(LLMKG_SERVERS.BRAIN_INSPIRED, {
      timeout: 15000,
      headers: { "X-Client": "LLMKG-Visualization" }
    });

    console.log(`Connected to brain server: ${brainServer.name}`);

    // Example: Complex brain visualization request
    const vizResult = await client.llmkg.brainVisualization({
      region: "prefrontal_cortex",
      type: "connectivity",
      timeRange: {
        start: new Date(Date.now() - 3600000), // 1 hour ago
        end: new Date()
      },
      resolution: "high"
    });

    console.log("Brain visualization result:", vizResult);

    // Example: Batch tool calls
    console.log("\nExecuting batch operations...");
    const batchResults = await Promise.allSettled([
      client.llmkg.analyzeSdr("sdr_12345", true, true),
      client.llmkg.analyzeConnectivity("hippocampus", "prefrontal_cortex", "functional"),
      client.llmkg.federatedMetrics({ metrics: ["participants", "data_points"], period: "5m" })
    ]);

    batchResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        console.log(`Batch operation ${index + 1} succeeded:`, result.value);
      } else {
        console.log(`Batch operation ${index + 1} failed:`, result.reason.message);
      }
    });

  } catch (error) {
    console.error("Advanced example failed:", error);
  } finally {
    await client.disconnectAll();
  }
}

/**
 * Real-time monitoring example
 */
async function realTimeMonitoringExample() {
  console.log("\n=== LLMKG MCP Client Real-Time Monitoring Example ===\n");

  const client = new MCPClient({
    enableTelemetry: true,
    autoDiscoverTools: true
  });

  // Set up real-time monitoring
  const monitoringInterval = setInterval(async () => {
    if (!client.isConnected) return;

    try {
      // Get real-time metrics
      const [activation, connectivity, federated] = await Promise.all([
        client.llmkg.getActivationPatterns("default_mode_network", 1000),
        client.llmkg.analyzeConnectivity("amygdala", "prefrontal_cortex"),
        client.llmkg.federatedMetrics({ metrics: ["active_clients"], period: "1m" })
      ]);

      console.log(`[${new Date().toISOString()}] Real-time update:`, {
        activation: activation ? "✅" : "❌",
        connectivity: connectivity ? "✅" : "❌", 
        federated: federated ? "✅" : "❌"
      });

    } catch (error) {
      console.log(`[${new Date().toISOString()}] Monitor error:`, (error as Error).message);
    }
  }, 5000); // Update every 5 seconds

  try {
    await client.connect(LLMKG_SERVERS.BRAIN_INSPIRED);
    console.log("Started real-time monitoring (press Ctrl+C to stop)...");

    // Run for 30 seconds
    await new Promise(resolve => setTimeout(resolve, 30000));

  } finally {
    clearInterval(monitoringInterval);
    await client.disconnectAll();
    console.log("Monitoring stopped.");
  }
}

/**
 * Main function to run examples
 */
async function main() {
  try {
    await basicUsageExample();
    await advancedUsageExample();
    await realTimeMonitoringExample();
  } catch (error) {
    console.error("Examples failed:", error);
    process.exit(1);
  }
}

// Run examples if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}