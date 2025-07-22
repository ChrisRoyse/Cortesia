/**
 * WebSocket Communication System Demo
 * 
 * This example demonstrates how to use the LLMKG WebSocket communication system
 * for real-time data streaming and visualization dashboard integration.
 */

import { 
  WebSocketManager, 
  WebSocketServer, 
  DashboardWebSocketClient,
  MessageType,
  DataTopic 
} from '../websocket';

/**
 * Demo 1: Basic WebSocket Server Setup
 */
async function basicServerDemo(): Promise<void> {
  console.log('🚀 Starting basic WebSocket server demo...');
  
  const server = new WebSocketServer({
    port: 8080,
    enableCompression: true,
    enableBuffering: true,
    maxConnections: 100
  });

  // Setup event handlers
  server.on('started', () => {
    console.log('✅ WebSocket server started on port 8080');
  });

  server.on('clientConnected', (client) => {
    console.log(`👤 Client connected: ${client.id}`);
  });

  server.on('clientDisconnected', ({ clientId }) => {
    console.log(`👋 Client disconnected: ${clientId}`);
  });

  // Start the server
  await server.start();

  // Simulate broadcasting LLMKG data
  setInterval(() => {
    // Broadcast cognitive pattern data
    server.broadcast(DataTopic.COGNITIVE_PATTERNS, {
      type: MessageType.COGNITIVE_PATTERN,
      data: {
        patternId: `pattern_${Date.now()}`,
        patternType: 'recognition',
        activation: Math.random(),
        confidence: Math.random(),
        context: { source: 'demo' }
      }
    });

    // Broadcast neural activity
    server.broadcast(DataTopic.NEURAL_ACTIVITY, {
      type: MessageType.NEURAL_ACTIVITY,
      data: {
        nodeId: `node_${Math.floor(Math.random() * 1000)}`,
        activityType: 'activation',
        intensity: Math.random(),
        connections: {
          incoming: [`input_${Math.floor(Math.random() * 100)}`],
          outgoing: [`output_${Math.floor(Math.random() * 100)}`]
        }
      }
    });
  }, 1000);

  console.log('🔄 Server is now broadcasting demo data every second');
}

/**
 * Demo 2: Dashboard Client Setup
 */
async function dashboardClientDemo(): Promise<void> {
  console.log('🖥️ Starting dashboard client demo...');

  const client = new DashboardWebSocketClient({
    url: 'ws://localhost:8080',
    autoReconnect: true,
    enableCompression: true
  });

  // Setup event handlers
  client.on('connected', () => {
    console.log('✅ Client connected to server');
  });

  client.on('disconnected', ({ reason }) => {
    console.log(`❌ Client disconnected: ${reason}`);
  });

  client.on('stateChange', ({ newState }) => {
    console.log(`🔄 Connection state changed to: ${newState}`);
  });

  // Connect to server
  await client.connect();

  // Subscribe to cognitive patterns with filtering
  await client.subscribe(
    [DataTopic.COGNITIVE_PATTERNS],
    (message, topic) => {
      console.log(`🧠 Cognitive Pattern [${topic}]:`, {
        patternId: message.data.patternId,
        type: message.data.patternType,
        activation: message.data.activation.toFixed(3),
        confidence: message.data.confidence.toFixed(3)
      });
    },
    { minConfidence: 0.5 }
  );

  // Subscribe to neural activity
  await client.subscribe(
    [DataTopic.NEURAL_ACTIVITY],
    (message, topic) => {
      console.log(`⚡ Neural Activity [${topic}]:`, {
        nodeId: message.data.nodeId,
        type: message.data.activityType,
        intensity: message.data.intensity.toFixed(3)
      });
    },
    { minIntensity: 0.3 }
  );

  // Subscribe to all data with wildcard
  await client.subscribe(
    ['system.*'],
    (message, topic) => {
      console.log(`📊 System Data [${topic}]:`, message.type);
    }
  );

  console.log('👂 Client is now listening for LLMKG data');
}

/**
 * Demo 3: Complete Manager Setup with Integration
 */
async function managerIntegrationDemo(): Promise<void> {
  console.log('🏗️ Starting WebSocket manager integration demo...');

  const manager = new WebSocketManager({
    server: {
      port: 8081,
      enableCompression: true,
      enableBuffering: true
    },
    enableServer: true,
    enableTelemetryIntegration: true,
    enableDataCollectionIntegration: true,
    enableMCPIntegration: true,
    dataStreamingInterval: 100, // 100ms for high-frequency updates
    batchingEnabled: true,
    compressionEnabled: true
  }, {
    // Stream configuration for different data types
    cognitivePatterns: {
      enabled: true,
      updateInterval: 50,
      minConfidence: 0.4
    },
    neuralActivity: {
      enabled: true,
      updateInterval: 25,
      minIntensity: 0.2
    },
    knowledgeGraph: {
      enabled: true,
      batchUpdates: true,
      maxBatchSize: 50
    },
    memoryMetrics: {
      enabled: true,
      updateInterval: 1000
    },
    attentionMechanisms: {
      enabled: true,
      trackFocusChanges: true,
      minIntensity: 0.3
    }
  });

  // Setup event handlers
  manager.on('initialized', () => {
    console.log('✅ WebSocket manager initialized');
  });

  manager.on('serverStarted', () => {
    console.log('🚀 WebSocket server started');
  });

  manager.on('streamingStarted', () => {
    console.log('📡 Data streaming started');
  });

  manager.on('error', (error) => {
    console.error('❌ Manager error:', error);
  });

  // Initialize the manager
  await manager.initialize();

  // Display statistics every 10 seconds
  setInterval(() => {
    const stats = manager.getStats();
    console.log('📊 Manager Statistics:', {
      totalDataStreamed: stats.manager.totalDataStreamed,
      totalClients: stats.manager.totalClientsServed,
      activeConnections: stats.server?.clients || 0,
      streamingUptime: Math.round(stats.streaming.uptime / 1000) + 's',
      compressionRatio: stats.server?.buffer?.compressionStats.ratio || 'N/A'
    });
  }, 10000);

  console.log('📈 Manager is now streaming integrated LLMKG data');
}

/**
 * Demo 4: Performance and Load Testing
 */
async function performanceDemo(): Promise<void> {
  console.log('⚡ Starting performance demo...');

  const server = new WebSocketServer({
    port: 8082,
    enableCompression: true,
    enableBuffering: true,
    maxConnections: 1000
  });

  await server.start();

  // Create multiple clients to test concurrent connections
  const clients: DashboardWebSocketClient[] = [];
  const numClients = 10;

  for (let i = 0; i < numClients; i++) {
    const client = new DashboardWebSocketClient({
      url: 'ws://localhost:8082',
      autoReconnect: true
    });

    await client.connect();

    // Subscribe each client to different topics
    await client.subscribe(
      [DataTopic.COGNITIVE_PATTERNS, DataTopic.NEURAL_ACTIVITY],
      (message, topic) => {
        // Process messages (in real app this would update visualization)
      }
    );

    clients.push(client);
  }

  console.log(`👥 ${numClients} clients connected`);

  // Generate high-frequency data
  let messageCount = 0;
  const startTime = Date.now();

  const highFrequencyTimer = setInterval(() => {
    // Broadcast 100 messages per interval
    for (let i = 0; i < 100; i++) {
      server.broadcast(DataTopic.COGNITIVE_PATTERNS, {
        type: MessageType.COGNITIVE_PATTERN,
        data: {
          patternId: `perf_${messageCount++}`,
          patternType: 'recognition',
          activation: Math.random(),
          confidence: Math.random(),
          context: { test: 'performance' }
        }
      });
    }
  }, 100); // Every 100ms = 1000 messages/second

  // Report performance stats
  setTimeout(() => {
    clearInterval(highFrequencyTimer);
    
    const duration = (Date.now() - startTime) / 1000;
    const messagesPerSecond = messageCount / duration;
    
    console.log('📊 Performance Results:', {
      duration: `${duration.toFixed(2)}s`,
      totalMessages: messageCount,
      messagesPerSecond: Math.round(messagesPerSecond),
      clients: numClients,
      serverStats: server.getStats()
    });

    // Cleanup
    clients.forEach(client => client.disconnect());
    server.stop();
  }, 30000); // Run for 30 seconds

  console.log('🏃‍♂️ Running high-frequency performance test for 30 seconds...');
}

/**
 * Main demo function
 */
async function runDemos(): Promise<void> {
  console.log('🎮 LLMKG WebSocket Communication System Demos\n');

  try {
    // Demo 1: Basic server (run in background)
    setTimeout(() => basicServerDemo(), 0);
    
    // Demo 2: Client connection (wait for server to start)
    setTimeout(() => dashboardClientDemo(), 2000);
    
    // Demo 3: Manager integration (different port)
    setTimeout(() => managerIntegrationDemo(), 4000);
    
    // Demo 4: Performance test (different port)
    setTimeout(() => performanceDemo(), 6000);

    console.log('🔄 All demos started. Check console output for real-time updates.');
    console.log('💡 Press Ctrl+C to stop all demos.');

  } catch (error) {
    console.error('❌ Demo error:', error);
    process.exit(1);
  }
}

// Run demos if this file is executed directly
if (require.main === module) {
  runDemos();
}

export {
  basicServerDemo,
  dashboardClientDemo,
  managerIntegrationDemo,
  performanceDemo,
  runDemos
};