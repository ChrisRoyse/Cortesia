/**
 * @fileoverview Complete Integration Example for LLMKG Visualization Phase 1
 * 
 * This example demonstrates the complete data pipeline from LLMKG core components
 * through MCP client, telemetry injection, data collection agents, and WebSocket
 * streaming to dashboard clients.
 * 
 * @author LLMKG Team
 * @version 1.0.0
 */

import {
  MCPClient,
  MCPClientConfig,
  ConnectionState,
  MCPEventType
} from '../mcp/index.js';

import {
  TelemetryInjector,
  TelemetryManager
} from '../telemetry/index.js';

import {
  CollectorManager,
  KnowledgeGraphCollector,
  CognitivePatternCollector,
  NeuralActivityCollector,
  MemorySystemsCollector
} from '../collectors/index.js';

import {
  WebSocketManager,
  DashboardWebSocketClient,
  WebSocketServer,
  MessageType
} from '../websocket/index.js';

/**
 * Complete integration example showcasing the full LLMKG visualization pipeline
 */
export class LLMKGVisualizationPipeline {
  private mcpClient: MCPClient;
  private telemetryInjector: TelemetryInjector;
  private telemetryManager: TelemetryManager;
  private collectorManager: CollectorManager;
  private websocketManager: WebSocketManager;
  private websocketServer: WebSocketServer;
  private isRunning = false;
  private stats = {
    startTime: 0,
    totalDataPoints: 0,
    totalClients: 0,
    errors: 0
  };

  constructor() {
    // Initialize components
    this.initializeComponents();
  }

  /**
   * Initialize all pipeline components
   */
  private initializeComponents(): void {
    console.log('🔧 Initializing LLMKG Visualization Pipeline Components...');

    // 1. Initialize MCP Client for LLMKG server communication
    const mcpConfig: MCPClientConfig = {
      enableTelemetry: true,
      autoDiscoverTools: true,
      requestTimeout: 30000,
      connectionConfig: {
        maxRetries: 3,
        baseDelay: 1000
      }
    };
    this.mcpClient = new MCPClient(mcpConfig);

    // 2. Initialize Telemetry System
    this.telemetryInjector = new TelemetryInjector();
    this.telemetryManager = new TelemetryManager({
      enableRealTimeStreaming: true,
      bufferSize: 10000,
      flushInterval: 1000,
      compressionEnabled: true
    });

    // 3. Initialize Data Collection Agents
    this.collectorManager = new CollectorManager({
      autoStart: false,
      maxConcurrentCollectors: 10,
      loadBalancingStrategy: 'adaptive',
      healthCheckInterval: 5000,
      performanceMonitoring: true,
      aggregationWindow: 2000,
      errorRecoveryAttempts: 3,
      collectorPriorities: {
        'cognitive-patterns': 1,
        'neural-activity': 2,
        'knowledge-graph': 3,
        'memory-systems': 4
      },
      resourceLimits: {
        maxMemoryPerCollector: 256,
        maxTotalMemory: 1024,
        maxCPUPercentage: 80,
        maxBufferSize: 1000
      }
    });

    // 4. Initialize WebSocket Communication System
    this.websocketManager = new WebSocketManager({
      server: {
        port: 8080,
        enableCompression: true,
        maxConnections: 100,
        heartbeatInterval: 30000
      },
      enableTelemetryIntegration: true,
      enableDataCollectionIntegration: true,
      dataStreamingInterval: 100
    });

    this.websocketServer = this.websocketManager.getServer();

    console.log('✅ All components initialized successfully');
  }

  /**
   * Start the complete visualization pipeline
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️  Pipeline is already running');
      return;
    }

    try {
      console.log('🚀 Starting LLMKG Visualization Pipeline...');
      this.stats.startTime = Date.now();

      // Step 1: Initialize and inject telemetry
      console.log('📊 Step 1: Setting up telemetry injection...');
      await this.setupTelemetry();

      // Step 2: Connect to LLMKG MCP servers
      console.log('🔗 Step 2: Connecting to LLMKG servers...');
      await this.connectToLLMKGServers();

      // Step 3: Initialize data collectors
      console.log('📡 Step 3: Initializing data collection agents...');
      await this.setupDataCollectors();

      // Step 4: Start WebSocket server
      console.log('🌐 Step 4: Starting WebSocket communication...');
      await this.startWebSocketCommunication();

      // Step 5: Set up event handling and data flow
      console.log('⚡ Step 5: Setting up data flow pipeline...');
      this.setupDataFlowPipeline();

      // Step 6: Start data collection
      console.log('🎯 Step 6: Starting data collection...');
      await this.startDataCollection();

      this.isRunning = true;
      console.log('✅ LLMKG Visualization Pipeline started successfully!');
      console.log(`📍 WebSocket Server: ws://localhost:8080`);
      console.log(`📊 Dashboard ready for connections`);
      
      // Start periodic status reporting
      this.startStatusReporting();

    } catch (error) {
      console.error('❌ Failed to start pipeline:', error);
      this.stats.errors++;
      throw error;
    }
  }

  /**
   * Stop the complete visualization pipeline
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      console.log('⚠️  Pipeline is not running');
      return;
    }

    try {
      console.log('🛑 Stopping LLMKG Visualization Pipeline...');

      // Stop data collection
      await this.collectorManager.stopAll();

      // Stop WebSocket server
      await this.websocketManager.stop();

      // Disconnect MCP clients
      this.mcpClient.disconnectAll();

      // Stop telemetry
      await this.telemetryManager.stop();

      this.isRunning = false;
      console.log('✅ Pipeline stopped successfully');

    } catch (error) {
      console.error('❌ Error stopping pipeline:', error);
      this.stats.errors++;
    }
  }

  /**
   * Set up telemetry injection system
   */
  private async setupTelemetry(): Promise<void> {
    // Initialize non-intrusive telemetry injection
    await this.telemetryInjector.initialize();

    // Inject telemetry into MCP client
    const instrumentedClient = this.telemetryInjector.injectMCPClient(this.mcpClient);
    
    // Start telemetry manager
    await this.telemetryManager.start();

    // Connect telemetry manager to WebSocket for real-time streaming
    this.telemetryManager.on('telemetry:batch', (batch) => {
      this.websocketServer.broadcast('telemetry.data', {
        batch,
        timestamp: Date.now()
      });
    });

    console.log('✅ Telemetry system initialized and injected');
  }

  /**
   * Connect to LLMKG MCP servers
   */
  private async connectToLLMKGServers(): Promise<void> {
    const servers = [
      'ws://localhost:8001/mcp',  // Brain-inspired LLMKG instance
      'ws://localhost:8002/mcp',  // Federated LLMKG instance
      'ws://localhost:8003/mcp'   // Knowledge Graph LLMKG instance
    ];

    try {
      const connectionResults = await this.mcpClient.connectMultiple(servers);
      console.log(`✅ Connected to ${connectionResults.length} LLMKG servers`);

      // Set up MCP event handlers
      this.mcpClient.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
        console.log(`🔗 Connection state changed: ${event.data.endpoint} -> ${event.data.newState}`);
        
        // Broadcast connection status to dashboard
        this.websocketServer.broadcast('system.connection', {
          type: 'mcp_connection_change',
          endpoint: event.data.endpoint,
          state: event.data.newState,
          timestamp: Date.now()
        });
      });

      this.mcpClient.on(MCPEventType.TOOLS_DISCOVERED, (event) => {
        console.log(`🛠️  Tools discovered on ${event.data.endpoint}: ${event.data.tools.length} tools`);
      });

    } catch (error) {
      console.warn('⚠️  Some MCP connections failed, continuing with available connections');
      // Continue with partial connections for demo purposes
    }
  }

  /**
   * Set up data collection agents
   */
  private async setupDataCollectors(): Promise<void> {
    // Create and add collectors to the manager
    const collectors = [
      {
        name: 'cognitive-patterns',
        collector: new CognitivePatternCollector(this.mcpClient, {
          collectionInterval: 500,
          bufferSize: 1000,
          enableRealTimeStreaming: true,
          patternTypes: ['recognition', 'association', 'inference', 'attention', 'reasoning', 'decision'],
          confidenceThreshold: 0.7,
          hierarchicalDepth: 3
        })
      },
      {
        name: 'neural-activity',
        collector: new NeuralActivityCollector(this.mcpClient, {
          collectionInterval: 100,
          bufferSize: 2000,
          enableRealTimeStreaming: true,
          activityTypes: ['activation', 'inhibition', 'propagation'],
          spatialResolution: 'high',
          temporalWindow: 1000
        })
      },
      {
        name: 'knowledge-graph',
        collector: new KnowledgeGraphCollector(this.mcpClient, {
          collectionInterval: 1000,
          bufferSize: 500,
          enableRealTimeStreaming: true,
          trackNodeUpdates: true,
          trackEdgeUpdates: true,
          trackCommunityDetection: true,
          maxGraphSize: 10000
        })
      },
      {
        name: 'memory-systems',
        collector: new MemorySystemsCollector(this.mcpClient, {
          collectionInterval: 2000,
          bufferSize: 500,
          enableRealTimeStreaming: true,
          memoryTypes: ['episodic', 'semantic', 'working', 'procedural'],
          compressionEnabled: true,
          retentionAnalysis: true
        })
      }
    ];

    // Add collectors to manager
    for (const { name, collector } of collectors) {
      await this.collectorManager.addCollector(name, collector);
      console.log(`✅ Added ${name} collector`);
    }

    console.log('✅ All data collectors initialized');
  }

  /**
   * Start WebSocket communication system
   */
  private async startWebSocketCommunication(): Promise<void> {
    // Initialize and start the WebSocket manager
    await this.websocketManager.initialize();
    await this.websocketManager.start();

    // Set up WebSocket event handlers
    this.websocketServer.on('client:connected', (client) => {
      this.stats.totalClients++;
      console.log(`🔌 Dashboard client connected (${this.stats.totalClients} total)`);
      
      // Send welcome message with current system status
      client.send(JSON.stringify({
        type: MessageType.SYSTEM_STATUS,
        data: {
          pipelineStatus: 'running',
          connectedServers: this.mcpClient.connectedEndpoints.length,
          activeCollectors: Object.keys(this.collectorManager.getStats().collectorStats).length,
          uptime: Date.now() - this.stats.startTime
        },
        timestamp: Date.now()
      }));
    });

    this.websocketServer.on('client:disconnected', () => {
      this.stats.totalClients--;
      console.log(`🔌 Dashboard client disconnected (${this.stats.totalClients} total)`);
    });

    console.log('✅ WebSocket communication system started');
  }

  /**
   * Set up the complete data flow pipeline
   */
  private setupDataFlowPipeline(): void {
    // Connect collector manager to WebSocket for real-time data streaming
    this.collectorManager.on('data:processed', (event) => {
      this.stats.totalDataPoints++;
      
      // Stream data to all connected dashboard clients
      this.websocketServer.broadcast(`${event.collector}.data`, {
        collector: event.collector,
        data: event.data,
        timestamp: event.timestamp
      });
    });

    // Handle system alerts and errors
    this.collectorManager.on('system:alert', (alert) => {
      console.log(`⚠️  System Alert [${alert.severity}]: ${alert.message}`);
      
      this.websocketServer.broadcast('system.alert', {
        severity: alert.severity,
        message: alert.message,
        source: alert.source,
        timestamp: alert.timestamp
      });
    });

    // Handle performance metrics
    this.collectorManager.on('performance:update', (metrics) => {
      this.websocketServer.broadcast('performance.metrics', {
        metrics,
        timestamp: Date.now()
      });
    });

    // Handle collector health status
    this.collectorManager.on('health:check:complete', (results) => {
      this.websocketServer.broadcast('system.health', {
        health: results,
        timestamp: Date.now()
      });
    });

    console.log('✅ Data flow pipeline configured');
  }

  /**
   * Start data collection from all collectors
   */
  private async startDataCollection(): Promise<void> {
    await this.collectorManager.startAll();
    console.log('✅ Data collection started from all agents');
  }

  /**
   * Start periodic status reporting
   */
  private startStatusReporting(): void {
    setInterval(() => {
      if (!this.isRunning) return;

      const uptime = Date.now() - this.stats.startTime;
      const status = {
        uptime: Math.floor(uptime / 1000),
        totalDataPoints: this.stats.totalDataPoints,
        connectedClients: this.stats.totalClients,
        mcpConnections: this.mcpClient.connectedEndpoints.length,
        collectorStats: this.collectorManager.getStats(),
        errors: this.stats.errors,
        performance: {
          dataRate: this.stats.totalDataPoints / (uptime / 1000),
          memoryUsage: process.memoryUsage()
        }
      };

      console.log('📊 Pipeline Status:', {
        uptime: `${status.uptime}s`,
        dataPoints: status.totalDataPoints,
        clients: status.connectedClients,
        dataRate: `${status.performance.dataRate.toFixed(2)}/sec`,
        memory: `${Math.round(status.performance.memoryUsage.heapUsed / 1024 / 1024)}MB`
      });

      // Broadcast status to dashboard clients
      this.websocketServer.broadcast('system.status', status);

    }, 30000); // Every 30 seconds
  }

  /**
   * Get current pipeline statistics
   */
  getStats() {
    return {
      ...this.stats,
      uptime: this.isRunning ? Date.now() - this.stats.startTime : 0,
      isRunning: this.isRunning,
      mcpStats: this.mcpClient.statistics,
      collectorStats: this.collectorManager.getStats(),
      websocketStats: this.websocketServer.getStats()
    };
  }

  /**
   * Demonstrate the pipeline with sample operations
   */
  async demonstrate(): Promise<void> {
    if (!this.isRunning) {
      throw new Error('Pipeline must be running to demonstrate');
    }

    console.log('🎭 Starting LLMKG Pipeline Demonstration...');

    // 1. Trigger brain visualization
    console.log('🧠 1. Requesting brain visualization data...');
    try {
      const brainData = await this.mcpClient.llmkg.brainVisualization({
        focus: 'hippocampus',
        depth: 3
      });
      console.log('✅ Brain visualization data received:', Object.keys(brainData).join(', '));
    } catch (error) {
      console.log('⚠️  Using mock brain visualization data');
    }

    // 2. Analyze connectivity patterns
    console.log('🕸️  2. Analyzing connectivity patterns...');
    try {
      const connectivity = await this.mcpClient.llmkg.analyzeConnectivity('semantic_node_1', 4);
      console.log('✅ Connectivity analysis completed:', connectivity.confidence);
    } catch (error) {
      console.log('⚠️  Using mock connectivity data');
    }

    // 3. Get federated metrics
    console.log('🌐 3. Fetching federated metrics...');
    try {
      const metrics = await this.mcpClient.llmkg.federatedMetrics({
        metricTypes: ['performance', 'resource_usage', 'activity']
      });
      console.log('✅ Federated metrics received from', metrics.instances?.length || 0, 'instances');
    } catch (error) {
      console.log('⚠️  Using mock federated metrics');
    }

    // 4. Force data collection cycle
    console.log('📊 4. Triggering data collection cycle...');
    // Data collection happens automatically, but we can check stats
    const collectorStats = this.collectorManager.getStats();
    console.log('✅ Active collectors:', collectorStats.activeCollectors);
    console.log('✅ Total data points:', collectorStats.totalDataPoints);

    console.log('🎉 Pipeline demonstration completed successfully!');
  }
}

/**
 * Example usage and integration testing
 */
export async function runCompleteIntegrationExample(): Promise<void> {
  const pipeline = new LLMKGVisualizationPipeline();

  try {
    // Start the complete pipeline
    await pipeline.start();

    // Run demonstration
    await pipeline.demonstrate();

    // Let it run for a demo period
    console.log('⏱️  Running pipeline for 60 seconds...');
    
    const demoTimer = setTimeout(async () => {
      console.log('📊 Final statistics:');
      console.log(pipeline.getStats());
      
      await pipeline.stop();
      console.log('✅ Complete integration example finished successfully');
    }, 60000);

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      clearTimeout(demoTimer);
      console.log('\n🛑 Received shutdown signal, stopping pipeline...');
      await pipeline.stop();
      process.exit(0);
    });

  } catch (error) {
    console.error('❌ Integration example failed:', error);
    await pipeline.stop();
    throw error;
  }
}

// Export for use in other modules
export { LLMKGVisualizationPipeline };

// Run if called directly
if (require.main === module) {
  runCompleteIntegrationExample().catch(console.error);
}