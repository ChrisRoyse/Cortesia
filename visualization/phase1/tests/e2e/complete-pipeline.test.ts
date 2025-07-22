/**
 * End-to-End Tests for Complete LLMKG Visualization Pipeline
 * 
 * Tests the complete data flow from LLMKG brain-inspired operations
 * through MCP, collectors, telemetry, and WebSocket streaming to visualization.
 */

import { MCPClient } from '../../src/mcp/client';
import { CollectorManager } from '../../src/collectors/manager';
import { CognitivePatternCollector } from '../../src/collectors/cognitive-patterns';
import { NeuralActivityCollector } from '../../src/collectors/neural-activity';
import { MemorySystemCollector } from '../../src/collectors/memory-systems';
import { WebSocketServer, ClientConnection } from '../../src/websocket/server';
import { WebSocketClient } from '../../src/websocket/client';
import { TelemetryInjector } from '../../src/telemetry/injector';
import { MockBrainInspiredMCPServer, MCPMessageType } from '../mocks/brain-inspired-server';
import { PerformanceTracker, TestHelpers, LLMKGDataGenerator, MockWebSocket } from '../config/test-utils';

// Mock WebSocket for testing
jest.mock('ws', () => ({
  Server: jest.fn().mockImplementation(() => ({
    on: jest.fn((event, handler) => {
      if (event === 'listening') setTimeout(handler, 10);
    }),
    close: jest.fn((callback) => callback && callback()),
    clients: new Set()
  })),
  default: MockWebSocket
}));

describe('Complete Pipeline E2E Tests', () => {
  let mockLLMKGServer: MockBrainInspiredMCPServer;
  let mcpClient: MCPClient;
  let collectorManager: CollectorManager;
  let websocketServer: WebSocketServer;
  let websocketClient: WebSocketClient;
  let telemetryInjector: TelemetryInjector;
  let performanceTracker: PerformanceTracker;

  beforeEach(async () => {
    performanceTracker = new PerformanceTracker();
    
    // Initialize mock LLMKG server with realistic brain-inspired capabilities
    mockLLMKGServer = new MockBrainInspiredMCPServer();
    await mockLLMKGServer.start();

    // Initialize telemetry system
    telemetryInjector = new TelemetryInjector();
    await telemetryInjector.initialize();

    // Initialize MCP client with telemetry injection
    mcpClient = new MCPClient({
      enableRealtimeUpdates: true,
      updateInterval: 100,
      serverUrl: 'ws://localhost:3000'
    });
    mcpClient = telemetryInjector.injectMCPClient(mcpClient);

    // Initialize WebSocket server
    websocketServer = new WebSocketServer({
      port: 8088,
      heartbeatInterval: 1000,
      enableBuffering: true,
      enableCompression: false
    });

    // Initialize WebSocket client
    websocketClient = new WebSocketClient({
      serverUrl: 'ws://localhost:8088',
      reconnectInterval: 1000,
      maxReconnectAttempts: 3
    });

    // Initialize collector manager
    collectorManager = new CollectorManager(mcpClient);
  });

  afterEach(async () => {
    if (collectorManager) {
      await collectorManager.stopAll();
    }
    if (websocketServer && websocketServer.isActive()) {
      await websocketServer.stop();
    }
    if (websocketClient && websocketClient.isConnected()) {
      await websocketClient.disconnect();
    }
    if (mcpClient) {
      mcpClient.disconnect();
    }
    if (mockLLMKGServer) {
      await mockLLMKGServer.stop();
    }
    if (telemetryInjector) {
      telemetryInjector.cleanup();
    }
    performanceTracker.clear();
  });

  describe('Complete LLMKG Data Pipeline', () => {
    it('should stream complete cognitive processing pipeline', async () => {
      const pipelineEvents: Array<{
        stage: string;
        data: any;
        timestamp: number;
      }> = [];

      // Setup collectors for different LLMKG components
      const cognitiveCollector = new CognitivePatternCollector(mcpClient, {
        name: 'e2e-cognitive',
        collectionInterval: 100,
        bufferSize: 1000
      });

      const neuralCollector = new NeuralActivityCollector(mcpClient, {
        name: 'e2e-neural',
        collectionInterval: 120,
        bufferSize: 1000
      });

      const memoryCollector = new MemorySystemCollector(mcpClient, {
        name: 'e2e-memory',
        collectionInterval: 150,
        bufferSize: 1000
      });

      // Track data flow through pipeline
      [cognitiveCollector, neuralCollector, memoryCollector].forEach(collector => {
        collector.on('data:collected', (event) => {
          pipelineEvents.push({
            stage: 'collection',
            data: {
              collector: event.collector,
              type: event.data.type,
              timestamp: event.data.timestamp
            },
            timestamp: Date.now()
          });

          // Forward to WebSocket
          websocketServer.broadcast(`${collector.getStats().name}_data`, event.data);
        });
      });

      // Setup WebSocket client to receive streamed data
      const receivedData: any[] = [];
      websocketClient.on('message', (message) => {
        pipelineEvents.push({
          stage: 'websocket_received',
          data: message,
          timestamp: Date.now()
        });
        receivedData.push(message);
      });

      // Start complete pipeline
      await websocketServer.start();
      await websocketClient.connect();

      await collectorManager.addCollector('cognitive', cognitiveCollector);
      await collectorManager.addCollector('neural', neuralCollector);
      await collectorManager.addCollector('memory', memoryCollector);
      await collectorManager.startAll();

      mcpClient.connect();

      // Let the pipeline run and process data
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify complete pipeline functionality
      expect(pipelineEvents.length).toBeGreaterThan(0);
      
      const collectionEvents = pipelineEvents.filter(e => e.stage === 'collection');
      const websocketEvents = pipelineEvents.filter(e => e.stage === 'websocket_received');
      
      expect(collectionEvents.length).toBeGreaterThan(0);
      expect(receivedData.length).toBeGreaterThan(0);

      // Verify data from all collectors
      const collectorTypes = new Set(collectionEvents.map(e => e.data.collector));
      expect(collectorTypes.size).toBeGreaterThan(1); // Multiple collectors active
    });

    it('should process real LLMKG brain-inspired operations', async () => {
      const llmkgOperations: Array<{
        operation: string;
        request: any;
        response: any;
        telemetry: any;
      }> = [];

      // Setup to track LLMKG operations
      mockLLMKGServer.on('telemetry', (telemetryEvent) => {
        if (telemetryEvent.event === 'tool_execution_start') {
          const operation = {
            operation: telemetryEvent.data.tool_name,
            request: telemetryEvent.data,
            response: null,
            telemetry: telemetryEvent.data
          };
          llmkgOperations.push(operation);
        }
      });

      // Setup collector to trigger LLMKG operations
      const brainInspiredCollector = new CognitivePatternCollector(mcpClient, {
        name: 'brain-inspired-e2e',
        collectionInterval: 200
      });

      const processedOperations: string[] = [];
      brainInspiredCollector.on('data:collected', async (event) => {
        // Simulate calling LLMKG brain-inspired tools
        try {
          // Simulate SDR encoding
          const sdrRequest = await mockLLMKGServer.handleMessage({
            type: MCPMessageType.TOOL_CALL,
            id: `sdr_encode_${Date.now()}`,
            method: 'sdr_encode',
            params: {
              input: `cognitive_pattern_${event.data.pattern_id}`,
              size: 2048,
              sparsity: 0.02
            }
          });
          processedOperations.push('sdr_encode');

          // Simulate cognitive processing
          const cognitiveRequest = await mockLLMKGServer.handleMessage({
            type: MCPMessageType.TOOL_CALL,
            id: `cognitive_process_${Date.now()}`,
            method: 'cognitive_process',
            params: {
              input_pattern: event.data,
              cortical_region: 'prefrontal',
              processing_mode: 'analytical'
            }
          });
          processedOperations.push('cognitive_process');

          // Stream results
          websocketServer.broadcast('llmkg_operations', {
            sdr_result: sdrRequest,
            cognitive_result: cognitiveRequest,
            timestamp: Date.now()
          });

        } catch (error) {
          console.error('LLMKG operation failed:', error);
        }
      });

      await websocketServer.start();
      await brainInspiredCollector.start();
      mcpClient.connect();

      // Process LLMKG operations
      await new Promise(resolve => setTimeout(resolve, 3000));

      await brainInspiredCollector.stop();

      // Verify LLMKG operations were processed
      expect(llmkgOperations.length).toBeGreaterThan(0);
      expect(processedOperations).toContain('sdr_encode');
      expect(processedOperations).toContain('cognitive_process');

      // Verify operation telemetry
      const sdrOperations = llmkgOperations.filter(op => op.operation === 'sdr_encode');
      const cognitiveOperations = llmkgOperations.filter(op => op.operation === 'cognitive_process');
      
      expect(sdrOperations.length).toBeGreaterThan(0);
      expect(cognitiveOperations.length).toBeGreaterThan(0);
    });

    it('should maintain data semantics through complete pipeline', async () => {
      const semanticValidations: Array<{
        stage: string;
        valid: boolean;
        dataType: string;
        reason?: string;
      }> = [];

      // Generate specific LLMKG data with known semantics
      const testCognitivePattern = LLMKGDataGenerator.generateCognitivePattern({
        pattern_id: 'semantic_test_001',
        cortical_region: 'temporal',
        activation_level: 0.75,
        pattern_strength: 85.5
      });

      const testSDRData = LLMKGDataGenerator.generateSDRData({
        sdr_id: 'semantic_sdr_001',
        size: 2048,
        sparsity: 0.02
      });

      // Setup semantic validation at each stage
      const semanticCollector = new CognitivePatternCollector(mcpClient, {
        name: 'semantic-validation',
        collectionInterval: 200
      });

      // Override collect to provide test data
      (semanticCollector as any).collect = async () => {
        return [
          {
            id: 'test-cognitive-001',
            timestamp: Date.now(),
            source: 'test',
            type: 'cognitive_pattern',
            data: testCognitivePattern,
            metadata: {
              collector: 'semantic-validation',
              method: 'test',
              tags: { test: 'semantic' }
            }
          },
          {
            id: 'test-sdr-001',
            timestamp: Date.now(),
            source: 'test',
            type: 'sdr_data',
            data: testSDRData,
            metadata: {
              collector: 'semantic-validation',
              method: 'test',
              tags: { test: 'semantic' }
            }
          }
        ];
      };

      // Validate at collection stage
      semanticCollector.on('data:collected', (event) => {
        const data = event.data;
        
        if (data.type === 'cognitive_pattern') {
          const isValid = 
            data.pattern_id === 'semantic_test_001' &&
            data.cortical_region === 'temporal' &&
            data.activation_level === 0.75 &&
            data.pattern_strength === 85.5;
            
          semanticValidations.push({
            stage: 'collection',
            valid: isValid,
            dataType: 'cognitive_pattern',
            reason: isValid ? undefined : 'Cognitive pattern semantics corrupted'
          });
        }

        if (data.type === 'sdr_data') {
          const isValid = 
            data.sdr_id === 'semantic_sdr_001' &&
            data.size === 2048 &&
            data.sparsity === 0.02 &&
            Array.isArray(data.active_bits);
            
          semanticValidations.push({
            stage: 'collection',
            valid: isValid,
            dataType: 'sdr_data',
            reason: isValid ? undefined : 'SDR data semantics corrupted'
          });
        }

        // Forward to WebSocket with semantic preservation
        websocketServer.broadcast('semantic_test', {
          original_data: data,
          semantic_validation: 'passed',
          timestamp: Date.now()
        });
      });

      // Validate at WebSocket stage
      websocketClient.on('message', (message) => {
        if (message.topic === 'semantic_test') {
          const originalData = message.data.original_data;
          
          if (originalData.type === 'cognitive_pattern') {
            const preserved = 
              originalData.pattern_id === 'semantic_test_001' &&
              originalData.cortical_region === 'temporal' &&
              originalData.activation_level === 0.75;
              
            semanticValidations.push({
              stage: 'websocket',
              valid: preserved,
              dataType: 'cognitive_pattern'
            });
          }
        }
      });

      await websocketServer.start();
      await websocketClient.connect();
      await semanticCollector.start();
      mcpClient.connect();

      // Process semantic validation data
      await new Promise(resolve => setTimeout(resolve, 1500));

      await semanticCollector.stop();

      // Verify semantic preservation
      const failedValidations = semanticValidations.filter(v => !v.valid);
      expect(failedValidations).toHaveLength(0);
      expect(semanticValidations.length).toBeGreaterThan(0);

      // Verify validation at multiple stages
      const stages = new Set(semanticValidations.map(v => v.stage));
      expect(stages.size).toBeGreaterThan(1);
    });
  });

  describe('Real-Time Visualization Streaming', () => {
    it('should provide real-time data suitable for live visualization', async () => {
      const visualizationFrames: Array<{
        frame_id: string;
        timestamp: number;
        data_points: number;
        latency: number;
        complete: boolean;
      }> = [];

      // Setup visualization-optimized collectors
      const visualizationCollectors = [
        new CognitivePatternCollector(mcpClient, {
          name: 'viz-cognitive',
          collectionInterval: 50, // 20 FPS
          bufferSize: 500,
          autoFlush: true,
          flushInterval: 50
        }),
        new NeuralActivityCollector(mcpClient, {
          name: 'viz-neural',
          collectionInterval: 33, // 30 FPS
          bufferSize: 500,
          autoFlush: true,
          flushInterval: 33
        })
      ];

      let frameCounter = 0;
      visualizationCollectors.forEach(collector => {
        collector.on('data:flushed', (event) => {
          const frameId = `frame_${frameCounter++}`;
          const frameTime = Date.now();
          
          // Create visualization frame
          const visualizationFrame = {
            frame_id: frameId,
            frame_time: frameTime,
            collector: event.collector,
            data_count: event.count,
            data_points: event.data?.map((item: any) => ({
              id: item.id,
              timestamp: item.timestamp,
              type: item.type,
              value: item.data.activation_level || item.data.firing_rate || Math.random()
            }))
          };

          // Stream to visualization clients
          websocketServer.broadcast('visualization_frame', visualizationFrame);
          
          visualizationFrames.push({
            frame_id: frameId,
            timestamp: frameTime,
            data_points: event.count,
            latency: frameTime - (event.data?.[0]?.timestamp || frameTime),
            complete: true
          });
        });
      });

      // Setup visualization client
      const visualizationUpdates: any[] = [];
      websocketClient.on('message', (message) => {
        if (message.topic === 'visualization_frame') {
          visualizationUpdates.push(message.data);
        }
      });

      await websocketServer.start();
      await websocketClient.connect();

      // Start visualization collectors
      for (let i = 0; i < visualizationCollectors.length; i++) {
        await collectorManager.addCollector(`viz-${i}`, visualizationCollectors[i]);
      }
      await collectorManager.startAll();
      
      mcpClient.connect();

      // Stream visualization data
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Analyze visualization streaming performance
      const avgLatency = visualizationFrames.reduce((sum, frame) => sum + frame.latency, 0) / visualizationFrames.length;
      const frameRate = visualizationFrames.length / 2; // frames per second
      const totalDataPoints = visualizationFrames.reduce((sum, frame) => sum + frame.data_points, 0);

      expect(visualizationFrames.length).toBeGreaterThan(20); // Reasonable frame count
      expect(avgLatency).toBeLessThan(100);                   // <100ms latency for visualization
      expect(frameRate).toBeGreaterThan(10);                  // >10 FPS
      expect(totalDataPoints).toBeGreaterThan(100);           // Sufficient data density
      expect(visualizationUpdates.length).toBeGreaterThan(0); // Client received updates
    });

    it('should support multiple concurrent visualization clients', async () => {
      const clientCount = 5;
      const clients: WebSocketClient[] = [];
      const clientData: Array<{ clientId: number; messages: any[] }> = [];

      // Create multiple visualization clients
      for (let i = 0; i < clientCount; i++) {
        const client = new WebSocketClient({
          serverUrl: 'ws://localhost:8088',
          reconnectInterval: 1000
        });

        const clientMessages: any[] = [];
        client.on('message', (message) => {
          clientMessages.push(message);
        });

        clients.push(client);
        clientData.push({ clientId: i, messages: clientMessages });
      }

      // Setup multi-client data streaming
      const multiClientCollector = new CognitivePatternCollector(mcpClient, {
        name: 'multi-client-test',
        collectionInterval: 100,
        autoFlush: true,
        flushInterval: 200
      });

      multiClientCollector.on('data:flushed', (event) => {
        // Broadcast to all connected clients
        websocketServer.broadcastToAll({
          type: 'multi_client_data',
          id: `multi_${Date.now()}`,
          timestamp: Date.now(),
          source: 'server',
          data: {
            batch_size: event.count,
            collector: event.collector,
            clients_expected: clientCount
          }
        });
      });

      await websocketServer.start();

      // Connect all clients
      for (const client of clients) {
        await client.connect();
      }

      await multiClientCollector.start();
      mcpClient.connect();

      // Stream to multiple clients
      await new Promise(resolve => setTimeout(resolve, 2000));

      await multiClientCollector.stop();

      // Disconnect all clients
      for (const client of clients) {
        await client.disconnect();
      }

      // Verify all clients received data
      expect(clientData.length).toBe(clientCount);
      clientData.forEach((client, index) => {
        expect(client.messages.length).toBeGreaterThan(0);
      });

      // Verify data consistency across clients
      const firstClientMessageCount = clientData[0].messages.length;
      clientData.forEach(client => {
        expect(Math.abs(client.messages.length - firstClientMessageCount)).toBeLessThan(5);
      });
    });

    it('should handle visualization data filtering and subscriptions', async () => {
      const subscriptionData: Record<string, any[]> = {
        cognitive_patterns: [],
        neural_activity: [],
        sdr_operations: [],
        memory_updates: []
      };

      // Setup topic-specific collectors
      const cognitiveCollector = new CognitivePatternCollector(mcpClient, {
        name: 'subscription-cognitive',
        collectionInterval: 100
      });

      const neuralCollector = new NeuralActivityCollector(mcpClient, {
        name: 'subscription-neural', 
        collectionInterval: 80
      });

      // Setup topic-based broadcasting
      cognitiveCollector.on('data:collected', (event) => {
        websocketServer.broadcast('cognitive_patterns', {
          type: 'cognitive_update',
          pattern_data: event.data,
          timestamp: Date.now()
        });
      });

      neuralCollector.on('data:collected', (event) => {
        websocketServer.broadcast('neural_activity', {
          type: 'neural_update',
          activity_data: event.data,
          timestamp: Date.now()
        });
      });

      // Setup selective client subscriptions
      const selectiveClient = new WebSocketClient({
        serverUrl: 'ws://localhost:8088'
      });

      selectiveClient.on('message', (message) => {
        const topic = message.topic;
        if (topic && subscriptionData[topic]) {
          subscriptionData[topic].push(message.data);
        }
      });

      // Simulate subscription management
      selectiveClient.on('connect', async () => {
        // Subscribe to specific topics
        await selectiveClient.send({
          type: 'subscribe',
          topics: ['cognitive_patterns', 'neural_activity'],
          filters: {
            min_activation: 0.5,
            cortical_regions: ['prefrontal', 'temporal']
          }
        });
      });

      await websocketServer.start();
      await selectiveClient.connect();

      await collectorManager.addCollector('cognitive', cognitiveCollector);
      await collectorManager.addCollector('neural', neuralCollector);
      await collectorManager.startAll();
      
      mcpClient.connect();

      // Generate subscription-filtered data
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Verify subscription filtering worked
      expect(subscriptionData.cognitive_patterns.length).toBeGreaterThan(0);
      expect(subscriptionData.neural_activity.length).toBeGreaterThan(0);
      expect(subscriptionData.sdr_operations.length).toBe(0); // Not subscribed
      expect(subscriptionData.memory_updates.length).toBe(0); // Not subscribed

      await selectiveClient.disconnect();
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should recover from component failures without data loss', async () => {
      const recoveryEvents: Array<{
        event: string;
        component: string;
        timestamp: number;
      }> = [];

      const dataLossTracker = {
        beforeFailure: 0,
        duringFailure: 0,
        afterRecovery: 0
      };

      // Setup resilient collector with failure simulation
      const resilientCollector = new CognitivePatternCollector(mcpClient, {
        name: 'resilient-test',
        collectionInterval: 50,
        bufferSize: 2000, // Large buffer to prevent data loss
        autoFlush: true,
        flushInterval: 100
      });

      let phase = 'normal';
      let failureCount = 0;

      // Inject controlled failures
      const originalFlush = (resilientCollector as any).processFlush.bind(resilientCollector);
      (resilientCollector as any).processFlush = async function(data: any[]) {
        if (phase === 'failure' && failureCount < 5) {
          failureCount++;
          recoveryEvents.push({
            event: 'flush_failure',
            component: 'collector',
            timestamp: Date.now()
          });
          throw new Error('Simulated flush failure');
        }
        return originalFlush(data);
      };

      resilientCollector.on('data:collected', (event) => {
        switch (phase) {
          case 'normal':
            dataLossTracker.beforeFailure++;
            break;
          case 'failure':
            dataLossTracker.duringFailure++;
            break;
          case 'recovery':
            dataLossTracker.afterRecovery++;
            break;
        }
      });

      resilientCollector.on('flush:error', (event) => {
        recoveryEvents.push({
          event: 'flush_error_handled',
          component: 'collector',
          timestamp: Date.now()
        });
      });

      resilientCollector.on('data:flushed', (event) => {
        websocketServer.broadcast('resilience_test', {
          batch_size: event.count,
          phase: phase,
          timestamp: Date.now()
        });

        if (phase === 'failure' && failureCount >= 5) {
          phase = 'recovery';
          recoveryEvents.push({
            event: 'entering_recovery',
            component: 'system',
            timestamp: Date.now()
          });
        }
      });

      await websocketServer.start();
      await resilientCollector.start();
      mcpClient.connect();

      // Phase 1: Normal operation
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Phase 2: Simulate failure
      phase = 'failure';
      recoveryEvents.push({
        event: 'failure_injected',
        component: 'system',
        timestamp: Date.now()
      });
      
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Phase 3: Let system recover
      await new Promise(resolve => setTimeout(resolve, 800));

      await resilientCollector.stop();

      // Verify recovery behavior
      expect(dataLossTracker.beforeFailure).toBeGreaterThan(0);
      expect(dataLossTracker.duringFailure).toBeGreaterThan(0);
      expect(dataLossTracker.afterRecovery).toBeGreaterThan(0);

      const failureEvents = recoveryEvents.filter(e => e.event === 'flush_failure');
      const errorHandlingEvents = recoveryEvents.filter(e => e.event === 'flush_error_handled');
      const recoveryStartEvents = recoveryEvents.filter(e => e.event === 'entering_recovery');

      expect(failureEvents.length).toBeGreaterThan(0);          // Failures occurred
      expect(errorHandlingEvents.length).toBeGreaterThan(0);    // Errors were handled
      expect(recoveryStartEvents.length).toBeGreaterThan(0);    // Recovery initiated
      
      // Data should continue flowing after recovery
      expect(dataLossTracker.afterRecovery).toBeGreaterThan(dataLossTracker.duringFailure * 0.5);
    });

    it('should maintain service availability during partial component failures', async () => {
      const availabilityMetrics = {
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        serviceUptime: 0
      };

      // Setup multiple collectors with different failure modes
      const collectors = [
        new CognitivePatternCollector(mcpClient, { 
          name: 'availability-cognitive',
          collectionInterval: 100 
        }),
        new NeuralActivityCollector(mcpClient, { 
          name: 'availability-neural',
          collectionInterval: 120 
        }),
        new MemorySystemCollector(mcpClient, { 
          name: 'availability-memory',
          collectionInterval: 150 
        })
      ];

      let failingCollectorIndex = 0;
      
      collectors.forEach((collector, index) => {
        collector.on('data:collected', (event) => {
          availabilityMetrics.totalRequests++;
          
          // Simulate one collector failing
          if (index === failingCollectorIndex && Math.random() < 0.3) {
            availabilityMetrics.failedRequests++;
            throw new Error('Simulated collection failure');
          }
          
          availabilityMetrics.successfulRequests++;
          
          websocketServer.broadcast('availability_test', {
            collector: event.collector,
            success: true,
            timestamp: Date.now()
          });
        });

        collector.on('collection:error', () => {
          // Error handled, service continues
        });
      });

      // Monitor service availability
      const serviceMonitor = setInterval(() => {
        availabilityMetrics.serviceUptime += 100;
        
        // Simulate switching which collector fails
        if (availabilityMetrics.serviceUptime % 2000 === 0) {
          failingCollectorIndex = (failingCollectorIndex + 1) % collectors.length;
        }
      }, 100);

      await websocketServer.start();
      
      // Start all collectors
      for (let i = 0; i < collectors.length; i++) {
        await collectorManager.addCollector(`availability-${i}`, collectors[i]);
      }
      await collectorManager.startAll();
      
      mcpClient.connect();

      // Run availability test
      await new Promise(resolve => setTimeout(resolve, 5000));

      clearInterval(serviceMonitor);

      // Calculate availability metrics
      const availability = availabilityMetrics.successfulRequests / availabilityMetrics.totalRequests;
      const failureRate = availabilityMetrics.failedRequests / availabilityMetrics.totalRequests;

      expect(availability).toBeGreaterThan(0.7);     // >70% availability despite failures
      expect(failureRate).toBeLessThan(0.3);         // <30% failure rate
      expect(availabilityMetrics.totalRequests).toBeGreaterThan(50); // Significant request volume
      expect(websocketServer.isActive()).toBe(true); // WebSocket service remained up
    });
  });

  describe('Performance Under Load', () => {
    it('should maintain performance with high concurrent visualization clients', async () => {
      const concurrentClientCount = 20;
      const clients: WebSocketClient[] = [];
      const performanceMetrics = {
        messagesPerClient: [] as number[],
        averageLatency: 0,
        totalThroughput: 0
      };

      // Create many concurrent clients
      for (let i = 0; i < concurrentClientCount; i++) {
        const client = new WebSocketClient({
          serverUrl: 'ws://localhost:8088'
        });

        let clientMessageCount = 0;
        client.on('message', () => {
          clientMessageCount++;
        });

        clients.push(client);
        performanceMetrics.messagesPerClient.push(clientMessageCount);
      }

      // Setup high-throughput collector
      const loadTestCollector = new NeuralActivityCollector(mcpClient, {
        name: 'load-test-collector',
        collectionInterval: 20, // 50 Hz
        bufferSize: 5000,
        autoFlush: true,
        flushInterval: 50
      });

      let totalBroadcasts = 0;
      loadTestCollector.on('data:flushed', (event) => {
        totalBroadcasts += event.count;
        
        // Broadcast to all concurrent clients
        websocketServer.broadcastToAll({
          type: 'load_test_data',
          id: `load_${Date.now()}`,
          timestamp: Date.now(),
          source: 'server',
          data: {
            batch_size: event.count,
            client_count: concurrentClientCount,
            performance_test: true
          }
        });
      });

      await websocketServer.start();

      // Connect all clients concurrently
      await Promise.all(clients.map(client => client.connect()));

      await loadTestCollector.start();
      mcpClient.connect();

      performanceTracker.start('concurrent_client_load');
      
      // Run under concurrent load
      await new Promise(resolve => setTimeout(resolve, 3000));

      const duration = performanceTracker.end('concurrent_client_load');

      // Disconnect all clients
      await Promise.all(clients.map(client => client.disconnect()));
      await loadTestCollector.stop();

      // Analyze concurrent performance
      const totalClientMessages = performanceMetrics.messagesPerClient.reduce((sum, count) => sum + count, 0);
      const avgMessagesPerClient = totalClientMessages / concurrentClientCount;
      const systemThroughput = totalBroadcasts / (duration / 1000);

      expect(avgMessagesPerClient).toBeGreaterThan(10);          // Each client got data
      expect(systemThroughput).toBeGreaterThan(500);             // System maintained throughput
      expect(totalClientMessages).toBeGreaterThan(200);         // Significant total volume
      
      // Verify fairly even distribution across clients
      const messageVariance = performanceMetrics.messagesPerClient.reduce((variance, count) => {
        return variance + Math.pow(count - avgMessagesPerClient, 2);
      }, 0) / concurrentClientCount;
      
      expect(messageVariance).toBeLessThan(avgMessagesPerClient * avgMessagesPerClient * 0.5);
    });
  });
});