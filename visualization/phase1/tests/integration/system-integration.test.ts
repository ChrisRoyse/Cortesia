/**
 * Integration tests for complete LLMKG Visualization system
 * 
 * Tests the integration of all Phase 1 components working together:
 * MCP Client -> Data Collectors -> Telemetry System -> WebSocket Communication
 */

import { MCPClient } from '../../src/mcp/client';
import { BaseCollector } from '../../src/collectors/base';
import { CollectorManager } from '../../src/collectors/manager';
import { CognitivePatternCollector } from '../../src/collectors/cognitive-patterns';
import { NeuralActivityCollector } from '../../src/collectors/neural-activity';
import { WebSocketServer } from '../../src/websocket/server';
import { TelemetryInjector } from '../../src/telemetry/injector';
import { MockBrainInspiredMCPServer } from '../mocks/brain-inspired-server';
import { PerformanceTracker, TestHelpers, LLMKGDataGenerator } from '../config/test-utils';

describe('System Integration Tests', () => {
  let mcpClient: MCPClient;
  let collectorManager: CollectorManager;
  let websocketServer: WebSocketServer;
  let telemetryInjector: TelemetryInjector;
  let mockLLMKGServer: MockBrainInspiredMCPServer;
  let performanceTracker: PerformanceTracker;

  beforeEach(async () => {
    performanceTracker = new PerformanceTracker();
    
    // Initialize mock LLMKG server
    mockLLMKGServer = new MockBrainInspiredMCPServer();
    await mockLLMKGServer.start();

    // Initialize MCP client
    mcpClient = new MCPClient({
      enableRealtimeUpdates: true,
      updateInterval: 100,
      serverUrl: 'ws://localhost:3000'
    });

    // Initialize telemetry system
    telemetryInjector = new TelemetryInjector();
    await telemetryInjector.initialize();

    // Inject telemetry into MCP client
    mcpClient = telemetryInjector.injectMCPClient(mcpClient);

    // Initialize WebSocket server
    websocketServer = new WebSocketServer({
      port: 8081,
      heartbeatInterval: 1000,
      enableBuffering: true
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

  describe('Complete Data Pipeline', () => {
    it('should stream data from MCP client through collectors to WebSocket', async () => {
      const dataReceivedSpy = jest.fn();
      const telemetryReceivedSpy = jest.fn();

      // Setup WebSocket data reception
      websocketServer.on('clientConnected', (client) => {
        websocketServer.sendToClient(client.id, {
          type: 'test_data',
          id: 'integration-test',
          timestamp: Date.now(),
          source: 'server',
          data: { message: 'WebSocket connected' }
        });
      });

      // Setup collector to forward data to WebSocket
      const cognitiveCollector = new CognitivePatternCollector(mcpClient);
      cognitiveCollector.on('data:collected', (event) => {
        dataReceivedSpy(event);
        websocketServer.broadcast('cognitive_patterns', event.data);
      });

      cognitiveCollector.on('telemetry', (event) => {
        telemetryReceivedSpy(event);
        websocketServer.broadcast('telemetry', event.data);
      });

      // Start all components
      await websocketServer.start();
      await collectorManager.addCollector('cognitive', cognitiveCollector);
      await collectorManager.startCollector('cognitive');
      mcpClient.connect();

      // Wait for data to flow through the pipeline
      await TestHelpers.waitFor(() => dataReceivedSpy.mock.calls.length > 0, 5000);

      expect(dataReceivedSpy).toHaveBeenCalled();
      expect(websocketServer.isActive()).toBe(true);
      expect(collectorManager.isRunning('cognitive')).toBe(true);
    });

    it('should handle high-frequency data streaming', async () => {
      const highFreqConfig = {
        enableRealtimeUpdates: true,
        updateInterval: 20, // 50 Hz
        serverUrl: 'ws://localhost:3000'
      };

      const highFreqClient = new MCPClient(highFreqConfig);
      const dataCounter = { count: 0 };

      const neuralCollector = new NeuralActivityCollector(highFreqClient, {
        name: 'high-freq-neural',
        collectionInterval: 20,
        bufferSize: 1000,
        autoFlush: true,
        flushInterval: 100
      });

      neuralCollector.on('data:collected', () => {
        dataCounter.count++;
      });

      neuralCollector.on('data:flushed', (event) => {
        websocketServer.broadcast('neural_activity_batch', {
          batch_size: event.count,
          timestamp: Date.now()
        });
      });

      await websocketServer.start();
      await neuralCollector.start();
      highFreqClient.connect();

      performanceTracker.start('high_frequency_streaming');

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      const duration = performanceTracker.end('high_frequency_streaming');
      const frequency = dataCounter.count / (duration / 1000);

      await neuralCollector.stop();
      highFreqClient.disconnect();

      // Should achieve close to target frequency
      expect(frequency).toBeGreaterThan(30); // Allow for some variance
      expect(dataCounter.count).toBeGreaterThan(60); // At least 60 data points in 2 seconds
    });

    it('should maintain data integrity across all components', async () => {
      const dataIntegrityTracker = new Map<string, any>();
      const integrityViolations: string[] = [];

      // Track data through each stage
      mcpClient.on('dataUpdate', (data) => {
        dataIntegrityTracker.set(data.timestamp.toString(), {
          stage: 'mcp_client',
          data: JSON.parse(JSON.stringify(data))
        });
      });

      const cognitiveCollector = new CognitivePatternCollector(mcpClient);
      cognitiveCollector.on('data:collected', (event) => {
        const originalData = dataIntegrityTracker.get(event.data.timestamp?.toString());
        if (originalData) {
          dataIntegrityTracker.set(event.data.timestamp.toString(), {
            ...originalData,
            stage: 'collector',
            collectedData: event.data
          });
        }
      });

      cognitiveCollector.on('data:flushed', (event) => {
        event.data?.forEach((item: any) => {
          const trackedData = dataIntegrityTracker.get(item.timestamp?.toString());
          if (trackedData) {
            // Verify data hasn't been corrupted
            if (JSON.stringify(trackedData.data) !== JSON.stringify(item.data)) {
              integrityViolations.push(`Data corruption detected at timestamp ${item.timestamp}`);
            }
          }
        });

        websocketServer.broadcast('cognitive_patterns', {
          count: event.count,
          timestamp: Date.now(),
          data: event.data
        });
      });

      await websocketServer.start();
      await cognitiveCollector.start();
      mcpClient.connect();

      // Let data flow for a while
      await new Promise(resolve => setTimeout(resolve, 1000));

      await cognitiveCollector.stop();

      expect(integrityViolations).toHaveLength(0);
      expect(dataIntegrityTracker.size).toBeGreaterThan(0);
    });
  });

  describe('Component Interaction', () => {
    it('should handle collector failures gracefully without affecting other components', async () => {
      const systemErrorSpy = jest.fn();
      const workingCollectorSpy = jest.fn();
      
      // Create a failing collector
      const failingCollector = new CognitivePatternCollector(mcpClient, {
        name: 'failing-collector',
        collectionInterval: 50
      });

      // Override collect method to fail
      (failingCollector as any).collect = async () => {
        throw new Error('Simulated collector failure');
      };

      failingCollector.on('collection:error', systemErrorSpy);

      // Create a working collector
      const workingCollector = new NeuralActivityCollector(mcpClient, {
        name: 'working-collector',
        collectionInterval: 50
      });

      workingCollector.on('data:collected', workingCollectorSpy);

      await websocketServer.start();
      
      // Add both collectors
      await collectorManager.addCollector('failing', failingCollector);
      await collectorManager.addCollector('working', workingCollector);
      
      await collectorManager.startAll();
      mcpClient.connect();

      // Wait for operations
      await new Promise(resolve => setTimeout(resolve, 500));

      // Failing collector should emit errors
      expect(systemErrorSpy).toHaveBeenCalled();
      
      // Working collector should continue functioning
      expect(workingCollectorSpy).toHaveBeenCalled();
      
      // WebSocket server should still be running
      expect(websocketServer.isActive()).toBe(true);
    });

    it('should coordinate multiple collectors efficiently', async () => {
      const collectors = [
        new CognitivePatternCollector(mcpClient, { name: 'cognitive-1', collectionInterval: 100 }),
        new NeuralActivityCollector(mcpClient, { name: 'neural-1', collectionInterval: 80 }),
        new CognitivePatternCollector(mcpClient, { name: 'cognitive-2', collectionInterval: 120 })
      ];

      const collectionEvents: Array<{ collector: string, timestamp: number }> = [];

      collectors.forEach((collector, index) => {
        collector.on('data:collected', (event) => {
          collectionEvents.push({
            collector: event.collector,
            timestamp: Date.now()
          });
        });
      });

      await websocketServer.start();
      
      // Add all collectors
      for (let i = 0; i < collectors.length; i++) {
        await collectorManager.addCollector(`collector-${i}`, collectors[i]);
      }
      
      await collectorManager.startAll();
      mcpClient.connect();

      // Run for sufficient time to see coordination
      await new Promise(resolve => setTimeout(resolve, 1000));

      await collectorManager.stopAll();

      // Should have events from all collectors
      const collectorNames = new Set(collectionEvents.map(e => e.collector));
      expect(collectorNames.size).toBe(3);

      // Events should be roughly distributed according to collection intervals
      const cognitiveEvents = collectionEvents.filter(e => e.collector.includes('cognitive'));
      const neuralEvents = collectionEvents.filter(e => e.collector.includes('neural'));
      
      expect(cognitiveEvents.length).toBeGreaterThan(0);
      expect(neuralEvents.length).toBeGreaterThan(0);
    });

    it('should handle telemetry injection across all components', async () => {
      const telemetryEvents: any[] = [];

      // Setup telemetry collection
      const cognitiveCollector = new CognitivePatternCollector(mcpClient);
      cognitiveCollector.on('telemetry', (event) => {
        telemetryEvents.push({
          source: 'collector',
          event: event
        });
      });

      // Mock telemetry from server
      mockLLMKGServer.on('telemetry', (event) => {
        telemetryEvents.push({
          source: 'server',
          event: event
        });
      });

      await websocketServer.start();
      await cognitiveCollector.start();
      mcpClient.connect();

      // Generate some operations to trigger telemetry
      await new Promise(resolve => setTimeout(resolve, 1000));

      await cognitiveCollector.stop();

      // Should have received telemetry from multiple sources
      const sources = new Set(telemetryEvents.map(e => e.source));
      expect(sources.size).toBeGreaterThan(0);
      expect(telemetryEvents.length).toBeGreaterThan(0);
    });
  });

  describe('Performance Integration', () => {
    it('should achieve end-to-end latency <100ms', async () => {
      const latencyMeasurements: number[] = [];
      
      // Configure for low latency
      const lowLatencyClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 50,
        serverUrl: 'ws://localhost:3000'
      });

      const collector = new CognitivePatternCollector(lowLatencyClient, {
        name: 'latency-test',
        collectionInterval: 50,
        bufferSize: 100,
        autoFlush: true,
        flushInterval: 50
      });

      collector.on('data:collected', (event) => {
        const now = Date.now();
        const dataTimestamp = event.data.timestamp;
        const latency = now - dataTimestamp;
        latencyMeasurements.push(latency);

        // Immediately forward to WebSocket (simulating real-time streaming)
        websocketServer.broadcast('real_time_cognitive', {
          ...event.data,
          processing_latency: latency
        });
      });

      await websocketServer.start();
      await collector.start();
      lowLatencyClient.connect();

      // Collect latency data
      await TestHelpers.waitFor(() => latencyMeasurements.length >= 10, 3000);

      await collector.stop();
      lowLatencyClient.disconnect();

      // Analyze latency distribution
      const avgLatency = latencyMeasurements.reduce((a, b) => a + b, 0) / latencyMeasurements.length;
      const maxLatency = Math.max(...latencyMeasurements);
      const p95Latency = latencyMeasurements.sort((a, b) => a - b)[Math.floor(latencyMeasurements.length * 0.95)];

      expect(avgLatency).toBeLessThan(50);  // Average <50ms
      expect(p95Latency).toBeLessThan(100); // 95th percentile <100ms
      expect(maxLatency).toBeLessThan(200); // Max <200ms (allowing for some variance)
    });

    it('should handle >1000 messages/second throughput', async () => {
      const messageCount = { sent: 0, received: 0 };
      
      // Configure for high throughput
      const highThroughputClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 10, // 100 Hz
        serverUrl: 'ws://localhost:3000'
      });

      const collector = new NeuralActivityCollector(highThroughputClient, {
        name: 'throughput-test',
        collectionInterval: 10,
        bufferSize: 10000,
        autoFlush: true,
        flushInterval: 100
      });

      collector.on('data:flushed', (event) => {
        messageCount.sent += event.count;
        websocketServer.broadcast('high_throughput_neural', {
          batch_count: event.count,
          timestamp: Date.now()
        });
      });

      websocketServer.on('message_sent', () => {
        messageCount.received++;
      });

      await websocketServer.start();
      await collector.start();
      highThroughputClient.connect();

      performanceTracker.start('throughput_test');

      // Run for 3 seconds
      await new Promise(resolve => setTimeout(resolve, 3000));

      const duration = performanceTracker.end('throughput_test');
      await collector.stop();
      highThroughputClient.disconnect();

      const throughput = messageCount.sent / (duration / 1000);

      expect(throughput).toBeGreaterThan(1000); // >1000 messages/second
      expect(messageCount.sent).toBeGreaterThan(3000); // At least 3000 messages in 3 seconds
    });

    it('should maintain stable memory usage under load', async () => {
      const memoryReadings: number[] = [];
      
      const loadTestClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 20,
        serverUrl: 'ws://localhost:3000'
      });

      const collectors = [
        new CognitivePatternCollector(loadTestClient, { 
          name: 'memory-test-1', 
          collectionInterval: 25,
          bufferSize: 5000
        }),
        new NeuralActivityCollector(loadTestClient, { 
          name: 'memory-test-2', 
          collectionInterval: 30,
          bufferSize: 5000
        })
      ];

      // Monitor memory usage
      const memoryMonitor = setInterval(() => {
        const usage = process.memoryUsage().heapUsed / (1024 * 1024); // MB
        memoryReadings.push(usage);
      }, 500);

      await websocketServer.start();
      
      for (let i = 0; i < collectors.length; i++) {
        await collectorManager.addCollector(`memory-test-${i}`, collectors[i]);
      }
      
      await collectorManager.startAll();
      loadTestClient.connect();

      // Run under load for 5 seconds
      await new Promise(resolve => setTimeout(resolve, 5000));

      clearInterval(memoryMonitor);
      await collectorManager.stopAll();
      loadTestClient.disconnect();

      // Analyze memory stability
      const initialMemory = memoryReadings[0];
      const finalMemory = memoryReadings[memoryReadings.length - 1];
      const maxMemory = Math.max(...memoryReadings);
      const memoryGrowth = finalMemory - initialMemory;

      expect(memoryGrowth).toBeLessThan(50); // <50MB growth
      expect(maxMemory).toBeLessThan(initialMemory + 100); // <100MB total increase
    });
  });

  describe('Error Recovery Integration', () => {
    it('should recover from temporary component failures', async () => {
      const recoveryEvents: string[] = [];
      
      const resilientCollector = new CognitivePatternCollector(mcpClient, {
        name: 'resilient-collector',
        collectionInterval: 100
      });

      resilientCollector.on('collection:error', () => {
        recoveryEvents.push('collector_error');
      });

      resilientCollector.on('data:collected', () => {
        recoveryEvents.push('collector_recovery');
      });

      // Inject periodic failures
      let failureCount = 0;
      const originalCollect = (resilientCollector as any).collect.bind(resilientCollector);
      (resilientCollector as any).collect = async () => {
        failureCount++;
        if (failureCount % 3 === 0) { // Fail every 3rd call
          throw new Error('Temporary failure');
        }
        return originalCollect();
      };

      await websocketServer.start();
      await resilientCollector.start();
      mcpClient.connect();

      // Run long enough to see failures and recoveries
      await new Promise(resolve => setTimeout(resolve, 2000));

      await resilientCollector.stop();

      const errors = recoveryEvents.filter(e => e === 'collector_error');
      const recoveries = recoveryEvents.filter(e => e === 'collector_recovery');

      expect(errors.length).toBeGreaterThan(0);
      expect(recoveries.length).toBeGreaterThan(0);
      expect(recoveries.length).toBeGreaterThan(errors.length); // More recoveries than errors
    });

    it('should handle network disconnections gracefully', async () => {
      const connectionEvents: string[] = [];
      
      mcpClient.on('connected', () => connectionEvents.push('connected'));
      mcpClient.on('disconnected', () => connectionEvents.push('disconnected'));

      const networkResilientCollector = new NeuralActivityCollector(mcpClient, {
        name: 'network-resilient',
        collectionInterval: 100
      });

      const connectionStates: boolean[] = [];
      networkResilientCollector.on('data:collected', () => {
        connectionStates.push(true);
      });

      await websocketServer.start();
      await networkResilientCollector.start();
      
      // Start connected
      mcpClient.connect();
      await new Promise(resolve => setTimeout(resolve, 300));

      // Simulate network disconnection
      mcpClient.disconnect();
      await new Promise(resolve => setTimeout(resolve, 300));

      // Reconnect
      mcpClient.connect();
      await new Promise(resolve => setTimeout(resolve, 300));

      await networkResilientCollector.stop();

      expect(connectionEvents).toContain('connected');
      expect(connectionEvents).toContain('disconnected');
      expect(connectionStates.length).toBeGreaterThan(0);
    });
  });

  describe('LLMKG Data Flow Integration', () => {
    it('should process complete LLMKG cognitive data pipeline', async () => {
      const cognitiveDataTypes = [
        'cognitive_pattern',
        'sdr_data', 
        'neural_activity',
        'memory_data',
        'attention_data',
        'knowledge_graph_data'
      ];

      const processedDataTypes = new Set<string>();
      
      // Setup collectors for different data types
      const cognitiveCollector = new CognitivePatternCollector(mcpClient);
      const neuralCollector = new NeuralActivityCollector(mcpClient);

      cognitiveCollector.on('data:collected', (event) => {
        processedDataTypes.add(event.data.type);
        websocketServer.broadcast('cognitive_pipeline', {
          data_type: event.data.type,
          timestamp: Date.now(),
          processed: true
        });
      });

      neuralCollector.on('data:collected', (event) => {
        processedDataTypes.add(event.data.type);
        websocketServer.broadcast('neural_pipeline', {
          data_type: event.data.type,
          timestamp: Date.now(),
          processed: true
        });
      });

      // Generate synthetic LLMKG data
      const dataGenerator = setInterval(() => {
        const dataType = cognitiveDataTypes[Math.floor(Math.random() * cognitiveDataTypes.length)];
        const data = LLMKGDataGenerator.generateMixedLLMKGBatch(1)[0];
        data.type = dataType;
        
        mcpClient.emit('dataUpdate', data);
      }, 50);

      await websocketServer.start();
      await cognitiveCollector.start();
      await neuralCollector.start();
      mcpClient.connect();

      // Process data for a while
      await new Promise(resolve => setTimeout(resolve, 2000));

      clearInterval(dataGenerator);
      await cognitiveCollector.stop();
      await neuralCollector.stop();

      // Should have processed multiple data types
      expect(processedDataTypes.size).toBeGreaterThan(0);
    });

    it('should maintain LLMKG data semantics across transformations', async () => {
      const semanticValidations: Array<{ valid: boolean, reason?: string }> = [];
      
      const semanticCollector = new CognitivePatternCollector(mcpClient);
      
      semanticCollector.on('data:collected', (event) => {
        const data = event.data;
        
        // Validate LLMKG-specific data semantics
        if (data.type === 'cognitive_pattern') {
          const isValid = 
            data.pattern_id &&
            typeof data.activation_level === 'number' &&
            data.activation_level >= 0 && data.activation_level <= 1 &&
            data.cortical_region &&
            Array.isArray(data.connections);
            
          semanticValidations.push({
            valid: isValid,
            reason: isValid ? undefined : 'Invalid cognitive pattern structure'
          });
        }
        
        if (data.type === 'sdr_data') {
          const isValid = 
            data.sdr_id &&
            typeof data.size === 'number' &&
            typeof data.sparsity === 'number' &&
            Array.isArray(data.active_bits);
            
          semanticValidations.push({
            valid: isValid,
            reason: isValid ? undefined : 'Invalid SDR data structure'
          });
        }
      });

      await websocketServer.start();
      await semanticCollector.start();
      mcpClient.connect();

      // Generate various LLMKG data types
      const testDataTypes = ['cognitive_pattern', 'sdr_data', 'neural_activity'];
      for (const dataType of testDataTypes) {
        const testData = LLMKGDataGenerator.generateMixedLLMKGBatch(5);
        testData.forEach(data => {
          data.type = dataType;
          mcpClient.emit('dataUpdate', data);
        });
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
      await semanticCollector.stop();

      // All semantic validations should pass
      const invalidItems = semanticValidations.filter(v => !v.valid);
      expect(invalidItems.length).toBe(0);
      expect(semanticValidations.length).toBeGreaterThan(0);
    });
  });
});