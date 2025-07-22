/**
 * Latency Performance Tests for LLMKG Visualization Phase 1
 * 
 * Tests to validate <100ms latency requirements across all components
 * and ensure real-time performance characteristics.
 */

import { MCPClient } from '../../src/mcp/client';
import { CognitivePatternCollector } from '../../src/collectors/cognitive-patterns';
import { NeuralActivityCollector } from '../../src/collectors/neural-activity';
import { WebSocketServer } from '../../src/websocket/server';
import { TelemetryInjector } from '../../src/telemetry/injector';
import { MockBrainInspiredMCPServer } from '../mocks/brain-inspired-server';
import { PerformanceTracker, LoadTestRunner, TestHelpers, LLMKGDataGenerator } from '../config/test-utils';

describe('Latency Performance Tests', () => {
  let performanceTracker: PerformanceTracker;
  let loadTestRunner: LoadTestRunner;

  beforeEach(() => {
    performanceTracker = new PerformanceTracker();
    loadTestRunner = new LoadTestRunner();
  });

  afterEach(() => {
    performanceTracker.clear();
    loadTestRunner.stop();
  });

  describe('MCP Client Latency', () => {
    let mcpClient: MCPClient;
    let mockServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockServer = new MockBrainInspiredMCPServer();
      // Configure for minimal processing delay
      mockServer.configureProcessingDelay(1, 5);
      await mockServer.start();

      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 10, // High frequency for latency testing
        serverUrl: 'ws://localhost:3000'
      });
    });

    afterEach(async () => {
      if (mcpClient) {
        mcpClient.disconnect();
      }
      if (mockServer) {
        await mockServer.stop();
      }
    });

    it('should achieve data generation latency <10ms', async () => {
      const latencyMeasurements: number[] = [];
      
      mcpClient.on('dataUpdate', (data) => {
        const now = Date.now();
        const latency = now - data.timestamp;
        latencyMeasurements.push(latency);
      });

      mcpClient.connect();

      // Collect latency measurements for 2 seconds
      await TestHelpers.waitFor(() => latencyMeasurements.length >= 50, 3000);

      mcpClient.disconnect();

      // Analyze latency distribution
      const avgLatency = latencyMeasurements.reduce((a, b) => a + b, 0) / latencyMeasurements.length;
      const maxLatency = Math.max(...latencyMeasurements);
      const p95Latency = latencyMeasurements.sort((a, b) => a - b)[Math.floor(latencyMeasurements.length * 0.95)];
      const p99Latency = latencyMeasurements[Math.floor(latencyMeasurements.length * 0.99)];

      expect(avgLatency).toBeLessThan(10); // Average <10ms
      expect(p95Latency).toBeLessThan(10); // 95th percentile <10ms
      expect(p99Latency).toBeLessThan(20); // 99th percentile <20ms
      expect(maxLatency).toBeLessThan(50); // Max <50ms
    });

    it('should maintain low latency under high frequency updates', async () => {
      // Configure for very high frequency
      const highFreqClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 5, // 200 Hz
        serverUrl: 'ws://localhost:3000'
      });

      const latencies: number[] = [];
      let totalUpdates = 0;

      highFreqClient.on('dataUpdate', (data) => {
        totalUpdates++;
        const latency = Date.now() - data.timestamp;
        latencies.push(latency);
      });

      performanceTracker.start('high_frequency_latency');
      highFreqClient.connect();

      // Run for 1 second
      await new Promise(resolve => setTimeout(resolve, 1000));

      const duration = performanceTracker.end('high_frequency_latency');
      highFreqClient.disconnect();

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const frequency = totalUpdates / (duration / 1000);

      expect(avgLatency).toBeLessThan(10); // Should maintain low latency
      expect(frequency).toBeGreaterThan(100); // Should achieve high frequency
      expect(totalUpdates).toBeGreaterThan(100);
    });

    it('should have minimal connection establishment latency', async () => {
      const connectionTimes: number[] = [];

      for (let i = 0; i < 10; i++) {
        const client = new MCPClient({
          enableRealtimeUpdates: false,
          serverUrl: 'ws://localhost:3000'
        });

        performanceTracker.start(`connection_${i}`);
        client.connect();
        
        // Connection is synchronous in mock, but measure the time anyway
        const connectionTime = performanceTracker.end(`connection_${i}`);
        connectionTimes.push(connectionTime);

        client.disconnect();
      }

      const avgConnectionTime = connectionTimes.reduce((a, b) => a + b, 0) / connectionTimes.length;
      const maxConnectionTime = Math.max(...connectionTimes);

      expect(avgConnectionTime).toBeLessThan(10); // Average connection <10ms
      expect(maxConnectionTime).toBeLessThan(50);  // Max connection <50ms
    });
  });

  describe('Data Collector Latency', () => {
    let mcpClient: MCPClient;
    let mockServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockServer = new MockBrainInspiredMCPServer();
      mockServer.configureProcessingDelay(1, 3);
      await mockServer.start();

      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 50,
        serverUrl: 'ws://localhost:3000'
      });
    });

    afterEach(async () => {
      if (mcpClient) {
        mcpClient.disconnect();
      }
      if (mockServer) {
        await mockServer.stop();
      }
    });

    it('should achieve collection processing latency <20ms', async () => {
      const collector = new CognitivePatternCollector(mcpClient, {
        name: 'latency-test-collector',
        collectionInterval: 50,
        autoFlush: false
      });

      const processingTimes: number[] = [];

      collector.on('data:collected', (event) => {
        const processingTime = event.data.metadata?.processingTime;
        if (processingTime !== undefined) {
          processingTimes.push(processingTime);
        }
      });

      await collector.start();
      mcpClient.connect();

      // Collect processing times
      await TestHelpers.waitFor(() => processingTimes.length >= 20, 3000);

      await collector.stop();

      const avgProcessingTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
      const maxProcessingTime = Math.max(...processingTimes);
      const p95ProcessingTime = processingTimes.sort((a, b) => a - b)[Math.floor(processingTimes.length * 0.95)];

      expect(avgProcessingTime).toBeLessThan(20);  // Average <20ms
      expect(p95ProcessingTime).toBeLessThan(20);  // 95th percentile <20ms
      expect(maxProcessingTime).toBeLessThan(50);  // Max <50ms
    });

    it('should maintain low latency during high-frequency collection', async () => {
      const highFreqCollector = new NeuralActivityCollector(mcpClient, {
        name: 'high-freq-latency-test',
        collectionInterval: 20, // 50 Hz
        bufferSize: 1000,
        autoFlush: true,
        flushInterval: 100
      });

      const collectionLatencies: number[] = [];
      let collectionCount = 0;

      highFreqCollector.on('data:collected', (event) => {
        collectionCount++;
        const collectionLatency = Date.now() - event.data.timestamp;
        collectionLatencies.push(collectionLatency);
      });

      await highFreqCollector.start();
      mcpClient.connect();

      performanceTracker.start('high_freq_collection');
      
      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      const duration = performanceTracker.end('high_freq_collection');
      await highFreqCollector.stop();

      const avgLatency = collectionLatencies.reduce((a, b) => a + b, 0) / collectionLatencies.length;
      const collectionRate = collectionCount / (duration / 1000);

      expect(avgLatency).toBeLessThan(30);     // Should maintain low latency
      expect(collectionRate).toBeGreaterThan(25); // Should achieve target rate
    });

    it('should have minimal buffer flush latency', async () => {
      const collector = new CognitivePatternCollector(mcpClient, {
        name: 'flush-latency-test',
        collectionInterval: 20,
        bufferSize: 50,  // Small buffer to trigger frequent flushes
        autoFlush: true,
        flushInterval: 100
      });

      const flushLatencies: number[] = [];

      collector.on('data:flushed', (event) => {
        // Measure flush processing time indirectly
        const flushTime = Date.now() - event.timestamp;
        flushLatencies.push(flushTime);
      });

      await collector.start();
      mcpClient.connect();

      // Wait for multiple flush operations
      await TestHelpers.waitFor(() => flushLatencies.length >= 5, 3000);

      await collector.stop();

      const avgFlushLatency = flushLatencies.reduce((a, b) => a + b, 0) / flushLatencies.length;
      const maxFlushLatency = Math.max(...flushLatencies);

      expect(avgFlushLatency).toBeLessThan(50);  // Average flush <50ms
      expect(maxFlushLatency).toBeLessThan(100); // Max flush <100ms
    });
  });

  describe('WebSocket Server Latency', () => {
    let server: WebSocketServer;

    beforeEach(() => {
      server = new WebSocketServer({
        port: 8082,
        enableBuffering: true,
        heartbeatInterval: 1000
      });
    });

    afterEach(async () => {
      if (server && server.isActive()) {
        await server.stop();
      }
    });

    it('should achieve message broadcast latency <5ms', async () => {
      // Mock WebSocket server functionality for testing
      jest.doMock('ws', () => ({
        Server: jest.fn().mockImplementation(() => ({
          on: jest.fn((event, handler) => {
            if (event === 'listening') {
              setTimeout(handler, 10);
            }
          }),
          close: jest.fn((callback) => callback && callback()),
          clients: new Set()
        }))
      }));

      await server.start();

      const broadcastLatencies: number[] = [];
      const testMessages = 100;

      // Simulate message broadcasting
      for (let i = 0; i < testMessages; i++) {
        const startTime = performance.now();
        
        server.broadcast('test_topic', {
          sequence: i,
          timestamp: Date.now(),
          data: LLMKGDataGenerator.generateCognitivePattern()
        });
        
        const latency = performance.now() - startTime;
        broadcastLatencies.push(latency);
      }

      const avgBroadcastLatency = broadcastLatencies.reduce((a, b) => a + b, 0) / broadcastLatencies.length;
      const maxBroadcastLatency = Math.max(...broadcastLatencies);
      const p95BroadcastLatency = broadcastLatencies.sort((a, b) => a - b)[Math.floor(broadcastLatencies.length * 0.95)];

      expect(avgBroadcastLatency).toBeLessThan(5);  // Average <5ms
      expect(p95BroadcastLatency).toBeLessThan(5);  // 95th percentile <5ms
      expect(maxBroadcastLatency).toBeLessThan(20); // Max <20ms
    });

    it('should maintain low latency with message buffering', async () => {
      const bufferingServer = new WebSocketServer({
        port: 8083,
        enableBuffering: true
      });

      // Mock the buffering system
      jest.doMock('ws', () => ({
        Server: jest.fn().mockImplementation(() => ({
          on: jest.fn((event, handler) => {
            if (event === 'listening') setTimeout(handler, 10);
          }),
          close: jest.fn((callback) => callback && callback()),
          clients: new Set()
        }))
      }));

      await bufferingServer.start();

      const bufferingLatencies: number[] = [];
      const messageCount = 200;

      // Send messages rapidly
      performanceTracker.start('buffering_test');

      for (let i = 0; i < messageCount; i++) {
        const messageStart = performance.now();
        
        bufferingServer.broadcast('buffered_topic', {
          id: i,
          timestamp: Date.now(),
          payload: LLMKGDataGenerator.generateSDRData()
        });
        
        const messageLatency = performance.now() - messageStart;
        bufferingLatencies.push(messageLatency);
      }

      const totalDuration = performanceTracker.end('buffering_test');
      
      await bufferingServer.stop();

      const avgBufferingLatency = bufferingLatencies.reduce((a, b) => a + b, 0) / bufferingLatencies.length;
      const throughput = messageCount / (totalDuration / 1000);

      expect(avgBufferingLatency).toBeLessThan(10);   // Should buffer quickly
      expect(throughput).toBeGreaterThan(1000);       // Should maintain high throughput
    });

    it('should have minimal server startup latency', async () => {
      const startupTimes: number[] = [];

      for (let i = 0; i < 5; i++) {
        const testServer = new WebSocketServer({
          port: 8084 + i,
          enableBuffering: false
        });

        // Mock for consistent testing
        jest.doMock('ws', () => ({
          Server: jest.fn().mockImplementation(() => ({
            on: jest.fn((event, handler) => {
              if (event === 'listening') setTimeout(handler, 5);
            }),
            close: jest.fn((callback) => callback && callback()),
            clients: new Set()
          }))
        }));

        performanceTracker.start(`startup_${i}`);
        await testServer.start();
        const startupTime = performanceTracker.end(`startup_${i}`);
        
        startupTimes.push(startupTime);
        await testServer.stop();
      }

      const avgStartupTime = startupTimes.reduce((a, b) => a + b, 0) / startupTimes.length;
      const maxStartupTime = Math.max(...startupTimes);

      expect(avgStartupTime).toBeLessThan(100); // Average startup <100ms
      expect(maxStartupTime).toBeLessThan(200); // Max startup <200ms
    });
  });

  describe('End-to-End Latency', () => {
    let mcpClient: MCPClient;
    let collector: CognitivePatternCollector;
    let server: WebSocketServer;
    let mockLLMKGServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockLLMKGServer = new MockBrainInspiredMCPServer();
      mockLLMKGServer.configureProcessingDelay(1, 5);
      await mockLLMKGServer.start();

      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 50,
        serverUrl: 'ws://localhost:3000'
      });

      collector = new CognitivePatternCollector(mcpClient, {
        name: 'e2e-latency-test',
        collectionInterval: 50,
        bufferSize: 100,
        autoFlush: true,
        flushInterval: 100
      });

      server = new WebSocketServer({
        port: 8085,
        enableBuffering: true
      });

      // Mock WebSocket
      jest.doMock('ws', () => ({
        Server: jest.fn().mockImplementation(() => ({
          on: jest.fn((event, handler) => {
            if (event === 'listening') setTimeout(handler, 10);
          }),
          close: jest.fn((callback) => callback && callback()),
          clients: new Set()
        }))
      }));
    });

    afterEach(async () => {
      if (collector) {
        await collector.stop();
      }
      if (server && server.isActive()) {
        await server.stop();
      }
      if (mcpClient) {
        mcpClient.disconnect();
      }
      if (mockLLMKGServer) {
        await mockLLMKGServer.stop();
      }
    });

    it('should achieve end-to-end latency <100ms (MCP → Collector → WebSocket)', async () => {
      const endToEndLatencies: number[] = [];

      // Setup data flow pipeline
      collector.on('data:collected', (event) => {
        const originalTimestamp = event.data.timestamp;
        const now = Date.now();
        
        // Forward to WebSocket
        server.broadcast('e2e_test', {
          ...event.data,
          e2e_timestamp: now
        });
        
        const endToEndLatency = now - originalTimestamp;
        endToEndLatencies.push(endToEndLatency);
      });

      await server.start();
      await collector.start();
      mcpClient.connect();

      // Collect end-to-end latency measurements
      await TestHelpers.waitFor(() => endToEndLatencies.length >= 10, 3000);

      const avgE2ELatency = endToEndLatencies.reduce((a, b) => a + b, 0) / endToEndLatencies.length;
      const maxE2ELatency = Math.max(...endToEndLatencies);
      const p95E2ELatency = endToEndLatencies.sort((a, b) => a - b)[Math.floor(endToEndLatencies.length * 0.95)];

      expect(avgE2ELatency).toBeLessThan(100); // Average E2E <100ms
      expect(p95E2ELatency).toBeLessThan(100); // 95th percentile <100ms  
      expect(maxE2ELatency).toBeLessThan(200);  // Max E2E <200ms
    });

    it('should maintain low latency under concurrent load', async () => {
      const concurrentLatencies: number[] = [];
      const concurrentCollectors: CognitivePatternCollector[] = [];

      // Create multiple concurrent collectors
      for (let i = 0; i < 3; i++) {
        const concurrentCollector = new CognitivePatternCollector(mcpClient, {
          name: `concurrent-${i}`,
          collectionInterval: 30,
          bufferSize: 50
        });

        concurrentCollector.on('data:collected', (event) => {
          const latency = Date.now() - event.data.timestamp;
          concurrentLatencies.push(latency);
          
          server.broadcast(`concurrent_${i}`, event.data);
        });

        concurrentCollectors.push(concurrentCollector);
      }

      await server.start();

      // Start all concurrent collectors
      for (const coll of concurrentCollectors) {
        await coll.start();
      }
      
      mcpClient.connect();

      // Run concurrent load
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Stop all collectors
      for (const coll of concurrentCollectors) {
        await coll.stop();
      }

      const avgConcurrentLatency = concurrentLatencies.reduce((a, b) => a + b, 0) / concurrentLatencies.length;
      const maxConcurrentLatency = Math.max(...concurrentLatencies);

      expect(avgConcurrentLatency).toBeLessThan(150); // Should maintain reasonable latency under load
      expect(maxConcurrentLatency).toBeLessThan(300);
      expect(concurrentLatencies.length).toBeGreaterThan(30); // Should process multiple messages
    });

    it('should achieve LLMKG real-time processing requirements', async () => {
      const llmkgLatencies: number[] = [];
      const dataTypes = ['cognitive_pattern', 'sdr_data', 'neural_activity'];

      collector.on('data:collected', (event) => {
        const processingLatency = Date.now() - event.data.timestamp;
        llmkgLatencies.push(processingLatency);

        // Simulate LLMKG-specific processing
        const llmkgProcessedData = {
          ...event.data,
          llmkg_processing_complete: Date.now(),
          processing_stage: 'visualization_ready'
        };

        server.broadcast('llmkg_processed', llmkgProcessedData);
      });

      await server.start();
      await collector.start();
      mcpClient.connect();

      // Generate LLMKG-specific data patterns
      const dataGenerator = setInterval(() => {
        const dataType = dataTypes[Math.floor(Math.random() * dataTypes.length)];
        const mockData = LLMKGDataGenerator.generateMixedLLMKGBatch(1)[0];
        mockData.type = dataType;
        
        mcpClient.emit('dataUpdate', mockData);
      }, 75);

      // Collect LLMKG processing latencies
      await new Promise(resolve => setTimeout(resolve, 1500));
      clearInterval(dataGenerator);

      const avgLLMKGLatency = llmkgLatencies.reduce((a, b) => a + b, 0) / llmkgLatencies.length;
      const p95LLMKGLatency = llmkgLatencies.sort((a, b) => a - b)[Math.floor(llmkgLatencies.length * 0.95)];

      // LLMKG should meet real-time cognitive processing requirements
      expect(avgLLMKGLatency).toBeLessThan(80);  // Average <80ms for cognitive data
      expect(p95LLMKGLatency).toBeLessThan(100); // 95th percentile <100ms
      expect(llmkgLatencies.length).toBeGreaterThan(10);
    });
  });

  describe('Latency Under Stress', () => {
    it('should maintain latency guarantees under memory pressure', async () => {
      const stressLatencies: number[] = [];
      
      // Create memory pressure
      const memoryPressure: any[] = [];
      for (let i = 0; i < 1000; i++) {
        memoryPressure.push(LLMKGDataGenerator.generateMixedLLMKGBatch(100));
      }

      const mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 25,
        serverUrl: 'ws://localhost:3000'
      });

      const stressCollector = new NeuralActivityCollector(mcpClient, {
        name: 'stress-test',
        collectionInterval: 25,
        bufferSize: 2000,
        maxMemoryUsage: 200 // Increased memory limit
      });

      stressCollector.on('data:collected', (event) => {
        const latency = Date.now() - event.data.timestamp;
        stressLatencies.push(latency);
      });

      await stressCollector.start();
      mcpClient.connect();

      // Run under memory pressure
      await new Promise(resolve => setTimeout(resolve, 1000));

      await stressCollector.stop();
      mcpClient.disconnect();

      // Clear memory pressure
      memoryPressure.length = 0;

      const avgStressLatency = stressLatencies.reduce((a, b) => a + b, 0) / stressLatencies.length;
      const maxStressLatency = Math.max(...stressLatencies);

      // Should maintain reasonable performance even under memory pressure
      expect(avgStressLatency).toBeLessThan(200); // Relaxed but still bounded
      expect(maxStressLatency).toBeLessThan(500);
    });

    it('should recover latency performance after stress periods', async () => {
      const beforeStressLatencies: number[] = [];
      const duringStressLatencies: number[] = [];
      const afterStressLatencies: number[] = [];

      const recoveryClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 50,
        serverUrl: 'ws://localhost:3000'
      });

      const recoveryCollector = new CognitivePatternCollector(recoveryClient, {
        name: 'recovery-test',
        collectionInterval: 50,
        bufferSize: 500
      });

      let phase = 'before';
      recoveryCollector.on('data:collected', (event) => {
        const latency = Date.now() - event.data.timestamp;
        
        switch (phase) {
          case 'before':
            beforeStressLatencies.push(latency);
            break;
          case 'during':
            duringStressLatencies.push(latency);
            break;
          case 'after':
            afterStressLatencies.push(latency);
            break;
        }
      });

      await recoveryCollector.start();
      recoveryClient.connect();

      // Phase 1: Normal operation
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Phase 2: Stress period (rapid data generation)
      phase = 'during';
      const stressGenerator = setInterval(() => {
        for (let i = 0; i < 10; i++) {
          recoveryClient.emit('dataUpdate', LLMKGDataGenerator.generateCognitivePattern());
        }
      }, 10);
      
      await new Promise(resolve => setTimeout(resolve, 500));
      clearInterval(stressGenerator);
      
      // Phase 3: Recovery period
      phase = 'after';
      await new Promise(resolve => setTimeout(resolve, 500));

      await recoveryCollector.stop();
      recoveryClient.disconnect();

      const avgBeforeLatency = beforeStressLatencies.reduce((a, b) => a + b, 0) / beforeStressLatencies.length;
      const avgDuringLatency = duringStressLatencies.reduce((a, b) => a + b, 0) / duringStressLatencies.length;
      const avgAfterLatency = afterStressLatencies.reduce((a, b) => a + b, 0) / afterStressLatencies.length;

      // Should recover to near-original performance
      expect(Math.abs(avgAfterLatency - avgBeforeLatency)).toBeLessThan(avgBeforeLatency * 0.5);
      expect(avgAfterLatency).toBeLessThan(avgDuringLatency);
    });
  });
});