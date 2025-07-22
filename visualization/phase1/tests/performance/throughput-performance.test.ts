/**
 * Throughput Performance Tests for LLMKG Visualization Phase 1
 * 
 * Tests to validate >1000 messages/second throughput requirements
 * and ensure system scalability under high load conditions.
 */

import { MCPClient } from '../../src/mcp/client';
import { CollectorManager } from '../../src/collectors/manager';
import { CognitivePatternCollector } from '../../src/collectors/cognitive-patterns';
import { NeuralActivityCollector } from '../../src/collectors/neural-activity';
import { WebSocketServer } from '../../src/websocket/server';
import { MockBrainInspiredMCPServer } from '../mocks/brain-inspired-server';
import { PerformanceTracker, LoadTestRunner, TestHelpers, LLMKGDataGenerator } from '../config/test-utils';

describe('Throughput Performance Tests', () => {
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

  describe('MCP Client Throughput', () => {
    let mcpClient: MCPClient;
    let mockServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockServer = new MockBrainInspiredMCPServer();
      // Configure for high throughput
      mockServer.configureProcessingDelay(1, 2);
      await mockServer.start();
    });

    afterEach(async () => {
      if (mcpClient) {
        mcpClient.disconnect();
      }
      if (mockServer) {
        await mockServer.stop();
      }
    });

    it('should achieve >1000 data updates/second', async () => {
      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 1, // 1000 Hz target
        serverUrl: 'ws://localhost:3000'
      });

      let messageCount = 0;
      mcpClient.on('dataUpdate', () => {
        messageCount++;
      });

      performanceTracker.start('high_throughput_client');
      mcpClient.connect();

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      const duration = performanceTracker.end('high_throughput_client');
      mcpClient.disconnect();

      const throughput = messageCount / (duration / 1000);

      expect(throughput).toBeGreaterThan(1000); // >1000 messages/second
      expect(messageCount).toBeGreaterThan(2000); // At least 2000 messages in 2 seconds
    });

    it('should sustain high throughput over extended periods', async () => {
      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 2, // 500 Hz
        serverUrl: 'ws://localhost:3000'
      });

      const throughputMeasurements: number[] = [];
      let totalMessages = 0;

      mcpClient.on('dataUpdate', () => {
        totalMessages++;
      });

      // Measure throughput every second for 10 seconds
      const measurementInterval = setInterval(() => {
        const currentThroughput = totalMessages;
        throughputMeasurements.push(currentThroughput);
        totalMessages = 0; // Reset counter
      }, 1000);

      mcpClient.connect();

      // Run for 10 seconds
      await new Promise(resolve => setTimeout(resolve, 10000));

      clearInterval(measurementInterval);
      mcpClient.disconnect();

      // Analyze sustained throughput
      const avgThroughput = throughputMeasurements.reduce((a, b) => a + b, 0) / throughputMeasurements.length;
      const minThroughput = Math.min(...throughputMeasurements);
      const maxThroughput = Math.max(...throughputMeasurements);
      const throughputVariance = maxThroughput - minThroughput;

      expect(avgThroughput).toBeGreaterThan(400);  // Sustained >400 messages/second
      expect(minThroughput).toBeGreaterThan(300);  // Minimum >300 messages/second
      expect(throughputVariance / avgThroughput).toBeLessThan(0.5); // <50% variance
    });

    it('should handle burst traffic effectively', async () => {
      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 10, // Normal rate
        serverUrl: 'ws://localhost:3000'
      });

      let burstMessageCount = 0;
      const burstMeasurements: number[] = [];

      mcpClient.on('dataUpdate', () => {
        burstMessageCount++;
      });

      mcpClient.connect();

      // Normal operation period
      await new Promise(resolve => setTimeout(resolve, 1000));
      const normalCount = burstMessageCount;
      burstMessageCount = 0;

      // Simulate burst by temporarily reducing interval
      (mcpClient as any).config.updateInterval = 1; // Burst mode
      
      performanceTracker.start('burst_handling');
      
      // Burst period
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const burstDuration = performanceTracker.end('burst_handling');
      const burstThroughput = burstMessageCount / (burstDuration / 1000);

      mcpClient.disconnect();

      expect(burstThroughput).toBeGreaterThan(500);           // Handle burst >500/sec
      expect(burstMessageCount).toBeGreaterThan(normalCount * 5); // Significant increase during burst
    });
  });

  describe('Data Collector Throughput', () => {
    let mcpClient: MCPClient;
    let mockServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockServer = new MockBrainInspiredMCPServer();
      mockServer.configureProcessingDelay(0.5, 1);
      await mockServer.start();

      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 5,
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

    it('should achieve >500 collections/second per collector', async () => {
      const collector = new CognitivePatternCollector(mcpClient, {
        name: 'throughput-test-collector',
        collectionInterval: 2, // 500 Hz target
        bufferSize: 10000,
        autoFlush: true,
        flushInterval: 100
      });

      let collectionCount = 0;
      collector.on('data:collected', () => {
        collectionCount++;
      });

      await collector.start();
      mcpClient.connect();

      performanceTracker.start('collector_throughput');

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      const duration = performanceTracker.end('collector_throughput');
      await collector.stop();

      const throughput = collectionCount / (duration / 1000);

      expect(throughput).toBeGreaterThan(500);    // >500 collections/second
      expect(collectionCount).toBeGreaterThan(1000); // At least 1000 collections in 2 seconds
    });

    it('should scale throughput with multiple concurrent collectors', async () => {
      const collectorCount = 5;
      const collectors: CognitivePatternCollector[] = [];
      const collectionCounts: number[] = new Array(collectorCount).fill(0);

      // Create multiple collectors
      for (let i = 0; i < collectorCount; i++) {
        const collector = new CognitivePatternCollector(mcpClient, {
          name: `concurrent-collector-${i}`,
          collectionInterval: 5, // 200 Hz each
          bufferSize: 5000,
          autoFlush: false
        });

        collector.on('data:collected', () => {
          collectionCounts[i]++;
        });

        collectors.push(collector);
      }

      // Start all collectors
      for (const collector of collectors) {
        await collector.start();
      }
      
      mcpClient.connect();

      performanceTracker.start('concurrent_throughput');

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      const duration = performanceTracker.end('concurrent_throughput');

      // Stop all collectors
      for (const collector of collectors) {
        await collector.stop();
      }

      const totalCollections = collectionCounts.reduce((a, b) => a + b, 0);
      const totalThroughput = totalCollections / (duration / 1000);
      const avgPerCollector = totalThroughput / collectorCount;

      expect(totalThroughput).toBeGreaterThan(1000);     // Combined >1000/second
      expect(avgPerCollector).toBeGreaterThan(100);      // Each collector contributing
      expect(totalCollections).toBeGreaterThan(2000);    // Total volume achieved
    });

    it('should maintain throughput during buffer flushes', async () => {
      const flushingCollector = new NeuralActivityCollector(mcpClient, {
        name: 'flushing-throughput-test',
        collectionInterval: 3, // ~333 Hz
        bufferSize: 100,  // Small buffer for frequent flushes
        autoFlush: true,
        flushInterval: 50 // Frequent flushes
      });

      let collectionCount = 0;
      let flushCount = 0;
      const throughputSamples: number[] = [];

      flushingCollector.on('data:collected', () => {
        collectionCount++;
      });

      flushingCollector.on('data:flushed', () => {
        flushCount++;
      });

      // Sample throughput every 500ms
      const samplingInterval = setInterval(() => {
        throughputSamples.push(collectionCount);
        collectionCount = 0; // Reset for next sample
      }, 500);

      await flushingCollector.start();
      mcpClient.connect();

      // Run for 3 seconds to see multiple flush cycles
      await new Promise(resolve => setTimeout(resolve, 3000));

      clearInterval(samplingInterval);
      await flushingCollector.stop();

      const avgThroughput = throughputSamples.reduce((a, b) => a + b, 0) / throughputSamples.length * 2; // *2 for per-second
      const throughputStdDev = Math.sqrt(
        throughputSamples.reduce((acc, val) => acc + Math.pow(val - avgThroughput/2, 2), 0) / throughputSamples.length
      );

      expect(avgThroughput).toBeGreaterThan(200);         // Maintain >200/sec during flushes
      expect(flushCount).toBeGreaterThan(10);             // Multiple flushes occurred
      expect(throughputStdDev / (avgThroughput/2)).toBeLessThan(0.5); // Stable throughput
    });
  });

  describe('WebSocket Server Throughput', () => {
    let server: WebSocketServer;

    beforeEach(() => {
      // Mock WebSocket for consistent testing
      jest.doMock('ws', () => ({
        Server: jest.fn().mockImplementation(() => ({
          on: jest.fn((event, handler) => {
            if (event === 'listening') setTimeout(handler, 10);
          }),
          close: jest.fn((callback) => callback && callback()),
          clients: new Set()
        }))
      }));

      server = new WebSocketServer({
        port: 8086,
        enableBuffering: true,
        enableCompression: false // Disable compression for pure throughput test
      });
    });

    afterEach(async () => {
      if (server && server.isActive()) {
        await server.stop();
      }
    });

    it('should achieve >2000 broadcasts/second', async () => {
      await server.start();

      let broadcastCount = 0;
      const testData = LLMKGDataGenerator.generateCognitivePattern();

      performanceTracker.start('broadcast_throughput');

      // Rapid broadcasting
      const broadcastInterval = setInterval(() => {
        for (let i = 0; i < 10; i++) {
          server.broadcast('throughput_test', {
            ...testData,
            sequence: broadcastCount++,
            timestamp: Date.now()
          });
        }
      }, 1);

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      clearInterval(broadcastInterval);
      const duration = performanceTracker.end('broadcast_throughput');

      const throughput = broadcastCount / (duration / 1000);

      expect(throughput).toBeGreaterThan(2000);      // >2000 broadcasts/second
      expect(broadcastCount).toBeGreaterThan(4000);  // At least 4000 broadcasts in 2 seconds
    });

    it('should handle concurrent client broadcasts efficiently', async () => {
      const clientCount = 10;
      const messagesPerClient = 100;

      await server.start();

      let totalMessages = 0;
      const clientMessages: number[] = new Array(clientCount).fill(0);

      performanceTracker.start('concurrent_client_throughput');

      // Simulate multiple clients sending messages concurrently
      const clientPromises = Array.from({ length: clientCount }, async (_, clientIndex) => {
        for (let msgIndex = 0; msgIndex < messagesPerClient; msgIndex++) {
          server.broadcast(`client_${clientIndex}`, {
            client_id: clientIndex,
            message_id: msgIndex,
            data: LLMKGDataGenerator.generateSDRData(),
            timestamp: Date.now()
          });
          
          clientMessages[clientIndex]++;
          totalMessages++;
          
          // Small delay to prevent overwhelming
          await new Promise(resolve => setImmediate(resolve));
        }
      });

      await Promise.all(clientPromises);
      const duration = performanceTracker.end('concurrent_client_throughput');

      const totalThroughput = totalMessages / (duration / 1000);
      const avgPerClient = totalThroughput / clientCount;

      expect(totalThroughput).toBeGreaterThan(1000);  // Combined >1000 messages/second
      expect(avgPerClient).toBeGreaterThan(50);       // Each client contributing
      expect(totalMessages).toBe(clientCount * messagesPerClient);
    });

    it('should maintain throughput with large message payloads', async () => {
      await server.start();

      const largePayload = {
        cognitive_patterns: LLMKGDataGenerator.generateMixedLLMKGBatch(50),
        neural_activity: Array.from({ length: 100 }, () => 
          LLMKGDataGenerator.generateNeuralActivity()
        ),
        knowledge_graph: LLMKGDataGenerator.generateKnowledgeGraphData(),
        metadata: {
          processing_chain: Array.from({ length: 20 }, (_, i) => ({
            stage: `stage_${i}`,
            timestamp: Date.now() - i * 1000,
            duration: Math.random() * 100
          }))
        }
      };

      let largeBroadcastCount = 0;

      performanceTracker.start('large_payload_throughput');

      // Broadcast large payloads rapidly
      const interval = setInterval(() => {
        server.broadcast('large_payload_test', {
          ...largePayload,
          sequence: largeBroadcastCount++,
          timestamp: Date.now()
        });
      }, 5);

      // Run for 2 seconds
      await new Promise(resolve => setTimeout(resolve, 2000));

      clearInterval(interval);
      const duration = performanceTracker.end('large_payload_throughput');

      const largeThroughput = largeBroadcastCount / (duration / 1000);

      expect(largeThroughput).toBeGreaterThan(100);    // >100 large messages/second
      expect(largeBroadcastCount).toBeGreaterThan(200); // At least 200 large messages
    });
  });

  describe('End-to-End System Throughput', () => {
    let mcpClient: MCPClient;
    let collectorManager: CollectorManager;
    let websocketServer: WebSocketServer;
    let mockLLMKGServer: MockBrainInspiredMCPServer;

    beforeEach(async () => {
      mockLLMKGServer = new MockBrainInspiredMCPServer();
      mockLLMKGServer.configureProcessingDelay(0.5, 1.5);
      await mockLLMKGServer.start();

      mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 2,
        serverUrl: 'ws://localhost:3000'
      });

      collectorManager = new CollectorManager(mcpClient);

      websocketServer = new WebSocketServer({
        port: 8087,
        enableBuffering: true,
        enableCompression: false
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
    });

    it('should achieve >1000 end-to-end messages/second', async () => {
      let endToEndCount = 0;
      
      // Setup collectors
      const cognitiveCollector = new CognitivePatternCollector(mcpClient, {
        name: 'e2e-throughput-cognitive',
        collectionInterval: 2,
        bufferSize: 5000,
        autoFlush: true,
        flushInterval: 50
      });

      const neuralCollector = new NeuralActivityCollector(mcpClient, {
        name: 'e2e-throughput-neural', 
        collectionInterval: 3,
        bufferSize: 5000,
        autoFlush: true,
        flushInterval: 50
      });

      // Setup data flow
      [cognitiveCollector, neuralCollector].forEach(collector => {
        collector.on('data:flushed', (event) => {
          endToEndCount += event.count;
          websocketServer.broadcast('e2e_throughput', {
            collector: event.collector,
            count: event.count,
            timestamp: Date.now()
          });
        });
      });

      await websocketServer.start();
      await collectorManager.addCollector('cognitive', cognitiveCollector);
      await collectorManager.addCollector('neural', neuralCollector);
      await collectorManager.startAll();
      
      performanceTracker.start('e2e_system_throughput');
      mcpClient.connect();

      // Run for 3 seconds
      await new Promise(resolve => setTimeout(resolve, 3000));

      const duration = performanceTracker.end('e2e_system_throughput');

      const e2eThroughput = endToEndCount / (duration / 1000);

      expect(e2eThroughput).toBeGreaterThan(1000);     // >1000 messages/second end-to-end
      expect(endToEndCount).toBeGreaterThan(3000);     // At least 3000 messages in 3 seconds
    });

    it('should scale throughput with system load', async () => {
      const scaleTestResults: Array<{
        collectorCount: number;
        throughput: number;
        efficiency: number;
      }> = [];

      // Test with different numbers of collectors
      for (const collectorCount of [1, 2, 4]) {
        let scaleThroughput = 0;
        const collectors: CognitivePatternCollector[] = [];

        // Create collectors
        for (let i = 0; i < collectorCount; i++) {
          const collector = new CognitivePatternCollector(mcpClient, {
            name: `scale-test-${i}`,
            collectionInterval: 5,
            bufferSize: 2000,
            autoFlush: true,
            flushInterval: 100
          });

          collector.on('data:collected', () => {
            scaleThroughput++;
            websocketServer.broadcast('scale_test', {
              collector: collector.getStats().name,
              timestamp: Date.now()
            });
          });

          collectors.push(collector);
        }

        await websocketServer.start();

        // Start all collectors
        for (const collector of collectors) {
          await collectorManager.addCollector(`scale-${collectors.indexOf(collector)}`, collector);
        }
        await collectorManager.startAll();

        performanceTracker.start(`scale_test_${collectorCount}`);
        mcpClient.connect();

        // Run for 2 seconds
        await new Promise(resolve => setTimeout(resolve, 2000));

        const duration = performanceTracker.end(`scale_test_${collectorCount}`);
        
        await collectorManager.stopAll();
        mcpClient.disconnect();

        const actualThroughput = scaleThroughput / (duration / 1000);
        const efficiency = actualThroughput / (collectorCount * 200); // Expected ~200/sec per collector

        scaleTestResults.push({
          collectorCount,
          throughput: actualThroughput,
          efficiency
        });

        // Reset for next iteration
        scaleThroughput = 0;
      }

      // Analyze scaling characteristics
      expect(scaleTestResults[0].throughput).toBeGreaterThan(100);   // 1 collector baseline
      expect(scaleTestResults[1].throughput).toBeGreaterThan(scaleTestResults[0].throughput * 1.5); // 2x scaling
      expect(scaleTestResults[2].throughput).toBeGreaterThan(scaleTestResults[0].throughput * 2.5); // 4x scaling

      // Efficiency should remain reasonable
      scaleTestResults.forEach(result => {
        expect(result.efficiency).toBeGreaterThan(0.3); // At least 30% efficiency
      });
    });

    it('should maintain throughput during peak LLMKG operations', async () => {
      let llmkgThroughput = 0;
      const llmkgDataTypes = ['cognitive_pattern', 'sdr_data', 'neural_activity', 'memory_data'];

      // Setup specialized LLMKG collectors
      const llmkgCollector = new CognitivePatternCollector(mcpClient, {
        name: 'llmkg-peak-test',
        collectionInterval: 1, // Very high frequency
        bufferSize: 10000,
        autoFlush: true,
        flushInterval: 25 // Frequent flushes
      });

      llmkgCollector.on('data:flushed', (event) => {
        llmkgThroughput += event.count;
        
        // Simulate LLMKG-specific processing
        websocketServer.broadcast('llmkg_peak', {
          batch_size: event.count,
          data_types: llmkgDataTypes,
          processing_stage: 'real_time_analysis',
          cognitive_load: Math.random(),
          timestamp: Date.now()
        });
      });

      await websocketServer.start();
      await llmkgCollector.start();

      performanceTracker.start('llmkg_peak_throughput');
      mcpClient.connect();

      // Generate intensive LLMKG workload
      const workloadGenerator = setInterval(() => {
        // Rapid data generation simulating peak cognitive activity
        for (let i = 0; i < 5; i++) {
          const dataType = llmkgDataTypes[Math.floor(Math.random() * llmkgDataTypes.length)];
          const data = LLMKGDataGenerator.generateMixedLLMKGBatch(1)[0];
          data.type = dataType;
          mcpClient.emit('dataUpdate', data);
        }
      }, 2);

      // Run peak load for 3 seconds
      await new Promise(resolve => setTimeout(resolve, 3000));

      clearInterval(workloadGenerator);
      const duration = performanceTracker.end('llmkg_peak_throughput');
      await llmkgCollector.stop();

      const peakThroughput = llmkgThroughput / (duration / 1000);

      expect(peakThroughput).toBeGreaterThan(800);     // Handle >800 LLMKG operations/second
      expect(llmkgThroughput).toBeGreaterThan(2400);   // At least 2400 operations in 3 seconds
    });
  });

  describe('Throughput Stress Testing', () => {
    it('should maintain performance under extended high-load periods', async () => {
      const extendedThroughputResults: number[] = [];

      const mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 2,
        serverUrl: 'ws://localhost:3000'
      });

      const stressCollector = new NeuralActivityCollector(mcpClient, {
        name: 'extended-stress-test',
        collectionInterval: 3,
        bufferSize: 15000, // Large buffer for sustained load
        maxMemoryUsage: 500, // Increased memory allowance
        autoFlush: true,
        flushInterval: 200
      });

      let messageCount = 0;
      stressCollector.on('data:collected', () => {
        messageCount++;
      });

      // Measure throughput every 2 seconds during extended test
      const measurementInterval = setInterval(() => {
        extendedThroughputResults.push(messageCount / 2); // Per second rate
        messageCount = 0;
      }, 2000);

      await stressCollector.start();
      mcpClient.connect();

      // Extended test: 20 seconds of continuous high load
      await new Promise(resolve => setTimeout(resolve, 20000));

      clearInterval(measurementInterval);
      await stressCollector.stop();
      mcpClient.disconnect();

      // Analyze sustained performance
      const avgSustainedThroughput = extendedThroughputResults.reduce((a, b) => a + b, 0) / extendedThroughputResults.length;
      const minSustainedThroughput = Math.min(...extendedThroughputResults);
      const throughputDegradation = (extendedThroughputResults[0] - extendedThroughputResults[extendedThroughputResults.length - 1]) / extendedThroughputResults[0];

      expect(avgSustainedThroughput).toBeGreaterThan(200);  // Sustained >200/sec
      expect(minSustainedThroughput).toBeGreaterThan(150);  // Never drop below 150/sec  
      expect(throughputDegradation).toBeLessThan(0.3);      // <30% degradation over time
      expect(extendedThroughputResults.length).toBeGreaterThan(8); // Multiple measurements
    });

    it('should recover throughput after memory pressure periods', async () => {
      const recoveryPhases: Array<{
        phase: string;
        throughput: number;
      }> = [];

      const mcpClient = new MCPClient({
        enableRealtimeUpdates: true,
        updateInterval: 3,
        serverUrl: 'ws://localhost:3000'
      });

      const recoveryCollector = new CognitivePatternCollector(mcpClient, {
        name: 'throughput-recovery-test',
        collectionInterval: 4,
        bufferSize: 5000,
        maxMemoryUsage: 100 // Limited memory to trigger pressure
      });

      let phaseMessageCount = 0;
      recoveryCollector.on('data:collected', () => {
        phaseMessageCount++;
      });

      await recoveryCollector.start();
      mcpClient.connect();

      // Phase 1: Normal operation
      await new Promise(resolve => setTimeout(resolve, 2000));
      recoveryPhases.push({
        phase: 'normal',
        throughput: phaseMessageCount / 2
      });
      phaseMessageCount = 0;

      // Phase 2: Memory pressure (generate large objects)
      const memoryPressure: any[] = [];
      for (let i = 0; i < 500; i++) {
        memoryPressure.push(LLMKGDataGenerator.generateMixedLLMKGBatch(50));
      }

      await new Promise(resolve => setTimeout(resolve, 2000));
      recoveryPhases.push({
        phase: 'pressure',
        throughput: phaseMessageCount / 2
      });
      phaseMessageCount = 0;

      // Phase 3: Recovery (clear memory pressure)
      memoryPressure.length = 0;
      global.gc && global.gc(); // Force garbage collection if available

      await new Promise(resolve => setTimeout(resolve, 2000));
      recoveryPhases.push({
        phase: 'recovery',
        throughput: phaseMessageCount / 2
      });

      await recoveryCollector.stop();
      mcpClient.disconnect();

      const normalThroughput = recoveryPhases.find(p => p.phase === 'normal')?.throughput || 0;
      const pressureThroughput = recoveryPhases.find(p => p.phase === 'pressure')?.throughput || 0;
      const recoveryThroughput = recoveryPhases.find(p => p.phase === 'recovery')?.throughput || 0;

      // Should show impact during pressure but recovery afterward
      expect(normalThroughput).toBeGreaterThan(50);
      expect(recoveryThroughput).toBeGreaterThan(pressureThroughput);
      expect(recoveryThroughput / normalThroughput).toBeGreaterThan(0.7); // At least 70% recovery
    });
  });
});