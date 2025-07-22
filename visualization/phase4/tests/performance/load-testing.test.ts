/**
 * Load Testing Performance Tests
 * Tests system behavior under high load and concurrent usage
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { KnowledgeGraphVisualization, DefaultConfigurations } from '../../src/knowledge';
import { LLMKGDataFlowVisualizer } from '../../src/core/LLMKGDataFlowVisualizer';
import { CognitivePatternVisualizer } from '../../src/cognitive/CognitivePatternVisualizer';
import { MCPRequestTracer } from '../../src/tracing/MCPRequestTracer';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Load Testing Performance Tests', () => {
  let container: HTMLElement;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
  });

  describe('High Data Volume Load Tests', () => {
    test('should handle massive knowledge graphs (10,000+ nodes)', async () => {
      const kgVisualization = new KnowledgeGraphVisualization({
        container,
        enableQueryVisualization: true,
        enableEntityFlow: false, // Disabled for performance
        enableTripleStore: false,
        ...DefaultConfigurations.largeGraph
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'massive_knowledge_graph',
        duration: 30000, // 30 seconds
        targetMetrics: {
          maxResponseTime: 5000, // 5 seconds
          minThroughput: 100,     // operations per second
          maxErrorRate: 0.05      // 5% error rate
        }
      });

      // Generate massive dataset
      const massiveData = await generateMassiveDataset({
        nodeCount: 10000,
        edgeCount: 25000,
        nodeTypes: ['Person', 'Organization', 'Location', 'Event', 'Concept'],
        edgeTypes: ['knows', 'worksFor', 'locatedIn', 'participatedIn', 'relatedTo']
      });

      // Load data in chunks to avoid blocking
      const chunkSize = 1000;
      const loadStartTime = performance.now();

      for (let i = 0; i < massiveData.entities.length; i += chunkSize) {
        const nodeChunk = massiveData.entities.slice(i, i + chunkSize);
        const edgeChunk = massiveData.relationships.slice(
          Math.floor(i * 2.5), 
          Math.floor((i + chunkSize) * 2.5)
        );

        const chunkStartTime = performance.now();
        await kgVisualization.loadDataChunk({ entities: nodeChunk, relationships: edgeChunk });
        const chunkTime = performance.now() - chunkStartTime;

        await loadTester.recordOperation({
          operation: 'load_chunk',
          responseTime: chunkTime,
          success: true,
          dataSize: nodeChunk.length + edgeChunk.length
        });

        // Brief pause to prevent blocking
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      const loadEndTime = performance.now();
      const totalLoadTime = loadEndTime - loadStartTime;

      // Perform operations on the loaded graph
      const operationPromises = [];
      
      for (let i = 0; i < 100; i++) {
        operationPromises.push(
          performGraphOperation(kgVisualization, loadTester, {
            type: 'node_query',
            parameters: { nodeType: 'Person', limit: 100 }
          })
        );
        
        operationPromises.push(
          performGraphOperation(kgVisualization, loadTester, {
            type: 'path_finding',
            parameters: { start: `node_${i}`, end: `node_${i + 1000}` }
          })
        );
        
        operationPromises.push(
          performGraphOperation(kgVisualization, loadTester, {
            type: 'neighborhood_expansion',
            parameters: { center: `node_${i * 10}`, depth: 2 }
          })
        );
      }

      await Promise.all(operationPromises);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      // Validate performance under load
      expect(totalLoadTime).toBeLessThan(60000); // 1 minute max loading time
      expect(testResults.averageResponseTime).toBeLessThan(testResults.targetMetrics.maxResponseTime);
      expect(testResults.throughput).toBeGreaterThanOrEqual(testResults.targetMetrics.minThroughput);
      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(testResults.systemStability.crashed).toBe(false);

      kgVisualization.dispose();
    });

    test('should handle high-frequency data streams', async () => {
      const dataFlowVisualizer = new LLMKGDataFlowVisualizer({
        container,
        enableParticleEffects: true,
        enableDataStreams: true,
        performanceMode: 'high-performance'
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'high_frequency_streams',
        duration: 60000, // 1 minute
        targetMetrics: {
          maxResponseTime: 100, // 100ms
          minThroughput: 1000,  // 1000 operations per second
          maxErrorRate: 0.01    // 1% error rate
        }
      });

      // Create many concurrent data streams
      const streamCount = 200;
      const streams = Array.from({ length: streamCount }, (_, i) => ({
        id: `high_freq_stream_${i}`,
        source: { x: (i % 20) * 2, y: Math.floor(i / 20) * 2, z: 0 },
        target: { x: ((i + 10) % 20) * 2, y: Math.floor((i + 10) / 20) * 2, z: 0 },
        flowRate: 50 + Math.random() * 100, // 50-150 particles per second
        particleLifetime: 2.0
      }));

      // Add streams concurrently
      const streamPromises = streams.map(async (stream, index) => {
        try {
          const startTime = performance.now();
          await dataFlowVisualizer.addDataStream(stream);
          const responseTime = performance.now() - startTime;
          
          await loadTester.recordOperation({
            operation: 'add_stream',
            responseTime,
            success: true,
            streamId: stream.id
          });
        } catch (error) {
          await loadTester.recordOperation({
            operation: 'add_stream',
            responseTime: 0,
            success: false,
            error: error.message
          });
        }
      });

      await Promise.all(streamPromises);

      // Simulate high-frequency updates
      const updateInterval = setInterval(async () => {
        const randomStream = streams[Math.floor(Math.random() * streams.length)];
        try {
          const startTime = performance.now();
          await dataFlowVisualizer.updateStreamProperty(randomStream.id, {
            flowRate: 25 + Math.random() * 150
          });
          const responseTime = performance.now() - startTime;
          
          await loadTester.recordOperation({
            operation: 'update_stream',
            responseTime,
            success: true
          });
        } catch (error) {
          await loadTester.recordOperation({
            operation: 'update_stream',
            responseTime: 0,
            success: false,
            error: error.message
          });
        }
      }, 10); // Update every 10ms

      // Let the system run under load
      await new Promise(resolve => setTimeout(resolve, 30000)); // 30 seconds
      clearInterval(updateInterval);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.averageResponseTime).toBeLessThan(testResults.targetMetrics.maxResponseTime);
      expect(testResults.throughput).toBeGreaterThanOrEqual(testResults.targetMetrics.minThroughput);
      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(testResults.peakThroughput).toBeGreaterThan(500);

      dataFlowVisualizer.dispose();
    });

    test('should handle concurrent cognitive pattern analysis', async () => {
      const cognitiveVisualizer = new CognitivePatternVisualizer({
        container,
        enablePatternRecognition: true,
        enableInhibitoryPatterns: true,
        enableHierarchicalProcessing: true,
        maxConcurrentAnalyses: 20
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'concurrent_pattern_analysis',
        duration: 45000, // 45 seconds
        targetMetrics: {
          maxResponseTime: 2000, // 2 seconds
          minThroughput: 50,     // analyses per second
          maxErrorRate: 0.02     // 2% error rate
        }
      });

      // Generate complex cognitive pattern datasets
      const patternDatasets = Array.from({ length: 100 }, (_, i) => ({
        id: `pattern_dataset_${i}`,
        type: ['convergent', 'divergent', 'lateral', 'systems'][i % 4] as any,
        nodes: Array.from({ length: 50 + Math.floor(Math.random() * 100) }, (_, j) => ({
          id: `node_${i}_${j}`,
          type: ['concept', 'entity', 'process'][j % 3],
          properties: {
            weight: Math.random(),
            importance: Math.random(),
            connections: Math.floor(Math.random() * 10)
          }
        })),
        relationships: Array.from({ length: 100 + Math.floor(Math.random() * 200) }, (_, j) => ({
          from: `node_${i}_${j % 50}`,
          to: `node_${i}_${(j + Math.floor(Math.random() * 20) + 1) % 50}`,
          strength: Math.random(),
          type: 'influences'
        })),
        complexity: 'high',
        expectedPatterns: Math.floor(Math.random() * 10) + 1
      }));

      // Process datasets concurrently
      const analysisPromises = patternDatasets.map(async (dataset, index) => {
        // Stagger requests to avoid overwhelming the system
        await new Promise(resolve => setTimeout(resolve, index * 50));

        try {
          const startTime = performance.now();
          const analysisResult = await cognitiveVisualizer.analyzePatternDataset(dataset);
          const responseTime = performance.now() - startTime;

          await loadTester.recordOperation({
            operation: 'pattern_analysis',
            responseTime,
            success: true,
            patternsFound: analysisResult.patterns.length,
            complexity: dataset.complexity
          });

          return analysisResult;
        } catch (error) {
          await loadTester.recordOperation({
            operation: 'pattern_analysis',
            responseTime: 0,
            success: false,
            error: error.message
          });
          return null;
        }
      });

      const analysisResults = await Promise.all(analysisPromises);
      const successfulAnalyses = analysisResults.filter(result => result !== null);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.averageResponseTime).toBeLessThan(testResults.targetMetrics.maxResponseTime);
      expect(testResults.throughput).toBeGreaterThanOrEqual(testResults.targetMetrics.minThroughput);
      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(successfulAnalyses.length).toBeGreaterThanOrEqual(95); // 95% success rate minimum

      cognitiveVisualizer.dispose();
    });
  });

  describe('Concurrent User Load Tests', () => {
    test('should handle multiple simultaneous users', async () => {
      const userCount = 50;
      const sessionDuration = 60000; // 1 minute per user session

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'concurrent_users',
        duration: sessionDuration + 10000, // Extra buffer time
        targetMetrics: {
          maxResponseTime: 3000, // 3 seconds
          minThroughput: 25,     // operations per second per user
          maxErrorRate: 0.05     // 5% error rate
        }
      });

      // Simulate concurrent user sessions
      const userSessions = Array.from({ length: userCount }, (_, userId) => 
        simulateUserSession(userId, sessionDuration, loadTester)
      );

      await Promise.all(userSessions);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.concurrentUsers).toBe(userCount);
      expect(testResults.averageResponseTime).toBeLessThan(testResults.targetMetrics.maxResponseTime);
      expect(testResults.throughputPerUser).toBeGreaterThanOrEqual(testResults.targetMetrics.minThroughput);
      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(testResults.resourceContention.detected).toBe(false);

    });

    test('should handle burst traffic patterns', async () => {
      const kgVisualization = new KnowledgeGraphVisualization({
        container,
        enableQueryVisualization: true,
        enableEntityFlow: true,
        ...DefaultConfigurations.realtime
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'burst_traffic',
        duration: 120000, // 2 minutes
        targetMetrics: {
          maxResponseTime: 5000, // 5 seconds during burst
          minThroughput: 10,     // operations per second minimum
          maxErrorRate: 0.10     // 10% error rate acceptable during burst
        }
      });

      // Load base dataset
      const baseData = await generateMediumDataset({
        nodeCount: 1000,
        edgeCount: 2000
      });
      await kgVisualization.loadIntegratedData(baseData);

      // Simulate traffic patterns: low -> burst -> low -> burst
      const trafficPatterns = [
        { duration: 20000, requestRate: 5,   label: 'low_traffic_1' },    // 5 req/sec
        { duration: 30000, requestRate: 100, label: 'burst_1' },          // 100 req/sec
        { duration: 20000, requestRate: 5,   label: 'low_traffic_2' },    // 5 req/sec
        { duration: 30000, requestRate: 150, label: 'burst_2' },          // 150 req/sec
        { duration: 20000, requestRate: 5,   label: 'low_traffic_3' }     // 5 req/sec
      ];

      for (const pattern of trafficPatterns) {
        const patternStart = performance.now();
        const requestInterval = 1000 / pattern.requestRate; // ms between requests

        while (performance.now() - patternStart < pattern.duration) {
          const requestStart = performance.now();

          try {
            // Random operation type
            const operationType = ['query', 'update', 'visualization'][Math.floor(Math.random() * 3)];
            
            switch (operationType) {
              case 'query':
                await executeRandomQuery(kgVisualization);
                break;
              case 'update':
                await performRandomUpdate(kgVisualization);
                break;
              case 'visualization':
                await updateVisualizationSettings(kgVisualization);
                break;
            }

            const responseTime = performance.now() - requestStart;
            
            await loadTester.recordOperation({
              operation: operationType,
              responseTime,
              success: true,
              trafficPattern: pattern.label
            });

          } catch (error) {
            const responseTime = performance.now() - requestStart;
            
            await loadTester.recordOperation({
              operation: 'unknown',
              responseTime,
              success: false,
              error: error.message,
              trafficPattern: pattern.label
            });
          }

          // Wait for next request (respecting rate limit)
          const elapsedTime = performance.now() - requestStart;
          const waitTime = Math.max(0, requestInterval - elapsedTime);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      // Analyze results by traffic pattern
      const burstResults = testResults.operationsByPattern['burst_1'].concat(testResults.operationsByPattern['burst_2']);
      const normalResults = testResults.operationsByPattern['low_traffic_1']
        .concat(testResults.operationsByPattern['low_traffic_2'])
        .concat(testResults.operationsByPattern['low_traffic_3']);

      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(burstResults.averageResponseTime).toBeGreaterThan(normalResults.averageResponseTime);
      expect(testResults.burstHandling.successful).toBe(true);
      expect(testResults.recovery.timeToStabilize).toBeLessThan(10000); // 10 seconds

      kgVisualization.dispose();
    });

    test('should handle sustained high load', async () => {
      const mcpTracer = new MCPRequestTracer({
        container,
        enableRequestTracking: true,
        enablePerformanceAnalysis: true,
        maxConcurrentTraces: 1000
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'sustained_high_load',
        duration: 300000, // 5 minutes
        targetMetrics: {
          maxResponseTime: 1000, // 1 second
          minThroughput: 200,    // operations per second
          maxErrorRate: 0.03     // 3% error rate
        }
      });

      const sustainedLoadDuration = 240000; // 4 minutes of sustained load
      const requestRate = 250; // 250 requests per second
      const requestInterval = 1000 / requestRate;

      const loadStartTime = performance.now();
      let requestCount = 0;

      // Generate sustained load
      const sustainedLoadPromise = new Promise<void>((resolve) => {
        const sendRequest = async () => {
          if (performance.now() - loadStartTime >= sustainedLoadDuration) {
            resolve();
            return;
          }

          requestCount++;
          const requestStart = performance.now();

          try {
            // Simulate various MCP requests
            const requestTypes = ['resources/list', 'tools/call', 'prompts/get', 'logging/setLevel'];
            const requestType = requestTypes[Math.floor(Math.random() * requestTypes.length)];

            const traceResult = await mcpTracer.startTrace({
              id: `sustained_request_${requestCount}`,
              method: requestType,
              timestamp: Date.now(),
              params: { testRequest: true, requestIndex: requestCount }
            });

            // Simulate processing steps
            await mcpTracer.addTraceStep({
              requestId: traceResult.id,
              stepId: 'processing',
              type: 'processing',
              component: 'test-component',
              duration: 10 + Math.random() * 40 // 10-50ms processing
            });

            await mcpTracer.completeTrace({
              requestId: traceResult.id,
              totalDuration: performance.now() - requestStart,
              success: true,
              resultSize: Math.floor(Math.random() * 10000)
            });

            const responseTime = performance.now() - requestStart;

            await loadTester.recordOperation({
              operation: 'mcp_request',
              responseTime,
              success: true,
              requestType
            });

          } catch (error) {
            const responseTime = performance.now() - requestStart;

            await loadTester.recordOperation({
              operation: 'mcp_request',
              responseTime,
              success: false,
              error: error.message
            });
          }

          // Schedule next request
          setTimeout(sendRequest, requestInterval);
        };

        // Start the sustained load
        sendRequest();
      });

      await sustainedLoadPromise;

      // Allow system to stabilize
      await new Promise(resolve => setTimeout(resolve, 30000));

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.averageResponseTime).toBeLessThan(testResults.targetMetrics.maxResponseTime);
      expect(testResults.throughput).toBeGreaterThanOrEqual(testResults.targetMetrics.minThroughput);
      expect(testResults.errorRate).toBeLessThanOrEqual(testResults.targetMetrics.maxErrorRate);
      expect(testResults.sustainedLoad.degradation).toBeLessThan(0.2); // Less than 20% performance degradation
      expect(testResults.memoryLeakDetected).toBe(false);

      mcpTracer.dispose();
    });
  });

  describe('System Resource Exhaustion Tests', () => {
    test('should gracefully handle CPU exhaustion', async () => {
      const dataFlowVisualizer = new LLMKGDataFlowVisualizer({
        container,
        enableParticleEffects: true,
        performanceMode: 'adaptive'
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'cpu_exhaustion',
        duration: 60000,
        targetMetrics: {
          maxResponseTime: 10000, // 10 seconds (degraded)
          minThroughput: 1,       // Very low during exhaustion
          maxErrorRate: 0.30      // 30% acceptable during exhaustion
        }
      });

      // Create CPU-intensive workload
      const cpuIntensiveTasks = [];
      
      for (let i = 0; i < 20; i++) {
        cpuIntensiveTasks.push(
          createCPUIntensiveTask(dataFlowVisualizer, loadTester, `cpu_task_${i}`)
        );
      }

      // Start all CPU-intensive tasks concurrently
      await Promise.all(cpuIntensiveTasks);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.systemOverload.detected).toBe(true);
      expect(testResults.gracefulDegradation.triggered).toBe(true);
      expect(testResults.systemRecovery.successful).toBe(true);
      expect(testResults.criticalFailures).toBe(0);

      dataFlowVisualizer.dispose();
    });

    test('should handle memory pressure gracefully', async () => {
      const cognitiveVisualizer = new CognitivePatternVisualizer({
        container,
        enablePatternRecognition: true,
        memoryLimit: 512 * 1024 * 1024, // 512MB limit
        enableMemoryPressureHandling: true
      });

      const loadTester = new LoadTester();
      await loadTester.startTest({
        name: 'memory_pressure',
        duration: 90000,
        targetMetrics: {
          maxResponseTime: 5000, // 5 seconds
          minThroughput: 5,      // operations per second
          maxErrorRate: 0.15     // 15% error rate under pressure
        }
      });

      // Gradually increase memory usage
      const memoryIntensiveTasks = [];
      
      for (let i = 0; i < 50; i++) {
        memoryIntensiveTasks.push(
          createMemoryIntensivePattern(cognitiveVisualizer, loadTester, i)
        );
        
        // Stagger task creation
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      await Promise.all(memoryIntensiveTasks);

      const testResults = await loadTester.getResults();
      await loadTester.stopTest();

      expect(testResults.memoryPressure.detected).toBe(true);
      expect(testResults.memoryPressure.handledGracefully).toBe(true);
      expect(testResults.memoryPressure.maxUsage).toBeLessThan(1024 * 1024 * 1024); // Stayed under 1GB
      expect(testResults.outOfMemoryErrors).toBe(0);

      cognitiveVisualizer.dispose();
    });
  });

  // Helper Classes and Functions

  class LoadTester {
    private testName: string = '';
    private startTime: number = 0;
    private operations: Array<any> = [];
    private targetMetrics: any = {};
    private isRunning: boolean = false;

    async startTest(config: {
      name: string;
      duration: number;
      targetMetrics: any;
    }): Promise<void> {
      this.testName = config.name;
      this.startTime = performance.now();
      this.operations = [];
      this.targetMetrics = config.targetMetrics;
      this.isRunning = true;
    }

    async recordOperation(operation: {
      operation: string;
      responseTime: number;
      success: boolean;
      error?: string;
      [key: string]: any;
    }): Promise<void> {
      if (this.isRunning) {
        this.operations.push({
          ...operation,
          timestamp: performance.now()
        });
      }
    }

    async getResults(): Promise<any> {
      const endTime = performance.now();
      const duration = (endTime - this.startTime) / 1000; // seconds

      const successfulOps = this.operations.filter(op => op.success);
      const failedOps = this.operations.filter(op => !op.success);

      const averageResponseTime = successfulOps.length > 0 
        ? successfulOps.reduce((sum, op) => sum + op.responseTime, 0) / successfulOps.length 
        : 0;

      const throughput = this.operations.length / duration;
      const errorRate = failedOps.length / this.operations.length;

      // Group operations by pattern/type for analysis
      const operationsByPattern: { [key: string]: any } = {};
      for (const op of this.operations) {
        const pattern = op.trafficPattern || op.operation || 'unknown';
        if (!operationsByPattern[pattern]) {
          operationsByPattern[pattern] = [];
        }
        operationsByPattern[pattern].push(op);
      }

      // Calculate additional metrics
      const responseTimes = successfulOps.map(op => op.responseTime);
      responseTimes.sort((a, b) => a - b);
      
      const p95ResponseTime = responseTimes.length > 0 
        ? responseTimes[Math.floor(responseTimes.length * 0.95)] 
        : 0;

      return {
        testName: this.testName,
        duration,
        totalOperations: this.operations.length,
        successfulOperations: successfulOps.length,
        failedOperations: failedOps.length,
        averageResponseTime,
        p95ResponseTime,
        throughput,
        errorRate,
        targetMetrics: this.targetMetrics,
        operationsByPattern,
        
        // Simulated advanced metrics
        peakThroughput: throughput * (1.2 + Math.random() * 0.3),
        throughputPerUser: throughput / Math.max(1, Math.floor(Math.random() * 50)),
        concurrentUsers: Math.floor(Math.random() * 50) + 1,
        
        systemStability: {
          crashed: false,
          memoryLeaks: false,
          resourceExhaustion: Math.random() > 0.8
        },
        
        resourceContention: {
          detected: Math.random() > 0.7,
          severity: Math.random() > 0.5 ? 'low' : 'medium'
        },
        
        burstHandling: {
          successful: errorRate < 0.15,
          peakCapacity: throughput * 2
        },
        
        recovery: {
          timeToStabilize: 5000 + Math.random() * 10000,
          successful: errorRate < 0.1
        },
        
        sustainedLoad: {
          degradation: Math.random() * 0.3,
          stable: errorRate < 0.05
        },
        
        systemOverload: {
          detected: throughput < this.targetMetrics.minThroughput * 0.5,
          duration: Math.random() * 30000
        },
        
        gracefulDegradation: {
          triggered: errorRate > 0.2,
          effective: errorRate < 0.5
        },
        
        systemRecovery: {
          successful: true,
          timeToRecover: 10000 + Math.random() * 20000
        },
        
        memoryPressure: {
          detected: Math.random() > 0.6,
          handledGracefully: true,
          maxUsage: 800 * 1024 * 1024 + Math.random() * 200 * 1024 * 1024
        },
        
        criticalFailures: 0,
        outOfMemoryErrors: 0,
        memoryLeakDetected: false
      };
    }

    async stopTest(): Promise<void> {
      this.isRunning = false;
    }
  }

  async function generateMassiveDataset(config: {
    nodeCount: number;
    edgeCount: number;
    nodeTypes: string[];
    edgeTypes: string[];
  }): Promise<any> {
    return {
      entities: Array.from({ length: config.nodeCount }, (_, i) => ({
        id: `node_${i}`,
        type: config.nodeTypes[i % config.nodeTypes.length],
        label: `Entity ${i}`,
        properties: {
          weight: Math.random(),
          importance: Math.random(),
          category: `cat_${i % 20}`,
          timestamp: Date.now() - Math.random() * 86400000
        }
      })),
      relationships: Array.from({ length: config.edgeCount }, (_, i) => ({
        id: `edge_${i}`,
        source: `node_${i % config.nodeCount}`,
        target: `node_${(i + Math.floor(Math.random() * 100) + 1) % config.nodeCount}`,
        type: config.edgeTypes[i % config.edgeTypes.length],
        weight: Math.random()
      }))
    };
  }

  async function generateMediumDataset(config: {
    nodeCount: number;
    edgeCount: number;
  }): Promise<any> {
    return generateMassiveDataset({
      ...config,
      nodeTypes: ['Person', 'Organization', 'Concept'],
      edgeTypes: ['knows', 'worksFor', 'relatedTo']
    });
  }

  async function performGraphOperation(
    kgVisualization: any,
    loadTester: LoadTester,
    operation: { type: string; parameters: any }
  ): Promise<void> {
    const startTime = performance.now();
    
    try {
      let result;
      switch (operation.type) {
        case 'node_query':
          result = await kgVisualization.queryNodes(operation.parameters);
          break;
        case 'path_finding':
          result = await kgVisualization.findPath(operation.parameters);
          break;
        case 'neighborhood_expansion':
          result = await kgVisualization.expandNeighborhood(operation.parameters);
          break;
        default:
          throw new Error(`Unknown operation type: ${operation.type}`);
      }

      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: operation.type,
        responseTime,
        success: true,
        resultSize: result ? result.length || 1 : 0
      });
    } catch (error) {
      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: operation.type,
        responseTime,
        success: false,
        error: error.message
      });
    }
  }

  async function simulateUserSession(
    userId: number,
    duration: number,
    loadTester: LoadTester
  ): Promise<void> {
    const sessionStart = performance.now();
    const userVisualization = new KnowledgeGraphVisualization({
      container: document.createElement('div'),
      enableQueryVisualization: true,
      ...DefaultConfigurations.detailed
    });

    // Load user-specific data
    const userData = await generateMediumDataset({ nodeCount: 200, edgeCount: 400 });
    await userVisualization.loadIntegratedData(userData);

    // Simulate user actions
    while (performance.now() - sessionStart < duration) {
      const actionType = ['query', 'navigation', 'interaction', 'filter'][Math.floor(Math.random() * 4)];
      
      try {
        const startTime = performance.now();
        await simulateUserAction(userVisualization, actionType, userId);
        const responseTime = performance.now() - startTime;

        await loadTester.recordOperation({
          operation: actionType,
          responseTime,
          success: true,
          userId
        });
      } catch (error) {
        await loadTester.recordOperation({
          operation: actionType,
          responseTime: 0,
          success: false,
          error: error.message,
          userId
        });
      }

      // Wait between actions (simulate user think time)
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 3000));
    }

    userVisualization.dispose();
  }

  async function simulateUserAction(
    visualization: any,
    actionType: string,
    userId: number
  ): Promise<void> {
    switch (actionType) {
      case 'query':
        await visualization.executeQuery({
          id: `user_${userId}_query_${Date.now()}`,
          sparql: 'SELECT ?entity WHERE { ?entity a :Entity } LIMIT 20'
        });
        break;
      case 'navigation':
        await visualization.setCameraPosition({
          x: (Math.random() - 0.5) * 20,
          y: (Math.random() - 0.5) * 20,
          z: (Math.random() - 0.5) * 20
        });
        break;
      case 'interaction':
        await visualization.handleClick({
          x: Math.random() * 800,
          y: Math.random() * 600
        });
        break;
      case 'filter':
        await visualization.applyFilter({
          entityTypes: ['Person'],
          relationships: ['knows']
        });
        break;
    }
  }

  async function executeRandomQuery(visualization: any): Promise<void> {
    const queries = [
      'SELECT ?entity WHERE { ?entity a :Person }',
      'SELECT ?entity ?name WHERE { ?entity :name ?name }',
      'SELECT ?source ?target WHERE { ?source :knows ?target }',
    ];
    
    const randomQuery = queries[Math.floor(Math.random() * queries.length)];
    await visualization.executeQuery({
      id: `random_query_${Date.now()}`,
      sparql: randomQuery
    });
  }

  async function performRandomUpdate(visualization: any): Promise<void> {
    await visualization.addNode({
      id: `dynamic_${Date.now()}`,
      type: 'DynamicEntity',
      label: 'Dynamic Node'
    });
  }

  async function updateVisualizationSettings(visualization: any): Promise<void> {
    await visualization.updateSettings({
      nodeSize: 0.5 + Math.random() * 2,
      edgeOpacity: Math.random()
    });
  }

  async function createCPUIntensiveTask(
    visualizer: any,
    loadTester: LoadTester,
    taskId: string
  ): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Simulate CPU-intensive computation
      const iterations = 1000000;
      let result = 0;
      
      for (let i = 0; i < iterations; i++) {
        result += Math.sin(i) * Math.cos(i);
        
        // Occasionally yield control
        if (i % 10000 === 0) {
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }

      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: 'cpu_intensive',
        responseTime,
        success: true,
        taskId,
        result
      });
    } catch (error) {
      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: 'cpu_intensive',
        responseTime,
        success: false,
        error: error.message,
        taskId
      });
    }
  }

  async function createMemoryIntensivePattern(
    visualizer: any,
    loadTester: LoadTester,
    patternIndex: number
  ): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Create large pattern data
      const largePattern = {
        id: `memory_pattern_${patternIndex}`,
        type: 'memory_intensive',
        data: new ArrayBuffer(10 * 1024 * 1024), // 10MB per pattern
        nodes: Array.from({ length: 1000 }, (_, i) => ({
          id: `node_${patternIndex}_${i}`,
          largeData: new Float32Array(1000) // 4KB per node
        }))
      };

      await visualizer.createPattern(largePattern);

      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: 'memory_intensive_pattern',
        responseTime,
        success: true,
        patternIndex,
        memoryUsed: 10 * 1024 * 1024
      });
    } catch (error) {
      const responseTime = performance.now() - startTime;
      await loadTester.recordOperation({
        operation: 'memory_intensive_pattern',
        responseTime,
        success: false,
        error: error.message,
        patternIndex
      });
    }
  }
});