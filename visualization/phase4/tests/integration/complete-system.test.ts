/**
 * Complete System Integration Tests
 * Tests the full Phase 4 visualization system end-to-end
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { KnowledgeGraphVisualization, DefaultConfigurations } from '../../src/knowledge';
import { LLMKGDataFlowVisualizer } from '../../src/core/LLMKGDataFlowVisualizer';
import { CognitivePatternVisualizer } from '../../src/cognitive/CognitivePatternVisualizer';
import { MemoryOperationVisualizer } from '../../src/memory/MemoryOperationVisualizer';
import { MCPRequestTracer } from '../../src/tracing/MCPRequestTracer';

// Setup DOM environment for testing
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Phase 4 Complete System Integration', () => {
  let container: HTMLElement;
  let visualization: KnowledgeGraphVisualization;
  let dataFlowVisualizer: LLMKGDataFlowVisualizer;
  let cognitiveVisualizer: CognitivePatternVisualizer;
  let memoryVisualizer: MemoryOperationVisualizer;
  let mcpTracer: MCPRequestTracer;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    // Initialize all visualization components
    visualization = new KnowledgeGraphVisualization({
      container,
      enableQueryVisualization: true,
      enableEntityFlow: true,
      enableTripleStore: true,
      ...DefaultConfigurations.detailed
    });

    dataFlowVisualizer = new LLMKGDataFlowVisualizer({
      container,
      enableParticleEffects: true,
      enableDataStreams: true,
      performanceMode: 'high-quality'
    });

    cognitiveVisualizer = new CognitivePatternVisualizer({
      container,
      enablePatternRecognition: true,
      enableInhibitoryPatterns: true,
      enableHierarchicalProcessing: true
    });

    memoryVisualizer = new MemoryOperationVisualizer({
      container,
      enableSDRVisualization: true,
      enableMemoryAnalytics: true,
      realTimeUpdates: true
    });

    mcpTracer = new MCPRequestTracer({
      container,
      enableRequestTracking: true,
      enablePerformanceAnalysis: true,
      visualizeDataFlow: true
    });
  });

  afterEach(() => {
    visualization?.dispose();
    dataFlowVisualizer?.dispose();
    cognitiveVisualizer?.dispose();
    memoryVisualizer?.dispose();
    mcpTracer?.dispose();
  });

  describe('System Initialization', () => {
    test('should initialize all components without errors', () => {
      expect(visualization).toBeDefined();
      expect(dataFlowVisualizer).toBeDefined();
      expect(cognitiveVisualizer).toBeDefined();
      expect(memoryVisualizer).toBeDefined();
      expect(mcpTracer).toBeDefined();
    });

    test('should have valid WebGL context', () => {
      const renderer = visualization.getRenderer();
      expect(renderer).toBeInstanceOf(THREE.WebGLRenderer);
      expect(renderer.getContext()).toBeDefined();
    });

    test('should initialize with correct configuration', () => {
      const config = visualization.getConfiguration();
      expect(config.nodeSize).toBeDefined();
      expect(config.edgeWidth).toBeDefined();
      expect(config.forceStrength).toBeGreaterThan(0);
    });
  });

  describe('Cross-Component Integration', () => {
    test('should share data between knowledge graph and data flow visualizer', async () => {
      // Add nodes to knowledge graph
      const nodeId = 'test-entity-1';
      visualization.addNode({
        id: nodeId,
        type: 'entity',
        position: new THREE.Vector3(0, 0, 0),
        metadata: { category: 'person', importance: 0.8 }
      });

      // Check if data flow visualizer can access the same node
      const dataFlowNode = await dataFlowVisualizer.getEntityData(nodeId);
      expect(dataFlowNode).toBeDefined();
      expect(dataFlowNode.id).toBe(nodeId);
    });

    test('should synchronize cognitive patterns with knowledge graph changes', async () => {
      // Create cognitive pattern
      const patternId = 'test-pattern-1';
      await cognitiveVisualizer.createPattern({
        id: patternId,
        type: 'convergent',
        strength: 0.7,
        nodes: ['node1', 'node2', 'node3']
      });

      // Add nodes to knowledge graph
      ['node1', 'node2', 'node3'].forEach(id => {
        visualization.addNode({
          id,
          type: 'concept',
          position: new THREE.Vector3(Math.random() * 10, Math.random() * 10, Math.random() * 10)
        });
      });

      // Verify pattern synchronization
      const pattern = await cognitiveVisualizer.getPattern(patternId);
      expect(pattern.activeNodes.length).toBe(3);
      expect(pattern.isActive).toBe(true);
    });

    test('should track memory operations in real-time', async () => {
      const operationId = 'memory-op-1';
      
      // Simulate memory operation
      await memoryVisualizer.trackOperation({
        id: operationId,
        type: 'sdr-encoding',
        inputSize: 1024,
        timestamp: Date.now(),
        metadata: { sparsity: 0.02, dimensions: 2048 }
      });

      // Verify operation tracking
      const operation = await memoryVisualizer.getOperation(operationId);
      expect(operation).toBeDefined();
      expect(operation.type).toBe('sdr-encoding');
      expect(operation.metrics).toBeDefined();
    });

    test('should trace MCP requests with visualization', async () => {
      const requestId = 'mcp-request-1';
      
      // Start tracing MCP request
      await mcpTracer.startTrace({
        id: requestId,
        method: 'resources/list',
        timestamp: Date.now(),
        params: { filter: 'knowledge-graph' }
      });

      // Add trace steps
      await mcpTracer.addTraceStep({
        requestId,
        stepId: 'step-1',
        type: 'processing',
        duration: 50,
        metadata: { component: 'knowledge-engine' }
      });

      // Complete trace
      await mcpTracer.completeTrace({
        requestId,
        totalDuration: 150,
        success: true,
        resultSize: 2048
      });

      // Verify trace visualization
      const trace = await mcpTracer.getTrace(requestId);
      expect(trace).toBeDefined();
      expect(trace.steps.length).toBeGreaterThan(0);
      expect(trace.isComplete).toBe(true);
    });
  });

  describe('Performance Integration', () => {
    test('should maintain target frame rate with all components active', async () => {
      const targetFPS = 60;
      const testDuration = 1000; // 1 second
      
      // Add substantial data to all components
      await Promise.all([
        addTestDataToKnowledgeGraph(100),
        addTestDataToCognitivePatterns(50),
        addTestMemoryOperations(200),
        addTestMCPTraces(30)
      ]);

      // Measure performance
      const frameCount = await measureFrameRate(testDuration);
      const actualFPS = (frameCount * 1000) / testDuration;
      
      expect(actualFPS).toBeGreaterThanOrEqual(targetFPS * 0.9); // Allow 10% tolerance
    });

    test('should handle memory usage within limits', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Load substantial test data
      await Promise.all([
        addTestDataToKnowledgeGraph(500),
        addTestDataToCognitivePatterns(100),
        addTestMemoryOperations(1000),
        addTestMCPTraces(100)
      ]);

      const peakMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = peakMemory - initialMemory;
      
      // Memory increase should be less than 1GB for typical workload
      expect(memoryIncrease).toBeLessThan(1024 * 1024 * 1024);
    });

    test('should respond to user interactions within latency limits', async () => {
      const maxLatency = 100; // 100ms
      
      const startTime = performance.now();
      
      // Simulate user interaction
      await visualization.handleClick({ x: 100, y: 100, button: 0 });
      
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      expect(latency).toBeLessThan(maxLatency);
    });
  });

  describe('Data Flow Integration', () => {
    test('should handle complete data pipeline', async () => {
      const pipelineId = 'test-pipeline-1';
      
      // Start MCP request trace
      await mcpTracer.startTrace({
        id: `${pipelineId}-request`,
        method: 'query/execute',
        timestamp: Date.now(),
        params: { query: 'SELECT * FROM entities' }
      });

      // Process cognitive pattern
      const patternResult = await cognitiveVisualizer.processPattern({
        type: 'systems-thinking',
        input: { entities: ['entity1', 'entity2'], relationships: ['rel1'] },
        expectedOutput: { insights: [], connections: [] }
      });

      // Execute memory operation
      const memoryResult = await memoryVisualizer.executeOperation({
        type: 'pattern-storage',
        data: patternResult,
        compression: true
      });

      // Update knowledge graph
      visualization.updateFromPipeline({
        pipelineId,
        patternResult,
        memoryResult,
        timestamp: Date.now()
      });

      // Verify end-to-end flow
      const pipelineMetrics = await dataFlowVisualizer.getPipelineMetrics(pipelineId);
      expect(pipelineMetrics.totalLatency).toBeLessThan(1000);
      expect(pipelineMetrics.throughput).toBeGreaterThan(0);
      expect(pipelineMetrics.errorRate).toBe(0);
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should gracefully handle component failures', async () => {
      // Simulate component failure
      const originalRender = visualization.render;
      visualization.render = jest.fn().mockImplementation(() => {
        throw new Error('Simulated render failure');
      });

      // System should continue operating
      expect(() => {
        dataFlowVisualizer.update();
        cognitiveVisualizer.update();
        memoryVisualizer.update();
      }).not.toThrow();

      // Restore original render
      visualization.render = originalRender;
    });

    test('should recover from WebGL context loss', async () => {
      const contextLossEvent = new Event('webglcontextlost');
      const contextRestoredEvent = new Event('webglcontextrestored');

      // Simulate context loss
      visualization.getRenderer().domElement.dispatchEvent(contextLossEvent);
      
      // Verify system handles context loss
      expect(visualization.isContextLost()).toBe(true);
      
      // Simulate context restoration
      visualization.getRenderer().domElement.dispatchEvent(contextRestoredEvent);
      
      // Verify system recovers
      expect(visualization.isContextLost()).toBe(false);
      expect(() => visualization.render()).not.toThrow();
    });
  });

  // Helper functions
  async function addTestDataToKnowledgeGraph(nodeCount: number): Promise<void> {
    for (let i = 0; i < nodeCount; i++) {
      visualization.addNode({
        id: `test-node-${i}`,
        type: Math.random() > 0.5 ? 'entity' : 'relation',
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20
        )
      });

      if (i > 0) {
        visualization.addEdge({
          id: `test-edge-${i}`,
          source: `test-node-${i - 1}`,
          target: `test-node-${i}`,
          type: 'connection'
        });
      }
    }
  }

  async function addTestDataToCognitivePatterns(patternCount: number): Promise<void> {
    const patternTypes = ['convergent', 'divergent', 'lateral', 'systems'];
    
    for (let i = 0; i < patternCount; i++) {
      await cognitiveVisualizer.createPattern({
        id: `test-pattern-${i}`,
        type: patternTypes[i % patternTypes.length] as any,
        strength: Math.random(),
        nodes: Array.from({ length: 3 + Math.floor(Math.random() * 5) }, (_, j) => `node-${i}-${j}`)
      });
    }
  }

  async function addTestMemoryOperations(operationCount: number): Promise<void> {
    const operationTypes = ['encoding', 'retrieval', 'consolidation', 'forgetting'];
    
    for (let i = 0; i < operationCount; i++) {
      await memoryVisualizer.trackOperation({
        id: `test-memory-op-${i}`,
        type: operationTypes[i % operationTypes.length] as any,
        inputSize: Math.floor(Math.random() * 2048) + 256,
        timestamp: Date.now() + i,
        metadata: { 
          sparsity: Math.random() * 0.1,
          dimensions: 1024 + Math.floor(Math.random() * 1024)
        }
      });
    }
  }

  async function addTestMCPTraces(traceCount: number): Promise<void> {
    const methods = ['resources/list', 'tools/call', 'prompts/get', 'query/execute'];
    
    for (let i = 0; i < traceCount; i++) {
      const requestId = `test-mcp-trace-${i}`;
      
      await mcpTracer.startTrace({
        id: requestId,
        method: methods[i % methods.length],
        timestamp: Date.now() + i * 10,
        params: { test: true }
      });

      await mcpTracer.completeTrace({
        requestId,
        totalDuration: Math.random() * 200 + 50,
        success: Math.random() > 0.1,
        resultSize: Math.floor(Math.random() * 4096)
      });
    }
  }

  async function measureFrameRate(duration: number): Promise<number> {
    let frameCount = 0;
    const startTime = performance.now();
    
    return new Promise((resolve) => {
      function frame() {
        frameCount++;
        
        if (performance.now() - startTime < duration) {
          requestAnimationFrame(frame);
        } else {
          resolve(frameCount);
        }
      }
      
      requestAnimationFrame(frame);
    });
  }
});