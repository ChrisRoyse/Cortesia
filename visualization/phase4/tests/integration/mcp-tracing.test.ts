/**
 * MCP Tracing Integration Tests
 * Tests MCP request tracing and visualization components
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { MCPRequestTracer } from '../../src/tracing/MCPRequestTracer';
import { RequestPathVisualizer } from '../../src/tracing/RequestPathVisualizer';
import { TraceAnalytics } from '../../src/tracing/TraceAnalytics';
import { ParticleEffects } from '../../src/tracing/ParticleEffects';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('MCP Tracing Integration Tests', () => {
  let container: HTMLElement;
  let mcpTracer: MCPRequestTracer;
  let pathVisualizer: RequestPathVisualizer;
  let analytics: TraceAnalytics;
  let particleEffects: ParticleEffects;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    mcpTracer = new MCPRequestTracer({
      container,
      enableRequestTracking: true,
      enablePerformanceAnalysis: true,
      visualizeDataFlow: true,
      maxConcurrentTraces: 100
    });

    pathVisualizer = new RequestPathVisualizer({
      container,
      enablePathHighlighting: true,
      enableStepAnimation: true,
      animationSpeed: 1.0
    });

    analytics = new TraceAnalytics({
      enableRealTimeAnalysis: true,
      retentionPeriod: 3600000, // 1 hour
      aggregationInterval: 1000 // 1 second
    });

    particleEffects = new ParticleEffects({
      container,
      maxParticles: 5000,
      enableTrails: true,
      enableGlow: true
    });
  });

  afterEach(() => {
    mcpTracer?.dispose();
    pathVisualizer?.dispose();
    analytics?.dispose();
    particleEffects?.dispose();
  });

  describe('MCP Request Tracing', () => {
    test('should start and track MCP request traces', async () => {
      const requestId = 'test-request-1';
      const traceConfig = {
        id: requestId,
        method: 'resources/list',
        timestamp: Date.now(),
        params: { 
          uri: 'llmkg://knowledge-base',
          filter: 'entities'
        },
        metadata: {
          client: 'llmkg-client',
          version: '1.0.0'
        }
      };

      await mcpTracer.startTrace(traceConfig);

      const trace = await mcpTracer.getTrace(requestId);
      expect(trace).toBeDefined();
      expect(trace.id).toBe(requestId);
      expect(trace.method).toBe('resources/list');
      expect(trace.isActive).toBe(true);
      expect(trace.startTime).toBeDefined();
    });

    test('should add trace steps with detailed information', async () => {
      const requestId = 'test-request-2';
      
      await mcpTracer.startTrace({
        id: requestId,
        method: 'tools/call',
        timestamp: Date.now(),
        params: { name: 'query-knowledge-graph' }
      });

      const steps = [
        {
          requestId,
          stepId: 'validation',
          type: 'pre-processing',
          component: 'mcp-server',
          timestamp: Date.now(),
          duration: 50,
          metadata: { validated: true }
        },
        {
          requestId,
          stepId: 'execution',
          type: 'processing',
          component: 'knowledge-engine',
          timestamp: Date.now() + 50,
          duration: 150,
          metadata: { 
            queriedEntities: 25,
            returndResults: 12
          }
        },
        {
          requestId,
          stepId: 'serialization',
          type: 'post-processing',
          component: 'mcp-server',
          timestamp: Date.now() + 200,
          duration: 25,
          metadata: { serializedSize: 2048 }
        }
      ];

      for (const step of steps) {
        await mcpTracer.addTraceStep(step);
      }

      const trace = await mcpTracer.getTrace(requestId);
      expect(trace.steps.length).toBe(3);
      expect(trace.steps[0].type).toBe('pre-processing');
      expect(trace.steps[1].type).toBe('processing');
      expect(trace.steps[2].type).toBe('post-processing');
    });

    test('should complete traces with summary information', async () => {
      const requestId = 'test-request-3';
      
      await mcpTracer.startTrace({
        id: requestId,
        method: 'prompts/get',
        timestamp: Date.now(),
        params: { name: 'analysis-prompt' }
      });

      await mcpTracer.addTraceStep({
        requestId,
        stepId: 'prompt-retrieval',
        type: 'processing',
        component: 'prompt-manager',
        duration: 100,
        metadata: { promptLength: 1024 }
      });

      const completionData = {
        requestId,
        totalDuration: 150,
        success: true,
        resultSize: 2048,
        errorCount: 0,
        warningCount: 1,
        metadata: {
          cacheHit: false,
          compressionRatio: 0.75
        }
      };

      await mcpTracer.completeTrace(completionData);

      const trace = await mcpTracer.getTrace(requestId);
      expect(trace.isActive).toBe(false);
      expect(trace.isComplete).toBe(true);
      expect(trace.totalDuration).toBe(150);
      expect(trace.success).toBe(true);
      expect(trace.resultSize).toBe(2048);
    });

    test('should handle concurrent traces', async () => {
      const traceCount = 50;
      const tracePromises = [];

      for (let i = 0; i < traceCount; i++) {
        const tracePromise = mcpTracer.startTrace({
          id: `concurrent-trace-${i}`,
          method: 'resources/list',
          timestamp: Date.now() + i,
          params: { index: i }
        });
        tracePromises.push(tracePromise);
      }

      await Promise.all(tracePromises);

      const activeTraces = await mcpTracer.getActiveTraces();
      expect(activeTraces.length).toBe(traceCount);

      // Complete all traces
      const completionPromises = activeTraces.map(trace => 
        mcpTracer.completeTrace({
          requestId: trace.id,
          totalDuration: 100 + Math.random() * 100,
          success: true,
          resultSize: 1024
        })
      );

      await Promise.all(completionPromises);

      const remainingActiveTraces = await mcpTracer.getActiveTraces();
      expect(remainingActiveTraces.length).toBe(0);
    });
  });

  describe('Request Path Visualization', () => {
    test('should visualize request paths through system components', async () => {
      const pathConfig = {
        id: 'test-path-1',
        name: 'Knowledge Query Path',
        components: [
          { id: 'mcp-client', position: { x: -10, y: 0, z: 0 }, type: 'client' },
          { id: 'mcp-server', position: { x: -5, y: 0, z: 0 }, type: 'server' },
          { id: 'knowledge-engine', position: { x: 0, y: 0, z: 0 }, type: 'engine' },
          { id: 'storage-system', position: { x: 5, y: 0, z: 0 }, type: 'storage' },
          { id: 'response-formatter', position: { x: 10, y: 0, z: 0 }, type: 'formatter' }
        ],
        connections: [
          { from: 'mcp-client', to: 'mcp-server', type: 'request' },
          { from: 'mcp-server', to: 'knowledge-engine', type: 'query' },
          { from: 'knowledge-engine', to: 'storage-system', type: 'data-access' },
          { from: 'storage-system', to: 'knowledge-engine', type: 'data-return' },
          { from: 'knowledge-engine', to: 'response-formatter', type: 'format-request' },
          { from: 'response-formatter', to: 'mcp-server', type: 'formatted-response' },
          { from: 'mcp-server', to: 'mcp-client', type: 'response' }
        ]
      };

      await pathVisualizer.createPath(pathConfig);

      const path = await pathVisualizer.getPath('test-path-1');
      expect(path).toBeDefined();
      expect(path.components.length).toBe(5);
      expect(path.connections.length).toBe(7);
    });

    test('should animate request flow along paths', async () => {
      const pathId = 'animated-path-1';
      
      await pathVisualizer.createPath({
        id: pathId,
        name: 'Animated Request Path',
        components: [
          { id: 'start', position: { x: 0, y: 0, z: 0 }, type: 'client' },
          { id: 'middle', position: { x: 5, y: 0, z: 0 }, type: 'server' },
          { id: 'end', position: { x: 10, y: 0, z: 0 }, type: 'storage' }
        ],
        connections: [
          { from: 'start', to: 'middle', type: 'request' },
          { from: 'middle', to: 'end', type: 'query' }
        ]
      });

      const animationConfig = {
        pathId,
        requestId: 'animated-request-1',
        flowSpeed: 2.0,
        particleCount: 50,
        highlightColor: new THREE.Color(0x00ff00),
        duration: 2000 // 2 seconds
      };

      const animation = await pathVisualizer.startAnimation(animationConfig);
      expect(animation).toBeDefined();
      expect(animation.isActive).toBe(true);

      // Wait for animation to complete
      await new Promise(resolve => setTimeout(resolve, 2100));

      const completedAnimation = await pathVisualizer.getAnimation(animationConfig.requestId);
      expect(completedAnimation.isActive).toBe(false);
      expect(completedAnimation.isComplete).toBe(true);
    });

    test('should highlight bottlenecks and performance issues', async () => {
      const pathId = 'performance-path-1';
      
      await pathVisualizer.createPath({
        id: pathId,
        name: 'Performance Analysis Path',
        components: [
          { id: 'fast-component', position: { x: 0, y: 0, z: 0 }, type: 'client' },
          { id: 'slow-component', position: { x: 5, y: 0, z: 0 }, type: 'server' },
          { id: 'normal-component', position: { x: 10, y: 0, z: 0 }, type: 'storage' }
        ],
        connections: [
          { from: 'fast-component', to: 'slow-component', type: 'request' },
          { from: 'slow-component', to: 'normal-component', type: 'query' }
        ]
      });

      const performanceData = {
        pathId,
        componentMetrics: [
          { componentId: 'fast-component', avgResponseTime: 50, errorRate: 0.01 },
          { componentId: 'slow-component', avgResponseTime: 1500, errorRate: 0.05 }, // Bottleneck
          { componentId: 'normal-component', avgResponseTime: 200, errorRate: 0.02 }
        ]
      };

      await pathVisualizer.updatePerformanceData(performanceData);

      const bottlenecks = await pathVisualizer.identifyBottlenecks(pathId);
      expect(bottlenecks).toBeDefined();
      expect(bottlenecks.length).toBe(1);
      expect(bottlenecks[0].componentId).toBe('slow-component');
      expect(bottlenecks[0].severity).toBe('high');
    });
  });

  describe('Trace Analytics', () => {
    test('should collect and analyze trace metrics', async () => {
      // Generate sample trace data
      const traces = [];
      for (let i = 0; i < 100; i++) {
        const trace = {
          id: `analysis-trace-${i}`,
          method: ['resources/list', 'tools/call', 'prompts/get'][i % 3],
          startTime: Date.now() - (Math.random() * 3600000), // Last hour
          duration: 50 + Math.random() * 500, // 50-550ms
          success: Math.random() > 0.1, // 90% success rate
          resultSize: Math.floor(Math.random() * 10000) + 100,
          steps: Math.floor(Math.random() * 5) + 1
        };
        traces.push(trace);
      }

      await analytics.ingestTraces(traces);

      const metrics = await analytics.getMetrics();
      expect(metrics.totalTraces).toBe(100);
      expect(metrics.avgDuration).toBeGreaterThan(0);
      expect(metrics.successRate).toBeGreaterThan(0.8);
      expect(metrics.errorRate).toBeLessThan(0.2);
    });

    test('should detect performance anomalies', async () => {
      // Create traces with some anomalous patterns
      const normalTraces = Array.from({ length: 90 }, (_, i) => ({
        id: `normal-trace-${i}`,
        method: 'resources/list',
        duration: 100 + Math.random() * 50, // 100-150ms (normal)
        success: true,
        timestamp: Date.now() - i * 1000
      }));

      const anomalousTraces = Array.from({ length: 10 }, (_, i) => ({
        id: `anomaly-trace-${i}`,
        method: 'resources/list',
        duration: 2000 + Math.random() * 1000, // 2000-3000ms (anomalous)
        success: Math.random() > 0.5, // 50% success rate
        timestamp: Date.now() - (90 + i) * 1000
      }));

      await analytics.ingestTraces([...normalTraces, ...anomalousTraces]);

      const anomalies = await analytics.detectAnomalies({
        timeWindow: 3600000, // 1 hour
        thresholds: {
          duration: { upper: 500 }, // > 500ms is anomalous
          errorRate: { upper: 0.2 }  // > 20% error rate is anomalous
        }
      });

      expect(anomalies.length).toBeGreaterThan(0);
      const durationAnomalies = anomalies.filter(a => a.type === 'duration');
      expect(durationAnomalies.length).toBeGreaterThan(0);
    });

    test('should provide trend analysis', async () => {
      // Generate trending data over time
      const timeWindows = 24; // 24 hours
      const traces = [];

      for (let hour = 0; hour < timeWindows; hour++) {
        const baseTime = Date.now() - (timeWindows - hour) * 3600000;
        const traceCount = 50 + Math.floor(Math.sin(hour / 4) * 20); // Simulate daily pattern
        
        for (let i = 0; i < traceCount; i++) {
          traces.push({
            id: `trend-trace-${hour}-${i}`,
            method: 'resources/list',
            timestamp: baseTime + (i * 60000), // Spread over the hour
            duration: 100 + Math.random() * 200,
            success: true
          });
        }
      }

      await analytics.ingestTraces(traces);

      const trends = await analytics.getTrends({
        timeWindow: 24 * 3600000, // 24 hours
        granularity: 3600000 // 1 hour buckets
      });

      expect(trends.length).toBe(24);
      expect(trends[0].timestamp).toBeDefined();
      expect(trends[0].metrics.traceCount).toBeGreaterThan(0);
      expect(trends[0].metrics.avgDuration).toBeGreaterThan(0);
    });
  });

  describe('Particle Effects Integration', () => {
    test('should create particle effects for trace visualization', () => {
      const effectConfig = {
        id: 'trace-flow-effect',
        type: 'flow',
        source: { x: -5, y: 0, z: 0 },
        target: { x: 5, y: 0, z: 0 },
        particleCount: 100,
        color: new THREE.Color(0x00aaff),
        speed: 2.0,
        lifetime: 3.0
      };

      particleEffects.createEffect(effectConfig);

      const effect = particleEffects.getEffect('trace-flow-effect');
      expect(effect).toBeDefined();
      expect(effect.particleCount).toBe(100);
      expect(effect.isActive).toBe(true);
    });

    test('should synchronize particle effects with trace events', async () => {
      const requestId = 'synchronized-request';
      
      // Start MCP trace
      await mcpTracer.startTrace({
        id: requestId,
        method: 'tools/call',
        timestamp: Date.now()
      });

      // Create synchronized particle effect
      await particleEffects.createSynchronizedEffect({
        traceId: requestId,
        type: 'pulse',
        intensity: 1.0,
        color: new THREE.Color(0xff4400),
        duration: 2000
      });

      // Add trace step and verify effect updates
      await mcpTracer.addTraceStep({
        requestId,
        stepId: 'processing-step',
        type: 'processing',
        component: 'knowledge-engine',
        duration: 150
      });

      // Check that particle effect responded to trace event
      const effect = particleEffects.getSynchronizedEffect(requestId);
      expect(effect).toBeDefined();
      expect(effect.isActive).toBe(true);
      expect(effect.currentIntensity).toBeGreaterThan(0);
    });

    test('should handle multiple concurrent particle effects', () => {
      const effectConfigs = Array.from({ length: 20 }, (_, i) => ({
        id: `concurrent-effect-${i}`,
        type: i % 2 === 0 ? 'flow' : 'burst',
        source: { x: (i % 5) * 2 - 4, y: 0, z: 0 },
        target: { x: (i % 5) * 2 - 2, y: 0, z: 0 },
        particleCount: 50,
        color: new THREE.Color().setHSL(i / 20, 1, 0.5),
        speed: 1 + Math.random()
      }));

      effectConfigs.forEach(config => {
        particleEffects.createEffect(config);
      });

      const activeEffects = particleEffects.getActiveEffects();
      expect(activeEffects.length).toBe(20);

      // Update all effects
      particleEffects.update(1/60); // 60 FPS

      // All effects should still be active
      const stillActiveEffects = particleEffects.getActiveEffects();
      expect(stillActiveEffects.length).toBe(20);
    });
  });

  describe('Integration and Performance', () => {
    test('should handle high-frequency trace events', async () => {
      const eventCount = 1000;
      const events = [];

      for (let i = 0; i < eventCount; i++) {
        events.push({
          id: `high-freq-trace-${i}`,
          method: 'resources/list',
          timestamp: Date.now() + i,
          params: { index: i }
        });
      }

      const startTime = performance.now();
      
      await Promise.all(events.map(event => mcpTracer.startTrace(event)));

      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // Should process 1000 events in reasonable time
      expect(processingTime).toBeLessThan(1000); // 1 second

      const activeTraces = await mcpTracer.getActiveTraces();
      expect(activeTraces.length).toBe(eventCount);
    });

    test('should maintain visualization performance with many traces', async () => {
      // Create substantial trace load
      for (let i = 0; i < 100; i++) {
        await mcpTracer.startTrace({
          id: `perf-trace-${i}`,
          method: 'tools/call',
          timestamp: Date.now() + i * 10
        });
      }

      // Create path visualizations
      for (let i = 0; i < 10; i++) {
        await pathVisualizer.createPath({
          id: `perf-path-${i}`,
          name: `Performance Path ${i}`,
          components: Array.from({ length: 5 }, (_, j) => ({
            id: `comp-${i}-${j}`,
            position: { x: j * 2, y: i, z: 0 },
            type: 'server'
          })),
          connections: Array.from({ length: 4 }, (_, j) => ({
            from: `comp-${i}-${j}`,
            to: `comp-${i}-${j + 1}`,
            type: 'request'
          }))
        });
      }

      // Measure render performance
      const frameCount = await measureRenderPerformance(1000); // 1 second
      const fps = frameCount;

      expect(fps).toBeGreaterThanOrEqual(30); // At least 30 FPS under load
    });
  });

  // Helper function
  async function measureRenderPerformance(duration: number): Promise<number> {
    let frameCount = 0;
    const startTime = performance.now();
    
    return new Promise((resolve) => {
      function frame() {
        frameCount++;
        pathVisualizer.render();
        particleEffects.update(1/60);
        
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