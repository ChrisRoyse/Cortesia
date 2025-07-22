/**
 * Rendering Performance Benchmark Tests
 * Tests rendering performance under various conditions and loads
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { LLMKGDataFlowVisualizer } from '../../src/core/LLMKGDataFlowVisualizer';
import { KnowledgeGraphAnimator } from '../../src/knowledge/KnowledgeGraphAnimator';
import { CognitivePatternVisualizer } from '../../src/cognitive/CognitivePatternVisualizer';
import { MemoryOperationVisualizer } from '../../src/memory/MemoryOperationVisualizer';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Rendering Performance Benchmark Tests', () => {
  let container: HTMLElement;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
  });

  describe('Target 60 FPS Performance', () => {
    test('should maintain 60 FPS with small knowledge graph (100 nodes)', async () => {
      const kgAnimator = new KnowledgeGraphAnimator({
        container,
        enablePhysicsSimulation: true,
        maxNodes: 1000,
        targetFPS: 60
      });

      // Create small graph
      const nodes = Array.from({ length: 100 }, (_, i) => ({
        id: `node_${i}`,
        type: 'Entity',
        label: `Node ${i}`,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 20
        )
      }));

      const edges = Array.from({ length: 200 }, (_, i) => ({
        id: `edge_${i}`,
        source: `node_${i % 100}`,
        target: `node_${(i + Math.floor(Math.random() * 50) + 1) % 100}`,
        type: 'CONNECTED_TO'
      }));

      await kgAnimator.loadGraph({ nodes, edges });

      // Benchmark rendering performance
      const benchmark = await measureRenderingPerformance(kgAnimator, {
        duration: 3000, // 3 seconds
        targetFPS: 60,
        tolerance: 0.9 // Allow 10% deviation
      });

      expect(benchmark.averageFPS).toBeGreaterThanOrEqual(54); // 60 * 0.9
      expect(benchmark.frameTimeP95).toBeLessThan(20); // 95th percentile < 20ms
      expect(benchmark.droppedFrames).toBeLessThan(5); // Less than 5 dropped frames

      kgAnimator.dispose();
    });

    test('should maintain acceptable performance with medium knowledge graph (500 nodes)', async () => {
      const kgAnimator = new KnowledgeGraphAnimator({
        container,
        enablePhysicsSimulation: true,
        maxNodes: 1000,
        targetFPS: 60
      });

      // Create medium graph
      const nodes = Array.from({ length: 500 }, (_, i) => ({
        id: `node_${i}`,
        type: ['Entity', 'Concept', 'Relation'][i % 3],
        label: `Node ${i}`,
        properties: { weight: Math.random(), importance: Math.random() }
      }));

      const edges = Array.from({ length: 1000 }, (_, i) => ({
        id: `edge_${i}`,
        source: `node_${i % 500}`,
        target: `node_${(i + Math.floor(Math.random() * 100) + 1) % 500}`,
        type: 'RELATED_TO',
        weight: Math.random()
      }));

      await kgAnimator.loadGraph({ nodes, edges });

      const benchmark = await measureRenderingPerformance(kgAnimator, {
        duration: 5000,
        targetFPS: 60,
        tolerance: 0.8 // Allow 20% deviation for larger graph
      });

      expect(benchmark.averageFPS).toBeGreaterThanOrEqual(48); // 60 * 0.8
      expect(benchmark.frameTimeP95).toBeLessThan(25);
      expect(benchmark.memoryUsage.peak).toBeLessThan(200 * 1024 * 1024); // 200MB

      kgAnimator.dispose();
    });

    test('should handle large knowledge graph (1000 nodes) with optimizations', async () => {
      const kgAnimator = new KnowledgeGraphAnimator({
        container,
        enablePhysicsSimulation: true,
        maxNodes: 1000,
        targetFPS: 60,
        enableLOD: true, // Level of Detail
        enableCulling: true, // Frustum culling
        enableInstancing: true // Instance rendering
      });

      // Create large graph
      const nodes = Array.from({ length: 1000 }, (_, i) => ({
        id: `node_${i}`,
        type: ['Entity', 'Concept', 'Relation', 'Process'][i % 4],
        label: `Node ${i}`,
        properties: { 
          weight: Math.random(), 
          importance: Math.random(),
          category: `category_${i % 20}`
        }
      }));

      const edges = Array.from({ length: 2000 }, (_, i) => ({
        id: `edge_${i}`,
        source: `node_${i % 1000}`,
        target: `node_${(i + Math.floor(Math.random() * 200) + 1) % 1000}`,
        type: ['RELATED_TO', 'PART_OF', 'INFLUENCES', 'CAUSES'][i % 4],
        weight: Math.random()
      }));

      await kgAnimator.loadGraph({ nodes, edges });

      const benchmark = await measureRenderingPerformance(kgAnimator, {
        duration: 10000, // 10 seconds for large graph
        targetFPS: 30, // Lower target for large graphs
        tolerance: 0.75
      });

      expect(benchmark.averageFPS).toBeGreaterThanOrEqual(22.5); // 30 * 0.75
      expect(benchmark.frameTimeP95).toBeLessThan(45);
      expect(benchmark.memoryUsage.peak).toBeLessThan(500 * 1024 * 1024); // 500MB
      expect(benchmark.gpuMemoryUsage).toBeLessThan(256 * 1024 * 1024); // 256MB GPU

      kgAnimator.dispose();
    });
  });

  describe('Multi-Component Rendering Performance', () => {
    test('should maintain performance with all visualization components active', async () => {
      const dataFlowVisualizer = new LLMKGDataFlowVisualizer({
        container,
        enableParticleEffects: true,
        enableDataStreams: true,
        performanceMode: 'high-quality'
      });

      const cognitiveVisualizer = new CognitivePatternVisualizer({
        container,
        enablePatternRecognition: true,
        enableRealTimeAnalysis: true
      });

      const memoryVisualizer = new MemoryOperationVisualizer({
        container,
        enableSDRVisualization: true,
        realTimeUpdates: true
      });

      // Load test data into each component
      await Promise.all([
        loadTestDataFlow(dataFlowVisualizer, 50),
        loadTestCognitivePatterns(cognitiveVisualizer, 20),
        loadTestMemoryOperations(memoryVisualizer, 100)
      ]);

      const benchmark = await measureMultiComponentPerformance([
        dataFlowVisualizer,
        cognitiveVisualizer,
        memoryVisualizer
      ], {
        duration: 5000,
        targetFPS: 60,
        tolerance: 0.8
      });

      expect(benchmark.combinedFPS).toBeGreaterThanOrEqual(48);
      expect(benchmark.renderTimeBreakdown.dataFlow).toBeLessThan(20);
      expect(benchmark.renderTimeBreakdown.cognitive).toBeLessThan(15);
      expect(benchmark.renderTimeBreakdown.memory).toBeLessThan(10);
      expect(benchmark.totalRenderTime).toBeLessThan(16.67); // Target 60 FPS

      // Cleanup
      dataFlowVisualizer.dispose();
      cognitiveVisualizer.dispose();
      memoryVisualizer.dispose();
    });

    test('should scale performance based on available resources', async () => {
      const scenarios = [
        { name: 'high-end', cpuCores: 8, gpuMemory: 8192, targetFPS: 60 },
        { name: 'mid-range', cpuCores: 4, gpuMemory: 4096, targetFPS: 45 },
        { name: 'low-end', cpuCores: 2, gpuMemory: 2048, targetFPS: 30 }
      ];

      for (const scenario of scenarios) {
        const visualizer = new LLMKGDataFlowVisualizer({
          container,
          performanceMode: 'auto',
          resourceConstraints: {
            cpuCores: scenario.cpuCores,
            gpuMemory: scenario.gpuMemory
          },
          targetFPS: scenario.targetFPS
        });

        await loadStandardTestData(visualizer);

        const benchmark = await measureRenderingPerformance(visualizer, {
          duration: 3000,
          targetFPS: scenario.targetFPS,
          tolerance: 0.85
        });

        const expectedMinFPS = scenario.targetFPS * 0.85;
        expect(benchmark.averageFPS).toBeGreaterThanOrEqual(expectedMinFPS);
        expect(benchmark.adaptiveOptimizations.applied.length).toBeGreaterThan(0);

        visualizer.dispose();
      }
    });
  });

  describe('Stress Testing', () => {
    test('should handle extreme load without crashing', async () => {
      const stressVisualizer = new LLMKGDataFlowVisualizer({
        container,
        enableParticleEffects: true,
        maxParticles: 50000,
        performanceMode: 'high-performance'
      });

      // Create extreme load
      const extremeDataStreams = Array.from({ length: 100 }, (_, i) => ({
        id: `stress_stream_${i}`,
        source: { x: (i % 10) * 5, y: 0, z: 0 },
        target: { x: ((i + 5) % 10) * 5, y: 10, z: 0 },
        flowRate: 200, // High particle flow
        particleCount: 500
      }));

      // Load data gradually to avoid overwhelming the system
      for (let i = 0; i < extremeDataStreams.length; i += 10) {
        const batch = extremeDataStreams.slice(i, i + 10);
        await Promise.all(batch.map(stream => stressVisualizer.addDataStream(stream)));
        
        // Brief pause to allow system to stabilize
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const stressBenchmark = await measureRenderingPerformance(stressVisualizer, {
        duration: 15000, // 15 seconds stress test
        targetFPS: 20, // Lower expectations under extreme load
        tolerance: 0.7
      });

      expect(stressBenchmark.didCrash).toBe(false);
      expect(stressBenchmark.averageFPS).toBeGreaterThanOrEqual(14); // 20 * 0.7
      expect(stressBenchmark.memoryLeaks.detected).toBe(false);
      expect(stressBenchmark.systemRecovery.successful).toBe(true);

      stressVisualizer.dispose();
    });

    test('should gracefully degrade performance under memory pressure', async () => {
      const memoryPressureVisualizer = new LLMKGDataFlowVisualizer({
        container,
        enableAdaptiveQuality: true,
        memoryLimit: 512 * 1024 * 1024, // 512MB limit
        performanceMode: 'adaptive'
      });

      // Gradually increase memory usage
      const memoryIntensiveData = [];
      let currentMemoryUsage = 0;
      const targetMemoryUsage = 600 * 1024 * 1024; // 600MB (above limit)

      while (currentMemoryUsage < targetMemoryUsage) {
        const largeDataSet = {
          id: `memory_data_${memoryIntensiveData.length}`,
          particles: new Float32Array(10000), // 40KB per dataset
          textures: Array.from({ length: 10 }, () => new Uint8Array(1024 * 1024)) // 10MB per dataset
        };
        
        memoryIntensiveData.push(largeDataSet);
        currentMemoryUsage += 10.04 * 1024 * 1024; // Approximate size

        await memoryPressureVisualizer.loadData(largeDataSet);
        
        const memoryStatus = await memoryPressureVisualizer.getMemoryStatus();
        if (memoryStatus.pressureLevel > 0.8) {
          break; // Stop before system becomes unstable
        }
      }

      const degradationBenchmark = await measureRenderingPerformance(memoryPressureVisualizer, {
        duration: 8000,
        targetFPS: 30,
        tolerance: 0.6, // Expect significant degradation
        trackDegradation: true
      });

      expect(degradationBenchmark.adaptiveDegradation.triggered).toBe(true);
      expect(degradationBenchmark.adaptiveDegradation.qualityReductions.length).toBeGreaterThan(0);
      expect(degradationBenchmark.memoryUsage.stayedBelowLimit).toBe(true);
      expect(degradationBenchmark.averageFPS).toBeGreaterThanOrEqual(18); // 30 * 0.6

      memoryPressureVisualizer.dispose();
    });
  });

  describe('Platform Compatibility Performance', () => {
    test('should optimize for WebGL1 vs WebGL2 capabilities', async () => {
      const webgl1Visualizer = new LLMKGDataFlowVisualizer({
        container,
        forceWebGL1: true,
        enableInstancedRendering: false, // Not available in WebGL1
        enableComputeShaders: false
      });

      const webgl2Visualizer = new LLMKGDataFlowVisualizer({
        container,
        preferWebGL2: true,
        enableInstancedRendering: true,
        enableComputeShaders: true
      });

      await loadStandardTestData(webgl1Visualizer);
      await loadStandardTestData(webgl2Visualizer);

      const [webgl1Benchmark, webgl2Benchmark] = await Promise.all([
        measureRenderingPerformance(webgl1Visualizer, { duration: 3000, targetFPS: 45 }),
        measureRenderingPerformance(webgl2Visualizer, { duration: 3000, targetFPS: 60 })
      ]);

      expect(webgl1Benchmark.averageFPS).toBeGreaterThanOrEqual(35);
      expect(webgl2Benchmark.averageFPS).toBeGreaterThanOrEqual(50);
      expect(webgl2Benchmark.averageFPS).toBeGreaterThan(webgl1Benchmark.averageFPS);

      // WebGL2 should use more advanced features
      expect(webgl2Benchmark.features.instancedRendering).toBe(true);
      expect(webgl1Benchmark.features.instancedRendering).toBe(false);

      webgl1Visualizer.dispose();
      webgl2Visualizer.dispose();
    });

    test('should adapt to different screen resolutions and pixel ratios', async () => {
      const resolutionConfigs = [
        { width: 1920, height: 1080, pixelRatio: 1.0, name: '1080p' },
        { width: 2560, height: 1440, pixelRatio: 1.0, name: '1440p' },
        { width: 3840, height: 2160, pixelRatio: 1.0, name: '4K' },
        { width: 1920, height: 1080, pixelRatio: 2.0, name: '1080p_2x' }
      ];

      const benchmarkResults = [];

      for (const config of resolutionConfigs) {
        const visualizer = new LLMKGDataFlowVisualizer({
          container,
          resolution: { width: config.width, height: config.height },
          pixelRatio: config.pixelRatio,
          enableAdaptiveResolution: true
        });

        await loadStandardTestData(visualizer);

        const benchmark = await measureRenderingPerformance(visualizer, {
          duration: 3000,
          targetFPS: 60,
          tolerance: 0.8
        });

        benchmark.configName = config.name;
        benchmark.totalPixels = config.width * config.height * config.pixelRatio;
        benchmarkResults.push(benchmark);

        visualizer.dispose();
      }

      // Verify performance scales reasonably with resolution
      benchmarkResults.sort((a, b) => a.totalPixels - b.totalPixels);
      
      expect(benchmarkResults[0].averageFPS).toBeGreaterThan(benchmarkResults[3].averageFPS);
      expect(benchmarkResults.every(b => b.averageFPS >= 40)).toBe(true); // Minimum acceptable FPS

      // 4K should still maintain reasonable performance
      const fourKBenchmark = benchmarkResults.find(b => b.configName === '4K');
      expect(fourKBenchmark.averageFPS).toBeGreaterThanOrEqual(30);
    });
  });

  // Helper functions
  async function measureRenderingPerformance(
    visualizer: any,
    config: {
      duration: number;
      targetFPS: number;
      tolerance?: number;
      trackDegradation?: boolean;
    }
  ): Promise<any> {
    const results = {
      averageFPS: 0,
      frameTimeP95: 0,
      droppedFrames: 0,
      memoryUsage: { initial: 0, peak: 0, final: 0 },
      gpuMemoryUsage: 0,
      didCrash: false,
      systemRecovery: { successful: false },
      adaptiveOptimizations: { applied: [] },
      adaptiveDegradation: { triggered: false, qualityReductions: [] },
      features: {},
      renderTimeBreakdown: {}
    };

    const frameTimes: number[] = [];
    const memoryReadings: number[] = [];
    let frameCount = 0;
    let lastFrameTime = performance.now();
    
    results.memoryUsage.initial = process.memoryUsage().heapUsed;

    const startTime = performance.now();
    let running = true;

    // Performance measurement loop
    const measureFrame = () => {
      if (!running) return;

      const currentTime = performance.now();
      const frameTime = currentTime - lastFrameTime;
      
      frameTimes.push(frameTime);
      frameCount++;
      
      // Record memory usage periodically
      if (frameCount % 60 === 0) {
        const memoryUsage = process.memoryUsage().heapUsed;
        memoryReadings.push(memoryUsage);
        results.memoryUsage.peak = Math.max(results.memoryUsage.peak, memoryUsage);
      }

      try {
        visualizer.render();
        lastFrameTime = currentTime;
      } catch (error) {
        results.didCrash = true;
        running = false;
        return;
      }

      if (currentTime - startTime < config.duration) {
        requestAnimationFrame(measureFrame);
      } else {
        running = false;
        calculateResults();
      }
    };

    const calculateResults = () => {
      const totalTime = (performance.now() - startTime) / 1000;
      results.averageFPS = frameCount / totalTime;

      // Calculate 95th percentile frame time
      frameTimes.sort((a, b) => a - b);
      const p95Index = Math.floor(frameTimes.length * 0.95);
      results.frameTimeP95 = frameTimes[p95Index];

      // Count dropped frames (assuming 60 FPS target)
      results.droppedFrames = Math.max(0, Math.floor(totalTime * config.targetFPS) - frameCount);

      results.memoryUsage.final = process.memoryUsage().heapUsed;
      
      // Detect memory leaks
      const memoryIncrease = results.memoryUsage.final - results.memoryUsage.initial;
      results.memoryLeaks = {
        detected: memoryIncrease > 100 * 1024 * 1024, // 100MB increase
        increase: memoryIncrease
      };

      // Check system recovery
      results.systemRecovery.successful = !results.didCrash && results.averageFPS > 0;

      // Get additional metrics from visualizer if available
      if (typeof visualizer.getPerformanceMetrics === 'function') {
        const metrics = visualizer.getPerformanceMetrics();
        results.gpuMemoryUsage = metrics.gpuMemoryUsage || 0;
        results.features = metrics.features || {};
        results.adaptiveOptimizations = metrics.adaptiveOptimizations || { applied: [] };
      }

      if (config.trackDegradation && typeof visualizer.getDegradationMetrics === 'function') {
        results.adaptiveDegradation = visualizer.getDegradationMetrics();
      }
    };

    return new Promise((resolve) => {
      requestAnimationFrame(measureFrame);
      setTimeout(() => {
        if (running) {
          running = false;
          calculateResults();
        }
        resolve(results);
      }, config.duration + 1000);
    });
  }

  async function measureMultiComponentPerformance(
    visualizers: any[],
    config: { duration: number; targetFPS: number; tolerance: number }
  ): Promise<any> {
    const results = {
      combinedFPS: 0,
      renderTimeBreakdown: {},
      totalRenderTime: 0,
      componentPerformance: []
    };

    let frameCount = 0;
    const startTime = performance.now();
    const componentTimes = visualizers.map(() => [] as number[]);

    const measureCombinedFrame = () => {
      const frameStartTime = performance.now();
      
      // Render each component and measure individual times
      visualizers.forEach((visualizer, index) => {
        const componentStartTime = performance.now();
        visualizer.render();
        const componentTime = performance.now() - componentStartTime;
        componentTimes[index].push(componentTime);
      });

      const frameTime = performance.now() - frameStartTime;
      frameCount++;

      if (performance.now() - startTime < config.duration) {
        requestAnimationFrame(measureCombinedFrame);
      } else {
        // Calculate results
        const totalTime = (performance.now() - startTime) / 1000;
        results.combinedFPS = frameCount / totalTime;

        // Calculate average render times per component
        componentTimes.forEach((times, index) => {
          const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
          const componentName = ['dataFlow', 'cognitive', 'memory'][index] || `component_${index}`;
          results.renderTimeBreakdown[componentName] = avgTime;
        });

        results.totalRenderTime = Object.values(results.renderTimeBreakdown)
          .reduce((sum: number, time: number) => sum + time, 0);
      }
    };

    return new Promise((resolve) => {
      requestAnimationFrame(measureCombinedFrame);
      setTimeout(() => resolve(results), config.duration + 500);
    });
  }

  async function loadTestDataFlow(visualizer: any, streamCount: number): Promise<void> {
    const streams = Array.from({ length: streamCount }, (_, i) => ({
      id: `test_stream_${i}`,
      source: { x: (i % 5) * 2, y: 0, z: 0 },
      target: { x: ((i + 2) % 5) * 2, y: 0, z: 0 },
      flowRate: 10 + Math.random() * 20,
      particleColor: new THREE.Color().setHSL(i / streamCount, 0.8, 0.5)
    }));

    for (const stream of streams) {
      await visualizer.addDataStream(stream);
    }
  }

  async function loadTestCognitivePatterns(visualizer: any, patternCount: number): Promise<void> {
    const patterns = Array.from({ length: patternCount }, (_, i) => ({
      id: `test_pattern_${i}`,
      type: ['convergent', 'divergent', 'lateral', 'systems'][i % 4],
      nodes: Array.from({ length: 3 + Math.floor(Math.random() * 5) }, (_, j) => `node_${i}_${j}`),
      strength: Math.random()
    }));

    for (const pattern of patterns) {
      await visualizer.createPattern(pattern);
    }
  }

  async function loadTestMemoryOperations(visualizer: any, operationCount: number): Promise<void> {
    const operations = Array.from({ length: operationCount }, (_, i) => ({
      id: `test_operation_${i}`,
      type: ['encoding', 'retrieval', 'consolidation'][i % 3],
      inputData: new Uint8Array(1024),
      timestamp: Date.now() - i * 100
    }));

    for (const operation of operations) {
      await visualizer.trackOperation(operation);
    }
  }

  async function loadStandardTestData(visualizer: any): Promise<void> {
    // Standard test data for consistent benchmarking
    await loadTestDataFlow(visualizer, 25);
    
    if (typeof visualizer.createPattern === 'function') {
      await loadTestCognitivePatterns(visualizer, 10);
    }
    
    if (typeof visualizer.trackOperation === 'function') {
      await loadTestMemoryOperations(visualizer, 50);
    }
  }
});