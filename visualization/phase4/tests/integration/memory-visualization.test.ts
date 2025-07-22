/**
 * Memory Visualization Integration Tests
 * Tests memory system visualization and SDR (Sparse Distributed Representation) components
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { MemoryOperationVisualizer } from '../../src/memory/MemoryOperationVisualizer';
import { SDRVisualizer } from '../../src/memory/SDRVisualizer';
import { MemoryAnalytics } from '../../src/memory/MemoryAnalytics';
import { StorageEfficiency } from '../../src/memory/StorageEfficiency';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Memory Visualization Integration Tests', () => {
  let container: HTMLElement;
  let memoryVisualizer: MemoryOperationVisualizer;
  let sdrVisualizer: SDRVisualizer;
  let memoryAnalytics: MemoryAnalytics;
  let storageEfficiency: StorageEfficiency;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    memoryVisualizer = new MemoryOperationVisualizer({
      container,
      enableSDRVisualization: true,
      enableMemoryAnalytics: true,
      realTimeUpdates: true,
      maxOperationHistory: 1000
    });

    sdrVisualizer = new SDRVisualizer({
      container,
      dimensions: 2048,
      sparsityLevel: 0.02, // 2% active bits
      enableBitPatternVisualization: true,
      enableSimilarityVisualization: true,
      enable3DVisualization: true
    });

    memoryAnalytics = new MemoryAnalytics({
      enableRealTimeAnalysis: true,
      retentionPeriod: 3600000, // 1 hour
      trackMemoryPatterns: true,
      analyzeEfficiency: true
    });

    storageEfficiency = new StorageEfficiency({
      enableCompressionAnalysis: true,
      enableAccessPatternAnalysis: true,
      enableGarbageCollectionVisualization: true,
      optimizationTarget: 'balanced' // balance between space and speed
    });
  });

  afterEach(() => {
    memoryVisualizer?.dispose();
    sdrVisualizer?.dispose();
    memoryAnalytics?.dispose();
    storageEfficiency?.dispose();
  });

  describe('Memory Operation Tracking', () => {
    test('should track and visualize memory operations', async () => {
      const operationId = 'test-memory-op-1';
      const operationData = {
        id: operationId,
        type: 'encoding',
        inputData: new Uint8Array(1024).fill(0).map(() => Math.random() > 0.5 ? 1 : 0),
        timestamp: Date.now(),
        metadata: {
          sparsity: 0.02,
          dimensions: 2048,
          patternId: 'test-pattern-1',
          confidence: 0.85
        }
      };

      await memoryVisualizer.trackOperation(operationData);

      const operation = await memoryVisualizer.getOperation(operationId);
      expect(operation).toBeDefined();
      expect(operation.id).toBe(operationId);
      expect(operation.type).toBe('encoding');
      expect(operation.status).toBe('completed');
      expect(operation.metrics).toBeDefined();
      expect(operation.metrics.processingTime).toBeGreaterThan(0);
    });

    test('should handle different memory operation types', async () => {
      const operationTypes = [
        {
          type: 'encoding',
          data: new Uint8Array(512).fill(0).map(() => Math.random() > 0.5 ? 1 : 0),
          expectedResult: 'sdr_pattern'
        },
        {
          type: 'retrieval',
          data: { queryPattern: new Uint8Array(512), threshold: 0.8 },
          expectedResult: 'matched_patterns'
        },
        {
          type: 'consolidation',
          data: { patternIds: ['pattern_1', 'pattern_2'], strength: 0.9 },
          expectedResult: 'consolidated_pattern'
        },
        {
          type: 'forgetting',
          data: { patternId: 'old_pattern', decayRate: 0.1 },
          expectedResult: 'pattern_weakened'
        }
      ];

      const operations = [];
      for (let i = 0; i < operationTypes.length; i++) {
        const operation = {
          id: `test-operation-${i}`,
          type: operationTypes[i].type as any,
          inputData: operationTypes[i].data,
          timestamp: Date.now() + i * 100,
          metadata: { operationIndex: i }
        };

        await memoryVisualizer.trackOperation(operation);
        operations.push(operation);
      }

      // Verify all operations were tracked
      const trackedOperations = await memoryVisualizer.getAllOperations();
      expect(trackedOperations.length).toBe(operationTypes.length);

      // Verify each operation type was handled correctly
      for (let i = 0; i < operationTypes.length; i++) {
        const operation = trackedOperations.find(op => op.id === `test-operation-${i}`);
        expect(operation).toBeDefined();
        expect(operation.type).toBe(operationTypes[i].type);
        expect(operation.result).toBeDefined();
      }
    });

    test('should maintain operation history', async () => {
      const operationCount = 100;
      const operations = [];

      // Create many operations
      for (let i = 0; i < operationCount; i++) {
        const operation = {
          id: `history-operation-${i}`,
          type: ['encoding', 'retrieval', 'consolidation', 'forgetting'][i % 4] as any,
          inputData: new Uint8Array(256),
          timestamp: Date.now() - ((operationCount - i) * 1000), // Spread over time
          metadata: { sequenceNumber: i }
        };

        await memoryVisualizer.trackOperation(operation);
        operations.push(operation);
      }

      const history = await memoryVisualizer.getOperationHistory({
        timeRange: { start: Date.now() - (operationCount * 1000), end: Date.now() },
        operationType: 'all',
        sortBy: 'timestamp'
      });

      expect(history.length).toBe(operationCount);
      
      // Verify chronological order
      for (let i = 1; i < history.length; i++) {
        expect(history[i].timestamp).toBeGreaterThanOrEqual(history[i - 1].timestamp);
      }
    });
  });

  describe('SDR Visualization', () => {
    test('should visualize sparse distributed representations', async () => {
      const dimensions = 2048;
      const sparsity = 0.02;
      const activeBits = Math.floor(dimensions * sparsity);

      // Create an SDR pattern
      const sdrPattern = new Uint8Array(dimensions);
      const activeBitIndices = new Set();
      while (activeBitIndices.size < activeBits) {
        activeBitIndices.add(Math.floor(Math.random() * dimensions));
      }
      activeBitIndices.forEach(index => sdrPattern[index] = 1);

      const visualizationId = 'test-sdr-viz-1';
      await sdrVisualizer.visualizeSDR({
        id: visualizationId,
        pattern: sdrPattern,
        dimensions,
        sparsity,
        metadata: {
          conceptName: 'test-concept',
          strength: 0.9,
          timestamp: Date.now()
        }
      });

      const visualization = await sdrVisualizer.getVisualization(visualizationId);
      expect(visualization).toBeDefined();
      expect(visualization.dimensions).toBe(dimensions);
      expect(visualization.activeBits).toBe(activeBits);
      expect(visualization.sparsity).toBeCloseTo(sparsity, 3);
      expect(visualization.visualElements.activeBitPositions.length).toBe(activeBits);
    });

    test('should calculate and visualize SDR similarity', async () => {
      const dimensions = 1024;
      const sparsity = 0.02;

      // Create two similar SDR patterns
      const basePattern = new Uint8Array(dimensions);
      const similarPattern = new Uint8Array(dimensions);
      const dissimilarPattern = new Uint8Array(dimensions);

      // Generate base pattern
      const activeBitIndices = [];
      for (let i = 0; i < Math.floor(dimensions * sparsity); i++) {
        let index;
        do {
          index = Math.floor(Math.random() * dimensions);
        } while (activeBitIndices.includes(index));
        activeBitIndices.push(index);
        basePattern[index] = 1;
      }

      // Create similar pattern (80% overlap)
      const overlapBits = Math.floor(activeBitIndices.length * 0.8);
      for (let i = 0; i < overlapBits; i++) {
        similarPattern[activeBitIndices[i]] = 1;
      }
      // Add some unique bits
      for (let i = overlapBits; i < activeBitIndices.length; i++) {
        let newIndex;
        do {
          newIndex = Math.floor(Math.random() * dimensions);
        } while (basePattern[newIndex] === 1 || similarPattern[newIndex] === 1);
        similarPattern[newIndex] = 1;
      }

      // Create dissimilar pattern (minimal overlap)
      const dissimilarActiveBits = [];
      for (let i = 0; i < Math.floor(dimensions * sparsity); i++) {
        let index;
        do {
          index = Math.floor(Math.random() * dimensions);
        } while (basePattern[index] === 1 || dissimilarActiveBits.includes(index));
        dissimilarActiveBits.push(index);
        dissimilarPattern[index] = 1;
      }

      await sdrVisualizer.visualizeSDR({ id: 'base', pattern: basePattern, dimensions, sparsity });
      await sdrVisualizer.visualizeSDR({ id: 'similar', pattern: similarPattern, dimensions, sparsity });
      await sdrVisualizer.visualizeSDR({ id: 'dissimilar', pattern: dissimilarPattern, dimensions, sparsity });

      // Calculate similarities
      const similaritySimilar = await sdrVisualizer.calculateSimilarity('base', 'similar');
      const similarityDissimilar = await sdrVisualizer.calculateSimilarity('base', 'dissimilar');

      expect(similaritySimilar).toBeGreaterThan(0.7); // High similarity
      expect(similarityDissimilar).toBeLessThan(0.2); // Low similarity
      expect(similaritySimilar).toBeGreaterThan(similarityDissimilar);

      // Visualize similarity comparison
      const comparisonViz = await sdrVisualizer.visualizeSimilarityComparison({
        referenceId: 'base',
        comparisonIds: ['similar', 'dissimilar'],
        visualizationType: 'heatmap'
      });

      expect(comparisonViz).toBeDefined();
      expect(comparisonViz.similarities.length).toBe(2);
      expect(comparisonViz.similarities[0].similarity).toBeGreaterThan(comparisonViz.similarities[1].similarity);
    });

    test('should handle 3D SDR visualization', async () => {
      const dimensions = 4096;
      const sparsity = 0.015;

      // Create complex SDR pattern
      const sdrPattern = new Uint8Array(dimensions);
      for (let i = 0; i < dimensions; i++) {
        if (Math.random() < sparsity) {
          sdrPattern[i] = 1;
        }
      }

      const viz3D = await sdrVisualizer.create3DVisualization({
        id: '3d-sdr-viz-1',
        pattern: sdrPattern,
        dimensions,
        layoutStrategy: 'spherical', // or 'cubic', 'cylindrical'
        colorMapping: 'activity-based',
        enableInteraction: true
      });

      expect(viz3D).toBeDefined();
      expect(viz3D.scene).toBeInstanceOf(THREE.Scene);
      expect(viz3D.camera).toBeInstanceOf(THREE.PerspectiveCamera);
      expect(viz3D.renderer).toBeInstanceOf(THREE.WebGLRenderer);
      expect(viz3D.activeBitMeshes.length).toBeGreaterThan(0);
      expect(viz3D.layoutStrategy).toBe('spherical');
    });

    test('should animate SDR pattern changes', async () => {
      const dimensions = 1024;
      const pattern1 = new Uint8Array(dimensions);
      const pattern2 = new Uint8Array(dimensions);

      // Create two different patterns
      for (let i = 0; i < dimensions; i++) {
        pattern1[i] = Math.random() < 0.02 ? 1 : 0;
        pattern2[i] = Math.random() < 0.02 ? 1 : 0;
      }

      const animationId = 'sdr-animation-1';
      const animation = await sdrVisualizer.animatePatternTransition({
        id: animationId,
        fromPattern: pattern1,
        toPattern: pattern2,
        duration: 2000,
        animationType: 'morph', // or 'fade', 'slide'
        easing: 'ease-in-out'
      });

      expect(animation).toBeDefined();
      expect(animation.isActive).toBe(true);
      expect(animation.progress).toBe(0);

      // Simulate animation progress
      await new Promise(resolve => setTimeout(resolve, 1000)); // Half duration
      
      const midAnimation = await sdrVisualizer.getAnimation(animationId);
      expect(midAnimation.progress).toBeGreaterThan(0);
      expect(midAnimation.progress).toBeLessThan(1);

      // Wait for completion
      await new Promise(resolve => setTimeout(resolve, 1100));
      
      const completedAnimation = await sdrVisualizer.getAnimation(animationId);
      expect(completedAnimation.progress).toBe(1);
      expect(completedAnimation.isComplete).toBe(true);
    });
  });

  describe('Memory Analytics', () => {
    test('should analyze memory usage patterns', async () => {
      // Generate diverse memory operations
      const operations = [];
      const operationTypes = ['encoding', 'retrieval', 'consolidation', 'forgetting'];
      
      for (let i = 0; i < 200; i++) {
        const operation = {
          id: `analytics-op-${i}`,
          type: operationTypes[i % 4] as any,
          timestamp: Date.now() - (Math.random() * 3600000), // Last hour
          inputSize: Math.floor(Math.random() * 4096) + 256,
          outputSize: Math.floor(Math.random() * 2048) + 128,
          processingTime: Math.random() * 100 + 10, // 10-110ms
          memoryUsage: Math.floor(Math.random() * 1024 * 1024) + 512 * 1024, // 512KB - 1.5MB
          success: Math.random() > 0.05 // 95% success rate
        };
        operations.push(operation);
      }

      await memoryAnalytics.analyzeOperations(operations);

      const analysis = await memoryAnalytics.getAnalysis();
      expect(analysis).toBeDefined();
      expect(analysis.totalOperations).toBe(200);
      expect(analysis.operationsByType).toBeDefined();
      expect(analysis.averageProcessingTime).toBeGreaterThan(0);
      expect(analysis.averageMemoryUsage).toBeGreaterThan(0);
      expect(analysis.successRate).toBeGreaterThan(0.9);

      // Check operation type breakdown
      expect(analysis.operationsByType.encoding).toBe(50);
      expect(analysis.operationsByType.retrieval).toBe(50);
      expect(analysis.operationsByType.consolidation).toBe(50);
      expect(analysis.operationsByType.forgetting).toBe(50);
    });

    test('should detect memory patterns and anomalies', async () => {
      // Create operations with some anomalous patterns
      const normalOps = Array.from({ length: 150 }, (_, i) => ({
        id: `normal-op-${i}`,
        type: 'encoding' as any,
        processingTime: 50 + Math.random() * 20, // 50-70ms normal
        memoryUsage: 1024 * 1024 + Math.random() * 512 * 1024, // 1-1.5MB normal
        timestamp: Date.now() - i * 1000
      }));

      const anomalousOps = Array.from({ length: 20 }, (_, i) => ({
        id: `anomaly-op-${i}`,
        type: 'encoding' as any,
        processingTime: 500 + Math.random() * 200, // 500-700ms anomalous
        memoryUsage: 10 * 1024 * 1024 + Math.random() * 5 * 1024 * 1024, // 10-15MB anomalous
        timestamp: Date.now() - (150 + i) * 1000
      }));

      await memoryAnalytics.analyzeOperations([...normalOps, ...anomalousOps]);

      const patterns = await memoryAnalytics.detectPatterns();
      expect(patterns).toBeDefined();
      expect(patterns.trends).toBeDefined();
      expect(patterns.anomalies.length).toBeGreaterThan(0);
      expect(patterns.correlations).toBeDefined();

      const processingTimeAnomalies = patterns.anomalies.filter(a => a.type === 'processing_time');
      const memoryUsageAnomalies = patterns.anomalies.filter(a => a.type === 'memory_usage');
      
      expect(processingTimeAnomalies.length).toBeGreaterThan(0);
      expect(memoryUsageAnomalies.length).toBeGreaterThan(0);
    });

    test('should provide optimization recommendations', async () => {
      const inefficientOperations = Array.from({ length: 100 }, (_, i) => ({
        id: `inefficient-op-${i}`,
        type: 'retrieval' as any,
        processingTime: 200 + Math.random() * 100, // Slow
        memoryUsage: 5 * 1024 * 1024, // High memory usage
        cacheHitRate: 0.1, // Poor cache performance
        timestamp: Date.now() - i * 1000
      }));

      await memoryAnalytics.analyzeOperations(inefficientOperations);

      const recommendations = await memoryAnalytics.getOptimizationRecommendations();
      expect(recommendations).toBeDefined();
      expect(recommendations.length).toBeGreaterThan(0);

      const cacheRecommendation = recommendations.find(r => r.type === 'cache_optimization');
      expect(cacheRecommendation).toBeDefined();
      expect(cacheRecommendation.priority).toBe('high');
      expect(cacheRecommendation.potentialImpact).toBeGreaterThan(0.5);

      const memoryRecommendation = recommendations.find(r => r.type === 'memory_optimization');
      expect(memoryRecommendation).toBeDefined();
    });
  });

  describe('Storage Efficiency', () => {
    test('should analyze storage compression efficiency', async () => {
      const storageData = {
        originalSize: 10 * 1024 * 1024, // 10MB
        compressedSize: 3 * 1024 * 1024, // 3MB
        compressionAlgorithm: 'lz4',
        compressionTime: 150, // ms
        decompressionTime: 50, // ms
        accessFrequency: 0.8, // Frequently accessed
        dataType: 'sdr_patterns'
      };

      await storageEfficiency.analyzeCompression(storageData);

      const efficiency = await storageEfficiency.getCompressionEfficiency();
      expect(efficiency).toBeDefined();
      expect(efficiency.compressionRatio).toBeCloseTo(0.3, 1); // ~70% compression
      expect(efficiency.spaceReduction).toBeCloseTo(0.7, 1);
      expect(efficiency.timeOverhead).toBeDefined();
      expect(efficiency.recommendation).toBeDefined();

      // Should recommend keeping compression for frequently accessed, well-compressing data
      expect(efficiency.recommendation.action).toBe('maintain_compression');
      expect(efficiency.recommendation.confidence).toBeGreaterThan(0.7);
    });

    test('should track access patterns', async () => {
      const accessLog = Array.from({ length: 500 }, (_, i) => ({
        id: `access-${i}`,
        dataId: `data_${Math.floor(i / 50)}`, // 10 different data items, 50 accesses each
        timestamp: Date.now() - (Math.random() * 3600000), // Last hour
        accessType: ['read', 'write', 'update'][Math.floor(Math.random() * 3)],
        size: Math.floor(Math.random() * 1024) + 256,
        latency: Math.random() * 50 + 5 // 5-55ms
      }));

      await storageEfficiency.trackAccessPatterns(accessLog);

      const patterns = await storageEfficiency.getAccessPatterns();
      expect(patterns).toBeDefined();
      expect(patterns.hotData.length).toBeGreaterThan(0); // Should identify frequently accessed data
      expect(patterns.coldData.length).toBeGreaterThan(0); // Should identify infrequently accessed data
      expect(patterns.temporalPatterns).toBeDefined();
      expect(patterns.spatialPatterns).toBeDefined();

      // Check pattern accuracy
      const totalAccesses = patterns.hotData.reduce((sum, item) => sum + item.accessCount, 0) +
                           patterns.coldData.reduce((sum, item) => sum + item.accessCount, 0);
      expect(totalAccesses).toBe(500);
    });

    test('should optimize storage layout', async () => {
      const storageLayout = {
        segments: [
          { id: 'segment_1', size: 2 * 1024 * 1024, accessFrequency: 0.9, dataAge: 86400 }, // Recent, hot
          { id: 'segment_2', size: 5 * 1024 * 1024, accessFrequency: 0.1, dataAge: 604800 }, // Old, cold
          { id: 'segment_3', size: 1 * 1024 * 1024, accessFrequency: 0.5, dataAge: 172800 }, // Medium
        ],
        totalSize: 8 * 1024 * 1024,
        fragmentation: 0.15, // 15% fragmentation
        compressionRatio: 0.4 // 60% compression
      };

      const optimization = await storageEfficiency.optimizeLayout(storageLayout);
      expect(optimization).toBeDefined();
      expect(optimization.recommendations.length).toBeGreaterThan(0);
      expect(optimization.projectedSavings).toBeGreaterThan(0);
      expect(optimization.riskAssessment).toBeDefined();

      // Should recommend moving hot data to faster storage tier
      const tierRecommendation = optimization.recommendations.find(r => r.type === 'storage_tiering');
      expect(tierRecommendation).toBeDefined();
      expect(tierRecommendation.affectedSegments).toContain('segment_1');

      // Should recommend defragmentation
      const defragRecommendation = optimization.recommendations.find(r => r.type === 'defragmentation');
      expect(defragRecommendation).toBeDefined();
    });

    test('should visualize garbage collection', async () => {
      const gcData = {
        collections: Array.from({ length: 20 }, (_, i) => ({
          id: `gc-${i}`,
          timestamp: Date.now() - (20 - i) * 60000, // Every minute
          type: i % 5 === 0 ? 'full' : 'incremental',
          duration: i % 5 === 0 ? 100 + Math.random() * 200 : 10 + Math.random() * 40,
          memoryFreed: Math.floor(Math.random() * 10 * 1024 * 1024), // 0-10MB
          totalMemory: 100 * 1024 * 1024 + Math.random() * 50 * 1024 * 1024 // 100-150MB
        })),
        currentMemoryUsage: 120 * 1024 * 1024,
        memoryPressure: 0.7 // 70% pressure
      };

      const gcVisualization = await storageEfficiency.visualizeGarbageCollection(gcData);
      expect(gcVisualization).toBeDefined();
      expect(gcVisualization.timeline.length).toBe(20);
      expect(gcVisualization.memoryTrend).toBeDefined();
      expect(gcVisualization.efficiencyMetrics.averageCollectionTime).toBeGreaterThan(0);
      expect(gcVisualization.recommendations).toBeDefined();

      // Should recommend optimization for high memory pressure
      expect(gcVisualization.recommendations.length).toBeGreaterThan(0);
      const pressureRecommendation = gcVisualization.recommendations.find(r => r.type === 'memory_pressure_reduction');
      expect(pressureRecommendation).toBeDefined();
    });
  });

  describe('Integration and Performance', () => {
    test('should handle concurrent memory operations efficiently', async () => {
      const concurrentOperations = 100;
      const operationPromises = [];

      for (let i = 0; i < concurrentOperations; i++) {
        const operation = memoryVisualizer.trackOperation({
          id: `concurrent-op-${i}`,
          type: ['encoding', 'retrieval'][i % 2] as any,
          inputData: new Uint8Array(1024),
          timestamp: Date.now() + i,
          metadata: { batchId: 'concurrent-batch-1' }
        });
        operationPromises.push(operation);
      }

      const startTime = performance.now();
      await Promise.all(operationPromises);
      const endTime = performance.now();

      const totalTime = endTime - startTime;
      expect(totalTime).toBeLessThan(5000); // Should complete within 5 seconds

      const completedOperations = await memoryVisualizer.getOperationsByBatch('concurrent-batch-1');
      expect(completedOperations.length).toBe(concurrentOperations);
    });

    test('should maintain visualization performance with large datasets', async () => {
      const largeDataset = {
        sdrPatterns: Array.from({ length: 100 }, (_, i) => ({
          id: `large-pattern-${i}`,
          pattern: new Uint8Array(4096).fill(0).map(() => Math.random() < 0.02 ? 1 : 0),
          dimensions: 4096,
          sparsity: 0.02
        })),
        memoryOperations: Array.from({ length: 1000 }, (_, i) => ({
          id: `large-op-${i}`,
          type: 'encoding' as any,
          timestamp: Date.now() - i * 100,
          processingTime: Math.random() * 100
        }))
      };

      // Load large dataset
      for (const pattern of largeDataset.sdrPatterns) {
        await sdrVisualizer.visualizeSDR(pattern);
      }

      await memoryAnalytics.analyzeOperations(largeDataset.memoryOperations);

      // Measure rendering performance
      const renderStart = performance.now();
      memoryVisualizer.render();
      sdrVisualizer.render();
      const renderEnd = performance.now();

      const renderTime = renderEnd - renderStart;
      expect(renderTime).toBeLessThan(100); // Should render within 100ms

      // Measure memory usage
      const memoryUsage = process.memoryUsage();
      expect(memoryUsage.heapUsed).toBeLessThan(500 * 1024 * 1024); // Less than 500MB
    });

    test('should integrate with LLMKG memory subsystem', async () => {
      // Mock LLMKG memory subsystem data
      const llmkgMemoryData = {
        cognitivePatterns: [
          {
            id: 'pattern_1',
            type: 'hierarchical_processing',
            sdrRepresentation: new Uint8Array(2048).fill(0).map(() => Math.random() < 0.02 ? 1 : 0),
            activationLevel: 0.85,
            lastAccessed: Date.now() - 60000
          },
          {
            id: 'pattern_2',
            type: 'lateral_thinking',
            sdrRepresentation: new Uint8Array(2048).fill(0).map(() => Math.random() < 0.02 ? 1 : 0),
            activationLevel: 0.60,
            lastAccessed: Date.now() - 300000
          }
        ],
        memoryConsolidation: {
          lastConsolidation: Date.now() - 3600000,
          patternsConsolidated: 25,
          strengthenedConnections: 150,
          weakenedConnections: 30
        },
        storageMetrics: {
          totalPatterns: 1000,
          activePatterns: 150,
          storageUsed: 50 * 1024 * 1024, // 50MB
          compressionRatio: 0.35
        }
      };

      const integration = await memoryVisualizer.integrateLLMKGData(llmkgMemoryData);
      expect(integration).toBeDefined();
      expect(integration.visualizedPatterns.length).toBe(2);
      expect(integration.consolidationVisualization).toBeDefined();
      expect(integration.storageVisualization).toBeDefined();

      // Check pattern visualization
      const pattern1Viz = integration.visualizedPatterns.find(p => p.id === 'pattern_1');
      expect(pattern1Viz).toBeDefined();
      expect(pattern1Viz.activationVisualization.intensity).toBeCloseTo(0.85, 2);

      // Check consolidation visualization
      expect(integration.consolidationVisualization.strengthenedConnections).toBe(150);
      expect(integration.consolidationVisualization.weakenedConnections).toBe(30);

      // Check storage visualization
      expect(integration.storageVisualization.efficiency).toBeDefined();
      expect(integration.storageVisualization.utilizationRate).toBeCloseTo(0.15, 2); // 150/1000
    });
  });
});