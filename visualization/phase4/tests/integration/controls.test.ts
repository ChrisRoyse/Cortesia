/**
 * Controls Integration Tests
 * Tests visualization controls, debug console, performance monitoring, and filtering systems
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { VisualizationControls } from '../../src/controls/VisualizationControls';
import { DebugConsole } from '../../src/controls/DebugConsole';
import { PerformanceMonitor } from '../../src/controls/PerformanceMonitor';
import { FilteringSystem } from '../../src/controls/FilteringSystem';
import { ExportTools } from '../../src/controls/ExportTools';
import { ControlsIntegration } from '../../src/controls/ControlsIntegration';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Controls Integration Tests', () => {
  let container: HTMLElement;
  let visualizationControls: VisualizationControls;
  let debugConsole: DebugConsole;
  let performanceMonitor: PerformanceMonitor;
  let filteringSystem: FilteringSystem;
  let exportTools: ExportTools;
  let controlsIntegration: ControlsIntegration;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    visualizationControls = new VisualizationControls({
      container,
      enableCameraControls: true,
      enableAnimationControls: true,
      enableLayerControls: true,
      enableColorControls: true,
      enablePhysicsControls: true
    });

    debugConsole = new DebugConsole({
      container,
      enableRealTimeLogging: true,
      enableCommandInput: true,
      enableSystemInspection: true,
      maxLogEntries: 1000
    });

    performanceMonitor = new PerformanceMonitor({
      container,
      enableFPSMonitoring: true,
      enableMemoryTracking: true,
      enableRenderTimeAnalysis: true,
      enableNetworkMonitoring: true,
      updateInterval: 1000
    });

    filteringSystem = new FilteringSystem({
      container,
      enableEntityFiltering: true,
      enableRelationshipFiltering: true,
      enableTimeRangeFiltering: true,
      enableSemanticFiltering: true
    });

    exportTools = new ExportTools({
      container,
      supportedFormats: ['png', 'jpg', 'svg', 'pdf', 'json', 'gltf'],
      enableBatchExport: true,
      enableCustomResolutions: true
    });

    controlsIntegration = new ControlsIntegration({
      container,
      components: [
        visualizationControls,
        debugConsole,
        performanceMonitor,
        filteringSystem,
        exportTools
      ]
    });
  });

  afterEach(() => {
    visualizationControls?.dispose();
    debugConsole?.dispose();
    performanceMonitor?.dispose();
    filteringSystem?.dispose();
    exportTools?.dispose();
    controlsIntegration?.dispose();
  });

  describe('Visualization Controls', () => {
    test('should provide comprehensive camera controls', async () => {
      const cameraSettings = {
        position: { x: 10, y: 5, z: 15 },
        target: { x: 0, y: 0, z: 0 },
        fov: 75,
        near: 0.1,
        far: 1000
      };

      await visualizationControls.setCameraSettings(cameraSettings);
      
      const currentSettings = await visualizationControls.getCameraSettings();
      expect(currentSettings.position).toEqual(cameraSettings.position);
      expect(currentSettings.target).toEqual(cameraSettings.target);
      expect(currentSettings.fov).toBe(75);

      // Test camera animations
      const animationConfig = {
        targetPosition: { x: -10, y: 10, z: 20 },
        duration: 2000,
        easing: 'ease-in-out'
      };

      const animation = await visualizationControls.animateCamera(animationConfig);
      expect(animation.isActive).toBe(true);
      
      // Wait for animation to complete
      await new Promise(resolve => setTimeout(resolve, 2100));
      
      const finalPosition = await visualizationControls.getCameraSettings();
      expect(finalPosition.position).toEqual(animationConfig.targetPosition);
    });

    test('should control animation playback', async () => {
      const animationData = {
        id: 'test-animation',
        duration: 5000,
        keyframes: [
          { time: 0, state: 'initial' },
          { time: 2500, state: 'middle' },
          { time: 5000, state: 'final' }
        ]
      };

      await visualizationControls.loadAnimation(animationData);

      // Test play controls
      await visualizationControls.playAnimation('test-animation');
      let playbackState = await visualizationControls.getPlaybackState('test-animation');
      expect(playbackState.isPlaying).toBe(true);
      expect(playbackState.currentTime).toBe(0);

      // Test pause
      await new Promise(resolve => setTimeout(resolve, 1000));
      await visualizationControls.pauseAnimation('test-animation');
      
      playbackState = await visualizationControls.getPlaybackState('test-animation');
      expect(playbackState.isPlaying).toBe(false);
      expect(playbackState.currentTime).toBeGreaterThan(0);

      // Test seek
      await visualizationControls.seekAnimation('test-animation', 3000);
      playbackState = await visualizationControls.getPlaybackState('test-animation');
      expect(playbackState.currentTime).toBe(3000);

      // Test speed control
      await visualizationControls.setPlaybackSpeed('test-animation', 2.0);
      playbackState = await visualizationControls.getPlaybackState('test-animation');
      expect(playbackState.speed).toBe(2.0);
    });

    test('should manage visualization layers', async () => {
      const layers = [
        { id: 'entities', name: 'Entity Nodes', visible: true, opacity: 1.0 },
        { id: 'relationships', name: 'Relationship Edges', visible: true, opacity: 0.8 },
        { id: 'labels', name: 'Node Labels', visible: false, opacity: 0.9 },
        { id: 'effects', name: 'Visual Effects', visible: true, opacity: 0.6 }
      ];

      await visualizationControls.initializeLayers(layers);

      // Test layer visibility
      await visualizationControls.setLayerVisibility('labels', true);
      const labelLayer = await visualizationControls.getLayer('labels');
      expect(labelLayer.visible).toBe(true);

      // Test layer opacity
      await visualizationControls.setLayerOpacity('effects', 0.3);
      const effectsLayer = await visualizationControls.getLayer('effects');
      expect(effectsLayer.opacity).toBe(0.3);

      // Test layer ordering
      await visualizationControls.moveLayerToFront('labels');
      const layerOrder = await visualizationControls.getLayerOrder();
      expect(layerOrder[layerOrder.length - 1]).toBe('labels');

      // Test batch layer operations
      await visualizationControls.setMultipleLayerProperties([
        { id: 'entities', visible: false },
        { id: 'relationships', opacity: 0.5 }
      ]);

      const entitiesLayer = await visualizationControls.getLayer('entities');
      const relationshipsLayer = await visualizationControls.getLayer('relationships');
      expect(entitiesLayer.visible).toBe(false);
      expect(relationshipsLayer.opacity).toBe(0.5);
    });

    test('should provide color scheme controls', async () => {
      const colorSchemes = {
        default: {
          background: new THREE.Color(0x000000),
          primaryNodes: new THREE.Color(0x3498db),
          secondaryNodes: new THREE.Color(0xe74c3c),
          edges: new THREE.Color(0x95a5a6)
        },
        highContrast: {
          background: new THREE.Color(0xffffff),
          primaryNodes: new THREE.Color(0x000000),
          secondaryNodes: new THREE.Color(0xff0000),
          edges: new THREE.Color(0x888888)
        }
      };

      await visualizationControls.loadColorSchemes(colorSchemes);

      // Test scheme switching
      await visualizationControls.applyColorScheme('highContrast');
      const currentScheme = await visualizationControls.getCurrentColorScheme();
      expect(currentScheme.name).toBe('highContrast');
      expect(currentScheme.colors.background.getHex()).toBe(0xffffff);

      // Test custom color assignment
      await visualizationControls.setNodeColor('node_1', new THREE.Color(0x00ff00));
      const nodeColor = await visualizationControls.getNodeColor('node_1');
      expect(nodeColor.getHex()).toBe(0x00ff00);

      // Test color interpolation
      const gradientColors = await visualizationControls.generateGradient({
        startColor: new THREE.Color(0xff0000),
        endColor: new THREE.Color(0x0000ff),
        steps: 10
      });
      
      expect(gradientColors.length).toBe(10);
      expect(gradientColors[0].getHex()).toBe(0xff0000);
      expect(gradientColors[9].getHex()).toBe(0x0000ff);
    });

    test('should control physics simulation parameters', async () => {
      const physicsConfig = {
        gravity: { x: 0, y: -9.8, z: 0 },
        damping: 0.95,
        springForce: 50.0,
        repulsionForce: 100.0,
        centeringForce: 0.01,
        maxVelocity: 5.0,
        enableCollisions: true,
        timeStep: 1/60
      };

      await visualizationControls.setPhysicsConfig(physicsConfig);
      
      const currentConfig = await visualizationControls.getPhysicsConfig();
      expect(currentConfig.springForce).toBe(50.0);
      expect(currentConfig.enableCollisions).toBe(true);

      // Test physics presets
      await visualizationControls.applyPhysicsPreset('orbital');
      const orbitalConfig = await visualizationControls.getPhysicsConfig();
      expect(orbitalConfig.centeringForce).toBeGreaterThan(0.01);

      await visualizationControls.applyPhysicsPreset('chaotic');
      const chaoticConfig = await visualizationControls.getPhysicsConfig();
      expect(chaoticConfig.repulsionForce).toBeGreaterThan(100.0);

      // Test dynamic physics adjustment
      await visualizationControls.adjustPhysicsForNodeCount(1000);
      const scaledConfig = await visualizationControls.getPhysicsConfig();
      expect(scaledConfig.springForce).toBeLessThan(physicsConfig.springForce); // Should scale down
    });
  });

  describe('Debug Console', () => {
    test('should log system events and provide command interface', async () => {
      // Test logging
      await debugConsole.log('info', 'System initialized');
      await debugConsole.log('warning', 'Performance degradation detected');
      await debugConsole.log('error', 'Failed to load resource');

      const logEntries = await debugConsole.getLogEntries();
      expect(logEntries.length).toBe(3);
      expect(logEntries[0].level).toBe('info');
      expect(logEntries[1].level).toBe('warning');
      expect(logEntries[2].level).toBe('error');

      // Test log filtering
      const errorLogs = await debugConsole.getLogEntries({ level: 'error' });
      expect(errorLogs.length).toBe(1);
      expect(errorLogs[0].message).toBe('Failed to load resource');

      // Test log search
      const searchResults = await debugConsole.searchLogs('performance');
      expect(searchResults.length).toBe(1);
      expect(searchResults[0].message).toContain('Performance');
    });

    test('should execute debug commands', async () => {
      const commands = [
        'status',
        'memory',
        'performance',
        'nodes.count',
        'edges.count',
        'camera.position',
        'physics.state'
      ];

      for (const command of commands) {
        const result = await debugConsole.executeCommand(command);
        expect(result).toBeDefined();
        expect(result.success).toBe(true);
        expect(result.output).toBeDefined();
      }

      // Test complex commands
      const complexResult = await debugConsole.executeCommand('query.analyze --complexity --performance');
      expect(complexResult.success).toBe(true);
      expect(complexResult.output.complexity).toBeDefined();
      expect(complexResult.output.performance).toBeDefined();

      // Test invalid command
      const invalidResult = await debugConsole.executeCommand('invalid.command');
      expect(invalidResult.success).toBe(false);
      expect(invalidResult.error).toBeDefined();
    });

    test('should provide system inspection capabilities', async () => {
      const systemState = await debugConsole.inspectSystem();
      expect(systemState).toBeDefined();
      expect(systemState.components).toBeDefined();
      expect(systemState.memory).toBeDefined();
      expect(systemState.performance).toBeDefined();

      // Inspect specific component
      const componentState = await debugConsole.inspectComponent('knowledge-graph');
      expect(componentState).toBeDefined();
      expect(componentState.isActive).toBeDefined();
      expect(componentState.nodeCount).toBeDefined();
      expect(componentState.edgeCount).toBeDefined();

      // Inspect memory usage
      const memoryState = await debugConsole.inspectMemory();
      expect(memoryState).toBeDefined();
      expect(memoryState.heapUsed).toBeGreaterThan(0);
      expect(memoryState.heapTotal).toBeGreaterThan(memoryState.heapUsed);

      // Inspect performance metrics
      const performanceState = await debugConsole.inspectPerformance();
      expect(performanceState).toBeDefined();
      expect(performanceState.fps).toBeDefined();
      expect(performanceState.frameTime).toBeDefined();
      expect(performanceState.renderTime).toBeDefined();
    });

    test('should support custom debug extensions', async () => {
      const customExtension = {
        name: 'cognitive-patterns',
        commands: [
          {
            name: 'patterns.list',
            handler: async () => ({ patterns: ['pattern1', 'pattern2'] })
          },
          {
            name: 'patterns.analyze',
            handler: async (args) => ({ analysis: `Analyzing ${args.pattern}` })
          }
        ],
        inspector: {
          name: 'patterns',
          handler: async () => ({ activePatterns: 5, totalPatterns: 20 })
        }
      };

      await debugConsole.registerExtension(customExtension);

      // Test custom commands
      const listResult = await debugConsole.executeCommand('patterns.list');
      expect(listResult.success).toBe(true);
      expect(listResult.output.patterns.length).toBe(2);

      const analyzeResult = await debugConsole.executeCommand('patterns.analyze pattern1');
      expect(analyzeResult.success).toBe(true);
      expect(analyzeResult.output.analysis).toContain('pattern1');

      // Test custom inspector
      const patternsState = await debugConsole.inspectComponent('patterns');
      expect(patternsState.activePatterns).toBe(5);
      expect(patternsState.totalPatterns).toBe(20);
    });
  });

  describe('Performance Monitor', () => {
    test('should track FPS and frame timing', async () => {
      // Simulate rendering frames
      for (let i = 0; i < 60; i++) {
        await performanceMonitor.recordFrame({
          frameNumber: i,
          timestamp: Date.now() + i * 16.67, // ~60 FPS
          renderTime: 10 + Math.random() * 5, // 10-15ms render time
          updateTime: 2 + Math.random() * 2 // 2-4ms update time
        });
      }

      const fpsMetrics = await performanceMonitor.getFPSMetrics();
      expect(fpsMetrics.currentFPS).toBeGreaterThan(55);
      expect(fpsMetrics.currentFPS).toBeLessThan(65);
      expect(fpsMetrics.averageFrameTime).toBeGreaterThan(15);
      expect(fpsMetrics.averageFrameTime).toBeLessThan(18);

      // Test performance alerts
      const alerts = await performanceMonitor.getPerformanceAlerts();
      expect(alerts.length).toBe(0); // Good performance, no alerts

      // Simulate performance degradation
      for (let i = 0; i < 10; i++) {
        await performanceMonitor.recordFrame({
          frameNumber: 60 + i,
          timestamp: Date.now() + (60 + i) * 50, // ~20 FPS
          renderTime: 40 + Math.random() * 10, // 40-50ms render time
          updateTime: 5 + Math.random() * 5 // 5-10ms update time
        });
      }

      const degradedAlerts = await performanceMonitor.getPerformanceAlerts();
      expect(degradedAlerts.length).toBeGreaterThan(0);
      expect(degradedAlerts.some(alert => alert.type === 'low_fps')).toBe(true);
    });

    test('should monitor memory usage', async () => {
      const memoryData = [
        { timestamp: Date.now() - 10000, heapUsed: 50 * 1024 * 1024, heapTotal: 100 * 1024 * 1024 },
        { timestamp: Date.now() - 8000, heapUsed: 60 * 1024 * 1024, heapTotal: 120 * 1024 * 1024 },
        { timestamp: Date.now() - 6000, heapUsed: 80 * 1024 * 1024, heapTotal: 150 * 1024 * 1024 },
        { timestamp: Date.now() - 4000, heapUsed: 90 * 1024 * 1024, heapTotal: 150 * 1024 * 1024 },
        { timestamp: Date.now() - 2000, heapUsed: 70 * 1024 * 1024, heapTotal: 150 * 1024 * 1024 }
      ];

      for (const data of memoryData) {
        await performanceMonitor.recordMemoryUsage(data);
      }

      const memoryMetrics = await performanceMonitor.getMemoryMetrics();
      expect(memoryMetrics.currentUsage).toBeDefined();
      expect(memoryMetrics.peakUsage).toBe(90 * 1024 * 1024);
      expect(memoryMetrics.averageUsage).toBeGreaterThan(0);
      expect(memoryMetrics.trend).toBeDefined(); // Should detect increasing then decreasing trend

      // Test memory leak detection
      const leakDetection = await performanceMonitor.detectMemoryLeaks();
      expect(leakDetection).toBeDefined();
      expect(leakDetection.suspiciousPatterns).toBeDefined();
      expect(leakDetection.recommendation).toBeDefined();
    });

    test('should analyze render performance', async () => {
      const renderData = [
        { component: 'knowledge-graph', renderTime: 12, triangles: 5000 },
        { component: 'particle-effects', renderTime: 8, particles: 1000 },
        { component: 'ui-overlay', renderTime: 3, elements: 50 },
        { component: 'debug-info', renderTime: 1, elements: 10 }
      ];

      await performanceMonitor.recordRenderData(renderData);

      const renderAnalysis = await performanceMonitor.getRenderAnalysis();
      expect(renderAnalysis.totalRenderTime).toBe(24);
      expect(renderAnalysis.bottlenecks[0].component).toBe('knowledge-graph');
      expect(renderAnalysis.recommendations.length).toBeGreaterThan(0);

      // Test GPU performance
      const gpuMetrics = await performanceMonitor.getGPUMetrics();
      expect(gpuMetrics).toBeDefined();
      expect(gpuMetrics.drawCalls).toBeDefined();
      expect(gpuMetrics.textureMemory).toBeDefined();
      expect(gpuMetrics.shaderCompileTime).toBeDefined();
    });

    test('should monitor network performance for data loading', async () => {
      const networkRequests = [
        { url: '/api/entities', method: 'GET', size: 1024 * 1024, duration: 500 },
        { url: '/api/relationships', method: 'GET', size: 2 * 1024 * 1024, duration: 800 },
        { url: '/api/query/execute', method: 'POST', size: 512 * 1024, duration: 1200 },
        { url: '/api/patterns', method: 'GET', size: 256 * 1024, duration: 300 }
      ];

      for (const request of networkRequests) {
        await performanceMonitor.recordNetworkRequest(request);
      }

      const networkMetrics = await performanceMonitor.getNetworkMetrics();
      expect(networkMetrics.totalRequests).toBe(4);
      expect(networkMetrics.totalDataTransferred).toBe(3.75 * 1024 * 1024);
      expect(networkMetrics.averageLatency).toBeGreaterThan(0);
      expect(networkMetrics.slowestRequest.url).toBe('/api/query/execute');

      // Test caching analysis
      const cachingAnalysis = await performanceMonitor.analyzeCachingEfficiency();
      expect(cachingAnalysis).toBeDefined();
      expect(cachingAnalysis.cacheableRequests).toBeDefined();
      expect(cachingAnalysis.potentialSavings).toBeDefined();
    });
  });

  describe('Filtering System', () => {
    test('should filter entities by type and properties', async () => {
      const entities = [
        { id: 'p1', type: 'Person', name: 'Alice', age: 30, department: 'Engineering' },
        { id: 'p2', type: 'Person', name: 'Bob', age: 25, department: 'Marketing' },
        { id: 'c1', type: 'Company', name: 'TechCorp', industry: 'Technology', size: 1000 },
        { id: 'c2', type: 'Company', name: 'HealthCorp', industry: 'Healthcare', size: 500 },
        { id: 'pr1', type: 'Project', name: 'AI Initiative', status: 'active', budget: 1000000 }
      ];

      await filteringSystem.loadEntities(entities);

      // Test type filtering
      const persons = await filteringSystem.filterByType(['Person']);
      expect(persons.length).toBe(2);
      expect(persons.every(e => e.type === 'Person')).toBe(true);

      // Test property filtering
      const engineers = await filteringSystem.filterByProperty('department', 'Engineering');
      expect(engineers.length).toBe(1);
      expect(engineers[0].id).toBe('p1');

      // Test range filtering
      const youngPeople = await filteringSystem.filterByPropertyRange('age', 20, 28);
      expect(youngPeople.length).toBe(1);
      expect(youngPeople[0].age).toBe(25);

      // Test complex filtering
      const complexFilter = await filteringSystem.applyComplexFilter({
        and: [
          { type: ['Person', 'Company'] },
          { or: [
            { property: 'department', value: 'Engineering' },
            { property: 'industry', value: 'Technology' }
          ]}
        ]
      });

      expect(complexFilter.length).toBe(2); // Alice and TechCorp
    });

    test('should filter relationships by type and strength', async () => {
      const relationships = [
        { id: 'r1', source: 'p1', target: 'c1', type: 'WORKS_FOR', strength: 0.9 },
        { id: 'r2', source: 'p2', target: 'c1', type: 'WORKS_FOR', strength: 0.8 },
        { id: 'r3', source: 'p1', target: 'p2', type: 'COLLABORATES_WITH', strength: 0.6 },
        { id: 'r4', source: 'c1', target: 'c2', type: 'COMPETES_WITH', strength: 0.4 },
        { id: 'r5', source: 'p1', target: 'pr1', type: 'LEADS', strength: 1.0 }
      ];

      await filteringSystem.loadRelationships(relationships);

      // Test relationship type filtering
      const workRelations = await filteringSystem.filterRelationshipsByType(['WORKS_FOR']);
      expect(workRelations.length).toBe(2);

      // Test strength filtering
      const strongRelations = await filteringSystem.filterRelationshipsByStrength(0.8, 1.0);
      expect(strongRelations.length).toBe(3); // r1, r2, r5

      // Test source/target filtering
      const p1Relations = await filteringSystem.filterRelationshipsByNode('p1');
      expect(p1Relations.length).toBe(3); // r1, r3, r5

      // Test bidirectional filtering
      const collaborativeRelations = await filteringSystem.filterBidirectionalRelationships(['COLLABORATES_WITH']);
      expect(collaborativeRelations.length).toBeGreaterThan(0);
    });

    test('should provide time-range filtering', async () => {
      const timeBasedData = [
        { id: 'e1', timestamp: Date.now() - 86400000, type: 'creation' }, // 1 day ago
        { id: 'e2', timestamp: Date.now() - 43200000, type: 'update' },   // 12 hours ago
        { id: 'e3', timestamp: Date.now() - 21600000, type: 'access' },   // 6 hours ago
        { id: 'e4', timestamp: Date.now() - 3600000, type: 'modification' }, // 1 hour ago
        { id: 'e5', timestamp: Date.now() - 1800000, type: 'view' }       // 30 minutes ago
      ];

      await filteringSystem.loadTimeBasedData(timeBasedData);

      // Filter last 4 hours
      const recentData = await filteringSystem.filterByTimeRange(
        Date.now() - 4 * 3600000, // 4 hours ago
        Date.now()
      );
      expect(recentData.length).toBe(2); // e4 and e5

      // Filter by time periods
      const lastHourData = await filteringSystem.filterByTimePeriod('last_hour');
      expect(lastHourData.length).toBe(2);

      const todayData = await filteringSystem.filterByTimePeriod('today');
      expect(todayData.length).toBe(5);

      // Test temporal patterns
      const patterns = await filteringSystem.analyzeTemporalPatterns();
      expect(patterns).toBeDefined();
      expect(patterns.activityDistribution).toBeDefined();
      expect(patterns.peakActivityPeriods).toBeDefined();
    });

    test('should support semantic filtering', async () => {
      const semanticData = [
        { id: 's1', content: 'artificial intelligence machine learning', category: 'technology' },
        { id: 's2', content: 'natural language processing deep learning', category: 'ai' },
        { id: 's3', content: 'data visualization user interface', category: 'ui' },
        { id: 's4', content: 'knowledge graph semantic web', category: 'knowledge' },
        { id: 's5', content: 'neural networks pattern recognition', category: 'ai' }
      ];

      await filteringSystem.loadSemanticData(semanticData);

      // Test keyword filtering
      const aiContent = await filteringSystem.filterByKeywords(['artificial', 'intelligence']);
      expect(aiContent.length).toBe(1);

      // Test semantic search
      const semanticResults = await filteringSystem.semanticSearch('machine learning algorithms');
      expect(semanticResults.length).toBeGreaterThan(0);
      expect(semanticResults[0].relevanceScore).toBeDefined();

      // Test category clustering
      const clusters = await filteringSystem.clusterBySemantics();
      expect(clusters.length).toBeGreaterThan(1);
      expect(clusters.find(c => c.category === 'ai').items.length).toBe(2);

      // Test similarity filtering
      const similarItems = await filteringSystem.findSimilarItems('s2', 0.5);
      expect(similarItems.length).toBeGreaterThan(0);
      expect(similarItems[0].similarity).toBeGreaterThan(0.5);
    });
  });

  describe('Export Tools', () => {
    test('should export visualizations in multiple formats', async () => {
      const visualizationData = {
        scene: new THREE.Scene(),
        camera: new THREE.PerspectiveCamera(75, 1, 0.1, 1000),
        renderer: new THREE.WebGLRenderer(),
        metadata: {
          title: 'Test Visualization',
          description: 'A test knowledge graph visualization',
          nodeCount: 100,
          edgeCount: 200
        }
      };

      // Test PNG export
      const pngExport = await exportTools.exportToPNG({
        data: visualizationData,
        resolution: { width: 1920, height: 1080 },
        quality: 0.9
      });
      expect(pngExport.success).toBe(true);
      expect(pngExport.data).toBeDefined();
      expect(pngExport.size).toBeGreaterThan(0);

      // Test SVG export  
      const svgExport = await exportTools.exportToSVG({
        data: visualizationData,
        includeLabels: true,
        scalable: true
      });
      expect(svgExport.success).toBe(true);
      expect(svgExport.data).toContain('<svg');

      // Test JSON export
      const jsonExport = await exportTools.exportToJSON({
        data: visualizationData,
        includeMetadata: true,
        compact: false
      });
      expect(jsonExport.success).toBe(true);
      expect(JSON.parse(jsonExport.data).metadata).toBeDefined();

      // Test 3D model export (glTF)
      const gltfExport = await exportTools.exportToGLTF({
        data: visualizationData,
        includeAnimations: true,
        compression: 'draco'
      });
      expect(gltfExport.success).toBe(true);
      expect(gltfExport.data.asset).toBeDefined();
    });

    test('should support batch export operations', async () => {
      const batchData = [
        { id: 'viz1', name: 'Network Overview', data: { nodes: 50, edges: 100 } },
        { id: 'viz2', name: 'Query Results', data: { nodes: 20, edges: 30 } },
        { id: 'viz3', name: 'Pattern Analysis', data: { nodes: 80, edges: 150 } }
      ];

      const batchExportConfig = {
        format: 'png',
        resolution: { width: 1024, height: 768 },
        includeMetadata: true,
        compressionLevel: 0.8
      };

      const batchResult = await exportTools.batchExport(batchData, batchExportConfig);
      expect(batchResult.success).toBe(true);
      expect(batchResult.exports.length).toBe(3);
      expect(batchResult.exports.every(exp => exp.success)).toBe(true);

      // Test batch progress tracking
      const progressCallback = jest.fn();
      await exportTools.batchExport(batchData, batchExportConfig, progressCallback);
      expect(progressCallback).toHaveBeenCalledTimes(3);
    });

    test('should provide export customization options', async () => {
      const customExportOptions = {
        template: 'research_paper',
        style: {
          background: '#ffffff',
          nodeColor: '#3498db',
          edgeColor: '#95a5a6',
          labelFont: 'Arial, 12px',
          showGrid: true,
          showLegend: true
        },
        annotations: [
          { type: 'title', text: 'Knowledge Graph Analysis', position: 'top' },
          { type: 'caption', text: 'Generated by LLMKG Phase 4', position: 'bottom' },
          { type: 'watermark', text: 'Confidential', opacity: 0.1 }
        ]
      };

      const customExport = await exportTools.exportWithCustomization({
        data: { nodes: [], edges: [] },
        format: 'pdf',
        options: customExportOptions
      });

      expect(customExport.success).toBe(true);
      expect(customExport.customizationsApplied).toEqual(expect.arrayContaining([
        'template', 'style', 'annotations'
      ]));

      // Test template library
      const availableTemplates = await exportTools.getAvailableTemplates();
      expect(availableTemplates.length).toBeGreaterThan(0);
      expect(availableTemplates.find(t => t.name === 'research_paper')).toBeDefined();
    });

    test('should handle high-resolution exports efficiently', async () => {
      const highResConfig = {
        resolution: { width: 4096, height: 4096 },
        dpi: 300,
        format: 'png',
        tileRendering: true, // For memory efficiency
        progressiveEncoding: true
      };

      const largeData = {
        nodes: Array.from({ length: 1000 }, (_, i) => ({ id: `node_${i}` })),
        edges: Array.from({ length: 2000 }, (_, i) => ({ id: `edge_${i}` }))
      };

      const startTime = performance.now();
      const highResExport = await exportTools.exportHighResolution(largeData, highResConfig);
      const endTime = performance.now();

      expect(highResExport.success).toBe(true);
      expect(endTime - startTime).toBeLessThan(30000); // Should complete within 30 seconds
      expect(highResExport.memoryPeak).toBeLessThan(512 * 1024 * 1024); // Less than 512MB peak memory
    });
  });

  describe('Controls Integration', () => {
    test('should coordinate between all control components', async () => {
      const testData = {
        nodes: Array.from({ length: 100 }, (_, i) => ({
          id: `node_${i}`,
          type: ['Person', 'Company', 'Project'][i % 3],
          timestamp: Date.now() - Math.random() * 86400000
        })),
        edges: Array.from({ length: 200 }, (_, i) => ({
          id: `edge_${i}`,
          source: `node_${i % 100}`,
          target: `node_${(i + 1) % 100}`,
          strength: Math.random()
        }))
      };

      await controlsIntegration.loadData(testData);

      // Test coordinated operations
      const coordinatedAction = await controlsIntegration.executeCoordinatedAction({
        action: 'filter_and_export',
        params: {
          filter: { type: ['Person'] },
          export: { format: 'svg', resolution: '1024x768' }
        }
      });

      expect(coordinatedAction.success).toBe(true);
      expect(coordinatedAction.componentsInvolved).toContain('filtering');
      expect(coordinatedAction.componentsInvolved).toContain('export');

      // Test state synchronization
      await controlsIntegration.synchronizeStates();
      const syncedStates = await controlsIntegration.getComponentStates();
      expect(syncedStates.filtering.activeFilters).toBeDefined();
      expect(syncedStates.performance.currentMetrics).toBeDefined();
      expect(syncedStates.debug.logCount).toBeGreaterThan(0);
    });

    test('should provide unified user interface', async () => {
      const uiConfig = {
        layout: 'dashboard',
        panels: [
          { component: 'visualization-controls', position: 'left', size: '300px' },
          { component: 'performance-monitor', position: 'top-right', size: '400px' },
          { component: 'debug-console', position: 'bottom', size: '200px' },
          { component: 'filtering-system', position: 'right', size: '250px' }
        ],
        theme: 'dark',
        responsive: true
      };

      const ui = await controlsIntegration.createUnifiedUI(uiConfig);
      expect(ui.success).toBe(true);
      expect(ui.layout.panels.length).toBe(4);
      expect(ui.eventHandlers).toBeDefined();

      // Test UI interactions
      const interactionResult = await controlsIntegration.handleUIInteraction({
        component: 'filtering-system',
        action: 'apply-filter',
        params: { type: 'Person' }
      });

      expect(interactionResult.success).toBe(true);
      expect(interactionResult.updatedComponents).toContain('visualization-controls');
    });

    test('should handle complex workflow automation', async () => {
      const workflow = {
        id: 'analysis-workflow',
        name: 'Complete Knowledge Graph Analysis',
        steps: [
          {
            id: 'load-data',
            component: 'data-loader',
            action: 'load',
            params: { source: 'test-dataset' }
          },
          {
            id: 'apply-filters',
            component: 'filtering-system',
            action: 'filter',
            params: { entityTypes: ['Person', 'Company'] },
            dependsOn: ['load-data']
          },
          {
            id: 'start-monitoring',
            component: 'performance-monitor',
            action: 'start',
            params: { metrics: ['fps', 'memory'] },
            dependsOn: ['load-data']
          },
          {
            id: 'visualize',
            component: 'visualization-controls',
            action: 'render',
            params: { layout: '3d-force' },
            dependsOn: ['apply-filters']
          },
          {
            id: 'export-results',
            component: 'export-tools',
            action: 'export',
            params: { format: 'png', resolution: '1920x1080' },
            dependsOn: ['visualize', 'start-monitoring']
          }
        ]
      };

      const workflowResult = await controlsIntegration.executeWorkflow(workflow);
      expect(workflowResult.success).toBe(true);
      expect(workflowResult.completedSteps.length).toBe(5);
      expect(workflowResult.results.get('export-results')).toBeDefined();

      // Test workflow status tracking
      const workflowStatus = await controlsIntegration.getWorkflowStatus('analysis-workflow');
      expect(workflowStatus.status).toBe('completed');
      expect(workflowStatus.progress).toBe(1.0);
    });

    test('should provide comprehensive system health monitoring', async () => {
      // Simulate system activity
      await new Promise(resolve => setTimeout(resolve, 2000));

      const healthReport = await controlsIntegration.generateHealthReport();
      expect(healthReport).toBeDefined();
      expect(healthReport.overall.status).toMatch(/^(healthy|warning|critical)$/);
      expect(healthReport.components.length).toBeGreaterThan(0);

      // Check individual component health
      const componentHealth = healthReport.components.find(c => c.name === 'performance-monitor');
      expect(componentHealth).toBeDefined();
      expect(componentHealth.metrics.responseTime).toBeDefined();
      expect(componentHealth.metrics.memoryUsage).toBeDefined();

      // Test automated health alerts
      const alerts = await controlsIntegration.getHealthAlerts();
      expect(Array.isArray(alerts)).toBe(true);

      // Test health trend analysis
      const trends = await controlsIntegration.analyzeHealthTrends({
        timeWindow: 300000, // 5 minutes
        metrics: ['performance', 'memory', 'errors']
      });
      expect(trends).toBeDefined();
      expect(trends.performance.trend).toBeDefined();
      expect(trends.memory.trend).toBeDefined();
    });
  });
});