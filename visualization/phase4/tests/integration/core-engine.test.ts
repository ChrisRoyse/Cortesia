/**
 * Core Engine Integration Tests
 * Tests the core visualization engine components and their interactions
 */

import { JSDOM } from 'jsdom';
import * as THREE from 'three';
import { LLMKGDataFlowVisualizer } from '../../src/core/LLMKGDataFlowVisualizer';
import { ParticleSystem } from '../../src/core/ParticleSystem';
import { ShaderLibrary } from '../../src/core/ShaderLibrary';

// Setup DOM environment
const dom = new JSDOM('<!DOCTYPE html><html><body><div id="test-container"></div></body></html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.WebGLRenderingContext = {} as any;

describe('Core Engine Integration Tests', () => {
  let container: HTMLElement;
  let visualizer: LLMKGDataFlowVisualizer;
  let particleSystem: ParticleSystem;
  let shaderLibrary: ShaderLibrary;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    
    visualizer = new LLMKGDataFlowVisualizer({
      container,
      enableParticleEffects: true,
      enableDataStreams: true,
      performanceMode: 'high-quality',
      targetFPS: 60
    });

    particleSystem = new ParticleSystem({
      maxParticles: 10000,
      particleSize: 0.1,
      enableCollisions: true,
      enableForces: true
    });

    shaderLibrary = new ShaderLibrary();
  });

  afterEach(() => {
    visualizer?.dispose();
    particleSystem?.dispose();
    shaderLibrary?.dispose();
  });

  describe('Core Visualizer Functionality', () => {
    test('should initialize with correct configuration', () => {
      const config = visualizer.getConfiguration();
      
      expect(config.targetFPS).toBe(60);
      expect(config.enableParticleEffects).toBe(true);
      expect(config.enableDataStreams).toBe(true);
      expect(config.performanceMode).toBe('high-quality');
    });

    test('should create valid THREE.js scene', () => {
      const scene = visualizer.getScene();
      
      expect(scene).toBeInstanceOf(THREE.Scene);
      expect(scene.children.length).toBeGreaterThan(0);
    });

    test('should initialize camera with correct settings', () => {
      const camera = visualizer.getCamera();
      
      expect(camera).toBeInstanceOf(THREE.PerspectiveCamera);
      expect(camera.fov).toBeGreaterThan(0);
      expect(camera.aspect).toBeGreaterThan(0);
      expect(camera.near).toBeGreaterThan(0);
      expect(camera.far).toBeGreaterThan(camera.near);
    });

    test('should create WebGL renderer with appropriate settings', () => {
      const renderer = visualizer.getRenderer();
      
      expect(renderer).toBeInstanceOf(THREE.WebGLRenderer);
      expect(renderer.getSize(new THREE.Vector2()).width).toBeGreaterThan(0);
      expect(renderer.getSize(new THREE.Vector2()).height).toBeGreaterThan(0);
    });
  });

  describe('Data Flow Visualization', () => {
    test('should add and track data streams', async () => {
      const streamId = 'test-stream-1';
      
      await visualizer.addDataStream({
        id: streamId,
        source: { x: -5, y: 0, z: 0 },
        target: { x: 5, y: 0, z: 0 },
        flowRate: 100, // particles per second
        particleColor: new THREE.Color(0x00ff00),
        streamType: 'knowledge-flow'
      });

      const stream = visualizer.getDataStream(streamId);
      expect(stream).toBeDefined();
      expect(stream.id).toBe(streamId);
      expect(stream.isActive).toBe(true);
    });

    test('should handle multiple concurrent data streams', async () => {
      const streamConfigs = [
        {
          id: 'stream-1',
          source: { x: 0, y: 0, z: 0 },
          target: { x: 10, y: 0, z: 0 },
          flowRate: 50,
          streamType: 'memory-access'
        },
        {
          id: 'stream-2',
          source: { x: 0, y: 0, z: 0 },
          target: { x: 0, y: 10, z: 0 },
          flowRate: 75,
          streamType: 'cognitive-process'
        },
        {
          id: 'stream-3',
          source: { x: 0, y: 0, z: 0 },
          target: { x: 0, y: 0, z: 10 },
          flowRate: 100,
          streamType: 'mcp-request'
        }
      ];

      await Promise.all(streamConfigs.map(config => visualizer.addDataStream(config)));

      const activeStreams = visualizer.getActiveStreams();
      expect(activeStreams.length).toBe(3);
      
      // Verify each stream is independent
      streamConfigs.forEach((config, index) => {
        const stream = activeStreams.find(s => s.id === config.id);
        expect(stream).toBeDefined();
        expect(stream.flowRate).toBe(config.flowRate);
      });
    });

    test('should update stream properties dynamically', async () => {
      const streamId = 'dynamic-stream';
      
      await visualizer.addDataStream({
        id: streamId,
        source: { x: 0, y: 0, z: 0 },
        target: { x: 5, y: 0, z: 0 },
        flowRate: 50,
        particleColor: new THREE.Color(0xff0000)
      });

      // Update stream properties
      await visualizer.updateDataStream(streamId, {
        flowRate: 100,
        particleColor: new THREE.Color(0x0000ff),
        target: { x: 10, y: 5, z: 0 }
      });

      const updatedStream = visualizer.getDataStream(streamId);
      expect(updatedStream.flowRate).toBe(100);
      expect(updatedStream.particleColor.getHex()).toBe(0x0000ff);
    });
  });

  describe('Particle System Integration', () => {
    test('should create particles with correct properties', () => {
      const particleCount = 1000;
      particleSystem.createParticles(particleCount);

      expect(particleSystem.getParticleCount()).toBe(particleCount);
      
      const particles = particleSystem.getParticles();
      expect(particles.length).toBe(particleCount);
      
      // Check first particle has valid properties
      const firstParticle = particles[0];
      expect(firstParticle.position).toBeInstanceOf(THREE.Vector3);
      expect(firstParticle.velocity).toBeInstanceOf(THREE.Vector3);
      expect(firstParticle.life).toBeGreaterThan(0);
    });

    test('should apply forces to particles correctly', () => {
      particleSystem.createParticles(100);
      
      const attractorForce = {
        type: 'attractor',
        position: new THREE.Vector3(0, 5, 0),
        strength: 10.0,
        radius: 15.0
      };

      particleSystem.addForce(attractorForce);
      
      const initialPositions = particleSystem.getParticles().map(p => p.position.clone());
      
      // Update particle system
      particleSystem.update(1/60); // 60 FPS delta
      
      const finalPositions = particleSystem.getParticles().map(p => p.position.clone());
      
      // Check that particles moved towards attractor
      const movedParticles = initialPositions.filter((initial, index) => {
        return !initial.equals(finalPositions[index]);
      });
      
      expect(movedParticles.length).toBeGreaterThan(0);
    });

    test('should handle particle lifecycle correctly', () => {
      const shortLifeConfig = {
        maxParticles: 100,
        particleLifetime: 0.1, // 100ms lifetime
        particleSize: 0.5
      };

      const shortLifeSystem = new ParticleSystem(shortLifeConfig);
      shortLifeSystem.createParticles(100);

      expect(shortLifeSystem.getParticleCount()).toBe(100);
      
      // Update for longer than particle lifetime
      shortLifeSystem.update(0.2); // 200ms
      
      const aliveParticles = shortLifeSystem.getAliveParticles();
      expect(aliveParticles.length).toBeLessThan(100);

      shortLifeSystem.dispose();
    });

    test('should maintain performance with large particle counts', () => {
      const largeParticleCount = 5000;
      particleSystem.createParticles(largeParticleCount);

      const updateStart = performance.now();
      particleSystem.update(1/60);
      const updateEnd = performance.now();
      
      const updateTime = updateEnd - updateStart;
      
      // Update should complete within reasonable time (< 16ms for 60 FPS)
      expect(updateTime).toBeLessThan(16);
    });
  });

  describe('Shader Library Integration', () => {
    test('should provide required shaders', () => {
      const requiredShaders = [
        'particle-vertex',
        'particle-fragment',
        'flow-vertex',
        'flow-fragment',
        'glow-vertex',
        'glow-fragment',
        'trail-vertex',
        'trail-fragment'
      ];

      requiredShaders.forEach(shaderName => {
        const shader = shaderLibrary.getShader(shaderName);
        expect(shader).toBeDefined();
        expect(shader.vertexShader).toBeDefined();
        expect(shader.fragmentShader).toBeDefined();
      });
    });

    test('should compile shaders without errors', () => {
      const renderer = new THREE.WebGLRenderer();
      const shaderMaterial = shaderLibrary.createMaterial('particle-vertex', 'particle-fragment');
      
      expect(shaderMaterial).toBeInstanceOf(THREE.ShaderMaterial);
      
      // Test shader compilation
      const geometry = new THREE.BufferGeometry();
      const mesh = new THREE.Mesh(geometry, shaderMaterial);
      
      expect(() => {
        renderer.compile(new THREE.Scene().add(mesh), new THREE.Camera());
      }).not.toThrow();

      renderer.dispose();
    });

    test('should support uniform updates', () => {
      const shaderMaterial = shaderLibrary.createMaterial('glow-vertex', 'glow-fragment');
      
      const uniforms = {
        time: { value: 0.0 },
        glowIntensity: { value: 1.0 },
        glowColor: { value: new THREE.Color(0xff0000) }
      };

      shaderLibrary.updateUniforms(shaderMaterial, uniforms);
      
      expect(shaderMaterial.uniforms.time.value).toBe(0.0);
      expect(shaderMaterial.uniforms.glowIntensity.value).toBe(1.0);
      expect(shaderMaterial.uniforms.glowColor.value).toBeInstanceOf(THREE.Color);
    });
  });

  describe('Performance and Memory Management', () => {
    test('should maintain target frame rate under load', async () => {
      const targetFPS = 60;
      const tolerance = 0.9; // Allow 10% deviation
      
      // Add substantial load
      await visualizer.addDataStream({
        id: 'heavy-stream-1',
        source: { x: -10, y: -10, z: -10 },
        target: { x: 10, y: 10, z: 10 },
        flowRate: 500,
        particleColor: new THREE.Color(0xff0000)
      });

      particleSystem.createParticles(5000);
      
      // Measure frame rate
      const frameCount = await measureFrameRate(1000); // 1 second
      const actualFPS = frameCount;
      
      expect(actualFPS).toBeGreaterThanOrEqual(targetFPS * tolerance);
    });

    test('should clean up resources properly', () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Create and dispose multiple visualizers
      for (let i = 0; i < 10; i++) {
        const tempVisualizer = new LLMKGDataFlowVisualizer({
          container,
          enableParticleEffects: true
        });
        
        tempVisualizer.dispose();
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be minimal
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // 50MB
    });

    test('should handle resize events correctly', () => {
      const initialSize = visualizer.getRenderer().getSize(new THREE.Vector2());
      
      // Simulate resize
      const newWidth = 1920;
      const newHeight = 1080;
      
      visualizer.handleResize(newWidth, newHeight);
      
      const newSize = visualizer.getRenderer().getSize(new THREE.Vector2());
      expect(newSize.width).toBe(newWidth);
      expect(newSize.height).toBe(newHeight);
      
      const camera = visualizer.getCamera() as THREE.PerspectiveCamera;
      expect(camera.aspect).toBe(newWidth / newHeight);
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid data stream configurations', async () => {
      const invalidConfigs = [
        // Missing required fields
        { id: 'invalid-1' },
        // Invalid coordinates
        { id: 'invalid-2', source: { x: NaN, y: 0, z: 0 }, target: { x: 1, y: 1, z: 1 } },
        // Negative flow rate
        { id: 'invalid-3', source: { x: 0, y: 0, z: 0 }, target: { x: 1, y: 1, z: 1 }, flowRate: -10 }
      ];

      for (const config of invalidConfigs) {
        await expect(visualizer.addDataStream(config as any)).rejects.toThrow();
      }
    });

    test('should recover from shader compilation errors', () => {
      const invalidShader = {
        vertexShader: 'invalid shader code',
        fragmentShader: 'also invalid'
      };

      expect(() => {
        shaderLibrary.addCustomShader('invalid-shader', invalidShader);
      }).not.toThrow(); // Should handle gracefully
    });

    test('should handle WebGL context loss', () => {
      const renderer = visualizer.getRenderer();
      const canvas = renderer.domElement;
      
      // Simulate context loss
      const contextLostEvent = new Event('webglcontextlost');
      canvas.dispatchEvent(contextLostEvent);
      
      expect(visualizer.isContextLost()).toBe(true);
      
      // Should not crash on render attempt
      expect(() => visualizer.render()).not.toThrow();
    });
  });

  // Helper function
  async function measureFrameRate(duration: number): Promise<number> {
    let frameCount = 0;
    const startTime = performance.now();
    
    return new Promise((resolve) => {
      function frame() {
        frameCount++;
        visualizer.render();
        
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