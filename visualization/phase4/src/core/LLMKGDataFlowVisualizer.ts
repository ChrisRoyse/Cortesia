/**
 * Main visualization engine for LLMKG Data Flow using Three.js
 * Orchestrates particle systems, shaders, and real-time data visualization
 */

import * as THREE from 'three';
import { ParticleSystem, ParticleConfig } from './ParticleSystem';
import { ShaderLibrary } from './ShaderLibrary';

export interface VisualizationConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  backgroundColor: number;
  cameraPosition: THREE.Vector3;
  targetFPS: number;
  enablePostProcessing: boolean;
  maxNodes: number;
  maxConnections: number;
}

export interface DataFlowNode {
  id: string;
  position: THREE.Vector3;
  type: 'input' | 'processing' | 'output' | 'cognitive';
  activation: number;
  connections: string[];
  metadata?: Record<string, any>;
}

export interface DataFlowConnection {
  id: string;
  source: string;
  target: string;
  strength: number;
  dataType: string;
  isActive: boolean;
}

export interface CognitivePattern {
  id: string;
  center: THREE.Vector3;
  complexity: number;
  strength: number;
  type: 'attention' | 'memory' | 'processing' | 'inhibition';
  nodes: string[];
}

export class LLMKGDataFlowVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private config: VisualizationConfig;
  
  private particleSystem: ParticleSystem;
  private shaderLibrary: ShaderLibrary;
  
  private nodes: Map<string, DataFlowNode> = new Map();
  private connections: Map<string, DataFlowConnection> = new Map();
  private cognitivePatterns: Map<string, CognitivePattern> = new Map();
  
  private nodeGeometry: THREE.SphereGeometry;
  private connectionGeometry: THREE.BufferGeometry;
  private nodeInstances: THREE.InstancedMesh;
  private connectionLines: THREE.LineSegments;
  
  private clock: THREE.Clock;
  private animationId: number | null = null;
  private isInitialized: boolean = false;
  private frameCount: number = 0;
  private lastFPSCheck: number = 0;
  private currentFPS: number = 0;

  constructor(config: VisualizationConfig) {
    this.config = { ...config };
    this.clock = new THREE.Clock();
    this.shaderLibrary = ShaderLibrary.getInstance();
    
    this.initializeRenderer();
    this.initializeScene();
    this.initializeCamera();
    this.initializeParticleSystem();
    this.initializeNodeSystem();
    this.setupEventListeners();
    
    this.isInitialized = true;
  }

  private initializeRenderer(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.config.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    
    this.renderer.setSize(this.config.width, this.config.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(this.config.backgroundColor, 1.0);
    this.renderer.shadowMap.enabled = false; // Disable for performance
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    
    // Optimize for particle rendering
    this.renderer.sortObjects = false;
  }

  private initializeScene(): void {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(this.config.backgroundColor, 50, 200);
    
    // Add ambient lighting for basic illumination
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    this.scene.add(ambientLight);
    
    // Add directional light for depth
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(10, 10, 5);
    this.scene.add(directionalLight);
  }

  private initializeCamera(): void {
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.config.width / this.config.height,
      0.1,
      1000
    );
    
    this.camera.position.copy(this.config.cameraPosition);
    this.camera.lookAt(0, 0, 0);
  }

  private initializeParticleSystem(): void {
    const particleConfig: ParticleConfig = {
      maxParticles: 10000,
      particleSize: 2.0,
      lifespan: 5.0,
      spawnRate: 50,
      initialVelocity: new THREE.Vector3(0, 1, 0),
      gravity: new THREE.Vector3(0, -0.1, 0),
      colorStart: new THREE.Color(0x00ffff),
      colorEnd: new THREE.Color(0xff4400),
      shaderName: 'dataFlow'
    };
    
    this.particleSystem = new ParticleSystem(particleConfig);
    this.scene.add(this.particleSystem.getPoints());
  }

  private initializeNodeSystem(): void {
    // Create geometry for nodes
    this.nodeGeometry = new THREE.SphereGeometry(0.5, 16, 12);
    
    // Create instanced mesh for efficient node rendering
    const nodeMaterial = this.shaderLibrary.createMaterial('neuralNetwork');
    if (nodeMaterial) {
      this.nodeInstances = new THREE.InstancedMesh(
        this.nodeGeometry,
        nodeMaterial,
        this.config.maxNodes
      );
      this.scene.add(this.nodeInstances);
    }
    
    // Initialize connection geometry
    this.connectionGeometry = new THREE.BufferGeometry();
    const connectionMaterial = new THREE.LineBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.6
    });
    
    this.connectionLines = new THREE.LineSegments(
      this.connectionGeometry,
      connectionMaterial
    );
    this.scene.add(this.connectionLines);
  }

  private setupEventListeners(): void {
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  private handleResize(): void {
    const width = this.config.canvas.clientWidth;
    const height = this.config.canvas.clientHeight;
    
    this.config.width = width;
    this.config.height = height;
    
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    
    this.renderer.setSize(width, height);
  }

  public addNode(node: DataFlowNode): void {
    this.nodes.set(node.id, node);
    this.updateNodeVisuals();
  }

  public removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.updateNodeVisuals();
  }

  public updateNode(nodeId: string, updates: Partial<DataFlowNode>): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      Object.assign(node, updates);
      this.updateNodeVisuals();
    }
  }

  public addConnection(connection: DataFlowConnection): void {
    this.connections.set(connection.id, connection);
    this.updateConnectionVisuals();
    
    // Spawn particles along the connection if active
    if (connection.isActive) {
      this.spawnConnectionParticles(connection);
    }
  }

  public removeConnection(connectionId: string): void {
    this.connections.delete(connectionId);
    this.updateConnectionVisuals();
  }

  public addCognitivePattern(pattern: CognitivePattern): void {
    this.cognitivePatterns.set(pattern.id, pattern);
    this.visualizeCognitivePattern(pattern);
  }

  public removeCognitivePattern(patternId: string): void {
    this.cognitivePatterns.delete(patternId);
  }

  private updateNodeVisuals(): void {
    if (!this.nodeInstances) return;
    
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    let instanceIndex = 0;
    
    for (const [nodeId, node] of this.nodes) {
      if (instanceIndex >= this.config.maxNodes) break;
      
      // Set position and scale based on activation
      const scale = 1.0 + node.activation * 0.5;
      matrix.makeScale(scale, scale, scale);
      matrix.setPosition(node.position);
      
      this.nodeInstances.setMatrixAt(instanceIndex, matrix);
      
      // Set color based on node type and activation
      switch (node.type) {
        case 'input':
          color.setHSL(0.3, 0.8, 0.4 + node.activation * 0.4);
          break;
        case 'processing':
          color.setHSL(0.6, 0.8, 0.4 + node.activation * 0.4);
          break;
        case 'output':
          color.setHSL(0.1, 0.8, 0.4 + node.activation * 0.4);
          break;
        case 'cognitive':
          color.setHSL(0.8, 0.8, 0.4 + node.activation * 0.4);
          break;
      }
      
      this.nodeInstances.setColorAt(instanceIndex, color);
      instanceIndex++;
    }
    
    this.nodeInstances.instanceMatrix.needsUpdate = true;
    if (this.nodeInstances.instanceColor) {
      this.nodeInstances.instanceColor.needsUpdate = true;
    }
    this.nodeInstances.count = instanceIndex;
  }

  private updateConnectionVisuals(): void {
    const positions: number[] = [];
    const colors: number[] = [];
    
    for (const [connectionId, connection] of this.connections) {
      const sourceNode = this.nodes.get(connection.source);
      const targetNode = this.nodes.get(connection.target);
      
      if (!sourceNode || !targetNode) continue;
      
      positions.push(
        sourceNode.position.x, sourceNode.position.y, sourceNode.position.z,
        targetNode.position.x, targetNode.position.y, targetNode.position.z
      );
      
      const intensity = connection.isActive ? connection.strength : 0.2;
      colors.push(
        0.3, 0.5 * intensity, 1.0 * intensity,
        0.3, 0.5 * intensity, 1.0 * intensity
      );
    }
    
    this.connectionGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3)
    );
    this.connectionGeometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(colors, 3)
    );
  }

  private spawnConnectionParticles(connection: DataFlowConnection): void {
    const sourceNode = this.nodes.get(connection.source);
    const targetNode = this.nodes.get(connection.target);
    
    if (!sourceNode || !targetNode) return;
    
    const direction = targetNode.position.clone().sub(sourceNode.position);
    const distance = direction.length();
    direction.normalize();
    
    // Spawn particles along the connection
    const particleCount = Math.ceil(distance * connection.strength * 2);
    for (let i = 0; i < particleCount; i++) {
      const t = i / (particleCount - 1);
      const position = sourceNode.position.clone().lerp(targetNode.position, t);
      const velocity = direction.clone().multiplyScalar(2 + Math.random());
      
      this.particleSystem.spawnParticle(
        position,
        velocity,
        connection.id,
        connection.strength
      );
    }
  }

  private visualizeCognitivePattern(pattern: CognitivePattern): void {
    // Create a burst of particles representing the cognitive pattern
    const particleCount = Math.floor(pattern.complexity * pattern.strength * 20);
    this.particleSystem.spawnDataFlowBurst(
      pattern.center,
      particleCount,
      pattern.complexity,
      pattern.id
    );
  }

  public animate(): void {
    if (!this.isInitialized) return;
    
    this.animationId = requestAnimationFrame(() => this.animate());
    
    const deltaTime = this.clock.getDelta();
    const elapsedTime = this.clock.getElapsedTime();
    
    // Update FPS tracking
    this.trackFPS(elapsedTime);
    
    // Update all systems
    this.update(deltaTime, elapsedTime);
    
    // Render the scene
    this.render();
  }

  private update(deltaTime: number, elapsedTime: number): void {
    // Update shader library time
    this.shaderLibrary.updateAllShadersTime(elapsedTime);
    
    // Update particle system
    this.particleSystem.update(deltaTime);
    
    // Update cognitive patterns (create periodic bursts)
    this.updateCognitivePatterns(elapsedTime);
    
    // Update camera controls if needed
    this.updateCamera(deltaTime);
  }

  private updateCognitivePatterns(elapsedTime: number): void {
    for (const [patternId, pattern] of this.cognitivePatterns) {
      // Create periodic visual effects for cognitive patterns
      if (Math.sin(elapsedTime * pattern.strength * 2) > 0.9) {
        this.visualizeCognitivePattern(pattern);
      }
    }
  }

  private updateCamera(deltaTime: number): void {
    // Simple camera orbit for demonstration
    const radius = this.config.cameraPosition.length();
    const angle = this.clock.getElapsedTime() * 0.1;
    
    this.camera.position.x = Math.cos(angle) * radius;
    this.camera.position.z = Math.sin(angle) * radius;
    this.camera.lookAt(0, 0, 0);
  }

  private render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  private trackFPS(elapsedTime: number): void {
    this.frameCount++;
    
    if (elapsedTime - this.lastFPSCheck >= 1.0) {
      this.currentFPS = this.frameCount;
      this.frameCount = 0;
      this.lastFPSCheck = elapsedTime;
    }
  }

  public start(): void {
    if (!this.isInitialized) return;
    this.animate();
  }

  public stop(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  public resize(width: number, height: number): void {
    this.config.width = width;
    this.config.height = height;
    this.handleResize();
  }

  public getPerformanceMetrics() {
    return {
      fps: this.currentFPS,
      targetFPS: this.config.targetFPS,
      nodeCount: this.nodes.size,
      connectionCount: this.connections.size,
      patternCount: this.cognitivePatterns.size,
      particles: this.particleSystem.getPerformanceMetrics(),
      renderer: {
        triangles: this.renderer.info.render.triangles,
        calls: this.renderer.info.render.calls,
        points: this.renderer.info.render.points
      }
    };
  }

  public dispose(): void {
    this.stop();
    
    // Dispose particle system
    this.particleSystem.dispose();
    
    // Dispose geometries
    this.nodeGeometry.dispose();
    this.connectionGeometry.dispose();
    
    // Dispose renderer
    this.renderer.dispose();
    
    // Clear collections
    this.nodes.clear();
    this.connections.clear();
    this.cognitivePatterns.clear();
    
    // Remove event listeners
    window.removeEventListener('resize', this.handleResize.bind(this));
    
    this.isInitialized = false;
  }
}

export default LLMKGDataFlowVisualizer;