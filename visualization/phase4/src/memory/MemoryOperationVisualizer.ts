/**
 * Memory Operation Visualizer for Phase 4
 * Animates memory operations (read, write, update, delete) with visual feedback
 */

import * as THREE from 'three';
import { ShaderLibrary } from '../core/ShaderLibrary';
import { ParticleSystem, ParticleConfig } from '../core/ParticleSystem';

export interface MemoryOperation {
  id: string;
  type: 'read' | 'write' | 'update' | 'delete';
  entityId: string;
  address: number;
  size: number;
  timestamp: number;
  duration?: number;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface MemoryBlock {
  address: number;
  size: number;
  type: 'allocated' | 'free' | 'reserved';
  entityId?: string;
  lastAccessed: number;
  accessCount: number;
}

export interface MemoryVisualizationConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  memorySize: number;
  blockHeight: number;
  animationDuration: number;
  colorScheme: {
    read: THREE.Color;
    write: THREE.Color;
    update: THREE.Color;
    delete: THREE.Color;
    allocated: THREE.Color;
    free: THREE.Color;
    reserved: THREE.Color;
  };
}

export interface MemoryStats {
  totalMemory: number;
  allocatedMemory: number;
  freeMemory: number;
  fragmentation: number;
  operationsPerSecond: number;
  averageOperationTime: number;
}

export class MemoryOperationVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private config: MemoryVisualizationConfig;
  private shaderLibrary: ShaderLibrary;
  private particleSystem: ParticleSystem;

  // Memory visualization
  private memoryBlocks: Map<number, MemoryBlock> = new Map();
  private activeOperations: Map<string, MemoryOperation> = new Map();
  private operationHistory: MemoryOperation[] = [];
  
  // Visual components
  private memoryBarGeometry: THREE.PlaneGeometry;
  private memoryBarMaterial: THREE.ShaderMaterial;
  private memoryBarMesh: THREE.Mesh;
  
  private operationGeometry: THREE.SphereGeometry;
  private operationInstances: THREE.InstancedMesh;
  private operationMaterial: THREE.ShaderMaterial;
  
  // Animation system
  private animationClock: THREE.Clock;
  private operationAnimations: Map<string, {
    startTime: number;
    endTime: number;
    startPosition: THREE.Vector3;
    endPosition: THREE.Vector3;
    operation: MemoryOperation;
  }> = new Map();
  
  // Performance tracking
  private stats: MemoryStats = {
    totalMemory: 0,
    allocatedMemory: 0,
    freeMemory: 0,
    fragmentation: 0,
    operationsPerSecond: 0,
    averageOperationTime: 0
  };
  
  private performanceWindow: number[] = [];
  private lastStatsUpdate: number = 0;

  constructor(config: MemoryVisualizationConfig) {
    this.config = { ...config };
    this.shaderLibrary = ShaderLibrary.getInstance();
    this.animationClock = new THREE.Clock();

    this.initializeRenderer();
    this.initializeScene();
    this.initializeCamera();
    this.initializeParticleSystem();
    this.initializeMemoryVisualization();
    this.initializeOperationVisualization();
    
    this.stats.totalMemory = config.memorySize;
    this.stats.freeMemory = config.memorySize;
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
    this.renderer.setClearColor(0x0f0f1f, 1.0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  private initializeScene(): void {
    this.scene = new THREE.Scene();
    
    // Add ambient lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    this.scene.add(ambientLight);
    
    // Add directional light for depth
    const directionalLight = new THREE.DirectionalLight(0x8888ff, 0.6);
    directionalLight.position.set(10, 10, 5);
    this.scene.add(directionalLight);
  }

  private initializeCamera(): void {
    const aspect = this.config.width / this.config.height;
    const viewSize = 50;
    
    this.camera = new THREE.OrthographicCamera(
      -viewSize * aspect,
      viewSize * aspect,
      viewSize,
      -viewSize,
      1,
      1000
    );
    
    this.camera.position.set(0, 0, 100);
    this.camera.lookAt(0, 0, 0);
  }

  private initializeParticleSystem(): void {
    const particleConfig: ParticleConfig = {
      maxParticles: 5000,
      particleSize: 1.0,
      lifespan: 2.0,
      spawnRate: 0,
      initialVelocity: new THREE.Vector3(0, 0, 0),
      gravity: new THREE.Vector3(0, 0, 0),
      colorStart: new THREE.Color(0x00ffff),
      colorEnd: new THREE.Color(0xff4400),
      shaderName: 'memoryOperation'
    };
    
    this.particleSystem = new ParticleSystem(particleConfig);
    this.scene.add(this.particleSystem.getPoints());
  }

  private initializeMemoryVisualization(): void {
    // Create memory bar geometry
    this.memoryBarGeometry = new THREE.PlaneGeometry(80, this.config.blockHeight);
    
    // Create shader material for memory visualization
    this.memoryBarMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        memorySize: { value: this.config.memorySize },
        blockHeight: { value: this.config.blockHeight },
        allocatedColor: { value: this.config.colorScheme.allocated },
        freeColor: { value: this.config.colorScheme.free },
        reservedColor: { value: this.config.colorScheme.reserved },
        memoryData: { value: new Float32Array(1000) }, // Memory block data
        operationIntensity: { value: 0.0 }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;
        
        void main() {
          vUv = uv;
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float memorySize;
        uniform float blockHeight;
        uniform vec3 allocatedColor;
        uniform vec3 freeColor;
        uniform vec3 reservedColor;
        uniform float memoryData[1000];
        uniform float operationIntensity;
        
        varying vec2 vUv;
        varying vec3 vPosition;
        
        void main() {
          float address = vUv.x * memorySize;
          int blockIndex = int(address / 1024.0); // 1KB blocks
          
          // Sample memory state from data array
          float blockState = memoryData[min(blockIndex, 999)];
          
          vec3 baseColor;
          if (blockState < 0.33) {
            baseColor = freeColor;
          } else if (blockState < 0.66) {
            baseColor = allocatedColor;
          } else {
            baseColor = reservedColor;
          }
          
          // Add operation intensity effect
          vec3 operationGlow = vec3(1.0, 0.5, 0.0) * operationIntensity;
          baseColor += operationGlow * smoothstep(0.0, 0.1, operationIntensity);
          
          // Add subtle animation
          float wave = sin(time * 2.0 + vUv.x * 10.0) * 0.05 + 1.0;
          baseColor *= wave;
          
          gl_FragColor = vec4(baseColor, 1.0);
        }
      `
    });
    
    this.memoryBarMesh = new THREE.Mesh(this.memoryBarGeometry, this.memoryBarMaterial);
    this.memoryBarMesh.position.set(0, 0, 0);
    this.scene.add(this.memoryBarMesh);
  }

  private initializeOperationVisualization(): void {
    // Create operation geometry (small spheres)
    this.operationGeometry = new THREE.SphereGeometry(1, 8, 6);
    
    // Create operation material
    this.operationMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        readColor: { value: this.config.colorScheme.read },
        writeColor: { value: this.config.colorScheme.write },
        updateColor: { value: this.config.colorScheme.update },
        deleteColor: { value: this.config.colorScheme.delete }
      },
      vertexShader: `
        attribute float operationType;
        attribute float operationProgress;
        attribute vec3 operationColor;
        
        varying float vOperationType;
        varying float vOperationProgress;
        varying vec3 vOperationColor;
        varying vec3 vPosition;
        
        uniform float time;
        
        void main() {
          vOperationType = operationType;
          vOperationProgress = operationProgress;
          vOperationColor = operationColor;
          
          vec4 worldPosition = instanceMatrix * vec4(position, 1.0);
          vPosition = worldPosition.xyz;
          
          // Animate based on operation progress
          float scale = 0.5 + operationProgress * 0.8;
          worldPosition.xyz *= scale;
          
          gl_Position = projectionMatrix * modelViewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 readColor;
        uniform vec3 writeColor;
        uniform vec3 updateColor;
        uniform vec3 deleteColor;
        
        varying float vOperationType;
        varying float vOperationProgress;
        varying vec3 vOperationColor;
        varying vec3 vPosition;
        
        void main() {
          vec3 color;
          
          // Select color based on operation type
          if (vOperationType < 0.25) {
            color = readColor;
          } else if (vOperationType < 0.5) {
            color = writeColor;
          } else if (vOperationType < 0.75) {
            color = updateColor;
          } else {
            color = deleteColor;
          }
          
          // Apply progress-based effects
          float intensity = vOperationProgress * 2.0;
          color *= intensity;
          
          // Add glow effect
          float glow = 1.0 - smoothstep(0.0, 1.0, length(gl_PointCoord - 0.5) * 2.0);
          color += vOperationColor * glow * 0.3;
          
          float alpha = vOperationProgress * glow;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending
    });
    
    // Create instanced mesh for operations
    this.operationInstances = new THREE.InstancedMesh(
      this.operationGeometry,
      this.operationMaterial,
      500 // Max concurrent operations
    );
    
    this.operationInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.operationInstances);
  }

  public addMemoryBlock(block: MemoryBlock): void {
    this.memoryBlocks.set(block.address, block);
    this.updateMemoryVisualization();
    this.updateStats();
  }

  public removeMemoryBlock(address: number): void {
    this.memoryBlocks.delete(address);
    this.updateMemoryVisualization();
    this.updateStats();
  }

  public startOperation(operation: MemoryOperation): void {
    this.activeOperations.set(operation.id, operation);
    
    // Calculate positions for animation
    const startPos = this.getMemoryPosition(operation.address);
    const endPos = startPos.clone().add(new THREE.Vector3(0, 10, 0));
    
    // Create animation entry
    this.operationAnimations.set(operation.id, {
      startTime: this.animationClock.getElapsedTime(),
      endTime: this.animationClock.getElapsedTime() + this.config.animationDuration,
      startPosition: startPos,
      endPosition: endPos,
      operation
    });
    
    // Spawn particles for the operation
    this.spawnOperationParticles(operation, startPos);
    
    this.updateOperationVisualization();
  }

  public completeOperation(operationId: string, success: boolean): void {
    const operation = this.activeOperations.get(operationId);
    if (!operation) return;
    
    operation.success = success;
    operation.duration = Date.now() - operation.timestamp;
    
    // Move to history
    this.operationHistory.push(operation);
    if (this.operationHistory.length > 1000) {
      this.operationHistory.shift();
    }
    
    // Clean up
    this.activeOperations.delete(operationId);
    this.operationAnimations.delete(operationId);
    
    // Update memory blocks based on operation
    this.applyOperationToMemory(operation);
    
    this.updateOperationVisualization();
    this.updatePerformanceMetrics();
  }

  private getMemoryPosition(address: number): THREE.Vector3 {
    // Convert memory address to visual position
    const normalizedAddress = address / this.config.memorySize;
    const x = (normalizedAddress - 0.5) * 80; // Memory bar width
    const y = 0;
    const z = 0;
    
    return new THREE.Vector3(x, y, z);
  }

  private spawnOperationParticles(operation: MemoryOperation, position: THREE.Vector3): void {
    const color = this.getOperationColor(operation.type);
    const particleCount = Math.ceil(operation.size / 1024) + 5; // More particles for larger operations
    
    for (let i = 0; i < particleCount; i++) {
      const velocity = new THREE.Vector3(
        (Math.random() - 0.5) * 2,
        Math.random() * 3 + 1,
        (Math.random() - 0.5) * 2
      );
      
      this.particleSystem.spawnParticle(
        position.clone().add(new THREE.Vector3(
          (Math.random() - 0.5) * 2,
          0,
          (Math.random() - 0.5) * 2
        )),
        velocity,
        operation.id,
        1.0
      );
    }
  }

  private getOperationColor(type: MemoryOperation['type']): THREE.Color {
    switch (type) {
      case 'read': return this.config.colorScheme.read;
      case 'write': return this.config.colorScheme.write;
      case 'update': return this.config.colorScheme.update;
      case 'delete': return this.config.colorScheme.delete;
      default: return new THREE.Color(0xffffff);
    }
  }

  private applyOperationToMemory(operation: MemoryOperation): void {
    const block = this.memoryBlocks.get(operation.address);
    if (!block) return;
    
    block.lastAccessed = operation.timestamp;
    block.accessCount++;
    
    switch (operation.type) {
      case 'write':
        block.type = 'allocated';
        block.entityId = operation.entityId;
        break;
      case 'delete':
        block.type = 'free';
        block.entityId = undefined;
        break;
      case 'update':
        if (block.type === 'allocated') {
          block.lastAccessed = operation.timestamp;
        }
        break;
    }
    
    this.updateMemoryVisualization();
  }

  private updateMemoryVisualization(): void {
    // Update memory data for shader
    const memoryData = new Float32Array(1000);
    
    for (let i = 0; i < 1000; i++) {
      const address = i * 1024; // 1KB blocks
      const block = this.findBlockAtAddress(address);
      
      if (block) {
        switch (block.type) {
          case 'free': memoryData[i] = 0.0; break;
          case 'allocated': memoryData[i] = 0.5; break;
          case 'reserved': memoryData[i] = 1.0; break;
        }
      } else {
        memoryData[i] = 0.0; // Free by default
      }
    }
    
    this.memoryBarMaterial.uniforms.memoryData.value = memoryData;
    
    // Update operation intensity based on current activity
    const operationIntensity = this.activeOperations.size / 10.0; // Scale factor
    this.memoryBarMaterial.uniforms.operationIntensity.value = operationIntensity;
  }

  private findBlockAtAddress(address: number): MemoryBlock | undefined {
    for (const [blockAddr, block] of this.memoryBlocks) {
      if (address >= blockAddr && address < blockAddr + block.size) {
        return block;
      }
    }
    return undefined;
  }

  private updateOperationVisualization(): void {
    let instanceIndex = 0;
    const currentTime = this.animationClock.getElapsedTime();
    
    for (const [operationId, animation] of this.operationAnimations) {
      if (instanceIndex >= this.operationInstances.count) break;
      
      const progress = Math.min(
        (currentTime - animation.startTime) / (animation.endTime - animation.startTime),
        1.0
      );
      
      // Interpolate position
      const position = animation.startPosition.clone().lerp(animation.endPosition, progress);
      
      // Create transformation matrix
      const matrix = new THREE.Matrix4();
      matrix.makeTranslation(position.x, position.y, position.z);
      
      this.operationInstances.setMatrixAt(instanceIndex, matrix);
      
      // Set color based on operation type
      const color = this.getOperationColor(animation.operation.type);
      this.operationInstances.setColorAt(instanceIndex, color);
      
      instanceIndex++;
      
      // Remove completed animations
      if (progress >= 1.0) {
        this.operationAnimations.delete(operationId);
      }
    }
    
    this.operationInstances.count = instanceIndex;
    this.operationInstances.instanceMatrix.needsUpdate = true;
    if (this.operationInstances.instanceColor) {
      this.operationInstances.instanceColor.needsUpdate = true;
    }
  }

  private updateStats(): void {
    let allocatedMemory = 0;
    let fragmentation = 0;
    let freeBlocks = 0;
    
    for (const [_, block] of this.memoryBlocks) {
      if (block.type === 'allocated') {
        allocatedMemory += block.size;
      } else if (block.type === 'free') {
        freeBlocks++;
      }
    }
    
    this.stats.allocatedMemory = allocatedMemory;
    this.stats.freeMemory = this.config.memorySize - allocatedMemory;
    this.stats.fragmentation = freeBlocks / Math.max(this.memoryBlocks.size, 1);
  }

  private updatePerformanceMetrics(): void {
    const currentTime = Date.now();
    this.performanceWindow.push(currentTime);
    
    // Keep only last second of operations
    this.performanceWindow = this.performanceWindow.filter(time => 
      currentTime - time <= 1000
    );
    
    this.stats.operationsPerSecond = this.performanceWindow.length;
    
    // Calculate average operation time
    const recentOperations = this.operationHistory.slice(-100);
    if (recentOperations.length > 0) {
      const totalTime = recentOperations.reduce((sum, op) => 
        sum + (op.duration || 0), 0
      );
      this.stats.averageOperationTime = totalTime / recentOperations.length;
    }
  }

  public animate(): void {
    const deltaTime = this.animationClock.getDelta();
    const elapsedTime = this.animationClock.getElapsedTime();
    
    // Update shader time uniforms
    this.memoryBarMaterial.uniforms.time.value = elapsedTime;
    this.operationMaterial.uniforms.time.value = elapsedTime;
    
    // Update particle system
    this.particleSystem.update(deltaTime);
    
    // Update operation animations
    this.updateOperationVisualization();
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }

  public getStats(): MemoryStats {
    return { ...this.stats };
  }

  public getOperationHistory(): MemoryOperation[] {
    return [...this.operationHistory];
  }

  public clearHistory(): void {
    this.operationHistory.length = 0;
    this.performanceWindow.length = 0;
  }

  public dispose(): void {
    // Dispose of all Three.js resources
    this.memoryBarGeometry.dispose();
    this.memoryBarMaterial.dispose();
    this.operationGeometry.dispose();
    this.operationMaterial.dispose();
    this.particleSystem.dispose();
    this.renderer.dispose();
    
    // Clear collections
    this.memoryBlocks.clear();
    this.activeOperations.clear();
    this.operationAnimations.clear();
    this.operationHistory.length = 0;
  }
}

export default MemoryOperationVisualizer;