/**
 * High-performance particle system for LLMKG data flow visualization
 * Optimized for 60 FPS rendering with memory-efficient particle management
 */

import * as THREE from 'three';
import { ShaderLibrary } from './ShaderLibrary';

export interface ParticleConfig {
  maxParticles: number;
  particleSize: number;
  lifespan: number;
  spawnRate: number;
  initialVelocity: THREE.Vector3;
  gravity: THREE.Vector3;
  colorStart: THREE.Color;
  colorEnd: THREE.Color;
  shaderName: string;
}

export interface DataFlowParticle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  lifecycle: number; // 0.0 to 1.0
  age: number;
  size: number;
  color: THREE.Color;
  phase: number;
  active: boolean;
  dataId?: string;
  cognitiveWeight?: number;
}

export class ParticleSystem {
  private particles: DataFlowParticle[] = [];
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
  private points: THREE.Points;
  private config: ParticleConfig;
  private shaderLibrary: ShaderLibrary;
  
  // Buffer attributes for performance
  private positionArray: Float32Array;
  private colorArray: Float32Array;
  private sizeArray: Float32Array;
  private lifecycleArray: Float32Array;
  private phaseArray: Float32Array;
  
  private activeParticleCount: number = 0;
  private spawnAccumulator: number = 0;
  private clock: THREE.Clock;

  constructor(config: ParticleConfig) {
    this.config = { ...config };
    this.shaderLibrary = ShaderLibrary.getInstance();
    this.clock = new THREE.Clock();
    
    this.initializeBuffers();
    this.createGeometry();
    this.createMaterial();
    this.createPoints();
  }

  private initializeBuffers(): void {
    const maxParticles = this.config.maxParticles;
    
    this.positionArray = new Float32Array(maxParticles * 3);
    this.colorArray = new Float32Array(maxParticles * 3);
    this.sizeArray = new Float32Array(maxParticles);
    this.lifecycleArray = new Float32Array(maxParticles);
    this.phaseArray = new Float32Array(maxParticles);
    
    // Initialize particle pool
    this.particles = Array.from({ length: maxParticles }, (_, i) => ({
      position: new THREE.Vector3(),
      velocity: new THREE.Vector3(),
      lifecycle: 0,
      age: 0,
      size: this.config.particleSize,
      color: new THREE.Color(),
      phase: Math.random() * Math.PI * 2,
      active: false,
      dataId: undefined,
      cognitiveWeight: 0
    }));
  }

  private createGeometry(): void {
    this.geometry = new THREE.BufferGeometry();
    
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positionArray, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colorArray, 3));
    this.geometry.setAttribute('size', new THREE.BufferAttribute(this.sizeArray, 1));
    this.geometry.setAttribute('lifecycle', new THREE.BufferAttribute(this.lifecycleArray, 1));
    this.geometry.setAttribute('phase', new THREE.BufferAttribute(this.phaseArray, 1));
    
    this.geometry.setDrawRange(0, 0);
  }

  private createMaterial(): void {
    const material = this.shaderLibrary.createMaterial(this.config.shaderName);
    if (!material) {
      throw new Error(`Shader '${this.config.shaderName}' not found in ShaderLibrary`);
    }
    
    this.material = material;
    this.material.vertexColors = true;
  }

  private createPoints(): void {
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
  }

  public spawnParticle(
    position: THREE.Vector3,
    velocity?: THREE.Vector3,
    dataId?: string,
    cognitiveWeight?: number
  ): boolean {
    const inactiveParticle = this.particles.find(p => !p.active);
    if (!inactiveParticle) return false;

    inactiveParticle.position.copy(position);
    inactiveParticle.velocity.copy(velocity || this.config.initialVelocity);
    inactiveParticle.lifecycle = 1.0;
    inactiveParticle.age = 0;
    inactiveParticle.active = true;
    inactiveParticle.dataId = dataId;
    inactiveParticle.cognitiveWeight = cognitiveWeight || Math.random();
    
    // Randomize some properties for visual variety
    inactiveParticle.size = this.config.particleSize * (0.8 + Math.random() * 0.4);
    inactiveParticle.phase = Math.random() * Math.PI * 2;
    
    this.activeParticleCount++;
    return true;
  }

  public spawnDataFlowBurst(
    center: THREE.Vector3,
    count: number,
    spread: number = 1.0,
    dataId?: string
  ): void {
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const radius = Math.random() * spread;
      
      const position = new THREE.Vector3(
        center.x + Math.cos(angle) * radius,
        center.y + (Math.random() - 0.5) * spread * 0.5,
        center.z + Math.sin(angle) * radius
      );
      
      const velocity = new THREE.Vector3(
        Math.cos(angle) * 2,
        Math.random() * 1 - 0.5,
        Math.sin(angle) * 2
      );
      
      this.spawnParticle(position, velocity, dataId, Math.random());
    }
  }

  public update(deltaTime: number): void {
    const time = this.clock.getElapsedTime();
    
    // Update shader time
    if (this.material.uniforms.time) {
      this.material.uniforms.time.value = time;
    }
    
    // Handle particle spawning
    this.handleSpawning(deltaTime);
    
    // Update particles
    this.updateParticles(deltaTime);
    
    // Update buffer attributes
    this.updateBuffers();
    
    // Update geometry draw range
    this.geometry.setDrawRange(0, this.activeParticleCount);
  }

  private handleSpawning(deltaTime: number): void {
    if (this.config.spawnRate <= 0) return;
    
    this.spawnAccumulator += deltaTime;
    const spawnInterval = 1.0 / this.config.spawnRate;
    
    while (this.spawnAccumulator >= spawnInterval) {
      this.spawnRandomParticle();
      this.spawnAccumulator -= spawnInterval;
    }
  }

  private spawnRandomParticle(): void {
    const position = new THREE.Vector3(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10
    );
    
    const velocity = this.config.initialVelocity.clone().multiplyScalar(0.5 + Math.random() * 0.5);
    
    this.spawnParticle(position, velocity);
  }

  private updateParticles(deltaTime: number): void {
    this.activeParticleCount = 0;
    
    for (const particle of this.particles) {
      if (!particle.active) continue;
      
      // Update lifecycle
      particle.age += deltaTime;
      particle.lifecycle = Math.max(0, 1.0 - (particle.age / this.config.lifespan));
      
      if (particle.lifecycle <= 0) {
        particle.active = false;
        continue;
      }
      
      // Update physics
      particle.velocity.add(
        this.config.gravity.clone().multiplyScalar(deltaTime)
      );
      
      particle.position.add(
        particle.velocity.clone().multiplyScalar(deltaTime)
      );
      
      // Update color based on lifecycle
      particle.color.lerpColors(
        this.config.colorStart,
        this.config.colorEnd,
        1.0 - particle.lifecycle
      );
      
      // Update size based on lifecycle
      particle.size = this.config.particleSize * particle.lifecycle;
      
      this.activeParticleCount++;
    }
  }

  private updateBuffers(): void {
    let activeIndex = 0;
    
    for (const particle of this.particles) {
      if (!particle.active) continue;
      
      const i3 = activeIndex * 3;
      
      // Position
      this.positionArray[i3] = particle.position.x;
      this.positionArray[i3 + 1] = particle.position.y;
      this.positionArray[i3 + 2] = particle.position.z;
      
      // Color
      this.colorArray[i3] = particle.color.r;
      this.colorArray[i3 + 1] = particle.color.g;
      this.colorArray[i3 + 2] = particle.color.b;
      
      // Size
      this.sizeArray[activeIndex] = particle.size;
      
      // Lifecycle
      this.lifecycleArray[activeIndex] = particle.lifecycle;
      
      // Phase
      this.phaseArray[activeIndex] = particle.phase;
      
      activeIndex++;
    }
    
    // Mark attributes as needing update
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.attributes.size.needsUpdate = true;
    this.geometry.attributes.lifecycle.needsUpdate = true;
    this.geometry.attributes.phase.needsUpdate = true;
  }

  public getPoints(): THREE.Points {
    return this.points;
  }

  public getActiveParticleCount(): number {
    return this.activeParticleCount;
  }

  public setConfig(newConfig: Partial<ParticleConfig>): void {
    Object.assign(this.config, newConfig);
  }

  public clearParticles(): void {
    this.particles.forEach(particle => {
      particle.active = false;
    });
    this.activeParticleCount = 0;
    this.geometry.setDrawRange(0, 0);
  }

  public dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.particles.length = 0;
    this.activeParticleCount = 0;
  }

  // Performance monitoring
  public getPerformanceMetrics() {
    return {
      activeParticles: this.activeParticleCount,
      maxParticles: this.config.maxParticles,
      utilizationPercentage: (this.activeParticleCount / this.config.maxParticles) * 100,
      memoryUsage: {
        positions: this.positionArray.length * 4, // bytes
        colors: this.colorArray.length * 4,
        total: (this.positionArray.length + this.colorArray.length + this.sizeArray.length + 
               this.lifecycleArray.length + this.phaseArray.length) * 4
      }
    };
  }
}

export default ParticleSystem;