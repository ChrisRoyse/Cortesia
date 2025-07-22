/**
 * ParticleEffects.ts
 * 
 * Advanced particle system for visualizing MCP request flow through cognitive systems.
 * Creates dynamic particle trails, bursts, and effects for request visualization.
 */

import * as THREE from 'three';
import { TraceEvent } from './MCPRequestTracer';

export interface Particle {
  id: string;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  life: number;
  maxLife: number;
  size: number;
  color: THREE.Color;
  opacity: number;
  type: 'trail' | 'burst' | 'error' | 'success' | 'processing';
  requestId: string;
}

export interface ParticleTrail {
  id: string;
  requestId: string;
  particles: Particle[];
  path: THREE.Vector3[];
  currentSegment: number;
  progress: number;
  speed: number;
  emissionRate: number;
  lastEmission: number;
  status: 'active' | 'complete' | 'error';
}

export interface ParticleSystemConfig {
  maxParticles: number;
  trailLength: number;
  particleSpeed: number;
  emissionRate: number;
  particleSize: number;
  particleLifetime: number;
  colors: {
    request: string;
    processing: string;
    success: string;
    error: string;
    cognitive: string;
  };
  effects: {
    gravity: number;
    turbulence: number;
    fadeRate: number;
    sizeVariation: number;
  };
}

export class ParticleEffects {
  private scene: THREE.Scene;
  private particles: Map<string, Particle> = new Map();
  private trails: Map<string, ParticleTrail> = new Map();
  private particleSystem: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry;
  private material: THREE.PointsMaterial;
  private animationFrame: number | null = null;
  private config: ParticleSystemConfig;
  private particlePool: Particle[] = [];

  // Particle attribute arrays
  private positions: Float32Array;
  private colors: Float32Array;
  private sizes: Float32Array;
  private opacities: Float32Array;

  constructor(scene: THREE.Scene, config?: Partial<ParticleSystemConfig>) {
    this.scene = scene;
    this.config = {
      maxParticles: 5000,
      trailLength: 50,
      particleSpeed: 20,
      emissionRate: 10, // particles per second
      particleSize: 2.0,
      particleLifetime: 3000, // milliseconds
      colors: {
        request: '#00BCD4',    // Cyan
        processing: '#FF9800', // Orange
        success: '#4CAF50',    // Green
        error: '#F44336',      // Red
        cognitive: '#2196F3'   // Blue
      },
      effects: {
        gravity: -0.1,
        turbulence: 0.5,
        fadeRate: 0.02,
        sizeVariation: 0.3
      },
      ...config
    };

    this.initializeParticleSystem();
    this.initializeParticlePool();
    this.startAnimation();
  }

  /**
   * Initialize Three.js particle system
   */
  private initializeParticleSystem(): void {
    this.geometry = new THREE.BufferGeometry();
    
    // Initialize attribute arrays
    this.positions = new Float32Array(this.config.maxParticles * 3);
    this.colors = new Float32Array(this.config.maxParticles * 3);
    this.sizes = new Float32Array(this.config.maxParticles);
    this.opacities = new Float32Array(this.config.maxParticles);

    // Set buffer attributes
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    this.geometry.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1));
    this.geometry.setAttribute('opacity', new THREE.BufferAttribute(this.opacities, 1));

    // Create particle material with custom shader
    this.material = new THREE.PointsMaterial({
      size: this.config.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    });

    // Create particle system
    this.particleSystem = new THREE.Points(this.geometry, this.material);
    this.scene.add(this.particleSystem);
  }

  /**
   * Initialize particle pool for efficient memory management
   */
  private initializeParticlePool(): void {
    for (let i = 0; i < this.config.maxParticles; i++) {
      this.particlePool.push(this.createParticle('', new THREE.Vector3(), 'trail', ''));
    }
  }

  /**
   * Create a new particle
   */
  private createParticle(
    id: string, 
    position: THREE.Vector3, 
    type: Particle['type'], 
    requestId: string
  ): Particle {
    const color = this.getParticleColor(type);
    const baseSize = this.config.particleSize;
    const sizeVariation = baseSize * this.config.effects.sizeVariation;
    
    return {
      id,
      position: position.clone(),
      velocity: new THREE.Vector3(),
      life: this.config.particleLifetime,
      maxLife: this.config.particleLifetime,
      size: baseSize + (Math.random() - 0.5) * sizeVariation,
      color: new THREE.Color(color),
      opacity: 1.0,
      type,
      requestId
    };
  }

  /**
   * Get particle color based on type
   */
  private getParticleColor(type: Particle['type']): string {
    switch (type) {
      case 'trail': return this.config.colors.request;
      case 'processing': return this.config.colors.processing;
      case 'burst': return this.config.colors.cognitive;
      case 'success': return this.config.colors.success;
      case 'error': return this.config.colors.error;
      default: return this.config.colors.request;
    }
  }

  /**
   * Process trace event and create appropriate particle effects
   */
  public processTraceEvent(event: TraceEvent): void {
    switch (event.type) {
      case 'request':
        this.createRequestTrail(event);
        break;
      case 'cognitive_activation':
        this.createCognitiveActivationBurst(event);
        break;
      case 'response':
        this.createResponseEffect(event);
        break;
      case 'error':
        this.createErrorEffect(event);
        break;
      case 'performance':
        this.createPerformanceIndicator(event);
        break;
    }
  }

  /**
   * Create particle trail for request flow
   */
  private createRequestTrail(event: TraceEvent): void {
    if (!event.coordinates) return;

    const startPosition = new THREE.Vector3(
      event.coordinates.x, 
      event.coordinates.y, 
      event.coordinates.z || 0
    );

    // Generate path points for the trail
    const path = this.generateTrailPath(startPosition, event);
    
    const trail: ParticleTrail = {
      id: `trail_${event.requestId}`,
      requestId: event.requestId,
      particles: [],
      path,
      currentSegment: 0,
      progress: 0,
      speed: this.config.particleSpeed,
      emissionRate: this.config.emissionRate,
      lastEmission: Date.now(),
      status: 'active'
    };

    this.trails.set(trail.id, trail);
  }

  /**
   * Generate path for particle trail
   */
  private generateTrailPath(startPosition: THREE.Vector3, event: TraceEvent): THREE.Vector3[] {
    const path = [startPosition];
    
    // Add intermediate points based on request data
    if (event.data.path && event.data.path.length > 0) {
      event.data.path.forEach((point: string, index: number) => {
        // Convert cognitive system names to coordinates
        const position = this.getCognitiveSystemPosition(point, index);
        path.push(position);
      });
    }

    // Add end position
    const endPosition = new THREE.Vector3(100, 0, 0); // Default exit
    path.push(endPosition);

    return path;
  }

  /**
   * Get position for cognitive system
   */
  private getCognitiveSystemPosition(systemName: string, index: number): THREE.Vector3 {
    const systemPositions: { [key: string]: THREE.Vector3 } = {
      'hierarchical_inhibitory': new THREE.Vector3(-50, 50, 0),
      'working_memory': new THREE.Vector3(-25, 25, 0),
      'knowledge_engine': new THREE.Vector3(0, 0, 0),
      'activation_engine': new THREE.Vector3(25, 25, 0),
      'triple_store': new THREE.Vector3(0, -25, 0),
      'sdr_storage': new THREE.Vector3(25, -25, 0),
      'zero_copy_engine': new THREE.Vector3(-25, -25, 0)
    };

    return systemPositions[systemName] || new THREE.Vector3(index * 20, 0, 0);
  }

  /**
   * Create burst effect for cognitive activation
   */
  private createCognitiveActivationBurst(event: TraceEvent): void {
    if (!event.coordinates) return;

    const center = new THREE.Vector3(
      event.coordinates.x,
      event.coordinates.y,
      event.coordinates.z || 0
    );

    // Create burst particles
    const burstCount = 20 + Math.random() * 30;
    for (let i = 0; i < burstCount; i++) {
      const particle = this.getParticleFromPool();
      if (!particle) continue;

      particle.id = `burst_${event.id}_${i}`;
      particle.requestId = event.requestId;
      particle.type = 'burst';
      particle.position.copy(center);
      particle.color = new THREE.Color(this.config.colors.cognitive);
      
      // Random burst direction
      const angle = (i / burstCount) * Math.PI * 2;
      const radius = 2 + Math.random() * 3;
      particle.velocity.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        (Math.random() - 0.5) * 2
      );

      particle.life = particle.maxLife;
      particle.opacity = 1.0;

      this.particles.set(particle.id, particle);
    }
  }

  /**
   * Create response completion effect
   */
  private createResponseEffect(event: TraceEvent): void {
    if (!event.coordinates) return;

    const isError = event.data.error;
    const effectType = isError ? 'error' : 'success';
    const color = isError ? this.config.colors.error : this.config.colors.success;

    const position = new THREE.Vector3(
      event.coordinates.x,
      event.coordinates.y,
      event.coordinates.z || 0
    );

    // Create completion effect
    this.createParticleExplosion(position, effectType, event.requestId, color);

    // Complete trail for this request
    const trailId = `trail_${event.requestId}`;
    const trail = this.trails.get(trailId);
    if (trail) {
      trail.status = isError ? 'error' : 'complete';
    }
  }

  /**
   * Create error effect
   */
  private createErrorEffect(event: TraceEvent): void {
    if (!event.coordinates) return;

    const position = new THREE.Vector3(
      event.coordinates.x,
      event.coordinates.y,
      event.coordinates.z || 0
    );

    this.createParticleExplosion(position, 'error', event.requestId, this.config.colors.error);
  }

  /**
   * Create performance indicator particles
   */
  private createPerformanceIndicator(event: TraceEvent): void {
    const performanceData = event.data;
    if (!performanceData.coordinates) return;

    const intensity = Math.min(performanceData.processingTime / 1000, 1); // Normalize to 0-1
    const particleCount = Math.floor(intensity * 10) + 1;

    for (let i = 0; i < particleCount; i++) {
      const particle = this.getParticleFromPool();
      if (!particle) continue;

      particle.id = `perf_${event.id}_${i}`;
      particle.requestId = event.requestId;
      particle.type = 'processing';
      particle.position.set(
        performanceData.coordinates.x + (Math.random() - 0.5) * 10,
        performanceData.coordinates.y + (Math.random() - 0.5) * 10,
        performanceData.coordinates.z || 0
      );

      // Color based on performance - green for fast, red for slow
      const hue = (1 - intensity) * 120; // 120 is green, 0 is red
      particle.color.setHSL(hue / 360, 0.8, 0.6);
      particle.life = particle.maxLife * 0.5; // Shorter life for performance indicators

      this.particles.set(particle.id, particle);
    }
  }

  /**
   * Create particle explosion effect
   */
  private createParticleExplosion(
    position: THREE.Vector3, 
    type: Particle['type'], 
    requestId: string, 
    color: string
  ): void {
    const explosionCount = 30 + Math.random() * 50;
    
    for (let i = 0; i < explosionCount; i++) {
      const particle = this.getParticleFromPool();
      if (!particle) continue;

      particle.id = `explosion_${requestId}_${i}`;
      particle.requestId = requestId;
      particle.type = type;
      particle.position.copy(position);
      particle.color = new THREE.Color(color);

      // Spherical explosion pattern
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const force = 5 + Math.random() * 10;

      particle.velocity.set(
        force * Math.sin(phi) * Math.cos(theta),
        force * Math.sin(phi) * Math.sin(theta),
        force * Math.cos(phi)
      );

      particle.life = particle.maxLife;
      particle.opacity = 1.0;

      this.particles.set(particle.id, particle);
    }
  }

  /**
   * Get particle from pool or create new one
   */
  private getParticleFromPool(): Particle | null {
    if (this.particles.size >= this.config.maxParticles) {
      return null; // Max particles reached
    }

    if (this.particlePool.length > 0) {
      return this.particlePool.pop()!;
    }

    return this.createParticle('', new THREE.Vector3(), 'trail', '');
  }

  /**
   * Return particle to pool
   */
  private returnParticleToPool(particle: Particle): void {
    particle.life = 0;
    particle.opacity = 0;
    this.particlePool.push(particle);
  }

  /**
   * Update particle system
   */
  private updateParticles(deltaTime: number): void {
    let particleIndex = 0;
    const particlesToRemove: string[] = [];

    // Update existing particles
    this.particles.forEach((particle, id) => {
      // Update particle physics
      particle.life -= deltaTime;
      
      if (particle.life <= 0) {
        particlesToRemove.push(id);
        return;
      }

      // Update position
      particle.position.add(particle.velocity.clone().multiplyScalar(deltaTime / 1000));

      // Apply effects
      particle.velocity.y += this.config.effects.gravity * deltaTime / 1000;
      
      // Add turbulence
      particle.velocity.add(new THREE.Vector3(
        (Math.random() - 0.5) * this.config.effects.turbulence,
        (Math.random() - 0.5) * this.config.effects.turbulence,
        (Math.random() - 0.5) * this.config.effects.turbulence
      ));

      // Update opacity based on life
      particle.opacity = particle.life / particle.maxLife;

      // Update buffer attributes
      if (particleIndex < this.config.maxParticles) {
        const i3 = particleIndex * 3;
        
        this.positions[i3] = particle.position.x;
        this.positions[i3 + 1] = particle.position.y;
        this.positions[i3 + 2] = particle.position.z;

        this.colors[i3] = particle.color.r;
        this.colors[i3 + 1] = particle.color.g;
        this.colors[i3 + 2] = particle.color.b;

        this.sizes[particleIndex] = particle.size * particle.opacity;
        this.opacities[particleIndex] = particle.opacity;

        particleIndex++;
      }
    });

    // Remove expired particles
    particlesToRemove.forEach(id => {
      const particle = this.particles.get(id);
      if (particle) {
        this.returnParticleToPool(particle);
        this.particles.delete(id);
      }
    });

    // Update trails
    this.updateTrails(deltaTime);

    // Update buffer attributes
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.attributes.size.needsUpdate = true;
    this.geometry.attributes.opacity.needsUpdate = true;

    // Set draw range
    this.geometry.setDrawRange(0, particleIndex);
  }

  /**
   * Update particle trails
   */
  private updateTrails(deltaTime: number): void {
    const now = Date.now();

    this.trails.forEach((trail, trailId) => {
      if (trail.status !== 'active') return;

      // Update trail progress
      trail.progress += (trail.speed * deltaTime) / 1000;

      // Emit new particles
      if (now - trail.lastEmission > 1000 / trail.emissionRate) {
        this.emitTrailParticle(trail);
        trail.lastEmission = now;
      }

      // Check if trail is complete
      if (trail.progress >= trail.path.length - 1) {
        trail.status = 'complete';
      }
    });
  }

  /**
   * Emit particle along trail path
   */
  private emitTrailParticle(trail: ParticleTrail): void {
    const particle = this.getParticleFromPool();
    if (!particle) return;

    // Calculate position along path
    const segmentIndex = Math.floor(trail.progress);
    const segmentProgress = trail.progress - segmentIndex;

    if (segmentIndex >= trail.path.length - 1) return;

    const startPoint = trail.path[segmentIndex];
    const endPoint = trail.path[segmentIndex + 1];
    const position = new THREE.Vector3().lerpVectors(startPoint, endPoint, segmentProgress);

    particle.id = `trail_${trail.id}_${Date.now()}`;
    particle.requestId = trail.requestId;
    particle.type = 'trail';
    particle.position.copy(position);
    particle.color = new THREE.Color(this.config.colors.request);
    particle.life = particle.maxLife * 0.8; // Shorter life for trail particles

    // Set velocity along path direction
    const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
    particle.velocity.copy(direction).multiplyScalar(trail.speed * 0.5);

    this.particles.set(particle.id, particle);
  }

  /**
   * Animation loop
   */
  private animate = (): void => {
    const deltaTime = 16; // ~60fps

    this.updateParticles(deltaTime);
    this.animationFrame = requestAnimationFrame(this.animate);
  };

  /**
   * Start animation loop
   */
  private startAnimation(): void {
    if (!this.animationFrame) {
      this.animate();
    }
  }

  /**
   * Stop animation loop
   */
  private stopAnimation(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Update particle system configuration
   */
  public updateConfig(config: Partial<ParticleSystemConfig>): void {
    this.config = { ...this.config, ...config };
    
    if (this.material) {
      this.material.size = this.config.particleSize;
    }
  }

  /**
   * Clear all particles for a specific request
   */
  public clearRequest(requestId: string): void {
    const particlesToRemove: string[] = [];
    
    this.particles.forEach((particle, id) => {
      if (particle.requestId === requestId) {
        particlesToRemove.push(id);
      }
    });

    particlesToRemove.forEach(id => {
      const particle = this.particles.get(id);
      if (particle) {
        this.returnParticleToPool(particle);
        this.particles.delete(id);
      }
    });

    // Remove trail
    const trailId = `trail_${requestId}`;
    this.trails.delete(trailId);
  }

  /**
   * Clear all particles and trails
   */
  public clearAll(): void {
    this.particles.forEach(particle => {
      this.returnParticleToPool(particle);
    });
    this.particles.clear();
    this.trails.clear();
  }

  /**
   * Get current particle count
   */
  public getParticleCount(): number {
    return this.particles.size;
  }

  /**
   * Get active trail count
   */
  public getTrailCount(): number {
    return Array.from(this.trails.values()).filter(t => t.status === 'active').length;
  }

  /**
   * Dispose and cleanup resources
   */
  public dispose(): void {
    this.stopAnimation();
    this.clearAll();

    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
    }

    if (this.geometry) {
      this.geometry.dispose();
    }

    if (this.material) {
      this.material.dispose();
    }

    this.particles.clear();
    this.trails.clear();
    this.particlePool.length = 0;
  }
}