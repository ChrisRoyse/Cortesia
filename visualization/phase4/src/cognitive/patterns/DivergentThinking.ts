import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class DivergentThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private radiationParticles: THREE.Points | null = null;
  private centerCore: THREE.Mesh;
  private expansionRings: THREE.Mesh[] = [];
  private branchingPaths: THREE.Line[] = [];
  private ideaBubbles: THREE.Mesh[] = [];
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private expansionRadius: number = 0;
  private branchCount: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 1200;
  private readonly MAX_RINGS = 5;
  private readonly MAX_BRANCHES = 16;
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'DivergentThinking';
    this.group.position.set(3, 2, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createCenterCore();
    this.createExpansionRings();
    this.createRadiationParticles();
    this.createBranchingPaths();
    this.createIdeaBubbles();
  }
  
  private createCenterCore(): void {
    // Central core from which all ideas radiate
    const geometry = new THREE.SphereGeometry(0.15, 16, 12);
    const material = new THREE.MeshBasicMaterial({
      color: 0xff4d80,
      transparent: true,
      opacity: 0,
      emissive: 0xaa2244,
      emissiveIntensity: 0
    });
    
    this.centerCore = new THREE.Mesh(geometry, material);
    this.centerCore.position.copy(this.CENTER_POSITION);
    this.group.add(this.centerCore);
  }
  
  private createExpansionRings(): void {
    // Expanding rings that show divergent waves
    for (let i = 0; i < this.MAX_RINGS; i++) {
      const radius = 0.5 + i * 0.4;
      const geometry = new THREE.RingGeometry(radius - 0.05, radius + 0.05, 32);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff4d80,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
      });
      
      const ring = new THREE.Mesh(geometry, material);
      ring.position.copy(this.CENTER_POSITION);
      ring.rotation.x = Math.PI / 2;
      this.expansionRings.push(ring);
      this.group.add(ring);
    }
  }
  
  private createRadiationParticles(): void {
    // Create particles that radiate outward from center
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const phases = new Float32Array(this.PARTICLE_COUNT);
    const directions = new Float32Array(this.PARTICLE_COUNT * 3);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Start particles at center
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
      
      // Random radiation directions
      const phi = Math.random() * Math.PI * 2;
      const theta = Math.random() * Math.PI;
      
      directions[i * 3] = Math.sin(theta) * Math.cos(phi);
      directions[i * 3 + 1] = Math.sin(theta) * Math.sin(phi);
      directions[i * 3 + 2] = Math.cos(theta);
      
      phases[i] = Math.random() * Math.PI * 2;
    }
    
    this.radiationParticles = this.effects.createParticleSystem('divergent', positions, {
      phase: phases,
      direction: directions
    });
    
    this.group.add(this.radiationParticles);
  }
  
  private createBranchingPaths(): void {
    // Create branching paths that spread ideas outward
    for (let i = 0; i < this.MAX_BRANCHES; i++) {
      const angle = (i / this.MAX_BRANCHES) * Math.PI * 2;
      const branchAngle = angle + (Math.random() - 0.5) * 0.5;
      
      const startPoint = this.CENTER_POSITION.clone();
      const midPoint = new THREE.Vector3(
        Math.cos(branchAngle) * 2,
        Math.sin(branchAngle) * 2,
        (Math.random() - 0.5) * 1.5
      );
      const endPoint = new THREE.Vector3(
        Math.cos(branchAngle) * 4,
        Math.sin(branchAngle) * 4,
        (Math.random() - 0.5) * 2
      );
      
      // Create branching curve
      const curve = new THREE.CatmullRomCurve3([
        startPoint,
        startPoint.clone().lerp(midPoint, 0.3),
        midPoint,
        midPoint.clone().lerp(endPoint, 0.7),
        endPoint
      ]);
      
      const points = curve.getPoints(30);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      
      const material = new THREE.LineBasicMaterial({
        color: 0xff4d80,
        transparent: true,
        opacity: 0,
        linewidth: 1.5
      });
      
      const line = new THREE.Line(geometry, material);
      this.branchingPaths.push(line);
      this.group.add(line);
    }
  }
  
  private createIdeaBubbles(): void {
    // Create bubbles representing different ideas
    const bubbleCount = 20;
    
    for (let i = 0; i < bubbleCount; i++) {
      const angle = (i / bubbleCount) * Math.PI * 2;
      const radius = 2.5 + Math.random() * 1.5;
      
      const geometry = new THREE.SphereGeometry(0.08 + Math.random() * 0.04, 8, 6);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff4d80,
        transparent: true,
        opacity: 0,
        emissive: 0x442244,
        emissiveIntensity: 0
      });
      
      const bubble = new THREE.Mesh(geometry, material);
      bubble.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        (Math.random() - 0.5) * 2
      );
      
      this.ideaBubbles.push(bubble);
      this.group.add(bubble);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update center core
    const coreMaterial = this.centerCore.material as THREE.MeshBasicMaterial;
    coreMaterial.opacity = activation;
    coreMaterial.emissiveIntensity = intensity;
    
    // Update expansion rings
    this.expansionRings.forEach((ring, index) => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (1 - index * 0.15);
    });
    
    // Update branching paths
    this.branchingPaths.forEach((path, index) => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.8;
    });
    
    // Update idea bubbles
    this.ideaBubbles.forEach((bubble, index) => {
      const material = bubble.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (0.4 + Math.random() * 0.4);
      material.emissiveIntensity = intensity * 0.3;
    });
    
    // Update particle system
    if (this.radiationParticles) {
      const material = this.radiationParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.expansionRadius = 0;
    this.branchCount = 0;
    
    // Fade out all elements
    const coreMaterial = this.centerCore.material as THREE.MeshBasicMaterial;
    coreMaterial.opacity = 0;
    coreMaterial.emissiveIntensity = 0;
    
    this.expansionRings.forEach(ring => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
    });
    
    this.branchingPaths.forEach(path => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.ideaBubbles.forEach(bubble => {
      const material = bubble.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    if (this.radiationParticles) {
      const material = this.radiationParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update expansion radius
    this.expansionRadius = Math.min(5, this.expansionRadius + 0.05);
    
    // Update branch count
    this.branchCount = Math.min(this.MAX_BRANCHES, this.branchCount + 0.3);
    
    // Animate center core
    this.animateCenterCore(time);
    
    // Animate expansion rings
    this.animateExpansionRings(time);
    
    // Animate branching paths
    this.animateBranchingPaths(time);
    
    // Animate idea bubbles
    this.animateIdeaBubbles(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateCenterCore(time: number): void {
    // Pulsing expansion effect
    const pulseScale = 1 + Math.sin(time * 3) * 0.4 * this.intensity;
    this.centerCore.scale.setScalar(pulseScale);
    
    // Slow rotation
    this.centerCore.rotation.y = time * 0.5;
    
    // Color intensity variation
    const material = this.centerCore.material as THREE.MeshBasicMaterial;
    material.emissiveIntensity = this.intensity * (0.7 + Math.sin(time * 2) * 0.3);
  }
  
  private animateExpansionRings(time: number): void {
    this.expansionRings.forEach((ring, index) => {
      // Expanding wave animation
      const wavePhase = time * 2 - index * 0.5;
      const expansion = Math.sin(wavePhase) * 0.2 + 1;
      ring.scale.setScalar(expansion * (1 + this.expansionRadius * 0.1));
      
      // Opacity waves
      const material = ring.material as THREE.MeshBasicMaterial;
      const opacityWave = Math.sin(wavePhase * 1.5) * 0.3 + 0.7;
      material.opacity = this.activation * (1 - index * 0.15) * opacityWave;
      
      // Color shift during expansion
      const colorShift = (this.expansionRadius / 5) * 0.3;
      material.color.setRGB(
        1.0,
        0.3 + colorShift,
        0.5 + colorShift
      );
    });
  }
  
  private animateBranchingPaths(time: number): void {
    this.branchingPaths.forEach((path, index) => {
      if (index < this.branchCount) {
        const material = path.material as THREE.LineBasicMaterial;
        
        // Growing path effect
        const growthProgress = Math.min(1, (time * 0.5 + index * 0.1) % 3);
        material.opacity = this.activation * 0.8 * growthProgress;
        
        // Color flow along branches
        const flowPhase = time * 1.5 + index * 0.3;
        const flowIntensity = Math.sin(flowPhase) * 0.4 + 0.6;
        material.color.setRGB(
          1.0,
          0.3 + flowIntensity * 0.3,
          0.5 + flowIntensity * 0.3
        );
      }
    });
  }
  
  private animateIdeaBubbles(time: number): void {
    this.ideaBubbles.forEach((bubble, index) => {
      // Floating motion
      const floatPhase = time * 0.8 + index * 0.4;
      bubble.position.y += Math.sin(floatPhase) * 0.01;
      bubble.position.x += Math.cos(floatPhase * 1.3) * 0.005;
      
      // Size variation
      const sizePhase = time * 1.2 + index * 0.6;
      const sizeVariation = Math.sin(sizePhase) * 0.2 + 1;
      bubble.scale.setScalar(sizeVariation);
      
      // Opacity flickering (like ideas popping)
      const material = bubble.material as THREE.MeshBasicMaterial;
      const flickerPhase = time * 2 + index * 0.8;
      const flicker = Math.sin(flickerPhase) * 0.3 + 0.7;
      material.opacity = this.activation * (0.4 + Math.random() * 0.4) * flicker;
      
      // Random color shifts
      if (Math.random() < 0.01) { // Occasional color shifts
        material.color.setHSL(
          (0.9 + Math.random() * 0.2) % 1, // Pink to purple range
          0.8,
          0.6
        );
      }
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Center core decay
    const coreMaterial = this.centerCore.material as THREE.MeshBasicMaterial;
    coreMaterial.opacity = currentActivation;
    coreMaterial.emissiveIntensity = currentIntensity;
    
    // Expansion rings decay
    this.expansionRings.forEach((ring, index) => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (1 - index * 0.15);
    });
    
    // Branching paths decay
    this.branchingPaths.forEach(path => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.8;
    });
    
    // Idea bubbles decay
    this.ideaBubbles.forEach(bubble => {
      const material = bubble.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (0.4 + Math.random() * 0.4);
      material.emissiveIntensity = currentIntensity * 0.3;
    });
    
    // Particle system decay
    if (this.radiationParticles) {
      const material = this.radiationParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'divergent',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      expansionRadius: this.expansionRadius,
      branchCount: this.branchCount,
      particleCount: this.PARTICLE_COUNT
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.centerCore.geometry.dispose();
    (this.centerCore.material as THREE.Material).dispose();
    
    this.expansionRings.forEach(ring => {
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
    });
    
    this.branchingPaths.forEach(path => {
      path.geometry.dispose();
      (path.material as THREE.Material).dispose();
    });
    
    this.ideaBubbles.forEach(bubble => {
      bubble.geometry.dispose();
      (bubble.material as THREE.Material).dispose();
    });
    
    if (this.radiationParticles) {
      this.radiationParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}