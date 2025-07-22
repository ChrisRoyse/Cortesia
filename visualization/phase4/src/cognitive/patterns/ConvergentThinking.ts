import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class ConvergentThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private beamParticles: THREE.Points | null = null;
  private focusPoint: THREE.Mesh;
  private dataStreams: THREE.Line[] = [];
  private convergenceField: THREE.Mesh;
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private convergenceProgress: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 800;
  private readonly STREAM_COUNT = 12;
  private readonly FOCUS_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'ConvergentThinking';
    this.group.position.set(-3, 2, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createFocusPoint();
    this.createConvergenceField();
    this.createBeamParticles();
    this.createDataStreams();
  }
  
  private createFocusPoint(): void {
    // Central focus point where all thinking converges
    const geometry = new THREE.SphereGeometry(0.1, 16, 12);
    const material = new THREE.MeshBasicMaterial({
      color: 0x3399ff,
      transparent: true,
      opacity: 0,
      emissive: 0x1166aa,
      emissiveIntensity: 0
    });
    
    this.focusPoint = new THREE.Mesh(geometry, material);
    this.focusPoint.position.copy(this.FOCUS_POSITION);
    this.group.add(this.focusPoint);
  }
  
  private createConvergenceField(): void {
    // Field effect around the focus point
    const geometry = new THREE.SphereGeometry(0.3, 32, 24);
    const material = new THREE.MeshBasicMaterial({
      color: 0x3399ff,
      transparent: true,
      opacity: 0,
      wireframe: true
    });
    
    this.convergenceField = new THREE.Mesh(geometry, material);
    this.convergenceField.position.copy(this.FOCUS_POSITION);
    this.group.add(this.convergenceField);
  }
  
  private createBeamParticles(): void {
    // Create particles that converge toward the focus point
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const phases = new Float32Array(this.PARTICLE_COUNT);
    const directions = new Float32Array(this.PARTICLE_COUNT * 3);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Start particles in a spherical distribution around the focus
      const phi = Math.random() * Math.PI * 2;
      const theta = Math.random() * Math.PI;
      const radius = 2 + Math.random() * 3;
      
      positions[i * 3] = Math.sin(theta) * Math.cos(phi) * radius;
      positions[i * 3 + 1] = Math.sin(theta) * Math.sin(phi) * radius;
      positions[i * 3 + 2] = Math.cos(theta) * radius;
      
      // Direction toward focus point
      const direction = new THREE.Vector3(
        -positions[i * 3],
        -positions[i * 3 + 1],
        -positions[i * 3 + 2]
      ).normalize();
      
      directions[i * 3] = direction.x;
      directions[i * 3 + 1] = direction.y;
      directions[i * 3 + 2] = direction.z;
      
      phases[i] = Math.random() * Math.PI * 2;
    }
    
    this.beamParticles = this.effects.createParticleSystem('convergent', positions, {
      phase: phases,
      direction: directions
    });
    
    this.group.add(this.beamParticles);
  }
  
  private createDataStreams(): void {
    // Create focused data streams converging to the center
    for (let i = 0; i < this.STREAM_COUNT; i++) {
      const angle = (i / this.STREAM_COUNT) * Math.PI * 2;
      const radius = 4;
      
      const startPoint = new THREE.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        (Math.random() - 0.5) * 2
      );
      
      const endPoint = this.FOCUS_POSITION.clone();
      
      // Create curved path toward focus
      const curve = new THREE.CatmullRomCurve3([
        startPoint,
        startPoint.clone().lerp(endPoint, 0.33).add(
          new THREE.Vector3(
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5,
            (Math.random() - 0.5) * 0.5
          )
        ),
        startPoint.clone().lerp(endPoint, 0.66).add(
          new THREE.Vector3(
            (Math.random() - 0.5) * 0.3,
            (Math.random() - 0.5) * 0.3,
            (Math.random() - 0.5) * 0.3
          )
        ),
        endPoint
      ]);
      
      const points = curve.getPoints(50);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      
      const material = new THREE.LineBasicMaterial({
        color: 0x3399ff,
        transparent: true,
        opacity: 0,
        linewidth: 2
      });
      
      const line = new THREE.Line(geometry, material);
      this.dataStreams.push(line);
      this.group.add(line);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update focus point
    const focusMaterial = this.focusPoint.material as THREE.MeshBasicMaterial;
    focusMaterial.opacity = activation;
    focusMaterial.emissiveIntensity = intensity;
    
    // Update convergence field
    const fieldMaterial = this.convergenceField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = activation * 0.3;
    
    // Update data streams
    this.dataStreams.forEach((stream, index) => {
      const material = stream.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.7;
    });
    
    // Update particle system
    if (this.beamParticles) {
      const material = this.beamParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.convergenceProgress = 0;
    
    // Fade out all elements
    const focusMaterial = this.focusPoint.material as THREE.MeshBasicMaterial;
    focusMaterial.opacity = 0;
    focusMaterial.emissiveIntensity = 0;
    
    const fieldMaterial = this.convergenceField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = 0;
    
    this.dataStreams.forEach(stream => {
      const material = stream.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    if (this.beamParticles) {
      const material = this.beamParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update convergence progress
    this.convergenceProgress = Math.min(1, this.convergenceProgress + 0.02);
    
    // Animate focus point
    this.animateFocusPoint(time);
    
    // Animate convergence field
    this.animateConvergenceField(time);
    
    // Animate data streams
    this.animateDataStreams(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateFocusPoint(time: number): void {
    // Pulsing effect based on convergence
    const pulseScale = 1 + Math.sin(time * 4) * 0.3 * this.convergenceProgress;
    this.focusPoint.scale.setScalar(pulseScale);
    
    // Color intensity based on convergence
    const material = this.focusPoint.material as THREE.MeshBasicMaterial;
    material.emissiveIntensity = this.intensity * (0.5 + this.convergenceProgress * 0.5);
  }
  
  private animateConvergenceField(time: number): void {
    // Slow rotation
    this.convergenceField.rotation.y = time * 0.2;
    
    // Scale based on convergence
    const scale = 1 - this.convergenceProgress * 0.3;
    this.convergenceField.scale.setScalar(scale);
    
    // Opacity pulses with convergence
    const material = this.convergenceField.material as THREE.MeshBasicMaterial;
    const basePulse = Math.sin(time * 2) * 0.1 + 0.1;
    material.opacity = this.activation * 0.3 + basePulse * this.convergenceProgress;
  }
  
  private animateDataStreams(time: number): void {
    this.dataStreams.forEach((stream, index) => {
      const material = stream.material as THREE.LineBasicMaterial;
      
      // Flowing light effect along streams
      const flowPhase = time * 2 + index * 0.5;
      const flowIntensity = Math.sin(flowPhase) * 0.3 + 0.7;
      material.opacity = this.activation * 0.7 * flowIntensity;
      
      // Color shift during convergence
      const convergenceColor = this.convergenceProgress;
      material.color.setRGB(
        0.2 + convergenceColor * 0.8,  // More white as convergence increases
        0.6 + convergenceColor * 0.4,
        1.0
      );
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Focus point decay
    const focusMaterial = this.focusPoint.material as THREE.MeshBasicMaterial;
    focusMaterial.opacity = currentActivation;
    focusMaterial.emissiveIntensity = currentIntensity * (0.5 + this.convergenceProgress * 0.5);
    
    // Convergence field decay
    const fieldMaterial = this.convergenceField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = currentActivation * 0.3;
    
    // Data streams decay
    this.dataStreams.forEach(stream => {
      const material = stream.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.7;
    });
    
    // Particle system decay
    if (this.beamParticles) {
      const material = this.beamParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'convergent',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      convergenceProgress: this.convergenceProgress,
      particleCount: this.PARTICLE_COUNT,
      streamCount: this.STREAM_COUNT
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.focusPoint.geometry.dispose();
    (this.focusPoint.material as THREE.Material).dispose();
    
    this.convergenceField.geometry.dispose();
    (this.convergenceField.material as THREE.Material).dispose();
    
    this.dataStreams.forEach(stream => {
      stream.geometry.dispose();
      (stream.material as THREE.Material).dispose();
    });
    
    if (this.beamParticles) {
      this.beamParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}