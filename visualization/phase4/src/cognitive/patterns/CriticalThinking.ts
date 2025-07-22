import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class CriticalThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private collisionParticles: THREE.Points | null = null;
  private argumentSpheres: THREE.Mesh[] = [];
  private contradictionZones: THREE.Mesh[] = [];
  private analysisBeams: THREE.Line[] = [];
  private evaluationRings: THREE.Mesh[] = [];
  private validationField: THREE.Mesh;
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private collisionEnergy: number = 0;
  private analysisDepth: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 800;
  private readonly ARGUMENT_COUNT = 6;
  private readonly ANALYSIS_BEAMS = 8;
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'CriticalThinking';
    this.group.position.set(3, -2, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createValidationField();
    this.createArgumentSpheres();
    this.createContradictionZones();
    this.createCollisionParticles();
    this.createAnalysisBeams();
    this.createEvaluationRings();
  }
  
  private createValidationField(): void {
    // Background field representing the validation space
    const geometry = new THREE.SphereGeometry(3, 32, 24);
    const material = new THREE.MeshBasicMaterial({
      color: 0xff6633,
      transparent: true,
      opacity: 0,
      wireframe: true,
      side: THREE.BackSide
    });
    
    this.validationField = new THREE.Mesh(geometry, material);
    this.validationField.position.copy(this.CENTER_POSITION);
    this.group.add(this.validationField);
  }
  
  private createArgumentSpheres(): void {
    // Create spheres representing different arguments/ideas being evaluated
    for (let i = 0; i < this.ARGUMENT_COUNT; i++) {
      const angle = (i / this.ARGUMENT_COUNT) * Math.PI * 2;
      const radius = 2;
      
      const geometry = new THREE.SphereGeometry(0.15, 16, 12);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff6633,
        transparent: true,
        opacity: 0,
        emissive: 0xaa3311,
        emissiveIntensity: 0
      });
      
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(
        Math.cos(angle) * radius,
        (Math.random() - 0.5) * 2,
        Math.sin(angle) * radius
      );
      
      // Store initial velocity for collision simulation
      sphere.userData = {
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02,
          (Math.random() - 0.5) * 0.02
        ),
        mass: 1 + Math.random()
      };
      
      this.argumentSpheres.push(sphere);
      this.group.add(sphere);
    }
  }
  
  private createContradictionZones(): void {
    // Create zones where contradictions are highlighted
    for (let i = 0; i < 4; i++) {
      const geometry = new THREE.OctahedronGeometry(0.3);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff3333,
        transparent: true,
        opacity: 0,
        emissive: 0x661111,
        emissiveIntensity: 0,
        wireframe: true
      });
      
      const zone = new THREE.Mesh(geometry, material);
      zone.position.set(
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 3
      );
      
      this.contradictionZones.push(zone);
      this.group.add(zone);
    }
  }
  
  private createCollisionParticles(): void {
    // Create particles for collision and analysis effects
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const collisionPhases = new Float32Array(this.PARTICLE_COUNT);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Distribute particles around collision zones
      const radius = 1 + Math.random() * 2.5;
      const phi = Math.random() * Math.PI * 2;
      const theta = Math.random() * Math.PI;
      
      positions[i * 3] = Math.sin(theta) * Math.cos(phi) * radius;
      positions[i * 3 + 1] = Math.sin(theta) * Math.sin(phi) * radius;
      positions[i * 3 + 2] = Math.cos(theta) * radius;
      
      collisionPhases[i] = Math.random() * Math.PI * 2;
    }
    
    this.collisionParticles = this.effects.createParticleSystem('critical', positions, {
      collisionPhase: collisionPhases
    });
    
    this.group.add(this.collisionParticles);
  }
  
  private createAnalysisBeams(): void {
    // Create beams that represent analytical examination
    for (let i = 0; i < this.ANALYSIS_BEAMS; i++) {
      const angle = (i / this.ANALYSIS_BEAMS) * Math.PI * 2;
      const beamLength = 4;
      
      const startPoint = new THREE.Vector3(0, 0, 0);
      const endPoint = new THREE.Vector3(
        Math.cos(angle) * beamLength,
        (Math.sin(i) * 0.5),
        Math.sin(angle) * beamLength
      );
      
      const geometry = new THREE.BufferGeometry().setFromPoints([startPoint, endPoint]);
      const material = new THREE.LineBasicMaterial({
        color: 0xff6633,
        transparent: true,
        opacity: 0,
        linewidth: 2
      });
      
      const beam = new THREE.Line(geometry, material);
      this.analysisBeams.push(beam);
      this.group.add(beam);
    }
  }
  
  private createEvaluationRings(): void {
    // Create rings that represent evaluation processes
    for (let i = 0; i < 3; i++) {
      const radius = 1 + i * 0.5;
      const geometry = new THREE.RingGeometry(radius - 0.05, radius + 0.05, 32);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff6633,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
      });
      
      const ring = new THREE.Mesh(geometry, material);
      ring.position.copy(this.CENTER_POSITION);
      ring.rotation.x = Math.PI / 2 + i * 0.3;
      ring.rotation.z = i * 0.5;
      
      this.evaluationRings.push(ring);
      this.group.add(ring);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update validation field
    const fieldMaterial = this.validationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = activation * 0.1;
    
    // Update argument spheres
    this.argumentSpheres.forEach((sphere, index) => {
      const material = sphere.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (0.7 + Math.random() * 0.3);
      material.emissiveIntensity = intensity * 0.5;
    });
    
    // Update contradiction zones
    this.contradictionZones.forEach(zone => {
      const material = zone.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.6;
      material.emissiveIntensity = intensity * 0.4;
    });
    
    // Update analysis beams
    this.analysisBeams.forEach(beam => {
      const material = beam.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.8;
    });
    
    // Update evaluation rings
    this.evaluationRings.forEach(ring => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.5;
    });
    
    // Update particle system
    if (this.collisionParticles) {
      const material = this.collisionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.collisionEnergy = 0;
    this.analysisDepth = 0;
    
    // Fade out all elements
    const fieldMaterial = this.validationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = 0;
    
    this.argumentSpheres.forEach(sphere => {
      const material = sphere.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.contradictionZones.forEach(zone => {
      const material = zone.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.analysisBeams.forEach(beam => {
      const material = beam.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.evaluationRings.forEach(ring => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
    });
    
    if (this.collisionParticles) {
      const material = this.collisionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update collision energy
    this.collisionEnergy = Math.min(1, this.collisionEnergy + 0.04);
    
    // Update analysis depth
    this.analysisDepth = Math.min(1, this.analysisDepth + 0.03);
    
    // Animate validation field
    this.animateValidationField(time);
    
    // Animate argument spheres (collision simulation)
    this.animateArgumentSpheres(time);
    
    // Animate contradiction zones
    this.animateContradictionZones(time);
    
    // Animate analysis beams
    this.animateAnalysisBeams(time);
    
    // Animate evaluation rings
    this.animateEvaluationRings(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateValidationField(time: number): void {
    // Pulsing validation space
    const pulseScale = 1 + Math.sin(time * 1.5) * 0.1 * this.intensity;
    this.validationField.scale.setScalar(pulseScale);
    
    // Rotation for dynamic analysis
    this.validationField.rotation.y = time * 0.1;
    this.validationField.rotation.x = Math.sin(time * 0.3) * 0.2;
  }
  
  private animateArgumentSpheres(time: number): void {
    // Simulate argument collision and interaction
    for (let i = 0; i < this.argumentSpheres.length; i++) {
      const sphere = this.argumentSpheres[i];
      const velocity = sphere.userData.velocity;
      
      // Apply velocity
      sphere.position.add(velocity.clone().multiplyScalar(this.collisionEnergy));
      
      // Boundary collision (critical analysis bounds)
      const boundaryRadius = 2.5;
      const distance = sphere.position.length();
      if (distance > boundaryRadius) {
        const normal = sphere.position.clone().normalize();
        velocity.reflect(normal).multiplyScalar(0.8); // Energy loss
        sphere.position.normalize().multiplyScalar(boundaryRadius);
      }
      
      // Inter-sphere collisions (argument conflicts)
      for (let j = i + 1; j < this.argumentSpheres.length; j++) {
        const other = this.argumentSpheres[j];
        const difference = sphere.position.clone().sub(other.position);
        const distance = difference.length();
        const minDistance = 0.4; // Minimum separation
        
        if (distance < minDistance) {
          // Collision response
          const normal = difference.normalize();
          const relativeVelocity = velocity.clone().sub(other.userData.velocity);
          const velocityAlongNormal = relativeVelocity.dot(normal);
          
          if (velocityAlongNormal > 0) continue; // Objects separating
          
          // Collision impulse
          const impulse = 2 * velocityAlongNormal / (sphere.userData.mass + other.userData.mass);
          velocity.add(normal.clone().multiplyScalar(-impulse * other.userData.mass));
          other.userData.velocity.add(normal.clone().multiplyScalar(impulse * sphere.userData.mass));
          
          // Create contradiction zone at collision point
          this.createCollisionEffect(sphere.position.clone().lerp(other.position, 0.5));
        }
      }
      
      // Argument evaluation pulsing
      const evaluationPhase = time * 3 + i * 0.8;
      const evaluationIntensity = Math.sin(evaluationPhase) * 0.3 + 1;
      sphere.scale.setScalar(evaluationIntensity);
      
      // Color shift based on collision energy
      const material = sphere.material as THREE.MeshBasicMaterial;
      const energyLevel = velocity.length() * 10;
      material.color.setRGB(
        1.0,
        0.4 - energyLevel * 0.2,
        0.2 - energyLevel * 0.1
      );
      material.emissiveIntensity = this.intensity * (0.5 + energyLevel * 0.3);
      
      // Damping
      velocity.multiplyScalar(0.995);
    }
  }
  
  private createCollisionEffect(position: THREE.Vector3): void {
    // Find nearest contradiction zone and activate it
    let nearestZone = this.contradictionZones[0];
    let nearestDistance = position.distanceTo(nearestZone.position);
    
    for (let i = 1; i < this.contradictionZones.length; i++) {
      const distance = position.distanceTo(this.contradictionZones[i].position);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestZone = this.contradictionZones[i];
      }
    }
    
    // Move zone to collision point and activate
    nearestZone.position.copy(position);
    const material = nearestZone.material as THREE.MeshBasicMaterial;
    material.emissiveIntensity = 1.0;
  }
  
  private animateContradictionZones(time: number): void {
    this.contradictionZones.forEach((zone, index) => {
      // Contradiction highlighting
      const highlightPhase = time * 4 + index * 1.2;
      const highlight = Math.sin(highlightPhase) * 0.5 + 0.5;
      
      const scale = 1 + highlight * 0.5 * this.collisionEnergy;
      zone.scale.setScalar(scale);
      
      // Rotation indicating analytical scrutiny
      zone.rotation.x = time * 1.5 + index;
      zone.rotation.y = time * 1.2 + index * 0.7;
      
      // Material intensity
      const material = zone.material as THREE.MeshBasicMaterial;
      material.emissiveIntensity = this.intensity * highlight * this.collisionEnergy;
      
      // Gradual fade of collision effects
      if (material.emissiveIntensity > this.intensity * 0.5) {
        material.emissiveIntensity *= 0.95;
      }
    });
  }
  
  private animateAnalysisBeams(time: number): void {
    this.analysisBeams.forEach((beam, index) => {
      const material = beam.material as THREE.LineBasicMaterial;
      
      // Scanning beam effect
      const scanPhase = time * 2 + index * 0.4;
      const scanIntensity = Math.sin(scanPhase) * 0.5 + 0.5;
      material.opacity = this.activation * 0.8 * scanIntensity * this.analysisDepth;
      
      // Beam rotation for comprehensive analysis
      const rotationPhase = time * 0.5 + index * 0.3;
      const geometry = beam.geometry as THREE.BufferGeometry;
      const positions = geometry.attributes.position.array as Float32Array;
      
      // Rotate end point
      const angle = (index / this.ANALYSIS_BEAMS) * Math.PI * 2 + rotationPhase;
      positions[3] = Math.cos(angle) * 4;
      positions[4] = Math.sin(rotationPhase) * 2;
      positions[5] = Math.sin(angle) * 4;
      
      geometry.attributes.position.needsUpdate = true;
      
      // Color intensity based on analysis depth
      material.color.setRGB(
        1.0,
        0.4 + this.analysisDepth * 0.2,
        0.2 + this.analysisDepth * 0.1
      );
    });
  }
  
  private animateEvaluationRings(time: number): void {
    this.evaluationRings.forEach((ring, index) => {
      // Ring rotation at different speeds
      ring.rotation.z += 0.01 * (index + 1);
      ring.rotation.x += 0.005 * (index + 1);
      
      // Size pulsing based on evaluation intensity
      const evaluationPhase = time * 1.8 + index * 0.6;
      const evaluationPulse = Math.sin(evaluationPhase) * 0.2 + 1;
      ring.scale.setScalar(evaluationPulse * (1 + this.analysisDepth * 0.3));
      
      // Material opacity waves
      const material = ring.material as THREE.MeshBasicMaterial;
      const opacityWave = Math.sin(time * 2.5 + index * 0.8) * 0.3 + 0.5;
      material.opacity = this.activation * 0.5 * opacityWave;
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Validation field decay
    const fieldMaterial = this.validationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = currentActivation * 0.1;
    
    // Argument spheres decay
    this.argumentSpheres.forEach(sphere => {
      const material = sphere.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (0.7 + Math.random() * 0.3);
      material.emissiveIntensity = currentIntensity * 0.5;
    });
    
    // Contradiction zones decay
    this.contradictionZones.forEach(zone => {
      const material = zone.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.6;
      material.emissiveIntensity = currentIntensity * 0.4;
    });
    
    // Analysis beams decay
    this.analysisBeams.forEach(beam => {
      const material = beam.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.8;
    });
    
    // Evaluation rings decay
    this.evaluationRings.forEach(ring => {
      const material = ring.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.5;
    });
    
    // Particle system decay
    if (this.collisionParticles) {
      const material = this.collisionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'critical',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      collisionEnergy: this.collisionEnergy,
      analysisDepth: this.analysisDepth,
      argumentCount: this.ARGUMENT_COUNT
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.validationField.geometry.dispose();
    (this.validationField.material as THREE.Material).dispose();
    
    this.argumentSpheres.forEach(sphere => {
      sphere.geometry.dispose();
      (sphere.material as THREE.Material).dispose();
    });
    
    this.contradictionZones.forEach(zone => {
      zone.geometry.dispose();
      (zone.material as THREE.Material).dispose();
    });
    
    this.analysisBeams.forEach(beam => {
      beam.geometry.dispose();
      (beam.material as THREE.Material).dispose();
    });
    
    this.evaluationRings.forEach(ring => {
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
    });
    
    if (this.collisionParticles) {
      this.collisionParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}