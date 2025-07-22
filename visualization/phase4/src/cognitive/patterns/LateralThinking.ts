import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class LateralThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private connectionParticles: THREE.Points | null = null;
  private lateralHubs: THREE.Mesh[] = [];
  private connectionLines: THREE.Line[] = [];
  private bridgeEffects: THREE.Mesh[] = [];
  private crossLinkingField: THREE.Mesh;
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private connectionStrength: number = 0;
  private crossLinkCount: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 600;
  private readonly HUB_COUNT = 8;
  private readonly MAX_CONNECTIONS = 20;
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'LateralThinking';
    this.group.position.set(-3, -2, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createCrossLinkingField();
    this.createLateralHubs();
    this.createConnectionParticles();
    this.createConnectionLines();
    this.createBridgeEffects();
  }
  
  private createCrossLinkingField(): void {
    // Background field showing the lateral connection space
    const geometry = new THREE.PlaneGeometry(6, 4);
    const material = new THREE.MeshBasicMaterial({
      color: 0xcc66ff,
      transparent: true,
      opacity: 0,
      wireframe: true,
      side: THREE.DoubleSide
    });
    
    this.crossLinkingField = new THREE.Mesh(geometry, material);
    this.crossLinkingField.position.copy(this.CENTER_POSITION);
    this.crossLinkingField.rotation.x = Math.PI / 4;
    this.group.add(this.crossLinkingField);
  }
  
  private createLateralHubs(): void {
    // Create hubs representing different domains/concepts
    for (let i = 0; i < this.HUB_COUNT; i++) {
      const angle = (i / this.HUB_COUNT) * Math.PI * 2;
      const radius = 2.5;
      
      const geometry = new THREE.SphereGeometry(0.12, 12, 8);
      const material = new THREE.MeshBasicMaterial({
        color: 0xcc66ff,
        transparent: true,
        opacity: 0,
        emissive: 0x664488,
        emissiveIntensity: 0
      });
      
      const hub = new THREE.Mesh(geometry, material);
      hub.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius * 0.5,
        (Math.random() - 0.5) * 1.5
      );
      
      this.lateralHubs.push(hub);
      this.group.add(hub);
    }
  }
  
  private createConnectionParticles(): void {
    // Create particles that flow between lateral connections
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const connectionStrengths = new Float32Array(this.PARTICLE_COUNT);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Position particles along potential connection paths
      const hubIndex1 = Math.floor(Math.random() * this.HUB_COUNT);
      const hubIndex2 = Math.floor(Math.random() * this.HUB_COUNT);
      
      if (hubIndex1 !== hubIndex2 && this.lateralHubs[hubIndex1] && this.lateralHubs[hubIndex2]) {
        const pos1 = this.lateralHubs[hubIndex1].position;
        const pos2 = this.lateralHubs[hubIndex2].position;
        const t = Math.random();
        
        positions[i * 3] = pos1.x + (pos2.x - pos1.x) * t;
        positions[i * 3 + 1] = pos1.y + (pos2.y - pos1.y) * t;
        positions[i * 3 + 2] = pos1.z + (pos2.z - pos1.z) * t;
      } else {
        positions[i * 3] = (Math.random() - 0.5) * 4;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
      }
      
      connectionStrengths[i] = Math.random();
    }
    
    this.connectionParticles = this.effects.createParticleSystem('lateral', positions, {
      connectionStrength: connectionStrengths
    });
    
    this.group.add(this.connectionParticles);
  }
  
  private createConnectionLines(): void {
    // Create dynamic connection lines between hubs
    for (let i = 0; i < this.MAX_CONNECTIONS; i++) {
      const hub1Index = Math.floor(Math.random() * this.HUB_COUNT);
      const hub2Index = Math.floor(Math.random() * this.HUB_COUNT);
      
      if (hub1Index !== hub2Index) {
        const pos1 = this.lateralHubs[hub1Index].position.clone();
        const pos2 = this.lateralHubs[hub2Index].position.clone();
        
        // Create curved lateral connection
        const midPoint = pos1.clone().lerp(pos2, 0.5);
        midPoint.add(new THREE.Vector3(
          (Math.random() - 0.5) * 1.5,
          (Math.random() - 0.5) * 1.5,
          (Math.random() - 0.5) * 1
        ));
        
        const curve = new THREE.QuadraticBezierCurve3(pos1, midPoint, pos2);
        const points = curve.getPoints(20);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        const material = new THREE.LineBasicMaterial({
          color: 0xcc66ff,
          transparent: true,
          opacity: 0,
          linewidth: 1.5
        });
        
        const line = new THREE.Line(geometry, material);
        this.connectionLines.push(line);
        this.group.add(line);
      }
    }
  }
  
  private createBridgeEffects(): void {
    // Create visual bridges representing lateral connections
    for (let i = 0; i < 6; i++) {
      const geometry = new THREE.CylinderGeometry(0.02, 0.02, 1.5, 8);
      const material = new THREE.MeshBasicMaterial({
        color: 0xcc66ff,
        transparent: true,
        opacity: 0,
        emissive: 0x442266,
        emissiveIntensity: 0
      });
      
      const bridge = new THREE.Mesh(geometry, material);
      
      // Position randomly between potential connection points
      const angle = (i / 6) * Math.PI * 2;
      bridge.position.set(
        Math.cos(angle) * 1.5,
        Math.sin(angle) * 0.8,
        (Math.random() - 0.5) * 1
      );
      
      bridge.rotation.z = angle + Math.PI / 2;
      
      this.bridgeEffects.push(bridge);
      this.group.add(bridge);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update cross-linking field
    const fieldMaterial = this.crossLinkingField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = activation * 0.1;
    
    // Update lateral hubs
    this.lateralHubs.forEach((hub, index) => {
      const material = hub.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (0.6 + Math.random() * 0.4);
      material.emissiveIntensity = intensity * 0.4;
    });
    
    // Update connection lines
    this.connectionLines.forEach((line, index) => {
      const material = line.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.6 * Math.random();
    });
    
    // Update bridge effects
    this.bridgeEffects.forEach((bridge, index) => {
      const material = bridge.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.5;
      material.emissiveIntensity = intensity * 0.3;
    });
    
    // Update particle system
    if (this.connectionParticles) {
      const material = this.connectionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.connectionStrength = 0;
    this.crossLinkCount = 0;
    
    // Fade out all elements
    const fieldMaterial = this.crossLinkingField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = 0;
    
    this.lateralHubs.forEach(hub => {
      const material = hub.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.connectionLines.forEach(line => {
      const material = line.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.bridgeEffects.forEach(bridge => {
      const material = bridge.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    if (this.connectionParticles) {
      const material = this.connectionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update connection strength
    this.connectionStrength = Math.min(1, this.connectionStrength + 0.03);
    
    // Update cross-link count
    this.crossLinkCount = Math.min(this.MAX_CONNECTIONS, this.crossLinkCount + 0.5);
    
    // Animate cross-linking field
    this.animateCrossLinkingField(time);
    
    // Animate lateral hubs
    this.animateLateralHubs(time);
    
    // Animate connection lines
    this.animateConnectionLines(time);
    
    // Animate bridge effects
    this.animateBridgeEffects(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateCrossLinkingField(time: number): void {
    // Wave-like deformation across the field
    this.crossLinkingField.rotation.y = Math.sin(time * 0.5) * 0.2;
    this.crossLinkingField.rotation.z = Math.cos(time * 0.3) * 0.1;
    
    // Opacity waves
    const material = this.crossLinkingField.material as THREE.MeshBasicMaterial;
    const wave = Math.sin(time * 1.5) * 0.05 + 0.05;
    material.opacity = this.activation * 0.1 + wave;
  }
  
  private animateLateralHubs(time: number): void {
    this.lateralHubs.forEach((hub, index) => {
      // Orbital motion suggesting different domains
      const orbitPhase = time * 0.3 + index * 0.8;
      const orbitRadius = 0.3;
      
      hub.position.x += Math.cos(orbitPhase) * orbitRadius * 0.02;
      hub.position.y += Math.sin(orbitPhase) * orbitRadius * 0.02;
      
      // Pulsing based on connection activity
      const pulsePhase = time * 2 + index * 0.5;
      const pulse = Math.sin(pulsePhase) * 0.3 + 1;
      hub.scale.setScalar(pulse * (1 + this.connectionStrength * 0.2));
      
      // Color shifts representing different domains
      const material = hub.material as THREE.MeshBasicMaterial;
      const colorPhase = time * 0.8 + index * 1.2;
      material.color.setHSL(
        (0.7 + Math.sin(colorPhase) * 0.1) % 1, // Purple range with variation
        0.8,
        0.6
      );
    });
  }
  
  private animateConnectionLines(time: number): void {
    this.connectionLines.forEach((line, index) => {
      if (index < this.crossLinkCount) {
        const material = line.material as THREE.LineBasicMaterial;
        
        // Connection strength waves
        const strengthPhase = time * 2.5 + index * 0.4;
        const strength = Math.sin(strengthPhase) * 0.4 + 0.6;
        material.opacity = this.activation * 0.6 * strength;
        
        // Color flow representing idea transfer
        const flowPhase = time * 1.8 + index * 0.3;
        const flow = Math.sin(flowPhase) * 0.3 + 0.7;
        material.color.setRGB(
          0.8 + flow * 0.2,
          0.4 + flow * 0.4,
          1.0
        );
      }
    });
  }
  
  private animateBridgeEffects(time: number): void {
    this.bridgeEffects.forEach((bridge, index) => {
      // Bridge activation sequence
      const activationPhase = time * 1.5 + index * 0.6;
      const activation = Math.sin(activationPhase) * 0.5 + 0.5;
      
      const material = bridge.material as THREE.MeshBasicMaterial;
      material.opacity = this.activation * 0.5 * activation;
      material.emissiveIntensity = this.intensity * 0.3 * activation;
      
      // Bridge extension/contraction
      const extensionPhase = time * 0.8 + index * 0.4;
      const extension = Math.sin(extensionPhase) * 0.2 + 1;
      bridge.scale.y = extension;
      
      // Bridge orientation shifts (lateral thinking jumps)
      if (Math.sin(time + index) > 0.95) {
        bridge.rotation.z += (Math.random() - 0.5) * 0.5;
      }
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Cross-linking field decay
    const fieldMaterial = this.crossLinkingField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = currentActivation * 0.1;
    
    // Lateral hubs decay
    this.lateralHubs.forEach(hub => {
      const material = hub.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (0.6 + Math.random() * 0.4);
      material.emissiveIntensity = currentIntensity * 0.4;
    });
    
    // Connection lines decay
    this.connectionLines.forEach(line => {
      const material = line.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.6;
    });
    
    // Bridge effects decay
    this.bridgeEffects.forEach(bridge => {
      const material = bridge.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.5;
      material.emissiveIntensity = currentIntensity * 0.3;
    });
    
    // Particle system decay
    if (this.connectionParticles) {
      const material = this.connectionParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'lateral',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      connectionStrength: this.connectionStrength,
      crossLinkCount: this.crossLinkCount,
      hubCount: this.HUB_COUNT
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.crossLinkingField.geometry.dispose();
    (this.crossLinkingField.material as THREE.Material).dispose();
    
    this.lateralHubs.forEach(hub => {
      hub.geometry.dispose();
      (hub.material as THREE.Material).dispose();
    });
    
    this.connectionLines.forEach(line => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    });
    
    this.bridgeEffects.forEach(bridge => {
      bridge.geometry.dispose();
      (bridge.material as THREE.Material).dispose();
    });
    
    if (this.connectionParticles) {
      this.connectionParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}