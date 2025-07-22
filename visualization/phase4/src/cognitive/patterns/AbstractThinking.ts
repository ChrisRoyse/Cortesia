import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class AbstractThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private patternParticles: THREE.Points | null = null;
  private abstractionLayers: THREE.Mesh[] = [];
  private patternNodes: THREE.Mesh[] = [];
  private conceptualLinks: THREE.Line[] = [];
  private emergentPatterns: THREE.Mesh[] = [];
  private abstractionField: THREE.Mesh;
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private abstractionLevel: number = 0;
  private patternComplexity: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 1000;
  private readonly ABSTRACTION_LAYERS = 4;
  private readonly PATTERN_NODES = 15;
  private readonly MAX_PATTERNS = 8;
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'AbstractThinking';
    this.group.position.set(0, -3, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createAbstractionField();
    this.createAbstractionLayers();
    this.createPatternNodes();
    this.createPatternParticles();
    this.createConceptualLinks();
    this.createEmergentPatterns();
  }
  
  private createAbstractionField(): void {
    // Background field representing the abstract thinking space
    const geometry = new THREE.TorusGeometry(3, 0.8, 16, 100);
    const material = new THREE.MeshBasicMaterial({
      color: 0x9966ff,
      transparent: true,
      opacity: 0,
      wireframe: true
    });
    
    this.abstractionField = new THREE.Mesh(geometry, material);
    this.abstractionField.position.copy(this.CENTER_POSITION);
    this.abstractionField.rotation.x = Math.PI / 2;
    this.group.add(this.abstractionField);
  }
  
  private createAbstractionLayers(): void {
    // Create layers representing different levels of abstraction
    for (let i = 0; i < this.ABSTRACTION_LAYERS; i++) {
      const radius = 1.5 + i * 0.8;
      const segments = 6 + i * 2;
      
      const geometry = new THREE.RingGeometry(radius - 0.1, radius + 0.1, segments);
      const material = new THREE.MeshBasicMaterial({
        color: 0x9966ff,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
      });
      
      const layer = new THREE.Mesh(geometry, material);
      layer.position.copy(this.CENTER_POSITION);
      layer.position.y = i * 0.3 - 0.6;
      layer.rotation.x = Math.PI / 2;
      layer.rotation.z = i * 0.4;
      
      this.abstractionLayers.push(layer);
      this.group.add(layer);
    }
  }
  
  private createPatternNodes(): void {
    // Create nodes representing discovered patterns
    for (let i = 0; i < this.PATTERN_NODES; i++) {
      const phi = Math.random() * Math.PI * 2;
      const theta = Math.random() * Math.PI;
      const radius = 1.5 + Math.random() * 2;
      
      const geometry = new THREE.OctahedronGeometry(0.08 + Math.random() * 0.04);
      const material = new THREE.MeshBasicMaterial({
        color: 0x9966ff,
        transparent: true,
        opacity: 0,
        emissive: 0x443377,
        emissiveIntensity: 0
      });
      
      const node = new THREE.Mesh(geometry, material);
      node.position.set(
        Math.sin(theta) * Math.cos(phi) * radius,
        Math.sin(theta) * Math.sin(phi) * radius,
        Math.cos(theta) * radius
      );
      
      // Store pattern metadata
      node.userData = {
        patternType: Math.floor(Math.random() * 5),
        complexity: Math.random(),
        confidence: Math.random(),
        connections: []
      };
      
      this.patternNodes.push(node);
      this.group.add(node);
    }
  }
  
  private createPatternParticles(): void {
    // Create particles for pattern detection visualization
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const patternPhases = new Float32Array(this.PARTICLE_COUNT);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Distribute particles in pattern-like formations
      const patternType = Math.floor(Math.random() * 3);
      
      switch (patternType) {
        case 0: // Spiral pattern
          const spiralAngle = (i / this.PARTICLE_COUNT) * Math.PI * 8;
          const spiralRadius = (i / this.PARTICLE_COUNT) * 3;
          positions[i * 3] = Math.cos(spiralAngle) * spiralRadius;
          positions[i * 3 + 1] = (i / this.PARTICLE_COUNT - 0.5) * 4;
          positions[i * 3 + 2] = Math.sin(spiralAngle) * spiralRadius;
          break;
          
        case 1: // Fractal-like distribution
          const fractalScale = Math.random() * 2 + 1;
          positions[i * 3] = (Math.random() - 0.5) * fractalScale;
          positions[i * 3 + 1] = (Math.random() - 0.5) * fractalScale;
          positions[i * 3 + 2] = (Math.random() - 0.5) * fractalScale;
          break;
          
        case 2: // Grid-like pattern with noise
          const gridSize = 8;
          const gridX = (i % gridSize) / gridSize - 0.5;
          const gridZ = Math.floor(i / gridSize) % gridSize / gridSize - 0.5;
          positions[i * 3] = gridX * 4 + (Math.random() - 0.5) * 0.3;
          positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
          positions[i * 3 + 2] = gridZ * 4 + (Math.random() - 0.5) * 0.3;
          break;
      }
      
      patternPhases[i] = Math.random() * Math.PI * 2;
    }
    
    this.patternParticles = this.effects.createParticleSystem('abstract', positions, {
      patternPhase: patternPhases
    });
    
    this.group.add(this.patternParticles);
  }
  
  private createConceptualLinks(): void {
    // Create links between pattern nodes
    for (let i = 0; i < this.PATTERN_NODES; i++) {
      const node1 = this.patternNodes[i];
      
      // Connect to 2-4 other nodes
      const connectionCount = 2 + Math.floor(Math.random() * 3);
      for (let j = 0; j < connectionCount; j++) {
        const targetIndex = Math.floor(Math.random() * this.PATTERN_NODES);
        if (targetIndex !== i) {
          const node2 = this.patternNodes[targetIndex];
          
          // Create curved conceptual link
          const start = node1.position.clone();
          const end = node2.position.clone();
          const mid = start.clone().lerp(end, 0.5);
          mid.add(new THREE.Vector3(
            (Math.random() - 0.5) * 0.8,
            (Math.random() - 0.5) * 0.8,
            (Math.random() - 0.5) * 0.8
          ));
          
          const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
          const points = curve.getPoints(20);
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          
          const material = new THREE.LineBasicMaterial({
            color: 0x9966ff,
            transparent: true,
            opacity: 0,
            linewidth: 1
          });
          
          const link = new THREE.Line(geometry, material);
          link.userData = {
            source: i,
            target: targetIndex,
            strength: Math.random()
          };
          
          this.conceptualLinks.push(link);
          this.group.add(link);
        }
      }
    }
  }
  
  private createEmergentPatterns(): void {
    // Create visual representations of emergent patterns
    for (let i = 0; i < this.MAX_PATTERNS; i++) {
      const patternType = Math.floor(Math.random() * 4);
      let geometry: THREE.BufferGeometry;
      
      switch (patternType) {
        case 0: // Geometric pattern
          geometry = new THREE.TetrahedronGeometry(0.2);
          break;
        case 1: // Complex pattern
          geometry = new THREE.DodecahedronGeometry(0.15);
          break;
        case 2: // Organic pattern
          geometry = new THREE.IcosahedronGeometry(0.18);
          break;
        default: // Abstract pattern
          geometry = new THREE.OctahedronGeometry(0.16);
      }
      
      const material = new THREE.MeshBasicMaterial({
        color: 0xaa66ff,
        transparent: true,
        opacity: 0,
        wireframe: true,
        emissive: 0x332255,
        emissiveIntensity: 0
      });
      
      const pattern = new THREE.Mesh(geometry, material);
      pattern.position.set(
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 4
      );
      
      pattern.userData = {
        type: patternType,
        emergenceTime: 0,
        confidence: 0
      };
      
      this.emergentPatterns.push(pattern);
      this.group.add(pattern);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update abstraction field
    const fieldMaterial = this.abstractionField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = activation * 0.2;
    
    // Update abstraction layers
    this.abstractionLayers.forEach((layer, index) => {
      const material = layer.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (0.3 - index * 0.05);
    });
    
    // Update pattern nodes
    this.patternNodes.forEach(node => {
      const material = node.material as THREE.MeshBasicMaterial;
      material.opacity = activation * (0.6 + node.userData.confidence * 0.4);
      material.emissiveIntensity = intensity * node.userData.confidence;
    });
    
    // Update conceptual links
    this.conceptualLinks.forEach(link => {
      const material = link.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.4 * link.userData.strength;
    });
    
    // Update emergent patterns
    this.emergentPatterns.forEach(pattern => {
      const material = pattern.material as THREE.MeshBasicMaterial;
      material.opacity = activation * pattern.userData.confidence * 0.7;
      material.emissiveIntensity = intensity * pattern.userData.confidence * 0.5;
    });
    
    // Update particle system
    if (this.patternParticles) {
      const material = this.patternParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.abstractionLevel = 0;
    this.patternComplexity = 0;
    
    // Fade out all elements
    const fieldMaterial = this.abstractionField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = 0;
    
    this.abstractionLayers.forEach(layer => {
      const material = layer.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
    });
    
    this.patternNodes.forEach(node => {
      const material = node.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.conceptualLinks.forEach(link => {
      const material = link.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.emergentPatterns.forEach(pattern => {
      const material = pattern.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    if (this.patternParticles) {
      const material = this.patternParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update abstraction level
    this.abstractionLevel = Math.min(1, this.abstractionLevel + 0.02);
    
    // Update pattern complexity
    this.patternComplexity = Math.min(1, this.patternComplexity + 0.025);
    
    // Animate abstraction field
    this.animateAbstractionField(time);
    
    // Animate abstraction layers
    this.animateAbstractionLayers(time);
    
    // Animate pattern nodes
    this.animatePatternNodes(time);
    
    // Animate conceptual links
    this.animateConceptualLinks(time);
    
    // Animate emergent patterns
    this.animateEmergentPatterns(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateAbstractionField(time: number): void {
    // Torus rotation representing continuous abstraction
    this.abstractionField.rotation.y = time * 0.2;
    this.abstractionField.rotation.z = Math.sin(time * 0.3) * 0.3;
    
    // Scale pulsing
    const scale = 1 + Math.sin(time * 1.2) * 0.1 * this.abstractionLevel;
    this.abstractionField.scale.setScalar(scale);
    
    // Material opacity waves
    const material = this.abstractionField.material as THREE.MeshBasicMaterial;
    const wave = Math.sin(time * 0.8) * 0.1 + 0.1;
    material.opacity = this.activation * 0.2 + wave * this.abstractionLevel;
  }
  
  private animateAbstractionLayers(time: number): void {
    this.abstractionLayers.forEach((layer, index) => {
      // Different rotation speeds for each layer
      layer.rotation.z += 0.005 * (index + 1) * this.abstractionLevel;
      
      // Vertical oscillation
      const oscillation = Math.sin(time * 0.6 + index * 1.2) * 0.2;
      layer.position.y = index * 0.3 - 0.6 + oscillation * this.abstractionLevel;
      
      // Opacity based on abstraction level
      const material = layer.material as THREE.MeshBasicMaterial;
      const layerActivation = Math.max(0, this.abstractionLevel - index * 0.2);
      material.opacity = this.activation * (0.3 - index * 0.05) * layerActivation;
    });
  }
  
  private animatePatternNodes(time: number): void {
    this.patternNodes.forEach((node, index) => {
      // Pattern recognition pulsing
      const recognitionPhase = time * 2.5 + index * 0.4;
      const recognition = Math.sin(recognitionPhase) * 0.4 + 0.6;
      
      // Update confidence over time (pattern becomes clearer)
      node.userData.confidence = Math.min(1, node.userData.confidence + 0.01 * this.patternComplexity);
      
      // Scale based on pattern confidence
      const confidenceScale = 1 + node.userData.confidence * recognition * 0.5;
      node.scale.setScalar(confidenceScale);
      
      // Orbital motion around abstraction center
      const orbitPhase = time * 0.3 + index * 0.6;
      const orbitRadius = 0.1 * node.userData.confidence;
      node.position.x += Math.cos(orbitPhase) * orbitRadius * 0.02;
      node.position.z += Math.sin(orbitPhase) * orbitRadius * 0.02;
      
      // Color shift based on pattern type and confidence
      const material = node.material as THREE.MeshBasicMaterial;
      const hue = (0.7 + node.userData.patternType * 0.1) % 1;
      material.color.setHSL(hue, 0.8, 0.5 + node.userData.confidence * 0.3);
      material.emissiveIntensity = this.intensity * node.userData.confidence;
    });
  }
  
  private animateConceptualLinks(time: number): void {
    this.conceptualLinks.forEach((link, index) => {
      const material = link.material as THREE.LineBasicMaterial;
      
      // Link strength pulsing
      const strengthPhase = time * 1.8 + index * 0.3;
      const strengthWave = Math.sin(strengthPhase) * 0.3 + 0.7;
      
      // Opacity based on connected nodes' confidence
      const sourceNode = this.patternNodes[link.userData.source];
      const targetNode = this.patternNodes[link.userData.target];
      const linkConfidence = (sourceNode.userData.confidence + targetNode.userData.confidence) * 0.5;
      
      material.opacity = this.activation * 0.4 * link.userData.strength * strengthWave * linkConfidence;
      
      // Color flow representing conceptual connection
      const flowPhase = time * 2 + index * 0.5;
      const flow = Math.sin(flowPhase) * 0.4 + 0.6;
      material.color.setRGB(
        0.6 + flow * 0.3,
        0.4 + flow * 0.2,
        1.0
      );
    });
  }
  
  private animateEmergentPatterns(time: number): void {
    this.emergentPatterns.forEach((pattern, index) => {
      // Pattern emergence animation
      pattern.userData.emergenceTime += 0.02 * this.patternComplexity;
      pattern.userData.confidence = Math.min(1, pattern.userData.emergenceTime);
      
      if (pattern.userData.confidence > 0.1) {
        // Rotation indicating pattern formation
        const rotationSpeed = 0.02 * (pattern.userData.type + 1);
        pattern.rotation.x += rotationSpeed;
        pattern.rotation.y += rotationSpeed * 0.7;
        pattern.rotation.z += rotationSpeed * 0.5;
        
        // Scale emergence
        const emergencePhase = time * 1.5 + index * 0.8;
        const emergence = Math.sin(emergencePhase) * 0.3 + 0.7;
        const scale = pattern.userData.confidence * emergence;
        pattern.scale.setScalar(scale);
        
        // Material updates
        const material = pattern.material as THREE.MeshBasicMaterial;
        material.opacity = this.activation * pattern.userData.confidence * 0.7;
        material.emissiveIntensity = this.intensity * pattern.userData.confidence * 0.5;
        
        // Color evolution as pattern emerges
        const evolutionHue = (0.7 + pattern.userData.type * 0.1 + pattern.userData.confidence * 0.2) % 1;
        material.color.setHSL(evolutionHue, 0.9, 0.6);
      }
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Abstraction field decay
    const fieldMaterial = this.abstractionField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = currentActivation * 0.2;
    
    // Abstraction layers decay
    this.abstractionLayers.forEach((layer, index) => {
      const material = layer.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (0.3 - index * 0.05);
    });
    
    // Pattern nodes decay
    this.patternNodes.forEach(node => {
      const material = node.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * (0.6 + node.userData.confidence * 0.4);
      material.emissiveIntensity = currentIntensity * node.userData.confidence;
    });
    
    // Conceptual links decay
    this.conceptualLinks.forEach(link => {
      const material = link.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.4 * link.userData.strength;
    });
    
    // Emergent patterns decay
    this.emergentPatterns.forEach(pattern => {
      const material = pattern.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * pattern.userData.confidence * 0.7;
      material.emissiveIntensity = currentIntensity * pattern.userData.confidence * 0.5;
    });
    
    // Particle system decay
    if (this.patternParticles) {
      const material = this.patternParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'abstract',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      abstractionLevel: this.abstractionLevel,
      patternComplexity: this.patternComplexity,
      detectedPatterns: this.emergentPatterns.filter(p => p.userData.confidence > 0.5).length
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.abstractionField.geometry.dispose();
    (this.abstractionField.material as THREE.Material).dispose();
    
    this.abstractionLayers.forEach(layer => {
      layer.geometry.dispose();
      (layer.material as THREE.Material).dispose();
    });
    
    this.patternNodes.forEach(node => {
      node.geometry.dispose();
      (node.material as THREE.Material).dispose();
    });
    
    this.conceptualLinks.forEach(link => {
      link.geometry.dispose();
      (link.material as THREE.Material).dispose();
    });
    
    this.emergentPatterns.forEach(pattern => {
      pattern.geometry.dispose();
      (pattern.material as THREE.Material).dispose();
    });
    
    if (this.patternParticles) {
      this.patternParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}