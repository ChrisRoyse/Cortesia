import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class SystemsThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private hierarchyParticles: THREE.Points | null = null;
  private rootNode: THREE.Mesh;
  private hierarchyLevels: THREE.Group[] = [];
  private inheritanceFlows: THREE.Line[] = [];
  private systemBoundaries: THREE.Mesh[] = [];
  private feedbackLoops: THREE.Line[] = [];
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private hierarchyDepth: number = 0;
  private flowStrength: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 900;
  private readonly MAX_LEVELS = 5;
  private readonly NODES_PER_LEVEL = [1, 3, 6, 12, 24];
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'SystemsThinking';
    this.group.position.set(0, 3, 0);
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createRootNode();
    this.createHierarchyLevels();
    this.createHierarchyParticles();
    this.createInheritanceFlows();
    this.createSystemBoundaries();
    this.createFeedbackLoops();
  }
  
  private createRootNode(): void {
    // Root node representing the top-level system
    const geometry = new THREE.SphereGeometry(0.2, 16, 12);
    const material = new THREE.MeshBasicMaterial({
      color: 0x33ff66,
      transparent: true,
      opacity: 0,
      emissive: 0x226633,
      emissiveIntensity: 0
    });
    
    this.rootNode = new THREE.Mesh(geometry, material);
    this.rootNode.position.copy(this.CENTER_POSITION);
    this.group.add(this.rootNode);
  }
  
  private createHierarchyLevels(): void {
    // Create hierarchical levels of the system
    for (let level = 0; level < this.MAX_LEVELS; level++) {
      const levelGroup = new THREE.Group();
      levelGroup.name = `Level${level}`;
      
      const nodeCount = this.NODES_PER_LEVEL[level];
      const levelHeight = -(level + 1) * 1.2;
      const radius = (level + 1) * 0.8;
      
      for (let i = 0; i < nodeCount; i++) {
        const angle = (i / nodeCount) * Math.PI * 2;
        
        const geometry = new THREE.SphereGeometry(
          0.08 + (this.MAX_LEVELS - level) * 0.02,
          12,
          8
        );
        const material = new THREE.MeshBasicMaterial({
          color: 0x33ff66,
          transparent: true,
          opacity: 0,
          emissive: 0x114422,
          emissiveIntensity: 0
        });
        
        const node = new THREE.Mesh(geometry, material);
        node.position.set(
          Math.cos(angle) * radius,
          levelHeight,
          Math.sin(angle) * radius
        );
        
        levelGroup.add(node);
      }
      
      this.hierarchyLevels.push(levelGroup);
      this.group.add(levelGroup);
    }
  }
  
  private createHierarchyParticles(): void {
    // Create particles that flow through the hierarchy
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const levels = new Float32Array(this.PARTICLE_COUNT);
    const branchIndices = new Float32Array(this.PARTICLE_COUNT);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      const level = Math.floor(Math.random() * this.MAX_LEVELS);
      const levelHeight = -(level + 1) * 1.2;
      const radius = (level + 1) * 0.8;
      const angle = Math.random() * Math.PI * 2;
      
      positions[i * 3] = Math.cos(angle) * radius * (0.5 + Math.random() * 0.5);
      positions[i * 3 + 1] = levelHeight + (Math.random() - 0.5) * 0.5;
      positions[i * 3 + 2] = Math.sin(angle) * radius * (0.5 + Math.random() * 0.5);
      
      levels[i] = level;
      branchIndices[i] = Math.floor(Math.random() * this.NODES_PER_LEVEL[level]);
    }
    
    this.hierarchyParticles = this.effects.createParticleSystem('systems', positions, {
      level: levels,
      branchIndex: branchIndices
    });
    
    this.group.add(this.hierarchyParticles);
  }
  
  private createInheritanceFlows(): void {
    // Create flows between hierarchical levels
    for (let level = 0; level < this.MAX_LEVELS - 1; level++) {
      const currentLevelHeight = -(level + 1) * 1.2;
      const nextLevelHeight = -(level + 2) * 1.2;
      const currentRadius = (level + 1) * 0.8;
      const nextRadius = (level + 2) * 0.8;
      
      const currentNodes = this.NODES_PER_LEVEL[level];
      const nextNodes = this.NODES_PER_LEVEL[level + 1];
      
      for (let i = 0; i < currentNodes; i++) {
        const currentAngle = (i / currentNodes) * Math.PI * 2;
        const currentPos = new THREE.Vector3(
          Math.cos(currentAngle) * currentRadius,
          currentLevelHeight,
          Math.sin(currentAngle) * currentRadius
        );
        
        // Connect to multiple child nodes
        const childrenPerNode = Math.ceil(nextNodes / currentNodes);
        for (let j = 0; j < childrenPerNode && i * childrenPerNode + j < nextNodes; j++) {
          const childIndex = i * childrenPerNode + j;
          const childAngle = (childIndex / nextNodes) * Math.PI * 2;
          const childPos = new THREE.Vector3(
            Math.cos(childAngle) * nextRadius,
            nextLevelHeight,
            Math.sin(childAngle) * nextRadius
          );
          
          // Create curved inheritance flow
          const midPoint = currentPos.clone().lerp(childPos, 0.5);
          midPoint.add(new THREE.Vector3(
            (Math.random() - 0.5) * 0.3,
            0,
            (Math.random() - 0.5) * 0.3
          ));
          
          const curve = new THREE.QuadraticBezierCurve3(currentPos, midPoint, childPos);
          const points = curve.getPoints(15);
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          
          const material = new THREE.LineBasicMaterial({
            color: 0x33ff66,
            transparent: true,
            opacity: 0,
            linewidth: 1
          });
          
          const line = new THREE.Line(geometry, material);
          this.inheritanceFlows.push(line);
          this.group.add(line);
        }
      }
    }
  }
  
  private createSystemBoundaries(): void {
    // Create boundaries for different system levels
    for (let level = 0; level < this.MAX_LEVELS; level++) {
      const radius = (level + 1) * 1.2;
      const height = 0.3;
      
      const geometry = new THREE.CylinderGeometry(radius, radius, height, 32, 1, true);
      const material = new THREE.MeshBasicMaterial({
        color: 0x33ff66,
        transparent: true,
        opacity: 0,
        wireframe: true,
        side: THREE.DoubleSide
      });
      
      const boundary = new THREE.Mesh(geometry, material);
      boundary.position.y = -(level + 1) * 1.2;
      
      this.systemBoundaries.push(boundary);
      this.group.add(boundary);
    }
  }
  
  private createFeedbackLoops(): void {
    // Create feedback loops between system levels
    for (let i = 0; i < 8; i++) {
      const startLevel = Math.floor(Math.random() * (this.MAX_LEVELS - 1));
      const endLevel = startLevel + 1 + Math.floor(Math.random() * (this.MAX_LEVELS - startLevel - 1));
      
      const startHeight = -(startLevel + 1) * 1.2;
      const endHeight = -(endLevel + 1) * 1.2;
      const startRadius = (startLevel + 1) * 0.8;
      const endRadius = (endLevel + 1) * 0.8;
      
      const angle = (i / 8) * Math.PI * 2;
      
      const startPos = new THREE.Vector3(
        Math.cos(angle) * startRadius,
        startHeight,
        Math.sin(angle) * startRadius
      );
      
      const endPos = new THREE.Vector3(
        Math.cos(angle + Math.PI) * endRadius,
        endHeight,
        Math.sin(angle + Math.PI) * endRadius
      );
      
      // Create curved feedback path
      const midPoint1 = startPos.clone().lerp(endPos, 0.33);
      const midPoint2 = startPos.clone().lerp(endPos, 0.66);
      midPoint1.add(new THREE.Vector3(2, 0.5, 0));
      midPoint2.add(new THREE.Vector3(-2, -0.5, 0));
      
      const curve = new THREE.CatmullRomCurve3([startPos, midPoint1, midPoint2, endPos]);
      const points = curve.getPoints(30);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      
      const material = new THREE.LineBasicMaterial({
        color: 0x66ff99,
        transparent: true,
        opacity: 0,
        linewidth: 1.5
      });
      
      const loop = new THREE.Line(geometry, material);
      this.feedbackLoops.push(loop);
      this.group.add(loop);
    }
  }
  
  public activate(activation: number, intensity: number, metadata?: any): void {
    this.isActive = true;
    this.activation = activation;
    this.intensity = intensity;
    
    // Update root node
    const rootMaterial = this.rootNode.material as THREE.MeshBasicMaterial;
    rootMaterial.opacity = activation;
    rootMaterial.emissiveIntensity = intensity;
    
    // Update hierarchy levels
    this.hierarchyLevels.forEach((levelGroup, levelIndex) => {
      levelGroup.children.forEach(node => {
        const material = (node as THREE.Mesh).material as THREE.MeshBasicMaterial;
        material.opacity = activation * (1 - levelIndex * 0.1);
        material.emissiveIntensity = intensity * (1 - levelIndex * 0.1);
      });
    });
    
    // Update inheritance flows
    this.inheritanceFlows.forEach(flow => {
      const material = flow.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.5;
    });
    
    // Update system boundaries
    this.systemBoundaries.forEach((boundary, index) => {
      const material = boundary.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.2 * (1 - index * 0.05);
    });
    
    // Update feedback loops
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.4;
    });
    
    // Update particle system
    if (this.hierarchyParticles) {
      const material = this.hierarchyParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.hierarchyDepth = 0;
    this.flowStrength = 0;
    
    // Fade out all elements
    const rootMaterial = this.rootNode.material as THREE.MeshBasicMaterial;
    rootMaterial.opacity = 0;
    rootMaterial.emissiveIntensity = 0;
    
    this.hierarchyLevels.forEach(levelGroup => {
      levelGroup.children.forEach(node => {
        const material = (node as THREE.Mesh).material as THREE.MeshBasicMaterial;
        material.opacity = 0;
        material.emissiveIntensity = 0;
      });
    });
    
    this.inheritanceFlows.forEach(flow => {
      const material = flow.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.systemBoundaries.forEach(boundary => {
      const material = boundary.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
    });
    
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    if (this.hierarchyParticles) {
      const material = this.hierarchyParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update hierarchy depth
    this.hierarchyDepth = Math.min(this.MAX_LEVELS, this.hierarchyDepth + 0.05);
    
    // Update flow strength
    this.flowStrength = Math.min(1, this.flowStrength + 0.02);
    
    // Animate root node
    this.animateRootNode(time);
    
    // Animate hierarchy levels
    this.animateHierarchyLevels(time);
    
    // Animate inheritance flows
    this.animateInheritanceFlows(time);
    
    // Animate system boundaries
    this.animateSystemBoundaries(time);
    
    // Animate feedback loops
    this.animateFeedbackLoops(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private animateRootNode(time: number): void {
    // Pulsing system core
    const pulseScale = 1 + Math.sin(time * 2) * 0.2 * this.intensity;
    this.rootNode.scale.setScalar(pulseScale);
    
    // Slow rotation
    this.rootNode.rotation.y = time * 0.3;
    
    // Brightness variation
    const material = this.rootNode.material as THREE.MeshBasicMaterial;
    material.emissiveIntensity = this.intensity * (0.8 + Math.sin(time * 1.5) * 0.2);
  }
  
  private animateHierarchyLevels(time: number): void {
    this.hierarchyLevels.forEach((levelGroup, levelIndex) => {
      if (levelIndex < this.hierarchyDepth) {
        // Level activation wave
        const activationPhase = time * 1.5 - levelIndex * 0.5;
        const activationWave = Math.sin(activationPhase) * 0.3 + 0.7;
        
        levelGroup.children.forEach((node, nodeIndex) => {
          const mesh = node as THREE.Mesh;
          
          // Node pulsing
          const pulsePhase = time * 2 + nodeIndex * 0.3;
          const pulse = Math.sin(pulsePhase) * 0.2 + 1;
          mesh.scale.setScalar(pulse * (1 + this.hierarchyDepth * 0.1));
          
          // Level-based color shift
          const material = mesh.material as THREE.MeshBasicMaterial;
          const colorShift = levelIndex / this.MAX_LEVELS;
          material.color.setRGB(
            0.2 + colorShift * 0.3,
            1.0 - colorShift * 0.3,
            0.4 + colorShift * 0.2
          );
          
          material.opacity = this.activation * activationWave * (1 - levelIndex * 0.1);
        });
      }
    });
  }
  
  private animateInheritanceFlows(time: number): void {
    this.inheritanceFlows.forEach((flow, index) => {
      const material = flow.material as THREE.LineBasicMaterial;
      
      // Flow wave animation
      const flowPhase = time * 2.5 + index * 0.2;
      const flowWave = Math.sin(flowPhase) * 0.4 + 0.6;
      material.opacity = this.activation * 0.5 * flowWave * this.flowStrength;
      
      // Color intensity based on flow
      const intensity = this.flowStrength * flowWave;
      material.color.setRGB(
        0.2 + intensity * 0.3,
        1.0,
        0.4 + intensity * 0.2
      );
    });
  }
  
  private animateSystemBoundaries(time: number): void {
    this.systemBoundaries.forEach((boundary, index) => {
      // Boundary pulsing
      const pulsePhase = time * 1.2 + index * 0.4;
      const pulse = Math.sin(pulsePhase) * 0.1 + 1;
      boundary.scale.setScalar(pulse);
      
      // Opacity waves
      const material = boundary.material as THREE.MeshBasicMaterial;
      const opacityWave = Math.sin(time * 0.8 + index * 0.3) * 0.1 + 0.1;
      material.opacity = this.activation * 0.2 * opacityWave;
    });
  }
  
  private animateFeedbackLoops(time: number): void {
    this.feedbackLoops.forEach((loop, index) => {
      const material = loop.material as THREE.LineBasicMaterial;
      
      // Feedback pulse animation
      const feedbackPhase = time * 1.8 + index * 0.6;
      const feedbackWave = Math.sin(feedbackPhase) * 0.5 + 0.5;
      material.opacity = this.activation * 0.4 * feedbackWave;
      
      // Color shift for feedback
      material.color.setRGB(
        0.4 + feedbackWave * 0.2,
        1.0,
        0.6 + feedbackWave * 0.3
      );
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Root node decay
    const rootMaterial = this.rootNode.material as THREE.MeshBasicMaterial;
    rootMaterial.opacity = currentActivation;
    rootMaterial.emissiveIntensity = currentIntensity;
    
    // Hierarchy levels decay
    this.hierarchyLevels.forEach((levelGroup, levelIndex) => {
      levelGroup.children.forEach(node => {
        const material = (node as THREE.Mesh).material as THREE.MeshBasicMaterial;
        material.opacity = currentActivation * (1 - levelIndex * 0.1);
        material.emissiveIntensity = currentIntensity * (1 - levelIndex * 0.1);
      });
    });
    
    // Inheritance flows decay
    this.inheritanceFlows.forEach(flow => {
      const material = flow.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.5;
    });
    
    // System boundaries decay
    this.systemBoundaries.forEach((boundary, index) => {
      const material = boundary.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.2 * (1 - index * 0.05);
    });
    
    // Feedback loops decay
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.4;
    });
    
    // Particle system decay
    if (this.hierarchyParticles) {
      const material = this.hierarchyParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'systems',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      hierarchyDepth: this.hierarchyDepth,
      flowStrength: this.flowStrength,
      levelCount: this.MAX_LEVELS
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.rootNode.geometry.dispose();
    (this.rootNode.material as THREE.Material).dispose();
    
    this.hierarchyLevels.forEach(levelGroup => {
      levelGroup.children.forEach(node => {
        const mesh = node as THREE.Mesh;
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
    });
    
    this.inheritanceFlows.forEach(flow => {
      flow.geometry.dispose();
      (flow.material as THREE.Material).dispose();
    });
    
    this.systemBoundaries.forEach(boundary => {
      boundary.geometry.dispose();
      (boundary.material as THREE.Material).dispose();
    });
    
    this.feedbackLoops.forEach(loop => {
      loop.geometry.dispose();
      (loop.material as THREE.Material).dispose();
    });
    
    if (this.hierarchyParticles) {
      this.hierarchyParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}