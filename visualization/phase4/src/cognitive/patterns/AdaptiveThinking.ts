import * as THREE from 'three';
import { PatternEffects } from '../PatternEffects';

export class AdaptiveThinking {
  private scene: THREE.Scene;
  private effects: PatternEffects;
  private group: THREE.Group;
  
  // Visual elements
  private adaptiveParticles: THREE.Points | null = null;
  private metaCognitiveCores: THREE.Mesh[] = [];
  private switchingPaths: THREE.Line[] = [];
  private patternSelectors: THREE.Mesh[] = [];
  private adaptationField: THREE.Mesh;
  private strategyNodes: THREE.Mesh[] = [];
  private feedbackLoops: THREE.Line[] = [];
  
  // Animation state
  private isActive: boolean = false;
  private activation: number = 0;
  private intensity: number = 0;
  private adaptationRate: number = 0;
  private metaCognitiveLevel: number = 0;
  private showMetaCognitive: boolean = false;
  
  // Pattern switching state
  private currentPattern: number = 0;
  private switchingProgress: number = 0;
  private targetPattern: number = 0;
  private switchCooldown: number = 0;
  
  // Configuration
  private readonly PARTICLE_COUNT = 1200;
  private readonly STRATEGY_COUNT = 7; // One for each cognitive pattern
  private readonly META_CORES = 3;
  private readonly SWITCHING_PATHS = 12;
  private readonly CENTER_POSITION = new THREE.Vector3(0, 0, 0);
  
  constructor(scene: THREE.Scene, effects: PatternEffects) {
    this.scene = scene;
    this.effects = effects;
    
    this.group = new THREE.Group();
    this.group.name = 'AdaptiveThinking';
    this.group.position.set(0, 0, 0); // Central position for meta-cognitive control
    this.scene.add(this.group);
    
    this.initializeVisuals();
  }
  
  private initializeVisuals(): void {
    this.createAdaptationField();
    this.createMetaCognitiveCores();
    this.createStrategyNodes();
    this.createAdaptiveParticles();
    this.createSwitchingPaths();
    this.createPatternSelectors();
    this.createFeedbackLoops();
  }
  
  private createAdaptationField(): void {
    // Field representing the adaptive thinking space
    const geometry = new THREE.SphereGeometry(4, 32, 24);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffcc33,
      transparent: true,
      opacity: 0,
      wireframe: true,
      side: THREE.BackSide
    });
    
    this.adaptationField = new THREE.Mesh(geometry, material);
    this.adaptationField.position.copy(this.CENTER_POSITION);
    this.group.add(this.adaptationField);
  }
  
  private createMetaCognitiveCores(): void {
    // Central cores managing meta-cognitive processes
    for (let i = 0; i < this.META_CORES; i++) {
      const angle = (i / this.META_CORES) * Math.PI * 2;
      const radius = 0.8;
      
      const geometry = new THREE.SphereGeometry(0.25, 16, 12);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffcc33,
        transparent: true,
        opacity: 0,
        emissive: 0xaa8800,
        emissiveIntensity: 0
      });
      
      const core = new THREE.Mesh(geometry, material);
      core.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle * 0.5) * 0.3,
        Math.sin(angle) * radius
      );
      
      this.metaCognitiveCores.push(core);
      this.group.add(core);
    }
  }
  
  private createStrategyNodes(): void {
    // Nodes representing different cognitive strategies
    const strategies = [
      { name: 'Convergent', color: 0x3399ff, position: new THREE.Vector3(-2.5, 1.5, 0) },
      { name: 'Divergent', color: 0xff4d80, position: new THREE.Vector3(2.5, 1.5, 0) },
      { name: 'Lateral', color: 0xcc66ff, position: new THREE.Vector3(-2.5, -1.5, 0) },
      { name: 'Systems', color: 0x33ff66, position: new THREE.Vector3(0, 2.5, 0) },
      { name: 'Critical', color: 0xff6633, position: new THREE.Vector3(2.5, -1.5, 0) },
      { name: 'Abstract', color: 0x9966ff, position: new THREE.Vector3(0, -2.5, 0) },
      { name: 'Meta', color: 0xffcc33, position: new THREE.Vector3(0, 0, 0) }
    ];
    
    strategies.forEach((strategy, index) => {
      const geometry = new THREE.CylinderGeometry(0.12, 0.12, 0.4, 8);
      const material = new THREE.MeshBasicMaterial({
        color: strategy.color,
        transparent: true,
        opacity: 0,
        emissive: new THREE.Color(strategy.color).multiplyScalar(0.5),
        emissiveIntensity: 0
      });
      
      const node = new THREE.Mesh(geometry, material);
      node.position.copy(strategy.position);
      node.userData = {
        strategyName: strategy.name,
        index: index,
        activation: 0,
        selected: index === 0 // Start with convergent
      };
      
      this.strategyNodes.push(node);
      this.group.add(node);
    });
  }
  
  private createAdaptiveParticles(): void {
    // Particles that demonstrate adaptive switching
    const positions = new Float32Array(this.PARTICLE_COUNT * 3);
    const switchPhases = new Float32Array(this.PARTICLE_COUNT);
    
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      // Start particles in a dynamic formation
      const strategyIndex = Math.floor(Math.random() * this.STRATEGY_COUNT);
      const strategyNode = this.strategyNodes[strategyIndex];
      
      if (strategyNode) {
        const offset = new THREE.Vector3(
          (Math.random() - 0.5) * 0.8,
          (Math.random() - 0.5) * 0.8,
          (Math.random() - 0.5) * 0.8
        );
        
        const position = strategyNode.position.clone().add(offset);
        positions[i * 3] = position.x;
        positions[i * 3 + 1] = position.y;
        positions[i * 3 + 2] = position.z;
      } else {
        positions[i * 3] = (Math.random() - 0.5) * 4;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 4;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 4;
      }
      
      switchPhases[i] = Math.random() * Math.PI * 2;
    }
    
    this.adaptiveParticles = this.effects.createParticleSystem('adaptive', positions, {
      switchPhase: switchPhases
    });
    
    this.group.add(this.adaptiveParticles);
  }
  
  private createSwitchingPaths(): void {
    // Paths between different cognitive strategies
    for (let i = 0; i < this.SWITCHING_PATHS; i++) {
      const startIndex = Math.floor(Math.random() * this.STRATEGY_COUNT);
      const endIndex = Math.floor(Math.random() * this.STRATEGY_COUNT);
      
      if (startIndex !== endIndex) {
        const startPos = this.strategyNodes[startIndex].position.clone();
        const endPos = this.strategyNodes[endIndex].position.clone();
        
        // Create curved switching path
        const midPoint = startPos.clone().lerp(endPos, 0.5);
        midPoint.add(new THREE.Vector3(
          (Math.random() - 0.5) * 1.5,
          (Math.random() - 0.5) * 1.5,
          (Math.random() - 0.5) * 1.5
        ));
        
        const curve = new THREE.QuadraticBezierCurve3(startPos, midPoint, endPos);
        const points = curve.getPoints(25);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        const material = new THREE.LineBasicMaterial({
          color: 0xffcc33,
          transparent: true,
          opacity: 0,
          linewidth: 2
        });
        
        const path = new THREE.Line(geometry, material);
        path.userData = {
          startStrategy: startIndex,
          endStrategy: endIndex,
          switchStrength: Math.random()
        };
        
        this.switchingPaths.push(path);
        this.group.add(path);
      }
    }
  }
  
  private createPatternSelectors(): void {
    // Visual selectors showing active pattern choice
    for (let i = 0; i < 3; i++) {
      const geometry = new THREE.RingGeometry(1.2 + i * 0.3, 1.4 + i * 0.3, 16);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffcc33,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
      });
      
      const selector = new THREE.Mesh(geometry, material);
      selector.position.copy(this.CENTER_POSITION);
      selector.rotation.x = Math.PI / 2;
      
      this.patternSelectors.push(selector);
      this.group.add(selector);
    }
  }
  
  private createFeedbackLoops(): void {
    // Feedback loops for adaptive learning
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI * 2;
      const radius = 3;
      
      const startPoint = new THREE.Vector3(
        Math.cos(angle) * radius,
        0,
        Math.sin(angle) * radius
      );
      
      const endPoint = new THREE.Vector3(
        Math.cos(angle + Math.PI) * radius * 0.8,
        1,
        Math.sin(angle + Math.PI) * radius * 0.8
      );
      
      // Create spiral feedback path
      const points: THREE.Vector3[] = [];
      for (let j = 0; j <= 30; j++) {
        const t = j / 30;
        const spiralAngle = angle + t * Math.PI * 3;
        const spiralRadius = radius * (1 - t * 0.2);
        const height = t * 1.5 - 0.75;
        
        points.push(new THREE.Vector3(
          Math.cos(spiralAngle) * spiralRadius,
          height,
          Math.sin(spiralAngle) * spiralRadius
        ));
      }
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color: 0xffdd44,
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
    
    // Update adaptation field
    const fieldMaterial = this.adaptationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = activation * 0.1;
    
    // Update meta-cognitive cores
    this.metaCognitiveCores.forEach(core => {
      const material = core.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.8;
      material.emissiveIntensity = intensity * 0.6;
    });
    
    // Update strategy nodes
    this.strategyNodes.forEach((node, index) => {
      const material = node.material as THREE.MeshBasicMaterial;
      const isSelected = index === this.currentPattern;
      material.opacity = activation * (isSelected ? 1.0 : 0.4);
      material.emissiveIntensity = intensity * (isSelected ? 0.8 : 0.2);
    });
    
    // Update switching paths
    this.switchingPaths.forEach(path => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.3 * path.userData.switchStrength;
    });
    
    // Update pattern selectors
    this.patternSelectors.forEach(selector => {
      const material = selector.material as THREE.MeshBasicMaterial;
      material.opacity = activation * 0.4;
    });
    
    // Update feedback loops
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = activation * 0.3;
    });
    
    // Update particle system
    if (this.adaptiveParticles) {
      const material = this.adaptiveParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = intensity;
      if (material.uniforms.metaCognitive) {
        material.uniforms.metaCognitive.value = this.showMetaCognitive ? 1.0 : 0.0;
      }
    }
  }
  
  public deactivate(): void {
    this.isActive = false;
    this.adaptationRate = 0;
    this.metaCognitiveLevel = 0;
    
    // Fade out all elements
    const fieldMaterial = this.adaptationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = 0;
    
    this.metaCognitiveCores.forEach(core => {
      const material = core.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.strategyNodes.forEach(node => {
      const material = node.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
      material.emissiveIntensity = 0;
    });
    
    this.switchingPaths.forEach(path => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    this.patternSelectors.forEach(selector => {
      const material = selector.material as THREE.MeshBasicMaterial;
      material.opacity = 0;
    });
    
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = 0;
    });
    
    if (this.adaptiveParticles) {
      const material = this.adaptiveParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = 0;
    }
  }
  
  public showMetaCognitiveEffects(show: boolean): void {
    this.showMetaCognitive = show;
    if (this.adaptiveParticles) {
      const material = this.adaptiveParticles.material as THREE.ShaderMaterial;
      if (material.uniforms.metaCognitive) {
        material.uniforms.metaCognitive.value = show ? 1.0 : 0.0;
      }
    }
  }
  
  public update(time: number, decayFactor: number): void {
    if (!this.isActive) return;
    
    // Update adaptation rate
    this.adaptationRate = Math.min(1, this.adaptationRate + 0.03);
    
    // Update meta-cognitive level
    this.metaCognitiveLevel = Math.min(1, this.metaCognitiveLevel + 0.02);
    
    // Handle pattern switching
    this.updatePatternSwitching(time);
    
    // Animate adaptation field
    this.animateAdaptationField(time);
    
    // Animate meta-cognitive cores
    this.animateMetaCognitiveCores(time);
    
    // Animate strategy nodes
    this.animateStrategyNodes(time);
    
    // Animate switching paths
    this.animateSwitchingPaths(time);
    
    // Animate pattern selectors
    this.animatePatternSelectors(time);
    
    // Animate feedback loops
    this.animateFeedbackLoops(time);
    
    // Apply decay
    this.applyDecay(decayFactor);
  }
  
  private updatePatternSwitching(time: number): void {
    // Cooldown management
    if (this.switchCooldown > 0) {
      this.switchCooldown -= 0.02;
      return;
    }
    
    // Automatic pattern switching simulation
    if (Math.random() < 0.005 * this.adaptationRate) { // 0.5% chance per frame when fully adapted
      this.targetPattern = Math.floor(Math.random() * this.STRATEGY_COUNT);
      if (this.targetPattern !== this.currentPattern) {
        this.switchingProgress = 0;
        this.switchCooldown = 1.0; // 1 second cooldown
      }
    }
    
    // Handle switching animation
    if (this.switchingProgress < 1 && this.targetPattern !== this.currentPattern) {
      this.switchingProgress += 0.05;
      
      if (this.switchingProgress >= 1) {
        // Complete the switch
        this.currentPattern = this.targetPattern;
        this.switchingProgress = 0;
        
        // Update strategy node selection
        this.strategyNodes.forEach((node, index) => {
          node.userData.selected = index === this.currentPattern;
        });
      }
    }
  }
  
  private animateAdaptationField(time: number): void {
    // Field pulsing with adaptation rate
    const pulse = Math.sin(time * 1.5) * 0.1 + 1;
    const scale = pulse * (1 + this.adaptationRate * 0.2);
    this.adaptationField.scale.setScalar(scale);
    
    // Rotation showing dynamic adaptation
    this.adaptationField.rotation.y = time * 0.1 * this.adaptationRate;
    this.adaptationField.rotation.x = Math.sin(time * 0.3) * 0.1 * this.metaCognitiveLevel;
  }
  
  private animateMetaCognitiveCores(time: number): void {
    this.metaCognitiveCores.forEach((core, index) => {
      // Orbital motion around center
      const orbitPhase = time * 0.4 + index * (Math.PI * 2 / 3);
      const orbitRadius = 0.8 + Math.sin(time * 1.2 + index) * 0.2 * this.metaCognitiveLevel;
      
      core.position.x = Math.cos(orbitPhase) * orbitRadius;
      core.position.z = Math.sin(orbitPhase) * orbitRadius;
      core.position.y = Math.sin(orbitPhase * 0.5) * 0.3 * this.metaCognitiveLevel;
      
      // Pulsing with meta-cognitive activity
      const pulsePhase = time * 2.5 + index * 0.8;
      const pulse = Math.sin(pulsePhase) * 0.3 + 1;
      core.scale.setScalar(pulse * (1 + this.metaCognitiveLevel * 0.3));
      
      // Intensity based on meta-cognitive level
      const material = core.material as THREE.MeshBasicMaterial;
      material.emissiveIntensity = this.intensity * 0.6 * (0.7 + this.metaCognitiveLevel * 0.3);
    });
  }
  
  private animateStrategyNodes(time: number): void {
    this.strategyNodes.forEach((node, index) => {
      const isSelected = index === this.currentPattern;
      const isSwitching = this.switchingProgress > 0 && 
                         (index === this.currentPattern || index === this.targetPattern);
      
      // Activation animation
      let activation = isSelected ? 1.0 : 0.3;
      if (isSwitching) {
        const switchProgress = index === this.targetPattern ? this.switchingProgress : 1 - this.switchingProgress;
        activation = 0.3 + switchProgress * 0.7;
      }
      
      // Scale based on activation
      const scalePhase = time * 2 + index * 0.5;
      const scaleVariation = Math.sin(scalePhase) * 0.2 + 1;
      node.scale.setScalar(scaleVariation * (0.8 + activation * 0.5));
      
      // Rotation indicating strategy consideration
      node.rotation.y = time * 0.5 * activation;
      
      // Material updates
      const material = node.material as THREE.MeshBasicMaterial;
      material.opacity = this.activation * (0.4 + activation * 0.6);
      material.emissiveIntensity = this.intensity * (0.2 + activation * 0.6);
      
      // Color intensity during switching
      if (isSwitching) {
        const switchGlow = Math.sin(time * 8) * 0.3 + 0.7;
        material.emissiveIntensity *= switchGlow;
      }
    });
  }
  
  private animateSwitchingPaths(time: number): void {
    this.switchingPaths.forEach((path, index) => {
      const material = path.material as THREE.LineBasicMaterial;
      
      // Path activation based on switching activity
      const pathActive = (path.userData.startStrategy === this.currentPattern || 
                         path.userData.endStrategy === this.currentPattern) ||
                        (this.switchingProgress > 0 && 
                         (path.userData.startStrategy === this.targetPattern || 
                          path.userData.endStrategy === this.targetPattern));
      
      // Flow animation
      const flowPhase = time * 3 + index * 0.4;
      const flow = Math.sin(flowPhase) * 0.4 + 0.6;
      
      let opacity = this.activation * 0.3 * path.userData.switchStrength * flow;
      if (pathActive) {
        opacity *= 2; // Highlight active switching paths
      }
      
      material.opacity = Math.min(1, opacity);
      
      // Color shift during active switching
      if (pathActive && this.switchingProgress > 0) {
        material.color.setRGB(
          1.0,
          0.8 + Math.sin(time * 10) * 0.2,
          0.2 + Math.sin(time * 10) * 0.3
        );
      } else {
        material.color.setRGB(1.0, 0.8, 0.2);
      }
    });
  }
  
  private animatePatternSelectors(time: number): void {
    this.patternSelectors.forEach((selector, index) => {
      // Rotation at different speeds
      selector.rotation.z += 0.01 * (index + 1) * this.adaptationRate;
      
      // Scale pulsing
      const pulsePhase = time * 1.8 + index * 0.6;
      const pulse = Math.sin(pulsePhase) * 0.2 + 1;
      selector.scale.setScalar(pulse * (1 + this.metaCognitiveLevel * 0.2));
      
      // Opacity waves
      const material = selector.material as THREE.MeshBasicMaterial;
      const opacityWave = Math.sin(time * 1.2 + index * 0.8) * 0.2 + 0.4;
      material.opacity = this.activation * opacityWave;
      
      // Color shift with switching
      if (this.switchingProgress > 0) {
        const switchHue = (0.15 + this.switchingProgress * 0.1) % 1;
        material.color.setHSL(switchHue, 0.9, 0.6);
      } else {
        material.color.setRGB(1.0, 0.8, 0.2);
      }
    });
  }
  
  private animateFeedbackLoops(time: number): void {
    this.feedbackLoops.forEach((loop, index) => {
      const material = loop.material as THREE.LineBasicMaterial;
      
      // Feedback flow animation
      const feedbackPhase = time * 2.2 + index * 0.5;
      const feedback = Math.sin(feedbackPhase) * 0.4 + 0.6;
      
      material.opacity = this.activation * 0.3 * feedback * this.adaptationRate;
      
      // Color flow representing learning feedback
      const flowColor = 0.5 + feedback * 0.5;
      material.color.setRGB(
        1.0,
        0.8 + flowColor * 0.2,
        0.3 + flowColor * 0.2
      );
    });
  }
  
  private applyDecay(decayFactor: number): void {
    // Apply temporal decay to all visual elements
    const currentActivation = this.activation * decayFactor;
    const currentIntensity = this.intensity * decayFactor;
    
    // Adaptation field decay
    const fieldMaterial = this.adaptationField.material as THREE.MeshBasicMaterial;
    fieldMaterial.opacity = currentActivation * 0.1;
    
    // Meta-cognitive cores decay
    this.metaCognitiveCores.forEach(core => {
      const material = core.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.8;
      material.emissiveIntensity = currentIntensity * 0.6;
    });
    
    // Strategy nodes decay
    this.strategyNodes.forEach((node, index) => {
      const material = node.material as THREE.MeshBasicMaterial;
      const isSelected = index === this.currentPattern;
      material.opacity = currentActivation * (isSelected ? 1.0 : 0.4);
      material.emissiveIntensity = currentIntensity * (isSelected ? 0.8 : 0.2);
    });
    
    // Switching paths decay
    this.switchingPaths.forEach(path => {
      const material = path.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.3 * path.userData.switchStrength;
    });
    
    // Pattern selectors decay
    this.patternSelectors.forEach(selector => {
      const material = selector.material as THREE.MeshBasicMaterial;
      material.opacity = currentActivation * 0.4;
    });
    
    // Feedback loops decay
    this.feedbackLoops.forEach(loop => {
      const material = loop.material as THREE.LineBasicMaterial;
      material.opacity = currentActivation * 0.3;
    });
    
    // Particle system decay
    if (this.adaptiveParticles) {
      const material = this.adaptiveParticles.material as THREE.ShaderMaterial;
      material.uniforms.intensity.value = currentIntensity;
    }
    
    // If decay is too low, deactivate
    if (decayFactor < 0.1) {
      this.deactivate();
    }
  }
  
  public getVisualizationData(): any {
    return {
      type: 'adaptive',
      isActive: this.isActive,
      activation: this.activation,
      intensity: this.intensity,
      adaptationRate: this.adaptationRate,
      metaCognitiveLevel: this.metaCognitiveLevel,
      currentPattern: this.currentPattern,
      switchingProgress: this.switchingProgress,
      showMetaCognitive: this.showMetaCognitive
    };
  }
  
  public dispose(): void {
    // Dispose geometries and materials
    this.adaptationField.geometry.dispose();
    (this.adaptationField.material as THREE.Material).dispose();
    
    this.metaCognitiveCores.forEach(core => {
      core.geometry.dispose();
      (core.material as THREE.Material).dispose();
    });
    
    this.strategyNodes.forEach(node => {
      node.geometry.dispose();
      (node.material as THREE.Material).dispose();
    });
    
    this.switchingPaths.forEach(path => {
      path.geometry.dispose();
      (path.material as THREE.Material).dispose();
    });
    
    this.patternSelectors.forEach(selector => {
      selector.geometry.dispose();
      (selector.material as THREE.Material).dispose();
    });
    
    this.feedbackLoops.forEach(loop => {
      loop.geometry.dispose();
      (loop.material as THREE.Material).dispose();
    });
    
    if (this.adaptiveParticles) {
      this.adaptiveParticles.geometry.dispose();
    }
    
    // Remove from scene
    this.scene.remove(this.group);
  }
}