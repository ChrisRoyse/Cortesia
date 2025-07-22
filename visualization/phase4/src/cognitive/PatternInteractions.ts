import * as THREE from 'three';
import { CognitivePatternData } from './CognitivePatternVisualizer';

export interface InteractionEffect {
  type: 'inhibition' | 'enhancement' | 'resonance' | 'interference';
  strength: number;
  source: string;
  target: string;
  duration: number;
  visualElements: THREE.Object3D[];
}

export class PatternInteractions {
  private scene: THREE.Scene;
  private interactionGroup: THREE.Group;
  private activeInteractions: Map<string, InteractionEffect> = new Map();
  private connectionLines: Map<string, THREE.Line> = new Map();
  private interactionParticles: Map<string, THREE.Points> = new Map();
  
  // Interaction matrix positions for visualization
  private patternPositions = new Map([
    ['convergent', new THREE.Vector3(-3, 2, 0)],
    ['divergent', new THREE.Vector3(3, 2, 0)],
    ['lateral', new THREE.Vector3(-3, -2, 0)],
    ['systems', new THREE.Vector3(0, 3, 0)],
    ['critical', new THREE.Vector3(3, -2, 0)],
    ['abstract', new THREE.Vector3(0, -3, 0)],
    ['adaptive', new THREE.Vector3(0, 0, 0)]
  ]);
  
  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.interactionGroup = new THREE.Group();
    this.interactionGroup.name = 'PatternInteractions';
    this.scene.add(this.interactionGroup);
  }
  
  public updateInteractions(
    activePatterns: CognitivePatternData[],
    interactionMatrix: number[][]
  ): void {
    this.clearExpiredInteractions();
    
    // Create interactions between active patterns
    for (let i = 0; i < activePatterns.length; i++) {
      for (let j = i + 1; j < activePatterns.length; j++) {
        const pattern1 = activePatterns[i];
        const pattern2 = activePatterns[j];
        
        const interactionStrength = this.getInteractionStrength(
          pattern1.type,
          pattern2.type,
          interactionMatrix
        );
        
        if (Math.abs(interactionStrength) > 0.1) {
          this.createInteraction(pattern1, pattern2, interactionStrength);
        }
      }
    }
  }
  
  private getInteractionStrength(
    type1: string,
    type2: string,
    matrix: number[][]
  ): number {
    const typeIndices = {
      convergent: 0,
      divergent: 1,
      lateral: 2,
      systems: 3,
      critical: 4,
      abstract: 5,
      adaptive: 6
    };
    
    const index1 = typeIndices[type1];
    const index2 = typeIndices[type2];
    
    if (index1 !== undefined && index2 !== undefined) {
      return matrix[index1][index2];
    }
    
    return 0;
  }
  
  private createInteraction(
    pattern1: CognitivePatternData,
    pattern2: CognitivePatternData,
    strength: number
  ): void {
    const interactionKey = `${pattern1.type}-${pattern2.type}`;
    
    // Skip if interaction already exists
    if (this.activeInteractions.has(interactionKey)) {
      return;
    }
    
    const interactionType = this.determineInteractionType(pattern1.type, pattern2.type, strength);
    const visualElements = this.createInteractionVisual(pattern1, pattern2, interactionType, strength);
    
    const interaction: InteractionEffect = {
      type: interactionType,
      strength: Math.abs(strength),
      source: pattern1.type,
      target: pattern2.type,
      duration: 3000, // 3 seconds
      visualElements
    };
    
    this.activeInteractions.set(interactionKey, interaction);
  }
  
  private determineInteractionType(
    type1: string,
    type2: string,
    strength: number
  ): InteractionEffect['type'] {
    // Define interaction types based on cognitive pattern relationships
    const inhibitoryPairs = [
      ['convergent', 'divergent'],
      ['critical', 'abstract'],
      ['systems', 'lateral']
    ];
    
    const enhancementPairs = [
      ['abstract', 'systems'],
      ['lateral', 'divergent'],
      ['convergent', 'critical']
    ];
    
    const resonancePairs = [
      ['adaptive', 'systems'],
      ['adaptive', 'abstract'],
      ['convergent', 'systems']
    ];
    
    const pairKey = [type1, type2].sort().join('-');
    
    if (strength < 0) {
      return 'inhibition';
    }
    
    for (const [a, b] of inhibitoryPairs) {
      if (pairKey === [a, b].sort().join('-')) {
        return 'inhibition';
      }
    }
    
    for (const [a, b] of enhancementPairs) {
      if (pairKey === [a, b].sort().join('-')) {
        return 'enhancement';
      }
    }
    
    for (const [a, b] of resonancePairs) {
      if (pairKey === [a, b].sort().join('-')) {
        return 'resonance';
      }
    }
    
    return 'interference';
  }
  
  private createInteractionVisual(
    pattern1: CognitivePatternData,
    pattern2: CognitivePatternData,
    interactionType: InteractionEffect['type'],
    strength: number
  ): THREE.Object3D[] {
    const visualElements: THREE.Object3D[] = [];
    const pos1 = this.patternPositions.get(pattern1.type);
    const pos2 = this.patternPositions.get(pattern2.type);
    
    if (!pos1 || !pos2) return visualElements;
    
    // Create connection line
    const connectionLine = this.createConnectionLine(pos1, pos2, interactionType, strength);
    visualElements.push(connectionLine);
    
    // Create interaction particles
    const particles = this.createInteractionParticles(pos1, pos2, interactionType, strength);
    visualElements.push(particles);
    
    // Add specific effects based on interaction type
    switch (interactionType) {
      case 'inhibition':
        visualElements.push(...this.createInhibitionEffect(pos1, pos2, strength));
        break;
      case 'enhancement':
        visualElements.push(...this.createEnhancementEffect(pos1, pos2, strength));
        break;
      case 'resonance':
        visualElements.push(...this.createResonanceEffect(pos1, pos2, strength));
        break;
      case 'interference':
        visualElements.push(...this.createInterferenceEffect(pos1, pos2, strength));
        break;
    }
    
    // Add all elements to the scene
    visualElements.forEach(element => {
      this.interactionGroup.add(element);
    });
    
    return visualElements;
  }
  
  private createConnectionLine(
    pos1: THREE.Vector3,
    pos2: THREE.Vector3,
    type: InteractionEffect['type'],
    strength: number
  ): THREE.Line {
    const points = [pos1.clone(), pos2.clone()];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    const colors = {
      inhibition: 0xff3333,
      enhancement: 0x33ff33,
      resonance: 0x3366ff,
      interference: 0xffaa33
    };
    
    const material = new THREE.LineBasicMaterial({
      color: colors[type],
      transparent: true,
      opacity: Math.abs(strength),
      linewidth: 2 + Math.abs(strength) * 3
    });
    
    return new THREE.Line(geometry, material);
  }
  
  private createInteractionParticles(
    pos1: THREE.Vector3,
    pos2: THREE.Vector3,
    type: InteractionEffect['type'],
    strength: number
  ): THREE.Points {
    const particleCount = Math.floor(20 + Math.abs(strength) * 30);
    const positions = new Float32Array(particleCount * 3);
    
    // Create particles along the connection line
    for (let i = 0; i < particleCount; i++) {
      const t = i / (particleCount - 1);
      const position = pos1.clone().lerp(pos2, t);
      
      // Add some randomness
      position.add(new THREE.Vector3(
        (Math.random() - 0.5) * 0.2,
        (Math.random() - 0.5) * 0.2,
        (Math.random() - 0.5) * 0.2
      ));
      
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y;
      positions[i * 3 + 2] = position.z;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const colors = {
      inhibition: new THREE.Color(1.0, 0.2, 0.2),
      enhancement: new THREE.Color(0.2, 1.0, 0.2),
      resonance: new THREE.Color(0.2, 0.4, 1.0),
      interference: new THREE.Color(1.0, 0.7, 0.2)
    };
    
    const material = new THREE.PointsMaterial({
      color: colors[type],
      size: 2 + Math.abs(strength) * 2,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });
    
    return new THREE.Points(geometry, material);
  }
  
  private createInhibitionEffect(pos1: THREE.Vector3, pos2: THREE.Vector3, strength: number): THREE.Object3D[] {
    const effects: THREE.Object3D[] = [];
    
    // Create pulsing rings at midpoint
    const midpoint = pos1.clone().lerp(pos2, 0.5);
    const ringGeometry = new THREE.RingGeometry(0.1, 0.3, 16);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0xff4444,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide
    });
    
    for (let i = 0; i < 3; i++) {
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.position.copy(midpoint);
      ring.rotation.x = Math.random() * Math.PI;
      ring.rotation.y = Math.random() * Math.PI;
      effects.push(ring);
    }
    
    return effects;
  }
  
  private createEnhancementEffect(pos1: THREE.Vector3, pos2: THREE.Vector3, strength: number): THREE.Object3D[] {
    const effects: THREE.Object3D[] = [];
    
    // Create expanding energy waves
    const midpoint = pos1.clone().lerp(pos2, 0.5);
    const sphereGeometry = new THREE.SphereGeometry(0.2, 8, 6);
    const sphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x44ff44,
      transparent: true,
      opacity: 0.4,
      wireframe: true
    });
    
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.copy(midpoint);
    effects.push(sphere);
    
    return effects;
  }
  
  private createResonanceEffect(pos1: THREE.Vector3, pos2: THREE.Vector3, strength: number): THREE.Object3D[] {
    const effects: THREE.Object3D[] = [];
    
    // Create standing wave pattern
    const wavePoints: THREE.Vector3[] = [];
    const segments = 20;
    
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const point = pos1.clone().lerp(pos2, t);
      
      // Add wave displacement
      const waveHeight = Math.sin(t * Math.PI * 4) * 0.3 * strength;
      point.y += waveHeight;
      
      wavePoints.push(point);
    }
    
    const waveGeometry = new THREE.BufferGeometry().setFromPoints(wavePoints);
    const waveMaterial = new THREE.LineBasicMaterial({
      color: 0x4466ff,
      transparent: true,
      opacity: 0.8
    });
    
    const waveLine = new THREE.Line(waveGeometry, waveMaterial);
    effects.push(waveLine);
    
    return effects;
  }
  
  private createInterferenceEffect(pos1: THREE.Vector3, pos2: THREE.Vector3, strength: number): THREE.Object3D[] {
    const effects: THREE.Object3D[] = [];
    
    // Create chaotic particle field
    const particleCount = 50;
    const positions = new Float32Array(particleCount * 3);
    
    const midpoint = pos1.clone().lerp(pos2, 0.5);
    
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const radius = 0.5 + Math.random() * 0.5;
      
      positions[i * 3] = midpoint.x + Math.cos(angle) * radius;
      positions[i * 3 + 1] = midpoint.y + Math.sin(angle) * radius;
      positions[i * 3 + 2] = midpoint.z + (Math.random() - 0.5) * 0.5;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({
      color: 0xffaa44,
      size: 1.5,
      transparent: true,
      opacity: 0.6
    });
    
    const particles = new THREE.Points(geometry, material);
    effects.push(particles);
    
    return effects;
  }
  
  public update(time: number): void {
    // Animate interaction effects
    for (const [key, interaction] of this.activeInteractions) {
      this.animateInteraction(interaction, time);
    }
  }
  
  private animateInteraction(interaction: InteractionEffect, time: number): void {
    interaction.visualElements.forEach((element, index) => {
      switch (interaction.type) {
        case 'inhibition':
          this.animateInhibition(element, time, index);
          break;
        case 'enhancement':
          this.animateEnhancement(element, time, index);
          break;
        case 'resonance':
          this.animateResonance(element, time, index);
          break;
        case 'interference':
          this.animateInterference(element, time, index);
          break;
      }
    });
  }
  
  private animateInhibition(element: THREE.Object3D, time: number, index: number): void {
    if (element instanceof THREE.Mesh) {
      // Pulsing effect
      const scale = 1 + Math.sin(time * 3 + index) * 0.3;
      element.scale.setScalar(scale);
      
      // Opacity pulsing
      const material = element.material as THREE.MeshBasicMaterial;
      material.opacity = 0.3 + Math.sin(time * 2 + index) * 0.3;
    }
  }
  
  private animateEnhancement(element: THREE.Object3D, time: number, index: number): void {
    if (element instanceof THREE.Mesh) {
      // Expanding effect
      const scale = 1 + Math.sin(time * 2) * 0.5;
      element.scale.setScalar(scale);
      
      // Rotation
      element.rotation.y = time * 0.5;
    }
  }
  
  private animateResonance(element: THREE.Object3D, time: number, index: number): void {
    if (element instanceof THREE.Line) {
      // Update wave points
      const geometry = element.geometry as THREE.BufferGeometry;
      const positions = geometry.attributes.position.array as Float32Array;
      
      for (let i = 0; i < positions.length; i += 3) {
        const waveOffset = Math.sin(time * 2 + i * 0.1) * 0.2;
        positions[i + 1] += waveOffset * 0.1;
      }
      
      geometry.attributes.position.needsUpdate = true;
    }
  }
  
  private animateInterference(element: THREE.Object3D, time: number, index: number): void {
    if (element instanceof THREE.Points) {
      // Chaotic movement
      const geometry = element.geometry as THREE.BufferGeometry;
      const positions = geometry.attributes.position.array as Float32Array;
      
      for (let i = 0; i < positions.length; i += 3) {
        positions[i] += Math.sin(time * 3 + i) * 0.01;
        positions[i + 1] += Math.cos(time * 2 + i) * 0.01;
        positions[i + 2] += Math.sin(time * 4 + i) * 0.005;
      }
      
      geometry.attributes.position.needsUpdate = true;
    }
  }
  
  private clearExpiredInteractions(): void {
    const currentTime = Date.now();
    
    for (const [key, interaction] of this.activeInteractions) {
      // Remove expired interactions (this would need timestamp tracking)
      // For now, we'll keep interactions active
    }
  }
  
  public clear(): void {
    // Remove all interaction visuals
    for (const interaction of this.activeInteractions.values()) {
      interaction.visualElements.forEach(element => {
        this.interactionGroup.remove(element);
        
        if (element instanceof THREE.Mesh || element instanceof THREE.Line || element instanceof THREE.Points) {
          element.geometry.dispose();
          if (Array.isArray(element.material)) {
            element.material.forEach(mat => mat.dispose());
          } else {
            element.material.dispose();
          }
        }
      });
    }
    
    this.activeInteractions.clear();
    this.connectionLines.clear();
    this.interactionParticles.clear();
  }
  
  public dispose(): void {
    this.clear();
    this.scene.remove(this.interactionGroup);
  }
}