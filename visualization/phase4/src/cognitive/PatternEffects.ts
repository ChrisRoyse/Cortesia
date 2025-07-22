import * as THREE from 'three';
import { CognitivePatternData } from './CognitivePatternVisualizer';

export interface EffectConfig {
  color: THREE.Color;
  intensity: number;
  size: number;
  duration: number;
  decay: number;
}

export class PatternEffects {
  private scene: THREE.Scene;
  private effectsGroup: THREE.Group;
  private particleSystems: Map<string, THREE.Points> = new Map();
  private trailSystems: Map<string, THREE.Line> = new Map();
  private shaderMaterials: Map<string, THREE.ShaderMaterial> = new Map();
  
  // Shader uniforms for global effects
  private globalUniforms = {
    time: { value: 0.0 },
    inhibition: { value: 0.0 },
    metaCognitive: { value: 0.0 }
  };
  
  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.effectsGroup = new THREE.Group();
    this.effectsGroup.name = 'PatternEffects';
    this.scene.add(this.effectsGroup);
    
    this.initializeShaders();
  }
  
  private initializeShaders(): void {
    // Convergent beam shader
    this.createBeamShader();
    
    // Divergent radiation shader
    this.createRadiationShader();
    
    // Lateral connection shader
    this.createConnectionShader();
    
    // Systems hierarchy shader
    this.createHierarchyShader();
    
    // Critical collision shader
    this.createCollisionShader();
    
    // Abstract pattern shader
    this.createPatternShader();
    
    // Adaptive switching shader
    this.createSwitchingShader();
  }
  
  private createBeamShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      uniform float inhibition;
      attribute float phase;
      attribute vec3 direction;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Convergent beam effect - particles move toward center
        float convergence = sin(time * 2.0 + phase) * intensity;
        pos += direction * convergence;
        
        // Apply inhibition
        pos *= (1.0 - inhibition * 0.5);
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 3.0 * intensity;
        
        // Color shifts based on convergence
        vColor = mix(vec3(0.2, 0.6, 1.0), vec3(1.0, 0.8, 0.2), convergence);
        vOpacity = intensity * (1.0 - inhibition);
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('convergent', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  private createRadiationShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      uniform float inhibition;
      attribute float phase;
      attribute vec3 direction;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Divergent radiation effect - particles move away from center
        float radiation = sin(time * 1.5 + phase) * intensity;
        pos += normalize(pos) * radiation * 2.0;
        
        // Apply inhibition
        pos *= (1.0 - inhibition * 0.3);
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 2.0 + intensity * 3.0;
        
        // Color shifts with radiation
        vColor = mix(vec3(1.0, 0.3, 0.8), vec3(0.3, 1.0, 0.5), radiation);
        vOpacity = intensity * (1.0 - inhibition);
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('divergent', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  private createConnectionShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      attribute float connectionStrength;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Lateral wave propagation
        float wave = sin(time * 3.0 + pos.x * 0.5) * intensity;
        pos.y += wave * 0.5;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        
        vColor = vec3(0.8, 0.4, 1.0);
        vOpacity = connectionStrength * intensity;
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        gl_FragColor = vec4(vColor, vOpacity);
      }
    `;
    
    this.shaderMaterials.set('lateral', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true
    }));
  }
  
  private createHierarchyShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      attribute float level;
      attribute float branchIndex;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Hierarchical flow animation
        float flow = sin(time * 2.0 - level * 0.5) * intensity;
        pos.z += flow * 0.3;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 2.0 + level * 1.5;
        
        // Color based on hierarchy level
        vColor = mix(vec3(0.2, 1.0, 0.3), vec3(0.2, 0.3, 1.0), level / 5.0);
        vOpacity = intensity;
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('systems', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  private createCollisionShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      attribute float collisionPhase;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Collision ripple effect
        float collision = sin(time * 4.0 + collisionPhase) * intensity;
        pos *= (1.0 + collision * 0.2);
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 3.0 * (1.0 + collision);
        
        // Red/orange collision colors
        vColor = mix(vec3(1.0, 0.2, 0.1), vec3(1.0, 0.6, 0.1), collision);
        vOpacity = intensity * (0.5 + collision * 0.5);
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('critical', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  private createPatternShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      attribute float patternPhase;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Abstract pattern detection waves
        float pattern = sin(time * 1.8 + patternPhase * 2.0) * intensity;
        pos += vec3(cos(patternPhase), sin(patternPhase), 0.0) * pattern * 0.5;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 2.0 + pattern * 2.0;
        
        // Purple/cyan abstract colors
        vColor = mix(vec3(0.6, 0.2, 1.0), vec3(0.2, 0.8, 1.0), pattern);
        vOpacity = intensity;
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('abstract', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  private createSwitchingShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float intensity;
      uniform float metaCognitive;
      attribute float switchPhase;
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        vec3 pos = position;
        
        // Meta-cognitive switching effect
        float switching = sin(time * 3.5 + switchPhase) * intensity;
        float meta = sin(time * 0.8) * metaCognitive;
        
        // Position morphing
        pos *= (1.0 + switching * 0.3 + meta * 0.2);
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 1.0 + switching * 4.0;
        
        // Multi-colored adaptive effect
        vColor = vec3(
          0.5 + sin(time + switchPhase) * 0.5,
          0.5 + cos(time + switchPhase * 1.3) * 0.5,
          0.5 + sin(time * 1.7 + switchPhase * 0.7) * 0.5
        );
        vOpacity = intensity * (0.7 + meta * 0.3);
      }
    `;
    
    const fragmentShader = `
      varying vec3 vColor;
      varying float vOpacity;
      
      void main() {
        float dist = distance(gl_PointCoord, vec2(0.5));
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha * vOpacity);
      }
    `;
    
    this.shaderMaterials.set('adaptive', new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        ...this.globalUniforms,
        intensity: { value: 1.0 },
        metaCognitive: { value: 0.0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending
    }));
  }
  
  // Public methods for creating effects
  
  public createParticleSystem(
    type: string,
    positions: Float32Array,
    attributes: { [key: string]: Float32Array }
  ): THREE.Points {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Add custom attributes
    for (const [name, data] of Object.entries(attributes)) {
      geometry.setAttribute(name, new THREE.BufferAttribute(data, 1));
    }
    
    const material = this.shaderMaterials.get(type);
    if (!material) {
      throw new Error(`Unknown pattern type: ${type}`);
    }
    
    const particles = new THREE.Points(geometry, material);
    this.effectsGroup.add(particles);
    this.particleSystems.set(`${type}_particles`, particles);
    
    return particles;
  }
  
  public createTrailSystem(type: string, points: THREE.Vector3[]): THREE.Line {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    const material = new THREE.LineBasicMaterial({
      color: this.getPatternColor(type),
      transparent: true,
      opacity: 0.6
    });
    
    const line = new THREE.Line(geometry, material);
    this.effectsGroup.add(line);
    this.trailSystems.set(`${type}_trail`, line);
    
    return line;
  }
  
  private getPatternColor(type: string): THREE.Color {
    const colors = {
      convergent: new THREE.Color(0x3399ff),
      divergent: new THREE.Color(0xff4d80),
      lateral: new THREE.Color(0xcc66ff),
      systems: new THREE.Color(0x33ff66),
      critical: new THREE.Color(0xff6633),
      abstract: new THREE.Color(0x9966ff),
      adaptive: new THREE.Color(0xffcc33)
    };
    
    return colors[type] || new THREE.Color(0xffffff);
  }
  
  public updateGlobalUniforms(time: number): void {
    this.globalUniforms.time.value = time;
    
    // Update all shader materials
    for (const material of this.shaderMaterials.values()) {
      material.uniforms.time.value = time;
    }
  }
  
  public applyGlobalInhibition(inhibition: number): void {
    this.globalUniforms.inhibition.value = inhibition;
    
    for (const material of this.shaderMaterials.values()) {
      if (material.uniforms.inhibition) {
        material.uniforms.inhibition.value = inhibition;
      }
    }
  }
  
  public setMetaCognitiveLevel(level: number): void {
    this.globalUniforms.metaCognitive.value = level;
    
    const adaptiveMaterial = this.shaderMaterials.get('adaptive');
    if (adaptiveMaterial && adaptiveMaterial.uniforms.metaCognitive) {
      adaptiveMaterial.uniforms.metaCognitive.value = level;
    }
  }
  
  public showPatternTrails(patterns: CognitivePatternData[]): void {
    // Create trail visualization from pattern history
    const trailPoints = patterns.map((pattern, index) => {
      const angle = (index / patterns.length) * Math.PI * 2;
      const radius = 3 + pattern.intensity * 2;
      return new THREE.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        pattern.activation * 2 - 1
      );
    });
    
    if (trailPoints.length > 1) {
      this.createTrailSystem('history', trailPoints);
    }
  }
  
  public clearPatternTrails(): void {
    for (const [key, trail] of this.trailSystems) {
      if (key.includes('trail')) {
        this.effectsGroup.remove(trail);
        trail.geometry.dispose();
        (trail.material as THREE.Material).dispose();
      }
    }
    this.trailSystems.clear();
  }
  
  public updateDecayEffects(time: number): void {
    // Update particle opacity based on age
    for (const particles of this.particleSystems.values()) {
      const material = particles.material as THREE.ShaderMaterial;
      if (material.uniforms.intensity) {
        const decay = Math.max(0.1, Math.sin(time * 0.5) * 0.5 + 0.5);
        material.uniforms.intensity.value = decay;
      }
    }
  }
  
  public dispose(): void {
    // Dispose all materials
    for (const material of this.shaderMaterials.values()) {
      material.dispose();
    }
    
    // Dispose all geometries
    for (const particles of this.particleSystems.values()) {
      particles.geometry.dispose();
    }
    
    for (const trail of this.trailSystems.values()) {
      trail.geometry.dispose();
      (trail.material as THREE.Material).dispose();
    }
    
    this.scene.remove(this.effectsGroup);
  }
}