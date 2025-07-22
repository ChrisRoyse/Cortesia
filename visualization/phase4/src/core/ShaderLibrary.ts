/**
 * Custom WebGL shader library for LLMKG brain-inspired visual effects
 * Provides optimized shaders for neural network and data flow visualization
 */

import * as THREE from 'three';

export interface ShaderConfig {
  vertexShader: string;
  fragmentShader: string;
  uniforms: Record<string, THREE.IUniform>;
}

export class ShaderLibrary {
  private static instance: ShaderLibrary;
  private shaders: Map<string, ShaderConfig> = new Map();

  private constructor() {
    this.initializeShaders();
  }

  public static getInstance(): ShaderLibrary {
    if (!ShaderLibrary.instance) {
      ShaderLibrary.instance = new ShaderLibrary();
    }
    return ShaderLibrary.instance;
  }

  private initializeShaders(): void {
    this.registerDataFlowShader();
    this.registerNeuralNetworkShader();
    this.registerParticleFlowShader();
    this.registerCognitivePatternShader();
  }

  public getShader(name: string): ShaderConfig | null {
    return this.shaders.get(name) || null;
  }

  public registerShader(name: string, config: ShaderConfig): void {
    this.shaders.set(name, config);
  }

  private registerDataFlowShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float speed;
      uniform float amplitude;
      
      attribute float size;
      attribute vec3 color;
      attribute float phase;
      
      varying vec3 vColor;
      varying float vSize;
      
      void main() {
        vColor = color;
        vSize = size;
        
        vec3 pos = position;
        float wave = sin(time * speed + phase) * amplitude;
        pos.y += wave;
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      uniform float time;
      uniform float opacity;
      
      varying vec3 vColor;
      varying float vSize;
      
      void main() {
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center);
        
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
        float pulse = 0.8 + 0.2 * sin(time * 3.0);
        
        gl_FragColor = vec4(vColor * pulse, alpha * opacity);
      }
    `;

    this.shaders.set('dataFlow', {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        speed: { value: 2.0 },
        amplitude: { value: 0.1 },
        opacity: { value: 1.0 }
      }
    });
  }

  private registerNeuralNetworkShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float connectionStrength;
      
      attribute float activation;
      attribute vec3 targetPosition;
      
      varying float vActivation;
      varying vec3 vColor;
      
      void main() {
        vActivation = activation;
        
        vec3 pos = mix(position, targetPosition, connectionStrength);
        float pulse = 0.5 + 0.5 * sin(time * 4.0 + activation * 10.0);
        
        vColor = vec3(
          0.3 + activation * 0.7,
          0.1 + activation * 0.9,
          1.0 - activation * 0.3
        ) * pulse;
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `;

    const fragmentShader = `
      varying float vActivation;
      varying vec3 vColor;
      
      void main() {
        float alpha = 0.6 + vActivation * 0.4;
        gl_FragColor = vec4(vColor, alpha);
      }
    `;

    this.shaders.set('neuralNetwork', {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        connectionStrength: { value: 1.0 }
      }
    });
  }

  private registerParticleFlowShader(): void {
    const vertexShader = `
      uniform float time;
      uniform vec3 flowDirection;
      uniform float flowSpeed;
      
      attribute float lifecycle;
      attribute vec3 velocity;
      
      varying vec3 vColor;
      varying float vLifecycle;
      
      void main() {
        vLifecycle = lifecycle;
        
        vec3 pos = position + velocity * time + flowDirection * flowSpeed * time;
        
        float age = 1.0 - lifecycle;
        vColor = mix(
          vec3(0.0, 1.0, 1.0), // Young particles (cyan)
          vec3(1.0, 0.5, 0.0), // Mature particles (orange)
          age
        );
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_PointSize = 5.0 * lifecycle;
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = `
      varying vec3 vColor;
      varying float vLifecycle;
      
      void main() {
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center);
        
        if (dist > 0.5) discard;
        
        float alpha = (1.0 - dist * 2.0) * vLifecycle;
        gl_FragColor = vec4(vColor, alpha);
      }
    `;

    this.shaders.set('particleFlow', {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        flowDirection: { value: new THREE.Vector3(1, 0, 0) },
        flowSpeed: { value: 1.0 }
      }
    });
  }

  private registerCognitivePatternShader(): void {
    const vertexShader = `
      uniform float time;
      uniform float patternComplexity;
      
      attribute float cognitiveWeight;
      attribute vec3 patternCenter;
      
      varying vec3 vColor;
      varying float vWeight;
      
      void main() {
        vWeight = cognitiveWeight;
        
        vec3 pos = position;
        float distToCenter = length(position - patternCenter);
        float influence = exp(-distToCenter * patternComplexity);
        
        pos += normalize(position - patternCenter) * influence * sin(time * 2.0) * 0.1;
        
        vColor = vec3(
          1.0 - cognitiveWeight * 0.5,
          cognitiveWeight,
          0.5 + cognitiveWeight * 0.5
        );
        
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `;

    const fragmentShader = `
      uniform float time;
      
      varying vec3 vColor;
      varying float vWeight;
      
      void main() {
        float pulse = 0.7 + 0.3 * sin(time * 1.5 + vWeight * 5.0);
        vec3 finalColor = vColor * pulse;
        
        gl_FragColor = vec4(finalColor, 0.8);
      }
    `;

    this.shaders.set('cognitivePattern', {
      vertexShader,
      fragmentShader,
      uniforms: {
        time: { value: 0.0 },
        patternComplexity: { value: 1.0 }
      }
    });
  }

  public updateShaderTime(shaderName: string, time: number): void {
    const shader = this.shaders.get(shaderName);
    if (shader && shader.uniforms.time) {
      shader.uniforms.time.value = time;
    }
  }

  public updateAllShadersTime(time: number): void {
    this.shaders.forEach((shader) => {
      if (shader.uniforms.time) {
        shader.uniforms.time.value = time;
      }
    });
  }

  public createMaterial(shaderName: string, additionalUniforms?: Record<string, THREE.IUniform>): THREE.ShaderMaterial | null {
    const shader = this.getShader(shaderName);
    if (!shader) return null;

    const uniforms = { ...shader.uniforms };
    if (additionalUniforms) {
      Object.assign(uniforms, additionalUniforms);
    }

    return new THREE.ShaderMaterial({
      vertexShader: shader.vertexShader,
      fragmentShader: shader.fragmentShader,
      uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
  }

  public getAvailableShaders(): string[] {
    return Array.from(this.shaders.keys());
  }
}

export default ShaderLibrary;