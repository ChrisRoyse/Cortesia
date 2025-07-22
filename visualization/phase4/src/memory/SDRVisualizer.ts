/**
 * SDR (Sparse Distributed Representation) Visualizer for Phase 4
 * Efficiently visualizes sparse matrices and SDR patterns with optimized rendering
 */

import * as THREE from 'three';
import { ShaderLibrary } from '../core/ShaderLibrary';

export interface SDRConfig {
  totalBits: number;
  activeBits: number;
  sparsity: number;
  overlapThreshold: number;
}

export interface SDRPattern {
  patternId: string;
  activeBits: Set<number>;
  totalBits: number;
  conceptName: string;
  confidence: number;
  usageCount: number;
  timestamp: number;
}

export interface SDRVisualizationConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  maxPatterns: number;
  cellSize: number;
  gridDimensions: { rows: number; cols: number };
  colorScheme: {
    active: THREE.Color;
    inactive: THREE.Color;
    highlight: THREE.Color;
  };
}

export interface SDRComparisonResult {
  patternA: string;
  patternB: string;
  overlap: number;
  similarity: number;
  uniqueBitsA: number;
  uniqueBitsB: number;
}

export class SDRVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private config: SDRVisualizationConfig;
  private shaderLibrary: ShaderLibrary;

  private patterns: Map<string, SDRPattern> = new Map();
  private visualizedPatterns: Map<string, THREE.Mesh> = new Map();
  
  // Efficient sparse matrix rendering
  private activeBitGeometry: THREE.InstancedBufferGeometry;
  private activeBitMaterial: THREE.ShaderMaterial;
  private activeBitMesh: THREE.InstancedMesh;
  
  private gridGeometry: THREE.BufferGeometry;
  private gridMaterial: THREE.LineBasicMaterial;
  private gridLines: THREE.LineSegments;
  
  // Animation and interaction
  private hoveredPattern: string | null = null;
  private selectedPattern: string | null = null;
  private animationClock: THREE.Clock;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;
  
  // Performance monitoring
  private renderStats = {
    activeBitsRendered: 0,
    patternsVisible: 0,
    frameTime: 0,
    memory: { usage: 0, allocated: 0 }
  };

  constructor(config: SDRVisualizationConfig) {
    this.config = { ...config };
    this.shaderLibrary = ShaderLibrary.getInstance();
    this.animationClock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this.initializeRenderer();
    this.initializeScene();
    this.initializeCamera();
    this.initializeSparseMatrixSystem();
    this.initializeGrid();
    this.setupEventListeners();
  }

  private initializeRenderer(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.config.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    
    this.renderer.setSize(this.config.width, this.config.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0a0a, 1.0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  private initializeScene(): void {
    this.scene = new THREE.Scene();
    
    // Add subtle ambient lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    this.scene.add(ambientLight);
    
    // Add directional light for depth
    const directionalLight = new THREE.DirectionalLight(0x8888ff, 0.4);
    directionalLight.position.set(0, 0, 10);
    this.scene.add(directionalLight);
  }

  private initializeCamera(): void {
    const aspect = this.config.width / this.config.height;
    const frustumSize = Math.max(this.config.gridDimensions.rows, this.config.gridDimensions.cols) * this.config.cellSize;
    
    this.camera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      -frustumSize / 2,
      1,
      1000
    );
    
    this.camera.position.set(0, 0, 100);
    this.camera.lookAt(0, 0, 0);
  }

  private initializeSparseMatrixSystem(): void {
    // Create instanced geometry for efficient active bit rendering
    const bitGeometry = new THREE.PlaneGeometry(this.config.cellSize * 0.8, this.config.cellSize * 0.8);
    this.activeBitGeometry = new THREE.InstancedBufferGeometry().copy(bitGeometry);
    
    // Create shader material for active bits
    this.activeBitMaterial = this.createSDRShaderMaterial();
    
    // Create instanced mesh for maximum efficiency
    this.activeBitMesh = new THREE.InstancedMesh(
      this.activeBitGeometry,
      this.activeBitMaterial,
      10000 // Max active bits across all patterns
    );
    
    this.activeBitMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.activeBitMesh);
  }

  private createSDRShaderMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        activeBrightness: { value: 1.0 },
        patternOpacity: { value: 1.0 },
        highlightColor: { value: this.config.colorScheme.highlight },
        activeColor: { value: this.config.colorScheme.active }
      },
      vertexShader: `
        attribute float bitIndex;
        attribute float patternId;
        attribute float activation;
        attribute vec3 patternColor;
        
        varying float vBitIndex;
        varying float vPatternId;
        varying float vActivation;
        varying vec3 vPatternColor;
        varying vec3 vWorldPosition;
        
        uniform float time;
        
        void main() {
          vBitIndex = bitIndex;
          vPatternId = patternId;
          vActivation = activation;
          vPatternColor = patternColor;
          
          vec4 worldPosition = instanceMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          
          // Add subtle pulsing animation for active bits
          float pulse = sin(time * 2.0 + bitIndex * 0.1) * 0.1 + 1.0;
          worldPosition.xy *= pulse;
          
          gl_Position = projectionMatrix * modelViewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float activeBrightness;
        uniform float patternOpacity;
        uniform vec3 highlightColor;
        uniform vec3 activeColor;
        
        varying float vBitIndex;
        varying float vPatternId;
        varying float vActivation;
        varying vec3 vPatternColor;
        varying vec3 vWorldPosition;
        
        void main() {
          // Create circular active bit visualization
          vec2 center = vec2(0.5);
          vec2 uv = gl_FragCoord.xy / vec2(textureSize(gl_FragTexCoordX, 0));
          float dist = distance(uv, center);
          
          if (dist > 0.4) discard;
          
          // Color based on pattern and activation
          vec3 color = mix(vPatternColor, activeColor, vActivation);
          
          // Add glow effect for highly activated bits
          float glow = smoothstep(0.4, 0.0, dist) * vActivation;
          color += highlightColor * glow * 0.3;
          
          // Fade based on distance from center
          float alpha = smoothstep(0.4, 0.2, dist) * patternOpacity;
          
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending
    });
  }

  private initializeGrid(): void {
    const gridPositions: number[] = [];
    const { rows, cols } = this.config.gridDimensions;
    const cellSize = this.config.cellSize;
    
    // Create grid lines
    for (let i = 0; i <= rows; i++) {
      const y = (i - rows / 2) * cellSize;
      gridPositions.push(
        -cols * cellSize / 2, y, 0,
        cols * cellSize / 2, y, 0
      );
    }
    
    for (let j = 0; j <= cols; j++) {
      const x = (j - cols / 2) * cellSize;
      gridPositions.push(
        x, -rows * cellSize / 2, 0,
        x, rows * cellSize / 2, 0
      );
    }
    
    this.gridGeometry = new THREE.BufferGeometry();
    this.gridGeometry.setAttribute('position', new THREE.Float32BufferAttribute(gridPositions, 3));
    
    this.gridMaterial = new THREE.LineBasicMaterial({
      color: 0x333333,
      transparent: true,
      opacity: 0.3
    });
    
    this.gridLines = new THREE.LineSegments(this.gridGeometry, this.gridMaterial);
    this.scene.add(this.gridLines);
  }

  private setupEventListeners(): void {
    this.config.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    this.config.canvas.addEventListener('click', this.onMouseClick.bind(this));
    window.addEventListener('resize', this.onWindowResize.bind(this));
  }

  public addSDRPattern(pattern: SDRPattern): void {
    this.patterns.set(pattern.patternId, pattern);
    this.updateVisualization();
  }

  public removeSDRPattern(patternId: string): void {
    this.patterns.delete(patternId);
    this.visualizedPatterns.delete(patternId);
    this.updateVisualization();
  }

  public updateSDRPattern(patternId: string, updates: Partial<SDRPattern>): void {
    const pattern = this.patterns.get(patternId);
    if (pattern) {
      Object.assign(pattern, updates);
      this.updateVisualization();
    }
  }

  public comparePatterns(patternIdA: string, patternIdB: string): SDRComparisonResult | null {
    const patternA = this.patterns.get(patternIdA);
    const patternB = this.patterns.get(patternIdB);
    
    if (!patternA || !patternB) return null;
    
    const intersection = new Set([...patternA.activeBits].filter(bit => patternB.activeBits.has(bit)));
    const union = new Set([...patternA.activeBits, ...patternB.activeBits]);
    
    const overlap = intersection.size;
    const similarity = overlap / union.size;
    
    return {
      patternA: patternIdA,
      patternB: patternIdB,
      overlap,
      similarity,
      uniqueBitsA: patternA.activeBits.size - overlap,
      uniqueBitsB: patternB.activeBits.size - overlap
    };
  }

  private updateVisualization(): void {
    const startTime = performance.now();
    
    // Clear previous visualization data
    this.renderStats.activeBitsRendered = 0;
    this.renderStats.patternsVisible = 0;
    
    // Create instanced data for all active bits
    const matrices: THREE.Matrix4[] = [];
    const colors: THREE.Color[] = [];
    const bitData: Float32Array[] = [];
    
    const { rows, cols } = this.config.gridDimensions;
    const cellSize = this.config.cellSize;
    
    for (const [patternId, pattern] of this.patterns) {
      const patternColor = this.getPatternColor(patternId, pattern);
      
      for (const bitIndex of pattern.activeBits) {
        const row = Math.floor(bitIndex / cols);
        const col = bitIndex % cols;
        
        if (row >= rows || col >= cols) continue;
        
        // Calculate world position
        const x = (col - cols / 2) * cellSize;
        const y = (row - rows / 2) * cellSize;
        
        // Create transformation matrix
        const matrix = new THREE.Matrix4();
        matrix.makeTranslation(x, y, 0);
        
        // Scale based on pattern confidence and usage
        const scale = 0.8 + (pattern.confidence * 0.4) + (Math.log(pattern.usageCount + 1) * 0.1);
        matrix.scale(new THREE.Vector3(scale, scale, 1));
        
        matrices.push(matrix);
        colors.push(patternColor);
        
        this.renderStats.activeBitsRendered++;
      }
      
      this.renderStats.patternsVisible++;
    }
    
    // Update instanced mesh
    for (let i = 0; i < matrices.length && i < this.activeBitMesh.count; i++) {
      this.activeBitMesh.setMatrixAt(i, matrices[i]);
      this.activeBitMesh.setColorAt(i, colors[i]);
    }
    
    this.activeBitMesh.count = Math.min(matrices.length, 10000);
    this.activeBitMesh.instanceMatrix.needsUpdate = true;
    if (this.activeBitMesh.instanceColor) {
      this.activeBitMesh.instanceColor.needsUpdate = true;
    }
    
    this.renderStats.frameTime = performance.now() - startTime;
  }

  private getPatternColor(patternId: string, pattern: SDRPattern): THREE.Color {
    if (patternId === this.selectedPattern) {
      return this.config.colorScheme.highlight.clone();
    }
    
    if (patternId === this.hoveredPattern) {
      return this.config.colorScheme.active.clone().lerp(this.config.colorScheme.highlight, 0.3);
    }
    
    // Generate consistent color based on pattern properties
    const hue = (pattern.conceptName.charCodeAt(0) * 137.508) % 360; // Golden angle for good distribution
    const saturation = 0.6 + (pattern.confidence * 0.4);
    const lightness = 0.4 + (Math.min(pattern.usageCount, 100) / 100) * 0.3;
    
    return new THREE.Color().setHSL(hue / 360, saturation, lightness);
  }

  private onMouseMove(event: MouseEvent): void {
    const rect = this.config.canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Raycast to find hovered patterns
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.activeBitMesh);
    
    if (intersects.length > 0) {
      // Find which pattern this bit belongs to
      const intersect = intersects[0];
      const instanceId = intersect.instanceId;
      
      // Update hover state
      const newHoveredPattern = this.findPatternForBit(instanceId);
      if (newHoveredPattern !== this.hoveredPattern) {
        this.hoveredPattern = newHoveredPattern;
        this.updateVisualization();
      }
    } else if (this.hoveredPattern) {
      this.hoveredPattern = null;
      this.updateVisualization();
    }
  }

  private onMouseClick(event: MouseEvent): void {
    if (this.hoveredPattern) {
      this.selectedPattern = this.selectedPattern === this.hoveredPattern ? null : this.hoveredPattern;
      this.updateVisualization();
      
      // Emit selection event
      this.config.canvas.dispatchEvent(new CustomEvent('sdr-pattern-selected', {
        detail: { 
          patternId: this.selectedPattern,
          pattern: this.selectedPattern ? this.patterns.get(this.selectedPattern) : null
        }
      }));
    }
  }

  private findPatternForBit(instanceId: number): string | null {
    // This would need to be implemented based on how we map instance IDs to patterns
    // For now, return the first pattern (simplified)
    return this.patterns.keys().next().value || null;
  }

  private onWindowResize(): void {
    const width = this.config.canvas.clientWidth;
    const height = this.config.canvas.clientHeight;
    
    this.config.width = width;
    this.config.height = height;
    
    const aspect = width / height;
    const frustumSize = Math.max(this.config.gridDimensions.rows, this.config.gridDimensions.cols) * this.config.cellSize;
    
    this.camera.left = -frustumSize * aspect / 2;
    this.camera.right = frustumSize * aspect / 2;
    this.camera.top = frustumSize / 2;
    this.camera.bottom = -frustumSize / 2;
    this.camera.updateProjectionMatrix();
    
    this.renderer.setSize(width, height);
  }

  public animate(): void {
    const deltaTime = this.animationClock.getDelta();
    const elapsedTime = this.animationClock.getElapsedTime();
    
    // Update shader uniforms
    if (this.activeBitMaterial.uniforms.time) {
      this.activeBitMaterial.uniforms.time.value = elapsedTime;
    }
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }

  public setSparsityFilter(minSparsity: number, maxSparsity: number): void {
    // Filter patterns by sparsity and update visualization
    for (const [patternId, pattern] of this.patterns) {
      const sparsity = pattern.activeBits.size / pattern.totalBits;
      const visible = sparsity >= minSparsity && sparsity <= maxSparsity;
      
      // Update pattern visibility
      // Implementation would depend on how we structure the instanced rendering
    }
  }

  public getPerformanceMetrics() {
    return {
      ...this.renderStats,
      memory: {
        usage: this.renderer.info.memory.geometries + this.renderer.info.memory.textures,
        allocated: this.patterns.size * 1000 // Rough estimate
      },
      patterns: {
        total: this.patterns.size,
        visible: this.renderStats.patternsVisible,
        selected: this.selectedPattern ? 1 : 0,
        hovered: this.hoveredPattern ? 1 : 0
      }
    };
  }

  public dispose(): void {
    // Dispose of all Three.js resources
    this.activeBitGeometry.dispose();
    this.activeBitMaterial.dispose();
    this.gridGeometry.dispose();
    this.gridMaterial.dispose();
    this.renderer.dispose();
    
    // Clear collections
    this.patterns.clear();
    this.visualizedPatterns.clear();
    
    // Remove event listeners
    this.config.canvas.removeEventListener('mousemove', this.onMouseMove.bind(this));
    this.config.canvas.removeEventListener('click', this.onMouseClick.bind(this));
    window.removeEventListener('resize', this.onWindowResize.bind(this));
  }
}

export default SDRVisualizer;