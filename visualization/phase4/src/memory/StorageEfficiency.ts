/**
 * Storage Efficiency Visualizer for Phase 4
 * Monitor and visualize storage efficiency and memory usage patterns
 */

import * as THREE from 'three';
import { ShaderLibrary } from '../core/ShaderLibrary';

export interface StorageMetrics {
  totalStorage: number;
  usedStorage: number;
  freeStorage: number;
  fragmentation: number;
  compressionRatio: number;
  ioOperations: number;
  cacheHitRate: number;
  indexEfficiency: number;
  zeroCopyRatio: number;
}

export interface StorageBlock {
  id: string;
  address: number;
  size: number;
  type: 'data' | 'index' | 'cache' | 'free';
  compressionLevel: number;
  accessFrequency: number;
  lastAccessed: number;
  fragmentLevel: number;
  entityIds: string[];
}

export interface StorageEfficiencyConfig {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  maxBlocks: number;
  updateInterval: number;
  colorScheme: {
    data: THREE.Color;
    index: THREE.Color;
    cache: THREE.Color;
    free: THREE.Color;
    fragmented: THREE.Color;
    compressed: THREE.Color;
  };
}

export interface EfficiencyTrend {
  timestamp: number;
  fragmentation: number;
  compression: number;
  cacheEfficiency: number;
  ioThroughput: number;
}

export class StorageEfficiency {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private config: StorageEfficiencyConfig;
  private shaderLibrary: ShaderLibrary;

  // Storage visualization
  private storageBlocks: Map<string, StorageBlock> = new Map();
  private metrics: StorageMetrics = {
    totalStorage: 0,
    usedStorage: 0,
    freeStorage: 0,
    fragmentation: 0,
    compressionRatio: 1.0,
    ioOperations: 0,
    cacheHitRate: 0,
    indexEfficiency: 0,
    zeroCopyRatio: 0
  };
  
  // Visual components
  private storageGridGeometry: THREE.InstancedBufferGeometry;
  private storageGridMaterial: THREE.ShaderMaterial;
  private storageGridMesh: THREE.InstancedMesh;
  
  private efficiencyBarGeometry: THREE.PlaneGeometry;
  private efficiencyBars: {
    fragmentation: THREE.Mesh;
    compression: THREE.Mesh;
    cache: THREE.Mesh;
    io: THREE.Mesh;
  };
  
  private trendLineGeometry: THREE.BufferGeometry;
  private trendLineMaterial: THREE.LineBasicMaterial;
  private trendLines: {
    fragmentation: THREE.Line;
    compression: THREE.Line;
    cache: THREE.Line;
    io: THREE.Line;
  };
  
  // Trend tracking
  private trendHistory: EfficiencyTrend[] = [];
  private maxTrendPoints: number = 100;
  
  // Animation and timing
  private animationClock: THREE.Clock;
  private lastUpdate: number = 0;
  private heatmapTexture: THREE.DataTexture;
  private heatmapData: Uint8Array;

  constructor(config: StorageEfficiencyConfig) {
    this.config = { ...config };
    this.shaderLibrary = ShaderLibrary.getInstance();
    this.animationClock = new THREE.Clock();

    this.initializeRenderer();
    this.initializeScene();
    this.initializeCamera();
    this.initializeStorageVisualization();
    this.initializeEfficiencyBars();
    this.initializeTrendLines();
    this.initializeHeatmap();
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
    const directionalLight = new THREE.DirectionalLight(0x6666ff, 0.4);
    directionalLight.position.set(5, 5, 10);
    this.scene.add(directionalLight);
  }

  private initializeCamera(): void {
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.config.width / this.config.height,
      0.1,
      1000
    );
    
    this.camera.position.set(0, 0, 50);
    this.camera.lookAt(0, 0, 0);
  }

  private initializeStorageVisualization(): void {
    // Create instanced geometry for storage blocks
    const blockGeometry = new THREE.BoxGeometry(1, 1, 0.1);
    this.storageGridGeometry = new THREE.InstancedBufferGeometry().copy(blockGeometry);
    
    // Create shader material for storage visualization
    this.storageGridMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0.0 },
        heatmapTexture: { value: null },
        dataColor: { value: this.config.colorScheme.data },
        indexColor: { value: this.config.colorScheme.index },
        cacheColor: { value: this.config.colorScheme.cache },
        freeColor: { value: this.config.colorScheme.free },
        fragmentedColor: { value: this.config.colorScheme.fragmented },
        compressedColor: { value: this.config.colorScheme.compressed },
        gridSize: { value: new THREE.Vector2(32, 32) }
      },
      vertexShader: `
        attribute float blockType;
        attribute float compressionLevel;
        attribute float accessFrequency;
        attribute float fragmentLevel;
        
        varying float vBlockType;
        varying float vCompressionLevel;
        varying float vAccessFrequency;
        varying float vFragmentLevel;
        varying vec2 vGridPos;
        varying vec3 vPosition;
        
        uniform float time;
        uniform vec2 gridSize;
        
        void main() {
          vBlockType = blockType;
          vCompressionLevel = compressionLevel;
          vAccessFrequency = accessFrequency;
          vFragmentLevel = fragmentLevel;
          
          // Calculate grid position
          float blockIndex = float(gl_InstanceID);
          vGridPos.x = mod(blockIndex, gridSize.x);
          vGridPos.y = floor(blockIndex / gridSize.x);
          
          // Transform to instance position
          vec4 worldPosition = instanceMatrix * vec4(position, 1.0);
          vPosition = worldPosition.xyz;
          
          // Add subtle animation based on access frequency
          float pulse = sin(time * accessFrequency * 2.0) * 0.1 + 1.0;
          worldPosition.xyz *= pulse;
          
          gl_Position = projectionMatrix * modelViewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform sampler2D heatmapTexture;
        uniform vec3 dataColor;
        uniform vec3 indexColor;
        uniform vec3 cacheColor;
        uniform vec3 freeColor;
        uniform vec3 fragmentedColor;
        uniform vec3 compressedColor;
        uniform vec2 gridSize;
        
        varying float vBlockType;
        varying float vCompressionLevel;
        varying float vAccessFrequency;
        varying float vFragmentLevel;
        varying vec2 vGridPos;
        varying vec3 vPosition;
        
        void main() {
          vec3 baseColor;
          
          // Select base color by block type
          if (vBlockType < 0.25) {
            baseColor = dataColor;
          } else if (vBlockType < 0.5) {
            baseColor = indexColor;
          } else if (vBlockType < 0.75) {
            baseColor = cacheColor;
          } else {
            baseColor = freeColor;
          }
          
          // Modify color based on properties
          if (vFragmentLevel > 0.5) {
            baseColor = mix(baseColor, fragmentedColor, vFragmentLevel * 0.7);
          }
          
          if (vCompressionLevel > 0.1) {
            baseColor = mix(baseColor, compressedColor, vCompressionLevel * 0.3);
          }
          
          // Add heatmap overlay
          vec2 heatmapUV = vGridPos / gridSize;
          vec3 heatmap = texture2D(heatmapTexture, heatmapUV).rgb;
          baseColor += heatmap * 0.4;
          
          // Add access frequency glow
          float glow = vAccessFrequency * 0.3;
          baseColor += vec3(glow, glow * 0.5, 0.0);
          
          // Add edge detection for fragmentation
          float edge = step(0.8, vFragmentLevel) * 0.5;
          baseColor += vec3(edge, 0.0, 0.0);
          
          gl_FragColor = vec4(baseColor, 1.0);
        }
      `
    });
    
    // Create instanced mesh
    this.storageGridMesh = new THREE.InstancedMesh(
      this.storageGridGeometry,
      this.storageGridMaterial,
      this.config.maxBlocks
    );
    
    this.storageGridMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.storageGridMesh);
  }

  private initializeEfficiencyBars(): void {
    this.efficiencyBarGeometry = new THREE.PlaneGeometry(2, 0.5);
    
    const createBar = (color: THREE.Color, position: THREE.Vector3) => {
      const material = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(this.efficiencyBarGeometry, material);
      mesh.position.copy(position);
      this.scene.add(mesh);
      return mesh;
    };
    
    this.efficiencyBars = {
      fragmentation: createBar(new THREE.Color(0xff4444), new THREE.Vector3(-15, 15, 0)),
      compression: createBar(new THREE.Color(0x44ff44), new THREE.Vector3(-15, 12, 0)),
      cache: createBar(new THREE.Color(0x4444ff), new THREE.Vector3(-15, 9, 0)),
      io: createBar(new THREE.Color(0xffff44), new THREE.Vector3(-15, 6, 0))
    };
  }

  private initializeTrendLines(): void {
    this.trendLineGeometry = new THREE.BufferGeometry();
    
    const createTrendLine = (color: THREE.Color) => {
      const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      const line = new THREE.Line(this.trendLineGeometry.clone(), material);
      this.scene.add(line);
      return line;
    };
    
    this.trendLines = {
      fragmentation: createTrendLine(new THREE.Color(0xff6666)),
      compression: createTrendLine(new THREE.Color(0x66ff66)),
      cache: createTrendLine(new THREE.Color(0x6666ff)),
      io: createTrendLine(new THREE.Color(0xffff66))
    };
  }

  private initializeHeatmap(): void {
    const size = 64;
    this.heatmapData = new Uint8Array(size * size * 4); // RGBA
    
    // Initialize with default values
    for (let i = 0; i < this.heatmapData.length; i += 4) {
      this.heatmapData[i] = 0;     // R
      this.heatmapData[i + 1] = 0; // G
      this.heatmapData[i + 2] = 0; // B
      this.heatmapData[i + 3] = 255; // A
    }
    
    this.heatmapTexture = new THREE.DataTexture(
      this.heatmapData,
      size,
      size,
      THREE.RGBAFormat
    );
    
    this.storageGridMaterial.uniforms.heatmapTexture.value = this.heatmapTexture;
  }

  public updateStorageBlock(block: StorageBlock): void {
    this.storageBlocks.set(block.id, block);
    this.updateVisualization();
    this.updateMetrics();
  }

  public removeStorageBlock(blockId: string): void {
    this.storageBlocks.delete(blockId);
    this.updateVisualization();
    this.updateMetrics();
  }

  public updateMetrics(): void {
    let totalStorage = 0;
    let usedStorage = 0;
    let fragmentedBlocks = 0;
    let compressedStorage = 0;
    let originalStorage = 0;
    let cacheBlocks = 0;
    let cacheHits = 0;
    let totalAccess = 0;
    
    for (const [_, block] of this.storageBlocks) {
      totalStorage += block.size;
      
      if (block.type !== 'free') {
        usedStorage += block.size;
        originalStorage += block.size;
        compressedStorage += block.size * (1 - block.compressionLevel);
      }
      
      if (block.fragmentLevel > 0.5) {
        fragmentedBlocks++;
      }
      
      if (block.type === 'cache') {
        cacheBlocks++;
        if (block.accessFrequency > 0.7) {
          cacheHits++;
        }
      }
      
      totalAccess += block.accessFrequency;
    }
    
    const totalBlocks = this.storageBlocks.size;
    
    this.metrics = {
      totalStorage,
      usedStorage,
      freeStorage: totalStorage - usedStorage,
      fragmentation: totalBlocks > 0 ? fragmentedBlocks / totalBlocks : 0,
      compressionRatio: originalStorage > 0 ? originalStorage / Math.max(compressedStorage, 1) : 1,
      ioOperations: this.metrics.ioOperations, // Maintained separately
      cacheHitRate: cacheBlocks > 0 ? cacheHits / cacheBlocks : 0,
      indexEfficiency: this.calculateIndexEfficiency(),
      zeroCopyRatio: this.calculateZeroCopyRatio()
    };
    
    // Add to trend history
    const currentTime = Date.now();
    this.trendHistory.push({
      timestamp: currentTime,
      fragmentation: this.metrics.fragmentation,
      compression: this.metrics.compressionRatio,
      cacheEfficiency: this.metrics.cacheHitRate,
      ioThroughput: this.metrics.ioOperations
    });
    
    // Limit history size
    if (this.trendHistory.length > this.maxTrendPoints) {
      this.trendHistory.shift();
    }
    
    this.updateTrendVisualization();
  }

  private calculateIndexEfficiency(): number {
    let indexBlocks = 0;
    let efficientIndexBlocks = 0;
    
    for (const [_, block] of this.storageBlocks) {
      if (block.type === 'index') {
        indexBlocks++;
        // Consider index efficient if it has high access frequency and low fragmentation
        if (block.accessFrequency > 0.6 && block.fragmentLevel < 0.3) {
          efficientIndexBlocks++;
        }
      }
    }
    
    return indexBlocks > 0 ? efficientIndexBlocks / indexBlocks : 1.0;
  }

  private calculateZeroCopyRatio(): number {
    let zeroCopyBlocks = 0;
    let totalDataBlocks = 0;
    
    for (const [_, block] of this.storageBlocks) {
      if (block.type === 'data') {
        totalDataBlocks++;
        // Assume zero-copy if compression is minimal and access is frequent
        if (block.compressionLevel < 0.1 && block.accessFrequency > 0.8) {
          zeroCopyBlocks++;
        }
      }
    }
    
    return totalDataBlocks > 0 ? zeroCopyBlocks / totalDataBlocks : 0;
  }

  private updateVisualization(): void {
    const matrix = new THREE.Matrix4();
    const gridSize = 32;
    let instanceIndex = 0;
    
    // Update storage grid
    for (const [_, block] of this.storageBlocks) {
      if (instanceIndex >= this.config.maxBlocks) break;
      
      const x = (instanceIndex % gridSize) - gridSize / 2;
      const y = Math.floor(instanceIndex / gridSize) - gridSize / 2;
      
      matrix.makeTranslation(x * 1.2, y * 1.2, 0);
      
      // Scale based on block size (logarithmic)
      const scale = 0.5 + Math.log(block.size + 1) / 20;
      matrix.scale(new THREE.Vector3(scale, scale, 1));
      
      this.storageGridMesh.setMatrixAt(instanceIndex, matrix);
      
      // Set instance attributes
      this.updateInstanceAttributes(instanceIndex, block);
      
      instanceIndex++;
    }
    
    this.storageGridMesh.count = instanceIndex;
    this.storageGridMesh.instanceMatrix.needsUpdate = true;
    
    // Update efficiency bars
    this.updateEfficiencyBars();
    
    // Update heatmap
    this.updateHeatmap();
  }

  private updateInstanceAttributes(index: number, block: StorageBlock): void {
    // These would be set through instance attributes
    // For now, we'll use the shader uniforms approach
  }

  private updateEfficiencyBars(): void {
    // Update bar scales based on metrics
    const updateBar = (bar: THREE.Mesh, value: number, maxValue: number = 1.0) => {
      const scale = Math.max(0.1, Math.min(value / maxValue, 1.0));
      bar.scale.setX(scale);
    };
    
    updateBar(this.efficiencyBars.fragmentation, 1 - this.metrics.fragmentation);
    updateBar(this.efficiencyBars.compression, this.metrics.compressionRatio, 3.0);
    updateBar(this.efficiencyBars.cache, this.metrics.cacheHitRate);
    updateBar(this.efficiencyBars.io, Math.min(this.metrics.ioOperations / 1000, 1.0));
  }

  private updateTrendVisualization(): void {
    if (this.trendHistory.length < 2) return;
    
    const points: THREE.Vector3[] = [];
    const maxPoints = Math.min(this.trendHistory.length, 50);
    
    for (let i = 0; i < maxPoints; i++) {
      const trend = this.trendHistory[this.trendHistory.length - maxPoints + i];
      const x = (i / (maxPoints - 1)) * 20 - 10; // Spread across 20 units
      const y = trend.fragmentation * 10 - 5; // Scale to visible range
      points.push(new THREE.Vector3(x, y, 5));
    }
    
    // Update trend line geometry
    this.trendLines.fragmentation.geometry.setFromPoints(points);
    
    // Similar updates for other trend lines with different metrics
    this.updateTrendLine(this.trendLines.compression, 'compression');
    this.updateTrendLine(this.trendLines.cache, 'cacheEfficiency');
    this.updateTrendLine(this.trendLines.io, 'ioThroughput');
  }

  private updateTrendLine(line: THREE.Line, metric: keyof EfficiencyTrend): void {
    if (this.trendHistory.length < 2) return;
    
    const points: THREE.Vector3[] = [];
    const maxPoints = Math.min(this.trendHistory.length, 50);
    
    for (let i = 0; i < maxPoints; i++) {
      const trend = this.trendHistory[this.trendHistory.length - maxPoints + i];
      const x = (i / (maxPoints - 1)) * 20 - 10;
      
      let y: number;
      switch (metric) {
        case 'compression':
          y = (trend.compression - 1) * 5; // Center around 1.0
          break;
        case 'cacheEfficiency':
          y = trend.cacheEfficiency * 10 - 5;
          break;
        case 'ioThroughput':
          y = (trend.ioThroughput / 1000) * 10 - 5;
          break;
        default:
          y = 0;
      }
      
      points.push(new THREE.Vector3(x, y + (metric === 'compression' ? 2 : metric === 'cacheEfficiency' ? -2 : -4), 5));
    }
    
    line.geometry.setFromPoints(points);
  }

  private updateHeatmap(): void {
    const size = 64;
    
    // Generate heatmap based on storage access patterns
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const index = (y * size + x) * 4;
        
        // Find blocks in this region
        const regionBlocks = Array.from(this.storageBlocks.values()).filter(block => {
          const blockX = Math.floor((block.address % 4096) / 64);
          const blockY = Math.floor(block.address / 4096) % 64;
          return Math.abs(blockX - x) <= 2 && Math.abs(blockY - y) <= 2;
        });
        
        let heat = 0;
        for (const block of regionBlocks) {
          heat += block.accessFrequency * (1 - block.fragmentLevel);
        }
        
        heat = Math.min(heat * 255, 255);
        
        this.heatmapData[index] = heat;     // R - Access frequency
        this.heatmapData[index + 1] = 0;    // G
        this.heatmapData[index + 2] = Math.min((1 - this.metrics.fragmentation) * 255, 255); // B - Efficiency
        this.heatmapData[index + 3] = 255;  // A
      }
    }
    
    this.heatmapTexture.needsUpdate = true;
  }

  public recordIOOperation(): void {
    this.metrics.ioOperations++;
  }

  public animate(): void {
    const deltaTime = this.animationClock.getDelta();
    const elapsedTime = this.animationClock.getElapsedTime();
    
    // Update shader time
    this.storageGridMaterial.uniforms.time.value = elapsedTime;
    
    // Update visualization if enough time has passed
    if (elapsedTime - this.lastUpdate > this.config.updateInterval) {
      this.updateVisualization();
      this.lastUpdate = elapsedTime;
    }
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }

  public getMetrics(): StorageMetrics {
    return { ...this.metrics };
  }

  public getStorageEfficiencyScore(): number {
    // Composite efficiency score
    const weights = {
      fragmentation: 0.3,
      compression: 0.2,
      cache: 0.2,
      index: 0.15,
      zeroCopy: 0.15
    };
    
    return (
      (1 - this.metrics.fragmentation) * weights.fragmentation +
      Math.min(this.metrics.compressionRatio / 2, 1) * weights.compression +
      this.metrics.cacheHitRate * weights.cache +
      this.metrics.indexEfficiency * weights.index +
      this.metrics.zeroCopyRatio * weights.zeroCopy
    );
  }

  public getTrendHistory(): EfficiencyTrend[] {
    return [...this.trendHistory];
  }

  public dispose(): void {
    // Dispose of all Three.js resources
    this.storageGridGeometry.dispose();
    this.storageGridMaterial.dispose();
    this.efficiencyBarGeometry.dispose();
    this.trendLineGeometry.dispose();
    this.heatmapTexture.dispose();
    
    // Dispose efficiency bars
    Object.values(this.efficiencyBars).forEach(bar => {
      bar.geometry.dispose();
      (bar.material as THREE.Material).dispose();
    });
    
    // Dispose trend lines
    Object.values(this.trendLines).forEach(line => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    });
    
    this.renderer.dispose();
    
    // Clear collections
    this.storageBlocks.clear();
    this.trendHistory.length = 0;
  }
}

export default StorageEfficiency;