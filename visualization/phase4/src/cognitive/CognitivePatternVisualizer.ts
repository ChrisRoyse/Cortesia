import * as THREE from 'three';
import { PatternEffects } from './PatternEffects';
import { PatternInteractions } from './PatternInteractions';
import { ConvergentThinking } from './patterns/ConvergentThinking';
import { DivergentThinking } from './patterns/DivergentThinking';
import { LateralThinking } from './patterns/LateralThinking';
import { SystemsThinking } from './patterns/SystemsThinking';
import { CriticalThinking } from './patterns/CriticalThinking';
import { AbstractThinking } from './patterns/AbstractThinking';
import { AdaptiveThinking } from './patterns/AdaptiveThinking';

export interface CognitivePatternData {
  type: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract' | 'adaptive';
  activation: number; // 0.0 - 1.0
  intensity: number; // 0.0 - 1.0
  duration: number; // milliseconds
  timestamp: number;
  metadata?: {
    trigger?: string;
    context?: string;
    interactions?: string[];
  };
}

export interface CognitiveState {
  activePatterns: Map<string, CognitivePatternData>;
  patternHistory: CognitivePatternData[];
  interactionMatrix: number[][];
  globalInhibition: number;
  metaCognitiveLevel: number;
}

export class CognitivePatternVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private effects: PatternEffects;
  private interactions: PatternInteractions;
  
  // Pattern visualizers
  private convergentPattern: ConvergentThinking;
  private divergentPattern: DivergentThinking;
  private lateralPattern: LateralThinking;
  private systemsPattern: SystemsThinking;
  private criticalPattern: CriticalThinking;
  private abstractPattern: AbstractThinking;
  private adaptivePattern: AdaptiveThinking;
  
  private cognitiveState: CognitiveState;
  private animationId: number | null = null;
  private time: number = 0;
  
  // Visual settings
  private showPatternHistory: boolean = false;
  private showInteractions: boolean = true;
  private replayMode: boolean = false;
  private replaySpeed: number = 1.0;
  
  constructor(container: HTMLElement) {
    this.initializeScene(container);
    this.initializePatterns();
    this.initializeState();
    this.setupEventListeners();
    this.startAnimation();
  }
  
  private initializeScene(container: HTMLElement): void {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);
    
    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 0, 10);
    
    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);
  }
  
  private initializePatterns(): void {
    this.effects = new PatternEffects(this.scene);
    this.interactions = new PatternInteractions(this.scene);
    
    // Initialize pattern visualizers
    this.convergentPattern = new ConvergentThinking(this.scene, this.effects);
    this.divergentPattern = new DivergentThinking(this.scene, this.effects);
    this.lateralPattern = new LateralThinking(this.scene, this.effects);
    this.systemsPattern = new SystemsThinking(this.scene, this.effects);
    this.criticalPattern = new CriticalThinking(this.scene, this.effects);
    this.abstractPattern = new AbstractThinking(this.scene, this.effects);
    this.adaptivePattern = new AdaptiveThinking(this.scene, this.effects);
  }
  
  private initializeState(): void {
    this.cognitiveState = {
      activePatterns: new Map(),
      patternHistory: [],
      interactionMatrix: Array(7).fill(null).map(() => Array(7).fill(0)),
      globalInhibition: 0,
      metaCognitiveLevel: 0
    };
  }
  
  private setupEventListeners(): void {
    window.addEventListener('resize', this.onWindowResize.bind(this));
    
    // Custom events for cognitive pattern data
    document.addEventListener('cognitivePatternActivated', this.onPatternActivated.bind(this));
    document.addEventListener('cognitivePatternDeactivated', this.onPatternDeactivated.bind(this));
    document.addEventListener('cognitiveStateUpdated', this.onStateUpdated.bind(this));
  }
  
  private onWindowResize(): void {
    const container = this.renderer.domElement.parentElement;
    if (!container) return;
    
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }
  
  private onPatternActivated(event: CustomEvent<CognitivePatternData>): void {
    const pattern = event.detail;
    this.cognitiveState.activePatterns.set(pattern.type, pattern);
    this.cognitiveState.patternHistory.push(pattern);
    this.updatePatternVisualization(pattern);
  }
  
  private onPatternDeactivated(event: CustomEvent<{ type: string }>): void {
    const { type } = event.detail;
    this.cognitiveState.activePatterns.delete(type);
    this.deactivatePatternVisualization(type);
  }
  
  private onStateUpdated(event: CustomEvent<Partial<CognitiveState>>): void {
    Object.assign(this.cognitiveState, event.detail);
    this.updateGlobalEffects();
  }
  
  private updatePatternVisualization(pattern: CognitivePatternData): void {
    const visualizer = this.getPatternVisualizer(pattern.type);
    if (visualizer) {
      visualizer.activate(pattern.activation, pattern.intensity, pattern.metadata);
    }
    
    // Update interactions if multiple patterns are active
    if (this.showInteractions && this.cognitiveState.activePatterns.size > 1) {
      this.interactions.updateInteractions(
        Array.from(this.cognitiveState.activePatterns.values()),
        this.cognitiveState.interactionMatrix
      );
    }
  }
  
  private deactivatePatternVisualization(type: string): void {
    const visualizer = this.getPatternVisualizer(type as CognitivePatternData['type']);
    if (visualizer) {
      visualizer.deactivate();
    }
  }
  
  private getPatternVisualizer(type: CognitivePatternData['type']) {
    switch (type) {
      case 'convergent': return this.convergentPattern;
      case 'divergent': return this.divergentPattern;
      case 'lateral': return this.lateralPattern;
      case 'systems': return this.systemsPattern;
      case 'critical': return this.criticalPattern;
      case 'abstract': return this.abstractPattern;
      case 'adaptive': return this.adaptivePattern;
      default: return null;
    }
  }
  
  private updateGlobalEffects(): void {
    // Global inhibition effects
    if (this.cognitiveState.globalInhibition > 0) {
      this.effects.applyGlobalInhibition(this.cognitiveState.globalInhibition);
    }
    
    // Meta-cognitive level effects
    if (this.cognitiveState.metaCognitiveLevel > 0.5) {
      this.adaptivePattern.showMetaCognitiveEffects(true);
    }
  }
  
  private startAnimation(): void {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate);
      this.time += 0.016; // Approximate 60fps
      
      this.updatePatterns();
      this.updateInteractions();
      this.updateDecayEffects();
      
      this.renderer.render(this.scene, this.camera);
    };
    
    animate();
  }
  
  private updatePatterns(): void {
    const currentTime = Date.now();
    
    // Update each active pattern
    for (const [type, pattern] of this.cognitiveState.activePatterns) {
      const visualizer = this.getPatternVisualizer(type);
      if (visualizer) {
        const elapsed = currentTime - pattern.timestamp;
        const decayFactor = Math.max(0, 1 - elapsed / pattern.duration);
        visualizer.update(this.time, decayFactor);
      }
    }
    
    // Clean up expired patterns
    for (const [type, pattern] of this.cognitiveState.activePatterns) {
      const elapsed = currentTime - pattern.timestamp;
      if (elapsed > pattern.duration) {
        this.cognitiveState.activePatterns.delete(type);
        this.deactivatePatternVisualization(type);
      }
    }
  }
  
  private updateInteractions(): void {
    if (this.showInteractions) {
      this.interactions.update(this.time);
    }
  }
  
  private updateDecayEffects(): void {
    this.effects.updateDecayEffects(this.time);
  }
  
  // Public API methods
  
  public setShowPatternHistory(show: boolean): void {
    this.showPatternHistory = show;
    if (show) {
      this.visualizePatternHistory();
    } else {
      this.clearPatternHistory();
    }
  }
  
  public setShowInteractions(show: boolean): void {
    this.showInteractions = show;
    if (!show) {
      this.interactions.clear();
    }
  }
  
  public startReplay(patterns: CognitivePatternData[], speed: number = 1.0): void {
    this.replayMode = true;
    this.replaySpeed = speed;
    this.replayPatternHistory(patterns);
  }
  
  public stopReplay(): void {
    this.replayMode = false;
    this.clearAllPatterns();
  }
  
  public exportVisualization(): string {
    // Export current visualization state as JSON
    return JSON.stringify({
      cognitiveState: {
        ...this.cognitiveState,
        activePatterns: Array.from(this.cognitiveState.activePatterns.entries())
      },
      settings: {
        showPatternHistory: this.showPatternHistory,
        showInteractions: this.showInteractions,
        replayMode: this.replayMode,
        replaySpeed: this.replaySpeed
      }
    });
  }
  
  private visualizePatternHistory(): void {
    // Implement pattern history visualization
    this.effects.showPatternTrails(this.cognitiveState.patternHistory);
  }
  
  private clearPatternHistory(): void {
    this.effects.clearPatternTrails();
  }
  
  private replayPatternHistory(patterns: CognitivePatternData[]): void {
    // Implement replay functionality
    let index = 0;
    const replayInterval = setInterval(() => {
      if (index >= patterns.length || !this.replayMode) {
        clearInterval(replayInterval);
        return;
      }
      
      const pattern = patterns[index];
      this.updatePatternVisualization(pattern);
      index++;
    }, (1000 / this.replaySpeed));
  }
  
  private clearAllPatterns(): void {
    for (const type of this.cognitiveState.activePatterns.keys()) {
      this.deactivatePatternVisualization(type);
    }
    this.cognitiveState.activePatterns.clear();
  }
  
  public dispose(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    
    // Dispose of all pattern visualizers
    this.convergentPattern.dispose();
    this.divergentPattern.dispose();
    this.lateralPattern.dispose();
    this.systemsPattern.dispose();
    this.criticalPattern.dispose();
    this.abstractPattern.dispose();
    this.adaptivePattern.dispose();
    
    this.effects.dispose();
    this.interactions.dispose();
    this.renderer.dispose();
  }
}