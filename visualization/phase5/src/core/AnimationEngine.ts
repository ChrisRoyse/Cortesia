import { ComponentNode, ComponentEdge } from './ArchitectureDiagramEngine';

export interface AnimationState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  progress: number;
  type: AnimationType;
  parameters: AnimationParameters;
}

export type AnimationType = 
  | 'node_pulse' 
  | 'edge_flow' 
  | 'status_change' 
  | 'layout_transition'
  | 'data_flow'
  | 'component_health'
  | 'path_trace'
  | 'drill_down'
  | 'zoom_focus';

export interface AnimationParameters {
  nodes?: string[];
  edges?: string[];
  startPositions?: Map<string, { x: number; y: number; z?: number }>;
  endPositions?: Map<string, { x: number; y: number; z?: number }>;
  colors?: Map<string, { from: string; to: string }>;
  opacity?: Map<string, { from: number; to: number }>;
  scale?: Map<string, { from: number; to: number }>;
  customProperties?: Map<string, any>;
}

export interface AnimationConfig {
  enableAnimations: boolean;
  globalSpeed: number;
  maxConcurrentAnimations: number;
  performanceMode: 'high_quality' | 'balanced' | 'performance';
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'bounce' | 'elastic';
  reduceMotion: boolean;
}

export interface RealTimeUpdate {
  timestamp: number;
  nodeUpdates: Map<string, Partial<ComponentNode>>;
  edgeUpdates: Map<string, Partial<ComponentEdge>>;
  systemMetrics: {
    performance: number;
    health: number;
    load: number;
    connections: number;
  };
}

export class AnimationEngine {
  private container: HTMLElement;
  private config: AnimationConfig;
  private activeAnimations: Map<string, AnimationState> = new Map();
  private animationQueue: Array<{ id: string; animation: AnimationState }> = [];
  private frameId: number | null = null;
  private lastFrameTime = 0;
  private targetFPS = 60;
  private frameTime = 1000 / this.targetFPS;
  private performanceMonitor = {
    frameCount: 0,
    totalTime: 0,
    averageFPS: 0,
    droppedFrames: 0
  };

  // Real-time data tracking
  private realTimeBuffer: RealTimeUpdate[] = [];
  private maxBufferSize = 100;
  private lastUpdateTime = 0;
  private updateInterval = 100; // 10 FPS for real-time updates

  // Animation caches for performance
  private positionCache = new Map<string, { x: number; y: number; z?: number }>();
  private colorCache = new Map<string, string>();
  private scaleCache = new Map<string, number>();
  private opacityCache = new Map<string, number>();

  constructor(container: HTMLElement, config: Partial<AnimationConfig> = {}) {
    this.container = container;
    this.config = {
      enableAnimations: true,
      globalSpeed: 1.0,
      maxConcurrentAnimations: 10,
      performanceMode: 'balanced',
      easing: 'ease-in-out',
      reduceMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
      ...config
    };

    this.initializeAnimationSystem();
  }

  private initializeAnimationSystem(): void {
    // Respect system preferences for reduced motion
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    mediaQuery.addEventListener('change', (e) => {
      this.config.reduceMotion = e.matches;
      if (e.matches) {
        this.pauseAllAnimations();
      }
    });

    // Start the animation loop
    this.startAnimationLoop();

    // Setup real-time update processing
    this.startRealTimeUpdateLoop();
  }

  private startAnimationLoop(): void {
    const animate = (currentTime: number) => {
      this.frameId = requestAnimationFrame(animate);
      
      if (currentTime - this.lastFrameTime >= this.frameTime) {
        this.updatePerformanceMetrics(currentTime);
        
        if (this.shouldSkipFrame()) {
          this.performanceMonitor.droppedFrames++;
          return;
        }
        
        this.processAnimationFrame(currentTime);
        this.lastFrameTime = currentTime;
      }
    };
    
    this.frameId = requestAnimationFrame(animate);
  }

  private startRealTimeUpdateLoop(): void {
    setInterval(() => {
      this.processRealTimeUpdates();
    }, this.updateInterval);
  }

  private updatePerformanceMetrics(currentTime: number): void {
    this.performanceMonitor.frameCount++;
    
    if (this.performanceMonitor.frameCount % 60 === 0) {
      const deltaTime = currentTime - this.performanceMonitor.totalTime;
      this.performanceMonitor.averageFPS = 1000 / (deltaTime / 60);
      this.performanceMonitor.totalTime = currentTime;
      
      // Adjust performance based on FPS
      this.adjustPerformanceSettings();
    }
  }

  private shouldSkipFrame(): boolean {
    if (this.config.performanceMode === 'performance') {
      return this.performanceMonitor.averageFPS < 30 && this.activeAnimations.size > 5;
    }
    return false;
  }

  private adjustPerformanceSettings(): void {
    const fps = this.performanceMonitor.averageFPS;
    
    if (fps < 30 && this.config.performanceMode !== 'performance') {
      console.warn('Low FPS detected, switching to performance mode');
      this.config.performanceMode = 'performance';
      this.frameTime = 1000 / 30; // Reduce to 30 FPS
    } else if (fps > 50 && this.config.performanceMode === 'performance') {
      this.config.performanceMode = 'balanced';
      this.frameTime = 1000 / 60; // Back to 60 FPS
    }
  }

  private processAnimationFrame(currentTime: number): void {
    // Process all active animations
    const completedAnimations: string[] = [];
    
    this.activeAnimations.forEach((animation, id) => {
      if (!animation.isPlaying) return;
      
      animation.currentTime = currentTime;
      animation.progress = Math.min(
        (currentTime - animation.parameters.customProperties?.get('startTime') || 0) / animation.duration,
        1.0
      );
      
      // Apply easing function
      const easedProgress = this.applyEasing(animation.progress, this.config.easing);
      
      // Update animation based on type
      this.updateAnimation(id, animation, easedProgress);
      
      // Check if animation is complete
      if (animation.progress >= 1.0) {
        completedAnimations.push(id);
      }
    });
    
    // Clean up completed animations
    completedAnimations.forEach(id => {
      this.completeAnimation(id);
    });
    
    // Start queued animations if slots available
    this.processAnimationQueue();
    
    // Emit frame update event
    this.emitAnimationEvent('frameUpdate', {
      currentTime,
      activeAnimations: this.activeAnimations.size,
      fps: this.performanceMonitor.averageFPS
    });
  }

  private updateAnimation(id: string, animation: AnimationState, progress: number): void {
    switch (animation.type) {
      case 'node_pulse':
        this.updateNodePulseAnimation(animation, progress);
        break;
      
      case 'edge_flow':
        this.updateEdgeFlowAnimation(animation, progress);
        break;
      
      case 'status_change':
        this.updateStatusChangeAnimation(animation, progress);
        break;
      
      case 'layout_transition':
        this.updateLayoutTransitionAnimation(animation, progress);
        break;
      
      case 'data_flow':
        this.updateDataFlowAnimation(animation, progress);
        break;
      
      case 'component_health':
        this.updateComponentHealthAnimation(animation, progress);
        break;
      
      case 'path_trace':
        this.updatePathTraceAnimation(animation, progress);
        break;
      
      case 'drill_down':
        this.updateDrillDownAnimation(animation, progress);
        break;
      
      case 'zoom_focus':
        this.updateZoomFocusAnimation(animation, progress);
        break;
    }
  }

  private updateNodePulseAnimation(animation: AnimationState, progress: number): void {
    const nodes = animation.parameters.nodes || [];
    const pulseScale = 1 + Math.sin(progress * Math.PI * 4) * 0.2;
    const pulseOpacity = 0.7 + Math.sin(progress * Math.PI * 4) * 0.3;
    
    nodes.forEach(nodeId => {
      this.scaleCache.set(nodeId, pulseScale);
      this.opacityCache.set(nodeId, pulseOpacity);
    });
    
    this.emitAnimationEvent('nodesPulsed', {
      nodes,
      scale: pulseScale,
      opacity: pulseOpacity
    });
  }

  private updateEdgeFlowAnimation(animation: AnimationState, progress: number): void {
    const edges = animation.parameters.edges || [];
    const flowOffset = progress * 100; // Move flow particles along edge
    
    edges.forEach(edgeId => {
      this.emitAnimationEvent('edgeFlowUpdate', {
        edgeId,
        flowOffset,
        progress
      });
    });
  }

  private updateStatusChangeAnimation(animation: AnimationState, progress: number): void {
    const colorMap = animation.parameters.colors;
    if (!colorMap) return;
    
    colorMap.forEach((colorTransition, elementId) => {
      const currentColor = this.interpolateColor(
        colorTransition.from,
        colorTransition.to,
        progress
      );
      
      this.colorCache.set(elementId, currentColor);
    });
    
    this.emitAnimationEvent('statusColorsUpdated', {
      colors: Object.fromEntries(this.colorCache)
    });
  }

  private updateLayoutTransitionAnimation(animation: AnimationState, progress: number): void {
    const startPositions = animation.parameters.startPositions;
    const endPositions = animation.parameters.endPositions;
    
    if (!startPositions || !endPositions) return;
    
    const currentPositions = new Map<string, { x: number; y: number; z?: number }>();
    
    endPositions.forEach((endPos, nodeId) => {
      const startPos = startPositions.get(nodeId);
      if (!startPos) return;
      
      const currentPos = {
        x: this.lerp(startPos.x, endPos.x, progress),
        y: this.lerp(startPos.y, endPos.y, progress),
        z: startPos.z !== undefined && endPos.z !== undefined 
          ? this.lerp(startPos.z, endPos.z, progress) 
          : undefined
      };
      
      currentPositions.set(nodeId, currentPos);
      this.positionCache.set(nodeId, currentPos);
    });
    
    this.emitAnimationEvent('layoutPositionsUpdated', {
      positions: Object.fromEntries(currentPositions)
    });
  }

  private updateDataFlowAnimation(animation: AnimationState, progress: number): void {
    // Simulate data packets flowing through the system
    const flowIntensity = Math.sin(progress * Math.PI * 2) * 0.5 + 0.5;
    const edges = animation.parameters.edges || [];
    
    edges.forEach(edgeId => {
      this.emitAnimationEvent('dataFlowUpdate', {
        edgeId,
        intensity: flowIntensity,
        progress
      });
    });
  }

  private updateComponentHealthAnimation(animation: AnimationState, progress: number): void {
    const nodes = animation.parameters.nodes || [];
    const healthPulse = Math.sin(progress * Math.PI * 6) * 0.1 + 0.9;
    
    nodes.forEach(nodeId => {
      this.emitAnimationEvent('componentHealthUpdate', {
        nodeId,
        healthIndicator: healthPulse,
        progress
      });
    });
  }

  private updatePathTraceAnimation(animation: AnimationState, progress: number): void {
    // Animate path highlighting with a traveling effect
    const nodes = animation.parameters.nodes || [];
    const segmentCount = nodes.length;
    
    nodes.forEach((nodeId, index) => {
      const segmentProgress = Math.max(0, Math.min(1, 
        (progress * segmentCount) - index
      ));
      
      const highlightIntensity = segmentProgress > 0 
        ? Math.sin(segmentProgress * Math.PI) 
        : 0;
      
      this.emitAnimationEvent('pathTraceUpdate', {
        nodeId,
        highlightIntensity,
        segmentProgress
      });
    });
  }

  private updateDrillDownAnimation(animation: AnimationState, progress: number): void {
    const targetNode = animation.parameters.nodes?.[0];
    if (!targetNode) return;
    
    const zoomScale = this.lerp(1, 3, progress);
    const focusOpacity = this.lerp(1, 0.3, progress);
    
    this.emitAnimationEvent('drillDownUpdate', {
      targetNode,
      zoomScale,
      focusOpacity,
      progress
    });
  }

  private updateZoomFocusAnimation(animation: AnimationState, progress: number): void {
    const targetNode = animation.parameters.nodes?.[0];
    if (!targetNode) return;
    
    const focusScale = 1 + Math.sin(progress * Math.PI * 2) * 0.1;
    
    this.emitAnimationEvent('zoomFocusUpdate', {
      targetNode,
      focusScale,
      progress
    });
  }

  private processRealTimeUpdates(): void {
    const now = Date.now();
    
    if (this.realTimeBuffer.length === 0 || now - this.lastUpdateTime < this.updateInterval) {
      return;
    }
    
    // Process buffered updates
    const updates = this.realTimeBuffer.splice(0, Math.min(5, this.realTimeBuffer.length));
    
    updates.forEach(update => {
      this.processRealTimeUpdate(update);
    });
    
    this.lastUpdateTime = now;
  }

  private processRealTimeUpdate(update: RealTimeUpdate): void {
    // Update node states with smooth transitions
    update.nodeUpdates.forEach((nodeUpdate, nodeId) => {
      if (nodeUpdate.status) {
        this.animateStatusChange(nodeId, nodeUpdate.status);
      }
      
      if (nodeUpdate.metrics) {
        this.animateMetricsUpdate(nodeId, nodeUpdate.metrics);
      }
    });
    
    // Update edge states
    update.edgeUpdates.forEach((edgeUpdate, edgeId) => {
      if (edgeUpdate.status) {
        this.animateEdgeStatusChange(edgeId, edgeUpdate.status);
      }
    });
    
    // Update system-level animations
    this.updateSystemHealthAnimation(update.systemMetrics);
  }

  private animateStatusChange(nodeId: string, newStatus: ComponentNode['status']): void {
    const statusColors = {
      healthy: '#4caf50',
      warning: '#ff9800',
      error: '#f44336',
      inactive: '#757575'
    };
    
    const currentColor = this.colorCache.get(nodeId) || statusColors.inactive;
    const targetColor = statusColors[newStatus];
    
    this.createAnimation(`status_${nodeId}`, 'status_change', {
      duration: 500,
      parameters: {
        colors: new Map([[nodeId, { from: currentColor, to: targetColor }]])
      }
    });
  }

  private animateMetricsUpdate(nodeId: string, metrics: ComponentNode['metrics']): void {
    if (!metrics) return;
    
    // Animate performance changes with pulse effect
    if (metrics.performance < 0.5) {
      this.createAnimation(`health_${nodeId}`, 'component_health', {
        duration: 2000,
        parameters: {
          nodes: [nodeId]
        }
      });
    }
    
    // Animate high load with pulsing
    if (metrics.load > 0.8) {
      this.createAnimation(`load_${nodeId}`, 'node_pulse', {
        duration: 1000,
        parameters: {
          nodes: [nodeId]
        }
      });
    }
  }

  private animateEdgeStatusChange(edgeId: string, newStatus: ComponentEdge['status']): void {
    if (newStatus === 'active') {
      this.createAnimation(`flow_${edgeId}`, 'edge_flow', {
        duration: 2000,
        parameters: {
          edges: [edgeId]
        }
      });
    }
  }

  private updateSystemHealthAnimation(metrics: RealTimeUpdate['systemMetrics']): void {
    // Create global health indicator animations
    if (metrics.health < 0.7) {
      this.createAnimation('system_warning', 'component_health', {
        duration: 3000,
        parameters: {
          nodes: [] // Will be populated by the system
        }
      });
    }
  }

  // Animation creation and management
  public createAnimation(
    id: string,
    type: AnimationType,
    options: {
      duration?: number;
      parameters?: Partial<AnimationParameters>;
      priority?: number;
      loop?: boolean;
    }
  ): void {
    if (this.config.reduceMotion && !this.isAccessibilityFriendlyAnimation(type)) {
      return;
    }
    
    const animation: AnimationState = {
      isPlaying: false,
      currentTime: 0,
      duration: options.duration || 1000,
      progress: 0,
      type,
      parameters: {
        nodes: [],
        edges: [],
        startPositions: new Map(),
        endPositions: new Map(),
        colors: new Map(),
        opacity: new Map(),
        scale: new Map(),
        customProperties: new Map([
          ['startTime', performance.now()],
          ['loop', options.loop || false],
          ['priority', options.priority || 0]
        ]),
        ...options.parameters
      }
    };
    
    // Handle animation queuing and prioritization
    if (this.activeAnimations.size >= this.config.maxConcurrentAnimations) {
      this.queueAnimation(id, animation);
    } else {
      this.startAnimation(id, animation);
    }
  }

  private queueAnimation(id: string, animation: AnimationState): void {
    // Remove existing queued animation with same id
    this.animationQueue = this.animationQueue.filter(item => item.id !== id);
    
    // Insert based on priority
    const priority = animation.parameters.customProperties?.get('priority') || 0;
    let insertIndex = this.animationQueue.length;
    
    for (let i = 0; i < this.animationQueue.length; i++) {
      const queuedPriority = this.animationQueue[i].animation.parameters.customProperties?.get('priority') || 0;
      if (priority > queuedPriority) {
        insertIndex = i;
        break;
      }
    }
    
    this.animationQueue.splice(insertIndex, 0, { id, animation });
  }

  private startAnimation(id: string, animation: AnimationState): void {
    // Stop existing animation with same id
    this.stopAnimation(id);
    
    animation.isPlaying = true;
    animation.parameters.customProperties?.set('startTime', performance.now());
    
    this.activeAnimations.set(id, animation);
    
    this.emitAnimationEvent('animationStarted', { id, type: animation.type });
  }

  private processAnimationQueue(): void {
    while (
      this.animationQueue.length > 0 && 
      this.activeAnimations.size < this.config.maxConcurrentAnimations
    ) {
      const { id, animation } = this.animationQueue.shift()!;
      this.startAnimation(id, animation);
    }
  }

  private completeAnimation(id: string): void {
    const animation = this.activeAnimations.get(id);
    if (!animation) return;
    
    const shouldLoop = animation.parameters.customProperties?.get('loop') === true;
    
    if (shouldLoop) {
      // Restart the animation
      animation.progress = 0;
      animation.currentTime = 0;
      animation.parameters.customProperties?.set('startTime', performance.now());
    } else {
      // Remove the animation
      this.activeAnimations.delete(id);
      this.emitAnimationEvent('animationCompleted', { id, type: animation.type });
    }
  }

  public stopAnimation(id: string): void {
    const animation = this.activeAnimations.get(id);
    if (animation) {
      animation.isPlaying = false;
      this.activeAnimations.delete(id);
      this.emitAnimationEvent('animationStopped', { id, type: animation.type });
    }
    
    // Remove from queue if present
    this.animationQueue = this.animationQueue.filter(item => item.id !== id);
  }

  public pauseAnimation(id: string): void {
    const animation = this.activeAnimations.get(id);
    if (animation) {
      animation.isPlaying = false;
      this.emitAnimationEvent('animationPaused', { id, type: animation.type });
    }
  }

  public resumeAnimation(id: string): void {
    const animation = this.activeAnimations.get(id);
    if (animation) {
      animation.isPlaying = true;
      animation.parameters.customProperties?.set('startTime', 
        performance.now() - (animation.progress * animation.duration)
      );
      this.emitAnimationEvent('animationResumed', { id, type: animation.type });
    }
  }

  public pauseAllAnimations(): void {
    this.activeAnimations.forEach((animation, id) => {
      animation.isPlaying = false;
    });
    this.emitAnimationEvent('allAnimationsPaused');
  }

  public resumeAllAnimations(): void {
    const now = performance.now();
    this.activeAnimations.forEach((animation, id) => {
      animation.isPlaying = true;
      animation.parameters.customProperties?.set('startTime', 
        now - (animation.progress * animation.duration)
      );
    });
    this.emitAnimationEvent('allAnimationsResumed');
  }

  // Real-time data integration
  public addRealTimeUpdate(update: RealTimeUpdate): void {
    this.realTimeBuffer.push(update);
    
    // Keep buffer size manageable
    if (this.realTimeBuffer.length > this.maxBufferSize) {
      this.realTimeBuffer.splice(0, this.realTimeBuffer.length - this.maxBufferSize);
    }
  }

  // Utility methods
  private applyEasing(progress: number, easing: AnimationConfig['easing']): number {
    switch (easing) {
      case 'linear':
        return progress;
      
      case 'ease-in':
        return progress * progress;
      
      case 'ease-out':
        return 1 - Math.pow(1 - progress, 2);
      
      case 'ease-in-out':
        return progress < 0.5 
          ? 2 * progress * progress 
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      
      case 'bounce':
        const n1 = 7.5625;
        const d1 = 2.75;
        
        if (progress < 1 / d1) {
          return n1 * progress * progress;
        } else if (progress < 2 / d1) {
          return n1 * (progress -= 1.5 / d1) * progress + 0.75;
        } else if (progress < 2.5 / d1) {
          return n1 * (progress -= 2.25 / d1) * progress + 0.9375;
        } else {
          return n1 * (progress -= 2.625 / d1) * progress + 0.984375;
        }
      
      case 'elastic':
        const c4 = (2 * Math.PI) / 3;
        return progress === 0 
          ? 0 
          : progress === 1 
          ? 1 
          : -Math.pow(2, 10 * progress - 10) * Math.sin((progress * 10 - 10.75) * c4);
      
      default:
        return progress;
    }
  }

  private lerp(start: number, end: number, progress: number): number {
    return start + (end - start) * progress;
  }

  private interpolateColor(startColor: string, endColor: string, progress: number): string {
    // Simple RGB interpolation
    const startRGB = this.hexToRgb(startColor);
    const endRGB = this.hexToRgb(endColor);
    
    if (!startRGB || !endRGB) return startColor;
    
    const r = Math.round(this.lerp(startRGB.r, endRGB.r, progress));
    const g = Math.round(this.lerp(startRGB.g, endRGB.g, progress));
    const b = Math.round(this.lerp(startRGB.b, endRGB.b, progress));
    
    return `rgb(${r}, ${g}, ${b})`;
  }

  private hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  }

  private isAccessibilityFriendlyAnimation(type: AnimationType): boolean {
    // These animations are considered essential for accessibility
    const accessibilityFriendlyTypes: AnimationType[] = [
      'status_change',
      'component_health'
    ];
    
    return accessibilityFriendlyTypes.includes(type);
  }

  private emitAnimationEvent(eventType: string, detail?: any): void {
    this.container.dispatchEvent(new CustomEvent(eventType, {
      detail,
      bubbles: true
    }));
  }

  // Public API methods
  public updateConfig(config: Partial<AnimationConfig>): void {
    this.config = { ...this.config, ...config };
    
    if (!this.config.enableAnimations) {
      this.pauseAllAnimations();
    }
    
    this.frameTime = 1000 / this.targetFPS / this.config.globalSpeed;
  }

  public getPerformanceMetrics() {
    return {
      ...this.performanceMonitor,
      activeAnimations: this.activeAnimations.size,
      queuedAnimations: this.animationQueue.length,
      realTimeBufferSize: this.realTimeBuffer.length
    };
  }

  public getActiveAnimations(): Map<string, AnimationState> {
    return new Map(this.activeAnimations);
  }

  public clearAnimationQueue(): void {
    this.animationQueue = [];
  }

  public dispose(): void {
    // Stop animation loop
    if (this.frameId) {
      cancelAnimationFrame(this.frameId);
    }
    
    // Clear all animations
    this.activeAnimations.clear();
    this.animationQueue = [];
    this.realTimeBuffer = [];
    
    // Clear caches
    this.positionCache.clear();
    this.colorCache.clear();
    this.scaleCache.clear();
    this.opacityCache.clear();
  }
}