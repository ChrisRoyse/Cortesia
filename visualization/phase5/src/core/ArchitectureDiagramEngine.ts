import * as d3 from 'd3';
import { gsap } from 'gsap';
import { Observable, Subject, BehaviorSubject } from 'rxjs';
import { throttleTime, debounceTime } from 'rxjs/operators';

import {
  ArchitectureData,
  ArchitectureNode,
  ConnectionEdge,
  LayoutType,
  ViewMode,
  ThemeConfiguration,
  DiagramState,
  ComponentUpdate,
  LayoutResult,
  Animation,
  NodeAnimation,
  FlowAnimation,
  LayoutAnimation,
  ExportFormat,
  InteractionType,
  PerformanceMetrics
} from '../types';

export interface ArchitectureEngineConfig {
  container: HTMLElement;
  svg: SVGSVGElement;
  width: number;
  height: number;
  theme: ThemeConfiguration;
  layout: LayoutType;
  enableAnimations: boolean;
  enableWebGL: boolean;
  maxNodes: number;
  optimizeForMobile: boolean;
}

export class ArchitectureDiagramEngine {
  private config: ArchitectureEngineConfig;
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private container: d3.Selection<HTMLElement, unknown, null, undefined>;
  private data: ArchitectureData | null = null;
  
  // Core components
  private layoutEngine: LayoutEngine;
  private animationEngine: AnimationEngine;
  private interactionEngine: InteractionEngine;
  private renderingEngine: RenderingEngine;
  
  // State management
  private currentState: DiagramState;
  private stateSubject = new BehaviorSubject<DiagramState>({
    layout: 'neural-layers',
    viewMode: 'overview',
    selectedNodes: new Set(),
    highlightedConnections: new Set(),
    zoom: { scale: 1, translate: { x: 0, y: 0 } },
    filters: {
      showLayers: [],
      showNodeTypes: ['subcortical', 'cortical', 'thalamic', 'mcp', 'storage', 'network'],
      showConnections: true,
      showMetrics: true,
    },
  });

  // Event subjects
  private nodeInteractionSubject = new Subject<{
    node: ArchitectureNode;
    interaction: InteractionType;
    event: Event;
  }>();
  
  private connectionInteractionSubject = new Subject<{
    connection: ConnectionEdge;
    interaction: InteractionType;
    event: Event;
  }>();
  
  private selectionChangeSubject = new Subject<string[]>();
  private zoomChangeSubject = new Subject<{ scale: number; translate: { x: number; y: number } }>();
  
  // Performance monitoring
  private performanceMetrics: PerformanceMetrics = {
    renderTime: 0,
    animationFPS: 60,
    memoryUsage: 0,
    nodeCount: 0,
    connectionCount: 0,
    lastMeasurement: Date.now()
  };

  private isInitialized = false;
  private animationTimeline: gsap.core.Timeline;

  constructor() {
    this.animationTimeline = gsap.timeline();
  }

  // Initialization
  async initialize(config: ArchitectureEngineConfig): Promise<void> {
    this.config = config;
    this.currentState = this.stateSubject.value;

    // Initialize D3 selections
    this.container = d3.select(config.container);
    this.svg = d3.select(config.svg);

    // Initialize core engines
    this.layoutEngine = new LayoutEngine();
    this.animationEngine = new AnimationEngine(this.animationTimeline, config.enableAnimations);
    this.interactionEngine = new InteractionEngine(this.svg, config);
    this.renderingEngine = new RenderingEngine(this.svg, config.theme);

    // Set up zoom behavior
    this.setupZoomBehavior();

    // Set up interaction handlers
    this.setupInteractionHandlers();

    // Set up performance monitoring
    this.setupPerformanceMonitoring();

    // Initialize viewport
    this.setupViewport();

    this.isInitialized = true;
  }

  // Data management
  async render(data: ArchitectureData): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Engine must be initialized before rendering');
    }

    const startTime = performance.now();

    this.data = data;
    this.updatePerformanceMetrics(data);

    // Calculate layout
    const layout = await this.layoutEngine.calculateLayout(
      data.nodes,
      data.connections,
      this.currentState.layout,
      {
        width: this.config.width,
        height: this.config.height,
        theme: this.config.theme
      }
    );

    // Render elements
    await this.renderingEngine.render(data, layout);

    // Apply initial animations
    if (this.config.enableAnimations) {
      this.animationEngine.playInitialAnimations(data.nodes);
    }

    const renderTime = performance.now() - startTime;
    this.performanceMetrics.renderTime = renderTime;
  }

  async updateData(data: ArchitectureData): Promise<void> {
    if (!this.data) {
      return this.render(data);
    }

    const startTime = performance.now();

    // Calculate differences
    const changes = this.calculateDataChanges(this.data, data);
    
    // Update data
    this.data = data;
    this.updatePerformanceMetrics(data);

    // Apply incremental updates
    if (changes.nodes.length > 0 || changes.connections.length > 0) {
      await this.applyIncrementalChanges(changes);
    }

    const renderTime = performance.now() - startTime;
    this.performanceMetrics.renderTime = renderTime;
  }

  async updateComponent(componentId: string, update: ComponentUpdate['changes']): Promise<void> {
    if (!this.data) return;

    // Find and update the node
    const nodeIndex = this.data.nodes.findIndex(n => n.id === componentId);
    if (nodeIndex === -1) return;

    const node = { ...this.data.nodes[nodeIndex] };
    
    // Apply updates
    if (update.status) node.status = update.status;
    if (update.metrics) node.metrics = { ...node.metrics, ...update.metrics };
    if (update.position) node.position = update.position;
    if (update.connections) node.connections = update.connections;

    // Update data
    this.data.nodes[nodeIndex] = node;

    // Animate the update
    if (this.config.enableAnimations) {
      await this.animationEngine.animateNodeUpdate(componentId, node);
    } else {
      // Direct update without animation
      await this.renderingEngine.updateNode(node);
    }
  }

  // Layout management
  async applyLayout(layoutType: LayoutType): Promise<void> {
    if (!this.data) return;

    this.currentState = { ...this.currentState, layout: layoutType };
    this.stateSubject.next(this.currentState);

    const newLayout = await this.layoutEngine.calculateLayout(
      this.data.nodes,
      this.data.connections,
      layoutType,
      {
        width: this.config.width,
        height: this.config.height,
        theme: this.config.theme
      }
    );

    if (this.config.enableAnimations) {
      await this.animationEngine.animateLayoutTransition(newLayout);
    }

    await this.renderingEngine.updateLayout(newLayout);
  }

  setViewMode(viewMode: ViewMode): void {
    this.currentState = { ...this.currentState, viewMode };
    this.stateSubject.next(this.currentState);
    
    this.renderingEngine.setViewMode(viewMode);
  }

  // Selection and interaction
  selectNodes(nodeIds: string[], mode: 'replace' | 'add' | 'toggle' = 'replace'): void {
    const selectedNodes = new Set(this.currentState.selectedNodes);

    switch (mode) {
      case 'replace':
        selectedNodes.clear();
        nodeIds.forEach(id => selectedNodes.add(id));
        break;
      case 'add':
        nodeIds.forEach(id => selectedNodes.add(id));
        break;
      case 'toggle':
        nodeIds.forEach(id => {
          if (selectedNodes.has(id)) {
            selectedNodes.delete(id);
          } else {
            selectedNodes.add(id);
          }
        });
        break;
    }

    this.currentState = { ...this.currentState, selectedNodes };
    this.stateSubject.next(this.currentState);
    
    this.renderingEngine.updateSelection(Array.from(selectedNodes));
    this.selectionChangeSubject.next(Array.from(selectedNodes));
  }

  focusOnNode(nodeId: string): void {
    if (!this.data) return;

    const node = this.data.nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Animate to center the node
    const targetTransform = {
      x: this.config.width / 2 - node.position.x,
      y: this.config.height / 2 - node.position.y,
      k: 1.5
    };

    this.animateToTransform(targetTransform);
  }

  // Zoom and pan controls
  zoomIn(): void {
    this.zoomBy(1.5);
  }

  zoomOut(): void {
    this.zoomBy(1 / 1.5);
  }

  zoomToFit(): void {
    if (!this.data || this.data.nodes.length === 0) return;

    // Calculate bounds
    const bounds = this.calculateNodeBounds(this.data.nodes);
    const padding = 50;

    const width = bounds.maxX - bounds.minX + padding * 2;
    const height = bounds.maxY - bounds.minY + padding * 2;

    const scale = Math.min(
      this.config.width / width,
      this.config.height / height,
      2 // Maximum zoom
    );

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    const targetTransform = {
      x: this.config.width / 2 - centerX * scale,
      y: this.config.height / 2 - centerY * scale,
      k: scale
    };

    this.animateToTransform(targetTransform);
  }

  resetView(): void {
    const targetTransform = { x: 0, y: 0, k: 1 };
    this.animateToTransform(targetTransform);
  }

  private zoomBy(factor: number): void {
    const currentTransform = this.currentState.zoom;
    const newScale = Math.max(0.1, Math.min(5, currentTransform.scale * factor));
    
    const targetTransform = {
      x: currentTransform.translate.x,
      y: currentTransform.translate.y,
      k: newScale
    };

    this.animateToTransform(targetTransform);
  }

  // Export functionality
  async exportDiagram(format: ExportFormat): Promise<Blob> {
    if (!this.svg) {
      throw new Error('SVG not available for export');
    }

    switch (format) {
      case 'svg':
        return this.exportSVG();
      case 'png':
        return this.exportPNG();
      case 'json':
        return this.exportJSON();
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  // Event handlers
  onNodeInteraction(handler: (node: ArchitectureNode, interaction: InteractionType, event: Event) => void): void {
    this.nodeInteractionSubject.subscribe(({ node, interaction, event }) => {
      handler(node, interaction, event);
    });
  }

  onConnectionInteraction(handler: (connection: ConnectionEdge, interaction: InteractionType, event: Event) => void): void {
    this.connectionInteractionSubject.subscribe(({ connection, interaction, event }) => {
      handler(connection, interaction, event);
    });
  }

  onSelectionChange(handler: (selectedIds: string[]) => void): void {
    this.selectionChangeSubject.subscribe(handler);
  }

  onZoomChange(handler: (zoom: { scale: number; translate: { x: number; y: number } }) => void): void {
    this.zoomChangeSubject.subscribe(handler);
  }

  // Performance monitoring
  getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  // Cleanup
  dispose(): void {
    this.animationTimeline.kill();
    this.interactionEngine?.dispose();
    this.nodeInteractionSubject.complete();
    this.connectionInteractionSubject.complete();
    this.selectionChangeSubject.complete();
    this.zoomChangeSubject.complete();
    this.stateSubject.complete();
  }

  // Private methods
  private setupZoomBehavior(): void {
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 5])
      .on('zoom', (event) => {
        const { transform } = event;
        
        this.currentState = {
          ...this.currentState,
          zoom: {
            scale: transform.k,
            translate: { x: transform.x, y: transform.y }
          }
        };

        this.stateSubject.next(this.currentState);
        this.zoomChangeSubject.next({
          scale: transform.k,
          translate: { x: transform.x, y: transform.y }
        });

        this.renderingEngine.applyTransform(transform);
      });

    this.svg.call(zoom);
  }

  private setupInteractionHandlers(): void {
    // Node interactions will be handled by the InteractionEngine
    // and forwarded through the subjects
  }

  private setupPerformanceMonitoring(): void {
    const updatePerformance = () => {
      const now = performance.now();
      const memory = (performance as any).memory;
      
      if (memory) {
        this.performanceMetrics.memoryUsage = memory.usedJSHeapSize;
      }
      
      this.performanceMetrics.lastMeasurement = now;
      
      // Schedule next update
      requestAnimationFrame(updatePerformance);
    };

    requestAnimationFrame(updatePerformance);
  }

  private setupViewport(): void {
    this.svg
      .attr('width', this.config.width)
      .attr('height', this.config.height)
      .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
  }

  private async animateToTransform(transform: { x: number; y: number; k: number }): Promise<void> {
    return new Promise((resolve) => {
      const svg = this.svg.node();
      if (!svg) {
        resolve();
        return;
      }

      const zoom = d3.zoom<SVGSVGElement, unknown>();
      const currentTransform = d3.zoomTransform(svg);

      this.svg
        .transition()
        .duration(750)
        .call(
          zoom.transform,
          d3.zoomIdentity
            .translate(transform.x, transform.y)
            .scale(transform.k)
        )
        .on('end', resolve);
    });
  }

  private calculateDataChanges(oldData: ArchitectureData, newData: ArchitectureData) {
    const nodeChanges = newData.nodes.filter(newNode => {
      const oldNode = oldData.nodes.find(n => n.id === newNode.id);
      return !oldNode || JSON.stringify(oldNode) !== JSON.stringify(newNode);
    });

    const connectionChanges = newData.connections.filter(newConn => {
      const oldConn = oldData.connections.find(c => c.id === newConn.id);
      return !oldConn || JSON.stringify(oldConn) !== JSON.stringify(newConn);
    });

    return {
      nodes: nodeChanges,
      connections: connectionChanges
    };
  }

  private async applyIncrementalChanges(changes: {
    nodes: ArchitectureNode[];
    connections: ConnectionEdge[];
  }): Promise<void> {
    if (this.config.enableAnimations) {
      const animations = changes.nodes.map(node => 
        this.animationEngine.animateNodeUpdate(node.id, node)
      );
      await Promise.all(animations);
    } else {
      // Apply changes directly
      for (const node of changes.nodes) {
        await this.renderingEngine.updateNode(node);
      }
      for (const connection of changes.connections) {
        await this.renderingEngine.updateConnection(connection);
      }
    }
  }

  private updatePerformanceMetrics(data: ArchitectureData): void {
    this.performanceMetrics.nodeCount = data.nodes.length;
    this.performanceMetrics.connectionCount = data.connections.length;
  }

  private calculateNodeBounds(nodes: ArchitectureNode[]) {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    nodes.forEach(node => {
      const radius = node.size || 30;
      minX = Math.min(minX, node.position.x - radius);
      maxX = Math.max(maxX, node.position.x + radius);
      minY = Math.min(minY, node.position.y - radius);
      maxY = Math.max(maxY, node.position.y + radius);
    });

    return { minX, maxX, minY, maxY };
  }

  private async exportSVG(): Promise<Blob> {
    const svgElement = this.svg.node();
    if (!svgElement) throw new Error('SVG element not found');

    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    
    return new Blob([svgString], { type: 'image/svg+xml' });
  }

  private async exportPNG(): Promise<Blob> {
    const svgBlob = await this.exportSVG();
    const svgUrl = URL.createObjectURL(svgBlob);
    
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;
        
        canvas.width = this.config.width * 2; // 2x resolution
        canvas.height = this.config.height * 2;
        ctx.scale(2, 2);
        
        ctx.drawImage(img, 0, 0);
        
        canvas.toBlob((blob) => {
          URL.revokeObjectURL(svgUrl);
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Failed to create PNG blob'));
          }
        }, 'image/png');
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(svgUrl);
        reject(new Error('Failed to load SVG for PNG conversion'));
      };
      
      img.src = svgUrl;
    });
  }

  private async exportJSON(): Promise<Blob> {
    const exportData = {
      ...this.data,
      state: this.currentState,
      metadata: {
        exported: Date.now(),
        version: '1.0.0',
        engine: 'ArchitectureDiagramEngine'
      }
    };

    const jsonString = JSON.stringify(exportData, null, 2);
    return new Blob([jsonString], { type: 'application/json' });
  }
}

// Placeholder classes that would be implemented separately
class LayoutEngine {
  async calculateLayout(
    nodes: ArchitectureNode[],
    connections: ConnectionEdge[],
    layoutType: LayoutType,
    constraints: { width: number; height: number; theme: ThemeConfiguration }
  ): Promise<LayoutResult> {
    // This would contain the actual layout algorithm implementations
    return {
      nodes: nodes.map(node => ({ id: node.id, position: node.position })),
      connections: connections.map(conn => ({
        id: conn.id,
        path: `M0,0 L100,100`, // Placeholder
        midpoint: { x: 50, y: 50 },
        angle: 0
      })),
      metadata: {
        algorithm: layoutType,
        quality: 1.0,
        executionTime: performance.now()
      }
    };
  }
}

class AnimationEngine {
  constructor(
    private timeline: gsap.core.Timeline,
    private enabled: boolean
  ) {}

  playInitialAnimations(nodes: ArchitectureNode[]): void {
    if (!this.enabled) return;
    // Implementation would go here
  }

  async animateNodeUpdate(nodeId: string, node: ArchitectureNode): Promise<void> {
    if (!this.enabled) return;
    // Implementation would go here
  }

  async animateLayoutTransition(layout: LayoutResult): Promise<void> {
    if (!this.enabled) return;
    // Implementation would go here
  }
}

class InteractionEngine {
  constructor(
    private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    private config: ArchitectureEngineConfig
  ) {}

  dispose(): void {
    // Cleanup event listeners
  }
}

class RenderingEngine {
  constructor(
    private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    private theme: ThemeConfiguration
  ) {}

  async render(data: ArchitectureData, layout: LayoutResult): Promise<void> {
    // Implementation would go here
  }

  async updateNode(node: ArchitectureNode): Promise<void> {
    // Implementation would go here
  }

  async updateConnection(connection: ConnectionEdge): Promise<void> {
    // Implementation would go here
  }

  async updateLayout(layout: LayoutResult): Promise<void> {
    // Implementation would go here
  }

  setViewMode(viewMode: ViewMode): void {
    // Implementation would go here
  }

  updateSelection(selectedNodeIds: string[]): void {
    // Implementation would go here
  }

  applyTransform(transform: d3.ZoomTransform): void {
    // Apply zoom/pan transform to the rendering
  }
}