import * as d3 from 'd3';
import cytoscape, { Core, NodeSingular, EdgeSingular } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import fcose from 'cytoscape-fcose';

// Register layout extensions
cytoscape.use(dagre);
cytoscape.use(fcose);

export interface ComponentNode {
  id: string;
  label: string;
  type: 'phase' | 'engine' | 'module' | 'layer';
  phase: number;
  status: 'healthy' | 'warning' | 'error' | 'inactive';
  position?: { x: number; y: number; z?: number };
  metrics?: {
    performance: number;
    memory: number;
    connections: number;
    load: number;
  };
  metadata?: Record<string, any>;
}

export interface ComponentEdge {
  id: string;
  source: string;
  target: string;
  type: 'data_flow' | 'control' | 'feedback' | 'inhibition';
  weight?: number;
  status: 'active' | 'inactive' | 'congested';
  latency?: number;
  throughput?: number;
}

export interface ArchitectureData {
  nodes: ComponentNode[];
  edges: ComponentEdge[];
  metadata: {
    timestamp: number;
    totalComponents: number;
    activeConnections: number;
    systemHealth: number;
  };
}

export interface VisualizationConfig {
  layout: 'hierarchical' | 'brain_inspired' | 'force_directed' | 'circular';
  dimensions: '2d' | '3d';
  showLabels: boolean;
  showMetrics: boolean;
  enableAnimations: boolean;
  colorScheme: 'default' | 'accessibility' | 'dark' | 'light';
  performanceMode: 'high_quality' | 'balanced' | 'performance';
}

export class ArchitectureDiagramEngine {
  private container: HTMLElement;
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private cy: Core;
  private data: ArchitectureData;
  private config: VisualizationConfig;
  private animationFrame: number | null = null;
  private renderCache: Map<string, any> = new Map();
  private lastRenderTime = 0;
  private readonly targetFPS = 60;
  private readonly minFrameTime = 1000 / this.targetFPS;

  constructor(container: HTMLElement, config: Partial<VisualizationConfig> = {}) {
    this.container = container;
    this.config = {
      layout: 'brain_inspired',
      dimensions: '2d',
      showLabels: true,
      showMetrics: true,
      enableAnimations: true,
      colorScheme: 'default',
      performanceMode: 'balanced',
      ...config
    };

    this.initializeVisualization();
    this.data = {
      nodes: [],
      edges: [],
      metadata: {
        timestamp: Date.now(),
        totalComponents: 0,
        activeConnections: 0,
        systemHealth: 1.0
      }
    };
  }

  private initializeVisualization(): void {
    // Clear container
    d3.select(this.container).selectAll('*').remove();

    if (this.config.dimensions === '2d') {
      this.initializeD3Visualization();
    }
    this.initializeCytoscapeGraph();
  }

  private initializeD3Visualization(): void {
    const rect = this.container.getBoundingClientRect();
    
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', rect.width)
      .attr('height', rect.height)
      .style('position', 'absolute')
      .style('top', 0)
      .style('left', 0);

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 5])
      .on('zoom', (event) => {
        this.svg.select('.main-group').attr('transform', event.transform);
      });

    this.svg.call(zoom);

    // Create main group for transformations
    this.svg.append('g').attr('class', 'main-group');

    // Add gradient definitions for brain-inspired styling
    this.addGradientDefinitions();
  }

  private initializeCytoscapeGraph(): void {
    const cytoscapeContainer = document.createElement('div');
    cytoscapeContainer.style.position = 'absolute';
    cytoscapeContainer.style.top = '0';
    cytoscapeContainer.style.left = '0';
    cytoscapeContainer.style.width = '100%';
    cytoscapeContainer.style.height = '100%';
    cytoscapeContainer.style.pointerEvents = this.config.dimensions === '2d' ? 'none' : 'all';
    
    this.container.appendChild(cytoscapeContainer);

    this.cy = cytoscape({
      container: cytoscapeContainer,
      style: this.getCytoscapeStyles(),
      layout: {
        name: 'preset'
      },
      wheelSensitivity: 0.2,
      maxZoom: 5,
      minZoom: 0.1
    });

    this.setupCytoscapeEventHandlers();
  }

  private addGradientDefinitions(): void {
    const defs = this.svg.append('defs');

    // Neural gradient for phase indicators
    const neuralGradient = defs.append('radialGradient')
      .attr('id', 'neural-gradient')
      .attr('cx', '50%')
      .attr('cy', '50%')
      .attr('r', '50%');

    neuralGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#4fc3f7')
      .attr('stop-opacity', 0.8);

    neuralGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#1976d2')
      .attr('stop-opacity', 0.3);

    // Status gradients
    const statusColors = {
      healthy: ['#4caf50', '#2e7d32'],
      warning: ['#ff9800', '#f57c00'],
      error: ['#f44336', '#d32f2f'],
      inactive: ['#757575', '#424242']
    };

    Object.entries(statusColors).forEach(([status, colors]) => {
      const gradient = defs.append('radialGradient')
        .attr('id', `status-${status}`)
        .attr('cx', '50%')
        .attr('cy', '50%')
        .attr('r', '50%');

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', colors[0])
        .attr('stop-opacity', 0.9);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', colors[1])
        .attr('stop-opacity', 0.6);
    });
  }

  private getCytoscapeStyles(): any[] {
    return [
      {
        selector: 'node',
        style: {
          'width': 'data(size)',
          'height': 'data(size)',
          'background-color': 'data(color)',
          'border-width': 2,
          'border-color': '#ffffff',
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          'font-weight': 'bold',
          'color': '#ffffff',
          'text-outline-width': 2,
          'text-outline-color': '#000000',
          'opacity': 0.9
        }
      },
      {
        selector: 'node[type = "phase"]',
        style: {
          'shape': 'ellipse',
          'width': 80,
          'height': 80,
          'background-color': '#1976d2',
          'border-width': 3
        }
      },
      {
        selector: 'node[type = "engine"]',
        style: {
          'shape': 'rectangle',
          'width': 60,
          'height': 40,
          'background-color': '#388e3c'
        }
      },
      {
        selector: 'node[type = "module"]',
        style: {
          'shape': 'diamond',
          'width': 50,
          'height': 50,
          'background-color': '#f57c00'
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 'data(weight)',
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'opacity': 0.8
        }
      },
      {
        selector: 'edge[type = "data_flow"]',
        style: {
          'line-color': '#2196f3',
          'target-arrow-color': '#2196f3'
        }
      },
      {
        selector: 'edge[type = "control"]',
        style: {
          'line-color': '#4caf50',
          'target-arrow-color': '#4caf50'
        }
      },
      {
        selector: 'edge[type = "feedback"]',
        style: {
          'line-color': '#ff9800',
          'target-arrow-color': '#ff9800',
          'line-style': 'dashed'
        }
      },
      {
        selector: 'edge[type = "inhibition"]',
        style: {
          'line-color': '#f44336',
          'target-arrow-color': '#f44336',
          'target-arrow-shape': 'tee'
        }
      },
      {
        selector: '.highlighted',
        style: {
          'border-width': 4,
          'border-color': '#ffeb3b',
          'z-index': 10
        }
      },
      {
        selector: '.selected',
        style: {
          'border-width': 5,
          'border-color': '#e91e63',
          'z-index': 15
        }
      }
    ];
  }

  private setupCytoscapeEventHandlers(): void {
    // Node hover effects
    this.cy.on('mouseover', 'node', (event) => {
      const node = event.target;
      node.addClass('highlighted');
      this.showNodeTooltip(node);
    });

    this.cy.on('mouseout', 'node', (event) => {
      const node = event.target;
      node.removeClass('highlighted');
      this.hideTooltip();
    });

    // Node selection
    this.cy.on('tap', 'node', (event) => {
      const node = event.target;
      this.cy.elements().removeClass('selected');
      node.addClass('selected');
      this.highlightConnectedComponents(node);
      this.emitNodeSelectedEvent(node.data());
    });

    // Edge selection
    this.cy.on('tap', 'edge', (event) => {
      const edge = event.target;
      this.emitEdgeSelectedEvent(edge.data());
    });

    // Background tap to clear selection
    this.cy.on('tap', (event) => {
      if (event.target === this.cy) {
        this.clearSelection();
      }
    });
  }

  private showNodeTooltip(node: NodeSingular): void {
    const nodeData = node.data();
    const position = node.renderedPosition();
    
    // Create or update tooltip
    let tooltip = document.querySelector('.architecture-tooltip') as HTMLElement;
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.className = 'architecture-tooltip';
      tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        z-index: 1000;
        max-width: 200px;
      `;
      this.container.appendChild(tooltip);
    }

    let content = `<strong>${nodeData.label}</strong><br>`;
    content += `Type: ${nodeData.type}<br>`;
    content += `Status: ${nodeData.status}<br>`;
    
    if (nodeData.metrics) {
      content += `Performance: ${(nodeData.metrics.performance * 100).toFixed(1)}%<br>`;
      content += `Memory: ${nodeData.metrics.memory.toFixed(1)}MB<br>`;
      content += `Connections: ${nodeData.metrics.connections}`;
    }

    tooltip.innerHTML = content;
    tooltip.style.left = `${position.x + 10}px`;
    tooltip.style.top = `${position.y - 10}px`;
    tooltip.style.display = 'block';
  }

  private hideTooltip(): void {
    const tooltip = document.querySelector('.architecture-tooltip') as HTMLElement;
    if (tooltip) {
      tooltip.style.display = 'none';
    }
  }

  private highlightConnectedComponents(node: NodeSingular): void {
    // Highlight connected edges and nodes
    const connectedEdges = node.connectedEdges();
    const connectedNodes = connectedEdges.connectedNodes().difference(node);
    
    connectedEdges.addClass('highlighted');
    connectedNodes.addClass('highlighted');
    
    // Remove highlights after 3 seconds
    setTimeout(() => {
      connectedEdges.removeClass('highlighted');
      connectedNodes.removeClass('highlighted');
    }, 3000);
  }

  private clearSelection(): void {
    this.cy.elements().removeClass('selected highlighted');
    this.emitSelectionClearedEvent();
  }

  public updateData(newData: ArchitectureData): void {
    this.data = newData;
    this.updateVisualization();
  }

  private updateVisualization(): void {
    const now = performance.now();
    if (now - this.lastRenderTime < this.minFrameTime) {
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
      }
      this.animationFrame = requestAnimationFrame(() => this.performUpdate());
      return;
    }
    
    this.performUpdate();
  }

  private performUpdate(): void {
    this.lastRenderTime = performance.now();
    
    // Update Cytoscape graph
    this.updateCytoscapeGraph();
    
    // Update D3 visualization if in 2D mode
    if (this.config.dimensions === '2d') {
      this.updateD3Visualization();
    }
  }

  private updateCytoscapeGraph(): void {
    const elements = this.prepareCytoscapeElements();
    
    // Batch update for performance
    this.cy.batch(() => {
      this.cy.elements().remove();
      this.cy.add(elements);
      this.cy.layout(this.getLayoutConfig()).run();
    });
  }

  private prepareCytoscapeElements(): any[] {
    const elements: any[] = [];
    
    // Add nodes
    this.data.nodes.forEach(node => {
      elements.push({
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          phase: node.phase,
          status: node.status,
          size: this.getNodeSize(node),
          color: this.getNodeColor(node),
          ...node.metrics
        },
        position: node.position
      });
    });
    
    // Add edges
    this.data.edges.forEach(edge => {
      elements.push({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          weight: Math.max(1, (edge.weight || 1) * 3),
          color: this.getEdgeColor(edge),
          status: edge.status
        }
      });
    });
    
    return elements;
  }

  private getNodeSize(node: ComponentNode): number {
    const baseSize = {
      phase: 80,
      engine: 60,
      module: 50,
      layer: 40
    }[node.type] || 50;
    
    // Scale based on metrics
    const performanceMultiplier = node.metrics ? (0.8 + node.metrics.performance * 0.4) : 1;
    return Math.round(baseSize * performanceMultiplier);
  }

  private getNodeColor(node: ComponentNode): string {
    const statusColors = {
      healthy: '#4caf50',
      warning: '#ff9800',
      error: '#f44336',
      inactive: '#757575'
    };
    
    return statusColors[node.status] || '#757575';
  }

  private getEdgeColor(edge: ComponentEdge): string {
    const typeColors = {
      data_flow: '#2196f3',
      control: '#4caf50',
      feedback: '#ff9800',
      inhibition: '#f44336'
    };
    
    const baseColor = typeColors[edge.type] || '#757575';
    
    // Adjust opacity based on status
    const opacity = {
      active: 1.0,
      inactive: 0.3,
      congested: 0.8
    }[edge.status] || 0.8;
    
    return baseColor; // Color opacity handled in CSS
  }

  private updateD3Visualization(): void {
    if (!this.svg) return;
    
    const mainGroup = this.svg.select('.main-group');
    
    // Update nodes with D3 for additional visual effects
    const nodeSelection = mainGroup.selectAll('.d3-node')
      .data(this.data.nodes, (d: any) => d.id);
    
    // Enter new nodes
    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', 'd3-node')
      .attr('transform', d => `translate(${d.position?.x || 0}, ${d.position?.y || 0})`);
    
    // Add visual enhancements for brain-inspired layout
    nodeEnter.append('circle')
      .attr('class', 'neural-glow')
      .attr('r', d => this.getNodeSize(d) / 2 + 10)
      .attr('fill', 'url(#neural-gradient)')
      .attr('opacity', 0.3);
    
    // Update existing nodes
    nodeSelection.merge(nodeEnter)
      .transition()
      .duration(this.config.enableAnimations ? 300 : 0)
      .attr('transform', d => `translate(${d.position?.x || 0}, ${d.position?.y || 0})`);
    
    // Remove old nodes
    nodeSelection.exit().remove();
  }

  private getLayoutConfig(): any {
    switch (this.config.layout) {
      case 'hierarchical':
        return {
          name: 'dagre',
          rankDir: 'TB',
          nodeSep: 50,
          rankSep: 100,
          animate: this.config.enableAnimations,
          animationDuration: 500
        };
      
      case 'brain_inspired':
        return {
          name: 'fcose',
          quality: 'default',
          randomize: false,
          animate: this.config.enableAnimations,
          animationDuration: 500,
          nodeDimensionsIncludeLabels: true,
          uniformNodeDimensions: false,
          packComponents: true,
          nodeRepulsion: 4500,
          idealEdgeLength: 150,
          edgeElasticity: 0.45,
          nestingFactor: 0.1,
          gravity: 0.25,
          numIter: 2500,
          tile: true,
          tilingPaddingVertical: 10,
          tilingPaddingHorizontal: 10
        };
      
      case 'force_directed':
        return {
          name: 'cose',
          animate: this.config.enableAnimations,
          animationDuration: 500,
          nodeRepulsion: 400000,
          nodeOverlap: 10,
          idealEdgeLength: 100,
          edgeElasticity: 100,
          nestingFactor: 5,
          gravity: 80,
          numIter: 1000
        };
      
      case 'circular':
        return {
          name: 'circle',
          animate: this.config.enableAnimations,
          animationDuration: 500,
          radius: 200
        };
      
      default:
        return { name: 'preset' };
    }
  }

  // Event emission methods
  private emitNodeSelectedEvent(nodeData: any): void {
    this.container.dispatchEvent(new CustomEvent('nodeSelected', {
      detail: nodeData
    }));
  }

  private emitEdgeSelectedEvent(edgeData: any): void {
    this.container.dispatchEvent(new CustomEvent('edgeSelected', {
      detail: edgeData
    }));
  }

  private emitSelectionClearedEvent(): void {
    this.container.dispatchEvent(new CustomEvent('selectionCleared'));
  }

  // Public API methods
  public setLayout(layout: VisualizationConfig['layout']): void {
    this.config.layout = layout;
    this.cy.layout(this.getLayoutConfig()).run();
  }

  public setDimensions(dimensions: VisualizationConfig['dimensions']): void {
    this.config.dimensions = dimensions;
    this.initializeVisualization();
    this.updateVisualization();
  }

  public zoomToFit(padding = 50): void {
    this.cy.fit(this.cy.elements(), padding);
  }

  public zoomToNode(nodeId: string, zoom = 2): void {
    const node = this.cy.getElementById(nodeId);
    if (node.length > 0) {
      this.cy.animate({
        zoom: zoom,
        center: { eles: node }
      }, {
        duration: 500
      });
    }
  }

  public exportImage(format: 'png' | 'jpg' = 'png'): string {
    return this.cy.png({
      output: 'base64',
      bg: 'white',
      full: true,
      scale: 2
    });
  }

  public getPerformanceMetrics(): {
    nodeCount: number;
    edgeCount: number;
    renderTime: number;
    memoryUsage: number;
  } {
    return {
      nodeCount: this.data.nodes.length,
      edgeCount: this.data.edges.length,
      renderTime: this.lastRenderTime,
      memoryUsage: this.renderCache.size
    };
  }

  public dispose(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    
    if (this.cy) {
      this.cy.destroy();
    }
    
    this.renderCache.clear();
    
    // Remove tooltip if exists
    const tooltip = document.querySelector('.architecture-tooltip');
    if (tooltip) {
      tooltip.remove();
    }
  }
}