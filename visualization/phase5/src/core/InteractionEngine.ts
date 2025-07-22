import { ComponentNode, ComponentEdge, ArchitectureDiagramEngine } from './ArchitectureDiagramEngine';

export interface InteractionState {
  mode: 'navigation' | 'selection' | 'path_tracing' | 'drilling';
  selectedNodes: Set<string>;
  selectedEdges: Set<string>;
  hoveredElement: { type: 'node' | 'edge'; id: string } | null;
  pathTrace: {
    active: boolean;
    startNode: string | null;
    endNode: string | null;
    path: string[];
    highlightedElements: Set<string>;
  };
  drillDown: {
    active: boolean;
    targetNode: string | null;
    level: number;
    history: string[];
  };
}

export interface InteractionConfig {
  enableMultiSelect: boolean;
  enablePathTracing: boolean;
  enableDrillDown: boolean;
  enableTooltips: boolean;
  enableContextMenu: boolean;
  selectionColor: string;
  hoverColor: string;
  pathTraceColor: string;
  animationDuration: number;
}

export interface PathTraceResult {
  path: ComponentNode[];
  edges: ComponentEdge[];
  totalLatency: number;
  bottlenecks: string[];
  pathEfficiency: number;
}

export interface DrillDownLevel {
  nodeId: string;
  children: ComponentNode[];
  internalEdges: ComponentEdge[];
  externalConnections: ComponentEdge[];
  metadata: {
    level: number;
    parentNode: string | null;
    viewType: 'internal' | 'detailed' | 'metrics';
  };
}

export class InteractionEngine {
  private container: HTMLElement;
  private diagramEngine: ArchitectureDiagramEngine;
  private state: InteractionState;
  private config: InteractionConfig;
  private eventListeners: Map<string, EventListener[]> = new Map();
  private contextMenu: HTMLElement | null = null;
  private pathHighlightTimeout: number | null = null;

  constructor(
    container: HTMLElement,
    diagramEngine: ArchitectureDiagramEngine,
    config: Partial<InteractionConfig> = {}
  ) {
    this.container = container;
    this.diagramEngine = diagramEngine;
    
    this.config = {
      enableMultiSelect: true,
      enablePathTracing: true,
      enableDrillDown: true,
      enableTooltips: true,
      enableContextMenu: true,
      selectionColor: '#e91e63',
      hoverColor: '#ffeb3b',
      pathTraceColor: '#00bcd4',
      animationDuration: 300,
      ...config
    };

    this.state = {
      mode: 'navigation',
      selectedNodes: new Set(),
      selectedEdges: new Set(),
      hoveredElement: null,
      pathTrace: {
        active: false,
        startNode: null,
        endNode: null,
        path: [],
        highlightedElements: new Set()
      },
      drillDown: {
        active: false,
        targetNode: null,
        level: 0,
        history: []
      }
    };

    this.initializeInteractions();
  }

  private initializeInteractions(): void {
    this.setupMouseEvents();
    this.setupKeyboardEvents();
    this.setupTouchEvents();
    this.setupContainerEvents();
    this.createContextMenu();
  }

  private setupMouseEvents(): void {
    // Node interactions
    this.container.addEventListener('mousedown', this.handleMouseDown.bind(this));
    this.container.addEventListener('mouseup', this.handleMouseUp.bind(this));
    this.container.addEventListener('mousemove', this.handleMouseMove.bind(this));
    this.container.addEventListener('wheel', this.handleWheel.bind(this));
    
    // Context menu
    this.container.addEventListener('contextmenu', this.handleContextMenu.bind(this));
    
    // Prevent default drag behavior
    this.container.addEventListener('dragstart', (e) => e.preventDefault());
  }

  private setupKeyboardEvents(): void {
    document.addEventListener('keydown', this.handleKeyDown.bind(this));
    document.addEventListener('keyup', this.handleKeyUp.bind(this));
  }

  private setupTouchEvents(): void {
    // Touch support for mobile/tablet devices
    this.container.addEventListener('touchstart', this.handleTouchStart.bind(this));
    this.container.addEventListener('touchmove', this.handleTouchMove.bind(this));
    this.container.addEventListener('touchend', this.handleTouchEnd.bind(this));
    
    // Prevent default touch behaviors
    this.container.addEventListener('touchstart', (e) => {
      if (e.touches.length > 1) {
        e.preventDefault(); // Prevent zoom
      }
    }, { passive: false });
  }

  private setupContainerEvents(): void {
    // Listen to diagram engine events
    this.container.addEventListener('nodeSelected', this.handleNodeSelected.bind(this));
    this.container.addEventListener('edgeSelected', this.handleEdgeSelected.bind(this));
    this.container.addEventListener('selectionCleared', this.handleSelectionCleared.bind(this));
    
    // Resize handler
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  private handleMouseDown(event: MouseEvent): void {
    const element = this.getElementAtPoint(event.clientX, event.clientY);
    
    if (!element) {
      if (!event.ctrlKey && !event.metaKey) {
        this.clearSelection();
      }
      return;
    }

    switch (this.state.mode) {
      case 'selection':
        this.handleSelectionModeClick(element, event);
        break;
      
      case 'path_tracing':
        this.handlePathTraceModeClick(element);
        break;
      
      case 'drilling':
        this.handleDrillModeClick(element);
        break;
      
      default:
        this.handleNavigationModeClick(element, event);
    }
  }

  private handleMouseUp(event: MouseEvent): void {
    // Handle drag end, selection completion, etc.
    this.emitInteractionEvent('mouseUp', {
      clientX: event.clientX,
      clientY: event.clientY,
      element: this.getElementAtPoint(event.clientX, event.clientY)
    });
  }

  private handleMouseMove(event: MouseEvent): void {
    const element = this.getElementAtPoint(event.clientX, event.clientY);
    
    // Update hover state
    if (element && (!this.state.hoveredElement || this.state.hoveredElement.id !== element.id)) {
      this.setHoveredElement(element);
    } else if (!element && this.state.hoveredElement) {
      this.clearHoveredElement();
    }

    // Update cursor
    this.updateCursor(element);
  }

  private handleWheel(event: WheelEvent): void {
    event.preventDefault();
    
    // Zoom functionality
    const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
    this.emitInteractionEvent('zoom', {
      factor: zoomFactor,
      centerX: event.clientX,
      centerY: event.clientY
    });
  }

  private handleContextMenu(event: MouseEvent): void {
    if (!this.config.enableContextMenu) return;
    
    event.preventDefault();
    const element = this.getElementAtPoint(event.clientX, event.clientY);
    
    this.showContextMenu(event.clientX, event.clientY, element);
  }

  private handleKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'Escape':
        this.clearSelection();
        this.exitPathTraceMode();
        this.exitDrillDownMode();
        break;
      
      case 'Delete':
      case 'Backspace':
        if (this.state.selectedNodes.size > 0 || this.state.selectedEdges.size > 0) {
          this.emitInteractionEvent('deleteSelection', {
            nodes: Array.from(this.state.selectedNodes),
            edges: Array.from(this.state.selectedEdges)
          });
        }
        break;
      
      case 'a':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.selectAll();
        }
        break;
      
      case 'f':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.emitInteractionEvent('focusRequest');
        }
        break;
      
      case 'p':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.togglePathTraceMode();
        }
        break;
      
      case 'd':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.toggleDrillDownMode();
        }
        break;
      
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
        if (!event.ctrlKey && !event.metaKey) {
          const mode = ['navigation', 'selection', 'path_tracing', 'drilling'][parseInt(event.key) - 1];
          if (mode) this.setInteractionMode(mode as InteractionState['mode']);
        }
        break;
    }
  }

  private handleKeyUp(event: KeyboardEvent): void {
    // Handle key release events if needed
  }

  private handleTouchStart(event: TouchEvent): void {
    if (event.touches.length === 1) {
      const touch = event.touches[0];
      this.handleMouseDown({
        clientX: touch.clientX,
        clientY: touch.clientY,
        ctrlKey: false,
        metaKey: false
      } as MouseEvent);
    } else if (event.touches.length === 2) {
      // Two-finger gestures (zoom, rotate)
      this.handleMultiTouchStart(event);
    }
  }

  private handleTouchMove(event: TouchEvent): void {
    if (event.touches.length === 1) {
      const touch = event.touches[0];
      this.handleMouseMove({
        clientX: touch.clientX,
        clientY: touch.clientY
      } as MouseEvent);
    } else if (event.touches.length === 2) {
      this.handleMultiTouchMove(event);
    }
  }

  private handleTouchEnd(event: TouchEvent): void {
    if (event.changedTouches.length === 1) {
      const touch = event.changedTouches[0];
      this.handleMouseUp({
        clientX: touch.clientX,
        clientY: touch.clientY
      } as MouseEvent);
    }
  }

  private handleMultiTouchStart(event: TouchEvent): void {
    // Store initial touch positions for gesture calculations
    const touch1 = event.touches[0];
    const touch2 = event.touches[1];
    
    this.emitInteractionEvent('multiTouchStart', {
      touch1: { x: touch1.clientX, y: touch1.clientY },
      touch2: { x: touch2.clientX, y: touch2.clientY }
    });
  }

  private handleMultiTouchMove(event: TouchEvent): void {
    event.preventDefault();
    
    const touch1 = event.touches[0];
    const touch2 = event.touches[1];
    
    // Calculate zoom and rotation
    const dx = touch2.clientX - touch1.clientX;
    const dy = touch2.clientY - touch1.clientY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);
    
    this.emitInteractionEvent('multiTouchMove', {
      touch1: { x: touch1.clientX, y: touch1.clientY },
      touch2: { x: touch2.clientX, y: touch2.clientY },
      distance,
      angle
    });
  }

  private handleResize(): void {
    // Update container dimensions and notify diagram engine
    this.emitInteractionEvent('containerResize', {
      width: this.container.clientWidth,
      height: this.container.clientHeight
    });
  }

  private handleNodeSelected(event: CustomEvent): void {
    const nodeData = event.detail;
    this.selectNode(nodeData.id, false);
  }

  private handleEdgeSelected(event: CustomEvent): void {
    const edgeData = event.detail;
    this.selectEdge(edgeData.id, false);
  }

  private handleSelectionCleared(event: CustomEvent): void {
    this.clearSelection();
  }

  private getElementAtPoint(x: number, y: number): { type: 'node' | 'edge'; id: string } | null {
    // This would typically interact with the rendering system
    // For now, we'll emit an event to request element information
    const result = { element: null as any };
    
    this.container.dispatchEvent(new CustomEvent('getElementAtPoint', {
      detail: { x, y, result }
    }));
    
    return result.element;
  }

  private handleSelectionModeClick(element: { type: 'node' | 'edge'; id: string }, event: MouseEvent): void {
    const isMultiSelect = this.config.enableMultiSelect && (event.ctrlKey || event.metaKey);
    
    if (element.type === 'node') {
      this.selectNode(element.id, !isMultiSelect);
    } else if (element.type === 'edge') {
      this.selectEdge(element.id, !isMultiSelect);
    }
  }

  private handlePathTraceModeClick(element: { type: 'node' | 'edge'; id: string }): void {
    if (element.type !== 'node') return;
    
    if (!this.state.pathTrace.startNode) {
      this.setPathTraceStart(element.id);
    } else if (!this.state.pathTrace.endNode && element.id !== this.state.pathTrace.startNode) {
      this.setPathTraceEnd(element.id);
      this.computeAndDisplayPath();
    } else {
      // Reset path tracing
      this.resetPathTrace();
      this.setPathTraceStart(element.id);
    }
  }

  private handleDrillModeClick(element: { type: 'node' | 'edge'; id: string }): void {
    if (element.type === 'node') {
      this.drillDownIntoNode(element.id);
    }
  }

  private handleNavigationModeClick(element: { type: 'node' | 'edge'; id: string }, event: MouseEvent): void {
    // In navigation mode, single click selects, double click drills down
    if (event.detail === 1) {
      // Single click
      setTimeout(() => {
        if (event.detail === 1) {
          if (element.type === 'node') {
            this.selectNode(element.id, true);
          } else if (element.type === 'edge') {
            this.selectEdge(element.id, true);
          }
        }
      }, 200);
    } else if (event.detail === 2) {
      // Double click
      if (element.type === 'node' && this.config.enableDrillDown) {
        this.drillDownIntoNode(element.id);
      }
    }
  }

  private setHoveredElement(element: { type: 'node' | 'edge'; id: string }): void {
    this.state.hoveredElement = element;
    this.emitInteractionEvent('elementHovered', element);
    
    if (this.config.enableTooltips) {
      this.showElementTooltip(element);
    }
  }

  private clearHoveredElement(): void {
    if (this.state.hoveredElement) {
      this.emitInteractionEvent('elementUnhovered', this.state.hoveredElement);
      this.state.hoveredElement = null;
      this.hideElementTooltip();
    }
  }

  private updateCursor(element: { type: 'node' | 'edge'; id: string } | null): void {
    let cursor = 'default';
    
    switch (this.state.mode) {
      case 'selection':
        cursor = element ? 'pointer' : 'default';
        break;
      case 'path_tracing':
        cursor = element?.type === 'node' ? 'crosshair' : 'default';
        break;
      case 'drilling':
        cursor = element?.type === 'node' ? 'zoom-in' : 'default';
        break;
      default:
        cursor = element ? 'pointer' : 'grab';
    }
    
    this.container.style.cursor = cursor;
  }

  private createContextMenu(): void {
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'architecture-context-menu';
    this.contextMenu.style.cssText = `
      position: absolute;
      background: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      padding: 4px 0;
      min-width: 120px;
      z-index: 1000;
      display: none;
      font-size: 14px;
      user-select: none;
    `;
    
    document.body.appendChild(this.contextMenu);
  }

  private showContextMenu(x: number, y: number, element: { type: 'node' | 'edge'; id: string } | null): void {
    if (!this.contextMenu) return;
    
    const menuItems = this.getContextMenuItems(element);
    
    this.contextMenu.innerHTML = '';
    menuItems.forEach(item => {
      const menuItem = document.createElement('div');
      menuItem.className = 'context-menu-item';
      menuItem.textContent = item.label;
      menuItem.style.cssText = `
        padding: 8px 16px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
      `;
      
      menuItem.addEventListener('mouseenter', () => {
        menuItem.style.backgroundColor = '#f0f0f0';
      });
      
      menuItem.addEventListener('mouseleave', () => {
        menuItem.style.backgroundColor = 'transparent';
      });
      
      menuItem.addEventListener('click', () => {
        item.action();
        this.hideContextMenu();
      });
      
      this.contextMenu!.appendChild(menuItem);
    });
    
    // Position menu
    this.contextMenu.style.left = `${x}px`;
    this.contextMenu.style.top = `${y}px`;
    this.contextMenu.style.display = 'block';
    
    // Hide menu when clicking outside
    const hideOnClick = (e: MouseEvent) => {
      if (!this.contextMenu!.contains(e.target as Node)) {
        this.hideContextMenu();
        document.removeEventListener('click', hideOnClick);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('click', hideOnClick);
    }, 0);
  }

  private hideContextMenu(): void {
    if (this.contextMenu) {
      this.contextMenu.style.display = 'none';
    }
  }

  private getContextMenuItems(element: { type: 'node' | 'edge'; id: string } | null) {
    const items: { label: string; action: () => void }[] = [];
    
    if (element) {
      if (element.type === 'node') {
        items.push(
          { label: 'Select Node', action: () => this.selectNode(element.id, true) },
          { label: 'Focus on Node', action: () => this.focusOnNode(element.id) },
          { label: 'Drill Down', action: () => this.drillDownIntoNode(element.id) },
          { label: 'Trace Paths From', action: () => this.startPathTraceFrom(element.id) },
          { label: 'Show Details', action: () => this.showNodeDetails(element.id) }
        );
      } else if (element.type === 'edge') {
        items.push(
          { label: 'Select Edge', action: () => this.selectEdge(element.id, true) },
          { label: 'Highlight Path', action: () => this.highlightEdgePath(element.id) },
          { label: 'Show Details', action: () => this.showEdgeDetails(element.id) }
        );
      }
      items.push({ label: 'Separator', action: () => {} });
    }
    
    items.push(
      { label: 'Fit to View', action: () => this.fitToView() },
      { label: 'Reset Zoom', action: () => this.resetZoom() },
      { label: 'Clear Selection', action: () => this.clearSelection() }
    );
    
    if (this.state.pathTrace.active) {
      items.push({ label: 'Exit Path Trace', action: () => this.exitPathTraceMode() });
    }
    
    if (this.state.drillDown.active) {
      items.push({ label: 'Go Back', action: () => this.drillDownGoBack() });
    }
    
    return items;
  }

  private showElementTooltip(element: { type: 'node' | 'edge'; id: string }): void {
    // This would show detailed tooltips - implementation depends on the specific visualization
    this.emitInteractionEvent('showTooltip', element);
  }

  private hideElementTooltip(): void {
    this.emitInteractionEvent('hideTooltip');
  }

  // Selection methods
  public selectNode(nodeId: string, clearOthers = true): void {
    if (clearOthers) {
      this.clearSelection();
    }
    
    this.state.selectedNodes.add(nodeId);
    this.emitInteractionEvent('nodeSelectionChanged', {
      nodeId,
      selected: true,
      allSelected: Array.from(this.state.selectedNodes)
    });
  }

  public selectEdge(edgeId: string, clearOthers = true): void {
    if (clearOthers) {
      this.clearSelection();
    }
    
    this.state.selectedEdges.add(edgeId);
    this.emitInteractionEvent('edgeSelectionChanged', {
      edgeId,
      selected: true,
      allSelected: Array.from(this.state.selectedEdges)
    });
  }

  public deselectNode(nodeId: string): void {
    this.state.selectedNodes.delete(nodeId);
    this.emitInteractionEvent('nodeSelectionChanged', {
      nodeId,
      selected: false,
      allSelected: Array.from(this.state.selectedNodes)
    });
  }

  public deselectEdge(edgeId: string): void {
    this.state.selectedEdges.delete(edgeId);
    this.emitInteractionEvent('edgeSelectionChanged', {
      edgeId,
      selected: false,
      allSelected: Array.from(this.state.selectedEdges)
    });
  }

  public clearSelection(): void {
    const hadSelection = this.state.selectedNodes.size > 0 || this.state.selectedEdges.size > 0;
    
    this.state.selectedNodes.clear();
    this.state.selectedEdges.clear();
    
    if (hadSelection) {
      this.emitInteractionEvent('selectionCleared');
    }
  }

  public selectAll(): void {
    // Request all nodes and edges from the diagram engine
    this.emitInteractionEvent('selectAllRequest');
  }

  // Path tracing methods
  public togglePathTraceMode(): void {
    if (this.state.mode === 'path_tracing') {
      this.exitPathTraceMode();
    } else {
      this.setInteractionMode('path_tracing');
    }
  }

  public setPathTraceStart(nodeId: string): void {
    this.state.pathTrace.startNode = nodeId;
    this.state.pathTrace.endNode = null;
    this.state.pathTrace.path = [];
    this.clearPathHighlight();
    
    this.emitInteractionEvent('pathTraceStartSet', { nodeId });
  }

  public setPathTraceEnd(nodeId: string): void {
    this.state.pathTrace.endNode = nodeId;
    this.emitInteractionEvent('pathTraceEndSet', { nodeId });
  }

  public computeAndDisplayPath(): void {
    if (!this.state.pathTrace.startNode || !this.state.pathTrace.endNode) return;
    
    this.emitInteractionEvent('computePathRequest', {
      startNode: this.state.pathTrace.startNode,
      endNode: this.state.pathTrace.endNode
    });
  }

  public displayPath(pathResult: PathTraceResult): void {
    this.state.pathTrace.path = pathResult.path.map(node => node.id);
    
    // Highlight path elements
    this.clearPathHighlight();
    pathResult.path.forEach(node => {
      this.state.pathTrace.highlightedElements.add(node.id);
    });
    pathResult.edges.forEach(edge => {
      this.state.pathTrace.highlightedElements.add(edge.id);
    });
    
    this.emitInteractionEvent('pathDisplayed', pathResult);
    
    // Auto-clear highlight after some time
    if (this.pathHighlightTimeout) {
      clearTimeout(this.pathHighlightTimeout);
    }
    this.pathHighlightTimeout = window.setTimeout(() => {
      this.clearPathHighlight();
    }, 10000);
  }

  public clearPathHighlight(): void {
    this.state.pathTrace.highlightedElements.clear();
    this.emitInteractionEvent('pathHighlightCleared');
  }

  public resetPathTrace(): void {
    this.state.pathTrace.startNode = null;
    this.state.pathTrace.endNode = null;
    this.state.pathTrace.path = [];
    this.clearPathHighlight();
    this.emitInteractionEvent('pathTraceReset');
  }

  public exitPathTraceMode(): void {
    this.resetPathTrace();
    this.state.pathTrace.active = false;
    if (this.state.mode === 'path_tracing') {
      this.setInteractionMode('navigation');
    }
  }

  public startPathTraceFrom(nodeId: string): void {
    this.setInteractionMode('path_tracing');
    this.setPathTraceStart(nodeId);
  }

  // Drill down methods
  public toggleDrillDownMode(): void {
    if (this.state.mode === 'drilling') {
      this.exitDrillDownMode();
    } else {
      this.setInteractionMode('drilling');
    }
  }

  public drillDownIntoNode(nodeId: string): void {
    this.state.drillDown.history.push(this.state.drillDown.targetNode || '');
    this.state.drillDown.targetNode = nodeId;
    this.state.drillDown.level++;
    this.state.drillDown.active = true;
    
    this.emitInteractionEvent('drillDownRequest', {
      nodeId,
      level: this.state.drillDown.level
    });
  }

  public drillDownGoBack(): void {
    if (this.state.drillDown.history.length === 0) {
      this.exitDrillDownMode();
      return;
    }
    
    this.state.drillDown.targetNode = this.state.drillDown.history.pop() || null;
    this.state.drillDown.level--;
    
    if (this.state.drillDown.level <= 0) {
      this.exitDrillDownMode();
    } else {
      this.emitInteractionEvent('drillDownBack', {
        nodeId: this.state.drillDown.targetNode,
        level: this.state.drillDown.level
      });
    }
  }

  public exitDrillDownMode(): void {
    this.state.drillDown.active = false;
    this.state.drillDown.targetNode = null;
    this.state.drillDown.level = 0;
    this.state.drillDown.history = [];
    
    if (this.state.mode === 'drilling') {
      this.setInteractionMode('navigation');
    }
    
    this.emitInteractionEvent('drillDownExit');
  }

  // Utility methods
  public focusOnNode(nodeId: string): void {
    this.emitInteractionEvent('focusOnNode', { nodeId });
  }

  public highlightEdgePath(edgeId: string): void {
    this.emitInteractionEvent('highlightEdgePath', { edgeId });
  }

  public showNodeDetails(nodeId: string): void {
    this.emitInteractionEvent('showNodeDetails', { nodeId });
  }

  public showEdgeDetails(edgeId: string): void {
    this.emitInteractionEvent('showEdgeDetails', { edgeId });
  }

  public fitToView(): void {
    this.emitInteractionEvent('fitToView');
  }

  public resetZoom(): void {
    this.emitInteractionEvent('resetZoom');
  }

  // Mode management
  public setInteractionMode(mode: InteractionState['mode']): void {
    const previousMode = this.state.mode;
    this.state.mode = mode;
    
    // Clean up previous mode
    switch (previousMode) {
      case 'path_tracing':
        if (mode !== 'path_tracing') {
          this.resetPathTrace();
        }
        break;
      case 'drilling':
        if (mode !== 'drilling') {
          // Keep drill down state but change interaction behavior
        }
        break;
    }
    
    this.emitInteractionEvent('interactionModeChanged', {
      previousMode,
      currentMode: mode
    });
  }

  public getInteractionState(): InteractionState {
    return { ...this.state };
  }

  public updateConfig(config: Partial<InteractionConfig>): void {
    this.config = { ...this.config, ...config };
    this.emitInteractionEvent('configUpdated', this.config);
  }

  private emitInteractionEvent(eventType: string, detail?: any): void {
    this.container.dispatchEvent(new CustomEvent(eventType, {
      detail,
      bubbles: true
    }));
  }

  public dispose(): void {
    // Remove all event listeners
    this.eventListeners.forEach((listeners, eventType) => {
      listeners.forEach(listener => {
        this.container.removeEventListener(eventType, listener);
      });
    });
    this.eventListeners.clear();
    
    // Remove global event listeners
    document.removeEventListener('keydown', this.handleKeyDown);
    document.removeEventListener('keyup', this.handleKeyUp);
    window.removeEventListener('resize', this.handleResize);
    
    // Clean up context menu
    if (this.contextMenu) {
      document.body.removeChild(this.contextMenu);
    }
    
    // Clear timeouts
    if (this.pathHighlightTimeout) {
      clearTimeout(this.pathHighlightTimeout);
    }
  }
}