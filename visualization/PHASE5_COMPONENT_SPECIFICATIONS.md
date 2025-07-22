# Phase 5: Component Specifications and Relationships

## Overview

This document provides detailed specifications for each component in the Phase 5 System Architecture Diagram visualization, including their interfaces, relationships, and implementation details.

## 1. Core Engine Components

### 1.1 ArchitectureDiagramEngine

**Purpose**: Main orchestrator for architecture diagram rendering and management

```typescript
class ArchitectureDiagramEngine {
  private renderingContext: RenderingContext;
  private layoutEngine: LayoutEngine;
  private interactionEngine: InteractionEngine;
  private animationEngine: AnimationEngine;
  private stateManager: StateManager;
  
  // Configuration
  public initialize(config: ArchitectureEngineConfig): Promise<void> {
    // Initialize D3.js rendering context
    // Set up WebGL context if enabled
    // Configure layout algorithms
    // Initialize event handlers
    // Start animation loop
  }
  
  // Rendering pipeline
  public render(data: ArchitectureData): void {
    const layout = this.layoutEngine.calculateLayout(data);
    const elements = this.createVisualElements(data, layout);
    this.renderElements(elements);
    this.animationEngine.applyInitialAnimations();
  }
  
  // Real-time updates
  public updateComponent(componentId: string, update: ComponentUpdate): void {
    const currentState = this.stateManager.getComponentState(componentId);
    const newState = this.mergeUpdates(currentState, update);
    this.stateManager.updateComponentState(componentId, newState);
    this.animationEngine.animateTransition(componentId, currentState, newState);
  }
  
  // Layout management
  public applyLayout(layoutType: LayoutType): void {
    const currentNodes = this.stateManager.getAllNodes();
    const newLayout = this.layoutEngine.calculateLayout(currentNodes, layoutType);
    this.animateLayoutTransition(newLayout);
  }
  
  // Export functionality
  public async exportDiagram(format: ExportFormat): Promise<Blob> {
    switch (format) {
      case 'svg':
        return this.exportSVG();
      case 'png':
        return this.exportPNG();
      case 'json':
        return this.exportConfiguration();
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }
}

interface ArchitectureEngineConfig {
  container: HTMLElement;
  width: number;
  height: number;
  theme: ThemeConfiguration;
  enableWebGL: boolean;
  maxNodes: number;
  updateThrottleMs: number;
  enableAnimations: boolean;
  layoutType: LayoutType;
  interactionTypes: InteractionType[];
}
```

**Dependencies**: LayoutEngine, InteractionEngine, AnimationEngine, StateManager
**Interfaces**: RenderingContext, ArchitectureData, ComponentUpdate
**Performance Requirements**: Handle 100+ nodes, update latency < 100ms

### 1.2 LayoutEngine

**Purpose**: Calculate optimal positioning for architecture components

```typescript
class LayoutEngine {
  private algorithms: Map<LayoutType, LayoutAlgorithm>;
  private constraints: LayoutConstraints;
  
  constructor() {
    this.algorithms = new Map([
      ['hierarchical', new HierarchicalLayout()],
      ['force-directed', new ForceDirectedLayout()],
      ['circular', new CircularLayout()],
      ['grid', new GridLayout()],
      ['neural-layers', new NeuralLayerLayout()], // LLMKG-specific
    ]);
  }
  
  public calculateLayout(
    nodes: ArchitectureNode[], 
    layoutType: LayoutType = 'neural-layers'
  ): LayoutResult {
    const algorithm = this.algorithms.get(layoutType);
    if (!algorithm) {
      throw new Error(`Unknown layout type: ${layoutType}`);
    }
    
    // Apply LLMKG-specific constraints
    const constrainedNodes = this.applyLLMKGConstraints(nodes);
    
    // Calculate positions
    const positions = algorithm.calculate(constrainedNodes, this.constraints);
    
    // Validate layout quality
    const quality = this.validateLayout(positions);
    if (quality.score < 0.7) {
      console.warn('Layout quality below threshold:', quality);
    }
    
    return {
      nodes: positions,
      connections: this.calculateConnectionPaths(positions),
      metadata: {
        algorithm: layoutType,
        quality: quality.score,
        executionTime: performance.now()
      }
    };
  }
  
  private applyLLMKGConstraints(nodes: ArchitectureNode[]): ConstrainedNode[] {
    return nodes.map(node => ({
      ...node,
      constraints: {
        // Cognitive layers must maintain hierarchical order
        layer: this.getCognitiveLayerConstraints(node.layer),
        // MCP components grouped together
        grouping: this.getMCPGroupingConstraints(node.type),
        // Minimum distances for readability
        minDistance: this.getMinimumDistanceConstraints(node),
        // Connection angle preferences
        connectionAngles: this.getConnectionAngleConstraints(node)
      }
    }));
  }
}

interface LayoutAlgorithm {
  calculate(nodes: ConstrainedNode[], constraints: LayoutConstraints): PositionedNode[];
  getName(): string;
  getComplexity(): 'O(n)' | 'O(n²)' | 'O(n log n)';
}

class NeuralLayerLayout implements LayoutAlgorithm {
  public calculate(nodes: ConstrainedNode[], constraints: LayoutConstraints): PositionedNode[] {
    // Group nodes by cognitive layer
    const layers = this.groupByLayer(nodes);
    
    // Position layers hierarchically
    const layerPositions = this.calculateLayerPositions(layers, constraints);
    
    // Position nodes within each layer
    const nodePositions = this.positionNodesInLayers(layers, layerPositions);
    
    // Optimize for minimal connection crossings
    return this.optimizeForConnections(nodePositions);
  }
  
  getName(): string { return 'Neural Layer Layout'; }
  getComplexity(): 'O(n log n)' { return 'O(n log n)'; }
}
```

**Key Features**:
- Multiple layout algorithms optimized for different visualization needs
- LLMKG-specific neural layer layout respecting cognitive hierarchy
- Real-time layout optimization with performance constraints
- Connection path optimization to minimize visual crossings

### 1.3 InteractionEngine

**Purpose**: Handle all user interactions with the architecture diagram

```typescript
class InteractionEngine {
  private eventHandlers: Map<InteractionType, EventHandler>;
  private selectionState: SelectionState;
  private focusState: FocusState;
  private interactionHistory: InteractionHistory;
  
  constructor(svgElement: SVGElement, config: InteractionConfig) {
    this.setupEventListeners(svgElement);
    this.initializeSelectionSystem();
    this.initializeKeyboardNavigation();
  }
  
  // Node interactions
  public onNodeInteraction(node: ArchitectureNode, interaction: InteractionType, event: Event): void {
    switch (interaction) {
      case 'hover':
        this.handleNodeHover(node, event as MouseEvent);
        break;
      case 'click':
        this.handleNodeClick(node, event as MouseEvent);
        break;
      case 'double-click':
        this.handleNodeDoubleClick(node, event as MouseEvent);
        break;
      case 'drag':
        this.handleNodeDrag(node, event as DragEvent);
        break;
      case 'context-menu':
        this.handleNodeContextMenu(node, event as MouseEvent);
        break;
    }
    
    // Record interaction for analytics
    this.recordInteraction(interaction, node.id, event);
  }
  
  // Selection management
  public selectNodes(nodeIds: string[], mode: SelectionMode = 'replace'): void {
    switch (mode) {
      case 'replace':
        this.selectionState.selectedNodes = new Set(nodeIds);
        break;
      case 'add':
        nodeIds.forEach(id => this.selectionState.selectedNodes.add(id));
        break;
      case 'toggle':
        nodeIds.forEach(id => {
          if (this.selectionState.selectedNodes.has(id)) {
            this.selectionState.selectedNodes.delete(id);
          } else {
            this.selectionState.selectedNodes.add(id);
          }
        });
        break;
    }
    
    this.updateSelectionVisuals();
    this.notifySelectionChange();
  }
  
  // Focus management for accessibility
  public focusNode(nodeId: string): void {
    this.focusState.focusedNode = nodeId;
    this.updateFocusVisuals();
    this.announceForScreenReader(nodeId);
  }
  
  // Path tracing interaction
  public startPathTrace(startNodeId: string): void {
    this.pathTraceState = {
      active: true,
      startNode: startNodeId,
      currentPath: [startNodeId],
      highlightedConnections: new Set()
    };
    
    this.updatePathTraceVisuals();
  }
  
  // Keyboard shortcuts
  private setupKeyboardShortcuts(): void {
    const shortcuts: KeyboardShortcut[] = [
      { key: 'Escape', action: () => this.clearSelection() },
      { key: 'Delete', action: () => this.hideSelectedNodes() },
      { key: 'f', modifiers: ['ctrl'], action: () => this.focusToFit() },
      { key: 'e', modifiers: ['ctrl'], action: () => this.exportDiagram() },
      { key: 'ArrowUp', action: () => this.navigateFocus('up') },
      { key: 'ArrowDown', action: () => this.navigateFocus('down') },
      { key: 'Enter', action: () => this.activateFocusedNode() }
    ];
    
    this.registerKeyboardShortcuts(shortcuts);
  }
}

interface InteractionConfig {
  enableMultiSelect: boolean;
  enableDragAndDrop: boolean;
  enablePathTracing: boolean;
  enableKeyboardNavigation: boolean;
  selectionColor: string;
  focusColor: string;
  hoverEffects: boolean;
  touchOptimizations: boolean;
}
```

**Interaction Types Supported**:
- Single and multi-node selection
- Drag and drop for layout customization
- Path tracing for connection analysis
- Keyboard navigation for accessibility
- Touch gestures for mobile devices

### 1.4 AnimationEngine

**Purpose**: Smooth animations and transitions for architecture diagram changes

```typescript
class AnimationEngine {
  private animationQueue: Animation[];
  private activeAnimations: Map<string, Animation>;
  private timeline: gsap.Timeline;
  
  constructor(config: AnimationConfig) {
    this.timeline = gsap.timeline();
    this.setupAnimationDefaults(config);
  }
  
  // Node animations
  public animateNodeUpdate(nodeId: string, from: NodeState, to: NodeState): void {
    const animation: NodeAnimation = {
      id: `node-${nodeId}-${Date.now()}`,
      type: 'node-update',
      target: nodeId,
      duration: 0.5,
      ease: 'power2.out',
      properties: {
        // Position changes
        x: { from: from.position.x, to: to.position.x },
        y: { from: from.position.y, to: to.position.y },
        // Visual state changes
        fill: { from: from.color, to: to.color },
        opacity: { from: from.opacity, to: to.opacity },
        scale: { from: from.scale, to: to.scale },
        // Status indicator changes
        statusColor: { from: from.status.color, to: to.status.color },
        statusSize: { from: from.status.size, to: to.status.size }
      }
    };
    
    this.queueAnimation(animation);
  }
  
  // Connection animations
  public animateConnectionFlow(connectionId: string, flowData: ConnectionFlow): void {
    const connection = this.getConnection(connectionId);
    if (!connection) return;
    
    const flowAnimation: FlowAnimation = {
      id: `flow-${connectionId}-${Date.now()}`,
      type: 'connection-flow',
      target: connectionId,
      duration: 2.0,
      repeat: -1, // Infinite repeat
      ease: 'none',
      properties: {
        // Animated particles along connection path
        particlePosition: { from: 0, to: 1 },
        particleOpacity: { from: 0, to: 1, to2: 0 }, // Fade in and out
        // Connection strength visualization
        strokeWidth: { from: flowData.minWidth, to: flowData.maxWidth },
        strokeOpacity: { from: 0.3, to: 0.8 }
      }
    };
    
    this.queueAnimation(flowAnimation);
  }
  
  // Layout transition animations
  public animateLayoutTransition(
    oldLayout: LayoutResult, 
    newLayout: LayoutResult
  ): Promise<void> {
    return new Promise((resolve) => {
      const transitionAnimation: LayoutAnimation = {
        id: `layout-transition-${Date.now()}`,
        type: 'layout-transition',
        duration: 1.0,
        ease: 'power2.inOut',
        onComplete: resolve,
        stages: [
          {
            // Stage 1: Fade out connections
            duration: 0.2,
            targets: '.connection',
            properties: { opacity: { to: 0 } }
          },
          {
            // Stage 2: Move nodes to new positions
            duration: 0.6,
            targets: '.node',
            properties: this.calculateNodeTransitions(oldLayout, newLayout)
          },
          {
            // Stage 3: Fade in connections with new paths
            duration: 0.2,
            targets: '.connection',
            properties: { 
              opacity: { to: 1 },
              d: this.calculateNewConnectionPaths(newLayout)
            }
          }
        ]
      };
      
      this.queueAnimation(transitionAnimation);
    });
  }
  
  // Performance-aware animation system
  private optimizeAnimationsForPerformance(): void {
    // Reduce animation complexity based on system performance
    const fps = this.getCurrentFPS();
    
    if (fps < 30) {
      // Disable complex animations
      this.disableParticleAnimations();
      this.reduceAnimationDuration(0.5);
    } else if (fps < 45) {
      // Reduce particle count
      this.setMaxParticleCount(50);
    }
    
    // Use requestAnimationFrame for smooth animations
    this.useRAFOptimization(true);
  }
  
  // Cognitive pattern-specific animations
  public animateCognitivePatternActivation(pattern: CognitivePattern): void {
    const affectedNodes = this.getNodesInPattern(pattern);
    
    // Create wave-like activation animation across the pattern
    const waveAnimation = affectedNodes.map((nodeId, index) => ({
      target: nodeId,
      delay: index * 0.1, // Stagger the activation
      duration: 0.3,
      properties: {
        scale: { from: 1, to: 1.2, back: 1 },
        glow: { from: 0, to: 1, back: 0.3 },
        strokeWidth: { from: 2, to: 4, back: 2 }
      }
    }));
    
    this.queueAnimationSequence(waveAnimation);
  }
}

interface AnimationConfig {
  globalDuration: number;
  enableComplexAnimations: boolean;
  maxConcurrentAnimations: number;
  performanceMode: 'smooth' | 'performance' | 'auto';
  particleAnimations: boolean;
  morphingTransitions: boolean;
}
```

**Animation Types**:
- Node state transitions (status, metrics, position)
- Connection flow animations with particles
- Layout transition animations
- Cognitive pattern activation waves
- Performance-adaptive animation complexity

## 2. React Component Specifications

### 2.1 SystemArchitectureDiagram (Main Component)

```typescript
interface SystemArchitectureDiagramProps {
  // Data
  architectureData: ArchitectureData;
  realTimeEnabled?: boolean;
  refreshInterval?: number;
  
  // Display configuration
  layout?: LayoutType;
  theme?: ThemeConfiguration;
  viewMode?: ViewMode;
  showMetrics?: boolean;
  showConnections?: boolean;
  
  // Interaction callbacks
  onNodeClick?: (node: ArchitectureNode) => void;
  onNodeDoubleClick?: (node: ArchitectureNode) => void;
  onConnectionClick?: (connection: ConnectionEdge) => void;
  onSelectionChange?: (selectedNodes: string[]) => void;
  onLayoutChange?: (layout: LayoutType) => void;
  
  // Integration props
  mcpClient?: MCPClient;
  telemetryStream?: Observable<TelemetryData>;
  websocketConnection?: WebSocketConnection;
  
  // Customization
  customNodeRenderer?: NodeRenderer;
  customConnectionRenderer?: ConnectionRenderer;
  customTooltipRenderer?: TooltipRenderer;
  
  // Performance
  maxNodes?: number;
  enableAnimations?: boolean;
  enableWebGL?: boolean;
  optimizeForMobile?: boolean;
}

const SystemArchitectureDiagram: React.FC<SystemArchitectureDiagramProps> = ({
  architectureData,
  realTimeEnabled = true,
  layout = 'neural-layers',
  theme = 'default',
  onNodeClick,
  onConnectionClick,
  ...props
}) => {
  // State management
  const [diagramState, setDiagramState] = useState<DiagramState>();
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<ArchitectureDiagramEngine | null>(null);
  
  // Custom hooks
  const { architectureMetrics } = useArchitectureMetrics();
  const { systemHealth } = useSystemHealth();
  const { realTimeUpdates } = useRealTimeUpdates(realTimeEnabled);
  
  // Initialize diagram engine
  useEffect(() => {
    if (!containerRef.current) return;
    
    const initializeEngine = async () => {
      try {
        setIsLoading(true);
        
        const engine = new ArchitectureDiagramEngine();
        await engine.initialize({
          container: containerRef.current!,
          theme,
          layout,
          enableAnimations: props.enableAnimations ?? true,
          maxNodes: props.maxNodes ?? 100
        });
        
        // Set up event handlers
        engine.onNodeClick((node) => {
          setSelectedNodes(prev => new Set([...prev, node.id]));
          onNodeClick?.(node);
        });
        
        engine.onConnectionClick((connection) => {
          onConnectionClick?.(connection);
        });
        
        engineRef.current = engine;
        setIsLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsLoading(false);
      }
    };
    
    initializeEngine();
    
    return () => {
      engineRef.current?.dispose();
    };
  }, [layout, theme, props.enableAnimations]);
  
  // Handle real-time updates
  useEffect(() => {
    if (!realTimeUpdates || !engineRef.current) return;
    
    const subscription = realTimeUpdates.subscribe(update => {
      engineRef.current?.updateComponent(update.componentId, update.changes);
    });
    
    return () => subscription.unsubscribe();
  }, [realTimeUpdates]);
  
  // Render loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="large" message="Loading architecture diagram..." />
      </div>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Failed to Load Architecture
          </h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="relative w-full h-full bg-gray-50">
      {/* Main diagram container */}
      <div 
        ref={containerRef} 
        className="w-full h-full overflow-hidden"
        role="img"
        aria-label="System Architecture Diagram"
      />
      
      {/* Control panel overlay */}
      <ArchitectureControls
        selectedLayout={layout}
        onLayoutChange={props.onLayoutChange}
        selectedNodes={Array.from(selectedNodes)}
        onExport={() => engineRef.current?.exportDiagram('svg')}
        className="absolute top-4 right-4"
      />
      
      {/* Legend panel */}
      <LegendPanel
        theme={theme}
        className="absolute bottom-4 left-4"
      />
      
      {/* Status indicator */}
      <div className="absolute top-4 left-4">
        <StatusIndicator
          status={systemHealth.overall}
          label={`${architectureData.nodes.length} Components`}
        />
      </div>
      
      {/* Selection info panel */}
      {selectedNodes.size > 0 && (
        <SelectionPanel
          selectedNodes={Array.from(selectedNodes)}
          onClearSelection={() => setSelectedNodes(new Set())}
          className="absolute bottom-4 right-4"
        />
      )}
    </div>
  );
};
```

### 2.2 ArchitectureNode Component

```typescript
interface ArchitectureNodeProps {
  node: ArchitectureNode;
  isSelected: boolean;
  isFocused: boolean;
  theme: ThemeConfiguration;
  showMetrics: boolean;
  onInteraction: (interaction: InteractionType, event: Event) => void;
}

const ArchitectureNode: React.FC<ArchitectureNodeProps> = ({
  node,
  isSelected,
  isFocused,
  theme,
  showMetrics,
  onInteraction
}) => {
  // Visual state based on node properties
  const nodeColor = useMemo(() => {
    return getNodeColor(node.type, node.status, theme);
  }, [node.type, node.status, theme]);
  
  const nodeSize = useMemo(() => {
    return getNodeSize(node.importance, node.metrics?.throughput || 0);
  }, [node.importance, node.metrics]);
  
  // Animation state
  const animatedProps = useSpring({
    scale: isSelected ? 1.1 : 1.0,
    opacity: node.status === 'offline' ? 0.5 : 1.0,
    strokeWidth: isFocused ? 4 : 2,
    config: { tension: 300, friction: 30 }
  });
  
  return (
    <g
      className="architecture-node"
      transform={`translate(${node.position.x}, ${node.position.y})`}
      tabIndex={0}
      role="button"
      aria-label={`${node.label} - ${node.status}`}
      onMouseEnter={(e) => onInteraction('hover', e)}
      onMouseLeave={(e) => onInteraction('hover-end', e)}
      onClick={(e) => onInteraction('click', e)}
      onDoubleClick={(e) => onInteraction('double-click', e)}
      onContextMenu={(e) => onInteraction('context-menu', e)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onInteraction('click', e);
        }
      }}
    >
      {/* Main node circle */}
      <animated.circle
        r={nodeSize}
        fill={nodeColor.primary}
        stroke={nodeColor.border}
        strokeWidth={animatedProps.strokeWidth}
        opacity={animatedProps.opacity}
        transform={animatedProps.scale.to(s => `scale(${s})`)}
        className={`
          transition-colors duration-200 cursor-pointer
          ${isSelected ? 'filter drop-shadow-lg' : ''}
          ${isFocused ? 'outline-none ring-2 ring-blue-500' : ''}
        `}
      />
      
      {/* Status indicator */}
      <StatusIndicator
        status={node.status}
        position={{ x: nodeSize * 0.7, y: -nodeSize * 0.7 }}
        size="small"
      />
      
      {/* Node label */}
      <text
        x={0}
        y={5}
        textAnchor="middle"
        className="text-sm font-medium fill-white pointer-events-none"
        fontSize={Math.max(10, nodeSize / 4)}
      >
        {node.label}
      </text>
      
      {/* Metrics overlay */}
      {showMetrics && node.metrics && (
        <MetricsOverlay
          metrics={node.metrics}
          position={{ x: 0, y: nodeSize + 20 }}
          compact={nodeSize < 30}
        />
      )}
      
      {/* Connection points */}
      {node.connections.map(connection => (
        <ConnectionPoint
          key={connection.id}
          connection={connection}
          nodeRadius={nodeSize}
          theme={theme}
        />
      ))}
      
      {/* Activity indicator for active processing */}
      {node.status === 'processing' && (
        <ActivityIndicator
          center={{ x: 0, y: 0 }}
          radius={nodeSize + 10}
          color={theme.colors.activity}
        />
      )}
    </g>
  );
};

// Helper function for node coloring based on type and status
function getNodeColor(type: NodeType, status: ComponentStatus, theme: ThemeConfiguration) {
  const baseColors = {
    'subcortical': theme.colors.cognitive.subcortical,
    'cortical': theme.colors.cognitive.cortical,
    'thalamic': theme.colors.cognitive.thalamic,
    'mcp': theme.colors.mcp.primary,
    'storage': theme.colors.storage.primary,
    'network': theme.colors.network.primary
  };
  
  const statusModifiers = {
    'healthy': { opacity: 1.0, saturation: 1.0 },
    'warning': { opacity: 0.8, saturation: 1.2 },
    'critical': { opacity: 0.9, saturation: 0.8, hue: 'red' },
    'offline': { opacity: 0.3, saturation: 0.2 }
  };
  
  const baseColor = baseColors[type] || theme.colors.default;
  const modifier = statusModifiers[status] || statusModifiers.healthy;
  
  return applyColorModifier(baseColor, modifier);
}
```

### 2.3 ConnectionEdge Component

```typescript
interface ConnectionEdgeProps {
  connection: ConnectionEdge;
  sourceNode: ArchitectureNode;
  targetNode: ArchitectureNode;
  isHighlighted: boolean;
  theme: ThemeConfiguration;
  showFlow: boolean;
  onInteraction: (interaction: InteractionType, event: Event) => void;
}

const ConnectionEdge: React.FC<ConnectionEdgeProps> = ({
  connection,
  sourceNode,
  targetNode,
  isHighlighted,
  theme,
  showFlow,
  onInteraction
}) => {
  // Calculate connection path
  const pathData = useMemo(() => {
    return calculateConnectionPath(
      sourceNode.position,
      targetNode.position,
      connection.type,
      sourceNode.size,
      targetNode.size
    );
  }, [sourceNode.position, targetNode.position, connection.type]);
  
  // Connection styling based on type and status
  const connectionStyle = useMemo(() => {
    const baseStyle = theme.connections[connection.type];
    return {
      ...baseStyle,
      strokeWidth: connection.strength * 5,
      opacity: connection.active ? 0.8 : 0.3,
      stroke: isHighlighted ? theme.colors.highlight : baseStyle.stroke
    };
  }, [connection, theme, isHighlighted]);
  
  // Animated flow particles
  const flowParticles = useSpring({
    opacity: showFlow && connection.active ? 1 : 0,
    config: { duration: 300 }
  });
  
  return (
    <g
      className="connection-edge"
      role="button"
      aria-label={`Connection from ${sourceNode.label} to ${targetNode.label}`}
      onMouseEnter={(e) => onInteraction('hover', e)}
      onMouseLeave={(e) => onInteraction('hover-end', e)}
      onClick={(e) => onInteraction('click', e)}
    >
      {/* Main connection path */}
      <path
        d={pathData.path}
        fill="none"
        stroke={connectionStyle.stroke}
        strokeWidth={connectionStyle.strokeWidth}
        strokeOpacity={connectionStyle.opacity}
        strokeDasharray={connection.type === 'inhibition' ? '5,5' : 'none'}
        markerEnd={connection.type !== 'bidirectional' ? 'url(#arrow)' : undefined}
        className="transition-all duration-200 cursor-pointer hover:stroke-opacity-100"
      />
      
      {/* Flow animation particles */}
      {showFlow && connection.active && (
        <animated.g opacity={flowParticles.opacity}>
          {[...Array(3)].map((_, index) => (
            <FlowParticle
              key={index}
              path={pathData.path}
              delay={index * 0.7}
              color={connectionStyle.stroke}
              size={Math.max(2, connectionStyle.strokeWidth / 3)}
            />
          ))}
        </animated.g>
      )}
      
      {/* Connection label */}
      {isHighlighted && connection.label && (
        <ConnectionLabel
          text={connection.label}
          position={pathData.midpoint}
          rotation={pathData.angle}
          theme={theme}
        />
      )}
      
      {/* Data flow indicator */}
      {connection.dataFlow > 0 && (
        <DataFlowIndicator
          flow={connection.dataFlow}
          path={pathData.path}
          direction={pathData.direction}
          theme={theme}
        />
      )}
    </g>
  );
};

// Flow particle animation component
const FlowParticle: React.FC<{
  path: string;
  delay: number;
  color: string;
  size: number;
}> = ({ path, delay, color, size }) => {
  const animatedProps = useSpring({
    from: { offset: 0, opacity: 0 },
    to: async (next) => {
      while (true) {
        await next({ offset: 0, opacity: 0 });
        await next({ offset: 0.1, opacity: 1 });
        await next({ offset: 0.9, opacity: 1 });
        await next({ offset: 1, opacity: 0 });
      }
    },
    config: { duration: 2000 },
    delay: delay * 1000
  });
  
  return (
    <animated.circle
      r={size}
      fill={color}
      opacity={animatedProps.opacity}
    >
      <animateMotion
        dur="2s"
        repeatCount="indefinite"
        begin={`${delay}s`}
      >
        <mpath href={`#${path}`} />
      </animateMotion>
    </animated.circle>
  );
};
```

## 3. Monitoring and Integration Components

### 3.1 RealTimeMonitor

```typescript
class RealTimeMonitor {
  private websocketConnection: WebSocketConnection;
  private metricsCollector: MetricsCollector;
  private alertSystem: AlertSystem;
  private subscribers: Map<string, Subscriber[]>;
  
  constructor(config: RealTimeMonitorConfig) {
    this.websocketConnection = new WebSocketConnection(config.websocketUrl);
    this.metricsCollector = new MetricsCollector(config.metricsConfig);
    this.alertSystem = new AlertSystem(config.alertConfig);
    this.subscribers = new Map();
    
    this.initializeDataStreams();
  }
  
  // Component monitoring
  public monitorComponent(componentId: string): ComponentMonitor {
    const monitor = new ComponentMonitor(componentId, {
      metricsCollector: this.metricsCollector,
      alertSystem: this.alertSystem,
      updateInterval: 1000
    });
    
    // Subscribe to real-time updates for this component
    this.websocketConnection.subscribe(`component.${componentId}`, (data) => {
      monitor.updateMetrics(data);
    });
    
    return monitor;
  }
  
  // System health monitoring
  public getSystemHealth(): Observable<SystemHealth> {
    return new Observable(observer => {
      const healthCheck = async () => {
        try {
          const components = await this.getAllComponents();
          const health = await this.calculateSystemHealth(components);
          observer.next(health);
        } catch (error) {
          observer.error(error);
        }
      };
      
      // Initial health check
      healthCheck();
      
      // Set up periodic health checks
      const interval = setInterval(healthCheck, 5000);
      
      return () => clearInterval(interval);
    });
  }
  
  // Alert management
  public configureAlerts(rules: AlertRule[]): void {
    rules.forEach(rule => {
      this.alertSystem.addRule(rule);
    });
    
    // Set up alert handlers
    this.alertSystem.onAlert((alert) => {
      this.notifySubscribers('alert', alert);
      this.logAlert(alert);
    });
  }
  
  // Performance bottleneck detection
  public detectBottlenecks(): Observable<Bottleneck[]> {
    return new Observable(observer => {
      const analyzer = new PerformanceAnalyzer({
        windowSize: 60000, // 1 minute window
        threshold: 0.8 // 80% utilization threshold
      });
      
      // Subscribe to metrics stream
      this.metricsCollector.getMetricsStream().subscribe(metrics => {
        const bottlenecks = analyzer.analyze(metrics);
        if (bottlenecks.length > 0) {
          observer.next(bottlenecks);
        }
      });
    });
  }
  
  private async calculateSystemHealth(components: ComponentState[]): Promise<SystemHealth> {
    const healthScores = components.map(c => c.healthScore);
    const avgHealth = healthScores.reduce((a, b) => a + b, 0) / healthScores.length;
    
    const criticalComponents = components.filter(c => c.status === 'critical');
    const warningComponents = components.filter(c => c.status === 'warning');
    
    return {
      overall: this.categorizeHealth(avgHealth),
      score: avgHealth,
      totalComponents: components.length,
      healthyComponents: components.filter(c => c.status === 'healthy').length,
      warningComponents: warningComponents.length,
      criticalComponents: criticalComponents.length,
      offlineComponents: components.filter(c => c.status === 'offline').length,
      recommendations: this.generateRecommendations(components),
      lastUpdated: Date.now()
    };
  }
}

class ComponentMonitor {
  private componentId: string;
  private metrics: ComponentMetrics;
  private healthHistory: HealthDataPoint[];
  private alertRules: AlertRule[];
  
  constructor(componentId: string, config: ComponentMonitorConfig) {
    this.componentId = componentId;
    this.metrics = this.initializeMetrics();
    this.healthHistory = [];
    this.alertRules = config.defaultAlertRules || [];
  }
  
  // Performance metrics
  public getCPUUsage(): number {
    return this.metrics.cpu.current;
  }
  
  public getMemoryUsage(): number {
    return this.metrics.memory.current;
  }
  
  public getThroughput(): number {
    return this.metrics.throughput.current;
  }
  
  public getLatency(): number {
    return this.metrics.latency.current;
  }
  
  // Health scoring
  public getHealthScore(): number {
    const weights = {
      cpu: 0.3,
      memory: 0.3,
      latency: 0.2,
      throughput: 0.1,
      errorRate: 0.1
    };
    
    const scores = {
      cpu: this.scoreCPUHealth(this.metrics.cpu.current),
      memory: this.scoreMemoryHealth(this.metrics.memory.current),
      latency: this.scoreLatencyHealth(this.metrics.latency.current),
      throughput: this.scoreThroughputHealth(this.metrics.throughput.current),
      errorRate: this.scoreErrorRateHealth(this.metrics.errorRate.current)
    };
    
    return Object.entries(weights).reduce(
      (total, [metric, weight]) => total + (scores[metric] * weight),
      0
    );
  }
  
  // Trend analysis
  public getHealthTrend(): HealthTrend {
    if (this.healthHistory.length < 10) {
      return 'stable';
    }
    
    const recent = this.healthHistory.slice(-10);
    const older = this.healthHistory.slice(-20, -10);
    
    const recentAvg = recent.reduce((a, b) => a + b.score, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b.score, 0) / older.length;
    
    const change = (recentAvg - olderAvg) / olderAvg;
    
    if (change > 0.05) return 'improving';
    if (change < -0.05) return 'degrading';
    return 'stable';
  }
  
  // Update metrics from real-time data
  public updateMetrics(data: ComponentUpdateData): void {
    this.metrics = {
      ...this.metrics,
      cpu: this.updateMetricValue(this.metrics.cpu, data.cpu),
      memory: this.updateMetricValue(this.metrics.memory, data.memory),
      latency: this.updateMetricValue(this.metrics.latency, data.latency),
      throughput: this.updateMetricValue(this.metrics.throughput, data.throughput),
      errorRate: this.updateMetricValue(this.metrics.errorRate, data.errorRate),
      lastUpdated: Date.now()
    };
    
    // Update health history
    const currentHealth = this.getHealthScore();
    this.healthHistory.push({
      timestamp: Date.now(),
      score: currentHealth
    });
    
    // Keep only recent history (last 24 hours)
    const cutoff = Date.now() - (24 * 60 * 60 * 1000);
    this.healthHistory = this.healthHistory.filter(h => h.timestamp > cutoff);
    
    // Check alert conditions
    this.checkAlertConditions();
  }
}
```

## 4. Integration Components

### 4.1 Phase Integration Bridges

```typescript
// Phase 1 Integration Bridge
class Phase1Integration implements IntegrationBridge {
  private telemetryClient: TelemetryClient;
  private dataTransformer: TelemetryDataTransformer;
  
  constructor(config: Phase1IntegrationConfig) {
    this.telemetryClient = new TelemetryClient(config.telemetryEndpoint);
    this.dataTransformer = new TelemetryDataTransformer();
  }
  
  public async getArchitectureMetrics(): Promise<ArchitectureMetrics> {
    const rawTelemetry = await this.telemetryClient.getLatestData();
    return this.dataTransformer.transformToArchitectureMetrics(rawTelemetry);
  }
  
  public subscribeToUpdates(): Observable<ArchitectureUpdate> {
    return this.telemetryClient.getDataStream()
      .pipe(
        map(data => this.dataTransformer.transformToArchitectureUpdate(data)),
        filter(update => this.isRelevantUpdate(update)),
        throttleTime(100) // Throttle to prevent excessive updates
      );
  }
  
  private isRelevantUpdate(update: ArchitectureUpdate): boolean {
    // Filter out irrelevant updates to reduce noise
    return update.changes.some(change => 
      change.type === 'status' || 
      change.type === 'metrics' || 
      change.type === 'health'
    );
  }
}

// Phase 2 Dashboard Integration Bridge
class Phase2Integration implements IntegrationBridge {
  private uiState: UIStateManager;
  private navigationSystem: NavigationSystem;
  private themeSystem: ThemeSystem;
  
  constructor(config: Phase2IntegrationConfig) {
    this.uiState = config.uiStateManager;
    this.navigationSystem = config.navigationSystem;
    this.themeSystem = config.themeSystem;
  }
  
  public registerArchitecturePage(): void {
    this.navigationSystem.addRoute({
      path: '/architecture',
      component: SystemArchitecturePage,
      title: 'System Architecture',
      icon: 'architecture',
      section: 'System'
    });
  }
  
  public integrateWithThemeSystem(): ThemeConfiguration {
    return this.themeSystem.getCurrentTheme();
  }
  
  public subscribeToThemeChanges(): Observable<ThemeConfiguration> {
    return this.themeSystem.getThemeUpdates();
  }
  
  public updateBreadcrumb(path: string[]): void {
    this.navigationSystem.updateBreadcrumb([
      { label: 'Dashboard', href: '/' },
      { label: 'System', href: '/system' },
      ...path.map(p => ({ label: p, href: `/architecture/${p.toLowerCase()}` }))
    ]);
  }
}

// Phase 3 Tools Integration Bridge
class Phase3Integration implements IntegrationBridge {
  private toolsRegistry: ToolsRegistry;
  private toolsMonitor: ToolsMonitor;
  
  constructor(config: Phase3IntegrationConfig) {
    this.toolsRegistry = config.toolsRegistry;
    this.toolsMonitor = config.toolsMonitor;
  }
  
  public getToolsArchitectureData(): Promise<ToolsArchitectureData> {
    return Promise.all([
      this.toolsRegistry.getAllTools(),
      this.toolsMonitor.getToolsStatus(),
      this.toolsRegistry.getToolsDependencies()
    ]).then(([tools, status, dependencies]) => ({
      tools: tools.map(tool => this.transformToolToNode(tool)),
      connections: dependencies.map(dep => this.transformDependencyToConnection(dep)),
      status: status
    }));
  }
  
  public launchToolFromArchitecture(toolId: string): Promise<void> {
    return this.toolsRegistry.launchTool(toolId);
  }
  
  public getToolDocumentation(toolId: string): Promise<ToolDocumentation> {
    return this.toolsRegistry.getToolDocumentation(toolId);
  }
  
  private transformToolToNode(tool: MCPTool): ArchitectureNode {
    return {
      id: `tool-${tool.id}`,
      type: 'mcp-tool',
      label: tool.name,
      description: tool.description,
      layer: 'integration',
      status: this.toolsMonitor.getToolStatus(tool.id),
      metrics: this.toolsMonitor.getToolMetrics(tool.id),
      metadata: {
        toolType: tool.type,
        version: tool.version,
        category: tool.category
      }
    };
  }
}

// Phase 4 Data Flow Integration Bridge
class Phase4Integration implements IntegrationBridge {
  private dataFlowEngine: DataFlowVisualizationEngine;
  private particleSystem: ParticleSystem;
  
  constructor(config: Phase4IntegrationConfig) {
    this.dataFlowEngine = config.dataFlowEngine;
    this.particleSystem = config.particleSystem;
  }
  
  public shareVisualizationEngine(): DataFlowEngineInterface {
    return {
      renderParticleFlow: this.dataFlowEngine.renderParticleFlow.bind(this.dataFlowEngine),
      animateDataPath: this.dataFlowEngine.animateDataPath.bind(this.dataFlowEngine),
      getShaderLibrary: this.dataFlowEngine.getShaderLibrary.bind(this.dataFlowEngine)
    };
  }
  
  public coordinateAnimations(architectureAnimations: Animation[]): void {
    // Coordinate Phase 5 animations with Phase 4 data flow animations
    this.dataFlowEngine.synchronizeAnimations(architectureAnimations);
  }
  
  public highlightDataFlowPath(startComponent: string, endComponent: string): void {
    const path = this.dataFlowEngine.findPath(startComponent, endComponent);
    this.dataFlowEngine.highlightPath(path);
  }
  
  public navigateToDataFlow(componentId: string): void {
    // Navigate to Phase 4 with specific component focused
    window.location.href = `/data-flow?focus=${componentId}`;
  }
}
```

This comprehensive component specifications document provides the detailed technical foundation for implementing Phase 5. Each component is designed with clear interfaces, responsibilities, and integration points that will enable parallel development by specialized agents while ensuring cohesive functionality across the entire system.

The specifications emphasize:
- **Modularity**: Clear separation of concerns
- **Integration**: Seamless connections with existing phases
- **Performance**: Optimized for real-time updates and large architectures
- **Accessibility**: Screen reader support and keyboard navigation
- **Extensibility**: Framework for future enhancements

These specifications will guide the implementation teams in creating a robust, scalable, and user-friendly system architecture visualization tool.