// Phase 5 System Architecture Types
import { Observable } from 'rxjs';

export type ComponentStatus = 'healthy' | 'warning' | 'critical' | 'offline' | 'processing';
export type NodeType = 'subcortical' | 'cortical' | 'thalamic' | 'mcp' | 'storage' | 'network' | 'mcp-tool';
export type LayoutType = 'hierarchical' | 'force-directed' | 'circular' | 'grid' | 'neural-layers';
export type ViewMode = 'overview' | 'detailed' | 'cognitive-layers' | 'connections-focus';
export type InteractionType = 'hover' | 'hover-end' | 'click' | 'double-click' | 'drag' | 'context-menu' | 'select' | 'focus';
export type SelectionMode = 'replace' | 'add' | 'toggle';
export type HealthTrend = 'improving' | 'stable' | 'degrading';
export type ExportFormat = 'svg' | 'png' | 'json';

export interface Position {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface ComponentMetrics {
  cpu: {
    current: number;
    average: number;
    peak: number;
  };
  memory: {
    current: number;
    average: number;
    peak: number;
  };
  throughput: {
    current: number;
    average: number;
    peak: number;
  };
  latency: {
    current: number;
    average: number;
    peak: number;
  };
  errorRate: {
    current: number;
    average: number;
    peak: number;
  };
  lastUpdated: number;
}

export interface ConnectionEdge {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'excitation' | 'inhibition' | 'bidirectional' | 'data-flow' | 'dependency';
  strength: number;
  active: boolean;
  dataFlow: number;
  label?: string;
  metadata?: Record<string, any>;
}

export interface ConnectionPoint {
  id: string;
  type: 'input' | 'output' | 'bidirectional';
  angle: number;
  active: boolean;
}

export interface ArchitectureNode {
  id: string;
  type: NodeType;
  label: string;
  description?: string;
  position: Position;
  size: number;
  layer: string;
  status: ComponentStatus;
  importance: number;
  metrics?: ComponentMetrics;
  connections: ConnectionPoint[];
  metadata?: Record<string, any>;
}

export interface ArchitectureData {
  nodes: ArchitectureNode[];
  connections: ConnectionEdge[];
  layers: LayerDefinition[];
  metadata: {
    lastUpdated: number;
    version: string;
    totalComponents: number;
  };
}

export interface LayerDefinition {
  id: string;
  name: string;
  description: string;
  position: Position;
  size: Size;
  color: string;
  phase: number;
  order: number;
  nodes: string[];
}

export interface ThemeConfiguration {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    highlight: string;
    activity: string;
    cognitive: {
      subcortical: string;
      cortical: string;
      thalamic: string;
    };
    mcp: {
      primary: string;
      secondary: string;
    };
    storage: {
      primary: string;
      secondary: string;
    };
    network: {
      primary: string;
      secondary: string;
    };
    default: string;
  };
  connections: {
    [key: string]: {
      stroke: string;
      strokeWidth: number;
      opacity: number;
    };
  };
  fonts: {
    primary: string;
    monospace: string;
  };
}

export interface NodeState {
  position: Position;
  color: string;
  opacity: number;
  scale: number;
  status: {
    color: string;
    size: number;
  };
}

export interface ComponentUpdate {
  componentId: string;
  changes: {
    status?: ComponentStatus;
    metrics?: Partial<ComponentMetrics>;
    position?: Position;
    connections?: ConnectionPoint[];
  };
}

export interface ArchitectureUpdate {
  type: 'component-update' | 'connection-update' | 'layout-change';
  timestamp: number;
  changes: ComponentUpdate[];
}

export interface SystemHealth {
  overall: ComponentStatus;
  score: number;
  totalComponents: number;
  healthyComponents: number;
  warningComponents: number;
  criticalComponents: number;
  offlineComponents: number;
  recommendations: string[];
  lastUpdated: number;
}

export interface LayoutResult {
  nodes: { id: string; position: Position }[];
  connections: { id: string; path: string; midpoint: Position; angle: number }[];
  metadata: {
    algorithm: string;
    quality: number;
    executionTime: number;
  };
}

export interface DiagramState {
  layout: LayoutType;
  viewMode: ViewMode;
  selectedNodes: Set<string>;
  focusedNode?: string;
  highlightedConnections: Set<string>;
  zoom: {
    scale: number;
    translate: Position;
  };
  filters: {
    showLayers: string[];
    showNodeTypes: NodeType[];
    showConnections: boolean;
    showMetrics: boolean;
  };
}

// Animation types
export interface Animation {
  id: string;
  type: 'node-update' | 'connection-flow' | 'layout-transition';
  target: string;
  duration: number;
  ease: string;
  onComplete?: () => void;
}

export interface NodeAnimation extends Animation {
  type: 'node-update';
  properties: {
    x?: { from: number; to: number };
    y?: { from: number; to: number };
    fill?: { from: string; to: string };
    opacity?: { from: number; to: number };
    scale?: { from: number; to: number };
    statusColor?: { from: string; to: string };
    statusSize?: { from: number; to: number };
  };
}

export interface ConnectionFlow {
  minWidth: number;
  maxWidth: number;
  particleCount: number;
  speed: number;
}

export interface FlowAnimation extends Animation {
  type: 'connection-flow';
  repeat: number;
  properties: {
    particlePosition?: { from: number; to: number };
    particleOpacity?: { from: number; to: number; to2?: number };
    strokeWidth?: { from: number; to: number };
    strokeOpacity?: { from: number; to: number };
  };
}

export interface LayoutAnimation extends Animation {
  type: 'layout-transition';
  stages: {
    duration: number;
    targets: string;
    properties: Record<string, any>;
  }[];
}

// Integration interfaces
export interface MCPClient {
  getTools(): Promise<MCPTool[]>;
  getToolStatus(toolId: string): Promise<ComponentStatus>;
  launchTool(toolId: string): Promise<void>;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  type: string;
  version: string;
  category: string;
  status: ComponentStatus;
}

export interface TelemetryData {
  timestamp: number;
  source: string;
  metrics: Record<string, number>;
  events: Array<{
    type: string;
    data: any;
  }>;
}

export interface WebSocketConnection {
  subscribe(channel: string, handler: (data: any) => void): void;
  unsubscribe(channel: string): void;
  send(channel: string, data: any): void;
  isConnected(): boolean;
}

// React component prop interfaces
export interface SystemArchitectureDiagramProps {
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
  
  // Performance
  maxNodes?: number;
  enableAnimations?: boolean;
  enableWebGL?: boolean;
  optimizeForMobile?: boolean;
  
  // Style
  className?: string;
  style?: React.CSSProperties;
}

export interface ArchitectureNodeProps {
  node: ArchitectureNode;
  isSelected: boolean;
  isFocused: boolean;
  isHighlighted: boolean;
  theme: ThemeConfiguration;
  showMetrics: boolean;
  scale: number;
  onInteraction: (interaction: InteractionType, event: Event) => void;
}

export interface ConnectionEdgeProps {
  connection: ConnectionEdge;
  sourceNode: ArchitectureNode;
  targetNode: ArchitectureNode;
  isHighlighted: boolean;
  theme: ThemeConfiguration;
  showFlow: boolean;
  scale: number;
  onInteraction: (interaction: InteractionType, event: Event) => void;
}

export interface ComponentDetailsProps {
  node?: ArchitectureNode;
  connections?: ConnectionEdge[];
  theme: ThemeConfiguration;
  onClose: () => void;
  onNavigateToComponent?: (componentId: string) => void;
}

export interface LayerVisualizationProps {
  layers: LayerDefinition[];
  nodes: ArchitectureNode[];
  theme: ThemeConfiguration;
  showLabels?: boolean;
  interactive?: boolean;
  onLayerClick?: (layer: LayerDefinition) => void;
}

export interface NavigationControlsProps {
  layout: LayoutType;
  viewMode: ViewMode;
  selectedNodes: string[];
  canUndo: boolean;
  canRedo: boolean;
  canExport: boolean;
  
  onLayoutChange: (layout: LayoutType) => void;
  onViewModeChange: (mode: ViewMode) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomToFit: () => void;
  onResetView: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onExport: (format: ExportFormat) => void;
  onToggleHelp: () => void;
  
  className?: string;
}

// Cognitive pattern types
export interface CognitivePattern {
  id: string;
  name: string;
  description: string;
  type: 'convergent' | 'divergent' | 'critical' | 'systems' | 'lateral' | 'adaptive' | 'abstract';
  nodes: string[];
  connections: string[];
  activation: number;
  strength: number;
  phase: number;
}

// Performance monitoring
export interface PerformanceMetrics {
  renderTime: number;
  animationFPS: number;
  memoryUsage: number;
  nodeCount: number;
  connectionCount: number;
  lastMeasurement: number;
}

export interface Bottleneck {
  id: string;
  component: string;
  type: 'cpu' | 'memory' | 'network' | 'rendering';
  severity: 'low' | 'medium' | 'high';
  description: string;
  recommendation: string;
  timestamp: number;
}