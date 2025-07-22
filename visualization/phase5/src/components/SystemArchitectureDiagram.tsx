import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { ArchitectureDiagramEngine, ArchitectureData, VisualizationConfig } from '../core/ArchitectureDiagramEngine';
import { LayoutEngine } from '../core/LayoutEngine';
import { InteractionEngine, InteractionConfig, PathTraceResult } from '../core/InteractionEngine';
import { AnimationEngine, AnimationConfig, RealTimeUpdate } from '../core/AnimationEngine';

export interface SystemArchitectureDiagramProps {
  data: ArchitectureData;
  visualization?: Partial<VisualizationConfig>;
  interaction?: Partial<InteractionConfig>;
  animation?: Partial<AnimationConfig>;
  realTimeData?: RealTimeUpdate[];
  onNodeSelected?: (nodeId: string, nodeData: any) => void;
  onEdgeSelected?: (edgeId: string, edgeData: any) => void;
  onPathTraced?: (result: PathTraceResult) => void;
  onDrillDown?: (nodeId: string, level: number) => void;
  onLayoutChanged?: (layoutType: string) => void;
  onPerformanceUpdate?: (metrics: any) => void;
  className?: string;
  style?: React.CSSProperties;
}

export interface DiagramControls {
  zoomIn: () => void;
  zoomOut: () => void;
  zoomToFit: () => void;
  resetView: () => void;
  setLayout: (layout: VisualizationConfig['layout']) => void;
  exportImage: (format?: 'png' | 'jpg') => string;
  toggleAnimations: () => void;
  focusOnNode: (nodeId: string) => void;
}

export const SystemArchitectureDiagram: React.FC<SystemArchitectureDiagramProps> = ({
  data,
  visualization = {},
  interaction = {},
  animation = {},
  realTimeData = [],
  onNodeSelected,
  onEdgeSelected,
  onPathTraced,
  onDrillDown,
  onLayoutChanged,
  onPerformanceUpdate,
  className,
  style
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const diagramEngineRef = useRef<ArchitectureDiagramEngine | null>(null);
  const layoutEngineRef = useRef<LayoutEngine | null>(null);
  const interactionEngineRef = useRef<InteractionEngine | null>(null);
  const animationEngineRef = useRef<AnimationEngine | null>(null);
  
  const [isInitialized, setIsInitialized] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);
  const [currentLayout, setCurrentLayout] = useState<VisualizationConfig['layout']>('brain_inspired');

  // Memoize configuration objects to prevent unnecessary re-renders
  const visualConfig = useMemo(() => ({
    layout: 'brain_inspired' as const,
    dimensions: '2d' as const,
    showLabels: true,
    showMetrics: true,
    enableAnimations: true,
    colorScheme: 'default' as const,
    performanceMode: 'balanced' as const,
    ...visualization
  }), [visualization]);

  const interactionConfig = useMemo(() => ({
    enableMultiSelect: true,
    enablePathTracing: true,
    enableDrillDown: true,
    enableTooltips: true,
    enableContextMenu: true,
    selectionColor: '#e91e63',
    hoverColor: '#ffeb3b',
    pathTraceColor: '#00bcd4',
    animationDuration: 300,
    ...interaction
  }), [interaction]);

  const animationConfig = useMemo(() => ({
    enableAnimations: true,
    globalSpeed: 1.0,
    maxConcurrentAnimations: 10,
    performanceMode: 'balanced' as const,
    easing: 'ease-in-out' as const,
    reduceMotion: false,
    ...animation
  }), [animation]);

  // Initialize engines
  const initializeEngines = useCallback(() => {
    if (!containerRef.current) return;

    try {
      setLoading(true);
      setError(null);

      // Initialize diagram engine
      diagramEngineRef.current = new ArchitectureDiagramEngine(
        containerRef.current,
        visualConfig
      );

      // Initialize layout engine
      const rect = containerRef.current.getBoundingClientRect();
      layoutEngineRef.current = new LayoutEngine(
        visualConfig.dimensions,
        rect.width,
        rect.height,
        600 // depth for 3D
      );

      // Initialize interaction engine
      interactionEngineRef.current = new InteractionEngine(
        containerRef.current,
        diagramEngineRef.current,
        interactionConfig
      );

      // Initialize animation engine
      animationEngineRef.current = new AnimationEngine(
        containerRef.current,
        animationConfig
      );

      setIsInitialized(true);
      setLoading(false);
    } catch (err) {
      setError(`Failed to initialize diagram: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLoading(false);
    }
  }, [visualConfig, interactionConfig, animationConfig]);

  // Setup event handlers
  const setupEventHandlers = useCallback(() => {
    if (!containerRef.current || !interactionEngineRef.current || !animationEngineRef.current) return;

    const container = containerRef.current;
    const interactionEngine = interactionEngineRef.current;
    const animationEngine = animationEngineRef.current;

    // Node selection events
    const handleNodeSelected = (event: CustomEvent) => {
      if (onNodeSelected) {
        onNodeSelected(event.detail.id, event.detail);
      }
    };

    // Edge selection events
    const handleEdgeSelected = (event: CustomEvent) => {
      if (onEdgeSelected) {
        onEdgeSelected(event.detail.id, event.detail);
      }
    };

    // Path tracing events
    const handlePathComputed = (event: CustomEvent) => {
      if (onPathTraced) {
        onPathTraced(event.detail);
      }
    };

    // Drill down events
    const handleDrillDown = (event: CustomEvent) => {
      if (onDrillDown) {
        onDrillDown(event.detail.nodeId, event.detail.level);
      }
    };

    // Layout change events
    const handleLayoutChanged = (event: CustomEvent) => {
      setCurrentLayout(event.detail.layout);
      if (onLayoutChanged) {
        onLayoutChanged(event.detail.layout);
      }
    };

    // Performance monitoring
    const handlePerformanceUpdate = () => {
      if (onPerformanceUpdate && diagramEngineRef.current && animationEngine) {
        const diagramMetrics = diagramEngineRef.current.getPerformanceMetrics();
        const animationMetrics = animationEngine.getPerformanceMetrics();
        
        const combined = {
          ...diagramMetrics,
          ...animationMetrics,
          timestamp: Date.now()
        };
        
        setPerformanceMetrics(combined);
        onPerformanceUpdate(combined);
      }
    };

    // Add event listeners
    container.addEventListener('nodeSelected', handleNodeSelected as EventListener);
    container.addEventListener('edgeSelected', handleEdgeSelected as EventListener);
    container.addEventListener('pathComputed', handlePathComputed as EventListener);
    container.addEventListener('drillDownRequest', handleDrillDown as EventListener);
    container.addEventListener('layoutChanged', handleLayoutChanged as EventListener);

    // Performance monitoring interval
    const performanceInterval = setInterval(handlePerformanceUpdate, 1000);

    // Cleanup function
    return () => {
      container.removeEventListener('nodeSelected', handleNodeSelected as EventListener);
      container.removeEventListener('edgeSelected', handleEdgeSelected as EventListener);
      container.removeEventListener('pathComputed', handlePathComputed as EventListener);
      container.removeEventListener('drillDownRequest', handleDrillDown as EventListener);
      container.removeEventListener('layoutChanged', handleLayoutChanged as EventListener);
      clearInterval(performanceInterval);
    };
  }, [onNodeSelected, onEdgeSelected, onPathTraced, onDrillDown, onLayoutChanged, onPerformanceUpdate]);

  // Initialize on mount
  useEffect(() => {
    initializeEngines();
    return () => {
      // Cleanup engines
      diagramEngineRef.current?.dispose();
      interactionEngineRef.current?.dispose();
      animationEngineRef.current?.dispose();
    };
  }, []); // Only run once on mount

  // Setup event handlers after initialization
  useEffect(() => {
    if (isInitialized) {
      return setupEventHandlers();
    }
  }, [isInitialized, setupEventHandlers]);

  // Update data when it changes
  useEffect(() => {
    if (isInitialized && diagramEngineRef.current && layoutEngineRef.current) {
      try {
        // Compute layout with brain-inspired positioning
        const positionedNodes = layoutEngineRef.current.computeBrainInspiredLayout(
          data.nodes,
          data.edges
        );

        // Update diagram with positioned data
        const updatedData = {
          ...data,
          nodes: positionedNodes
        };

        diagramEngineRef.current.updateData(updatedData);

        // Create entrance animations for new nodes
        if (animationEngineRef.current) {
          const newNodes = positionedNodes.filter(node => 
            node.status === 'healthy' || node.status === 'warning'
          );
          
          if (newNodes.length > 0) {
            animationEngineRef.current.createAnimation('entrance', 'layout_transition', {
              duration: 1000,
              parameters: {
                nodes: newNodes.map(n => n.id)
              }
            });
          }
        }
      } catch (err) {
        setError(`Failed to update diagram data: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    }
  }, [data, isInitialized]);

  // Handle real-time data updates
  useEffect(() => {
    if (isInitialized && animationEngineRef.current && realTimeData.length > 0) {
      realTimeData.forEach(update => {
        animationEngineRef.current!.addRealTimeUpdate(update);
      });
    }
  }, [realTimeData, isInitialized]);

  // Update configuration when props change
  useEffect(() => {
    if (isInitialized && diagramEngineRef.current) {
      diagramEngineRef.current.setLayout(visualConfig.layout);
      diagramEngineRef.current.setDimensions(visualConfig.dimensions);
    }
  }, [visualConfig, isInitialized]);

  useEffect(() => {
    if (isInitialized && interactionEngineRef.current) {
      interactionEngineRef.current.updateConfig(interactionConfig);
    }
  }, [interactionConfig, isInitialized]);

  useEffect(() => {
    if (isInitialized && animationEngineRef.current) {
      animationEngineRef.current.updateConfig(animationConfig);
    }
  }, [animationConfig, isInitialized]);

  // Create controls object for parent components
  const controls = useMemo<DiagramControls>(() => ({
    zoomIn: () => {
      if (diagramEngineRef.current) {
        // Zoom in by 20%
        containerRef.current?.dispatchEvent(new CustomEvent('zoom', {
          detail: { factor: 1.2, centerX: 0, centerY: 0 }
        }));
      }
    },
    zoomOut: () => {
      if (diagramEngineRef.current) {
        // Zoom out by 20%
        containerRef.current?.dispatchEvent(new CustomEvent('zoom', {
          detail: { factor: 0.8, centerX: 0, centerY: 0 }
        }));
      }
    },
    zoomToFit: () => {
      diagramEngineRef.current?.zoomToFit();
    },
    resetView: () => {
      diagramEngineRef.current?.zoomToFit();
      interactionEngineRef.current?.clearSelection();
    },
    setLayout: (layout: VisualizationConfig['layout']) => {
      if (diagramEngineRef.current && layoutEngineRef.current) {
        diagramEngineRef.current.setLayout(layout);
        setCurrentLayout(layout);
      }
    },
    exportImage: (format = 'png') => {
      return diagramEngineRef.current?.exportImage(format) || '';
    },
    toggleAnimations: () => {
      if (animationEngineRef.current) {
        const currentConfig = { ...animationConfig };
        currentConfig.enableAnimations = !currentConfig.enableAnimations;
        animationEngineRef.current.updateConfig(currentConfig);
      }
    },
    focusOnNode: (nodeId: string) => {
      diagramEngineRef.current?.zoomToNode(nodeId);
      interactionEngineRef.current?.focusOnNode(nodeId);
    }
  }), [animationConfig]);

  // Expose controls via ref
  React.useImperativeHandle(React.forwardRef(() => null).ref, () => controls);

  if (error) {
    return (
      <div className={`architecture-diagram-error ${className || ''}`} style={style}>
        <div className="error-message">
          <h3>Diagram Error</h3>
          <p>{error}</p>
          <button onClick={initializeEngines}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div 
      className={`architecture-diagram-container ${className || ''}`} 
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        backgroundColor: '#fafafa',
        ...style
      }}
    >
      {/* Main diagram container */}
      <div
        ref={containerRef}
        className="architecture-diagram"
        style={{
          width: '100%',
          height: '100%',
          position: 'relative'
        }}
      />

      {/* Loading overlay */}
      {loading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            zIndex: 1000
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div className="loading-spinner" style={{
              width: '40px',
              height: '40px',
              border: '4px solid #f3f3f3',
              borderTop: '4px solid #1976d2',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 16px'
            }} />
            <p>Loading Architecture Diagram...</p>
          </div>
        </div>
      )}

      {/* Performance indicator */}
      {performanceMetrics && onPerformanceUpdate && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            fontFamily: 'monospace',
            zIndex: 999
          }}
        >
          <div>FPS: {performanceMetrics.averageFPS?.toFixed(1) || 'N/A'}</div>
          <div>Nodes: {performanceMetrics.nodeCount || 0}</div>
          <div>Edges: {performanceMetrics.edgeCount || 0}</div>
          <div>Animations: {performanceMetrics.activeAnimations || 0}</div>
        </div>
      )}

      {/* Status indicator */}
      <div
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          display: 'flex',
          alignItems: 'center',
          background: 'rgba(255, 255, 255, 0.9)',
          padding: '6px 12px',
          borderRadius: '20px',
          fontSize: '12px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          zIndex: 999
        }}
      >
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: isInitialized ? '#4caf50' : loading ? '#ff9800' : '#f44336',
            marginRight: '8px'
          }}
        />
        <span>
          {loading ? 'Loading...' : error ? 'Error' : isInitialized ? 'Ready' : 'Initializing...'}
        </span>
        {currentLayout && (
          <span style={{ marginLeft: '12px', color: '#666' }}>
            Layout: {currentLayout}
          </span>
        )}
      </div>

      {/* Add CSS animations */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .architecture-diagram-container {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .architecture-diagram-error {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 100%;
          height: 100%;
          background: #fafafa;
        }
        
        .error-message {
          text-align: center;
          padding: 24px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          max-width: 400px;
        }
        
        .error-message h3 {
          margin: 0 0 16px 0;
          color: #f44336;
        }
        
        .error-message p {
          margin: 0 0 16px 0;
          color: #666;
        }
        
        .error-message button {
          background: #1976d2;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .error-message button:hover {
          background: #1565c0;
        }
      `}</style>
    </div>
  );
};

export default SystemArchitectureDiagram;