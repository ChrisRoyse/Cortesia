import React, { useState, useRef, useEffect } from 'react';
import { SystemArchitectureDiagram, DiagramControls } from '../src/components/SystemArchitectureDiagram';
import { useArchitectureVisualization } from '../src/hooks/useArchitectureVisualization';
import { ArchitectureData, ComponentNode, ComponentEdge } from '../src/core/ArchitectureDiagramEngine';
import { PathTraceResult } from '../src/core/InteractionEngine';

// Sample architecture data representing LLMKG system
const generateSampleArchitectureData = (): ArchitectureData => ({
  nodes: [
    // Phase 1 nodes
    {
      id: 'websocket_server',
      label: 'WebSocket Server',
      type: 'engine',
      phase: 1,
      status: 'healthy',
      position: { x: 100, y: 200 },
      metrics: {
        performance: 0.95,
        memory: 45.2,
        connections: 8,
        load: 0.3
      }
    },
    {
      id: 'input_processor',
      label: 'Input Processor',
      type: 'module',
      phase: 1,
      status: 'healthy',
      position: { x: 250, y: 150 },
      metrics: {
        performance: 0.87,
        memory: 32.1,
        connections: 12,
        load: 0.5
      }
    },
    
    // Phase 2 nodes
    {
      id: 'cognitive_engine',
      label: 'Cognitive Engine',
      type: 'engine',
      phase: 2,
      status: 'warning',
      position: { x: 400, y: 180 },
      metrics: {
        performance: 0.75,
        memory: 128.4,
        connections: 15,
        load: 0.8
      }
    },
    {
      id: 'activation_engine',
      label: 'Activation Engine',
      type: 'engine',
      phase: 2,
      status: 'healthy',
      position: { x: 350, y: 300 },
      metrics: {
        performance: 0.92,
        memory: 67.8,
        connections: 10,
        load: 0.4
      }
    },
    
    // Phase 3 nodes
    {
      id: 'knowledge_store',
      label: 'Knowledge Store',
      type: 'module',
      phase: 3,
      status: 'healthy',
      position: { x: 550, y: 120 },
      metrics: {
        performance: 0.88,
        memory: 256.7,
        connections: 20,
        load: 0.6
      }
    },
    {
      id: 'memory_engine',
      label: 'Memory Engine',
      type: 'engine',
      phase: 3,
      status: 'healthy',
      position: { x: 600, y: 250 },
      metrics: {
        performance: 0.91,
        memory: 89.3,
        connections: 14,
        load: 0.45
      }
    },
    
    // Phase 4 nodes
    {
      id: 'inhibitory_control',
      label: 'Inhibitory Control',
      type: 'layer',
      phase: 4,
      status: 'healthy',
      position: { x: 750, y: 200 },
      metrics: {
        performance: 0.94,
        memory: 34.6,
        connections: 8,
        load: 0.25
      }
    },
    {
      id: 'executive_control',
      label: 'Executive Control',
      type: 'engine',
      phase: 4,
      status: 'error',
      position: { x: 700, y: 350 },
      metrics: {
        performance: 0.45,
        memory: 156.2,
        connections: 6,
        load: 0.95
      }
    },
    
    // Phase 5 nodes
    {
      id: 'visualization_engine',
      label: 'Visualization Engine',
      type: 'engine',
      phase: 5,
      status: 'healthy',
      position: { x: 900, y: 280 },
      metrics: {
        performance: 0.89,
        memory: 78.9,
        connections: 5,
        load: 0.35
      }
    },
    {
      id: 'output_formatter',
      label: 'Output Formatter',
      type: 'module',
      phase: 5,
      status: 'healthy',
      position: { x: 850, y: 150 },
      metrics: {
        performance: 0.96,
        memory: 23.4,
        connections: 3,
        load: 0.2
      }
    }
  ],
  edges: [
    // Phase 1 to 2 connections
    {
      id: 'ws_to_cognitive',
      source: 'websocket_server',
      target: 'cognitive_engine',
      type: 'data_flow',
      status: 'active',
      weight: 2,
      latency: 15,
      throughput: 1200
    },
    {
      id: 'input_to_activation',
      source: 'input_processor',
      target: 'activation_engine',
      type: 'data_flow',
      status: 'active',
      weight: 3,
      latency: 8,
      throughput: 2400
    },
    
    // Phase 2 to 3 connections
    {
      id: 'cognitive_to_knowledge',
      source: 'cognitive_engine',
      target: 'knowledge_store',
      type: 'data_flow',
      status: 'congested',
      weight: 4,
      latency: 45,
      throughput: 800
    },
    {
      id: 'activation_to_memory',
      source: 'activation_engine',
      target: 'memory_engine',
      type: 'control',
      status: 'active',
      weight: 2,
      latency: 12,
      throughput: 1500
    },
    
    // Phase 3 to 4 connections
    {
      id: 'knowledge_to_inhibitory',
      source: 'knowledge_store',
      target: 'inhibitory_control',
      type: 'feedback',
      status: 'active',
      weight: 2,
      latency: 20,
      throughput: 900
    },
    {
      id: 'memory_to_executive',
      source: 'memory_engine',
      target: 'executive_control',
      type: 'control',
      status: 'inactive',
      weight: 1,
      latency: 100,
      throughput: 200
    },
    
    // Phase 4 to 5 connections
    {
      id: 'inhibitory_to_viz',
      source: 'inhibitory_control',
      target: 'visualization_engine',
      type: 'control',
      status: 'active',
      weight: 3,
      latency: 18,
      throughput: 1100
    },
    {
      id: 'executive_to_output',
      source: 'executive_control',
      target: 'output_formatter',
      type: 'data_flow',
      status: 'inactive',
      weight: 1,
      latency: 200,
      throughput: 50
    },
    
    // Cross-phase feedback loops
    {
      id: 'viz_to_cognitive_feedback',
      source: 'visualization_engine',
      target: 'cognitive_engine',
      type: 'feedback',
      status: 'active',
      weight: 1,
      latency: 25,
      throughput: 600
    },
    {
      id: 'inhibitory_feedback',
      source: 'inhibitory_control',
      target: 'activation_engine',
      type: 'inhibition',
      status: 'active',
      weight: 2,
      latency: 10,
      throughput: 1800
    }
  ],
  metadata: {
    timestamp: Date.now(),
    totalComponents: 10,
    activeConnections: 8,
    systemHealth: 0.78
  }
});

const ArchitectureDashboard: React.FC = () => {
  const diagramControlsRef = useRef<DiagramControls | null>(null);
  const [selectedLayout, setSelectedLayout] = useState<'hierarchical' | 'brain_inspired' | 'force_directed' | 'circular'>('brain_inspired');
  const [showPerformance, setShowPerformance] = useState(true);
  const [animationsEnabled, setAnimationsEnabled] = useState(true);

  // Initialize the architecture visualization hook
  const [visualizationState, visualizationActions] = useArchitectureVisualization({
    dataSource: {
      type: 'static',
      static: {
        data: generateSampleArchitectureData()
      }
    },
    updateInterval: 1000,
    maxRealTimeBufferSize: 50,
    enablePerformanceMonitoring: true,
    autoReconnect: false
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate performance fluctuations
      visualizationState.data.nodes.forEach(node => {
        if (node.metrics) {
          const performanceChange = (Math.random() - 0.5) * 0.1;
          const loadChange = (Math.random() - 0.5) * 0.2;
          
          visualizationActions.updateNode(node.id, {
            metrics: {
              ...node.metrics,
              performance: Math.max(0, Math.min(1, node.metrics.performance + performanceChange)),
              load: Math.max(0, Math.min(1, node.metrics.load + loadChange))
            }
          });

          // Occasionally change status based on performance
          if (node.metrics.performance < 0.6) {
            visualizationActions.updateNode(node.id, { status: 'warning' });
          } else if (node.metrics.performance < 0.4) {
            visualizationActions.updateNode(node.id, { status: 'error' });
          } else if (node.metrics.performance > 0.8) {
            visualizationActions.updateNode(node.id, { status: 'healthy' });
          }
        }
      });

      // Simulate edge status changes
      visualizationState.data.edges.forEach(edge => {
        if (Math.random() < 0.1) { // 10% chance of status change
          const statuses: ComponentEdge['status'][] = ['active', 'inactive', 'congested'];
          const newStatus = statuses[Math.floor(Math.random() * statuses.length)];
          visualizationActions.updateEdge(edge.id, { status: newStatus });
        }
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [visualizationState.data, visualizationActions]);

  // Event handlers
  const handleNodeSelected = (nodeId: string, nodeData: any) => {
    console.log('Node selected:', nodeId, nodeData);
  };

  const handleEdgeSelected = (edgeId: string, edgeData: any) => {
    console.log('Edge selected:', edgeId, edgeData);
  };

  const handlePathTraced = (result: PathTraceResult) => {
    console.log('Path traced:', result);
  };

  const handleDrillDown = (nodeId: string, level: number) => {
    console.log('Drill down into:', nodeId, 'at level:', level);
    visualizationActions.drillDownInto(nodeId);
  };

  const handleLayoutChanged = (layoutType: string) => {
    console.log('Layout changed to:', layoutType);
    setSelectedLayout(layoutType as any);
  };

  const handlePerformanceUpdate = (metrics: any) => {
    if (showPerformance) {
      console.log('Performance metrics:', metrics);
    }
  };

  // Control panel actions
  const handleLayoutChange = (layout: typeof selectedLayout) => {
    setSelectedLayout(layout);
    // This would be handled by the DiagramControls ref if properly implemented
  };

  const handleZoomToFit = () => {
    // diagramControlsRef.current?.zoomToFit();
  };

  const handleExportImage = () => {
    // const imageData = diagramControlsRef.current?.exportImage('png');
    // if (imageData) {
    //   const link = document.createElement('a');
    //   link.download = 'llmkg_architecture.png';
    //   link.href = imageData;
    //   link.click();
    // }
  };

  const handleToggleAnimations = () => {
    setAnimationsEnabled(!animationsEnabled);
    // diagramControlsRef.current?.toggleAnimations();
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: '#f5f5f5',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px 24px',
        backgroundColor: '#1976d2',
        color: 'white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ margin: 0, fontSize: '20px' }}>
          LLMKG Phase 5 - System Architecture Dashboard
        </h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ fontSize: '14px' }}>
            System Health: {(visualizationState.data.metadata.systemHealth * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '14px' }}>
            Components: {visualizationState.data.nodes.length}
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '8px 24px',
        backgroundColor: 'white',
        borderBottom: '1px solid #e0e0e0',
        gap: '16px'
      }}>
        {/* Layout Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '14px', fontWeight: '500' }}>Layout:</label>
          <select 
            value={selectedLayout} 
            onChange={(e) => handleLayoutChange(e.target.value as any)}
            style={{
              padding: '4px 8px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '14px'
            }}
          >
            <option value="brain_inspired">Brain Inspired</option>
            <option value="hierarchical">Hierarchical</option>
            <option value="force_directed">Force Directed</option>
            <option value="circular">Circular</option>
          </select>
        </div>

        {/* View Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            onClick={handleZoomToFit}
            style={{
              padding: '6px 12px',
              border: '1px solid #1976d2',
              borderRadius: '4px',
              backgroundColor: 'white',
              color: '#1976d2',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Fit to View
          </button>
          
          <button 
            onClick={handleExportImage}
            style={{
              padding: '6px 12px',
              border: '1px solid #1976d2',
              borderRadius: '4px',
              backgroundColor: 'white',
              color: '#1976d2',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Export PNG
          </button>
        </div>

        {/* Settings */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '14px' }}>
            <input 
              type="checkbox" 
              checked={animationsEnabled}
              onChange={handleToggleAnimations}
            />
            Animations
          </label>
          
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '14px' }}>
            <input 
              type="checkbox" 
              checked={showPerformance}
              onChange={(e) => setShowPerformance(e.target.checked)}
            />
            Performance Monitor
          </label>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ 
        display: 'flex', 
        flex: 1, 
        overflow: 'hidden',
        gap: '0'
      }}>
        {/* Architecture Diagram */}
        <div style={{ flex: 1, position: 'relative' }}>
          <SystemArchitectureDiagram
            data={visualizationState.data}
            realTimeData={visualizationState.realTimeUpdates}
            visualization={{
              layout: selectedLayout,
              dimensions: '2d',
              enableAnimations: animationsEnabled,
              showLabels: true,
              showMetrics: true
            }}
            interaction={{
              enableMultiSelect: true,
              enablePathTracing: true,
              enableDrillDown: true,
              enableTooltips: true,
              enableContextMenu: true
            }}
            animation={{
              enableAnimations: animationsEnabled,
              globalSpeed: 1.0,
              performanceMode: 'balanced'
            }}
            onNodeSelected={handleNodeSelected}
            onEdgeSelected={handleEdgeSelected}
            onPathTraced={handlePathTraced}
            onDrillDown={handleDrillDown}
            onLayoutChanged={handleLayoutChanged}
            onPerformanceUpdate={showPerformance ? handlePerformanceUpdate : undefined}
          />
        </div>

        {/* Side Panel */}
        <div style={{
          width: '300px',
          backgroundColor: 'white',
          borderLeft: '1px solid #e0e0e0',
          padding: '16px',
          overflow: 'auto'
        }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>System Status</h3>
          
          {/* Overall Health */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '14px', marginBottom: '4px' }}>System Health</div>
            <div style={{
              width: '100%',
              height: '8px',
              backgroundColor: '#e0e0e0',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${visualizationState.data.metadata.systemHealth * 100}%`,
                height: '100%',
                backgroundColor: visualizationState.data.metadata.systemHealth > 0.8 ? '#4caf50' : 
                               visualizationState.data.metadata.systemHealth > 0.6 ? '#ff9800' : '#f44336',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>

          {/* Phase Status */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '14px', marginBottom: '8px' }}>Phase Status</div>
            {[1, 2, 3, 4, 5].map(phase => {
              const phaseNodes = visualizationState.data.nodes.filter(n => n.phase === phase);
              const healthyCount = phaseNodes.filter(n => n.status === 'healthy').length;
              const warningCount = phaseNodes.filter(n => n.status === 'warning').length;
              const errorCount = phaseNodes.filter(n => n.status === 'error').length;
              
              return (
                <div key={phase} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '4px 8px',
                  margin: '2px 0',
                  backgroundColor: '#f5f5f5',
                  borderRadius: '4px',
                  fontSize: '12px'
                }}>
                  <span>Phase {phase}</span>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <span style={{ color: '#4caf50' }}>✓{healthyCount}</span>
                    {warningCount > 0 && <span style={{ color: '#ff9800' }}>⚠{warningCount}</span>}
                    {errorCount > 0 && <span style={{ color: '#f44336' }}>✗{errorCount}</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Selected Items */}
          {(visualizationState.selectedNodes.length > 0 || visualizationState.selectedEdges.length > 0) && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '14px', marginBottom: '8px' }}>Selection</div>
              {visualizationState.selectedNodes.map(nodeId => {
                const node = visualizationState.data.nodes.find(n => n.id === nodeId);
                return node ? (
                  <div key={nodeId} style={{
                    padding: '4px 8px',
                    backgroundColor: '#e3f2fd',
                    borderRadius: '4px',
                    fontSize: '12px',
                    marginBottom: '2px'
                  }}>
                    {node.label} (Phase {node.phase})
                  </div>
                ) : null;
              })}
              {visualizationState.selectedEdges.map(edgeId => {
                const edge = visualizationState.data.edges.find(e => e.id === edgeId);
                return edge ? (
                  <div key={edgeId} style={{
                    padding: '4px 8px',
                    backgroundColor: '#fff3e0',
                    borderRadius: '4px',
                    fontSize: '12px',
                    marginBottom: '2px'
                  }}>
                    Connection: {edge.type}
                  </div>
                ) : null;
              })}
            </div>
          )}

          {/* Path Trace Results */}
          {visualizationState.pathTrace && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '14px', marginBottom: '8px' }}>Path Analysis</div>
              <div style={{
                padding: '8px',
                backgroundColor: '#f5f5f5',
                borderRadius: '4px',
                fontSize: '12px'
              }}>
                <div>Path Length: {visualizationState.pathTrace.path.length} nodes</div>
                <div>Total Latency: {visualizationState.pathTrace.totalLatency}ms</div>
                <div>Efficiency: {(visualizationState.pathTrace.pathEfficiency * 100).toFixed(1)}%</div>
                {visualizationState.pathTrace.bottlenecks.length > 0 && (
                  <div>Bottlenecks: {visualizationState.pathTrace.bottlenecks.length}</div>
                )}
              </div>
            </div>
          )}

          {/* Error Display */}
          {visualizationState.error && (
            <div style={{
              padding: '8px',
              backgroundColor: '#ffebee',
              borderLeft: '4px solid #f44336',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#c62828',
              marginBottom: '16px'
            }}>
              {visualizationState.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ArchitectureDashboard;