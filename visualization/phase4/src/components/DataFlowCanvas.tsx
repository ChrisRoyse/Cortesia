/**
 * React component for LLMKG Data Flow Visualization Canvas
 * Integrates the 3D visualization engine with React component lifecycle
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { useVisualizationEngine } from '../hooks/useVisualizationEngine';
import { DataFlowNode, DataFlowConnection, CognitivePattern } from '../core/LLMKGDataFlowVisualizer';

export interface DataFlowCanvasProps {
  width?: number;
  height?: number;
  websocketUrl?: string;
  autoStart?: boolean;
  showControls?: boolean;
  showPerformanceMetrics?: boolean;
  enableDemoMode?: boolean;
  className?: string;
  style?: React.CSSProperties;
  onNodeClick?: (node: DataFlowNode) => void;
  onConnectionClick?: (connection: DataFlowConnection) => void;
  onPatternClick?: (pattern: CognitivePattern) => void;
  onPerformanceUpdate?: (metrics: any) => void;
}

export const DataFlowCanvas: React.FC<DataFlowCanvasProps> = ({
  width = 800,
  height = 600,
  websocketUrl,
  autoStart = true,
  showControls = false,
  showPerformanceMetrics = false,
  enableDemoMode = false,
  className = '',
  style = {},
  onNodeClick,
  onConnectionClick,
  onPatternClick,
  onPerformanceUpdate
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width, height });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { state, actions, visualizer } = useVisualizationEngine({
    canvasRef,
    width: canvasSize.width,
    height: canvasSize.height,
    websocketUrl,
    autoStart,
    performanceMonitoring: showPerformanceMetrics,
    config: {
      backgroundColor: 0x0a0a0a,
      cameraPosition: new THREE.Vector3(0, 5, 15),
      targetFPS: 60,
      enablePostProcessing: true,
      maxNodes: 1000,
      maxConnections: 2000
    }
  });

  // Handle performance updates
  useEffect(() => {
    if (state.performanceMetrics && onPerformanceUpdate) {
      onPerformanceUpdate(state.performanceMetrics);
    }
  }, [state.performanceMetrics, onPerformanceUpdate]);

  // Handle canvas resize
  const handleResize = useCallback(() => {
    if (!canvasRef.current) return;

    const container = canvasRef.current.parentElement;
    if (container) {
      const rect = container.getBoundingClientRect();
      const newSize = {
        width: Math.floor(rect.width),
        height: Math.floor(rect.height)
      };

      if (newSize.width !== canvasSize.width || newSize.height !== canvasSize.height) {
        setCanvasSize(newSize);
      }
    }
  }, [canvasSize]);

  // Setup resize observer
  useEffect(() => {
    const resizeObserver = new ResizeObserver(handleResize);
    
    if (canvasRef.current?.parentElement) {
      resizeObserver.observe(canvasRef.current.parentElement);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [handleResize]);

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!canvasRef.current) return;

    if (!document.fullscreenElement) {
      canvasRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      });
    }
  }, []);

  // Handle canvas interaction (mouse/touch events)
  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!visualizer || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Create a raycaster to detect clicked objects
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(x, y);
    
    // This would require access to the visualizer's camera and scene
    // Implementation would depend on exposing these from the visualizer
    console.log('Canvas clicked at:', { x, y });
  }, [visualizer]);

  // Generate demo data for testing
  const generateDemoData = useCallback(() => {
    actions.generateDemoData();
  }, [actions]);

  // Control panel component
  const ControlPanel = () => (
    <div className="data-flow-controls" style={{
      position: 'absolute',
      top: 10,
      left: 10,
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '10px',
      borderRadius: '5px',
      color: 'white',
      fontSize: '12px',
      zIndex: 1000
    }}>
      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={state.isRunning ? actions.stop : actions.start}
          style={{
            marginRight: '10px',
            padding: '5px 10px',
            background: state.isRunning ? '#ff4444' : '#44ff44',
            border: 'none',
            color: 'white',
            borderRadius: '3px',
            cursor: 'pointer'
          }}
        >
          {state.isRunning ? 'Stop' : 'Start'}
        </button>

        <button
          onClick={toggleFullscreen}
          style={{
            marginRight: '10px',
            padding: '5px 10px',
            background: '#4488ff',
            border: 'none',
            color: 'white',
            borderRadius: '3px',
            cursor: 'pointer'
          }}
        >
          {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>

        {enableDemoMode && (
          <button
            onClick={generateDemoData}
            style={{
              padding: '5px 10px',
              background: '#ffaa44',
              border: 'none',
              color: 'white',
              borderRadius: '3px',
              cursor: 'pointer'
            }}
          >
            Generate Demo Data
          </button>
        )}
      </div>

      <div>
        Status: {state.isInitialized ? (state.isRunning ? 'Running' : 'Ready') : 'Initializing'}
      </div>
      
      {state.error && (
        <div style={{ color: '#ff6666', marginTop: '5px' }}>
          Error: {state.error}
        </div>
      )}
    </div>
  );

  // Performance metrics component
  const PerformanceMetrics = () => {
    if (!showPerformanceMetrics || !state.performanceMetrics) return null;

    const metrics = state.performanceMetrics;
    
    return (
      <div className="performance-metrics" style={{
        position: 'absolute',
        top: 10,
        right: 10,
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '10px',
        borderRadius: '5px',
        color: 'white',
        fontSize: '11px',
        zIndex: 1000,
        minWidth: '200px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
          Performance Metrics
        </div>
        
        <div>FPS: {metrics.fps} / {metrics.targetFPS}</div>
        <div>Nodes: {metrics.nodeCount}</div>
        <div>Connections: {metrics.connectionCount}</div>
        <div>Patterns: {metrics.patternCount}</div>
        
        {metrics.particles && (
          <>
            <div style={{ marginTop: '5px', fontWeight: 'bold' }}>Particles:</div>
            <div>Active: {metrics.particles.activeParticles}</div>
            <div>Max: {metrics.particles.maxParticles}</div>
            <div>Utilization: {metrics.particles.utilizationPercentage.toFixed(1)}%</div>
          </>
        )}
        
        {metrics.renderer && (
          <>
            <div style={{ marginTop: '5px', fontWeight: 'bold' }}>Renderer:</div>
            <div>Draw Calls: {metrics.renderer.calls}</div>
            <div>Triangles: {metrics.renderer.triangles}</div>
            <div>Points: {metrics.renderer.points}</div>
          </>
        )}
      </div>
    );
  };

  return (
    <div 
      className={`data-flow-canvas-container ${className}`}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        ...style
      }}
    >
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        onClick={handleCanvasClick}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          cursor: 'pointer'
        }}
      />
      
      {showControls && <ControlPanel />}
      {showPerformanceMetrics && <PerformanceMetrics />}
      
      {!state.isInitialized && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '5px',
          textAlign: 'center'
        }}>
          {state.error ? `Error: ${state.error}` : 'Initializing visualization...'}
        </div>
      )}
    </div>
  );
};

// Helper function to create a demo node
export const createDemoNode = (id: string, type: DataFlowNode['type']): DataFlowNode => ({
  id,
  position: new THREE.Vector3(
    (Math.random() - 0.5) * 10,
    (Math.random() - 0.5) * 5,
    (Math.random() - 0.5) * 10
  ),
  type,
  activation: Math.random(),
  connections: []
});

// Helper function to create a demo connection
export const createDemoConnection = (
  id: string,
  source: string,
  target: string
): DataFlowConnection => ({
  id,
  source,
  target,
  strength: 0.5 + Math.random() * 0.5,
  dataType: 'neural_signal',
  isActive: Math.random() > 0.3
});

// Helper function to create a demo cognitive pattern
export const createDemoCognitivePattern = (
  id: string,
  type: CognitivePattern['type']
): CognitivePattern => ({
  id,
  center: new THREE.Vector3(
    (Math.random() - 0.5) * 5,
    (Math.random() - 0.5) * 5,
    (Math.random() - 0.5) * 5
  ),
  complexity: 0.5 + Math.random() * 0.5,
  strength: 0.6 + Math.random() * 0.4,
  type,
  nodes: []
});

export default DataFlowCanvas;