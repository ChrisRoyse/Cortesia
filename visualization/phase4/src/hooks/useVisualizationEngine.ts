/**
 * React hook for managing LLMKG visualization engine state and lifecycle
 * Provides integration with Phase 1 WebSocket data and performance monitoring
 */

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import * as THREE from 'three';
import { 
  LLMKGDataFlowVisualizer,
  VisualizationConfig,
  DataFlowNode,
  DataFlowConnection,
  CognitivePattern
} from '../core/LLMKGDataFlowVisualizer';

export interface VisualizationState {
  isInitialized: boolean;
  isRunning: boolean;
  performanceMetrics: any;
  nodeCount: number;
  connectionCount: number;
  patternCount: number;
  error: string | null;
}

export interface WebSocketData {
  type: 'node' | 'connection' | 'pattern' | 'update';
  payload: any;
  timestamp: number;
}

export interface UseVisualizationEngineOptions {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  width: number;
  height: number;
  websocketUrl?: string;
  autoStart?: boolean;
  performanceMonitoring?: boolean;
  config?: Partial<VisualizationConfig>;
}

export function useVisualizationEngine(options: UseVisualizationEngineOptions) {
  const {
    canvasRef,
    width,
    height,
    websocketUrl,
    autoStart = true,
    performanceMonitoring = true,
    config = {}
  } = options;

  const visualizerRef = useRef<LLMKGDataFlowVisualizer | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const animationRef = useRef<number | null>(null);
  const performanceIntervalRef = useRef<number | null>(null);

  const [state, setState] = useState<VisualizationState>({
    isInitialized: false,
    isRunning: false,
    performanceMetrics: null,
    nodeCount: 0,
    connectionCount: 0,
    patternCount: 0,
    error: null
  });

  // Default visualization configuration
  const defaultConfig: VisualizationConfig = useMemo(() => ({
    canvas: canvasRef.current!,
    width,
    height,
    backgroundColor: 0x0a0a0a,
    cameraPosition: new THREE.Vector3(0, 5, 10),
    targetFPS: 60,
    enablePostProcessing: true,
    maxNodes: 1000,
    maxConnections: 2000,
    ...config
  }), [width, height, config, canvasRef]);

  // Initialize visualizer
  const initializeVisualizer = useCallback(() => {
    if (!canvasRef.current) {
      setState(prev => ({ ...prev, error: 'Canvas ref not available' }));
      return false;
    }

    try {
      const visualizer = new LLMKGDataFlowVisualizer({
        ...defaultConfig,
        canvas: canvasRef.current
      });

      visualizerRef.current = visualizer;
      
      setState(prev => ({
        ...prev,
        isInitialized: true,
        error: null
      }));

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to initialize visualizer'
      }));
      return false;
    }
  }, [defaultConfig, canvasRef]);

  // Start visualization
  const startVisualization = useCallback(() => {
    if (!visualizerRef.current || !state.isInitialized) return false;

    try {
      visualizerRef.current.start();
      
      setState(prev => ({
        ...prev,
        isRunning: true,
        error: null
      }));

      // Start performance monitoring
      if (performanceMonitoring) {
        performanceIntervalRef.current = window.setInterval(() => {
          if (visualizerRef.current) {
            const metrics = visualizerRef.current.getPerformanceMetrics();
            setState(prev => ({
              ...prev,
              performanceMetrics: metrics,
              nodeCount: metrics.nodeCount,
              connectionCount: metrics.connectionCount,
              patternCount: metrics.patternCount
            }));
          }
        }, 1000);
      }

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start visualization'
      }));
      return false;
    }
  }, [state.isInitialized, performanceMonitoring]);

  // Stop visualization
  const stopVisualization = useCallback(() => {
    if (visualizerRef.current) {
      visualizerRef.current.stop();
    }

    if (performanceIntervalRef.current) {
      clearInterval(performanceIntervalRef.current);
      performanceIntervalRef.current = null;
    }

    setState(prev => ({
      ...prev,
      isRunning: false
    }));
  }, []);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (!websocketUrl) return;

    try {
      const ws = new WebSocket(websocketUrl);
      websocketRef.current = ws;

      ws.onopen = () => {
        console.log('Connected to LLMKG WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketData = JSON.parse(event.data);
          handleWebSocketData(data);
        } catch (error) {
          console.error('Error parsing WebSocket data:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        // Attempt reconnection after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setState(prev => ({
          ...prev,
          error: 'WebSocket connection error'
        }));
      };
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to connect WebSocket'
      }));
    }
  }, [websocketUrl]);

  // Handle WebSocket data
  const handleWebSocketData = useCallback((data: WebSocketData) => {
    if (!visualizerRef.current) return;

    try {
      switch (data.type) {
        case 'node':
          if (data.payload.action === 'add') {
            visualizerRef.current.addNode(data.payload.node as DataFlowNode);
          } else if (data.payload.action === 'update') {
            visualizerRef.current.updateNode(
              data.payload.nodeId,
              data.payload.updates
            );
          } else if (data.payload.action === 'remove') {
            visualizerRef.current.removeNode(data.payload.nodeId);
          }
          break;

        case 'connection':
          if (data.payload.action === 'add') {
            visualizerRef.current.addConnection(data.payload.connection as DataFlowConnection);
          } else if (data.payload.action === 'remove') {
            visualizerRef.current.removeConnection(data.payload.connectionId);
          }
          break;

        case 'pattern':
          if (data.payload.action === 'add') {
            visualizerRef.current.addCognitivePattern(data.payload.pattern as CognitivePattern);
          } else if (data.payload.action === 'remove') {
            visualizerRef.current.removeCognitivePattern(data.payload.patternId);
          }
          break;

        default:
          console.warn('Unknown WebSocket data type:', data.type);
      }
    } catch (error) {
      console.error('Error handling WebSocket data:', error);
    }
  }, []);

  // Add node manually
  const addNode = useCallback((node: DataFlowNode) => {
    if (visualizerRef.current) {
      visualizerRef.current.addNode(node);
    }
  }, []);

  // Add connection manually
  const addConnection = useCallback((connection: DataFlowConnection) => {
    if (visualizerRef.current) {
      visualizerRef.current.addConnection(connection);
    }
  }, []);

  // Add cognitive pattern manually
  const addCognitivePattern = useCallback((pattern: CognitivePattern) => {
    if (visualizerRef.current) {
      visualizerRef.current.addCognitivePattern(pattern);
    }
  }, []);

  // Update node
  const updateNode = useCallback((nodeId: string, updates: Partial<DataFlowNode>) => {
    if (visualizerRef.current) {
      visualizerRef.current.updateNode(nodeId, updates);
    }
  }, []);

  // Resize handler
  const handleResize = useCallback((newWidth: number, newHeight: number) => {
    if (visualizerRef.current) {
      visualizerRef.current.resize(newWidth, newHeight);
    }
  }, []);

  // Generate demo data for testing
  const generateDemoData = useCallback(() => {
    if (!visualizerRef.current) return;

    // Add demo nodes
    for (let i = 0; i < 10; i++) {
      const node: DataFlowNode = {
        id: `demo_node_${i}`,
        position: new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 10
        ),
        type: ['input', 'processing', 'output', 'cognitive'][Math.floor(Math.random() * 4)] as any,
        activation: Math.random(),
        connections: []
      };
      addNode(node);
    }

    // Add demo connections
    for (let i = 0; i < 5; i++) {
      const connection: DataFlowConnection = {
        id: `demo_connection_${i}`,
        source: `demo_node_${i}`,
        target: `demo_node_${(i + 1) % 10}`,
        strength: 0.5 + Math.random() * 0.5,
        dataType: 'neural_signal',
        isActive: Math.random() > 0.3
      };
      addConnection(connection);
    }

    // Add demo cognitive pattern
    const pattern: CognitivePattern = {
      id: 'demo_attention_pattern',
      center: new THREE.Vector3(0, 0, 0),
      complexity: 0.7,
      strength: 0.8,
      type: 'attention',
      nodes: ['demo_node_0', 'demo_node_1', 'demo_node_2']
    };
    addCognitivePattern(pattern);
  }, [addNode, addConnection, addCognitivePattern]);

  // Initialize on mount
  useEffect(() => {
    if (!state.isInitialized) {
      initializeVisualizer();
    }
  }, [initializeVisualizer, state.isInitialized]);

  // Auto-start if enabled
  useEffect(() => {
    if (state.isInitialized && autoStart && !state.isRunning) {
      startVisualization();
    }
  }, [state.isInitialized, autoStart, state.isRunning, startVisualization]);

  // Connect WebSocket if URL provided
  useEffect(() => {
    if (state.isInitialized && websocketUrl) {
      connectWebSocket();
    }

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [state.isInitialized, websocketUrl, connectWebSocket]);

  // Handle resize
  useEffect(() => {
    handleResize(width, height);
  }, [width, height, handleResize]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopVisualization();
      
      if (visualizerRef.current) {
        visualizerRef.current.dispose();
      }

      if (websocketRef.current) {
        websocketRef.current.close();
      }

      if (performanceIntervalRef.current) {
        clearInterval(performanceIntervalRef.current);
      }
    };
  }, [stopVisualization]);

  return {
    state,
    actions: {
      start: startVisualization,
      stop: stopVisualization,
      addNode,
      addConnection,
      addCognitivePattern,
      updateNode,
      resize: handleResize,
      generateDemoData
    },
    visualizer: visualizerRef.current
  };
}

export default useVisualizationEngine;