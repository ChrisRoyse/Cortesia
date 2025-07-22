import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { ArchitectureData, ComponentNode, ComponentEdge } from '../core/ArchitectureDiagramEngine';
import { RealTimeUpdate } from '../core/AnimationEngine';
import { PathTraceResult } from '../core/InteractionEngine';

export interface VisualizationState {
  data: ArchitectureData;
  realTimeUpdates: RealTimeUpdate[];
  selectedNodes: string[];
  selectedEdges: string[];
  pathTrace: PathTraceResult | null;
  drillDownHistory: string[];
  performanceMetrics: any;
  isLoading: boolean;
  error: string | null;
}

export interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
}

export interface DataSourceConfig {
  type: 'websocket' | 'polling' | 'static';
  websocket?: WebSocketConfig;
  polling?: {
    url: string;
    interval: number;
  };
  static?: {
    data: ArchitectureData;
  };
}

export interface VisualizationHookOptions {
  dataSource: DataSourceConfig;
  updateInterval?: number;
  maxRealTimeBufferSize?: number;
  enablePerformanceMonitoring?: boolean;
  autoReconnect?: boolean;
}

export interface VisualizationActions {
  // Data management
  updateData: (data: Partial<ArchitectureData>) => void;
  addNode: (node: ComponentNode) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, updates: Partial<ComponentNode>) => void;
  addEdge: (edge: ComponentEdge) => void;
  removeEdge: (edgeId: string) => void;
  updateEdge: (edgeId: string, updates: Partial<ComponentEdge>) => void;
  
  // Selection management
  selectNode: (nodeId: string, multiSelect?: boolean) => void;
  selectEdge: (edgeId: string, multiSelect?: boolean) => void;
  clearSelection: () => void;
  
  // Path tracing
  tracePathBetween: (startNodeId: string, endNodeId: string) => Promise<PathTraceResult | null>;
  clearPathTrace: () => void;
  
  // Drill down
  drillDownInto: (nodeId: string) => void;
  drillDownBack: () => void;
  resetDrillDown: () => void;
  
  // Real-time updates
  startRealTimeUpdates: () => void;
  stopRealTimeUpdates: () => void;
  addRealTimeUpdate: (update: RealTimeUpdate) => void;
  
  // Error handling
  clearError: () => void;
  retryConnection: () => void;
}

export const useArchitectureVisualization = (
  options: VisualizationHookOptions
): [VisualizationState, VisualizationActions] => {
  
  // State management
  const [state, setState] = useState<VisualizationState>({
    data: {
      nodes: [],
      edges: [],
      metadata: {
        timestamp: Date.now(),
        totalComponents: 0,
        activeConnections: 0,
        systemHealth: 1.0
      }
    },
    realTimeUpdates: [],
    selectedNodes: [],
    selectedEdges: [],
    pathTrace: null,
    drillDownHistory: [],
    performanceMetrics: null,
    isLoading: false,
    error: null
  });

  // WebSocket connection management
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);

  // Path finding cache
  const pathCache = useRef(new Map<string, PathTraceResult>());

  // WebSocket connection setup
  const connectWebSocket = useCallback(() => {
    if (!options.dataSource.websocket) return;

    const { url, maxReconnectAttempts, heartbeatInterval } = options.dataSource.websocket;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected to LLMKG Phase 1');
        setState(prev => ({ ...prev, isLoading: false, error: null }));
        reconnectAttempts.current = 0;

        // Send heartbeat
        if (heartbeatInterval > 0) {
          heartbeatIntervalRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'heartbeat', timestamp: Date.now() }));
            }
          }, heartbeatInterval);
        }

        // Request initial data
        ws.send(JSON.stringify({
          type: 'subscribe',
          topics: ['architecture_updates', 'component_status', 'system_metrics']
        }));
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (err) {
          console.warn('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setState(prev => ({ 
          ...prev, 
          error: 'WebSocket connection error',
          isLoading: false 
        }));
      };

      ws.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        setState(prev => ({ ...prev, isLoading: false }));

        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        // Auto-reconnect if enabled
        if (options.autoReconnect && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectAttempts.current++;
          
          setState(prev => ({ 
            ...prev, 
            error: `Connection lost, reconnecting in ${delay/1000}s... (attempt ${reconnectAttempts.current})`
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setState(prev => ({ 
            ...prev, 
            error: 'Failed to reconnect after maximum attempts'
          }));
        }
      };

    } catch (err) {
      setState(prev => ({ 
        ...prev, 
        error: `Failed to connect: ${err instanceof Error ? err.message : 'Unknown error'}`,
        isLoading: false 
      }));
    }
  }, [options.dataSource.websocket, options.autoReconnect]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'architecture_update':
        setState(prev => ({
          ...prev,
          data: message.data,
          realTimeUpdates: []
        }));
        break;

      case 'component_status_update':
        const realTimeUpdate: RealTimeUpdate = {
          timestamp: message.timestamp || Date.now(),
          nodeUpdates: new Map(Object.entries(message.nodeUpdates || {})),
          edgeUpdates: new Map(Object.entries(message.edgeUpdates || {})),
          systemMetrics: message.systemMetrics || {
            performance: 1.0,
            health: 1.0,
            load: 0.0,
            connections: 0
          }
        };

        setState(prev => {
          const newUpdates = [...prev.realTimeUpdates, realTimeUpdate];
          // Keep buffer size manageable
          if (newUpdates.length > (options.maxRealTimeBufferSize || 100)) {
            newUpdates.splice(0, newUpdates.length - (options.maxRealTimeBufferSize || 100));
          }
          return { ...prev, realTimeUpdates: newUpdates };
        });
        break;

      case 'system_metrics':
        setState(prev => ({
          ...prev,
          performanceMetrics: message.metrics
        }));
        break;

      case 'path_trace_result':
        setState(prev => ({
          ...prev,
          pathTrace: message.result
        }));
        break;

      case 'error':
        setState(prev => ({
          ...prev,
          error: message.message
        }));
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  }, [options.maxRealTimeBufferSize]);

  // Polling setup
  const startPolling = useCallback(() => {
    if (!options.dataSource.polling) return;

    const { url, interval } = options.dataSource.polling;

    const poll = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true }));
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        setState(prev => ({
          ...prev,
          data,
          isLoading: false,
          error: null
        }));
      } catch (err) {
        setState(prev => ({
          ...prev,
          error: `Polling failed: ${err instanceof Error ? err.message : 'Unknown error'}`,
          isLoading: false
        }));
      }
    };

    // Initial poll
    poll();

    // Setup interval
    pollingIntervalRef.current = setInterval(poll, interval);
  }, [options.dataSource.polling]);

  // Path finding algorithm
  const findPath = useCallback(async (
    startNodeId: string, 
    endNodeId: string
  ): Promise<PathTraceResult | null> => {
    const cacheKey = `${startNodeId}->${endNodeId}`;
    
    // Check cache first
    if (pathCache.current.has(cacheKey)) {
      return pathCache.current.get(cacheKey)!;
    }

    const { nodes, edges } = state.data;
    
    // Build adjacency map
    const adjacencyMap = new Map<string, { nodeId: string; edge: ComponentEdge }[]>();
    nodes.forEach(node => adjacencyMap.set(node.id, []));
    
    edges.forEach(edge => {
      adjacencyMap.get(edge.source)?.push({ nodeId: edge.target, edge });
      // For undirected edges, add reverse connection
      if (edge.type !== 'inhibition') {
        adjacencyMap.get(edge.target)?.push({ nodeId: edge.source, edge });
      }
    });

    // Dijkstra's algorithm for shortest path
    const distances = new Map<string, number>();
    const previous = new Map<string, { nodeId: string; edge: ComponentEdge } | null>();
    const unvisited = new Set(nodes.map(n => n.id));

    // Initialize distances
    nodes.forEach(node => {
      distances.set(node.id, node.id === startNodeId ? 0 : Infinity);
      previous.set(node.id, null);
    });

    while (unvisited.size > 0) {
      // Find unvisited node with minimum distance
      let currentNode = '';
      let minDistance = Infinity;
      
      unvisited.forEach(nodeId => {
        const distance = distances.get(nodeId) || Infinity;
        if (distance < minDistance) {
          minDistance = distance;
          currentNode = nodeId;
        }
      });

      if (currentNode === '' || minDistance === Infinity) break;
      
      unvisited.delete(currentNode);

      // If we reached the target, break early
      if (currentNode === endNodeId) break;

      // Update distances to neighbors
      const neighbors = adjacencyMap.get(currentNode) || [];
      neighbors.forEach(({ nodeId, edge }) => {
        if (!unvisited.has(nodeId)) return;

        const edgeWeight = edge.weight || 1;
        const latency = edge.latency || 0;
        const weight = edgeWeight + latency * 0.1; // Factor in latency

        const altDistance = (distances.get(currentNode) || 0) + weight;
        if (altDistance < (distances.get(nodeId) || Infinity)) {
          distances.set(nodeId, altDistance);
          previous.set(nodeId, { nodeId: currentNode, edge });
        }
      });
    }

    // Reconstruct path
    if (!previous.has(endNodeId) || previous.get(endNodeId) === null) {
      return null; // No path found
    }

    const path: ComponentNode[] = [];
    const pathEdges: ComponentEdge[] = [];
    let current = endNodeId;
    let totalLatency = 0;
    const bottlenecks: string[] = [];

    while (current !== startNodeId) {
      const node = nodes.find(n => n.id === current);
      if (!node) break;
      
      path.unshift(node);
      
      const prev = previous.get(current);
      if (!prev) break;
      
      pathEdges.unshift(prev.edge);
      totalLatency += prev.edge.latency || 0;
      
      // Identify bottlenecks (high latency or congested edges)
      if ((prev.edge.latency || 0) > 100 || prev.edge.status === 'congested') {
        bottlenecks.push(prev.edge.id);
      }
      
      current = prev.nodeId;
    }

    // Add start node
    const startNode = nodes.find(n => n.id === startNodeId);
    if (startNode) path.unshift(startNode);

    const result: PathTraceResult = {
      path,
      edges: pathEdges,
      totalLatency,
      bottlenecks,
      pathEfficiency: pathEdges.length > 0 ? 1 / (totalLatency / pathEdges.length) : 0
    };

    // Cache the result
    pathCache.current.set(cacheKey, result);
    
    return result;
  }, [state.data]);

  // Actions
  const actions = useMemo<VisualizationActions>(() => ({
    updateData: (data: Partial<ArchitectureData>) => {
      setState(prev => ({
        ...prev,
        data: { ...prev.data, ...data }
      }));
    },

    addNode: (node: ComponentNode) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          nodes: [...prev.data.nodes, node],
          metadata: {
            ...prev.data.metadata,
            totalComponents: prev.data.nodes.length + 1
          }
        }
      }));
    },

    removeNode: (nodeId: string) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          nodes: prev.data.nodes.filter(n => n.id !== nodeId),
          edges: prev.data.edges.filter(e => e.source !== nodeId && e.target !== nodeId),
          metadata: {
            ...prev.data.metadata,
            totalComponents: prev.data.nodes.length - 1
          }
        },
        selectedNodes: prev.selectedNodes.filter(id => id !== nodeId)
      }));
    },

    updateNode: (nodeId: string, updates: Partial<ComponentNode>) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          nodes: prev.data.nodes.map(node =>
            node.id === nodeId ? { ...node, ...updates } : node
          )
        }
      }));
    },

    addEdge: (edge: ComponentEdge) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          edges: [...prev.data.edges, edge],
          metadata: {
            ...prev.data.metadata,
            activeConnections: prev.data.edges.length + 1
          }
        }
      }));
    },

    removeEdge: (edgeId: string) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          edges: prev.data.edges.filter(e => e.id !== edgeId),
          metadata: {
            ...prev.data.metadata,
            activeConnections: prev.data.edges.length - 1
          }
        },
        selectedEdges: prev.selectedEdges.filter(id => id !== edgeId)
      }));
    },

    updateEdge: (edgeId: string, updates: Partial<ComponentEdge>) => {
      setState(prev => ({
        ...prev,
        data: {
          ...prev.data,
          edges: prev.data.edges.map(edge =>
            edge.id === edgeId ? { ...edge, ...updates } : edge
          )
        }
      }));
    },

    selectNode: (nodeId: string, multiSelect = false) => {
      setState(prev => ({
        ...prev,
        selectedNodes: multiSelect 
          ? prev.selectedNodes.includes(nodeId)
            ? prev.selectedNodes.filter(id => id !== nodeId)
            : [...prev.selectedNodes, nodeId]
          : [nodeId],
        selectedEdges: multiSelect ? prev.selectedEdges : []
      }));
    },

    selectEdge: (edgeId: string, multiSelect = false) => {
      setState(prev => ({
        ...prev,
        selectedEdges: multiSelect
          ? prev.selectedEdges.includes(edgeId)
            ? prev.selectedEdges.filter(id => id !== edgeId)
            : [...prev.selectedEdges, edgeId]
          : [edgeId],
        selectedNodes: multiSelect ? prev.selectedNodes : []
      }));
    },

    clearSelection: () => {
      setState(prev => ({
        ...prev,
        selectedNodes: [],
        selectedEdges: []
      }));
    },

    tracePathBetween: async (startNodeId: string, endNodeId: string) => {
      const result = await findPath(startNodeId, endNodeId);
      setState(prev => ({
        ...prev,
        pathTrace: result
      }));
      return result;
    },

    clearPathTrace: () => {
      setState(prev => ({
        ...prev,
        pathTrace: null
      }));
    },

    drillDownInto: (nodeId: string) => {
      setState(prev => ({
        ...prev,
        drillDownHistory: [...prev.drillDownHistory, nodeId]
      }));
    },

    drillDownBack: () => {
      setState(prev => ({
        ...prev,
        drillDownHistory: prev.drillDownHistory.slice(0, -1)
      }));
    },

    resetDrillDown: () => {
      setState(prev => ({
        ...prev,
        drillDownHistory: []
      }));
    },

    startRealTimeUpdates: () => {
      if (options.dataSource.type === 'websocket') {
        connectWebSocket();
      } else if (options.dataSource.type === 'polling') {
        startPolling();
      }
    },

    stopRealTimeUpdates: () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    },

    addRealTimeUpdate: (update: RealTimeUpdate) => {
      setState(prev => {
        const newUpdates = [...prev.realTimeUpdates, update];
        if (newUpdates.length > (options.maxRealTimeBufferSize || 100)) {
          newUpdates.splice(0, newUpdates.length - (options.maxRealTimeBufferSize || 100));
        }
        return { ...prev, realTimeUpdates: newUpdates };
      });
    },

    clearError: () => {
      setState(prev => ({ ...prev, error: null }));
    },

    retryConnection: () => {
      reconnectAttempts.current = 0;
      if (options.dataSource.type === 'websocket') {
        connectWebSocket();
      } else if (options.dataSource.type === 'polling') {
        startPolling();
      }
    }
  }), [findPath, connectWebSocket, startPolling, options.dataSource.type, options.maxRealTimeBufferSize]);

  // Initialize data source on mount
  useEffect(() => {
    if (options.dataSource.type === 'static' && options.dataSource.static) {
      setState(prev => ({
        ...prev,
        data: options.dataSource.static!.data
      }));
    } else {
      actions.startRealTimeUpdates();
    }

    return () => {
      actions.stopRealTimeUpdates();
      
      // Cleanup timeouts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, []);

  // Clear path cache when data changes significantly
  useEffect(() => {
    pathCache.current.clear();
  }, [state.data.nodes.length, state.data.edges.length]);

  return [state, actions];
};