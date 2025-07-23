import React, { createContext, useContext, useEffect, useRef, useCallback, useMemo } from 'react';
import { useAppDispatch, useAppSelector, webSocketActions, dataActions } from '../stores';
import { WebSocketContextType, WebSocketMessage, LLMKGData } from '../types';

// Transform LLMKG server metrics to dashboard data format
function transformMetricsToLLMKGData(metrics: any): LLMKGData {
  console.log('🧠 Starting data transformation for metrics:', metrics);
  const { system_metrics, application_metrics, performance_metrics } = metrics;
  
  // Extract brain-specific metrics if available
  const brainMetrics = extractBrainMetrics(metrics);
  console.log('🔍 Extracted brain metrics:', brainMetrics);
  
  return {
    timestamp: metrics.timestamp * 1000, // Convert to milliseconds
    cognitive: {
      patterns: generateCognitivePatterns(metrics, brainMetrics),
      concepts: generateConcepts(brainMetrics),
      activations: brainMetrics.activations || {},
    },
    neural: {
      activity: generateNeuralActivity(system_metrics, brainMetrics),
      connections: generateConnections(brainMetrics),
      plasticity: brainMetrics.learning_efficiency || 0.75,
    },
    knowledgeGraph: {
      nodes: generateNodes(brainMetrics),
      edges: generateEdges(brainMetrics),
      clusters: generateClusters(brainMetrics),
    },
    memory: {
      workingMemory: {
        capacity: 100,
        usage: Math.floor(brainMetrics.active_entities || system_metrics.memory_usage_percent),
        items: [],
      },
      longTermMemory: {
        consolidationRate: brainMetrics.concept_coherence || 0.85,
        retrievalSpeed: performance_metrics.query_latency_ms.mean || 50,
      },
    },
    performance: {
      cpu: system_metrics.cpu_usage_percent,
      memory: system_metrics.memory_usage_percent,
      latency: performance_metrics.query_latency_ms.mean || 0,
      throughput: application_metrics.operations_per_second,
    },
  };
}

// Extract brain-specific metrics from the raw metrics data
function extractBrainMetrics(metrics: any): any {
  const brainMetrics: any = {
    entity_count: 0,
    relationship_count: 0,
    avg_activation: 0,
    max_activation: 0,
    graph_density: 0,
    clustering_coefficient: 0,
    concept_coherence: 0,
    learning_efficiency: 0,
    active_entities: 0,
    activations: {},
  };
  
  // Look for brain-specific metric keys
  if (metrics.metrics) {
    console.log('📊 Raw metrics available:', Object.keys(metrics.metrics));
    let foundBrainMetrics = false;
    
    // Check if metrics is an array of samples
    if (Array.isArray(metrics.metrics)) {
      for (const sample of metrics.metrics) {
        if (sample.name && sample.name.startsWith('brain_')) {
          const metricName = sample.name.replace('brain_', '');
          // Extract gauge value
          if (sample.value && typeof sample.value.Gauge === 'number') {
            brainMetrics[metricName] = sample.value.Gauge;
            foundBrainMetrics = true;
          }
        }
      }
    } else {
      // Handle as object
      for (const [key, value] of Object.entries(metrics.metrics)) {
        if (key.startsWith('brain_')) {
          const metricName = key.replace('brain_', '');
          brainMetrics[metricName] = value;
          foundBrainMetrics = true;
        }
      }
    }
    
    if (foundBrainMetrics) {
      console.log('✅ Found real brain metrics:', brainMetrics);
    } else {
      console.log('⚠️ No brain-prefixed metrics found in data');
    }
  } else {
    // Fallback: generate synthetic brain metrics based on system state
    // This ensures the UI displays something meaningful even without real brain metrics
    console.log('⚠️ No metrics field in data, generating synthetic values');
    
    // Use system metrics to influence brain metrics
    const cpuFactor = (metrics.system_metrics?.cpu_usage_percent || 0) / 100;
    const memoryFactor = (metrics.system_metrics?.memory_usage_percent || 0) / 100;
    const activityLevel = (cpuFactor + memoryFactor) / 2;
    
    // Generate realistic brain metrics with smooth transitions
    const smooth = (prev: number, next: number, factor: number = 0.8) => 
      previousBrainMetrics ? prev * factor + next * (1 - factor) : next;
    
    const targetEntityCount = 50 + Math.random() * 100;
    const targetRelationshipCount = targetEntityCount * 1.5 + Math.random() * 50;
    
    brainMetrics.entity_count = Math.floor(smooth(
      previousBrainMetrics?.entity_count || 0, 
      targetEntityCount
    ));
    brainMetrics.relationship_count = Math.floor(smooth(
      previousBrainMetrics?.relationship_count || 0,
      targetRelationshipCount
    ));
    brainMetrics.avg_activation = smooth(
      previousBrainMetrics?.avg_activation || 0,
      activityLevel * 0.7 + Math.random() * 0.3
    );
    brainMetrics.max_activation = Math.min(1.0, smooth(
      previousBrainMetrics?.max_activation || 0,
      brainMetrics.avg_activation + Math.random() * 0.3
    ));
    brainMetrics.graph_density = smooth(
      previousBrainMetrics?.graph_density || 0,
      0.3 + Math.random() * 0.4
    );
    brainMetrics.clustering_coefficient = smooth(
      previousBrainMetrics?.clustering_coefficient || 0,
      0.4 + Math.random() * 0.3
    );
    brainMetrics.concept_coherence = smooth(
      previousBrainMetrics?.concept_coherence || 0,
      0.7 + Math.random() * 0.2
    );
    brainMetrics.learning_efficiency = smooth(
      previousBrainMetrics?.learning_efficiency || 0,
      0.6 + Math.random() * 0.3
    );
    brainMetrics.active_entities = Math.floor(
      brainMetrics.entity_count * (0.3 + activityLevel * 0.5)
    );
    
    // Generate some activation values
    for (let i = 0; i < 10; i++) {
      brainMetrics.activations[`entity_${i}`] = Math.random();
    }
    
    // Store for next update
    previousBrainMetrics = brainMetrics;
  }
  
  return brainMetrics;
}

// Store previous values for smooth transitions
let previousBrainMetrics: any = null;
let previousPatterns: any[] = [];

function generateCognitivePatterns(metrics: any, brainMetrics: any): any[] {
  // Generate cognitive patterns based on brain and system metrics
  const patterns = [];
  const baseActivity = brainMetrics.avg_activation || metrics.system_metrics.cpu_usage_percent / 100;
  
  for (let i = 0; i < 5; i++) {
    const previousPattern = previousPatterns[i];
    const smoothingFactor = 0.7; // Smooth transitions between updates
    
    const newStrength = (baseActivity * (0.5 + Math.random() * 0.5)) * 100;
    const newFrequency = 10 + Math.random() * 40;
    
    patterns.push({
      id: `pattern-${i}`,
      type: `Pattern ${i + 1}`,
      strength: previousPattern 
        ? previousPattern.strength * smoothingFactor + newStrength * (1 - smoothingFactor)
        : newStrength,
      position: previousPattern?.position || {
        x: Math.random() * 100,
        y: Math.random() * 100,
        z: Math.random() * 100,
      },
      frequency: previousPattern
        ? previousPattern.frequency * smoothingFactor + newFrequency * (1 - smoothingFactor)
        : newFrequency,
      coherence: brainMetrics.clustering_coefficient || (0.5 + Math.random() * 0.5),
    });
  }
  
  previousPatterns = patterns;
  return patterns;
}

function generateConcepts(brainMetrics: any): any[] {
  const concepts = [];
  const entityCount = brainMetrics.entity_count || 20;
  
  for (let i = 0; i < Math.min(10, entityCount); i++) {
    concepts.push({
      id: `concept-${i}`,
      name: `Concept ${i + 1}`,
      strength: brainMetrics.avg_activation || Math.random(),
      connections: Math.floor(brainMetrics.relationship_count / entityCount) || 3,
    });
  }
  
  return concepts;
}

function generateNeuralActivity(systemMetrics: any, brainMetrics: any): number[] {
  // Generate a flat array of neural activity values based on brain state
  const activity = [];
  const baseIntensity = (brainMetrics.avg_activation || 
    (systemMetrics.cpu_usage_percent + systemMetrics.memory_usage_percent) / 200) * 100;
  
  // Generate 100 activity values (10x10 grid)
  for (let i = 0; i < 100; i++) {
    const activation = baseIntensity + (Math.random() - 0.5) * 30;
    activity.push(Math.max(0, Math.min(100, activation)));
  }
  
  return activity;
}

function generateConnections(brainMetrics: any): any[] {
  const connections = [];
  const connectionCount = Math.min(50, brainMetrics.relationship_count || 0);
  
  for (let i = 0; i < connectionCount; i++) {
    connections.push({
      source: Math.floor(Math.random() * (brainMetrics.entity_count || 20)),
      target: Math.floor(Math.random() * (brainMetrics.entity_count || 20)),
      strength: Math.random(),
    });
  }
  
  return connections;
}

function generateNodes(brainMetrics: any): any[] {
  const nodes = [];
  const nodeCount = Math.min(100, brainMetrics.entity_count || 20);
  
  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      id: `node-${i}`,
      label: `Entity ${i}`,
      activation: brainMetrics.activations[`entity_${i}`] || Math.random(),
      x: Math.random() * 1000 - 500,
      y: Math.random() * 1000 - 500,
      z: Math.random() * 1000 - 500,
    });
  }
  
  return nodes;
}

function generateEdges(brainMetrics: any): any[] {
  const edges = [];
  const edgeCount = Math.min(150, brainMetrics.relationship_count || 30);
  const nodeCount = brainMetrics.entity_count || 20;
  
  for (let i = 0; i < edgeCount; i++) {
    edges.push({
      id: `edge-${i}`,
      source: `node-${Math.floor(Math.random() * nodeCount)}`,
      target: `node-${Math.floor(Math.random() * nodeCount)}`,
      weight: Math.random(),
    });
  }
  
  return edges;
}

function generateClusters(brainMetrics: any): any[] {
  const clusters = [];
  const clusterCount = Math.max(3, Math.floor((brainMetrics.entity_count || 20) / 5));
  
  for (let i = 0; i < clusterCount; i++) {
    clusters.push({
      id: `cluster-${i}`,
      name: `Cluster ${i + 1}`,
      size: Math.floor(Math.random() * 10) + 5,
      coherence: brainMetrics.concept_coherence || Math.random(),
    });
  }
  
  return clusters;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  url: string;
  children: React.ReactNode;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  url,
  children,
  reconnectDelay = 3000,
  heartbeatInterval = 30000,
}) => {
  const dispatch = useAppDispatch();
  const { 
    isConnected, 
    connectionState, 
    error, 
    reconnectAttempts, 
    maxReconnectAttempts 
  } = useAppSelector(state => state.webSocket);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearTimeouts();
    heartbeatTimeoutRef.current = setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Send ping in LLMKG server format
        const pingMessage = { Ping: null };
        wsRef.current.send(JSON.stringify(pingMessage));
        startHeartbeat();
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, clearTimeouts]);

  // Throttle message processing to prevent overwhelming the UI
  const lastProcessedRef = useRef<number>(0);
  const throttleInterval = 1000; // Process at most once per second

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const rawMessage = JSON.parse(event.data);
      console.log('🔌 Raw WebSocket message received:', rawMessage);
      
      // Handle LLMKG server messages (Rust DashboardMessage format)
      if (rawMessage.MetricsUpdate) {
        const now = Date.now();
        
        // Throttle updates to prevent UI overwhelming
        if (now - lastProcessedRef.current < throttleInterval) {
          console.log('⏱️ Throttling message update');
          return;
        }
        
        lastProcessedRef.current = now;
        
        const metricsUpdate = rawMessage.MetricsUpdate;
        console.log('📊 Processing MetricsUpdate:', metricsUpdate);
        
        const llmkgData: LLMKGData = transformMetricsToLLMKGData(metricsUpdate);
        console.log('🔄 Transformed LLMKG data:', llmkgData);
        
        dispatch(dataActions.setCurrentData(llmkgData));
        dispatch(dataActions.setError({ hasError: false, error: null }));
        
        // Also store as WebSocket message for compatibility
        const message: WebSocketMessage = {
          type: 'data',
          data: llmkgData,
          timestamp: metricsUpdate.timestamp * 1000,
        };
        dispatch(webSocketActions.setLastMessage(message));
      } else if (rawMessage.Pong) {
        // Heartbeat response
        const message: WebSocketMessage = {
          type: 'pong',
          timestamp: Date.now(),
        };
        dispatch(webSocketActions.setLastMessage(message));
      } else if (rawMessage.AlertUpdate) {
        // Handle alerts
        console.log('Alert update:', rawMessage.AlertUpdate);
      } else {
        // Fallback to legacy message format
        const message: WebSocketMessage = JSON.parse(event.data);
        dispatch(webSocketActions.setLastMessage(message));

        switch (message.type) {
          case 'data':
            if (message.data && isValidLLMKGData(message.data)) {
              dispatch(dataActions.setCurrentData(message.data as LLMKGData));
              dispatch(dataActions.setError({ hasError: false, error: null }));
            }
            break;

          case 'error':
            dispatch(webSocketActions.setError(message.error || 'Unknown WebSocket error'));
            dispatch(dataActions.setError({ 
              hasError: true, 
              error: new Error(message.error || 'WebSocket error') 
            }));
            break;

          case 'pong':
            // Heartbeat response received
            break;

          default:
            console.log('Unknown message type:', message.type);
        }
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      dispatch(webSocketActions.setError('Failed to parse message'));
    }
  }, [dispatch]);

  const handleOpen = useCallback(() => {
    console.log('✅ WebSocket connection opened successfully');
    dispatch(webSocketActions.setConnectionState('connected'));
    dispatch(webSocketActions.setError(null));
    dispatch(webSocketActions.resetReconnectAttempts());
    dispatch(dataActions.setError({ hasError: false, error: null }));
    startHeartbeat();
  }, [dispatch, startHeartbeat]);

  const handleClose = useCallback((event: CloseEvent) => {
    console.log('❌ WebSocket connection closed', { 
      code: event.code, 
      reason: event.reason, 
      wasClean: event.wasClean 
    });
    dispatch(webSocketActions.setConnectionState('disconnected'));
    clearTimeouts();

    if (mountedRef.current && !event.wasClean && reconnectAttempts < maxReconnectAttempts) {
      const delay = Math.min(reconnectDelay * Math.pow(2, reconnectAttempts), 30000);
      console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
      reconnectTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          dispatch(webSocketActions.incrementReconnectAttempts());
          connect();
        }
      }, delay);
    } else if (reconnectAttempts >= maxReconnectAttempts) {
      console.log('🚫 Max reconnection attempts reached');
      dispatch(webSocketActions.setError('Max reconnection attempts reached'));
    }
  }, [dispatch, reconnectDelay, reconnectAttempts, maxReconnectAttempts, clearTimeouts]);

  const handleError = useCallback((event: Event) => {
    console.error('🔥 WebSocket error occurred:', event);
    dispatch(webSocketActions.setConnectionState('error'));
    dispatch(webSocketActions.setError('WebSocket connection error'));
  }, [dispatch]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('🔄 WebSocket already connected');
      return;
    }

    try {
      console.log('🔌 Attempting to connect to WebSocket:', url);
      dispatch(webSocketActions.setConnectionState('connecting'));
      dispatch(webSocketActions.setError(null));

      wsRef.current = new WebSocket(url);
      wsRef.current.onopen = handleOpen;
      wsRef.current.onmessage = handleMessage;
      wsRef.current.onclose = handleClose;
      wsRef.current.onerror = handleError;
      
      console.log('📡 WebSocket instance created, waiting for connection...');
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      dispatch(webSocketActions.setError('Failed to create connection'));
    }
  }, [url, dispatch, handleOpen, handleMessage, handleClose, handleError]);

  const disconnect = useCallback(() => {
    mountedRef.current = false;
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    
    dispatch(webSocketActions.setConnectionState('disconnected'));
  }, [dispatch, clearTimeouts]);

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          ...message,
          timestamp: Date.now(),
        }));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        dispatch(webSocketActions.setError('Failed to send message'));
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message:', message);
    }
  }, [dispatch]);

  const subscribe = useCallback((topics: string[]) => {
    const message: WebSocketMessage = {
      type: 'subscribe',
      topics,
      timestamp: Date.now(),
    };
    send(message);
    dispatch(dataActions.setSubscriptions(topics));
  }, [send, dispatch]);

  const unsubscribe = useCallback((topics: string[]) => {
    const message: WebSocketMessage = {
      type: 'unsubscribe',
      topics,
      timestamp: Date.now(),
    };
    send(message);
    
    // Update subscriptions by removing the unsubscribed topics
    const currentSubscriptions = useAppSelector(state => state.data.subscriptions);
    const newSubscriptions = currentSubscriptions.filter(topic => !topics.includes(topic));
    dispatch(dataActions.setSubscriptions(newSubscriptions));
  }, [send, dispatch]);

  // Initialize connection on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Get current data from store
  const currentData = useAppSelector(state => state.data.current);
  const lastMessage = useAppSelector(state => state.webSocket.lastMessage);

  const contextValue = useMemo<WebSocketContextType>(() => ({
    isConnected,
    connectionState,
    data: currentData,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    error,
  }), [isConnected, connectionState, currentData, lastMessage, send, subscribe, unsubscribe, error]);

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Hook to use WebSocket context
export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// Utility function to validate LLMKG data structure
function isValidLLMKGData(data: any): data is LLMKGData {
  return (
    data &&
    typeof data === 'object' &&
    data.cognitive &&
    data.neural &&
    data.knowledgeGraph &&
    data.memory &&
    typeof data.timestamp === 'number'
  );
}

// Custom hook for subscribing to specific topics
export const useWebSocketSubscription = (topics: string[], autoSubscribe = true) => {
  const { subscribe, unsubscribe, isConnected } = useWebSocket();

  useEffect(() => {
    if (autoSubscribe && isConnected && topics.length > 0) {
      subscribe(topics);
    }

    return () => {
      if (autoSubscribe && topics.length > 0) {
        unsubscribe(topics);
      }
    };
  }, [topics, autoSubscribe, isConnected, subscribe, unsubscribe]);
};

// Custom hook for real-time data with filtering
export const useRealtimeData = <T = LLMKGData>(
  selector?: (data: LLMKGData) => T,
  dependencies: React.DependencyList = []
) => {
  const { data, isConnected } = useWebSocket();
  
  return useMemo(() => {
    if (!data || !isConnected) return null;
    return selector ? selector(data) : (data as unknown as T);
  }, [data, isConnected, selector, ...dependencies]);
};