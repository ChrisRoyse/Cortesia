import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { message } from 'antd';
import { useAppDispatch } from '../stores';
import { setConnectionStatus, addAlert } from '../stores/systemSlice';
import { updateMetrics } from '../stores/performanceSlice';

// Enhanced types for the visualization core
export interface LLMKGVisualizationConfig {
  mcp: {
    endpoint: string;
    protocol: 'ws' | 'http';
    authentication?: {
      type: 'bearer' | 'api-key';
      token: string;
    };
    reconnect?: {
      enabled: boolean;
      maxAttempts: number;
      delay: number;
    };
  };
  visualization: {
    theme: 'light' | 'dark' | 'auto';
    updateInterval: number;
    maxDataPoints: number;
    enableAnimations: boolean;
    enableDebugMode: boolean;
  };
  performance: {
    enableProfiling: boolean;
    sampleRate: number;
    maxMemoryUsage: number;
    enableLazyLoading: boolean;
  };
  features: {
    enabledPhases: string[];
    experimentalFeatures: string[];
  };
}

export interface LLMKGContext {
  config: LLMKGVisualizationConfig;
  mcpClient: any | null;
  sdrProcessor: any | null;
  cognitiveEngine: any | null;
  knowledgeGraph: any | null;
  connected: boolean;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';
  error: Error | null;
  lastUpdate: number;
  systemHealth: number;
}

// Enhanced mock implementations with more realistic behavior
class EnhancedMockMCPClient {
  private connected = false;
  private callbacks: Map<string, Function[]> = new Map();
  private heartbeatInterval?: number;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  async connect() {
    this.connected = true;
    this.startHeartbeat();
    this.emit('connected');
    
    // Simulate occasional connection issues
    setTimeout(() => {
      if (Math.random() < 0.1) { // 10% chance of connection issue
        this.simulateConnectionIssue();
      }
    }, 10000 + Math.random() * 20000);
  }

  async disconnect() {
    this.connected = false;
    this.stopHeartbeat();
    this.emit('disconnected');
  }

  private startHeartbeat() {
    this.heartbeatInterval = window.setInterval(() => {
      if (this.connected) {
        this.emit('heartbeat', { timestamp: Date.now() });
      }
    }, 5000);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = undefined;
    }
  }

  private simulateConnectionIssue() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.connected = false;
      this.emit('disconnected');
      this.reconnectAttempts++;
      
      setTimeout(() => {
        this.emit('reconnecting');
        setTimeout(() => {
          this.connected = true;
          this.emit('connected');
          this.reconnectAttempts = 0;
        }, 2000 + Math.random() * 3000);
      }, 1000);
    }
  }

  async request(method: string, params: any) {
    if (!this.connected) {
      throw new Error('Not connected to LLMKG system');
    }

    // Simulate network latency
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));

    // Mock responses for different methods with realistic data
    switch (method) {
      case 'visualization/getState':
        return {
          cognitive: {
            layers: [
              { name: 'subcortical', activation: Math.random(), neuronCount: 1000 },
              { name: 'cortical', activation: Math.random(), neuronCount: 5000 },
              { name: 'thalamic', activation: Math.random(), neuronCount: 800 }
            ],
            patterns: [
              { type: 'convergent', strength: Math.random(), activeNodes: 150 },
              { type: 'divergent', strength: Math.random(), activeNodes: 200 }
            ],
            attention: { 
              focus: Math.random(), 
              scope: Math.random(),
              targetRegion: 'cortical'
            }
          },
          sdr: {
            activeSDRs: Array.from({ length: 10 }, (_, i) => ({
              id: `sdr_${i}`,
              sparsity: 0.02 + Math.random() * 0.03,
              bits: 2048,
              activeBits: Math.floor(40 + Math.random() * 20)
            })),
            operations: [
              { type: 'encode', duration: Math.random() * 10, success: true },
              { type: 'overlap', duration: Math.random() * 5, success: true }
            ],
            overlaps: [
              ['sdr_0', 'sdr_1', Math.random() * 0.5],
              ['sdr_1', 'sdr_2', Math.random() * 0.3]
            ]
          },
          graph: {
            entities: Array.from({ length: 500 }, (_, i) => ({
              id: `entity_${i}`,
              type: 'concept',
              connections: Math.floor(Math.random() * 10)
            })),
            relations: Array.from({ length: 1200 }, (_, i) => ({
              id: `rel_${i}`,
              source: `entity_${Math.floor(Math.random() * 500)}`,
              target: `entity_${Math.floor(Math.random() * 500)}`,
              weight: Math.random()
            })),
            modifications: []
          }
        };

      case 'system/metrics':
        return {
          memoryUsage: 50 + Math.random() * 40,
          cpuUsage: 20 + Math.random() * 30,
          networkLatency: 50 + Math.random() * 100,
          throughput: 100 + Math.random() * 50,
          errorRate: Math.random() * 2,
          uptime: Date.now() - (7 * 24 * 60 * 60 * 1000) + Math.random() * 1000000
        };

      case 'version/getGitInfo':
        return {
          branch: 'main',
          commit: 'abc123def456789',
          shortCommit: 'abc123d',
          author: 'LLMKG Developer',
          email: 'dev@llmkg.com',
          message: 'feat(phase11): implement production-ready dashboard with performance optimizations',
          timestamp: new Date().toISOString(),
          isDirty: false,
          tags: ['v2.0.0', 'production-ready']
        };

      case 'performance/getMetrics':
        return {
          system: {
            cpuUsage: 20 + Math.random() * 30,
            memoryUsage: 50 + Math.random() * 40,
            diskUsage: 30 + Math.random() * 20,
            networkLatency: 50 + Math.random() * 100
          },
          cognitive: {
            averageLatency: 10 + Math.random() * 20,
            throughput: 100 + Math.random() * 50,
            errorRate: Math.random() * 2
          },
          components: {
            memoryDashboard: { renderTime: 8 + Math.random() * 8 },
            cognitiveDashboard: { renderTime: 12 + Math.random() * 10 },
            debugDashboard: { renderTime: 15 + Math.random() * 10 }
          }
        };

      default:
        return {};
    }
  }

  subscribe(event: string, callback: Function) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event)!.push(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  private emit(event: string, data?: any) {
    const callbacks = this.callbacks.get(event) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in MCP callback:', error);
      }
    });
  }

  isConnected() {
    return this.connected;
  }
}

class EnhancedMockSDRProcessor {
  private activeSDRs: any[] = [];

  getState() {
    return {
      totalSDRs: 1000 + Math.floor(Math.random() * 500),
      activeSDRs: 150 + Math.floor(Math.random() * 100),
      averageSparsity: 0.02 + Math.random() * 0.01,
      compressionRatio: 0.7 + Math.random() * 0.2,
      memoryUsage: 50 + Math.random() * 30,
      operationsPerSecond: 100 + Math.random() * 50
    };
  }

  async encode(text: string) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 20));
    
    const activeBits = Math.floor(40 + Math.random() * 20);
    return {
      bits: new Array(2048).fill(0).map((_, i) => i < activeBits ? 1 : 0),
      sparsity: activeBits / 2048,
      hash: btoa(text).slice(0, 8),
      processingTime: 10 + Math.random() * 20
    };
  }

  overlap(sdr1: any, sdr2: any) {
    return Math.random() * 0.6; // More realistic overlap range
  }

  getActiveSDRs() {
    return Array.from({ length: 10 }, (_, i) => ({
      id: `sdr_${i}`,
      sparsity: 0.02 + Math.random() * 0.03,
      bits: 2048,
      activeBits: Math.floor(40 + Math.random() * 20),
      lastAccessed: Date.now() - Math.random() * 10000
    }));
  }
}

class EnhancedMockCognitiveEngine {
  private currentState: any = null;

  getCurrentState() {
    this.currentState = {
      layers: [
        { 
          name: 'subcortical', 
          activation: Math.random(),
          neuronCount: 1000,
          activeNeurons: Math.floor(200 + Math.random() * 300)
        },
        { 
          name: 'cortical', 
          activation: Math.random(),
          neuronCount: 5000,
          activeNeurons: Math.floor(800 + Math.random() * 1200)
        },
        { 
          name: 'thalamic', 
          activation: Math.random(),
          neuronCount: 800,
          activeNeurons: Math.floor(100 + Math.random() * 200)
        }
      ],
      patterns: [
        { type: 'convergent', strength: Math.random(), confidence: 0.7 + Math.random() * 0.3 },
        { type: 'divergent', strength: Math.random(), confidence: 0.6 + Math.random() * 0.4 },
        { type: 'lateral', strength: Math.random(), confidence: 0.5 + Math.random() * 0.5 }
      ],
      attention: { 
        focus: Math.random(), 
        scope: Math.random(),
        targetRegion: ['cortical', 'subcortical', 'thalamic'][Math.floor(Math.random() * 3)]
      },
      inhibitionExcitationBalance: 0.4 + Math.random() * 0.2
    };
    
    return this.currentState;
  }

  subscribe(event: string, callback: Function) {
    const interval = setInterval(() => {
      callback(this.getCurrentState());
    }, 1000 + Math.random() * 1000);
    
    return () => clearInterval(interval);
  }
}

class EnhancedMockKnowledgeGraph {
  getState() {
    return {
      entityCount: 500 + Math.floor(Math.random() * 200),
      relationCount: 1200 + Math.floor(Math.random() * 400),
      avgConnectivity: 2.0 + Math.random() * 1.0,
      clusteringCoefficient: 0.2 + Math.random() * 0.2,
      memoryUsage: 30 + Math.random() * 20,
      queryLatency: 5 + Math.random() * 10
    };
  }

  async getEntities() {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
    
    return Array.from({ length: 20 }, (_, i) => ({
      id: `entity_${i}`,
      type: ['concept', 'person', 'place', 'event'][Math.floor(Math.random() * 4)],
      data: { 
        name: `Entity ${i}`,
        confidence: 0.7 + Math.random() * 0.3,
        connections: Math.floor(Math.random() * 15)
      }
    }));
  }

  async getRelations(filter?: any) {
    await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 70));
    
    return Array.from({ length: 30 }, (_, i) => ({
      id: `relation_${i}`,
      source: `entity_${i % 20}`,
      target: `entity_${(i + 1) % 20}`,
      type: ['related_to', 'part_of', 'instance_of', 'causes'][Math.floor(Math.random() * 4)],
      weight: Math.random(),
      confidence: 0.6 + Math.random() * 0.4
    }));
  }
}

const LLMKGVisualizationContext = createContext<LLMKGContext | null>(null);

export const LLMKGVisualizationProvider: React.FC<{
  config: LLMKGVisualizationConfig;
  children: React.ReactNode;
}> = ({ config, children }) => {
  const dispatch = useAppDispatch();
  
  const [context, setContext] = useState<LLMKGContext>({
    config,
    mcpClient: null,
    sdrProcessor: null,
    cognitiveEngine: null,
    knowledgeGraph: null,
    connected: false,
    connectionStatus: 'disconnected',
    error: null,
    lastUpdate: 0,
    systemHealth: 0,
  });

  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const metricsIntervalRef = useRef<number | null>(null);

  const connect = useCallback(async () => {
    try {
      dispatch(setConnectionStatus('connecting'));
      setContext(prev => ({ ...prev, connectionStatus: 'connecting', error: null }));

      // Simulate connection time
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

      // Initialize enhanced mock components
      const mcpClient = new EnhancedMockMCPClient();
      await mcpClient.connect();

      const sdrProcessor = new EnhancedMockSDRProcessor();
      const cognitiveEngine = new EnhancedMockCognitiveEngine();
      const knowledgeGraph = new EnhancedMockKnowledgeGraph();

      // Set up event listeners
      mcpClient.subscribe('connected', () => {
        dispatch(setConnectionStatus('connected'));
        dispatch(addAlert({ type: 'success', message: 'Successfully connected to LLMKG system' }));
      });

      mcpClient.subscribe('disconnected', () => {
        dispatch(setConnectionStatus('disconnected'));
        dispatch(addAlert({ type: 'warning', message: 'Connection to LLMKG system lost' }));
      });

      mcpClient.subscribe('reconnecting', () => {
        dispatch(setConnectionStatus('reconnecting'));
      });

      setContext(prev => ({
        ...prev,
        mcpClient,
        sdrProcessor,
        cognitiveEngine,
        knowledgeGraph,
        connected: true,
        connectionStatus: 'connected',
        lastUpdate: Date.now(),
        systemHealth: 85 + Math.random() * 15, // 85-100% health
      }));

      dispatch(setConnectionStatus('connected'));
      reconnectAttemptsRef.current = 0;
      message.success('Connected to LLMKG system');

      // Start metrics collection
      startMetricsCollection(mcpClient);

    } catch (error) {
      const errorObj = error as Error;
      setContext(prev => ({
        ...prev,
        error: errorObj,
        connected: false,
        connectionStatus: 'error',
        systemHealth: 0,
      }));

      dispatch(setConnectionStatus('error'));
      dispatch(addAlert({ type: 'error', message: `Connection failed: ${errorObj.message}` }));
      message.error(`Connection failed: ${errorObj.message}`);

      // Handle reconnection
      if (config.mcp.reconnect?.enabled && reconnectAttemptsRef.current < (config.mcp.reconnect.maxAttempts || 5)) {
        reconnectAttemptsRef.current++;
        dispatch(setConnectionStatus('reconnecting'));
        setContext(prev => ({ ...prev, connectionStatus: 'reconnecting' }));
        
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, config.mcp.reconnect.delay || 5000);
      }
    }
  }, [config, dispatch]);

  const startMetricsCollection = useCallback((mcpClient: any) => {
    if (metricsIntervalRef.current) {
      clearInterval(metricsIntervalRef.current);
    }

    metricsIntervalRef.current = window.setInterval(async () => {
      try {
        const metrics = await mcpClient.request('performance/getMetrics', {});
        
        dispatch(updateMetrics({
          renderTime: metrics.components?.memoryDashboard?.renderTime || 0,
          memoryUsage: metrics.system?.memoryUsage || 0,
          networkLatency: metrics.system?.networkLatency || 0,
          dataProcessingTime: metrics.cognitive?.averageLatency || 0,
          updateFrequency: metrics.cognitive?.throughput || 0,
          componentCount: Object.keys(metrics.components || {}).length,
        }));

        setContext(prev => ({
          ...prev,
          lastUpdate: Date.now(),
          systemHealth: Math.max(0, Math.min(100, 100 - (metrics.system?.cpuUsage || 0) * 0.5 - (metrics.cognitive?.errorRate || 0) * 10)),
        }));

      } catch (error) {
        console.error('Failed to collect metrics:', error);
      }
    }, config.visualization.updateInterval || 1000);
  }, [dispatch, config.visualization.updateInterval]);

  const disconnect = useCallback(async () => {
    try {
      if (context.mcpClient && 'disconnect' in context.mcpClient) {
        await context.mcpClient.disconnect();
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      if (metricsIntervalRef.current) {
        clearInterval(metricsIntervalRef.current);
        metricsIntervalRef.current = null;
      }

      setContext(prev => ({
        ...prev,
        mcpClient: null,
        sdrProcessor: null,
        cognitiveEngine: null,
        knowledgeGraph: null,
        connected: false,
        connectionStatus: 'disconnected',
        error: null,
        systemHealth: 0,
      }));

      dispatch(setConnectionStatus('disconnected'));
      message.info('Disconnected from LLMKG system');
    } catch (error) {
      console.error('Error during disconnect:', error);
    }
  }, [context.mcpClient, dispatch]);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <LLMKGVisualizationContext.Provider value={context}>
      {children}
    </LLMKGVisualizationContext.Provider>
  );
};

export const useLLMKG = () => {
  const context = useContext(LLMKGVisualizationContext);
  if (!context) {
    throw new Error('useLLMKG must be used within LLMKGVisualizationProvider');
  }
  return context;
};

// Enhanced hooks
export const useConnectionStatus = () => {
  const { connected, connectionStatus, error, systemHealth } = useLLMKG();
  return { connected, connectionStatus, error, systemHealth };
};

export const useRealTimeData = <T,>(
  dataSource: string,
  defaultValue: T,
  updateInterval = 1000
): [T, boolean, Error | null] => {
  const { mcpClient, connected } = useLLMKG();
  const [data, setData] = useState<T>(defaultValue);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!mcpClient || !connected) return;

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await mcpClient.request(`data/${dataSource}`, {});
        setData(result || defaultValue);
      } catch (err) {
        const error = err as Error;
        setError(error);
        console.error(`Failed to fetch ${dataSource}:`, error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, updateInterval);

    return () => clearInterval(interval);
  }, [mcpClient, connected, dataSource, updateInterval, defaultValue]);

  return [data, loading, error];
};