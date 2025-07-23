import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { message } from 'antd';
import { LLMKGVisualizationConfig, LLMKGContext, ConnectionStatus, VisualizationUpdate } from '@/types';

// Mock implementations for development - replace with actual implementations
class MockMCPClient {
  private connected = false;
  private callbacks: Map<string, Function[]> = new Map();

  async connect() {
    this.connected = true;
    this.emit('connected');
  }

  async disconnect() {
    this.connected = false;
    this.emit('disconnected');
  }

  async request(method: string, params: any) {
    // Mock responses for different methods
    switch (method) {
      case 'visualization/getState':
        return {
          cognitive: { layers: [], patterns: [], attention: null },
          sdr: { activeSDRs: [], operations: [], overlaps: [] },
          graph: { entities: [], relations: [], modifications: [] }
        };
      case 'version/getGitInfo':
        return {
          branch: 'main',
          commit: 'abc123def456',
          shortCommit: 'abc123d',
          author: 'Developer',
          email: 'dev@llmkg.com',
          message: 'feat: implement unified visualization system',
          timestamp: new Date().toISOString(),
          isDirty: false,
          tags: ['v1.0.0']
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
  }

  private emit(event: string, data?: any) {
    const callbacks = this.callbacks.get(event) || [];
    callbacks.forEach(callback => callback(data));
  }

  isConnected() {
    return this.connected;
  }
}

class MockSDRProcessor {
  getState() {
    return {
      totalSDRs: 1000,
      activeSDRs: 150,
      averageSparsity: 0.02,
      compressionRatio: 0.8
    };
  }

  async encode(text: string) {
    return {
      bits: new Array(2048).fill(0).map(() => Math.random() < 0.02 ? 1 : 0),
      sparsity: 0.02,
      hash: text.length.toString()
    };
  }

  overlap(sdr1: any, sdr2: any) {
    return Math.random() * 0.5;
  }
}

class MockCognitiveEngine {
  getCurrentState() {
    return {
      layers: [
        { name: 'subcortical', activation: Math.random() },
        { name: 'cortical', activation: Math.random() },
        { name: 'thalamic', activation: Math.random() }
      ],
      patterns: [
        { type: 'convergent', strength: Math.random() },
        { type: 'divergent', strength: Math.random() }
      ],
      attention: { focus: Math.random(), scope: Math.random() }
    };
  }

  subscribe(event: string, callback: Function) {
    // Mock subscription
    const interval = setInterval(() => {
      callback(this.getCurrentState());
    }, 1000);
    return () => clearInterval(interval);
  }
}

class MockKnowledgeGraph {
  getState() {
    return {
      entityCount: 500,
      relationCount: 1200,
      avgConnectivity: 2.4,
      clusteringCoefficient: 0.3
    };
  }

  async getEntities() {
    return Array.from({ length: 10 }, (_, i) => ({
      id: `entity_${i}`,
      type: 'concept',
      data: { name: `Concept ${i}` }
    }));
  }

  async getRelations(filter?: any) {
    return Array.from({ length: 20 }, (_, i) => ({
      id: `relation_${i}`,
      source: `entity_${i % 10}`,
      target: `entity_${(i + 1) % 10}`,
      type: 'related_to'
    }));
  }
}

const LLMKGVisualizationContext = createContext<LLMKGContext | null>(null);

export const LLMKGVisualizationProvider: React.FC<{
  config: LLMKGVisualizationConfig;
  children: React.ReactNode;
}> = ({ config, children }) => {
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
  });

  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(async () => {
    try {
      setContext(prev => ({ ...prev, connectionStatus: 'connecting', error: null }));

      // Initialize MCP Client (using mock for now)
      const mcpClient = new MockMCPClient();
      await mcpClient.connect();

      // Initialize other components
      const sdrProcessor = new MockSDRProcessor();
      const cognitiveEngine = new MockCognitiveEngine();
      const knowledgeGraph = new MockKnowledgeGraph();

      setContext(prev => ({
        ...prev,
        mcpClient,
        sdrProcessor,
        cognitiveEngine,
        knowledgeGraph,
        connected: true,
        connectionStatus: 'connected',
        lastUpdate: Date.now(),
      }));

      reconnectAttemptsRef.current = 0;
      message.success('Connected to LLMKG system');

    } catch (error) {
      const errorObj = error as Error;
      setContext(prev => ({
        ...prev,
        error: errorObj,
        connected: false,
        connectionStatus: 'error',
      }));

      message.error(`Connection failed: ${errorObj.message}`);

      // Handle reconnection
      if (config.mcp.reconnect?.enabled && reconnectAttemptsRef.current < (config.mcp.reconnect.maxAttempts || 5)) {
        reconnectAttemptsRef.current++;
        setContext(prev => ({ ...prev, connectionStatus: 'reconnecting' }));
        
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, config.mcp.reconnect.delay || 5000);
      }
    }
  }, [config]);

  const disconnect = useCallback(async () => {
    try {
      if (context.mcpClient && 'disconnect' in context.mcpClient) {
        await context.mcpClient.disconnect();
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
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
      }));

      message.info('Disconnected from LLMKG system');
    } catch (error) {
      console.error('Error during disconnect:', error);
    }
  }, [context.mcpClient]);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Subscribe to updates
  useEffect(() => {
    if (!context.mcpClient || !context.connected) return;

    const unsubscribe = context.mcpClient.subscribe('update', (update: VisualizationUpdate) => {
      setContext(prev => ({ ...prev, lastUpdate: Date.now() }));
    });

    return unsubscribe;
  }, [context.mcpClient, context.connected]);

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

// Connection status hook
export const useConnectionStatus = () => {
  const { connected, connectionStatus, error } = useLLMKG();
  return { connected, connectionStatus, error };
};

// Real-time data hook
export const useRealTimeData = <T,>(
  dataSource: string,
  defaultValue: T,
  updateInterval = 1000
): [T, boolean] => {
  const { mcpClient, connected } = useLLMKG();
  const [data, setData] = useState<T>(defaultValue);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!mcpClient || !connected) return;

    const fetchData = async () => {
      try {
        setLoading(true);
        const result = await mcpClient.request(`data/${dataSource}`, {});
        setData(result);
      } catch (error) {
        console.error(`Failed to fetch ${dataSource}:`, error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, updateInterval);

    return () => clearInterval(interval);
  }, [mcpClient, connected, dataSource, updateInterval]);

  return [data, loading];
};

// Component registration system
const componentRegistry = new Map<string, any>();

export const registerComponent = (id: string, component: any, metadata?: any) => {
  componentRegistry.set(id, { component, metadata });
};

export const getComponent = (id: string) => {
  return componentRegistry.get(id);
};

export const getRegisteredComponents = () => {
  return Array.from(componentRegistry.entries()).map(([id, { component, metadata }]) => ({
    id,
    component,
    metadata,
  }));
};