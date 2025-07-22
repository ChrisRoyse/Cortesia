import { useState, useEffect, useCallback, useRef } from 'react';
import { useAppDispatch, useAppSelector } from '../../../stores/hooks';
import { MCPTool } from '../types';
import ToolDiscoveryService from '../services/ToolDiscoveryService';
import ToolRegistry from '../services/ToolRegistry';
import { setTools, setLoading, setError } from '../stores/toolsSlice';

interface UseToolDiscoveryOptions {
  endpoints?: string[];
  autoDiscover?: boolean;
  discoveryInterval?: number;
  onDiscoveryComplete?: (tools: MCPTool[]) => void;
  onDiscoveryError?: (error: Error) => void;
}

interface UseToolDiscoveryResult {
  tools: MCPTool[];
  isDiscovering: boolean;
  error: string | null;
  discoverTools: () => Promise<void>;
  refreshTool: (toolId: string) => Promise<void>;
  clearCache: () => void;
  lastDiscovery: Date | null;
  discoveryStats: {
    totalDiscovered: number;
    successfulEndpoints: number;
    failedEndpoints: number;
    discoveryTime: number;
  };
}

// Default LLMKG endpoints
const DEFAULT_ENDPOINTS = [
  'http://localhost:8080',  // Knowledge Graph Server
  'http://localhost:8081',  // Cognitive Server
  'http://localhost:8082',  // Neural Server
  'http://localhost:8083',  // Memory Server
  'http://localhost:8084',  // Federation Server
  'http://localhost:8085',  // Analysis Server
];

export function useToolDiscovery(options: UseToolDiscoveryOptions = {}): UseToolDiscoveryResult {
  const {
    endpoints = DEFAULT_ENDPOINTS,
    autoDiscover = true,
    discoveryInterval = 5 * 60 * 1000, // 5 minutes
    onDiscoveryComplete,
    onDiscoveryError,
  } = options;

  const dispatch = useAppDispatch();
  const { tools: registeredTools, loading, error } = useAppSelector(state => state.tools);
  
  const [lastDiscovery, setLastDiscovery] = useState<Date | null>(null);
  const [discoveryStats, setDiscoveryStats] = useState({
    totalDiscovered: 0,
    successfulEndpoints: 0,
    failedEndpoints: 0,
    discoveryTime: 0,
  });

  const discoveryInProgress = useRef(false);
  const intervalRef = useRef<NodeJS.Timeout>();

  // Main discovery function
  const discoverTools = useCallback(async () => {
    if (discoveryInProgress.current) {
      console.log('Discovery already in progress, skipping...');
      return;
    }

    discoveryInProgress.current = true;
    dispatch(setLoading(true));
    dispatch(setError(null));

    const startTime = Date.now();
    const endpointResults: Map<string, { success: boolean; count: number }> = new Map();

    try {
      // Discover tools from all endpoints
      const discoveryPromises = endpoints.map(async (endpoint) => {
        try {
          const tools = await ToolDiscoveryService.discoverTools([endpoint], {
            timeout: 10000,
            retries: 2,
            retryDelay: 1000,
          });

          endpointResults.set(endpoint, { success: true, count: tools.length });
          return tools;
        } catch (error) {
          console.error(`Failed to discover from ${endpoint}:`, error);
          endpointResults.set(endpoint, { success: false, count: 0 });
          return [];
        }
      });

      const results = await Promise.all(discoveryPromises);
      const allTools = results.flat();

      // Validate and register tools
      const validTools: MCPTool[] = [];
      for (const tool of allTools) {
        const isValid = await ToolDiscoveryService.validateTool(tool);
        if (isValid) {
          validTools.push(tool);
          ToolRegistry.registerTool(tool);
        } else {
          console.warn(`Invalid tool skipped: ${tool.name}`);
        }
      }

      // Update stats
      const successfulEndpoints = Array.from(endpointResults.values()).filter(r => r.success).length;
      const failedEndpoints = endpoints.length - successfulEndpoints;
      const discoveryTime = Date.now() - startTime;

      setDiscoveryStats({
        totalDiscovered: validTools.length,
        successfulEndpoints,
        failedEndpoints,
        discoveryTime,
      });

      setLastDiscovery(new Date());

      // Update Redux store
      dispatch(setTools(validTools));

      // Callback
      onDiscoveryComplete?.(validTools);

      console.log(`Discovery complete: ${validTools.length} tools from ${successfulEndpoints} endpoints in ${discoveryTime}ms`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown discovery error';
      dispatch(setError(errorMessage));
      onDiscoveryError?.(error as Error);
      console.error('Tool discovery failed:', error);
    } finally {
      dispatch(setLoading(false));
      discoveryInProgress.current = false;
    }
  }, [endpoints, dispatch, onDiscoveryComplete, onDiscoveryError]);

  // Refresh a specific tool
  const refreshTool = useCallback(async (toolId: string) => {
    const tool = ToolRegistry.getTool(toolId);
    if (!tool || !tool.endpoint) {
      console.warn(`Tool ${toolId} not found or has no endpoint`);
      return;
    }

    try {
      const [refreshedTool] = await ToolDiscoveryService.discoverTools([tool.endpoint], {
        timeout: 5000,
        retries: 1,
      });

      if (refreshedTool && refreshedTool.name === tool.name) {
        ToolRegistry.registerTool(refreshedTool);
        
        // Update Redux store
        const allTools = ToolRegistry.getAllTools();
        dispatch(setTools(allTools));
      }
    } catch (error) {
      console.error(`Failed to refresh tool ${toolId}:`, error);
    }
  }, [dispatch]);

  // Clear discovery cache
  const clearCache = useCallback(() => {
    ToolDiscoveryService.clearCache();
    console.log('Discovery cache cleared');
  }, []);

  // Auto-discovery effect
  useEffect(() => {
    if (!autoDiscover) return;

    // Initial discovery
    discoverTools();

    // Set up interval for periodic discovery
    if (discoveryInterval > 0) {
      intervalRef.current = setInterval(() => {
        discoverTools();
      }, discoveryInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoDiscover, discoveryInterval, discoverTools]);

  // Subscribe to WebSocket updates for real-time tool changes
  useEffect(() => {
    const wsState = (window as any).__LLMKG_WS_STATE__;
    if (!wsState?.client) return;

    const handleToolUpdate = (message: any) => {
      if (message.type === 'tool-update' && message.data) {
        const { toolId, status, metrics } = message.data;
        
        // Update tool status in registry
        if (toolId && status) {
          ToolRegistry.updateToolStatus(toolId, status);
        }

        // Update tool metrics
        if (toolId && metrics) {
          ToolRegistry.updateToolMetrics(toolId, metrics);
        }

        // Refresh Redux store
        const allTools = ToolRegistry.getAllTools();
        dispatch(setTools(allTools));
      }
    };

    wsState.client.on('message', handleToolUpdate);

    return () => {
      wsState.client.off('message', handleToolUpdate);
    };
  }, [dispatch]);

  return {
    tools: registeredTools,
    isDiscovering: loading,
    error,
    discoverTools,
    refreshTool,
    clearCache,
    lastDiscovery,
    discoveryStats,
  };
}

// Hook for discovering tools from specific categories
export function useToolDiscoveryByCategory(
  categories: string[],
  options?: UseToolDiscoveryOptions
): UseToolDiscoveryResult {
  const result = useToolDiscovery(options);
  
  // Filter tools by category
  const filteredTools = result.tools.filter(tool => 
    categories.includes(tool.category)
  );

  return {
    ...result,
    tools: filteredTools,
  };
}

// Hook for monitoring tool health
export function useToolHealth(toolIds?: string[]) {
  const [healthStatus, setHealthStatus] = useState<Record<string, any>>({});
  const checkIntervalRef = useRef<NodeJS.Timeout>();

  const checkHealth = useCallback(async () => {
    const idsToCheck = toolIds || ToolRegistry.getAllTools().map(t => t.id);
    const newHealthStatus: Record<string, any> = {};

    for (const id of idsToCheck) {
      const tool = ToolRegistry.getTool(id);
      if (tool) {
        newHealthStatus[id] = tool.status;
      }
    }

    setHealthStatus(newHealthStatus);
  }, [toolIds]);

  useEffect(() => {
    // Initial check
    checkHealth();

    // Set up periodic health checks
    checkIntervalRef.current = setInterval(checkHealth, 30000); // 30 seconds

    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, [checkHealth]);

  return {
    healthStatus,
    checkHealth,
  };
}