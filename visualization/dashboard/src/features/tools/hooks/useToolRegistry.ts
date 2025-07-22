import { useState, useEffect, useCallback, useMemo } from 'react';
import { useAppDispatch, useAppSelector } from '../../../stores/hooks';
import { MCPTool, ToolCategory, ToolFilter, ToolExecution } from '../types';
import ToolRegistry from '../services/ToolRegistry';
import { setTools, addExecution, updateExecution } from '../stores/toolsSlice';

interface UseToolRegistryOptions {
  autoSync?: boolean;
  syncInterval?: number;
}

interface UseToolRegistryResult {
  tools: MCPTool[];
  toolCount: number;
  categories: ToolCategory[];
  searchTools: (query: string) => MCPTool[];
  filterTools: (filter: ToolFilter) => MCPTool[];
  getTool: (id: string) => MCPTool | undefined;
  getToolsByCategory: (category: ToolCategory) => MCPTool[];
  executeTool: (toolId: string, params: any) => Promise<any>;
  getExecutionHistory: (toolId?: string) => ToolExecution[];
  registryStats: any;
}

export function useToolRegistry(options: UseToolRegistryOptions = {}): UseToolRegistryResult {
  const { autoSync = true, syncInterval = 5000 } = options;
  
  const dispatch = useAppDispatch();
  const { tools: reduxTools, executions, executionHistory } = useAppSelector(state => state.tools);
  const [registryStats, setRegistryStats] = useState(ToolRegistry.getStats());

  // Sync registry with Redux store
  const syncWithRedux = useCallback(() => {
    const allTools = ToolRegistry.getAllTools();
    dispatch(setTools(allTools));
    setRegistryStats(ToolRegistry.getStats());
  }, [dispatch]);

  // Subscribe to registry events
  useEffect(() => {
    const unsubscribe = ToolRegistry.subscribe((event) => {
      switch (event.type) {
        case 'tool-registered':
        case 'tool-updated':
        case 'tool-removed':
        case 'batch-registered':
        case 'registry-cleared':
          syncWithRedux();
          break;
      }
    });

    return unsubscribe;
  }, [syncWithRedux]);

  // Auto-sync with Redux store
  useEffect(() => {
    if (!autoSync) return;

    const intervalId = setInterval(syncWithRedux, syncInterval);
    return () => clearInterval(intervalId);
  }, [autoSync, syncInterval, syncWithRedux]);

  // Search tools
  const searchTools = useCallback((query: string): MCPTool[] => {
    return ToolRegistry.searchTools(query);
  }, []);

  // Filter tools
  const filterTools = useCallback((filter: ToolFilter): MCPTool[] => {
    return ToolRegistry.filterTools(filter);
  }, []);

  // Get tool by ID
  const getTool = useCallback((id: string): MCPTool | undefined => {
    return ToolRegistry.getTool(id);
  }, []);

  // Get tools by category
  const getToolsByCategory = useCallback((category: ToolCategory): MCPTool[] => {
    return ToolRegistry.getToolsByCategory(category);
  }, []);

  // Execute a tool
  const executeTool = useCallback(async (toolId: string, params: any): Promise<any> => {
    const tool = ToolRegistry.getTool(toolId);
    if (!tool || !tool.endpoint) {
      throw new Error(`Tool ${toolId} not found or has no endpoint`);
    }

    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    // Add execution to store
    dispatch(addExecution({
      id: executionId,
      toolId,
      toolName: tool.name,
      input: params,
      startTime,
      status: 'running',
    }));

    try {
      // Execute tool via endpoint
      const response = await fetch(`${tool.endpoint}/mcp/tools/${tool.name}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      const endTime = Date.now();

      // Update execution with success
      dispatch(updateExecution({
        id: executionId,
        output: result,
        endTime,
        duration: endTime - startTime,
        status: 'success',
      }));

      // Update tool metrics
      const currentMetrics = tool.metrics;
      const newExecution = {
        totalExecutions: currentMetrics.totalExecutions + 1,
        successRate: ((currentMetrics.successRate * currentMetrics.totalExecutions) + 100) / (currentMetrics.totalExecutions + 1),
        averageResponseTime: ((currentMetrics.averageResponseTime * currentMetrics.totalExecutions) + (endTime - startTime)) / (currentMetrics.totalExecutions + 1),
        lastExecutionTime: new Date(),
      };

      ToolRegistry.updateToolMetrics(toolId, newExecution);

      return result;
    } catch (error) {
      const endTime = Date.now();
      
      // Update execution with error
      dispatch(updateExecution({
        id: executionId,
        error: {
          code: 'EXECUTION_ERROR',
          message: error instanceof Error ? error.message : 'Unknown error',
          details: error,
        },
        endTime,
        duration: endTime - startTime,
        status: 'error',
      }));

      // Update tool error metrics
      const currentMetrics = tool.metrics;
      ToolRegistry.updateToolMetrics(toolId, {
        totalExecutions: currentMetrics.totalExecutions + 1,
        successRate: (currentMetrics.successRate * currentMetrics.totalExecutions) / (currentMetrics.totalExecutions + 1),
        errorCount: currentMetrics.errorCount + 1,
      });

      throw error;
    }
  }, [dispatch]);

  // Get execution history
  const getExecutionHistory = useCallback((toolId?: string): ToolExecution[] => {
    if (toolId) {
      return executionHistory.filter(exec => exec.toolId === toolId);
    }
    return executionHistory;
  }, [executionHistory]);

  // Get unique categories
  const categories = useMemo(() => {
    const categorySet = new Set(reduxTools.map(tool => tool.category));
    return Array.from(categorySet) as ToolCategory[];
  }, [reduxTools]);

  return {
    tools: reduxTools,
    toolCount: reduxTools.length,
    categories,
    searchTools,
    filterTools,
    getTool,
    getToolsByCategory,
    executeTool,
    getExecutionHistory,
    registryStats,
  };
}

// Hook for working with a specific tool
export function useTool(toolId: string) {
  const { getTool, executeTool, getExecutionHistory } = useToolRegistry();
  const [tool, setTool] = useState<MCPTool | undefined>(getTool(toolId));
  const [executing, setExecuting] = useState(false);
  const [lastExecution, setLastExecution] = useState<ToolExecution | null>(null);

  // Update tool when registry changes
  useEffect(() => {
    const unsubscribe = ToolRegistry.subscribe((event) => {
      if (
        (event.type === 'tool-updated' || event.type === 'tool-status-updated') &&
        event.toolId === toolId
      ) {
        setTool(getTool(toolId));
      }
    });

    return unsubscribe;
  }, [toolId, getTool]);

  // Execute tool wrapper
  const execute = useCallback(async (params: any) => {
    setExecuting(true);
    try {
      const result = await executeTool(toolId, params);
      const history = getExecutionHistory(toolId);
      if (history.length > 0) {
        setLastExecution(history[history.length - 1]);
      }
      return result;
    } finally {
      setExecuting(false);
    }
  }, [toolId, executeTool, getExecutionHistory]);

  // Get tool-specific execution history
  const history = useMemo(() => {
    return getExecutionHistory(toolId);
  }, [toolId, getExecutionHistory]);

  return {
    tool,
    execute,
    executing,
    history,
    lastExecution,
  };
}

// Hook for tool statistics
export function useToolStats(toolId?: string) {
  const { tools, registryStats } = useToolRegistry();
  
  const toolStats = useMemo(() => {
    if (toolId) {
      const tool = tools.find(t => t.id === toolId);
      if (!tool) return null;

      return {
        metrics: tool.metrics,
        status: tool.status,
        category: tool.category,
        lastUpdated: tool.updatedAt,
      };
    }

    // Aggregate stats for all tools
    const totalExecutions = tools.reduce((sum, tool) => sum + tool.metrics.totalExecutions, 0);
    const avgSuccessRate = tools.reduce((sum, tool) => sum + tool.metrics.successRate, 0) / tools.length;
    const avgResponseTime = tools.reduce((sum, tool) => sum + tool.metrics.averageResponseTime, 0) / tools.length;

    return {
      totalTools: tools.length,
      totalExecutions,
      avgSuccessRate,
      avgResponseTime,
      byCategory: registryStats.categoryCounts,
      byStatus: registryStats.statusCounts,
    };
  }, [toolId, tools, registryStats]);

  return toolStats;
}

// Hook for tool favorites
export function useToolFavorites() {
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    const stored = localStorage.getItem('llmkg-tool-favorites');
    return stored ? new Set(JSON.parse(stored)) : new Set();
  });

  const toggleFavorite = useCallback((toolId: string) => {
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(toolId)) {
        newFavorites.delete(toolId);
      } else {
        newFavorites.add(toolId);
      }
      
      // Persist to localStorage
      localStorage.setItem('llmkg-tool-favorites', JSON.stringify(Array.from(newFavorites)));
      
      return newFavorites;
    });
  }, []);

  const isFavorite = useCallback((toolId: string): boolean => {
    return favorites.has(toolId);
  }, [favorites]);

  return {
    favorites: Array.from(favorites),
    toggleFavorite,
    isFavorite,
  };
}