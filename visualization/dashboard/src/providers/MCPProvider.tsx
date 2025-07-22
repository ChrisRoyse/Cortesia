import React, { createContext, useContext, useEffect, useCallback, useMemo } from 'react';
import { useAppDispatch, useAppSelector, mcpActions } from '../stores';
import { MCPContextType, MCPTool } from '../types';

const MCPContext = createContext<MCPContextType | null>(null);

interface MCPProviderProps {
  children: React.ReactNode;
  serverUrl?: string;
  refreshInterval?: number;
}

export const MCPProvider: React.FC<MCPProviderProps> = ({
  children,
  serverUrl = 'http://localhost:3000',
  refreshInterval = 30000,
}) => {
  const dispatch = useAppDispatch();
  const { tools, loading, error } = useAppSelector(state => state.mcp);

  // Fetch available MCP tools
  const fetchTools = useCallback(async (): Promise<MCPTool[]> => {
    try {
      const response = await fetch(`${serverUrl}/api/mcp/tools`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      return data.tools || [];
    } catch (error) {
      console.error('Failed to fetch MCP tools:', error);
      throw error;
    }
  }, [serverUrl]);

  // Refresh tools from server
  const refreshTools = useCallback(async () => {
    try {
      dispatch(mcpActions.setLoading(true));
      dispatch(mcpActions.setError(null));
      
      const fetchedTools = await fetchTools();
      dispatch(mcpActions.setTools(fetchedTools));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch tools';
      dispatch(mcpActions.setError(errorMessage));
    } finally {
      dispatch(mcpActions.setLoading(false));
    }
  }, [dispatch, fetchTools]);

  // Execute an MCP tool
  const executeTool = useCallback(async (
    toolName: string, 
    parameters: Record<string, any>
  ): Promise<any> => {
    const executionId = `${toolName}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    try {
      // Add execution to tracking
      dispatch(mcpActions.addExecution({
        id: executionId,
        toolName,
        parameters,
        startTime,
      }));

      // Make API call to execute tool
      const response = await fetch(`${serverUrl}/api/mcp/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tool: toolName,
          parameters,
          executionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      const endTime = Date.now();

      // Complete execution tracking
      dispatch(mcpActions.completeExecution({
        id: executionId,
        result,
        endTime,
      }));

      return result;
    } catch (error) {
      const endTime = Date.now();
      const errorResult = {
        error: error instanceof Error ? error.message : 'Execution failed',
        success: false,
      };

      // Complete execution with error
      dispatch(mcpActions.completeExecution({
        id: executionId,
        result: errorResult,
        endTime,
      }));

      throw error;
    }
  }, [dispatch, serverUrl]);

  // Initialize tools on mount and set up refresh interval
  useEffect(() => {
    refreshTools();

    const interval = setInterval(refreshTools, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshTools, refreshInterval]);

  // Create context value
  const contextValue = useMemo<MCPContextType>(() => ({
    tools,
    loading,
    error,
    executeTool,
    refreshTools,
  }), [tools, loading, error, executeTool, refreshTools]);

  return (
    <MCPContext.Provider value={contextValue}>
      {children}
    </MCPContext.Provider>
  );
};

// Hook to use MCP context
export const useMCP = (): MCPContextType => {
  const context = useContext(MCPContext);
  if (!context) {
    throw new Error('useMCP must be used within an MCPProvider');
  }
  return context;
};

// Custom hook for filtering tools by category
export const useMCPToolsByCategory = (category?: MCPTool['category']) => {
  const { tools } = useMCP();
  
  return useMemo(() => {
    if (!category) return tools;
    return tools.filter(tool => tool.category === category);
  }, [tools, category]);
};

// Custom hook for finding a specific tool
export const useMCPTool = (toolName: string) => {
  const { tools } = useMCP();
  
  return useMemo(() => {
    return tools.find(tool => tool.name === toolName);
  }, [tools, toolName]);
};

// Custom hook for executing tools with enhanced error handling
export const useMCPExecution = () => {
  const { executeTool } = useMCP();
  const executionHistory = useAppSelector(state => state.mcp.executionHistory);
  
  const executeWithRetry = useCallback(async (
    toolName: string, 
    parameters: Record<string, any>,
    retries = 2
  ) => {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        return await executeTool(toolName, parameters);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        
        if (attempt < retries) {
          // Wait before retrying (exponential backoff)
          const delay = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  }, [executeTool]);

  return {
    execute: executeTool,
    executeWithRetry,
    history: executionHistory,
  };
};

// Custom hook for batch tool execution
export const useMCPBatch = () => {
  const { executeTool } = useMCP();
  
  const executeBatch = useCallback(async (
    executions: Array<{ toolName: string; parameters: Record<string, any> }>
  ) => {
    const results = await Promise.allSettled(
      executions.map(({ toolName, parameters }) => 
        executeTool(toolName, parameters)
      )
    );
    
    return results.map((result, index) => ({
      toolName: executions[index].toolName,
      parameters: executions[index].parameters,
      success: result.status === 'fulfilled',
      result: result.status === 'fulfilled' ? result.value : null,
      error: result.status === 'rejected' ? result.reason : null,
    }));
  }, [executeTool]);

  return { executeBatch };
};

// Mock data for development (when MCP server is not available)
const mockTools: MCPTool[] = [
  {
    name: 'cognitive-analyze',
    description: 'Analyze cognitive patterns in the current knowledge state',
    parameters: {
      depth: { type: 'number', default: 3 },
      includeInhibitory: { type: 'boolean', default: true },
    },
    category: 'cognitive',
  },
  {
    name: 'neural-simulate',
    description: 'Run neural network simulation with given parameters',
    parameters: {
      iterations: { type: 'number', default: 100 },
      learningRate: { type: 'number', default: 0.01 },
    },
    category: 'neural',
  },
  {
    name: 'knowledge-query',
    description: 'Query the knowledge graph for specific patterns',
    parameters: {
      query: { type: 'string', required: true },
      maxResults: { type: 'number', default: 50 },
    },
    category: 'knowledge',
  },
  {
    name: 'memory-optimize',
    description: 'Optimize memory usage and performance',
    parameters: {
      aggressive: { type: 'boolean', default: false },
    },
    category: 'memory',
  },
  {
    name: 'system-status',
    description: 'Get comprehensive system status and health metrics',
    parameters: {},
    category: 'utility',
  },
];

// Development mode provider that uses mock data
export const MCPProviderDev: React.FC<MCPProviderProps> = ({ children }) => {
  const dispatch = useAppDispatch();
  
  useEffect(() => {
    // Load mock tools in development
    dispatch(mcpActions.setTools(mockTools));
  }, [dispatch]);

  const executeTool = useCallback(async (
    toolName: string, 
    parameters: Record<string, any>
  ) => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
    
    // Return mock result based on tool
    const mockResults = {
      'cognitive-analyze': {
        patterns: [
          { id: 'p1', type: 'hierarchical', strength: 0.8 },
          { id: 'p2', type: 'lateral', strength: 0.6 },
        ],
        inhibitoryLevel: 0.3,
        success: true,
      },
      'neural-simulate': {
        iterations: parameters.iterations || 100,
        finalLoss: Math.random() * 0.1,
        convergence: Math.random() > 0.2,
        success: true,
      },
      'knowledge-query': {
        results: [
          { id: 'n1', label: 'Neural Network', relevance: 0.9 },
          { id: 'n2', label: 'Machine Learning', relevance: 0.8 },
        ],
        totalResults: 2,
        success: true,
      },
      'memory-optimize': {
        memoryFreed: Math.floor(Math.random() * 1000) + 'MB',
        performance: 'improved',
        success: true,
      },
      'system-status': {
        uptime: '2d 14h 32m',
        cpu: Math.floor(Math.random() * 30) + 20,
        memory: Math.floor(Math.random() * 40) + 30,
        status: 'healthy',
        success: true,
      },
    };
    
    return mockResults[toolName as keyof typeof mockResults] || { 
      message: 'Tool executed successfully',
      parameters,
      success: true,
    };
  }, []);

  const contextValue = useMemo<MCPContextType>(() => ({
    tools: mockTools,
    loading: false,
    error: null,
    executeTool,
    refreshTools: async () => {},
  }), [executeTool]);

  return (
    <MCPContext.Provider value={contextValue}>
      {children}
    </MCPContext.Provider>
  );
};