import { useToolDiscovery, useToolRegistry } from '../features/tools/hooks';
import { MCPTool } from '../features/tools/types';

interface MCPToolsHookResult {
  tools: MCPTool[];
  recentExecutions: any[];
  toolStats: {
    executionsToday: number;
    executionGrowth: number;
    successRate: number;
    avgResponseTime: number;
  };
  isLoading: boolean;
  error: string | null;
  executeTools: (toolId: string, params: any) => Promise<void>;
  refreshTools: () => Promise<void>;
}

/**
 * Compatibility hook for existing code that uses useMCPTools
 * This wraps our new tool discovery and registry hooks
 */
export function useMCPTools(): MCPToolsHookResult {
  const { tools, isDiscovering, error, discoverTools } = useToolDiscovery();
  const { executeTool, getExecutionHistory, registryStats } = useToolRegistry();

  // Calculate tool statistics
  const toolStats = {
    executionsToday: tools.reduce((sum, tool) => sum + (tool.metrics?.totalExecutions || 0), 0),
    executionGrowth: 15, // Mock growth percentage
    successRate: Math.round(
      tools.reduce((sum, tool) => sum + (tool.metrics?.successRate || 0), 0) / (tools.length || 1)
    ),
    avgResponseTime: Math.round(
      tools.reduce((sum, tool) => sum + (tool.metrics?.averageResponseTime || 0), 0) / (tools.length || 1)
    ),
  };

  const executeTools = async (toolId: string, params: any) => {
    await executeTool(toolId, params);
  };

  const refreshTools = async () => {
    await discoverTools();
  };

  return {
    tools,
    recentExecutions: getExecutionHistory(),
    toolStats,
    isLoading: isDiscovering,
    error,
    executeTools,
    refreshTools,
  };
}