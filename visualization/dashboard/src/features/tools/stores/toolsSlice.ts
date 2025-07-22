import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { MCPTool, ToolExecution, ToolError } from '../types';

interface ToolsState {
  tools: MCPTool[];
  loading: boolean;
  error: string | null;
  executions: Record<string, ToolExecution>;
  executionHistory: ToolExecution[];
  selectedToolId: string | null;
  filters: {
    searchTerm: string;
    categories: string[];
    status: string[];
    tags: string[];
  };
  view: 'grid' | 'list' | 'table';
  sortBy: 'name' | 'category' | 'status' | 'performance' | 'lastUsed';
  sortOrder: 'asc' | 'desc';
}

const initialState: ToolsState = {
  tools: [],
  loading: false,
  error: null,
  executions: {},
  executionHistory: [],
  selectedToolId: null,
  filters: {
    searchTerm: '',
    categories: [],
    status: [],
    tags: [],
  },
  view: 'grid',
  sortBy: 'name',
  sortOrder: 'asc',
};

const toolsSlice = createSlice({
  name: 'tools',
  initialState,
  reducers: {
    // Tool management
    setTools: (state, action: PayloadAction<MCPTool[]>) => {
      state.tools = action.payload;
    },
    addTool: (state, action: PayloadAction<MCPTool>) => {
      const existingIndex = state.tools.findIndex(t => t.id === action.payload.id);
      if (existingIndex >= 0) {
        state.tools[existingIndex] = action.payload;
      } else {
        state.tools.push(action.payload);
      }
    },
    updateTool: (state, action: PayloadAction<{ id: string; updates: Partial<MCPTool> }>) => {
      const tool = state.tools.find(t => t.id === action.payload.id);
      if (tool) {
        Object.assign(tool, action.payload.updates);
      }
    },
    removeTool: (state, action: PayloadAction<string>) => {
      state.tools = state.tools.filter(t => t.id !== action.payload);
    },

    // Loading and error states
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    // Execution management
    addExecution: (state, action: PayloadAction<{
      id: string;
      toolId: string;
      toolName: string;
      input: Record<string, any>;
      startTime: number;
      status: 'pending' | 'running';
    }>) => {
      const execution: ToolExecution = {
        id: action.payload.id,
        toolId: action.payload.toolId,
        toolName: action.payload.toolName,
        input: action.payload.input,
        startTime: new Date(action.payload.startTime),
        status: action.payload.status,
      };
      state.executions[execution.id] = execution;
    },
    updateExecution: (state, action: PayloadAction<{
      id: string;
      output?: Record<string, any>;
      error?: ToolError;
      endTime?: number;
      duration?: number;
      status?: 'success' | 'error';
    }>) => {
      const execution = state.executions[action.payload.id];
      if (execution) {
        if (action.payload.output !== undefined) execution.output = action.payload.output;
        if (action.payload.error !== undefined) execution.error = action.payload.error;
        if (action.payload.endTime !== undefined) execution.endTime = new Date(action.payload.endTime);
        if (action.payload.duration !== undefined) execution.duration = action.payload.duration;
        if (action.payload.status !== undefined) execution.status = action.payload.status;

        // Move to history if completed
        if (execution.status === 'success' || execution.status === 'error') {
          state.executionHistory.push({ ...execution });
          delete state.executions[action.payload.id];

          // Keep only last 100 executions
          if (state.executionHistory.length > 100) {
            state.executionHistory = state.executionHistory.slice(-100);
          }
        }
      }
    },
    clearExecutionHistory: (state) => {
      state.executionHistory = [];
    },

    // UI state
    selectTool: (state, action: PayloadAction<string | null>) => {
      state.selectedToolId = action.payload;
    },
    setView: (state, action: PayloadAction<'grid' | 'list' | 'table'>) => {
      state.view = action.payload;
    },
    setSortBy: (state, action: PayloadAction<typeof initialState.sortBy>) => {
      state.sortBy = action.payload;
    },
    setSortOrder: (state, action: PayloadAction<'asc' | 'desc'>) => {
      state.sortOrder = action.payload;
    },
    toggleSortOrder: (state) => {
      state.sortOrder = state.sortOrder === 'asc' ? 'desc' : 'asc';
    },

    // Filtering
    setSearchTerm: (state, action: PayloadAction<string>) => {
      state.filters.searchTerm = action.payload;
    },
    setFilterCategories: (state, action: PayloadAction<string[]>) => {
      state.filters.categories = action.payload;
    },
    toggleFilterCategory: (state, action: PayloadAction<string>) => {
      const category = action.payload;
      const index = state.filters.categories.indexOf(category);
      if (index >= 0) {
        state.filters.categories.splice(index, 1);
      } else {
        state.filters.categories.push(category);
      }
    },
    setFilterStatus: (state, action: PayloadAction<string[]>) => {
      state.filters.status = action.payload;
    },
    toggleFilterStatus: (state, action: PayloadAction<string>) => {
      const status = action.payload;
      const index = state.filters.status.indexOf(status);
      if (index >= 0) {
        state.filters.status.splice(index, 1);
      } else {
        state.filters.status.push(status);
      }
    },
    setFilterTags: (state, action: PayloadAction<string[]>) => {
      state.filters.tags = action.payload;
    },
    toggleFilterTag: (state, action: PayloadAction<string>) => {
      const tag = action.payload;
      const index = state.filters.tags.indexOf(tag);
      if (index >= 0) {
        state.filters.tags.splice(index, 1);
      } else {
        state.filters.tags.push(tag);
      }
    },
    clearFilters: (state) => {
      state.filters = {
        searchTerm: '',
        categories: [],
        status: [],
        tags: [],
      };
    },

    // Batch operations
    updateMultipleToolStatus: (state, action: PayloadAction<Array<{ id: string; status: any }>>) => {
      action.payload.forEach(({ id, status }) => {
        const tool = state.tools.find(t => t.id === id);
        if (tool) {
          tool.status = status;
          tool.updatedAt = new Date();
        }
      });
    },
    updateMultipleToolMetrics: (state, action: PayloadAction<Array<{ id: string; metrics: any }>>) => {
      action.payload.forEach(({ id, metrics }) => {
        const tool = state.tools.find(t => t.id === id);
        if (tool) {
          tool.metrics = { ...tool.metrics, ...metrics };
          tool.updatedAt = new Date();
        }
      });
    },
    updateToolStats: (state, action: PayloadAction<{ toolId: string; responseTime: number }>) => {
      const tool = state.tools.find(t => t.id === action.payload.toolId);
      if (tool) {
        tool.responseTime = action.payload.responseTime;
        tool.updatedAt = new Date();
      }
    },
  },
});

// Export actions
export const {
  setTools,
  addTool,
  updateTool,
  removeTool,
  setLoading,
  setError,
  addExecution,
  updateExecution,
  clearExecutionHistory,
  selectTool,
  setView,
  setSortBy,
  setSortOrder,
  toggleSortOrder,
  setSearchTerm,
  setFilterCategories,
  toggleFilterCategory,
  setFilterStatus,
  toggleFilterStatus,
  setFilterTags,
  toggleFilterTag,
  clearFilters,
  updateMultipleToolStatus,
  updateMultipleToolMetrics,
  updateToolStats,
} = toolsSlice.actions;

// Selectors
export const selectAllTools = (state: { tools: ToolsState }) => state.tools.tools;
export const selectToolById = (state: { tools: ToolsState }, id: string) => 
  state.tools.tools.find(tool => tool.id === id);
export const selectFilteredTools = (state: { tools: ToolsState }) => {
  let filtered = state.tools.tools;

  // Apply search filter
  if (state.tools.filters.searchTerm) {
    const searchLower = state.tools.filters.searchTerm.toLowerCase();
    filtered = filtered.filter(tool =>
      tool.name.toLowerCase().includes(searchLower) ||
      tool.description.toLowerCase().includes(searchLower) ||
      tool.tags?.some(tag => tag.toLowerCase().includes(searchLower))
    );
  }

  // Apply category filter
  if (state.tools.filters.categories.length > 0) {
    filtered = filtered.filter(tool =>
      state.tools.filters.categories.includes(tool.category)
    );
  }

  // Apply status filter
  if (state.tools.filters.status.length > 0) {
    filtered = filtered.filter(tool =>
      state.tools.filters.status.includes(tool.status.health)
    );
  }

  // Apply tag filter
  if (state.tools.filters.tags.length > 0) {
    filtered = filtered.filter(tool =>
      tool.tags?.some(tag => state.tools.filters.tags.includes(tag))
    );
  }

  // Apply sorting
  const sorted = [...filtered].sort((a, b) => {
    let comparison = 0;

    switch (state.tools.sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'category':
        comparison = a.category.localeCompare(b.category);
        break;
      case 'status':
        const statusOrder = { healthy: 0, degraded: 1, unknown: 2, unavailable: 3 };
        comparison = statusOrder[a.status.health] - statusOrder[b.status.health];
        break;
      case 'performance':
        comparison = a.metrics.averageResponseTime - b.metrics.averageResponseTime;
        break;
      case 'lastUsed':
        const aTime = a.metrics.lastExecutionTime?.getTime() || 0;
        const bTime = b.metrics.lastExecutionTime?.getTime() || 0;
        comparison = bTime - aTime; // Most recent first
        break;
    }

    return state.tools.sortOrder === 'desc' ? -comparison : comparison;
  });

  return sorted;
};

export const selectToolsGroupedByCategory = (state: { tools: ToolsState }) => {
  const grouped: Record<string, MCPTool[]> = {};
  
  state.tools.tools.forEach(tool => {
    if (!grouped[tool.category]) {
      grouped[tool.category] = [];
    }
    grouped[tool.category].push(tool);
  });

  return grouped;
};

export const selectExecutionHistory = (state: { tools: ToolsState }) => state.tools.executionHistory;
export const selectActiveExecutions = (state: { tools: ToolsState }) => Object.values(state.tools.executions);
export const selectToolsLoading = (state: { tools: ToolsState }) => state.tools.loading;
export const selectToolsError = (state: { tools: ToolsState }) => state.tools.error;

export default toolsSlice.reducer;