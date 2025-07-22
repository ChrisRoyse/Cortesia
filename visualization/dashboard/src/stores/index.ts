import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import { 
  LLMKGData, 
  DashboardState, 
  DashboardConfig, 
  WebSocketMessage,
  MCPTool,
  ErrorState,
  LoadingState 
} from '../types';
import layoutReducer from './slices/layoutSlice';
import realtimeReducer from './slices/realtimeSlice';
import toolsReducer from '../features/tools/stores/toolsSlice';

// Dashboard Slice
const initialDashboardState: DashboardState = {
  config: {
    theme: 'dark',
    refreshRate: 1000,
    maxDataPoints: 1000,
    enableAnimations: true,
  },
  activeView: 'overview',
  isFullscreen: false,
  sidebarCollapsed: false,
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState: initialDashboardState,
  reducers: {
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'auto'>) => {
      state.config.theme = action.payload;
    },
    setRefreshRate: (state, action: PayloadAction<number>) => {
      state.config.refreshRate = action.payload;
    },
    setMaxDataPoints: (state, action: PayloadAction<number>) => {
      state.config.maxDataPoints = action.payload;
    },
    toggleAnimations: (state) => {
      state.config.enableAnimations = !state.config.enableAnimations;
    },
    setActiveView: (state, action: PayloadAction<string>) => {
      state.activeView = action.payload;
    },
    toggleFullscreen: (state) => {
      state.isFullscreen = !state.isFullscreen;
    },
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    updateConfig: (state, action: PayloadAction<Partial<DashboardConfig>>) => {
      state.config = { ...state.config, ...action.payload };
    },
  },
});

// Data Slice - for LLMKG real-time data
interface DataState {
  current: LLMKGData | null;
  history: LLMKGData[];
  loading: LoadingState;
  error: ErrorState;
  lastUpdate: number;
  subscriptions: string[];
}

const initialDataState: DataState = {
  current: null,
  history: [],
  loading: { isLoading: false },
  error: { hasError: false, error: null },
  lastUpdate: 0,
  subscriptions: [],
};

const dataSlice = createSlice({
  name: 'data',
  initialState: initialDataState,
  reducers: {
    setCurrentData: (state, action: PayloadAction<LLMKGData>) => {
      state.current = action.payload;
      state.history.push(action.payload);
      state.lastUpdate = Date.now();
      
      // Keep only the last maxDataPoints entries
      const maxPoints = 1000; // This should come from dashboard config
      if (state.history.length > maxPoints) {
        state.history = state.history.slice(-maxPoints);
      }
    },
    setLoading: (state, action: PayloadAction<LoadingState>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<ErrorState>) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = { hasError: false, error: null };
    },
    clearHistory: (state) => {
      state.history = [];
    },
    setSubscriptions: (state, action: PayloadAction<string[]>) => {
      state.subscriptions = action.payload;
    },
    addSubscription: (state, action: PayloadAction<string>) => {
      if (!state.subscriptions.includes(action.payload)) {
        state.subscriptions.push(action.payload);
      }
    },
    removeSubscription: (state, action: PayloadAction<string>) => {
      state.subscriptions = state.subscriptions.filter(sub => sub !== action.payload);
    },
  },
});

// WebSocket Slice
interface WebSocketState {
  isConnected: boolean;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastMessage: WebSocketMessage | null;
  error: string | null;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
}

const initialWebSocketState: WebSocketState = {
  isConnected: false,
  connectionState: 'disconnected',
  lastMessage: null,
  error: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 5,
};

const webSocketSlice = createSlice({
  name: 'webSocket',
  initialState: initialWebSocketState,
  reducers: {
    setConnectionState: (state, action: PayloadAction<'connecting' | 'connected' | 'disconnected' | 'error'>) => {
      state.connectionState = action.payload;
      state.isConnected = action.payload === 'connected';
    },
    setLastMessage: (state, action: PayloadAction<WebSocketMessage>) => {
      state.lastMessage = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    incrementReconnectAttempts: (state) => {
      state.reconnectAttempts += 1;
    },
    resetReconnectAttempts: (state) => {
      state.reconnectAttempts = 0;
    },
  },
});

// MCP Slice
interface MCPState {
  tools: MCPTool[];
  loading: boolean;
  error: string | null;
  executions: Record<string, any>;
  executionHistory: Array<{
    toolName: string;
    parameters: Record<string, any>;
    result: any;
    timestamp: number;
    duration: number;
  }>;
}

const initialMCPState: MCPState = {
  tools: [],
  loading: false,
  error: null,
  executions: {},
  executionHistory: [],
};

const mcpSlice = createSlice({
  name: 'mcp',
  initialState: initialMCPState,
  reducers: {
    setTools: (state, action: PayloadAction<MCPTool[]>) => {
      state.tools = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    addExecution: (state, action: PayloadAction<{
      id: string;
      toolName: string;
      parameters: Record<string, any>;
      startTime: number;
    }>) => {
      state.executions[action.payload.id] = action.payload;
    },
    completeExecution: (state, action: PayloadAction<{
      id: string;
      result: any;
      endTime: number;
    }>) => {
      const execution = state.executions[action.payload.id];
      if (execution) {
        const historyEntry = {
          toolName: execution.toolName,
          parameters: execution.parameters,
          result: action.payload.result,
          timestamp: execution.startTime,
          duration: action.payload.endTime - execution.startTime,
        };
        state.executionHistory.push(historyEntry);
        
        // Keep only last 100 executions
        if (state.executionHistory.length > 100) {
          state.executionHistory = state.executionHistory.slice(-100);
        }
        
        delete state.executions[action.payload.id];
      }
    },
    clearExecutionHistory: (state) => {
      state.executionHistory = [];
    },
  },
});

// Configure Store
export const store = configureStore({
  reducer: {
    dashboard: dashboardSlice.reducer,
    data: dataSlice.reducer,
    webSocket: webSocketSlice.reducer,
    mcp: mcpSlice.reducer,
    layout: layoutReducer,
    realtime: realtimeReducer,
    tools: toolsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['data/setCurrentData', 'webSocket/setLastMessage', 'tools/setTools', 'tools/addTool', 'tools/updateTool'],
        ignoredPaths: ['data.current', 'data.history', 'webSocket.lastMessage', 'tools.tools', 'tools.executions', 'tools.executionHistory'],
      },
    }),
});

// Export types and actions
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

// Export actions
export const dashboardActions = dashboardSlice.actions;
export const dataActions = dataSlice.actions;
export const webSocketActions = webSocketSlice.actions;
export const mcpActions = mcpSlice.actions;

// Selectors
export const selectDashboardConfig = (state: RootState) => state.dashboard.config;
export const selectActiveView = (state: RootState) => state.dashboard.activeView;
export const selectCurrentData = (state: RootState) => state.data.current;
export const selectDataHistory = (state: RootState) => state.data.history;
export const selectWebSocketState = (state: RootState) => state.webSocket;
export const selectMCPTools = (state: RootState) => state.mcp.tools;
export const selectMCPExecutionHistory = (state: RootState) => state.mcp.executionHistory;