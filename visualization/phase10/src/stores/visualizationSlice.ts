import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { LLMKGVisualizationConfig, ThemeConfig, DashboardLayout } from '@/types';

interface VisualizationState {
  config: LLMKGVisualizationConfig;
  theme: ThemeConfig;
  currentLayout: DashboardLayout | null;
  availableLayouts: DashboardLayout[];
  sidebarCollapsed: boolean;
  enabledPhases: string[];
  debugMode: boolean;
  lastUpdate: number;
}

const defaultTheme: ThemeConfig = {
  primaryColor: '#1890ff',
  backgroundColor: '#001529',
  textColor: '#ffffff',
  borderColor: '#303030',
  accentColor: '#52c41a',
  chartColors: ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2'],
  darkMode: true,
};

const defaultConfig: LLMKGVisualizationConfig = {
  mcp: {
    endpoint: 'ws://localhost:8080',
    protocol: 'ws',
    reconnect: {
      enabled: true,
      maxAttempts: 5,
      delay: 5000,
    },
  },
  visualization: {
    theme: 'dark',
    updateInterval: 1000,
    maxDataPoints: 1000,
    enableAnimations: true,
    enableDebugMode: false,
  },
  performance: {
    enableProfiling: false,
    sampleRate: 1,
    maxMemoryUsage: 512 * 1024 * 1024,
    enableLazyLoading: true,
  },
  features: {
    enabledPhases: ['phase7', 'phase8', 'phase9', 'phase10'],
    experimentalFeatures: [],
  },
};

const initialState: VisualizationState = {
  config: defaultConfig,
  theme: defaultTheme,
  currentLayout: null,
  availableLayouts: [],
  sidebarCollapsed: false,
  enabledPhases: ['phase7', 'phase8', 'phase9', 'phase10'],
  debugMode: false,
  lastUpdate: 0,
};

const visualizationSlice = createSlice({
  name: 'visualization',
  initialState,
  reducers: {
    updateConfig: (state, action: PayloadAction<Partial<LLMKGVisualizationConfig>>) => {
      state.config = { ...state.config, ...action.payload };
      state.lastUpdate = Date.now();
    },
    
    updateTheme: (state, action: PayloadAction<Partial<ThemeConfig>>) => {
      state.theme = { ...state.theme, ...action.payload };
      state.lastUpdate = Date.now();
    },
    
    setCurrentLayout: (state, action: PayloadAction<DashboardLayout>) => {
      state.currentLayout = action.payload;
    },
    
    addLayout: (state, action: PayloadAction<DashboardLayout>) => {
      const existingIndex = state.availableLayouts.findIndex(
        layout => layout.id === action.payload.id
      );
      if (existingIndex >= 0) {
        state.availableLayouts[existingIndex] = action.payload;
      } else {
        state.availableLayouts.push(action.payload);
      }
    },
    
    removeLayout: (state, action: PayloadAction<string>) => {
      state.availableLayouts = state.availableLayouts.filter(
        layout => layout.id !== action.payload
      );
      if (state.currentLayout?.id === action.payload) {
        state.currentLayout = null;
      }
    },
    
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    
    enablePhase: (state, action: PayloadAction<string>) => {
      if (!state.enabledPhases.includes(action.payload)) {
        state.enabledPhases.push(action.payload);
        state.config.features.enabledPhases = state.enabledPhases;
      }
    },
    
    disablePhase: (state, action: PayloadAction<string>) => {
      state.enabledPhases = state.enabledPhases.filter(
        phase => phase !== action.payload
      );
      state.config.features.enabledPhases = state.enabledPhases;
    },
    
    toggleDebugMode: (state) => {
      state.debugMode = !state.debugMode;
      state.config.visualization.enableDebugMode = state.debugMode;
    },
    
    setDebugMode: (state, action: PayloadAction<boolean>) => {
      state.debugMode = action.payload;
      state.config.visualization.enableDebugMode = action.payload;
    },
    
    updateVisualizationSettings: (state, action: PayloadAction<{
      updateInterval?: number;
      maxDataPoints?: number;
      enableAnimations?: boolean;
    }>) => {
      state.config.visualization = {
        ...state.config.visualization,
        ...action.payload,
      };
      state.lastUpdate = Date.now();
    },
    
    updatePerformanceSettings: (state, action: PayloadAction<{
      enableProfiling?: boolean;
      sampleRate?: number;
      maxMemoryUsage?: number;
      enableLazyLoading?: boolean;
    }>) => {
      state.config.performance = {
        ...state.config.performance,
        ...action.payload,
      };
      state.lastUpdate = Date.now();
    },
    
    updateMCPSettings: (state, action: PayloadAction<{
      endpoint?: string;
      protocol?: 'ws' | 'http';
      reconnect?: {
        enabled: boolean;
        maxAttempts: number;
        delay: number;
      };
    }>) => {
      state.config.mcp = {
        ...state.config.mcp,
        ...action.payload,
      };
      state.lastUpdate = Date.now();
    },
    
    resetToDefaults: (state) => {
      state.config = defaultConfig;
      state.theme = defaultTheme;
      state.enabledPhases = ['phase7', 'phase8', 'phase9', 'phase10'];
      state.debugMode = false;
      state.lastUpdate = Date.now();
    },
    
    importConfig: (state, action: PayloadAction<LLMKGVisualizationConfig>) => {
      state.config = action.payload;
      state.enabledPhases = action.payload.features.enabledPhases;
      state.debugMode = action.payload.visualization.enableDebugMode;
      state.lastUpdate = Date.now();
    },
  },
});

export const {
  updateConfig,
  updateTheme,
  setCurrentLayout,
  addLayout,
  removeLayout,
  toggleSidebar,
  setSidebarCollapsed,
  enablePhase,
  disablePhase,
  toggleDebugMode,
  setDebugMode,
  updateVisualizationSettings,
  updatePerformanceSettings,
  updateMCPSettings,
  resetToDefaults,
  importConfig,
} = visualizationSlice.actions;

export default visualizationSlice.reducer;

// Selectors
export const selectConfig = (state: { visualization: VisualizationState }) => 
  state.visualization.config;

export const selectTheme = (state: { visualization: VisualizationState }) => 
  state.visualization.theme;

export const selectCurrentLayout = (state: { visualization: VisualizationState }) => 
  state.visualization.currentLayout;

export const selectAvailableLayouts = (state: { visualization: VisualizationState }) => 
  state.visualization.availableLayouts;

export const selectSidebarCollapsed = (state: { visualization: VisualizationState }) => 
  state.visualization.sidebarCollapsed;

export const selectEnabledPhases = (state: { visualization: VisualizationState }) => 
  state.visualization.enabledPhases;

export const selectDebugMode = (state: { visualization: VisualizationState }) => 
  state.visualization.debugMode;

export const selectLastUpdate = (state: { visualization: VisualizationState }) => 
  state.visualization.lastUpdate;