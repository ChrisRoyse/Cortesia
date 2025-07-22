import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Layout, Layouts } from 'react-grid-layout';
import { LayoutPreset } from '../../components/Layout/LayoutManager';
import { loadLayoutFromStorage, saveLayoutToStorage } from '../../utils/layoutUtils';

export interface LayoutSettings {
  isDraggable: boolean;
  isResizable: boolean;
  compactType: 'vertical' | 'horizontal' | null;
  preventCollision: boolean;
  margin: [number, number];
  containerPadding: [number, number];
  rowHeight: number;
  autoSize: boolean;
  useCSSTransforms: boolean;
}

export interface LayoutState {
  currentLayout: Layouts;
  layoutHistory: Layouts[];
  presets: LayoutPreset[];
  activePreset: string | null;
  settings: LayoutSettings;
  isLoading: boolean;
  error: string | null;
  lastSaved: string | null;
}

const defaultSettings: LayoutSettings = {
  isDraggable: true,
  isResizable: true,
  compactType: 'vertical',
  preventCollision: false,
  margin: [10, 10],
  containerPadding: [10, 10],
  rowHeight: 150,
  autoSize: true,
  useCSSTransforms: true
};

const defaultLayouts: Layouts = {
  lg: [],
  md: [],
  sm: [],
  xs: [],
  xxs: []
};

// Load initial state from localStorage
const loadInitialState = (): Partial<LayoutState> => {
  try {
    const storedLayout = loadLayoutFromStorage('llmkg-dashboard-layout');
    const storedPresets = localStorage.getItem('llmkg-layout-presets');
    const storedSettings = localStorage.getItem('llmkg-layout-settings');
    
    return {
      currentLayout: storedLayout?.layout || defaultLayouts,
      presets: storedPresets ? JSON.parse(storedPresets) : [],
      settings: storedSettings ? { ...defaultSettings, ...JSON.parse(storedSettings) } : defaultSettings,
      lastSaved: storedLayout ? new Date().toISOString() : null
    };
  } catch (error) {
    console.warn('Failed to load layout state from storage:', error);
    return {};
  }
};

const initialState: LayoutState = {
  currentLayout: defaultLayouts,
  layoutHistory: [],
  presets: [],
  activePreset: null,
  settings: defaultSettings,
  isLoading: false,
  error: null,
  lastSaved: null,
  ...loadInitialState()
};

const layoutSlice = createSlice({
  name: 'layout',
  initialState,
  reducers: {
    setLayout: (state, action: PayloadAction<{ layouts: Layouts; items?: any[] }>) => {
      // Add current layout to history
      if (Object.keys(state.currentLayout).length > 0) {
        state.layoutHistory.push(state.currentLayout);
        // Keep only last 10 layouts in history
        if (state.layoutHistory.length > 10) {
          state.layoutHistory.shift();
        }
      }
      
      state.currentLayout = action.payload.layouts;
      state.activePreset = null; // Clear active preset when layout changes manually
      state.lastSaved = new Date().toISOString();
      state.error = null;
      
      // Save to localStorage
      saveLayoutToStorage('llmkg-dashboard-layout', action.payload.layouts, action.payload.items);
    },

    updateLayoutSettings: (state, action: PayloadAction<Partial<LayoutSettings>>) => {
      state.settings = { ...state.settings, ...action.payload };
      
      try {
        localStorage.setItem('llmkg-layout-settings', JSON.stringify(state.settings));
      } catch (error) {
        console.warn('Failed to save layout settings:', error);
      }
    },

    saveLayoutPreset: (state, action: PayloadAction<LayoutPreset>) => {
      const existingIndex = state.presets.findIndex(preset => preset.id === action.payload.id);
      
      if (existingIndex >= 0) {
        state.presets[existingIndex] = action.payload;
      } else {
        state.presets.push(action.payload);
      }
      
      try {
        localStorage.setItem('llmkg-layout-presets', JSON.stringify(state.presets));
      } catch (error) {
        console.warn('Failed to save layout preset:', error);
        state.error = 'Failed to save preset';
      }
    },

    loadLayoutPreset: (state, action: PayloadAction<string>) => {
      const preset = state.presets.find(p => p.id === action.payload);
      
      if (preset) {
        // Add current layout to history
        if (Object.keys(state.currentLayout).length > 0) {
          state.layoutHistory.push(state.currentLayout);
        }
        
        state.currentLayout = preset.layout;
        state.activePreset = action.payload;
        state.lastSaved = new Date().toISOString();
        state.error = null;
        
        // Save to localStorage
        saveLayoutToStorage('llmkg-dashboard-layout', preset.layout, preset.items);
      } else {
        state.error = `Preset not found: ${action.payload}`;
      }
    },

    deleteLayoutPreset: (state, action: PayloadAction<string>) => {
      state.presets = state.presets.filter(preset => preset.id !== action.payload);
      
      if (state.activePreset === action.payload) {
        state.activePreset = null;
      }
      
      try {
        localStorage.setItem('llmkg-layout-presets', JSON.stringify(state.presets));
      } catch (error) {
        console.warn('Failed to delete layout preset:', error);
        state.error = 'Failed to delete preset';
      }
    },

    resetLayout: (state) => {
      // Add current layout to history
      if (Object.keys(state.currentLayout).length > 0) {
        state.layoutHistory.push(state.currentLayout);
      }
      
      state.currentLayout = defaultLayouts;
      state.activePreset = null;
      state.error = null;
      
      // Clear from localStorage
      try {
        localStorage.removeItem('llmkg-dashboard-layout');
      } catch (error) {
        console.warn('Failed to clear layout from storage:', error);
      }
    },

    undoLayout: (state) => {
      if (state.layoutHistory.length > 0) {
        const previousLayout = state.layoutHistory.pop()!;
        state.currentLayout = previousLayout;
        state.activePreset = null;
        state.lastSaved = new Date().toISOString();
        
        // Save to localStorage
        saveLayoutToStorage('llmkg-dashboard-layout', previousLayout);
      }
    },

    clearLayoutHistory: (state) => {
      state.layoutHistory = [];
    },

    setActivePreset: (state, action: PayloadAction<string | null>) => {
      state.activePreset = action.payload;
    },

    setLayoutError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    setLayoutLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    importLayoutPresets: (state, action: PayloadAction<LayoutPreset[]>) => {
      // Merge with existing presets, avoiding duplicates
      const existingIds = new Set(state.presets.map(p => p.id));
      const newPresets = action.payload.filter(preset => !existingIds.has(preset.id));
      
      state.presets = [...state.presets, ...newPresets];
      
      try {
        localStorage.setItem('llmkg-layout-presets', JSON.stringify(state.presets));
      } catch (error) {
        console.warn('Failed to import layout presets:', error);
        state.error = 'Failed to import presets';
      }
    },

    exportLayoutPresets: (state) => {
      try {
        const exportData = {
          presets: state.presets,
          currentLayout: state.currentLayout,
          settings: state.settings,
          timestamp: new Date().toISOString(),
          version: '1.0'
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
          type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `llmkg-layout-presets-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (error) {
        console.warn('Failed to export layout presets:', error);
        state.error = 'Failed to export presets';
      }
    },

    optimizeCurrentLayout: (state, action: PayloadAction<{ containerCols: number; allowResize?: boolean; allowReorder?: boolean }>) => {
      const { containerCols, allowResize = true, allowReorder = true } = action.payload;
      
      // This would integrate with the optimization utilities
      // For now, we'll just mark it as optimized
      state.lastSaved = new Date().toISOString();
      state.error = null;
      
      // In a real implementation, you'd call the optimization utilities here
      // const optimizedLayout = optimizeLayout(state.currentLayout.lg, containerCols, allowResize, allowReorder);
      // state.currentLayout.lg = optimizedLayout;
    },

    duplicateLayoutPreset: (state, action: PayloadAction<{ sourceId: string; newId: string; newName: string }>) => {
      const { sourceId, newId, newName } = action.payload;
      const sourcePreset = state.presets.find(p => p.id === sourceId);
      
      if (sourcePreset) {
        const duplicatedPreset: LayoutPreset = {
          ...sourcePreset,
          id: newId,
          name: newName,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        };
        
        state.presets.push(duplicatedPreset);
        
        try {
          localStorage.setItem('llmkg-layout-presets', JSON.stringify(state.presets));
        } catch (error) {
          console.warn('Failed to duplicate layout preset:', error);
          state.error = 'Failed to duplicate preset';
        }
      } else {
        state.error = `Source preset not found: ${sourceId}`;
      }
    },

    updateLayoutPreset: (state, action: PayloadAction<{ id: string; updates: Partial<LayoutPreset> }>) => {
      const { id, updates } = action.payload;
      const presetIndex = state.presets.findIndex(p => p.id === id);
      
      if (presetIndex >= 0) {
        state.presets[presetIndex] = {
          ...state.presets[presetIndex],
          ...updates,
          updatedAt: new Date().toISOString()
        };
        
        try {
          localStorage.setItem('llmkg-layout-presets', JSON.stringify(state.presets));
        } catch (error) {
          console.warn('Failed to update layout preset:', error);
          state.error = 'Failed to update preset';
        }
      } else {
        state.error = `Preset not found: ${id}`;
      }
    }
  }
});

export const {
  setLayout,
  updateLayoutSettings,
  saveLayoutPreset,
  loadLayoutPreset,
  deleteLayoutPreset,
  resetLayout,
  undoLayout,
  clearLayoutHistory,
  setActivePreset,
  setLayoutError,
  setLayoutLoading,
  importLayoutPresets,
  exportLayoutPresets,
  optimizeCurrentLayout,
  duplicateLayoutPreset,
  updateLayoutPreset
} = layoutSlice.actions;

// Selectors
export const selectCurrentLayout = (state: { layout: LayoutState }) => state.layout.currentLayout;
export const selectLayoutPresets = (state: { layout: LayoutState }) => state.layout.presets;
export const selectActivePreset = (state: { layout: LayoutState }) => state.layout.activePreset;
export const selectLayoutSettings = (state: { layout: LayoutState }) => state.layout.settings;
export const selectLayoutHistory = (state: { layout: LayoutState }) => state.layout.layoutHistory;
export const selectLayoutError = (state: { layout: LayoutState }) => state.layout.error;
export const selectLayoutLoading = (state: { layout: LayoutState }) => state.layout.isLoading;
export const selectLastSaved = (state: { layout: LayoutState }) => state.layout.lastSaved;

// Computed selectors
export const selectCanUndo = (state: { layout: LayoutState }) => state.layout.layoutHistory.length > 0;
export const selectLayoutStats = (state: { layout: LayoutState }) => ({
  totalPresets: state.layout.presets.length,
  customPresets: state.layout.presets.filter(p => !p.isDefault).length,
  historySize: state.layout.layoutHistory.length,
  lastSaved: state.layout.lastSaved,
  hasActivePreset: !!state.layout.activePreset
});

export default layoutSlice.reducer;