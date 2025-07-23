import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ComponentRegistration } from '@/types';

interface ComponentsState {
  registered: ComponentRegistration[];
  active: string[];
  loading: string[];
  errors: Record<string, string>;
  lastRegistered: number;
  performance: Record<string, {
    renderTime: number;
    memoryUsage: number;
    updateCount: number;
    lastUpdate: number;
  }>;
}

const initialState: ComponentsState = {
  registered: [],
  active: [],
  loading: [],
  errors: {},
  lastRegistered: 0,
  performance: {},
};

const componentsSlice = createSlice({
  name: 'components',
  initialState,
  reducers: {
    registerComponent: (state, action: PayloadAction<ComponentRegistration>) => {
      const existingIndex = state.registered.findIndex(
        comp => comp.id === action.payload.id
      );
      
      if (existingIndex >= 0) {
        state.registered[existingIndex] = action.payload;
      } else {
        state.registered.push(action.payload);
      }
      
      state.lastRegistered = Date.now();
      
      // Clear any previous errors for this component
      delete state.errors[action.payload.id];
    },
    
    unregisterComponent: (state, action: PayloadAction<string>) => {
      state.registered = state.registered.filter(
        comp => comp.id !== action.payload
      );
      state.active = state.active.filter(id => id !== action.payload);
      state.loading = state.loading.filter(id => id !== action.payload);
      delete state.errors[action.payload];
      delete state.performance[action.payload];
    },
    
    enableComponent: (state, action: PayloadAction<string>) => {
      const component = state.registered.find(comp => comp.id === action.payload);
      if (component) {
        component.enabled = true;
        if (!state.active.includes(action.payload)) {
          state.active.push(action.payload);
        }
      }
    },
    
    disableComponent: (state, action: PayloadAction<string>) => {
      const component = state.registered.find(comp => comp.id === action.payload);
      if (component) {
        component.enabled = false;
        state.active = state.active.filter(id => id !== action.payload);
      }
    },
    
    setComponentLoading: (state, action: PayloadAction<{ id: string; loading: boolean }>) => {
      const { id, loading } = action.payload;
      
      if (loading) {
        if (!state.loading.includes(id)) {
          state.loading.push(id);
        }
      } else {
        state.loading = state.loading.filter(loadingId => loadingId !== id);
      }
    },
    
    setComponentError: (state, action: PayloadAction<{ id: string; error: string }>) => {
      state.errors[action.payload.id] = action.payload.error;
      
      // Remove from active and loading if there's an error
      state.active = state.active.filter(id => id !== action.payload.id);
      state.loading = state.loading.filter(id => id !== action.payload.id);
    },
    
    clearComponentError: (state, action: PayloadAction<string>) => {
      delete state.errors[action.payload];
    },
    
    updateComponentProps: (state, action: PayloadAction<{ id: string; props: Record<string, any> }>) => {
      const component = state.registered.find(comp => comp.id === action.payload.id);
      if (component) {
        component.props = { ...component.props, ...action.payload.props };
      }
    },
    
    updateComponentPerformance: (state, action: PayloadAction<{
      id: string;
      renderTime?: number;
      memoryUsage?: number;
    }>) => {
      const { id, renderTime, memoryUsage } = action.payload;
      
      if (!state.performance[id]) {
        state.performance[id] = {
          renderTime: 0,
          memoryUsage: 0,
          updateCount: 0,
          lastUpdate: 0,
        };
      }
      
      const perf = state.performance[id];
      
      if (renderTime !== undefined) {
        perf.renderTime = renderTime;
      }
      
      if (memoryUsage !== undefined) {
        perf.memoryUsage = memoryUsage;
      }
      
      perf.updateCount++;
      perf.lastUpdate = Date.now();
    },
    
    resetComponentPerformance: (state, action: PayloadAction<string>) => {
      if (state.performance[action.payload]) {
        state.performance[action.payload] = {
          renderTime: 0,
          memoryUsage: 0,
          updateCount: 0,
          lastUpdate: Date.now(),
        };
      }
    },
    
    clearAllPerformance: (state) => {
      state.performance = {};
    },
    
    bulkRegisterComponents: (state, action: PayloadAction<ComponentRegistration[]>) => {
      action.payload.forEach(component => {
        const existingIndex = state.registered.findIndex(
          comp => comp.id === component.id
        );
        
        if (existingIndex >= 0) {
          state.registered[existingIndex] = component;
        } else {
          state.registered.push(component);
        }
      });
      
      state.lastRegistered = Date.now();
    },
    
    bulkEnableComponents: (state, action: PayloadAction<string[]>) => {
      action.payload.forEach(id => {
        const component = state.registered.find(comp => comp.id === id);
        if (component) {
          component.enabled = true;
          if (!state.active.includes(id)) {
            state.active.push(id);
          }
        }
      });
    },
    
    bulkDisableComponents: (state, action: PayloadAction<string[]>) => {
      action.payload.forEach(id => {
        const component = state.registered.find(comp => comp.id === id);
        if (component) {
          component.enabled = false;
          state.active = state.active.filter(activeId => activeId !== id);
        }
      });
    },
    
    clearAllErrors: (state) => {
      state.errors = {};
    },
    
    resetComponents: (state) => {
      state.registered = [];
      state.active = [];
      state.loading = [];
      state.errors = {};
      state.performance = {};
      state.lastRegistered = 0;
    },
  },
});

export const {
  registerComponent,
  unregisterComponent,
  enableComponent,
  disableComponent,
  setComponentLoading,
  setComponentError,
  clearComponentError,
  updateComponentProps,
  updateComponentPerformance,
  resetComponentPerformance,
  clearAllPerformance,
  bulkRegisterComponents,
  bulkEnableComponents,
  bulkDisableComponents,
  clearAllErrors,
  resetComponents,
} = componentsSlice.actions;

export default componentsSlice.reducer;

// Selectors
export const selectAllComponents = (state: { components: ComponentsState }) => 
  state.components.registered;

export const selectActiveComponents = (state: { components: ComponentsState }) => 
  state.components.registered.filter(comp => comp.enabled);

export const selectComponentsByPhase = (phase: string) => 
  (state: { components: ComponentsState }) =>
    state.components.registered.filter(comp => comp.phase === phase);

export const selectComponent = (id: string) => 
  (state: { components: ComponentsState }) =>
    state.components.registered.find(comp => comp.id === id);

export const selectComponentsWithErrors = (state: { components: ComponentsState }) => 
  state.components.registered.filter(comp => state.components.errors[comp.id]);

export const selectLoadingComponents = (state: { components: ComponentsState }) => 
  state.components.loading;

export const selectComponentErrors = (state: { components: ComponentsState }) => 
  state.components.errors;

export const selectComponentPerformance = (state: { components: ComponentsState }) => 
  state.components.performance;

export const selectComponentPerformanceById = (id: string) => 
  (state: { components: ComponentsState }) =>
    state.components.performance[id];

export const selectEnabledComponentsCount = (state: { components: ComponentsState }) => 
  state.components.registered.filter(comp => comp.enabled).length;

export const selectComponentsCount = (state: { components: ComponentsState }) => 
  state.components.registered.length;