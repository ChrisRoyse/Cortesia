import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ConnectionStatus, VisualizationUpdate, ErrorInfo } from '@/types';

interface SystemState {
  connectionStatus: ConnectionStatus;
  connected: boolean;
  lastConnected: number | null;
  lastUpdate: number;
  errors: ErrorInfo[];
  updates: VisualizationUpdate[];
  systemHealth: {
    overall: number;
    components: Record<string, number>;
    lastCheck: number;
  };
  metrics: {
    memoryUsage: number;
    cpuUsage: number;
    networkLatency: number;
    throughput: number;
    errorRate: number;
    uptime: number;
  };
  alerts: {
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: number;
    dismissed: boolean;
  }[];
}

const initialState: SystemState = {
  connectionStatus: 'disconnected',
  connected: false,
  lastConnected: null,
  lastUpdate: 0,
  errors: [],
  updates: [],
  systemHealth: {
    overall: 0,
    components: {},
    lastCheck: 0,
  },
  metrics: {
    memoryUsage: 0,
    cpuUsage: 0,
    networkLatency: 0,
    throughput: 0,
    errorRate: 0,
    uptime: 0,
  },
  alerts: [],
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setConnectionStatus: (state, action: PayloadAction<ConnectionStatus>) => {
      state.connectionStatus = action.payload;
      state.connected = action.payload === 'connected';
      
      if (action.payload === 'connected') {
        state.lastConnected = Date.now();
        state.errors = state.errors.filter(error => error.type !== 'connection');
      }
      
      state.lastUpdate = Date.now();
    },
    
    addError: (state, action: PayloadAction<ErrorInfo>) => {
      state.errors.push(action.payload);
      
      // Keep only the last 100 errors
      if (state.errors.length > 100) {
        state.errors = state.errors.slice(-100);
      }
      
      // Add alert for critical errors
      if (action.payload.severity === 'critical' || action.payload.severity === 'high') {
        state.alerts.push({
          id: action.payload.id,
          type: 'error',
          message: action.payload.message,
          timestamp: action.payload.timestamp,
          dismissed: false,
        });
      }
    },
    
    clearErrors: (state) => {
      state.errors = [];
    },
    
    removeError: (state, action: PayloadAction<string>) => {
      state.errors = state.errors.filter(error => error.id !== action.payload);
    },
    
    addUpdate: (state, action: PayloadAction<VisualizationUpdate>) => {
      state.updates.push(action.payload);
      state.lastUpdate = action.payload.timestamp;
      
      // Keep only the last 1000 updates
      if (state.updates.length > 1000) {
        state.updates = state.updates.slice(-1000);
      }
    },
    
    clearUpdates: (state) => {
      state.updates = [];
    },
    
    updateSystemHealth: (state, action: PayloadAction<{
      overall: number;
      components: Record<string, number>;
    }>) => {
      state.systemHealth.overall = action.payload.overall;
      state.systemHealth.components = action.payload.components;
      state.systemHealth.lastCheck = Date.now();
      
      // Add alert if health drops below threshold
      if (action.payload.overall < 70) {
        state.alerts.push({
          id: `health-${Date.now()}`,
          type: action.payload.overall < 50 ? 'error' : 'warning',
          message: `System health dropped to ${action.payload.overall}%`,
          timestamp: Date.now(),
          dismissed: false,
        });
      }
    },
    
    updateMetrics: (state, action: PayloadAction<Partial<SystemState['metrics']>>) => {
      state.metrics = { ...state.metrics, ...action.payload };
      
      // Check for performance alerts
      if (action.payload.memoryUsage && action.payload.memoryUsage > 90) {
        state.alerts.push({
          id: `memory-${Date.now()}`,
          type: 'warning',
          message: `High memory usage: ${action.payload.memoryUsage}%`,
          timestamp: Date.now(),
          dismissed: false,
        });
      }
      
      if (action.payload.errorRate && action.payload.errorRate > 5) {
        state.alerts.push({
          id: `errors-${Date.now()}`,
          type: 'error',
          message: `High error rate: ${action.payload.errorRate}%`,
          timestamp: Date.now(),
          dismissed: false,
        });
      }
    },
    
    addAlert: (state, action: PayloadAction<{
      type: 'info' | 'warning' | 'error' | 'success';
      message: string;
    }>) => {
      state.alerts.push({
        id: `alert-${Date.now()}`,
        type: action.payload.type,
        message: action.payload.message,
        timestamp: Date.now(),
        dismissed: false,
      });
      
      // Keep only the last 50 alerts
      if (state.alerts.length > 50) {
        state.alerts = state.alerts.slice(-50);
      }
    },
    
    dismissAlert: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(alert => alert.id === action.payload);
      if (alert) {
        alert.dismissed = true;
      }
    },
    
    clearAlerts: (state) => {
      state.alerts = [];
    },
    
    clearDismissedAlerts: (state) => {
      state.alerts = state.alerts.filter(alert => !alert.dismissed);
    },
    
    resetSystem: (state) => {
      state.connectionStatus = 'disconnected';
      state.connected = false;
      state.lastConnected = null;
      state.errors = [];
      state.updates = [];
      state.alerts = [];
      state.systemHealth = {
        overall: 0,
        components: {},
        lastCheck: 0,
      };
      state.metrics = {
        memoryUsage: 0,
        cpuUsage: 0,
        networkLatency: 0,
        throughput: 0,
        errorRate: 0,
        uptime: 0,
      };
      state.lastUpdate = Date.now();
    },
  },
});

export const {
  setConnectionStatus,
  addError,
  clearErrors,
  removeError,
  addUpdate,
  clearUpdates,
  updateSystemHealth,
  updateMetrics,
  addAlert,
  dismissAlert,
  clearAlerts,
  clearDismissedAlerts,
  resetSystem,
} = systemSlice.actions;

export default systemSlice.reducer;

// Selectors
export const selectConnectionStatus = (state: { system: SystemState }) => 
  state.system.connectionStatus;

export const selectConnected = (state: { system: SystemState }) => 
  state.system.connected;

export const selectLastConnected = (state: { system: SystemState }) => 
  state.system.lastConnected;

export const selectErrors = (state: { system: SystemState }) => 
  state.system.errors;

export const selectRecentErrors = (state: { system: SystemState }) => 
  state.system.errors.slice(-10);

export const selectCriticalErrors = (state: { system: SystemState }) => 
  state.system.errors.filter(error => 
    error.severity === 'critical' || error.severity === 'high'
  );

export const selectUpdates = (state: { system: SystemState }) => 
  state.system.updates;

export const selectRecentUpdates = (state: { system: SystemState }) => 
  state.system.updates.slice(-20);

export const selectSystemHealth = (state: { system: SystemState }) => 
  state.system.systemHealth;

export const selectMetrics = (state: { system: SystemState }) => 
  state.system.metrics;

export const selectAlerts = (state: { system: SystemState }) => 
  state.system.alerts;

export const selectActiveAlerts = (state: { system: SystemState }) => 
  state.system.alerts.filter(alert => !alert.dismissed);

export const selectLastUpdate = (state: { system: SystemState }) => 
  state.system.lastUpdate;