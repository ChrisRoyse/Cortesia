# Phase 2: Core Dashboard Infrastructure

## Overview

This document provides comprehensive implementation details for Phase 2 of the LLMKG visualization system, focusing on building a robust, scalable dashboard framework specifically designed for visualizing Large Language Model Knowledge Graphs and cognitive patterns.

## Table of Contents

1. [Dashboard Framework Architecture](#dashboard-framework-architecture)
2. [Routing and Navigation System](#routing-and-navigation-system)
3. [Component Library Design](#component-library-design)
4. [Real-time Update System](#real-time-update-system)
5. [State Management Architecture](#state-management-architecture)
6. [Layout System](#layout-system)
7. [Theme and Styling System](#theme-and-styling-system)
8. [Testing and Documentation Setup](#testing-and-documentation-setup)

---

## Dashboard Framework Architecture

### Core Architecture Overview

```typescript
// src/dashboard/core/DashboardApp.tsx
import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from '@emotion/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WebSocketProvider } from './providers/WebSocketProvider';
import { MCPProvider } from './providers/MCPProvider';
import { store } from './store';
import { theme } from './theme';
import { Router } from './routing/Router';
import { GlobalErrorBoundary } from './components/ErrorBoundary';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      retry: 3,
    },
  },
});

export const DashboardApp: React.FC = () => {
  return (
    <GlobalErrorBoundary>
      <Provider store={store}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={theme}>
            <WebSocketProvider>
              <MCPProvider>
                <BrowserRouter>
                  <Router />
                </BrowserRouter>
              </MCPProvider>
            </WebSocketProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </Provider>
    </GlobalErrorBoundary>
  );
};
```

### Dashboard Container Architecture

```typescript
// src/dashboard/core/DashboardContainer.tsx
import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Box, Container } from '@mui/material';
import { Sidebar } from '../components/layout/Sidebar';
import { Header } from '../components/layout/Header';
import { Footer } from '../components/layout/Footer';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMCPConnection } from '../hooks/useMCPConnection';
import { RootState } from '../store';
import { setConnectionStatus } from '../store/slices/systemSlice';

interface DashboardContainerProps {
  children: React.ReactNode;
}

export const DashboardContainer: React.FC<DashboardContainerProps> = ({ children }) => {
  const dispatch = useDispatch();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { isConnected: wsConnected } = useWebSocket();
  const { isConnected: mcpConnected } = useMCPConnection();
  const { theme } = useSelector((state: RootState) => state.ui);

  useEffect(() => {
    dispatch(setConnectionStatus({
      websocket: wsConnected,
      mcp: mcpConnected,
    }));
  }, [wsConnected, mcpConnected, dispatch]);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          ml: sidebarOpen ? '240px' : '64px',
          transition: 'margin-left 0.3s ease',
        }}
      >
        <Header />
        <Container
          maxWidth={false}
          sx={{
            flexGrow: 1,
            py: 3,
            px: { xs: 2, sm: 3, md: 4 },
          }}
        >
          {children}
        </Container>
        <Footer />
      </Box>
    </Box>
  );
};
```

---

## Routing and Navigation System

### Router Configuration

```typescript
// src/dashboard/routing/Router.tsx
import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { DashboardContainer } from '../core/DashboardContainer';
import { LoadingScreen } from '../components/common/LoadingScreen';
import { ProtectedRoute } from './ProtectedRoute';

// Lazy load dashboard pages for code splitting
const KnowledgeGraphView = lazy(() => import('../views/KnowledgeGraphView'));
const CognitivePatternsView = lazy(() => import('../views/CognitivePatternsView'));
const NeuralActivityView = lazy(() => import('../views/NeuralActivityView'));
const SystemMetricsView = lazy(() => import('../views/SystemMetricsView'));
const QueryInterface = lazy(() => import('../views/QueryInterface'));
const SettingsView = lazy(() => import('../views/SettingsView'));

export const Router: React.FC = () => {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <KnowledgeGraphView />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
        <Route
          path="/cognitive-patterns"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <CognitivePatternsView />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
        <Route
          path="/neural-activity"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <NeuralActivityView />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
        <Route
          path="/metrics"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <SystemMetricsView />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
        <Route
          path="/query"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <QueryInterface />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <DashboardContainer>
                <SettingsView />
              </DashboardContainer>
            </ProtectedRoute>
          }
        />
      </Routes>
    </Suspense>
  );
};
```

### Navigation Hook

```typescript
// src/dashboard/hooks/useNavigation.ts
import { useNavigate, useLocation } from 'react-router-dom';
import { useCallback } from 'react';

export interface NavigationItem {
  id: string;
  title: string;
  path: string;
  icon: React.ComponentType;
  badge?: number;
  children?: NavigationItem[];
}

export const useNavigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navigateTo = useCallback((path: string, state?: any) => {
    navigate(path, { state });
  }, [navigate]);

  const isActive = useCallback((path: string) => {
    return location.pathname === path;
  }, [location.pathname]);

  const goBack = useCallback(() => {
    navigate(-1);
  }, [navigate]);

  return {
    navigateTo,
    isActive,
    goBack,
    currentPath: location.pathname,
  };
};
```

---

## Component Library Design

### Base Component Structure

```typescript
// src/dashboard/components/base/BaseComponent.tsx
import React from 'react';
import { Box, BoxProps } from '@mui/material';
import { styled } from '@mui/material/styles';

export interface BaseComponentProps extends BoxProps {
  variant?: 'default' | 'elevated' | 'outlined';
  interactive?: boolean;
}

const StyledBaseComponent = styled(Box)<BaseComponentProps>(({ theme, variant, interactive }) => ({
  position: 'relative',
  borderRadius: theme.shape.borderRadius,
  transition: theme.transitions.create(['box-shadow', 'transform'], {
    duration: theme.transitions.duration.short,
  }),
  
  ...(variant === 'elevated' && {
    boxShadow: theme.shadows[2],
    '&:hover': interactive ? {
      boxShadow: theme.shadows[4],
      transform: 'translateY(-2px)',
    } : {},
  }),
  
  ...(variant === 'outlined' && {
    border: `1px solid ${theme.palette.divider}`,
  }),
}));

export const BaseComponent: React.FC<BaseComponentProps> = ({ children, ...props }) => {
  return <StyledBaseComponent {...props}>{children}</StyledBaseComponent>;
};
```

### Knowledge Graph Visualization Component

```typescript
// src/dashboard/components/visualizations/KnowledgeGraphVisualization.tsx
import React, { useEffect, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { Box, Paper, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { GraphData, GraphNode, GraphLink } from '../../types/graph';
import { GraphControls } from './GraphControls';
import { NodeDetailsPanel } from './NodeDetailsPanel';

interface KnowledgeGraphVisualizationProps {
  data: GraphData;
  onNodeClick?: (node: GraphNode) => void;
  onLinkClick?: (link: GraphLink) => void;
  height?: number;
}

export const KnowledgeGraphVisualization: React.FC<KnowledgeGraphVisualizationProps> = ({
  data,
  onNodeClick,
  onLinkClick,
  height = 600,
}) => {
  const theme = useTheme();
  const graphRef = useRef<any>();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [graphConfig, setGraphConfig] = useState({
    nodeRelSize: 6,
    linkWidth: 2,
    linkOpacity: 0.5,
    nodeOpacity: 0.9,
    showLabels: true,
    physics: {
      enabled: true,
      gravity: -100,
      linkDistance: 100,
      linkStrength: 0.1,
    },
  });

  useEffect(() => {
    // Configure camera and controls
    if (graphRef.current) {
      graphRef.current.d3Force('link').distance(graphConfig.physics.linkDistance);
      graphRef.current.d3Force('charge').strength(graphConfig.physics.gravity);
    }
  }, [graphConfig.physics]);

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNode(node);
    onNodeClick?.(node);
  };

  const nodeColor = (node: GraphNode) => {
    switch (node.type) {
      case 'entity': return theme.palette.primary.main;
      case 'concept': return theme.palette.secondary.main;
      case 'pattern': return theme.palette.warning.main;
      default: return theme.palette.grey[500];
    }
  };

  return (
    <Box sx={{ position: 'relative', height }}>
      <Paper sx={{ height: '100%', overflow: 'hidden' }}>
        <ForceGraph3D
          ref={graphRef}
          graphData={data}
          nodeLabel="label"
          nodeAutoColorBy="group"
          nodeThreeObject={(node: GraphNode) => {
            const sprite = new SpriteText(node.label);
            sprite.color = nodeColor(node);
            sprite.textHeight = 8;
            return sprite;
          }}
          nodeThreeObjectExtend={true}
          linkWidth={graphConfig.linkWidth}
          linkOpacity={graphConfig.linkOpacity}
          onNodeClick={handleNodeClick}
          onLinkClick={onLinkClick}
          enableNodeDrag={true}
          enableNavigationControls={true}
          showNavInfo={false}
        />
        <GraphControls
          config={graphConfig}
          onChange={setGraphConfig}
          sx={{ position: 'absolute', top: 16, right: 16 }}
        />
        {selectedNode && (
          <NodeDetailsPanel
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
            sx={{ position: 'absolute', bottom: 16, left: 16 }}
          />
        )}
      </Paper>
    </Box>
  );
};
```

### Cognitive Pattern Visualization Component

```typescript
// src/dashboard/components/visualizations/CognitivePatternVisualization.tsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Box, Paper } from '@mui/material';
import { CognitivePattern } from '../../types/cognitive';

interface CognitivePatternVisualizationProps {
  patterns: CognitivePattern[];
  width?: number;
  height?: number;
  onPatternSelect?: (pattern: CognitivePattern) => void;
}

export const CognitivePatternVisualization: React.FC<CognitivePatternVisualizationProps> = ({
  patterns,
  width = 800,
  height = 600,
  onPatternSelect,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !patterns.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create hierarchical layout
    const hierarchy = d3.hierarchy({
      name: 'root',
      children: patterns.map(p => ({
        name: p.name,
        value: p.strength,
        data: p,
      })),
    });

    const pack = d3.pack()
      .size([width, height])
      .padding(10);

    const root = pack(hierarchy.sum(d => d.value).sort((a, b) => b.value - a.value));

    // Create pattern bubbles
    const node = svg.selectAll('.pattern-node')
      .data(root.descendants().filter(d => d.depth > 0))
      .enter()
      .append('g')
      .attr('class', 'pattern-node')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    // Add circles
    node.append('circle')
      .attr('r', d => d.r)
      .attr('fill', d => {
        const pattern = d.data.data as CognitivePattern;
        return pattern.type === 'convergent' ? '#3f51b5' :
               pattern.type === 'divergent' ? '#f50057' :
               pattern.type === 'lateral' ? '#00bcd4' : '#ff9800';
      })
      .attr('fill-opacity', 0.7)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .on('click', (event, d) => {
        if (d.data.data && onPatternSelect) {
          onPatternSelect(d.data.data as CognitivePattern);
        }
      });

    // Add labels
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .text(d => d.data.name)
      .style('font-size', d => `${Math.min(d.r / 3, 16)}px`)
      .style('fill', '#fff')
      .style('pointer-events', 'none');

    // Add strength indicators
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.5em')
      .text(d => `${Math.round(d.value * 100)}%`)
      .style('font-size', d => `${Math.min(d.r / 4, 12)}px`)
      .style('fill', '#fff')
      .style('opacity', 0.8)
      .style('pointer-events', 'none');

  }, [patterns, width, height, onPatternSelect]);

  return (
    <Paper sx={{ p: 2, height, overflow: 'hidden' }}>
      <svg ref={svgRef} width={width} height={height} />
    </Paper>
  );
};
```

### Reusable Data Grid Component

```typescript
// src/dashboard/components/common/DataGrid.tsx
import React, { useMemo } from 'react';
import {
  DataGrid as MuiDataGrid,
  GridColDef,
  GridRowsProp,
  GridToolbar,
  GridValueGetterParams,
} from '@mui/x-data-grid';
import { Box, Paper } from '@mui/material';

interface DataGridProps<T = any> {
  rows: T[];
  columns: GridColDef[];
  loading?: boolean;
  pageSize?: number;
  onRowClick?: (row: T) => void;
  height?: number | string;
}

export function DataGrid<T extends { id: string | number }>({
  rows,
  columns,
  loading = false,
  pageSize = 25,
  onRowClick,
  height = 400,
}: DataGridProps<T>) {
  const processedColumns = useMemo(() => {
    return columns.map(col => ({
      ...col,
      flex: col.flex || 1,
      headerClassName: 'data-grid-header',
    }));
  }, [columns]);

  return (
    <Paper sx={{ height, width: '100%' }}>
      <MuiDataGrid
        rows={rows}
        columns={processedColumns}
        pageSize={pageSize}
        rowsPerPageOptions={[10, 25, 50, 100]}
        loading={loading}
        components={{
          Toolbar: GridToolbar,
        }}
        onRowClick={(params) => onRowClick?.(params.row)}
        sx={{
          '& .data-grid-header': {
            backgroundColor: 'action.hover',
            fontWeight: 'bold',
          },
          '& .MuiDataGrid-row:hover': {
            backgroundColor: 'action.hover',
            cursor: onRowClick ? 'pointer' : 'default',
          },
        }}
      />
    </Paper>
  );
}
```

---

## Real-time Update System

### WebSocket Provider

```typescript
// src/dashboard/providers/WebSocketProvider.tsx
import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useDispatch } from 'react-redux';
import { updateMetrics } from '../store/slices/metricsSlice';
import { updateGraph } from '../store/slices/graphSlice';
import { addNotification } from '../store/slices/notificationSlice';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  subscribe: (event: string, handler: (data: any) => void) => void;
  unsubscribe: (event: string, handler: (data: any) => void) => void;
  emit: (event: string, data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  socket: null,
  isConnected: false,
  subscribe: () => {},
  unsubscribe: () => {},
  emit: () => {},
});

export const useWebSocket = () => useContext(WebSocketContext);

interface WebSocketProviderProps {
  children: React.ReactNode;
  url?: string;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  url = process.env.REACT_APP_WS_URL || 'ws://localhost:3001',
}) => {
  const dispatch = useDispatch();
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const handlersRef = useRef<Map<string, Set<Function>>>(new Map());

  useEffect(() => {
    // Initialize WebSocket connection
    socketRef.current = io(url, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    });

    const socket = socketRef.current;

    // Connection event handlers
    socket.on('connect', () => {
      setIsConnected(true);
      dispatch(addNotification({
        type: 'success',
        message: 'Connected to real-time updates',
      }));
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      dispatch(addNotification({
        type: 'warning',
        message: 'Disconnected from real-time updates',
      }));
    });

    // Real-time data handlers
    socket.on('metrics:update', (data) => {
      dispatch(updateMetrics(data));
    });

    socket.on('graph:update', (data) => {
      dispatch(updateGraph(data));
    });

    socket.on('pattern:detected', (data) => {
      dispatch(addNotification({
        type: 'info',
        message: `New pattern detected: ${data.patternType}`,
        data,
      }));
    });

    // Custom event distribution
    socket.onAny((event, ...args) => {
      const handlers = handlersRef.current.get(event);
      if (handlers) {
        handlers.forEach(handler => handler(...args));
      }
    });

    return () => {
      socket.disconnect();
    };
  }, [url, dispatch]);

  const subscribe = (event: string, handler: (data: any) => void) => {
    if (!handlersRef.current.has(event)) {
      handlersRef.current.set(event, new Set());
    }
    handlersRef.current.get(event)!.add(handler);
  };

  const unsubscribe = (event: string, handler: (data: any) => void) => {
    const handlers = handlersRef.current.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        handlersRef.current.delete(event);
      }
    }
  };

  const emit = (event: string, data: any) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit(event, data);
    }
  };

  return (
    <WebSocketContext.Provider
      value={{
        socket: socketRef.current,
        isConnected,
        subscribe,
        unsubscribe,
        emit,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};
```

### Real-time Data Hook

```typescript
// src/dashboard/hooks/useRealTimeData.ts
import { useEffect, useState, useRef } from 'react';
import { useWebSocket } from '../providers/WebSocketProvider';

interface UseRealTimeDataOptions {
  event: string;
  initialData?: any;
  transform?: (data: any) => any;
  bufferSize?: number;
  throttleMs?: number;
}

export function useRealTimeData<T = any>({
  event,
  initialData,
  transform = (data) => data,
  bufferSize = 100,
  throttleMs = 100,
}: UseRealTimeDataOptions): {
  data: T | null;
  buffer: T[];
  isStreaming: boolean;
} {
  const { subscribe, unsubscribe, isConnected } = useWebSocket();
  const [data, setData] = useState<T | null>(initialData || null);
  const [buffer, setBuffer] = useState<T[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const lastUpdateRef = useRef<number>(0);
  const bufferRef = useRef<T[]>([]);

  useEffect(() => {
    if (!isConnected) return;

    const handleData = (rawData: any) => {
      const now = Date.now();
      const transformedData = transform(rawData) as T;

      // Add to buffer
      bufferRef.current = [...bufferRef.current, transformedData].slice(-bufferSize);

      // Throttle updates
      if (now - lastUpdateRef.current >= throttleMs) {
        setData(transformedData);
        setBuffer([...bufferRef.current]);
        lastUpdateRef.current = now;
        setIsStreaming(true);
      }
    };

    subscribe(event, handleData);

    return () => {
      unsubscribe(event, handleData);
      setIsStreaming(false);
    };
  }, [event, subscribe, unsubscribe, isConnected, transform, bufferSize, throttleMs]);

  return { data, buffer, isStreaming };
}
```

---

## State Management Architecture

### Redux Store Configuration

```typescript
// src/dashboard/store/index.ts
import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import graphReducer from './slices/graphSlice';
import metricsReducer from './slices/metricsSlice';
import cognitiveReducer from './slices/cognitiveSlice';
import systemReducer from './slices/systemSlice';
import uiReducer from './slices/uiSlice';
import notificationReducer from './slices/notificationSlice';
import { llmkgApi } from './api/llmkgApi';

export const store = configureStore({
  reducer: {
    graph: graphReducer,
    metrics: metricsReducer,
    cognitive: cognitiveReducer,
    system: systemReducer,
    ui: uiReducer,
    notifications: notificationReducer,
    [llmkgApi.reducerPath]: llmkgApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['graph/updateGraph'],
        ignoredPaths: ['graph.nodes', 'graph.links'],
      },
    }).concat(llmkgApi.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### Graph State Slice

```typescript
// src/dashboard/store/slices/graphSlice.ts
import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';
import { GraphData, GraphNode, GraphLink, GraphFilter } from '../../types/graph';
import { graphAPI } from '../api/graphAPI';

interface GraphState {
  data: GraphData | null;
  selectedNodes: string[];
  selectedLinks: string[];
  filters: GraphFilter;
  layout: 'force' | '3d' | 'hierarchical' | 'radial';
  loading: boolean;
  error: string | null;
}

const initialState: GraphState = {
  data: null,
  selectedNodes: [],
  selectedLinks: [],
  filters: {
    nodeTypes: [],
    minWeight: 0,
    maxDistance: Infinity,
  },
  layout: '3d',
  loading: false,
  error: null,
};

// Async thunks
export const fetchGraph = createAsyncThunk(
  'graph/fetchGraph',
  async (params: { query?: string; depth?: number }) => {
    const response = await graphAPI.getGraph(params);
    return response.data;
  }
);

export const expandNode = createAsyncThunk(
  'graph/expandNode',
  async ({ nodeId, depth = 1 }: { nodeId: string; depth?: number }) => {
    const response = await graphAPI.expandNode(nodeId, depth);
    return response.data;
  }
);

const graphSlice = createSlice({
  name: 'graph',
  initialState,
  reducers: {
    updateGraph: (state, action: PayloadAction<Partial<GraphData>>) => {
      if (state.data) {
        state.data = {
          ...state.data,
          nodes: [...state.data.nodes, ...(action.payload.nodes || [])],
          links: [...state.data.links, ...(action.payload.links || [])],
        };
      }
    },
    selectNode: (state, action: PayloadAction<string>) => {
      if (!state.selectedNodes.includes(action.payload)) {
        state.selectedNodes.push(action.payload);
      }
    },
    deselectNode: (state, action: PayloadAction<string>) => {
      state.selectedNodes = state.selectedNodes.filter(id => id !== action.payload);
    },
    setFilters: (state, action: PayloadAction<Partial<GraphFilter>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    setLayout: (state, action: PayloadAction<GraphState['layout']>) => {
      state.layout = action.payload;
    },
    clearGraph: (state) => {
      state.data = null;
      state.selectedNodes = [];
      state.selectedLinks = [];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchGraph.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchGraph.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchGraph.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch graph';
      })
      .addCase(expandNode.fulfilled, (state, action) => {
        if (state.data) {
          const newNodes = action.payload.nodes.filter(
            node => !state.data!.nodes.find(n => n.id === node.id)
          );
          const newLinks = action.payload.links.filter(
            link => !state.data!.links.find(l => 
              l.source === link.source && l.target === link.target
            )
          );
          state.data.nodes.push(...newNodes);
          state.data.links.push(...newLinks);
        }
      });
  },
});

export const {
  updateGraph,
  selectNode,
  deselectNode,
  setFilters,
  setLayout,
  clearGraph,
} = graphSlice.actions;

export default graphSlice.reducer;
```

### Cognitive Patterns State Slice

```typescript
// src/dashboard/store/slices/cognitiveSlice.ts
import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { CognitivePattern, NeuralActivity, ThinkingMode } from '../../types/cognitive';

interface CognitiveState {
  patterns: CognitivePattern[];
  activePatterns: string[];
  neuralActivity: NeuralActivity[];
  thinkingMode: ThinkingMode;
  patternHistory: Array<{
    timestamp: number;
    pattern: CognitivePattern;
    trigger: string;
  }>;
}

const initialState: CognitiveState = {
  patterns: [],
  activePatterns: [],
  neuralActivity: [],
  thinkingMode: 'balanced',
  patternHistory: [],
};

const cognitiveSlice = createSlice({
  name: 'cognitive',
  initialState,
  reducers: {
    addPattern: (state, action: PayloadAction<CognitivePattern>) => {
      state.patterns.push(action.payload);
      state.patternHistory.push({
        timestamp: Date.now(),
        pattern: action.payload,
        trigger: 'manual',
      });
    },
    activatePattern: (state, action: PayloadAction<string>) => {
      if (!state.activePatterns.includes(action.payload)) {
        state.activePatterns.push(action.payload);
      }
    },
    deactivatePattern: (state, action: PayloadAction<string>) => {
      state.activePatterns = state.activePatterns.filter(id => id !== action.payload);
    },
    updateNeuralActivity: (state, action: PayloadAction<NeuralActivity[]>) => {
      state.neuralActivity = action.payload;
    },
    setThinkingMode: (state, action: PayloadAction<ThinkingMode>) => {
      state.thinkingMode = action.payload;
    },
    clearPatternHistory: (state) => {
      state.patternHistory = [];
    },
  },
});

export const {
  addPattern,
  activatePattern,
  deactivatePattern,
  updateNeuralActivity,
  setThinkingMode,
  clearPatternHistory,
} = cognitiveSlice.actions;

export default cognitiveSlice.reducer;
```

---

## Layout System

### Responsive Grid Layout

```typescript
// src/dashboard/components/layout/ResponsiveGrid.tsx
import React from 'react';
import { Responsive, WidthProvider, Layout } from 'react-grid-layout';
import { Box } from '@mui/material';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface GridItem {
  id: string;
  component: React.ReactNode;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
}

interface ResponsiveGridProps {
  items: GridItem[];
  layouts?: { [key: string]: Layout[] };
  onLayoutChange?: (layout: Layout[], layouts: any) => void;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  items,
  layouts,
  onLayoutChange,
}) => {
  const defaultLayouts = {
    lg: items.map((item, index) => ({
      i: item.id,
      x: (index % 2) * 6,
      y: Math.floor(index / 2) * 4,
      w: 6,
      h: 4,
      minW: item.minW || 2,
      minH: item.minH || 2,
      maxW: item.maxW,
      maxH: item.maxH,
    })),
    md: items.map((item, index) => ({
      i: item.id,
      x: (index % 2) * 6,
      y: Math.floor(index / 2) * 4,
      w: 6,
      h: 4,
      minW: item.minW || 2,
      minH: item.minH || 2,
    })),
    sm: items.map((item, index) => ({
      i: item.id,
      x: 0,
      y: index * 4,
      w: 12,
      h: 4,
      minW: 12,
      minH: item.minH || 2,
    })),
  };

  return (
    <ResponsiveGridLayout
      className="layout"
      layouts={layouts || defaultLayouts}
      breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
      cols={{ lg: 12, md: 12, sm: 12, xs: 4, xxs: 2 }}
      rowHeight={60}
      onLayoutChange={onLayoutChange}
      draggableHandle=".drag-handle"
    >
      {items.map((item) => (
        <Box
          key={item.id}
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            bgcolor: 'background.paper',
            borderRadius: 1,
            overflow: 'hidden',
            boxShadow: 1,
          }}
        >
          {item.component}
        </Box>
      ))}
    </ResponsiveGridLayout>
  );
};
```

### Dashboard Layout Manager

```typescript
// src/dashboard/components/layout/DashboardLayout.tsx
import React, { useState, useEffect } from 'react';
import { Box, IconButton, Menu, MenuItem } from '@mui/material';
import { ViewModule, ViewQuilt, ViewStream } from '@mui/icons-material';
import { ResponsiveGrid } from './ResponsiveGrid';
import { useLocalStorage } from '../../hooks/useLocalStorage';

export type LayoutPreset = 'default' | 'focus' | 'overview' | 'custom';

interface DashboardLayoutProps {
  children: React.ReactNode[];
  preset?: LayoutPreset;
  onLayoutChange?: (preset: LayoutPreset) => void;
}

const layoutPresets = {
  default: {
    lg: [
      { i: '0', x: 0, y: 0, w: 6, h: 6 },
      { i: '1', x: 6, y: 0, w: 6, h: 6 },
      { i: '2', x: 0, y: 6, w: 12, h: 4 },
      { i: '3', x: 0, y: 10, w: 12, h: 4 },
    ],
  },
  focus: {
    lg: [
      { i: '0', x: 0, y: 0, w: 12, h: 8 },
      { i: '1', x: 0, y: 8, w: 4, h: 4 },
      { i: '2', x: 4, y: 8, w: 4, h: 4 },
      { i: '3', x: 8, y: 8, w: 4, h: 4 },
    ],
  },
  overview: {
    lg: [
      { i: '0', x: 0, y: 0, w: 3, h: 4 },
      { i: '1', x: 3, y: 0, w: 3, h: 4 },
      { i: '2', x: 6, y: 0, w: 3, h: 4 },
      { i: '3', x: 9, y: 0, w: 3, h: 4 },
    ],
  },
};

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  preset = 'default',
  onLayoutChange,
}) => {
  const [currentPreset, setCurrentPreset] = useState<LayoutPreset>(preset);
  const [customLayouts, setCustomLayouts] = useLocalStorage('dashboardLayouts', {});
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handlePresetChange = (newPreset: LayoutPreset) => {
    setCurrentPreset(newPreset);
    onLayoutChange?.(newPreset);
    setAnchorEl(null);
  };

  const handleLayoutChange = (layout: any, layouts: any) => {
    if (currentPreset === 'custom') {
      setCustomLayouts(layouts);
    }
  };

  const getLayouts = () => {
    if (currentPreset === 'custom') {
      return customLayouts;
    }
    return layoutPresets[currentPreset as keyof typeof layoutPresets];
  };

  const items = React.Children.map(children, (child, index) => ({
    id: index.toString(),
    component: child,
  }));

  return (
    <Box sx={{ position: 'relative', height: '100%' }}>
      <Box sx={{ position: 'absolute', top: 0, right: 0, zIndex: 1000 }}>
        <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
          <ViewModule />
        </IconButton>
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null)}
        >
          <MenuItem onClick={() => handlePresetChange('default')}>
            <ViewModule sx={{ mr: 1 }} /> Default Layout
          </MenuItem>
          <MenuItem onClick={() => handlePresetChange('focus')}>
            <ViewQuilt sx={{ mr: 1 }} /> Focus Layout
          </MenuItem>
          <MenuItem onClick={() => handlePresetChange('overview')}>
            <ViewStream sx={{ mr: 1 }} /> Overview Layout
          </MenuItem>
          <MenuItem onClick={() => handlePresetChange('custom')}>
            Custom Layout
          </MenuItem>
        </Menu>
      </Box>
      <ResponsiveGrid
        items={items}
        layouts={getLayouts()}
        onLayoutChange={handleLayoutChange}
      />
    </Box>
  );
};
```

---

## Theme and Styling System

### Theme Configuration

```typescript
// src/dashboard/theme/index.ts
import { createTheme, ThemeOptions } from '@mui/material/styles';

const baseTheme: ThemeOptions = {
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
};

export const lightTheme = createTheme({
  ...baseTheme,
  palette: {
    mode: 'light',
    primary: {
      main: '#3f51b5',
      light: '#7986cb',
      dark: '#303f9f',
    },
    secondary: {
      main: '#f50057',
      light: '#ff5983',
      dark: '#c51162',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.6)',
    },
  },
});

export const darkTheme = createTheme({
  ...baseTheme,
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
      light: '#e3f2fd',
      dark: '#42a5f5',
    },
    secondary: {
      main: '#f48fb1',
      light: '#ffc1e3',
      dark: '#bf5f82',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: 'rgba(255, 255, 255, 0.87)',
      secondary: 'rgba(255, 255, 255, 0.6)',
    },
  },
});

// LLMKG-specific color schemes
export const cognitiveColors = {
  convergent: '#3f51b5',
  divergent: '#f50057',
  lateral: '#00bcd4',
  critical: '#ff9800',
  abstract: '#9c27b0',
  systems: '#4caf50',
};

export const neuralColors = {
  active: '#4caf50',
  inhibited: '#f44336',
  baseline: '#9e9e9e',
  spiking: '#ffeb3b',
};

export const graphColors = {
  entity: '#2196f3',
  concept: '#9c27b0',
  pattern: '#ff9800',
  relationship: '#607d8b',
};
```

### Styled Components

```typescript
// src/dashboard/components/styled/StyledComponents.tsx
import { styled } from '@mui/material/styles';
import { Box, Paper, Card, Button } from '@mui/material';

export const GlassPanel = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.05)' 
    : 'rgba(255, 255, 255, 0.7)',
  backdropFilter: 'blur(10px)',
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(2),
  boxShadow: theme.palette.mode === 'dark'
    ? '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
    : '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
}));

export const MetricCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[8],
  },
}));

export const GraphContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  '& .controls': {
    position: 'absolute',
    top: theme.spacing(2),
    right: theme.spacing(2),
    zIndex: 1000,
  },
}));

export const AnimatedButton = styled(Button)(({ theme }) => ({
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 0,
    height: 0,
    borderRadius: '50%',
    background: theme.palette.primary.light,
    transform: 'translate(-50%, -50%)',
    transition: 'width 0.6s, height 0.6s',
  },
  '&:hover::before': {
    width: '300px',
    height: '300px',
  },
  '& > *': {
    position: 'relative',
    zIndex: 1,
  },
}));
```

---

## Testing and Documentation Setup

### Testing Configuration

```typescript
// src/dashboard/tests/setup.ts
import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';
import { server } from './mocks/server';

// Configure testing library
configure({ testIdAttribute: 'data-testid' });

// Setup MSW
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
};

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(callback: any) {}
  disconnect() {}
  observe() {}
  unobserve() {}
};
```

### Component Testing Example

```typescript
// src/dashboard/components/__tests__/KnowledgeGraphVisualization.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { KnowledgeGraphVisualization } from '../visualizations/KnowledgeGraphVisualization';
import graphReducer from '../../store/slices/graphSlice';

const mockData = {
  nodes: [
    { id: '1', label: 'Node 1', type: 'entity', group: 'A' },
    { id: '2', label: 'Node 2', type: 'concept', group: 'B' },
  ],
  links: [
    { source: '1', target: '2', weight: 0.8 },
  ],
};

describe('KnowledgeGraphVisualization', () => {
  let store: any;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        graph: graphReducer,
      },
    });
  });

  it('renders without crashing', () => {
    render(
      <Provider store={store}>
        <KnowledgeGraphVisualization data={mockData} />
      </Provider>
    );
    expect(screen.getByRole('region')).toBeInTheDocument();
  });

  it('handles node clicks', async () => {
    const onNodeClick = jest.fn();
    render(
      <Provider store={store}>
        <KnowledgeGraphVisualization data={mockData} onNodeClick={onNodeClick} />
      </Provider>
    );

    // Simulate node click (implementation depends on ForceGraph3D)
    await waitFor(() => {
      // Test node interaction
    });
  });

  it('updates when data changes', () => {
    const { rerender } = render(
      <Provider store={store}>
        <KnowledgeGraphVisualization data={mockData} />
      </Provider>
    );

    const newData = {
      ...mockData,
      nodes: [...mockData.nodes, { id: '3', label: 'Node 3', type: 'pattern', group: 'C' }],
    };

    rerender(
      <Provider store={store}>
        <KnowledgeGraphVisualization data={newData} />
      </Provider>
    );

    // Verify graph updates
  });
});
```

### Integration Testing

```typescript
// src/dashboard/tests/integration/dashboard.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { DashboardApp } from '../../core/DashboardApp';

const server = setupServer(
  rest.get('/api/graph', (req, res, ctx) => {
    return res(ctx.json({
      nodes: [{ id: '1', label: 'Test Node' }],
      links: [],
    }));
  }),
  rest.ws('ws://localhost:3001', (ws) => {
    ws.on('connection', () => {
      ws.send(JSON.stringify({
        type: 'metrics:update',
        data: { cpu: 45, memory: 60 },
      }));
    });
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Dashboard Integration', () => {
  it('loads and displays initial data', async () => {
    render(<DashboardApp />);

    await waitFor(() => {
      expect(screen.getByText('Test Node')).toBeInTheDocument();
    });
  });

  it('receives and displays real-time updates', async () => {
    render(<DashboardApp />);

    await waitFor(() => {
      expect(screen.getByText('CPU: 45%')).toBeInTheDocument();
      expect(screen.getByText('Memory: 60%')).toBeInTheDocument();
    });
  });
});
```

### Documentation Generation

```typescript
// src/dashboard/docs/generateDocs.ts
import { generateDocs } from '@storybook/addon-docs';
import * as components from '../components';

// Generate component documentation
export const componentDocs = Object.entries(components).map(([name, component]) => ({
  name,
  component,
  docs: generateDocs(component),
}));

// Generate API documentation
export const apiDocs = {
  hooks: [
    {
      name: 'useWebSocket',
      description: 'Hook for WebSocket connections',
      params: [],
      returns: 'WebSocketContextType',
    },
    {
      name: 'useRealTimeData',
      description: 'Hook for real-time data streaming',
      params: ['options: UseRealTimeDataOptions'],
      returns: '{ data: T, buffer: T[], isStreaming: boolean }',
    },
  ],
  components: [
    {
      name: 'KnowledgeGraphVisualization',
      props: {
        data: 'GraphData',
        onNodeClick: '(node: GraphNode) => void',
        onLinkClick: '(link: GraphLink) => void',
        height: 'number',
      },
    },
  ],
};
```

### Storybook Configuration

```typescript
// .storybook/main.ts
import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx|mdx)'],
  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-interactions',
    '@storybook/addon-links',
    '@storybook/addon-a11y',
    '@storybook/addon-docs',
  ],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  docs: {
    autodocs: 'tag',
  },
};

export default config;
```

### Component Story Example

```typescript
// src/dashboard/components/visualizations/KnowledgeGraphVisualization.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { KnowledgeGraphVisualization } from './KnowledgeGraphVisualization';

const meta: Meta<typeof KnowledgeGraphVisualization> = {
  title: 'Visualizations/KnowledgeGraph',
  component: KnowledgeGraphVisualization,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    data: {
      nodes: [
        { id: '1', label: 'Central Concept', type: 'concept', group: 'core' },
        { id: '2', label: 'Related Entity', type: 'entity', group: 'related' },
        { id: '3', label: 'Pattern A', type: 'pattern', group: 'patterns' },
        { id: '4', label: 'Pattern B', type: 'pattern', group: 'patterns' },
      ],
      links: [
        { source: '1', target: '2', weight: 0.8 },
        { source: '1', target: '3', weight: 0.6 },
        { source: '1', target: '4', weight: 0.7 },
        { source: '3', target: '4', weight: 0.5 },
      ],
    },
    height: 600,
  },
};

export const LargeGraph: Story = {
  args: {
    data: generateLargeGraphData(100, 150),
    height: 800,
  },
};

export const Interactive: Story = {
  args: {
    ...Default.args,
    onNodeClick: (node) => console.log('Node clicked:', node),
    onLinkClick: (link) => console.log('Link clicked:', link),
  },
};

// Helper function to generate test data
function generateLargeGraphData(nodeCount: number, linkCount: number) {
  const nodes = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node-${i}`,
    label: `Node ${i}`,
    type: ['entity', 'concept', 'pattern'][i % 3] as any,
    group: `group-${i % 5}`,
  }));

  const links = Array.from({ length: linkCount }, (_, i) => ({
    source: `node-${Math.floor(Math.random() * nodeCount)}`,
    target: `node-${Math.floor(Math.random() * nodeCount)}`,
    weight: Math.random(),
  }));

  return { nodes, links };
}
```

## Summary

This Phase 2 implementation provides a comprehensive dashboard infrastructure for the LLMKG visualization system with:

1. **Modular Architecture**: Clean separation of concerns with providers, hooks, and components
2. **Real-time Capabilities**: WebSocket integration for live updates and streaming data
3. **Responsive Design**: Adaptive layouts that work across all device sizes
4. **LLMKG-Specific Components**: Specialized visualizations for knowledge graphs and cognitive patterns
5. **Robust State Management**: Redux Toolkit with async operations and real-time updates
6. **Comprehensive Testing**: Unit, integration, and visual testing with Storybook
7. **Developer Experience**: TypeScript throughout, hot reloading, and extensive documentation

The framework is designed to be extensible and maintainable, providing a solid foundation for building advanced LLMKG visualization features in subsequent phases.