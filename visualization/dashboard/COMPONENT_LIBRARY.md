# LLMKG Component Library

A comprehensive React component library specifically designed for visualizing Large Language Model Knowledge Graphs (LLMKG) with brain-inspired architecture patterns. Built with TypeScript, D3.js, Three.js, and React Three Fiber.

## 🚀 Features

- **Real-time Visualizations**: High-performance components optimized for live data streaming
- **Brain-Inspired Architecture**: Components designed for cognitive patterns, neural activities, and memory systems
- **3D Knowledge Graphs**: Interactive Three.js visualizations with physics simulation
- **Performance Optimized**: Virtual scrolling, WebGL acceleration, and 60 FPS animations
- **Dark/Light Theme Support**: Automatic theme adaptation based on user preferences
- **TypeScript First**: Comprehensive type definitions for all components and data structures
- **Accessibility Ready**: Keyboard navigation and screen reader support

## 📦 Installation

The components are part of the LLMKG dashboard and require the following dependencies:

```bash
npm install @react-three/fiber @react-three/drei three d3 @types/three @types/d3 react-virtualized-auto-sizer react-window
```

## 🧩 Components

### Visualization Components

#### KnowledgeGraph3D
Interactive 3D knowledge graph visualization with physics simulation.

```tsx
import { KnowledgeGraph3D } from '@/components/visualizations';

<KnowledgeGraph3D
  nodes={knowledgeGraphNodes}
  edges={knowledgeGraphEdges}
  width={800}
  height={600}
  onNodeClick={(node) => console.log(node)}
  enablePhysics={true}
  showLabels={true}
  interactive={true}
/>
```

**Props:**
- `nodes`: Array of knowledge nodes with 3D positioning
- `edges`: Array of connections between nodes
- `onNodeClick`: Callback for node interaction
- `enablePhysics`: Enable force-directed physics simulation
- `showLabels`: Display node labels on hover/selection
- `interactive`: Enable user interaction (zoom, pan, select)

#### CognitivePatternViz
D3.js-based visualization for cognitive patterns and inhibitory mechanisms.

```tsx
import { CognitivePatternViz } from '@/components/visualizations';

<CognitivePatternViz
  patterns={cognitivePatterns}
  inhibitoryLevels={inhibitoryData}
  width={800}
  height={400}
  timeWindow={5000}
  showLegend={true}
/>
```

**Props:**
- `patterns`: Array of cognitive patterns (hierarchical, lateral, feedback)
- `inhibitoryLevels`: Inhibitory control data
- `timeWindow`: Time range for pattern display (ms)
- `showLegend`: Display pattern type legend

#### NeuralActivityHeatmap
Real-time heatmap for neural network activity visualization.

```tsx
import { NeuralActivityHeatmap } from '@/components/visualizations';

<NeuralActivityHeatmap
  neuralData={neuralActivityData}
  width={800}
  height={600}
  gridSize={20}
  updateInterval={100}
  showGrid={true}
/>
```

**Props:**
- `neuralData`: Neural network activity data
- `gridSize`: Grid resolution for heatmap
- `updateInterval`: Refresh rate in milliseconds
- `showGrid`: Display grid lines

#### MemorySystemChart
Comprehensive memory system performance visualization.

```tsx
import { MemorySystemChart } from '@/components/visualizations';

<MemorySystemChart
  memoryData={memorySystemData}
  width={800}
  height={600}
  timeRange={60000}
  showTrends={true}
  showBreakdown={true}
/>
```

**Props:**
- `memoryData`: Memory usage and performance metrics
- `timeRange`: Historical data range (ms)
- `showTrends`: Display trend charts
- `showBreakdown`: Show memory breakdown by type

### Common Components

#### MetricCard
Reusable metric display card with trends and status indicators.

```tsx
import { MetricCard, CPUMetricCard, MemoryMetricCard } from '@/components/common';

<MetricCard
  title="CPU Usage"
  value={0.65}
  format="percentage"
  trend="up"
  trendValue={0.05}
  status="warning"
  showProgress={true}
  sparklineData={historicalData}
/>

// Preset cards
<CPUMetricCard value={0.75} trend="up" />
<MemoryMetricCard value={8589934592} showProgress />
```

**Props:**
- `title`: Metric name
- `value`: Current metric value
- `format`: Value formatting (number, percentage, bytes, duration)
- `trend`: Trend direction (up, down, stable)
- `status`: Status indicator (normal, warning, critical, success)
- `sparklineData`: Mini chart data points

#### StatusIndicator
Real-time status indicators with multiple display variants.

```tsx
import { 
  StatusIndicator, 
  WebSocketStatusIndicator,
  SystemStatusIndicator 
} from '@/components/common';

<StatusIndicator
  status="online"
  label="LLMKG Core"
  variant="badge"
  animated={true}
/>

<WebSocketStatusIndicator variant="card" />
```

**Props:**
- `status`: Status type (online, offline, connecting, error, warning)
- `variant`: Display style (dot, badge, card, pulse)
- `animated`: Enable pulse animations
- `interactive`: Enable click interactions

#### DataGrid
High-performance data grid with virtual scrolling and advanced features.

```tsx
import { DataGrid, Column } from '@/components/common';

const columns: Column[] = [
  { key: 'id', title: 'ID', width: 80, sortable: true },
  { key: 'name', title: 'Name', width: 200, filterable: true },
  { key: 'status', title: 'Status', render: (value) => <StatusBadge status={value} /> }
];

<DataGrid
  data={tableData}
  columns={columns}
  height={400}
  virtualScrolling={true}
  sortable={true}
  filterable={true}
  selectable={true}
  pagination={true}
  onRowSelect={(rows) => console.log(rows)}
/>
```

**Props:**
- `data`: Array of data objects
- `columns`: Column definitions with render functions
- `virtualScrolling`: Enable virtual scrolling for large datasets
- `sortable`: Enable column sorting
- `filterable`: Enable column filtering
- `selectable`: Enable row selection

## 🎨 Theming

All components automatically adapt to the dashboard theme (dark/light) using the Redux store:

```tsx
const theme = useAppSelector(state => state.dashboard.config.theme);
```

Theme colors are applied consistently across all components with proper contrast ratios for accessibility.

## 🔄 Real-time Data Integration

Components are designed to work with WebSocket data streams:

```tsx
// Example: Using with WebSocket data
const currentData = useAppSelector(state => state.data.current);

<NeuralActivityHeatmap neuralData={currentData?.neural} />
<MemorySystemChart memoryData={currentData?.memory} />
```

## 🎯 Performance Optimization

### Virtual Scrolling
Large datasets are handled efficiently with react-window:

```tsx
<DataGrid
  data={largeDataset} // 10k+ rows
  virtualScrolling={true}
  height={400}
/>
```

### WebGL Acceleration
3D visualizations use WebGL through Three.js:

```tsx
<KnowledgeGraph3D
  nodes={nodes}
  edges={edges}
  enablePhysics={true} // 60 FPS physics simulation
/>
```

### Memoization
Components use React.memo and useMemo for optimal re-rendering:

```tsx
const ProcessedData = useMemo(() => {
  return expensiveDataProcessing(rawData);
}, [rawData]);
```

## 📊 Data Formats

### Knowledge Graph Data
```typescript
interface KnowledgeNode {
  id: string;
  label: string;
  type: 'concept' | 'entity' | 'relation' | 'property';
  weight: number;
  position: { x: number; y: number };
  metadata: Record<string, any>;
}

interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  confidence: number;
}
```

### Neural Activity Data
```typescript
interface NeuralActivity {
  nodeId: string;
  activation: number;
  position: { x: number; y: number; z?: number };
  layer: number;
}

interface NeuralData {
  activity: NeuralActivity[];
  layers: NeuralLayer[];
  connections: NeuralConnection[];
  overallActivity: number;
}
```

### Cognitive Pattern Data
```typescript
interface CognitivePattern {
  id: string;
  type: 'hierarchical' | 'lateral' | 'feedback';
  strength: number;
  activeNodes: string[];
  timestamp: number;
}
```

## 🧪 Testing

Components include comprehensive testing patterns:

```typescript
// Example test
import { render, screen } from '@testing-library/react';
import { MetricCard } from '@/components/common';

test('displays metric value correctly', () => {
  render(<MetricCard title="CPU" value={0.75} format="percentage" />);
  expect(screen.getByText('75.0%')).toBeInTheDocument();
});
```

## 🚀 Usage Examples

### Basic Dashboard Layout
```tsx
import React from 'react';
import {
  KnowledgeGraph3D,
  NeuralActivityHeatmap,
  MetricCard,
  StatusIndicator,
  DataGrid
} from '@/components';

export const LLMKGDashboard = () => {
  return (
    <div className="dashboard-grid">
      {/* Status Bar */}
      <header>
        <StatusIndicator status="online" label="System Status" />
        <WebSocketStatusIndicator />
      </header>

      {/* Metrics */}
      <section className="metrics">
        <CPUMetricCard value={cpuUsage} />
        <MemoryMetricCard value={memoryUsage} />
        <LatencyMetricCard value={latency} />
      </section>

      {/* Main Visualization */}
      <main>
        <KnowledgeGraph3D
          nodes={knowledgeNodes}
          edges={knowledgeEdges}
          width="100%"
          height={600}
        />
      </main>

      {/* Side Panel */}
      <aside>
        <NeuralActivityHeatmap neuralData={neuralData} />
        <DataGrid data={nodeDetails} columns={columns} />
      </aside>
    </div>
  );
};
```

### Real-time Updates
```tsx
import { useEffect, useState } from 'react';
import { useAppSelector } from '@/stores';

export const RealTimeVisualization = () => {
  const wsData = useAppSelector(state => state.data.current);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (wsData) {
      setHistory(prev => [wsData, ...prev.slice(0, 99)]);
    }
  }, [wsData]);

  return (
    <MemorySystemChart
      memoryData={wsData?.memory}
      timeRange={60000}
      showTrends={true}
    />
  );
};
```

## 📝 Development Guidelines

### Adding New Components

1. **Create component file** in appropriate directory
2. **Export from index.ts** for easy importing
3. **Include TypeScript types** for all props and data
4. **Add theme support** using useAppSelector
5. **Implement error boundaries** for robustness
6. **Add accessibility attributes** (ARIA labels, keyboard nav)
7. **Include usage examples** in documentation

### Performance Best Practices

- Use `React.memo` for expensive components
- Implement `useMemo` for complex calculations
- Use `useCallback` for event handlers
- Implement virtual scrolling for large datasets
- Use WebGL for 3D visualizations
- Debounce real-time updates when necessary

### Accessibility Requirements

- Keyboard navigation support
- ARIA labels and descriptions
- Screen reader compatible
- High contrast color schemes
- Focus management
- Alternative text for visualizations

## 🔧 Configuration

Components can be configured via the Redux store:

```typescript
// Dashboard configuration
interface DashboardConfig {
  theme: 'light' | 'dark' | 'auto';
  refreshRate: number;
  maxDataPoints: number;
  enableAnimations: boolean;
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Performance Issues**: Enable virtual scrolling and reduce animation complexity
2. **Memory Leaks**: Ensure proper cleanup of D3 selections and Three.js objects
3. **Theme Issues**: Check Redux store connection and theme provider
4. **Data Loading**: Implement proper loading states and error boundaries

### Debug Mode

Enable debug logging:

```typescript
// In development
if (process.env.NODE_ENV === 'development') {
  console.log('Component render:', { props, state });
}
```

## 📚 Resources

- [Three.js Documentation](https://threejs.org/docs/)
- [D3.js Documentation](https://d3js.org/)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/)
- [React Window](https://github.com/bvaughn/react-window)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-component`)
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## 📄 License

This component library is part of the LLMKG project and follows the same licensing terms.

---

**Note**: This component library is specifically designed for LLMKG's brain-inspired architecture and may require adaptation for other use cases.