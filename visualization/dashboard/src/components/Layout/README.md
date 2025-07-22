# LLMKG Advanced Layout System

A sophisticated, responsive layout system designed specifically for LLMKG Dashboard's complex brain-inspired visualizations. This system provides drag-and-drop grid layouts, viewport optimization, and customizable dashboard arrangements.

## Features

### 🎯 Core Components
- **GridLayout**: Drag-and-drop grid system with responsive breakpoints
- **ResizablePanel**: Interactive resizable panels with constraints
- **ResponsiveContainer**: Adaptive containers with breakpoint management  
- **LayoutManager**: State management with preset saving/loading
- **ViewportOptimizer**: Performance optimization with lazy loading

### 📱 Responsive Design
- **Mobile First**: Optimized for all device sizes
- **Breakpoint System**: xs, sm, md, lg, xl, xxl breakpoints
- **Orientation Support**: Portrait/landscape handling
- **Touch Friendly**: Mobile gesture support

### ⚡ Performance Features
- **Viewport Optimization**: Only render visible components
- **Lazy Loading**: Load components as needed
- **Virtualization**: Handle large datasets efficiently
- **Memory Management**: Proper cleanup and disposal
- **60 FPS Animations**: Smooth layout transitions

## Quick Start

### Basic Grid Layout

```tsx
import { GridLayout, type LayoutItem } from '@/components/Layout';

const items: LayoutItem[] = [
  {
    i: 'neural-heatmap',
    x: 0, y: 0, w: 6, h: 4,
    component: <NeuralActivityHeatmap />
  },
  {
    i: 'knowledge-graph',
    x: 6, y: 0, w: 6, h: 4,
    component: <KnowledgeGraph3D />
  }
];

function MyDashboard() {
  return (
    <GridLayout
      items={items}
      onLayoutChange={(layout, layouts) => {
        console.log('Layout changed:', layouts);
      }}
      isDraggable={true}
      isResizable={true}
    />
  );
}
```

### Layout Manager with Presets

```tsx
import { LayoutManager } from '@/components/Layout';

function DashboardWithPresets() {
  return (
    <LayoutManager
      items={items}
      allowPresets={true}
      allowCustomization={true}
      category="cognitive" // cognitive, neural, knowledge, memory, overview
    />
  );
}
```

### Resizable Panels

```tsx
import { ResizablePanel } from '@/components/Layout';

function ResizableComponent() {
  return (
    <ResizablePanel
      width={400}
      height={300}
      minWidth={200}
      minHeight={150}
      maxWidth={800}
      maxHeight={600}
      isResizable={true}
      aspectRatio={1.33}
      maintainAspectRatio={true}
      onResize={(dimensions) => {
        console.log('New size:', dimensions);
      }}
    >
      <MyVisualization />
    </ResizablePanel>
  );
}
```

### Responsive Container

```tsx
import { ResponsiveContainer } from '@/components/Layout';

function ResponsiveComponent() {
  return (
    <ResponsiveContainer
      minWidth={300}
      maxWidth={1200}
      aspectRatio={16/9}
      fillContainer={true}
      centerContent={true}
      onBreakpointChange={(breakpoint) => {
        console.log('Breakpoint changed:', breakpoint);
      }}
    >
      <MyContent />
    </ResponsiveContainer>
  );
}
```

### Viewport Optimization

```tsx
import { ViewportOptimizer } from '@/components/Layout';

function OptimizedList() {
  return (
    <ViewportOptimizer
      enableLazyLoading={true}
      enableVirtualization={true}
      itemHeight={200}
      maxVisibleItems={20}
      onVisibilityChange={(visibleIds) => {
        console.log('Visible:', visibleIds);
      }}
      onPerformanceMetrics={(metrics) => {
        console.log('FPS:', metrics.fps);
      }}
    >
      {largeDataSet.map(item => (
        <ExpensiveComponent key={item.id} data={item} />
      ))}
    </ViewportOptimizer>
  );
}
```

## LLMKG-Specific Layouts

### Predefined Layout Presets

The system includes optimized presets for different LLMKG use cases:

#### 1. Cognitive Analysis Layout
- **Focus**: Pattern recognition and cognitive monitoring
- **Components**: Pattern recognition (8x4), cognitive load (4x2), attention heatmap (4x2), inhibition control (6x3), memory trace (6x3)

#### 2. Neural Monitoring Layout  
- **Focus**: Real-time neural activity visualization
- **Components**: Neural heatmap (12x5), spike patterns (6x3), connection strength (6x3), activity timeline (12x2)

#### 3. Knowledge Graph Layout
- **Focus**: 3D knowledge graph exploration
- **Components**: 3D graph (9x8), node inspector (3x4), relation explorer (3x4), metrics (6x2), search (6x2)

#### 4. Memory Analytics Layout
- **Focus**: Memory system performance optimization
- **Components**: Memory usage (6x3), cache efficiency (6x3), GC metrics (4x3), allocation timeline (8x3), leak detection (12x2)

#### 5. System Overview Layout
- **Focus**: Comprehensive system monitoring
- **Components**: Status (3x2), performance (3x2), resources (3x2), alerts (3x2), neural overview (6x4), cognitive overview (6x4), summaries (4x3 each)

## Responsive Breakpoints

| Breakpoint | Min Width | Typical Use | Grid Columns |
|------------|-----------|-------------|--------------|
| **xxs**    | 0px       | Small phones | 1-2 |
| **xs**     | 480px     | Phones | 2-4 |
| **sm**     | 768px     | Tablets | 6 |
| **md**     | 992px     | Small laptops | 8-10 |
| **lg**     | 1200px    | Desktops | 12 |
| **xl**     | 1600px    | Large screens | 12 |

## API Reference

### GridLayout Props

```tsx
interface GridLayoutProps {
  items: LayoutItem[];
  onLayoutChange: (layout: Layout[], layouts: Layouts) => void;
  breakpoints?: Breakpoints;
  cols?: { [key: string]: number };
  isDraggable?: boolean;
  isResizable?: boolean;
  rowHeight?: number;
  margin?: [number, number];
  containerPadding?: [number, number];
  layouts?: Layouts;
  compactType?: 'vertical' | 'horizontal' | null;
  preventCollision?: boolean;
  autoSize?: boolean;
}
```

### LayoutItem Interface

```tsx
interface LayoutItem {
  i: string;              // Unique identifier
  x: number;              // Grid column position
  y: number;              // Grid row position  
  w: number;              // Width in grid columns
  h: number;              // Height in grid rows
  minW?: number;          // Minimum width
  minH?: number;          // Minimum height
  maxW?: number;          // Maximum width
  maxH?: number;          // Maximum height
  static?: boolean;       // Cannot be moved/resized
  isDraggable?: boolean;  // Can be dragged
  isResizable?: boolean;  // Can be resized
  component: React.ReactNode; // React component to render
}
```

### ResizablePanel Props

```tsx
interface ResizablePanelProps {
  children: React.ReactNode;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  isResizable?: boolean;
  handles?: ResizeHandle[];
  onResize?: (dimensions: PanelDimensions) => void;
  onResizeStart?: (dimensions: PanelDimensions) => void;
  onResizeEnd?: (dimensions: PanelDimensions) => void;
  aspectRatio?: number;
  maintainAspectRatio?: boolean;
  snap?: { x?: number[]; y?: number[]; };
}
```

## State Management

The layout system integrates with Redux for persistent state management:

```tsx
// Redux slice actions
import {
  setLayout,
  saveLayoutPreset,
  loadLayoutPreset,
  resetLayout,
  updateLayoutSettings
} from '@/stores/slices/layoutSlice';

// Selectors
import {
  selectCurrentLayout,
  selectLayoutPresets,
  selectActivePreset,
  selectLayoutSettings
} from '@/stores/slices/layoutSlice';
```

## Performance Optimization

### Viewport Optimization
- Only renders components within viewport
- Configurable buffer zones for smooth scrolling
- Performance metrics monitoring
- Memory usage tracking

### Layout Calculations
- Efficient grid calculations
- Optimized re-render cycles
- Debounced resize handling
- Memoized component rendering

### Browser Compatibility
- Modern browsers (Chrome 80+, Firefox 75+, Safari 13+)
- Fallbacks for older browsers
- Touch device optimization
- High DPI display support

## Best Practices

### Layout Design
1. **Use appropriate grid sizes**: Start with 12-column grid for desktop
2. **Set minimum constraints**: Prevent components from becoming unusable
3. **Consider aspect ratios**: Maintain visual proportions for visualizations
4. **Plan for mobile**: Design mobile-first responsive layouts

### Performance
1. **Use viewport optimization**: For lists with >20 items
2. **Implement lazy loading**: For expensive components
3. **Limit active components**: Keep visible components under 50
4. **Monitor memory usage**: Use performance metrics in development

### User Experience
1. **Provide visual feedback**: Show resize handles and drag states
2. **Include loading states**: For lazy-loaded components
3. **Offer preset options**: Common layouts for different use cases
4. **Enable customization**: Save/load custom arrangements

## Examples

See `/examples/LayoutSystemDemo.tsx` for comprehensive usage examples including:
- Basic drag-and-drop layouts
- Preset management
- Resizable panels
- Viewport optimization
- Responsive breakpoints

## Troubleshooting

### Common Issues

#### Layout Not Saving
```tsx
// Ensure store is properly configured
import { store } from '@/stores';
import { Provider } from 'react-redux';

<Provider store={store}>
  <App />
</Provider>
```

#### Components Not Rendering
```tsx
// Check that all LayoutItems have components
const items: LayoutItem[] = [
  {
    i: 'my-component',
    x: 0, y: 0, w: 4, h: 3,
    component: <MyComponent /> // ← Required
  }
];
```

#### Performance Issues
```tsx
// Enable viewport optimization for large lists
<ViewportOptimizer
  enableLazyLoading={true}
  enableVirtualization={true}
  maxVisibleItems={20}
>
  {/* Large dataset */}
</ViewportOptimizer>
```

## Contributing

When contributing to the layout system:

1. **Follow TypeScript best practices**
2. **Include comprehensive tests**  
3. **Document new features**
4. **Consider performance implications**
5. **Test on multiple devices/browsers**

## License

Part of the LLMKG Dashboard project. See main project LICENSE for details.