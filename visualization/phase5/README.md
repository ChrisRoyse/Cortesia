# LLMKG Phase 5: Interactive System Architecture Diagram

Phase 5 provides a comprehensive, interactive system architecture visualization for Large Language Model Knowledge Graphs (LLMKG). This package includes React components for visualizing the brain-inspired cognitive architecture with real-time monitoring, performance analytics, and intuitive user interactions.

## Features

### 🧠 Brain-Inspired Architecture Visualization
- **Hierarchical Cognitive Layers**: Visual representation of Phase 1-4 cognitive systems
- **Neural Layer Layout**: Brain-inspired positioning respecting cognitive hierarchy
- **Cognitive Pattern Visualization**: Active pattern highlighting with wave-like animations
- **MCP Tool Integration**: Seamless visualization of Model Context Protocol tools

### 🎮 Interactive Controls
- **Multi-Selection Support**: Click, Ctrl+click, and drag selection
- **Zoom & Pan**: Smooth navigation with mouse/trackpad support
- **Drag-and-Drop**: Custom layout adjustments
- **Keyboard Navigation**: Full accessibility support
- **Touch Gestures**: Mobile-optimized interactions

### 📊 Real-Time Monitoring
- **Live System Health**: Component status updates
- **Performance Metrics**: CPU, memory, latency, throughput tracking
- **Connection Flow**: Animated data flow between components
- **Alert System**: Automatic bottleneck and health issue detection

### 🎨 Rich Visual Features
- **Adaptive Themes**: Customizable color schemes and styling
- **Smooth Animations**: GSAP-powered transitions and effects
- **Status Indicators**: Visual health and activity feedback
- **Layer Separation**: Clear visual hierarchy between cognitive phases

### 🔧 Developer-Friendly
- **TypeScript**: Full type safety and IntelliSense support
- **Modular Design**: Composable components and hooks
- **Performance Optimized**: Handles 100+ nodes with smooth interactions
- **Extensible**: Easy customization and extension points

## Installation

```bash
npm install --save @llmkg/phase5-architecture
```

### Dependencies
The package has peer dependencies on:
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "@reduxjs/toolkit": "^2.0.1",
  "react-redux": "^9.0.4"
}
```

## Quick Start

```tsx
import React from 'react';
import { SystemArchitectureDiagram, defaultTheme } from '@llmkg/phase5-architecture';

const architectureData = {
  nodes: [
    {
      id: 'cortical-1',
      type: 'cortical',
      label: 'Cortical Processor',
      position: { x: 100, y: 100 },
      size: 40,
      layer: 'cognitive-cortical',
      status: 'healthy',
      importance: 0.8,
      connections: [],
      metrics: {
        cpu: { current: 45, average: 42, peak: 67 },
        memory: { current: 62, average: 58, peak: 78 },
        throughput: { current: 1250, average: 1180, peak: 1450 },
        latency: { current: 23, average: 28, peak: 45 },
        errorRate: { current: 0.2, average: 0.3, peak: 1.2 },
        lastUpdated: Date.now()
      }
    }
    // ... more nodes
  ],
  connections: [
    {
      id: 'conn-1',
      sourceId: 'cortical-1',
      targetId: 'subcortical-1',
      type: 'excitation',
      strength: 0.8,
      active: true,
      dataFlow: 0.6
    }
    // ... more connections
  ],
  layers: [
    {
      id: 'cognitive-cortical',
      name: 'Cortical Layer',
      description: 'High-level cognitive processing',
      position: { x: 50, y: 50 },
      size: { width: 400, height: 200 },
      color: '#2563eb',
      phase: 2,
      order: 1,
      nodes: ['cortical-1']
    }
    // ... more layers
  ],
  metadata: {
    lastUpdated: Date.now(),
    version: '1.0.0',
    totalComponents: 1
  }
};

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <SystemArchitectureDiagram
        architectureData={architectureData}
        theme={defaultTheme}
        layout="neural-layers"
        realTimeEnabled={true}
        onNodeClick={(node) => console.log('Node clicked:', node)}
        onConnectionClick={(connection) => console.log('Connection clicked:', connection)}
      />
    </div>
  );
}
```

## Core Components

### SystemArchitectureDiagram
The main container component that orchestrates the entire visualization.

```tsx
import { SystemArchitectureDiagram } from '@llmkg/phase5-architecture';

<SystemArchitectureDiagram
  architectureData={data}
  realTimeEnabled={true}
  layout="neural-layers"
  theme={customTheme}
  onNodeClick={handleNodeClick}
  onConnectionClick={handleConnectionClick}
  onSelectionChange={handleSelectionChange}
/>
```

### ArchitectureNode
Individual component visualization with status, metrics, and interaction support.

### ConnectionEdge
Data flow connections with animation, strength visualization, and flow indicators.

### ComponentDetails
Detailed information panel with metrics, connections, and quick actions.

### LayerVisualization
Hierarchical layer representation showing cognitive phases and component groupings.

### NavigationControls
Zoom, pan, layout, and export controls with keyboard shortcuts.

## Custom Hooks

### useArchitectureMetrics
Real-time architecture metrics and health calculations.

```tsx
import { useArchitectureMetrics } from '@llmkg/phase5-architecture';

const { architectureMetrics, healthScore, alerts } = useArchitectureMetrics(data);
```

### useSystemHealth
Comprehensive system health monitoring with recommendations.

```tsx
import { useSystemHealth } from '@llmkg/phase5-architecture';

const { systemHealth, layerHealth, criticalComponents } = useSystemHealth(data);
```

### useRealTimeUpdates
WebSocket and telemetry integration for live updates.

```tsx
import { useRealTimeUpdates } from '@llmkg/phase5-architecture';

const { realTimeUpdates, isConnected, statistics } = useRealTimeUpdates(
  true,
  websocketConnection,
  telemetryStream
);
```

### usePerformanceMonitoring
Performance tracking and bottleneck detection.

```tsx
import { usePerformanceMonitoring } from '@llmkg/phase5-architecture';

const { 
  performanceMetrics, 
  bottlenecks, 
  optimizationSuggestions 
} = usePerformanceMonitoring();
```

## Layout Algorithms

Phase 5 supports multiple layout algorithms optimized for different use cases:

- **Neural Layers**: Brain-inspired hierarchical layout (default)
- **Hierarchical**: Traditional top-down tree structure
- **Force-Directed**: Physics-based automatic positioning
- **Circular**: Circular arrangement for relationship focus
- **Grid**: Regular grid pattern for systematic viewing

## Customization

### Custom Themes
```tsx
import { createCustomTheme } from '@llmkg/phase5-architecture';

const customTheme = createCustomTheme({
  colors: {
    primary: '#your-color',
    cognitive: {
      cortical: '#custom-cortical-color'
    }
  }
});
```

### Performance Configuration
```tsx
const performanceConfig = {
  maxNodes: 200,
  enableAnimations: true,
  enableWebGL: false,
  optimizeForMobile: true
};
```

## Integration with Other Phases

Phase 5 is designed to integrate seamlessly with other LLMKG phases:

- **Phase 1**: WebSocket infrastructure for real-time data
- **Phase 2**: Dashboard routing and state management  
- **Phase 3**: Tool catalog for MCP component information
- **Phase 4**: Data flow visualization for connection animations

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Performance Guidelines

- **Recommended**: 100-200 nodes for optimal performance
- **Maximum**: 500+ nodes with performance monitoring
- **Memory**: ~50MB for typical architectures
- **Frame Rate**: 60 FPS target with graceful degradation

## Development

### Building
```bash
npm run build
```

### Testing
```bash
npm test
```

### Type Checking
```bash
npm run type-check
```

## Architecture

Phase 5 follows a modular architecture:

```
src/
├── components/          # React UI components
├── hooks/              # Custom React hooks
├── core/               # Core engine and algorithms
├── types/              # TypeScript type definitions
└── index.ts            # Main exports
```

## Contributing

Please see the main LLMKG repository for contribution guidelines.

## Getting Started

### Installation

```bash
# Navigate to phase5 directory
cd visualization/phase5

# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev

# The example app will open at http://localhost:5175
```

### Building

```bash
# Build for production
npm run build

# Preview production build
npm run start
```

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Type Checking

```bash
# Check TypeScript types
npm run type-check
```

### Linting

```bash
# Run ESLint
npm run lint

# Fix linting issues
npm run lint:fix
```

## Project Structure

```
phase5/
├── src/
│   ├── components/         # React UI components
│   │   ├── SystemArchitectureDiagram.tsx
│   │   ├── ArchitectureNode.tsx
│   │   ├── ConnectionEdge.tsx
│   │   ├── ComponentDetails.tsx
│   │   ├── LayerVisualization.tsx
│   │   └── NavigationControls.tsx
│   ├── core/              # Core engines
│   │   ├── ArchitectureDiagramEngine.ts
│   │   ├── LayoutEngine.ts
│   │   ├── InteractionEngine.ts
│   │   └── AnimationEngine.ts
│   ├── hooks/             # Custom React hooks
│   │   ├── useArchitectureMetrics.ts
│   │   ├── useSystemHealth.ts
│   │   ├── useRealTimeUpdates.ts
│   │   └── usePerformanceMonitoring.ts
│   ├── monitoring/        # Real-time monitoring
│   │   ├── RealTimeMonitor.ts
│   │   ├── ComponentMonitor.tsx
│   │   ├── HealthVisualization.tsx
│   │   ├── PerformanceMetrics.tsx
│   │   ├── AlertSystem.ts
│   │   └── SystemDashboard.tsx
│   ├── health/           # Health monitoring
│   │   ├── HealthMetrics.ts
│   │   └── SystemHealthEngine.ts
│   ├── types/            # TypeScript definitions
│   │   ├── index.ts
│   │   └── MonitoringTypes.ts
│   ├── utils/            # Utility functions
│   │   ├── colorUtils.ts
│   │   ├── layoutUtils.ts
│   │   ├── animationUtils.ts
│   │   ├── geometryUtils.ts
│   │   └── performanceUtils.ts
│   └── index.ts          # Main exports
├── example/              # Example implementation
│   ├── App.tsx
│   └── main.tsx
├── tests/               # Test files
│   └── setup.ts
├── index.html           # Entry HTML
├── package.json         # Package configuration
├── tsconfig.json        # TypeScript config
├── vite.config.ts       # Vite configuration
└── jest.config.js       # Jest configuration
```

## Implementation Status

### ✅ Completed Components

1. **Architecture Visualization** - Interactive system diagram with brain-inspired layout
2. **Real-Time Monitoring** - Sub-100ms update latency with WebSocket integration
3. **Health Visualization** - Comprehensive health metrics and status indicators
4. **Performance Metrics** - Advanced performance monitoring and analytics
5. **Alert System** - Configurable alerts with multiple notification channels
6. **System Dashboard** - Integrated dashboard with multiple views
7. **LLMKG Integration** - Specialized monitoring for cognitive patterns and brain components

### 🚀 Key Features

- **Brain-Inspired Visualization**: Hierarchical cognitive layers with neural connections
- **Real-Time Updates**: Live component status and performance metrics
- **Interactive Controls**: Zoom, pan, layout selection, and export functionality
- **Cognitive Pattern Monitoring**: Visualization of thinking patterns and activation levels
- **Memory System Analytics**: SDR utilization and memory performance tracking
- **Federation Health**: Multi-node synchronization and trust monitoring
- **Advanced Animations**: Smooth transitions and data flow visualization

## License

Part of the LLMKG project. See main repository for license information.