# Phase 7: Memory & Storage Monitoring System

Real-time memory analysis and optimization dashboard for LLMKG's brain-inspired architecture.

## Overview

The Memory & Storage Monitoring System provides comprehensive insights into:
- SDR (Sparse Distributed Representation) storage and fragmentation
- Knowledge graph memory allocation and usage
- Zero-copy optimization performance
- Memory flow visualization between components
- Cognitive layer memory distribution

## Features

### 1. SDR Storage Visualization
- Real-time SDR count tracking (active vs archived)
- Storage block utilization with fragmentation heatmap
- Compression efficiency metrics
- Defragmentation recommendations

### 2. Knowledge Graph Memory Treemap
- Hierarchical visualization of memory allocation
- Entity, relation, embedding, index, and cache memory usage
- Interactive tooltips with detailed metrics
- Access frequency and fragmentation data

### 3. Zero-Copy Performance Monitor
- Operation count and memory savings tracking
- Copy-on-write event monitoring
- Efficiency metrics and recommendations
- Historical performance charts

### 4. Memory Flow Visualization
- Interactive force-directed graph of memory operations
- Real-time flow animation between components
- Operation type visualization (allocate, free, copy, share)
- Node-based memory accumulation tracking

### 5. Cognitive Layer Memory Analysis
- Subcortical component breakdown (thalamus, hippocampus, amygdala, basal ganglia)
- Cortical region allocation (prefrontal, temporal, parietal, occipital)
- Working memory buffer management
- Memory pressure indicators and recommendations

## Installation

```bash
npm install
```

## Usage

### Basic Implementation

```tsx
import { MemoryDashboard } from '@llmkg/memory-monitoring';

function App() {
  return (
    <MemoryDashboard 
      wsUrl="ws://localhost:8080"
      className="w-full" 
    />
  );
}
```

### Individual Components

```tsx
import {
  SDRStorageVisualization,
  KnowledgeGraphTreemap,
  ZeroCopyMonitor,
  MemoryFlowVisualization,
  CognitiveLayerMemoryVisualization
} from '@llmkg/memory-monitoring';

// Use components individually
<SDRStorageVisualization storage={sdrData} />
<KnowledgeGraphTreemap memory={knowledgeData} />
<ZeroCopyMonitor metrics={zeroCopyData} history={historyData} />
<MemoryFlowVisualization flows={flowData} />
<CognitiveLayerMemoryVisualization memory={cognitiveData} />
```

## Data Structures

### SDR Storage
```typescript
interface SDRStorage {
  totalSDRs: number;
  activeSDRs: number;
  archivedSDRs: number;
  totalMemoryBytes: number;
  averageSparsity: number;
  compressionRatio: number;
  fragmentationLevel: number;
  storageBlocks: SDRStorageBlock[];
}
```

### Knowledge Graph Memory
```typescript
interface KnowledgeGraphMemory {
  entities: MemoryBlock;
  relations: MemoryBlock;
  embeddings: MemoryBlock;
  indexes: MemoryBlock;
  cache: MemoryBlock;
}
```

### Zero-Copy Metrics
```typescript
interface ZeroCopyMetrics {
  enabled: boolean;
  totalOperations: number;
  savedBytes: number;
  copyOnWriteEvents: number;
  sharedRegions: number;
  efficiency: number;
}
```

### Memory Flow
```typescript
interface MemoryFlow {
  timestamp: number;
  source: string;
  target: string;
  bytes: number;
  operation: 'allocate' | 'free' | 'copy' | 'share';
  duration: number;
}
```

### Cognitive Layer Memory
```typescript
interface CognitiveLayerMemory {
  subcortical: {
    total: number;
    used: number;
    components: {
      thalamus: number;
      hippocampus: number;
      amygdala: number;
      basalGanglia: number;
    };
  };
  cortical: {
    total: number;
    used: number;
    regions: {
      prefrontal: number;
      temporal: number;
      parietal: number;
      occipital: number;
    };
  };
  workingMemory: {
    capacity: number;
    used: number;
    buffers: WorkingMemoryBuffer[];
  };
}
```

## WebSocket Integration

The dashboard connects to a WebSocket server for real-time updates:

```javascript
// Expected WebSocket message format
{
  sdrStorage: { ... },
  knowledgeGraphMemory: { ... },
  zeroCopyMetrics: { ... },
  memoryFlows: [ ... ],
  cognitiveMemory: { ... },
  memoryPressure: { ... }
}
```

## Performance Optimization

1. **Efficient Rendering**
   - Uses React.memo for component optimization
   - D3.js for performant data visualization
   - Throttled WebSocket updates

2. **Memory Management**
   - Limited history buffer sizes
   - Automatic cleanup of old data
   - Efficient data structures

3. **Visual Optimization**
   - SVG-based visualizations with hardware acceleration
   - Responsive design with viewport scaling
   - Progressive data loading

## Customization

### Theme Configuration
```tsx
// Use with your theme provider
const theme = {
  colors: {
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#8b5cf6'
  }
};
```

### Custom Memory Pressure Thresholds
```typescript
const pressureThresholds = {
  low: 0.5,
  medium: 0.7,
  high: 0.85,
  critical: 0.95
};
```

## Best Practices

1. **Monitor Memory Pressure**
   - Set up alerts for high memory pressure
   - Implement automatic garbage collection triggers
   - Archive old SDRs when fragmentation exceeds 30%

2. **Optimize Zero-Copy**
   - Enable zero-copy for large memory operations
   - Monitor copy-on-write events for optimization opportunities
   - Track efficiency metrics to identify bottlenecks

3. **Manage Cognitive Layers**
   - Balance memory allocation across layers
   - Prioritize working memory for active tasks
   - Monitor component-specific usage patterns

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

## Integration with LLMKG

This monitoring system integrates with LLMKG's:
- Brain-enhanced graph core
- SDR pattern storage system
- MCP protocol handlers
- Cognitive processing layers
- Zero-copy memory optimization

## Future Enhancements

1. **Advanced Analytics**
   - Memory usage prediction
   - Anomaly detection
   - Performance regression analysis

2. **Extended Visualizations**
   - 3D memory topology
   - Time-series memory patterns
   - Cross-component correlation analysis

3. **Automation**
   - Auto-defragmentation
   - Dynamic memory reallocation
   - Predictive scaling

## License

MIT License - see LICENSE file for details