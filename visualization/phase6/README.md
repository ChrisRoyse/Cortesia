# LLMKG Phase 6: Performance Monitoring

Phase 6 provides comprehensive performance monitoring for LLMKG's brain-inspired architecture, including real-time metrics visualization, performance profiling, and bottleneck detection.

## Features

### 🎯 Real-Time Performance Metrics
- **Cognitive Layer Monitoring**: Track processing latency, throughput, and error rates across subcortical, cortical, and thalamic layers
- **SDR Performance**: Monitor creation rate, sparsity, overlap ratios, and memory efficiency
- **MCP Protocol Analysis**: Message rate, latency, error tracking, and queue monitoring
- **System Resources**: CPU, memory, disk I/O, and network utilization tracking

### 📊 Advanced Visualizations
- **Performance Dashboard**: Unified view of all system metrics
- **Interactive Charts**: Line charts, area charts, bar charts, and heatmaps
- **Real-Time Updates**: Sub-second metric updates via WebSocket
- **Historical Analysis**: Trend detection and performance comparison

### 🚨 Alert Management
- **Multi-Level Alerts**: Info, warning, critical, and emergency severity levels
- **Configurable Thresholds**: Customizable alert rules for each metric
- **Alert Actions**: Acknowledgment, resolution tracking, and notifications
- **Smart Filtering**: Filter alerts by severity, component, or status

### 🔧 Performance Optimization
- **Optimization Suggestions**: AI-driven recommendations for performance improvements
- **Impact Analysis**: Estimate improvement potential and implementation effort
- **Implementation Tracking**: Monitor optimization progress and results
- **Performance Snapshots**: Capture and compare system states

## Installation

```bash
cd visualization/phase6
npm install
```

## Development

```bash
# Start development server
npm run dev

# The example app will open at http://localhost:5176
```

## Building

```bash
# Build for production
npm run build

# Preview production build
npm run start
```

## Usage

### Basic Integration

```tsx
import { PerformanceDashboard } from '@llmkg/phase6-performance';

function App() {
  return (
    <PerformanceDashboard
      websocketUrl="ws://localhost:8080/performance"
      showAlerts={true}
      showOptimizations={true}
      refreshInterval={1000}
    />
  );
}
```

### Custom Metrics Hook

```tsx
import { usePerformanceMetrics } from '@llmkg/phase6-performance';

function MyComponent() {
  const { 
    metrics, 
    currentMetrics, 
    trends, 
    alerts 
  } = usePerformanceMetrics({
    websocketUrl: 'ws://localhost:8080/performance',
    historySize: 100
  });

  return (
    <div>
      <h3>System Health: {calculateHealth(currentMetrics)}%</h3>
      <p>Active Alerts: {alerts.length}</p>
    </div>
  );
}
```

### Performance Service

```typescript
import { PerformanceService } from '@llmkg/phase6-performance';

const service = new PerformanceService({
  websocketUrl: 'ws://localhost:8080/performance',
  historySize: 1000,
  thresholds: {
    cognitive: {
      maxLatency: 100,
      minThroughput: 100,
      maxErrorRate: 0.01,
      hebbianLearningRange: [0.1, 0.9]
    },
    // ... other thresholds
  }
});

// Subscribe to metrics
const unsubscribe = service.subscribe((metrics) => {
  console.log('New metrics:', metrics);
});

// Create performance snapshot
const snapshot = service.createSnapshot('Before optimization');

// Generate performance report
const report = await service.generateReport({
  start: new Date('2024-01-01'),
  end: new Date()
});
```

## Components

### PerformanceDashboard
Main dashboard component with tabbed interface for different views.

### CognitivePerformanceChart
Visualizes processing latency and throughput across cognitive layers.

### SDRMetricsChart
Displays SDR creation rate, sparsity, and memory efficiency.

### MCPProtocolChart
Shows MCP message rate, latency, and protocol health.

### SystemResourceChart
Tracks CPU, memory, disk, and network utilization.

### PerformanceHeatmap
Heatmap visualization of performance metrics over time.

### AlertsPanel
Manages and displays performance alerts with filtering and actions.

### OptimizationPanel
Shows optimization recommendations with impact analysis.

### PerformanceSnapshot
Create and compare performance snapshots.

## Configuration

### Thresholds Configuration

```typescript
const thresholds: PerformanceThresholds = {
  cognitive: {
    maxLatency: 100,        // milliseconds
    minThroughput: 100,     // operations/second
    maxErrorRate: 0.01,     // 1%
    hebbianLearningRange: [0.1, 0.9]
  },
  sdr: {
    sparsityRange: [0.02, 0.05],  // 2-5%
    maxMemoryUsage: 1000000000,   // 1GB
    minCompressionRatio: 5
  },
  mcp: {
    maxLatency: 50,         // milliseconds
    maxErrorRate: 0.01,     // 1%
    maxQueueLength: 100
  },
  system: {
    maxCPU: 80,            // 80%
    maxMemory: 80,         // 80%
    maxDiskIO: 80,         // 80%
    maxNetworkIO: 80       // 80%
  }
};
```

### Chart Configuration

```typescript
const chartConfig: ChartConfiguration = {
  timeWindow: 300,        // 5 minutes in seconds
  refreshRate: 1000,      // 1 second in milliseconds
  metrics: ['latency', 'throughput', 'errors'],
  chartType: 'line',
  aggregation: 'average',
  showPredictions: true,
  enableZoom: true,
  showAnomalies: true
};
```

## LLMKG-Specific Features

### Cognitive Metrics
- **Hebbian Learning Rate**: Track synaptic plasticity in neural connections
- **Attention Focus**: Monitor thalamic attention switching efficiency
- **Cognitive Load**: Estimate overall cognitive processing burden
- **Inhibition/Excitation Balance**: Ensure proper neural circuit regulation

### SDR Optimization
- **Sparsity Monitoring**: Maintain optimal 2-5% sparsity for SDRs
- **Semantic Preservation**: Track accuracy of semantic encoding
- **Compression Efficiency**: Monitor storage optimization ratios
- **Overlap Analysis**: Ensure proper semantic similarity representation

### Brain-Inspired Monitoring
- **Layer Interaction**: Track cross-layer communication latency
- **Pattern Recognition Speed**: Measure cognitive pattern detection
- **Arousal System**: Monitor attention and alertness regulation
- **Memory Formation**: Track SDR creation and storage efficiency

## Performance Considerations

- **Data Sampling**: High-frequency metrics use intelligent sampling
- **Batching**: Updates are batched for efficiency
- **Memory Management**: Automatic cleanup of old metrics
- **WebSocket Optimization**: Message compression and batching
- **Rendering Performance**: Canvas-based charts for smooth animations

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## API Reference

### Types

```typescript
interface PerformanceMetrics {
  timestamp: number;
  cognitive: CognitiveMetrics;
  sdr: SDRMetrics;
  mcp: MCPMetrics;
  system: SystemMetrics;
}

interface PerformanceAlert {
  id: string;
  timestamp: number;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  component: string;
  metric: string;
  value: number;
  threshold: number;
  message: string;
}

interface PerformanceOptimization {
  id: string;
  category: 'cognitive' | 'sdr' | 'mcp' | 'system';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  effort: 'low' | 'medium' | 'high';
  estimatedImprovement: number;
  status: 'suggested' | 'in_progress' | 'completed' | 'rejected';
}
```

## Integration with Other Phases

- **Phase 1**: Receives telemetry data via WebSocket
- **Phase 2**: Uses dashboard UI framework
- **Phase 3**: Monitors MCP tool performance
- **Phase 4**: Tracks data flow visualization performance
- **Phase 5**: Monitors architecture component health

## License

Part of the LLMKG project. See main repository for license information.