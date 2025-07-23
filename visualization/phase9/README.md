# Phase 9: Advanced Debugging Tools

Comprehensive debugging and analysis tools for LLMKG's brain-inspired cognitive systems.

## Overview

The Advanced Debugging Tools provide deep insights into system behavior through:
- Distributed tracing for request flow analysis
- Time-travel debugging for state exploration
- Query analysis and optimization recommendations
- Comprehensive error logging and monitoring

## Features

### 1. Distributed Tracing
- Real-time trace visualization with waterfall, graph, and timeline views
- Service dependency mapping
- Performance bottleneck identification
- Error propagation tracking
- Span-level details with tags and logs

### 2. Time-Travel Debugging
- System state snapshots with automatic and manual triggers
- Interactive timeline navigation with playback controls
- State comparison between different points in time
- Performance metrics tracking across snapshots
- Change detection and impact analysis

### 3. Query Analyzer
- Query execution plan visualization
- Performance profiling with detailed metrics
- Bottleneck identification and analysis
- Optimization suggestions with implementation guidance
- Cost analysis and resource usage tracking

### 4. Error Logging Dashboard
- Centralized error aggregation and categorization
- Real-time error trend analysis
- Error frequency and impact metrics
- Resolution tracking and management
- Context-aware error details with stack traces

## Installation

```bash
npm install
```

## Usage

### Basic Implementation

```tsx
import { DebuggingDashboard } from '@llmkg/advanced-debugging-tools';

function App() {
  return (
    <DebuggingDashboard 
      wsUrl="ws://localhost:8080"
      className="w-full" 
    />
  );
}
```

### Individual Components

```tsx
import {
  DistributedTracing,
  TimeTravelDebugger,
  QueryAnalyzer,
  ErrorLoggingDashboard
} from '@llmkg/advanced-debugging-tools';

// Use components individually
<DistributedTracing traces={traces} />
<TimeTravelDebugger session={session} onSnapshotChange={handleChange} />
<QueryAnalyzer analyses={queryAnalyses} onOptimize={handleOptimize} />
<ErrorLoggingDashboard errors={errorLogs} stats={errorStats} />
```

## Data Structures

### Distributed Trace
```typescript
interface DistributedTrace {
  traceId: string;
  spans: TraceSpan[];
  rootSpan: TraceSpan;
  services: string[];
  startTime: number;
  endTime: number;
  duration: number;
  spanCount: number;
  errorCount: number;
}
```

### Time-Travel Snapshot
```typescript
interface TimeTravelSnapshot {
  id: string;
  timestamp: number;
  label: string;
  state: {
    patterns: any[];
    connections: any[];
    memory: any;
    activations: any;
  };
  metadata: {
    trigger: string;
    changes: string[];
    performance: {
      cpu: number;
      memory: number;
    };
  };
}
```

### Query Analysis
```typescript
interface QueryAnalysis {
  queryId: string;
  query: string;
  timestamp: number;
  executionTime: number;
  plan: QueryPlan;
  profile: QueryProfile;
  suggestions: OptimizationSuggestion[];
  bottlenecks: Bottleneck[];
}
```

### Error Log
```typescript
interface ErrorLog {
  id: string;
  timestamp: number;
  level: 'warning' | 'error' | 'critical';
  category: string;
  message: string;
  stack?: string;
  context: {
    service: string;
    operation: string;
    userId?: string;
    requestId?: string;
    metadata?: Record<string, any>;
  };
  frequency: number;
  resolved: boolean;
}
```

## WebSocket Integration

Expected WebSocket message format:

```javascript
{
  traces: DistributedTrace[],
  snapshot: TimeTravelSnapshot,
  queryAnalysis: QueryAnalysis,
  errors: ErrorLog[]
}
```

## Visualization Features

### Distributed Tracing
- **Waterfall View**: Timeline-based span visualization
- **Graph View**: Service dependency graph with force-directed layout
- **Timeline View**: Chronological event sequence

### Time-Travel Debugging
- **Interactive Timeline**: Draggable timeline with performance metrics
- **Playback Controls**: Play, pause, step, and speed controls
- **State Comparison**: Side-by-side snapshot analysis

### Query Analysis
- **Execution Plan Tree**: Hierarchical query plan visualization
- **Performance Metrics**: Comprehensive timing and resource analysis
- **Bottleneck Identification**: Visual bottleneck highlighting

### Error Dashboard
- **Trend Analysis**: Time-series error frequency charts
- **Category Distribution**: Pie chart for error categorization
- **Error Details**: Expandable error information with stack traces

## Advanced Features

### Smart Debugging
- Automatic anomaly detection in traces
- Pattern recognition in error logs
- Performance regression detection
- Predictive failure analysis

### Integration Capabilities
- OpenTelemetry standard compliance
- Custom instrumentation support
- Third-party APM tool integration
- Slack/PagerDuty alerting

### Performance Optimization
- Efficient data streaming
- Lazy loading for large datasets
- Virtualized lists for performance
- Configurable retention policies

## Best Practices

### Distributed Tracing
1. **Trace Sampling**: Configure appropriate sampling rates
2. **Context Propagation**: Ensure trace context flows through all services
3. **Span Naming**: Use consistent and descriptive span names
4. **Tag Strategy**: Add meaningful tags for filtering and analysis

### Time-Travel Debugging
1. **Snapshot Strategy**: Balance frequency with storage requirements
2. **Change Detection**: Focus on significant state changes
3. **Performance Impact**: Monitor overhead of snapshot collection
4. **Retention Policy**: Configure appropriate retention periods

### Query Analysis
1. **Query Normalization**: Normalize similar queries for pattern analysis
2. **Threshold Tuning**: Set appropriate slow query thresholds
3. **Index Recommendations**: Implement suggested optimizations
4. **Regular Analysis**: Schedule periodic query performance reviews

### Error Management
1. **Error Classification**: Implement consistent error categorization
2. **Resolution Tracking**: Maintain resolution status and history
3. **Alert Fatigue**: Configure intelligent alerting thresholds
4. **Root Cause Analysis**: Link related errors for pattern identification

## Configuration

### Trace Configuration
```typescript
const traceConfig = {
  samplingRate: 0.1,
  maxSpansPerTrace: 1000,
  retentionHours: 24,
  excludeRoutes: ['/health', '/metrics']
};
```

### Snapshot Configuration
```typescript
const snapshotConfig = {
  autoSnapshotInterval: 60000, // 1 minute
  maxSnapshots: 100,
  performanceThreshold: 80, // CPU/Memory %
  triggerOnError: true
};
```

### Query Analysis Configuration
```typescript
const queryConfig = {
  slowQueryThreshold: 1000, // ms
  enablePlanAnalysis: true,
  maxAnalysisHistory: 1000,
  autoOptimize: false
};
```

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

This debugging system integrates with:
- Brain-enhanced graph processing
- Cognitive pattern execution
- MCP protocol operations
- Memory management systems
- SDR processing pipelines

## Troubleshooting

### Common Issues

1. **WebSocket Connection**: Ensure WebSocket server is running on correct port
2. **Memory Usage**: Monitor snapshot storage and implement cleanup
3. **Performance Impact**: Adjust tracing and snapshot frequencies
4. **Data Volume**: Configure appropriate retention and sampling rates

### Performance Tuning

1. **Sampling**: Use statistical sampling for high-volume systems
2. **Batching**: Batch telemetry data for network efficiency
3. **Compression**: Enable data compression for storage and transport
4. **Indexing**: Implement efficient indexing for query performance

## Future Enhancements

1. **Machine Learning Integration**
   - Anomaly detection algorithms
   - Predictive failure analysis
   - Automated optimization recommendations

2. **Advanced Visualization**
   - 3D trace topology
   - Real-time collaboration features
   - Custom dashboard creation

3. **Enterprise Features**
   - Multi-tenant support
   - Role-based access control
   - Advanced alerting and escalation

## License

MIT License - see LICENSE file for details