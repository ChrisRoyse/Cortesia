# MCP Request Tracing System

A comprehensive visualization and analytics system for tracking MCP (Model Context Protocol) requests through the LLMKG cognitive architecture in real-time.

## Overview

This system provides four integrated components that work together to visualize, analyze, and monitor MCP request flow through cognitive systems:

1. **MCPRequestTracer** - Captures MCP requests from Phase 1 WebSocket
2. **RequestPathVisualizer** - Animates request paths through cognitive systems  
3. **ParticleEffects** - Creates particle trails following request flow
4. **TraceAnalytics** - Collects performance and path analytics

## Features

### Real-time Request Visualization
- Live capture of MCP requests from Phase 1 WebSocket connection
- Animated pathways showing request flow through cognitive systems
- Beautiful particle effects with trails, bursts, and explosions
- Color-coded visualization for different request states and types

### Advanced Analytics
- Performance monitoring with latency, throughput, and error rate tracking
- Cognitive load analysis and path efficiency scoring
- Bottleneck identification and optimization suggestions
- System health monitoring with configurable alerts

### Interactive Features
- Real-time particle system with physics simulation
- Dynamic path visualization with curved segments
- Node activation animations with pulsing effects
- Error state visualization with red particles and paths

## Architecture

```
MCPTracingSystem
├── MCPRequestTracer (WebSocket connection & event processing)
├── RequestPathVisualizer (3D path animation)
├── ParticleEffects (GPU-accelerated particle system)
└── TraceAnalytics (Performance monitoring & insights)
```

## Quick Start

```typescript
import * as THREE from 'three';
import { MCPTracingSystem } from './tracing';

// Create Three.js scene
const scene = new THREE.Scene();

// Initialize tracing system
const tracingSystem = new MCPTracingSystem(scene, {
  websocketUrl: 'ws://localhost:8080/mcp',
  visualization: {
    pathWidth: 0.5,
    animationSpeed: 1.0,
    colors: {
      entry: '#4CAF50',
      cognitive: '#2196F3',
      processing: '#FF9800',
      exit: '#9C27B0',
      error: '#F44336'
    }
  },
  particles: {
    maxParticles: 5000,
    particleSpeed: 20,
    emissionRate: 10
  },
  analytics: {
    healthCheckInterval: 30000,
    performanceThresholds: {
      latencyWarning: 1000,
      latencyCritical: 5000
    }
  }
});

// Check system status
const status = tracingSystem.getSystemStatus();
console.log('Tracing system status:', status);
```

## Component Details

### MCPRequestTracer

Connects to Phase 1 WebSocket and processes incoming MCP data:

```typescript
const tracer = new MCPRequestTracer('ws://localhost:8080/mcp');

// Listen for events
tracer.addEventListener('request', (event) => {
  console.log('New request:', event.data);
});

tracer.addEventListener('response', (event) => {
  console.log('Request completed:', event.data);
});

// Simulate requests for testing
tracer.simulateRequest({
  method: 'tools/call',
  params: { name: 'test_tool' }
});
```

### RequestPathVisualizer

Creates animated 3D paths between cognitive system nodes:

```typescript
const visualizer = new RequestPathVisualizer(scene, {
  pathWidth: 0.5,
  pathOpacity: 0.7,
  animationSpeed: 1.0,
  colors: {
    pathActive: '#00BCD4',
    pathComplete: '#8BC34A',
    pathError: '#FF5722'
  }
});

// Process trace events
visualizer.processTraceEvent(traceEvent);

// Get active paths
const activePaths = visualizer.getActivePaths();
```

### ParticleEffects

GPU-accelerated particle system with physics simulation:

```typescript
const particles = new ParticleEffects(scene, {
  maxParticles: 5000,
  particleSpeed: 20,
  emissionRate: 10,
  particleLifetime: 3000,
  effects: {
    gravity: -0.1,
    turbulence: 0.5,
    fadeRate: 0.02
  }
});

// Get particle statistics
console.log('Active particles:', particles.getParticleCount());
console.log('Active trails:', particles.getTrailCount());
```

### TraceAnalytics

Performance monitoring and system health analysis:

```typescript
const analytics = new TraceAnalytics({
  performanceThresholds: {
    latencyWarning: 1000,
    latencyCritical: 5000,
    errorRateWarning: 0.05
  }
});

// Get system health
const health = analytics.getCurrentHealth();
console.log('System health:', health.overallHealth);

// Get performance metrics
const metrics = analytics.getAggregatedMetrics('latency', 60000); // Last minute
console.log('Average latency:', metrics.avg, 'ms');

// Get bottlenecks
const bottlenecks = analytics.getTopBottlenecks(5);
console.log('Top bottlenecks:', bottlenecks);

// Handle alerts
const alerts = analytics.getActiveAlerts();
alerts.forEach(alert => {
  console.warn(`Alert: ${alert.message}`, alert.context);
  analytics.acknowledgeAlert(alert.id);
});
```

## Data Flow

1. **Request Initiation**: MCP request received from Phase 1 WebSocket
2. **Path Creation**: Request path visualized with animated segments
3. **Particle Emission**: Particle trail starts following the path
4. **Cognitive Processing**: System activations create particle bursts
5. **Performance Tracking**: Metrics collected for analysis
6. **Completion**: Success/error effects and path completion
7. **Analytics Update**: Performance data aggregated and health calculated

## Visualization Elements

### Node Types
- **Entry Nodes** (Green): Request entry points (tools, resources, prompts)
- **Cognitive Nodes** (Blue): Cognitive system components
- **Processing Nodes** (Orange): Data processing systems
- **Exit Nodes** (Purple/Red): Success or error exits

### Particle Types
- **Trail Particles** (Cyan): Follow request path
- **Burst Particles** (Blue): Cognitive activation effects
- **Success Particles** (Green): Successful completion
- **Error Particles** (Red): Error states
- **Processing Particles** (Orange): Performance indicators

### Path States
- **Active Paths** (Cyan): Currently processing requests
- **Complete Paths** (Light Green): Successfully completed
- **Error Paths** (Deep Orange): Failed requests

## Performance Optimization

### Particle System
- Object pooling for efficient memory management
- GPU-accelerated rendering with Three.js Points
- Configurable particle limits and lifetimes
- Automatic cleanup of expired particles

### Path Visualization  
- Bezier curves for smooth path animation
- Efficient geometry updates using BufferGeometry
- Selective rendering based on visibility
- Automatic path cleanup after completion

### Analytics
- Configurable retention periods for metrics
- Efficient time-window queries
- Background health monitoring
- Alert throttling and acknowledgment

## Configuration Options

### Visualization Config
```typescript
{
  pathWidth: number;           // Path tube radius
  pathOpacity: number;         // Path transparency
  animationSpeed: number;      // Animation speed multiplier
  nodeSize: number;           // Node sphere radius
  colors: {                   // Color scheme
    entry: string;
    cognitive: string;
    processing: string;
    exit: string;
    error: string;
  }
}
```

### Particle Config
```typescript
{
  maxParticles: number;        // Maximum active particles
  trailLength: number;         // Particles in trail
  particleSpeed: number;       // Movement speed
  emissionRate: number;        // Particles per second
  particleSize: number;        // Base particle size
  particleLifetime: number;    // Particle lifespan (ms)
  effects: {
    gravity: number;           // Downward force
    turbulence: number;        // Random movement
    fadeRate: number;          // Opacity decay rate
  }
}
```

### Analytics Config
```typescript
{
  metricsRetentionPeriod: number;  // Data retention time
  healthCheckInterval: number;     // Health check frequency
  performanceThresholds: {
    latencyWarning: number;        // Latency warning threshold
    latencyCritical: number;       // Latency critical threshold
    errorRateWarning: number;      // Error rate warning
    errorRateCritical: number;     // Error rate critical
  }
}
```

## Integration with Phase 1

The system connects to Phase 1 via WebSocket to receive real-time MCP data:

```typescript
// Expected message format from Phase 1
{
  type: 'mcp_request' | 'mcp_response' | 'cognitive_activation' | 'error',
  payload: {
    id: string;
    method?: string;
    params?: any;
    timestamp: number;
    requestId?: string;
    pattern?: string;  // For cognitive activations
    error?: string;    // For errors
    // ... other fields
  }
}
```

## Debugging and Monitoring

### System Status
```typescript
const status = tracingSystem.getSystemStatus();
// Returns: connected, activeRequests, activePaths, particles, systemHealth, alerts
```

### Export Analytics
```typescript
const data = analytics.exportAnalyticsData();
// Returns complete analytics dataset for external analysis
```

### Simulation Mode
```typescript
// Test without Phase 1 connection
tracingSystem.simulateRequest({
  method: 'tools/call',
  params: { name: 'example_tool' },
  cognitivePattern: 'knowledge_engine'
});
```

## Browser Compatibility

- Requires WebGL support for 3D visualization
- Uses modern JavaScript features (ES2020+)
- Optimized for Chrome/Firefox/Safari
- Mobile support with reduced particle counts

## Performance Considerations

### Recommended Limits
- **Desktop**: 5000 particles, 50 concurrent paths
- **Mobile**: 1000 particles, 20 concurrent paths
- **Low-end devices**: 500 particles, 10 concurrent paths

### Memory Management
- Automatic particle pooling and reuse
- Configurable data retention periods
- Efficient geometry buffer updates
- WebSocket connection management

## Troubleshooting

### Connection Issues
- Check Phase 1 WebSocket server is running
- Verify websocketUrl configuration
- Check browser console for connection errors

### Performance Issues
- Reduce maxParticles configuration
- Lower emissionRate for fewer particles
- Increase particleLifetime for longer trails
- Disable expensive effects on low-end devices

### Visualization Issues
- Ensure WebGL is supported and enabled
- Check Three.js scene setup and camera positioning
- Verify proper disposal of resources on cleanup

This tracing system provides comprehensive visualization and analytics for MCP request flow through the LLMKG cognitive architecture, enabling developers to understand, debug, and optimize system performance in real-time.