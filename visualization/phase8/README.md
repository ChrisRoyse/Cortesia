# Phase 8: Cognitive Pattern Visualization System

Advanced visualization system for brain-inspired cognitive patterns in LLMKG.

## Overview

The Cognitive Pattern Visualization System provides deep insights into:
- Real-time 3D pattern activation and propagation
- Pattern classification and type analysis
- Inhibition/excitation balance monitoring
- Temporal pattern detection and prediction
- Pattern connectivity and correlation analysis

## Features

### 1. 3D Pattern Activation Visualization
- Interactive 3D space showing pattern relationships
- Real-time activation levels with pulsing animations
- Connection strength visualization (excitatory/inhibitory)
- Multiple view modes (3D, top-down, side view)
- Pattern clustering based on activation levels

### 2. Pattern Classification & Analysis
- Sunburst chart for pattern type distribution
- Radar chart showing pattern characteristics:
  - Activation levels
  - Confidence scores
  - Complexity metrics
  - Resource usage
  - Success rates
  - Connectivity
- Detailed pattern type descriptions and icons
- Performance metrics summary

### 3. Inhibition/Excitation Balance Monitor
- Real-time balance gauge with optimal range indicators
- Time series visualization of balance history
- Regional balance breakdown by brain regions
- Active pattern lists for excitatory and inhibitory states
- Automatic alerts for imbalanced states
- Recommendations for balance optimization

### 4. Temporal Pattern Analysis
- Interactive timeline visualization
- Pattern correlation matrix
- Sequence detection and prediction
- Frequency and predictability metrics
- Time window controls (1h, 6h, 24h)
- Next event prediction

## Installation

```bash
npm install
```

## Usage

### Basic Implementation

```tsx
import { CognitivePatternDashboard } from '@llmkg/cognitive-pattern-visualization';

function App() {
  return (
    <CognitivePatternDashboard 
      wsUrl="ws://localhost:8080"
      className="w-full" 
    />
  );
}
```

### Individual Components

```tsx
import {
  PatternActivation3D,
  PatternClassification,
  InhibitionExcitationBalance,
  TemporalPatternAnalysis
} from '@llmkg/cognitive-pattern-visualization';

// Use components individually
<PatternActivation3D patterns={patterns} connections={connections} />
<PatternClassification patterns={patterns} metrics={metrics} />
<InhibitionExcitationBalance balanceData={history} currentBalance={balance} />
<TemporalPatternAnalysis patterns={temporalPatterns} events={events} />
```

## Data Structures

### Cognitive Pattern
```typescript
interface CognitivePattern {
  id: string;
  type: PatternType;
  name: string;
  activation: number; // 0-1
  confidence: number; // 0-1
  timestamp: number;
  connections: PatternConnection[];
  metadata: PatternMetadata;
}
```

### Pattern Types
```typescript
type PatternType = 
  | 'convergent'      // Focused problem-solving
  | 'divergent'       // Creative exploration
  | 'lateral'         // Non-linear connections
  | 'systems'         // Holistic understanding
  | 'critical'        // Analytical evaluation
  | 'abstract'        // High-level concepts
  | 'adaptive'        // Dynamic response
  | 'chain_of_thought' // Sequential reasoning
  | 'tree_of_thoughts'; // Branching exploration
```

### Inhibition/Excitation Balance
```typescript
interface InhibitionExcitationBalance {
  timestamp: number;
  excitation: {
    total: number;
    byRegion: Record<string, number>;
    patterns: string[];
  };
  inhibition: {
    total: number;
    byRegion: Record<string, number>;
    patterns: string[];
  };
  balance: number; // -1 to 1
  optimalRange: [number, number];
}
```

### Temporal Pattern
```typescript
interface TemporalPattern {
  id: string;
  sequence: TemporalEvent[];
  frequency: number;
  duration: number;
  predictability: number; // 0-1
  nextPredicted?: TemporalEvent;
}
```

## WebSocket Integration

Expected WebSocket message format:

```javascript
{
  patterns: CognitivePattern[],
  connections: PatternConnection[],
  metrics: CognitiveMetrics,
  balance: InhibitionExcitationBalance,
  temporalPatterns: TemporalPattern[],
  temporalEvents: TemporalEvent[]
}
```

## Visualization Techniques

### 3D Pattern Space
- Force-directed layout for natural clustering
- Perspective projection with rotation
- Color coding by pattern type
- Size based on activation level
- Opacity based on confidence

### Balance Visualization
- Gauge chart for current balance
- Time series with optimal range overlay
- Regional heatmaps
- Gradient fills for visual impact

### Temporal Analysis
- Timeline with event markers
- Pattern sequence connections
- Correlation matrix heatmap
- Predictive indicators

## Performance Optimization

1. **3D Rendering**
   - Efficient D3.js force simulation
   - Throttled animation frames
   - Level-of-detail rendering

2. **Data Management**
   - Limited history buffers
   - Efficient data structures
   - Incremental updates

3. **Visual Performance**
   - Hardware-accelerated SVG
   - Optimized transitions
   - Selective rendering

## Customization

### Pattern Type Configuration
```typescript
const customPatternTypes = {
  myPattern: {
    color: '#ff6b6b',
    icon: '🔥',
    description: 'Custom pattern type'
  }
};
```

### Balance Thresholds
```typescript
const balanceConfig = {
  optimal: [-0.2, 0.2],
  warning: [-0.5, 0.5],
  critical: [-0.8, 0.8]
};
```

## Best Practices

1. **Pattern Monitoring**
   - Track activation trends over time
   - Monitor pattern type distribution
   - Identify dominant patterns

2. **Balance Management**
   - Maintain balance within optimal range
   - Respond to imbalance alerts
   - Monitor regional variations

3. **Temporal Analysis**
   - Look for recurring sequences
   - Use predictions for optimization
   - Identify pattern correlations

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

This visualization system integrates with:
- Brain-enhanced graph cognitive layers
- Pattern recognition systems
- Neural activation propagation
- Hebbian learning mechanisms
- Inhibitory circuit modeling

## Advanced Features

### Pattern Recognition
- Automatic pattern clustering
- Anomaly detection
- Emergence identification

### Predictive Analytics
- Next pattern prediction
- Sequence completion
- Trend forecasting

### Optimization Insights
- Resource usage analysis
- Performance bottlenecks
- Efficiency recommendations

## Future Enhancements

1. **Advanced Visualizations**
   - VR/AR pattern exploration
   - Multi-dimensional scaling
   - Network topology analysis

2. **Machine Learning Integration**
   - Pattern classification ML
   - Predictive modeling
   - Anomaly detection AI

3. **Real-time Optimization**
   - Automatic rebalancing
   - Pattern pruning
   - Dynamic resource allocation

## License

MIT License - see LICENSE file for details