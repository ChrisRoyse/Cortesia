# LLMKG Tool Visualization Components

This directory contains advanced visualization components for displaying MCP tool requests and responses with special handling for LLMKG-specific data types.

## Components Overview

### 1. RequestResponseView
The main container component that orchestrates all visualizations.

**Features:**
- Multiple view modes: Split, Tabs, and Diff
- Automatic detection of LLMKG data types (graphs, neural data)
- Export functionality (JSON, SVG for visualizations)
- Fullscreen mode
- Execution time display

**Usage:**
```typescript
<RequestResponseView
  request={requestData}
  response={responseData}
  tool={mcpTool}
  executionTime={1234}
  viewMode="split"
  onViewModeChange={(mode) => setViewMode(mode)}
/>
```

### 2. JsonViewer
Enhanced JSON viewer with syntax highlighting and interactive features.

**Features:**
- Syntax highlighting with theme support
- Collapsible/expandable nodes
- Search functionality
- Click-to-copy at any level
- Path tracking for nested values
- Custom highlighting rules

**Usage:**
```typescript
<JsonViewer
  data={jsonData}
  theme="dark"
  expandLevel={2}
  onNodeClick={(path, value) => console.log('Clicked:', path, value)}
  highlighting={[
    { key: 'error', color: '#ff0000' },
    { path: ['data', 'status'], color: '#00ff00' }
  ]}
/>
```

### 3. GraphVisualization
Interactive knowledge graph visualization using D3.js.

**Features:**
- Force-directed, hierarchical, and circular layouts
- Interactive node/edge selection
- Zoom and pan controls
- Export as SVG
- Real-time layout adjustments
- Node and edge type coloring

**Data Format:**
```typescript
{
  nodes: [
    { id: 'node1', type: 'entity', label: 'Person', properties: {...} }
  ],
  edges: [
    { source: 'node1', target: 'node2', type: 'knows', weight: 0.8 }
  ]
}
```

### 4. NeuralDataViewer
Specialized viewer for neural activity and cognitive data.

**Features:**
- Neural activity heatmaps
- SDR (Sparse Distributed Representation) visualization
- Memory consolidation progress tracking
- Cognitive pattern strength displays
- Playback controls for temporal data

**Supported Data Types:**
- `neural_activity`: Time-series neural activation data
- `sdr_data`: Sparse matrix visualizations
- `memory_data`: Consolidation progress and usage
- `cognitive_patterns`: Pattern strength over time

### 5. DiffViewer
Side-by-side or unified diff view for comparing request/response data.

**Features:**
- Unified and split view modes
- Line-by-line comparison
- Addition/deletion highlighting
- Search within diff
- Statistics display

## Data Formatting Utilities

The `dataFormatters.ts` file provides utility functions for:

- **Time formatting**: `formatExecutionTime()`, `formatTimestamp()`
- **Data export**: `copyToClipboard()`, `exportAsJson()`
- **Data manipulation**: `flattenObject()`, `getNestedValue()`
- **Visualization helpers**: `normalizeGraphData()`, `formatNeuralData()`
- **Security**: `sanitizeData()` for removing sensitive information

## Integration Example

```typescript
import { RequestResponseView } from './visualization/RequestResponseView';
import { MCPTool } from '../types';

const ToolExecutionViewer = ({ execution }) => {
  const [viewMode, setViewMode] = useState('split');
  
  return (
    <RequestResponseView
      request={execution.request}
      response={execution.response}
      tool={execution.tool}
      executionTime={execution.duration}
      viewMode={viewMode}
      onViewModeChange={setViewMode}
    />
  );
};
```

## LLMKG-Specific Visualizations

### Knowledge Graph Results
When the response contains `nodes` and `edges`, the GraphVisualization component automatically renders an interactive graph with:
- Node clustering by type
- Edge weight visualization
- Metadata tooltips
- Layout options

### Neural Activity Data
For responses with neural data, the NeuralDataViewer provides:
- Activity heatmaps over time
- Spike train visualizations
- Pattern strength indicators
- Playback controls

### Memory Operations
Memory consolidation data is visualized with:
- Progress circles
- Usage bars
- Consolidation history timeline

## Performance Considerations

1. **Large Datasets**: Components use virtualization and lazy loading for datasets > 10,000 items
2. **Memoization**: Expensive computations are memoized using React.useMemo
3. **Debouncing**: Search operations are debounced to prevent excessive re-renders
4. **SVG Optimization**: Graph visualizations use D3's enter/update/exit pattern for efficient updates

## Styling and Theming

All components support Material-UI theming and adapt to light/dark modes automatically. Custom styling can be applied through sx props or theme overrides.

## Accessibility

- Keyboard navigation support
- ARIA labels for interactive elements
- High contrast mode compatibility
- Screen reader announcements for state changes