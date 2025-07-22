# Phase 5: System Architecture Diagram

## Overview
Phase 5 focuses on creating comprehensive system architecture diagrams that visualize LLMKG's brain-inspired architecture, including the hierarchical cognitive layers, MCP protocol integration, and data flow patterns.

## Objectives
1. **Create Interactive Architecture Diagrams**
   - Visualize the complete LLMKG system architecture
   - Show hierarchical cognitive layers (Subcortical, Cortical, Thalamic)
   - Display MCP protocol integration points
   - Illustrate data flow between components

2. **Component Relationship Visualization**
   - Map dependencies between cognitive modules
   - Show activation pathways and inhibitory circuits
   - Visualize SDR (Sparse Distributed Representation) flow
   - Display knowledge graph integration points

3. **Real-time System State**
   - Live component status indicators
   - Performance metrics overlay
   - Resource utilization visualization
   - Active connection highlighting

## Technical Implementation

### Architecture Visualization Component
```typescript
// src/components/SystemArchitecture.tsx
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useMCPConnection } from '../hooks/useMCPConnection';

interface ArchitectureNode {
  id: string;
  type: 'subcortical' | 'cortical' | 'thalamic' | 'mcp' | 'storage';
  label: string;
  layer: number;
  x?: number;
  y?: number;
  status: 'active' | 'idle' | 'processing';
  metrics?: {
    cpu: number;
    memory: number;
    throughput: number;
  };
}

interface ArchitectureLink {
  source: string;
  target: string;
  type: 'activation' | 'inhibition' | 'data' | 'control';
  strength: number;
  active: boolean;
}

export const SystemArchitecture: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<ArchitectureNode[]>([]);
  const [links, setLinks] = useState<ArchitectureLink[]>([]);
  const { client, connected } = useMCPConnection();

  useEffect(() => {
    if (!connected || !client) return;

    // Initialize architecture nodes
    const architectureNodes: ArchitectureNode[] = [
      // Subcortical Layer
      {
        id: 'input_gate',
        type: 'subcortical',
        label: 'Input Gate',
        layer: 0,
        status: 'active'
      },
      {
        id: 'pattern_detector',
        type: 'subcortical',
        label: 'Pattern Detector',
        layer: 0,
        status: 'active'
      },
      {
        id: 'arousal_system',
        type: 'subcortical',
        label: 'Arousal System',
        layer: 0,
        status: 'idle'
      },
      
      // Cortical Layer
      {
        id: 'entity_processor',
        type: 'cortical',
        label: 'Entity Processor',
        layer: 1,
        status: 'processing'
      },
      {
        id: 'relation_mapper',
        type: 'cortical',
        label: 'Relation Mapper',
        layer: 1,
        status: 'active'
      },
      {
        id: 'concept_former',
        type: 'cortical',
        label: 'Concept Former',
        layer: 1,
        status: 'active'
      },
      
      // Thalamic Layer
      {
        id: 'attention_controller',
        type: 'thalamic',
        label: 'Attention Controller',
        layer: 2,
        status: 'active'
      },
      {
        id: 'context_switcher',
        type: 'thalamic',
        label: 'Context Switcher',
        layer: 2,
        status: 'idle'
      },
      
      // MCP Integration
      {
        id: 'mcp_interface',
        type: 'mcp',
        label: 'MCP Interface',
        layer: 3,
        status: 'active'
      },
      
      // Storage Layer
      {
        id: 'sdr_storage',
        type: 'storage',
        label: 'SDR Storage',
        layer: 4,
        status: 'active'
      },
      {
        id: 'knowledge_graph',
        type: 'storage',
        label: 'Knowledge Graph',
        layer: 4,
        status: 'active'
      }
    ];

    // Initialize architecture links
    const architectureLinks: ArchitectureLink[] = [
      // Subcortical to Cortical
      { source: 'input_gate', target: 'entity_processor', type: 'activation', strength: 0.8, active: true },
      { source: 'pattern_detector', target: 'relation_mapper', type: 'activation', strength: 0.7, active: true },
      { source: 'arousal_system', target: 'concept_former', type: 'activation', strength: 0.5, active: false },
      
      // Cortical interconnections
      { source: 'entity_processor', target: 'relation_mapper', type: 'data', strength: 0.9, active: true },
      { source: 'relation_mapper', target: 'concept_former', type: 'data', strength: 0.8, active: true },
      
      // Thalamic control
      { source: 'attention_controller', target: 'entity_processor', type: 'control', strength: 0.6, active: true },
      { source: 'attention_controller', target: 'relation_mapper', type: 'control', strength: 0.6, active: true },
      { source: 'context_switcher', target: 'concept_former', type: 'control', strength: 0.5, active: false },
      
      // Inhibitory connections
      { source: 'attention_controller', target: 'arousal_system', type: 'inhibition', strength: 0.4, active: true },
      
      // MCP connections
      { source: 'mcp_interface', target: 'input_gate', type: 'data', strength: 0.9, active: true },
      { source: 'concept_former', target: 'mcp_interface', type: 'data', strength: 0.8, active: true },
      
      // Storage connections
      { source: 'entity_processor', target: 'sdr_storage', type: 'data', strength: 0.7, active: true },
      { source: 'concept_former', target: 'knowledge_graph', type: 'data', strength: 0.8, active: true }
    ];

    setNodes(architectureNodes);
    setLinks(architectureLinks);

    // Subscribe to real-time updates
    const updateInterval = setInterval(async () => {
      try {
        const systemState = await client.request('system/getArchitectureState', {});
        updateNodeStates(systemState);
      } catch (error) {
        console.error('Failed to update architecture state:', error);
      }
    }, 1000);

    return () => clearInterval(updateInterval);
  }, [connected, client]);

  const updateNodeStates = (systemState: any) => {
    setNodes(prevNodes => 
      prevNodes.map(node => ({
        ...node,
        status: systemState.components[node.id]?.status || node.status,
        metrics: systemState.components[node.id]?.metrics || node.metrics
      }))
    );

    setLinks(prevLinks =>
      prevLinks.map(link => ({
        ...link,
        active: systemState.connections[`${link.source}-${link.target}`]?.active || link.active,
        strength: systemState.connections[`${link.source}-${link.target}`]?.strength || link.strength
      }))
    );
  };

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = 1200;
    const height = 800;
    const layerHeight = height / 6;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create layer backgrounds
    const layers = [
      { name: 'Subcortical', y: 0, color: '#1a1a2e' },
      { name: 'Cortical', y: layerHeight, color: '#16213e' },
      { name: 'Thalamic', y: layerHeight * 2, color: '#0f3460' },
      { name: 'MCP', y: layerHeight * 3, color: '#533483' },
      { name: 'Storage', y: layerHeight * 4, color: '#e94560' }
    ];

    const layerGroup = svg.append('g').attr('class', 'layers');
    
    layerGroup.selectAll('rect')
      .data(layers)
      .enter()
      .append('rect')
      .attr('x', 0)
      .attr('y', d => d.y)
      .attr('width', width)
      .attr('height', layerHeight)
      .attr('fill', d => d.color)
      .attr('opacity', 0.1);

    layerGroup.selectAll('text')
      .data(layers)
      .enter()
      .append('text')
      .attr('x', 20)
      .attr('y', d => d.y + 30)
      .attr('fill', '#666')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(d => d.name);

    // Position nodes
    const positionedNodes = nodes.map((node, i) => {
      const nodesInLayer = nodes.filter(n => n.layer === node.layer);
      const indexInLayer = nodesInLayer.indexOf(node);
      return {
        ...node,
        x: (width / (nodesInLayer.length + 1)) * (indexInLayer + 1),
        y: layerHeight * node.layer + layerHeight / 2
      };
    });

    // Create force simulation for fine-tuning positions
    const simulation = d3.forceSimulation(positionedNodes)
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX(d => d.x!).strength(0.5))
      .force('y', d3.forceY(d => d.y!).strength(0.5))
      .stop();

    // Run simulation
    for (let i = 0; i < 50; i++) simulation.tick();

    // Draw links
    const linkGroup = svg.append('g').attr('class', 'links');
    
    const linkElements = linkGroup.selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('x1', d => positionedNodes.find(n => n.id === d.source)?.x || 0)
      .attr('y1', d => positionedNodes.find(n => n.id === d.source)?.y || 0)
      .attr('x2', d => positionedNodes.find(n => n.id === d.target)?.x || 0)
      .attr('y2', d => positionedNodes.find(n => n.id === d.target)?.y || 0)
      .attr('stroke', d => {
        switch (d.type) {
          case 'activation': return '#00ff00';
          case 'inhibition': return '#ff0000';
          case 'data': return '#0080ff';
          case 'control': return '#ffaa00';
          default: return '#666';
        }
      })
      .attr('stroke-width', d => d.strength * 5)
      .attr('opacity', d => d.active ? 0.8 : 0.2)
      .attr('stroke-dasharray', d => d.type === 'inhibition' ? '5,5' : 'none');

    // Add link animations for active connections
    linkElements.filter(d => d.active)
      .append('animate')
      .attr('attributeName', 'stroke-opacity')
      .attr('values', '0.8;0.3;0.8')
      .attr('dur', '2s')
      .attr('repeatCount', 'indefinite');

    // Draw nodes
    const nodeGroup = svg.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup.selectAll('g')
      .data(positionedNodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    // Node circles
    nodeElements.append('circle')
      .attr('r', 40)
      .attr('fill', d => {
        switch (d.type) {
          case 'subcortical': return '#4a90e2';
          case 'cortical': return '#50c878';
          case 'thalamic': return '#ff6b6b';
          case 'mcp': return '#9b59b6';
          case 'storage': return '#f39c12';
          default: return '#666';
        }
      })
      .attr('stroke', d => {
        switch (d.status) {
          case 'active': return '#00ff00';
          case 'processing': return '#ffaa00';
          case 'idle': return '#666';
          default: return '#333';
        }
      })
      .attr('stroke-width', 3);

    // Status indicators
    nodeElements.append('circle')
      .attr('r', 8)
      .attr('cx', 30)
      .attr('cy', -30)
      .attr('fill', d => {
        switch (d.status) {
          case 'active': return '#00ff00';
          case 'processing': return '#ffaa00';
          case 'idle': return '#666';
          default: return '#333';
        }
      })
      .append('animate')
      .attr('attributeName', 'opacity')
      .attr('values', d => d.status === 'processing' ? '1;0.3;1' : '1')
      .attr('dur', '1s')
      .attr('repeatCount', 'indefinite');

    // Node labels
    nodeElements.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .text(d => d.label);

    // Metrics display
    nodeElements.filter(d => d.metrics)
      .append('g')
      .attr('transform', 'translate(0, 60)')
      .each(function(d) {
        const metricsGroup = d3.select(this);
        
        metricsGroup.append('rect')
          .attr('x', -40)
          .attr('y', -10)
          .attr('width', 80)
          .attr('height', 40)
          .attr('fill', 'rgba(0,0,0,0.8)')
          .attr('rx', 5);

        metricsGroup.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', 5)
          .attr('fill', 'white')
          .attr('font-size', '10px')
          .text(`CPU: ${d.metrics?.cpu || 0}%`);

        metricsGroup.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', 18)
          .attr('fill', 'white')
          .attr('font-size', '10px')
          .text(`Mem: ${d.metrics?.memory || 0}MB`);
      });

    // Add interactivity
    nodeElements
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        console.log('Node clicked:', d);
        // Implement node detail view
      })
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle').attr('r', 45);
        
        // Highlight connected links
        linkElements
          .attr('opacity', l => 
            (l.source === d.id || l.target === d.id) ? 1 : 0.1
          );
      })
      .on('mouseout', function() {
        d3.select(this).select('circle').attr('r', 40);
        
        // Reset link opacity
        linkElements
          .attr('opacity', d => d.active ? 0.8 : 0.2);
      });

  }, [nodes, links]);

  return (
    <div className="system-architecture">
      <h2>LLMKG System Architecture</h2>
      <svg ref={svgRef} width={1200} height={800} />
      <div className="architecture-legend">
        <h3>Legend</h3>
        <div className="legend-item">
          <span className="node-type subcortical">●</span> Subcortical Layer
        </div>
        <div className="legend-item">
          <span className="node-type cortical">●</span> Cortical Layer
        </div>
        <div className="legend-item">
          <span className="node-type thalamic">●</span> Thalamic Layer
        </div>
        <div className="legend-item">
          <span className="link-type activation">—</span> Activation
        </div>
        <div className="legend-item">
          <span className="link-type inhibition">- -</span> Inhibition
        </div>
        <div className="legend-item">
          <span className="link-type data">—</span> Data Flow
        </div>
        <div className="legend-item">
          <span className="link-type control">—</span> Control Signal
        </div>
      </div>
    </div>
  );
};
```

### Architecture State Provider
```rust
// src/visualization/architecture_state.rs
use crate::cognitive::{CognitiveLayer, LayerType};
use crate::mcp::MCPInterface;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureState {
    pub components: HashMap<String, ComponentState>,
    pub connections: HashMap<String, ConnectionState>,
    pub overall_health: SystemHealth,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentState {
    pub id: String,
    pub status: ComponentStatus,
    pub metrics: ComponentMetrics,
    pub last_activity: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ComponentStatus {
    Active,
    Idle,
    Processing,
    Error(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub cpu: f32,
    pub memory: u64,
    pub throughput: f32,
    pub latency: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionState {
    pub source: String,
    pub target: String,
    pub active: bool,
    pub strength: f32,
    pub data_flow: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub bottlenecks: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
}

impl ArchitectureState {
    pub fn collect_current_state(layers: &[CognitiveLayer]) -> Self {
        let mut components = HashMap::new();
        let mut connections = HashMap::new();

        // Collect component states
        for layer in layers {
            let component_id = format!("{:?}", layer.layer_type);
            
            components.insert(
                component_id.clone(),
                ComponentState {
                    id: component_id,
                    status: Self::determine_status(layer),
                    metrics: Self::collect_metrics(layer),
                    last_activity: layer.last_activity_timestamp(),
                }
            );
        }

        // Collect connection states
        for layer in layers {
            for connection in layer.get_connections() {
                let connection_id = format!("{}-{}", connection.source, connection.target);
                
                connections.insert(
                    connection_id,
                    ConnectionState {
                        source: connection.source.clone(),
                        target: connection.target.clone(),
                        active: connection.is_active(),
                        strength: connection.strength(),
                        data_flow: connection.current_flow(),
                    }
                );
            }
        }

        // Analyze system health
        let overall_health = Self::analyze_health(&components, &connections);

        ArchitectureState {
            components,
            connections,
            overall_health,
        }
    }

    fn determine_status(layer: &CognitiveLayer) -> ComponentStatus {
        if layer.has_errors() {
            ComponentStatus::Error(layer.get_error_message())
        } else if layer.is_processing() {
            ComponentStatus::Processing
        } else if layer.is_active() {
            ComponentStatus::Active
        } else {
            ComponentStatus::Idle
        }
    }

    fn collect_metrics(layer: &CognitiveLayer) -> ComponentMetrics {
        ComponentMetrics {
            cpu: layer.cpu_usage(),
            memory: layer.memory_usage(),
            throughput: layer.throughput(),
            latency: layer.average_latency(),
        }
    }

    fn analyze_health(
        components: &HashMap<String, ComponentState>,
        connections: &HashMap<String, ConnectionState>
    ) -> SystemHealth {
        let mut bottlenecks = Vec::new();
        let mut recommendations = Vec::new();

        // Check for high latency components
        for (id, component) in components {
            if component.metrics.latency > 100.0 {
                bottlenecks.push(format!("{} has high latency", id));
                recommendations.push(format!("Consider optimizing {}", id));
            }

            if component.metrics.cpu > 80.0 {
                bottlenecks.push(format!("{} has high CPU usage", id));
                recommendations.push(format!("Scale or optimize {}", id));
            }
        }

        // Check for inactive critical connections
        for (id, connection) in connections {
            if !connection.active && connection.strength > 0.7 {
                bottlenecks.push(format!("Critical connection {} is inactive", id));
                recommendations.push(format!("Investigate connection {}", id));
            }
        }

        let overall_status = if bottlenecks.is_empty() {
            HealthStatus::Healthy
        } else if bottlenecks.len() > 3 {
            HealthStatus::Critical
        } else {
            HealthStatus::Degraded
        };

        SystemHealth {
            overall_status,
            bottlenecks,
            recommendations,
        }
    }
}
```

## LLMKG-Specific Features

### 1. Cognitive Layer Visualization
- **Hierarchical Display**: Show subcortical, cortical, and thalamic layers
- **Activation Patterns**: Visualize neural activation spreading
- **Inhibitory Circuits**: Display inhibitory connections and their effects

### 2. MCP Protocol Integration
- **Protocol State**: Show MCP connection status and message flow
- **Tool Integration**: Visualize available tools and their usage
- **Message Routing**: Display how messages flow through the system

### 3. SDR Representation
- **Sparse Patterns**: Visualize SDR bit patterns and overlaps
- **Semantic Distance**: Show similarity between different SDRs
- **Memory Formation**: Display how SDRs form and evolve

### 4. Knowledge Graph Integration
- **Entity Relationships**: Show how entities connect in the graph
- **Concept Hierarchy**: Display hierarchical concept organization
- **Dynamic Updates**: Real-time graph modifications

## Testing Procedures

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_state_collection() {
        let layers = create_test_layers();
        let state = ArchitectureState::collect_current_state(&layers);
        
        assert!(!state.components.is_empty());
        assert!(!state.connections.is_empty());
        assert!(matches!(state.overall_health.overall_status, HealthStatus::Healthy));
    }

    #[test]
    fn test_component_status_determination() {
        let layer = create_test_layer();
        let status = ArchitectureState::determine_status(&layer);
        
        assert!(matches!(status, ComponentStatus::Active));
    }

    #[test]
    fn test_health_analysis() {
        let mut components = HashMap::new();
        components.insert(
            "test_component".to_string(),
            ComponentState {
                id: "test_component".to_string(),
                status: ComponentStatus::Active,
                metrics: ComponentMetrics {
                    cpu: 90.0, // High CPU
                    memory: 1024,
                    throughput: 100.0,
                    latency: 150.0, // High latency
                },
                last_activity: 0,
            }
        );

        let connections = HashMap::new();
        let health = ArchitectureState::analyze_health(&components, &connections);
        
        assert_eq!(health.bottlenecks.len(), 2);
        assert!(matches!(health.overall_status, HealthStatus::Degraded));
    }
}
```

### Integration Tests
```typescript
// tests/architecture.test.tsx
describe('System Architecture Visualization', () => {
  it('should render all cognitive layers', async () => {
    const { container } = render(<SystemArchitecture />);
    
    await waitFor(() => {
      expect(container.querySelector('.subcortical')).toBeInTheDocument();
      expect(container.querySelector('.cortical')).toBeInTheDocument();
      expect(container.querySelector('.thalamic')).toBeInTheDocument();
    });
  });

  it('should update component states in real-time', async () => {
    const { container } = render(<SystemArchitecture />);
    
    // Wait for initial render
    await waitFor(() => {
      const activeNodes = container.querySelectorAll('[stroke="#00ff00"]');
      expect(activeNodes.length).toBeGreaterThan(0);
    });

    // Simulate state update
    mockMCPClient.emit('architectureUpdate', {
      components: {
        'input_gate': { status: 'processing' }
      }
    });

    await waitFor(() => {
      const processingNode = container.querySelector('[stroke="#ffaa00"]');
      expect(processingNode).toBeInTheDocument();
    });
  });

  it('should handle connection highlighting on hover', async () => {
    const { container } = render(<SystemArchitecture />);
    
    const node = container.querySelector('.node');
    fireEvent.mouseOver(node);

    await waitFor(() => {
      const highlightedLinks = container.querySelectorAll('line[opacity="1"]');
      expect(highlightedLinks.length).toBeGreaterThan(0);
    });
  });
});
```

## Performance Considerations

### 1. Rendering Optimization
- Use React.memo for component memoization
- Implement virtual scrolling for large architectures
- Batch D3 updates for smooth animations

### 2. Data Management
- Cache architecture state locally
- Implement delta updates instead of full refreshes
- Use WebSocket for real-time updates

### 3. Visual Performance
- Limit animation complexity based on node count
- Use CSS transforms for better GPU acceleration
- Implement level-of-detail rendering

## Deliverables Checklist

- [ ] Interactive system architecture diagram component
- [ ] Real-time state monitoring integration
- [ ] Cognitive layer visualization with proper hierarchy
- [ ] MCP protocol flow visualization
- [ ] Connection strength and type indicators
- [ ] Component health metrics display
- [ ] Interactive node selection and detail views
- [ ] Performance optimization for large architectures
- [ ] Comprehensive test coverage
- [ ] Documentation and usage examples
- [ ] Integration with existing visualization framework
- [ ] Export functionality for architecture snapshots