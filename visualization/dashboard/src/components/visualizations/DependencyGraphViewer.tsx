/**
 * LLMKG Dependency Graph Visualization Component
 * 
 * Real-time interactive visualization of LLMKG codebase dependencies
 * using D3.js force-directed graph layout.
 * 
 * Features:
 * - Interactive force-directed graph visualization
 * - Real-time updates from WebSocket data stream
 * - Module complexity and coupling visualization
 * - Dependency path finding and impact analysis
 * - Export functionality for dependency graphs
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { Card, Button, Select, Slider, Switch, Typography, Space, Tooltip } from 'antd';
import { DownloadOutlined, ExpandOutlined, SettingOutlined, SearchOutlined } from '@ant-design/icons';
import { SearchBox } from '../common';
import type { SelectProps } from 'antd';

const { Title, Text } = Typography;
const { Option } = Select;

// Types matching the Rust backend structures
interface ModuleInfo {
  name: string;
  path: string;
  exports: string[];
  imports: string[];
  internal_calls: number;
  external_calls: number;
}

interface DependencyEdge {
  from: string;
  to: string;
  dependency_type: 'Import' | 'FunctionCall' | 'StructUsage' | 'TraitImplementation';
  strength: number;
}

interface DependencyGraph {
  modules: Record<string, ModuleInfo>;
  edges: DependencyEdge[];
}

interface CodebaseMetrics {
  total_files: number;
  total_lines: number;
  total_functions: number;
  total_structs: number;
  total_enums: number;
  total_modules: number;
  dependency_graph: DependencyGraph;
}

// D3 Node and Link types
interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  path: string;
  module: ModuleInfo;
  type: 'core' | 'cognitive' | 'storage' | 'embedding' | 'external' | 'other';
  complexity: number;
  connections: number;
  x?: number;
  y?: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  edge: DependencyEdge;
  strength: number;
}

interface DependencyGraphViewerProps {
  codebaseMetrics?: CodebaseMetrics;
  onNodeSelect?: (node: GraphNode) => void;
  onPathFind?: (from: string, to: string) => void;
  className?: string;
}

const DependencyGraphViewer: React.FC<DependencyGraphViewerProps> = ({
  codebaseMetrics,
  onNodeSelect,
  onPathFind,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [highlightedPath, setHighlightedPath] = useState<string[]>([]);
  const [filterType, setFilterType] = useState<string>('all');
  const [showExternalDeps, setShowExternalDeps] = useState<boolean>(true);
  const [linkDistance, setLinkDistance] = useState<number>(100);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Process codebase metrics into D3-compatible format
  const { nodes, links } = useMemo(() => {
    if (!codebaseMetrics?.dependency_graph) {
      return { nodes: [], links: [] };
    }

    const { modules, edges } = codebaseMetrics.dependency_graph;
    
    // Create nodes from modules
    const nodeArray: GraphNode[] = Object.values(modules).map(module => {
      const moduleType = determineModuleType(module.name);
      const complexity = calculateModuleComplexity(module);
      const connections = edges.filter(e => e.from === module.name || e.to === module.name).length;

      return {
        id: module.name,
        name: module.name,
        path: module.path,
        module,
        type: moduleType,
        complexity,
        connections,
      };
    });

    // Filter external dependencies if needed
    const filteredNodes = showExternalDeps 
      ? nodeArray 
      : nodeArray.filter(node => node.type !== 'external');

    // Create links from edges
    const linkArray: GraphLink[] = edges
      .filter(edge => {
        const sourceExists = filteredNodes.some(n => n.id === edge.from);
        const targetExists = filteredNodes.some(n => n.id === edge.to);
        return sourceExists && targetExists;
      })
      .map(edge => ({
        source: edge.from,
        target: edge.to,
        edge,
        strength: edge.strength,
      }));

    return { nodes: filteredNodes, links: linkArray };
  }, [codebaseMetrics, showExternalDeps]);

  // Filter nodes based on type and search
  const filteredNodes = useMemo(() => {
    let filtered = nodes;

    if (filterType !== 'all') {
      filtered = filtered.filter(node => node.type === filterType);
    }

    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter(node => 
        node.name.toLowerCase().includes(search) ||
        node.path.toLowerCase().includes(search)
      );
    }

    return filtered;
  }, [nodes, filterType, searchTerm]);

  // Filter links to match filtered nodes
  const filteredLinks = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    return links.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });
  }, [filteredNodes, links]);

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current || filteredNodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    svg.attr('width', width).attr('height', height);
    svg.selectAll('*').remove();

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Main group for zooming/panning
    const g = svg.append('g');

    // Create force simulation
    const simulation = d3.forceSimulation<GraphNode>(filteredNodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(filteredLinks)
        .id(d => d.id)
        .distance(linkDistance)
        .strength(link => link.strength))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.complexity) * 3 + 5));

    // Create arrow markers for directed edges
    const defs = g.append('defs');
    
    const arrowMarker = defs.append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto');

    arrowMarker.append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('class', 'dependency-arrow');

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredLinks)
      .enter().append('line')
      .attr('class', 'dependency-link')
      .attr('marker-end', 'url(#arrow)')
      .style('stroke', d => getDependencyColor(d.edge.dependency_type))
      .style('stroke-width', d => Math.max(1, d.strength * 3))
      .style('stroke-opacity', 0.6);

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(filteredNodes)
      .enter().append('circle')
      .attr('class', 'dependency-node')
      .attr('r', d => Math.sqrt(d.complexity) * 3 + 5)
      .style('fill', d => getModuleColor(d.type))
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .call(d3.drag<SVGCircleElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    // Add labels
    const label = g.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(filteredNodes)
      .enter().append('text')
      .attr('class', 'dependency-label')
      .text(d => d.name.split('::').pop() || d.name)
      .style('font-size', '10px')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none')
      .style('fill', '#333');

    // Node click handlers
    node.on('click', (event, d) => {
      setSelectedNode(d);
      onNodeSelect?.(d);
      
      // Highlight connected nodes
      const connectedNodes = new Set([d.id]);
      filteredLinks.forEach(link => {
        const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
        const targetId = typeof link.target === 'string' ? link.target : link.target.id;
        
        if (sourceId === d.id) connectedNodes.add(targetId);
        if (targetId === d.id) connectedNodes.add(sourceId);
      });

      node.style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.3);
      link.style('opacity', l => {
        const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
        const targetId = typeof l.target === 'string' ? l.target : l.target.id;
        return (sourceId === d.id || targetId === d.id) ? 1 : 0.1;
      });
      label.style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.3);
    });

    // Double-click to reset highlights
    svg.on('dblclick', () => {
      setSelectedNode(null);
      node.style('opacity', 1);
      link.style('opacity', 0.6);
      label.style('opacity', 1);
    });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x!)
        .attr('y1', d => (d.source as GraphNode).y!)
        .attr('x2', d => (d.target as GraphNode).x!)
        .attr('y2', d => (d.target as GraphNode).y!);

      node
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      label
        .attr('x', d => d.x!)
        .attr('y', d => d.y! + 20);
    });

    return () => {
      simulation.stop();
    };

  }, [filteredNodes, filteredLinks, linkDistance, onNodeSelect]);

  // Helper functions
  const determineModuleType = (moduleName: string): GraphNode['type'] => {
    if (moduleName.startsWith('external::')) return 'external';
    if (moduleName.includes('core')) return 'core';
    if (moduleName.includes('cognitive')) return 'cognitive';
    if (moduleName.includes('storage')) return 'storage';
    if (moduleName.includes('embedding')) return 'embedding';
    return 'other';
  };

  const calculateModuleComplexity = (module: ModuleInfo): number => {
    return module.internal_calls + module.external_calls + module.imports.length + module.exports.length;
  };

  const getModuleColor = (type: GraphNode['type']): string => {
    const colors = {
      core: '#ff6b6b',
      cognitive: '#4ecdc4',
      storage: '#45b7d1',
      embedding: '#96ceb4',
      external: '#feca57',
      other: '#a8a8a8'
    };
    return colors[type];
  };

  const getDependencyColor = (depType: DependencyEdge['dependency_type']): string => {
    const colors = {
      Import: '#3498db',
      FunctionCall: '#e74c3c',
      StructUsage: '#2ecc71',
      TraitImplementation: '#f39c12'
    };
    return colors[depType];
  };

  const exportGraph = useCallback(async () => {
    if (!svgRef.current) return;

    const svg = svgRef.current;
    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    canvas.width = svg.clientWidth;
    canvas.height = svg.clientHeight;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'llmkg-dependency-graph.png';
          a.click();
          URL.revokeObjectURL(url);
        }
      });
    };

    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    img.src = url;
  }, []);

  const moduleTypeOptions: SelectProps['options'] = [
    { label: 'All Modules', value: 'all' },
    { label: 'Core Modules', value: 'core' },
    { label: 'Cognitive Modules', value: 'cognitive' },
    { label: 'Storage Modules', value: 'storage' },
    { label: 'Embedding Modules', value: 'embedding' },
    { label: 'External Dependencies', value: 'external' },
    { label: 'Other Modules', value: 'other' },
  ];

  return (
    <Card 
      className={`dependency-graph-viewer ${className}`}
      title={
        <Space>
          <Title level={4} style={{ margin: 0 }}>
            LLMKG Dependency Graph
          </Title>
          <Text type="secondary">
            {filteredNodes.length} modules, {filteredLinks.length} dependencies
          </Text>
        </Space>
      }
      extra={
        <Space>
          <Tooltip title="Export as PNG">
            <Button icon={<DownloadOutlined />} onClick={exportGraph} />
          </Tooltip>
          <Tooltip title="Toggle Fullscreen">
            <Button 
              icon={<ExpandOutlined />} 
              onClick={() => setIsFullscreen(!isFullscreen)} 
            />
          </Tooltip>
        </Space>
      }
    >
      <div style={{ marginBottom: 16 }}>
        <Space wrap>
          <SearchBox
            placeholder="Search modules..."
            value={searchTerm}
            onChange={setSearchTerm}
            style={{ width: 200 }}
          />
          
          <Select
            value={filterType}
            onChange={setFilterType}
            options={moduleTypeOptions}
            style={{ width: 150 }}
          />

          <Space>
            <Text>Show External:</Text>
            <Switch 
              checked={showExternalDeps}
              onChange={setShowExternalDeps}
              size="small"
            />
          </Space>

          <Space>
            <Text>Link Distance:</Text>
            <Slider
              min={50}
              max={200}
              value={linkDistance}
              onChange={setLinkDistance}
              style={{ width: 100 }}
            />
          </Space>
        </Space>
      </div>

      <div 
        ref={containerRef} 
        style={{ 
          height: isFullscreen ? '80vh' : 600, 
          border: '1px solid #d9d9d9',
          borderRadius: 6,
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
        
        {/* Legend */}
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(255, 255, 255, 0.9)',
          padding: 12,
          borderRadius: 6,
          fontSize: '12px',
          border: '1px solid #d9d9d9'
        }}>
          <div><strong>Module Types:</strong></div>
          <div><span style={{color: '#ff6b6b'}}>●</span> Core</div>
          <div><span style={{color: '#4ecdc4'}}>●</span> Cognitive</div>
          <div><span style={{color: '#45b7d1'}}>●</span> Storage</div>
          <div><span style={{color: '#96ceb4'}}>●</span> Embedding</div>
          <div><span style={{color: '#feca57'}}>●</span> External</div>
          <div><span style={{color: '#a8a8a8'}}>●</span> Other</div>
        </div>

        {selectedNode && (
          <div style={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            background: 'rgba(255, 255, 255, 0.95)',
            padding: 12,
            borderRadius: 6,
            maxWidth: 300,
            border: '1px solid #d9d9d9'
          }}>
            <div><strong>{selectedNode.name}</strong></div>
            <div><small>{selectedNode.path}</small></div>
            <div>Type: {selectedNode.type}</div>
            <div>Complexity: {selectedNode.complexity}</div>
            <div>Connections: {selectedNode.connections}</div>
            <div>Imports: {selectedNode.module.imports.length}</div>
            <div>Exports: {selectedNode.module.exports.length}</div>
          </div>
        )}
      </div>

      <style jsx>{`
        .dependency-link {
          stroke: #999;
          stroke-opacity: 0.6;
        }
        
        .dependency-arrow {
          fill: #999;
          stroke: #999;
        }
        
        .dependency-node {
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .dependency-node:hover {
          stroke-width: 3px !important;
        }
        
        .dependency-label {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
          user-select: none;
        }
      `}</style>
    </Card>
  );
};

export default DependencyGraphViewer;