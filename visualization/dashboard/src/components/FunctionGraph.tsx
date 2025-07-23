import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Card, Space, Button, Select, Tooltip, Badge, Slider, Input, Switch } from 'antd';
import { FunctionOutlined, SearchOutlined, SettingOutlined, ZoomInOutlined, ZoomOutOutlined, ReloadOutlined } from '@ant-design/icons';
import * as d3 from 'd3';

interface FunctionGraphProps {
  functionMap: Record<string, FunctionInfo>;
  onFunctionSelect?: (functionName: string) => void;
  selectedFunction?: string | null;
  height?: number;
}

interface FunctionInfo {
  name: string;
  file_path: string;
  line_number: number;
  parameters: string[];
  return_type?: string;
  complexity: number;
  is_public: boolean;
  is_async: boolean;
  calls: string[];
  called_by: string[];
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  complexity: number;
  is_public: boolean;
  is_async: boolean;
  file_path: string;
  calls: string[];
  called_by: string[];
  group: string;
  size: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  type: 'calls' | 'called_by';
  strength: number;
}

export const FunctionGraph: React.FC<FunctionGraphProps> = ({
  functionMap,
  onFunctionSelect,
  selectedFunction,
  height = 600
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [complexityFilter, setComplexityFilter] = useState<[number, number]>([0, 20]);
  const [showPrivate, setShowPrivate] = useState(true);
  const [showAsync, setShowAsync] = useState(true);
  const [layoutType, setLayoutType] = useState<'force' | 'hierarchical' | 'circular'>('force');
  const [zoomLevel, setZoomLevel] = useState(1);

  const processedData = useMemo(() => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    const nodeMap = new Map<string, GraphNode>();

    // Filter functions based on criteria
    const filteredFunctions = Object.entries(functionMap).filter(([name, info]) => {
      const matchesSearch = searchTerm === '' || name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesComplexity = info.complexity >= complexityFilter[0] && info.complexity <= complexityFilter[1];
      const matchesVisibility = showPrivate || info.is_public;
      const matchesAsync = showAsync || !info.is_async;
      
      return matchesSearch && matchesComplexity && matchesVisibility && matchesAsync;
    });

    // Create nodes
    filteredFunctions.forEach(([name, info]) => {
      const group = info.file_path.split('/').slice(-2, -1)[0] || 'root';
      const size = Math.max(5, Math.min(25, info.complexity * 2));
      
      const node: GraphNode = {
        id: name,
        name,
        complexity: info.complexity,
        is_public: info.is_public,
        is_async: info.is_async,
        file_path: info.file_path,
        calls: info.calls,
        called_by: info.called_by,
        group,
        size
      };
      
      nodes.push(node);
      nodeMap.set(name, node);
    });

    // Create links
    filteredFunctions.forEach(([name, info]) => {
      info.calls.forEach(calledFunction => {
        if (nodeMap.has(calledFunction)) {
          links.push({
            source: name,
            target: calledFunction,
            type: 'calls',
            strength: 1
          });
        }
      });
    });

    return { nodes, links };
  }, [functionMap, searchTerm, complexityFilter, showPrivate, showAsync]);

  useEffect(() => {
    if (!svgRef.current || processedData.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const actualHeight = height - 100;

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    svg.call(zoom);

    const container = svg.append('g');

    // Create simulation based on layout type
    let simulation: d3.Simulation<GraphNode, GraphLink>;

    switch (layoutType) {
      case 'hierarchical':
        simulation = d3.forceSimulation<GraphNode>(processedData.nodes)
          .force('link', d3.forceLink<GraphNode, GraphLink>(processedData.links).id(d => d.id).strength(0.5))
          .force('charge', d3.forceManyBody().strength(-100))
          .force('x', d3.forceX(width / 2).strength(0.1))
          .force('y', d3.forceY().strength(0.3).y(d => (d as GraphNode).complexity * 20));
        break;
      
      case 'circular':
        simulation = d3.forceSimulation<GraphNode>(processedData.nodes)
          .force('link', d3.forceLink<GraphNode, GraphLink>(processedData.links).id(d => d.id).strength(0.3))
          .force('charge', d3.forceManyBody().strength(-50))
          .force('center', d3.forceCenter(width / 2, actualHeight / 2))
          .force('collision', d3.forceCollide().radius(d => (d as GraphNode).size + 2));
        break;
      
      default: // force
        simulation = d3.forceSimulation<GraphNode>(processedData.nodes)
          .force('link', d3.forceLink<GraphNode, GraphLink>(processedData.links).id(d => d.id).distance(50))
          .force('charge', d3.forceManyBody().strength(-200))
          .force('center', d3.forceCenter(width / 2, actualHeight / 2))
          .force('collision', d3.forceCollide().radius(d => (d as GraphNode).size + 5));
    }

    // Color scale for groups
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Create links
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(processedData.links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 1)
      .attr('marker-end', 'url(#arrowhead)');

    // Create arrowhead marker
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999')
      .style('stroke', 'none');

    // Create nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(processedData.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag<SVGGElement, GraphNode>()
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

    // Add circles to nodes
    const circles = node.append('circle')
      .attr('r', d => d.size)
      .attr('fill', d => colorScale(d.group))
      .attr('stroke', d => selectedFunction === d.id ? '#ff4d4f' : d.is_public ? '#52c41a' : '#faad14')
      .attr('stroke-width', d => selectedFunction === d.id ? 3 : 2)
      .attr('opacity', d => d.is_async ? 0.8 : 1);

    // Add text labels
    const labels = node.append('text')
      .attr('dx', d => d.size + 5)
      .attr('dy', 4)
      .style('font-size', '12px')
      .style('font-family', 'monospace')
      .style('fill', '#333')
      .text(d => d.name);

    // Add complexity indicators
    node.filter(d => d.complexity > 10)
      .append('circle')
      .attr('r', 3)
      .attr('cx', d => d.size - 5)
      .attr('cy', d => -d.size + 5)
      .attr('fill', '#ff4d4f')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // Add async indicators
    node.filter(d => d.is_async)
      .append('rect')
      .attr('x', d => -d.size / 2)
      .attr('y', d => d.size - 2)
      .attr('width', d => d.size)
      .attr('height', 2)
      .attr('fill', '#1890ff');

    // Node interactions
    node
      .on('click', (event, d) => {
        onFunctionSelect?.(d.id);
      })
      .on('mouseover', function(event, d) {
        // Highlight connected nodes
        const connectedNodes = new Set([d.id]);
        processedData.links.forEach(link => {
          if (link.source === d.id || (typeof link.source === 'object' && (link.source as GraphNode).id === d.id)) {
            connectedNodes.add(typeof link.target === 'string' ? link.target : (link.target as GraphNode).id);
          }
          if (link.target === d.id || (typeof link.target === 'object' && (link.target as GraphNode).id === d.id)) {
            connectedNodes.add(typeof link.source === 'string' ? link.source : (link.source as GraphNode).id);
          }
        });

        // Highlight effect
        circles
          .attr('opacity', n => connectedNodes.has(n.id) ? 1 : 0.3)
          .attr('stroke-width', n => connectedNodes.has(n.id) ? 3 : 1);

        link
          .attr('opacity', l => {
            const sourceId = typeof l.source === 'string' ? l.source : (l.source as GraphNode).id;
            const targetId = typeof l.target === 'string' ? l.target : (l.target as GraphNode).id;
            return (sourceId === d.id || targetId === d.id) ? 1 : 0.2;
          })
          .attr('stroke-width', l => {
            const sourceId = typeof l.source === 'string' ? l.source : (l.source as GraphNode).id;
            const targetId = typeof l.target === 'string' ? l.target : (l.target as GraphNode).id;
            return (sourceId === d.id || targetId === d.id) ? 3 : 1;
          });

        // Show tooltip
        const tooltip = d3.select('body').selectAll('.function-tooltip')
          .data([d])
          .join('div')
          .attr('class', 'function-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', 1000);

        tooltip.html(`
          <div><strong>${d.name}</strong></div>
          <div>Complexity: ${d.complexity}</div>
          <div>File: ${d.file_path.split('/').pop()}</div>
          <div>Calls: ${d.calls.length} functions</div>
          <div>Called by: ${d.called_by.length} functions</div>
          <div>Visibility: ${d.is_public ? 'Public' : 'Private'}</div>
          ${d.is_async ? '<div>Async function</div>' : ''}
        `);

        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        // Reset highlighting
        circles
          .attr('opacity', d => d.is_async ? 0.8 : 1)
          .attr('stroke-width', d => selectedFunction === d.id ? 3 : 2);

        link
          .attr('opacity', 0.6)
          .attr('stroke-width', 1);

        // Remove tooltip
        d3.select('body').selectAll('.function-tooltip').remove();
      });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x!)
        .attr('y1', d => (d.source as GraphNode).y!)
        .attr('x2', d => (d.target as GraphNode).x!)
        .attr('y2', d => (d.target as GraphNode).y!);

      node
        .attr('transform', d => `translate(${d.x!},${d.y!})`);
    });

    return () => {
      simulation.stop();
    };
  }, [processedData, layoutType, selectedFunction, height]);

  const handleZoomIn = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
        1.5
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
        1 / 1.5
      );
    }
  };

  const handleReset = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform as any,
        d3.zoomIdentity
      );
    }
  };

  return (
    <Card
      title={
        <Space>
          <FunctionOutlined />
          Function Call Graph
          <Badge count={processedData.nodes.length} style={{ backgroundColor: '#52c41a' }} />
        </Space>
      }
      extra={
        <Space>
          <Input
            placeholder="Search functions..."
            prefix={<SearchOutlined />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: 200 }}
            size="small"
          />
          <Select
            value={layoutType}
            onChange={setLayoutType}
            size="small"
            style={{ width: 120 }}
          >
            <Select.Option value="force">Force Layout</Select.Option>
            <Select.Option value="hierarchical">Hierarchical</Select.Option>
            <Select.Option value="circular">Circular</Select.Option>
          </Select>
          <Button.Group size="small">
            <Button icon={<ZoomInOutlined />} onClick={handleZoomIn} />
            <Button icon={<ZoomOutOutlined />} onClick={handleZoomOut} />
            <Button icon={<ReloadOutlined />} onClick={handleReset} />
          </Button.Group>
        </Space>
      }
    >
      <div style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span>Complexity Range:</span>
            <Slider
              range
              min={0}
              max={20}
              value={complexityFilter}
              onChange={(value: number | number[]) => setComplexityFilter(value as [number, number])}
              style={{ width: 200 }}
            />
            <span>{complexityFilter[0]} - {complexityFilter[1]}</span>
          </div>
          <Space>
            <Switch
              checked={showPrivate}
              onChange={setShowPrivate}
              size="small"
            />
            <span>Show Private Functions</span>
            <Switch
              checked={showAsync}
              onChange={setShowAsync}
              size="small"
            />
            <span>Show Async Functions</span>
            <span style={{ marginLeft: 16, fontSize: '12px', color: '#666' }}>
              Zoom: {(zoomLevel * 100).toFixed(0)}%
            </span>
          </Space>
        </Space>
      </div>

      <div style={{ border: '1px solid #f0f0f0', borderRadius: 6 }}>
        <svg
          ref={svgRef}
          width="100%"
          height={height - 100}
          style={{ display: 'block' }}
        />
      </div>

      <div style={{ marginTop: 12, fontSize: '12px', color: '#666' }}>
        <Space>
          <span>🟢 Public Function</span>
          <span>🟡 Private Function</span>
          <span>🔵 Async Function (blue underline)</span>
          <span>🔴 High Complexity (red dot)</span>
          <span>➡️ Function Call</span>
        </Space>
      </div>
    </Card>
  );
};

export default FunctionGraph;