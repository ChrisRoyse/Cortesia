import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { MemoryFlow } from '../types/memory';

interface MemoryFlowVisualizationProps {
  flows: MemoryFlow[];
  className?: string;
}

export function MemoryFlowVisualization({ flows, className = '' }: MemoryFlowVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || flows.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Extract unique nodes
    const nodes = Array.from(new Set(flows.flatMap(f => [f.source, f.target])))
      .map(id => ({ id }));

    // Create links from flows
    const links = flows.map(flow => ({
      source: flow.source,
      target: flow.target,
      value: flow.bytes,
      operation: flow.operation,
      duration: flow.duration
    }));

    // Create force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(centerX, centerY))
      .force('collision', d3.forceCollide().radius(50));

    // Create container
    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Create gradient definitions for different operations
    const defs = svg.append('defs');
    
    const operationColors = {
      allocate: '#10b981',
      free: '#ef4444',
      copy: '#f59e0b',
      share: '#3b82f6'
    };

    Object.entries(operationColors).forEach(([operation, color]) => {
      const gradient = defs.append('linearGradient')
        .attr('id', `gradient-${operation}`)
        .attr('gradientUnits', 'userSpaceOnUse');

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', color)
        .attr('stop-opacity', 0.1);

      gradient.append('stop')
        .attr('offset', '50%')
        .attr('stop-color', color)
        .attr('stop-opacity', 0.5);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', color)
        .attr('stop-opacity', 0.1);
    });

    // Scale for link width
    const linkWidthScale = d3.scaleLog()
      .domain([1, d3.max(links, d => d.value) || 1])
      .range([1, 10]);

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => operationColors[d.operation as keyof typeof operationColors])
      .attr('stroke-width', d => linkWidthScale(d.value))
      .attr('stroke-opacity', 0.6)
      .attr('class', 'transition-all duration-300');

    // Create node groups
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    // Add circles
    node.append('circle')
      .attr('r', 30)
      .attr('fill', d => getNodeColor(d.id))
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .on('mouseover', function(event, d) {
        setSelectedNode(d.id);
        d3.select(this).attr('r', 35);
        highlightConnections(d.id);
      })
      .on('mouseout', function() {
        setSelectedNode(null);
        d3.select(this).attr('r', 30);
        resetHighlight();
      });

    // Add labels
    node.append('text')
      .text(d => d.id)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .attr('pointer-events', 'none');

    // Add memory size labels
    const nodeMemory = new Map<string, number>();
    flows.forEach(flow => {
      if (flow.operation === 'allocate') {
        nodeMemory.set(flow.target, (nodeMemory.get(flow.target) || 0) + flow.bytes);
      } else if (flow.operation === 'free') {
        nodeMemory.set(flow.source, Math.max(0, (nodeMemory.get(flow.source) || 0) - flow.bytes));
      }
    });

    node.append('text')
      .text(d => formatBytes(nodeMemory.get(d.id) || 0))
      .attr('text-anchor', 'middle')
      .attr('dy', '1.5em')
      .attr('fill', 'rgba(255, 255, 255, 0.6)')
      .attr('font-size', '10px')
      .attr('pointer-events', 'none');

    // Animation for active flows
    const animateFlow = () => {
      const activeFlows = flows.slice(-10); // Last 10 flows
      
      activeFlows.forEach((flow, i) => {
        const sourceNode = nodes.find(n => n.id === flow.source);
        const targetNode = nodes.find(n => n.id === flow.target);
        
        if (sourceNode && targetNode) {
          const particle = g.append('circle')
            .attr('r', 4)
            .attr('fill', operationColors[flow.operation as keyof typeof operationColors])
            .attr('cx', (sourceNode as any).x)
            .attr('cy', (sourceNode as any).y);

          particle.transition()
            .delay(i * 100)
            .duration(1000)
            .attr('cx', (targetNode as any).x)
            .attr('cy', (targetNode as any).y)
            .remove();
        }
      });
    };

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y);

      node.attr('transform', d => `translate(${(d as any).x},${(d as any).y})`);
    });

    // Start animation
    const animationInterval = setInterval(animateFlow, 2000);

    // Helper functions
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    function highlightConnections(nodeId: string) {
      link.attr('stroke-opacity', d => 
        (d.source as any).id === nodeId || (d.target as any).id === nodeId ? 1 : 0.1
      );
      node.attr('opacity', d => 
        d.id === nodeId || links.some(l => 
          ((l.source as any).id === nodeId && (l.target as any).id === d.id) ||
          ((l.target as any).id === nodeId && (l.source as any).id === d.id)
        ) ? 1 : 0.3
      );
    }

    function resetHighlight() {
      link.attr('stroke-opacity', 0.6);
      node.attr('opacity', 1);
    }

    return () => {
      clearInterval(animationInterval);
      simulation.stop();
    };
  }, [flows]);

  const getNodeColor = (nodeId: string): string => {
    if (nodeId.includes('cortical')) return '#3b82f6';
    if (nodeId.includes('subcortical')) return '#10b981';
    if (nodeId.includes('cache')) return '#f59e0b';
    if (nodeId.includes('index')) return '#ef4444';
    if (nodeId.includes('embedding')) return '#8b5cf6';
    return '#6b7280';
  };

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)}${units[unitIndex]}`;
  };

  const flowStats = React.useMemo(() => {
    const stats = {
      allocate: 0,
      free: 0,
      copy: 0,
      share: 0,
      totalBytes: 0
    };

    flows.forEach(flow => {
      stats[flow.operation]++;
      stats.totalBytes += flow.bytes;
    });

    return stats;
  }, [flows]);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Memory Flow Visualization</h3>
        <div className="text-sm text-gray-400">
          {flows.length} operations | {formatBytes(flowStats.totalBytes)} transferred
        </div>
      </div>

      {/* Flow Stats */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="bg-gray-800 rounded p-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">Allocations</span>
          <span className="text-sm text-green-400 font-medium">{flowStats.allocate}</span>
        </div>
        <div className="bg-gray-800 rounded p-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">Frees</span>
          <span className="text-sm text-red-400 font-medium">{flowStats.free}</span>
        </div>
        <div className="bg-gray-800 rounded p-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">Copies</span>
          <span className="text-sm text-orange-400 font-medium">{flowStats.copy}</span>
        </div>
        <div className="bg-gray-800 rounded p-2 flex items-center justify-between">
          <span className="text-xs text-gray-400">Shares</span>
          <span className="text-sm text-blue-400 font-medium">{flowStats.share}</span>
        </div>
      </div>

      {/* Flow Diagram */}
      <div className="bg-gray-800 rounded p-4">
        <svg 
          ref={svgRef}
          width="800"
          height="600"
          className="w-full h-auto"
          viewBox="0 0 800 600"
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center space-x-6 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
          <span className="text-gray-400">Allocate</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
          <span className="text-gray-400">Free</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-orange-500 rounded-full mr-2"></div>
          <span className="text-gray-400">Copy</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
          <span className="text-gray-400">Share</span>
        </div>
      </div>

      {selectedNode && (
        <div className="mt-4 bg-gray-800 rounded p-3">
          <div className="text-sm text-gray-400">Selected Node: <span className="text-white font-medium">{selectedNode}</span></div>
        </div>
      )}
    </div>
  );
}