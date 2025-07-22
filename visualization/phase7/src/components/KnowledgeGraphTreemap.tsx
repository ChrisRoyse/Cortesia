import React, { useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { KnowledgeGraphMemory, MemoryBlock } from '../types/memory';

interface KnowledgeGraphTreemapProps {
  memory: KnowledgeGraphMemory;
  width?: number;
  height?: number;
  className?: string;
}

export function KnowledgeGraphTreemap({ 
  memory, 
  width = 800, 
  height = 600,
  className = '' 
}: KnowledgeGraphTreemapProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const hierarchicalData = useMemo(() => {
    return {
      name: 'Knowledge Graph',
      children: [
        { ...memory.entities, name: 'Entities' },
        { ...memory.relations, name: 'Relations' },
        { ...memory.embeddings, name: 'Embeddings' },
        { ...memory.indexes, name: 'Indexes' },
        { ...memory.cache, name: 'Cache' }
      ]
    };
  }, [memory]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const root = d3.hierarchy(hierarchicalData)
      .sum(d => d.size || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const treemap = d3.treemap<any>()
      .size([width, height])
      .padding(2)
      .round(true);

    treemap(root);

    const colorScale = d3.scaleOrdinal()
      .domain(['Entities', 'Relations', 'Embeddings', 'Indexes', 'Cache'])
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']);

    const g = svg.append('g');

    const cells = g.selectAll('g')
      .data(root.leaves())
      .join('g')
      .attr('transform', d => `translate(${d.x0},${d.y0})`);

    cells.append('rect')
      .attr('width', d => d.x1 - d.x0)
      .attr('height', d => d.y1 - d.y0)
      .attr('fill', d => colorScale(d.parent?.data.name || '') as string)
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
        showTooltip(event, d);
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 1);
        hideTooltip();
      });

    cells.append('text')
      .attr('x', 4)
      .attr('y', 20)
      .text(d => d.data.name)
      .attr('font-size', d => {
        const width = d.x1 - d.x0;
        return Math.min(16, width / 6) + 'px';
      })
      .attr('fill', 'white')
      .attr('pointer-events', 'none');

    cells.append('text')
      .attr('x', 4)
      .attr('y', 40)
      .text(d => formatBytes(d.data.used || 0))
      .attr('font-size', d => {
        const width = d.x1 - d.x0;
        return Math.min(14, width / 8) + 'px';
      })
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .attr('pointer-events', 'none');

    // Tooltip functions
    const tooltip = d3.select('body').append('div')
      .attr('class', 'treemap-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('border-radius', '5px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    function showTooltip(event: MouseEvent, d: any) {
      const data = d.data;
      const usage = data.used / data.size * 100;
      
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <div style="font-weight: bold; margin-bottom: 5px;">${data.name}</div>
        <div>Size: ${formatBytes(data.size)}</div>
        <div>Used: ${formatBytes(data.used)} (${usage.toFixed(1)}%)</div>
        ${data.metadata ? `
          <div>Access Count: ${data.metadata.accessCount}</div>
          <div>Fragmentation: ${(data.metadata.fragmentation * 100).toFixed(1)}%</div>
        ` : ''}
      `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px');
    }

    function hideTooltip() {
      tooltip.transition().duration(200).style('opacity', 0);
    }

    return () => {
      tooltip.remove();
    };
  }, [hierarchicalData, width, height]);

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

  const totalMemory = useMemo(() => {
    return Object.values(memory).reduce((sum, block) => sum + block.size, 0);
  }, [memory]);

  const usedMemory = useMemo(() => {
    return Object.values(memory).reduce((sum, block) => sum + block.used, 0);
  }, [memory]);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Knowledge Graph Memory</h3>
        <div className="text-sm text-gray-400">
          {formatBytes(usedMemory)} / {formatBytes(totalMemory)} ({((usedMemory / totalMemory) * 100).toFixed(1)}%)
        </div>
      </div>
      
      <div className="bg-gray-800 rounded p-2">
        <svg 
          ref={svgRef} 
          width={width} 
          height={height}
          className="w-full h-auto"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      <div className="mt-4 grid grid-cols-5 gap-2">
        {Object.entries(memory).map(([key, block]) => (
          <div key={key} className="flex items-center justify-between bg-gray-800 rounded p-2">
            <span className="text-xs text-gray-400 capitalize">{key}</span>
            <span className="text-xs text-white">{((block.used / block.size) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}