import React, { useEffect, useRef } from 'react';
import { Card } from 'antd';
import * as d3 from 'd3';
import { KnowledgeGraphMemory, HierarchicalMemoryData } from '../../types/memory';

interface KnowledgeGraphTreemapProps {
  memory: KnowledgeGraphMemory;
}

export const KnowledgeGraphTreemap: React.FC<KnowledgeGraphTreemapProps> = ({ memory }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 400;
    const margin = { top: 10, right: 10, bottom: 10, left: 10 };

    // Convert memory data to hierarchical format
    const data: HierarchicalMemoryData = {
      name: 'Knowledge Graph Memory',
      size: 0,
      children: Object.entries(memory).map(([key, block]) => ({
        name: block.name,
        size: block.used,
        value: block.used
      }))
    };

    const root = d3.hierarchy<HierarchicalMemoryData>(data)
      .sum(d => d.value || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const treemap = d3.treemap<HierarchicalMemoryData>()
      .size([width - margin.left - margin.right, height - margin.top - margin.bottom])
      .padding(2);

    treemap(root);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const leaves = g.selectAll('g')
      .data(root.leaves())
      .enter().append('g')
      .attr('transform', d => `translate(${(d as any).x0},${(d as any).y0})`);

    leaves.append('rect')
      .attr('width', d => (d as any).x1 - (d as any).x0)
      .attr('height', d => (d as any).y1 - (d as any).y0)
      .attr('fill', d => color(d.data.name))
      .attr('opacity', 0.7)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    leaves.append('text')
      .attr('x', 4)
      .attr('y', 14)
      .text(d => d.data.name)
      .attr('font-size', '10px')
      .attr('fill', '#fff');

    leaves.append('text')
      .attr('x', 4)
      .attr('y', 28)
      .text(d => formatBytes(d.value || 0))
      .attr('font-size', '8px')
      .attr('fill', '#fff')
      .attr('opacity', 0.8);

  }, [memory]);

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  return (
    <Card title="Knowledge Graph Memory Distribution">
      <svg
        ref={svgRef}
        width={800}
        height={400}
        style={{ border: '1px solid #d9d9d9', borderRadius: '4px' }}
      />
    </Card>
  );
};