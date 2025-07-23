import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { CognitivePattern, PatternType, CognitiveMetrics } from '../types/cognitive';

interface PatternClassificationProps {
  patterns: CognitivePattern[];
  metrics: CognitiveMetrics;
  className?: string;
}

export function PatternClassification({ patterns, metrics, className = '' }: PatternClassificationProps) {
  const chartRef = useRef<SVGSVGElement>(null);
  const radarRef = useRef<SVGSVGElement>(null);

  // Pattern type descriptions
  const patternDescriptions: Record<PatternType, string> = {
    convergent: 'Focused problem-solving and synthesis',
    divergent: 'Creative exploration and ideation',
    lateral: 'Non-linear connections and insights',
    systems: 'Holistic understanding of complex systems',
    critical: 'Analytical evaluation and judgment',
    abstract: 'High-level conceptualization',
    adaptive: 'Dynamic response to changing contexts',
    chain_of_thought: 'Sequential reasoning steps',
    tree_of_thoughts: 'Branching exploration paths'
  };

  const patternIcons: Record<PatternType, string> = {
    convergent: '🎯',
    divergent: '💡',
    lateral: '🔀',
    systems: '🌐',
    critical: '🔍',
    abstract: '🎨',
    adaptive: '🔄',
    chain_of_thought: '🔗',
    tree_of_thoughts: '🌳'
  };

  // Sunburst chart for pattern distribution
  useEffect(() => {
    if (!chartRef.current) return;

    const width = 400;
    const height = 400;
    const radius = Math.min(width, height) / 2;

    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height/2})`);

    // Prepare hierarchical data
    const hierarchicalData = {
      name: 'root',
      children: Object.entries(metrics.patternDistribution).map(([type, count]) => ({
        name: type,
        value: count,
        patterns: patterns.filter(p => p.type === type)
      }))
    };

    const root = d3.hierarchy(hierarchicalData)
      .sum(d => d.value || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const partition = d3.partition<any>()
      .size([2 * Math.PI, radius]);

    partition(root);

    const colorScale = d3.scaleOrdinal()
      .domain(Object.keys(metrics.patternDistribution))
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#6366f1', '#84cc16']);

    const arc = d3.arc<any>()
      .startAngle(d => d.x0)
      .endAngle(d => d.x1)
      .padAngle(0.01)
      .padRadius(radius / 2)
      .innerRadius(d => d.y0)
      .outerRadius(d => d.y1 - 1);

    // Draw arcs
    const paths = g.selectAll('path')
      .data(root.descendants().slice(1))
      .join('path')
      .attr('fill', d => colorScale(d.data.name) as string)
      .attr('fill-opacity', 0.8)
      .attr('d', arc)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill-opacity', 1)
          .attr('transform', 'scale(1.05)');
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill-opacity', 0.8)
          .attr('transform', 'scale(1)');
      });

    // Add labels
    const labels = g.selectAll('text')
      .data(root.descendants().slice(1))
      .join('text')
      .attr('transform', d => {
        const x = (d.x0 + d.x1) / 2 * 180 / Math.PI;
        const y = (d.y0 + d.y1) / 2;
        return `rotate(${x - 90}) translate(${y},0) rotate(${x < 180 ? 0 : 180})`;
      })
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '10px')
      .text(d => d.data.name);

    // Center text
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.5em')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text(patterns.length);

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.5em')
      .attr('font-size', '12px')
      .attr('fill', '#9ca3af')
      .text('Total Patterns');
  }, [patterns, metrics]);

  // Radar chart for pattern characteristics
  useEffect(() => {
    if (!radarRef.current) return;

    const width = 400;
    const height = 400;
    const radius = Math.min(width, height) / 2 - 40;

    const svg = d3.select(radarRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height/2})`);

    // Define axes for radar chart
    const axes = [
      'Activation',
      'Confidence',
      'Complexity',
      'Resource Usage',
      'Success Rate',
      'Connectivity'
    ];

    const angleSlice = Math.PI * 2 / axes.length;

    // Scales
    const rScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);

    // Draw grid
    const levels = 5;
    for (let level = 0; level < levels; level++) {
      const levelFactor = radius * ((level + 1) / levels);
      
      g.selectAll(`.level-${level}`)
        .data(axes)
        .join('line')
        .attr('x1', (d, i) => levelFactor * Math.cos(angleSlice * i - Math.PI / 2))
        .attr('y1', (d, i) => levelFactor * Math.sin(angleSlice * i - Math.PI / 2))
        .attr('x2', (d, i) => levelFactor * Math.cos(angleSlice * (i + 1) - Math.PI / 2))
        .attr('y2', (d, i) => levelFactor * Math.sin(angleSlice * (i + 1) - Math.PI / 2))
        .attr('stroke', '#374151')
        .attr('stroke-width', '1px');
    }

    // Draw axes
    const axis = g.selectAll('.axis')
      .data(axes)
      .join('g')
      .attr('class', 'axis');

    axis.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', (d, i) => radius * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y2', (d, i) => radius * Math.sin(angleSlice * i - Math.PI / 2))
      .attr('stroke', '#374151')
      .attr('stroke-width', '1px');

    axis.append('text')
      .attr('x', (d, i) => (radius + 20) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y', (d, i) => (radius + 20) * Math.sin(angleSlice * i - Math.PI / 2))
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '12px')
      .attr('fill', '#9ca3af')
      .text(d => d);

    // Calculate average values for each pattern type
    const patternTypeData = Object.keys(metrics.patternDistribution).map(type => {
      const typePatterns = patterns.filter(p => p.type === type as PatternType);
      if (typePatterns.length === 0) return null;

      const avgActivation = typePatterns.reduce((sum, p) => sum + p.activation, 0) / typePatterns.length;
      const avgConfidence = typePatterns.reduce((sum, p) => sum + p.confidence, 0) / typePatterns.length;
      const avgComplexity = typePatterns.reduce((sum, p) => sum + p.metadata.complexity, 0) / typePatterns.length;
      const avgResourceUsage = typePatterns.reduce((sum, p) => 
        sum + (p.metadata.resourceUsage.cpu + p.metadata.resourceUsage.memory) / 2, 0
      ) / typePatterns.length;
      const successRate = metrics.performanceMetrics.successRate;
      const connectivity = typePatterns.reduce((sum, p) => sum + p.connections.length, 0) / typePatterns.length / 10;

      return {
        type,
        values: [
          avgActivation,
          avgConfidence,
          avgComplexity,
          avgResourceUsage,
          successRate,
          Math.min(1, connectivity)
        ]
      };
    }).filter(d => d !== null);

    const colorScale = d3.scaleOrdinal()
      .domain(Object.keys(metrics.patternDistribution))
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#6366f1', '#84cc16']);

    // Draw data
    patternTypeData.forEach((d, idx) => {
      if (!d) return;

      const dataLine = d3.lineRadial<number>()
        .radius((value, i) => rScale(value))
        .angle((value, i) => i * angleSlice)
        .curve(d3.curveLinearClosed);

      g.append('path')
        .datum(d.values)
        .attr('d', dataLine as any)
        .attr('fill', colorScale(d.type) as string)
        .attr('fill-opacity', 0.3)
        .attr('stroke', colorScale(d.type) as string)
        .attr('stroke-width', 2);

      // Add dots
      g.selectAll(`.dots-${idx}`)
        .data(d.values)
        .join('circle')
        .attr('cx', (value, i) => rScale(value) * Math.cos(angleSlice * i - Math.PI / 2))
        .attr('cy', (value, i) => rScale(value) * Math.sin(angleSlice * i - Math.PI / 2))
        .attr('r', 4)
        .attr('fill', colorScale(d.type) as string);
    });
  }, [patterns, metrics]);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <h3 className="text-xl font-semibold text-white mb-4">Pattern Classification & Analysis</h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sunburst Chart */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Pattern Distribution</h4>
          <svg 
            ref={chartRef}
            width="400"
            height="400"
            className="w-full h-auto"
            viewBox="0 0 400 400"
            preserveAspectRatio="xMidYMid meet"
          />
        </div>

        {/* Radar Chart */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Pattern Characteristics</h4>
          <svg 
            ref={radarRef}
            width="400"
            height="400"
            className="w-full h-auto"
            viewBox="0 0 400 400"
            preserveAspectRatio="xMidYMid meet"
          />
        </div>
      </div>

      {/* Pattern Type Cards */}
      <div className="mt-6">
        <h4 className="text-lg font-medium text-white mb-3">Pattern Types</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(metrics.patternDistribution).map(([type, count]) => {
            const typePatterns = patterns.filter(p => p.type === type as PatternType);
            const avgActivation = typePatterns.length > 0
              ? typePatterns.reduce((sum, p) => sum + p.activation, 0) / typePatterns.length
              : 0;

            return (
              <div key={type} className="bg-gray-800 rounded p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <span className="text-2xl mr-2">{patternIcons[type as PatternType]}</span>
                    <h5 className="text-white font-medium capitalize">
                      {type.replace('_', ' ')}
                    </h5>
                  </div>
                  <span className="text-2xl font-bold text-white">{count}</span>
                </div>
                <p className="text-xs text-gray-400 mb-2">
                  {patternDescriptions[type as PatternType]}
                </p>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Avg Activation</span>
                  <span className="text-white">{(avgActivation * 100).toFixed(1)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Performance Summary */}
      <div className="mt-6 bg-gray-800 rounded p-4">
        <h4 className="text-lg font-medium text-white mb-3">Performance Summary</h4>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-400">Success Rate</div>
            <div className="text-2xl font-bold text-green-400">
              {(metrics.performanceMetrics.successRate * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Avg Latency</div>
            <div className="text-2xl font-bold text-blue-400">
              {metrics.performanceMetrics.averageLatency.toFixed(0)}ms
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Resource Efficiency</div>
            <div className="text-2xl font-bold text-orange-400">
              {(metrics.performanceMetrics.resourceEfficiency * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}