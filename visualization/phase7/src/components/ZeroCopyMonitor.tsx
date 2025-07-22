import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { ZeroCopyMetrics } from '../types/memory';

interface ZeroCopyMonitorProps {
  metrics: ZeroCopyMetrics;
  history: ZeroCopyMetrics[];
  className?: string;
}

export function ZeroCopyMonitor({ metrics, history, className = '' }: ZeroCopyMonitorProps) {
  const chartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!chartRef.current || history.length === 0) return;

    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 80, bottom: 30, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, history.length - 1])
      .range([0, width]);

    const yScaleOps = d3.scaleLinear()
      .domain([0, d3.max(history, d => d.totalOperations) || 0])
      .range([height, 0]);

    const yScaleBytes = d3.scaleLinear()
      .domain([0, d3.max(history, d => d.savedBytes) || 0])
      .range([height, 0]);

    // Lines
    const lineOps = d3.line<ZeroCopyMetrics>()
      .x((d, i) => xScale(i))
      .y(d => yScaleOps(d.totalOperations));

    const lineBytes = d3.line<ZeroCopyMetrics>()
      .x((d, i) => xScale(i))
      .y(d => yScaleBytes(d.savedBytes));

    const lineCOW = d3.line<ZeroCopyMetrics>()
      .x((d, i) => xScale(i))
      .y(d => yScaleOps(d.copyOnWriteEvents));

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Time');

    g.append('g')
      .call(d3.axisLeft(yScaleOps))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height / 2)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Operations');

    g.append('g')
      .attr('transform', `translate(${width}, 0)`)
      .call(d3.axisRight(yScaleBytes))
      .append('text')
      .attr('transform', 'rotate(90)')
      .attr('y', -40)
      .attr('x', height / 2)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Bytes Saved');

    // Add lines
    g.append('path')
      .datum(history)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', lineOps);

    g.append('path')
      .datum(history)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)
      .attr('d', lineBytes);

    g.append('path')
      .datum(history)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('d', lineCOW);

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${width - 100}, 10)`);

    const legendItems = [
      { color: '#3b82f6', label: 'Operations' },
      { color: '#10b981', label: 'Bytes Saved' },
      { color: '#ef4444', label: 'COW Events' }
    ];

    legendItems.forEach((item, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendRow.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('stroke', item.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', item.label === 'COW Events' ? '3,3' : '');

      legendRow.append('text')
        .attr('x', 25)
        .attr('y', 5)
        .attr('font-size', '12px')
        .attr('fill', '#9ca3af')
        .text(item.label);
    });
  }, [history]);

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Zero-Copy Performance</h3>
        <div className={`px-3 py-1 rounded text-sm ${metrics.enabled ? 'bg-green-500/20 text-green-400' : 'bg-gray-700 text-gray-400'}`}>
          {metrics.enabled ? 'Enabled' : 'Disabled'}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Total Operations</div>
          <div className="text-2xl font-bold text-white">{metrics.totalOperations.toLocaleString()}</div>
          <div className="text-xs text-gray-500">Since startup</div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Memory Saved</div>
          <div className="text-2xl font-bold text-green-400">{formatBytes(metrics.savedBytes)}</div>
          <div className="text-xs text-gray-500">Via zero-copy</div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Efficiency</div>
          <div className="text-2xl font-bold text-blue-400">{(metrics.efficiency * 100).toFixed(1)}%</div>
          <div className="text-xs text-gray-500">Copy avoidance rate</div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">COW Events</div>
          <div className="text-2xl font-bold text-orange-400">{metrics.copyOnWriteEvents}</div>
          <div className="text-xs text-gray-500">Write triggers</div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Shared Regions</div>
          <div className="text-2xl font-bold text-purple-400">{metrics.sharedRegions}</div>
          <div className="text-xs text-gray-500">Active regions</div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Avg Save/Op</div>
          <div className="text-2xl font-bold text-white">
            {metrics.totalOperations > 0 ? formatBytes(metrics.savedBytes / metrics.totalOperations) : '0 B'}
          </div>
          <div className="text-xs text-gray-500">Per operation</div>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-gray-800 rounded p-4">
        <svg 
          ref={chartRef}
          width="600"
          height="300"
          className="w-full h-auto"
          viewBox="0 0 600 300"
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      {/* Recommendations */}
      {metrics.efficiency < 0.8 && (
        <div className="mt-4 bg-yellow-500/10 border border-yellow-500/20 rounded p-4">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-yellow-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-yellow-400 text-sm">
              Zero-copy efficiency below 80%. Consider enabling for more memory operations.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}