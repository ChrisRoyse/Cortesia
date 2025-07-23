import React, { useState, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { ErrorLog, ErrorStats } from '../types/debugging';

interface ErrorLoggingDashboardProps {
  errors: ErrorLog[];
  stats: ErrorStats;
  onResolve?: (errorId: string) => void;
  onFilter?: (category: string | null, level: string | null) => void;
  className?: string;
}

export function ErrorLoggingDashboard({ 
  errors, 
  stats, 
  onResolve,
  onFilter,
  className = '' 
}: ErrorLoggingDashboardProps) {
  const trendChartRef = useRef<SVGSVGElement>(null);
  const distributionRef = useRef<SVGSVGElement>(null);
  const [selectedError, setSelectedError] = useState<string | null>(null);
  const [filterLevel, setFilterLevel] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('24h');

  // Trend chart
  useEffect(() => {
    if (!trendChartRef.current || stats.trend.length === 0) return;

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const svg = d3.select(trendChartRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Filter data based on time range
    const now = Date.now();
    const rangeMs = timeRange === '1h' ? 3600000 : timeRange === '24h' ? 86400000 : 604800000;
    const filteredTrend = stats.trend.filter(d => now - d.timestamp < rangeMs);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(filteredTrend, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(filteredTrend, d => d.count) || 0])
      .range([height, 0]);

    // Line
    const line = d3.line<{timestamp: number; count: number}>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.count))
      .curve(d3.curveMonotoneX);

    // Area
    const area = d3.area<{timestamp: number; count: number}>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(height)
      .y1(d => yScale(d.count))
      .curve(d3.curveMonotoneX);

    // Gradient
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'error-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#ef4444')
      .attr('stop-opacity', 0.8);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#ef4444')
      .attr('stop-opacity', 0.1);

    // Draw area
    g.append('path')
      .datum(filteredTrend)
      .attr('fill', 'url(#error-gradient)')
      .attr('d', area);

    // Draw line
    g.append('path')
      .datum(filteredTrend)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')));

    g.append('g')
      .call(d3.axisLeft(yScale));

    // Alert threshold line
    const threshold = 50;
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(threshold))
      .attr('y2', yScale(threshold))
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '5,5');

    g.append('text')
      .attr('x', width)
      .attr('y', yScale(threshold) - 5)
      .attr('text-anchor', 'end')
      .attr('fill', '#f59e0b')
      .attr('font-size', '10px')
      .text('Alert Threshold');

  }, [stats.trend, timeRange]);

  // Distribution chart
  useEffect(() => {
    if (!distributionRef.current) return;

    const width = 300;
    const height = 300;
    const radius = Math.min(width, height) / 2;

    const svg = d3.select(distributionRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height/2})`);

    // Prepare data
    const data = Object.entries(stats.byCategory)
      .map(([category, count]) => ({ category, count }))
      .sort((a, b) => b.count - a.count);

    // Pie layout
    const pie = d3.pie<{category: string; count: number}>()
      .value(d => d.count)
      .sort(null);

    const arc = d3.arc<any>()
      .innerRadius(radius * 0.6)
      .outerRadius(radius * 0.9);

    const outerArc = d3.arc<any>()
      .innerRadius(radius * 0.9)
      .outerRadius(radius * 0.9);

    const colorScale = d3.scaleOrdinal()
      .domain(data.map(d => d.category))
      .range(d3.schemeSet3);

    // Slices
    const slices = g.selectAll('.slice')
      .data(pie(data))
      .join('g')
      .attr('class', 'slice');

    slices.append('path')
      .attr('d', arc)
      .attr('fill', d => colorScale(d.data.category) as string)
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setFilterCategory(d.data.category);
        if (onFilter) {
          onFilter(d.data.category, filterLevel);
        }
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('d', d3.arc<any>()
            .innerRadius(radius * 0.6)
            .outerRadius(radius * 0.95)
          );
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('d', arc);
      });

    // Labels
    const labels = g.selectAll('.label')
      .data(pie(data))
      .join('g')
      .attr('class', 'label');

    labels.append('text')
      .attr('transform', d => {
        const pos = outerArc.centroid(d);
        const midAngle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        pos[0] = radius * 0.95 * (midAngle < Math.PI ? 1 : -1);
        return `translate(${pos})`;
      })
      .style('text-anchor', d => {
        const midAngle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        return midAngle < Math.PI ? 'start' : 'end';
      })
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .text(d => d.data.category);

    // Polylines
    labels.append('polyline')
      .attr('points', d => {
        const pos = outerArc.centroid(d);
        const midAngle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        pos[0] = radius * 0.95 * (midAngle < Math.PI ? 1 : -1);
        const innerPos = arc.centroid(d);
        const outerPos = outerArc.centroid(d);
        return [innerPos, outerPos, pos].map(p => p.join(',')).join(' ');
      })
      .attr('fill', 'none')
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1);

    // Center text
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.5em')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text(stats.total);

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1.5em')
      .attr('font-size', '12px')
      .attr('fill', '#9ca3af')
      .text('Total Errors');

  }, [stats, onFilter, filterLevel]);

  const filteredErrors = errors.filter(error => {
    if (filterLevel && error.level !== filterLevel) return false;
    if (filterCategory && error.category !== filterCategory) return false;
    return true;
  });

  const selectedErrorData = errors.find(e => e.id === selectedError);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Error Logging Dashboard</h3>
        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {(['1h', '24h', '7d'] as const).map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  timeRange === range 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Total Errors</div>
          <div className="text-2xl font-bold text-white">{stats.total}</div>
          <div className="text-xs text-gray-500 mt-1">
            {stats.topErrors.filter(e => !e.resolved).length} unresolved
          </div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Critical</div>
          <div className="text-2xl font-bold text-red-400">
            {stats.byLevel.critical || 0}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {((stats.byLevel.critical || 0) / stats.total * 100).toFixed(1)}% of total
          </div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Error Rate</div>
          <div className="text-2xl font-bold text-orange-400">
            {stats.trend.length > 0 
              ? (stats.trend[stats.trend.length - 1].count / (timeRange === '1h' ? 60 : timeRange === '24h' ? 1440 : 10080)).toFixed(2)
              : 0}/min
          </div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Top Category</div>
          <div className="text-2xl font-bold text-white">
            {Object.entries(stats.byCategory).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Error Trend */}
        <div className="lg:col-span-2 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Error Trend</h4>
          <svg 
            ref={trendChartRef}
            width="600"
            height="200"
            className="w-full h-auto"
            viewBox="0 0 600 200"
            preserveAspectRatio="xMidYMid meet"
          />
        </div>

        {/* Category Distribution */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">By Category</h4>
          <svg 
            ref={distributionRef}
            width="300"
            height="300"
            className="w-full h-auto"
            viewBox="0 0 300 300"
            preserveAspectRatio="xMidYMid meet"
          />
        </div>
      </div>

      {/* Filters */}
      <div className="mt-6 bg-gray-800 rounded p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-lg font-medium text-white">Error Logs</h4>
          <div className="flex items-center space-x-4">
            <select
              value={filterLevel || ''}
              onChange={(e) => {
                setFilterLevel(e.target.value || null);
                if (onFilter) onFilter(filterCategory, e.target.value || null);
              }}
              className="bg-gray-700 text-white rounded px-3 py-1 text-sm"
            >
              <option value="">All Levels</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="critical">Critical</option>
            </select>
            
            <select
              value={filterCategory || ''}
              onChange={(e) => {
                setFilterCategory(e.target.value || null);
                if (onFilter) onFilter(e.target.value || null, filterLevel);
              }}
              className="bg-gray-700 text-white rounded px-3 py-1 text-sm"
            >
              <option value="">All Categories</option>
              {Object.keys(stats.byCategory).map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>

            {(filterLevel || filterCategory) && (
              <button
                onClick={() => {
                  setFilterLevel(null);
                  setFilterCategory(null);
                  if (onFilter) onFilter(null, null);
                }}
                className="text-gray-400 hover:text-white text-sm"
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>

        {/* Error List */}
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredErrors.map(error => (
            <div
              key={error.id}
              className={`p-3 rounded cursor-pointer transition-all ${
                selectedError === error.id 
                  ? 'bg-blue-500/20 border border-blue-500' 
                  : 'bg-gray-700 hover:bg-gray-600'
              } ${error.resolved ? 'opacity-50' : ''}`}
              onClick={() => setSelectedError(error.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className={`text-xs px-2 py-1 rounded ${
                      error.level === 'critical' ? 'bg-red-500/20 text-red-400' :
                      error.level === 'error' ? 'bg-orange-500/20 text-orange-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {error.level.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-400">{error.category}</span>
                    <span className="text-xs text-gray-500">
                      {new Date(error.timestamp).toLocaleTimeString()}
                    </span>
                    {error.frequency > 1 && (
                      <span className="text-xs bg-gray-600 px-2 py-1 rounded">
                        {error.frequency}x
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-sm text-white">{error.message}</div>
                  <div className="mt-1 text-xs text-gray-400">
                    {error.context.service} → {error.context.operation}
                  </div>
                </div>
                {!error.resolved && onResolve && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onResolve(error.id);
                    }}
                    className="ml-2 text-green-400 hover:text-green-300"
                    title="Mark as resolved"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Error Details */}
      {selectedErrorData && (
        <div className="mt-6 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Error Details</h4>
          <div className="space-y-4">
            <div>
              <div className="text-sm text-gray-400">Message</div>
              <div className="text-white">{selectedErrorData.message}</div>
            </div>
            
            {selectedErrorData.stack && (
              <div>
                <div className="text-sm text-gray-400">Stack Trace</div>
                <pre className="mt-1 bg-gray-900 rounded p-3 text-xs text-gray-300 overflow-x-auto">
                  {selectedErrorData.stack}
                </pre>
              </div>
            )}
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-400">First Seen</div>
                <div className="text-white">
                  {new Date(selectedErrorData.firstSeen).toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400">Last Seen</div>
                <div className="text-white">
                  {new Date(selectedErrorData.lastSeen).toLocaleString()}
                </div>
              </div>
            </div>
            
            {selectedErrorData.context.metadata && (
              <div>
                <div className="text-sm text-gray-400">Additional Context</div>
                <div className="mt-1 bg-gray-700 rounded p-3">
                  {Object.entries(selectedErrorData.context.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-gray-400">{key}:</span>
                      <span className="text-white">{JSON.stringify(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}