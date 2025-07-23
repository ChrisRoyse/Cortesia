import React, { useState, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { QueryAnalysis, PlanNode, OptimizationSuggestion } from '../types/debugging';

interface QueryAnalyzerProps {
  analyses: QueryAnalysis[];
  onOptimize?: (queryId: string, suggestion: OptimizationSuggestion) => void;
  className?: string;
}

export function QueryAnalyzer({ analyses, onOptimize, className = '' }: QueryAnalyzerProps) {
  const planRef = useRef<SVGSVGElement>(null);
  const [selectedQuery, setSelectedQuery] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'plan' | 'profile' | 'bottlenecks'>('plan');

  const selectedAnalysis = analyses.find(a => a.queryId === selectedQuery);

  // Query plan visualization
  useEffect(() => {
    if (!planRef.current || !selectedAnalysis || viewMode !== 'plan') return;

    const width = 800;
    const height = 600;

    const svg = d3.select(planRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Create hierarchy from plan nodes
    const root = d3.hierarchy(selectedAnalysis.plan.nodes[0], d => d.children);

    // Tree layout
    const treeLayout = d3.tree<PlanNode>()
      .size([width - 100, height - 100]);

    treeLayout(root);

    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Color scale based on cost
    const maxCost = d3.max(root.descendants(), d => d.data.cost) || 1;
    const colorScale = d3.scaleSequential()
      .domain([0, maxCost])
      .interpolator(d3.interpolateRdYlGn)
      .clamp(true);

    // Links
    const links = g.append('g')
      .selectAll('path')
      .data(root.links())
      .join('path')
      .attr('d', d3.linkVertical<any, any>()
        .x(d => d.x + 50)
        .y(d => d.y + 50)
      )
      .attr('fill', 'none')
      .attr('stroke', '#4b5563')
      .attr('stroke-width', 2);

    // Nodes
    const nodes = g.append('g')
      .selectAll('g')
      .data(root.descendants())
      .join('g')
      .attr('transform', d => `translate(${d.x + 50},${d.y + 50})`);

    // Node rectangles
    nodes.append('rect')
      .attr('x', -60)
      .attr('y', -25)
      .attr('width', 120)
      .attr('height', 50)
      .attr('fill', d => colorScale(maxCost - d.data.cost))
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .attr('rx', 5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke-width', 3);
        showNodeTooltip(event, d.data);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 2);
        hideTooltip();
      });

    // Node labels
    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-5')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text(d => d.data.operation);

    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '10')
      .attr('fill', 'rgba(255,255,255,0.8)')
      .attr('font-size', '10px')
      .text(d => `Cost: ${d.data.cost.toFixed(0)}`);

    // Cost indicators
    nodes.filter(d => d.data.cost > selectedAnalysis!.plan.estimatedCost * 0.3)
      .append('circle')
      .attr('cx', 55)
      .attr('cy', -20)
      .attr('r', 8)
      .attr('fill', '#ef4444')
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .append('title')
      .text('High cost operation');

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'query-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('border-radius', '5px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    function showNodeTooltip(event: MouseEvent, node: PlanNode) {
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <div style="font-weight: bold;">${node.operation}</div>
        <div>Type: ${node.type}</div>
        <div>Cost: ${node.cost.toFixed(2)}</div>
        <div>Rows: ${node.rows.toLocaleString()}</div>
        <div>Width: ${node.width}</div>
        ${Object.entries(node.details).map(([k, v]) => 
          `<div>${k}: ${v}</div>`
        ).join('')}
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
  }, [selectedAnalysis, viewMode]);

  // Bottleneck visualization
  useEffect(() => {
    if (!planRef.current || !selectedAnalysis || viewMode !== 'bottlenecks') return;

    const margin = { top: 20, right: 150, bottom: 40, left: 200 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(planRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = selectedAnalysis.bottlenecks.sort((a, b) => b.duration - a.duration);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.duration) || 0])
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(data.map(d => d.operation))
      .range([0, height])
      .padding(0.1);

    const colorScale = d3.scaleSequential()
      .domain([0, 100])
      .interpolator(d3.interpolateReds);

    // Bars
    g.selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', 0)
      .attr('y', d => yScale(d.operation) || 0)
      .attr('width', d => xScale(d.duration))
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.percentage))
      .attr('rx', 4);

    // Labels
    g.selectAll('.label')
      .data(data)
      .join('text')
      .attr('x', -10)
      .attr('y', d => (yScale(d.operation) || 0) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .text(d => d.component);

    // Duration labels
    g.selectAll('.duration')
      .data(data)
      .join('text')
      .attr('x', d => xScale(d.duration) + 5)
      .attr('y', d => (yScale(d.operation) || 0) + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text(d => `${d.duration}ms (${d.percentage.toFixed(1)}%)`);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${d}ms`));

  }, [selectedAnalysis, viewMode]);

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Query Analyzer</h3>
        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {(['plan', 'profile', 'bottlenecks'] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 rounded text-sm capitalize transition-colors ${
                  viewMode === mode 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Query List */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Recent Queries</h4>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {analyses.map(analysis => (
              <div
                key={analysis.queryId}
                className={`p-3 rounded cursor-pointer transition-all ${
                  selectedQuery === analysis.queryId 
                    ? 'bg-blue-500/20 border border-blue-500' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
                onClick={() => setSelectedQuery(analysis.queryId)}
              >
                <div className="text-xs text-gray-400">
                  {new Date(analysis.timestamp).toLocaleTimeString()}
                </div>
                <div className="mt-1 text-sm text-white truncate" title={analysis.query}>
                  {analysis.query.substring(0, 50)}...
                </div>
                <div className="mt-2 flex justify-between items-center">
                  <span className={`text-xs ${
                    analysis.executionTime > 1000 ? 'text-red-400' : 
                    analysis.executionTime > 500 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {formatDuration(analysis.executionTime)}
                  </span>
                  <span className="text-xs text-gray-400">
                    {analysis.profile.rowsProcessed.toLocaleString()} rows
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Analysis View */}
        <div className="lg:col-span-3">
          {selectedAnalysis ? (
            <>
              {/* Query Details */}
              <div className="bg-gray-800 rounded p-4 mb-6">
                <h4 className="text-lg font-medium text-white mb-3">Query Details</h4>
                <div className="bg-gray-700 rounded p-3 mb-4">
                  <code className="text-sm text-gray-300 whitespace-pre-wrap">
                    {selectedAnalysis.query}
                  </code>
                </div>
                
                {viewMode === 'plan' && (
                  <div className="bg-gray-700 rounded p-4">
                    <svg 
                      ref={planRef}
                      width="800"
                      height="600"
                      className="w-full h-auto"
                      viewBox="0 0 800 600"
                      preserveAspectRatio="xMidYMid meet"
                    />
                  </div>
                )}

                {viewMode === 'profile' && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Total Time</div>
                      <div className="text-xl font-bold text-white">
                        {formatDuration(selectedAnalysis.profile.actualTime)}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Planning</div>
                      <div className="text-xl font-bold text-blue-400">
                        {formatDuration(selectedAnalysis.profile.planningTime)}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Execution</div>
                      <div className="text-xl font-bold text-green-400">
                        {formatDuration(selectedAnalysis.profile.executionTime)}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Cache Hit Rate</div>
                      <div className="text-xl font-bold text-orange-400">
                        {selectedAnalysis.profile.cacheHits > 0 
                          ? ((selectedAnalysis.profile.cacheHits / 
                             (selectedAnalysis.profile.cacheHits + selectedAnalysis.profile.cacheMisses)) * 100).toFixed(1)
                          : 0}%
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Rows Processed</div>
                      <div className="text-xl font-bold text-white">
                        {selectedAnalysis.profile.rowsProcessed.toLocaleString()}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Bytes Processed</div>
                      <div className="text-xl font-bold text-white">
                        {formatBytes(selectedAnalysis.profile.bytesProcessed)}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Memory Used</div>
                      <div className="text-xl font-bold text-white">
                        {formatBytes(selectedAnalysis.profile.memoryUsed)}
                      </div>
                    </div>
                    <div className="bg-gray-700 rounded p-4">
                      <div className="text-sm text-gray-400">Cost</div>
                      <div className="text-xl font-bold text-white">
                        {selectedAnalysis.plan.estimatedCost.toFixed(0)}
                      </div>
                    </div>
                  </div>
                )}

                {viewMode === 'bottlenecks' && (
                  <div className="bg-gray-700 rounded p-4">
                    <svg 
                      ref={planRef}
                      width="800"
                      height="400"
                      className="w-full h-auto"
                      viewBox="0 0 800 400"
                      preserveAspectRatio="xMidYMid meet"
                    />
                  </div>
                )}
              </div>

              {/* Optimization Suggestions */}
              {selectedAnalysis.suggestions.length > 0 && (
                <div className="bg-gray-800 rounded p-4">
                  <h4 className="text-lg font-medium text-white mb-3">Optimization Suggestions</h4>
                  <div className="space-y-3">
                    {selectedAnalysis.suggestions.map((suggestion, i) => (
                      <div 
                        key={i} 
                        className={`border rounded p-4 ${
                          suggestion.priority === 'high' ? 'border-red-500/50 bg-red-500/10' :
                          suggestion.priority === 'medium' ? 'border-yellow-500/50 bg-yellow-500/10' :
                          'border-blue-500/50 bg-blue-500/10'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center">
                              <span className={`text-sm font-medium ${
                                suggestion.priority === 'high' ? 'text-red-400' :
                                suggestion.priority === 'medium' ? 'text-yellow-400' :
                                'text-blue-400'
                              }`}>
                                {suggestion.type.toUpperCase()} - {suggestion.priority.toUpperCase()}
                              </span>
                            </div>
                            <div className="mt-1 text-white">{suggestion.description}</div>
                            <div className="mt-2 text-sm text-gray-400">
                              <strong>Impact:</strong> {suggestion.impact}
                            </div>
                            <div className="mt-1 text-sm text-gray-400">
                              <strong>Implementation:</strong> {suggestion.implementation}
                            </div>
                          </div>
                          {onOptimize && (
                            <button
                              onClick={() => onOptimize(selectedAnalysis.queryId, suggestion)}
                              className="ml-4 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition-colors"
                            >
                              Apply
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-gray-800 rounded p-20 text-center">
              <div className="text-gray-500">Select a query to analyze</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(2)} ${units[unitIndex]}`;
}