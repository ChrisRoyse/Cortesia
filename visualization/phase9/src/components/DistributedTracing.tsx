import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { DistributedTrace, TraceSpan } from '../types/debugging';

interface DistributedTracingProps {
  traces: DistributedTrace[];
  className?: string;
}

export function DistributedTracing({ traces, className = '' }: DistributedTracingProps) {
  const timelineRef = useRef<SVGSVGElement>(null);
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const [selectedSpan, setSelectedSpan] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'timeline' | 'waterfall' | 'graph'>('waterfall');

  const selectedTraceData = traces.find(t => t.traceId === selectedTrace);

  // Waterfall view
  useEffect(() => {
    if (!timelineRef.current || !selectedTraceData || viewMode !== 'waterfall') return;

    const margin = { top: 20, right: 200, bottom: 40, left: 200 };
    const width = 1000 - margin.left - margin.right;
    const height = Math.max(400, selectedTraceData.spans.length * 30) - margin.top - margin.bottom;

    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Sort spans by start time
    const sortedSpans = [...selectedTraceData.spans].sort((a, b) => a.startTime - b.startTime);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, selectedTraceData.duration])
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(sortedSpans.map(s => s.spanId))
      .range([0, height])
      .padding(0.1);

    const colorScale = d3.scaleOrdinal()
      .domain(['success', 'warning', 'error'])
      .range(['#10b981', '#f59e0b', '#ef4444']);

    // Grid lines
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => `${d}ms`);

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis);

    // Service labels
    const serviceGroups = d3.group(sortedSpans, d => d.serviceName);
    let yOffset = 0;
    
    serviceGroups.forEach((spans, service) => {
      const serviceHeight = spans.length * yScale.bandwidth() + (spans.length - 1) * yScale.padding();
      
      g.append('rect')
        .attr('x', -margin.left)
        .attr('y', yOffset)
        .attr('width', margin.left - 10)
        .attr('height', serviceHeight)
        .attr('fill', '#374151')
        .attr('rx', 4);
      
      g.append('text')
        .attr('x', -margin.left / 2)
        .attr('y', yOffset + serviceHeight / 2)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(service);
      
      yOffset += serviceHeight + yScale.padding();
    });

    // Span bars
    const spanGroups = g.selectAll('.span')
      .data(sortedSpans)
      .join('g')
      .attr('class', 'span')
      .attr('transform', d => `translate(0, ${yScale(d.spanId)})`);

    spanGroups.append('rect')
      .attr('x', d => xScale(d.startTime - selectedTraceData.startTime))
      .attr('width', d => xScale(d.duration))
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.status) as string)
      .attr('rx', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => setSelectedSpan(d.spanId))
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
        showTooltip(event, d);
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 1);
        hideTooltip();
      });

    spanGroups.append('text')
      .attr('x', d => xScale(d.startTime - selectedTraceData.startTime) + 5)
      .attr('y', yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('fill', 'white')
      .attr('font-size', '12px')
      .text(d => d.operationName)
      .style('pointer-events', 'none');

    // Duration labels
    spanGroups.append('text')
      .attr('x', d => xScale(d.startTime - selectedTraceData.startTime + d.duration) + 5)
      .attr('y', yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text(d => `${d.duration}ms`)
      .style('pointer-events', 'none');

    // Parent-child connections
    sortedSpans.forEach(span => {
      if (span.parentSpanId) {
        const parent = sortedSpans.find(s => s.spanId === span.parentSpanId);
        if (parent) {
          const parentY = yScale(parent.spanId)! + yScale.bandwidth() / 2;
          const childY = yScale(span.spanId)! + yScale.bandwidth() / 2;
          const x = xScale(span.startTime - selectedTraceData.startTime);

          g.append('path')
            .attr('d', `M${x},${parentY} L${x},${childY}`)
            .attr('stroke', '#4b5563')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '2,2');
        }
      }
    });

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'trace-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('border-radius', '5px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    function showTooltip(event: MouseEvent, d: TraceSpan) {
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <div style="font-weight: bold;">${d.operationName}</div>
        <div>Service: ${d.serviceName}</div>
        <div>Duration: ${d.duration}ms</div>
        <div>Status: ${d.status}</div>
        ${Object.entries(d.tags).map(([k, v]) => 
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
  }, [selectedTraceData, viewMode]);

  // Graph view
  useEffect(() => {
    if (!timelineRef.current || !selectedTraceData || viewMode !== 'graph') return;

    const width = 1000;
    const height = 600;

    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Create nodes and links
    const nodes = selectedTraceData.spans.map(span => ({
      id: span.spanId,
      span,
      group: span.serviceName
    }));

    const links = selectedTraceData.spans
      .filter(span => span.parentSpanId)
      .map(span => ({
        source: span.parentSpanId!,
        target: span.spanId,
        value: span.duration
      }));

    // Force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    const colorScale = d3.scaleOrdinal()
      .domain(selectedTraceData.services)
      .range(d3.schemeCategory10);

    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#4b5563')
      .attr('stroke-width', d => Math.sqrt(d.value / 100));

    // Nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    node.append('circle')
      .attr('r', d => 10 + Math.sqrt(d.span.duration / 10))
      .attr('fill', d => colorScale(d.group) as string)
      .attr('stroke', d => d.span.status === 'error' ? '#ef4444' : '#1f2937')
      .attr('stroke-width', 2);

    node.append('text')
      .text(d => d.span.operationName.substring(0, 20))
      .attr('x', 0)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '10px');

    // Update positions
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y);

      node.attr('transform', d => `translate(${(d as any).x},${(d as any).y})`);
    });

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

    return () => {
      simulation.stop();
    };
  }, [selectedTraceData, viewMode]);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Distributed Tracing</h3>
        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {(['waterfall', 'graph', 'timeline'] as const).map(mode => (
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
        {/* Trace List */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Recent Traces</h4>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {traces.map(trace => (
              <div
                key={trace.traceId}
                className={`p-3 rounded cursor-pointer transition-all ${
                  selectedTrace === trace.traceId 
                    ? 'bg-blue-500/20 border border-blue-500' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
                onClick={() => setSelectedTrace(trace.traceId)}
              >
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">
                    {new Date(trace.startTime).toLocaleTimeString()}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    trace.errorCount > 0 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                  }`}>
                    {trace.errorCount > 0 ? `${trace.errorCount} errors` : 'Success'}
                  </span>
                </div>
                <div className="mt-1 text-sm text-white truncate">
                  {trace.rootSpan.operationName}
                </div>
                <div className="mt-1 flex justify-between text-xs text-gray-400">
                  <span>{trace.duration}ms</span>
                  <span>{trace.spanCount} spans</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trace Visualization */}
        <div className="lg:col-span-3 bg-gray-800 rounded p-4">
          {selectedTraceData ? (
            <>
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-lg font-medium text-white">
                  Trace: {selectedTraceData.traceId.substring(0, 8)}...
                </h4>
                <div className="flex items-center space-x-4 text-sm">
                  <span className="text-gray-400">
                    Duration: <span className="text-white">{selectedTraceData.duration}ms</span>
                  </span>
                  <span className="text-gray-400">
                    Services: <span className="text-white">{selectedTraceData.services.length}</span>
                  </span>
                  <span className="text-gray-400">
                    Spans: <span className="text-white">{selectedTraceData.spanCount}</span>
                  </span>
                </div>
              </div>
              
              <svg 
                ref={timelineRef}
                width="1000"
                height={viewMode === 'waterfall' ? Math.max(400, selectedTraceData.spans.length * 30) : 600}
                className="w-full h-auto"
                viewBox={`0 0 1000 ${viewMode === 'waterfall' ? Math.max(400, selectedTraceData.spans.length * 30) : 600}`}
                preserveAspectRatio="xMidYMid meet"
              />

              {/* Span Details */}
              {selectedSpan && (
                <div className="mt-4 bg-gray-700 rounded p-4">
                  <h5 className="text-lg font-medium text-white mb-2">Span Details</h5>
                  {(() => {
                    const span = selectedTraceData.spans.find(s => s.spanId === selectedSpan);
                    if (!span) return null;
                    
                    return (
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-400">Operation:</span>
                          <span className="text-white ml-2">{span.operationName}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Service:</span>
                          <span className="text-white ml-2">{span.serviceName}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Duration:</span>
                          <span className="text-white ml-2">{span.duration}ms</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Status:</span>
                          <span className={`ml-2 ${
                            span.status === 'error' ? 'text-red-400' : 
                            span.status === 'warning' ? 'text-yellow-400' : 'text-green-400'
                          }`}>
                            {span.status}
                          </span>
                        </div>
                        {span.logs.length > 0 && (
                          <div className="col-span-2">
                            <div className="text-gray-400 mb-2">Logs:</div>
                            <div className="space-y-1">
                              {span.logs.map((log, i) => (
                                <div key={i} className="text-xs">
                                  <span className={`${
                                    log.level === 'error' ? 'text-red-400' :
                                    log.level === 'warn' ? 'text-yellow-400' :
                                    'text-gray-300'
                                  }`}>
                                    [{log.level}]
                                  </span>
                                  <span className="text-gray-300 ml-2">{log.message}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </>
          ) : (
            <div className="text-gray-500 text-center py-20">
              Select a trace to view details
            </div>
          )}
        </div>
      </div>

      {/* Service Summary */}
      {selectedTraceData && (
        <div className="mt-6 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Service Summary</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {selectedTraceData.services.map(service => {
              const serviceSpans = selectedTraceData.spans.filter(s => s.serviceName === service);
              const totalDuration = serviceSpans.reduce((sum, s) => sum + s.duration, 0);
              const errorCount = serviceSpans.filter(s => s.status === 'error').length;
              
              return (
                <div key={service} className="bg-gray-700 rounded p-3">
                  <div className="text-sm font-medium text-white">{service}</div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">Spans:</span>
                      <span className="text-white ml-1">{serviceSpans.length}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Time:</span>
                      <span className="text-white ml-1">{totalDuration}ms</span>
                    </div>
                    <div className="col-span-2">
                      <span className="text-gray-400">Errors:</span>
                      <span className={`ml-1 ${errorCount > 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {errorCount}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}