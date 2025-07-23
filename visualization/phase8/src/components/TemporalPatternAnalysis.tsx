import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { TemporalPattern, TemporalEvent } from '../types/cognitive';

interface TemporalPatternAnalysisProps {
  patterns: TemporalPattern[];
  events: TemporalEvent[];
  className?: string;
}

export function TemporalPatternAnalysis({ 
  patterns, 
  events, 
  className = '' 
}: TemporalPatternAnalysisProps) {
  const timelineRef = useRef<SVGSVGElement>(null);
  const matrixRef = useRef<SVGSVGElement>(null);
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [timeWindow, setTimeWindow] = useState<'1h' | '6h' | '24h'>('6h');

  // Timeline visualization
  useEffect(() => {
    if (!timelineRef.current || events.length === 0) return;

    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Filter events based on time window
    const now = Date.now();
    const windowMs = timeWindow === '1h' ? 3600000 : timeWindow === '6h' ? 21600000 : 86400000;
    const filteredEvents = events.filter(e => now - e.timestamp < windowMs);

    // Scales
    const xScale = d3.scaleTime()
      .domain([now - windowMs, now])
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(Array.from(new Set(filteredEvents.map(e => e.patternId))))
      .range([0, height])
      .padding(0.1);

    const colorScale = d3.scaleOrdinal()
      .domain(Array.from(new Set(filteredEvents.map(e => e.patternId))))
      .range(d3.schemeCategory10);

    // Background
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#1f2937')
      .attr('opacity', 0.5);

    // Grid lines
    const xAxis = d3.axisBottom(xScale)
      .ticks(d3.timeHour.every(timeWindow === '1h' ? 0.25 : timeWindow === '6h' ? 1 : 4))
      .tickFormat(d3.timeFormat('%H:%M'));

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .append('text')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Time');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('font-size', '10px');

    // Event rectangles
    const eventGroups = g.selectAll('.event')
      .data(filteredEvents)
      .join('g')
      .attr('class', 'event');

    eventGroups.append('rect')
      .attr('x', d => xScale(d.timestamp))
      .attr('y', d => yScale(d.patternId) || 0)
      .attr('width', 3)
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.patternId) as string)
      .attr('opacity', d => d.activation)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('width', 6)
          .attr('opacity', 1);
        
        showTooltip(event, d);
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('width', 3)
          .attr('opacity', d.activation);
        
        hideTooltip();
      })
      .on('click', (event, d) => {
        const pattern = patterns.find(p => p.sequence.some(e => e.patternId === d.patternId));
        if (pattern) setSelectedPattern(pattern.id);
      });

    // Pattern connections
    patterns.forEach(pattern => {
      const patternEvents = pattern.sequence
        .filter(e => filteredEvents.some(fe => 
          fe.patternId === e.patternId && 
          Math.abs(fe.timestamp - e.timestamp) < 1000
        ))
        .sort((a, b) => a.timestamp - b.timestamp);

      if (patternEvents.length > 1) {
        const line = d3.line<TemporalEvent>()
          .x(d => xScale(d.timestamp) + 1.5)
          .y(d => (yScale(d.patternId) || 0) + yScale.bandwidth() / 2)
          .curve(d3.curveBasis);

        g.append('path')
          .datum(patternEvents)
          .attr('fill', 'none')
          .attr('stroke', '#6366f1')
          .attr('stroke-width', 2)
          .attr('stroke-opacity', 0.3)
          .attr('d', line);
      }
    });

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'temporal-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('border-radius', '5px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    function showTooltip(event: MouseEvent, d: TemporalEvent) {
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <div style="font-weight: bold;">${d.patternId}</div>
        <div>Time: ${new Date(d.timestamp).toLocaleTimeString()}</div>
        <div>Activation: ${(d.activation * 100).toFixed(1)}%</div>
        <div>Context: ${d.context.join(', ')}</div>
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
  }, [events, patterns, timeWindow]);

  // Pattern correlation matrix
  useEffect(() => {
    if (!matrixRef.current || patterns.length === 0) return;

    const size = 300;
    const margin = 50;
    const cellSize = (size - 2 * margin) / patterns.length;

    const svg = d3.select(matrixRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin},${margin})`);

    // Calculate correlations
    const correlations: number[][] = patterns.map((p1, i) => 
      patterns.map((p2, j) => {
        if (i === j) return 1;
        
        // Calculate temporal correlation based on sequence overlap
        const overlap = p1.sequence.filter(e1 => 
          p2.sequence.some(e2 => 
            Math.abs(e1.timestamp - e2.timestamp) < 5000 && 
            e1.patternId === e2.patternId
          )
        ).length;
        
        return overlap / Math.max(p1.sequence.length, p2.sequence.length);
      })
    );

    // Color scale
    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateBlues);

    // Draw cells
    patterns.forEach((p1, i) => {
      patterns.forEach((p2, j) => {
        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', colorScale(correlations[i][j]))
          .style('cursor', 'pointer')
          .on('mouseover', function(event) {
            d3.select(this)
              .attr('stroke', 'white')
              .attr('stroke-width', 2);
            
            // Highlight row and column
            g.selectAll('rect')
              .attr('opacity', (d, idx) => {
                const row = Math.floor(idx / patterns.length);
                const col = idx % patterns.length;
                return row === i || col === j ? 1 : 0.3;
              });
          })
          .on('mouseout', function() {
            d3.select(this)
              .attr('stroke', 'none');
            
            g.selectAll('rect').attr('opacity', 1);
          });
      });
    });

    // Labels
    patterns.forEach((p, i) => {
      // Row labels
      g.append('text')
        .attr('x', -5)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dy', '0.35em')
        .attr('font-size', '10px')
        .attr('fill', '#9ca3af')
        .text(p.id.substring(0, 8));

      // Column labels
      g.append('text')
        .attr('x', i * cellSize + cellSize / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', '#9ca3af')
        .attr('transform', `rotate(-45, ${i * cellSize + cellSize / 2}, -5)`)
        .text(p.id.substring(0, 8));
    });
  }, [patterns]);

  const selectedPatternData = patterns.find(p => p.id === selectedPattern);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Temporal Pattern Analysis</h3>
        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {(['1h', '6h', '24h'] as const).map(window => (
              <button
                key={window}
                onClick={() => setTimeWindow(window)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  timeWindow === window 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                {window}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Pattern Timeline */}
      <div className="bg-gray-800 rounded p-4 mb-6">
        <h4 className="text-lg font-medium text-white mb-3">Pattern Timeline</h4>
        <svg 
          ref={timelineRef}
          width="800"
          height="400"
          className="w-full h-auto"
          viewBox="0 0 800 400"
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pattern List */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Detected Patterns</h4>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {patterns.map(pattern => (
              <div
                key={pattern.id}
                className={`p-3 rounded cursor-pointer transition-all ${
                  selectedPattern === pattern.id 
                    ? 'bg-blue-500/20 border border-blue-500' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
                onClick={() => setSelectedPattern(pattern.id)}
              >
                <div className="flex justify-between items-center">
                  <span className="text-sm text-white">{pattern.id}</span>
                  <span className="text-xs text-gray-400">
                    {pattern.sequence.length} events
                  </span>
                </div>
                <div className="mt-1 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Frequency:</span>
                    <span className="text-gray-300 ml-1">{pattern.frequency}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Predictability:</span>
                    <span className="text-gray-300 ml-1">
                      {(pattern.predictability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Pattern Correlations</h4>
          <svg 
            ref={matrixRef}
            width="300"
            height="300"
            className="w-full h-auto"
            viewBox="0 0 300 300"
            preserveAspectRatio="xMidYMid meet"
          />
          <div className="mt-2 text-xs text-gray-500 text-center">
            Darker = Higher Correlation
          </div>
        </div>

        {/* Selected Pattern Details */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Pattern Details</h4>
          {selectedPatternData ? (
            <div className="space-y-4">
              <div>
                <div className="text-sm text-gray-400">Pattern ID</div>
                <div className="text-white">{selectedPatternData.id}</div>
              </div>
              
              <div>
                <div className="text-sm text-gray-400">Sequence Length</div>
                <div className="text-white">{selectedPatternData.sequence.length} events</div>
              </div>
              
              <div>
                <div className="text-sm text-gray-400">Duration</div>
                <div className="text-white">{(selectedPatternData.duration / 1000).toFixed(1)}s</div>
              </div>
              
              <div>
                <div className="text-sm text-gray-400">Frequency</div>
                <div className="text-white">{selectedPatternData.frequency} occurrences</div>
              </div>
              
              <div>
                <div className="text-sm text-gray-400">Predictability</div>
                <div className="flex items-center">
                  <div className="flex-1 bg-gray-700 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full bg-blue-500"
                      style={{ width: `${selectedPatternData.predictability * 100}%` }}
                    />
                  </div>
                  <span className="ml-2 text-white">
                    {(selectedPatternData.predictability * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              
              {selectedPatternData.nextPredicted && (
                <div>
                  <div className="text-sm text-gray-400">Next Predicted Event</div>
                  <div className="mt-1 p-2 bg-gray-700 rounded">
                    <div className="text-xs text-white">{selectedPatternData.nextPredicted.patternId}</div>
                    <div className="text-xs text-gray-400">
                      in ~{((selectedPatternData.nextPredicted.timestamp - Date.now()) / 1000).toFixed(0)}s
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-center py-8">
              Select a pattern to view details
            </div>
          )}
        </div>
      </div>

      {/* Pattern Statistics */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Total Patterns</div>
          <div className="text-2xl font-bold text-white">{patterns.length}</div>
        </div>
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Total Events</div>
          <div className="text-2xl font-bold text-blue-400">{events.length}</div>
        </div>
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Avg Predictability</div>
          <div className="text-2xl font-bold text-green-400">
            {patterns.length > 0 
              ? (patterns.reduce((sum, p) => sum + p.predictability, 0) / patterns.length * 100).toFixed(0)
              : 0}%
          </div>
        </div>
        <div className="bg-gray-800 rounded p-4">
          <div className="text-sm text-gray-400">Avg Duration</div>
          <div className="text-2xl font-bold text-orange-400">
            {patterns.length > 0
              ? (patterns.reduce((sum, p) => sum + p.duration, 0) / patterns.length / 1000).toFixed(1)
              : 0}s
          </div>
        </div>
      </div>
    </div>
  );
}