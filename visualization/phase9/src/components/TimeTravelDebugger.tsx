import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { TimeTravelSession, TimeTravelSnapshot } from '../types/debugging';

interface TimeTravelDebuggerProps {
  session: TimeTravelSession;
  onSnapshotChange: (snapshot: TimeTravelSnapshot) => void;
  onCompare?: (base: string, compare: string) => void;
  className?: string;
}

export function TimeTravelDebugger({ 
  session, 
  onSnapshotChange,
  onCompare,
  className = '' 
}: TimeTravelDebuggerProps) {
  const timelineRef = useRef<SVGSVGElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedSnapshots, setSelectedSnapshots] = useState<string[]>([]);
  const [hoveredSnapshot, setHoveredSnapshot] = useState<string | null>(null);

  const currentSnapshot = session.snapshots[session.currentIndex];

  // Timeline visualization
  useEffect(() => {
    if (!timelineRef.current || session.snapshots.length === 0) return;

    const margin = { top: 40, right: 20, bottom: 40, left: 60 };
    const width = 800 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain([
        new Date(session.snapshots[0].timestamp),
        new Date(session.snapshots[session.snapshots.length - 1].timestamp)
      ])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0]);

    // Background
    g.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#1f2937')
      .attr('opacity', 0.5);

    // CPU usage area
    const cpuArea = d3.area<TimeTravelSnapshot>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(height)
      .y1(d => yScale(d.metadata.performance.cpu))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(session.snapshots)
      .attr('fill', '#3b82f6')
      .attr('opacity', 0.3)
      .attr('d', cpuArea);

    // Memory usage area
    const memoryArea = d3.area<TimeTravelSnapshot>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(height)
      .y1(d => yScale(d.metadata.performance.memory))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(session.snapshots)
      .attr('fill', '#10b981')
      .attr('opacity', 0.3)
      .attr('d', memoryArea);

    // Snapshot markers
    const markers = g.selectAll('.snapshot-marker')
      .data(session.snapshots)
      .join('g')
      .attr('class', 'snapshot-marker')
      .attr('transform', d => `translate(${xScale(new Date(d.timestamp))}, ${height / 2})`);

    markers.append('line')
      .attr('y1', -height / 2)
      .attr('y2', height / 2)
      .attr('stroke', d => {
        if (d.id === currentSnapshot.id) return '#3b82f6';
        if (selectedSnapshots.includes(d.id)) return '#f59e0b';
        return '#6b7280';
      })
      .attr('stroke-width', d => d.id === currentSnapshot.id ? 3 : 1)
      .attr('stroke-dasharray', d => d.metadata.trigger === 'auto' ? '2,2' : '');

    markers.append('circle')
      .attr('r', d => {
        if (d.id === currentSnapshot.id) return 8;
        if (selectedSnapshots.includes(d.id)) return 6;
        return 4;
      })
      .attr('fill', d => {
        if (d.id === currentSnapshot.id) return '#3b82f6';
        if (selectedSnapshots.includes(d.id)) return '#f59e0b';
        return '#9ca3af';
      })
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (compareMode) {
          const newSelection = selectedSnapshots.includes(d.id)
            ? selectedSnapshots.filter(id => id !== d.id)
            : [...selectedSnapshots, d.id].slice(-2);
          setSelectedSnapshots(newSelection);
          
          if (newSelection.length === 2 && onCompare) {
            onCompare(newSelection[0], newSelection[1]);
          }
        } else {
          const index = session.snapshots.findIndex(s => s.id === d.id);
          session.currentIndex = index;
          onSnapshotChange(d);
        }
      })
      .on('mouseover', (event, d) => {
        setHoveredSnapshot(d.id);
        showTooltip(event, d);
      })
      .on('mouseout', () => {
        setHoveredSnapshot(null);
        hideTooltip();
      });

    // Labels for significant snapshots
    markers.filter(d => d.metadata.trigger !== 'auto')
      .append('text')
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '10px')
      .text(d => d.label);

    // Time axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat('%H:%M:%S'));

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis);

    // Current time indicator
    const currentTime = new Date(currentSnapshot.timestamp);
    g.append('line')
      .attr('x1', xScale(currentTime))
      .attr('x2', xScale(currentTime))
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2);

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'timetravel-tooltip')
      .style('position', 'absolute')
      .style('padding', '10px')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('color', 'white')
      .style('border-radius', '5px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    function showTooltip(event: MouseEvent, d: TimeTravelSnapshot) {
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <div style="font-weight: bold;">${d.label}</div>
        <div>Time: ${new Date(d.timestamp).toLocaleTimeString()}</div>
        <div>Trigger: ${d.metadata.trigger}</div>
        <div>CPU: ${d.metadata.performance.cpu.toFixed(1)}%</div>
        <div>Memory: ${d.metadata.performance.memory.toFixed(1)}%</div>
        <div>Changes: ${d.metadata.changes.length}</div>
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
  }, [session, currentSnapshot, selectedSnapshots, compareMode]);

  // Playback control
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      const nextIndex = (session.currentIndex + 1) % session.snapshots.length;
      session.currentIndex = nextIndex;
      onSnapshotChange(session.snapshots[nextIndex]);
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, session, onSnapshotChange]);

  const handleStepBackward = () => {
    if (session.currentIndex > 0) {
      session.currentIndex--;
      onSnapshotChange(session.snapshots[session.currentIndex]);
    }
  };

  const handleStepForward = () => {
    if (session.currentIndex < session.snapshots.length - 1) {
      session.currentIndex++;
      onSnapshotChange(session.snapshots[session.currentIndex]);
    }
  };

  const handleJumpToStart = () => {
    session.currentIndex = 0;
    onSnapshotChange(session.snapshots[0]);
  };

  const handleJumpToEnd = () => {
    session.currentIndex = session.snapshots.length - 1;
    onSnapshotChange(session.snapshots[session.currentIndex]);
  };

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Time-Travel Debugger</h3>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              compareMode ? 'bg-orange-500 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {compareMode ? 'Exit Compare' : 'Compare Mode'}
          </button>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-gray-800 rounded p-4 mb-6">
        <svg 
          ref={timelineRef}
          width="800"
          height="200"
          className="w-full h-auto"
          viewBox="0 0 800 200"
          preserveAspectRatio="xMidYMid meet"
        />
      </div>

      {/* Playback Controls */}
      <div className="bg-gray-800 rounded p-4 mb-6">
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={handleJumpToStart}
            className="p-2 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Jump to Start"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M8.445 14.832A1 1 0 0010 14v-8a1 1 0 00-1.555-.832L3 9.168V6a1 1 0 00-2 0v8a1 1 0 002 0v-3.168l5.445 4z"/>
            </svg>
          </button>
          
          <button
            onClick={handleStepBackward}
            className="p-2 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Step Backward"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M12.707 14.707a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 1.414L9.414 10l3.293 3.293a1 1 0 010 1.414z"/>
            </svg>
          </button>
          
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-3 rounded-full bg-blue-500 hover:bg-blue-600 text-white transition-colors"
          >
            {isPlaying ? (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd"/>
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd"/>
              </svg>
            )}
          </button>
          
          <button
            onClick={handleStepForward}
            className="p-2 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Step Forward"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"/>
            </svg>
          </button>
          
          <button
            onClick={handleJumpToEnd}
            className="p-2 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Jump to End"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M11.555 5.168A1 1 0 0010 6v8a1 1 0 001.555.832L17 10.832V14a1 1 0 102 0V6a1 1 0 10-2 0v3.168l-5.445-4z"/>
            </svg>
          </button>
        </div>

        <div className="flex items-center justify-center mt-4 space-x-4">
          <span className="text-sm text-gray-400">Speed:</span>
          <div className="flex bg-gray-700 rounded-lg p-1">
            {[0.5, 1, 2, 4].map(speed => (
              <button
                key={speed}
                onClick={() => setPlaybackSpeed(speed)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  playbackSpeed === speed 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Current Snapshot Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Current Snapshot</h4>
          <div className="space-y-3">
            <div>
              <div className="text-sm text-gray-400">Label</div>
              <div className="text-white">{currentSnapshot.label}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Time</div>
              <div className="text-white">
                {new Date(currentSnapshot.timestamp).toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Trigger</div>
              <div className="text-white capitalize">{currentSnapshot.metadata.trigger}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Position</div>
              <div className="text-white">
                {session.currentIndex + 1} / {session.snapshots.length}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">State Summary</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-400">Patterns</span>
              <span className="text-white">{currentSnapshot.state.patterns.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-400">Connections</span>
              <span className="text-white">{currentSnapshot.state.connections.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-400">CPU Usage</span>
              <span className="text-white">{currentSnapshot.metadata.performance.cpu.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-400">Memory Usage</span>
              <span className="text-white">{currentSnapshot.metadata.performance.memory.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Changes in Current Snapshot */}
      {currentSnapshot.metadata.changes.length > 0 && (
        <div className="mt-6 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Changes in This Snapshot</h4>
          <div className="space-y-2">
            {currentSnapshot.metadata.changes.map((change, i) => (
              <div key={i} className="flex items-center text-sm">
                <svg className="w-4 h-4 text-blue-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M12.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd"/>
                </svg>
                <span className="text-gray-300">{change}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compare Mode Info */}
      {compareMode && selectedSnapshots.length === 2 && (
        <div className="mt-6 bg-orange-500/10 border border-orange-500/20 rounded p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-orange-400 font-medium">Compare Mode Active</div>
              <div className="text-sm text-gray-300 mt-1">
                Comparing snapshots at {new Date(session.snapshots.find(s => s.id === selectedSnapshots[0])!.timestamp).toLocaleTimeString()}
                {' and '}
                {new Date(session.snapshots.find(s => s.id === selectedSnapshots[1])!.timestamp).toLocaleTimeString()}
              </div>
            </div>
            <button
              onClick={() => {
                if (onCompare) {
                  onCompare(selectedSnapshots[0], selectedSnapshots[1]);
                }
              }}
              className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 transition-colors"
            >
              View Comparison
            </button>
          </div>
        </div>
      )}
    </div>
  );
}