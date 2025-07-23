import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { CognitivePattern, PatternConnection } from '../types/cognitive';

interface PatternActivation3DProps {
  patterns: CognitivePattern[];
  connections: PatternConnection[];
  className?: string;
}

export function PatternActivation3D({ patterns, connections, className = '' }: PatternActivation3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'3d' | 'top' | 'side'>('3d');

  useEffect(() => {
    if (!containerRef.current) return;

    const width = 800;
    const height = 600;
    const depth = 400;

    // Clear previous content
    d3.select(containerRef.current).selectAll('*').remove();

    const svg = d3.select(containerRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height/2})`);

    // Pattern type colors
    const colorScale = d3.scaleOrdinal()
      .domain(['convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract', 'adaptive', 'chain_of_thought', 'tree_of_thoughts'])
      .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#6366f1', '#84cc16']);

    // Position patterns in 3D space
    const positionedPatterns = patterns.map((pattern, i) => {
      const theta = (i / patterns.length) * 2 * Math.PI;
      const phi = (i / patterns.length) * Math.PI;
      const r = 150 + pattern.activation * 100;

      return {
        ...pattern,
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi)
      };
    });

    // Project 3D to 2D
    const project = (x: number, y: number, z: number) => {
      const perspective = 800;
      const rotX = rotation.x * Math.PI / 180;
      const rotY = rotation.y * Math.PI / 180;

      // Rotate around Y axis
      const x1 = x * Math.cos(rotY) - z * Math.sin(rotY);
      const z1 = x * Math.sin(rotY) + z * Math.cos(rotY);

      // Rotate around X axis
      const y1 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
      const z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);

      // Apply perspective
      const scale = perspective / (perspective + z2);
      
      if (viewMode === 'top') {
        return { x: x1 * scale, y: -z1 * scale, scale };
      } else if (viewMode === 'side') {
        return { x: z1 * scale, y: y1 * scale, scale };
      }
      
      return { x: x1 * scale, y: y1 * scale, scale };
    };

    // Draw connections
    const connectionsGroup = g.append('g').attr('class', 'connections');
    
    connections.forEach(conn => {
      const source = positionedPatterns.find(p => p.id === conn.sourceId);
      const target = positionedPatterns.find(p => p.id === conn.targetId);
      
      if (source && target) {
        const sourceProj = project(source.x, source.y, source.z);
        const targetProj = project(target.x, target.y, target.z);
        
        connectionsGroup.append('line')
          .attr('x1', sourceProj.x)
          .attr('y1', sourceProj.y)
          .attr('x2', targetProj.x)
          .attr('y2', targetProj.y)
          .attr('stroke', conn.type === 'excitatory' ? '#10b981' : '#ef4444')
          .attr('stroke-width', conn.strength * 2)
          .attr('stroke-opacity', 0.3)
          .attr('class', 'connection-line');
      }
    });

    // Draw patterns
    const patternsGroup = g.append('g').attr('class', 'patterns');
    
    const patternNodes = patternsGroup.selectAll('g')
      .data(positionedPatterns)
      .join('g')
      .attr('transform', d => {
        const proj = project(d.x, d.y, d.z);
        return `translate(${proj.x}, ${proj.y})`;
      })
      .attr('cursor', 'pointer')
      .on('click', (event, d) => setSelectedPattern(d.id))
      .on('mouseover', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d => 15 + d.activation * 25);
        
        // Highlight connections
        connectionsGroup.selectAll('.connection-line')
          .attr('stroke-opacity', line => {
            const lineData = d3.select(line).data()[0] as any;
            return lineData.sourceId === d.id || lineData.targetId === d.id ? 0.8 : 0.1;
          });
      })
      .on('mouseout', function(event, d) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', d => 10 + d.activation * 20);
        
        connectionsGroup.selectAll('.connection-line')
          .attr('stroke-opacity', 0.3);
      });

    // Pattern circles
    patternNodes.append('circle')
      .attr('r', d => 10 + d.activation * 20)
      .attr('fill', d => colorScale(d.type) as string)
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .attr('opacity', d => 0.7 + d.confidence * 0.3);

    // Activation pulse animation
    patternNodes.append('circle')
      .attr('r', d => 10 + d.activation * 20)
      .attr('fill', 'none')
      .attr('stroke', d => colorScale(d.type) as string)
      .attr('stroke-width', 2)
      .attr('opacity', 0)
      .each(function(d) {
        if (d.activation > 0.5) {
          d3.select(this)
            .transition()
            .duration(1000)
            .attr('r', d => 20 + d.activation * 40)
            .attr('opacity', 0)
            .transition()
            .duration(0)
            .attr('r', d => 10 + d.activation * 20)
            .on('end', function repeat() {
              d3.select(this)
                .transition()
                .duration(1000)
                .attr('r', d => 20 + d.activation * 40)
                .attr('opacity', 0)
                .transition()
                .duration(0)
                .attr('r', d => 10 + d.activation * 20)
                .on('end', repeat);
            });
        }
      });

    // Pattern labels
    patternNodes.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', -20)
      .attr('font-size', d => {
        const proj = project(d.x, d.y, d.z);
        return 10 + proj.scale * 4;
      })
      .attr('fill', 'white')
      .attr('opacity', 0.8);

    // Auto-rotation
    let animationFrame: number;
    const animate = () => {
      if (viewMode === '3d') {
        setRotation(prev => ({
          x: prev.x,
          y: (prev.y + 0.5) % 360
        }));
      }
      animationFrame = requestAnimationFrame(animate);
    };
    
    if (viewMode === '3d') {
      animate();
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [patterns, connections, rotation, viewMode]);

  const patternStats = React.useMemo(() => {
    const stats = {
      total: patterns.length,
      active: patterns.filter(p => p.activation > 0.1).length,
      highConfidence: patterns.filter(p => p.confidence > 0.8).length,
      avgActivation: patterns.reduce((sum, p) => sum + p.activation, 0) / patterns.length
    };
    return stats;
  }, [patterns]);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-semibold text-white">Cognitive Pattern Activation</h3>
        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {['3d', 'top', 'side'].map(mode => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  viewMode === mode 
                    ? 'bg-blue-500 text-white' 
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-gray-800 rounded p-3">
          <div className="text-xs text-gray-400">Total Patterns</div>
          <div className="text-lg font-bold text-white">{patternStats.total}</div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="text-xs text-gray-400">Active</div>
          <div className="text-lg font-bold text-green-400">{patternStats.active}</div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="text-xs text-gray-400">High Confidence</div>
          <div className="text-lg font-bold text-blue-400">{patternStats.highConfidence}</div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="text-xs text-gray-400">Avg Activation</div>
          <div className="text-lg font-bold text-orange-400">{(patternStats.avgActivation * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* 3D Visualization */}
      <div className="bg-gray-800 rounded p-4">
        <div ref={containerRef} className="w-full" />
      </div>

      {/* Selected Pattern Details */}
      {selectedPattern && (
        <div className="mt-4 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-2">Selected Pattern</h4>
          {(() => {
            const pattern = patterns.find(p => p.id === selectedPattern);
            if (!pattern) return null;
            
            return (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-400">Name</div>
                  <div className="text-white">{pattern.name}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Type</div>
                  <div className="text-white capitalize">{pattern.type.replace('_', ' ')}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Activation</div>
                  <div className="text-white">{(pattern.activation * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Confidence</div>
                  <div className="text-white">{(pattern.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* View Controls */}
      {viewMode === '3d' && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          Auto-rotating • Click patterns for details • Hover to highlight connections
        </div>
      )}
    </div>
  );
}