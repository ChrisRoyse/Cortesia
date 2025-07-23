import React, { useRef, useEffect, memo } from 'react';
import * as d3 from 'd3';
import { Box } from '@mui/material';

interface Pattern {
  id: string;
  type: string;
  strength: number;
  position: { x: number; y: number; z: number };
}

interface Props {
  patterns: Pattern[];
}

const CognitivePatternVisualizerComponent: React.FC<Props> = ({ patterns }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !patterns || patterns.length === 0) return;
    
    // Validate and sanitize patterns data
    const validPatterns = patterns.filter(pattern => 
      pattern && 
      pattern.position && 
      typeof pattern.position.x === 'number' && 
      typeof pattern.position.y === 'number' &&
      typeof pattern.strength === 'number' &&
      pattern.type
    );
    
    if (validPatterns.length === 0) {
      console.warn('No valid patterns data available for visualization');
      return;
    }

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([50, width - 50]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([50, height - 50]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    const radiusScale = d3.scaleLinear()
      .domain([0, 100])
      .range([10, 40]);

    // Create gradient definitions
    const defs = svg.append('defs');
    validPatterns.forEach((pattern, i) => {
      const gradient = defs.append('radialGradient')
        .attr('id', `pattern-gradient-${i}`)
        .attr('cx', '50%')
        .attr('cy', '50%')
        .attr('r', '50%');

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', colorScale(pattern.type))
        .attr('stop-opacity', 0.8);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', colorScale(pattern.type))
        .attr('stop-opacity', 0.2);
    });

    // Draw patterns
    const nodes = svg.selectAll('.pattern-node')
      .data(validPatterns)
      .enter()
      .append('g')
      .attr('class', 'pattern-node')
      .attr('transform', d => `translate(${xScale(d.position.x)}, ${yScale(d.position.y)})`);

    // Add circles
    nodes.append('circle')
      .attr('r', d => radiusScale(d.strength))
      .attr('fill', (d, i) => `url(#pattern-gradient-${i})`)
      .attr('stroke', d => colorScale(d.type))
      .attr('stroke-width', 2)
      .style('filter', 'url(#glow)');

    // Add labels
    nodes.append('text')
      .text(d => d.type)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .style('fill', 'white')
      .style('font-size', '12px')
      .style('font-weight', 'bold');

    // Add glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
      .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');

    // Animate patterns
    nodes.selectAll('circle')
      .style('opacity', 0)
      .transition()
      .duration(1000)
      .delay((d, i) => i * 100)
      .style('opacity', 1);

  }, [patterns]);

  // Show loading state if no valid patterns
  if (!patterns || patterns.length === 0) {
    return (
      <Box sx={{ 
        width: '100%', 
        height: 'calc(100% - 40px)', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: 'text.secondary'
      }}>
        No cognitive patterns data available
      </Box>
    );
  }

  return (
    <Box 
      sx={{ width: '100%', height: 'calc(100% - 40px)' }}
      data-testid="cognitive-pattern-visualizer"
      data-pattern-count={patterns?.length || 0}
    >
      <svg ref={svgRef} width="100%" height="100%" />
    </Box>
  );
};

// Memoize the component to prevent unnecessary re-renders
export const CognitivePatternVisualizer = memo(CognitivePatternVisualizerComponent, (prevProps, nextProps) => {
  // Custom comparison to avoid re-rendering if patterns are the same
  if (prevProps.patterns.length !== nextProps.patterns.length) return false;
  
  return prevProps.patterns.every((pattern, index) => {
    const nextPattern = nextProps.patterns[index];
    return pattern.id === nextPattern.id && 
           pattern.strength === nextPattern.strength &&
           pattern.position.x === nextPattern.position.x &&
           pattern.position.y === nextPattern.position.y;
  });
});