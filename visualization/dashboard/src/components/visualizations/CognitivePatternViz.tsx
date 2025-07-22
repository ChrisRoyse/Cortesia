import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useAppSelector } from '../../stores';
import { CognitivePattern, CognitiveData } from '../../types';

interface CognitivePatternVizProps {
  patterns: CognitivePattern[];
  inhibitoryLevels: CognitiveData;
  timeWindow?: number;
  width?: number;
  height?: number;
  interactive?: boolean;
  showLegend?: boolean;
  className?: string;
}

interface ProcessedPattern extends CognitivePattern {
  x: number;
  y: number;
  radius: number;
  opacity: number;
  connections: string[];
}

interface InhibitoryLayer {
  level: number;
  strength: number;
  color: string;
  patterns: ProcessedPattern[];
}

// Pattern visualization configuration
const VIZ_CONFIG = {
  margin: { top: 40, right: 60, bottom: 40, left: 60 },
  patternRadius: {
    min: 5,
    max: 25,
  },
  inhibitoryLayers: {
    count: 5,
    spacing: 0.15,
  },
  animation: {
    duration: 500,
    delay: 50,
  },
  colors: {
    hierarchical: '#61dafb',
    lateral: '#f093fb',
    feedback: '#a8e6cf',
    inhibitory: '#ff6b6b',
    activation: '#4ecdc4',
  },
};

const CognitivePatternViz: React.FC<CognitivePatternVizProps> = ({
  patterns,
  inhibitoryLevels,
  timeWindow = 5000,
  width = 800,
  height = 600,
  interactive = true,
  showLegend = true,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const theme = useAppSelector(state => state.dashboard.config.theme);
  const enableAnimations = useAppSelector(state => state.dashboard.config.enableAnimations);
  
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [hoveredPattern, setHoveredPattern] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  // Calculate chart dimensions
  const chartDimensions = useMemo(() => {
    return {
      width: width - VIZ_CONFIG.margin.left - VIZ_CONFIG.margin.right,
      height: height - VIZ_CONFIG.margin.top - VIZ_CONFIG.margin.bottom,
    };
  }, [width, height]);

  // Process patterns for visualization
  const processedPatterns = useMemo((): ProcessedPattern[] => {
    const currentTime = Date.now();
    
    return patterns
      .filter(pattern => currentTime - pattern.timestamp <= timeWindow)
      .map((pattern, index) => {
        // Calculate position based on pattern type and hierarchy
        const layerIndex = pattern.type === 'hierarchical' ? 0 : pattern.type === 'lateral' ? 1 : 2;
        const layerY = chartDimensions.height * (0.2 + layerIndex * 0.3);
        
        // X position based on timestamp (timeline)
        const timeRange = timeWindow;
        const timePosition = (currentTime - pattern.timestamp) / timeRange;
        const x = chartDimensions.width * (1 - timePosition);
        
        // Add some vertical jitter based on pattern strength
        const yJitter = (pattern.strength - 0.5) * 40;
        const y = layerY + yJitter;
        
        // Radius based on pattern strength and active nodes
        const baseRadius = VIZ_CONFIG.patternRadius.min + 
          (pattern.strength * (VIZ_CONFIG.patternRadius.max - VIZ_CONFIG.patternRadius.min));
        const nodeCountMultiplier = Math.sqrt(pattern.activeNodes.length) / 3;
        const radius = Math.max(baseRadius * nodeCountMultiplier, VIZ_CONFIG.patternRadius.min);
        
        // Opacity based on age and strength
        const ageMultiplier = 1 - (timePosition * 0.7);
        const opacity = Math.max(pattern.strength * ageMultiplier, 0.1);
        
        return {
          ...pattern,
          x,
          y,
          radius,
          opacity,
          connections: pattern.activeNodes,
        };
      });
  }, [patterns, timeWindow, chartDimensions]);

  // Create inhibitory layers data
  const inhibitoryLayers = useMemo((): InhibitoryLayer[] => {
    const layers: InhibitoryLayer[] = [];
    
    for (let i = 0; i < VIZ_CONFIG.inhibitoryLayers.count; i++) {
      const level = i / (VIZ_CONFIG.inhibitoryLayers.count - 1);
      const strength = Math.max(0, inhibitoryLevels.inhibitoryLevel - level);
      
      // Color intensity based on strength
      const baseColor = theme === 'dark' ? '#ff6b6b' : '#dc2626';
      const color = d3.color(baseColor)!.copy({ opacity: strength * 0.3 }).toString();
      
      // Filter patterns for this inhibitory level
      const layerPatterns = processedPatterns.filter(pattern => {
        const patternLevel = pattern.strength;
        return Math.abs(patternLevel - level) < VIZ_CONFIG.inhibitoryLayers.spacing;
      });
      
      layers.push({
        level,
        strength,
        color,
        patterns: layerPatterns,
      });
    }
    
    return layers;
  }, [inhibitoryLevels, processedPatterns, theme]);

  // Color scale for pattern types
  const colorScale = useMemo(() => {
    return d3.scaleOrdinal<string>()
      .domain(['hierarchical', 'lateral', 'feedback'])
      .range([
        VIZ_CONFIG.colors.hierarchical,
        VIZ_CONFIG.colors.lateral,
        VIZ_CONFIG.colors.feedback,
      ]);
  }, []);

  // Event handlers
  const handlePatternClick = useCallback((pattern: ProcessedPattern) => {
    if (!interactive) return;
    setSelectedPattern(selectedPattern === pattern.id ? null : pattern.id);
  }, [interactive, selectedPattern]);

  const handlePatternHover = useCallback((pattern: ProcessedPattern | null, event?: MouseEvent) => {
    if (!interactive) return;
    
    setHoveredPattern(pattern?.id || null);
    
    if (pattern && event && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setTooltip({
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        content: `${pattern.type.charAt(0).toUpperCase() + pattern.type.slice(1)} Pattern
Strength: ${(pattern.strength * 100).toFixed(1)}%
Active Nodes: ${pattern.activeNodes.length}
Age: ${((Date.now() - pattern.timestamp) / 1000).toFixed(1)}s`,
      });
    } else {
      setTooltip(null);
    }
  }, [interactive]);

  // D3 visualization effect
  useEffect(() => {
    if (!svgRef.current || !processedPatterns.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group with margins
    const g = svg
      .append('g')
      .attr('transform', `translate(${VIZ_CONFIG.margin.left},${VIZ_CONFIG.margin.top})`);

    // Background gradient
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'backgroundGradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', chartDimensions.width).attr('y2', chartDimensions.height);
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', theme === 'dark' ? '#1e293b' : '#f8fafc')
      .attr('stop-opacity', 0.8);
    
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', theme === 'dark' ? '#0f172a' : '#e2e8f0')
      .attr('stop-opacity', 0.4);

    // Background
    g.append('rect')
      .attr('width', chartDimensions.width)
      .attr('height', chartDimensions.height)
      .attr('fill', 'url(#backgroundGradient)')
      .attr('rx', 8);

    // Draw inhibitory layers
    inhibitoryLayers.forEach((layer, layerIndex) => {
      if (layer.strength > 0.1) {
        g.append('rect')
          .attr('x', 0)
          .attr('y', layerIndex * chartDimensions.height / VIZ_CONFIG.inhibitoryLayers.count)
          .attr('width', chartDimensions.width)
          .attr('height', chartDimensions.height / VIZ_CONFIG.inhibitoryLayers.count)
          .attr('fill', layer.color)
          .attr('stroke', theme === 'dark' ? '#374151' : '#d1d5db')
          .attr('stroke-width', 0.5)
          .attr('stroke-dasharray', '2,2');
      }
    });

    // Draw connections between patterns
    const connections = g.append('g').attr('class', 'connections');
    
    processedPatterns.forEach((pattern, i) => {
      processedPatterns.slice(i + 1).forEach(otherPattern => {
        const sharedNodes = pattern.activeNodes.filter(node => 
          otherPattern.activeNodes.includes(node)
        );
        
        if (sharedNodes.length > 0) {
          const line = connections.append('line')
            .attr('x1', pattern.x)
            .attr('y1', pattern.y)
            .attr('x2', otherPattern.x)
            .attr('y2', otherPattern.y)
            .attr('stroke', theme === 'dark' ? '#6b7280' : '#9ca3af')
            .attr('stroke-width', Math.sqrt(sharedNodes.length))
            .attr('stroke-opacity', 0.3);
          
          if (enableAnimations) {
            line.attr('stroke-dasharray', '3,3')
              .append('animateTransform')
              .attr('attributeName', 'stroke-dashoffset')
              .attr('values', '0;6')
              .attr('dur', '2s')
              .attr('repeatCount', 'indefinite');
          }
        }
      });
    });

    // Draw patterns
    const patternsGroup = g.append('g').attr('class', 'patterns');
    
    const patternCircles = patternsGroup
      .selectAll('.pattern')
      .data(processedPatterns)
      .enter()
      .append('g')
      .attr('class', 'pattern')
      .attr('transform', d => `translate(${d.x},${d.y})`);

    // Pattern circles
    patternCircles
      .append('circle')
      .attr('r', 0)
      .attr('fill', d => colorScale(d.type))
      .attr('fill-opacity', d => d.opacity)
      .attr('stroke', d => selectedPattern === d.id ? '#ffffff' : colorScale(d.type))
      .attr('stroke-width', d => selectedPattern === d.id ? 3 : 1)
      .attr('stroke-opacity', 0.8)
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('click', interactive ? (event, d) => handlePatternClick(d) : null)
      .on('mouseover', interactive ? (event, d) => handlePatternHover(d, event) : null)
      .on('mouseout', interactive ? () => handlePatternHover(null) : null);

    // Animate pattern appearance
    if (enableAnimations) {
      patternCircles.select('circle')
        .transition()
        .duration(VIZ_CONFIG.animation.duration)
        .delay((_, i) => i * VIZ_CONFIG.animation.delay)
        .attr('r', d => d.radius)
        .ease(d3.easeElasticOut);
    } else {
      patternCircles.select('circle').attr('r', d => d.radius);
    }

    // Pattern strength indicators (inner circles)
    patternCircles
      .append('circle')
      .attr('r', d => d.radius * d.strength * 0.6)
      .attr('fill', VIZ_CONFIG.colors.activation)
      .attr('fill-opacity', 0.4);

    // Active node count text
    patternCircles
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', d => Math.max(d.radius * 0.4, 10))
      .attr('font-weight', 'bold')
      .text(d => d.activeNodes.length);

    // Timeline axis
    const timeScale = d3.scaleLinear()
      .domain([0, timeWindow])
      .range([chartDimensions.width, 0]);

    const timeAxis = d3.axisBottom(timeScale)
      .tickFormat(d => `${(d as number / 1000).toFixed(1)}s ago`)
      .ticks(5);

    g.append('g')
      .attr('transform', `translate(0,${chartDimensions.height + 20})`)
      .call(timeAxis)
      .attr('color', theme === 'dark' ? '#9ca3af' : '#6b7280');

    // Y-axis for pattern types
    const typeScale = d3.scaleBand()
      .domain(['Hierarchical', 'Lateral', 'Feedback'])
      .range([0, chartDimensions.height])
      .padding(0.1);

    const typeAxis = d3.axisLeft(typeScale);

    g.append('g')
      .attr('transform', 'translate(-20,0)')
      .call(typeAxis)
      .attr('color', theme === 'dark' ? '#9ca3af' : '#6b7280');

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', VIZ_CONFIG.margin.top / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Cognitive Pattern Activity');

    // Inhibitory level indicator
    const inhibitoryIndicator = svg.append('g')
      .attr('transform', `translate(${width - VIZ_CONFIG.margin.right / 2}, ${VIZ_CONFIG.margin.top})`);

    inhibitoryIndicator.append('rect')
      .attr('x', -10)
      .attr('y', 0)
      .attr('width', 20)
      .attr('height', chartDimensions.height)
      .attr('fill', 'url(#inhibitoryGradient)')
      .attr('stroke', theme === 'dark' ? '#374151' : '#d1d5db');

    // Inhibitory gradient
    const inhibitoryGradient = defs.append('linearGradient')
      .attr('id', 'inhibitoryGradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', chartDimensions.height);
    
    inhibitoryGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', VIZ_CONFIG.colors.inhibitory)
      .attr('stop-opacity', inhibitoryLevels.inhibitoryLevel);
    
    inhibitoryGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', VIZ_CONFIG.colors.inhibitory)
      .attr('stop-opacity', 0);

  }, [processedPatterns, inhibitoryLayers, chartDimensions, theme, enableAnimations, selectedPattern, interactive, colorScale, timeWindow, inhibitoryLevels, width]);

  // Legend component
  const renderLegend = () => {
    if (!showLegend) return null;

    return (
      <div 
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: theme === 'dark' ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          borderRadius: '6px',
          padding: '8px',
          fontSize: '12px',
          color: theme === 'dark' ? '#ffffff' : '#374151',
        }}
      >
        <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>Pattern Types</div>
        {['hierarchical', 'lateral', 'feedback'].map(type => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <div
              style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                backgroundColor: colorScale(type),
                marginRight: '6px',
              }}
            />
            <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
          </div>
        ))}
        <div style={{ marginTop: '8px', fontSize: '11px', opacity: 0.7 }}>
          Circle size = pattern strength × active nodes
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className={`relative ${className}`} style={{ width, height }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          background: theme === 'dark' ? '#0f172a' : '#f8fafc',
          borderRadius: '8px',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        }}
      />
      
      {renderLegend()}
      
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x + 10,
            top: tooltip.y - 10,
            background: theme === 'dark' ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.95)',
            color: theme === 'dark' ? '#ffffff' : '#374151',
            padding: '8px 10px',
            borderRadius: '4px',
            fontSize: '12px',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
            whiteSpace: 'pre-line',
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
};

export default CognitivePatternViz;