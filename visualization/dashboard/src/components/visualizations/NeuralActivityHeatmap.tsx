import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useAppSelector } from '../../stores';
import { NeuralActivity, NeuralData } from '../../types';

interface NeuralActivityHeatmapProps {
  neuralData: NeuralData;
  width?: number;
  height?: number;
  gridSize?: number;
  updateInterval?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  interactive?: boolean;
  className?: string;
}

interface HeatmapCell {
  x: number;
  y: number;
  value: number;
  nodeId: string;
  layer: number;
  gridX: number;
  gridY: number;
}

interface LayerInfo {
  id: string;
  name: string;
  color: string;
  yOffset: number;
  height: number;
}

// Heatmap configuration
const HEATMAP_CONFIG = {
  margin: { top: 40, right: 80, bottom: 40, left: 60 },
  cell: {
    padding: 1,
    borderRadius: 2,
  },
  colorScales: {
    activity: {
      low: [0, 0.3],
      medium: [0.3, 0.7],
      high: [0.7, 1.0],
    },
  },
  animation: {
    duration: 300,
    stagger: 20,
  },
  layers: {
    maxVisible: 8,
    colors: [
      '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', 
      '#ffeaa7', '#dda0dd', '#98d8c8', '#fad2e1'
    ],
  },
};

const NeuralActivityHeatmap: React.FC<NeuralActivityHeatmapProps> = ({
  neuralData,
  width = 800,
  height = 600,
  gridSize = 20,
  updateInterval = 100,
  showGrid = true,
  showLegend = true,
  interactive = true,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const animationRef = useRef<number>();
  const theme = useAppSelector(state => state.dashboard.config.theme);
  const enableAnimations = useAppSelector(state => state.dashboard.config.enableAnimations);
  
  const [hoveredCell, setHoveredCell] = useState<HeatmapCell | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [activityHistory, setActivityHistory] = useState<NeuralActivity[][]>([]);

  // Calculate chart dimensions
  const chartDimensions = useMemo(() => {
    return {
      width: width - HEATMAP_CONFIG.margin.left - HEATMAP_CONFIG.margin.right,
      height: height - HEATMAP_CONFIG.margin.top - HEATMAP_CONFIG.margin.bottom,
    };
  }, [width, height]);

  // Process neural activity data into heatmap grid
  const heatmapData = useMemo((): HeatmapCell[] => {
    if (!neuralData.activity.length) return [];

    // Calculate grid dimensions
    const cellSize = Math.min(
      chartDimensions.width / gridSize,
      chartDimensions.height / gridSize
    );
    
    // Group activities by layer
    const layerGroups = d3.group(neuralData.activity, d => d.layer);
    const layers = Array.from(layerGroups.keys()).sort((a, b) => a - b);
    
    const cells: HeatmapCell[] = [];
    
    layers.forEach((layerIndex, layerPosition) => {
      const layerActivities = layerGroups.get(layerIndex) || [];
      
      // Calculate layer position
      const layerHeight = chartDimensions.height / Math.min(layers.length, HEATMAP_CONFIG.layers.maxVisible);
      const yOffset = layerPosition * layerHeight;
      
      // Create spatial grid for this layer
      const gridWidth = Math.floor(chartDimensions.width / cellSize);
      const gridHeight = Math.floor(layerHeight / cellSize);
      
      // Map neural activities to grid positions
      layerActivities.forEach(activity => {
        // Normalize position to grid coordinates
        const normalizedX = activity.position.x || Math.random();
        const normalizedY = activity.position.y || Math.random();
        
        const gridX = Math.floor(normalizedX * gridWidth);
        const gridY = Math.floor(normalizedY * gridHeight);
        
        const x = gridX * cellSize;
        const y = yOffset + gridY * cellSize;
        
        cells.push({
          x,
          y,
          value: activity.activation,
          nodeId: activity.nodeId,
          layer: layerIndex,
          gridX,
          gridY,
        });
      });
    });
    
    return cells;
  }, [neuralData.activity, chartDimensions, gridSize]);

  // Process layer information
  const layerInfo = useMemo((): LayerInfo[] => {
    const layers = neuralData.layers.slice(0, HEATMAP_CONFIG.layers.maxVisible);
    const layerHeight = chartDimensions.height / layers.length;
    
    return layers.map((layer, index) => ({
      id: layer.id,
      name: layer.name,
      color: HEATMAP_CONFIG.layers.colors[index % HEATMAP_CONFIG.layers.colors.length],
      yOffset: index * layerHeight,
      height: layerHeight,
    }));
  }, [neuralData.layers, chartDimensions.height]);

  // Color scale for activity intensity
  const activityColorScale = useMemo(() => {
    return d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateReds);
  }, []);

  // Layer color scale
  const layerColorScale = useMemo(() => {
    return d3.scaleOrdinal<string>()
      .domain(layerInfo.map(l => l.id))
      .range(HEATMAP_CONFIG.layers.colors);
  }, [layerInfo]);

  // Update activity history for trend analysis
  useEffect(() => {
    const interval = setInterval(() => {
      if (neuralData.activity.length > 0) {
        setActivityHistory(prev => {
          const newHistory = [neuralData.activity, ...prev.slice(0, 49)]; // Keep last 50 snapshots
          return newHistory;
        });
      }
    }, updateInterval);

    return () => clearInterval(interval);
  }, [neuralData.activity, updateInterval]);

  // Event handlers
  const handleCellHover = useCallback((cell: HeatmapCell | null, event?: MouseEvent) => {
    if (!interactive) return;
    
    setHoveredCell(cell);
    
    if (cell && event && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      const layer = neuralData.layers.find(l => l.id === cell.layer.toString());
      
      setTooltip({
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        content: `Node: ${cell.nodeId}
Layer: ${layer?.name || `Layer ${cell.layer}`}
Activation: ${(cell.value * 100).toFixed(1)}%
Position: (${cell.gridX}, ${cell.gridY})`,
      });
    } else {
      setTooltip(null);
    }
  }, [interactive, neuralData.layers]);

  const handleLayerSelect = useCallback((layerId: string) => {
    if (!interactive) return;
    setSelectedLayer(selectedLayer === layerId ? null : layerId);
  }, [interactive, selectedLayer]);

  // Main D3 visualization effect
  useEffect(() => {
    if (!svgRef.current || !heatmapData.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group with margins
    const g = svg
      .append('g')
      .attr('transform', `translate(${HEATMAP_CONFIG.margin.left},${HEATMAP_CONFIG.margin.top})`);

    // Background
    g.append('rect')
      .attr('width', chartDimensions.width)
      .attr('height', chartDimensions.height)
      .attr('fill', theme === 'dark' ? '#1e293b' : '#f8fafc')
      .attr('stroke', theme === 'dark' ? '#374151' : '#e5e7eb')
      .attr('rx', 4);

    // Draw layer backgrounds
    layerInfo.forEach(layer => {
      const layerGroup = g.append('g').attr('class', `layer-${layer.id}`);
      
      // Layer background
      layerGroup
        .append('rect')
        .attr('x', 0)
        .attr('y', layer.yOffset)
        .attr('width', chartDimensions.width)
        .attr('height', layer.height)
        .attr('fill', layer.color)
        .attr('fill-opacity', selectedLayer === layer.id ? 0.2 : 0.05)
        .attr('stroke', layer.color)
        .attr('stroke-width', selectedLayer === layer.id ? 2 : 1)
        .attr('stroke-opacity', 0.3)
        .style('cursor', interactive ? 'pointer' : 'default')
        .on('click', interactive ? () => handleLayerSelect(layer.id) : null);

      // Layer label
      layerGroup
        .append('text')
        .attr('x', 5)
        .attr('y', layer.yOffset + 15)
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .attr('font-weight', selectedLayer === layer.id ? 'bold' : 'normal')
        .text(layer.name);
    });

    // Grid lines
    if (showGrid) {
      const gridGroup = g.append('g').attr('class', 'grid');
      const cellSize = Math.min(
        chartDimensions.width / gridSize,
        chartDimensions.height / gridSize
      );

      // Vertical grid lines
      for (let i = 0; i <= gridSize; i++) {
        gridGroup
          .append('line')
          .attr('x1', i * cellSize)
          .attr('y1', 0)
          .attr('x2', i * cellSize)
          .attr('y2', chartDimensions.height)
          .attr('stroke', theme === 'dark' ? '#374151' : '#e5e7eb')
          .attr('stroke-width', 0.5)
          .attr('opacity', 0.3);
      }

      // Horizontal grid lines
      for (let i = 0; i <= gridSize; i++) {
        gridGroup
          .append('line')
          .attr('x1', 0)
          .attr('y1', i * (chartDimensions.height / gridSize))
          .attr('x2', chartDimensions.width)
          .attr('y2', i * (chartDimensions.height / gridSize))
          .attr('stroke', theme === 'dark' ? '#374151' : '#e5e7eb')
          .attr('stroke-width', 0.5)
          .attr('opacity', 0.3);
      }
    }

    // Draw heatmap cells
    const cellsGroup = g.append('g').attr('class', 'cells');
    
    const cells = cellsGroup
      .selectAll('.cell')
      .data(heatmapData.filter(cell => 
        !selectedLayer || cell.layer.toString() === selectedLayer
      ))
      .enter()
      .append('g')
      .attr('class', 'cell');

    // Cell rectangles
    const cellRects = cells
      .append('rect')
      .attr('x', d => d.x + HEATMAP_CONFIG.cell.padding)
      .attr('y', d => d.y + HEATMAP_CONFIG.cell.padding)
      .attr('width', 0)
      .attr('height', 0)
      .attr('fill', d => activityColorScale(d.value))
      .attr('stroke', d => layerColorScale(d.layer.toString()))
      .attr('stroke-width', 1)
      .attr('rx', HEATMAP_CONFIG.cell.borderRadius)
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('mouseover', interactive ? (event, d) => handleCellHover(d, event) : null)
      .on('mouseout', interactive ? () => handleCellHover(null) : null);

    // Calculate cell size
    const cellSize = Math.min(
      chartDimensions.width / gridSize,
      chartDimensions.height / gridSize
    ) - (HEATMAP_CONFIG.cell.padding * 2);

    // Animate cell appearance
    if (enableAnimations) {
      cellRects
        .transition()
        .duration(HEATMAP_CONFIG.animation.duration)
        .delay((_, i) => i * HEATMAP_CONFIG.animation.stagger)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .ease(d3.easeElasticOut);
    } else {
      cellRects
        .attr('width', cellSize)
        .attr('height', cellSize);
    }

    // Activity intensity indicators (pulse animation for high activity)
    cells
      .filter(d => d.value > 0.7)
      .append('circle')
      .attr('cx', d => d.x + cellSize / 2)
      .attr('cy', d => d.y + cellSize / 2)
      .attr('r', 2)
      .attr('fill', '#ffffff')
      .attr('fill-opacity', 0.8);

    if (enableAnimations) {
      cells
        .filter(d => d.value > 0.7)
        .select('circle')
        .append('animate')
        .attr('attributeName', 'r')
        .attr('values', '2;4;2')
        .attr('dur', '1s')
        .attr('repeatCount', 'indefinite');
    }

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', HEATMAP_CONFIG.margin.top / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Neural Activity Heatmap');

    // Overall activity indicator
    const overallActivity = neuralData.overallActivity || 0;
    const activityIndicator = svg.append('g')
      .attr('transform', `translate(${width - 30}, ${HEATMAP_CONFIG.margin.top})`);

    activityIndicator.append('circle')
      .attr('cx', 0)
      .attr('cy', 0)
      .attr('r', 15)
      .attr('fill', activityColorScale(overallActivity))
      .attr('stroke', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('stroke-width', 2);

    activityIndicator.append('text')
      .attr('x', 0)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '10px')
      .text(`${(overallActivity * 100).toFixed(0)}%`);

  }, [heatmapData, layerInfo, chartDimensions, theme, enableAnimations, selectedLayer, showGrid, activityColorScale, layerColorScale, gridSize, interactive, neuralData.overallActivity, width]);

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
          minWidth: '120px',
        }}
      >
        <div style={{ marginBottom: '8px', fontWeight: 'bold' }}>Activity Scale</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {[
            { level: 'High', color: '#dc2626', threshold: '70-100%' },
            { level: 'Medium', color: '#f59e0b', threshold: '30-70%' },
            { level: 'Low', color: '#fee2e2', threshold: '0-30%' },
          ].map(item => (
            <div key={item.level} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div
                style={{
                  width: '12px',
                  height: '12px',
                  backgroundColor: item.color,
                  borderRadius: '2px',
                }}
              />
              <span>{item.level} ({item.threshold})</span>
            </div>
          ))}
        </div>
        
        {layerInfo.length > 0 && (
          <>
            <div style={{ marginTop: '12px', marginBottom: '6px', fontWeight: 'bold' }}>
              Layers ({layerInfo.length})
            </div>
            <div style={{ fontSize: '11px', opacity: 0.8 }}>
              Click layer to highlight
            </div>
          </>
        )}
      </div>
    );
  };

  // Performance stats
  const renderStats = () => (
    <div
      style={{
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        background: theme === 'dark' ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)',
        border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        borderRadius: '6px',
        padding: '8px',
        fontSize: '11px',
        color: theme === 'dark' ? '#ffffff' : '#374151',
      }}
    >
      <div>Active Nodes: {neuralData.activity.length}</div>
      <div>Layers: {neuralData.layers.length}</div>
      <div>Connections: {neuralData.connections.length}</div>
      <div>Update Rate: {Math.round(1000 / updateInterval)}Hz</div>
    </div>
  );

  if (!neuralData.activity.length) {
    return (
      <div 
        className={`flex items-center justify-center ${className}`}
        style={{
          width,
          height,
          background: theme === 'dark' ? '#1e293b' : '#f8fafc',
          borderRadius: '8px',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          color: theme === 'dark' ? '#9ca3af' : '#6b7280',
        }}
      >
        No neural activity data available
      </div>
    );
  }

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
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
      {renderStats()}
      
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

export default NeuralActivityHeatmap;