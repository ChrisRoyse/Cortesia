import React, { useMemo, useCallback } from 'react';
import { useSpring, animated } from '@react-spring/web';
import {
  LayerVisualizationProps,
  LayerDefinition,
  ArchitectureNode,
  ThemeConfiguration
} from '../types';

const LayerVisualization: React.FC<LayerVisualizationProps> = ({
  layers,
  nodes,
  theme,
  showLabels = true,
  interactive = false,
  onLayerClick
}) => {
  // Sort layers by phase and order
  const sortedLayers = useMemo(() => {
    return [...layers].sort((a, b) => {
      if (a.phase !== b.phase) {
        return a.phase - b.phase;
      }
      return a.order - b.order;
    });
  }, [layers]);

  // Calculate layer statistics
  const layerStats = useMemo(() => {
    return layers.map(layer => {
      const layerNodes = nodes.filter(node => layer.nodes.includes(node.id));
      const healthyNodes = layerNodes.filter(node => node.status === 'healthy').length;
      const activeNodes = layerNodes.filter(node => node.status !== 'offline').length;
      
      return {
        ...layer,
        nodeCount: layerNodes.length,
        healthyNodes,
        activeNodes,
        healthPercentage: layerNodes.length > 0 ? (healthyNodes / layerNodes.length) * 100 : 0,
        averageImportance: layerNodes.reduce((sum, node) => sum + (node.importance || 0), 0) / layerNodes.length || 0
      };
    });
  }, [layers, nodes]);

  return (
    <g className="layer-visualization">
      {/* Background layers */}
      {sortedLayers.map((layer, index) => (
        <LayerBackground
          key={layer.id}
          layer={layer}
          index={index}
          totalLayers={sortedLayers.length}
          theme={theme}
          interactive={interactive}
          onClick={onLayerClick}
        />
      ))}

      {/* Layer labels and information */}
      {showLabels && (
        <g className="layer-labels">
          {layerStats.map((layer, index) => (
            <LayerLabel
              key={layer.id}
              layer={layer}
              index={index}
              theme={theme}
              interactive={interactive}
              onClick={onLayerClick}
            />
          ))}
        </g>
      )}

      {/* Phase separators */}
      <PhaseSeparators layers={sortedLayers} theme={theme} />

      {/* Layer connections/flow indicators */}
      <LayerConnections layers={sortedLayers} theme={theme} />
    </g>
  );
};

// Individual layer background component
const LayerBackground: React.FC<{
  layer: LayerDefinition;
  index: number;
  totalLayers: number;
  theme: ThemeConfiguration;
  interactive: boolean;
  onClick?: (layer: LayerDefinition) => void;
}> = ({ layer, index, totalLayers, theme, interactive, onClick }) => {
  // Animation for layer appearance
  const layerSpring = useSpring({
    opacity: 1,
    scale: 1,
    from: { opacity: 0, scale: 0.95 },
    delay: index * 100,
    config: { tension: 280, friction: 60 }
  });

  const handleClick = useCallback(() => {
    if (interactive && onClick) {
      onClick(layer);
    }
  }, [interactive, onClick, layer]);

  // Calculate layer opacity based on depth
  const baseOpacity = 0.1 + (index / totalLayers) * 0.2;
  const layerColor = layer.color || theme.colors.primary;

  return (
    <animated.g
      className={`layer-background ${interactive ? 'cursor-pointer' : ''}`}
      style={layerSpring}
      onClick={handleClick}
    >
      {/* Main layer rectangle */}
      <rect
        x={layer.position.x}
        y={layer.position.y}
        width={layer.size.width}
        height={layer.size.height}
        fill={layerColor}
        fillOpacity={baseOpacity}
        stroke={layerColor}
        strokeWidth={1}
        strokeOpacity={0.3}
        rx={8}
        ry={8}
        className={`transition-all duration-200 ${
          interactive ? 'hover:fill-opacity-20 hover:stroke-opacity-50' : ''
        }`}
      />

      {/* Layer depth indicator (3D effect) */}
      <rect
        x={layer.position.x + 4}
        y={layer.position.y + 4}
        width={layer.size.width}
        height={layer.size.height}
        fill="none"
        stroke={layerColor}
        strokeWidth={0.5}
        strokeOpacity={0.2}
        rx={8}
        ry={8}
        className="pointer-events-none"
      />

      {/* Phase indicator strip */}
      <rect
        x={layer.position.x}
        y={layer.position.y}
        width={8}
        height={layer.size.height}
        fill={getPhaseColor(layer.phase, theme)}
        fillOpacity={0.7}
        rx={4}
        className="pointer-events-none"
      />
    </animated.g>
  );
};

// Layer label and statistics component
const LayerLabel: React.FC<{
  layer: LayerDefinition & {
    nodeCount: number;
    healthyNodes: number;
    activeNodes: number;
    healthPercentage: number;
    averageImportance: number;
  };
  index: number;
  theme: ThemeConfiguration;
  interactive: boolean;
  onClick?: (layer: LayerDefinition) => void;
}> = ({ layer, index, theme, interactive, onClick }) => {
  const labelX = layer.position.x + 20;
  const labelY = layer.position.y + 25;

  const handleClick = useCallback(() => {
    if (interactive && onClick) {
      onClick(layer);
    }
  }, [interactive, onClick, layer]);

  return (
    <g 
      className={`layer-label ${interactive ? 'cursor-pointer' : ''}`}
      onClick={handleClick}
    >
      {/* Layer name */}
      <text
        x={labelX}
        y={labelY}
        fontSize={16}
        fontWeight="600"
        fill={theme.colors.text}
        fontFamily={theme.fonts.primary}
        className={interactive ? 'hover:fill-blue-600' : ''}
      >
        {layer.name}
      </text>

      {/* Phase indicator */}
      <text
        x={labelX}
        y={labelY + 18}
        fontSize={12}
        fill={getPhaseColor(layer.phase, theme)}
        fontFamily={theme.fonts.primary}
        fontWeight="500"
      >
        Phase {layer.phase}
      </text>

      {/* Layer description */}
      <text
        x={labelX}
        y={labelY + 35}
        fontSize={11}
        fill={theme.colors.secondary}
        fontFamily={theme.fonts.primary}
      >
        {truncateText(layer.description, 40)}
      </text>

      {/* Statistics panel */}
      <LayerStatsPanel
        layer={layer}
        position={{ x: labelX, y: labelY + 50 }}
        theme={theme}
      />
    </g>
  );
};

// Layer statistics panel
const LayerStatsPanel: React.FC<{
  layer: {
    nodeCount: number;
    healthyNodes: number;
    activeNodes: number;
    healthPercentage: number;
    averageImportance: number;
  };
  position: { x: number; y: number };
  theme: ThemeConfiguration;
}> = ({ layer, position, theme }) => {
  const panelWidth = 180;
  const panelHeight = 60;

  return (
    <g className="layer-stats-panel">
      {/* Background */}
      <rect
        x={position.x}
        y={position.y}
        width={panelWidth}
        height={panelHeight}
        fill="rgba(255, 255, 255, 0.95)"
        stroke={theme.colors.secondary}
        strokeWidth={1}
        rx={4}
        className="drop-shadow-sm"
      />

      {/* Component count */}
      <g className="stat-item">
        <circle
          cx={position.x + 10}
          cy={position.y + 15}
          r={4}
          fill={theme.colors.primary}
        />
        <text
          x={position.x + 20}
          y={position.y + 18}
          fontSize={10}
          fill={theme.colors.text}
          fontFamily={theme.fonts.primary}
        >
          {layer.nodeCount} Components
        </text>
      </g>

      {/* Health indicator */}
      <g className="stat-item">
        <circle
          cx={position.x + 10}
          cy={position.y + 30}
          r={4}
          fill={getHealthColor(layer.healthPercentage)}
        />
        <text
          x={position.x + 20}
          y={position.y + 33}
          fontSize={10}
          fill={theme.colors.text}
          fontFamily={theme.fonts.primary}
        >
          {layer.healthPercentage.toFixed(0)}% Healthy
        </text>
      </g>

      {/* Importance indicator */}
      <g className="stat-item">
        <rect
          x={position.x + 10}
          y={position.y + 42}
          width={layer.averageImportance * 60}
          height={4}
          fill={theme.colors.highlight}
          rx={2}
        />
        <text
          x={position.x + 80}
          y={position.y + 48}
          fontSize={10}
          fill={theme.colors.text}
          fontFamily={theme.fonts.primary}
        >
          Importance: {(layer.averageImportance * 100).toFixed(0)}%
        </text>
      </g>
    </g>
  );
};

// Phase separators
const PhaseSeparators: React.FC<{
  layers: LayerDefinition[];
  theme: ThemeConfiguration;
}> = ({ layers, theme }) => {
  const phases = useMemo(() => {
    const phaseGroups = layers.reduce((groups, layer) => {
      if (!groups[layer.phase]) {
        groups[layer.phase] = [];
      }
      groups[layer.phase].push(layer);
      return groups;
    }, {} as Record<number, LayerDefinition[]>);

    return Object.entries(phaseGroups).map(([phase, phaseLayers]) => ({
      phase: parseInt(phase),
      layers: phaseLayers,
      minY: Math.min(...phaseLayers.map(l => l.position.y)),
      maxY: Math.max(...phaseLayers.map(l => l.position.y + l.size.height)),
      minX: Math.min(...phaseLayers.map(l => l.position.x)),
      maxX: Math.max(...phaseLayers.map(l => l.position.x + l.size.width))
    }));
  }, [layers]);

  return (
    <g className="phase-separators">
      {phases.map((phaseGroup, index) => {
        if (index === phases.length - 1) return null; // No separator after last phase

        const nextPhase = phases[index + 1];
        const separatorX = (phaseGroup.maxX + nextPhase.minX) / 2;

        return (
          <g key={`separator-${phaseGroup.phase}-${nextPhase.phase}`}>
            {/* Separator line */}
            <line
              x1={separatorX}
              y1={Math.min(phaseGroup.minY, nextPhase.minY) - 20}
              x2={separatorX}
              y2={Math.max(phaseGroup.maxY, nextPhase.maxY) + 20}
              stroke={theme.colors.secondary}
              strokeWidth={2}
              strokeDasharray="5,5"
              strokeOpacity={0.5}
            />

            {/* Phase transition label */}
            <rect
              x={separatorX - 40}
              y={Math.min(phaseGroup.minY, nextPhase.minY) - 35}
              width={80}
              height={20}
              fill={theme.colors.background}
              stroke={theme.colors.secondary}
              strokeWidth={1}
              rx={10}
            />
            <text
              x={separatorX}
              y={Math.min(phaseGroup.minY, nextPhase.minY) - 22}
              textAnchor="middle"
              fontSize={10}
              fill={theme.colors.text}
              fontFamily={theme.fonts.primary}
              fontWeight="500"
            >
              Phase {phaseGroup.phase} → {nextPhase.phase}
            </text>
          </g>
        );
      })}
    </g>
  );
};

// Layer connections showing data flow between layers
const LayerConnections: React.FC<{
  layers: LayerDefinition[];
  theme: ThemeConfiguration;
}> = ({ layers, theme }) => {
  const connections = useMemo(() => {
    const sortedLayers = [...layers].sort((a, b) => a.phase - b.phase || a.order - b.order);
    const layerConnections = [];

    for (let i = 0; i < sortedLayers.length - 1; i++) {
      const sourceLayer = sortedLayers[i];
      const targetLayer = sortedLayers[i + 1];

      // Only connect layers from adjacent phases
      if (targetLayer.phase === sourceLayer.phase + 1) {
        const sourcePoint = {
          x: sourceLayer.position.x + sourceLayer.size.width,
          y: sourceLayer.position.y + sourceLayer.size.height / 2
        };

        const targetPoint = {
          x: targetLayer.position.x,
          y: targetLayer.position.y + targetLayer.size.height / 2
        };

        layerConnections.push({
          id: `${sourceLayer.id}-${targetLayer.id}`,
          source: sourcePoint,
          target: targetPoint,
          sourceLayer,
          targetLayer
        });
      }
    }

    return layerConnections;
  }, [layers]);

  return (
    <g className="layer-connections">
      {connections.map(connection => (
        <LayerConnection
          key={connection.id}
          connection={connection}
          theme={theme}
        />
      ))}
    </g>
  );
};

// Individual layer connection
const LayerConnection: React.FC<{
  connection: {
    id: string;
    source: { x: number; y: number };
    target: { x: number; y: number };
    sourceLayer: LayerDefinition;
    targetLayer: LayerDefinition;
  };
  theme: ThemeConfiguration;
}> = ({ connection, theme }) => {
  const { source, target } = connection;
  
  // Create a curved path between layers
  const midX = (source.x + target.x) / 2;
  const path = `M${source.x},${source.y} Q${midX},${source.y} ${midX},${target.y} Q${midX},${target.y} ${target.x},${target.y}`;

  return (
    <g className="layer-connection">
      <path
        d={path}
        fill="none"
        stroke={theme.colors.primary}
        strokeWidth={1}
        strokeOpacity={0.3}
        strokeDasharray="3,3"
        className="pointer-events-none"
      />
      
      {/* Flow indicator */}
      <FlowIndicator
        path={path}
        color={theme.colors.primary}
        speed={2}
      />
    </g>
  );
};

// Animated flow indicator for layer connections
const FlowIndicator: React.FC<{
  path: string;
  color: string;
  speed: number;
}> = ({ path, color, speed }) => {
  const flowSpring = useSpring({
    from: { offset: 0 },
    to: async (next) => {
      while (true) {
        await next({ offset: 1 });
        await next({ offset: 0 });
      }
    },
    config: { duration: 3000 / speed },
    loop: true
  });

  return (
    <g className="flow-indicator">
      <animated.circle
        r={2}
        fill={color}
        opacity={0.6}
      >
        <animateMotion
          dur={`${3000 / speed}ms`}
          repeatCount="indefinite"
        >
          <mpath href={`data:image/svg+xml,<path d="${encodeURIComponent(path)}"/>`} />
        </animateMotion>
      </animated.circle>
    </g>
  );
};

// Utility functions
function getPhaseColor(phase: number, theme: ThemeConfiguration): string {
  const phaseColors = {
    1: '#ef4444', // Red
    2: '#f97316', // Orange
    3: '#eab308', // Yellow
    4: '#22c55e', // Green
    5: '#3b82f6', // Blue
    6: '#8b5cf6', // Purple
    7: '#ec4899', // Pink
  };
  
  return phaseColors[phase] || theme.colors.primary;
}

function getHealthColor(healthPercentage: number): string {
  if (healthPercentage >= 90) return '#10b981'; // Green
  if (healthPercentage >= 70) return '#f59e0b'; // Yellow
  if (healthPercentage >= 50) return '#f97316'; // Orange
  return '#ef4444'; // Red
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

export default LayerVisualization;