import React, { useMemo, useCallback, useState } from 'react';
import { useSpring, animated } from '@react-spring/web';
import {
  ArchitectureNodeProps,
  InteractionType,
  ComponentStatus,
  NodeType,
  ThemeConfiguration
} from '../types';

const ArchitectureNode: React.FC<ArchitectureNodeProps> = ({
  node,
  isSelected,
  isFocused,
  isHighlighted,
  theme,
  showMetrics,
  scale,
  onInteraction
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Calculate node visual properties
  const nodeVisuals = useMemo(() => {
    const baseSize = Math.max(20, Math.min(60, node.size || 30));
    const adjustedSize = baseSize * Math.sqrt(node.importance || 1);
    
    // Color based on type and status
    const colors = getNodeColors(node.type, node.status, theme);
    
    return {
      radius: adjustedSize,
      colors,
      strokeWidth: isSelected ? 4 : isFocused ? 3 : 2,
      opacity: node.status === 'offline' ? 0.5 : 1.0
    };
  }, [node.type, node.status, node.size, node.importance, theme, isSelected, isFocused]);

  // Animation springs
  const nodeSpring = useSpring({
    scale: isSelected ? 1.15 : isHovered ? 1.05 : 1.0,
    opacity: nodeVisuals.opacity,
    strokeWidth: nodeVisuals.strokeWidth,
    glowIntensity: isHighlighted ? 1 : 0,
    config: { tension: 300, friction: 30 }
  });

  const pulseSpring = useSpring({
    scale: node.status === 'processing' ? [1, 1.1, 1] : [1],
    loop: node.status === 'processing',
    config: { duration: 1500 }
  });

  // Event handlers
  const handleMouseEnter = useCallback((e: React.MouseEvent) => {
    setIsHovered(true);
    onInteraction('hover', e.nativeEvent);
  }, [onInteraction]);

  const handleMouseLeave = useCallback((e: React.MouseEvent) => {
    setIsHovered(false);
    onInteraction('hover-end', e.nativeEvent);
  }, [onInteraction]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onInteraction('click', e.nativeEvent);
  }, [onInteraction]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onInteraction('double-click', e.nativeEvent);
  }, [onInteraction]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    onInteraction('context-menu', e.nativeEvent);
  }, [onInteraction]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onInteraction('click', e.nativeEvent);
    }
  }, [onInteraction]);

  // Calculate label font size based on node size and zoom
  const labelFontSize = Math.max(10, Math.min(16, nodeVisuals.radius / 2.5 * scale));
  
  // Determine if label should be visible
  const showLabel = scale > 0.5 && nodeVisuals.radius > 15;

  return (
    <g
      className="architecture-node cursor-pointer select-none"
      transform={`translate(${node.position.x}, ${node.position.y})`}
      tabIndex={0}
      role="button"
      aria-label={`${node.label} - ${node.status} - ${node.type}`}
      aria-selected={isSelected}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
      onKeyDown={handleKeyDown}
    >
      {/* Glow effect for highlighted nodes */}
      {isHighlighted && (
        <animated.circle
          r={nodeVisuals.radius + 8}
          fill="none"
          stroke={theme.colors.highlight}
          strokeWidth={2}
          opacity={nodeSpring.glowIntensity.to(i => i * 0.6)}
          filter="url(#glow)"
          className="pointer-events-none"
        />
      )}

      {/* Selection ring */}
      {isSelected && (
        <animated.circle
          r={nodeVisuals.radius + 6}
          fill="none"
          stroke={theme.colors.primary}
          strokeWidth={3}
          strokeDasharray="4,2"
          opacity={0.8}
          className="pointer-events-none animate-spin"
          style={{ animationDuration: '3s' }}
        />
      )}

      {/* Focus ring for accessibility */}
      {isFocused && (
        <circle
          r={nodeVisuals.radius + 4}
          fill="none"
          stroke={theme.colors.primary}
          strokeWidth={2}
          strokeDasharray="2,2"
          opacity={0.7}
          className="pointer-events-none"
        />
      )}

      {/* Main node circle with animation */}
      <animated.circle
        r={nodeVisuals.radius}
        fill={nodeVisuals.colors.primary}
        stroke={nodeVisuals.colors.border}
        strokeWidth={nodeSpring.strokeWidth}
        opacity={nodeSpring.opacity}
        transform={nodeSpring.scale.to(s => `scale(${s})`)}
        className="transition-colors duration-200"
        style={{
          filter: node.status === 'critical' ? 'drop-shadow(0 0 8px rgba(220, 38, 38, 0.6))' : undefined
        }}
      />

      {/* Processing animation overlay */}
      {node.status === 'processing' && (
        <animated.circle
          r={nodeVisuals.radius}
          fill="none"
          stroke={theme.colors.activity}
          strokeWidth={2}
          opacity={0.6}
          transform={pulseSpring.scale.to((scales: number[]) => 
            scales.length > 1 ? `scale(${scales[1]})` : `scale(${scales[0]})`
          )}
          strokeDasharray="8,4"
          className="pointer-events-none animate-spin"
          style={{ animationDuration: '2s' }}
        />
      )}

      {/* Node type indicator */}
      <NodeTypeIndicator
        nodeType={node.type}
        radius={nodeVisuals.radius}
        theme={theme}
        scale={scale}
      />

      {/* Status indicator */}
      <StatusIndicator
        status={node.status}
        position={{
          x: nodeVisuals.radius * 0.7,
          y: -nodeVisuals.radius * 0.7
        }}
        theme={theme}
        scale={scale}
      />

      {/* Node label */}
      {showLabel && (
        <text
          x={0}
          y={nodeVisuals.radius + 20}
          textAnchor="middle"
          fontSize={labelFontSize}
          fill={theme.colors.text}
          fontFamily={theme.fonts.primary}
          fontWeight="500"
          className="pointer-events-none select-none"
        >
          <tspan x={0} dy={0}>
            {truncateLabel(node.label, nodeVisuals.radius)}
          </tspan>
          {node.description && scale > 0.8 && (
            <tspan
              x={0}
              dy={labelFontSize + 2}
              fontSize={labelFontSize * 0.8}
              fill={theme.colors.secondary}
              fontWeight="400"
            >
              {truncateLabel(node.description, nodeVisuals.radius, 20)}
            </tspan>
          )}
        </text>
      )}

      {/* Metrics overlay */}
      {showMetrics && node.metrics && scale > 0.6 && (
        <MetricsOverlay
          metrics={node.metrics}
          position={{ x: 0, y: nodeVisuals.radius + (showLabel ? 40 : 25) }}
          theme={theme}
          compact={nodeVisuals.radius < 35}
          scale={scale}
        />
      )}

      {/* Connection points */}
      {node.connections.map((connection, index) => (
        <ConnectionPoint
          key={connection.id}
          connection={connection}
          nodeRadius={nodeVisuals.radius}
          angle={(Math.PI * 2 * index) / node.connections.length}
          theme={theme}
          active={connection.active}
          scale={scale}
        />
      ))}

      {/* Activity indicator for real-time updates */}
      {node.metadata?.recentActivity && (
        <ActivityRipple
          center={{ x: 0, y: 0 }}
          maxRadius={nodeVisuals.radius * 2}
          color={theme.colors.activity}
          intensity={node.metadata.recentActivity}
        />
      )}
    </g>
  );
};

// Helper component for node type indicators
const NodeTypeIndicator: React.FC<{
  nodeType: NodeType;
  radius: number;
  theme: ThemeConfiguration;
  scale: number;
}> = ({ nodeType, radius, theme, scale }) => {
  if (scale < 0.7) return null;

  const iconSize = Math.max(8, radius / 3);
  const icon = getNodeTypeIcon(nodeType);

  return (
    <g className="node-type-indicator pointer-events-none">
      <text
        x={0}
        y={iconSize / 3}
        textAnchor="middle"
        fontSize={iconSize}
        fill="white"
        fontWeight="bold"
        style={{ filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.3))' }}
      >
        {icon}
      </text>
    </g>
  );
};

// Helper component for status indicators
const StatusIndicator: React.FC<{
  status: ComponentStatus;
  position: { x: number; y: number };
  theme: ThemeConfiguration;
  scale: number;
}> = ({ status, position, theme, scale }) => {
  if (scale < 0.6) return null;

  const size = Math.max(4, 8 * scale);
  const colors = {
    healthy: '#10b981',
    warning: '#f59e0b',
    critical: '#dc2626',
    offline: '#6b7280',
    processing: '#3b82f6'
  };

  return (
    <g className="status-indicator pointer-events-none">
      <circle
        cx={position.x}
        cy={position.y}
        r={size}
        fill={colors[status]}
        stroke="white"
        strokeWidth={1}
        className={status === 'processing' ? 'animate-pulse' : undefined}
      />
      {status === 'critical' && (
        <text
          x={position.x}
          y={position.y + 2}
          textAnchor="middle"
          fontSize={size * 1.2}
          fill="white"
          fontWeight="bold"
        >
          !
        </text>
      )}
    </g>
  );
};

// Helper component for metrics overlay
const MetricsOverlay: React.FC<{
  metrics: any;
  position: { x: number; y: number };
  theme: ThemeConfiguration;
  compact: boolean;
  scale: number;
}> = ({ metrics, position, theme, compact, scale }) => {
  if (scale < 0.5) return null;

  const fontSize = compact ? 8 : 10;
  const lineHeight = fontSize + 2;

  const displayMetrics = [
    { label: 'CPU', value: `${metrics.cpu.current.toFixed(1)}%`, color: getCPUColor(metrics.cpu.current) },
    { label: 'MEM', value: `${metrics.memory.current.toFixed(1)}%`, color: getMemoryColor(metrics.memory.current) },
    { label: 'LAT', value: `${metrics.latency.current.toFixed(0)}ms`, color: getLatencyColor(metrics.latency.current) }
  ];

  return (
    <g className="metrics-overlay pointer-events-none">
      <rect
        x={position.x - 25}
        y={position.y - 5}
        width={50}
        height={displayMetrics.length * lineHeight + 5}
        fill="rgba(255, 255, 255, 0.9)"
        stroke={theme.colors.secondary}
        strokeWidth={1}
        rx={3}
        opacity={0.9}
      />
      {displayMetrics.map((metric, index) => (
        <g key={metric.label}>
          <text
            x={position.x - 20}
            y={position.y + lineHeight * index + lineHeight - 2}
            fontSize={fontSize}
            fill={metric.color}
            fontFamily={theme.fonts.monospace}
            fontWeight="500"
          >
            {metric.label}
          </text>
          <text
            x={position.x + 20}
            y={position.y + lineHeight * index + lineHeight - 2}
            fontSize={fontSize}
            fill={metric.color}
            fontFamily={theme.fonts.monospace}
            textAnchor="end"
          >
            {metric.value}
          </text>
        </g>
      ))}
    </g>
  );
};

// Helper component for connection points
const ConnectionPoint: React.FC<{
  connection: any;
  nodeRadius: number;
  angle: number;
  theme: ThemeConfiguration;
  active: boolean;
  scale: number;
}> = ({ connection, nodeRadius, angle, theme, active, scale }) => {
  if (scale < 0.8) return null;

  const x = Math.cos(angle) * (nodeRadius + 5);
  const y = Math.sin(angle) * (nodeRadius + 5);
  const size = active ? 3 : 2;

  return (
    <circle
      cx={x}
      cy={y}
      r={size}
      fill={active ? theme.colors.activity : theme.colors.secondary}
      stroke="white"
      strokeWidth={0.5}
      className={`pointer-events-none transition-all duration-200 ${active ? 'animate-pulse' : ''}`}
    />
  );
};

// Helper component for activity ripples
const ActivityRipple: React.FC<{
  center: { x: number; y: number };
  maxRadius: number;
  color: string;
  intensity: number;
}> = ({ center, maxRadius, color, intensity }) => {
  const rippleSpring = useSpring({
    from: { radius: 0, opacity: 0.8 },
    to: async (next) => {
      while (true) {
        await next({ radius: maxRadius, opacity: 0 });
        await next({ radius: 0, opacity: 0.8 });
      }
    },
    config: { duration: 2000 },
    loop: true
  });

  return (
    <animated.circle
      cx={center.x}
      cy={center.y}
      r={rippleSpring.radius}
      fill="none"
      stroke={color}
      strokeWidth={2}
      opacity={rippleSpring.opacity.to(o => o * intensity)}
      className="pointer-events-none"
    />
  );
};

// Utility functions
function getNodeColors(type: NodeType, status: ComponentStatus, theme: ThemeConfiguration) {
  const baseColors = {
    'subcortical': theme.colors.cognitive.subcortical,
    'cortical': theme.colors.cognitive.cortical,
    'thalamic': theme.colors.cognitive.thalamic,
    'mcp': theme.colors.mcp.primary,
    'mcp-tool': theme.colors.mcp.secondary,
    'storage': theme.colors.storage.primary,
    'network': theme.colors.network.primary
  };

  let primary = baseColors[type] || theme.colors.default;
  let border = primary;

  // Adjust colors based on status
  switch (status) {
    case 'warning':
      border = '#f59e0b';
      break;
    case 'critical':
      border = '#dc2626';
      primary = mixColors(primary, '#dc2626', 0.3);
      break;
    case 'offline':
      primary = desaturateColor(primary, 0.7);
      border = '#9ca3af';
      break;
    case 'processing':
      border = theme.colors.activity;
      break;
  }

  return { primary, border };
}

function getNodeTypeIcon(type: NodeType): string {
  const icons = {
    'subcortical': '⚡',
    'cortical': '🧠',
    'thalamic': '🔄',
    'mcp': '🔧',
    'mcp-tool': '⚙️',
    'storage': '💾',
    'network': '🌐'
  };
  return icons[type] || '⬢';
}

function getCPUColor(value: number): string {
  if (value > 80) return '#dc2626';
  if (value > 60) return '#f59e0b';
  return '#10b981';
}

function getMemoryColor(value: number): string {
  if (value > 85) return '#dc2626';
  if (value > 70) return '#f59e0b';
  return '#3b82f6';
}

function getLatencyColor(value: number): string {
  if (value > 200) return '#dc2626';
  if (value > 100) return '#f59e0b';
  return '#10b981';
}

function truncateLabel(text: string, nodeRadius: number, maxLength?: number): string {
  const maxChars = maxLength || Math.max(8, Math.floor(nodeRadius / 3));
  if (text.length <= maxChars) return text;
  return text.substring(0, maxChars - 3) + '...';
}

function mixColors(color1: string, color2: string, ratio: number): string {
  // Simple color mixing - in production, use a proper color manipulation library
  return color1; // Placeholder
}

function desaturateColor(color: string, amount: number): string {
  // Simple desaturation - in production, use a proper color manipulation library
  return color; // Placeholder
}

export default ArchitectureNode;