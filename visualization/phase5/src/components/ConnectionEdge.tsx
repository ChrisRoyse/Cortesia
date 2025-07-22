import React, { useMemo, useCallback, useState } from 'react';
import { useSpring, animated } from '@react-spring/web';
import * as d3 from 'd3';
import {
  ConnectionEdgeProps,
  InteractionType,
  ThemeConfiguration,
  ArchitectureNode,
  ConnectionEdge
} from '../types';

const ConnectionEdgeComponent: React.FC<ConnectionEdgeProps> = ({
  connection,
  sourceNode,
  targetNode,
  isHighlighted,
  theme,
  showFlow,
  scale,
  onInteraction
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Calculate connection path and properties
  const pathData = useMemo(() => {
    return calculateConnectionPath(
      sourceNode.position,
      targetNode.position,
      connection.type,
      sourceNode.size || 30,
      targetNode.size || 30,
      connection.metadata?.curvature || 0
    );
  }, [sourceNode, targetNode, connection.type, connection.metadata?.curvature]);

  // Connection styling
  const connectionStyle = useMemo(() => {
    const baseStyle = theme.connections[connection.type] || theme.connections['data-flow'];
    const strengthMultiplier = Math.max(0.3, Math.min(2.0, connection.strength));
    
    return {
      stroke: isHighlighted ? theme.colors.highlight : baseStyle.stroke,
      strokeWidth: baseStyle.strokeWidth * strengthMultiplier * (isHovered ? 1.5 : 1),
      opacity: connection.active ? baseStyle.opacity : baseStyle.opacity * 0.5,
      dashArray: getDashArray(connection.type),
      markerEnd: getMarkerEnd(connection.type)
    };
  }, [connection, theme, isHighlighted, isHovered]);

  // Animation springs
  const connectionSpring = useSpring({
    strokeWidth: connectionStyle.strokeWidth,
    opacity: connectionStyle.opacity,
    stroke: connectionStyle.stroke,
    config: { tension: 300, friction: 30 }
  });

  const flowSpring = useSpring({
    opacity: showFlow && connection.active ? 1 : 0,
    config: { duration: 300 }
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

  // Don't render if connection is too weak or scale is too small
  if (connection.strength < 0.1 || scale < 0.3) {
    return null;
  }

  return (
    <g
      className="connection-edge cursor-pointer"
      role="button"
      aria-label={`Connection from ${sourceNode.label} to ${targetNode.label}: ${connection.type}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
    >
      {/* Invisible thick path for easier interaction */}
      <path
        d={pathData.path}
        fill="none"
        stroke="transparent"
        strokeWidth={Math.max(10, connectionStyle.strokeWidth * 3)}
        className="pointer-events-stroke"
      />

      {/* Main connection path */}
      <animated.path
        d={pathData.path}
        fill="none"
        stroke={connectionSpring.stroke}
        strokeWidth={connectionSpring.strokeWidth}
        strokeOpacity={connectionSpring.opacity}
        strokeDasharray={connectionStyle.dashArray}
        markerEnd={connectionStyle.markerEnd}
        className="transition-all duration-200"
      />

      {/* Connection strength indicator */}
      {isHighlighted && connection.strength > 0.7 && (
        <StrengthIndicator
          path={pathData.path}
          midpoint={pathData.midpoint}
          strength={connection.strength}
          theme={theme}
        />
      )}

      {/* Data flow visualization */}
      {showFlow && connection.active && connection.dataFlow > 0 && (
        <animated.g opacity={flowSpring.opacity}>
          <DataFlowVisualization
            path={pathData.path}
            flow={connection.dataFlow}
            direction={pathData.direction}
            color={connectionStyle.stroke}
            speed={connection.metadata?.flowSpeed || 1}
          />
        </animated.g>
      )}

      {/* Connection label */}
      {(isHovered || isHighlighted) && connection.label && scale > 0.6 && (
        <ConnectionLabel
          text={connection.label}
          position={pathData.midpoint}
          rotation={pathData.angle}
          theme={theme}
          background={true}
        />
      )}

      {/* Connection metrics (for data flow connections) */}
      {connection.type === 'data-flow' && connection.metadata?.metrics && scale > 0.8 && (
        <ConnectionMetrics
          metrics={connection.metadata.metrics}
          position={pathData.midpoint}
          theme={theme}
          visible={isHovered || isHighlighted}
        />
      )}

      {/* Inhibitory connection indicator */}
      {connection.type === 'inhibition' && scale > 0.7 && (
        <InhibitoryIndicator
          position={pathData.midpoint}
          rotation={pathData.angle}
          theme={theme}
          active={connection.active}
        />
      )}

      {/* Bidirectional indicator */}
      {connection.type === 'bidirectional' && scale > 0.7 && (
        <BidirectionalIndicator
          startPos={sourceNode.position}
          endPos={targetNode.position}
          theme={theme}
          active={connection.active}
        />
      )}
    </g>
  );
};

// Helper component for connection strength visualization
const StrengthIndicator: React.FC<{
  path: string;
  midpoint: { x: number; y: number };
  strength: number;
  theme: ThemeConfiguration;
}> = ({ path, midpoint, strength, theme }) => {
  const size = 4 + (strength * 6);
  
  return (
    <g className="strength-indicator">
      <circle
        cx={midpoint.x}
        cy={midpoint.y}
        r={size}
        fill={theme.colors.highlight}
        stroke="white"
        strokeWidth={2}
        opacity={0.8}
        className="animate-pulse"
      />
      <text
        x={midpoint.x}
        y={midpoint.y + 2}
        textAnchor="middle"
        fontSize={size}
        fill="white"
        fontWeight="bold"
      >
        {Math.round(strength * 100)}
      </text>
    </g>
  );
};

// Helper component for data flow visualization
const DataFlowVisualization: React.FC<{
  path: string;
  flow: number;
  direction: 'forward' | 'backward' | 'bidirectional';
  color: string;
  speed: number;
}> = ({ path, flow, direction, color, speed }) => {
  const particleCount = Math.max(1, Math.min(5, Math.floor(flow * 3)));
  const particles = Array.from({ length: particleCount }, (_, i) => ({
    id: i,
    delay: (i * 0.5) / speed,
    size: 2 + (flow * 2)
  }));

  return (
    <g className="data-flow">
      {particles.map(particle => (
        <FlowParticle
          key={particle.id}
          path={path}
          delay={particle.delay}
          color={color}
          size={particle.size}
          direction={direction}
          speed={speed}
        />
      ))}
    </g>
  );
};

// Animated particle for data flow
const FlowParticle: React.FC<{
  path: string;
  delay: number;
  color: string;
  size: number;
  direction: 'forward' | 'backward' | 'bidirectional';
  speed: number;
}> = ({ path, delay, color, size, direction, speed }) => {
  const pathId = `flow-path-${Math.random().toString(36).substr(2, 9)}`;
  const duration = 2000 / speed;

  const particleSpring = useSpring({
    from: { opacity: 0 },
    to: async (next) => {
      while (true) {
        await next({ opacity: 0.8 });
        await next({ opacity: 0 });
      }
    },
    config: { duration: duration / 4 },
    delay: delay * 1000,
    loop: true
  });

  return (
    <g className="flow-particle">
      <defs>
        <path id={pathId} d={path} />
      </defs>
      <animated.circle
        r={size}
        fill={color}
        opacity={particleSpring.opacity}
        filter="url(#glow)"
      >
        <animateMotion
          dur={`${duration}ms`}
          repeatCount="indefinite"
          begin={`${delay}s`}
          rotate={direction === 'backward' ? 'reverse' : 'auto'}
        >
          <mpath href={`#${pathId}`} />
        </animateMotion>
      </animated.circle>
    </g>
  );
};

// Helper component for connection labels
const ConnectionLabel: React.FC<{
  text: string;
  position: { x: number; y: number };
  rotation: number;
  theme: ThemeConfiguration;
  background?: boolean;
}> = ({ text, position, rotation, theme, background = false }) => {
  const fontSize = 11;
  const padding = 4;
  const textWidth = text.length * fontSize * 0.6;

  return (
    <g className="connection-label pointer-events-none">
      {background && (
        <rect
          x={position.x - textWidth / 2 - padding}
          y={position.y - fontSize / 2 - padding}
          width={textWidth + padding * 2}
          height={fontSize + padding * 2}
          fill="rgba(255, 255, 255, 0.9)"
          stroke={theme.colors.secondary}
          strokeWidth={1}
          rx={3}
        />
      )}
      <text
        x={position.x}
        y={position.y + fontSize / 3}
        textAnchor="middle"
        fontSize={fontSize}
        fill={theme.colors.text}
        fontFamily={theme.fonts.primary}
        fontWeight="500"
        transform={`rotate(${rotation}, ${position.x}, ${position.y})`}
      >
        {text}
      </text>
    </g>
  );
};

// Helper component for connection metrics
const ConnectionMetrics: React.FC<{
  metrics: any;
  position: { x: number; y: number };
  theme: ThemeConfiguration;
  visible: boolean;
}> = ({ metrics, position, theme, visible }) => {
  if (!visible) return null;

  const metricsData = [
    { label: 'Throughput', value: `${metrics.throughput?.toFixed(1) || 0} ops/s` },
    { label: 'Latency', value: `${metrics.latency?.toFixed(0) || 0}ms` },
    { label: 'Errors', value: `${metrics.errorRate?.toFixed(1) || 0}%` }
  ];

  return (
    <g className="connection-metrics pointer-events-none">
      <rect
        x={position.x - 40}
        y={position.y - 25}
        width={80}
        height={50}
        fill="rgba(0, 0, 0, 0.8)"
        stroke={theme.colors.primary}
        strokeWidth={1}
        rx={5}
      />
      {metricsData.map((metric, index) => (
        <g key={metric.label}>
          <text
            x={position.x}
            y={position.y - 15 + (index * 12)}
            textAnchor="middle"
            fontSize={9}
            fill="white"
            fontFamily={theme.fonts.monospace}
          >
            {metric.label}: {metric.value}
          </text>
        </g>
      ))}
    </g>
  );
};

// Helper component for inhibitory connections
const InhibitoryIndicator: React.FC<{
  position: { x: number; y: number };
  rotation: number;
  theme: ThemeConfiguration;
  active: boolean;
}> = ({ position, rotation, theme, active }) => {
  const size = active ? 8 : 6;
  
  return (
    <g 
      className="inhibitory-indicator pointer-events-none"
      transform={`translate(${position.x}, ${position.y}) rotate(${rotation})`}
    >
      <circle
        cx={0}
        cy={0}
        r={size}
        fill={theme.connections.inhibition.stroke}
        stroke="white"
        strokeWidth={1}
        opacity={active ? 1 : 0.6}
      />
      <line
        x1={-size * 0.5}
        y1={0}
        x2={size * 0.5}
        y2={0}
        stroke="white"
        strokeWidth={2}
        strokeLinecap="round"
      />
    </g>
  );
};

// Helper component for bidirectional connections
const BidirectionalIndicator: React.FC<{
  startPos: { x: number; y: number };
  endPos: { x: number; y: number };
  theme: ThemeConfiguration;
  active: boolean;
}> = ({ startPos, endPos, theme, active }) => {
  const midpoint = {
    x: (startPos.x + endPos.x) / 2,
    y: (startPos.y + endPos.y) / 2
  };
  
  const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x) * 180 / Math.PI;
  
  return (
    <g className="bidirectional-indicator pointer-events-none">
      {/* Forward arrow */}
      <g transform={`translate(${midpoint.x + 10}, ${midpoint.y}) rotate(${angle})`}>
        <path
          d="M0,-3 L6,0 L0,3"
          fill={theme.colors.primary}
          opacity={active ? 0.8 : 0.4}
        />
      </g>
      
      {/* Backward arrow */}
      <g transform={`translate(${midpoint.x - 10}, ${midpoint.y}) rotate(${angle + 180})`}>
        <path
          d="M0,-3 L6,0 L0,3"
          fill={theme.colors.primary}
          opacity={active ? 0.8 : 0.4}
        />
      </g>
    </g>
  );
};

// Utility functions
function calculateConnectionPath(
  source: { x: number; y: number },
  target: { x: number; y: number },
  type: string,
  sourceRadius: number,
  targetRadius: number,
  curvature: number = 0
): {
  path: string;
  midpoint: { x: number; y: number };
  angle: number;
  direction: 'forward' | 'backward' | 'bidirectional';
} {
  // Calculate connection points on node edges
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  if (distance === 0) {
    return {
      path: '',
      midpoint: source,
      angle: 0,
      direction: 'forward'
    };
  }
  
  const unitX = dx / distance;
  const unitY = dy / distance;
  
  const sourcePoint = {
    x: source.x + unitX * sourceRadius,
    y: source.y + unitY * sourceRadius
  };
  
  const targetPoint = {
    x: target.x - unitX * targetRadius,
    y: target.y - unitY * targetRadius
  };
  
  const midpoint = {
    x: (sourcePoint.x + targetPoint.x) / 2,
    y: (sourcePoint.y + targetPoint.y) / 2
  };
  
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  
  let path: string;
  
  if (curvature === 0 || type === 'dependency') {
    // Straight line
    path = `M${sourcePoint.x},${sourcePoint.y} L${targetPoint.x},${targetPoint.y}`;
  } else {
    // Curved path
    const controlDistance = distance * Math.abs(curvature);
    const normalX = -unitY; // Perpendicular to connection line
    const normalY = unitX;
    
    const controlPoint = {
      x: midpoint.x + normalX * controlDistance,
      y: midpoint.y + normalY * controlDistance
    };
    
    path = `M${sourcePoint.x},${sourcePoint.y} Q${controlPoint.x},${controlPoint.y} ${targetPoint.x},${targetPoint.y}`;
  }
  
  return {
    path,
    midpoint,
    angle,
    direction: type === 'bidirectional' ? 'bidirectional' : 'forward'
  };
}

function getDashArray(connectionType: string): string {
  switch (connectionType) {
    case 'inhibition':
      return '5,5';
    case 'dependency':
      return '2,3';
    default:
      return 'none';
  }
}

function getMarkerEnd(connectionType: string): string {
  if (connectionType === 'bidirectional') {
    return undefined;
  }
  return 'url(#arrow)';
}

export default ConnectionEdgeComponent;