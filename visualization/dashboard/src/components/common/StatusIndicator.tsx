import React, { useMemo, useState, useEffect } from 'react';
import { useAppSelector } from '../../stores';

export type StatusType = 'online' | 'offline' | 'connecting' | 'error' | 'warning' | 'maintenance' | 'idle' | 'active';
export type StatusSize = 'small' | 'medium' | 'large';
export type StatusVariant = 'dot' | 'badge' | 'card' | 'pulse';

interface StatusIndicatorProps {
  status: StatusType;
  label?: string;
  description?: string;
  size?: StatusSize;
  variant?: StatusVariant;
  showLabel?: boolean;
  showTimestamp?: boolean;
  lastUpdated?: Date;
  animated?: boolean;
  blinking?: boolean;
  interactive?: boolean;
  onClick?: () => void;
  className?: string;
  style?: React.CSSProperties;
}

interface ConnectionStatusProps extends Omit<StatusIndicatorProps, 'status'> {
  isConnected: boolean;
  connectionState?: 'connecting' | 'connected' | 'disconnected' | 'error';
  reconnectAttempts?: number;
  maxReconnectAttempts?: number;
}

interface SystemStatusProps extends Omit<StatusIndicatorProps, 'status'> {
  health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  uptime?: number;
  lastHealthCheck?: Date;
}

// Status configuration
const STATUS_CONFIG = {
  online: {
    color: '#10b981',
    backgroundColor: '#ecfdf5',
    label: 'Online',
    icon: '●',
    pulse: true,
  },
  offline: {
    color: '#ef4444',
    backgroundColor: '#fef2f2',
    label: 'Offline',
    icon: '●',
    pulse: false,
  },
  connecting: {
    color: '#f59e0b',
    backgroundColor: '#fffbeb',
    label: 'Connecting',
    icon: '◐',
    pulse: true,
  },
  error: {
    color: '#ef4444',
    backgroundColor: '#fef2f2',
    label: 'Error',
    icon: '⚠',
    pulse: true,
  },
  warning: {
    color: '#f59e0b',
    backgroundColor: '#fffbeb',
    label: 'Warning',
    icon: '⚠',
    pulse: false,
  },
  maintenance: {
    color: '#8b5cf6',
    backgroundColor: '#faf5ff',
    label: 'Maintenance',
    icon: '🔧',
    pulse: false,
  },
  idle: {
    color: '#6b7280',
    backgroundColor: '#f9fafb',
    label: 'Idle',
    icon: '⏸',
    pulse: false,
  },
  active: {
    color: '#3b82f6',
    backgroundColor: '#eff6ff',
    label: 'Active',
    icon: '▶',
    pulse: true,
  },
};

// Size configuration
const SIZE_CONFIG = {
  small: {
    dot: 8,
    badge: { padding: '2px 6px', fontSize: '10px' },
    card: { padding: '8px', fontSize: '12px' },
    icon: '12px',
  },
  medium: {
    dot: 12,
    badge: { padding: '4px 8px', fontSize: '12px' },
    card: { padding: '12px', fontSize: '14px' },
    icon: '16px',
  },
  large: {
    dot: 16,
    badge: { padding: '6px 12px', fontSize: '14px' },
    card: { padding: '16px', fontSize: '16px' },
    icon: '20px',
  },
};

// Utility function to format time ago
const timeAgo = (date: Date): string => {
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

// Pulse animation component
const PulseAnimation: React.FC<{ 
  color: string; 
  size: number; 
  enabled: boolean;
  theme: string;
}> = ({ color, size, enabled, theme }) => {
  if (!enabled) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: size * 2,
        height: size * 2,
        borderRadius: '50%',
        backgroundColor: color,
        opacity: 0.3,
        animation: 'pulse 2s infinite',
        zIndex: -1,
      }}
    />
  );
};

// Dot variant component
const StatusDot: React.FC<{
  config: typeof STATUS_CONFIG[StatusType];
  size: StatusSize;
  animated: boolean;
  blinking: boolean;
  theme: string;
}> = ({ config, size, animated, blinking, theme }) => {
  const dotSize = SIZE_CONFIG[size].dot;
  
  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      {animated && config.pulse && (
        <PulseAnimation 
          color={config.color} 
          size={dotSize} 
          enabled={true}
          theme={theme}
        />
      )}
      <div
        style={{
          width: dotSize,
          height: dotSize,
          borderRadius: '50%',
          backgroundColor: config.color,
          animation: blinking ? 'blink 1s infinite' : 'none',
          transition: 'all 0.2s ease-in-out',
        }}
      />
    </div>
  );
};

// Badge variant component
const StatusBadge: React.FC<{
  config: typeof STATUS_CONFIG[StatusType];
  label: string;
  size: StatusSize;
  theme: string;
}> = ({ config, label, size, theme }) => {
  const sizeConfig = SIZE_CONFIG[size].badge;
  
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        padding: sizeConfig.padding,
        borderRadius: '12px',
        backgroundColor: theme === 'dark' ? config.color + '20' : config.backgroundColor,
        color: config.color,
        fontSize: sizeConfig.fontSize,
        fontWeight: '500',
        border: `1px solid ${config.color}40`,
      }}
    >
      <span style={{ fontSize: SIZE_CONFIG[size].icon }}>{config.icon}</span>
      <span>{label}</span>
    </div>
  );
};

// Card variant component
const StatusCard: React.FC<{
  config: typeof STATUS_CONFIG[StatusType];
  label: string;
  description?: string;
  showTimestamp: boolean;
  lastUpdated?: Date;
  size: StatusSize;
  theme: string;
  interactive: boolean;
  onClick?: () => void;
}> = ({ config, label, description, showTimestamp, lastUpdated, size, theme, interactive, onClick }) => {
  const sizeConfig = SIZE_CONFIG[size].card;
  
  return (
    <div
      style={{
        padding: sizeConfig.padding,
        borderRadius: '8px',
        backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
        border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        borderLeftColor: config.color,
        borderLeftWidth: '4px',
        cursor: interactive ? 'pointer' : 'default',
        transition: 'all 0.2s ease-in-out',
      }}
      onClick={interactive ? onClick : undefined}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <span 
          style={{ 
            fontSize: SIZE_CONFIG[size].icon, 
            color: config.color 
          }}
        >
          {config.icon}
        </span>
        <span 
          style={{ 
            fontSize: sizeConfig.fontSize, 
            fontWeight: '600',
            color: theme === 'dark' ? '#ffffff' : '#374151',
          }}
        >
          {label}
        </span>
      </div>
      
      {description && (
        <p 
          style={{
            fontSize: size === 'small' ? '11px' : '12px',
            color: theme === 'dark' ? '#d1d5db' : '#6b7280',
            margin: '0 0 4px 0',
            lineHeight: '1.4',
          }}
        >
          {description}
        </p>
      )}
      
      {showTimestamp && lastUpdated && (
        <p 
          style={{
            fontSize: '10px',
            color: theme === 'dark' ? '#9ca3af' : '#9ca3af',
            margin: 0,
          }}
        >
          Updated {timeAgo(lastUpdated)}
        </p>
      )}
    </div>
  );
};

// Pulse variant component
const StatusPulse: React.FC<{
  config: typeof STATUS_CONFIG[StatusType];
  size: StatusSize;
  theme: string;
}> = ({ config, size, theme }) => {
  const dotSize = SIZE_CONFIG[size].dot;
  
  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}>
      <div
        style={{
          width: dotSize,
          height: dotSize,
          borderRadius: '50%',
          backgroundColor: config.color,
          position: 'relative',
          zIndex: 1,
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: dotSize * 3,
          height: dotSize * 3,
          borderRadius: '50%',
          backgroundColor: config.color,
          opacity: 0.2,
          animation: 'pulse 2s infinite',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: dotSize * 2,
          height: dotSize * 2,
          borderRadius: '50%',
          backgroundColor: config.color,
          opacity: 0.4,
          animation: 'pulse 2s infinite 0.5s',
        }}
      />
    </div>
  );
};

// Main StatusIndicator component
export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  description,
  size = 'medium',
  variant = 'dot',
  showLabel = true,
  showTimestamp = false,
  lastUpdated,
  animated = true,
  blinking = false,
  interactive = false,
  onClick,
  className = '',
  style,
}) => {
  const theme = useAppSelector(state => state.dashboard.config.theme);
  const enableAnimations = useAppSelector(state => state.dashboard.config.enableAnimations);
  
  const config = STATUS_CONFIG[status];
  const effectiveLabel = label || config.label;
  const effectiveAnimated = animated && enableAnimations;

  // Add CSS for animations
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0% {
          opacity: 0.6;
          transform: translate(-50%, -50%) scale(0.8);
        }
        50% {
          opacity: 0.3;
          transform: translate(-50%, -50%) scale(1.2);
        }
        100% {
          opacity: 0;
          transform: translate(-50%, -50%) scale(1.5);
        }
      }
      
      @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
      }
      
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const renderVariant = () => {
    switch (variant) {
      case 'badge':
        return (
          <StatusBadge
            config={config}
            label={effectiveLabel}
            size={size}
            theme={theme}
          />
        );
      
      case 'card':
        return (
          <StatusCard
            config={config}
            label={effectiveLabel}
            description={description}
            showTimestamp={showTimestamp}
            lastUpdated={lastUpdated}
            size={size}
            theme={theme}
            interactive={interactive}
            onClick={onClick}
          />
        );
      
      case 'pulse':
        return (
          <StatusPulse
            config={config}
            size={size}
            theme={theme}
          />
        );
      
      case 'dot':
      default:
        return (
          <StatusDot
            config={config}
            size={size}
            animated={effectiveAnimated}
            blinking={blinking}
            theme={theme}
          />
        );
    }
  };

  return (
    <div 
      className={`status-indicator ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: showLabel && variant === 'dot' ? '6px' : '0',
        ...style,
      }}
    >
      {renderVariant()}
      
      {showLabel && variant === 'dot' && (
        <span 
          style={{
            fontSize: SIZE_CONFIG[size].card.fontSize,
            color: theme === 'dark' ? '#ffffff' : '#374151',
          }}
        >
          {effectiveLabel}
        </span>
      )}
    </div>
  );
};

// ConnectionStatusIndicator component
export const ConnectionStatusIndicator: React.FC<ConnectionStatusProps> = ({
  isConnected,
  connectionState = isConnected ? 'connected' : 'disconnected',
  reconnectAttempts = 0,
  maxReconnectAttempts = 5,
  ...props
}) => {
  const status: StatusType = useMemo(() => {
    if (connectionState === 'error') return 'error';
    if (connectionState === 'connecting') return 'connecting';
    if (connectionState === 'connected') return 'online';
    return 'offline';
  }, [connectionState]);

  const description = useMemo(() => {
    if (connectionState === 'connecting' && reconnectAttempts > 0) {
      return `Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`;
    }
    if (connectionState === 'connected') {
      return 'Connected to WebSocket';
    }
    if (connectionState === 'error') {
      return 'Connection error occurred';
    }
    return 'Disconnected from server';
  }, [connectionState, reconnectAttempts, maxReconnectAttempts]);

  return (
    <StatusIndicator
      status={status}
      description={description}
      animated={connectionState === 'connecting'}
      blinking={connectionState === 'error'}
      {...props}
    />
  );
};

// SystemStatusIndicator component
export const SystemStatusIndicator: React.FC<SystemStatusProps> = ({
  health,
  uptime,
  lastHealthCheck,
  ...props
}) => {
  const status: StatusType = useMemo(() => {
    switch (health) {
      case 'healthy': return 'online';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      case 'unknown': return 'idle';
      default: return 'offline';
    }
  }, [health]);

  const description = useMemo(() => {
    const parts = [];
    if (uptime) {
      const hours = Math.floor(uptime / 3600000);
      const minutes = Math.floor((uptime % 3600000) / 60000);
      parts.push(`Uptime: ${hours}h ${minutes}m`);
    }
    if (lastHealthCheck) {
      parts.push(`Last check: ${timeAgo(lastHealthCheck)}`);
    }
    return parts.join(' • ');
  }, [uptime, lastHealthCheck]);

  return (
    <StatusIndicator
      status={status}
      description={description}
      {...props}
    />
  );
};

// Preset status indicators for common LLMKG components
export const WebSocketStatusIndicator: React.FC<Omit<ConnectionStatusProps, 'label'>> = (props) => {
  const webSocketState = useAppSelector(state => state.webSocket);
  
  return (
    <div data-testid="ws-status">
      <ConnectionStatusIndicator
        label="WebSocket"
        isConnected={webSocketState.isConnected}
        connectionState={webSocketState.connectionState}
        reconnectAttempts={webSocketState.reconnectAttempts}
        maxReconnectAttempts={webSocketState.maxReconnectAttempts}
        lastUpdated={new Date()}
        {...props}
      />
    </div>
  );
};

export const MCPStatusIndicator: React.FC<Omit<StatusIndicatorProps, 'status' | 'label'>> = (props) => {
  const mcpState = useAppSelector(state => state.mcp);
  
  const status: StatusType = useMemo(() => {
    if (mcpState.loading) return 'connecting';
    if (mcpState.error) return 'error';
    if (mcpState.tools.length > 0) return 'online';
    return 'offline';
  }, [mcpState]);

  return (
    <StatusIndicator
      status={status}
      label="MCP Tools"
      description={`${mcpState.tools.length} tools available`}
      animated={mcpState.loading}
      {...props}
    />
  );
};

export default StatusIndicator;