import React from 'react';
import {
  Box,
  Chip,
  CircularProgress,
  Tooltip,
  Typography,
  keyframes
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Block as BlockIcon,
  Speed as SpeedIcon,
  SignalCellularAlt as SignalIcon
} from '@mui/icons-material';
import { ToolStatusInfo, ToolStatus } from '../../types';

interface StatusIndicatorProps {
  status: ToolStatusInfo;
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  showDetails?: boolean;
  animated?: boolean;
  onClick?: () => void;
}

// Animation keyframes
const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const rotate = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  size = 'medium',
  showLabel = false,
  showDetails = false,
  animated = true,
  onClick
}) => {
  // Get status color
  const getStatusColor = (health: ToolStatus): string => {
    switch (health) {
      case 'healthy':
        return '#4caf50';
      case 'degraded':
        return '#ff9800';
      case 'unavailable':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  // Get status icon
  const getStatusIcon = (health: ToolStatus, iconSize: number) => {
    const color = getStatusColor(health);
    const shouldAnimate = animated && (health === 'degraded' || health === 'unavailable');
    
    switch (health) {
      case 'healthy':
        return (
          <CheckCircleIcon 
            sx={{ 
              fontSize: iconSize, 
              color,
              animation: animated ? `${pulse} 3s ease-in-out infinite` : 'none'
            }} 
          />
        );
      case 'degraded':
        return (
          <WarningIcon 
            sx={{ 
              fontSize: iconSize, 
              color,
              animation: shouldAnimate ? `${pulse} 2s ease-in-out infinite` : 'none'
            }} 
          />
        );
      case 'unavailable':
        return (
          <ErrorIcon 
            sx={{ 
              fontSize: iconSize, 
              color,
              animation: shouldAnimate ? `${pulse} 1s ease-in-out infinite` : 'none'
            }} 
          />
        );
      default:
        return (
          <BlockIcon 
            sx={{ 
              fontSize: iconSize, 
              color,
              animation: animated ? `${rotate} 4s linear infinite` : 'none'
            }} 
          />
        );
    }
  };

  // Get icon size based on prop
  const getIconSize = (): number => {
    switch (size) {
      case 'small':
        return 16;
      case 'large':
        return 32;
      default:
        return 24;
    }
  };

  // Format response time
  const formatResponseTime = (ms: number): string => {
    if (ms < 1000) {
      return `${ms.toFixed(0)}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Build tooltip content
  const tooltipContent = (
    <Box>
      <Typography variant="body2" fontWeight="bold">
        Status: {status.health}
      </Typography>
      <Typography variant="caption" component="div">
        Response Time: {formatResponseTime(status.responseTime)}
      </Typography>
      <Typography variant="caption" component="div">
        Error Rate: {(status.errorRate * 100).toFixed(1)}%
      </Typography>
      <Typography variant="caption" component="div">
        Available: {status.available ? 'Yes' : 'No'}
      </Typography>
      <Typography variant="caption" component="div">
        Last Checked: {new Date(status.lastChecked).toLocaleTimeString()}
      </Typography>
      {status.message && (
        <Typography variant="caption" component="div" sx={{ mt: 0.5 }}>
          {status.message}
        </Typography>
      )}
    </Box>
  );

  // Response time indicator
  const ResponseTimeIndicator = () => {
    const getColor = () => {
      if (status.responseTime < 200) return 'success';
      if (status.responseTime < 500) return 'warning';
      return 'error';
    };

    return (
      <Chip
        icon={<SpeedIcon />}
        label={formatResponseTime(status.responseTime)}
        size="small"
        color={getColor()}
        variant="outlined"
        sx={{ ml: 1 }}
      />
    );
  };

  // Signal strength indicator
  const SignalStrengthIndicator = () => {
    const strength = status.available ? (1 - status.errorRate) * 100 : 0;
    
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
        <SignalIcon 
          sx={{ 
            fontSize: 16,
            color: strength > 95 ? 'success.main' : 
                   strength > 90 ? 'warning.main' : 'error.main'
          }} 
        />
        <Typography variant="caption" sx={{ ml: 0.5 }}>
          {strength.toFixed(0)}%
        </Typography>
      </Box>
    );
  };

  // Main render
  if (showDetails) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          cursor: onClick ? 'pointer' : 'default',
          '&:hover': onClick ? { opacity: 0.8 } : {}
        }}
        onClick={onClick}
      >
        <Tooltip title={tooltipContent} arrow placement="top">
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {getStatusIcon(status.health, getIconSize())}
            {showLabel && (
              <Typography
                variant={size === 'small' ? 'caption' : 'body2'}
                sx={{ ml: 1, color: getStatusColor(status.health), fontWeight: 'medium' }}
              >
                {status.health}
              </Typography>
            )}
          </Box>
        </Tooltip>
        <ResponseTimeIndicator />
        <SignalStrengthIndicator />
      </Box>
    );
  }

  // Simple indicator
  return (
    <Tooltip title={tooltipContent} arrow placement="top">
      <Box
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          cursor: onClick ? 'pointer' : 'default',
          '&:hover': onClick ? { opacity: 0.8 } : {}
        }}
        onClick={onClick}
      >
        {getStatusIcon(status.health, getIconSize())}
        {showLabel && (
          <Typography
            variant={size === 'small' ? 'caption' : 'body2'}
            sx={{ ml: 1, color: getStatusColor(status.health), fontWeight: 'medium' }}
          >
            {status.health}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
};

// Mini status indicator for compact displays
export const MiniStatusIndicator: React.FC<{ status: ToolStatus }> = ({ status }) => {
  const getColor = () => {
    switch (status) {
      case 'healthy':
        return '#4caf50';
      case 'degraded':
        return '#ff9800';
      case 'unavailable':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  return (
    <Box
      sx={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        backgroundColor: getColor(),
        animation: status === 'unavailable' ? `${pulse} 1s ease-in-out infinite` : 'none'
      }}
    />
  );
};

// Status progress ring
export const StatusProgressRing: React.FC<{ 
  status: ToolStatusInfo;
  size?: number;
  thickness?: number;
}> = ({ status, size = 40, thickness = 4 }) => {
  const normalizedErrorRate = Math.min(status.errorRate * 100, 100);
  const successRate = 100 - normalizedErrorRate;

  const getColor = () => {
    if (successRate >= 99) return 'success.main';
    if (successRate >= 95) return 'warning.main';
    return 'error.main';
  };

  return (
    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
      <CircularProgress
        variant="determinate"
        value={successRate}
        size={size}
        thickness={thickness}
        sx={{
          color: getColor(),
          [`& .MuiCircularProgress-circle`]: {
            strokeLinecap: 'round'
          }
        }}
      />
      <Box
        sx={{
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          position: 'absolute',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="caption" component="div" color="text.secondary">
          {successRate.toFixed(0)}%
        </Typography>
      </Box>
    </Box>
  );
};

export default StatusIndicator;