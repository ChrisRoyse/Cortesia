import React, { useMemo, useState, useEffect } from 'react';
import { useAppSelector } from '../../stores';

export type MetricTrend = 'up' | 'down' | 'stable';
export type MetricStatus = 'normal' | 'warning' | 'critical' | 'success';
export type MetricSize = 'small' | 'medium' | 'large';

interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  subtitle?: string;
  description?: string;
  trend?: MetricTrend;
  trendValue?: number;
  trendPeriod?: string;
  status?: MetricStatus;
  size?: MetricSize;
  icon?: React.ReactNode;
  showProgress?: boolean;
  progressMax?: number;
  sparklineData?: number[];
  format?: 'number' | 'percentage' | 'bytes' | 'duration' | 'custom';
  precision?: number;
  interactive?: boolean;
  onClick?: () => void;
  className?: string;
  style?: React.CSSProperties;
}

// Utility functions for formatting values
const formatValue = (
  value: number | string, 
  format: MetricCardProps['format'] = 'number',
  precision: number = 1
): string => {
  if (typeof value === 'string') return value;
  
  switch (format) {
    case 'percentage':
      return `${(value * 100).toFixed(precision)}%`;
    
    case 'bytes':
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      let size = value;
      let unitIndex = 0;
      
      while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
      }
      
      return `${size.toFixed(precision)} ${units[unitIndex]}`;
    
    case 'duration':
      if (value < 1000) return `${value.toFixed(precision)}ms`;
      if (value < 60000) return `${(value / 1000).toFixed(precision)}s`;
      if (value < 3600000) return `${(value / 60000).toFixed(precision)}m`;
      return `${(value / 3600000).toFixed(precision)}h`;
    
    case 'number':
    default:
      if (value >= 1000000) return `${(value / 1000000).toFixed(precision)}M`;
      if (value >= 1000) return `${(value / 1000).toFixed(precision)}K`;
      return value.toFixed(precision);
  }
};

// Trend arrow component
const TrendArrow: React.FC<{ trend: MetricTrend; theme: string }> = ({ trend, theme }) => {
  const getArrowColor = () => {
    switch (trend) {
      case 'up': return '#10b981';
      case 'down': return '#ef4444';
      case 'stable': return theme === 'dark' ? '#6b7280' : '#9ca3af';
      default: return theme === 'dark' ? '#6b7280' : '#9ca3af';
    }
  };

  const getArrowPath = () => {
    switch (trend) {
      case 'up': return 'M5 15l5-5 5 5H5z';
      case 'down': return 'M5 9l5 5 5-5H5z';
      case 'stable': return 'M4 12h12M4 12l4-4M4 12l4 4';
      default: return '';
    }
  };

  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill={getArrowColor()}>
      <path d={getArrowPath()} />
    </svg>
  );
};

// Sparkline component
const Sparkline: React.FC<{ 
  data: number[]; 
  width: number; 
  height: number; 
  color: string;
  theme: string;
}> = ({ data, width, height, color, theme }) => {
  const points = useMemo(() => {
    if (data.length < 2) return '';
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    
    return data
      .map((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * height;
        return `${x},${y}`;
      })
      .join(' ');
  }, [data, width, height]);

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id="sparklineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0.1" />
        </linearGradient>
      </defs>
      
      {/* Fill area */}
      {points && (
        <polygon
          points={`0,${height} ${points} ${width},${height}`}
          fill="url(#sparklineGradient)"
          stroke="none"
        />
      )}
      
      {/* Line */}
      {points && (
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
    </svg>
  );
};

// Progress bar component
const ProgressBar: React.FC<{
  value: number;
  max: number;
  status: MetricStatus;
  theme: string;
  showValue?: boolean;
}> = ({ value, max, status, theme, showValue = true }) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  const getProgressColor = () => {
    switch (status) {
      case 'success': return '#10b981';
      case 'warning': return '#f59e0b';
      case 'critical': return '#ef4444';
      case 'normal':
      default: return '#3b82f6';
    }
  };

  return (
    <div className="w-full">
      <div 
        className="h-2 rounded-full overflow-hidden"
        style={{ 
          backgroundColor: theme === 'dark' ? '#374151' : '#e5e7eb' 
        }}
      >
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{
            width: `${percentage}%`,
            backgroundColor: getProgressColor(),
          }}
        />
      </div>
      {showValue && (
        <div className="flex justify-between mt-1 text-xs opacity-75">
          <span>{formatValue(value, 'number')}</span>
          <span>{formatValue(max, 'number')}</span>
        </div>
      )}
    </div>
  );
};

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  description,
  trend,
  trendValue,
  trendPeriod = '24h',
  status = 'normal',
  size = 'medium',
  icon,
  showProgress = false,
  progressMax = 100,
  sparklineData,
  format = 'number',
  precision = 1,
  interactive = false,
  onClick,
  className = '',
  style,
}) => {
  const theme = useAppSelector(state => state.dashboard.config.theme);
  const enableAnimations = useAppSelector(state => state.dashboard.config.enableAnimations);
  
  const [isHovered, setIsHovered] = useState(false);
  const [animatedValue, setAnimatedValue] = useState<number>(0);

  // Animate value changes
  useEffect(() => {
    if (!enableAnimations || typeof value !== 'number') {
      setAnimatedValue(typeof value === 'number' ? value : 0);
      return;
    }

    const targetValue = value;
    const startValue = animatedValue;
    const duration = 1000; // 1 second
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (ease-out)
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      
      const currentValue = startValue + (targetValue - startValue) * easedProgress;
      setAnimatedValue(currentValue);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, enableAnimations, animatedValue]);

  // Card styling based on size and theme
  const cardStyles = useMemo(() => {
    const baseStyles: React.CSSProperties = {
      backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
      borderColor: theme === 'dark' ? '#374151' : '#e5e7eb',
      color: theme === 'dark' ? '#ffffff' : '#374151',
      borderRadius: '8px',
      border: '1px solid',
      transition: enableAnimations ? 'all 0.2s ease-in-out' : 'none',
      cursor: interactive ? 'pointer' : 'default',
    };

    // Status-based border color
    if (status !== 'normal') {
      const statusColors = {
        success: '#10b981',
        warning: '#f59e0b',
        critical: '#ef4444',
        normal: baseStyles.borderColor,
      };
      baseStyles.borderLeftColor = statusColors[status];
      baseStyles.borderLeftWidth = '4px';
    }

    // Size-based padding and dimensions
    const sizeStyles = {
      small: { padding: '12px', minWidth: '120px' },
      medium: { padding: '16px', minWidth: '160px' },
      large: { padding: '20px', minWidth: '200px' },
    };
    
    Object.assign(baseStyles, sizeStyles[size]);

    // Hover effect
    if (interactive && isHovered) {
      baseStyles.transform = 'translateY(-2px)';
      baseStyles.boxShadow = theme === 'dark' 
        ? '0 8px 25px rgba(0, 0, 0, 0.3)' 
        : '0 8px 25px rgba(0, 0, 0, 0.1)';
      baseStyles.borderColor = theme === 'dark' ? '#6b7280' : '#d1d5db';
    }

    return baseStyles;
  }, [theme, size, status, interactive, isHovered, enableAnimations]);

  // Value display
  const displayValue = useMemo(() => {
    const val = enableAnimations ? animatedValue : (typeof value === 'number' ? value : value);
    return formatValue(val, format, precision);
  }, [value, animatedValue, format, precision, enableAnimations]);

  // Status indicator color
  const statusColor = useMemo(() => {
    const colors = {
      normal: theme === 'dark' ? '#6b7280' : '#9ca3af',
      success: '#10b981',
      warning: '#f59e0b',
      critical: '#ef4444',
    };
    return colors[status];
  }, [status, theme]);

  return (
    <div
      className={`metric-card ${className}`}
      style={{ ...cardStyles, ...style }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={interactive ? onClick : undefined}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon && (
            <div 
              className="flex-shrink-0"
              style={{ color: statusColor }}
            >
              {icon}
            </div>
          )}
          <div>
            <h3 
              className={`font-semibold ${size === 'small' ? 'text-sm' : size === 'large' ? 'text-lg' : 'text-base'}`}
              style={{ color: theme === 'dark' ? '#ffffff' : '#374151' }}
            >
              {title}
            </h3>
            {subtitle && (
              <p 
                className="text-xs opacity-75 mt-1"
                style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}
              >
                {subtitle}
              </p>
            )}
          </div>
        </div>
        
        {/* Status indicator */}
        {status !== 'normal' && (
          <div
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{ backgroundColor: statusColor }}
            title={`Status: ${status}`}
          />
        )}
      </div>

      {/* Value */}
      <div className="mb-3">
        <div className="flex items-baseline gap-1">
          <span 
            className={`font-bold ${size === 'small' ? 'text-xl' : size === 'large' ? 'text-4xl' : 'text-2xl'}`}
            style={{ color: theme === 'dark' ? '#ffffff' : '#111827' }}
          >
            {displayValue}
          </span>
          {unit && (
            <span 
              className={`${size === 'small' ? 'text-sm' : 'text-lg'} opacity-75`}
              style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}
            >
              {unit}
            </span>
          )}
        </div>
        
        {/* Trend indicator */}
        {trend && (
          <div className="flex items-center gap-2 mt-2">
            <TrendArrow trend={trend} theme={theme} />
            {trendValue && (
              <span 
                className="text-sm"
                style={{ 
                  color: trend === 'up' ? '#10b981' : trend === 'down' ? '#ef4444' : (theme === 'dark' ? '#d1d5db' : '#6b7280')
                }}
              >
                {trend === 'up' ? '+' : trend === 'down' ? '-' : ''}
                {formatValue(Math.abs(trendValue), format, precision)} ({trendPeriod})
              </span>
            )}
          </div>
        )}
      </div>

      {/* Progress bar */}
      {showProgress && typeof value === 'number' && (
        <div className="mb-3">
          <ProgressBar
            value={value}
            max={progressMax}
            status={status}
            theme={theme}
          />
        </div>
      )}

      {/* Sparkline */}
      {sparklineData && sparklineData.length > 1 && (
        <div className="mb-3">
          <Sparkline
            data={sparklineData}
            width={size === 'small' ? 80 : size === 'large' ? 120 : 100}
            height={size === 'small' ? 20 : size === 'large' ? 30 : 25}
            color={statusColor}
            theme={theme}
          />
        </div>
      )}

      {/* Description */}
      {description && (
        <p 
          className="text-xs opacity-75 leading-relaxed"
          style={{ color: theme === 'dark' ? '#d1d5db' : '#6b7280' }}
        >
          {description}
        </p>
      )}
    </div>
  );
};

// Preset metric cards for common LLMKG metrics
export const CPUMetricCard: React.FC<Omit<MetricCardProps, 'title' | 'format'>> = (props) => (
  <MetricCard
    title="CPU Usage"
    format="percentage"
    icon={<span>⚡</span>}
    {...props}
  />
);

export const MemoryMetricCard: React.FC<Omit<MetricCardProps, 'title' | 'format'>> = (props) => (
  <MetricCard
    title="Memory"
    format="bytes"
    icon={<span>🧠</span>}
    {...props}
  />
);

export const LatencyMetricCard: React.FC<Omit<MetricCardProps, 'title' | 'format'>> = (props) => (
  <MetricCard
    title="Latency"
    format="duration"
    icon={<span>⏱️</span>}
    {...props}
  />
);

export const ThroughputMetricCard: React.FC<Omit<MetricCardProps, 'title' | 'unit'>> = (props) => (
  <MetricCard
    title="Throughput"
    unit="ops/s"
    icon={<span>📊</span>}
    {...props}
  />
);

export const ErrorRateMetricCard: React.FC<Omit<MetricCardProps, 'title' | 'format'>> = (props) => (
  <MetricCard
    title="Error Rate"
    format="percentage"
    icon={<span>⚠️</span>}
    {...props}
  />
);

export default MetricCard;