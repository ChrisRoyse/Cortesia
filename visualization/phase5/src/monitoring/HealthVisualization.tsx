/**
 * Phase 5 Health Visualization
 * 
 * Visual health indicators using color coding, progress bars, and trend charts.
 * Specialized for LLMKG brain-inspired architecture with cognitive pattern
 * visualization and real-time health status indicators.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  ComponentHealth,
  ComponentStatus,
  SystemHealthSummary,
  CognitivePatternActivation,
  BrainComponentHealth,
  MemorySystemMetrics,
  FederationNodeHealth,
  LLMKGComponentType,
  HealthVisualizationConfig
} from '../types/MonitoringTypes';

// Health Status Icon Components
const HealthStatusIcon: React.FC<{ status: ComponentStatus; size?: 'sm' | 'md' | 'lg' }> = ({ 
  status, 
  size = 'md' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const statusConfig = {
    active: { color: 'text-green-500', bgColor: 'bg-green-100', pulse: 'animate-pulse' },
    idle: { color: 'text-blue-500', bgColor: 'bg-blue-100', pulse: '' },
    processing: { color: 'text-yellow-500', bgColor: 'bg-yellow-100', pulse: 'animate-pulse' },
    error: { color: 'text-red-500', bgColor: 'bg-red-100', pulse: 'animate-bounce' },
    degraded: { color: 'text-orange-500', bgColor: 'bg-orange-100', pulse: '' },
    offline: { color: 'text-gray-400', bgColor: 'bg-gray-100', pulse: '' }
  };

  const config = statusConfig[status];

  return (
    <div className={`${sizeClasses[size]} ${config.bgColor} ${config.pulse} rounded-full flex items-center justify-center`}>
      <div className={`w-2 h-2 ${config.color} rounded-full`} />
    </div>
  );
};

// Health Score Gauge Component
interface HealthScoreGaugeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  animated?: boolean;
}

const HealthScoreGauge: React.FC<HealthScoreGaugeProps> = ({ 
  score, 
  size = 'md', 
  showText = true,
  animated = true 
}) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  
  useEffect(() => {
    if (animated) {
      const timer = setTimeout(() => setAnimatedScore(score), 100);
      return () => clearTimeout(timer);
    } else {
      setAnimatedScore(score);
    }
  }, [score, animated]);

  const sizeConfig = {
    sm: { size: 60, strokeWidth: 4, fontSize: 'text-xs' },
    md: { size: 80, strokeWidth: 6, fontSize: 'text-sm' },
    lg: { size: 120, strokeWidth: 8, fontSize: 'text-lg' }
  };

  const { size: circleSize, strokeWidth, fontSize } = sizeConfig[size];
  const radius = (circleSize - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (animatedScore / 100) * circumference;

  const getScoreColor = (score: number): string => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    if (score >= 40) return 'text-orange-500';
    return 'text-red-500';
  };

  const getStrokeColor = (score: number): string => {
    if (score >= 80) return 'stroke-green-500';
    if (score >= 60) return 'stroke-yellow-500';
    if (score >= 40) return 'stroke-orange-500';
    return 'stroke-red-500';
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={circleSize} height={circleSize} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={circleSize / 2}
          cy={circleSize / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          className="text-gray-200"
        />
        {/* Progress circle */}
        <circle
          cx={circleSize / 2}
          cy={circleSize / 2}
          r={radius}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={`${getStrokeColor(animatedScore)} transition-all duration-1000 ease-in-out`}
        />
      </svg>
      
      {showText && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`font-bold ${fontSize} ${getScoreColor(animatedScore)}`}>
            {Math.round(animatedScore)}
          </span>
        </div>
      )}
    </div>
  );
};

// Health Trend Indicator
interface HealthTrendProps {
  trend: 'improving' | 'stable' | 'degrading';
  size?: 'sm' | 'md' | 'lg';
}

const HealthTrendIndicator: React.FC<HealthTrendProps> = ({ trend, size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6', 
    lg: 'w-8 h-8'
  };

  const trendConfig = {
    improving: { 
      color: 'text-green-500', 
      icon: '↗', 
      bgColor: 'bg-green-100',
      label: 'Improving' 
    },
    stable: { 
      color: 'text-blue-500', 
      icon: '→', 
      bgColor: 'bg-blue-100',
      label: 'Stable' 
    },
    degrading: { 
      color: 'text-red-500', 
      icon: '↘', 
      bgColor: 'bg-red-100',
      label: 'Degrading' 
    }
  };

  const config = trendConfig[trend];

  return (
    <div className={`${sizeClasses[size]} ${config.bgColor} ${config.color} rounded flex items-center justify-center font-bold`}>
      {config.icon}
    </div>
  );
};

// System Health Overview Component
interface SystemHealthOverviewProps {
  healthSummary: SystemHealthSummary;
  compact?: boolean;
}

const SystemHealthOverview: React.FC<SystemHealthOverviewProps> = ({ 
  healthSummary, 
  compact = false 
}) => {
  if (compact) {
    return (
      <div className="bg-white rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <HealthStatusIcon status={healthSummary.overall} size="lg" />
            <div>
              <h3 className="font-semibold text-gray-900">System Health</h3>
              <p className="text-sm text-gray-500">
                {healthSummary.activeComponents}/{healthSummary.totalComponents} Active
              </p>
            </div>
          </div>
          <HealthScoreGauge score={healthSummary.healthScore} size="md" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">System Health Overview</h2>
        <p className="text-gray-600">Real-time monitoring of LLMKG architecture components</p>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Overall Health Score */}
          <div className="text-center">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Overall Health</h3>
            <HealthScoreGauge score={healthSummary.healthScore} size="lg" />
            <div className="mt-4 flex items-center justify-center space-x-2">
              <HealthStatusIcon status={healthSummary.overall} />
              <span className="text-sm font-medium text-gray-700 capitalize">
                {healthSummary.overall}
              </span>
            </div>
          </div>

          {/* Component Statistics */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Component Status</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Total Components</span>
                <span className="font-semibold">{healthSummary.totalComponents}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Active</span>
                <span className="font-semibold text-green-600">{healthSummary.activeComponents}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Degraded</span>
                <span className="font-semibold text-orange-600">{healthSummary.degradedComponents}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Offline</span>
                <span className="font-semibold text-gray-600">{healthSummary.offlineComponents}</span>
              </div>
            </div>
          </div>

          {/* LLMKG-Specific Health Metrics */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">LLMKG Systems</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Cognitive Patterns</span>
                <div className="flex items-center space-x-2">
                  <div className="w-12 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full transition-all duration-500" 
                      style={{ width: `${healthSummary.cognitivePatternActivity}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{healthSummary.cognitivePatternActivity}%</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Memory System</span>
                <div className="flex items-center space-x-2">
                  <div className="w-12 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500" 
                      style={{ width: `${healthSummary.memorySystemHealth}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{healthSummary.memorySystemHealth}%</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Federation</span>
                <div className="flex items-center space-x-2">
                  <div className="w-12 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full transition-all duration-500" 
                      style={{ width: `${healthSummary.federationHealth}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">{healthSummary.federationHealth}%</span>
                </div>
              </div>

              {healthSummary.activeAlerts > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Active Alerts</span>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    {healthSummary.activeAlerts}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6 text-sm text-gray-500 text-center">
          Last updated: {new Date(healthSummary.lastUpdated).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

// Component Health Grid
interface ComponentHealthGridProps {
  components: ComponentHealth[];
  onComponentClick?: (componentId: string) => void;
}

const ComponentHealthGrid: React.FC<ComponentHealthGridProps> = ({ 
  components, 
  onComponentClick 
}) => {
  const sortedComponents = useMemo(() => 
    components.sort((a, b) => {
      // Sort by status priority, then by health score
      const statusPriority = { 
        error: 0, 
        degraded: 1, 
        processing: 2, 
        active: 3, 
        idle: 4, 
        offline: 5 
      };
      
      const aPriority = statusPriority[a.status];
      const bPriority = statusPriority[b.status];
      
      if (aPriority !== bPriority) {
        return aPriority - bPriority;
      }
      
      return b.healthScore - a.healthScore;
    }), [components]
  );

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">Component Health Matrix</h2>
        <p className="text-gray-600">Individual component health status and metrics</p>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {sortedComponents.map((component) => (
            <div
              key={component.componentId}
              className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => onComponentClick?.(component.componentId)}
            >
              <div className="flex items-center justify-between mb-3">
                <HealthStatusIcon status={component.status} />
                <HealthScoreGauge score={component.healthScore} size="sm" showText={false} />
              </div>
              
              <h3 className="font-medium text-gray-900 text-sm mb-1 truncate">
                {component.componentId}
              </h3>
              
              <div className="text-xs text-gray-500 mb-2">
                <span className="capitalize">{component.status}</span> • 
                Score: {component.healthScore}
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">CPU</span>
                  <span className="font-medium">{component.cpuUsage.toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Memory</span>
                  <span className="font-medium">{component.memoryUsage.toFixed(1)}%</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Latency</span>
                  <span className="font-medium">{component.responseTime.toFixed(1)}ms</span>
                </div>
              </div>

              {component.trend !== 'stable' && (
                <div className="mt-2 flex items-center justify-center">
                  <HealthTrendIndicator trend={component.trend} size="sm" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Cognitive Pattern Activity Visualization
interface CognitivePatternVisualizationProps {
  patterns: CognitivePatternActivation[];
  brainComponents: BrainComponentHealth[];
}

const CognitivePatternVisualization: React.FC<CognitivePatternVisualizationProps> = ({
  patterns,
  brainComponents
}) => {
  const activePatterns = patterns.filter(p => 
    Date.now() - p.timestamp < 5 * 60 * 1000 // Last 5 minutes
  );

  const patternTypeColors = {
    abstract: 'bg-purple-500',
    convergent: 'bg-blue-500',
    divergent: 'bg-green-500',
    critical: 'bg-red-500',
    lateral: 'bg-yellow-500',
    systems: 'bg-indigo-500',
    adaptive: 'bg-pink-500'
  };

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">Cognitive Pattern Activity</h2>
        <p className="text-gray-600">Real-time brain-inspired cognitive pattern monitoring</p>
      </div>

      <div className="p-6">
        {activePatterns.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">🧠</div>
            <h3 className="text-lg font-medium text-gray-900">No Active Patterns</h3>
            <p className="text-gray-500">Cognitive patterns will appear here when activated</p>
          </div>
        ) : (
          <div className="space-y-4">
            {activePatterns.map((pattern, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div 
                      className={`w-4 h-4 rounded-full ${patternTypeColors[pattern.patternType]} animate-pulse`} 
                    />
                    <h3 className="font-medium text-gray-900 capitalize">
                      {pattern.patternType} Pattern
                    </h3>
                  </div>
                  <div className="text-sm text-gray-500">
                    {new Date(pattern.timestamp).toLocaleTimeString()}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-gray-500">Activation Level</p>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${patternTypeColors[pattern.patternType]} transition-all duration-500`}
                          style={{ width: `${pattern.activationLevel * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium">
                        {Math.round(pattern.activationLevel * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs text-gray-500">Confidence</p>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${pattern.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium">
                        {Math.round(pattern.confidence * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <p className="text-xs text-gray-500">Duration</p>
                    <p className="text-sm font-medium">{pattern.duration}ms</p>
                  </div>

                  <div>
                    <p className="text-xs text-gray-500">Affected Components</p>
                    <p className="text-sm font-medium">{pattern.affectedComponents.length}</p>
                  </div>
                </div>

                {pattern.inhibitionLevel > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">Inhibition Level</span>
                      <span className="font-medium text-red-600">
                        {Math.round(pattern.inhibitionLevel * 100)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Memory System Health Visualization
interface MemorySystemVisualizationProps {
  memoryMetrics: MemorySystemMetrics;
}

const MemorySystemVisualization: React.FC<MemorySystemVisualizationProps> = ({
  memoryMetrics
}) => {
  const metrics = [
    { 
      label: 'SDR Utilization', 
      value: memoryMetrics.sdrUtilization, 
      color: 'bg-blue-500',
      description: 'Sparse Distributed Representation usage'
    },
    { 
      label: 'Working Memory Load', 
      value: memoryMetrics.workingMemoryLoad, 
      color: 'bg-green-500',
      description: 'Active working memory utilization'
    },
    { 
      label: 'Long-term Storage', 
      value: memoryMetrics.longTermStorageUsage, 
      color: 'bg-purple-500',
      description: 'Persistent storage utilization'
    },
    { 
      label: 'Index Efficiency', 
      value: memoryMetrics.indexEfficiency, 
      color: 'bg-indigo-500',
      description: 'Search index performance'
    }
  ];

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">Memory System Health</h2>
        <p className="text-gray-600">LLMKG memory subsystem performance metrics</p>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Memory utilization metrics */}
          <div className="space-y-4">
            {metrics.map((metric, index) => (
              <div key={index}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-700">{metric.label}</span>
                  <span className="text-sm font-semibold">{Math.round(metric.value * 100)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`${metric.color} h-3 rounded-full transition-all duration-1000 ease-in-out`}
                    style={{ width: `${metric.value * 100}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">{metric.description}</p>
              </div>
            ))}
          </div>

          {/* Performance metrics */}
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <p className="text-sm text-gray-500">Hit Rate</p>
                <p className="text-2xl font-bold text-green-600">
                  {Math.round(memoryMetrics.hitRate * 100)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">Compression</p>
                <p className="text-2xl font-bold text-blue-600">
                  {memoryMetrics.compressionRatio.toFixed(1)}x
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <p className="text-sm text-gray-500">Access Latency</p>
                <p className="text-2xl font-bold text-gray-900">
                  {memoryMetrics.accessLatency.toFixed(1)}ms
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">Fragmentation</p>
                <p className="text-2xl font-bold text-orange-600">
                  {Math.round(memoryMetrics.fragmentationLevel * 100)}%
                </p>
              </div>
            </div>

            <div className="text-center">
              <p className="text-sm text-gray-500">Eviction Rate</p>
              <p className="text-lg font-bold text-red-600">
                {memoryMetrics.evictionRate.toFixed(2)}/min
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Health Visualization Component
interface HealthVisualizationProps {
  systemHealth: SystemHealthSummary;
  componentHealth: ComponentHealth[];
  cognitivePatterns: CognitivePatternActivation[];
  brainComponents: BrainComponentHealth[];
  memoryMetrics: MemorySystemMetrics;
  config?: HealthVisualizationConfig;
  onComponentClick?: (componentId: string) => void;
}

const HealthVisualization: React.FC<HealthVisualizationProps> = ({
  systemHealth,
  componentHealth,
  cognitivePatterns,
  brainComponents,
  memoryMetrics,
  config = {
    showTrends: true,
    animateChanges: true,
    colorScheme: 'default',
    updateFrequency: 1000,
    showDetails: true
  },
  onComponentClick
}) => {
  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <SystemHealthOverview healthSummary={systemHealth} />

      {/* Component Health Grid */}
      <ComponentHealthGrid 
        components={componentHealth} 
        onComponentClick={onComponentClick}
      />

      {/* Cognitive Pattern Visualization */}
      <CognitivePatternVisualization 
        patterns={cognitivePatterns}
        brainComponents={brainComponents}
      />

      {/* Memory System Visualization */}
      <MemorySystemVisualization memoryMetrics={memoryMetrics} />
    </div>
  );
};

export {
  HealthVisualization,
  SystemHealthOverview,
  ComponentHealthGrid,
  CognitivePatternVisualization,
  MemorySystemVisualization,
  HealthStatusIcon,
  HealthScoreGauge,
  HealthTrendIndicator
};

export default HealthVisualization;