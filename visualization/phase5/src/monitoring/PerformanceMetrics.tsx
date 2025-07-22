/**
 * Phase 5 Performance Metrics Dashboard
 * 
 * Performance dashboards showing latency, throughput, resource usage with
 * historical trending and LLMKG-specific cognitive performance analysis.
 * Includes real-time charts, heatmaps, and predictive analytics.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  PerformanceMetrics,
  ComponentHealth,
  HistoricalDataPoint,
  PerformanceChartConfig,
  LLMKGComponentType,
  CognitivePatternActivation,
  BrainComponentHealth
} from '../types/MonitoringTypes';

// Performance Chart Component using Canvas for high performance
interface PerformanceChartProps {
  data: HistoricalDataPoint[];
  config: PerformanceChartConfig;
  width: number;
  height: number;
  title: string;
  yAxisLabel: string;
  color: string;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data,
  config,
  width,
  height,
  title,
  yAxisLabel,
  color
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; value: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set up high DPI rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Calculate data ranges
    const timeRange = Math.max(...data.map(d => d.timestamp)) - Math.min(...data.map(d => d.timestamp));
    const minValue = Math.min(...data.map(d => d.value));
    const maxValue = Math.max(...data.map(d => d.value));
    const valueRange = maxValue - minValue || 1;

    // Draw grid lines
    ctx.strokeStyle = '#f3f4f6';
    ctx.lineWidth = 1;

    // Vertical grid lines (time)
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i * chartWidth) / 10;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();
    }

    // Horizontal grid lines (values)
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i * chartHeight) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw data line/area
    if (data.length > 1) {
      const points = data.map(d => ({
        x: padding.left + ((d.timestamp - Math.min(...data.map(p => p.timestamp))) / timeRange) * chartWidth,
        y: padding.top + chartHeight - ((d.value - minValue) / valueRange) * chartHeight
      }));

      if (config.chartType === 'area') {
        // Draw area fill
        ctx.fillStyle = `${color}20`;
        ctx.beginPath();
        ctx.moveTo(points[0].x, padding.top + chartHeight);
        points.forEach(point => ctx.lineTo(point.x, point.y));
        ctx.lineTo(points[points.length - 1].x, padding.top + chartHeight);
        ctx.closePath();
        ctx.fill();
      }

      // Draw line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.stroke();

      // Draw data points
      ctx.fillStyle = color;
      points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui, sans-serif';

    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const value = minValue + (i * valueRange) / 5;
      const y = padding.top + chartHeight - (i * chartHeight) / 5;
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(1), padding.left - 10, y + 4);
    }

    // X-axis labels (time)
    const now = Date.now();
    for (let i = 0; i <= 5; i++) {
      const timeAgo = (config.timeWindow / 5) * i;
      const time = new Date(now - timeAgo * 1000);
      const x = padding.left + chartWidth - (i * chartWidth) / 5;
      ctx.textAlign = 'center';
      ctx.fillText(time.toLocaleTimeString(), x, padding.top + chartHeight + 20);
    }

    // Y-axis title
    ctx.save();
    ctx.translate(20, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yAxisLabel, 0, 0);
    ctx.restore();

  }, [data, config, width, height, color, yAxisLabel]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate which data point is closest to mouse
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const timeRange = Math.max(...data.map(d => d.timestamp)) - Math.min(...data.map(d => d.timestamp));
    
    if (x >= padding.left && x <= padding.left + chartWidth) {
      const relativeX = (x - padding.left) / chartWidth;
      const targetTime = Math.min(...data.map(d => d.timestamp)) + relativeX * timeRange;
      
      // Find closest data point
      const closest = data.reduce((prev, curr) => 
        Math.abs(curr.timestamp - targetTime) < Math.abs(prev.timestamp - targetTime) ? curr : prev
      );
      
      setHoveredPoint({ x, y, value: closest.value });
    } else {
      setHoveredPoint(null);
    }
  };

  return (
    <div className="relative">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredPoint(null)}
        className="border border-gray-200 rounded cursor-crosshair"
      />
      
      {hoveredPoint && (
        <div
          className="absolute bg-black text-white text-xs rounded px-2 py-1 pointer-events-none z-10"
          style={{ 
            left: hoveredPoint.x + 10, 
            top: hoveredPoint.y - 30,
            transform: hoveredPoint.x > width - 100 ? 'translateX(-100%)' : 'none'
          }}
        >
          Value: {hoveredPoint.value.toFixed(2)}
        </div>
      )}
    </div>
  );
};

// Performance Metrics Heatmap
interface PerformanceHeatmapProps {
  componentMetrics: Map<string, PerformanceMetrics[]>;
  metric: 'cpu' | 'memory' | 'latency' | 'throughput';
  width: number;
  height: number;
}

const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({
  componentMetrics,
  metric,
  width,
  height
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set up high DPI rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    const components = Array.from(componentMetrics.keys());
    const cellWidth = width / components.length;
    const cellHeight = 20;
    const maxRows = Math.floor(height / cellHeight);

    // Get all values for normalization
    const allValues: number[] = [];
    componentMetrics.forEach(metrics => {
      metrics.slice(-maxRows).forEach(m => {
        allValues.push(m[metric]);
      });
    });

    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const valueRange = maxValue - minValue || 1;

    // Draw heatmap
    components.forEach((componentId, componentIndex) => {
      const metrics = componentMetrics.get(componentId) || [];
      const recentMetrics = metrics.slice(-maxRows);

      recentMetrics.forEach((m, timeIndex) => {
        const intensity = (m[metric] - minValue) / valueRange;
        const red = Math.round(255 * intensity);
        const green = Math.round(255 * (1 - intensity));
        const blue = 0;

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(
          componentIndex * cellWidth,
          (maxRows - timeIndex - 1) * cellHeight,
          cellWidth - 1,
          cellHeight - 1
        );
      });
    });

    // Draw component labels
    ctx.fillStyle = '#374151';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';

    components.forEach((componentId, index) => {
      const x = index * cellWidth + cellWidth / 2;
      ctx.save();
      ctx.translate(x, height - 5);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(componentId, 0, 0);
      ctx.restore();
    });

  }, [componentMetrics, metric, width, height]);

  return (
    <div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        {metric.toUpperCase()} Heatmap
      </h3>
      <canvas
        ref={canvasRef}
        className="border border-gray-200 rounded"
      />
      <div className="flex justify-between mt-2 text-xs text-gray-500">
        <span>Low</span>
        <span>High</span>
      </div>
    </div>
  );
};

// Real-time Performance Metrics Card
interface MetricsCardProps {
  title: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  color: string;
  icon?: string;
}

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  unit,
  trend,
  trendValue,
  color,
  icon = '📊'
}) => {
  const trendColor = {
    up: trend === 'up' && title.toLowerCase().includes('error') ? 'text-red-600' : 'text-green-600',
    down: trend === 'down' && title.toLowerCase().includes('error') ? 'text-green-600' : 'text-red-600',
    stable: 'text-gray-600'
  };

  const trendIcon = {
    up: '↗',
    down: '↘',
    stable: '→'
  };

  return (
    <div className="bg-white rounded-lg border shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="text-3xl">{icon}</div>
        <div className={`text-sm font-medium ${trendColor[trend]}`}>
          {trendIcon[trend]} {Math.abs(trendValue).toFixed(1)}%
        </div>
      </div>
      
      <h3 className="text-sm font-medium text-gray-500 mb-1">{title}</h3>
      <div className="flex items-baseline">
        <span className={`text-2xl font-bold ${color}`}>
          {value.toFixed(value < 10 ? 2 : value < 100 ? 1 : 0)}
        </span>
        <span className="ml-1 text-sm text-gray-500">{unit}</span>
      </div>
    </div>
  );
};

// Cognitive Performance Analysis
interface CognitivePerformanceProps {
  cognitivePatterns: CognitivePatternActivation[];
  brainComponents: BrainComponentHealth[];
}

const CognitivePerformanceAnalysis: React.FC<CognitivePerformanceProps> = ({
  cognitivePatterns,
  brainComponents
}) => {
  const cognitiveMetrics = useMemo(() => {
    const recentPatterns = cognitivePatterns.filter(p => 
      Date.now() - p.timestamp < 60 * 60 * 1000 // Last hour
    );

    const avgActivationLevel = recentPatterns.length > 0
      ? recentPatterns.reduce((sum, p) => sum + p.activationLevel, 0) / recentPatterns.length
      : 0;

    const avgConfidence = recentPatterns.length > 0
      ? recentPatterns.reduce((sum, p) => sum + p.confidence, 0) / recentPatterns.length
      : 0;

    const avgDuration = recentPatterns.length > 0
      ? recentPatterns.reduce((sum, p) => sum + p.duration, 0) / recentPatterns.length
      : 0;

    const patternTypeDistribution = recentPatterns.reduce((acc, p) => {
      acc[p.patternType] = (acc[p.patternType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      avgActivationLevel,
      avgConfidence,
      avgDuration,
      totalPatterns: recentPatterns.length,
      patternTypeDistribution
    };
  }, [cognitivePatterns]);

  const brainMetrics = useMemo(() => {
    if (brainComponents.length === 0) {
      return {
        avgNeuralActivity: 0,
        avgSynapticStrength: 0,
        avgPlasticity: 0,
        avgInhibitionBalance: 0,
        avgExcitationBalance: 0
      };
    }

    return {
      avgNeuralActivity: brainComponents.reduce((sum, c) => sum + c.neuralActivityLevel, 0) / brainComponents.length,
      avgSynapticStrength: brainComponents.reduce((sum, c) => sum + c.synapticStrength, 0) / brainComponents.length,
      avgPlasticity: brainComponents.reduce((sum, c) => sum + c.plasticityScore, 0) / brainComponents.length,
      avgInhibitionBalance: brainComponents.reduce((sum, c) => sum + c.inhibitionBalance, 0) / brainComponents.length,
      avgExcitationBalance: brainComponents.reduce((sum, c) => sum + c.excitationBalance, 0) / brainComponents.length
    };
  }, [brainComponents]);

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">Cognitive Performance Analysis</h2>
        <p className="text-gray-600">Brain-inspired cognitive processing metrics</p>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <MetricsCard
            title="Activation Level"
            value={cognitiveMetrics.avgActivationLevel * 100}
            unit="%"
            trend="stable"
            trendValue={0}
            color="text-purple-600"
            icon="🧠"
          />

          <MetricsCard
            title="Confidence Score"
            value={cognitiveMetrics.avgConfidence * 100}
            unit="%"
            trend="stable"
            trendValue={0}
            color="text-blue-600"
            icon="🎯"
          />

          <MetricsCard
            title="Avg Duration"
            value={cognitiveMetrics.avgDuration}
            unit="ms"
            trend="stable"
            trendValue={0}
            color="text-green-600"
            icon="⏱️"
          />

          <MetricsCard
            title="Active Patterns"
            value={cognitiveMetrics.totalPatterns}
            unit="patterns"
            trend="stable"
            trendValue={0}
            color="text-indigo-600"
            icon="🔮"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pattern Type Distribution */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Pattern Type Distribution</h3>
            <div className="space-y-3">
              {Object.entries(cognitiveMetrics.patternTypeDistribution).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 capitalize">{type}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full" 
                        style={{ width: `${(count / cognitiveMetrics.totalPatterns) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Brain Component Health */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Brain Component Health</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Neural Activity</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full" 
                      style={{ width: `${brainMetrics.avgNeuralActivity * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">
                    {Math.round(brainMetrics.avgNeuralActivity * 100)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Synaptic Strength</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-indigo-600 h-2 rounded-full" 
                      style={{ width: `${brainMetrics.avgSynapticStrength * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">
                    {Math.round(brainMetrics.avgSynapticStrength * 100)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Plasticity</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-pink-600 h-2 rounded-full" 
                      style={{ width: `${brainMetrics.avgPlasticity * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">
                    {Math.round(brainMetrics.avgPlasticity * 100)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">I/E Balance</span>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-red-600">
                    I: {brainMetrics.avgInhibitionBalance.toFixed(2)}
                  </span>
                  <span className="text-xs text-blue-600">
                    E: {brainMetrics.avgExcitationBalance.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Performance Metrics Dashboard
interface PerformanceMetricsDashboardProps {
  componentMetrics: Map<string, PerformanceMetrics[]>;
  componentHealth: ComponentHealth[];
  cognitivePatterns: CognitivePatternActivation[];
  brainComponents: BrainComponentHealth[];
  config?: PerformanceChartConfig;
  onComponentClick?: (componentId: string) => void;
}

const PerformanceMetricsDashboard: React.FC<PerformanceMetricsDashboardProps> = ({
  componentMetrics,
  componentHealth,
  cognitivePatterns,
  brainComponents,
  config = {
    timeWindow: 300, // 5 minutes
    refreshRate: 1000,
    metrics: ['cpu', 'memory', 'latency', 'throughput'],
    showPredictions: false,
    enableZoom: true,
    chartType: 'line'
  },
  onComponentClick
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'cpu' | 'memory' | 'latency' | 'throughput'>('cpu');
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [timeWindow, setTimeWindow] = useState(config.timeWindow);

  // Calculate aggregate metrics
  const aggregateMetrics = useMemo(() => {
    const allMetrics: PerformanceMetrics[] = [];
    componentMetrics.forEach(metrics => allMetrics.push(...metrics));
    
    if (allMetrics.length === 0) {
      return {
        avgCPU: 0,
        avgMemory: 0,
        avgLatency: 0,
        avgThroughput: 0,
        maxCPU: 0,
        maxMemory: 0,
        maxLatency: 0,
        totalThroughput: 0
      };
    }

    // Get recent metrics (last 5 minutes)
    const cutoff = Date.now() - (5 * 60 * 1000);
    const recentMetrics = allMetrics.filter(m => m.timestamp > cutoff);

    return {
      avgCPU: recentMetrics.reduce((sum, m) => sum + m.cpu, 0) / recentMetrics.length,
      avgMemory: recentMetrics.reduce((sum, m) => sum + m.memory, 0) / recentMetrics.length,
      avgLatency: recentMetrics.reduce((sum, m) => sum + m.latency, 0) / recentMetrics.length,
      avgThroughput: recentMetrics.reduce((sum, m) => sum + m.throughput, 0) / recentMetrics.length,
      maxCPU: Math.max(...recentMetrics.map(m => m.cpu)),
      maxMemory: Math.max(...recentMetrics.map(m => m.memory)),
      maxLatency: Math.max(...recentMetrics.map(m => m.latency)),
      totalThroughput: recentMetrics.reduce((sum, m) => sum + m.throughput, 0)
    };
  }, [componentMetrics]);

  // Convert performance metrics to historical data points
  const getHistoricalData = (metric: keyof PerformanceMetrics, componentId?: string): HistoricalDataPoint[] => {
    if (componentId && componentMetrics.has(componentId)) {
      return componentMetrics.get(componentId)!.map(m => ({
        timestamp: m.timestamp,
        value: m[metric] as number,
        metric,
        componentId
      }));
    } else {
      // Aggregate all components
      const allData: HistoricalDataPoint[] = [];
      componentMetrics.forEach((metrics, id) => {
        metrics.forEach(m => {
          allData.push({
            timestamp: m.timestamp,
            value: m[metric] as number,
            metric,
            componentId: id
          });
        });
      });
      return allData.sort((a, b) => a.timestamp - b.timestamp);
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="Average CPU"
          value={aggregateMetrics.avgCPU}
          unit="%"
          trend="stable"
          trendValue={0}
          color="text-blue-600"
          icon="💻"
        />

        <MetricsCard
          title="Average Memory"
          value={aggregateMetrics.avgMemory}
          unit="%"
          trend="stable"
          trendValue={0}
          color="text-green-600"
          icon="🧮"
        />

        <MetricsCard
          title="Average Latency"
          value={aggregateMetrics.avgLatency}
          unit="ms"
          trend="stable"
          trendValue={0}
          color="text-yellow-600"
          icon="⚡"
        />

        <MetricsCard
          title="Total Throughput"
          value={aggregateMetrics.totalThroughput}
          unit="req/s"
          trend="stable"
          trendValue={0}
          color="text-purple-600"
          icon="🚀"
        />
      </div>

      {/* Performance Charts */}
      <div className="bg-white rounded-lg border shadow-sm">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Performance Trends</h2>
            
            <div className="flex items-center space-x-4">
              {/* Metric Selector */}
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value as any)}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm"
              >
                <option value="cpu">CPU Usage</option>
                <option value="memory">Memory Usage</option>
                <option value="latency">Latency</option>
                <option value="throughput">Throughput</option>
              </select>

              {/* Time Window Selector */}
              <select
                value={timeWindow}
                onChange={(e) => setTimeWindow(Number(e.target.value))}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm"
              >
                <option value={60}>1 Minute</option>
                <option value={300}>5 Minutes</option>
                <option value={900}>15 Minutes</option>
                <option value={3600}>1 Hour</option>
              </select>
            </div>
          </div>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Main Performance Chart */}
            <PerformanceChart
              data={getHistoricalData(selectedMetric, selectedComponent || undefined)}
              config={{ ...config, timeWindow }}
              width={400}
              height={300}
              title={`${selectedMetric.toUpperCase()} Over Time`}
              yAxisLabel={selectedMetric === 'latency' ? 'Milliseconds' : 
                         selectedMetric === 'throughput' ? 'Requests/sec' : 'Percentage'}
              color={selectedMetric === 'cpu' ? '#3b82f6' :
                     selectedMetric === 'memory' ? '#10b981' :
                     selectedMetric === 'latency' ? '#f59e0b' : '#8b5cf6'}
            />

            {/* Performance Heatmap */}
            <PerformanceHeatmap
              componentMetrics={componentMetrics}
              metric={selectedMetric}
              width={400}
              height={300}
            />
          </div>
        </div>
      </div>

      {/* Cognitive Performance Analysis */}
      <CognitivePerformanceAnalysis
        cognitivePatterns={cognitivePatterns}
        brainComponents={brainComponents}
      />
    </div>
  );
};

export {
  PerformanceMetricsDashboard,
  PerformanceChart,
  PerformanceHeatmap,
  MetricsCard,
  CognitivePerformanceAnalysis
};

export default PerformanceMetricsDashboard;