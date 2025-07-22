/**
 * Real-time performance monitoring and optimization for LLMKG Phase 4
 * Provides comprehensive performance tracking and automatic optimization
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

interface PerformanceMetrics {
  fps: number;
  frameTime: number; // ms
  memoryUsage: {
    used: number; // MB
    total: number; // MB
    percentage: number;
  };
  gpuUsage?: {
    utilization: number; // percentage
    memory: number; // MB
  };
  networkActivity: {
    bytesReceived: number;
    bytesSent: number;
    requestsPerSecond: number;
  };
  renderingStats: {
    drawCalls: number;
    triangles: number;
    particles: number;
    textureMemory: number; // MB
  };
  componentPerformance: {
    [componentName: string]: {
      renderTime: number;
      updateTime: number;
      memoryUsage: number;
    };
  };
}

interface PerformanceAlert {
  id: string;
  type: 'warning' | 'error' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  suggestion?: string;
  action?: () => void;
}

interface OptimizationSettings {
  autoOptimize: boolean;
  targetFPS: number;
  maxMemoryUsage: number; // MB
  enableAdaptiveQuality: boolean;
  enableGPUOptimization: boolean;
  enableNetworkOptimization: boolean;
  alertThresholds: {
    lowFPS: number;
    highMemory: number; // percentage
    highGPU: number; // percentage
    slowRender: number; // ms
  };
}

interface PerformanceMonitorProps {
  isVisible: boolean;
  onClose: () => void;
  onOptimizationChange: (settings: OptimizationSettings) => void;
  onQualityAdjustment: (quality: 'low' | 'medium' | 'high' | 'auto') => void;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  isVisible,
  onClose,
  onOptimizationChange,
  onQualityAdjustment
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    frameTime: 16.67,
    memoryUsage: { used: 0, total: 0, percentage: 0 },
    networkActivity: { bytesReceived: 0, bytesSent: 0, requestsPerSecond: 0 },
    renderingStats: { drawCalls: 0, triangles: 0, particles: 0, textureMemory: 0 },
    componentPerformance: {}
  });

  const [historicalData, setHistoricalData] = useState<{
    fps: number[];
    memory: number[];
    frameTime: number[];
    timestamps: number[];
  }>({
    fps: [],
    memory: [],
    frameTime: [],
    timestamps: []
  });

  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [settings, setSettings] = useState<OptimizationSettings>({
    autoOptimize: true,
    targetFPS: 60,
    maxMemoryUsage: 512,
    enableAdaptiveQuality: true,
    enableGPUOptimization: true,
    enableNetworkOptimization: true,
    alertThresholds: {
      lowFPS: 30,
      highMemory: 80,
      highGPU: 85,
      slowRender: 33.33 // 30 FPS equivalent
    }
  });

  const [activeTab, setActiveTab] = useState<'metrics' | 'charts' | 'alerts' | 'optimization'>('metrics');
  const [isRecording, setIsRecording] = useState(false);

  const metricsIntervalRef = useRef<NodeJS.Timeout>();
  const frameTimeRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(performance.now());
  const networkStatsRef = useRef({ bytesReceived: 0, bytesSent: 0, requestCount: 0 });

  // Performance measurement
  useEffect(() => {
    if (!isVisible) return;

    startPerformanceMonitoring();

    return () => {
      stopPerformanceMonitoring();
    };
  }, [isVisible]);

  const startPerformanceMonitoring = useCallback(() => {
    // Start FPS monitoring
    const measureFrame = () => {
      const now = performance.now();
      const frameTime = now - lastFrameTimeRef.current;
      frameTimeRef.current = frameTime;
      lastFrameTimeRef.current = now;
      requestAnimationFrame(measureFrame);
    };
    measureFrame();

    // Start metrics collection
    metricsIntervalRef.current = setInterval(() => {
      collectMetrics();
    }, 1000); // Update every second

    setIsRecording(true);
  }, []);

  const stopPerformanceMonitoring = useCallback(() => {
    if (metricsIntervalRef.current) {
      clearInterval(metricsIntervalRef.current);
    }
    setIsRecording(false);
  }, []);

  const collectMetrics = useCallback(async () => {
    try {
      const now = performance.now();
      const fps = frameTimeRef.current > 0 ? 1000 / frameTimeRef.current : 60;

      // Memory metrics
      const memoryInfo = await getMemoryInfo();
      
      // GPU metrics (if available)
      const gpuInfo = await getGPUInfo();
      
      // Network metrics
      const networkInfo = getNetworkInfo();
      
      // Rendering metrics
      const renderingInfo = getRenderingInfo();
      
      // Component performance
      const componentInfo = getComponentPerformance();

      const newMetrics: PerformanceMetrics = {
        fps: Math.round(fps),
        frameTime: frameTimeRef.current,
        memoryUsage: memoryInfo,
        gpuUsage: gpuInfo,
        networkActivity: networkInfo,
        renderingStats: renderingInfo,
        componentPerformance: componentInfo
      };

      setMetrics(newMetrics);

      // Update historical data
      setHistoricalData(prev => {
        const maxDataPoints = 60; // Keep 1 minute of data
        const newFps = [...prev.fps, newMetrics.fps].slice(-maxDataPoints);
        const newMemory = [...prev.memory, newMetrics.memoryUsage.percentage].slice(-maxDataPoints);
        const newFrameTime = [...prev.frameTime, newMetrics.frameTime].slice(-maxDataPoints);
        const newTimestamps = [...prev.timestamps, now].slice(-maxDataPoints);

        return {
          fps: newFps,
          memory: newMemory,
          frameTime: newFrameTime,
          timestamps: newTimestamps
        };
      });

      // Check for performance issues
      checkPerformanceAlerts(newMetrics);

      // Apply automatic optimizations
      if (settings.autoOptimize) {
        applyAutoOptimizations(newMetrics);
      }

    } catch (error) {
      console.error('Error collecting performance metrics:', error);
    }
  }, [settings]);

  // Memory info using Performance API
  const getMemoryInfo = async (): Promise<PerformanceMetrics['memoryUsage']> => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const used = memory.usedJSHeapSize / (1024 * 1024); // Convert to MB
      const total = memory.totalJSHeapSize / (1024 * 1024);
      return {
        used: Math.round(used),
        total: Math.round(total),
        percentage: Math.round((used / total) * 100)
      };
    }
    
    // Fallback estimation
    return {
      used: 50,
      total: 100,
      percentage: 50
    };
  };

  // GPU info using WebGL context
  const getGPUInfo = async (): Promise<PerformanceMetrics['gpuUsage']> => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          // This is a rough estimation - actual GPU usage is not directly accessible
          return {
            utilization: Math.random() * 20 + 40, // Mock data: 40-60%
            memory: Math.random() * 100 + 200 // Mock data: 200-300MB
          };
        }
      }
    } catch (error) {
      // GPU info not available
    }
    
    return undefined;
  };

  // Network info
  const getNetworkInfo = (): PerformanceMetrics['networkActivity'] => {
    // Use Navigation Timing API
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      return {
        bytesReceived: networkStatsRef.current.bytesReceived,
        bytesSent: networkStatsRef.current.bytesSent,
        requestsPerSecond: networkStatsRef.current.requestCount
      };
    }

    return {
      bytesReceived: 0,
      bytesSent: 0,
      requestsPerSecond: 0
    };
  };

  // Rendering info (mock data - would integrate with actual 3D engine)
  const getRenderingInfo = (): PerformanceMetrics['renderingStats'] => {
    return {
      drawCalls: Math.floor(Math.random() * 100) + 50,
      triangles: Math.floor(Math.random() * 10000) + 5000,
      particles: Math.floor(Math.random() * 1000) + 2000,
      textureMemory: Math.floor(Math.random() * 50) + 25
    };
  };

  // Component performance tracking
  const getComponentPerformance = (): PerformanceMetrics['componentPerformance'] => {
    // Mock data - would integrate with React DevTools profiling
    return {
      'VisualizationCanvas': {
        renderTime: Math.random() * 5 + 10,
        updateTime: Math.random() * 2 + 1,
        memoryUsage: Math.random() * 10 + 5
      },
      'FilteringSystem': {
        renderTime: Math.random() * 2 + 1,
        updateTime: Math.random() * 1 + 0.5,
        memoryUsage: Math.random() * 5 + 2
      },
      'DataProcessor': {
        renderTime: Math.random() * 3 + 2,
        updateTime: Math.random() * 4 + 2,
        memoryUsage: Math.random() * 8 + 3
      }
    };
  };

  // Performance alert system
  const checkPerformanceAlerts = useCallback((metrics: PerformanceMetrics) => {
    const newAlerts: PerformanceAlert[] = [];

    // Low FPS alert
    if (metrics.fps < settings.alertThresholds.lowFPS) {
      newAlerts.push({
        id: `low-fps-${Date.now()}`,
        type: 'warning',
        title: 'Low Frame Rate',
        message: `FPS dropped to ${metrics.fps}, below target of ${settings.alertThresholds.lowFPS}`,
        timestamp: new Date(),
        suggestion: 'Consider reducing particle count or disabling post-processing',
        action: () => onQualityAdjustment('low')
      });
    }

    // High memory usage alert
    if (metrics.memoryUsage.percentage > settings.alertThresholds.highMemory) {
      newAlerts.push({
        id: `high-memory-${Date.now()}`,
        type: 'error',
        title: 'High Memory Usage',
        message: `Memory usage at ${metrics.memoryUsage.percentage}%, exceeding ${settings.alertThresholds.highMemory}% threshold`,
        timestamp: new Date(),
        suggestion: 'Clear unused data or reduce visualization complexity',
        action: () => triggerMemoryCleanup()
      });
    }

    // GPU usage alert
    if (metrics.gpuUsage && metrics.gpuUsage.utilization > settings.alertThresholds.highGPU) {
      newAlerts.push({
        id: `high-gpu-${Date.now()}`,
        type: 'warning',
        title: 'High GPU Usage',
        message: `GPU utilization at ${metrics.gpuUsage.utilization}%`,
        timestamp: new Date(),
        suggestion: 'Disable shadows or reduce anti-aliasing',
        action: () => optimizeGPUSettings()
      });
    }

    // Slow render alert
    if (metrics.frameTime > settings.alertThresholds.slowRender) {
      newAlerts.push({
        id: `slow-render-${Date.now()}`,
        type: 'info',
        title: 'Slow Rendering',
        message: `Frame time increased to ${metrics.frameTime.toFixed(2)}ms`,
        timestamp: new Date(),
        suggestion: 'Enable adaptive quality or reduce render complexity',
        action: () => enableAdaptiveQuality()
      });
    }

    if (newAlerts.length > 0) {
      setAlerts(prev => [...prev, ...newAlerts].slice(-10)); // Keep last 10 alerts
    }
  }, [settings.alertThresholds, onQualityAdjustment]);

  // Auto-optimization system
  const applyAutoOptimizations = useCallback((metrics: PerformanceMetrics) => {
    if (!settings.enableAdaptiveQuality) return;

    // Adaptive quality based on FPS
    if (metrics.fps < settings.targetFPS * 0.8) { // 80% of target
      onQualityAdjustment('low');
    } else if (metrics.fps > settings.targetFPS * 1.1 && metrics.memoryUsage.percentage < 60) {
      onQualityAdjustment('high');
    }

    // Memory management
    if (metrics.memoryUsage.percentage > 85) {
      triggerMemoryCleanup();
    }

  }, [settings, onQualityAdjustment]);

  // Optimization actions
  const triggerMemoryCleanup = useCallback(() => {
    // Trigger garbage collection hint
    if ('gc' in window) {
      (window as any).gc();
    }
    
    // Clear caches
    // This would integrate with actual caching systems
    console.log('Triggering memory cleanup...');
  }, []);

  const optimizeGPUSettings = useCallback(() => {
    // Automatically reduce GPU-intensive settings
    onQualityAdjustment('medium');
  }, [onQualityAdjustment]);

  const enableAdaptiveQuality = useCallback(() => {
    setSettings(prev => ({
      ...prev,
      enableAdaptiveQuality: true
    }));
  }, []);

  // Settings management
  const updateSettings = useCallback((updates: Partial<OptimizationSettings>) => {
    const newSettings = { ...settings, ...updates };
    setSettings(newSettings);
    onOptimizationChange(newSettings);
  }, [settings, onOptimizationChange]);

  const dismissAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);

  const clearAllAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  // Chart component for historical data
  const MetricsChart: React.FC<{ title: string; data: number[]; color: string; unit: string }> = ({ 
    title, data, color, unit 
  }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
      if (!canvasRef.current || data.length === 0) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d')!;
      const { width, height } = canvas;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw background grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw data line
      if (data.length > 1) {
        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const range = max - min || 1;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
          const x = (width / (data.length - 1)) * i;
          const y = height - ((data[i] - min) / range) * height;
          
          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        
        ctx.stroke();

        // Add current value indicator
        if (data.length > 0) {
          const lastValue = data[data.length - 1];
          const lastX = width - 10;
          const lastY = height - ((lastValue - min) / range) * height;
          
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
          ctx.fill();

          // Value label
          ctx.fillStyle = '#374151';
          ctx.font = '12px sans-serif';
          ctx.textAlign = 'right';
          ctx.fillText(`${lastValue.toFixed(1)}${unit}`, width - 15, lastY - 10);
        }
      }
    }, [data, color, unit]);

    return (
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-gray-700">{title}</h4>
        <canvas
          ref={canvasRef}
          width={280}
          height={120}
          className="border border-gray-200 rounded"
        />
      </div>
    );
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-4 right-4 w-96 bg-white rounded-lg shadow-xl border z-50">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center bg-gray-50 rounded-t-lg">
        <div className="flex items-center space-x-2">
          <h2 className="text-lg font-semibold text-gray-800">Performance Monitor</h2>
          {isRecording && (
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-red-600">Recording</span>
            </div>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-gray-600 hover:text-gray-800"
        >
          ✕
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b">
        {([
          { key: 'metrics', label: 'Metrics' },
          { key: 'charts', label: 'Charts' },
          { key: 'alerts', label: `Alerts${alerts.length > 0 ? ` (${alerts.length})` : ''}` },
          { key: 'optimization', label: 'Settings' }
        ] as const).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex-1 px-3 py-2 text-sm border-b-2 ${
              activeTab === key
                ? 'border-blue-500 text-blue-600 bg-blue-50'
                : 'border-transparent text-gray-600 hover:text-gray-800'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="max-h-96 overflow-auto">
        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="p-4 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{metrics.fps}</div>
                <div className="text-sm text-blue-800">FPS</div>
              </div>
              <div className="p-3 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{metrics.frameTime.toFixed(1)}</div>
                <div className="text-sm text-green-800">Frame Time (ms)</div>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm">
                  <span>Memory Usage</span>
                  <span>{metrics.memoryUsage.used}MB / {metrics.memoryUsage.total}MB</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div
                    className={`h-2 rounded-full ${
                      metrics.memoryUsage.percentage > 80 ? 'bg-red-500' :
                      metrics.memoryUsage.percentage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${metrics.memoryUsage.percentage}%` }}
                  />
                </div>
              </div>

              {metrics.gpuUsage && (
                <div>
                  <div className="flex justify-between text-sm">
                    <span>GPU Usage</span>
                    <span>{metrics.gpuUsage.utilization.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                    <div
                      className={`h-2 rounded-full ${
                        metrics.gpuUsage.utilization > 80 ? 'bg-red-500' :
                        metrics.gpuUsage.utilization > 60 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${metrics.gpuUsage.utilization}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="text-sm space-y-1">
                <div className="flex justify-between">
                  <span>Draw Calls:</span>
                  <span>{metrics.renderingStats.drawCalls}</span>
                </div>
                <div className="flex justify-between">
                  <span>Particles:</span>
                  <span>{metrics.renderingStats.particles.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Triangles:</span>
                  <span>{metrics.renderingStats.triangles.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Charts Tab */}
        {activeTab === 'charts' && (
          <div className="p-4 space-y-4">
            <MetricsChart
              title="FPS"
              data={historicalData.fps}
              color="#3b82f6"
              unit=""
            />
            <MetricsChart
              title="Memory Usage"
              data={historicalData.memory}
              color="#10b981"
              unit="%"
            />
            <MetricsChart
              title="Frame Time"
              data={historicalData.frameTime}
              color="#f59e0b"
              unit="ms"
            />
          </div>
        )}

        {/* Alerts Tab */}
        {activeTab === 'alerts' && (
          <div className="p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-medium">Performance Alerts</h3>
              {alerts.length > 0 && (
                <button
                  onClick={clearAllAlerts}
                  className="text-sm text-red-600 hover:text-red-800"
                >
                  Clear All
                </button>
              )}
            </div>

            {alerts.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <div className="text-2xl mb-2">✓</div>
                <div>No performance issues detected</div>
              </div>
            ) : (
              <div className="space-y-3">
                {alerts.map(alert => (
                  <div
                    key={alert.id}
                    className={`p-3 rounded-lg border ${
                      alert.type === 'error' ? 'bg-red-50 border-red-200' :
                      alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                      'bg-blue-50 border-blue-200'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="font-medium text-sm">{alert.title}</div>
                        <div className="text-xs text-gray-600 mt-1">{alert.message}</div>
                        {alert.suggestion && (
                          <div className="text-xs text-gray-500 mt-2">
                            💡 {alert.suggestion}
                          </div>
                        )}
                        {alert.action && (
                          <button
                            onClick={alert.action}
                            className="mt-2 text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                          >
                            Apply Fix
                          </button>
                        )}
                      </div>
                      <button
                        onClick={() => dismissAlert(alert.id)}
                        className="text-gray-400 hover:text-gray-600 ml-2"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="text-xs text-gray-400 mt-2">
                      {alert.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Optimization Tab */}
        {activeTab === 'optimization' && (
          <div className="p-4 space-y-4">
            <div>
              <h3 className="font-medium mb-3">Optimization Settings</h3>
              
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={settings.autoOptimize}
                    onChange={(e) => updateSettings({ autoOptimize: e.target.checked })}
                    className="rounded"
                  />
                  <span className="text-sm">Auto Optimize Performance</span>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={settings.enableAdaptiveQuality}
                    onChange={(e) => updateSettings({ enableAdaptiveQuality: e.target.checked })}
                    className="rounded"
                  />
                  <span className="text-sm">Adaptive Quality</span>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={settings.enableGPUOptimization}
                    onChange={(e) => updateSettings({ enableGPUOptimization: e.target.checked })}
                    className="rounded"
                  />
                  <span className="text-sm">GPU Optimization</span>
                </label>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Performance Targets</h4>
              <div className="space-y-3">
                <label className="block">
                  <span className="text-sm">Target FPS: {settings.targetFPS}</span>
                  <input
                    type="range"
                    min="30"
                    max="120"
                    step="10"
                    value={settings.targetFPS}
                    onChange={(e) => updateSettings({ targetFPS: parseInt(e.target.value) })}
                    className="w-full mt-1"
                  />
                </label>

                <label className="block">
                  <span className="text-sm">Max Memory: {settings.maxMemoryUsage}MB</span>
                  <input
                    type="range"
                    min="128"
                    max="2048"
                    step="128"
                    value={settings.maxMemoryUsage}
                    onChange={(e) => updateSettings({ maxMemoryUsage: parseInt(e.target.value) })}
                    className="w-full mt-1"
                  />
                </label>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Alert Thresholds</h4>
              <div className="space-y-3">
                <label className="block">
                  <span className="text-sm">Low FPS Warning: {settings.alertThresholds.lowFPS}</span>
                  <input
                    type="range"
                    min="15"
                    max="60"
                    step="5"
                    value={settings.alertThresholds.lowFPS}
                    onChange={(e) => updateSettings({
                      alertThresholds: { ...settings.alertThresholds, lowFPS: parseInt(e.target.value) }
                    })}
                    className="w-full mt-1"
                  />
                </label>

                <label className="block">
                  <span className="text-sm">High Memory Warning: {settings.alertThresholds.highMemory}%</span>
                  <input
                    type="range"
                    min="50"
                    max="95"
                    step="5"
                    value={settings.alertThresholds.highMemory}
                    onChange={(e) => updateSettings({
                      alertThresholds: { ...settings.alertThresholds, highMemory: parseInt(e.target.value) }
                    })}
                    className="w-full mt-1"
                  />
                </label>
              </div>
            </div>

            <div className="pt-3 border-t">
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => onQualityAdjustment('auto')}
                  className="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Auto Quality
                </button>
                <button
                  onClick={triggerMemoryCleanup}
                  className="px-3 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Clean Memory
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PerformanceMonitor;