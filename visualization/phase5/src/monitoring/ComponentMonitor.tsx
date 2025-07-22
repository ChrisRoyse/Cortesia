/**
 * Phase 5 Component Monitor
 * 
 * Individual component monitoring with health, performance, and status tracking.
 * Provides specialized monitoring for LLMKG brain-inspired components including
 * cognitive patterns, neural bridges, and MCP tools.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ComponentHealth,
  PerformanceMetrics,
  BrainComponentHealth,
  CognitivePatternActivation,
  MCPToolHealth,
  MemorySystemMetrics,
  LLMKGComponentType,
  ComponentStatus
} from '../types/MonitoringTypes';

// Component Monitor Core Class
export class ComponentMonitor {
  private componentId: string;
  private componentType: LLMKGComponentType;
  private health: ComponentHealth | null = null;
  private performanceHistory: PerformanceMetrics[] = [];
  private brainHealth: BrainComponentHealth | null = null;
  private cognitiveActivations: CognitivePatternActivation[] = [];
  private mcpToolHealth: MCPToolHealth | null = null;
  private updateCallbacks: Set<(data: any) => void> = new Set();
  private maxHistorySize = 100;

  constructor(componentId: string, componentType: LLMKGComponentType) {
    this.componentId = componentId;
    this.componentType = componentType;
  }

  // Core monitoring methods
  public updateHealth(health: ComponentHealth): void {
    this.health = health;
    this.notifyCallbacks('health', health);
  }

  public updatePerformanceMetrics(metrics: PerformanceMetrics): void {
    this.performanceHistory.push(metrics);
    
    // Maintain history size
    if (this.performanceHistory.length > this.maxHistorySize) {
      this.performanceHistory.shift();
    }
    
    this.notifyCallbacks('performance', metrics);
  }

  public updateBrainHealth(brainHealth: BrainComponentHealth): void {
    this.brainHealth = brainHealth;
    this.notifyCallbacks('brain_health', brainHealth);
  }

  public updateCognitiveActivation(activation: CognitivePatternActivation): void {
    this.cognitiveActivations.push(activation);
    
    // Keep only recent activations (last hour)
    const cutoff = Date.now() - (60 * 60 * 1000);
    this.cognitiveActivations = this.cognitiveActivations.filter(a => a.timestamp > cutoff);
    
    this.notifyCallbacks('cognitive_activation', activation);
  }

  public updateMCPToolHealth(toolHealth: MCPToolHealth): void {
    this.mcpToolHealth = toolHealth;
    this.notifyCallbacks('mcp_tool_health', toolHealth);
  }

  // Health scoring and analysis
  public getHealthScore(): number {
    if (!this.health) return 0;
    return this.health.healthScore;
  }

  public getHealthTrend(): 'improving' | 'stable' | 'degrading' {
    if (!this.health) return 'stable';
    return this.health.trend;
  }

  public getCurrentStatus(): ComponentStatus {
    if (!this.health) return 'offline';
    return this.health.status;
  }

  // Performance analysis
  public getAverageCPUUsage(minutes: number = 5): number {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentMetrics = this.performanceHistory.filter(m => m.timestamp > cutoff);
    
    if (recentMetrics.length === 0) return 0;
    
    return recentMetrics.reduce((sum, m) => sum + m.cpu, 0) / recentMetrics.length;
  }

  public getAverageMemoryUsage(minutes: number = 5): number {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentMetrics = this.performanceHistory.filter(m => m.timestamp > cutoff);
    
    if (recentMetrics.length === 0) return 0;
    
    return recentMetrics.reduce((sum, m) => sum + m.memory, 0) / recentMetrics.length;
  }

  public getAverageLatency(minutes: number = 5): number {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentMetrics = this.performanceHistory.filter(m => m.timestamp > cutoff);
    
    if (recentMetrics.length === 0) return 0;
    
    return recentMetrics.reduce((sum, m) => sum + m.latency, 0) / recentMetrics.length;
  }

  public getThroughput(): number {
    const latest = this.performanceHistory[this.performanceHistory.length - 1];
    return latest ? latest.throughput : 0;
  }

  // Brain-specific analysis
  public getNeuralActivityLevel(): number {
    if (!this.brainHealth) return 0;
    return this.brainHealth.neuralActivityLevel;
  }

  public getSynapticStrength(): number {
    if (!this.brainHealth) return 0;
    return this.brainHealth.synapticStrength;
  }

  public getInhibitionExcitationBalance(): { inhibition: number; excitation: number } {
    if (!this.brainHealth) return { inhibition: 0, excitation: 0 };
    return {
      inhibition: this.brainHealth.inhibitionBalance,
      excitation: this.brainHealth.excitationBalance
    };
  }

  public getPlasticityScore(): number {
    if (!this.brainHealth) return 0;
    return this.brainHealth.plasticityScore;
  }

  // Cognitive pattern analysis
  public getActiveCognitivePatterns(): CognitivePatternActivation[] {
    const activeThreshold = Date.now() - (5 * 60 * 1000); // Last 5 minutes
    return this.cognitiveActivations.filter(a => a.timestamp > activeThreshold);
  }

  public getCognitivePatternFrequency(patternType: string): number {
    const lastHour = Date.now() - (60 * 60 * 1000);
    const patternCount = this.cognitiveActivations.filter(
      a => a.patternType === patternType && a.timestamp > lastHour
    ).length;
    
    return patternCount;
  }

  // MCP Tool specific analysis
  public getMCPToolAvailability(): boolean {
    return this.mcpToolHealth ? this.mcpToolHealth.isAvailable : false;
  }

  public getMCPToolSuccessRate(): number {
    return this.mcpToolHealth ? this.mcpToolHealth.successRate : 0;
  }

  public getMCPToolResponseTime(): number {
    return this.mcpToolHealth ? this.mcpToolHealth.responseTime : 0;
  }

  // Event subscription
  public onUpdate(callback: (type: string, data: any) => void): () => void {
    this.updateCallbacks.add(callback);
    return () => this.updateCallbacks.delete(callback);
  }

  private notifyCallbacks(type: string, data: any): void {
    this.updateCallbacks.forEach(callback => callback(type, data));
  }

  // Data export
  public exportData(): ComponentMonitorData {
    return {
      componentId: this.componentId,
      componentType: this.componentType,
      health: this.health,
      performanceHistory: this.performanceHistory,
      brainHealth: this.brainHealth,
      cognitiveActivations: this.cognitiveActivations,
      mcpToolHealth: this.mcpToolHealth,
      exportTimestamp: Date.now()
    };
  }
}

// React Component for Component Monitor UI
interface ComponentMonitorProps {
  componentId: string;
  componentType: LLMKGComponentType;
  monitor: ComponentMonitor;
  compact?: boolean;
  showBrainMetrics?: boolean;
  showCognitivePatterns?: boolean;
  className?: string;
  onComponentClick?: (componentId: string) => void;
}

export const ComponentMonitorComponent: React.FC<ComponentMonitorProps> = ({
  componentId,
  componentType,
  monitor,
  compact = false,
  showBrainMetrics = true,
  showCognitivePatterns = true,
  className = '',
  onComponentClick
}) => {
  const [health, setHealth] = useState<ComponentHealth | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [brainHealth, setBrainHealth] = useState<BrainComponentHealth | null>(null);
  const [cognitiveActivations, setCognitiveActivations] = useState<CognitivePatternActivation[]>([]);
  const [mcpToolHealth, setMcpToolHealth] = useState<MCPToolHealth | null>(null);

  // Subscribe to monitor updates
  useEffect(() => {
    const unsubscribe = monitor.onUpdate((type: string, data: any) => {
      switch (type) {
        case 'health':
          setHealth(data);
          break;
        case 'performance':
          setPerformanceMetrics(data);
          break;
        case 'brain_health':
          setBrainHealth(data);
          break;
        case 'cognitive_activation':
          setCognitiveActivations(monitor.getActiveCognitivePatterns());
          break;
        case 'mcp_tool_health':
          setMcpToolHealth(data);
          break;
      }
    });

    return unsubscribe;
  }, [monitor]);

  // Calculate derived metrics
  const averageCPU = useMemo(() => monitor.getAverageCPUUsage(5), [monitor, performanceMetrics]);
  const averageMemory = useMemo(() => monitor.getAverageMemoryUsage(5), [monitor, performanceMetrics]);
  const averageLatency = useMemo(() => monitor.getAverageLatency(5), [monitor, performanceMetrics]);
  const throughput = useMemo(() => monitor.getThroughput(), [monitor, performanceMetrics]);

  // Status color mapping
  const getStatusColor = (status: ComponentStatus): string => {
    const colors = {
      active: 'text-green-600 bg-green-50',
      idle: 'text-blue-600 bg-blue-50',
      processing: 'text-yellow-600 bg-yellow-50',
      error: 'text-red-600 bg-red-50',
      degraded: 'text-orange-600 bg-orange-50',
      offline: 'text-gray-600 bg-gray-50'
    };
    return colors[status] || colors.offline;
  };

  // Health score color
  const getHealthScoreColor = (score: number): string => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    if (score >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  if (compact) {
    return (
      <div 
        className={`bg-white rounded-lg border shadow-sm p-4 cursor-pointer hover:shadow-md transition-shadow ${className}`}
        onClick={() => onComponentClick?.(componentId)}
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium text-gray-900">{componentId}</h3>
            <p className="text-sm text-gray-500">{componentType}</p>
          </div>
          
          {health && (
            <div className="flex items-center space-x-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(health.status)}`}>
                {health.status}
              </span>
              <span className={`text-lg font-semibold ${getHealthScoreColor(health.healthScore)}`}>
                {health.healthScore}
              </span>
            </div>
          )}
        </div>
        
        {performanceMetrics && (
          <div className="mt-2 grid grid-cols-4 gap-2 text-sm">
            <div className="text-center">
              <p className="text-gray-500">CPU</p>
              <p className="font-medium">{averageCPU.toFixed(1)}%</p>
            </div>
            <div className="text-center">
              <p className="text-gray-500">Memory</p>
              <p className="font-medium">{averageMemory.toFixed(1)}%</p>
            </div>
            <div className="text-center">
              <p className="text-gray-500">Latency</p>
              <p className="font-medium">{averageLatency.toFixed(1)}ms</p>
            </div>
            <div className="text-center">
              <p className="text-gray-500">Throughput</p>
              <p className="font-medium">{throughput.toFixed(0)}/s</p>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg border shadow-sm ${className}`}>
      {/* Component Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{componentId}</h2>
            <p className="text-gray-600">{componentType.replace(/_/g, ' ').toUpperCase()}</p>
          </div>
          
          {health && (
            <div className="flex items-center space-x-4">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(health.status)}`}>
                {health.status.toUpperCase()}
              </span>
              
              <div className="text-center">
                <p className="text-sm text-gray-500">Health Score</p>
                <p className={`text-2xl font-bold ${getHealthScoreColor(health.healthScore)}`}>
                  {health.healthScore}
                </p>
              </div>
              
              <div className="text-center">
                <p className="text-sm text-gray-500">Trend</p>
                <p className={`text-sm font-medium ${
                  health.trend === 'improving' ? 'text-green-600' :
                  health.trend === 'degrading' ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {health.trend.toUpperCase()}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Performance Metrics */}
      {performanceMetrics && (
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">CPU Usage</p>
              <p className="text-2xl font-bold text-gray-900">{averageCPU.toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full" 
                  style={{ width: `${Math.min(averageCPU, 100)}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Memory Usage</p>
              <p className="text-2xl font-bold text-gray-900">{averageMemory.toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-green-600 h-2 rounded-full" 
                  style={{ width: `${Math.min(averageMemory, 100)}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Avg Latency</p>
              <p className="text-2xl font-bold text-gray-900">{averageLatency.toFixed(1)}</p>
              <p className="text-xs text-gray-500">milliseconds</p>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Throughput</p>
              <p className="text-2xl font-bold text-gray-900">{throughput.toFixed(0)}</p>
              <p className="text-xs text-gray-500">requests/sec</p>
            </div>
          </div>
        </div>
      )}

      {/* Brain-Specific Metrics */}
      {showBrainMetrics && brainHealth && (
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Brain Component Health</h3>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Neural Activity</p>
              <p className="text-2xl font-bold text-gray-900">{(brainHealth.neuralActivityLevel * 100).toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full" 
                  style={{ width: `${brainHealth.neuralActivityLevel * 100}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Synaptic Strength</p>
              <p className="text-2xl font-bold text-gray-900">{(brainHealth.synapticStrength * 100).toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-indigo-600 h-2 rounded-full" 
                  style={{ width: `${brainHealth.synapticStrength * 100}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Plasticity Score</p>
              <p className="text-2xl font-bold text-gray-900">{(brainHealth.plasticityScore * 100).toFixed(1)}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-pink-600 h-2 rounded-full" 
                  style={{ width: `${brainHealth.plasticityScore * 100}%` }}
                />
              </div>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Inhibition/Excitation</p>
              <div className="flex justify-center space-x-2">
                <span className="text-sm text-red-600">I: {brainHealth.inhibitionBalance.toFixed(2)}</span>
                <span className="text-sm text-blue-600">E: {brainHealth.excitationBalance.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cognitive Patterns */}
      {showCognitivePatterns && cognitiveActivations.length > 0 && (
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Cognitive Patterns</h3>
          
          <div className="space-y-3">
            {cognitiveActivations.slice(0, 5).map((activation, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{activation.patternType}</p>
                  <p className="text-sm text-gray-500">
                    Activation Level: {(activation.activationLevel * 100).toFixed(1)}% | 
                    Confidence: {(activation.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">
                    {new Date(activation.timestamp).toLocaleTimeString()}
                  </p>
                  <p className="text-sm text-gray-500">
                    Duration: {activation.duration}ms
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MCP Tool Health */}
      {mcpToolHealth && (
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">MCP Tool Health</h3>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Availability</p>
              <p className={`text-lg font-bold ${mcpToolHealth.isAvailable ? 'text-green-600' : 'text-red-600'}`}>
                {mcpToolHealth.isAvailable ? 'Online' : 'Offline'}
              </p>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Success Rate</p>
              <p className="text-lg font-bold text-gray-900">{(mcpToolHealth.successRate * 100).toFixed(1)}%</p>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Response Time</p>
              <p className="text-lg font-bold text-gray-900">{mcpToolHealth.responseTime.toFixed(1)}ms</p>
            </div>
            
            <div className="text-center">
              <p className="text-sm font-medium text-gray-500">Usage Frequency</p>
              <p className="text-lg font-bold text-gray-900">{mcpToolHealth.usageFrequency}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Data type for component monitor export
interface ComponentMonitorData {
  componentId: string;
  componentType: LLMKGComponentType;
  health: ComponentHealth | null;
  performanceHistory: PerformanceMetrics[];
  brainHealth: BrainComponentHealth | null;
  cognitiveActivations: CognitivePatternActivation[];
  mcpToolHealth: MCPToolHealth | null;
  exportTimestamp: number;
}

export default ComponentMonitor;