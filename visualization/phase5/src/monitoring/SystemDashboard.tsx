/**
 * Phase 5 System Dashboard
 * 
 * Integrated dashboard combining all monitoring views with real-time updates,
 * comprehensive system health visualization, and LLMKG-specific cognitive
 * architecture monitoring in a unified interface.
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { RealTimeMonitor } from './RealTimeMonitor';
import { ComponentMonitor } from './ComponentMonitor';
import { AlertSystem, AlertRule, AlertHistory } from './AlertSystem';
import {
  HealthVisualization,
  SystemHealthOverview,
  ComponentHealthGrid,
  CognitivePatternVisualization,
  MemorySystemVisualization
} from './HealthVisualization';
import {
  PerformanceMetricsDashboard,
  PerformanceChart,
  MetricsCard
} from './PerformanceMetrics';
import {
  SystemHealthSummary,
  ComponentHealth,
  PerformanceMetrics,
  CognitivePatternActivation,
  BrainComponentHealth,
  MemorySystemMetrics,
  FederationNodeHealth,
  MCPToolHealth,
  SystemAlert,
  MonitoringConfig,
  AlertThresholds
} from '../types/MonitoringTypes';

// Dashboard Configuration
interface SystemDashboardConfig {
  refreshInterval: number;
  enableRealTimeUpdates: boolean;
  enableAlerts: boolean;
  enableCognitiveMonitoring: boolean;
  enableBrainComponentMonitoring: boolean;
  enableMemorySystemMonitoring: boolean;
  enableFederationMonitoring: boolean;
  enableMCPToolMonitoring: boolean;
  layout: 'grid' | 'tabs' | 'split';
  theme: 'light' | 'dark' | 'auto';
}

// Dashboard Layout Components
interface DashboardLayoutProps {
  children: React.ReactNode;
  layout: 'grid' | 'tabs' | 'split';
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children, layout }) => {
  switch (layout) {
    case 'grid':
      return <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">{children}</div>;
    case 'split':
      return <div className="flex flex-col lg:flex-row gap-6">{children}</div>;
    case 'tabs':
    default:
      return <div className="space-y-6">{children}</div>;
  }
};

// Alert Panel Component
interface AlertPanelProps {
  alerts: SystemAlert[];
  alertHistory: AlertHistory[];
  onAcknowledgeAlert: (alertId: string) => void;
  onResolveAlert: (alertId: string, resolution?: string) => void;
  compact?: boolean;
}

const AlertPanel: React.FC<AlertPanelProps> = ({
  alerts,
  alertHistory,
  onAcknowledgeAlert,
  onResolveAlert,
  compact = false
}) => {
  const [selectedAlert, setSelectedAlert] = useState<SystemAlert | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const severityColors = {
    info: 'bg-blue-50 text-blue-700 border-blue-200',
    warning: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    critical: 'bg-red-50 text-red-700 border-red-200',
    emergency: 'bg-purple-50 text-purple-700 border-purple-200'
  };

  const severityIcons = {
    info: 'ℹ️',
    warning: '⚠️',
    critical: '🚨',
    emergency: '🆘'
  };

  if (compact) {
    return (
      <div className="bg-white rounded-lg border shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900">Active Alerts</h3>
          <span className="text-sm text-gray-500">{alerts.length} active</span>
        </div>
        
        {alerts.length === 0 ? (
          <p className="text-sm text-gray-500 text-center py-4">No active alerts</p>
        ) : (
          <div className="space-y-2">
            {alerts.slice(0, 3).map(alert => (
              <div key={alert.id} className={`p-2 rounded border ${severityColors[alert.severity]}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span>{severityIcons[alert.severity]}</span>
                    <span className="text-sm font-medium">{alert.title}</span>
                  </div>
                  <span className="text-xs">{alert.componentId}</span>
                </div>
              </div>
            ))}
            {alerts.length > 3 && (
              <div className="text-center">
                <span className="text-sm text-gray-500">+{alerts.length - 3} more alerts</span>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">System Alerts</h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50"
            >
              {showHistory ? 'Current' : 'History'}
            </button>
            <span className="text-sm text-gray-500">
              {alerts.length} active
            </span>
          </div>
        </div>
      </div>

      <div className="p-6">
        {alerts.length === 0 && !showHistory ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">✅</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">All Systems Operational</h3>
            <p className="text-gray-500">No active alerts at this time</p>
          </div>
        ) : (
          <div className="space-y-4">
            {(showHistory ? alertHistory.slice(0, 10) : alerts).map(item => {
              const alert = 'alert' in item ? item.alert : item;
              const history = 'alert' in item ? item : null;
              
              return (
                <div
                  key={alert.id}
                  className={`border rounded-lg p-4 ${severityColors[alert.severity]} cursor-pointer hover:shadow-md transition-shadow`}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      <span className="text-2xl">{severityIcons[alert.severity]}</span>
                      <div>
                        <h3 className="font-semibold">{alert.title}</h3>
                        <p className="text-sm mt-1">{alert.message}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs">
                          <span>Component: {alert.componentId}</span>
                          <span>Time: {new Date(alert.timestamp).toLocaleString()}</span>
                          {alert.acknowledged && (
                            <span className="text-green-600">✓ Acknowledged</span>
                          )}
                          {history?.resolvedAt && (
                            <span className="text-blue-600">✓ Resolved</span>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    {!alert.acknowledged && !showHistory && (
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onAcknowledgeAlert(alert.id);
                          }}
                          className="px-3 py-1 text-xs bg-white border rounded hover:bg-gray-50"
                        >
                          Acknowledge
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onResolveAlert(alert.id, 'Manually resolved');
                          }}
                          className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
                        >
                          Resolve
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Alert Details</h2>
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className={`p-4 rounded border ${severityColors[selectedAlert.severity]} mb-4`}>
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-2xl">{severityIcons[selectedAlert.severity]}</span>
                  <span className="font-semibold text-lg">{selectedAlert.title}</span>
                </div>
                <p className="mb-3">{selectedAlert.message}</p>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Component:</span> {selectedAlert.componentId}
                  </div>
                  <div>
                    <span className="font-medium">Type:</span> {selectedAlert.componentType}
                  </div>
                  <div>
                    <span className="font-medium">Severity:</span> {selectedAlert.severity}
                  </div>
                  <div>
                    <span className="font-medium">Time:</span> {new Date(selectedAlert.timestamp).toLocaleString()}
                  </div>
                </div>
                
                {selectedAlert.metadata && Object.keys(selectedAlert.metadata).length > 0 && (
                  <div className="mt-3">
                    <span className="font-medium text-sm">Additional Info:</span>
                    <pre className="text-xs bg-gray-100 p-2 rounded mt-1 overflow-auto">
                      {JSON.stringify(selectedAlert.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>

              <div className="flex justify-end space-x-2">
                {!selectedAlert.acknowledged && (
                  <button
                    onClick={() => {
                      onAcknowledgeAlert(selectedAlert.id);
                      setSelectedAlert(null);
                    }}
                    className="px-4 py-2 border rounded hover:bg-gray-50"
                  >
                    Acknowledge
                  </button>
                )}
                <button
                  onClick={() => {
                    onResolveAlert(selectedAlert.id, 'Manually resolved from detail view');
                    setSelectedAlert(null);
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Resolve
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Main System Dashboard Component
interface SystemDashboardProps {
  config?: Partial<SystemDashboardConfig>;
  monitoringConfig?: MonitoringConfig;
  alertThresholds?: AlertThresholds;
  className?: string;
}

const SystemDashboard: React.FC<SystemDashboardProps> = ({
  config: userConfig = {},
  monitoringConfig,
  alertThresholds,
  className = ''
}) => {
  // Configuration with defaults
  const config: SystemDashboardConfig = useMemo(() => ({
    refreshInterval: 5000,
    enableRealTimeUpdates: true,
    enableAlerts: true,
    enableCognitiveMonitoring: true,
    enableBrainComponentMonitoring: true,
    enableMemorySystemMonitoring: true,
    enableFederationMonitoring: true,
    enableMCPToolMonitoring: true,
    layout: 'grid',
    theme: 'light',
    ...userConfig
  }), [userConfig]);

  // Core system state
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Monitoring data state
  const [systemHealth, setSystemHealth] = useState<SystemHealthSummary | null>(null);
  const [componentHealth, setComponentHealth] = useState<ComponentHealth[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<Map<string, PerformanceMetrics[]>>(new Map());
  const [cognitivePatterns, setCognitivePatterns] = useState<CognitivePatternActivation[]>([]);
  const [brainComponents, setBrainComponents] = useState<BrainComponentHealth[]>([]);
  const [memoryMetrics, setMemoryMetrics] = useState<MemorySystemMetrics | null>(null);
  const [federationHealth, setFederationHealth] = useState<FederationNodeHealth[]>([]);
  const [mcpToolHealth, setMcpToolHealth] = useState<MCPToolHealth[]>([]);

  // Alert system state
  const [activeAlerts, setActiveAlerts] = useState<SystemAlert[]>([]);
  const [alertHistory, setAlertHistory] = useState<AlertHistory[]>([]);

  // Core system instances
  const realTimeMonitor = useRef<RealTimeMonitor | null>(null);
  const alertSystem = useRef<AlertSystem | null>(null);

  // Dashboard UI state
  const [selectedView, setSelectedView] = useState<'overview' | 'components' | 'performance' | 'cognitive' | 'alerts'>('overview');
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);

  // Initialize monitoring systems
  useEffect(() => {
    const initializeSystems = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Initialize Real-Time Monitor
        if (config.enableRealTimeUpdates && monitoringConfig) {
          realTimeMonitor.current = new RealTimeMonitor();
          await realTimeMonitor.current.startMonitoring(monitoringConfig);

          // Set up event listeners
          realTimeMonitor.current.addEventListener('health_update', (health: ComponentHealth) => {
            setComponentHealth(prev => {
              const updated = prev.filter(h => h.componentId !== health.componentId);
              return [...updated, health];
            });
          });

          realTimeMonitor.current.addEventListener('performance_metrics', (metrics: PerformanceMetrics) => {
            setPerformanceMetrics(prev => {
              const updated = new Map(prev);
              const componentMetrics = updated.get(metrics.componentId) || [];
              updated.set(metrics.componentId, [...componentMetrics.slice(-99), metrics]);
              return updated;
            });
          });

          realTimeMonitor.current.addEventListener('cognitive_activation', (activation: CognitivePatternActivation) => {
            setCognitivePatterns(prev => [...prev.slice(-99), activation]);
          });

          realTimeMonitor.current.addEventListener('system_status', (status: SystemHealthSummary) => {
            setSystemHealth(status);
          });
        }

        // Initialize Alert System
        if (config.enableAlerts && alertThresholds) {
          alertSystem.current = new AlertSystem(alertThresholds);

          // Set up alert event listeners
          alertSystem.current.onAlert((alert) => {
            setActiveAlerts(prev => [...prev, alert]);
          });

          alertSystem.current.onResolve((alert) => {
            setActiveAlerts(prev => prev.filter(a => a.id !== alert.id));
          });

          alertSystem.current.onAcknowledge((alert) => {
            setActiveAlerts(prev => prev.map(a => a.id === alert.id ? { ...a, acknowledged: true } : a));
          });
        }

        // Load initial data
        await loadInitialData();

        setIsInitialized(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize monitoring systems');
        console.error('Dashboard initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initializeSystems();

    // Cleanup on unmount
    return () => {
      realTimeMonitor.current?.stopMonitoring();
      alertSystem.current?.cleanup();
    };
  }, [config, monitoringConfig, alertThresholds]);

  // Load initial data
  const loadInitialData = async (): Promise<void> => {
    if (realTimeMonitor.current) {
      try {
        const [
          healthSummary,
          cognitiveActivity,
          memorySystemMetrics,
          federationNodes
        ] = await Promise.all([
          realTimeMonitor.current.getSystemHealth(),
          realTimeMonitor.current.getCognitivePatternActivity(),
          realTimeMonitor.current.getMemorySystemMetrics().catch(() => null),
          realTimeMonitor.current.getFederationHealth()
        ]);

        setSystemHealth(healthSummary);
        setCognitivePatterns(cognitiveActivity);
        if (memorySystemMetrics) setMemoryMetrics(memorySystemMetrics);
        setFederationHealth(federationNodes);
      } catch (error) {
        console.warn('Failed to load some initial data:', error);
      }
    }
  };

  // Alert handling
  const handleAcknowledgeAlert = useCallback((alertId: string) => {
    alertSystem.current?.acknowledgeAlert(alertId, 'Dashboard User');
  }, []);

  const handleResolveAlert = useCallback((alertId: string, resolution?: string) => {
    alertSystem.current?.resolveAlert(alertId, resolution);
  }, []);

  // Component click handler
  const handleComponentClick = useCallback((componentId: string) => {
    setSelectedComponent(componentId);
    setSelectedView('components');
  }, []);

  // Render loading state
  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-lg font-medium text-gray-900">Initializing Monitoring Systems</h3>
          <p className="text-gray-500">Setting up real-time monitoring and alerts...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Monitoring System Error</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
          >
            Retry Initialization
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Dashboard Header */}
      <div className="bg-white rounded-lg border shadow-sm p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">LLMKG System Monitor</h1>
            <p className="text-gray-600">Real-time monitoring of cognitive architecture components</p>
          </div>
          
          {systemHealth && (
            <div className="flex items-center space-x-4">
              <div className="text-center">
                <p className="text-sm text-gray-500">System Health</p>
                <p className="text-2xl font-bold text-green-600">{systemHealth.healthScore}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">Active Alerts</p>
                <p className={`text-2xl font-bold ${activeAlerts.length > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {activeAlerts.length}
                </p>
              </div>
            </div>
          )}
        </div>
        
        {/* Navigation Tabs */}
        <div className="flex space-x-4 mt-6 border-b border-gray-200">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'components', label: 'Components', icon: '🔧' },
            { id: 'performance', label: 'Performance', icon: '⚡' },
            { id: 'cognitive', label: 'Cognitive', icon: '🧠' },
            { id: 'alerts', label: 'Alerts', icon: '🚨' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedView(tab.id as any)}
              className={`flex items-center space-x-2 px-4 py-2 border-b-2 font-medium text-sm ${
                selectedView === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Dashboard Content */}
      <DashboardLayout layout={config.layout}>
        {selectedView === 'overview' && systemHealth && (
          <>
            <SystemHealthOverview healthSummary={systemHealth} />
            {activeAlerts.length > 0 && (
              <AlertPanel
                alerts={activeAlerts}
                alertHistory={alertHistory}
                onAcknowledgeAlert={handleAcknowledgeAlert}
                onResolveAlert={handleResolveAlert}
                compact
              />
            )}
          </>
        )}

        {selectedView === 'components' && (
          <ComponentHealthGrid
            components={componentHealth}
            onComponentClick={handleComponentClick}
          />
        )}

        {selectedView === 'performance' && (
          <PerformanceMetricsDashboard
            componentMetrics={performanceMetrics}
            componentHealth={componentHealth}
            cognitivePatterns={cognitivePatterns}
            brainComponents={brainComponents}
            onComponentClick={handleComponentClick}
          />
        )}

        {selectedView === 'cognitive' && (
          <>
            <CognitivePatternVisualization
              patterns={cognitivePatterns}
              brainComponents={brainComponents}
            />
            {memoryMetrics && (
              <MemorySystemVisualization memoryMetrics={memoryMetrics} />
            )}
          </>
        )}

        {selectedView === 'alerts' && (
          <AlertPanel
            alerts={activeAlerts}
            alertHistory={alertHistory}
            onAcknowledgeAlert={handleAcknowledgeAlert}
            onResolveAlert={handleResolveAlert}
          />
        )}
      </DashboardLayout>

      {/* Status Bar */}
      <div className="bg-gray-100 rounded-lg p-4">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div className="flex items-center space-x-4">
            <span>🔄 Real-time updates: {config.enableRealTimeUpdates ? 'ON' : 'OFF'}</span>
            <span>🚨 Alerts: {config.enableAlerts ? 'ON' : 'OFF'}</span>
            <span>📊 Components: {componentHealth.length}</span>
          </div>
          <div>
            Last updated: {systemHealth ? new Date(systemHealth.lastUpdated).toLocaleString() : 'Never'}
          </div>
        </div>
      </div>
    </div>
  );
};

export {
  SystemDashboard,
  AlertPanel,
  DashboardLayout
};

export default SystemDashboard;