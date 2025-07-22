import React, { useState, useMemo } from 'react';
import {
  CubeTransparentIcon,
  ServerIcon,
  CircuitBoardIcon,
  DatabaseIcon,
  CloudIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon,
  ChartBarIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { useSystemArchitecture } from '../../hooks/useSystemArchitecture';
import { StatusIndicator } from '../../components/common/StatusIndicator';
import { MetricCard } from '../../components/common/MetricCard';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { ProgressBar } from '../../components/common/ProgressBar';
import { Badge } from '../../components/common/Badge';
import { NetworkTopology } from '../../components/Architecture/NetworkTopology';
import { ComponentDependencyGraph } from '../../components/Architecture/ComponentDependencyGraph';
import { SystemHealthMatrix } from '../../components/Architecture/SystemHealthMatrix';

interface ComponentViewMode {
  id: 'overview' | 'topology' | 'dependencies' | 'health';
  name: string;
  icon: React.ComponentType<any>;
}

const viewModes: ComponentViewMode[] = [
  { id: 'overview', name: 'Overview', icon: CubeTransparentIcon },
  { id: 'topology', name: 'Network Topology', icon: CloudIcon },
  { id: 'dependencies', name: 'Dependencies', icon: CircuitBoardIcon },
  { id: 'health', name: 'Health Matrix', icon: ChartBarIcon },
];

const ArchitecturePage: React.FC = () => {
  const [selectedView, setSelectedView] = useState<ComponentViewMode['id']>('overview');
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  
  const {
    components,
    systemMetrics,
    dependencies,
    healthChecks,
    networkTopology,
    isLoading,
    error,
    refreshArchitecture,
  } = useSystemArchitecture();

  const criticalComponents = useMemo(() => {
    return components.filter(c => c.criticality === 'critical');
  }, [components]);

  const componentsByStatus = useMemo(() => {
    return {
      healthy: components.filter(c => c.status === 'healthy').length,
      warning: components.filter(c => c.status === 'warning').length,
      critical: components.filter(c => c.status === 'critical').length,
      offline: components.filter(c => c.status === 'offline').length,
    };
  }, [components]);

  const getComponentIcon = (type: string) => {
    switch (type) {
      case 'cognitive':
        return CircuitBoardIcon;
      case 'neural':
        return CircuitBoardIcon;
      case 'memory':
        return DatabaseIcon;
      case 'knowledge':
        return CubeTransparentIcon;
      case 'api':
        return CloudIcon;
      case 'security':
        return ShieldCheckIcon;
      default:
        return ServerIcon;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'critical':
        return 'text-red-600 bg-red-100';
      case 'offline':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const renderOverview = () => (
    <div className="space-y-8">
      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Components"
          value={components.length.toString()}
          icon={CubeTransparentIcon}
          className="bg-gradient-to-r from-blue-500 to-blue-600"
        />
        <MetricCard
          title="Healthy Components"
          value={componentsByStatus.healthy.toString()}
          change={`${Math.round((componentsByStatus.healthy / components.length) * 100)}%`}
          changeType="neutral"
          icon={CheckCircleIcon}
          className="bg-gradient-to-r from-green-500 to-green-600"
        />
        <MetricCard
          title="System Uptime"
          value={`${systemMetrics.uptime.toFixed(1)}%`}
          icon={ClockIcon}
          className="bg-gradient-to-r from-purple-500 to-purple-600"
        />
        <MetricCard
          title="Response Time"
          value={`${systemMetrics.avgResponseTime}ms`}
          change={systemMetrics.responseTimeChange > 0 ? `+${systemMetrics.responseTimeChange}%` : `${systemMetrics.responseTimeChange}%`}
          changeType={systemMetrics.responseTimeChange > 0 ? 'increase' : 'decrease'}
          icon={ArrowPathIcon}
          className="bg-gradient-to-r from-orange-500 to-orange-600"
        />
      </div>

      {/* Component Grid */}
      <div className="bg-white rounded-lg shadow-sm">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">System Components</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {components.map((component) => {
              const IconComponent = getComponentIcon(component.type);
              return (
                <div
                  key={component.id}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => setSelectedComponent(component.id)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${getStatusColor(component.status)}`}>
                        <IconComponent className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          {component.name}
                        </h3>
                        <p className="text-sm text-gray-600">{component.version}</p>
                      </div>
                    </div>
                    <StatusIndicator status={component.status as any} />
                  </div>

                  <p className="text-sm text-gray-600 mb-3">
                    {component.description}
                  </p>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">CPU Usage</span>
                      <span className="font-medium">{component.metrics.cpu}%</span>
                    </div>
                    <ProgressBar 
                      value={component.metrics.cpu} 
                      max={100}
                      color={component.metrics.cpu > 80 ? 'red' : component.metrics.cpu > 60 ? 'yellow' : 'green'}
                      size="sm"
                    />

                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-500">Memory</span>
                      <span className="font-medium">{component.metrics.memory}%</span>
                    </div>
                    <ProgressBar 
                      value={component.metrics.memory} 
                      max={100}
                      color={component.metrics.memory > 80 ? 'red' : component.metrics.memory > 60 ? 'yellow' : 'green'}
                      size="sm"
                    />
                  </div>

                  <div className="mt-3 flex items-center justify-between">
                    <div className="flex space-x-1">
                      <Badge 
                        variant={component.criticality === 'critical' ? 'destructive' : 'secondary'}
                        className="text-xs"
                      >
                        {component.criticality}
                      </Badge>
                    </div>
                    <span className="text-xs text-gray-500">
                      Updated {new Date(component.lastUpdated).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Critical Components Alert */}
      {componentsByStatus.critical > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-start">
            <ExclamationTriangleIcon className="w-6 h-6 text-red-500 mr-3 mt-0.5" />
            <div>
              <h3 className="text-lg font-medium text-red-900 mb-2">
                Critical Components Detected
              </h3>
              <p className="text-red-800 mb-4">
                {componentsByStatus.critical} component{componentsByStatus.critical !== 1 ? 's' : ''} require immediate attention.
              </p>
              <div className="space-y-2">
                {components
                  .filter(c => c.status === 'critical')
                  .map(component => (
                    <div key={component.id} className="flex items-center space-x-2">
                      <XCircleIcon className="w-4 h-4 text-red-500" />
                      <span className="text-red-900 font-medium">{component.name}</span>
                      <span className="text-red-700 text-sm">- {component.statusMessage}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderContent = () => {
    switch (selectedView) {
      case 'topology':
        return (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Network Topology</h2>
            <NetworkTopology topology={networkTopology} />
          </div>
        );
      case 'dependencies':
        return (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Component Dependencies</h2>
            <ComponentDependencyGraph 
              components={components} 
              dependencies={dependencies}
              onComponentSelect={setSelectedComponent}
            />
          </div>
        );
      case 'health':
        return (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">System Health Matrix</h2>
            <SystemHealthMatrix 
              components={components}
              healthChecks={healthChecks}
            />
          </div>
        );
      default:
        return renderOverview();
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <LoadingSpinner size="large" message="Loading system architecture..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <XCircleIcon className="w-6 h-6 text-red-500 mr-3" />
            <div>
              <h3 className="text-lg font-medium text-red-900">Error Loading Architecture</h3>
              <p className="text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button
            onClick={refreshArchitecture}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">System Architecture</h1>
            <p className="mt-2 text-gray-600">
              Monitor system components, dependencies, and overall health
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={refreshArchitecture}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
            >
              <ArrowPathIcon className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </div>

      {/* View Mode Tabs */}
      <div className="mb-8">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8">
            {viewModes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setSelectedView(mode.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  selectedView === mode.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <mode.icon className="w-5 h-5" />
                <span>{mode.name}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      {renderContent()}

      {/* Component Detail Modal */}
      {selectedComponent && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity bg-gray-500 bg-opacity-75" onClick={() => setSelectedComponent(null)} />
            
            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              {/* Component detail modal content would go here */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Component Details
                </h3>
                <p className="text-gray-600">
                  Detailed information about component {selectedComponent}
                </p>
                <div className="mt-6 flex justify-end">
                  <button
                    onClick={() => setSelectedComponent(null)}
                    className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ArchitecturePage;