import React, { useState, useMemo } from 'react';
import { useSpring, animated } from '@react-spring/web';
import {
  ComponentDetailsProps,
  ArchitectureNode,
  ConnectionEdge,
  ComponentStatus,
  ThemeConfiguration
} from '../types';

const ComponentDetails: React.FC<ComponentDetailsProps> = ({
  node,
  connections = [],
  theme,
  onClose,
  onNavigateToComponent
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'connections' | 'history'>('overview');

  // Animation for panel entrance
  const panelSpring = useSpring({
    opacity: 1,
    transform: 'translateX(0%)',
    from: { opacity: 0, transform: 'translateX(100%)' },
    config: { tension: 300, friction: 30 }
  });

  // Calculate connection statistics
  const connectionStats = useMemo(() => {
    const incoming = connections.filter(conn => conn.targetId === node?.id);
    const outgoing = connections.filter(conn => conn.sourceId === node?.id);
    const bidirectional = connections.filter(conn => conn.type === 'bidirectional');
    
    return {
      total: connections.length,
      incoming: incoming.length,
      outgoing: outgoing.length,
      bidirectional: bidirectional.length,
      active: connections.filter(conn => conn.active).length,
      types: {
        excitation: connections.filter(conn => conn.type === 'excitation').length,
        inhibition: connections.filter(conn => conn.type === 'inhibition').length,
        dataFlow: connections.filter(conn => conn.type === 'data-flow').length,
        dependency: connections.filter(conn => conn.type === 'dependency').length,
      }
    };
  }, [connections, node?.id]);

  if (!node) {
    return null;
  }

  return (
    <animated.div
      style={panelSpring}
      className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-50 overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <ComponentIcon type={node.type} size={24} />
          <div>
            <h2 className="text-lg font-semibold truncate">{node.label}</h2>
            <p className="text-blue-100 text-sm">{getNodeTypeLabel(node.type)}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <StatusBadge status={node.status} />
          <button
            onClick={onClose}
            className="text-white hover:text-blue-200 text-xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-blue-600 transition-colors"
          >
            ×
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-gray-50 border-b">
        <div className="flex">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'metrics', label: 'Metrics', icon: '📈' },
            { id: 'connections', label: 'Connections', icon: '🔗', count: connectionStats.total },
            { id: 'history', label: 'History', icon: '📋' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              <span className="mr-1">{tab.icon}</span>
              {tab.label}
              {tab.count !== undefined && (
                <span className="ml-1 bg-gray-500 text-white text-xs rounded-full px-2 py-0.5">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'overview' && (
          <OverviewTab node={node} theme={theme} onNavigateToComponent={onNavigateToComponent} />
        )}
        {activeTab === 'metrics' && (
          <MetricsTab node={node} theme={theme} />
        )}
        {activeTab === 'connections' && (
          <ConnectionsTab 
            node={node} 
            connections={connections} 
            connectionStats={connectionStats}
            theme={theme} 
            onNavigateToComponent={onNavigateToComponent} 
          />
        )}
        {activeTab === 'history' && (
          <HistoryTab node={node} theme={theme} />
        )}
      </div>
    </animated.div>
  );
};

// Overview Tab Component
const OverviewTab: React.FC<{
  node: ArchitectureNode;
  theme: ThemeConfiguration;
  onNavigateToComponent?: (componentId: string) => void;
}> = ({ node, theme, onNavigateToComponent }) => {
  return (
    <div className="space-y-6">
      {/* Basic Information */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Component Information</h3>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Type</div>
              <div className="font-medium">{getNodeTypeLabel(node.type)}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Layer</div>
              <div className="font-medium">{node.layer}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Importance</div>
              <div className="font-medium">{(node.importance * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <div className="text-sm text-gray-600">Status</div>
              <StatusBadge status={node.status} showLabel />
            </div>
          </div>
          
          {node.description && (
            <div className="bg-blue-50 border-l-4 border-blue-400 p-4">
              <div className="text-sm text-gray-600 mb-1">Description</div>
              <div className="text-gray-800">{node.description}</div>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Quick Actions</h3>
        <div className="grid grid-cols-2 gap-2">
          {node.type === 'mcp-tool' && (
            <button className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors">
              Launch Tool
            </button>
          )}
          <button 
            onClick={() => onNavigateToComponent?.(node.id)}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
          >
            Focus on Map
          </button>
          <button className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition-colors">
            View Logs
          </button>
          <button className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors">
            Debug
          </button>
        </div>
      </div>

      {/* Metadata */}
      {node.metadata && Object.keys(node.metadata).length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Metadata</h3>
          <div className="bg-gray-50 p-3 rounded font-mono text-sm">
            <pre>{JSON.stringify(node.metadata, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

// Metrics Tab Component
const MetricsTab: React.FC<{
  node: ArchitectureNode;
  theme: ThemeConfiguration;
}> = ({ node, theme }) => {
  if (!node.metrics) {
    return (
      <div className="text-center py-8 text-gray-500">
        <div className="text-4xl mb-4">📊</div>
        <p>No metrics available for this component</p>
      </div>
    );
  }

  const metrics = node.metrics;

  return (
    <div className="space-y-6">
      {/* Current Metrics */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Current Performance</h3>
        <div className="space-y-4">
          <MetricCard
            label="CPU Usage"
            current={metrics.cpu.current}
            average={metrics.cpu.average}
            peak={metrics.cpu.peak}
            unit="%"
            color={getCPUColor(metrics.cpu.current)}
          />
          <MetricCard
            label="Memory Usage"
            current={metrics.memory.current}
            average={metrics.memory.average}
            peak={metrics.memory.peak}
            unit="%"
            color={getMemoryColor(metrics.memory.current)}
          />
          <MetricCard
            label="Throughput"
            current={metrics.throughput.current}
            average={metrics.throughput.average}
            peak={metrics.throughput.peak}
            unit="ops/s"
            color={getThroughputColor(metrics.throughput.current)}
          />
          <MetricCard
            label="Latency"
            current={metrics.latency.current}
            average={metrics.latency.average}
            peak={metrics.latency.peak}
            unit="ms"
            color={getLatencyColor(metrics.latency.current)}
          />
          <MetricCard
            label="Error Rate"
            current={metrics.errorRate.current}
            average={metrics.errorRate.average}
            peak={metrics.errorRate.peak}
            unit="%"
            color={getErrorRateColor(metrics.errorRate.current)}
          />
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-sm text-gray-500">
        Last updated: {new Date(metrics.lastUpdated).toLocaleString()}
      </div>
    </div>
  );
};

// Connections Tab Component
const ConnectionsTab: React.FC<{
  node: ArchitectureNode;
  connections: ConnectionEdge[];
  connectionStats: any;
  theme: ThemeConfiguration;
  onNavigateToComponent?: (componentId: string) => void;
}> = ({ node, connections, connectionStats, theme, onNavigateToComponent }) => {
  const [filter, setFilter] = useState<'all' | 'incoming' | 'outgoing' | 'active'>('all');

  const filteredConnections = useMemo(() => {
    switch (filter) {
      case 'incoming':
        return connections.filter(conn => conn.targetId === node.id);
      case 'outgoing':
        return connections.filter(conn => conn.sourceId === node.id);
      case 'active':
        return connections.filter(conn => conn.active);
      default:
        return connections;
    }
  }, [connections, node.id, filter]);

  return (
    <div className="space-y-6">
      {/* Connection Statistics */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Connection Overview</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 p-3 rounded">
            <div className="text-2xl font-bold text-blue-600">{connectionStats.total}</div>
            <div className="text-sm text-blue-600">Total Connections</div>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <div className="text-2xl font-bold text-green-600">{connectionStats.active}</div>
            <div className="text-sm text-green-600">Active</div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="text-2xl font-bold text-purple-600">{connectionStats.incoming}</div>
            <div className="text-sm text-purple-600">Incoming</div>
          </div>
          <div className="bg-orange-50 p-3 rounded">
            <div className="text-2xl font-bold text-orange-600">{connectionStats.outgoing}</div>
            <div className="text-sm text-orange-600">Outgoing</div>
          </div>
        </div>
      </div>

      {/* Connection Types */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Connection Types</h3>
        <div className="space-y-2">
          {Object.entries(connectionStats.types).map(([type, count]) => (
            <div key={type} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div className="flex items-center space-x-2">
                <ConnectionTypeIcon type={type} />
                <span className="capitalize">{type.replace(/([A-Z])/g, ' $1').toLowerCase()}</span>
              </div>
              <span className="font-medium">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Filter Controls */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Connection Details</h3>
        <div className="flex space-x-2 mb-4">
          {[
            { key: 'all', label: 'All', count: connectionStats.total },
            { key: 'incoming', label: 'Incoming', count: connectionStats.incoming },
            { key: 'outgoing', label: 'Outgoing', count: connectionStats.outgoing },
            { key: 'active', label: 'Active', count: connectionStats.active },
          ].map(option => (
            <button
              key={option.key}
              onClick={() => setFilter(option.key as any)}
              className={`px-3 py-1 text-sm rounded-full ${
                filter === option.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {option.label} ({option.count})
            </button>
          ))}
        </div>

        {/* Connection List */}
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {filteredConnections.map(connection => (
            <ConnectionItem
              key={connection.id}
              connection={connection}
              currentNodeId={node.id}
              theme={theme}
              onNavigateToComponent={onNavigateToComponent}
            />
          ))}
          {filteredConnections.length === 0 && (
            <div className="text-center py-4 text-gray-500">
              No connections match the selected filter
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// History Tab Component
const HistoryTab: React.FC<{
  node: ArchitectureNode;
  theme: ThemeConfiguration;
}> = ({ node, theme }) => {
  // Mock history data - in a real implementation, this would come from telemetry
  const historyEvents = [
    { timestamp: Date.now() - 300000, type: 'status_change', message: 'Status changed to healthy' },
    { timestamp: Date.now() - 600000, type: 'metric_alert', message: 'CPU usage exceeded 80%' },
    { timestamp: Date.now() - 900000, type: 'connection_added', message: 'New connection established' },
    { timestamp: Date.now() - 1200000, type: 'initialization', message: 'Component initialized' },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Recent Activity</h3>
      <div className="space-y-3">
        {historyEvents.map((event, index) => (
          <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded">
            <div className="flex-shrink-0 mt-1">
              <EventIcon type={event.type} />
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">{event.message}</div>
              <div className="text-xs text-gray-500">
                {new Date(event.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Helper Components
const MetricCard: React.FC<{
  label: string;
  current: number;
  average: number;
  peak: number;
  unit: string;
  color: string;
}> = ({ label, current, average, peak, unit, color }) => (
  <div className="bg-white border rounded-lg p-4">
    <div className="flex justify-between items-start mb-2">
      <h4 className="font-medium text-gray-700">{label}</h4>
      <div className={`text-2xl font-bold ${color}`}>
        {current.toFixed(1)}{unit}
      </div>
    </div>
    <div className="flex justify-between text-sm text-gray-500">
      <span>Avg: {average.toFixed(1)}{unit}</span>
      <span>Peak: {peak.toFixed(1)}{unit}</span>
    </div>
    <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
      <div 
        className={`h-2 rounded-full ${color.replace('text-', 'bg-')}`}
        style={{ width: `${Math.min(100, (current / peak) * 100)}%` }}
      ></div>
    </div>
  </div>
);

const StatusBadge: React.FC<{ status: ComponentStatus; showLabel?: boolean }> = ({ 
  status, 
  showLabel = false 
}) => {
  const config = {
    healthy: { bg: 'bg-green-100', text: 'text-green-800', dot: 'bg-green-400', label: 'Healthy' },
    warning: { bg: 'bg-yellow-100', text: 'text-yellow-800', dot: 'bg-yellow-400', label: 'Warning' },
    critical: { bg: 'bg-red-100', text: 'text-red-800', dot: 'bg-red-400', label: 'Critical' },
    offline: { bg: 'bg-gray-100', text: 'text-gray-800', dot: 'bg-gray-400', label: 'Offline' },
    processing: { bg: 'bg-blue-100', text: 'text-blue-800', dot: 'bg-blue-400', label: 'Processing' },
  };

  const { bg, text, dot, label } = config[status];

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${bg} ${text}`}>
      <span className={`w-2 h-2 ${dot} rounded-full mr-1`}></span>
      {showLabel && label}
    </span>
  );
};

const ConnectionItem: React.FC<{
  connection: ConnectionEdge;
  currentNodeId: string;
  theme: ThemeConfiguration;
  onNavigateToComponent?: (componentId: string) => void;
}> = ({ connection, currentNodeId, theme, onNavigateToComponent }) => {
  const isIncoming = connection.targetId === currentNodeId;
  const otherNodeId = isIncoming ? connection.sourceId : connection.targetId;
  const direction = isIncoming ? '←' : '→';

  return (
    <div className="flex items-center justify-between p-3 bg-white border rounded hover:bg-gray-50 transition-colors">
      <div className="flex items-center space-x-3">
        <div className="text-lg">{direction}</div>
        <div>
          <div className="font-medium text-sm">{connection.label || `${connection.type} connection`}</div>
          <div className="text-xs text-gray-500">
            Strength: {(connection.strength * 100).toFixed(0)}% | 
            {connection.active ? ' Active' : ' Inactive'}
          </div>
        </div>
      </div>
      <button
        onClick={() => onNavigateToComponent?.(otherNodeId)}
        className="text-blue-600 hover:text-blue-800 text-sm"
      >
        View
      </button>
    </div>
  );
};

// Utility functions and components
const ComponentIcon: React.FC<{ type: string; size: number }> = ({ type, size }) => {
  const icons = {
    'subcortical': '⚡',
    'cortical': '🧠',
    'thalamic': '🔄',
    'mcp': '🔧',
    'mcp-tool': '⚙️',
    'storage': '💾',
    'network': '🌐'
  };
  
  return (
    <div style={{ fontSize: `${size}px` }}>
      {icons[type] || '⬢'}
    </div>
  );
};

const ConnectionTypeIcon: React.FC<{ type: string }> = ({ type }) => {
  const icons = {
    excitation: '⚡',
    inhibition: '🚫',
    dataFlow: '📊',
    dependency: '🔗',
    bidirectional: '↔️'
  };
  
  return <span>{icons[type] || '🔗'}</span>;
};

const EventIcon: React.FC<{ type: string }> = ({ type }) => {
  const icons = {
    status_change: '🔄',
    metric_alert: '⚠️',
    connection_added: '🔗',
    initialization: '🚀',
    error: '❌',
    update: '📝'
  };
  
  return <span className="text-lg">{icons[type] || '📋'}</span>;
};

function getNodeTypeLabel(type: string): string {
  const labels = {
    'subcortical': 'Subcortical System',
    'cortical': 'Cortical System',
    'thalamic': 'Thalamic Relay',
    'mcp': 'MCP Component',
    'mcp-tool': 'MCP Tool',
    'storage': 'Storage System',
    'network': 'Network Component'
  };
  return labels[type] || type;
}

function getCPUColor(value: number): string {
  if (value > 80) return 'text-red-600';
  if (value > 60) return 'text-yellow-600';
  return 'text-green-600';
}

function getMemoryColor(value: number): string {
  if (value > 85) return 'text-red-600';
  if (value > 70) return 'text-yellow-600';
  return 'text-blue-600';
}

function getThroughputColor(value: number): string {
  if (value > 1000) return 'text-green-600';
  if (value > 100) return 'text-blue-600';
  return 'text-gray-600';
}

function getLatencyColor(value: number): string {
  if (value > 200) return 'text-red-600';
  if (value > 100) return 'text-yellow-600';
  return 'text-green-600';
}

function getErrorRateColor(value: number): string {
  if (value > 5) return 'text-red-600';
  if (value > 1) return 'text-yellow-600';
  return 'text-green-600';
}

export default ComponentDetails;