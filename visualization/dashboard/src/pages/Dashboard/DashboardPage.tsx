import React from 'react';
import { Link } from 'react-router-dom';
import {
  CpuChipIcon,
  CircuitBoardIcon,
  ShareIcon,
  DatabaseIcon,
  WrenchScrewdriverIcon,
  CubeTransparentIcon,
  ChartBarIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { useRealTimeData } from '../../hooks/useRealTimeData';
import { useRealTimeStatus } from '../../hooks/useRealTimeStatus';
import { MetricCard } from '../../components/common/MetricCard';
import { Chart } from '../../components/common/Chart';
import { StatusIndicator } from '../../components/common/StatusIndicator';
import { ProgressBar } from '../../components/common/ProgressBar';
import { ActivityFeed } from '../../components/common/ActivityFeed';

interface QuickActionItem {
  id: string;
  title: string;
  description: string;
  path: string;
  icon: React.ComponentType<any>;
  status?: 'active' | 'warning' | 'error';
  gradient: string;
}

const quickActions: QuickActionItem[] = [
  {
    id: 'cognitive',
    title: 'Cognitive Patterns',
    description: 'Pattern recognition and inhibitory mechanisms',
    path: '/cognitive',
    icon: CpuChipIcon,
    gradient: 'from-blue-500 to-cyan-600',
  },
  {
    id: 'neural',
    title: 'Neural Activity',
    description: 'Real-time neural network visualization',
    path: '/neural',
    icon: CircuitBoardIcon,
    gradient: 'from-purple-500 to-pink-600',
  },
  {
    id: 'knowledge-graph',
    title: 'Knowledge Graph',
    description: '3D semantic relationship visualization',
    path: '/knowledge-graph',
    icon: ShareIcon,
    gradient: 'from-green-500 to-teal-600',
  },
  {
    id: 'memory',
    title: 'Memory Systems',
    description: 'Memory consolidation and performance',
    path: '/memory',
    icon: DatabaseIcon,
    gradient: 'from-orange-500 to-red-600',
  },
  {
    id: 'tools',
    title: 'MCP Tools',
    description: 'Model Context Protocol tool catalog',
    path: '/tools',
    icon: WrenchScrewdriverIcon,
    gradient: 'from-indigo-500 to-purple-600',
  },
  {
    id: 'architecture',
    title: 'System Architecture',
    description: 'Component health and dependencies',
    path: '/architecture',
    icon: CubeTransparentIcon,
    gradient: 'from-gray-600 to-gray-800',
  },
];

const DashboardPage: React.FC = () => {
  const { 
    cognitiveData, 
    neuralData, 
    memoryData, 
    knowledgeGraphData,
    isLoading 
  } = useRealTimeData();
  
  const { systemStatus, componentStatuses, recentActivities } = useRealTimeStatus();

  // Calculate system metrics
  const systemMetrics = {
    totalNodes: knowledgeGraphData?.nodeCount || 0,
    activeConnections: neuralData?.activeConnections || 0,
    memoryUsage: memoryData?.usage || 0,
    cognitiveLoad: cognitiveData?.load || 0,
    uptime: systemStatus.uptime || 0,
    responseTime: systemStatus.performance?.responseTime || 0,
  };

  const chartData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    datasets: [
      {
        label: 'Neural Activity',
        data: [65, 59, 80, 81, 56, 55],
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
      },
      {
        label: 'Cognitive Load',
        data: [28, 48, 40, 19, 86, 27],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
      },
    ],
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                LLMKG Dashboard
              </h1>
              <p className="mt-2 text-lg text-gray-600">
                Brain-inspired Large Language Model Knowledge Graph
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <StatusIndicator 
                status={systemStatus.overall} 
                showLabel 
                size="lg"
              />
              <div className="text-right">
                <div className="text-sm text-gray-500">System Uptime</div>
                <div className="text-lg font-semibold text-gray-900">
                  {Math.floor(systemMetrics.uptime / 3600)}h {Math.floor((systemMetrics.uptime % 3600) / 60)}m
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Knowledge Nodes"
            value={systemMetrics.totalNodes.toLocaleString()}
            change={"+12%"}
            changeType="increase"
            icon={ShareIcon}
            className="bg-gradient-to-r from-blue-500 to-cyan-600"
          />
          <MetricCard
            title="Neural Connections"
            value={systemMetrics.activeConnections.toLocaleString()}
            change={"+8%"}
            changeType="increase"
            icon={CircuitBoardIcon}
            className="bg-gradient-to-r from-purple-500 to-pink-600"
          />
          <MetricCard
            title="Memory Usage"
            value={`${systemMetrics.memoryUsage}%`}
            change={"-3%"}
            changeType="decrease"
            icon={DatabaseIcon}
            className="bg-gradient-to-r from-green-500 to-teal-600"
          />
          <MetricCard
            title="Cognitive Load"
            value={`${systemMetrics.cognitiveLoad}%`}
            change={"+5%"}
            changeType="increase"
            icon={CpuChipIcon}
            className="bg-gradient-to-r from-orange-500 to-red-600"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">
                System Components
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {quickActions.map((action) => {
                  const status = componentStatuses[action.id]?.status || 'active';
                  return (
                    <Link
                      key={action.id}
                      to={action.path}
                      className="group relative overflow-hidden rounded-lg bg-white border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all duration-200"
                    >
                      <div className="p-6">
                        <div className="flex items-center justify-between mb-4">
                          <div className={`p-3 rounded-lg bg-gradient-to-r ${action.gradient} text-white`}>
                            <action.icon className="w-6 h-6" />
                          </div>
                          <StatusIndicator status={status} />
                        </div>
                        <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                          {action.title}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {action.description}
                        </p>
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>

            {/* System Performance Chart */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">
                  System Performance
                </h2>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500">Last 24 hours</span>
                  <ChartBarIcon className="w-5 h-5 text-gray-400" />
                </div>
              </div>
              <div className="h-80">
                <Chart
                  type="line"
                  data={chartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top' as const,
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100,
                      },
                    },
                  }}
                />
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* System Health */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                System Health
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">CPU Usage</span>
                    <span className="text-sm font-medium">
                      {systemStatus.performance?.cpu || 0}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={systemStatus.performance?.cpu || 0} 
                    max={100}
                    color="blue"
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Memory</span>
                    <span className="text-sm font-medium">
                      {systemStatus.performance?.memory || 0}%
                    </span>
                  </div>
                  <ProgressBar 
                    value={systemStatus.performance?.memory || 0} 
                    max={100}
                    color="purple"
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Network</span>
                    <span className="text-sm font-medium">
                      {systemStatus.performance?.network || 0} Mbps
                    </span>
                  </div>
                  <ProgressBar 
                    value={(systemStatus.performance?.network || 0) / 10} 
                    max={100}
                    color="green"
                  />
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Recent Activity
                </h3>
                <ClockIcon className="w-5 h-5 text-gray-400" />
              </div>
              <ActivityFeed activities={recentActivities.slice(0, 10)} />
            </div>

            {/* System Alerts */}
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                System Alerts
              </h3>
              <div className="space-y-3">
                <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                  <CheckCircleIcon className="w-5 h-5 text-green-500 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-green-900">
                      All systems operational
                    </p>
                    <p className="text-xs text-green-700 mt-1">
                      No critical issues detected
                    </p>
                  </div>
                </div>
                {systemStatus.warnings?.length > 0 && (
                  <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                    <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-yellow-900">
                        {systemStatus.warnings.length} warnings
                      </p>
                      <p className="text-xs text-yellow-700 mt-1">
                        Check system logs for details
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;