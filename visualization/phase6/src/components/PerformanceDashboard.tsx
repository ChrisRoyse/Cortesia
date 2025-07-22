import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Alert, Space, Tabs, Button } from 'antd';
import { 
  ThunderboltOutlined, 
  DatabaseOutlined, 
  ClockCircleOutlined,
  WarningOutlined,
  DashboardOutlined,
  BarChartOutlined,
  LineChartOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { usePerformanceMetrics } from '../hooks/usePerformanceMetrics';
import { CognitivePerformanceChart } from './CognitivePerformanceChart';
import { SDRMetricsChart } from './SDRMetricsChart';
import { MCPProtocolChart } from './MCPProtocolChart';
import { SystemResourceChart } from './SystemResourceChart';
import { PerformanceHeatmap } from './PerformanceHeatmap';
import { AlertsPanel } from './AlertsPanel';
import { OptimizationPanel } from './OptimizationPanel';
import { PerformanceSnapshot } from './PerformanceSnapshot';
import type { PerformanceMetrics } from '../types';

const { TabPane } = Tabs;

export interface PerformanceDashboardProps {
  websocketUrl?: string;
  theme?: 'light' | 'dark';
  showAlerts?: boolean;
  showOptimizations?: boolean;
  refreshInterval?: number;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  websocketUrl = 'ws://localhost:8080/performance',
  theme = 'light',
  showAlerts = true,
  showOptimizations = true,
  refreshInterval = 1000
}) => {
  const { 
    metrics, 
    currentMetrics, 
    trends, 
    alerts, 
    isLoading, 
    error 
  } = usePerformanceMetrics({
    websocketUrl,
    refreshInterval
  });

  const [activeTab, setActiveTab] = useState('overview');
  const [timeWindow, setTimeWindow] = useState(300); // 5 minutes

  const calculateSystemHealth = (metrics: PerformanceMetrics | null): number => {
    if (!metrics) return 0;

    const factors = [
      1 - (metrics.system.cpuUsage / 100),
      1 - (metrics.system.memoryUsage / 100),
      metrics.mcp.errorRate < 0.01 ? 1 : 0.5,
      metrics.cognitive.subcortical.processingLatency < 50 ? 1 : 0.5
    ];

    return factors.reduce((a, b) => a + b, 0) / factors.length;
  };

  const getHealthColor = (value: number): string => {
    if (value >= 0.8) return '#52c41a';
    if (value >= 0.6) return '#faad14';
    return '#f5222d';
  };

  if (isLoading) {
    return (
      <div className="performance-dashboard loading">
        <Card>
          <Space direction="vertical" align="center" style={{ width: '100%' }}>
            <DashboardOutlined style={{ fontSize: 48 }} />
            <h3>Loading Performance Metrics...</h3>
          </Space>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="performance-dashboard error">
        <Alert
          message="Performance Monitoring Error"
          description={error.message}
          type="error"
          showIcon
        />
      </div>
    );
  }

  return (
    <div className={`performance-dashboard ${theme}`}>
      {/* Header Stats */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="System Health"
              value={(calculateSystemHealth(currentMetrics) * 100).toFixed(0)}
              suffix="%"
              prefix={<DashboardOutlined />}
              valueStyle={{ color: getHealthColor(calculateSystemHealth(currentMetrics)) }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Cognitive Latency"
              value={currentMetrics?.cognitive.cortical.processingLatency || 0}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
              precision={1}
              valueStyle={{ 
                color: (currentMetrics?.cognitive.cortical.processingLatency || 0) > 50 ? '#f5222d' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="SDR Creation Rate"
              value={currentMetrics?.sdr.creationRate || 0}
              suffix="/s"
              prefix={<ThunderboltOutlined />}
              precision={0}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="MCP Messages"
              value={currentMetrics?.mcp.messageRate || 0}
              suffix="msg/s"
              prefix={<DatabaseOutlined />}
              precision={0}
            />
          </Card>
        </Col>
      </Row>

      {/* Alerts Section */}
      {showAlerts && alerts.length > 0 && (
        <div className="alerts-section" style={{ marginBottom: 24 }}>
          {alerts.slice(0, 3).map((alert) => (
            <Alert
              key={alert.id}
              message={alert.message}
              type={alert.severity === 'critical' ? 'error' : alert.severity}
              icon={<WarningOutlined />}
              showIcon
              closable
              style={{ marginBottom: 8 }}
            />
          ))}
          {alerts.length > 3 && (
            <Button type="link" onClick={() => setActiveTab('alerts')}>
              View all {alerts.length} alerts
            </Button>
          )}
        </div>
      )}

      {/* Main Content Tabs */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={<span><LineChartOutlined />Overview</span>} 
          key="overview"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Cognitive Layer Performance">
                <CognitivePerformanceChart 
                  metrics={metrics}
                  timeWindow={timeWindow}
                  height={300}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="SDR Performance Metrics">
                <SDRMetricsChart 
                  metrics={metrics}
                  timeWindow={timeWindow}
                  height={300}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="MCP Protocol Performance">
                <MCPProtocolChart 
                  metrics={metrics}
                  timeWindow={timeWindow}
                  height={300}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="System Resources">
                <SystemResourceChart 
                  metrics={metrics}
                  timeWindow={timeWindow}
                  height={300}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={<span><BarChartOutlined />Detailed Analysis</span>} 
          key="analysis"
        >
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <Card title="Performance Heatmap">
                <PerformanceHeatmap 
                  metrics={metrics}
                  height={400}
                />
              </Card>
            </Col>
            <Col xs={24}>
              <Card title="Performance Trends">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {trends.map((trend) => (
                    <div key={trend.metric} className="trend-item">
                      <span className="trend-metric">{trend.metric}</span>
                      <span className={`trend-direction ${trend.trend}`}>
                        {trend.trend === 'increasing' ? '↑' : trend.trend === 'decreasing' ? '↓' : '→'}
                        {Math.abs(trend.changePercent).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        {showAlerts && (
          <TabPane 
            tab={<span><WarningOutlined />Alerts ({alerts.length})</span>} 
            key="alerts"
          >
            <AlertsPanel alerts={alerts} />
          </TabPane>
        )}

        {showOptimizations && (
          <TabPane 
            tab={<span><SettingOutlined />Optimizations</span>} 
            key="optimizations"
          >
            <OptimizationPanel />
          </TabPane>
        )}

        <TabPane 
          tab={<span><DatabaseOutlined />Snapshots</span>} 
          key="snapshots"
        >
          <PerformanceSnapshot currentMetrics={currentMetrics} />
        </TabPane>
      </Tabs>

      {/* Time Window Controls */}
      <div className="time-window-controls" style={{ marginTop: 24, textAlign: 'center' }}>
        <Space>
          <span>Time Window:</span>
          <Button 
            type={timeWindow === 60 ? 'primary' : 'default'}
            onClick={() => setTimeWindow(60)}
          >
            1m
          </Button>
          <Button 
            type={timeWindow === 300 ? 'primary' : 'default'}
            onClick={() => setTimeWindow(300)}
          >
            5m
          </Button>
          <Button 
            type={timeWindow === 900 ? 'primary' : 'default'}
            onClick={() => setTimeWindow(900)}
          >
            15m
          </Button>
          <Button 
            type={timeWindow === 3600 ? 'primary' : 'default'}
            onClick={() => setTimeWindow(3600)}
          >
            1h
          </Button>
        </Space>
      </div>
    </div>
  );
};