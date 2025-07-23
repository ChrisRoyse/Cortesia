import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Timeline, Tag, Space, Typography, Alert } from 'antd';
import { 
  ThunderboltOutlined,
  DatabaseOutlined,
  BranchesOutlined,
  BugOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { useLLMKG, useRealTimeData } from '@/integration/VisualizationCore';

const { Title, Text, Paragraph } = Typography;

interface SystemMetrics {
  memoryUsage: number;
  cognitiveLoad: number;
  knowledgeGraphSize: number;
  activeTasks: number;
  systemHealth: number;
  uptime: number;
}

interface SystemEvent {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  timestamp: string;
  component: string;
}

export const SystemOverview: React.FC = () => {
  const { connected, config } = useLLMKG();
  const [metrics] = useRealTimeData<SystemMetrics>('system/metrics', {
    memoryUsage: 0,
    cognitiveLoad: 0,
    knowledgeGraphSize: 0,
    activeTasks: 0,
    systemHealth: 0,
    uptime: 0,
  });

  const [events] = useRealTimeData<SystemEvent[]>('system/events', []);

  // Mock data for demonstration
  const mockMetrics: SystemMetrics = {
    memoryUsage: 68,
    cognitiveLoad: 45,
    knowledgeGraphSize: 1247,
    activeTasks: 8,
    systemHealth: 92,
    uptime: 1440 * 7, // 7 days in minutes
  };

  const mockEvents: SystemEvent[] = [
    {
      id: '1',
      type: 'success',
      message: 'Phase 10 unified system initialized successfully',
      timestamp: new Date().toISOString(),
      component: 'Core System'
    },
    {
      id: '2',
      type: 'info',
      message: 'Memory monitoring active - Phase 7 components loaded',
      timestamp: new Date(Date.now() - 300000).toISOString(),
      component: 'Memory System'
    },
    {
      id: '3',
      type: 'info',
      message: 'Cognitive pattern analysis running - Phase 8 components active',
      timestamp: new Date(Date.now() - 600000).toISOString(),
      component: 'Cognitive Engine'
    },
    {
      id: '4',
      type: 'info',
      message: 'Advanced debugging tools ready - Phase 9 components loaded',
      timestamp: new Date(Date.now() - 900000).toISOString(),
      component: 'Debug System'
    },
  ];

  const formatUptime = (minutes: number): string => {
    const days = Math.floor(minutes / (24 * 60));
    const hours = Math.floor((minutes % (24 * 60)) / 60);
    const mins = minutes % 60;
    return `${days}d ${hours}h ${mins}m`;
  };

  const getHealthColor = (health: number): string => {
    if (health >= 90) return '#52c41a';
    if (health >= 70) return '#faad14';
    if (health >= 50) return '#fa8c16';
    return '#ff4d4f';
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const currentMetrics = connected ? metrics : mockMetrics;
  const currentEvents = connected ? events : mockEvents;

  return (
    <div style={{ padding: '16px 0' }}>
      <Title level={2}>System Overview</Title>
      <Paragraph>
        Comprehensive view of the LLMKG brain-inspired cognitive architecture visualization system.
        Monitor real-time performance, system health, and component status across all phases.
      </Paragraph>

      {!connected && (
        <Alert
          message="Demo Mode"
          description="Displaying mock data. Connect to LLMKG system for real-time metrics."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* System Health Overview */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="System Health"
              value={currentMetrics.systemHealth}
              suffix="%"
              valueStyle={{ color: getHealthColor(currentMetrics.systemHealth) }}
              prefix={<ThunderboltOutlined />}
            />
            <Progress
              percent={currentMetrics.systemHealth}
              strokeColor={getHealthColor(currentMetrics.systemHealth)}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic
              title="Memory Usage"
              value={currentMetrics.memoryUsage}
              suffix="%"
              valueStyle={{ color: currentMetrics.memoryUsage > 80 ? '#ff4d4f' : '#1890ff' }}
              prefix={<DatabaseOutlined />}
            />
            <Progress
              percent={currentMetrics.memoryUsage}
              strokeColor={currentMetrics.memoryUsage > 80 ? '#ff4d4f' : '#1890ff'}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="Cognitive Load"
              value={currentMetrics.cognitiveLoad}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<BranchesOutlined />}
            />
            <Progress
              percent={currentMetrics.cognitiveLoad}
              strokeColor="#52c41a"
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="Active Tasks"
              value={currentMetrics.activeTasks}
              valueStyle={{ color: '#722ed1' }}
              prefix={<BugOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Background processes
            </Text>
          </Card>
        </Col>
      </Row>

      {/* Detailed Metrics */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="Knowledge Graph Metrics" size="small">
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Entities"
                  value={currentMetrics.knowledgeGraphSize}
                  valueStyle={{ fontSize: '18px' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Relations"
                  value={currentMetrics.knowledgeGraphSize * 2.4}
                  precision={0}
                  valueStyle={{ fontSize: '18px' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Uptime"
                  value={formatUptime(currentMetrics.uptime)}
                  valueStyle={{ fontSize: '14px' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="Phase Status" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Phase 7 - Memory Systems</Text>
                <Tag color="success">Active</Tag>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Phase 8 - Cognitive Patterns</Text>
                <Tag color="success">Active</Tag>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Phase 9 - Advanced Debugging</Text>
                <Tag color="success">Active</Tag>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Phase 10 - Integration</Text>
                <Tag color="processing">In Progress</Tag>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Phase 11 - Optimization</Text>
                <Tag color="default">Pending</Tag>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* System Events Timeline */}
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="Recent System Events" size="small">
            <Timeline>
              {currentEvents.map(event => (
                <Timeline.Item
                  key={event.id}
                  dot={getEventIcon(event.type)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Text strong>{event.message}</Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {event.component} • {new Date(event.timestamp).toLocaleString()}
                      </Text>
                    </div>
                    <Tag 
                      color={
                        event.type === 'success' ? 'success' :
                        event.type === 'warning' ? 'warning' :
                        event.type === 'error' ? 'error' : 'processing'
                      }
                    >
                      {event.type.toUpperCase()}
                    </Tag>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      {/* Configuration Summary */}
      <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="Configuration Summary" size="small">
            <Row gutter={16}>
              <Col span={8}>
                <Text strong>MCP Endpoint:</Text>
                <br />
                <Text type="secondary">{config.mcp.endpoint}</Text>
              </Col>
              <Col span={8}>
                <Text strong>Update Interval:</Text>
                <br />
                <Text type="secondary">{config.visualization.updateInterval}ms</Text>
              </Col>
              <Col span={8}>
                <Text strong>Theme:</Text>
                <br />
                <Text type="secondary">{config.visualization.theme}</Text>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};