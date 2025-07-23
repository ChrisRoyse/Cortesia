import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag, Space, Button, Select, Alert, Typography } from 'antd';
import { 
  RocketOutlined,
  ThunderboltOutlined,
  MemoryOutlined,
  NetworkOutlined,
  ClockCircleOutlined,
  ExclamationTriangleOutlined,
  BarChartOutlined,
  OptimizationOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useAppSelector, useAppDispatch } from '@/stores';
import { 
  selectCurrentMetrics, 
  selectPerformanceHistory, 
  selectPerformanceAlerts,
  selectProfiling,
  updateMetrics,
  startProfiling,
  stopProfiling,
  optimizePerformance
} from '@/stores/performanceSlice';

const { Title, Text } = Typography;
const { Option } = Select;

export const PerformanceMonitor: React.FC = () => {
  const dispatch = useAppDispatch();
  const currentMetrics = useAppSelector(selectCurrentMetrics);
  const history = useAppSelector(selectPerformanceHistory);
  const alerts = useAppSelector(selectPerformanceAlerts);
  const profiling = useAppSelector(selectProfiling);
  
  const [timeRange, setTimeRange] = useState<'5m' | '30m' | '1h' | '24h'>('30m');

  // Mock real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const mockMetrics = {
        renderTime: Math.random() * 20 + 5, // 5-25ms
        componentCount: Math.floor(Math.random() * 50) + 10, // 10-60 components
        memoryUsage: Math.random() * 100 * 1024 * 1024 + 50 * 1024 * 1024, // 50-150MB
        updateFrequency: Math.random() * 10 + 5, // 5-15 updates/sec
        dataProcessingTime: Math.random() * 50 + 10, // 10-60ms
        networkLatency: Math.random() * 100 + 50, // 50-150ms
      };
      
      dispatch(updateMetrics(mockMetrics));
    }, 1000);

    return () => clearInterval(interval);
  }, [dispatch]);

  const getTimeRangeData = () => {
    const now = Date.now();
    const ranges = {
      '5m': 5 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
    };
    
    const cutoff = now - ranges[timeRange];
    return history
      .filter(entry => entry.timestamp > cutoff)
      .map(entry => ({
        ...entry.metrics,
        timestamp: new Date(entry.timestamp).toLocaleTimeString(),
      }))
      .slice(-100); // Limit to last 100 points for performance
  };

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)}${units[unitIndex]}`;
  };

  const getPerformanceColor = (value: number, threshold: number): string => {
    const percentage = (value / threshold) * 100;
    if (percentage < 50) return '#52c41a';
    if (percentage < 80) return '#faad14';
    return '#ff4d4f';
  };

  const chartData = getTimeRangeData();

  const alertColumns = [
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={
          type === 'memory' ? 'red' :
          type === 'render' ? 'orange' :
          type === 'network' ? 'blue' : 'purple'
        }>
          {type.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
    },
    {
      title: 'Value',
      key: 'value',
      render: (_, record: any) => (
        <Text>
          {record.type === 'memory' ? formatBytes(record.currentValue) : 
           `${record.currentValue.toFixed(2)}ms`}
        </Text>
      ),
    },
    {
      title: 'Threshold',
      key: 'threshold',
      render: (_, record: any) => (
        <Text type="secondary">
          {record.type === 'memory' ? formatBytes(record.threshold) : 
           `${record.threshold}ms`}
        </Text>
      ),
    },
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: number) => (
        <Text style={{ fontSize: '12px' }}>
          {new Date(timestamp).toLocaleTimeString()}
        </Text>
      ),
    },
  ];

  return (
    <div style={{ padding: '16px 0' }}>
      <Title level={2}>
        <RocketOutlined /> Performance Monitor
      </Title>
      
      <Alert
        message="Real-Time Performance Monitoring"
        description="Monitor system performance metrics, identify bottlenecks, and optimize visualization components in real-time."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      {/* Control Panel */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <Select value={timeRange} onChange={setTimeRange} style={{ width: 120 }}>
            <Option value="5m">5 minutes</Option>
            <Option value="30m">30 minutes</Option>
            <Option value="1h">1 hour</Option>
            <Option value="24h">24 hours</Option>
          </Select>
          
          <Button 
            icon={<BarChartOutlined />}
            onClick={() => profiling.enabled ? dispatch(stopProfiling()) : dispatch(startProfiling({}))}
            type={profiling.enabled ? 'primary' : 'default'}
          >
            {profiling.enabled ? 'Stop Profiling' : 'Start Profiling'}
          </Button>
          
          <Button 
            icon={<OptimizationOutlined />}
            onClick={() => dispatch(optimizePerformance())}
          >
            Optimize
          </Button>
        </Space>
      </Card>

      {/* Current Metrics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Render Time"
              value={currentMetrics.renderTime}
              suffix="ms"
              precision={2}
              valueStyle={{ color: getPerformanceColor(currentMetrics.renderTime, 16) }}
              prefix={<ThunderboltOutlined />}
            />
            <Progress
              percent={Math.min((currentMetrics.renderTime / 16) * 100, 100)}
              strokeColor={getPerformanceColor(currentMetrics.renderTime, 16)}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic
              title="Memory Usage"
              value={formatBytes(currentMetrics.memoryUsage)}
              valueStyle={{ color: getPerformanceColor(currentMetrics.memoryUsage, 100 * 1024 * 1024) }}
              prefix={<MemoryOutlined />}
            />
            <Progress
              percent={Math.min((currentMetrics.memoryUsage / (100 * 1024 * 1024)) * 100, 100)}
              strokeColor={getPerformanceColor(currentMetrics.memoryUsage, 100 * 1024 * 1024)}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="Network Latency"
              value={currentMetrics.networkLatency}
              suffix="ms"
              precision={1}
              valueStyle={{ color: getPerformanceColor(currentMetrics.networkLatency, 200) }}
              prefix={<NetworkOutlined />}
            />
            <Progress
              percent={Math.min((currentMetrics.networkLatency / 200) * 100, 100)}
              strokeColor={getPerformanceColor(currentMetrics.networkLatency, 200)}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="Update Frequency"
              value={currentMetrics.updateFrequency}
              suffix="/sec"
              precision={1}
              valueStyle={{ color: '#1890ff' }}
              prefix={<ClockCircleOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {currentMetrics.componentCount} components
            </Text>
          </Card>
        </Col>
      </Row>

      {/* Performance Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="Render Performance" size="small">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="renderTime" 
                  stroke="#1890ff" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="Memory Usage" size="small">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis tickFormatter={(value) => formatBytes(value)} />
                <Tooltip formatter={(value) => formatBytes(value as number)} />
                <Area 
                  type="monotone" 
                  dataKey="memoryUsage" 
                  stroke="#52c41a" 
                  fill="#52c41a" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="Network & Processing" size="small">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="networkLatency" 
                  stroke="#faad14" 
                  strokeWidth={2}
                  dot={false}
                  name="Network Latency"
                />
                <Line 
                  type="monotone" 
                  dataKey="dataProcessingTime" 
                  stroke="#722ed1" 
                  strokeWidth={2}
                  dot={false}
                  name="Data Processing"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col span={12}>
          <Card 
            title={
              <Space>
                <ExclamationTriangleOutlined />
                Performance Alerts
                <Tag color="error">{alerts.length}</Tag>
              </Space>
            } 
            size="small"
          >
            <Table
              dataSource={alerts.slice(-10)}
              columns={alertColumns}
              rowKey="id"
              pagination={false}
              size="small"
              scroll={{ y: 200 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Profiling Status */}
      {profiling.enabled && (
        <Card 
          title="Profiling Active" 
          style={{ marginTop: 16 }}
          extra={<Tag color="processing">RECORDING</Tag>}
        >
          <Text>
            Performance profiling started at {' '}
            {profiling.startTime ? new Date(profiling.startTime).toLocaleTimeString() : 'Unknown'}
          </Text>
          <br />
          <Text type="secondary">
            Sample rate: {profiling.sampleRate}x
          </Text>
        </Card>
      )}
    </div>
  );
};