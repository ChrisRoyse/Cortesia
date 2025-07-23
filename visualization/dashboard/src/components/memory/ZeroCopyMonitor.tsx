import React from 'react';
import { Card, Row, Col, Statistic, Progress, Table } from 'antd';
import { ThunderboltOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { ZeroCopyMetrics } from '../../types/memory';

interface ZeroCopyMonitorProps {
  metrics: ZeroCopyMetrics;
  history: ZeroCopyMetrics[];
}

export const ZeroCopyMonitor: React.FC<ZeroCopyMonitorProps> = ({ metrics, history }) => {
  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  const columns = [
    {
      title: 'Time',
      key: 'time',
      render: (_: any, _record: any, index: number) => {
        const timeAgo = (history.length - index - 1) * 2; // Assuming 2-second intervals
        return `${timeAgo}s ago`;
      },
    },
    {
      title: 'Operations',
      dataIndex: 'totalOperations',
      key: 'totalOperations',
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: 'Saved Bytes',
      dataIndex: 'savedBytes',
      key: 'savedBytes',
      render: (value: number) => formatBytes(value),
    },
    {
      title: 'Efficiency',
      dataIndex: 'efficiency',
      key: 'efficiency',
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
  ];

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Status"
              value={metrics.enabled ? 'Enabled' : 'Disabled'}
              prefix={<CheckCircleOutlined style={{ color: metrics.enabled ? '#52c41a' : '#ff4d4f' }} />}
              valueStyle={{ color: metrics.enabled ? '#52c41a' : '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Operations"
              value={metrics.totalOperations.toLocaleString()}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Saved Bytes"
              value={formatBytes(metrics.savedBytes)}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Efficiency"
              value={(metrics.efficiency * 100).toFixed(1)}
              suffix="%"
            />
            <Progress
              percent={metrics.efficiency * 100}
              size="small"
              strokeColor="#1890ff"
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card title="Copy-on-Write Events">
            <Statistic
              title="Total Events"
              value={metrics.copyOnWriteEvents.toLocaleString()}
            />
            <div style={{ marginTop: 16 }}>
              <div>Shared Regions: <strong>{metrics.sharedRegions.toLocaleString()}</strong></div>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card title="Recent History">
            <Table
              dataSource={history.slice(-5).reverse()}
              columns={columns}
              size="small"
              pagination={false}
              rowKey={(record, index) => `history-${index}`}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};