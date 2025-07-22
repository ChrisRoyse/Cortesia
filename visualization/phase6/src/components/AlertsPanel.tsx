import React, { useState } from 'react';
import { Table, Tag, Button, Space, Modal, Typography } from 'antd';
import { CheckOutlined, DeleteOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import { format } from 'date-fns';
import type { PerformanceAlert } from '../types';

const { Text } = Typography;

export interface AlertsPanelProps {
  alerts: PerformanceAlert[];
  onAcknowledge?: (alertId: string) => void;
  onDelete?: (alertId: string) => void;
}

export const AlertsPanel: React.FC<AlertsPanelProps> = ({
  alerts,
  onAcknowledge,
  onDelete
}) => {
  const [selectedAlert, setSelectedAlert] = useState<PerformanceAlert | null>(null);

  const getSeverityColor = (severity: PerformanceAlert['severity']) => {
    switch (severity) {
      case 'info': return 'blue';
      case 'warning': return 'orange';
      case 'critical': return 'red';
      case 'emergency': return 'magenta';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: number) => format(new Date(timestamp), 'HH:mm:ss'),
      sorter: (a: PerformanceAlert, b: PerformanceAlert) => a.timestamp - b.timestamp,
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: PerformanceAlert['severity']) => (
        <Tag color={getSeverityColor(severity)}>{severity.toUpperCase()}</Tag>
      ),
      filters: [
        { text: 'Emergency', value: 'emergency' },
        { text: 'Critical', value: 'critical' },
        { text: 'Warning', value: 'warning' },
        { text: 'Info', value: 'info' },
      ],
      onFilter: (value: any, record: PerformanceAlert) => record.severity === value,
    },
    {
      title: 'Component',
      dataIndex: 'component',
      key: 'component',
      width: 150,
      filters: [...new Set(alerts.map(a => a.component))].map(c => ({ text: c, value: c })),
      onFilter: (value: any, record: PerformanceAlert) => record.component === value,
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      render: (message: string, record: PerformanceAlert) => (
        <Space>
          <Text>{message}</Text>
          <Button 
            type="link" 
            size="small" 
            onClick={() => setSelectedAlert(record)}
          >
            Details
          </Button>
        </Space>
      ),
    },
    {
      title: 'Value',
      key: 'value',
      width: 120,
      render: (record: PerformanceAlert) => (
        <Space direction="vertical" size={0}>
          <Text type="danger">{record.value.toFixed(2)}</Text>
          <Text type="secondary" style={{ fontSize: 10 }}>
            Threshold: {record.threshold.toFixed(2)}
          </Text>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'acknowledged',
      key: 'acknowledged',
      width: 100,
      render: (acknowledged: boolean, record: PerformanceAlert) => (
        acknowledged ? (
          <Tag color="green">Acknowledged</Tag>
        ) : (
          <Tag color="red">Active</Tag>
        )
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (record: PerformanceAlert) => (
        <Space>
          {!record.acknowledged && (
            <Button
              type="primary"
              size="small"
              icon={<CheckOutlined />}
              onClick={() => onAcknowledge?.(record.id)}
            >
              Ack
            </Button>
          )}
          <Button
            danger
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => onDelete?.(record.id)}
          />
        </Space>
      ),
    },
  ];

  return (
    <div className="alerts-panel">
      <Table
        columns={columns}
        dataSource={alerts}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total) => `Total ${total} alerts`,
        }}
        rowClassName={(record) => 
          record.severity === 'emergency' ? 'alert-emergency' :
          record.severity === 'critical' ? 'alert-critical' : ''
        }
      />

      <Modal
        title="Alert Details"
        visible={!!selectedAlert}
        onCancel={() => setSelectedAlert(null)}
        footer={[
          <Button key="close" onClick={() => setSelectedAlert(null)}>
            Close
          </Button>,
          selectedAlert && !selectedAlert.acknowledged && (
            <Button
              key="acknowledge"
              type="primary"
              onClick={() => {
                onAcknowledge?.(selectedAlert.id);
                setSelectedAlert(null);
              }}
            >
              Acknowledge
            </Button>
          ),
        ]}
      >
        {selectedAlert && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Alert ID:</Text> {selectedAlert.id}
            </div>
            <div>
              <Text strong>Time:</Text> {format(new Date(selectedAlert.timestamp), 'yyyy-MM-dd HH:mm:ss')}
            </div>
            <div>
              <Text strong>Severity:</Text>{' '}
              <Tag color={getSeverityColor(selectedAlert.severity)}>
                {selectedAlert.severity.toUpperCase()}
              </Tag>
            </div>
            <div>
              <Text strong>Component:</Text> {selectedAlert.component}
            </div>
            <div>
              <Text strong>Metric:</Text> {selectedAlert.metric}
            </div>
            <div>
              <Text strong>Current Value:</Text>{' '}
              <Text type="danger">{selectedAlert.value.toFixed(2)}</Text>
            </div>
            <div>
              <Text strong>Threshold:</Text> {selectedAlert.threshold.toFixed(2)}
            </div>
            <div>
              <Text strong>Message:</Text> {selectedAlert.message}
            </div>
            {selectedAlert.acknowledged && (
              <div>
                <Text strong>Status:</Text>{' '}
                <Tag color="green">Acknowledged</Tag>
              </div>
            )}
            {selectedAlert.resolvedAt && (
              <div>
                <Text strong>Resolved At:</Text>{' '}
                {format(new Date(selectedAlert.resolvedAt), 'yyyy-MM-dd HH:mm:ss')}
              </div>
            )}
          </Space>
        )}
      </Modal>

      <style jsx>{`
        .alert-emergency {
          background-color: #fff2e8;
        }
        .alert-critical {
          background-color: #ffebe6;
        }
      `}</style>
    </div>
  );
};