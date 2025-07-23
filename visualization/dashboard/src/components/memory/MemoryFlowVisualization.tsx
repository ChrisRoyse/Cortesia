import React from 'react';
import { Card, Table, Tag } from 'antd';
import { MemoryFlow } from '../../types/memory';

interface MemoryFlowVisualizationProps {
  flows: MemoryFlow[];
}

export const MemoryFlowVisualization: React.FC<MemoryFlowVisualizationProps> = ({ flows }) => {
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

  const getOperationColor = (operation: string) => {
    switch (operation) {
      case 'allocate': return 'green';
      case 'free': return 'red';
      case 'copy': return 'blue';
      case 'share': return 'orange';
      default: return 'default';
    }
  };

  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: number) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
      },
      sorter: (a: MemoryFlow, b: MemoryFlow) => a.timestamp - b.timestamp,
    },
    {
      title: 'Source',
      dataIndex: 'source',
      key: 'source',
    },
    {
      title: 'Target',
      dataIndex: 'target',
      key: 'target',
    },
    {
      title: 'Operation',
      dataIndex: 'operation',
      key: 'operation',
      render: (operation: string) => (
        <Tag color={getOperationColor(operation)}>
          {operation.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Bytes',
      dataIndex: 'bytes',
      key: 'bytes',
      render: (bytes: number) => formatBytes(bytes),
      sorter: (a: MemoryFlow, b: MemoryFlow) => a.bytes - b.bytes,
    },
    {
      title: 'Duration (ms)',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => duration.toFixed(2),
      sorter: (a: MemoryFlow, b: MemoryFlow) => a.duration - b.duration,
    },
  ];

  const recentFlows = flows.slice(-50).sort((a, b) => b.timestamp - a.timestamp);

  return (
    <Card title="Memory Flow Operations">
      <Table
        dataSource={recentFlows}
        columns={columns}
        rowKey={(record) => `${record.timestamp}-${record.source}-${record.target}`}
        pagination={{
          pageSize: 20,
          showTotal: (total, range) => 
            `${range[0]}-${range[1]} of ${total} operations`,
        }}
        scroll={{ x: 800 }}
      />
    </Card>
  );
};