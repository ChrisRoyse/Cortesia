import React from 'react';
import { Card, Row, Col, Progress, Table } from 'antd';
import { CognitiveLayerMemory } from '../../types/memory';

interface CognitiveLayerMemoryVisualizationProps {
  memory: CognitiveLayerMemory;
}

export const CognitiveLayerMemoryVisualization: React.FC<CognitiveLayerMemoryVisualizationProps> = ({ memory }) => {
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

  const subcorticalUsagePercent = (memory.subcortical.used / memory.subcortical.total) * 100;
  const corticalUsagePercent = (memory.cortical.used / memory.cortical.total) * 100;
  const workingMemoryUsagePercent = (memory.workingMemory.used / memory.workingMemory.capacity) * 100;

  const workingMemoryColumns = [
    {
      title: 'Buffer ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Content',
      dataIndex: 'content',
      key: 'content',
      ellipsis: true,
    },
    {
      title: 'Size',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatBytes(size),
    },
    {
      title: 'Age (ms)',
      dataIndex: 'age',
      key: 'age',
      render: (age: number) => age.toLocaleString(),
    },
    {
      title: 'Access Count',
      dataIndex: 'accessCount',
      key: 'accessCount',
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: number) => priority.toFixed(3),
    },
  ];

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card title="Subcortical Memory">
            <div style={{ marginBottom: 16 }}>
              <div>Used: {formatBytes(memory.subcortical.used)}</div>
              <div>Total: {formatBytes(memory.subcortical.total)}</div>
            </div>
            <Progress
              percent={subcorticalUsagePercent}
              strokeColor="#1890ff"
              style={{ marginBottom: 16 }}
            />
            <div>
              {Object.entries(memory.subcortical.components).map(([name, size]) => (
                <div key={name} style={{ marginBottom: 4 }}>
                  <span style={{ textTransform: 'capitalize' }}>{name}: </span>
                  <span>{formatBytes(size)}</span>
                </div>
              ))}
            </div>
          </Card>
        </Col>
        
        <Col xs={24} md={8}>
          <Card title="Cortical Memory">
            <div style={{ marginBottom: 16 }}>
              <div>Used: {formatBytes(memory.cortical.used)}</div>
              <div>Total: {formatBytes(memory.cortical.total)}</div>
            </div>
            <Progress
              percent={corticalUsagePercent}
              strokeColor="#52c41a"
              style={{ marginBottom: 16 }}
            />
            <div>
              {Object.entries(memory.cortical.regions).map(([name, size]) => (
                <div key={name} style={{ marginBottom: 4 }}>
                  <span style={{ textTransform: 'capitalize' }}>{name}: </span>
                  <span>{formatBytes(size)}</span>
                </div>
              ))}
            </div>
          </Card>
        </Col>
        
        <Col xs={24} md={8}>
          <Card title="Working Memory">
            <div style={{ marginBottom: 16 }}>
              <div>Used: {formatBytes(memory.workingMemory.used)}</div>
              <div>Capacity: {formatBytes(memory.workingMemory.capacity)}</div>
              <div>Buffers: {memory.workingMemory.buffers.length}</div>
            </div>
            <Progress
              percent={workingMemoryUsagePercent}
              strokeColor="#fa8c16"
            />
          </Card>
        </Col>
      </Row>

      <Card title="Working Memory Buffers">
        <Table
          dataSource={memory.workingMemory.buffers}
          columns={workingMemoryColumns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} buffers`,
          }}
          scroll={{ x: 800 }}
        />
      </Card>
    </div>
  );
};