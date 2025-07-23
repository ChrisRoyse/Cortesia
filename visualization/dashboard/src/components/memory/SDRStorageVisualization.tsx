import React from 'react';
import { Card, Row, Col, Progress, Statistic } from 'antd';
import { SDRStorage } from '../../types/memory';

interface SDRStorageVisualizationProps {
  storage: SDRStorage;
}

export const SDRStorageVisualization: React.FC<SDRStorageVisualizationProps> = ({ storage }) => {
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

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total SDRs"
              value={storage.totalSDRs.toLocaleString()}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active SDRs"
              value={storage.activeSDRs.toLocaleString()}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Memory"
              value={formatBytes(storage.totalMemoryBytes)}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Compression Ratio"
              value={(storage.compressionRatio * 100).toFixed(1)}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      <Card title="Storage Blocks">
        <Row gutter={[16, 16]}>
          {storage.storageBlocks.slice(0, 12).map((block) => (
            <Col xs={24} sm={12} md={8} lg={6} key={block.id}>
              <Card size="small" title={`Block ${block.id.split('-')[1]}`}>
                <Progress
                  percent={Math.round((block.used / block.size) * 100)}
                  size="small"
                  strokeColor="#1890ff"
                />
                <div style={{ marginTop: 8, fontSize: '12px' }}>
                  <div>Used: {formatBytes(block.used)}</div>
                  <div>Size: {formatBytes(block.size)}</div>
                  <div>Patterns: {block.patterns.toLocaleString()}</div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
};