import React, { useState } from 'react';
import { Card, Button, Input, Modal, List, Space, Tag, Typography, message } from 'antd';
import { CameraOutlined, CompareOutlined, SaveOutlined } from '@ant-design/icons';
import { format } from 'date-fns';
import type { PerformanceMetrics, PerformanceSnapshot as SnapshotType } from '../types';

const { Text, Paragraph } = Typography;

export interface PerformanceSnapshotProps {
  currentMetrics: PerformanceMetrics | null;
}

export const PerformanceSnapshot: React.FC<PerformanceSnapshotProps> = ({
  currentMetrics
}) => {
  const [snapshots, setSnapshots] = useState<SnapshotType[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [snapshotName, setSnapshotName] = useState('');
  const [snapshotDescription, setSnapshotDescription] = useState('');
  const [compareSnapshots, setCompareSnapshots] = useState<[string, string]>(['', '']);

  const createSnapshot = () => {
    if (!currentMetrics) {
      message.error('No metrics available for snapshot');
      return;
    }

    if (!snapshotName.trim()) {
      message.error('Please enter a snapshot name');
      return;
    }

    const newSnapshot: SnapshotType = {
      id: `snapshot-${Date.now()}`,
      timestamp: Date.now(),
      name: snapshotName,
      description: snapshotDescription,
      metrics: currentMetrics
    };

    setSnapshots([...snapshots, newSnapshot]);
    setShowCreateModal(false);
    setSnapshotName('');
    setSnapshotDescription('');
    message.success('Snapshot created successfully');
  };

  const compareSelectedSnapshots = () => {
    const [id1, id2] = compareSnapshots;
    if (!id1 || !id2) {
      message.error('Please select two snapshots to compare');
      return;
    }

    const snapshot1 = snapshots.find(s => s.id === id1);
    const snapshot2 = snapshots.find(s => s.id === id2);

    if (!snapshot1 || !snapshot2) {
      message.error('Selected snapshots not found');
      return;
    }

    // Calculate differences
    const differences: Record<string, number> = {};
    
    // Compare cognitive metrics
    Object.entries(snapshot2.metrics.cognitive).forEach(([layer, metrics]) => {
      const oldMetrics = snapshot1.metrics.cognitive[layer as keyof typeof snapshot1.metrics.cognitive];
      differences[`cognitive.${layer}.latency`] = 
        ((metrics.processingLatency - oldMetrics.processingLatency) / oldMetrics.processingLatency) * 100;
      differences[`cognitive.${layer}.throughput`] = 
        ((metrics.throughput - oldMetrics.throughput) / oldMetrics.throughput) * 100;
    });

    // Compare SDR metrics
    differences['sdr.creationRate'] = 
      ((snapshot2.metrics.sdr.creationRate - snapshot1.metrics.sdr.creationRate) / 
       snapshot1.metrics.sdr.creationRate) * 100;
    differences['sdr.memoryUsage'] = 
      ((snapshot2.metrics.sdr.memoryUsage - snapshot1.metrics.sdr.memoryUsage) / 
       snapshot1.metrics.sdr.memoryUsage) * 100;

    // Update comparison in snapshot2
    const updatedSnapshots = snapshots.map(s => 
      s.id === id2 
        ? { ...s, comparison: { baseline: id1, differences } }
        : s
    );
    
    setSnapshots(updatedSnapshots);
    setShowCompareModal(false);
    message.success('Comparison completed');
  };

  const deleteSnapshot = (id: string) => {
    setSnapshots(snapshots.filter(s => s.id !== id));
    message.success('Snapshot deleted');
  };

  return (
    <div className="performance-snapshot">
      <Card
        title="Performance Snapshots"
        extra={
          <Space>
            <Button
              type="primary"
              icon={<CameraOutlined />}
              onClick={() => setShowCreateModal(true)}
              disabled={!currentMetrics}
            >
              Create Snapshot
            </Button>
            <Button
              icon={<CompareOutlined />}
              onClick={() => setShowCompareModal(true)}
              disabled={snapshots.length < 2}
            >
              Compare
            </Button>
          </Space>
        }
      >
        <List
          dataSource={snapshots}
          renderItem={(snapshot) => (
            <List.Item
              actions={[
                <Button
                  key="view"
                  type="link"
                  onClick={() => {
                    // View snapshot details
                  }}
                >
                  View
                </Button>,
                <Button
                  key="delete"
                  type="link"
                  danger
                  onClick={() => deleteSnapshot(snapshot.id)}
                >
                  Delete
                </Button>
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Text strong>{snapshot.name}</Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {format(new Date(snapshot.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                    </Text>
                  </Space>
                }
                description={
                  <Space direction="vertical" size={4}>
                    {snapshot.description && (
                      <Paragraph ellipsis style={{ marginBottom: 8 }}>
                        {snapshot.description}
                      </Paragraph>
                    )}
                    <Space wrap>
                      <Tag>CPU: {snapshot.metrics.system.cpuUsage.toFixed(1)}%</Tag>
                      <Tag>Memory: {snapshot.metrics.system.memoryUsage.toFixed(1)}%</Tag>
                      <Tag>Latency: {snapshot.metrics.cognitive.cortical.processingLatency.toFixed(1)}ms</Tag>
                      <Tag>SDR Rate: {snapshot.metrics.sdr.creationRate.toFixed(0)}/s</Tag>
                    </Space>
                    {snapshot.comparison && (
                      <div style={{ marginTop: 8 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          Compared with: {snapshots.find(s => s.id === snapshot.comparison!.baseline)?.name}
                        </Text>
                        <div style={{ marginTop: 4 }}>
                          {Object.entries(snapshot.comparison.differences).slice(0, 3).map(([key, value]) => (
                            <Tag 
                              key={key}
                              color={value > 0 ? 'green' : 'red'}
                              style={{ fontSize: 11 }}
                            >
                              {key}: {value > 0 ? '+' : ''}{value.toFixed(1)}%
                            </Tag>
                          ))}
                        </div>
                      </div>
                    )}
                  </Space>
                }
              />
            </List.Item>
          )}
          locale={{ emptyText: 'No snapshots created yet' }}
        />
      </Card>

      {/* Create Snapshot Modal */}
      <Modal
        title="Create Performance Snapshot"
        visible={showCreateModal}
        onCancel={() => {
          setShowCreateModal(false);
          setSnapshotName('');
          setSnapshotDescription('');
        }}
        onOk={createSnapshot}
      >
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <div>
            <Text strong>Snapshot Name *</Text>
            <Input
              placeholder="Enter snapshot name"
              value={snapshotName}
              onChange={(e) => setSnapshotName(e.target.value)}
              style={{ marginTop: 8 }}
            />
          </div>
          <div>
            <Text strong>Description (Optional)</Text>
            <Input.TextArea
              placeholder="Enter description"
              value={snapshotDescription}
              onChange={(e) => setSnapshotDescription(e.target.value)}
              rows={3}
              style={{ marginTop: 8 }}
            />
          </div>
          {currentMetrics && (
            <div style={{ padding: 12, backgroundColor: '#f5f5f5', borderRadius: 8 }}>
              <Text strong>Current Metrics Summary:</Text>
              <div style={{ marginTop: 8, fontSize: 12 }}>
                <div>System Health: {((1 - currentMetrics.system.cpuUsage / 100) * 100).toFixed(0)}%</div>
                <div>Cognitive Latency: {currentMetrics.cognitive.cortical.processingLatency.toFixed(1)}ms</div>
                <div>SDR Creation Rate: {currentMetrics.sdr.creationRate.toFixed(0)}/s</div>
                <div>MCP Message Rate: {currentMetrics.mcp.messageRate.toFixed(0)} msg/s</div>
              </div>
            </div>
          )}
        </Space>
      </Modal>

      {/* Compare Snapshots Modal */}
      <Modal
        title="Compare Snapshots"
        visible={showCompareModal}
        onCancel={() => {
          setShowCompareModal(false);
          setCompareSnapshots(['', '']);
        }}
        onOk={compareSelectedSnapshots}
        width={600}
      >
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <div>
            <Text strong>Baseline Snapshot:</Text>
            <select
              value={compareSnapshots[0]}
              onChange={(e) => setCompareSnapshots([e.target.value, compareSnapshots[1]])}
              style={{ width: '100%', marginTop: 8, padding: 8 }}
            >
              <option value="">Select baseline snapshot</option>
              {snapshots.map(s => (
                <option key={s.id} value={s.id}>
                  {s.name} ({format(new Date(s.timestamp), 'yyyy-MM-dd HH:mm')})
                </option>
              ))}
            </select>
          </div>
          <div>
            <Text strong>Compare With:</Text>
            <select
              value={compareSnapshots[1]}
              onChange={(e) => setCompareSnapshots([compareSnapshots[0], e.target.value])}
              style={{ width: '100%', marginTop: 8, padding: 8 }}
            >
              <option value="">Select snapshot to compare</option>
              {snapshots
                .filter(s => s.id !== compareSnapshots[0])
                .map(s => (
                  <option key={s.id} value={s.id}>
                    {s.name} ({format(new Date(s.timestamp), 'yyyy-MM-dd HH:mm')})
                  </option>
                ))}
            </select>
          </div>
          {compareSnapshots[0] && compareSnapshots[1] && (
            <div style={{ padding: 12, backgroundColor: '#f5f5f5', borderRadius: 8 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                This will calculate the percentage change in key metrics between the selected snapshots.
              </Text>
            </div>
          )}
        </Space>
      </Modal>
    </div>
  );
};