import React, { useState, useEffect } from 'react';
import { Card, List, Tag, Button, Progress, Space, Modal, Typography } from 'antd';
import { RocketOutlined, ClockCircleOutlined, ToolOutlined } from '@ant-design/icons';
import type { PerformanceOptimization } from '../types';

const { Text, Paragraph } = Typography;

export const OptimizationPanel: React.FC = () => {
  const [optimizations, setOptimizations] = useState<PerformanceOptimization[]>([]);
  const [selectedOptimization, setSelectedOptimization] = useState<PerformanceOptimization | null>(null);

  useEffect(() => {
    // Mock optimizations - in real implementation, these would come from the service
    setOptimizations([
      {
        id: 'opt-1',
        category: 'cognitive',
        title: 'Parallelize Cortical Layer Processing',
        description: 'Implement parallel processing for cortical layer computations to reduce latency',
        impact: 'high',
        effort: 'medium',
        estimatedImprovement: 35,
        status: 'suggested'
      },
      {
        id: 'opt-2',
        category: 'sdr',
        title: 'Implement SDR Compression',
        description: 'Use advanced compression techniques for SDR storage to reduce memory usage',
        impact: 'medium',
        effort: 'low',
        estimatedImprovement: 40,
        status: 'suggested'
      },
      {
        id: 'opt-3',
        category: 'mcp',
        title: 'Enable Message Batching',
        description: 'Batch multiple MCP messages to reduce protocol overhead',
        impact: 'high',
        effort: 'low',
        estimatedImprovement: 25,
        status: 'in_progress'
      },
      {
        id: 'opt-4',
        category: 'system',
        title: 'Implement Caching Layer',
        description: 'Add intelligent caching to reduce redundant computations',
        impact: 'high',
        effort: 'high',
        estimatedImprovement: 50,
        status: 'suggested'
      }
    ]);
  }, []);

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'green';
      default: return 'default';
    }
  };

  const getEffortColor = (effort: string) => {
    switch (effort) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'green';
      default: return 'default';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'cognitive': return '🧠';
      case 'sdr': return '💾';
      case 'mcp': return '📡';
      case 'system': return '⚙️';
      default: return '📊';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'suggested': return '#1890ff';
      case 'in_progress': return '#faad14';
      case 'completed': return '#52c41a';
      case 'rejected': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  return (
    <div className="optimization-panel">
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Text>Total Optimizations: {optimizations.length}</Text>
          <Text>
            In Progress: {optimizations.filter(o => o.status === 'in_progress').length}
          </Text>
          <Text>
            Completed: {optimizations.filter(o => o.status === 'completed').length}
          </Text>
        </Space>
      </div>

      <List
        grid={{ gutter: 16, xs: 1, sm: 2, md: 2, lg: 3, xl: 3, xxl: 4 }}
        dataSource={optimizations}
        renderItem={(optimization) => (
          <List.Item>
            <Card
              hoverable
              onClick={() => setSelectedOptimization(optimization)}
              style={{ height: '100%' }}
              actions={[
                <Button
                  key="apply"
                  type="primary"
                  size="small"
                  disabled={optimization.status !== 'suggested'}
                  onClick={(e) => {
                    e.stopPropagation();
                    // Apply optimization
                  }}
                >
                  Apply
                </Button>,
                <Button
                  key="details"
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedOptimization(optimization);
                  }}
                >
                  Details
                </Button>
              ]}
            >
              <Card.Meta
                avatar={<span style={{ fontSize: 24 }}>{getCategoryIcon(optimization.category)}</span>}
                title={
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Text strong>{optimization.title}</Text>
                    <Progress
                      percent={optimization.status === 'completed' ? 100 : 
                              optimization.status === 'in_progress' ? 50 : 0}
                      strokeColor={getStatusColor(optimization.status)}
                      showInfo={false}
                      strokeWidth={4}
                    />
                  </Space>
                }
                description={
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 8 }}>
                      {optimization.description}
                    </Paragraph>
                    <Space wrap>
                      <Tag color={getImpactColor(optimization.impact)}>
                        Impact: {optimization.impact}
                      </Tag>
                      <Tag color={getEffortColor(optimization.effort)}>
                        Effort: {optimization.effort}
                      </Tag>
                    </Space>
                    <div style={{ marginTop: 8 }}>
                      <RocketOutlined /> Expected Improvement: 
                      <Text strong style={{ marginLeft: 8 }}>
                        +{optimization.estimatedImprovement}%
                      </Text>
                    </div>
                  </Space>
                }
              />
            </Card>
          </List.Item>
        )}
      />

      <Modal
        title="Optimization Details"
        visible={!!selectedOptimization}
        onCancel={() => setSelectedOptimization(null)}
        width={600}
        footer={[
          <Button key="close" onClick={() => setSelectedOptimization(null)}>
            Close
          </Button>,
          selectedOptimization?.status === 'suggested' && (
            <Button key="apply" type="primary">
              Apply Optimization
            </Button>
          )
        ]}
      >
        {selectedOptimization && (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <div>
              <Text strong style={{ fontSize: 18 }}>
                {getCategoryIcon(selectedOptimization.category)} {selectedOptimization.title}
              </Text>
            </div>
            
            <Paragraph>{selectedOptimization.description}</Paragraph>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Card size="small">
                <Statistic
                  title="Expected Improvement"
                  value={selectedOptimization.estimatedImprovement}
                  suffix="%"
                  prefix="+"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
              <Card size="small">
                <Statistic
                  title="Implementation Effort"
                  value={selectedOptimization.effort.toUpperCase()}
                  valueStyle={{ 
                    color: selectedOptimization.effort === 'low' ? '#52c41a' :
                           selectedOptimization.effort === 'medium' ? '#faad14' : '#ff4d4f',
                    fontSize: 16
                  }}
                />
              </Card>
            </div>

            <div>
              <Text strong>Implementation Steps:</Text>
              <ol style={{ marginTop: 8 }}>
                <li>Analyze current {selectedOptimization.category} performance metrics</li>
                <li>Implement the optimization in a test environment</li>
                <li>Run performance benchmarks to validate improvement</li>
                <li>Deploy to production with monitoring</li>
                <li>Monitor for any regressions or issues</li>
              </ol>
            </div>

            <div>
              <Text strong>Potential Risks:</Text>
              <ul style={{ marginTop: 8 }}>
                <li>Temporary performance degradation during implementation</li>
                <li>Increased complexity in the {selectedOptimization.category} subsystem</li>
                <li>Potential compatibility issues with existing components</li>
              </ul>
            </div>

            <div>
              <Text strong>Status:</Text>{' '}
              <Tag color={getStatusColor(selectedOptimization.status)}>
                {selectedOptimization.status.replace('_', ' ').toUpperCase()}
              </Tag>
            </div>
          </Space>
        )}
      </Modal>
    </div>
  );
};

// Statistic component (simple implementation)
const Statistic: React.FC<{
  title: string;
  value: string | number;
  suffix?: string;
  prefix?: string;
  valueStyle?: React.CSSProperties;
}> = ({ title, value, suffix, prefix, valueStyle }) => (
  <div>
    <div style={{ fontSize: 12, color: '#666' }}>{title}</div>
    <div style={{ fontSize: 24, fontWeight: 'bold', ...valueStyle }}>
      {prefix}{value}{suffix}
    </div>
  </div>
);