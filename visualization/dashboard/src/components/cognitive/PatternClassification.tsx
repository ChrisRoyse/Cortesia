import React from 'react';
import { Card, Table, Tag, Progress, Row, Col } from 'antd';
import { CognitivePattern, CognitiveMetrics, PatternType } from '../../types/cognitive';

interface PatternClassificationProps {
  patterns: CognitivePattern[];
  metrics: CognitiveMetrics;
}

export const PatternClassification: React.FC<PatternClassificationProps> = ({ patterns, metrics }) => {
  const getPatternTypeColor = (type: PatternType): string => {
    const colors: Record<PatternType, string> = {
      convergent: 'blue',
      divergent: 'green',
      lateral: 'orange',
      systems: 'purple',
      critical: 'red',
      abstract: 'cyan',
      adaptive: 'geekblue',
      chain_of_thought: 'magenta',
      tree_of_thoughts: 'lime'
    };
    return colors[type] || 'default';
  };

  const columns = [
    {
      title: 'Pattern ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: PatternType) => (
        <Tag color={getPatternTypeColor(type)}>
          {type.replace('_', ' ').toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Activation',
      dataIndex: 'activation',
      key: 'activation',
      render: (activation: number) => (
        <div>
          <Progress
            percent={activation * 100}
            size="small"
            strokeColor={activation > 0.7 ? '#ff4d4f' : activation > 0.4 ? '#faad14' : '#52c41a'}
          />
          <span style={{ fontSize: '12px' }}>{(activation * 100).toFixed(1)}%</span>
        </div>
      ),
      sorter: (a: CognitivePattern, b: CognitivePattern) => a.activation - b.activation,
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
      sorter: (a: CognitivePattern, b: CognitivePattern) => a.confidence - b.confidence,
    },
    {
      title: 'Connections',
      dataIndex: 'connections',
      key: 'connections',
      render: (connections: any[]) => connections.length,
    },
    {
      title: 'Complexity',
      dataIndex: ['metadata', 'complexity'],
      key: 'complexity',
      render: (complexity: number) => complexity.toFixed(1),
      sorter: (a: CognitivePattern, b: CognitivePattern) => a.metadata.complexity - b.metadata.complexity,
    },
  ];

  const typeStats = Object.entries(metrics.patternDistribution).map(([type, count]) => ({
    type: type as PatternType,
    count,
    percentage: (count / metrics.totalPatterns) * 100,
    avgActivation: patterns
      .filter(p => p.type === type)
      .reduce((sum, p) => sum + p.activation, 0) / count || 0
  }));

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Pattern Type Statistics">
            <Row gutter={[16, 16]}>
              {typeStats.map(({ type, count, percentage, avgActivation }) => (
                <Col xs={24} sm={12} md={8} lg={6} key={type}>
                  <Card size="small">
                    <div style={{ marginBottom: 8 }}>
                      <Tag color={getPatternTypeColor(type)}>
                        {type.replace('_', ' ').toUpperCase()}
                      </Tag>
                    </div>
                    <div style={{ marginBottom: 4 }}>
                      <strong>Count: {count}</strong>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ fontSize: '12px' }}>
                        {percentage.toFixed(1)}% of total
                      </span>
                    </div>
                    <Progress
                      percent={avgActivation * 100}
                      size="small"
                      strokeColor={getPatternTypeColor(type)}
                      format={() => `${(avgActivation * 100).toFixed(1)}%`}
                    />
                    <div style={{ fontSize: '11px', marginTop: 4 }}>
                      Avg Activation
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      <Card title="Pattern Details">
        <Table
          dataSource={patterns}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 20,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} patterns`,
          }}
          scroll={{ x: 800 }}
        />
      </Card>
    </div>
  );
};