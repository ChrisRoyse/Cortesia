import React from 'react';
import { Card, Table, Progress, Statistic, Row, Col } from 'antd';
import { ClockCircleOutlined, LineChartOutlined } from '@ant-design/icons';
import { TemporalPattern } from '../../types/cognitive';

interface TemporalPatternAnalysisProps {
  patterns: TemporalPattern[];
}

export const TemporalPatternAnalysis: React.FC<TemporalPatternAnalysisProps> = ({ patterns }) => {
  const columns = [
    {
      title: 'Pattern ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Sequence Length',
      dataIndex: 'sequence',
      key: 'sequenceLength',
      render: (sequence: any[]) => sequence.length,
      sorter: (a: TemporalPattern, b: TemporalPattern) => a.sequence.length - b.sequence.length,
    },
    {
      title: 'Frequency (Hz)',
      dataIndex: 'frequency',
      key: 'frequency',
      render: (frequency: number) => frequency.toFixed(2),
      sorter: (a: TemporalPattern, b: TemporalPattern) => a.frequency - b.frequency,
    },
    {
      title: 'Duration (ms)',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => duration.toFixed(0),
      sorter: (a: TemporalPattern, b: TemporalPattern) => a.duration - b.duration,
    },
    {
      title: 'Predictability',
      dataIndex: 'predictability',
      key: 'predictability',
      render: (predictability: number) => (
        <div>
          <Progress
            percent={predictability * 100}
            size="small"
            strokeColor={predictability > 0.7 ? '#52c41a' : predictability > 0.4 ? '#faad14' : '#ff4d4f'}
          />
          <span style={{ fontSize: '12px' }}>{(predictability * 100).toFixed(1)}%</span>
        </div>
      ),
      sorter: (a: TemporalPattern, b: TemporalPattern) => a.predictability - b.predictability,
    },
    {
      title: 'Next Predicted',
      dataIndex: 'nextPredicted',
      key: 'nextPredicted',
      render: (nextPredicted: any) => nextPredicted ? 'Yes' : 'No',
    },
  ];

  const avgFrequency = patterns.reduce((sum, p) => sum + p.frequency, 0) / patterns.length;
  const avgDuration = patterns.reduce((sum, p) => sum + p.duration, 0) / patterns.length;
  const avgPredictability = patterns.reduce((sum, p) => sum + p.predictability, 0) / patterns.length;
  const patternsWithPredictions = patterns.filter(p => p.nextPredicted).length;

  return (
    <div>
      {/* Summary Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Patterns"
              value={patterns.length}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Frequency"
              value={avgFrequency.toFixed(2)}
              suffix="Hz"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Duration"
              value={avgDuration.toFixed(0)}
              suffix="ms"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Predictability"
              value={(avgPredictability * 100).toFixed(1)}
              suffix="%"
              valueStyle={{ 
                color: avgPredictability > 0.7 ? '#52c41a' : 
                       avgPredictability > 0.4 ? '#faad14' : '#ff4d4f' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Prediction Status */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={12}>
          <Card title="Prediction Status">
            <div style={{ marginBottom: 16 }}>
              <div>Patterns with Predictions: <strong>{patternsWithPredictions}</strong></div>
              <div>Patterns without Predictions: <strong>{patterns.length - patternsWithPredictions}</strong></div>
            </div>
            <Progress
              percent={(patternsWithPredictions / patterns.length) * 100}
              strokeColor="#1890ff"
              format={() => `${patternsWithPredictions}/${patterns.length}`}
            />
          </Card>
        </Col>
        
        <Col xs={24} md={12}>
          <Card title="Predictability Distribution">
            <div>
              {[
                { label: 'High (>70%)', min: 0.7, color: '#52c41a' },
                { label: 'Medium (40-70%)', min: 0.4, max: 0.7, color: '#faad14' },
                { label: 'Low (<40%)', max: 0.4, color: '#ff4d4f' }
              ].map(({ label, min = 0, max = 1, color }) => {
                const count = patterns.filter(p => p.predictability >= min && p.predictability < max).length;
                const percentage = (count / patterns.length) * 100;
                
                return (
                  <div key={label} style={{ marginBottom: 8 }}>
                    <div style={{ marginBottom: 4 }}>
                      {label}: <strong>{count}</strong>
                    </div>
                    <Progress
                      percent={percentage}
                      strokeColor={color}
                      size="small"
                      format={() => `${percentage.toFixed(1)}%`}
                    />
                  </div>
                );
              })}
            </div>
          </Card>
        </Col>
      </Row>

      {/* Detailed Pattern Table */}
      <Card title="Temporal Pattern Details">
        <Table
          dataSource={patterns}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 15,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} patterns`,
          }}
          scroll={{ x: 800 }}
        />
      </Card>

      {/* Pattern Visualization Placeholder */}
      <Card title="Temporal Pattern Visualization" style={{ marginTop: 16 }}>
        <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
          Advanced temporal pattern visualization will be implemented here
        </div>
      </Card>
    </div>
  );
};