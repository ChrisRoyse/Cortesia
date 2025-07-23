import React, { useState, useEffect } from 'react';
import { Card, Tabs, Row, Col, Statistic, Progress, Tag, Space, Typography } from 'antd';
import { 
  BulbOutlined, 
  NodeIndexOutlined, 
  ThunderboltOutlined,
  BarChartOutlined,
  RadarChartOutlined
} from '@ant-design/icons';
import { PatternActivation3D } from './PatternActivation3D';
import { PatternClassification } from './PatternClassification';
import { InhibitionExcitationBalance as InhibitionExcitationBalanceViz } from './InhibitionExcitationBalance';
import { TemporalPatternAnalysis } from './TemporalPatternAnalysis';
import {
  CognitivePattern,
  PatternType,
  CognitiveMetrics,
  InhibitionExcitationBalance,
  TemporalPattern,
  PatternActivation
} from '../../types/cognitive';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface CognitivePatternDashboardProps {
  wsUrl?: string;
  className?: string;
}

export const CognitivePatternDashboard: React.FC<CognitivePatternDashboardProps> = ({ 
  wsUrl = 'ws://localhost:8081', 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Cognitive data states
  const [patterns, setPatterns] = useState<CognitivePattern[]>(generateMockPatterns());
  const [metrics, setMetrics] = useState<CognitiveMetrics>(generateMockMetrics());
  const [balance, setBalance] = useState<InhibitionExcitationBalance>(generateMockBalance());
  const [temporalPatterns, setTemporalPatterns] = useState<TemporalPattern[]>(generateMockTemporalPatterns());
  const [activations, setActivations] = useState<PatternActivation[]>(generateMockActivations());

  // WebSocket connection (similar to memory dashboard)
  useEffect(() => {
    // Simulate updates
    const simulationInterval = setInterval(() => {
      simulateCognitiveUpdates();
    }, 2000);

    return () => {
      clearInterval(simulationInterval);
    };
  }, []);

  const simulateCognitiveUpdates = () => {
    // Update pattern activations
    setPatterns(prev => prev.map(pattern => ({
      ...pattern,
      activation: Math.max(0, Math.min(1, pattern.activation + (Math.random() * 0.2 - 0.1))),
      confidence: Math.max(0, Math.min(1, pattern.confidence + (Math.random() * 0.1 - 0.05)))
    })));

    // Update metrics
    setMetrics(prev => ({
      ...prev,
      averageActivation: Math.random() * 0.8 + 0.1,
      performanceMetrics: {
        ...prev.performanceMetrics,
        successRate: Math.random() * 0.2 + 0.8,
        averageLatency: Math.random() * 50 + 10
      }
    }));

    setLastUpdate(new Date());
  };

  const getPatternTypeColor = (type: PatternType): string => {
    const colors: Record<PatternType, string> = {
      convergent: '#1890ff',
      divergent: '#52c41a',
      lateral: '#faad14',
      systems: '#722ed1',
      critical: '#f5222d',
      abstract: '#13c2c2',
      adaptive: '#fa8c16',
      chain_of_thought: '#eb2f96',
      tree_of_thoughts: '#a0d911'
    };
    return colors[type] || '#d9d9d9';
  };

  const renderOverview = () => (
    <div>
      {/* Metrics Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Patterns"
              value={metrics.totalPatterns}
              prefix={<BulbOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Patterns"
              value={metrics.activePatterns}
              prefix={<NodeIndexOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Average Activation"
              value={(metrics.averageActivation * 100).toFixed(1)}
              suffix="%"
              prefix={<ThunderboltOutlined />}
            />
            <Progress
              percent={metrics.averageActivation * 100}
              size="small"
              strokeColor="#1890ff"
              style={{ marginTop: 8 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Success Rate"
              value={(metrics.performanceMetrics.successRate * 100).toFixed(1)}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Pattern Distribution */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Pattern Type Distribution">
            <div>
              {Object.entries(metrics.patternDistribution).map(([type, count]) => (
                <div key={type} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Space>
                      <Tag color={getPatternTypeColor(type as PatternType)}>
                        {type.replace('_', ' ').toUpperCase()}
                      </Tag>
                    </Space>
                    <Text strong>{count}</Text>
                  </div>
                  <Progress
                    percent={(count / metrics.totalPatterns) * 100}
                    size="small"
                    strokeColor={getPatternTypeColor(type as PatternType)}
                    showInfo={false}
                    style={{ marginTop: 4 }}
                  />
                </div>
              ))}
            </div>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Performance Metrics">
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Success Rate"
                  value={(metrics.performanceMetrics.successRate * 100).toFixed(1)}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Avg Latency"
                  value={metrics.performanceMetrics.averageLatency.toFixed(1)}
                  suffix="ms"
                />
              </Col>
              <Col xs={24} sm={8}>
                <Statistic
                  title="Resource Efficiency"
                  value={(metrics.performanceMetrics.resourceEfficiency * 100).toFixed(1)}
                  suffix="%"
                />
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Active Patterns */}
      <Card title="Most Active Patterns">
        <Row gutter={[16, 16]}>
          {patterns
            .sort((a, b) => b.activation - a.activation)
            .slice(0, 6)
            .map((pattern) => (
              <Col xs={24} sm={12} md={8} key={pattern.id}>
                <Card size="small">
                  <div style={{ marginBottom: 8 }}>
                    <Tag color={getPatternTypeColor(pattern.type)}>
                      {pattern.type.replace('_', ' ').toUpperCase()}
                    </Tag>
                  </div>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>{pattern.name}</Text>
                  </div>
                  <Progress
                    percent={pattern.activation * 100}
                    size="small"
                    strokeColor={getPatternTypeColor(pattern.type)}
                  />
                  <div style={{ marginTop: 4, fontSize: '12px' }}>
                    <Text type="secondary">
                      Confidence: {(pattern.confidence * 100).toFixed(1)}%
                    </Text>
                  </div>
                </Card>
              </Col>
            ))}
        </Row>
      </Card>
    </div>
  );

  return (
    <div className={className}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>Cognitive Pattern Analysis</Title>
        <Text type="secondary">
          Real-time cognitive pattern visualization and analysis
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={
            <span>
              <BarChartOutlined />
              Overview
            </span>
          } 
          key="overview"
        >
          {renderOverview()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <RadarChartOutlined />
              3D Activation
            </span>
          } 
          key="activation"
        >
          <PatternActivation3D patterns={patterns} activations={activations} />
        </TabPane>
        
        <TabPane tab="Classification" key="classification">
          <PatternClassification patterns={patterns} metrics={metrics} />
        </TabPane>
        
        <TabPane tab="Balance Analysis" key="balance">
          <InhibitionExcitationBalanceViz balance={balance} />
        </TabPane>
        
        <TabPane tab="Temporal Analysis" key="temporal">
          <TemporalPatternAnalysis patterns={temporalPatterns} />
        </TabPane>
      </Tabs>
    </div>
  );
};

// Mock data generators
function generateMockPatterns(): CognitivePattern[] {
  const types: PatternType[] = [
    'convergent', 'divergent', 'lateral', 'systems', 'critical', 
    'abstract', 'adaptive', 'chain_of_thought', 'tree_of_thoughts'
  ];

  return Array.from({ length: 50 }, (_, i) => ({
    id: `pattern-${i}`,
    type: types[Math.floor(Math.random() * types.length)],
    name: `Pattern ${i + 1}`,
    activation: Math.random(),
    confidence: Math.random(),
    timestamp: Date.now() - Math.random() * 3600000,
    connections: Array.from({ length: Math.floor(Math.random() * 5) }, (_, j) => ({
      sourceId: `pattern-${i}`,
      targetId: `pattern-${Math.floor(Math.random() * 50)}`,
      strength: Math.random(),
      type: Math.random() > 0.5 ? 'excitatory' : 'inhibitory' as const
    })),
    metadata: {
      complexity: Math.random() * 100,
      resourceUsage: {
        cpu: Math.random() * 100,
        memory: Math.random() * 1000000,
        duration: Math.random() * 1000
      },
      parameters: {},
      tags: [`tag${Math.floor(Math.random() * 5)}`]
    }
  }));
}

function generateMockMetrics(): CognitiveMetrics {
  return {
    totalPatterns: 50,
    activePatterns: 28,
    averageActivation: 0.65,
    patternDistribution: {
      convergent: 8,
      divergent: 6,
      lateral: 5,
      systems: 7,
      critical: 4,
      abstract: 6,
      adaptive: 5,
      chain_of_thought: 4,
      tree_of_thoughts: 5
    },
    performanceMetrics: {
      successRate: 0.87,
      averageLatency: 45.2,
      resourceEfficiency: 0.78
    }
  };
}

function generateMockBalance(): InhibitionExcitationBalance {
  return {
    timestamp: Date.now(),
    excitation: {
      total: 0.65,
      byRegion: {
        prefrontal: 0.7,
        temporal: 0.6,
        parietal: 0.65,
        occipital: 0.5
      },
      patterns: ['pattern-1', 'pattern-3', 'pattern-7']
    },
    inhibition: {
      total: 0.35,
      byRegion: {
        prefrontal: 0.3,
        temporal: 0.4,
        parietal: 0.35,
        occipital: 0.5
      },
      patterns: ['pattern-2', 'pattern-5']
    },
    balance: 0.3,
    optimalRange: [0.2, 0.4]
  };
}

function generateMockTemporalPatterns(): TemporalPattern[] {
  return Array.from({ length: 10 }, (_, i) => ({
    id: `temporal-${i}`,
    sequence: Array.from({ length: 5 }, (_, j) => ({
      patternId: `pattern-${j}`,
      timestamp: Date.now() - j * 1000,
      activation: Math.random(),
      context: [`context-${j}`]
    })),
    frequency: Math.random() * 10,
    duration: Math.random() * 5000,
    predictability: Math.random()
  }));
}

function generateMockActivations(): PatternActivation[] {
  return Array.from({ length: 20 }, (_, i) => ({
    patternId: `pattern-${i}`,
    timestamp: Date.now() - Math.random() * 60000,
    activationLevel: Math.random(),
    propagation: Array.from({ length: 3 }, (_, j) => ({
      nodeId: `node-${j}`,
      activation: Math.random(),
      timestamp: Date.now() - j * 100,
      depth: j
    })),
    outcome: ['success', 'failure', 'partial'][Math.floor(Math.random() * 3)] as any
  }));
}