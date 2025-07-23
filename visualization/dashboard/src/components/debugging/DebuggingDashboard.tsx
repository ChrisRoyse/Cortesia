import React, { useState, useEffect } from 'react';
import { Card, Tabs, Row, Col, Statistic, Table, Tag, Button, Space, Typography, Alert } from 'antd';
import { 
  BugOutlined, 
  ClockCircleOutlined, 
  SearchOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';
import {
  DistributedTrace,
  TimeTravelSession,
  QueryAnalysis,
  ErrorLog,
  ErrorStats
} from '../../types/debugging';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface DebuggingDashboardProps {
  className?: string;
}

export const DebuggingDashboard: React.FC<DebuggingDashboardProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [traces, setTraces] = useState<DistributedTrace[]>(generateMockTraces());
  const [timeTravelSession, setTimeTravelSession] = useState<TimeTravelSession>(generateMockTimeTravelSession());
  const [queryAnalyses, setQueryAnalyses] = useState<QueryAnalysis[]>(generateMockQueryAnalyses());
  const [errorStats, setErrorStats] = useState<ErrorStats>(generateMockErrorStats());

  const renderOverview = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Traces"
              value={traces.length}
              prefix={<SearchOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Error Rate"
              value={(errorStats.total > 0 ? (errorStats.byLevel.error || 0) / errorStats.total * 100 : 0).toFixed(1)}
              suffix="%"
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: errorStats.byLevel.error > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Query Time"
              value={queryAnalyses.reduce((sum, q) => sum + q.executionTime, 0) / queryAnalyses.length || 0}
              suffix="ms"
              precision={1}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Time Travel Snapshots"
              value={timeTravelSession.snapshots.length}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {errorStats.byLevel.error > 0 && (
        <Alert
          message="Active Errors Detected"
          description={`${errorStats.byLevel.error} errors found across ${Object.keys(errorStats.byService).length} services`}
          type="error"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}
    </div>
  );

  return (
    <div className={className}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>Advanced Debugging Tools</Title>
        <Text type="secondary">Distributed tracing, time-travel debugging, and query analysis</Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={
            <span>
              <BugOutlined />
              Overview
            </span>
          } 
          key="overview"
        >
          {renderOverview()}
        </TabPane>
        
        <TabPane tab="Distributed Tracing" key="tracing">
          <DistributedTracingView traces={traces} />
        </TabPane>
        
        <TabPane tab="Time Travel Debug" key="timetravel">
          <TimeTravelDebugView session={timeTravelSession} />
        </TabPane>
        
        <TabPane tab="Query Analysis" key="queries">
          <QueryAnalysisView analyses={queryAnalyses} />
        </TabPane>
        
        <TabPane tab="Error Logging" key="errors">
          <ErrorLoggingView stats={errorStats} />
        </TabPane>
      </Tabs>
    </div>
  );
};

// Component implementations
const DistributedTracingView: React.FC<{ traces: DistributedTrace[] }> = ({ traces }) => {
  const columns = [
    { title: 'Trace ID', dataIndex: 'traceId', key: 'traceId', ellipsis: true },
    { title: 'Duration (ms)', dataIndex: 'duration', key: 'duration', render: (d: number) => d.toFixed(1) },
    { title: 'Spans', dataIndex: 'spanCount', key: 'spanCount' },
    { title: 'Services', dataIndex: 'services', key: 'services', render: (s: string[]) => s.length },
    { title: 'Errors', dataIndex: 'errorCount', key: 'errorCount', render: (e: number) => e > 0 ? <Tag color="red">{e}</Tag> : <Tag color="green">0</Tag> },
  ];

  return <Card title="Distributed Traces"><Table dataSource={traces} columns={columns} rowKey="traceId" /></Card>;
};

const TimeTravelDebugView: React.FC<{ session: TimeTravelSession }> = ({ session }) => (
  <Card title="Time Travel Debugging">
    <Space style={{ marginBottom: 16 }}>
      <Button icon={session.isPlaying ? <PauseCircleOutlined /> : <PlayCircleOutlined />}>
        {session.isPlaying ? 'Pause' : 'Play'}
      </Button>
      <Text>Speed: {session.playbackSpeed}x</Text>
      <Text>Snapshot {session.currentIndex + 1} of {session.snapshots.length}</Text>
    </Space>
    <div>Current snapshot: {session.snapshots[session.currentIndex]?.label || 'None'}</div>
  </Card>
);

const QueryAnalysisView: React.FC<{ analyses: QueryAnalysis[] }> = ({ analyses }) => {
  const columns = [
    { title: 'Query ID', dataIndex: 'queryId', key: 'queryId' },
    { title: 'Execution Time (ms)', dataIndex: 'executionTime', key: 'executionTime', render: (t: number) => t.toFixed(1) },
    { title: 'Rows Processed', dataIndex: ['profile', 'rowsProcessed'], key: 'rowsProcessed' },
    { title: 'Suggestions', dataIndex: 'suggestions', key: 'suggestions', render: (s: any[]) => s.length },
  ];

  return <Card title="Query Analysis"><Table dataSource={analyses} columns={columns} rowKey="queryId" /></Card>;
};

const ErrorLoggingView: React.FC<{ stats: ErrorStats }> = ({ stats }) => {
  const columns = [
    { title: 'Message', dataIndex: 'message', key: 'message', ellipsis: true },
    { title: 'Level', dataIndex: 'level', key: 'level', render: (level: string) => <Tag color={level === 'error' ? 'red' : level === 'warning' ? 'orange' : 'blue'}>{level}</Tag> },
    { title: 'Service', dataIndex: ['context', 'service'], key: 'service' },
    { title: 'Frequency', dataIndex: 'frequency', key: 'frequency' },
    { title: 'Last Seen', dataIndex: 'lastSeen', key: 'lastSeen', render: (t: number) => new Date(t).toLocaleString() },
  ];

  return <Card title="Error Logs"><Table dataSource={stats.topErrors} columns={columns} rowKey="id" /></Card>;
};

// Mock data generators
function generateMockTraces(): DistributedTrace[] {
  return Array.from({ length: 10 }, (_, i) => ({
    traceId: `trace-${i}`,
    spans: [],
    rootSpan: {} as any,
    services: [`service-${i % 3}`, `service-${(i + 1) % 3}`],
    startTime: Date.now() - Math.random() * 60000,
    endTime: Date.now(),
    duration: Math.random() * 1000,
    spanCount: Math.floor(Math.random() * 20) + 1,
    errorCount: Math.floor(Math.random() * 3)
  }));
}

function generateMockTimeTravelSession(): TimeTravelSession {
  return {
    sessionId: 'session-1',
    snapshots: Array.from({ length: 10 }, (_, i) => ({
      id: `snapshot-${i}`,
      timestamp: Date.now() - (10 - i) * 5000,
      label: `Snapshot ${i + 1}`,
      state: { patterns: [], connections: [], memory: {}, activations: {} },
      metadata: { trigger: 'manual', changes: [], performance: { cpu: 50, memory: 60 } }
    })),
    currentIndex: 5,
    playbackSpeed: 1,
    isPlaying: false
  };
}

function generateMockQueryAnalyses(): QueryAnalysis[] {
  return Array.from({ length: 5 }, (_, i) => ({
    queryId: `query-${i}`,
    query: `SELECT * FROM table_${i}`,
    timestamp: Date.now() - Math.random() * 60000,
    executionTime: Math.random() * 500,
    plan: { nodes: [], estimatedCost: 100, estimatedRows: 1000 },
    profile: { actualTime: 100, planningTime: 10, executionTime: 90, rowsProcessed: 1000, bytesProcessed: 50000, memoryUsed: 1024, cacheHits: 800, cacheMisses: 200 },
    suggestions: [],
    bottlenecks: []
  }));
}

function generateMockErrorStats(): ErrorStats {
  return {
    total: 25,
    byLevel: { warning: 15, error: 8, critical: 2 },
    byCategory: { 'database': 10, 'network': 8, 'auth': 7 },
    byService: { 'api-service': 12, 'auth-service': 8, 'db-service': 5 },
    trend: [],
    topErrors: Array.from({ length: 5 }, (_, i) => ({
      id: `error-${i}`,
      timestamp: Date.now() - Math.random() * 60000,
      level: ['warning', 'error', 'critical'][Math.floor(Math.random() * 3)] as any,
      category: 'database',
      message: `Database connection timeout ${i}`,
      context: { service: 'db-service', operation: 'query' },
      frequency: Math.floor(Math.random() * 10) + 1,
      firstSeen: Date.now() - 86400000,
      lastSeen: Date.now() - Math.random() * 60000,
      resolved: false
    }))
  };
}