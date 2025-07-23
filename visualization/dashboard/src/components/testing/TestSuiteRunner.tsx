import React, { useState, useEffect } from 'react';
import {
  Card, Button, Table, Tag, Progress, Space, Typography, 
  Tooltip, Modal, List, Alert, Badge, Drawer, Tabs, 
  Descriptions, Statistic, Timeline, Switch
} from 'antd';
import {
  PlayCircleOutlined, PauseCircleOutlined, StopOutlined,
  CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined,
  InfoCircleOutlined, ExclamationCircleOutlined, BugOutlined,
  EyeOutlined, ReloadOutlined, SettingOutlined, WifiOutlined
} from '@ant-design/icons';
import { TestExecutionTracker, TestSuiteDefinition, TestExecution, TestExecutionLog } from '../../services/TestExecutionTracker';
import { useTestStreaming } from '../../hooks/useTestStreaming';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface TestSuiteRunnerProps {
  tracker: TestExecutionTracker;
  onTestComplete?: (execution: TestExecution) => void;
}

export const TestSuiteRunner: React.FC<TestSuiteRunnerProps> = ({ tracker, onTestComplete }) => {
  const [testSuites, setTestSuites] = useState<TestSuiteDefinition[]>([]);
  const [activeExecutions, setActiveExecutions] = useState<TestExecution[]>([]);
  const [executionHistory, setExecutionHistory] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [selectedExecution, setSelectedExecution] = useState<TestExecution | null>(null);
  const [executionDrawerVisible, setExecutionDrawerVisible] = useState(false);
  const [testOptions, setTestOptions] = useState({
    release: false,
    nocapture: false,
    ignored: false,
    features: [] as string[]
  });

  // Use test streaming for real-time updates
  const [streamingState, streamingActions] = useTestStreaming(undefined, {
    autoConnect: true,
    maxRecentEvents: 200,
    maxExecutionLogs: 100
  });

  useEffect(() => {
    loadTestSuites();
    loadExecutionHistory();
    
    // Subscribe to all test executions for real-time updates
    if (streamingState.connected) {
      streamingActions.subscribeToAllExecutions();
    }
    
    // Set up event listeners
    const handleExecutionStarted = (execution: TestExecution) => {
      setActiveExecutions(prev => [...prev, execution]);
    };
    
    const handleExecutionCompleted = (execution: TestExecution) => {
      setActiveExecutions(prev => prev.filter(e => e.id !== execution.id));
      loadExecutionHistory();
      onTestComplete?.(execution);
    };
    
    const handleExecutionFailed = (execution: TestExecution) => {
      setActiveExecutions(prev => prev.filter(e => e.id !== execution.id));
      loadExecutionHistory();
    };
    
    tracker.on('executionStarted', handleExecutionStarted);
    tracker.on('executionCompleted', handleExecutionCompleted);
    tracker.on('executionFailed', handleExecutionFailed);
    
    return () => {
      tracker.off('executionStarted', handleExecutionStarted);
      tracker.off('executionCompleted', handleExecutionCompleted);
      tracker.off('executionFailed', handleExecutionFailed);
    };
  }, [tracker, onTestComplete, streamingState.connected, streamingActions]);

  // Merge streaming executions with local executions
  useEffect(() => {
    const streamingExecutions = Array.from(streamingState.activeExecutions.values());
    setActiveExecutions(prev => {
      const merged = [...prev];
      
      // Add streaming executions that aren't already in local state
      streamingExecutions.forEach(streamingExec => {
        if (!merged.some(exec => exec.id === streamingExec.id)) {
          merged.push(streamingExec);
        }
      });
      
      return merged;
    });
  }, [streamingState.activeExecutions]);

  const loadTestSuites = async () => {
    try {
      setLoading(true);
      const suites = await tracker.getTestSuites();
      setTestSuites(suites);
    } catch (error) {
      console.error('Failed to load test suites:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadExecutionHistory = async () => {
    try {
      const history = tracker.getExecutionHistory();
      setExecutionHistory(history);
    } catch (error) {
      console.error('Failed to load execution history:', error);
    }
  };

  const runTestSuite = async (suiteId: string) => {
    try {
      await tracker.executeTestSuite(suiteId, testOptions);
    } catch (error) {
      console.error('Failed to run test suite:', error);
    }
  };

  const cancelExecution = async (executionId: string) => {
    try {
      await tracker.cancelExecution(executionId);
    } catch (error) {
      console.error('Failed to cancel execution:', error);
    }
  };

  const viewExecutionDetails = (execution: TestExecution) => {
    setSelectedExecution(execution);
    setExecutionDrawerVisible(true);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <ClockCircleOutlined spin style={{ color: '#1890ff' }} />;
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'cancelled':
        return <StopOutlined style={{ color: '#faad14' }} />;
      default:
        return <InfoCircleOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'processing';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'warning';
      default:
        return 'default';
    }
  };

  const suiteColumns = [
    {
      title: 'Suite Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: TestSuiteDefinition) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (category: string) => (
        <Tag color="blue">{category}</Tag>
      )
    },
    {
      title: 'Tests',
      dataIndex: 'totalTests',
      key: 'totalTests',
      width: 80,
      render: (count: number) => (
        <Badge count={count} style={{ backgroundColor: '#52c41a' }} />
      )
    },
    {
      title: 'Modules',
      dataIndex: 'modules',
      key: 'modules',
      width: 100,
      render: (modules: any[]) => (
        <Tooltip title={modules.map(m => m.relativePath).join(', ')}>
          <Badge count={modules.length} style={{ backgroundColor: '#1890ff' }} />
        </Tooltip>
      )
    },
    {
      title: 'Tags',
      dataIndex: 'tags',
      key: 'tags',
      width: 150,
      render: (tags: string[]) => (
        <Space wrap>
          {tags.slice(0, 2).map(tag => (
            <Tag key={tag} size="small">{tag}</Tag>
          ))}
          {tags.length > 2 && (
            <Tooltip title={tags.slice(2).join(', ')}>
              <Tag size="small">+{tags.length - 2}</Tag>
            </Tooltip>
          )}
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_: any, record: TestSuiteDefinition) => {
        const isRunning = activeExecutions.some(e => 
          e.testPattern === record.testPattern || e.category === record.category
        );
        
        return (
          <Space>
            <Button
              type="primary"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => runTestSuite(record.id)}
              loading={isRunning}
              disabled={!record.enabled}
            >
              Run
            </Button>
            <Button
              size="small"
              icon={<InfoCircleOutlined />}
              onClick={() => {
                // Show suite details
              }}
            >
              Info
            </Button>
          </Space>
        );
      }
    }
  ];

  const executionColumns = [
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag icon={getStatusIcon(status)} color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Suite',
      dataIndex: 'category',
      key: 'category',
      render: (category: string, record: TestExecution) => (
        <div>
          <Text strong>{category || 'Unknown'}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.testPattern}
          </Text>
        </div>
      )
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 150,
      render: (_: any, record: TestExecution) => (
        <div>
          <Progress
            percent={Math.round((record.progress.current / record.progress.total) * 100)}
            size="small"
            status={record.status === 'failed' ? 'exception' : 
                   record.status === 'running' ? 'active' : 'success'}
          />
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.progress.current}/{record.progress.total}
          </Text>
        </div>
      )
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 100,
      render: (_: any, record: TestExecution) => {
        const duration = record.endTime 
          ? record.endTime.getTime() - record.startTime.getTime()
          : Date.now() - record.startTime.getTime();
        return <Text>{Math.round(duration / 1000)}s</Text>;
      }
    },
    {
      title: 'Results',
      key: 'results',
      width: 120,
      render: (_: any, record: TestExecution) => {
        if (!record.summary) {
          return <Text type="secondary">Running...</Text>;
        }
        
        return (
          <Space>
            <Text style={{ color: '#52c41a' }}>{record.summary.passed}</Text>
            <Text>/</Text>
            <Text style={{ color: '#ff4d4f' }}>{record.summary.failed}</Text>
            <Text>/</Text>
            <Text style={{ color: '#faad14' }}>{record.summary.ignored}</Text>
          </Space>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_: any, record: TestExecution) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => viewExecutionDetails(record)}
          >
            View
          </Button>
          {record.status === 'running' && (
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => cancelExecution(record.id)}
            >
              Stop
            </Button>
          )}
        </Space>
      )
    }
  ];

  const renderExecutionDetails = () => (
    <Drawer
      title="Test Execution Details"
      width={800}
      onClose={() => setExecutionDrawerVisible(false)}
      open={executionDrawerVisible}
    >
      {selectedExecution && (
        <Tabs defaultActiveKey="overview">
          <TabPane tab="Overview" key="overview">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Descriptions bordered size="small">
                <Descriptions.Item label="Execution ID" span={2}>
                  {selectedExecution.id}
                </Descriptions.Item>
                <Descriptions.Item label="Status">
                  <Tag icon={getStatusIcon(selectedExecution.status)} color={getStatusColor(selectedExecution.status)}>
                    {selectedExecution.status.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Started">
                  {selectedExecution.startTime.toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="Duration">
                  {selectedExecution.endTime 
                    ? `${Math.round((selectedExecution.endTime.getTime() - selectedExecution.startTime.getTime()) / 1000)}s`
                    : 'Running...'
                  }
                </Descriptions.Item>
                <Descriptions.Item label="Category">
                  {selectedExecution.category || 'N/A'}
                </Descriptions.Item>
              </Descriptions>
              
              {selectedExecution.summary && (
                <Card title="Test Results" size="small">
                  <Space size="large">
                    <Statistic
                      title="Passed"
                      value={selectedExecution.summary.passed}
                      valueStyle={{ color: '#52c41a' }}
                      prefix={<CheckCircleOutlined />}
                    />
                    <Statistic
                      title="Failed"
                      value={selectedExecution.summary.failed}
                      valueStyle={{ color: '#ff4d4f' }}
                      prefix={<CloseCircleOutlined />}
                    />
                    <Statistic
                      title="Ignored"
                      value={selectedExecution.summary.ignored}
                      valueStyle={{ color: '#faad14' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                    <Statistic
                      title="Total Time"
                      value={selectedExecution.summary.executionTime}
                      suffix="ms"
                    />
                  </Space>
                </Card>
              )}
            </Space>
          </TabPane>
          
          <TabPane tab="Test Results" key="results">
            {selectedExecution.summary?.results && (
              <List
                itemLayout="vertical"
                dataSource={selectedExecution.summary.results}
                renderItem={(result) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={getStatusIcon(result.outcome)}
                      title={
                        <Space>
                          <Text strong>{result.testName}</Text>
                          <Tag color={getStatusColor(result.outcome)}>
                            {result.outcome.toUpperCase()}
                          </Tag>
                          <Text type="secondary">{result.executionTime}ms</Text>
                        </Space>
                      }
                      description={
                        result.failureMessage && (
                          <Alert
                            type="error"
                            message="Test Failure"
                            description={<pre style={{ fontSize: '11px' }}>{result.failureMessage}</pre>}
                            showIcon
                          />
                        )
                      }
                    />
                  </List.Item>
                )}
              />
            )}
          </TabPane>
          
          <TabPane tab="Logs" key="logs">
            <Timeline>
              {selectedExecution.logs.map((log, index) => (
                <Timeline.Item
                  key={index}
                  color={log.level === 'error' ? 'red' : log.level === 'warning' ? 'orange' : 'blue'}
                >
                  <div>
                    <Text strong>{log.timestamp.toLocaleTimeString()}</Text>
                    <Tag style={{ marginLeft: 8 }}>{log.level.toUpperCase()}</Tag>
                  </div>
                  <div>{log.message}</div>
                  {log.testName && (
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      Test: {log.testName}
                    </Text>
                  )}
                </Timeline.Item>
              ))}
            </Timeline>
          </TabPane>
        </Tabs>
      )}
    </Drawer>
  );

  return (
    <div>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Connection Status & Test Options */}
        <Card 
          title="Test Configuration" 
          size="small"
          extra={
            <Space>
              <Tag 
                icon={<WifiOutlined />} 
                color={streamingState.connected ? 'success' : streamingState.connecting ? 'processing' : 'error'}
              >
                {streamingState.connected ? 'Connected' : streamingState.connecting ? 'Connecting...' : 'Disconnected'}
              </Tag>
              {!streamingState.connected && (
                <Button 
                  size="small" 
                  onClick={streamingActions.connect}
                  loading={streamingState.connecting}
                >
                  Reconnect
                </Button>
              )}
            </Space>
          }
        >
          <Space>
            <Switch
              checked={testOptions.release}
              onChange={(checked) => setTestOptions(prev => ({ ...prev, release: checked }))}
            />
            <Text>Release Mode</Text>
            
            <Switch
              checked={testOptions.nocapture}
              onChange={(checked) => setTestOptions(prev => ({ ...prev, nocapture: checked }))}
            />
            <Text>No Capture</Text>
            
            <Switch
              checked={testOptions.ignored}
              onChange={(checked) => setTestOptions(prev => ({ ...prev, ignored: checked }))}
            />
            <Text>Include Ignored</Text>
          </Space>
          
          {streamingState.error && (
            <Alert
              type="warning"
              message="Real-time Updates Unavailable"
              description={streamingState.error}
              showIcon
              style={{ marginTop: 16 }}
            />
          )}
        </Card>

        {/* Active Executions */}
        {activeExecutions.length > 0 && (
          <Card 
            title="Running Tests" 
            size="small"
            extra={<Badge count={activeExecutions.length} />}
          >
            <Table
              dataSource={activeExecutions}
              columns={executionColumns}
              size="small"
              pagination={false}
              rowKey="id"
            />
          </Card>
        )}

        {/* Test Suites */}
        <Card
          title="Available Test Suites"
          size="small"
          extra={
            <Space>
              <Button 
                icon={<ReloadOutlined />} 
                onClick={loadTestSuites}
                loading={loading}
              >
                Refresh
              </Button>
              <Button icon={<SettingOutlined />}>
                Settings
              </Button>
            </Space>
          }
        >
          <Table
            dataSource={testSuites}
            columns={suiteColumns}
            size="small"
            loading={loading}
            pagination={{ pageSize: 10 }}
            rowKey="id"
          />
        </Card>

        {/* Execution History */}
        <Card title="Recent Executions" size="small">
          <Table
            dataSource={executionHistory.executions?.slice(0, 10) || []}
            columns={executionColumns}
            size="small"
            pagination={false}
            rowKey="id"
          />
        </Card>
      </Space>

      {renderExecutionDetails()}
    </div>
  );
};