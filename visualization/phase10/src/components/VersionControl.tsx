import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tag, Timeline, Table, Space, Button, Modal, Descriptions, Typography, Alert } from 'antd';
import { 
  BranchesOutlined,
  TagOutlined,
  UserOutlined,
  ClockCircleOutlined,
  CodeOutlined,
  DeploymentUnitOutlined,
  HistoryOutlined,
  DiffOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { useLLMKG } from '@/integration/VisualizationCore';
import { GitInfo, BuildInfo, ChangelogEntry, DeploymentRecord } from '@/types';

const { Title, Text, Paragraph } = Typography;

export const VersionControl: React.FC = () => {
  const { mcpClient, connected } = useLLMKG();
  const [gitInfo, setGitInfo] = useState<GitInfo | null>(null);
  const [buildInfo, setBuildInfo] = useState<BuildInfo | null>(null);
  const [changelog, setChangelog] = useState<ChangelogEntry[]>([]);
  const [deployments, setDeployments] = useState<DeploymentRecord[]>([]);
  const [compareModalVisible, setCompareModalVisible] = useState(false);
  const [selectedCommits, setSelectedCommits] = useState<string[]>([]);

  useEffect(() => {
    loadVersionInfo();
    loadChangelog();
    loadDeploymentHistory();
  }, [connected]);

  const loadVersionInfo = async () => {
    try {
      if (connected && mcpClient) {
        const git = await mcpClient.request('version/getGitInfo', {});
        setGitInfo(git);

        const build = await mcpClient.request('version/getBuildInfo', {});
        setBuildInfo(build);
      } else {
        // Mock data for demonstration
        setGitInfo({
          branch: 'main',
          commit: 'abc123def456789',
          shortCommit: 'abc123d',
          author: 'LLMKG Developer',
          email: 'dev@llmkg.com',
          message: 'feat(phase10): implement unified visualization system with cross-phase integration',
          timestamp: new Date().toISOString(),
          isDirty: false,
          tags: ['v1.0.0', 'phase10-release']
        });

        setBuildInfo({
          version: '1.0.0',
          buildNumber: `${Date.now()}-abc123d`,
          buildDate: new Date().toISOString(),
          environment: 'development',
          gitInfo: {
            branch: 'main',
            commit: 'abc123def456789',
            shortCommit: 'abc123d',
            author: 'LLMKG Developer',
            email: 'dev@llmkg.com',
            message: 'feat(phase10): implement unified visualization system',
            timestamp: new Date().toISOString(),
            isDirty: false,
            tags: ['v1.0.0']
          }
        });
      }
    } catch (error) {
      console.error('Failed to load version info:', error);
    }
  };

  const loadChangelog = async () => {
    try {
      if (connected && mcpClient) {
        const changes = await mcpClient.request('version/getChangelog', { limit: 20 });
        setChangelog(changes);
      } else {
        // Mock changelog data
        setChangelog([
          {
            version: '1.0.0',
            date: new Date().toISOString(),
            author: 'LLMKG Team',
            changes: [
              {
                type: 'feature',
                description: 'Unified visualization system with cross-phase integration',
                component: 'Phase 10'
              },
              {
                type: 'feature',
                description: 'Component registry and performance monitoring',
                component: 'Phase 10'
              },
              {
                type: 'enhancement',
                description: 'Real-time state management with Redux Toolkit',
                component: 'Core'
              }
            ]
          },
          {
            version: '0.9.0',
            date: new Date(Date.now() - 86400000).toISOString(),
            author: 'LLMKG Team',
            changes: [
              {
                type: 'feature',
                description: 'Advanced debugging tools with time-travel capabilities',
                component: 'Phase 9'
              },
              {
                type: 'feature',
                description: 'Distributed tracing visualization',
                component: 'Phase 9'
              },
              {
                type: 'fix',
                description: 'Fixed memory leaks in pattern visualization',
                component: 'Phase 8'
              }
            ]
          },
          {
            version: '0.8.0',
            date: new Date(Date.now() - 172800000).toISOString(),
            author: 'LLMKG Team',
            changes: [
              {
                type: 'feature',
                description: 'Cognitive pattern visualization with 3D support',
                component: 'Phase 8'
              },
              {
                type: 'feature',
                description: 'Inhibition/excitation balance monitoring',
                component: 'Phase 8'
              }
            ]
          }
        ]);
      }
    } catch (error) {
      console.error('Failed to load changelog:', error);
    }
  };

  const loadDeploymentHistory = async () => {
    try {
      if (connected && mcpClient) {
        const history = await mcpClient.request('deployment/getHistory', { limit: 10 });
        setDeployments(history);
      } else {
        // Mock deployment data
        setDeployments([
          {
            id: 'deploy-1',
            version: '1.0.0',
            environment: 'development',
            timestamp: new Date().toISOString(),
            author: 'LLMKG Developer',
            status: 'success',
            duration: 180
          },
          {
            id: 'deploy-2',
            version: '0.9.0',
            environment: 'staging',
            timestamp: new Date(Date.now() - 86400000).toISOString(),
            author: 'LLMKG Developer',
            status: 'success',
            duration: 240
          },
          {
            id: 'deploy-3',
            version: '0.8.0',
            environment: 'development',
            timestamp: new Date(Date.now() - 172800000).toISOString(),
            author: 'LLMKG Developer',
            status: 'success',
            duration: 195
          }
        ]);
      }
    } catch (error) {
      console.error('Failed to load deployment history:', error);
    }
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'feature':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'fix':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'enhancement':
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#722ed1' }} />;
    }
  };

  const deploymentColumns = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string) => <Tag color="blue">{version}</Tag>
    },
    {
      title: 'Environment',
      dataIndex: 'environment',
      key: 'environment',
      render: (env: string) => (
        <Tag color={
          env === 'production' ? 'red' :
          env === 'staging' ? 'orange' : 'green'
        }>
          {env.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={
          status === 'success' ? 'success' :
          status === 'failed' ? 'error' : 'processing'
        }>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration?: number) => duration ? `${Math.floor(duration / 60)}m ${duration % 60}s` : '-'
    },
    {
      title: 'Date',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleString()
    },
    {
      title: 'Author',
      dataIndex: 'author',
      key: 'author'
    }
  ];

  return (
    <div style={{ padding: '16px 0' }}>
      <Title level={2}>
        <BranchesOutlined /> Version Control & Deployment
      </Title>
      <Paragraph>
        Track version history, deployment status, and system changes across all phases.
        Monitor git repository state and deployment pipeline health.
      </Paragraph>

      {!connected && (
        <Alert
          message="Demo Mode"
          description="Displaying mock version control data. Connect to LLMKG system for real repository information."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Current Version Display */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Current Version Information" size="default">
            {buildInfo && gitInfo ? (
              <Row gutter={16}>
                <Col span={8}>
                  <Descriptions title="Build Information" column={1} size="small">
                    <Descriptions.Item label="Version">
                      <Tag color="blue">{buildInfo.version}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="Build Number">
                      <Text style={{ fontFamily: 'monospace' }}>{buildInfo.buildNumber}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="Environment">
                      <Tag color={
                        buildInfo.environment === 'production' ? 'red' :
                        buildInfo.environment === 'staging' ? 'orange' : 'green'
                      }>
                        {buildInfo.environment.toUpperCase()}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="Build Date">
                      {new Date(buildInfo.buildDate).toLocaleString()}
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
                
                <Col span={8}>
                  <Descriptions title="Git Information" column={1} size="small">
                    <Descriptions.Item label="Branch">
                      <Space>
                        <BranchesOutlined />
                        <Tag color="cyan">{gitInfo.branch}</Tag>
                        {gitInfo.isDirty && <Tag color="warning">Modified</Tag>}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit">
                      <Space>
                        <CodeOutlined />
                        <Text style={{ fontFamily: 'monospace' }}>
                          {gitInfo.shortCommit}
                        </Text>
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Author">
                      <Space>
                        <UserOutlined />
                        {gitInfo.author}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit Time">
                      <Space>
                        <ClockCircleOutlined />
                        {new Date(gitInfo.timestamp).toLocaleString()}
                      </Space>
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
                
                <Col span={8}>
                  <Descriptions title="Tags & Labels" column={1} size="small">
                    <Descriptions.Item label="Git Tags">
                      <Space wrap>
                        {gitInfo.tags.map(tag => (
                          <Tag key={tag} icon={<TagOutlined />} color="geekblue">
                            {tag}
                          </Tag>
                        ))}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit Message" span={2}>
                      <div style={{ 
                        background: '#f5f5f5', 
                        padding: '8px 12px', 
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}>
                        {gitInfo.message}
                      </div>
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
              </Row>
            ) : (
              <div>Loading version information...</div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Recent Changes and Deployment History */}
      <Row gutter={[24, 24]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="Recent Changes" size="default" extra={
            <Button 
              icon={<HistoryOutlined />}
              onClick={() => setCompareModalVisible(true)}
            >
              Compare Versions
            </Button>
          }>
            <Timeline>
              {changelog.slice(0, 5).map((entry, index) => (
                <Timeline.Item
                  key={entry.version}
                  color={index === 0 ? 'green' : 'blue'}
                  dot={index === 0 ? <ClockCircleOutlined style={{ fontSize: '16px' }} /> : undefined}
                >
                  <div>
                    <Space>
                      <Tag color="purple">{entry.version}</Tag>
                      <Text style={{ fontSize: '12px', color: '#666' }}>
                        {new Date(entry.date).toLocaleDateString()}
                      </Text>
                      <Text style={{ fontSize: '12px', color: '#666' }}>
                        by {entry.author}
                      </Text>
                    </Space>
                    <div style={{ marginTop: '8px' }}>
                      {entry.changes.slice(0, 3).map((change, idx) => (
                        <div key={idx} style={{ marginBottom: '4px' }}>
                          <Tag 
                            size="small" 
                            color={
                              change.type === 'feature' ? 'green' :
                              change.type === 'fix' ? 'red' :
                              change.type === 'enhancement' ? 'blue' : 'orange'
                            }
                          >
                            {change.type}
                          </Tag>
                          <Text style={{ fontSize: '12px' }}>
                            {change.description}
                            {change.component && (
                              <Tag size="small" style={{ marginLeft: '8px' }}>
                                {change.component}
                              </Tag>
                            )}
                          </Text>
                        </div>
                      ))}
                      {entry.changes.length > 3 && (
                        <div style={{ fontSize: '12px', color: '#888' }}>
                          +{entry.changes.length - 3} more changes
                        </div>
                      )}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="Deployment History" size="default" extra={
            <Button 
              icon={<DeploymentUnitOutlined />}
              type="primary"
              disabled
            >
              Deploy
            </Button>
          }>
            <Table
              dataSource={deployments}
              columns={deploymentColumns}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* Version Comparison Modal */}
      <Modal
        title="Version Comparison"
        open={compareModalVisible}
        onCancel={() => setCompareModalVisible(false)}
        width={1000}
        footer={null}
      >
        <Alert
          message="Version Comparison"
          description="Version comparison functionality will be available in a future update. This will allow you to compare changes between different versions and commits."
          type="info"
          showIcon
        />
      </Modal>
    </div>
  );
};