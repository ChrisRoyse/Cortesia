import React, { useState, useEffect } from 'react';
import { Layout, Menu, Breadcrumb, Card, Row, Col, Alert, Spin, Typography, Space, Tag } from 'antd';
import { 
  DashboardOutlined,
  BranchesOutlined,
  BulbOutlined,
  BugOutlined,
  MemoryOutlined,
  ShareAltOutlined,
  SettingOutlined,
  BookOutlined,
  HeartOutlined,
  RocketOutlined
} from '@ant-design/icons';
import { Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { useLLMKG, useConnectionStatus } from '@/integration/VisualizationCore';

// Import dashboard components from different phases
import { MemoryDashboard } from '@phase7/components/MemoryDashboard';
import { CognitivePatternDashboard } from '@phase8/components/CognitivePatternDashboard';
import { DebuggingDashboard } from '@phase9/components/DebuggingDashboard';

// Import Phase 10 specific components
import { SystemOverview } from './SystemOverview';
import { ComponentRegistry } from './ComponentRegistry';
import { PerformanceMonitor } from './PerformanceMonitor';
import { DocumentationHub } from './DocumentationHub';
import { VersionControl } from './VersionControl';
import { ConnectionStatus } from './ConnectionStatus';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;

interface NavigationItem {
  key: string;
  icon: React.ReactNode;
  label: string;
  path: string;
  phase?: string;
  disabled?: boolean;
}

const navigationItems: NavigationItem[] = [
  {
    key: 'overview',
    icon: <DashboardOutlined />,
    label: 'System Overview',
    path: '/overview',
  },
  {
    key: 'memory',
    icon: <MemoryOutlined />,
    label: 'Memory Systems',
    path: '/memory',
    phase: 'Phase 7',
  },
  {
    key: 'cognitive',
    icon: <BulbOutlined />,
    label: 'Cognitive Patterns',
    path: '/cognitive',
    phase: 'Phase 8',
  },
  {
    key: 'debugging',
    icon: <BugOutlined />,
    label: 'Advanced Debugging',
    path: '/debugging',
    phase: 'Phase 9',
  },
  {
    key: 'components',
    icon: <ShareAltOutlined />,
    label: 'Component Registry',
    path: '/components',
    phase: 'Phase 10',
  },
  {
    key: 'performance',
    icon: <RocketOutlined />,
    label: 'Performance Monitor',
    path: '/performance',
    phase: 'Phase 10',
  },
  {
    key: 'version',
    icon: <BranchesOutlined />,
    label: 'Version Control',
    path: '/version',
    phase: 'Phase 10',
  },
  {
    key: 'docs',
    icon: <BookOutlined />,
    label: 'Documentation',
    path: '/docs',
    phase: 'Phase 10',
  },
  {
    key: 'settings',
    icon: <SettingOutlined />,
    label: 'Settings',
    path: '/settings',
    disabled: true,
  },
];

export const UnifiedDashboard: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKeys, setSelectedKeys] = useState<string[]>(['overview']);
  const location = useLocation();
  const { config } = useLLMKG();
  const { connected, connectionStatus, error } = useConnectionStatus();

  useEffect(() => {
    const currentPath = location.pathname;
    const currentItem = navigationItems.find(item => item.path === currentPath);
    if (currentItem) {
      setSelectedKeys([currentItem.key]);
    }
  }, [location.pathname]);

  const renderConnectionAlert = () => {
    if (connectionStatus === 'connecting') {
      return (
        <Alert
          message="Connecting to LLMKG System"
          description="Establishing connection to the brain-inspired cognitive architecture..."
          type="info"
          icon={<Spin size="small" />}
          showIcon
          style={{ margin: '16px' }}
        />
      );
    }

    if (connectionStatus === 'reconnecting') {
      return (
        <Alert
          message="Reconnecting"
          description="Connection lost, attempting to reconnect..."
          type="warning"
          icon={<Spin size="small" />}
          showIcon
          style={{ margin: '16px' }}
        />
      );
    }

    if (error) {
      return (
        <Alert
          message="Connection Error"
          description={error.message}
          type="error"
          showIcon
          style={{ margin: '16px' }}
        />
      );
    }

    return null;
  };

  const getBreadcrumbItems = () => {
    const currentPath = location.pathname;
    const currentItem = navigationItems.find(item => item.path === currentPath);
    
    const items = [
      {
        title: (
          <Link to="/overview">
            <DashboardOutlined /> LLMKG Visualization
          </Link>
        ),
      },
    ];

    if (currentItem && currentItem.key !== 'overview') {
      items.push({
        title: (
          <Space>
            {currentItem.icon}
            {currentItem.label}
            {currentItem.phase && (
              <Tag color="blue" size="small">{currentItem.phase}</Tag>
            )}
          </Space>
        ),
      });
    }

    return items;
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        theme="dark"
        width={250}
      >
        <div style={{ 
          padding: '16px', 
          textAlign: 'center',
          borderBottom: '1px solid #303030'
        }}>
          <Title 
            level={collapsed ? 5 : 4} 
            style={{ 
              color: 'white', 
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}
          >
            {collapsed ? 'LLMKG' : 'LLMKG Visualization'}
          </Title>
          {!collapsed && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Brain-Inspired Cognitive Architecture
            </Text>
          )}
        </div>

        <ConnectionStatus collapsed={collapsed} />

        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={selectedKeys}
          style={{ borderRight: 0 }}
        >
          {navigationItems.map(item => (
            <Menu.Item 
              key={item.key} 
              icon={item.icon}
              disabled={item.disabled}
            >
              <Link to={item.path}>
                <Space>
                  {item.label}
                  {!collapsed && item.phase && (
                    <Tag size="small" color="processing">
                      {item.phase}
                    </Tag>
                  )}
                </Space>
              </Link>
            </Menu.Item>
          ))}
        </Menu>
      </Sider>

      <Layout>
        <Header style={{ 
          background: '#fff', 
          padding: '0 16px',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <Breadcrumb items={getBreadcrumbItems()} />
          
          <Space>
            <Tag 
              color={connected ? 'success' : 'error'} 
              icon={<HeartOutlined />}
            >
              {connected ? 'Connected' : 'Disconnected'}
            </Tag>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Theme: {config.visualization.theme}
            </Text>
          </Space>
        </Header>

        <Content style={{ padding: '16px' }}>
          {renderConnectionAlert()}

          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<SystemOverview />} />
            <Route path="/memory" element={<MemoryDashboard />} />
            <Route path="/cognitive" element={<CognitivePatternDashboard />} />
            <Route path="/debugging" element={<DebuggingDashboard />} />
            <Route path="/components" element={<ComponentRegistry />} />
            <Route path="/performance" element={<PerformanceMonitor />} />
            <Route path="/version" element={<VersionControl />} />
            <Route path="/docs" element={<DocumentationHub />} />
            <Route path="/settings" element={
              <Card>
                <Alert
                  message="Settings Panel"
                  description="Configuration settings will be available in a future update."
                  type="info"
                  showIcon
                />
              </Card>
            } />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
};

export default UnifiedDashboard;