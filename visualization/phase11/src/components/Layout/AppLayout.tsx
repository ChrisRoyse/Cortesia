import React, { useState, useEffect } from 'react';
import { Layout, Menu, Breadcrumb, Typography, Space, Tag, Avatar, Dropdown, Button } from 'antd';
import { 
  DashboardOutlined,
  MemoryOutlined,
  BulbOutlined,
  BugOutlined,
  RocketOutlined,
  AppstoreOutlined,
  BranchesOutlined,
  BookOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  LogoutOutlined,
  QuestionCircleOutlined,
  HeartOutlined,
  WifiOutlined,
  DisconnectOutlined,
} from '@ant-design/icons';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppSelector, useAppDispatch } from '../../stores';
import { toggleSidebar, selectSidebarCollapsed } from '../../stores/visualizationSlice';
import { selectConnectionStatus } from '../../stores/systemSlice';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

interface AppLayoutProps {
  children: React.ReactNode;
}

interface NavigationItem {
  key: string;
  icon: React.ReactNode;
  label: string;
  path: string;
  phase?: string;
  badge?: string;
}

const navigationItems: NavigationItem[] = [
  {
    key: 'overview',
    icon: <DashboardOutlined />,
    label: 'System Overview',
    path: '/overview',
    badge: 'live',
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
    key: 'performance',
    icon: <RocketOutlined />,
    label: 'Performance Monitor',
    path: '/performance',
    phase: 'Phase 11',
  },
  {
    key: 'registry',
    icon: <AppstoreOutlined />,
    label: 'Component Registry',
    path: '/registry',
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
  },
  {
    key: 'settings',
    icon: <SettingOutlined />,
    label: 'Settings',
    path: '/settings',
  },
];

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const location = useLocation();
  const dispatch = useAppDispatch();
  const collapsed = useAppSelector(selectSidebarCollapsed);
  const connectionStatus = useAppSelector(selectConnectionStatus);
  
  const [selectedKeys, setSelectedKeys] = useState<string[]>(['overview']);

  useEffect(() => {
    const currentPath = location.pathname;
    const currentItem = navigationItems.find(item => item.path === currentPath);
    if (currentItem) {
      setSelectedKeys([currentItem.key]);
    }
  }, [location.pathname]);

  const handleMenuClick = () => {
    if (window.innerWidth <= 768 && !collapsed) {
      dispatch(toggleSidebar());
    }
  };

  const getBreadcrumbItems = () => {
    const currentPath = location.pathname;
    const currentItem = navigationItems.find(item => item.path === currentPath);
    
    return [
      {
        title: (
          <Link to="/overview">
            <Space>
              <DashboardOutlined />
              LLMKG
            </Space>
          </Link>
        ),
      },
      ...(currentItem && currentItem.key !== 'overview' ? [{
        title: (
          <Space>
            {currentItem.icon}
            {currentItem.label}
            {currentItem.phase && (
              <Tag color="blue" size="small">{currentItem.phase}</Tag>
            )}
            {currentItem.badge && (
              <Tag color="red" size="small">{currentItem.badge}</Tag>
            )}
          </Space>
        ),
      }] : []),
    ];
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <WifiOutlined style={{ color: '#52c41a' }} />;
      case 'connecting':
      case 'reconnecting':
        return <WifiOutlined style={{ color: '#faad14' }} className="spin" />;
      case 'error':
        return <DisconnectOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <DisconnectOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const getConnectionText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return 'Reconnecting...';
      case 'error':
        return 'Connection Error';
      default:
        return 'Disconnected';
    }
  };

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
    },
    {
      key: 'help',
      icon: <QuestionCircleOutlined />,
      label: 'Help & Support',
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      danger: true,
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* Sidebar */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={280}
        collapsedWidth={80}
        style={{
          background: 'var(--background-color, #001529)',
          borderRight: '1px solid var(--border-color, #303030)',
        }}
        breakpoint="lg"
        onBreakpoint={(broken) => {
          if (broken && !collapsed) {
            dispatch(toggleSidebar());
          }
        }}
      >
        {/* Logo Section */}
        <motion.div
          style={{
            padding: collapsed ? '16px 8px' : '16px 24px',
            borderBottom: '1px solid var(--border-color, #303030)',
            textAlign: collapsed ? 'center' : 'left',
            transition: 'all 0.2s ease',
          }}
          animate={{
            padding: collapsed ? '16px 8px' : '16px 24px',
          }}
        >
          <Space size={collapsed ? 0 : 12} align="center">
            <motion.div
              style={{
                fontSize: collapsed ? '24px' : '32px',
                transition: 'font-size 0.2s ease',
              }}
              animate={{
                fontSize: collapsed ? '24px' : '32px',
              }}
            >
              🧠
            </motion.div>
            
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  transition={{ duration: 0.2 }}
                  style={{ overflow: 'hidden' }}
                >
                  <div>
                    <Title 
                      level={4} 
                      style={{ 
                        color: 'white', 
                        margin: 0,
                        whiteSpace: 'nowrap'
                      }}
                    >
                      LLMKG Viz
                    </Title>
                    <Text 
                      type="secondary" 
                      style={{ 
                        fontSize: '12px',
                        whiteSpace: 'nowrap'
                      }}
                    >
                      Brain-Inspired Architecture
                    </Text>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Space>
        </motion.div>

        {/* Connection Status */}
        <motion.div
          style={{
            padding: collapsed ? '8px' : '12px 24px',
            borderBottom: '1px solid var(--border-color, #303030)',
            textAlign: collapsed ? 'center' : 'left',
            background: connectionStatus === 'connected' 
              ? 'rgba(82, 196, 26, 0.1)' 
              : connectionStatus === 'error' 
              ? 'rgba(255, 77, 79, 0.1)' 
              : 'transparent',
          }}
        >
          <Space size={collapsed ? 0 : 8} align="center">
            {getConnectionIcon()}
            
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  transition={{ duration: 0.2 }}
                  style={{ overflow: 'hidden' }}
                >
                  <Text 
                    style={{ 
                      color: 'white', 
                      fontSize: '12px',
                      fontWeight: 500,
                      whiteSpace: 'nowrap'
                    }}
                  >
                    {getConnectionText()}
                  </Text>
                </motion.div>
              )}
            </AnimatePresence>
          </Space>
        </motion.div>

        {/* Navigation Menu */}
        <Menu
          mode="inline"
          selectedKeys={selectedKeys}
          style={{
            background: 'transparent',
            border: 'none',
            padding: '16px 8px',
          }}
          onClick={handleMenuClick}
        >
          {navigationItems.map(item => (
            <Menu.Item 
              key={item.key} 
              icon={item.icon}
              style={{
                margin: '4px 0',
                borderRadius: '6px',
                height: 'auto',
                lineHeight: 'normal',
                padding: collapsed ? '12px 8px' : '12px 16px',
              }}
            >
              <Link to={item.path} style={{ textDecoration: 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>{item.label}</span>
                  {!collapsed && (
                    <Space size={4}>
                      {item.phase && (
                        <Tag size="small" color="blue">
                          {item.phase}
                        </Tag>
                      )}
                      {item.badge && (
                        <Tag size="small" color="red">
                          {item.badge}
                        </Tag>
                      )}
                    </Space>
                  )}
                </div>
              </Link>
            </Menu.Item>
          ))}
        </Menu>

        {/* Version Info */}
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                position: 'absolute',
                bottom: '16px',
                left: '24px',
                right: '24px',
                textAlign: 'center',
              }}
            >
              <Text 
                type="secondary" 
                style={{ 
                  fontSize: '11px',
                  display: 'block',
                  marginBottom: '4px'
                }}
              >
                Version 2.0.0
              </Text>
              <Text 
                type="secondary" 
                style={{ 
                  fontSize: '10px',
                  display: 'block'
                }}
              >
                Production Ready
              </Text>
            </motion.div>
          )}
        </AnimatePresence>
      </Sider>

      {/* Main Layout */}
      <Layout>
        {/* Header */}
        <Header
          style={{
            background: 'var(--background-color, #001529)',
            borderBottom: '1px solid var(--border-color, #303030)',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            height: '64px',
          }}
        >
          <Space align="center">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => dispatch(toggleSidebar())}
              style={{
                fontSize: '16px',
                width: 40,
                height: 40,
                color: 'white',
              }}
            />
            
            <Breadcrumb 
              items={getBreadcrumbItems()}
              style={{ margin: '0 16px' }}
            />
          </Space>

          <Space align="center">
            <Tag 
              icon={<HeartOutlined />}
              color={connectionStatus === 'connected' ? 'success' : 'error'}
            >
              System {connectionStatus === 'connected' ? 'Healthy' : 'Issues'}
            </Tag>

            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: ({ key }) => {
                  if (key === 'logout') {
                    console.log('Logout clicked');
                  }
                },
              }}
              placement="bottomRight"
            >
              <Button
                type="text"
                style={{
                  height: 40,
                  padding: '0 8px',
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Avatar 
                  size="small" 
                  icon={<UserOutlined />}
                  style={{ backgroundColor: '#1890ff' }}
                />
                <span style={{ fontSize: '14px' }}>Admin</span>
              </Button>
            </Dropdown>
          </Space>
        </Header>

        {/* Content */}
        <Content
          style={{
            background: 'var(--background-color, #001529)',
            padding: '24px',
            overflow: 'auto',
            position: 'relative',
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ height: '100%' }}
          >
            {children}
          </motion.div>
        </Content>
      </Layout>
    </Layout>
  );
};