import React from 'react';
import { Space, Typography, Tag, Tooltip } from 'antd';
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  LoadingOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useConnectionStatus } from '@/integration/VisualizationCore';

const { Text } = Typography;

interface ConnectionStatusProps {
  collapsed?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ collapsed = false }) => {
  const { connected, connectionStatus, error } = useConnectionStatus();

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'connecting':
      case 'reconnecting':
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'disconnected':
      default:
        return <DisconnectOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'success';
      case 'connecting':
      case 'reconnecting':
        return 'processing';
      case 'error':
        return 'error';
      case 'disconnected':
      default:
        return 'default';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return 'Reconnecting...';
      case 'error':
        return 'Error';
      case 'disconnected':
      default:
        return 'Disconnected';
    }
  };

  const getTooltipContent = () => {
    const baseInfo = `Status: ${getStatusText()}`;
    if (error) {
      return `${baseInfo}\nError: ${error.message}`;
    }
    if (connectionStatus === 'connected') {
      return `${baseInfo}\nLLMKG System is responding normally`;
    }
    return baseInfo;
  };

  if (collapsed) {
    return (
      <div style={{ 
        padding: '8px', 
        textAlign: 'center',
        borderBottom: '1px solid #303030'
      }}>
        <Tooltip title={getTooltipContent()} placement="right">
          <Tag color={getStatusColor()} icon={getStatusIcon()}>
            {connectionStatus === 'connected' ? '●' : connectionStatus === 'error' ? '✕' : '○'}
          </Tag>
        </Tooltip>
      </div>
    );
  }

  return (
    <div style={{ 
      padding: '12px 16px', 
      borderBottom: '1px solid #303030',
      backgroundColor: connectionStatus === 'connected' ? 'rgba(82, 196, 26, 0.1)' : 
                      connectionStatus === 'error' ? 'rgba(255, 77, 79, 0.1)' : 
                      'transparent'
    }}>
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space>
          {getStatusIcon()}
          <Text style={{ color: 'white', fontSize: '12px', fontWeight: 500 }}>
            {getStatusText()}
          </Text>
        </Space>
        
        {connectionStatus === 'connected' && (
          <Text style={{ color: '#52c41a', fontSize: '11px' }}>
            LLMKG System Active
          </Text>
        )}
        
        {connectionStatus === 'error' && error && (
          <Tooltip title={error.message}>
            <Text style={{ 
              color: '#ff4d4f', 
              fontSize: '10px',
              display: 'block',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              {error.message}
            </Text>
          </Tooltip>
        )}
        
        {(connectionStatus === 'connecting' || connectionStatus === 'reconnecting') && (
          <Text style={{ color: '#1890ff', fontSize: '11px' }}>
            {connectionStatus === 'reconnecting' ? 'Attempting reconnection...' : 'Initializing connection...'}
          </Text>
        )}
      </Space>
    </div>
  );
};