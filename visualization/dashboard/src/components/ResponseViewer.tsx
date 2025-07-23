import React, { useState } from 'react';
import { Card, Tabs, Tag, Space, Typography, Button, Tooltip, Descriptions } from 'antd';
import { 
  CodeOutlined, TableOutlined, EyeOutlined, CopyOutlined, 
  CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined 
} from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

const { TabPane } = Tabs;
const { Text } = Typography;

interface ResponseViewerProps {
  response: {
    status_code: number;
    headers: Record<string, string>;
    body?: string;
    size_bytes: number;
  };
  duration?: number;
}

export const ResponseViewer: React.FC<ResponseViewerProps> = ({ response, duration }) => {
  const [activeTab, setActiveTab] = useState('body');

  const getStatusColor = (status: number) => {
    if (status >= 200 && status < 300) return 'success';
    if (status >= 300 && status < 400) return 'warning';
    if (status >= 400 && status < 500) return 'error';
    if (status >= 500) return 'error';
    return 'default';
  };

  const getStatusIcon = (status: number) => {
    if (status >= 200 && status < 300) return <CheckCircleOutlined />;
    if (status >= 400) return <CloseCircleOutlined />;
    return <ClockCircleOutlined />;
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      // Could show a notification here
    });
  };

  const formatJSON = (text: string) => {
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      return text;
    }
  };

  const isJSON = (text: string) => {
    try {
      JSON.parse(text);
      return true;
    } catch {
      return false;
    }
  };

  const getContentType = () => {
    const contentType = response.headers['content-type'] || response.headers['Content-Type'] || '';
    return contentType.split(';')[0];
  };

  const renderBody = () => {
    if (!response.body) {
      return (
        <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
          <Text type="secondary">No response body</Text>
        </div>
      );
    }

    const contentType = getContentType();
    const body = response.body;

    if (contentType.includes('json') || isJSON(body)) {
      return (
        <div>
          <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tag color="blue">JSON</Tag>
            <Button 
              size="small" 
              icon={<CopyOutlined />} 
              onClick={() => copyToClipboard(formatJSON(body))}
            >
              Copy
            </Button>
          </div>
          <SyntaxHighlighter
            language="json"
            style={tomorrow}
            customStyle={{
              borderRadius: 6,
              border: '1px solid #f0f0f0',
              maxHeight: 400,
              overflow: 'auto'
            }}
          >
            {formatJSON(body)}
          </SyntaxHighlighter>
        </div>
      );
    }

    if (contentType.includes('html')) {
      return (
        <div>
          <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tag color="orange">HTML</Tag>
            <Button 
              size="small" 
              icon={<CopyOutlined />} 
              onClick={() => copyToClipboard(body)}
            >
              Copy
            </Button>
          </div>
          <SyntaxHighlighter
            language="html"
            style={tomorrow}
            customStyle={{
              borderRadius: 6,
              border: '1px solid #f0f0f0',
              maxHeight: 400,
              overflow: 'auto'
            }}
          >
            {body}
          </SyntaxHighlighter>
        </div>
      );
    }

    if (contentType.includes('xml')) {
      return (
        <div>
          <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tag color="purple">XML</Tag>
            <Button 
              size="small" 
              icon={<CopyOutlined />} 
              onClick={() => copyToClipboard(body)}
            >
              Copy
            </Button>
          </div>
          <SyntaxHighlighter
            language="xml"
            style={tomorrow}
            customStyle={{
              borderRadius: 6,
              border: '1px solid #f0f0f0',
              maxHeight: 400,
              overflow: 'auto'
            }}
          >
            {body}
          </SyntaxHighlighter>
        </div>
      );
    }

    // Plain text or unknown content type
    return (
      <div>
        <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Tag>{contentType || 'text/plain'}</Tag>
          <Button 
            size="small" 
            icon={<CopyOutlined />} 
            onClick={() => copyToClipboard(body)}
          >
            Copy
          </Button>
        </div>
        <pre style={{
          background: '#f5f5f5',
          padding: 16,
          borderRadius: 6,
          border: '1px solid #f0f0f0',
          maxHeight: 400,
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word'
        }}>
          {body}
        </pre>
      </div>
    );
  };

  const renderHeaders = () => (
    <div style={{ maxHeight: 400, overflow: 'auto' }}>
      {Object.entries(response.headers).length === 0 ? (
        <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
          <Text type="secondary">No response headers</Text>
        </div>
      ) : (
        <Descriptions bordered size="small" column={1}>
          {Object.entries(response.headers).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              <Text copyable={{ text: value }} style={{ fontFamily: 'monospace' }}>
                {value}
              </Text>
            </Descriptions.Item>
          ))}
        </Descriptions>
      )}
    </div>
  );

  const renderStatus = () => (
    <div>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Descriptions bordered size="small">
          <Descriptions.Item label="Status Code" span={2}>
            <Space>
              {getStatusIcon(response.status_code)}
              <Tag color={getStatusColor(response.status_code)}>
                {response.status_code}
              </Tag>
              <Text type="secondary">
                {response.status_code >= 200 && response.status_code < 300 && 'Success'}
                {response.status_code >= 300 && response.status_code < 400 && 'Redirect'}
                {response.status_code >= 400 && response.status_code < 500 && 'Client Error'}
                {response.status_code >= 500 && 'Server Error'}
              </Text>
            </Space>
          </Descriptions.Item>
          <Descriptions.Item label="Response Time" span={1}>
            <Text strong>{formatDuration(duration)}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Content Length" span={1}>
            <Text>{formatBytes(response.size_bytes)}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Content Type" span={2}>
            <Tag>{getContentType() || 'Unknown'}</Tag>
          </Descriptions.Item>
        </Descriptions>

        {duration && (
          <div style={{ marginTop: 16 }}>
            <Text strong>Performance Analysis:</Text>
            <div style={{ marginTop: 8 }}>
              {duration < 100 && (
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  Excellent (&lt; 100ms)
                </Tag>
              )}
              {duration >= 100 && duration < 500 && (
                <Tag color="warning" icon={<ClockCircleOutlined />}>
                  Good (100-500ms)
                </Tag>
              )}
              {duration >= 500 && duration < 1000 && (
                <Tag color="orange" icon={<ClockCircleOutlined />}>
                  Slow (500ms-1s)
                </Tag>
              )}
              {duration >= 1000 && (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  Very Slow (&gt; 1s)
                </Tag>
              )}
            </div>
          </div>
        )}
      </Space>
    </div>
  );

  return (
    <div style={{ border: '1px solid #f0f0f0', borderRadius: 6 }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        padding: '12px 16px',
        borderBottom: '1px solid #f0f0f0',
        background: '#fafafa'
      }}>
        <Space>
          {getStatusIcon(response.status_code)}
          <Tag color={getStatusColor(response.status_code)}>
            {response.status_code}
          </Tag>
          <Text strong>{formatBytes(response.size_bytes)}</Text>
          {duration && (
            <>
              <Text type="secondary">•</Text>
              <Text strong>{formatDuration(duration)}</Text>
            </>
          )}
        </Space>
        <Button 
          size="small" 
          icon={<CopyOutlined />} 
          onClick={() => copyToClipboard(JSON.stringify(response, null, 2))}
        >
          Copy Response
        </Button>
      </div>

      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        size="small"
        style={{ margin: 0 }}
        tabBarStyle={{ margin: '0 16px', paddingTop: 8 }}
      >
        <TabPane 
          tab={
            <span>
              <CodeOutlined />
              Body
            </span>
          } 
          key="body"
        >
          <div style={{ padding: 16 }}>
            {renderBody()}
          </div>
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <TableOutlined />
              Headers ({Object.keys(response.headers).length})
            </span>
          } 
          key="headers"
        >
          <div style={{ padding: 16 }}>
            {renderHeaders()}
          </div>
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <EyeOutlined />
              Status
            </span>
          } 
          key="status"
        >
          <div style={{ padding: 16 }}>
            {renderStatus()}
          </div>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ResponseViewer;