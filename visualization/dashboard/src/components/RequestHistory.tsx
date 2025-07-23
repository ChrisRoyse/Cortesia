import React, { useState } from 'react';
import { Table, Tag, Button, Space, Typography, Tooltip, Modal, Input, Select } from 'antd';
import { 
  ReloadOutlined, EyeOutlined, CopyOutlined, DeleteOutlined, 
  FilterOutlined, SearchOutlined, HistoryOutlined 
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { ResponseViewer } from './ResponseViewer';

const { Text } = Typography;
const { Search } = Input;
const { Option } = Select;

interface ApiRequest {
  id: string;
  endpoint: string;
  method: string;
  timestamp: number;
  headers: Record<string, string>;
  query_params: Record<string, string>;
  body?: string;
  response?: ApiResponse;
  duration?: number;
  error?: string;
}

interface ApiResponse {
  status_code: number;
  headers: Record<string, string>;
  body?: string;
  size_bytes: number;
}

interface RequestHistoryProps {
  requests: ApiRequest[];
  onReplayRequest: (request: ApiRequest) => void;
  onClearHistory?: () => void;
}

export const RequestHistory: React.FC<RequestHistoryProps> = ({
  requests,
  onReplayRequest,
  onClearHistory
}) => {
  const [selectedRequest, setSelectedRequest] = useState<ApiRequest | null>(null);
  const [showResponseModal, setShowResponseModal] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [methodFilter, setMethodFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  const getStatusColor = (status?: number) => {
    if (!status) return 'default';
    if (status >= 200 && status < 300) return 'success';
    if (status >= 300 && status < 400) return 'warning';
    if (status >= 400 && status < 500) return 'error';
    if (status >= 500) return 'error';
    return 'default';
  };

  const getMethodColor = (method: string) => {
    switch (method.toLowerCase()) {
      case 'get': return 'blue';
      case 'post': return 'green';
      case 'put': return 'orange';
      case 'delete': return 'red';
      case 'patch': return 'purple';
      default: return 'default';
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const filteredRequests = requests.filter(request => {
    const matchesSearch = searchTerm === '' || 
      request.endpoint.toLowerCase().includes(searchTerm.toLowerCase()) ||
      request.method.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesMethod = methodFilter === '' || request.method === methodFilter;
    
    const matchesStatus = statusFilter === '' || 
      (statusFilter === 'success' && request.response?.status_code && request.response.status_code >= 200 && request.response.status_code < 300) ||
      (statusFilter === 'error' && request.response?.status_code && request.response.status_code >= 400) ||
      (statusFilter === 'failed' && request.error);

    return matchesSearch && matchesMethod && matchesStatus;
  });

  const columns: ColumnsType<ApiRequest> = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: number) => (
        <Tooltip title={formatTimestamp(timestamp)}>
          <Text style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleTimeString()}
          </Text>
        </Tooltip>
      ),
      sorter: (a, b) => a.timestamp - b.timestamp,
      defaultSortOrder: 'descend'
    },
    {
      title: 'Method',
      dataIndex: 'method',
      key: 'method',
      width: 80,
      render: (method: string) => (
        <Tag color={getMethodColor(method)} style={{ fontSize: '11px' }}>
          {method}
        </Tag>
      ),
      filters: [
        { text: 'GET', value: 'GET' },
        { text: 'POST', value: 'POST' },
        { text: 'PUT', value: 'PUT' },
        { text: 'DELETE', value: 'DELETE' },
        { text: 'PATCH', value: 'PATCH' }
      ],
      onFilter: (value, record) => record.method === value
    },
    {
      title: 'Endpoint',
      dataIndex: 'endpoint',
      key: 'endpoint',
      ellipsis: {
        showTitle: false
      },
      render: (endpoint: string) => (
        <Tooltip title={endpoint}>
          <Text style={{ fontFamily: 'monospace', fontSize: '12px' }}>
            {endpoint}
          </Text>
        </Tooltip>
      )
    },
    {
      title: 'Status',
      key: 'status',
      width: 80,
      render: (_, record: ApiRequest) => {
        if (record.error) {
          return <Tag color="error">ERROR</Tag>;
        }
        if (record.response) {
          return (
            <Tag color={getStatusColor(record.response.status_code)}>
              {record.response.status_code}
            </Tag>
          );
        }
        return <Tag color="default">PENDING</Tag>;
      },
      filters: [
        { text: 'Success (2xx)', value: 'success' },
        { text: 'Error (4xx/5xx)', value: 'error' },
        { text: 'Failed', value: 'failed' }
      ],
      onFilter: (value, record) => {
        if (value === 'success') return record.response?.status_code && record.response.status_code >= 200 && record.response.status_code < 300;
        if (value === 'error') return record.response?.status_code && record.response.status_code >= 400;
        if (value === 'failed') return !!record.error;
        return false;
      }
    },
    {
      title: 'Duration',
      dataIndex: 'duration',
      key: 'duration',
      width: 80,
      render: (duration?: number) => (
        <Text style={{ fontSize: '12px', fontFamily: 'monospace' }}>
          {formatDuration(duration)}
        </Text>
      ),
      sorter: (a, b) => (a.duration || 0) - (b.duration || 0)
    },
    {
      title: 'Size',
      key: 'size',
      width: 80,
      render: (_, record: ApiRequest) => {
        if (!record.response) return <Text type="secondary">-</Text>;
        const bytes = record.response.size_bytes;
        const formatted = bytes < 1024 ? `${bytes}B` : 
                         bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)}KB` :
                         `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
        return (
          <Text style={{ fontSize: '12px', fontFamily: 'monospace' }}>
            {formatted}
          </Text>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_, record: ApiRequest) => (
        <Space size="small">
          <Tooltip title="Replay Request">
            <Button
              type="text"
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => onReplayRequest(record)}
            />
          </Tooltip>
          {record.response && (
            <Tooltip title="View Response">
              <Button
                type="text"
                size="small"
                icon={<EyeOutlined />}
                onClick={() => {
                  setSelectedRequest(record);
                  setShowResponseModal(true);
                }}
              />
            </Tooltip>
          )}
          <Tooltip title="Copy as cURL">
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => copyAsCurl(record)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const copyAsCurl = (request: ApiRequest) => {
    let curl = `curl -X ${request.method}`;
    
    // Add headers
    Object.entries(request.headers).forEach(([key, value]) => {
      curl += ` -H "${key}: ${value}"`;
    });
    
    // Add query parameters
    const url = new URL(request.endpoint, window.location.origin);
    Object.entries(request.query_params).forEach(([key, value]) => {
      if (value) url.searchParams.set(key, value);
    });
    
    // Add body for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(request.method) && request.body) {
      curl += ` -d '${request.body}'`;
    }
    
    curl += ` "${url.toString()}"`;
    
    navigator.clipboard.writeText(curl).then(() => {
      // Could show a success message
    });
  };

  const uniqueMethods = [...new Set(requests.map(r => r.method))];

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Search
            placeholder="Search requests..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: 250 }}
            allowClear
          />
          
          <Select
            placeholder="Filter by method"
            value={methodFilter}
            onChange={setMethodFilter}
            style={{ width: 120 }}
            allowClear
          >
            {uniqueMethods.map(method => (
              <Option key={method} value={method}>{method}</Option>
            ))}
          </Select>
          
          <Select
            placeholder="Filter by status"
            value={statusFilter}
            onChange={setStatusFilter}
            style={{ width: 140 }}
            allowClear
          >
            <Option value="success">Success (2xx)</Option>
            <Option value="error">Error (4xx/5xx)</Option>
            <Option value="failed">Failed</Option>
          </Select>
        </Space>
        
        <Space>
          <Text type="secondary">
            {filteredRequests.length} of {requests.length} requests
          </Text>
          {onClearHistory && (
            <Button
              size="small"
              icon={<DeleteOutlined />}
              onClick={onClearHistory}
              danger
            >
              Clear History
            </Button>
          )}
        </Space>
      </div>

      <Table
        dataSource={filteredRequests}
        columns={columns}
        rowKey="id"
        size="small"
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showQuickJumper: true,
          showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} requests`
        }}
        scroll={{ x: 800 }}
        locale={{
          emptyText: (
            <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
              <HistoryOutlined style={{ fontSize: 48, marginBottom: 16 }} />
              <div>No request history</div>
              <div style={{ fontSize: '12px', marginTop: 8 }}>
                Make some API requests to see them here
              </div>
            </div>
          )
        }}
      />

      <Modal
        title={
          selectedRequest && (
            <Space>
              <Tag color={getMethodColor(selectedRequest.method)}>
                {selectedRequest.method}
              </Tag>
              <Text style={{ fontFamily: 'monospace' }}>
                {selectedRequest.endpoint}
              </Text>
            </Space>
          )
        }
        open={showResponseModal}
        onCancel={() => {
          setShowResponseModal(false);
          setSelectedRequest(null);
        }}
        footer={null}
        width={800}
        style={{ top: 20 }}
      >
        {selectedRequest?.response && (
          <ResponseViewer 
            response={selectedRequest.response} 
            duration={selectedRequest.duration}
          />
        )}
        
        {selectedRequest?.error && (
          <div style={{ padding: 16, background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: 6 }}>
            <Text type="danger" strong>Error:</Text>
            <div style={{ marginTop: 8, fontFamily: 'monospace', fontSize: '12px' }}>
              {selectedRequest.error}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default RequestHistory;