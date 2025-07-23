import React, { useState, useEffect } from 'react';
import { 
  Card, Row, Col, Button, Input, Select, Space, Typography, Tabs, Table, 
  Tag, Badge, Alert, Spin, Tooltip, Form, Modal, Tree, Progress, Divider, 
  Switch, Slider, notification
} from 'antd';
import { 
  ApiOutlined, SendOutlined, HistoryOutlined, SettingOutlined, 
  PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined, 
  ExportOutlined, ImportOutlined, CodeOutlined, BugOutlined,
  ClockCircleOutlined, CheckCircleOutlined, CloseCircleOutlined
} from '@ant-design/icons';
import { useRealTimeData } from '../../hooks/useRealTimeData';
import { JSONEditor } from '../../components/JSONEditor';
import { ResponseViewer } from '../../components/ResponseViewer';
import { RequestHistory } from '../../components/RequestHistory';
import { TestSuiteRunner } from '../../components/testing/TestSuiteRunner';
import { TestExecutionTracker } from '../../services/TestExecutionTracker';
import { apiService, ApiEndpoint as IApiEndpoint } from '../../services/api';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;

interface ApiEndpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS';
  handler_function: string;
  parameters: ApiParameter[];
  response_schema?: string;
  auth_required: boolean;
  rate_limit?: RateLimit;
  documentation: string;
  tags: string[];
}

interface ApiParameter {
  name: string;
  param_type: 'Query' | 'Path' | 'Body' | 'Header' | 'FormData';
  data_type: string;
  required: boolean;
  description: string;
  example?: string;
}

interface RateLimit {
  requests_per_window: number;
  window_duration: { secs: number; nanos: number };
}

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

interface TestSuite {
  name: string;
  tests: ApiTestCase[];
  environment: Record<string, string>;
}

interface ApiTestCase {
  id: string;
  name: string;
  endpoint: string;
  method: string;
  headers: Record<string, string>;
  query_params: Record<string, string>;
  body?: string;
  expected_status?: number;
  assertions: TestAssertion[];
  timeout: number;
}

interface TestAssertion {
  type: 'status' | 'header' | 'body' | 'response_time';
  field?: string;
  operator: 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater_than' | 'less_than';
  expected: string;
}

export const APITestingPage: React.FC = () => {
  const { data: apiData, loading } = useRealTimeData<any>('api_metrics');
  const [selectedEndpoint, setSelectedEndpoint] = useState<ApiEndpoint | null>(null);
  const [requestData, setRequestData] = useState({
    method: 'GET',
    url: '',
    headers: {} as Record<string, string>,
    queryParams: {} as Record<string, string>,
    body: ''
  });
  const [response, setResponse] = useState<any>(null);
  const [requestHistory, setRequestHistory] = useState<ApiRequest[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('manual');
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [currentTest, setCurrentTest] = useState<ApiTestCase | null>(null);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [testExecutionTracker] = useState(() => new TestExecutionTracker());
  const [discoveredEndpoints, setDiscoveredEndpoints] = useState<IApiEndpoint[]>([]);
  const [loadingEndpoints, setLoadingEndpoints] = useState(true);

  const httpMethods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'];

  const endpoints = apiData?.endpoints ? Object.values(apiData.endpoints) as ApiEndpoint[] : [];

  useEffect(() => {
    // Load test suites from storage or API
    loadTestSuites();
    // Discover real API endpoints
    discoverEndpoints();
  }, []);
  
  const discoverEndpoints = async () => {
    setLoadingEndpoints(true);
    try {
      const result = await apiService.getApiDiscovery();
      if (result.status === 'success' && result.data) {
        setDiscoveredEndpoints(result.data.endpoints);
      } else {
        notification.error({
          message: 'Failed to discover API endpoints',
          description: result.error || 'Unknown error',
        });
      }
    } catch (error) {
      console.error('Failed to discover endpoints:', error);
    } finally {
      setLoadingEndpoints(false);
    }
  };

  const loadTestSuites = () => {
    // Mock test suites for demonstration
    const mockSuites: TestSuite[] = [
      {
        name: 'Health Check Tests',
        environment: { baseUrl: 'http://localhost:3001' },
        tests: [
          {
            id: 'health-1',
            name: 'Health Endpoint Available',
            endpoint: '/api/health',
            method: 'GET',
            headers: {},
            query_params: {},
            expected_status: 200,
            assertions: [
              { type: 'status', operator: 'equals', expected: '200' },
              { type: 'response_time', operator: 'less_than', expected: '100' }
            ],
            timeout: 5000
          }
        ]
      },
      {
        name: 'Metrics API Tests',
        environment: { baseUrl: 'http://localhost:3001' },
        tests: [
          {
            id: 'metrics-1',
            name: 'Get System Metrics',
            endpoint: '/api/metrics',
            method: 'GET',
            headers: {},
            query_params: {},
            expected_status: 200,
            assertions: [
              { type: 'status', operator: 'equals', expected: '200' },
              { type: 'header', field: 'content-type', operator: 'contains', expected: 'application/json' }
            ],
            timeout: 5000
          }
        ]
      }
    ];
    setTestSuites(mockSuites);
  };

  const sendRequest = async () => {
    setIsLoading(true);
    const startTime = Date.now();

    try {
      const url = new URL(requestData.url, window.location.origin);
      
      // Add query parameters
      Object.entries(requestData.queryParams).forEach(([key, value]) => {
        if (value) url.searchParams.set(key, value);
      });

      const options: RequestInit = {
        method: requestData.method,
        headers: {
          'Content-Type': 'application/json',
          ...requestData.headers
        }
      };

      if (['POST', 'PUT', 'PATCH'].includes(requestData.method) && requestData.body) {
        options.body = requestData.body;
      }

      const response = await fetch(url.toString(), options);
      const duration = Date.now() - startTime;
      
      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      let responseBody: string;
      const contentType = response.headers.get('content-type');
      
      if (contentType?.includes('application/json')) {
        responseBody = JSON.stringify(await response.json(), null, 2);
      } else {
        responseBody = await response.text();
      }

      const apiResponse: ApiResponse = {
        status_code: response.status,
        headers: responseHeaders,
        body: responseBody,
        size_bytes: new Blob([responseBody]).size
      };

      const request: ApiRequest = {
        id: Date.now().toString(),
        endpoint: requestData.url,
        method: requestData.method,
        timestamp: startTime,
        headers: requestData.headers,
        query_params: requestData.queryParams,
        body: requestData.body || undefined,
        response: apiResponse,
        duration
      };

      setResponse(apiResponse);
      setRequestHistory(prev => [request, ...prev.slice(0, 49)]); // Keep last 50 requests

      notification.success({
        message: 'Request Completed',
        description: `${requestData.method} ${requestData.url} - ${response.status} (${duration}ms)`
      });

    } catch (error) {
      const request: ApiRequest = {
        id: Date.now().toString(),
        endpoint: requestData.url,
        method: requestData.method,
        timestamp: startTime,
        headers: requestData.headers,
        query_params: requestData.queryParams,
        body: requestData.body || undefined,
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      setRequestHistory(prev => [request, ...prev.slice(0, 49)]);
      setResponse(null);

      notification.error({
        message: 'Request Failed',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const runTestSuite = async (suite: TestSuite) => {
    setIsRunningTests(true);
    const results: Record<string, any> = {};

    for (const test of suite.tests) {
      try {
        const result = await runSingleTest(test, suite.environment);
        results[test.id] = result;
      } catch (error) {
        results[test.id] = {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    }

    setTestResults(results);
    setIsRunningTests(false);

    const passedTests = Object.values(results).filter((r: any) => r.success).length;
    const totalTests = suite.tests.length;

    notification.info({
      message: 'Test Suite Completed',
      description: `${passedTests}/${totalTests} tests passed`
    });
  };

  const runSingleTest = async (test: ApiTestCase, environment: Record<string, string>) => {
    const startTime = Date.now();
    const baseUrl = environment.baseUrl || window.location.origin;
    const url = new URL(test.endpoint, baseUrl);

    // Add query parameters
    Object.entries(test.query_params).forEach(([key, value]) => {
      if (value) url.searchParams.set(key, value);
    });

    const options: RequestInit = {
      method: test.method,
      headers: {
        'Content-Type': 'application/json',
        ...test.headers
      }
    };

    if (['POST', 'PUT', 'PATCH'].includes(test.method) && test.body) {
      options.body = test.body;
    }

    const response = await fetch(url.toString(), options);
    const duration = Date.now() - startTime;
    const responseText = await response.text();

    // Run assertions
    const assertionResults = test.assertions.map(assertion => {
      return runAssertion(assertion, {
        status: response.status,
        headers: Object.fromEntries(response.headers.entries()),
        body: responseText,
        responseTime: duration
      });
    });

    const success = assertionResults.every(result => result.passed);

    return {
      success,
      status: response.status,
      responseTime: duration,
      assertions: assertionResults,
      response: responseText
    };
  };

  const runAssertion = (assertion: TestAssertion, response: any) => {
    let actualValue: any;
    
    switch (assertion.type) {
      case 'status':
        actualValue = response.status.toString();
        break;
      case 'header':
        actualValue = response.headers[assertion.field || ''];
        break;
      case 'body':
        actualValue = response.body;
        break;
      case 'response_time':
        actualValue = response.responseTime.toString();
        break;
      default:
        return { passed: false, message: 'Unknown assertion type' };
    }

    const expected = assertion.expected;
    let passed = false;

    switch (assertion.operator) {
      case 'equals':
        passed = actualValue === expected;
        break;
      case 'not_equals':
        passed = actualValue !== expected;
        break;
      case 'contains':
        passed = actualValue?.includes(expected);
        break;
      case 'not_contains':
        passed = !actualValue?.includes(expected);
        break;
      case 'greater_than':
        passed = parseFloat(actualValue) > parseFloat(expected);
        break;
      case 'less_than':
        passed = parseFloat(actualValue) < parseFloat(expected);
        break;
    }

    return {
      passed,
      message: passed ? 'Passed' : `Expected ${actualValue} ${assertion.operator} ${expected}`,
      actual: actualValue,
      expected
    };
  };

  const handleEndpointSelect = (endpoint: ApiEndpoint) => {
    setSelectedEndpoint(endpoint);
    setRequestData({
      method: endpoint.method,
      url: endpoint.path,
      headers: endpoint.auth_required ? { 'Authorization': 'Bearer <token>' } : {},
      queryParams: {},
      body: endpoint.method === 'POST' || endpoint.method === 'PUT' ? '{\n  \n}' : ''
    });
  };

  const addHeaderRow = () => {
    const key = `header_${Date.now()}`;
    setRequestData(prev => ({
      ...prev,
      headers: { ...prev.headers, [key]: '' }
    }));
  };

  const updateHeader = (oldKey: string, newKey: string, value: string) => {
    setRequestData(prev => {
      const newHeaders = { ...prev.headers };
      if (oldKey !== newKey) {
        delete newHeaders[oldKey];
      }
      newHeaders[newKey] = value;
      return { ...prev, headers: newHeaders };
    });
  };

  const removeHeader = (key: string) => {
    setRequestData(prev => {
      const newHeaders = { ...prev.headers };
      delete newHeaders[key];
      return { ...prev, headers: newHeaders };
    });
  };

  const endpointColumns = [
    {
      title: 'Method',
      dataIndex: 'method',
      key: 'method',
      width: 80,
      render: (method: string) => (
        <Tag color={
          method === 'GET' ? 'blue' :
          method === 'POST' ? 'green' :
          method === 'PUT' ? 'orange' :
          method === 'DELETE' ? 'red' : 'default'
        }>
          {method}
        </Tag>
      )
    },
    {
      title: 'Path',
      dataIndex: 'path',
      key: 'path',
      ellipsis: true
    },
    {
      title: 'Description',
      dataIndex: 'documentation',
      key: 'documentation',
      ellipsis: true
    },
    {
      title: 'Auth',
      dataIndex: 'auth_required',
      key: 'auth_required',
      width: 60,
      render: (auth: boolean) => auth ? <Tag color="orange">Auth</Tag> : <Tag>Public</Tag>
    },
    {
      title: 'Action',
      key: 'action',
      width: 80,
      render: (_: any, record: ApiEndpoint) => (
        <Button 
          type="link" 
          size="small" 
          onClick={() => handleEndpointSelect(record)}
        >
          Test
        </Button>
      )
    }
  ];

  const renderManualTestingTab = () => (
    <Row gutter={[16, 16]}>
      <Col span={8}>
        <Card title="API Endpoints" size="small">
          <Table
            dataSource={endpoints}
            columns={endpointColumns}
            size="small"
            pagination={{ pageSize: 10 }}
            rowKey="path"
          />
        </Card>
      </Col>
      
      <Col span={16}>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Card 
            title="Request Builder" 
            size="small"
            extra={
              <Button 
                type="primary" 
                icon={<SendOutlined />} 
                onClick={sendRequest}
                loading={isLoading}
              >
                Send Request
              </Button>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Select
                  value={requestData.method}
                  onChange={(value) => setRequestData(prev => ({ ...prev, method: value }))}
                  style={{ width: 100 }}
                >
                  {httpMethods.map(method => (
                    <Option key={method} value={method}>{method}</Option>
                  ))}
                </Select>
                <Input
                  placeholder="Enter URL (e.g., /api/health)"
                  value={requestData.url}
                  onChange={(e) => setRequestData(prev => ({ ...prev, url: e.target.value }))}
                  style={{ flex: 1 }}
                />
              </Space>

              <Tabs size="small">
                <TabPane tab="Headers" key="headers">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {Object.entries(requestData.headers).map(([key, value]) => (
                      <Space key={key} style={{ width: '100%' }}>
                        <Input
                          placeholder="Header name"
                          value={key}
                          onChange={(e) => updateHeader(key, e.target.value, value)}
                          style={{ width: 150 }}
                        />
                        <Input
                          placeholder="Header value"
                          value={value}
                          onChange={(e) => updateHeader(key, key, e.target.value)}
                          style={{ flex: 1 }}
                        />
                        <Button 
                          size="small" 
                          danger 
                          onClick={() => removeHeader(key)}
                        >
                          Remove
                        </Button>
                      </Space>
                    ))}
                    <Button size="small" onClick={addHeaderRow}>Add Header</Button>
                  </Space>
                </TabPane>
                
                <TabPane tab="Query Params" key="params">
                  <Text type="secondary">Query parameters (added to URL)</Text>
                </TabPane>
                
                {['POST', 'PUT', 'PATCH'].includes(requestData.method) && (
                  <TabPane tab="Body" key="body">
                    <TextArea
                      placeholder="Request body (JSON)"
                      value={requestData.body}
                      onChange={(e) => setRequestData(prev => ({ ...prev, body: e.target.value }))}
                      rows={6}
                      style={{ fontFamily: 'monospace' }}
                    />
                  </TabPane>
                )}
              </Tabs>
            </Space>
          </Card>

          {response && (
            <Card title="Response" size="small">
              <ResponseViewer response={response} />
            </Card>
          )}
        </Space>
      </Col>
    </Row>
  );

  const renderTestSuitesTab = () => (
    <TestSuiteRunner 
      tracker={testExecutionTracker}
      onTestComplete={(execution) => {
        notification.info({
          message: 'Test Suite Completed',
          description: `${execution.summary?.passed || 0} passed, ${execution.summary?.failed || 0} failed`,
          duration: 5
        });
      }}
    />
  );

  const renderHistoryTab = () => (
    <Card title="Request History">
      <RequestHistory requests={requestHistory} onReplayRequest={(request) => {
        setRequestData({
          method: request.method,
          url: request.endpoint,
          headers: request.headers,
          queryParams: request.query_params,
          body: request.body || ''
        });
        setActiveTab('manual');
      }} />
    </Card>
  );

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin size="large" />
        <div style={{ marginTop: 20 }}>Loading API data...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ApiOutlined style={{ marginRight: 8 }} />
          API Testing & Monitoring
        </Title>
        <Text type="secondary">
          Interactive API testing, monitoring, and validation
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
        <TabPane 
          tab={
            <span>
              <SendOutlined />
              Manual Testing
            </span>
          } 
          key="manual"
        >
          {renderManualTestingTab()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <PlayCircleOutlined />
              LLMKG Tests
            </span>
          } 
          key="suites"
        >
          {renderTestSuitesTab()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <HistoryOutlined />
              History
            </span>
          } 
          key="history"
        >
          {renderHistoryTab()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default APITestingPage;