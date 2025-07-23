import React, { useState, useEffect } from 'react';
import { Card, Table, Tag, Space, Button, Modal, Descriptions, Typography, Alert, Input, Select } from 'antd';
import { 
  AppstoreOutlined,
  EyeOutlined,
  SettingOutlined,
  BugOutlined,
  RocketOutlined,
  SearchOutlined
} from '@ant-design/icons';
import { getRegisteredComponents } from '@/integration/VisualizationCore';

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;
const { Option } = Select;

interface RegisteredComponent {
  id: string;
  component: React.ComponentType<any>;
  metadata?: {
    name: string;
    description: string;
    phase: string;
    version: string;
    author: string;
    dependencies: string[];
    props: Record<string, any>;
    status: 'active' | 'inactive' | 'error';
    lastUpdated: string;
  };
}

export const ComponentRegistry: React.FC = () => {
  const [components, setComponents] = useState<RegisteredComponent[]>([]);
  const [filteredComponents, setFilteredComponents] = useState<RegisteredComponent[]>([]);
  const [selectedComponent, setSelectedComponent] = useState<RegisteredComponent | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [phaseFilter, setPhaseFilter] = useState<string>('all');

  useEffect(() => {
    loadComponents();
  }, []);

  useEffect(() => {
    filterComponents();
  }, [components, searchTerm, phaseFilter]);

  const loadComponents = () => {
    // Get registered components and add mock metadata for demonstration
    const registeredComponents = getRegisteredComponents();
    
    const mockComponents: RegisteredComponent[] = [
      {
        id: 'memory-dashboard',
        component: {} as any,
        metadata: {
          name: 'Memory Dashboard',
          description: 'Comprehensive memory system monitoring and visualization',
          phase: 'Phase 7',
          version: '1.2.0',
          author: 'LLMKG Team',
          dependencies: ['@phase7/components', 'd3', 'react'],
          props: {
            updateInterval: 1000,
            enableRealTime: true,
            theme: 'dark'
          },
          status: 'active',
          lastUpdated: '2024-01-15T10:30:00Z'
        }
      },
      {
        id: 'sdr-storage-visualization',
        component: {} as any,
        metadata: {
          name: 'SDR Storage Visualization',
          description: 'Sparse Distributed Representation storage analysis and fragmentation monitoring',
          phase: 'Phase 7',
          version: '1.1.5',
          author: 'LLMKG Team',
          dependencies: ['@phase7/types', 'd3', 'lodash'],
          props: {
            storageBlocks: [],
            fragmentationThreshold: 0.3
          },
          status: 'active',
          lastUpdated: '2024-01-12T14:20:00Z'
        }
      },
      {
        id: 'knowledge-graph-treemap',
        component: {} as any,
        metadata: {
          name: 'Knowledge Graph Treemap',
          description: 'Hierarchical visualization of knowledge graph memory allocation',
          phase: 'Phase 7',
          version: '1.0.8',
          author: 'LLMKG Team',
          dependencies: ['@phase7/types', 'd3'],
          props: {
            width: 800,
            height: 600,
            showTooltips: true
          },
          status: 'active',
          lastUpdated: '2024-01-10T09:15:00Z'
        }
      },
      {
        id: 'cognitive-pattern-dashboard',
        component: {} as any,
        metadata: {
          name: 'Cognitive Pattern Dashboard',
          description: 'Brain-inspired cognitive pattern analysis and visualization',
          phase: 'Phase 8',
          version: '2.0.1',
          author: 'LLMKG Team',
          dependencies: ['@phase8/components', 'three', 'd3'],
          props: {
            enable3D: true,
            patternTypes: ['convergent', 'divergent', 'lateral']
          },
          status: 'active',
          lastUpdated: '2024-01-14T16:45:00Z'
        }
      },
      {
        id: 'pattern-activation-3d',
        component: {} as any,
        metadata: {
          name: 'Pattern Activation 3D',
          description: '3D visualization of cognitive pattern activation in real-time',
          phase: 'Phase 8',
          version: '1.5.0',
          author: 'LLMKG Team',
          dependencies: ['@phase8/types', 'three', 'react-three-fiber'],
          props: {
            enableRotation: true,
            animationSpeed: 1.0,
            showConnections: true
          },
          status: 'active',
          lastUpdated: '2024-01-13T11:30:00Z'
        }
      },
      {
        id: 'debugging-dashboard',
        component: {} as any,
        metadata: {
          name: 'Advanced Debugging Dashboard',
          description: 'Comprehensive debugging tools with time-travel capabilities',
          phase: 'Phase 9',
          version: '1.3.2',
          author: 'LLMKG Team',
          dependencies: ['@phase9/components', 'monaco-editor'],
          props: {
            enableTimeTravel: true,
            maxStateHistory: 1000
          },
          status: 'active',
          lastUpdated: '2024-01-11T13:20:00Z'
        }
      },
      {
        id: 'distributed-tracing',
        component: {} as any,
        metadata: {
          name: 'Distributed Tracing',
          description: 'Visual tracing of distributed operations across the LLMKG system',
          phase: 'Phase 9',
          version: '1.1.0',
          author: 'LLMKG Team',
          dependencies: ['@phase9/types', 'd3'],
          props: {
            traceDepth: 10,
            showTimeline: true
          },
          status: 'active',
          lastUpdated: '2024-01-09T15:10:00Z'
        }
      },
      {
        id: 'unified-dashboard',
        component: {} as any,
        metadata: {
          name: 'Unified Dashboard',
          description: 'Main dashboard integrating all visualization phases',
          phase: 'Phase 10',
          version: '1.0.0',
          author: 'LLMKG Team',
          dependencies: ['@phase7/components', '@phase8/components', '@phase9/components'],
          props: {
            enableAllPhases: true,
            layout: 'responsive'
          },
          status: 'active',
          lastUpdated: '2024-01-15T18:00:00Z'
        }
      }
    ];

    setComponents(mockComponents);
  };

  const filterComponents = () => {
    let filtered = components;

    if (searchTerm) {
      filtered = filtered.filter(comp => 
        comp.metadata?.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        comp.metadata?.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        comp.id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (phaseFilter !== 'all') {
      filtered = filtered.filter(comp => 
        comp.metadata?.phase === phaseFilter
      );
    }

    setFilteredComponents(filtered);
  };

  const handleViewComponent = (component: RegisteredComponent) => {
    setSelectedComponent(component);
    setModalVisible(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'default';
      case 'error':
        return 'error';
      default:
        return 'processing';
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'Phase 7':
        return 'blue';
      case 'Phase 8':
        return 'green';
      case 'Phase 9':
        return 'orange';
      case 'Phase 10':
        return 'purple';
      case 'Phase 11':
        return 'gold';
      default:
        return 'default';
    }
  };

  const phases = ['Phase 7', 'Phase 8', 'Phase 9', 'Phase 10', 'Phase 11'];

  const columns = [
    {
      title: 'Component',
      key: 'component',
      render: (_, record: RegisteredComponent) => (
        <div>
          <Text strong>{record.metadata?.name || record.id}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.metadata?.description}
          </Text>
        </div>
      ),
    },
    {
      title: 'Phase',
      dataIndex: ['metadata', 'phase'],
      key: 'phase',
      render: (phase: string) => (
        <Tag color={getPhaseColor(phase)}>{phase}</Tag>
      ),
      filters: phases.map(phase => ({ text: phase, value: phase })),
      onFilter: (value: any, record: RegisteredComponent) => 
        record.metadata?.phase === value,
    },
    {
      title: 'Version',
      dataIndex: ['metadata', 'version'],
      key: 'version',
      render: (version: string) => (
        <Text style={{ fontFamily: 'monospace' }}>{version}</Text>
      ),
    },
    {
      title: 'Status',
      dataIndex: ['metadata', 'status'],
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status?.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Last Updated',
      dataIndex: ['metadata', 'lastUpdated'],
      key: 'lastUpdated',
      render: (date: string) => (
        <Text style={{ fontSize: '12px' }}>
          {new Date(date).toLocaleDateString()}
        </Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: RegisteredComponent) => (
        <Space>
          <Button 
            icon={<EyeOutlined />} 
            size="small"
            onClick={() => handleViewComponent(record)}
          >
            View
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            size="small"
            disabled
          >
            Configure
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '16px 0' }}>
      <Title level={2}>
        <AppstoreOutlined /> Component Registry
      </Title>
      <Paragraph>
        Manage and monitor all registered visualization components across different phases.
        View component details, dependencies, and configuration options.
      </Paragraph>

      <Alert
        message="Component Registry"
        description="This registry shows all available visualization components. Components are automatically registered when loaded and can be configured through this interface."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Card>
        <Space style={{ marginBottom: 16, width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <Search
              placeholder="Search components..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: 300 }}
              prefix={<SearchOutlined />}
            />
            <Select
              value={phaseFilter}
              onChange={setPhaseFilter}
              style={{ width: 150 }}
            >
              <Option value="all">All Phases</Option>
              {phases.map(phase => (
                <Option key={phase} value={phase}>{phase}</Option>
              ))}
            </Select>
          </Space>

          <Space>
            <Text type="secondary">
              {filteredComponents.length} of {components.length} components
            </Text>
          </Space>
        </Space>

        <Table
          dataSource={filteredComponents}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} components`,
          }}
        />
      </Card>

      <Modal
        title={selectedComponent?.metadata?.name || 'Component Details'}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedComponent && (
          <div>
            <Descriptions title="Component Information" column={2}>
              <Descriptions.Item label="ID">
                <Text code>{selectedComponent.id}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Version">
                <Text>{selectedComponent.metadata?.version}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Phase">
                <Tag color={getPhaseColor(selectedComponent.metadata?.phase || '')}>
                  {selectedComponent.metadata?.phase}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={getStatusColor(selectedComponent.metadata?.status || '')}>
                  {selectedComponent.metadata?.status?.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Author">
                <Text>{selectedComponent.metadata?.author}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="Last Updated">
                <Text>
                  {selectedComponent.metadata?.lastUpdated ? 
                    new Date(selectedComponent.metadata.lastUpdated).toLocaleString() : 
                    'Unknown'
                  }
                </Text>
              </Descriptions.Item>
            </Descriptions>

            <Descriptions title="Description" column={1}>
              <Descriptions.Item label="">
                <Text>{selectedComponent.metadata?.description}</Text>
              </Descriptions.Item>
            </Descriptions>

            <Descriptions title="Dependencies" column={1}>
              <Descriptions.Item label="">
                <Space wrap>
                  {selectedComponent.metadata?.dependencies?.map(dep => (
                    <Tag key={dep}>{dep}</Tag>
                  )) || <Text type="secondary">No dependencies</Text>}
                </Space>
              </Descriptions.Item>
            </Descriptions>

            <Descriptions title="Default Props" column={1}>
              <Descriptions.Item label="">
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: '8px', 
                  borderRadius: '4px',
                  fontSize: '12px',
                  overflow: 'auto'
                }}>
                  {JSON.stringify(selectedComponent.metadata?.props || {}, null, 2)}
                </pre>
              </Descriptions.Item>
            </Descriptions>
          </div>
        )}
      </Modal>
    </div>
  );
};