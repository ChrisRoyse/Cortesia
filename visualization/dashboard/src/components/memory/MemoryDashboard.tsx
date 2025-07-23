import React, { useState, useEffect } from 'react';
import { Card, Tabs, Row, Col, Statistic, Tag, Progress, Space, Typography, Alert } from 'antd';
import { 
  DatabaseOutlined, 
  ThunderboltOutlined, 
  NodeIndexOutlined, 
  WifiOutlined, 
  DisconnectOutlined 
} from '@ant-design/icons';
import { SDRStorageVisualization } from './SDRStorageVisualization';
import { KnowledgeGraphTreemap } from './KnowledgeGraphTreemap';
import { ZeroCopyMonitor } from './ZeroCopyMonitor';
import { MemoryFlowVisualization } from './MemoryFlowVisualization';
import { CognitiveLayerMemoryVisualization } from './CognitiveLayerMemory';
import {
  SDRStorage,
  KnowledgeGraphMemory,
  ZeroCopyMetrics,
  MemoryFlow,
  CognitiveLayerMemory,
  MemoryPressure
} from '../../types/memory';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface MemoryDashboardProps {
  wsUrl?: string;
  className?: string;
}

export const MemoryDashboard: React.FC<MemoryDashboardProps> = ({ 
  wsUrl = 'ws://localhost:8080', 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Memory data states
  const [sdrStorage, setSDRStorage] = useState<SDRStorage>(generateMockSDRStorage());
  const [knowledgeGraphMemory, setKnowledgeGraphMemory] = useState<KnowledgeGraphMemory>(generateMockKnowledgeGraphMemory());
  const [zeroCopyMetrics, setZeroCopyMetrics] = useState<ZeroCopyMetrics>(generateMockZeroCopyMetrics());
  const [zeroCopyHistory, setZeroCopyHistory] = useState<ZeroCopyMetrics[]>([generateMockZeroCopyMetrics()]);
  const [memoryFlows, setMemoryFlows] = useState<MemoryFlow[]>(generateMockMemoryFlows());
  const [cognitiveMemory, setCognitiveMemory] = useState<CognitiveLayerMemory>(generateMockCognitiveMemory());
  const [memoryPressure, setMemoryPressure] = useState<MemoryPressure>(generateMockMemoryPressure());

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setIsConnected(true);
          console.log('Connected to memory monitoring WebSocket');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleMemoryUpdate(data);
            setLastUpdate(new Date());
          } catch (error) {
            console.error('Error parsing memory data:', error);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          console.log('Disconnected from memory monitoring WebSocket');
          reconnectTimer = window.setTimeout(connect, 5000);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
      } catch (error) {
        console.error('Failed to connect:', error);
        reconnectTimer = window.setTimeout(connect, 5000);
      }
    };

    connect();

    // Simulate updates if not connected
    const simulationInterval = setInterval(() => {
      if (!isConnected) {
        simulateMemoryUpdates();
      }
    }, 2000);

    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      clearInterval(simulationInterval);
    };
  }, [wsUrl, isConnected]);

  const handleMemoryUpdate = (data: any) => {
    if (data.sdrStorage) setSDRStorage(data.sdrStorage);
    if (data.knowledgeGraphMemory) setKnowledgeGraphMemory(data.knowledgeGraphMemory);
    if (data.zeroCopyMetrics) {
      setZeroCopyMetrics(data.zeroCopyMetrics);
      setZeroCopyHistory(prev => [...prev.slice(-49), data.zeroCopyMetrics]);
    }
    if (data.memoryFlows) setMemoryFlows(prev => [...prev.slice(-50), ...data.memoryFlows]);
    if (data.cognitiveMemory) setCognitiveMemory(data.cognitiveMemory);
    if (data.memoryPressure) setMemoryPressure(data.memoryPressure);
  };

  const simulateMemoryUpdates = () => {
    // Simulate SDR storage changes
    setSDRStorage(prev => ({
      ...prev,
      activeSDRs: prev.activeSDRs + Math.floor(Math.random() * 10 - 5),
      totalMemoryBytes: prev.totalMemoryBytes + Math.floor(Math.random() * 1000000 - 500000),
      fragmentationLevel: Math.max(0, Math.min(1, prev.fragmentationLevel + (Math.random() * 0.02 - 0.01)))
    }));

    // Simulate zero-copy metrics
    const newZeroCopyMetrics = {
      ...zeroCopyMetrics,
      totalOperations: zeroCopyMetrics.totalOperations + Math.floor(Math.random() * 100),
      savedBytes: zeroCopyMetrics.savedBytes + Math.floor(Math.random() * 1000000),
      copyOnWriteEvents: zeroCopyMetrics.copyOnWriteEvents + Math.floor(Math.random() * 5)
    };
    setZeroCopyMetrics(newZeroCopyMetrics);
    setZeroCopyHistory(prev => [...prev.slice(-49), newZeroCopyMetrics]);

    // Simulate memory flows
    const newFlow: MemoryFlow = {
      timestamp: Date.now(),
      source: ['cortical', 'subcortical', 'cache', 'index'][Math.floor(Math.random() * 4)],
      target: ['cortical', 'subcortical', 'cache', 'embedding'][Math.floor(Math.random() * 4)],
      bytes: Math.floor(Math.random() * 1000000),
      operation: ['allocate', 'free', 'copy', 'share'][Math.floor(Math.random() * 4)] as any,
      duration: Math.random() * 100
    };
    setMemoryFlows(prev => [...prev.slice(-99), newFlow]);
  };

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  const totalMemory = 
    sdrStorage.totalMemoryBytes +
    Object.values(knowledgeGraphMemory).reduce((sum, block) => sum + block.size, 0) +
    cognitiveMemory.subcortical.total +
    cognitiveMemory.cortical.total +
    cognitiveMemory.workingMemory.capacity;

  const renderOverview = () => (
    <div>
      {/* Connection Status */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          {isConnected ? <WifiOutlined style={{ color: '#52c41a' }} /> : <DisconnectOutlined style={{ color: '#ff4d4f' }} />}
          <Text type={isConnected ? 'success' : 'danger'}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </Text>
          <Text type="secondary">
            Last update: {lastUpdate.toLocaleTimeString()}
          </Text>
        </Space>
      </Card>

      {/* System Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Memory"
              value={formatBytes(totalMemory)}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Memory Pressure"
              value={memoryPressure.level.toUpperCase()}
              valueStyle={{
                color: memoryPressure.level === 'critical' ? '#ff4d4f' :
                       memoryPressure.level === 'high' ? '#fa8c16' :
                       memoryPressure.level === 'medium' ? '#fadb14' :
                       '#52c41a'
              }}
            />
            <Progress 
              percent={memoryPressure.percentage} 
              size="small"
              strokeColor={
                memoryPressure.level === 'critical' ? '#ff4d4f' :
                memoryPressure.level === 'high' ? '#fa8c16' :
                memoryPressure.level === 'medium' ? '#fadb14' :
                '#52c41a'
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Zero-Copy Savings"
              value={formatBytes(zeroCopyMetrics.savedBytes)}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Fragmentation"
              value={(sdrStorage.fragmentationLevel * 100).toFixed(1)}
              suffix="%"
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Quick Stats */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} md={8}>
          <Card title="SDR Storage" size="small">
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Active SDRs: </Text>
              <Text strong>{sdrStorage.activeSDRs.toLocaleString()}</Text>
            </div>
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Compression: </Text>
              <Text strong>{(sdrStorage.compressionRatio * 100).toFixed(1)}%</Text>
            </div>
            <div>
              <Text type="secondary">Avg Sparsity: </Text>
              <Text strong>{(sdrStorage.averageSparsity * 100).toFixed(1)}%</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card title="Knowledge Graph" size="small">
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Entities: </Text>
              <Text strong>{formatBytes(knowledgeGraphMemory.entities.used)}</Text>
            </div>
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Relations: </Text>
              <Text strong>{formatBytes(knowledgeGraphMemory.relations.used)}</Text>
            </div>
            <div>
              <Text type="secondary">Cache Hit Rate: </Text>
              <Text strong>87.3%</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card title="Cognitive Layers" size="small">
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Subcortical: </Text>
              <Text strong>{((cognitiveMemory.subcortical.used / cognitiveMemory.subcortical.total) * 100).toFixed(1)}%</Text>
            </div>
            <div style={{ marginBottom: 8 }}>
              <Text type="secondary">Cortical: </Text>
              <Text strong>{((cognitiveMemory.cortical.used / cognitiveMemory.cortical.total) * 100).toFixed(1)}%</Text>
            </div>
            <div>
              <Text type="secondary">Working Memory: </Text>
              <Text strong>{cognitiveMemory.workingMemory.buffers.length} buffers</Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Memory Pressure Recommendations */}
      {memoryPressure.recommendations.length > 0 && (
        <Alert
          message="Memory Optimization Recommendations"
          description={
            <ul style={{ marginTop: 8, marginBottom: 0 }}>
              {memoryPressure.recommendations.map((rec, i) => (
                <li key={i}>{rec}</li>
              ))}
            </ul>
          }
          type="warning"
          showIcon
        />
      )}
    </div>
  );

  return (
    <div className={className}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>Memory & Storage Monitoring</Title>
        <Text type="secondary">Real-time memory analysis and optimization</Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={
            <span>
              <DatabaseOutlined />
              Overview
            </span>
          } 
          key="overview"
        >
          {renderOverview()}
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <NodeIndexOutlined />
              SDR Storage
            </span>
          } 
          key="sdr"
        >
          <SDRStorageVisualization storage={sdrStorage} />
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <DatabaseOutlined />
              Knowledge Graph
            </span>
          } 
          key="knowledge"
        >
          <KnowledgeGraphTreemap memory={knowledgeGraphMemory} />
        </TabPane>
        
        <TabPane 
          tab={
            <span>
              <ThunderboltOutlined />
              Zero-Copy
            </span>
          } 
          key="zerocopy"
        >
          <ZeroCopyMonitor metrics={zeroCopyMetrics} history={zeroCopyHistory} />
        </TabPane>
        
        <TabPane tab="Memory Flow" key="flow">
          <MemoryFlowVisualization flows={memoryFlows} />
        </TabPane>
        
        <TabPane tab="Cognitive Layers" key="cognitive">
          <CognitiveLayerMemoryVisualization memory={cognitiveMemory} />
        </TabPane>
      </Tabs>
    </div>
  );
};

// Mock data generators (same as original)
function generateMockSDRStorage(): SDRStorage {
  return {
    totalSDRs: 125000,
    activeSDRs: 45000,
    archivedSDRs: 80000,
    totalMemoryBytes: 2147483648, // 2GB
    averageSparsity: 0.02,
    compressionRatio: 0.65,
    fragmentationLevel: 0.15,
    storageBlocks: Array.from({ length: 20 }, (_, i) => ({
      id: `block-${i}`,
      size: 134217728, // 128MB
      used: Math.floor(Math.random() * 134217728),
      fragmented: Math.floor(Math.random() * 13421772),
      patterns: Math.floor(Math.random() * 5000),
      lastAccess: Date.now() - Math.floor(Math.random() * 3600000),
      compressionType: ['overlap', 'dictionary', 'none'][Math.floor(Math.random() * 3)] as any
    }))
  };
}

function generateMockKnowledgeGraphMemory(): KnowledgeGraphMemory {
  return {
    entities: {
      name: 'Entities',
      size: 1073741824, // 1GB
      used: 805306368,
      metadata: {
        accessCount: 1250000,
        lastAccess: Date.now(),
        fragmentation: 0.08
      }
    },
    relations: {
      name: 'Relations',
      size: 536870912, // 512MB
      used: 402653184,
      metadata: {
        accessCount: 890000,
        lastAccess: Date.now() - 1000,
        fragmentation: 0.12
      }
    },
    embeddings: {
      name: 'Embeddings',
      size: 2147483648, // 2GB
      used: 1610612736,
      metadata: {
        accessCount: 450000,
        lastAccess: Date.now() - 2000,
        fragmentation: 0.05
      }
    },
    indexes: {
      name: 'Indexes',
      size: 268435456, // 256MB
      used: 201326592,
      metadata: {
        accessCount: 2100000,
        lastAccess: Date.now(),
        fragmentation: 0.18
      }
    },
    cache: {
      name: 'Cache',
      size: 1073741824, // 1GB
      used: 536870912,
      metadata: {
        accessCount: 5400000,
        lastAccess: Date.now(),
        fragmentation: 0.25
      }
    }
  };
}

function generateMockZeroCopyMetrics(): ZeroCopyMetrics {
  return {
    enabled: true,
    totalOperations: 1250000,
    savedBytes: 5368709120, // 5GB
    copyOnWriteEvents: 12500,
    sharedRegions: 856,
    efficiency: 0.85
  };
}

function generateMockMemoryFlows(): MemoryFlow[] {
  const nodes = ['cortical', 'subcortical', 'cache', 'index', 'embedding', 'working'];
  const operations: Array<'allocate' | 'free' | 'copy' | 'share'> = ['allocate', 'free', 'copy', 'share'];
  
  return Array.from({ length: 50 }, () => ({
    timestamp: Date.now() - Math.floor(Math.random() * 60000),
    source: nodes[Math.floor(Math.random() * nodes.length)],
    target: nodes[Math.floor(Math.random() * nodes.length)],
    bytes: Math.floor(Math.random() * 10485760), // Up to 10MB
    operation: operations[Math.floor(Math.random() * operations.length)],
    duration: Math.random() * 100
  }));
}

function generateMockCognitiveMemory(): CognitiveLayerMemory {
  return {
    subcortical: {
      total: 536870912, // 512MB
      used: 402653184,
      components: {
        thalamus: 134217728,
        hippocampus: 100663296,
        amygdala: 83886080,
        basalGanglia: 83886080
      }
    },
    cortical: {
      total: 2147483648, // 2GB
      used: 1610612736,
      regions: {
        prefrontal: 536870912,
        temporal: 402653184,
        parietal: 335544320,
        occipital: 335544320
      }
    },
    workingMemory: {
      capacity: 67108864, // 64MB
      used: 50331648,
      buffers: Array.from({ length: 8 }, (_, i) => ({
        id: `buffer-${i}`,
        content: `Working memory content ${i}`,
        size: 6291456, // 6MB
        age: Math.floor(Math.random() * 10000),
        accessCount: Math.floor(Math.random() * 100),
        priority: Math.random()
      }))
    }
  };
}

function generateMockMemoryPressure(): MemoryPressure {
  return {
    level: 'medium',
    percentage: 65,
    swapUsed: 134217728, // 128MB
    pageCache: 268435456, // 256MB
    recommendations: [
      'Consider clearing unused embeddings cache',
      'Archive old SDR patterns to reduce active memory',
      'Enable more aggressive garbage collection'
    ]
  };
}