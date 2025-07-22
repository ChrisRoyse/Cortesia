import React, { useState, useEffect } from 'react';
import {
  KnowledgeGraph3D,
  CognitivePatternViz,
  NeuralActivityHeatmap,
  MemorySystemChart,
  MetricCard,
  CPUMetricCard,
  MemoryMetricCard,
  LatencyMetricCard,
  ThroughputMetricCard,
  StatusIndicator,
  WebSocketStatusIndicator,
  DataGrid,
  Column,
} from '../components';
import { 
  LLMKGData, 
  KnowledgeNode, 
  KnowledgeEdge, 
  CognitivePattern,
  NeuralData,
  MemoryData 
} from '../types';

// Mock data generators
const generateMockKnowledgeGraph = () => {
  const nodes: KnowledgeNode[] = [];
  const edges: KnowledgeEdge[] = [];
  
  // Generate nodes
  for (let i = 0; i < 50; i++) {
    nodes.push({
      id: `node_${i}`,
      label: `Concept ${i}`,
      type: ['concept', 'entity', 'relation', 'property'][Math.floor(Math.random() * 4)] as any,
      weight: Math.random(),
      position: { x: Math.random(), y: Math.random() },
      metadata: { category: `Category ${Math.floor(Math.random() * 5)}` },
    });
  }
  
  // Generate edges
  for (let i = 0; i < 80; i++) {
    const source = nodes[Math.floor(Math.random() * nodes.length)];
    const target = nodes[Math.floor(Math.random() * nodes.length)];
    if (source.id !== target.id) {
      edges.push({
        id: `edge_${i}`,
        source: source.id,
        target: target.id,
        type: 'relates_to',
        weight: Math.random(),
        confidence: Math.random(),
      });
    }
  }
  
  return { nodes, edges };
};

const generateMockCognitiveData = () => {
  const patterns: CognitivePattern[] = [];
  const currentTime = Date.now();
  
  for (let i = 0; i < 20; i++) {
    patterns.push({
      id: `pattern_${i}`,
      type: ['hierarchical', 'lateral', 'feedback'][Math.floor(Math.random() * 3)] as any,
      strength: Math.random(),
      activeNodes: Array.from({ length: Math.floor(Math.random() * 10) + 1 }, (_, j) => `node_${j}`),
      timestamp: currentTime - Math.random() * 10000,
    });
  }
  
  return {
    patterns,
    inhibitoryLevel: Math.random() * 0.8,
    activationThreshold: 0.5,
    hierarchicalDepth: 3,
  };
};

const generateMockNeuralData = (): NeuralData => {
  const activities = [];
  const layers = [
    { id: '0', name: 'Input Layer', nodeCount: 128, averageActivation: Math.random() },
    { id: '1', name: 'Hidden Layer 1', nodeCount: 256, averageActivation: Math.random() },
    { id: '2', name: 'Hidden Layer 2', nodeCount: 256, averageActivation: Math.random() },
    { id: '3', name: 'Output Layer', nodeCount: 64, averageActivation: Math.random() },
  ];
  
  layers.forEach((layer, layerIndex) => {
    for (let i = 0; i < Math.min(layer.nodeCount, 100); i++) {
      activities.push({
        nodeId: `${layer.id}_${i}`,
        activation: Math.random(),
        position: { x: Math.random(), y: Math.random() },
        layer: layerIndex,
      });
    }
  });
  
  const connections = [];
  for (let i = 0; i < 200; i++) {
    connections.push({
      from: activities[Math.floor(Math.random() * activities.length)].nodeId,
      to: activities[Math.floor(Math.random() * activities.length)].nodeId,
      weight: Math.random(),
      active: Math.random() > 0.5,
    });
  }
  
  return {
    activity: activities,
    layers,
    connections,
    overallActivity: Math.random(),
  };
};

const generateMockMemoryData = (): MemoryData => {
  const totalMemory = 8 * 1024 * 1024 * 1024; // 8GB
  const usedMemory = totalMemory * (0.3 + Math.random() * 0.4); // 30-70% usage
  
  return {
    usage: {
      total: totalMemory,
      used: usedMemory,
      available: totalMemory - usedMemory,
      percentage: usedMemory / totalMemory,
    },
    performance: {
      latency: 50 + Math.random() * 200,
      throughput: 1000 + Math.random() * 5000,
      errorRate: Math.random() * 0.05,
      uptime: 0.95 + Math.random() * 0.05,
    },
    stores: [
      {
        id: 'sdr_store',
        name: 'SDR Store',
        type: 'sdr',
        size: 1024 * 1024 * 1024,
        utilization: Math.random() * 0.8,
        accessCount: Math.floor(Math.random() * 10000),
      },
      {
        id: 'zce_store',
        name: 'Zero Copy Engine',
        type: 'zce',
        size: 2 * 1024 * 1024 * 1024,
        utilization: Math.random() * 0.9,
        accessCount: Math.floor(Math.random() * 50000),
      },
      {
        id: 'cache_store',
        name: 'Cache Store',
        type: 'cache',
        size: 512 * 1024 * 1024,
        utilization: Math.random() * 0.95,
        accessCount: Math.floor(Math.random() * 100000),
      },
    ],
  };
};

const generateMockTableData = () => {
  const data = [];
  for (let i = 0; i < 1000; i++) {
    data.push({
      id: i,
      name: `Node ${i}`,
      type: ['concept', 'entity', 'relation'][Math.floor(Math.random() * 3)],
      activation: Math.random(),
      connections: Math.floor(Math.random() * 20),
      lastUpdated: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      status: ['active', 'inactive', 'processing'][Math.floor(Math.random() * 3)],
    });
  }
  return data;
};

const ComponentShowcase: React.FC = () => {
  const [knowledgeGraph, setKnowledgeGraph] = useState(generateMockKnowledgeGraph());
  const [cognitiveData, setCognitiveData] = useState(generateMockCognitiveData());
  const [neuralData, setNeuralData] = useState(generateMockNeuralData());
  const [memoryData, setMemoryData] = useState(generateMockMemoryData());
  const [tableData] = useState(generateMockTableData());
  const [selectedNodes, setSelectedNodes] = useState<any[]>([]);
  
  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setNeuralData(generateMockNeuralData());
      setMemoryData(generateMockMemoryData());
      setCognitiveData(generateMockCognitiveData());
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const tableColumns: Column[] = [
    {
      key: 'id',
      title: 'ID',
      width: 80,
      sortable: true,
    },
    {
      key: 'name',
      title: 'Name',
      width: 150,
      sortable: true,
      filterable: true,
    },
    {
      key: 'type',
      title: 'Type',
      width: 120,
      sortable: true,
      filterable: true,
      filter: {
        type: 'select',
        options: [
          { label: 'Concept', value: 'concept' },
          { label: 'Entity', value: 'entity' },
          { label: 'Relation', value: 'relation' },
        ],
      },
    },
    {
      key: 'activation',
      title: 'Activation',
      width: 100,
      sortable: true,
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      key: 'connections',
      title: 'Connections',
      width: 100,
      sortable: true,
    },
    {
      key: 'lastUpdated',
      title: 'Last Updated',
      width: 180,
      sortable: true,
      render: (value: string) => new Date(value).toLocaleString(),
    },
    {
      key: 'status',
      title: 'Status',
      width: 120,
      render: (value: string) => (
        <StatusIndicator 
          status={value === 'active' ? 'online' : value === 'processing' ? 'connecting' : 'offline'}
          label={value}
          variant="badge"
          size="small"
        />
      ),
    },
  ];

  return (
    <div style={{ padding: '20px', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <h1 style={{ marginBottom: '30px', color: '#333' }}>LLMKG Component Library Showcase</h1>
      
      {/* Metric Cards Section */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Metric Cards</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
          <CPUMetricCard
            value={0.65}
            trend="up"
            trendValue={0.05}
            status="warning"
            showProgress
            sparklineData={Array.from({ length: 20 }, () => Math.random())}
          />
          <MemoryMetricCard
            value={memoryData.usage.used}
            subtitle="of 8.00 GB total"
            trend="stable"
            status="normal"
            showProgress
            progressMax={memoryData.usage.total}
          />
          <LatencyMetricCard
            value={memoryData.performance.latency}
            trend="down"
            trendValue={20}
            status="success"
            sparklineData={Array.from({ length: 20 }, () => 50 + Math.random() * 200)}
          />
          <ThroughputMetricCard
            value={memoryData.performance.throughput}
            trend="up"
            trendValue={500}
            status="normal"
            sparklineData={Array.from({ length: 20 }, () => 1000 + Math.random() * 5000)}
          />
        </div>
      </section>

      {/* Status Indicators Section */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Status Indicators</h2>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'center' }}>
          <WebSocketStatusIndicator variant="badge" />
          <StatusIndicator status="online" label="LLMKG Core" variant="badge" />
          <StatusIndicator status="warning" label="Memory Warning" variant="badge" />
          <StatusIndicator status="connecting" label="Syncing..." variant="pulse" />
          <StatusIndicator 
            status="active" 
            label="Neural Network" 
            variant="card" 
            description="Processing cognitive patterns"
            showTimestamp
            lastUpdated={new Date()}
            size="large"
          />
        </div>
      </section>

      {/* 3D Knowledge Graph */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>3D Knowledge Graph</h2>
        <KnowledgeGraph3D
          nodes={knowledgeGraph.nodes}
          edges={knowledgeGraph.edges}
          width={800}
          height={600}
          onNodeClick={(node) => console.log('Node clicked:', node)}
          onNodeHover={(node) => console.log('Node hovered:', node)}
          enablePhysics={true}
          showLabels={true}
        />
      </section>

      {/* Cognitive Pattern Visualization */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Cognitive Pattern Visualization</h2>
        <CognitivePatternViz
          patterns={cognitiveData.patterns}
          inhibitoryLevels={cognitiveData}
          width={800}
          height={400}
          timeWindow={10000}
          showLegend={true}
        />
      </section>

      {/* Neural Activity Heatmap */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Neural Activity Heatmap</h2>
        <NeuralActivityHeatmap
          neuralData={neuralData}
          width={800}
          height={500}
          gridSize={20}
          updateInterval={1000}
          showGrid={true}
          showLegend={true}
        />
      </section>

      {/* Memory System Chart */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Memory System Chart</h2>
        <MemorySystemChart
          memoryData={memoryData}
          width={800}
          height={600}
          timeRange={60000}
          showTrends={true}
          showBreakdown={true}
        />
      </section>

      {/* Data Grid */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>High-Performance Data Grid</h2>
        <DataGrid
          data={tableData}
          columns={tableColumns}
          height={400}
          virtualScrolling={true}
          sortable={true}
          filterable={true}
          selectable={true}
          pagination={true}
          pageSize={25}
          onRowSelect={(rows, keys) => setSelectedNodes(rows)}
          onRowClick={(record) => console.log('Row clicked:', record)}
        />
        {selectedNodes.length > 0 && (
          <div style={{ marginTop: '10px', color: '#666' }}>
            Selected {selectedNodes.length} row(s)
          </div>
        )}
      </section>

      {/* Component Integration Example */}
      <section>
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Integrated Dashboard Example</h2>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr', 
          gap: '20px',
          background: 'white',
          padding: '20px',
          borderRadius: '8px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div>
            <h3 style={{ color: '#333', marginBottom: '15px' }}>System Status</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <WebSocketStatusIndicator variant="card" />
              <StatusIndicator 
                status="active" 
                label="LLMKG Engine" 
                variant="card"
                description="All systems operational"
                showTimestamp
                lastUpdated={new Date()}
              />
            </div>
          </div>
          
          <div>
            <h3 style={{ color: '#333', marginBottom: '15px' }}>Performance Metrics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <MetricCard
                title="Active Nodes"
                value={neuralData.activity.length}
                size="small"
                status="normal"
                icon={<span>🧠</span>}
              />
              <MetricCard
                title="Throughput"
                value={memoryData.performance.throughput}
                unit="ops/s"
                size="small"
                format="number"
                precision={0}
                status="success"
                icon={<span>⚡</span>}
              />
              <MetricCard
                title="Memory Usage"
                value={memoryData.usage.percentage}
                format="percentage"
                size="small"
                status={memoryData.usage.percentage > 0.8 ? "warning" : "normal"}
                showProgress
                icon={<span>💾</span>}
              />
              <MetricCard
                title="Error Rate"
                value={memoryData.performance.errorRate}
                format="percentage"
                size="small"
                status={memoryData.performance.errorRate > 0.01 ? "critical" : "success"}
                icon={<span>⚠️</span>}
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ComponentShowcase;