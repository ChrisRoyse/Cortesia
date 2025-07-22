import React, { useState, useEffect } from 'react';
import { 
  SystemArchitectureDiagram, 
  SystemDashboard,
  defaultTheme,
  createCustomTheme 
} from '../src';
import type { 
  ArchitectureData, 
  ArchitectureNode, 
  ConnectionEdge,
  LayerDefinition,
  ComponentStatus 
} from '../src/types';

// Generate sample architecture data for LLMKG
function generateSampleArchitectureData(): ArchitectureData {
  const nodes: ArchitectureNode[] = [
    // Subcortical Layer (Phase 1)
    {
      id: 'input-gate',
      type: 'subcortical',
      label: 'Input Gate',
      description: 'Primary sensory input processing',
      position: { x: 200, y: 100 },
      size: 40,
      layer: 'subcortical',
      status: 'active' as ComponentStatus,
      importance: 0.9,
      connections: [
        { id: 'input-1', type: 'output', angle: 0, active: true }
      ],
      metrics: {
        cpu: { current: 45, average: 42, peak: 67 },
        memory: { current: 62, average: 58, peak: 78 },
        throughput: { current: 1250, average: 1180, peak: 1450 },
        latency: { current: 23, average: 28, peak: 45 },
        errorRate: { current: 0.2, average: 0.3, peak: 1.2 },
        lastUpdated: Date.now()
      }
    },
    {
      id: 'pattern-detector',
      type: 'subcortical',
      label: 'Pattern Detector',
      description: 'Low-level pattern recognition',
      position: { x: 400, y: 100 },
      size: 40,
      layer: 'subcortical',
      status: 'processing' as ComponentStatus,
      importance: 0.8,
      connections: [
        { id: 'pattern-1', type: 'output', angle: 0, active: true }
      ]
    },
    {
      id: 'arousal-system',
      type: 'subcortical',
      label: 'Arousal System',
      description: 'Attention and alertness regulation',
      position: { x: 600, y: 100 },
      size: 40,
      layer: 'subcortical',
      status: 'idle' as ComponentStatus,
      importance: 0.7,
      connections: [
        { id: 'arousal-1', type: 'output', angle: 0, active: false }
      ]
    },
    
    // Cortical Layer (Phase 2)
    {
      id: 'entity-processor',
      type: 'cortical',
      label: 'Entity Processor',
      description: 'Entity extraction and processing',
      position: { x: 150, y: 300 },
      size: 45,
      layer: 'cortical',
      status: 'active' as ComponentStatus,
      importance: 0.9,
      connections: [
        { id: 'entity-in-1', type: 'input', angle: 180, active: true },
        { id: 'entity-out-1', type: 'output', angle: 0, active: true }
      ],
      metrics: {
        cpu: { current: 67, average: 65, peak: 89 },
        memory: { current: 128, average: 125, peak: 156 },
        throughput: { current: 890, average: 850, peak: 1100 },
        latency: { current: 34, average: 38, peak: 67 },
        errorRate: { current: 0.5, average: 0.6, peak: 2.1 },
        lastUpdated: Date.now()
      }
    },
    {
      id: 'relation-mapper',
      type: 'cortical',
      label: 'Relation Mapper',
      description: 'Relationship extraction and mapping',
      position: { x: 350, y: 300 },
      size: 45,
      layer: 'cortical',
      status: 'active' as ComponentStatus,
      importance: 0.85,
      connections: [
        { id: 'relation-in-1', type: 'input', angle: 180, active: true },
        { id: 'relation-out-1', type: 'output', angle: 0, active: true }
      ]
    },
    {
      id: 'concept-former',
      type: 'cortical',
      label: 'Concept Former',
      description: 'Abstract concept formation',
      position: { x: 550, y: 300 },
      size: 45,
      layer: 'cortical',
      status: 'processing' as ComponentStatus,
      importance: 0.8,
      connections: [
        { id: 'concept-in-1', type: 'input', angle: 180, active: true },
        { id: 'concept-out-1', type: 'output', angle: 0, active: true }
      ]
    },
    
    // Thalamic Layer (Phase 3)
    {
      id: 'attention-controller',
      type: 'thalamic',
      label: 'Attention Controller',
      description: 'Attention focus and switching',
      position: { x: 250, y: 500 },
      size: 50,
      layer: 'thalamic',
      status: 'active' as ComponentStatus,
      importance: 0.95,
      connections: [
        { id: 'attention-in-1', type: 'input', angle: 180, active: true },
        { id: 'attention-out-1', type: 'output', angle: 0, active: true },
        { id: 'attention-control-1', type: 'bidirectional', angle: 90, active: true }
      ]
    },
    {
      id: 'context-switcher',
      type: 'thalamic',
      label: 'Context Switcher',
      description: 'Context management and switching',
      position: { x: 450, y: 500 },
      size: 50,
      layer: 'thalamic',
      status: 'idle' as ComponentStatus,
      importance: 0.8,
      connections: [
        { id: 'context-in-1', type: 'input', angle: 180, active: false },
        { id: 'context-out-1', type: 'output', angle: 0, active: false }
      ]
    },
    
    // MCP Interface (Phase 4)
    {
      id: 'mcp-interface',
      type: 'mcp',
      label: 'MCP Interface',
      description: 'Model Context Protocol interface',
      position: { x: 350, y: 700 },
      size: 55,
      layer: 'mcp',
      status: 'active' as ComponentStatus,
      importance: 1.0,
      connections: [
        { id: 'mcp-in-1', type: 'input', angle: 180, active: true },
        { id: 'mcp-out-1', type: 'output', angle: 0, active: true }
      ],
      metrics: {
        cpu: { current: 23, average: 25, peak: 45 },
        memory: { current: 89, average: 85, peak: 120 },
        throughput: { current: 2100, average: 2000, peak: 2500 },
        latency: { current: 12, average: 15, peak: 28 },
        errorRate: { current: 0.1, average: 0.2, peak: 0.8 },
        lastUpdated: Date.now()
      }
    },
    
    // Storage Layer (Phase 5)
    {
      id: 'sdr-storage',
      type: 'storage',
      label: 'SDR Storage',
      description: 'Sparse Distributed Representation storage',
      position: { x: 200, y: 900 },
      size: 45,
      layer: 'storage',
      status: 'active' as ComponentStatus,
      importance: 0.9,
      connections: [
        { id: 'sdr-in-1', type: 'input', angle: 180, active: true }
      ]
    },
    {
      id: 'knowledge-graph',
      type: 'storage',
      label: 'Knowledge Graph',
      description: 'Graph-based knowledge storage',
      position: { x: 500, y: 900 },
      size: 45,
      layer: 'storage',
      status: 'active' as ComponentStatus,
      importance: 0.9,
      connections: [
        { id: 'kg-in-1', type: 'input', angle: 180, active: true }
      ]
    }
  ];

  const connections: ConnectionEdge[] = [
    // Subcortical to Cortical
    { 
      id: 'conn-1', 
      sourceId: 'input-gate', 
      targetId: 'entity-processor', 
      type: 'excitation',
      strength: 0.8,
      active: true,
      dataFlow: 0.6
    },
    { 
      id: 'conn-2', 
      sourceId: 'pattern-detector', 
      targetId: 'relation-mapper', 
      type: 'excitation',
      strength: 0.7,
      active: true,
      dataFlow: 0.5
    },
    { 
      id: 'conn-3', 
      sourceId: 'arousal-system', 
      targetId: 'concept-former', 
      type: 'excitation',
      strength: 0.5,
      active: false,
      dataFlow: 0
    },
    
    // Cortical interconnections
    { 
      id: 'conn-4', 
      sourceId: 'entity-processor', 
      targetId: 'relation-mapper', 
      type: 'data-flow',
      strength: 0.9,
      active: true,
      dataFlow: 0.7
    },
    { 
      id: 'conn-5', 
      sourceId: 'relation-mapper', 
      targetId: 'concept-former', 
      type: 'data-flow',
      strength: 0.8,
      active: true,
      dataFlow: 0.6
    },
    
    // Thalamic control
    { 
      id: 'conn-6', 
      sourceId: 'attention-controller', 
      targetId: 'entity-processor', 
      type: 'bidirectional',
      strength: 0.6,
      active: true,
      dataFlow: 0.3
    },
    { 
      id: 'conn-7', 
      sourceId: 'attention-controller', 
      targetId: 'relation-mapper', 
      type: 'bidirectional',
      strength: 0.6,
      active: true,
      dataFlow: 0.3
    },
    { 
      id: 'conn-8', 
      sourceId: 'context-switcher', 
      targetId: 'concept-former', 
      type: 'bidirectional',
      strength: 0.5,
      active: false,
      dataFlow: 0
    },
    
    // Inhibitory connections
    { 
      id: 'conn-9', 
      sourceId: 'attention-controller', 
      targetId: 'arousal-system', 
      type: 'inhibition',
      strength: 0.4,
      active: true,
      dataFlow: 0.2
    },
    
    // MCP connections
    { 
      id: 'conn-10', 
      sourceId: 'mcp-interface', 
      targetId: 'input-gate', 
      type: 'data-flow',
      strength: 0.9,
      active: true,
      dataFlow: 0.8
    },
    { 
      id: 'conn-11', 
      sourceId: 'concept-former', 
      targetId: 'mcp-interface', 
      type: 'data-flow',
      strength: 0.8,
      active: true,
      dataFlow: 0.6
    },
    
    // Storage connections
    { 
      id: 'conn-12', 
      sourceId: 'entity-processor', 
      targetId: 'sdr-storage', 
      type: 'data-flow',
      strength: 0.7,
      active: true,
      dataFlow: 0.5
    },
    { 
      id: 'conn-13', 
      sourceId: 'concept-former', 
      targetId: 'knowledge-graph', 
      type: 'data-flow',
      strength: 0.8,
      active: true,
      dataFlow: 0.6
    }
  ];

  const layers: LayerDefinition[] = [
    {
      id: 'subcortical',
      name: 'Subcortical Layer',
      description: 'Low-level processing and pattern detection',
      position: { x: 50, y: 50 },
      size: { width: 700, height: 150 },
      color: '#dc2626',
      phase: 1,
      order: 0,
      nodes: ['input-gate', 'pattern-detector', 'arousal-system']
    },
    {
      id: 'cortical',
      name: 'Cortical Layer',
      description: 'High-level cognitive processing',
      position: { x: 50, y: 250 },
      size: { width: 700, height: 150 },
      color: '#2563eb',
      phase: 2,
      order: 1,
      nodes: ['entity-processor', 'relation-mapper', 'concept-former']
    },
    {
      id: 'thalamic',
      name: 'Thalamic Layer',
      description: 'Attention and context management',
      position: { x: 50, y: 450 },
      size: { width: 700, height: 150 },
      color: '#7c3aed',
      phase: 3,
      order: 2,
      nodes: ['attention-controller', 'context-switcher']
    },
    {
      id: 'mcp',
      name: 'MCP Interface Layer',
      description: 'Model Context Protocol integration',
      position: { x: 50, y: 650 },
      size: { width: 700, height: 150 },
      color: '#059669',
      phase: 4,
      order: 3,
      nodes: ['mcp-interface']
    },
    {
      id: 'storage',
      name: 'Storage Layer',
      description: 'Persistent knowledge and memory storage',
      position: { x: 50, y: 850 },
      size: { width: 700, height: 150 },
      color: '#ea580c',
      phase: 5,
      order: 4,
      nodes: ['sdr-storage', 'knowledge-graph']
    }
  ];

  return {
    nodes,
    connections,
    layers,
    metadata: {
      lastUpdated: Date.now(),
      version: '1.0.0',
      totalComponents: nodes.length
    }
  };
}

export default function App() {
  const [architectureData, setArchitectureData] = useState<ArchitectureData>(
    generateSampleArchitectureData()
  );
  const [showDashboard, setShowDashboard] = useState(false);
  const [theme, setTheme] = useState(defaultTheme);
  const [darkMode, setDarkMode] = useState(false);

  // Create dark theme
  const darkTheme = createCustomTheme({
    name: 'llmkg-dark',
    colors: {
      ...defaultTheme.colors,
      background: '#0f172a',
      surface: '#1e293b',
      text: '#f1f5f9'
    }
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setArchitectureData(prev => ({
        ...prev,
        nodes: prev.nodes.map(node => ({
          ...node,
          metrics: node.metrics ? {
            ...node.metrics,
            cpu: {
              ...node.metrics.cpu,
              current: Math.max(0, Math.min(100, node.metrics.cpu.current + (Math.random() - 0.5) * 10))
            },
            memory: {
              ...node.metrics.memory,
              current: Math.max(0, Math.min(200, node.metrics.memory.current + (Math.random() - 0.5) * 20))
            },
            throughput: {
              ...node.metrics.throughput,
              current: Math.max(0, node.metrics.throughput.current + (Math.random() - 0.5) * 100)
            },
            lastUpdated: Date.now()
          } : undefined
        })),
        metadata: {
          ...prev.metadata,
          lastUpdated: Date.now()
        }
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      backgroundColor: darkMode ? darkTheme.colors.background : theme.colors.background,
      color: darkMode ? darkTheme.colors.text : theme.colors.text,
      fontFamily: theme.fonts.primary
    }}>
      <div style={{ 
        padding: '20px', 
        borderBottom: `1px solid ${darkMode ? '#334155' : '#e2e8f0'}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0 }}>LLMKG Phase 5 - Interactive System Architecture</h1>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setShowDashboard(!showDashboard)}
            style={{
              padding: '8px 16px',
              backgroundColor: darkMode ? '#3b82f6' : '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            {showDashboard ? 'Show Diagram' : 'Show Dashboard'}
          </button>
          <button
            onClick={() => {
              setDarkMode(!darkMode);
              setTheme(darkMode ? defaultTheme : darkTheme);
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: darkMode ? '#6366f1' : '#7c3aed',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            {darkMode ? '☀️ Light' : '🌙 Dark'}
          </button>
        </div>
      </div>
      
      <div style={{ height: 'calc(100vh - 81px)' }}>
        {showDashboard ? (
          <SystemDashboard
            realTimeEnabled={true}
            theme={darkMode ? darkTheme : theme}
          />
        ) : (
          <SystemArchitectureDiagram
            architectureData={architectureData}
            theme={darkMode ? darkTheme : theme}
            layout="neural-layers"
            realTimeEnabled={true}
            showMetrics={true}
            showConnections={true}
            enableAnimations={true}
            onNodeClick={(node) => console.log('Node clicked:', node)}
            onConnectionClick={(connection) => console.log('Connection clicked:', connection)}
            onSelectionChange={(selectedNodes) => console.log('Selection changed:', selectedNodes)}
          />
        )}
      </div>
    </div>
  );
}