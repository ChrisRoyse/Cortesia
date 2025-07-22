# Phase 10: Code Integration & Documentation

## Overview
Phase 10 focuses on integrating all visualization components into the LLMKG codebase, creating comprehensive documentation, establishing best practices, and ensuring seamless interaction between visualization tools and the core system.

## Objectives
1. **Component Integration**
   - Integrate visualization components with LLMKG core
   - Establish data flow pipelines
   - Create unified API interfaces
   - Implement component communication

2. **Documentation System**
   - Generate interactive API documentation
   - Create component usage guides
   - Document data formats and protocols
   - Provide integration examples

3. **Developer Experience**
   - Create development workflows
   - Implement hot-reloading for visualizations
   - Provide debugging utilities
   - Establish testing frameworks

4. **Production Deployment**
   - Optimize bundle sizes
   - Implement lazy loading
   - Configure production builds
   - Set up monitoring

## Technical Implementation

### Integration Architecture
```typescript
// src/integration/VisualizationCore.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { MCPClient } from '@llmkg/mcp-client';
import { SDRProcessor } from '@llmkg/sdr-processor';
import { CognitiveEngine } from '@llmkg/cognitive-engine';
import { KnowledgeGraph } from '@llmkg/knowledge-graph';

interface LLMKGVisualizationConfig {
  mcp: {
    endpoint: string;
    protocol: 'ws' | 'http';
    authentication?: {
      type: 'bearer' | 'api-key';
      token: string;
    };
  };
  visualization: {
    theme: 'light' | 'dark';
    updateInterval: number;
    maxDataPoints: number;
    enableAnimations: boolean;
  };
  performance: {
    enableProfiling: boolean;
    sampleRate: number;
    maxMemoryUsage: number;
  };
}

interface LLMKGContext {
  config: LLMKGVisualizationConfig;
  mcpClient: MCPClient | null;
  sdrProcessor: SDRProcessor | null;
  cognitiveEngine: CognitiveEngine | null;
  knowledgeGraph: KnowledgeGraph | null;
  connected: boolean;
  error: Error | null;
}

const LLMKGVisualizationContext = createContext<LLMKGContext | null>(null);

export const LLMKGVisualizationProvider: React.FC<{
  config: LLMKGVisualizationConfig;
  children: React.ReactNode;
}> = ({ config, children }) => {
  const [context, setContext] = useState<LLMKGContext>({
    config,
    mcpClient: null,
    sdrProcessor: null,
    cognitiveEngine: null,
    knowledgeGraph: null,
    connected: false,
    error: null,
  });

  useEffect(() => {
    const initializeConnections = async () => {
      try {
        // Initialize MCP Client
        const mcpClient = new MCPClient({
          endpoint: config.mcp.endpoint,
          protocol: config.mcp.protocol,
          authentication: config.mcp.authentication,
        });

        await mcpClient.connect();

        // Initialize other components through MCP
        const sdrProcessor = await mcpClient.getComponent<SDRProcessor>('sdr-processor');
        const cognitiveEngine = await mcpClient.getComponent<CognitiveEngine>('cognitive-engine');
        const knowledgeGraph = await mcpClient.getComponent<KnowledgeGraph>('knowledge-graph');

        setContext({
          ...context,
          mcpClient,
          sdrProcessor,
          cognitiveEngine,
          knowledgeGraph,
          connected: true,
        });

      } catch (error) {
        setContext({
          ...context,
          error: error as Error,
          connected: false,
        });
      }
    };

    initializeConnections();

    return () => {
      context.mcpClient?.disconnect();
    };
  }, [config]);

  return (
    <LLMKGVisualizationContext.Provider value={context}>
      {children}
    </LLMKGVisualizationContext.Provider>
  );
};

export const useLLMKG = () => {
  const context = useContext(LLMKGVisualizationContext);
  if (!context) {
    throw new Error('useLLMKG must be used within LLMKGVisualizationProvider');
  }
  return context;
};

// Main visualization dashboard
export const LLMKGDashboard: React.FC = () => {
  const { connected, error } = useLLMKG();

  if (error) {
    return <ErrorDisplay error={error} />;
  }

  if (!connected) {
    return <ConnectionLoader />;
  }

  return (
    <div className="llmkg-dashboard">
      <Header />
      <NavigationMenu />
      <div className="dashboard-content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraphExplorer />} />
          <Route path="/neural-network" element={<NeuralNetworkVisualizer />} />
          <Route path="/cognitive-flow" element={<CognitiveFlowChart />} />
          <Route path="/sdr-patterns" element={<SDRPatternAnalyzer />} />
          <Route path="/architecture" element={<SystemArchitecture />} />
          <Route path="/performance" element={<PerformanceDashboard />} />
          <Route path="/memory" element={<MemoryMonitoring />} />
          <Route path="/patterns" element={<CognitivePatternVisualizer />} />
          <Route path="/debug" element={<AdvancedDebugger />} />
          <Route path="/docs" element={<InteractiveDocs />} />
        </Routes>
      </div>
    </div>
  );
};
```

### Unified Data Pipeline
```rust
// src/visualization/data_pipeline.rs
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};
use crate::mcp::MCPInterface;
use crate::cognitive::CognitiveEngine;
use crate::sdr::SDRProcessor;
use crate::knowledge_graph::KnowledgeGraph;

#[derive(Debug, Clone)]
pub struct VisualizationPipeline {
    mcp_interface: Arc<MCPInterface>,
    cognitive_engine: Arc<Mutex<CognitiveEngine>>,
    sdr_processor: Arc<Mutex<SDRProcessor>>,
    knowledge_graph: Arc<Mutex<KnowledgeGraph>>,
    update_channel: broadcast::Sender<VisualizationUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationUpdate {
    CognitiveState(CognitiveStateUpdate),
    SDRUpdate(SDRUpdate),
    KnowledgeGraphUpdate(GraphUpdate),
    PerformanceMetrics(MetricsUpdate),
    SystemEvent(SystemEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveStateUpdate {
    pub timestamp: u64,
    pub layer_states: Vec<LayerState>,
    pub active_patterns: Vec<Pattern>,
    pub attention_focus: Option<AttentionState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRUpdate {
    pub timestamp: u64,
    pub active_sdrs: Vec<SDRInfo>,
    pub operations: Vec<SDROperation>,
    pub overlaps: Vec<(String, String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphUpdate {
    pub timestamp: u64,
    pub new_entities: Vec<Entity>,
    pub new_relations: Vec<Relation>,
    pub modifications: Vec<GraphModification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsUpdate {
    pub timestamp: u64,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub throughput: f32,
    pub latency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    pub timestamp: u64,
    pub event_type: String,
    pub severity: EventSeverity,
    pub message: String,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl VisualizationPipeline {
    pub fn new(
        mcp_interface: Arc<MCPInterface>,
        cognitive_engine: Arc<Mutex<CognitiveEngine>>,
        sdr_processor: Arc<Mutex<SDRProcessor>>,
        knowledge_graph: Arc<Mutex<KnowledgeGraph>>,
    ) -> Self {
        let (tx, _) = broadcast::channel(1000);
        
        Self {
            mcp_interface,
            cognitive_engine,
            sdr_processor,
            knowledge_graph,
            update_channel: tx,
        }
    }

    pub fn start(&self) {
        // Start cognitive state monitoring
        self.start_cognitive_monitoring();
        
        // Start SDR monitoring
        self.start_sdr_monitoring();
        
        // Start knowledge graph monitoring
        self.start_graph_monitoring();
        
        // Start performance monitoring
        self.start_performance_monitoring();
        
        // Set up MCP handlers
        self.setup_mcp_handlers();
    }

    fn start_cognitive_monitoring(&self) {
        let engine = self.cognitive_engine.clone();
        let tx = self.update_channel.clone();
        
        tokio::spawn(async move {
            loop {
                let engine_guard = engine.lock().unwrap();
                let state = engine_guard.get_current_state();
                drop(engine_guard);
                
                let update = VisualizationUpdate::CognitiveState(CognitiveStateUpdate {
                    timestamp: current_timestamp(),
                    layer_states: state.layers,
                    active_patterns: state.patterns,
                    attention_focus: state.attention,
                });
                
                let _ = tx.send(update);
                
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        });
    }

    fn start_sdr_monitoring(&self) {
        let processor = self.sdr_processor.clone();
        let tx = self.update_channel.clone();
        
        tokio::spawn(async move {
            loop {
                let processor_guard = processor.lock().unwrap();
                let sdrs = processor_guard.get_active_sdrs();
                let operations = processor_guard.get_recent_operations(10);
                let overlaps = processor_guard.calculate_overlaps(&sdrs);
                drop(processor_guard);
                
                let update = VisualizationUpdate::SDRUpdate(SDRUpdate {
                    timestamp: current_timestamp(),
                    active_sdrs: sdrs,
                    operations,
                    overlaps,
                });
                
                let _ = tx.send(update);
                
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
    }

    fn start_graph_monitoring(&self) {
        let graph = self.knowledge_graph.clone();
        let tx = self.update_channel.clone();
        
        tokio::spawn(async move {
            let mut last_version = 0;
            
            loop {
                let graph_guard = graph.lock().unwrap();
                let current_version = graph_guard.version();
                
                if current_version > last_version {
                    let changes = graph_guard.get_changes_since(last_version);
                    drop(graph_guard);
                    
                    let update = VisualizationUpdate::KnowledgeGraphUpdate(GraphUpdate {
                        timestamp: current_timestamp(),
                        new_entities: changes.entities,
                        new_relations: changes.relations,
                        modifications: changes.modifications,
                    });
                    
                    let _ = tx.send(update);
                    last_version = current_version;
                }
                
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
        });
    }

    fn start_performance_monitoring(&self) {
        let tx = self.update_channel.clone();
        
        tokio::spawn(async move {
            loop {
                let metrics = collect_system_metrics();
                
                let update = VisualizationUpdate::PerformanceMetrics(MetricsUpdate {
                    timestamp: current_timestamp(),
                    cpu_usage: metrics.cpu,
                    memory_usage: metrics.memory,
                    throughput: metrics.throughput,
                    latency: metrics.latency,
                });
                
                let _ = tx.send(update);
                
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    }

    fn setup_mcp_handlers(&self) {
        let mcp = self.mcp_interface.clone();
        let tx = self.update_channel.clone();
        
        // Register visualization-specific MCP handlers
        mcp.register_handler("visualization/getState", {
            let engine = self.cognitive_engine.clone();
            let processor = self.sdr_processor.clone();
            let graph = self.knowledge_graph.clone();
            
            move |_params| {
                let cognitive_state = engine.lock().unwrap().get_current_state();
                let sdr_state = processor.lock().unwrap().get_state();
                let graph_state = graph.lock().unwrap().get_state();
                
                json!({
                    "cognitive": cognitive_state,
                    "sdr": sdr_state,
                    "graph": graph_state,
                })
            }
        });

        mcp.register_handler("visualization/subscribe", {
            let tx = tx.clone();
            
            move |params| {
                let update_types = params["types"].as_array()
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>())
                    .unwrap_or_default();
                
                let mut rx = tx.subscribe();
                
                tokio::spawn(async move {
                    while let Ok(update) = rx.recv().await {
                        // Filter updates based on subscription
                        if should_send_update(&update, &update_types) {
                            // Send through MCP connection
                            send_mcp_update(update).await;
                        }
                    }
                });
                
                json!({ "subscribed": true })
            }
        });
    }

    pub fn subscribe(&self) -> broadcast::Receiver<VisualizationUpdate> {
        self.update_channel.subscribe()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn collect_system_metrics() -> SystemMetrics {
    // Implementation would use actual system monitoring
    SystemMetrics {
        cpu: 0.0,
        memory: 0,
        throughput: 0.0,
        latency: 0.0,
    }
}

fn should_send_update(update: &VisualizationUpdate, types: &[&str]) -> bool {
    if types.is_empty() {
        return true;
    }
    
    match update {
        VisualizationUpdate::CognitiveState(_) => types.contains(&"cognitive"),
        VisualizationUpdate::SDRUpdate(_) => types.contains(&"sdr"),
        VisualizationUpdate::KnowledgeGraphUpdate(_) => types.contains(&"graph"),
        VisualizationUpdate::PerformanceMetrics(_) => types.contains(&"performance"),
        VisualizationUpdate::SystemEvent(_) => types.contains(&"events"),
    }
}
```

### Interactive Documentation System
```typescript
// src/docs/InteractiveDocs.tsx
import React, { useState, useEffect } from 'react';
import { Tabs, Card, Input, Button, Typography, Space, Divider } from 'antd';
import { CodeOutlined, ApiOutlined, BookOutlined, PlayCircleOutlined } from '@ant-design/icons';
import MonacoEditor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import { useLLMKG } from '../integration/VisualizationCore';

const { TabPane } = Tabs;
const { Title, Paragraph, Text } = Typography;
const { Search } = Input;

interface DocSection {
  id: string;
  title: string;
  content: string;
  examples: CodeExample[];
  api?: APIEndpoint[];
}

interface CodeExample {
  title: string;
  language: string;
  code: string;
  runnable: boolean;
}

interface APIEndpoint {
  method: string;
  path: string;
  description: string;
  parameters: Parameter[];
  response: ResponseSchema;
}

interface Parameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

interface ResponseSchema {
  type: string;
  properties: Record<string, any>;
}

export const InteractiveDocs: React.FC = () => {
  const [sections, setSections] = useState<DocSection[]>([]);
  const [selectedSection, setSelectedSection] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [runOutput, setRunOutput] = useState<string>('');
  const { mcpClient } = useLLMKG();

  useEffect(() => {
    // Load documentation sections
    loadDocumentation();
  }, []);

  const loadDocumentation = async () => {
    const docs: DocSection[] = [
      {
        id: 'getting-started',
        title: 'Getting Started',
        content: `
# Getting Started with LLMKG Visualization

## Installation

\`\`\`bash
npm install @llmkg/visualization
# or
yarn add @llmkg/visualization
\`\`\`

## Basic Setup

\`\`\`typescript
import { LLMKGVisualizationProvider, LLMKGDashboard } from '@llmkg/visualization';

const config = {
  mcp: {
    endpoint: 'ws://localhost:8080',
    protocol: 'ws'
  },
  visualization: {
    theme: 'dark',
    updateInterval: 100,
    maxDataPoints: 1000,
    enableAnimations: true
  }
};

function App() {
  return (
    <LLMKGVisualizationProvider config={config}>
      <LLMKGDashboard />
    </LLMKGVisualizationProvider>
  );
}
\`\`\`
        `,
        examples: [
          {
            title: 'Basic Connection',
            language: 'typescript',
            code: `const { mcpClient, connected } = useLLMKG();

if (connected) {
  const state = await mcpClient.request('visualization/getState', {});
  console.log('Current state:', state);
}`,
            runnable: true
          }
        ]
      },
      {
        id: 'knowledge-graph',
        title: 'Knowledge Graph Visualization',
        content: `
# Knowledge Graph Visualization

The Knowledge Graph Explorer provides interactive visualization of entities and their relationships.

## Features
- 3D force-directed graph layout
- Real-time updates
- Entity filtering and search
- Relationship type filtering
- Graph analytics

## Usage

The knowledge graph automatically connects to the LLMKG backend and displays entities as they are created.
        `,
        examples: [
          {
            title: 'Query Knowledge Graph',
            language: 'typescript',
            code: `const { knowledgeGraph } = useLLMKG();

// Get all entities
const entities = await knowledgeGraph.getEntities();

// Query specific entity
const entity = await knowledgeGraph.getEntity('entity_123');

// Get relationships
const relations = await knowledgeGraph.getRelations({
  source: 'entity_123'
});`,
            runnable: true
          }
        ],
        api: [
          {
            method: 'GET',
            path: '/api/knowledge-graph/entities',
            description: 'Retrieve all entities in the knowledge graph',
            parameters: [
              {
                name: 'limit',
                type: 'number',
                required: false,
                description: 'Maximum number of entities to return'
              },
              {
                name: 'offset',
                type: 'number',
                required: false,
                description: 'Offset for pagination'
              }
            ],
            response: {
              type: 'object',
              properties: {
                entities: 'Entity[]',
                total: 'number'
              }
            }
          }
        ]
      },
      {
        id: 'cognitive-patterns',
        title: 'Cognitive Pattern Visualization',
        content: `
# Cognitive Pattern Visualization

Visualize the brain-inspired cognitive processing patterns in LLMKG.

## Layer Types
- **Subcortical**: Basic pattern detection and arousal
- **Cortical**: Higher-level processing and concept formation
- **Thalamic**: Attention control and context switching

## Pattern Types
- **Excitatory**: Activation spreading patterns
- **Inhibitory**: Competition and suppression patterns
- **Modulatory**: Attention and context modulation
        `,
        examples: [
          {
            title: 'Monitor Cognitive Patterns',
            language: 'typescript',
            code: `const { cognitiveEngine } = useLLMKG();

// Subscribe to pattern updates
cognitiveEngine.subscribe('patterns', (patterns) => {
  console.log('Active patterns:', patterns);
  
  // Find burst patterns
  const bursts = patterns.filter(p => p.type === 'burst');
  console.log('Burst patterns:', bursts);
});

// Get current cognitive state
const state = await cognitiveEngine.getCurrentState();
console.log('Layer activations:', state.layers);`,
            runnable: true
          }
        ]
      },
      {
        id: 'sdr-analysis',
        title: 'SDR Analysis',
        content: `
# Sparse Distributed Representation (SDR) Analysis

SDRs are the fundamental data structure in LLMKG, representing information in a brain-like sparse format.

## Key Concepts
- **Sparsity**: Typically 2% of bits are active
- **Overlap**: Semantic similarity measured by bit overlap
- **Capacity**: High capacity with collision resistance

## Visualization Features
- Bit pattern visualization
- Overlap matrix
- Semantic clustering
- Union/intersection operations
        `,
        examples: [
          {
            title: 'SDR Operations',
            language: 'typescript',
            code: `const { sdrProcessor } = useLLMKG();

// Create SDR from text
const sdr1 = await sdrProcessor.encode('machine learning');
const sdr2 = await sdrProcessor.encode('deep learning');

// Calculate overlap
const overlap = sdrProcessor.overlap(sdr1, sdr2);
console.log('Semantic similarity:', overlap);

// Union operation
const union = sdrProcessor.union(sdr1, sdr2);
console.log('Union sparsity:', union.sparsity);`,
            runnable: true
          }
        ]
      },
      {
        id: 'performance',
        title: 'Performance Monitoring',
        content: `
# Performance Monitoring

Monitor system performance and identify bottlenecks.

## Metrics Tracked
- **Cognitive Processing**: Layer latencies, throughput
- **SDR Operations**: Creation rate, overlap calculations
- **Memory Usage**: Per-component memory allocation
- **MCP Protocol**: Message rates, latencies

## Optimization Tips
1. Adjust sparsity for memory/accuracy tradeoff
2. Tune inhibition parameters for stability
3. Configure batch sizes for throughput
        `,
        examples: [
          {
            title: 'Performance Metrics',
            language: 'typescript',
            code: `const { mcpClient } = useLLMKG();

// Get current metrics
const metrics = await mcpClient.request('performance/getMetrics', {
  detailed: true
});

console.log('CPU Usage:', metrics.system.cpuUsage);
console.log('Memory:', metrics.system.memoryUsage);
console.log('Cognitive Latency:', metrics.cognitive.averageLatency);

// Subscribe to performance alerts
mcpClient.subscribe('performance.alerts', (alert) => {
  console.warn('Performance alert:', alert);
});`,
            runnable: true
          }
        ]
      },
      {
        id: 'debugging',
        title: 'Advanced Debugging',
        content: `
# Advanced Debugging Tools

Debug complex cognitive behaviors and trace execution paths.

## Features
- **Time-Travel Debugging**: Step forward/backward through states
- **Breakpoints**: Conditional breaks on patterns
- **State Inspection**: Deep inspection of all components
- **Diff Analysis**: Compare states to find changes

## Debug Workflow
1. Start debug session
2. Set breakpoints on interesting conditions
3. Run operations
4. Analyze captured states
5. Export session for sharing
        `,
        examples: [
          {
            title: 'Debug Session',
            language: 'typescript',
            code: `const { mcpClient } = useLLMKG();

// Start debug session
const session = await mcpClient.request('debug/startSession', {
  recordStates: true,
  enableBreakpoints: true
});

// Set breakpoint
await mcpClient.request('debug/addBreakpoint', {
  sessionId: session.id,
  type: 'cognitive',
  condition: 'cortical.activation > 0.8'
});

// Later, analyze states
const states = await mcpClient.request('debug/getStates', {
  sessionId: session.id,
  range: [0, 100]
});

console.log('Captured states:', states.length);`,
            runnable: true
          }
        ]
      }
    ];

    setSections(docs);
    setSelectedSection(docs[0].id);
  };

  const runExample = async (code: string) => {
    try {
      setRunOutput('Running...\n');
      
      // Create a safe execution context
      const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
      const fn = new AsyncFunction('useLLMKG', 'console', code);
      
      // Capture console output
      const output: string[] = [];
      const mockConsole = {
        log: (...args: any[]) => output.push(args.join(' ')),
        warn: (...args: any[]) => output.push(`WARN: ${args.join(' ')}`),
        error: (...args: any[]) => output.push(`ERROR: ${args.join(' ')}`)
      };
      
      // Execute with injected dependencies
      await fn(useLLMKG, mockConsole);
      
      setRunOutput(output.join('\n'));
    } catch (error) {
      setRunOutput(`Error: ${error.message}`);
    }
  };

  const filteredSections = sections.filter(s =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const currentSection = sections.find(s => s.id === selectedSection);

  return (
    <div className="interactive-docs">
      <Title level={2}>Interactive Documentation</Title>
      
      <Row gutter={24}>
        <Col span={6}>
          <Card>
            <Search
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ marginBottom: 16 }}
            />
            
            <Menu
              mode="vertical"
              selectedKeys={[selectedSection]}
              onClick={({ key }) => setSelectedSection(key)}
            >
              {filteredSections.map(section => (
                <Menu.Item key={section.id} icon={<BookOutlined />}>
                  {section.title}
                </Menu.Item>
              ))}
            </Menu>
          </Card>
        </Col>
        
        <Col span={18}>
          {currentSection && (
            <Card>
              <div className="doc-content">
                <ReactMarkdown>{currentSection.content}</ReactMarkdown>
                
                {currentSection.examples.length > 0 && (
                  <>
                    <Divider />
                    <Title level={3}>Examples</Title>
                    
                    {currentSection.examples.map((example, idx) => (
                      <Card key={idx} size="small" style={{ marginBottom: 16 }}>
                        <Title level={4}>{example.title}</Title>
                        
                        <MonacoEditor
                          height="200px"
                          language={example.language}
                          value={example.code}
                          theme="vs-dark"
                          options={{
                            readOnly: !example.runnable,
                            minimap: { enabled: false }
                          }}
                        />
                        
                        {example.runnable && (
                          <>
                            <Button
                              icon={<PlayCircleOutlined />}
                              onClick={() => runExample(example.code)}
                              style={{ marginTop: 8 }}
                            >
                              Run Example
                            </Button>
                            
                            {runOutput && (
                              <pre style={{ 
                                marginTop: 16, 
                                padding: 12, 
                                background: '#1e1e1e',
                                color: '#d4d4d4',
                                borderRadius: 4
                              }}>
                                {runOutput}
                              </pre>
                            )}
                          </>
                        )}
                      </Card>
                    ))}
                  </>
                )}
                
                {currentSection.api && currentSection.api.length > 0 && (
                  <>
                    <Divider />
                    <Title level={3}>API Reference</Title>
                    
                    {currentSection.api.map((endpoint, idx) => (
                      <Card key={idx} size="small" style={{ marginBottom: 16 }}>
                        <Space>
                          <Tag color={endpoint.method === 'GET' ? 'green' : 'blue'}>
                            {endpoint.method}
                          </Tag>
                          <Text code>{endpoint.path}</Text>
                        </Space>
                        
                        <Paragraph style={{ marginTop: 8 }}>
                          {endpoint.description}
                        </Paragraph>
                        
                        {endpoint.parameters.length > 0 && (
                          <>
                            <Title level={5}>Parameters</Title>
                            <Table
                              dataSource={endpoint.parameters}
                              columns={[
                                { title: 'Name', dataIndex: 'name', key: 'name' },
                                { title: 'Type', dataIndex: 'type', key: 'type' },
                                { title: 'Required', dataIndex: 'required', key: 'required', render: (v) => v ? 'Yes' : 'No' },
                                { title: 'Description', dataIndex: 'description', key: 'description' }
                              ]}
                              pagination={false}
                              size="small"
                            />
                          </>
                        )}
                      </Card>
                    ))}
                  </>
                )}
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};
```

### Development Workflow Configuration
```javascript
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');

module.exports = (env, argv) => {
  const isDevelopment = argv.mode === 'development';
  const isProduction = argv.mode === 'production';

  return {
    entry: './src/index.tsx',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? '[name].[contenthash].js' : '[name].js',
      chunkFilename: isProduction ? '[name].[contenthash].chunk.js' : '[name].chunk.js',
      clean: true
    },
    
    module: {
      rules: [
        {
          test: /\.(ts|tsx)$/,
          use: 'ts-loader',
          exclude: /node_modules/
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader', 'postcss-loader']
        },
        {
          test: /\.(png|svg|jpg|jpeg|gif)$/i,
          type: 'asset/resource'
        },
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource'
        }
      ]
    },
    
    resolve: {
      extensions: ['.tsx', '.ts', '.js'],
      alias: {
        '@components': path.resolve(__dirname, 'src/components'),
        '@hooks': path.resolve(__dirname, 'src/hooks'),
        '@utils': path.resolve(__dirname, 'src/utils'),
        '@llmkg': path.resolve(__dirname, 'src/llmkg')
      }
    },
    
    plugins: [
      new HtmlWebpackPlugin({
        template: './public/index.html',
        favicon: './public/favicon.ico'
      }),
      
      ...(isProduction ? [
        new CompressionPlugin({
          algorithm: 'gzip',
          test: /\.(js|css|html|svg)$/,
          threshold: 8192,
          minRatio: 0.8
        }),
        new BundleAnalyzerPlugin({
          analyzerMode: 'static',
          openAnalyzer: false,
          reportFilename: 'bundle-report.html'
        })
      ] : []),
      
      ...(isDevelopment ? [
        new webpack.HotModuleReplacementPlugin()
      ] : [])
    ],
    
    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            compress: {
              drop_console: isProduction
            }
          }
        })
      ],
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: -10
          },
          d3: {
            test: /[\\/]node_modules[\\/](d3|d3-.*)[\\/]/,
            name: 'd3',
            priority: 10
          },
          react: {
            test: /[\\/]node_modules[\\/](react|react-dom|react-router)[\\/]/,
            name: 'react',
            priority: 10
          },
          antd: {
            test: /[\\/]node_modules[\\/](antd|@ant-design)[\\/]/,
            name: 'antd',
            priority: 10
          }
        }
      },
      runtimeChunk: 'single'
    },
    
    devServer: {
      contentBase: './dist',
      historyApiFallback: true,
      hot: true,
      port: 3000,
      proxy: {
        '/api': {
          target: 'http://localhost:8080',
          changeOrigin: true
        },
        '/ws': {
          target: 'ws://localhost:8080',
          ws: true,
          changeOrigin: true
        }
      }
    },
    
    devtool: isDevelopment ? 'eval-source-map' : 'source-map'
  };
};
```

### Integration Tests
```typescript
// tests/integration/full-stack.test.ts
import { render, waitFor } from '@testing-library/react';
import { LLMKGVisualizationProvider, LLMKGDashboard } from '../src';
import { MockMCPServer } from './mocks/MockMCPServer';

describe('Full Stack Integration', () => {
  let mockServer: MockMCPServer;

  beforeAll(async () => {
    mockServer = new MockMCPServer();
    await mockServer.start(8080);
  });

  afterAll(async () => {
    await mockServer.stop();
  });

  it('should connect to MCP server and display data', async () => {
    const config = {
      mcp: {
        endpoint: 'ws://localhost:8080',
        protocol: 'ws' as const
      },
      visualization: {
        theme: 'light' as const,
        updateInterval: 100,
        maxDataPoints: 1000,
        enableAnimations: true
      },
      performance: {
        enableProfiling: false,
        sampleRate: 1,
        maxMemoryUsage: 512 * 1024 * 1024
      }
    };

    const { container } = render(
      <LLMKGVisualizationProvider config={config}>
        <LLMKGDashboard />
      </LLMKGVisualizationProvider>
    );

    // Wait for connection
    await waitFor(() => {
      expect(container.querySelector('.connection-status')).toHaveTextContent('Connected');
    });

    // Verify components loaded
    expect(container.querySelector('.knowledge-graph-explorer')).toBeInTheDocument();
    expect(container.querySelector('.neural-network-visualizer')).toBeInTheDocument();
    expect(container.querySelector('.performance-dashboard')).toBeInTheDocument();
  });

  it('should handle real-time updates', async () => {
    const { container } = render(
      <LLMKGVisualizationProvider config={config}>
        <KnowledgeGraphExplorer />
      </LLMKGVisualizationProvider>
    );

    // Simulate entity creation
    mockServer.createEntity({
      id: 'test_entity_1',
      type: 'concept',
      data: { name: 'Test Concept' }
    });

    await waitFor(() => {
      const entity = container.querySelector('[data-entity-id="test_entity_1"]');
      expect(entity).toBeInTheDocument();
      expect(entity).toHaveTextContent('Test Concept');
    });
  });

  it('should handle errors gracefully', async () => {
    // Stop server to simulate connection error
    await mockServer.stop();

    const { container } = render(
      <LLMKGVisualizationProvider config={config}>
        <LLMKGDashboard />
      </LLMKGVisualizationProvider>
    );

    await waitFor(() => {
      expect(container.querySelector('.error-display')).toBeInTheDocument();
      expect(container.querySelector('.error-message')).toHaveTextContent('Connection failed');
    });
  });
});
```

## LLMKG-Specific Features

### 1. Unified Data Access
- **Single Context Provider**: All components access LLMKG through unified context
- **Automatic Reconnection**: Handle connection failures gracefully
- **State Synchronization**: Keep all visualizations in sync

### 2. Component Communication
- **Event Bus**: Components communicate through centralized event system
- **State Sharing**: Shared state management for coordinated updates
- **Cross-Component Actions**: Actions in one component affect others

### 3. Developer Tools
- **Hot Module Replacement**: Instant updates during development
- **State DevTools**: Inspect and modify LLMKG state
- **Performance Profiling**: Built-in performance monitoring

### 4. Production Optimization
- **Code Splitting**: Load components on demand
- **Bundle Analysis**: Identify and eliminate bloat
- **Compression**: Gzip/Brotli compression for assets

## Documentation Structure

### 1. API Documentation
- **Auto-generated**: From TypeScript/Rust code
- **Interactive Examples**: Run code directly in docs
- **Type Information**: Full type documentation

### 2. Component Documentation
- **Usage Examples**: Real-world usage patterns
- **Props Documentation**: All component properties
- **Best Practices**: Recommended patterns

### 3. Integration Guides
- **Setup Instructions**: Step-by-step setup
- **Configuration Options**: All config parameters
- **Troubleshooting**: Common issues and solutions

## Testing Strategy

### 1. Unit Tests
- Component-level testing
- Hook testing
- Utility function testing

### 2. Integration Tests
- Full-stack testing with mock server
- Real-time update testing
- Error handling verification

### 3. E2E Tests
- Complete user workflows
- Performance benchmarks
- Cross-browser testing

## Deployment

### 1. Development
```bash
npm run dev
# Starts development server with HMR
```

### 2. Production Build
```bash
npm run build
# Creates optimized production bundle
```

### 3. Docker Deployment
```dockerfile
FROM node:16-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 4. Version Control Information & Deployment Tracking

### Git Integration Dashboard
```typescript
// src/components/VersionControl/GitDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tag, Timeline, Table, Space, Button, Modal, Descriptions } from 'antd';
import { 
  BranchesOutlined, 
  TagOutlined, 
  UserOutlined, 
  ClockCircleOutlined,
  CodeOutlined,
  DeploymentUnitOutlined,
  HistoryOutlined,
  DiffOutlined
} from '@ant-design/icons';
import { useLLMKG } from '../../integration/VisualizationCore';

interface GitInfo {
  branch: string;
  commit: string;
  shortCommit: string;
  author: string;
  email: string;
  message: string;
  timestamp: string;
  isDirty: boolean;
  tags: string[];
}

interface BuildInfo {
  version: string;
  buildNumber: string;
  buildDate: string;
  environment: 'development' | 'staging' | 'production';
  gitInfo: GitInfo;
}

interface ChangelogEntry {
  version: string;
  date: string;
  author: string;
  changes: {
    type: 'feature' | 'fix' | 'enhancement' | 'breaking';
    description: string;
    component?: string;
  }[];
}

interface DeploymentRecord {
  id: string;
  version: string;
  environment: string;
  timestamp: string;
  author: string;
  status: 'success' | 'failed' | 'in-progress';
  duration?: number;
  rollbackFrom?: string;
}

export const GitDashboard: React.FC = () => {
  const [gitInfo, setGitInfo] = useState<GitInfo | null>(null);
  const [buildInfo, setBuildInfo] = useState<BuildInfo | null>(null);
  const [changelog, setChangelog] = useState<ChangelogEntry[]>([]);
  const [deployments, setDeployments] = useState<DeploymentRecord[]>([]);
  const [compareModalVisible, setCompareModalVisible] = useState(false);
  const [selectedCommits, setSelectedCommits] = useState<string[]>([]);
  const { mcpClient } = useLLMKG();

  useEffect(() => {
    loadVersionInfo();
    loadChangelog();
    loadDeploymentHistory();
  }, []);

  const loadVersionInfo = async () => {
    try {
      // Fetch current Git information
      const git = await mcpClient?.request('version/getGitInfo', {});
      setGitInfo(git);

      // Fetch build information
      const build = await mcpClient?.request('version/getBuildInfo', {});
      setBuildInfo(build);
    } catch (error) {
      console.error('Failed to load version info:', error);
    }
  };

  const loadChangelog = async () => {
    try {
      const changes = await mcpClient?.request('version/getChangelog', {
        limit: 20
      });
      setChangelog(changes);
    } catch (error) {
      console.error('Failed to load changelog:', error);
    }
  };

  const loadDeploymentHistory = async () => {
    try {
      const history = await mcpClient?.request('deployment/getHistory', {
        limit: 10
      });
      setDeployments(history);
    } catch (error) {
      console.error('Failed to load deployment history:', error);
    }
  };

  const handleCompareVersions = async () => {
    if (selectedCommits.length === 2) {
      try {
        const comparison = await mcpClient?.request('version/compareCommits', {
          from: selectedCommits[0],
          to: selectedCommits[1]
        });
        
        // Display comparison modal with diff information
        setCompareModalVisible(true);
      } catch (error) {
        console.error('Failed to compare versions:', error);
      }
    }
  };

  return (
    <div className="git-dashboard">
      <Row gutter={[24, 24]}>
        {/* Current Version Display */}
        <Col span={24}>
          <Card title="Current Version Information" size="default">
            {buildInfo && gitInfo ? (
              <Row gutter={16}>
                <Col span={8}>
                  <Descriptions title="Build Information" column={1} size="small">
                    <Descriptions.Item label="Version">
                      <Tag color="blue">{buildInfo.version}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="Build Number">
                      {buildInfo.buildNumber}
                    </Descriptions.Item>
                    <Descriptions.Item label="Environment">
                      <Tag color={
                        buildInfo.environment === 'production' ? 'red' :
                        buildInfo.environment === 'staging' ? 'orange' : 'green'
                      }>
                        {buildInfo.environment.toUpperCase()}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="Build Date">
                      {new Date(buildInfo.buildDate).toLocaleString()}
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
                
                <Col span={8}>
                  <Descriptions title="Git Information" column={1} size="small">
                    <Descriptions.Item label="Branch">
                      <Space>
                        <BranchesOutlined />
                        <Tag color="cyan">{gitInfo.branch}</Tag>
                        {gitInfo.isDirty && <Tag color="warning">Modified</Tag>}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit">
                      <Space>
                        <CodeOutlined />
                        <span style={{ fontFamily: 'monospace' }}>
                          {gitInfo.shortCommit}
                        </span>
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Author">
                      <Space>
                        <UserOutlined />
                        {gitInfo.author}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit Time">
                      <Space>
                        <ClockCircleOutlined />
                        {new Date(gitInfo.timestamp).toLocaleString()}
                      </Space>
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
                
                <Col span={8}>
                  <Descriptions title="Tags & Labels" column={1} size="small">
                    <Descriptions.Item label="Git Tags">
                      <Space wrap>
                        {gitInfo.tags.map(tag => (
                          <Tag key={tag} icon={<TagOutlined />} color="geekblue">
                            {tag}
                          </Tag>
                        ))}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Commit Message" span={2}>
                      <div style={{ 
                        background: '#f5f5f5', 
                        padding: '8px 12px', 
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}>
                        {gitInfo.message}
                      </div>
                    </Descriptions.Item>
                  </Descriptions>
                </Col>
              </Row>
            ) : (
              <div>Loading version information...</div>
            )}
          </Card>
        </Col>

        {/* Recent Changes Timeline */}
        <Col span={12}>
          <Card title="Recent Changes" size="default" extra={
            <Button 
              icon={<HistoryOutlined />}
              onClick={() => setCompareModalVisible(true)}
            >
              Compare Versions
            </Button>
          }>
            <Timeline>
              {changelog.slice(0, 5).map((entry, index) => (
                <Timeline.Item
                  key={entry.version}
                  color={index === 0 ? 'green' : 'blue'}
                  dot={index === 0 ? <ClockCircleOutlined style={{ fontSize: '16px' }} /> : undefined}
                >
                  <div>
                    <Space>
                      <Tag color="purple">{entry.version}</Tag>
                      <span style={{ fontSize: '12px', color: '#666' }}>
                        {new Date(entry.date).toLocaleDateString()}
                      </span>
                      <span style={{ fontSize: '12px', color: '#666' }}>
                        by {entry.author}
                      </span>
                    </Space>
                    <div style={{ marginTop: '8px' }}>
                      {entry.changes.slice(0, 3).map((change, idx) => (
                        <div key={idx} style={{ marginBottom: '4px' }}>
                          <Tag 
                            size="small" 
                            color={
                              change.type === 'feature' ? 'green' :
                              change.type === 'fix' ? 'red' :
                              change.type === 'enhancement' ? 'blue' : 'orange'
                            }
                          >
                            {change.type}
                          </Tag>
                          <span style={{ fontSize: '12px' }}>
                            {change.description}
                            {change.component && (
                              <Tag size="small" style={{ marginLeft: '8px' }}>
                                {change.component}
                              </Tag>
                            )}
                          </span>
                        </div>
                      ))}
                      {entry.changes.length > 3 && (
                        <div style={{ fontSize: '12px', color: '#888' }}>
                          +{entry.changes.length - 3} more changes
                        </div>
                      )}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>

        {/* Deployment History */}
        <Col span={12}>
          <Card title="Deployment History" size="default" extra={
            <Button 
              icon={<DeploymentUnitOutlined />}
              type="primary"
              onClick={() => {/* Trigger new deployment */}}
            >
              Deploy
            </Button>
          }>
            <Table
              dataSource={deployments}
              columns={[
                {
                  title: 'Version',
                  dataIndex: 'version',
                  key: 'version',
                  render: (version) => <Tag color="blue">{version}</Tag>
                },
                {
                  title: 'Environment',
                  dataIndex: 'environment',
                  key: 'environment',
                  render: (env) => (
                    <Tag color={
                      env === 'production' ? 'red' :
                      env === 'staging' ? 'orange' : 'green'
                    }>
                      {env.toUpperCase()}
                    </Tag>
                  )
                },
                {
                  title: 'Status',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status) => (
                    <Tag color={
                      status === 'success' ? 'success' :
                      status === 'failed' ? 'error' : 'processing'
                    }>
                      {status.toUpperCase()}
                    </Tag>
                  )
                },
                {
                  title: 'Duration',
                  dataIndex: 'duration',
                  key: 'duration',
                  render: (duration) => duration ? `${Math.floor(duration / 60)}m ${duration % 60}s` : '-'
                },
                {
                  title: 'Date',
                  dataIndex: 'timestamp',
                  key: 'timestamp',
                  render: (timestamp) => new Date(timestamp).toLocaleString()
                }
              ]}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>

        {/* MCP Tools Impact Analysis */}
        <Col span={24}>
          <Card title="Recent MCP Tools Modifications" size="default">
            <MCPModificationTracker />
          </Card>
        </Col>
      </Row>

      {/* Version Comparison Modal */}
      <Modal
        title="Version Comparison"
        visible={compareModalVisible}
        onCancel={() => setCompareModalVisible(false)}
        width={1000}
        footer={null}
      >
        <VersionComparison />
      </Modal>
    </div>
  );
};

// MCP Tools Modification Tracker Component
const MCPModificationTracker: React.FC = () => {
  const [modifications, setModifications] = useState<any[]>([]);
  const { mcpClient } = useLLMKG();

  useEffect(() => {
    loadMCPModifications();
  }, []);

  const loadMCPModifications = async () => {
    try {
      const mods = await mcpClient?.request('version/getMCPModifications', {
        since: Date.now() - (7 * 24 * 60 * 60 * 1000) // Last 7 days
      });
      setModifications(mods || []);
    } catch (error) {
      console.error('Failed to load MCP modifications:', error);
    }
  };

  return (
    <Table
      dataSource={modifications}
      columns={[
        {
          title: 'Tool/Handler',
          dataIndex: 'tool',
          key: 'tool',
          render: (tool) => <Tag color="geekblue">{tool}</Tag>
        },
        {
          title: 'Modification Type',
          dataIndex: 'type',
          key: 'type',
          render: (type) => (
            <Tag color={
              type === 'added' ? 'success' :
              type === 'modified' ? 'warning' : 'error'
            }>
              {type.toUpperCase()}
            </Tag>
          )
        },
        {
          title: 'Description',
          dataIndex: 'description',
          key: 'description'
        },
        {
          title: 'Impact Level',
          dataIndex: 'impact',
          key: 'impact',
          render: (impact) => (
            <Tag color={
              impact === 'high' ? 'red' :
              impact === 'medium' ? 'orange' : 'green'
            }>
              {impact.toUpperCase()}
            </Tag>
          )
        },
        {
          title: 'Modified',
          dataIndex: 'timestamp',
          key: 'timestamp',
          render: (timestamp) => new Date(timestamp).toLocaleDateString()
        },
        {
          title: 'Author',
          dataIndex: 'author',
          key: 'author'
        }
      ]}
      pagination={{ pageSize: 5 }}
      size="small"
    />
  );
};

// Version Comparison Component
const VersionComparison: React.FC = () => {
  const [comparison, setComparison] = useState<any>(null);
  const { mcpClient } = useLLMKG();

  return (
    <div className="version-comparison">
      <Row gutter={16}>
        <Col span={12}>
          <Card title="Files Changed" size="small">
            {/* File diff view */}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="Summary" size="small">
            {/* Change summary */}
          </Card>
        </Col>
      </Row>
    </div>
  );
};
```

### Git API Integration
```rust
// src/version/git_api.rs
use std::process::Command;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::mcp::MCPInterface;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInfo {
    pub branch: String,
    pub commit: String,
    pub short_commit: String,
    pub author: String,
    pub email: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub is_dirty: bool,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub build_number: String,
    pub build_date: DateTime<Utc>,
    pub environment: Environment,
    pub git_info: GitInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub version: String,
    pub date: DateTime<Utc>,
    pub author: String,
    pub changes: Vec<ChangeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeItem {
    pub change_type: ChangeType,
    pub description: String,
    pub component: Option<String>,
    pub breaking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Feature,
    Fix,
    Enhancement,
    Documentation,
    Refactor,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecord {
    pub id: String,
    pub version: String,
    pub environment: String,
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub status: DeploymentStatus,
    pub duration: Option<u64>,
    pub rollback_from: Option<String>,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Success,
    Failed,
    InProgress,
    Cancelled,
}

pub struct GitIntegration {
    mcp_interface: Arc<MCPInterface>,
    repository_path: PathBuf,
}

impl GitIntegration {
    pub fn new(mcp_interface: Arc<MCPInterface>, repo_path: PathBuf) -> Self {
        Self {
            mcp_interface,
            repository_path: repo_path,
        }
    }

    pub fn setup_mcp_handlers(&self) {
        let mcp = self.mcp_interface.clone();
        let repo_path = self.repository_path.clone();

        // Get current Git information
        mcp.register_handler("version/getGitInfo", {
            let repo_path = repo_path.clone();
            move |_params| {
                let git_info = get_git_info(&repo_path)?;
                Ok(serde_json::to_value(git_info)?)
            }
        });

        // Get build information
        mcp.register_handler("version/getBuildInfo", {
            let repo_path = repo_path.clone();
            move |_params| {
                let build_info = get_build_info(&repo_path)?;
                Ok(serde_json::to_value(build_info)?)
            }
        });

        // Get changelog
        mcp.register_handler("version/getChangelog", {
            let repo_path = repo_path.clone();
            move |params| {
                let limit = params.get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                
                let changelog = get_changelog(&repo_path, limit)?;
                Ok(serde_json::to_value(changelog)?)
            }
        });

        // Compare commits
        mcp.register_handler("version/compareCommits", {
            let repo_path = repo_path.clone();
            move |params| {
                let from = params["from"].as_str().unwrap();
                let to = params["to"].as_str().unwrap();
                
                let comparison = compare_commits(&repo_path, from, to)?;
                Ok(serde_json::to_value(comparison)?)
            }
        });

        // Get MCP tool modifications
        mcp.register_handler("version/getMCPModifications", {
            let repo_path = repo_path.clone();
            move |params| {
                let since = params.get("since")
                    .and_then(|v| v.as_u64())
                    .map(|ts| DateTime::<Utc>::from_timestamp(ts as i64, 0).unwrap())
                    .unwrap_or_else(|| Utc::now() - chrono::Duration::days(7));
                
                let modifications = get_mcp_modifications(&repo_path, since)?;
                Ok(serde_json::to_value(modifications)?)
            }
        });

        // Get deployment history
        mcp.register_handler("deployment/getHistory", {
            move |params| {
                let limit = params.get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                
                let history = get_deployment_history(limit)?;
                Ok(serde_json::to_value(history)?)
            }
        });
    }
}

fn get_git_info(repo_path: &Path) -> Result<GitInfo, Box<dyn std::error::Error>> {
    let branch = run_git_command(repo_path, &["branch", "--show-current"])?
        .trim().to_string();
    
    let commit = run_git_command(repo_path, &["rev-parse", "HEAD"])?
        .trim().to_string();
    
    let short_commit = run_git_command(repo_path, &["rev-parse", "--short", "HEAD"])?
        .trim().to_string();
    
    let author_info = run_git_command(repo_path, &["log", "-1", "--format=%an|%ae|%s|%ct"])?;
    let parts: Vec<&str> = author_info.trim().split('|').collect();
    
    let author = parts[0].to_string();
    let email = parts[1].to_string();
    let message = parts[2].to_string();
    let timestamp = DateTime::<Utc>::from_timestamp(parts[3].parse()?, 0).unwrap();
    
    let is_dirty = !run_git_command(repo_path, &["status", "--porcelain"])?
        .trim().is_empty();
    
    let tags = run_git_command(repo_path, &["tag", "--points-at", "HEAD"])?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(GitInfo {
        branch,
        commit,
        short_commit,
        author,
        email,
        message,
        timestamp,
        is_dirty,
        tags,
    })
}

fn get_build_info(repo_path: &Path) -> Result<BuildInfo, Box<dyn std::error::Error>> {
    let git_info = get_git_info(repo_path)?;
    
    // Read version from Cargo.toml or package.json
    let version = read_version_from_manifest(repo_path)?;
    
    // Generate build number from timestamp + commit
    let build_number = format!("{}-{}", 
        Utc::now().timestamp(), 
        git_info.short_commit
    );
    
    // Determine environment from branch or ENV var
    let environment = determine_environment(&git_info.branch)?;

    Ok(BuildInfo {
        version,
        build_number,
        build_date: Utc::now(),
        environment,
        git_info,
    })
}

fn get_changelog(repo_path: &Path, limit: usize) -> Result<Vec<ChangelogEntry>, Box<dyn std::error::Error>> {
    // Parse git log with conventional commits format
    let log_output = run_git_command(repo_path, &[
        "log", 
        &format!("-{}", limit),
        "--format=%H|%an|%ct|%s",
        "--reverse"
    ])?;

    let mut changelog = Vec::new();
    
    for line in log_output.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 4 {
            let commit = parts[0];
            let author = parts[1];
            let timestamp = DateTime::<Utc>::from_timestamp(parts[2].parse()?, 0).unwrap();
            let message = parts[3];
            
            // Parse conventional commit message
            if let Some(change_item) = parse_conventional_commit(message) {
                // Group by version/tag if available
                let version = get_version_for_commit(repo_path, commit)?
                    .unwrap_or_else(|| format!("dev-{}", commit[..8].to_string()));
                
                // Find or create changelog entry for this version
                if let Some(entry) = changelog.iter_mut().find(|e| e.version == version) {
                    entry.changes.push(change_item);
                } else {
                    changelog.push(ChangelogEntry {
                        version,
                        date: timestamp,
                        author: author.to_string(),
                        changes: vec![change_item],
                    });
                }
            }
        }
    }

    Ok(changelog)
}

fn get_mcp_modifications(repo_path: &Path, since: DateTime<Utc>) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    // Find modifications to MCP-related files since the given timestamp
    let since_str = since.format("%Y-%m-%d").to_string();
    
    let log_output = run_git_command(repo_path, &[
        "log",
        &format!("--since={}", since_str),
        "--name-only",
        "--format=%H|%an|%ct|%s",
        "--",
        "src/mcp/",
        "**/*mcp*",
        "**/tools/",
        "**/handlers/"
    ])?;

    let mut modifications = Vec::new();
    let lines: Vec<&str> = log_output.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].contains('|') {
            // Parse commit info
            let parts: Vec<&str> = lines[i].split('|').collect();
            if parts.len() >= 4 {
                let commit = parts[0];
                let author = parts[1];
                let timestamp = DateTime::<Utc>::from_timestamp(parts[2].parse()?, 0).unwrap();
                let message = parts[3];
                
                i += 1;
                
                // Collect modified files
                let mut files = Vec::new();
                while i < lines.len() && !lines[i].contains('|') && !lines[i].is_empty() {
                    files.push(lines[i]);
                    i += 1;
                }
                
                // Analyze impact of changes
                for file in files {
                    if is_mcp_related_file(file) {
                        modifications.push(serde_json::json!({
                            "tool": extract_tool_name(file),
                            "type": determine_modification_type(message),
                            "description": message,
                            "impact": assess_impact_level(file, message),
                            "timestamp": timestamp.timestamp_millis(),
                            "author": author,
                            "file": file,
                            "commit": commit
                        }));
                    }
                }
            }
        }
        i += 1;
    }

    Ok(modifications)
}

fn run_git_command(repo_path: &Path, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(args)
        .output()?;
    
    if !output.status.success() {
        return Err(format!(
            "Git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ).into());
    }
    
    Ok(String::from_utf8(output.stdout)?)
}

// Helper functions
fn parse_conventional_commit(message: &str) -> Option<ChangeItem> {
    // Parse conventional commit format: type(scope): description
    let re = regex::Regex::new(r"^(feat|fix|docs|style|refactor|test|chore)(\([^)]+\))?: (.+)$").ok()?;
    
    if let Some(captures) = re.captures(message) {
        let change_type = match &captures[1] {
            "feat" => ChangeType::Feature,
            "fix" => ChangeType::Fix,
            "docs" => ChangeType::Documentation,
            "refactor" => ChangeType::Refactor,
            "test" => ChangeType::Test,
            _ => ChangeType::Enhancement,
        };
        
        let component = captures.get(2)
            .map(|m| m.as_str().trim_matches(|c| c == '(' || c == ')').to_string());
        
        let description = captures[3].to_string();
        let breaking = message.contains("BREAKING CHANGE") || message.contains('!');
        
        return Some(ChangeItem {
            change_type,
            description,
            component,
            breaking,
        });
    }
    
    None
}

fn determine_environment(branch: &str) -> Result<Environment, Box<dyn std::error::Error>> {
    match branch {
        "main" | "master" => Ok(Environment::Production),
        "staging" | "develop" => Ok(Environment::Staging),
        _ => Ok(Environment::Development),
    }
}

fn is_mcp_related_file(file: &str) -> bool {
    file.contains("mcp") || 
    file.contains("tools/") || 
    file.contains("handlers/") ||
    file.ends_with("_mcp.rs") ||
    file.ends_with("_tool.rs")
}

fn extract_tool_name(file: &str) -> String {
    // Extract tool name from file path
    if let Some(name) = file.split('/').last() {
        name.replace("_mcp.rs", "")
            .replace("_tool.rs", "")
            .replace(".rs", "")
    } else {
        "unknown".to_string()
    }
}
```

### Version Comparison Features
```typescript
// src/components/VersionControl/VersionComparison.tsx
import React, { useState, useEffect } from 'react';
import { Card, Select, Button, Row, Col, Tabs, Table, Tag, Space } from 'antd';
import { DiffOutlined, DownloadOutlined, ShareAltOutlined } from '@ant-design/icons';
import MonacoEditor from '@monaco-editor/react';
import { useLLMKG } from '../../integration/VisualizationCore';

const { Option } = Select;
const { TabPane } = Tabs;

interface ComparisonData {
  filesChanged: FileChange[];
  additions: number;
  deletions: number;
  summary: {
    features: number;
    fixes: number;
    breaking: number;
  };
}

interface FileChange {
  path: string;
  status: 'added' | 'deleted' | 'modified' | 'renamed';
  additions: number;
  deletions: number;
  diff: string;
}

export const VersionComparison: React.FC = () => {
  const [availableVersions, setAvailableVersions] = useState<string[]>([]);
  const [fromVersion, setFromVersion] = useState<string>('');
  const [toVersion, setToVersion] = useState<string>('');
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(false);
  const { mcpClient } = useLLMKG();

  useEffect(() => {
    loadAvailableVersions();
  }, []);

  const loadAvailableVersions = async () => {
    try {
      const versions = await mcpClient?.request('version/getAvailableVersions', {});
      setAvailableVersions(versions || []);
    } catch (error) {
      console.error('Failed to load versions:', error);
    }
  };

  const performComparison = async () => {
    if (!fromVersion || !toVersion) return;
    
    setLoading(true);
    try {
      const result = await mcpClient?.request('version/compareVersions', {
        from: fromVersion,
        to: toVersion
      });
      setComparison(result);
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportComparison = () => {
    if (!comparison) return;
    
    const data = {
      comparison,
      metadata: {
        fromVersion,
        toVersion,
        generatedAt: new Date().toISOString()
      }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `comparison-${fromVersion}-to-${toVersion}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="version-comparison">
      <Card title="Version Comparison Tool" size="default">
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Select
              placeholder="From Version"
              style={{ width: '100%' }}
              value={fromVersion}
              onChange={setFromVersion}
            >
              {availableVersions.map(version => (
                <Option key={version} value={version}>{version}</Option>
              ))}
            </Select>
          </Col>
          <Col span={6}>
            <Select
              placeholder="To Version"
              style={{ width: '100%' }}
              value={toVersion}
              onChange={setToVersion}
            >
              {availableVersions.map(version => (
                <Option key={version} value={version}>{version}</Option>
              ))}
            </Select>
          </Col>
          <Col span={6}>
            <Button 
              type="primary" 
              icon={<DiffOutlined />}
              onClick={performComparison}
              loading={loading}
              disabled={!fromVersion || !toVersion}
            >
              Compare
            </Button>
          </Col>
          <Col span={6}>
            <Space>
              <Button 
                icon={<DownloadOutlined />}
                onClick={exportComparison}
                disabled={!comparison}
              >
                Export
              </Button>
              <Button 
                icon={<ShareAltOutlined />}
                disabled={!comparison}
              >
                Share
              </Button>
            </Space>
          </Col>
        </Row>

        {comparison && (
          <Tabs defaultActiveKey="summary">
            <TabPane tab="Summary" key="summary">
              <Row gutter={16}>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', color: '#52c41a' }}>
                        +{comparison.additions}
                      </div>
                      <div>Additions</div>
                    </div>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', color: '#ff4d4f' }}>
                        -{comparison.deletions}
                      </div>
                      <div>Deletions</div>
                    </div>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', color: '#1890ff' }}>
                        {comparison.filesChanged.length}
                      </div>
                      <div>Files Changed</div>
                    </div>
                  </Card>
                </Col>
              </Row>

              <Card title="Change Summary" style={{ marginTop: 16 }} size="small">
                <Space wrap>
                  <Tag color="green">
                    {comparison.summary.features} Features
                  </Tag>
                  <Tag color="blue">
                    {comparison.summary.fixes} Fixes
                  </Tag>
                  {comparison.summary.breaking > 0 && (
                    <Tag color="red">
                      {comparison.summary.breaking} Breaking Changes
                    </Tag>
                  )}
                </Space>
              </Card>
            </TabPane>

            <TabPane tab="Files Changed" key="files">
              <Table
                dataSource={comparison.filesChanged}
                columns={[
                  {
                    title: 'File',
                    dataIndex: 'path',
                    key: 'path',
                    render: (path) => <span style={{ fontFamily: 'monospace' }}>{path}</span>
                  },
                  {
                    title: 'Status',
                    dataIndex: 'status',
                    key: 'status',
                    render: (status) => (
                      <Tag color={
                        status === 'added' ? 'green' :
                        status === 'deleted' ? 'red' :
                        status === 'modified' ? 'blue' : 'orange'
                      }>
                        {status.toUpperCase()}
                      </Tag>
                    )
                  },
                  {
                    title: 'Changes',
                    key: 'changes',
                    render: (_, record) => (
                      <Space>
                        <span style={{ color: '#52c41a' }}>+{record.additions}</span>
                        <span style={{ color: '#ff4d4f' }}>-{record.deletions}</span>
                      </Space>
                    )
                  }
                ]}
                expandable={{
                  expandedRowRender: (record) => (
                    <MonacoEditor
                      height="300px"
                      language="diff"
                      value={record.diff}
                      theme="vs-dark"
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        scrollBeyondLastLine: false
                      }}
                    />
                  )
                }}
                size="small"
              />
            </TabPane>
          </Tabs>
        )}
      </Card>
    </div>
  );
};
```

## Deliverables Checklist

- [ ] Unified visualization framework with context provider
- [ ] Data pipeline for real-time updates
- [ ] Interactive documentation system
- [ ] Version control information & deployment tracking
- [ ] Development workflow configuration
- [ ] Production build optimization
- [ ] Component communication system
- [ ] Error handling and recovery
- [ ] Performance monitoring integration
- [ ] Comprehensive test suite
- [ ] Deployment configurations
- [ ] API documentation generation
- [ ] Developer onboarding guide