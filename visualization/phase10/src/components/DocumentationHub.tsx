import React, { useState, useEffect } from 'react';
import { Card, Tabs, Typography, Space, Input, Menu, Row, Col, Tag, Button, Modal, Alert } from 'antd';
import { 
  BookOutlined,
  ApiOutlined,
  CodeOutlined,
  PlayCircleOutlined,
  SearchOutlined,
  ExportOutlined,
  ShareAltOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useLLMKG } from '@/integration/VisualizationCore';

const { TabPane } = Tabs;
const { Title, Text, Paragraph } = Typography;
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
  response: any;
}

interface Parameter {
  name: string;
  type: string;
  required: boolean;
  description: string;
}

export const DocumentationHub: React.FC = () => {
  const [sections, setSections] = useState<DocSection[]>([]);
  const [selectedSection, setSelectedSection] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [runOutput, setRunOutput] = useState<string>('');
  const [modalVisible, setModalVisible] = useState(false);
  const { mcpClient, connected } = useLLMKG();

  useEffect(() => {
    loadDocumentation();
  }, []);

  const loadDocumentation = () => {
    const docs: DocSection[] = [
      {
        id: 'getting-started',
        title: 'Getting Started',
        content: `
# Getting Started with LLMKG Visualization

## Overview

The LLMKG Visualization System is a comprehensive brain-inspired cognitive architecture visualization platform that provides real-time monitoring and analysis across multiple phases:

- **Phase 7**: Storage & Memory Monitoring
- **Phase 8**: Cognitive Pattern Visualization  
- **Phase 9**: Advanced Debugging Tools
- **Phase 10**: Unified Integration System

## Quick Start

### Installation

\`\`\`bash
npm install @llmkg/visualization-unified
# or
yarn add @llmkg/visualization-unified
\`\`\`

### Basic Setup

\`\`\`typescript
import { LLMKGVisualizationProvider, UnifiedDashboard } from '@llmkg/visualization-unified';

const config = {
  mcp: {
    endpoint: 'ws://localhost:8080',
    protocol: 'ws'
  },
  visualization: {
    theme: 'dark',
    updateInterval: 1000,
    enableAnimations: true
  }
};

function App() {
  return (
    <LLMKGVisualizationProvider config={config}>
      <UnifiedDashboard />
    </LLMKGVisualizationProvider>
  );
}
\`\`\`

## Key Features

- **Real-time Monitoring**: Live updates from LLMKG brain-inspired cognitive system
- **Cross-Phase Integration**: Seamless navigation between all visualization phases
- **Performance Optimization**: Built-in performance monitoring and optimization
- **Component Registry**: Dynamic component management and configuration
- **Version Control**: Git integration and deployment tracking
        `,
        examples: [
          {
            title: 'Basic Configuration',
            language: 'typescript',
            code: `import { createVisualizationConfig } from '@llmkg/visualization-unified';

const config = createVisualizationConfig({
  mcp: {
    endpoint: 'ws://localhost:8080',
    reconnect: {
      enabled: true,
      maxAttempts: 5,
      delay: 5000
    }
  },
  visualization: {
    theme: 'dark',
    updateInterval: 1000,
    maxDataPoints: 1000,
    enableAnimations: true
  },
  features: {
    enabledPhases: ['phase7', 'phase8', 'phase9', 'phase10']
  }
});`,
            runnable: false
          }
        ]
      },
      {
        id: 'memory-monitoring',
        title: 'Memory System Monitoring (Phase 7)',
        content: `
# Memory System Monitoring

Phase 7 provides comprehensive monitoring of LLMKG's memory systems including SDR storage, knowledge graph memory allocation, and zero-copy optimizations.

## Components

### SDR Storage Visualization
Monitor sparse distributed representation storage with fragmentation analysis.

### Knowledge Graph Memory Treemap
Hierarchical visualization of entity, relation, and embedding memory usage.

### Zero-Copy Performance Monitor
Track zero-copy operations and memory efficiency metrics.

## Usage

\`\`\`typescript
import { MemoryDashboard } from '@llmkg/visualization-unified';

<MemoryDashboard 
  updateInterval={1000}
  enableRealTime={true}
  showFragmentation={true}
/>
\`\`\`
        `,
        examples: [
          {
            title: 'Memory Dashboard Integration',
            language: 'typescript',
            code: `import { MemoryDashboard, useLLMKG } from '@llmkg/visualization-unified';

function MyMemoryMonitor() {
  const { sdrProcessor, knowledgeGraph } = useLLMKG();
  
  return (
    <MemoryDashboard 
      sdrProcessor={sdrProcessor}
      knowledgeGraph={knowledgeGraph}
      updateInterval={1000}
      enableRealTime={true}
    />
  );
}`,
            runnable: true
          }
        ]
      },
      {
        id: 'cognitive-patterns',
        title: 'Cognitive Pattern Visualization (Phase 8)',
        content: `
# Cognitive Pattern Visualization

Phase 8 focuses on visualizing brain-inspired cognitive processing patterns including activation flows, pattern classification, and inhibition/excitation balance.

## Key Concepts

### Pattern Types
- **Convergent**: Information integration patterns
- **Divergent**: Creative exploration patterns  
- **Lateral**: Cross-domain association patterns
- **Systems**: Holistic system-level patterns
- **Critical**: Analytical evaluation patterns

### 3D Visualization
Interactive 3D space showing real-time pattern activation with force-directed layouts.

## Implementation

\`\`\`typescript
import { CognitivePatternDashboard } from '@llmkg/visualization-unified';

<CognitivePatternDashboard
  enable3D={true}
  showInhibition={true}
  patternTypes={['convergent', 'divergent', 'lateral']}
/>
\`\`\`
        `,
        examples: [
          {
            title: 'Pattern Monitoring',
            language: 'typescript',
            code: `import { PatternActivation3D, useLLMKG } from '@llmkg/visualization-unified';

function CognitiveMonitor() {
  const { cognitiveEngine } = useLLMKG();
  
  useEffect(() => {
    const unsubscribe = cognitiveEngine?.subscribe('patterns', (patterns) => {
      console.log('Active patterns:', patterns);
    });
    
    return unsubscribe;
  }, [cognitiveEngine]);
  
  return (
    <PatternActivation3D
      enableRotation={true}
      showConnections={true}
      animationSpeed={1.0}
    />
  );
}`,
            runnable: true
          }
        ]
      },
      {
        id: 'debugging-tools',
        title: 'Advanced Debugging (Phase 9)',
        content: `
# Advanced Debugging Tools

Phase 9 provides sophisticated debugging capabilities including distributed tracing, time-travel debugging, and query analysis.

## Features

### Time-Travel Debugging
Step forward/backward through system states with state snapshots and comparison.

### Distributed Tracing
Visual tracing of operations across distributed LLMKG components.

### Query Analysis
Execution plan visualization with performance optimization suggestions.

## Usage

\`\`\`typescript
import { DebuggingDashboard } from '@llmkg/visualization-unified';

<DebuggingDashboard
  enableTimeTravel={true}
  maxStateHistory={1000}
  enableTracing={true}
/>
\`\`\`
        `,
        examples: [
          {
            title: 'Debug Session Setup',
            language: 'typescript',
            code: `import { TimeTravelDebugger, useLLMKG } from '@llmkg/visualization-unified';

function DebugSession() {
  const { mcpClient } = useLLMKG();
  
  const startDebugSession = async () => {
    const session = await mcpClient?.request('debug/startSession', {
      recordStates: true,
      enableBreakpoints: true
    });
    
    // Set breakpoint on cognitive activation
    await mcpClient?.request('debug/addBreakpoint', {
      sessionId: session.id,
      type: 'cognitive',
      condition: 'cortical.activation > 0.8'
    });
  };
  
  return (
    <TimeTravelDebugger
      onStateChange={(state) => console.log('State:', state)}
      enablePlayback={true}
    />
  );
}`,
            runnable: true
          }
        ]
      },
      {
        id: 'integration-system',
        title: 'Unified Integration (Phase 10)',
        content: `
# Unified Integration System

Phase 10 provides the integration layer that unifies all visualization phases into a cohesive system.

## Architecture

### Component Registry
Central registry for managing and configuring visualization components across phases.

### State Management
Redux-based state management with cross-phase data sharing.

### Performance Monitoring
Real-time performance metrics and optimization suggestions.

### Version Control Integration
Git repository tracking and deployment monitoring.

## Core Services

\`\`\`typescript
import { useAppSelector, useAppDispatch } from '@llmkg/visualization-unified';
import { registerComponent, enablePhase } from '@llmkg/visualization-unified';

// Register a custom component
dispatch(registerComponent({
  id: 'my-custom-viz',
  name: 'Custom Visualization',
  phase: 'phase10',
  component: MyCustomComponent,
  enabled: true
}));

// Enable specific phases
dispatch(enablePhase('phase8'));
\`\`\`
        `,
        examples: [
          {
            title: 'Custom Component Registration',
            language: 'typescript',
            code: `import { registerComponent, useAppDispatch } from '@llmkg/visualization-unified';

function MyApp() {
  const dispatch = useAppDispatch();
  
  useEffect(() => {
    // Register custom visualization component
    dispatch(registerComponent({
      id: 'custom-memory-viz',
      name: 'Custom Memory Visualization',
      phase: 'phase7',
      component: CustomMemoryComponent,
      props: {
        updateInterval: 500,
        theme: 'light'
      },
      dependencies: ['d3', 'react'],
      enabled: true
    }));
  }, [dispatch]);
  
  return <UnifiedDashboard />;
}`,
            runnable: true
          }
        ]
      },
      {
        id: 'api-reference',
        title: 'API Reference',
        content: `
# API Reference

## Core Hooks

### useLLMKG()
Access the main LLMKG context and services.

\`\`\`typescript
const { 
  mcpClient, 
  sdrProcessor, 
  cognitiveEngine, 
  knowledgeGraph,
  connected,
  connectionStatus 
} = useLLMKG();
\`\`\`

### useRealTimeData()
Subscribe to real-time data updates.

\`\`\`typescript
const [data, loading] = useRealTimeData('system/metrics', defaultValue, 1000);
\`\`\`

### useConnectionStatus()
Monitor connection status.

\`\`\`typescript
const { connected, connectionStatus, error } = useConnectionStatus();
\`\`\`

## State Management

### Visualization Slice
\`\`\`typescript
import { updateConfig, toggleDebugMode } from '@llmkg/visualization-unified';

dispatch(updateConfig({ visualization: { theme: 'light' } }));
dispatch(toggleDebugMode());
\`\`\`

### Performance Slice
\`\`\`typescript
import { updateMetrics, startProfiling } from '@llmkg/visualization-unified';

dispatch(updateMetrics({ renderTime: 16.7 }));
dispatch(startProfiling({ sampleRate: 1 }));
\`\`\`
        `,
        examples: [
          {
            title: 'Real-time Data Subscription',
            language: 'typescript',
            code: `import { useRealTimeData, useLLMKG } from '@llmkg/visualization-unified';

function MetricsDisplay() {
  const { connected } = useLLMKG();
  const [metrics, loading] = useRealTimeData('system/metrics', {
    memoryUsage: 0,
    cpuUsage: 0,
    networkLatency: 0
  }, 1000);
  
  if (!connected) return <div>Disconnected</div>;
  if (loading) return <div>Loading...</div>;
  
  return (
    <div>
      <p>Memory: {metrics.memoryUsage}%</p>
      <p>CPU: {metrics.cpuUsage}%</p>
      <p>Latency: {metrics.networkLatency}ms</p>
    </div>
  );
}`,
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
      setRunOutput('Running example...\n');
      
      // Mock execution for demonstration
      setTimeout(() => {
        setRunOutput(`Example executed successfully!\n\nOutput:\n${code.slice(0, 100)}...`);
      }, 1000);
      
    } catch (error) {
      setRunOutput(`Error: ${error}`);
    }
  };

  const filteredSections = sections.filter(s =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const currentSection = sections.find(s => s.id === selectedSection);

  return (
    <div style={{ padding: '16px 0' }}>
      <Title level={2}>
        <BookOutlined /> Documentation Hub
      </Title>
      <Paragraph>
        Comprehensive documentation for the LLMKG Visualization System.
        Explore guides, API references, and interactive examples.
      </Paragraph>

      <Alert
        message="Interactive Documentation"
        description="This documentation hub provides comprehensive guides, API references, and runnable examples for all visualization phases."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={24}>
        <Col span={6}>
          <Card size="small">
            <Search
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ marginBottom: 16 }}
              prefix={<SearchOutlined />}
            />
            
            <Menu
              mode="vertical"
              selectedKeys={[selectedSection]}
              onClick={({ key }) => setSelectedSection(key)}
              style={{ border: 'none' }}
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
              <div style={{ marginBottom: 16 }}>
                <Space>
                  <Button icon={<ExportOutlined />} size="small">
                    Export
                  </Button>
                  <Button icon={<ShareAltOutlined />} size="small">
                    Share
                  </Button>
                </Space>
              </div>

              <div className="doc-content">
                <ReactMarkdown
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={dark}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {currentSection.content}
                </ReactMarkdown>
                
                {currentSection.examples.length > 0 && (
                  <div style={{ marginTop: 32 }}>
                    <Title level={3}>Examples</Title>
                    
                    {currentSection.examples.map((example, idx) => (
                      <Card key={idx} size="small" style={{ marginBottom: 16 }}>
                        <div style={{ marginBottom: 8 }}>
                          <Space>
                            <Text strong>{example.title}</Text>
                            {example.runnable && (
                              <Button
                                icon={<PlayCircleOutlined />}
                                size="small"
                                onClick={() => runExample(example.code)}
                              >
                                Run Example
                              </Button>
                            )}
                          </Space>
                        </div>
                        
                        <SyntaxHighlighter
                          language={example.language}
                          style={dark}
                          customStyle={{ fontSize: '12px' }}
                        >
                          {example.code}
                        </SyntaxHighlighter>
                        
                        {runOutput && example.runnable && (
                          <div style={{ marginTop: 8 }}>
                            <Text strong>Output:</Text>
                            <pre style={{ 
                              background: '#f5f5f5', 
                              padding: '8px', 
                              borderRadius: '4px',
                              fontSize: '12px',
                              marginTop: '4px'
                            }}>
                              {runOutput}
                            </pre>
                          </div>
                        )}
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};