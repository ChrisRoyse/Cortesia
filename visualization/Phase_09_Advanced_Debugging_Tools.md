# Phase 9: Advanced Debugging Tools

## Overview
Phase 9 focuses on developing sophisticated debugging tools specifically designed for LLMKG's brain-inspired architecture, including SDR inspection, cognitive state debugging, MCP protocol analysis, and time-travel debugging capabilities.

## Objectives
1. **SDR Debugging Tools**
   - Inspect individual SDR bit patterns
   - Visualize SDR overlaps and unions
   - Debug SDR encoding/decoding issues
   - Analyze semantic similarity preservation

2. **Cognitive State Debugger**
   - Step through cognitive processing
   - Inspect layer states at each step
   - Debug activation spreading
   - Analyze inhibitory circuit behavior

3. **MCP Protocol Analyzer**
   - Monitor MCP message flow
   - Debug protocol handshakes
   - Analyze message latency
   - Inspect tool invocations

4. **Time-Travel Debugging**
   - Record and replay system states
   - Step backward through processing
   - Compare different execution paths
   - Identify state divergence points

## Technical Implementation

### Advanced Debugging Dashboard
```typescript
// src/components/AdvancedDebugger.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Tabs, Card, Row, Col, Button, Select, Input, Table, Tag, Timeline, Collapse } from 'antd';
import { 
  BugOutlined, 
  StepBackwardOutlined, 
  StepForwardOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SearchOutlined
} from '@ant-design/icons';
import MonacoEditor from '@monaco-editor/react';
import { useMCPConnection } from '../hooks/useMCPConnection';
import SDRInspector from './debugger/SDRInspector';
import CognitiveStateDebugger from './debugger/CognitiveStateDebugger';
import MCPAnalyzer from './debugger/MCPAnalyzer';
import TimeTravelDebugger from './debugger/TimeTravelDebugger';

const { TabPane } = Tabs;
const { Panel } = Collapse;

interface DebugSession {
  id: string;
  startTime: number;
  endTime?: number;
  states: SystemState[];
  breakpoints: Breakpoint[];
  currentIndex: number;
}

interface SystemState {
  timestamp: number;
  cognitive: CognitiveState;
  sdrs: SDRState;
  mcp: MCPState;
  memory: MemorySnapshot;
}

interface Breakpoint {
  id: string;
  type: 'cognitive' | 'sdr' | 'mcp' | 'conditional';
  condition?: string;
  location?: string;
  enabled: boolean;
}

export const AdvancedDebugger: React.FC = () => {
  const { client, connected } = useMCPConnection();
  const [activeTab, setActiveTab] = useState('sdr');
  const [session, setSession] = useState<DebugSession | null>(null);
  const [recording, setRecording] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [breakpoints, setBreakpoints] = useState<Breakpoint[]>([]);

  const startDebugging = async () => {
    if (!client) return;

    try {
      const response = await client.request('debug/startSession', {
        recordStates: true,
        enableBreakpoints: true
      });

      setSession({
        id: response.sessionId,
        startTime: Date.now(),
        states: [],
        breakpoints: [],
        currentIndex: 0
      });

      setRecording(true);
    } catch (error) {
      console.error('Failed to start debug session:', error);
    }
  };

  const stopDebugging = async () => {
    if (!client || !session) return;

    try {
      await client.request('debug/stopSession', {
        sessionId: session.id
      });

      setRecording(false);
      setPlaying(false);
    } catch (error) {
      console.error('Failed to stop debug session:', error);
    }
  };

  return (
    <div className="advanced-debugger">
      <h2>Advanced Debugging Tools</h2>

      {/* Debug Controls */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={12}>
            <Button
              icon={recording ? <PauseCircleOutlined /> : <BugOutlined />}
              onClick={recording ? stopDebugging : startDebugging}
              type={recording ? 'danger' : 'primary'}
            >
              {recording ? 'Stop Debugging' : 'Start Debugging'}
            </Button>
            
            {session && (
              <>
                <Button
                  icon={<StepBackwardOutlined />}
                  onClick={() => stepBackward()}
                  disabled={session.currentIndex === 0}
                  style={{ marginLeft: 8 }}
                >
                  Step Back
                </Button>
                <Button
                  icon={playing ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                  onClick={() => setPlaying(!playing)}
                  style={{ marginLeft: 8 }}
                >
                  {playing ? 'Pause' : 'Play'}
                </Button>
                <Button
                  icon={<StepForwardOutlined />}
                  onClick={() => stepForward()}
                  disabled={session.currentIndex >= session.states.length - 1}
                  style={{ marginLeft: 8 }}
                >
                  Step Forward
                </Button>
              </>
            )}
          </Col>
          <Col span={12} style={{ textAlign: 'right' }}>
            {session && (
              <Tag color="blue">
                State: {session.currentIndex + 1} / {session.states.length}
              </Tag>
            )}
          </Col>
        </Row>
      </Card>

      {/* Debug Tabs */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="SDR Inspector" key="sdr">
          <SDRInspector session={session} />
        </TabPane>
        <TabPane tab="Cognitive State" key="cognitive">
          <CognitiveStateDebugger session={session} />
        </TabPane>
        <TabPane tab="MCP Analyzer" key="mcp">
          <MCPAnalyzer session={session} />
        </TabPane>
        <TabPane tab="Time Travel" key="timetravel">
          <TimeTravelDebugger session={session} onStateChange={handleStateChange} />
        </TabPane>
        <TabPane tab="Breakpoints" key="breakpoints">
          <BreakpointManager 
            breakpoints={breakpoints} 
            onBreakpointsChange={setBreakpoints} 
          />
        </TabPane>
      </Tabs>
    </div>
  );

  function stepBackward() {
    if (!session || session.currentIndex === 0) return;
    
    setSession({
      ...session,
      currentIndex: session.currentIndex - 1
    });
  }

  function stepForward() {
    if (!session || session.currentIndex >= session.states.length - 1) return;
    
    setSession({
      ...session,
      currentIndex: session.currentIndex + 1
    });
  }

  function handleStateChange(index: number) {
    if (!session) return;
    
    setSession({
      ...session,
      currentIndex: index
    });
  }
};

// SDR Inspector Component
const SDRInspector: React.FC<{ session: DebugSession | null }> = ({ session }) => {
  const [selectedSDR, setSelectedSDR] = useState<string | null>(null);
  const [comparisonSDR, setComparisonSDR] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentState = session?.states[session.currentIndex];
  const sdrs = currentState?.sdrs?.active || [];

  useEffect(() => {
    if (!canvasRef.current || !selectedSDR) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw SDR bit pattern
    const sdr = sdrs.find(s => s.id === selectedSDR);
    if (!sdr) return;

    const bitSize = 4;
    const cols = Math.sqrt(sdr.bits.length);
    const rows = Math.ceil(sdr.bits.length / cols);

    canvas.width = cols * bitSize;
    canvas.height = rows * bitSize;

    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw bits
    sdr.bits.forEach((bit, idx) => {
      const x = (idx % cols) * bitSize;
      const y = Math.floor(idx / cols) * bitSize;
      
      if (bit) {
        ctx.fillStyle = '#00ff00';
        ctx.fillRect(x, y, bitSize - 1, bitSize - 1);
      }
      
      // Draw comparison overlay
      if (comparisonSDR) {
        const compSdr = sdrs.find(s => s.id === comparisonSDR);
        if (compSdr && compSdr.bits[idx] && !bit) {
          ctx.fillStyle = '#ff0000';
          ctx.fillRect(x, y, bitSize - 1, bitSize - 1);
        } else if (compSdr && compSdr.bits[idx] && bit) {
          ctx.fillStyle = '#ffff00';
          ctx.fillRect(x, y, bitSize - 1, bitSize - 1);
        }
      }
    });

  }, [selectedSDR, comparisonSDR, sdrs]);

  return (
    <div>
      <Row gutter={16}>
        <Col span={8}>
          <Card title="Active SDRs" style={{ height: 600, overflow: 'auto' }}>
            <Table
              dataSource={sdrs}
              columns={[
                {
                  title: 'ID',
                  dataIndex: 'id',
                  key: 'id',
                  render: (id) => (
                    <a onClick={() => setSelectedSDR(id)}>{id}</a>
                  )
                },
                {
                  title: 'Sparsity',
                  dataIndex: 'sparsity',
                  key: 'sparsity',
                  render: (s) => `${(s * 100).toFixed(2)}%`
                },
                {
                  title: 'Active Bits',
                  dataIndex: 'activeBits',
                  key: 'activeBits'
                }
              ]}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
        
        <Col span={16}>
          <Card title="SDR Bit Pattern Visualization">
            <Row>
              <Col span={12}>
                <h4>Selected SDR: {selectedSDR || 'None'}</h4>
                <canvas ref={canvasRef} style={{ border: '1px solid #ccc' }} />
              </Col>
              <Col span={12}>
                {selectedSDR && (
                  <div>
                    <h4>SDR Details</h4>
                    <Select
                      placeholder="Compare with..."
                      style={{ width: '100%', marginBottom: 16 }}
                      value={comparisonSDR}
                      onChange={setComparisonSDR}
                      allowClear
                    >
                      {sdrs.filter(s => s.id !== selectedSDR).map(s => (
                        <Select.Option key={s.id} value={s.id}>
                          {s.id}
                        </Select.Option>
                      ))}
                    </Select>
                    
                    {comparisonSDR && (
                      <SDRComparison 
                        sdr1={sdrs.find(s => s.id === selectedSDR)!}
                        sdr2={sdrs.find(s => s.id === comparisonSDR)!}
                      />
                    )}
                  </div>
                )}
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// SDR Comparison Component
const SDRComparison: React.FC<{ sdr1: any; sdr2: any }> = ({ sdr1, sdr2 }) => {
  const overlap = calculateOverlap(sdr1.bits, sdr2.bits);
  const union = calculateUnion(sdr1.bits, sdr2.bits);
  const similarity = overlap / union;

  return (
    <div>
      <p>Overlap: {overlap} bits</p>
      <p>Union: {union} bits</p>
      <p>Similarity: {(similarity * 100).toFixed(2)}%</p>
      <div style={{ marginTop: 16 }}>
        <Tag color="green">SDR1 Only</Tag>
        <Tag color="red">SDR2 Only</Tag>
        <Tag color="yellow">Both</Tag>
      </div>
    </div>
  );
};

// Breakpoint Manager Component
const BreakpointManager: React.FC<{
  breakpoints: Breakpoint[];
  onBreakpointsChange: (breakpoints: Breakpoint[]) => void;
}> = ({ breakpoints, onBreakpointsChange }) => {
  const [newBreakpoint, setNewBreakpoint] = useState<Partial<Breakpoint>>({
    type: 'cognitive',
    enabled: true
  });

  const addBreakpoint = () => {
    const breakpoint: Breakpoint = {
      id: `bp_${Date.now()}`,
      type: newBreakpoint.type as any,
      condition: newBreakpoint.condition,
      location: newBreakpoint.location,
      enabled: true
    };

    onBreakpointsChange([...breakpoints, breakpoint]);
    setNewBreakpoint({ type: 'cognitive', enabled: true });
  };

  return (
    <div>
      <Card title="Add Breakpoint" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Select
              value={newBreakpoint.type}
              onChange={(type) => setNewBreakpoint({ ...newBreakpoint, type })}
              style={{ width: '100%' }}
            >
              <Select.Option value="cognitive">Cognitive State</Select.Option>
              <Select.Option value="sdr">SDR Operation</Select.Option>
              <Select.Option value="mcp">MCP Message</Select.Option>
              <Select.Option value="conditional">Conditional</Select.Option>
            </Select>
          </Col>
          <Col span={12}>
            <Input
              placeholder="Condition (e.g., activation > 0.8)"
              value={newBreakpoint.condition}
              onChange={(e) => setNewBreakpoint({ ...newBreakpoint, condition: e.target.value })}
            />
          </Col>
          <Col span={6}>
            <Button type="primary" onClick={addBreakpoint} block>
              Add Breakpoint
            </Button>
          </Col>
        </Row>
      </Card>

      <Card title="Active Breakpoints">
        <Table
          dataSource={breakpoints}
          columns={[
            {
              title: 'Type',
              dataIndex: 'type',
              key: 'type',
              render: (type) => <Tag color="blue">{type}</Tag>
            },
            {
              title: 'Condition',
              dataIndex: 'condition',
              key: 'condition'
            },
            {
              title: 'Enabled',
              dataIndex: 'enabled',
              key: 'enabled',
              render: (enabled, record) => (
                <Switch
                  checked={enabled}
                  onChange={(checked) => {
                    const updated = breakpoints.map(bp =>
                      bp.id === record.id ? { ...bp, enabled: checked } : bp
                    );
                    onBreakpointsChange(updated);
                  }}
                />
              )
            },
            {
              title: 'Actions',
              key: 'actions',
              render: (_, record) => (
                <Button
                  type="link"
                  danger
                  onClick={() => {
                    onBreakpointsChange(breakpoints.filter(bp => bp.id !== record.id));
                  }}
                >
                  Remove
                </Button>
              )
            }
          ]}
          pagination={false}
        />
      </Card>
    </div>
  );
};

// Helper functions
function calculateOverlap(bits1: boolean[], bits2: boolean[]): number {
  return bits1.reduce((count, bit, idx) => 
    count + (bit && bits2[idx] ? 1 : 0), 0
  );
}

function calculateUnion(bits1: boolean[], bits2: boolean[]): number {
  return bits1.reduce((count, bit, idx) => 
    count + (bit || bits2[idx] ? 1 : 0), 0
  );
}
```

### Debugging Backend Implementation
```rust
// src/debugging/advanced_debugger.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use crate::cognitive::{CognitiveLayer, LayerType};
use crate::sdr::{SDR, SDRProcessor};
use crate::mcp::MCPInterface;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSession {
    pub id: String,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub states: VecDeque<SystemState>,
    pub breakpoints: Vec<Breakpoint>,
    pub current_index: usize,
    pub max_states: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub timestamp: u64,
    pub cognitive: CognitiveState,
    pub sdrs: SDRState,
    pub mcp: MCPState,
    pub memory: MemorySnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub layers: HashMap<String, LayerState>,
    pub active_patterns: Vec<Pattern>,
    pub inhibitory_state: InhibitoryState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerState {
    pub layer_type: LayerType,
    pub neurons: Vec<NeuronState>,
    pub activation_sum: f32,
    pub inhibition_sum: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub id: usize,
    pub activation: f32,
    pub potential: f32,
    pub refractory: bool,
    pub connections: Vec<(usize, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub neurons: Vec<usize>,
    pub strength: f32,
    pub layer: LayerType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InhibitoryState {
    pub global_inhibition: f32,
    pub lateral_inhibition: HashMap<usize, f32>,
    pub competitive_winners: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRState {
    pub active: Vec<SDRInfo>,
    pub recent_operations: Vec<SDROperation>,
    pub overlap_matrix: HashMap<(String, String), f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDRInfo {
    pub id: String,
    pub bits: Vec<bool>,
    pub sparsity: f32,
    pub active_bits: usize,
    pub semantic_fingerprint: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDROperation {
    pub timestamp: u64,
    pub operation_type: String,
    pub input_sdrs: Vec<String>,
    pub output_sdr: Option<String>,
    pub duration_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPState {
    pub active_connections: Vec<MCPConnection>,
    pub message_queue: Vec<MCPMessage>,
    pub tool_invocations: Vec<ToolInvocation>,
    pub protocol_stats: ProtocolStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConnection {
    pub id: String,
    pub status: String,
    pub latency_ms: f32,
    pub messages_sent: u32,
    pub messages_received: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMessage {
    pub id: String,
    pub timestamp: u64,
    pub message_type: String,
    pub source: String,
    pub destination: String,
    pub payload_size: usize,
    pub processing_time_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub tool_name: String,
    pub timestamp: u64,
    pub arguments: HashMap<String, String>,
    pub result: Option<String>,
    pub duration_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStats {
    pub total_messages: u64,
    pub average_latency_ms: f32,
    pub error_rate: f32,
    pub throughput_per_second: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub total_allocated: usize,
    pub cognitive_memory: usize,
    pub sdr_memory: usize,
    pub mcp_memory: usize,
    pub allocations: Vec<MemoryAllocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub address: usize,
    pub size: usize,
    pub component: String,
    pub allocation_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub id: String,
    pub breakpoint_type: BreakpointType,
    pub condition: Option<String>,
    pub location: Option<String>,
    pub enabled: bool,
    pub hit_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointType {
    Cognitive,
    SDR,
    MCP,
    Conditional,
}

pub struct AdvancedDebugger {
    sessions: Arc<Mutex<HashMap<String, DebugSession>>>,
    cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
    sdr_processor: Arc<Mutex<SDRProcessor>>,
    mcp_interface: Arc<Mutex<MCPInterface>>,
    breakpoint_evaluator: BreakpointEvaluator,
}

impl AdvancedDebugger {
    pub fn new(
        cognitive_layers: Vec<Arc<Mutex<CognitiveLayer>>>,
        sdr_processor: Arc<Mutex<SDRProcessor>>,
        mcp_interface: Arc<Mutex<MCPInterface>>,
    ) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            cognitive_layers,
            sdr_processor,
            mcp_interface,
            breakpoint_evaluator: BreakpointEvaluator::new(),
        }
    }

    pub fn start_session(&self, record_states: bool, enable_breakpoints: bool) -> String {
        let session_id = format!("debug_{}", uuid::Uuid::new_v4());
        
        let session = DebugSession {
            id: session_id.clone(),
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            end_time: None,
            states: VecDeque::new(),
            breakpoints: Vec::new(),
            current_index: 0,
            max_states: 10000, // Keep last 10k states
        };

        self.sessions.lock().unwrap().insert(session_id.clone(), session);

        // Start recording thread if enabled
        if record_states {
            self.start_recording(session_id.clone());
        }

        session_id
    }

    fn start_recording(&self, session_id: String) {
        let sessions = self.sessions.clone();
        let cognitive_layers = self.cognitive_layers.clone();
        let sdr_processor = self.sdr_processor.clone();
        let mcp_interface = self.mcp_interface.clone();

        std::thread::spawn(move || {
            loop {
                // Check if session still exists
                let mut sessions_guard = sessions.lock().unwrap();
                if let Some(session) = sessions_guard.get_mut(&session_id) {
                    // Capture current state
                    let state = Self::capture_system_state(
                        &cognitive_layers,
                        &sdr_processor,
                        &mcp_interface,
                    );

                    // Add to session states
                    session.states.push_back(state);
                    
                    // Limit state history
                    if session.states.len() > session.max_states {
                        session.states.pop_front();
                    }
                } else {
                    // Session ended
                    break;
                }
                drop(sessions_guard);

                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
    }

    fn capture_system_state(
        cognitive_layers: &[Arc<Mutex<CognitiveLayer>>],
        sdr_processor: &Arc<Mutex<SDRProcessor>>,
        mcp_interface: &Arc<Mutex<MCPInterface>>,
    ) -> SystemState {
        let cognitive = Self::capture_cognitive_state(cognitive_layers);
        let sdrs = Self::capture_sdr_state(sdr_processor);
        let mcp = Self::capture_mcp_state(mcp_interface);
        let memory = Self::capture_memory_snapshot();

        SystemState {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            cognitive,
            sdrs,
            mcp,
            memory,
        }
    }

    fn capture_cognitive_state(layers: &[Arc<Mutex<CognitiveLayer>>]) -> CognitiveState {
        let mut layer_states = HashMap::new();
        let mut active_patterns = Vec::new();
        let mut inhibitory_state = InhibitoryState {
            global_inhibition: 0.0,
            lateral_inhibition: HashMap::new(),
            competitive_winners: Vec::new(),
        };

        for layer in layers {
            let layer_guard = layer.lock().unwrap();
            let layer_type = layer_guard.layer_type();
            
            // Capture neuron states
            let neurons: Vec<NeuronState> = layer_guard.get_neurons()
                .iter()
                .enumerate()
                .map(|(id, neuron)| NeuronState {
                    id,
                    activation: neuron.activation,
                    potential: neuron.potential,
                    refractory: neuron.is_refractory(),
                    connections: neuron.get_connections()
                        .iter()
                        .map(|&(target, weight)| (target, weight))
                        .collect(),
                })
                .collect();

            let activation_sum = neurons.iter().map(|n| n.activation).sum();
            let inhibition_sum = layer_guard.get_inhibition_sum();

            layer_states.insert(
                format!("{:?}", layer_type),
                LayerState {
                    layer_type,
                    neurons,
                    activation_sum,
                    inhibition_sum,
                }
            );

            // Extract active patterns
            let patterns = layer_guard.get_active_patterns();
            active_patterns.extend(patterns.into_iter().map(|p| Pattern {
                id: p.id,
                neurons: p.active_neurons,
                strength: p.strength,
                layer: layer_type,
            }));

            // Capture inhibitory state
            if let Some(inhibition) = layer_guard.get_inhibitory_state() {
                inhibitory_state.global_inhibition = inhibition.global_level;
                inhibitory_state.lateral_inhibition.extend(inhibition.lateral_pairs);
                inhibitory_state.competitive_winners.extend(inhibition.winners);
            }
        }

        CognitiveState {
            layers: layer_states,
            active_patterns,
            inhibitory_state,
        }
    }

    fn capture_sdr_state(processor: &Arc<Mutex<SDRProcessor>>) -> SDRState {
        let processor_guard = processor.lock().unwrap();
        
        let active_sdrs = processor_guard.get_active_sdrs();
        let active: Vec<SDRInfo> = active_sdrs.iter().map(|sdr| {
            SDRInfo {
                id: sdr.id.clone(),
                bits: sdr.bits.clone(),
                sparsity: sdr.sparsity(),
                active_bits: sdr.active_bit_count(),
                semantic_fingerprint: sdr.semantic_fingerprint(),
            }
        }).collect();

        let recent_operations = processor_guard.get_recent_operations(100);
        
        // Calculate overlap matrix for active SDRs
        let mut overlap_matrix = HashMap::new();
        for i in 0..active.len() {
            for j in i+1..active.len() {
                let overlap = SDR::overlap(&active[i].bits, &active[j].bits);
                overlap_matrix.insert(
                    (active[i].id.clone(), active[j].id.clone()),
                    overlap
                );
            }
        }

        SDRState {
            active,
            recent_operations,
            overlap_matrix,
        }
    }

    fn capture_mcp_state(interface: &Arc<Mutex<MCPInterface>>) -> MCPState {
        let interface_guard = interface.lock().unwrap();
        
        MCPState {
            active_connections: interface_guard.get_active_connections(),
            message_queue: interface_guard.get_message_queue(50),
            tool_invocations: interface_guard.get_recent_tool_invocations(20),
            protocol_stats: interface_guard.get_protocol_stats(),
        }
    }

    fn capture_memory_snapshot() -> MemorySnapshot {
        // This would integrate with actual memory tracking
        MemorySnapshot {
            total_allocated: 0,
            cognitive_memory: 0,
            sdr_memory: 0,
            mcp_memory: 0,
            allocations: Vec::new(),
        }
    }

    pub fn stop_session(&self, session_id: &str) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.end_time = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            );
        }
    }

    pub fn add_breakpoint(&self, session_id: &str, breakpoint: Breakpoint) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.breakpoints.push(breakpoint);
        }
    }

    pub fn evaluate_breakpoints(&self, session_id: &str, state: &SystemState) -> Vec<String> {
        let sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get(session_id) {
            self.breakpoint_evaluator.evaluate(&session.breakpoints, state)
        } else {
            Vec::new()
        }
    }

    pub fn get_session_state(&self, session_id: &str) -> Option<DebugSession> {
        self.sessions.lock().unwrap().get(session_id).cloned()
    }

    pub fn step_forward(&self, session_id: &str) -> Option<SystemState> {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            if session.current_index < session.states.len() - 1 {
                session.current_index += 1;
                session.states.get(session.current_index).cloned()
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn step_backward(&self, session_id: &str) -> Option<SystemState> {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            if session.current_index > 0 {
                session.current_index -= 1;
                session.states.get(session.current_index).cloned()
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn jump_to_state(&self, session_id: &str, index: usize) -> Option<SystemState> {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            if index < session.states.len() {
                session.current_index = index;
                session.states.get(index).cloned()
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub struct BreakpointEvaluator {
    expression_evaluator: ExpressionEvaluator,
}

impl BreakpointEvaluator {
    pub fn new() -> Self {
        Self {
            expression_evaluator: ExpressionEvaluator::new(),
        }
    }

    pub fn evaluate(&self, breakpoints: &[Breakpoint], state: &SystemState) -> Vec<String> {
        let mut triggered = Vec::new();

        for breakpoint in breakpoints {
            if !breakpoint.enabled {
                continue;
            }

            let should_break = match &breakpoint.breakpoint_type {
                BreakpointType::Cognitive => self.evaluate_cognitive_breakpoint(breakpoint, &state.cognitive),
                BreakpointType::SDR => self.evaluate_sdr_breakpoint(breakpoint, &state.sdrs),
                BreakpointType::MCP => self.evaluate_mcp_breakpoint(breakpoint, &state.mcp),
                BreakpointType::Conditional => {
                    if let Some(condition) = &breakpoint.condition {
                        self.expression_evaluator.evaluate(condition, state)
                    } else {
                        false
                    }
                }
            };

            if should_break {
                triggered.push(breakpoint.id.clone());
            }
        }

        triggered
    }

    fn evaluate_cognitive_breakpoint(&self, breakpoint: &Breakpoint, state: &CognitiveState) -> bool {
        // Check for specific cognitive conditions
        if let Some(condition) = &breakpoint.condition {
            // Example: "subcortical.activation > 0.8"
            if condition.contains("activation") {
                for (layer_name, layer_state) in &state.layers {
                    if condition.contains(layer_name) {
                        let activation = layer_state.activation_sum / layer_state.neurons.len() as f32;
                        if condition.contains(">") {
                            if let Some(threshold) = condition.split('>').nth(1).and_then(|s| s.trim().parse::<f32>().ok()) {
                                return activation > threshold;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn evaluate_sdr_breakpoint(&self, breakpoint: &Breakpoint, state: &SDRState) -> bool {
        if let Some(condition) = &breakpoint.condition {
            // Example: "sparsity < 0.02"
            if condition.contains("sparsity") {
                for sdr in &state.active {
                    if condition.contains("<") {
                        if let Some(threshold) = condition.split('<').nth(1).and_then(|s| s.trim().parse::<f32>().ok()) {
                            if sdr.sparsity < threshold {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn evaluate_mcp_breakpoint(&self, breakpoint: &Breakpoint, state: &MCPState) -> bool {
        if let Some(condition) = &breakpoint.condition {
            // Example: "latency > 100"
            if condition.contains("latency") {
                if condition.contains(">") {
                    if let Some(threshold) = condition.split('>').nth(1).and_then(|s| s.trim().parse::<f32>().ok()) {
                        return state.protocol_stats.average_latency_ms > threshold;
                    }
                }
            }
        }
        false
    }
}

struct ExpressionEvaluator;

impl ExpressionEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate(&self, expression: &str, state: &SystemState) -> bool {
        // This would implement a proper expression parser/evaluator
        // For now, just a simple implementation
        false
    }
}
```

## LLMKG-Specific Features

### 1. SDR Debugging
- **Bit-Level Inspection**: Visualize individual bits in SDRs
- **Overlap Analysis**: Compare SDRs to understand semantic relationships
- **Encoding/Decoding Trace**: Track how data is converted to/from SDRs
- **Sparsity Validation**: Ensure SDRs maintain proper sparsity

### 2. Cognitive State Debugging
- **Layer-by-Layer Stepping**: Step through each cognitive layer
- **Activation Visualization**: See how activation spreads
- **Inhibition Analysis**: Debug inhibitory circuit behavior
- **Pattern Matching**: Identify when specific patterns emerge

### 3. MCP Protocol Analysis
- **Message Flow Visualization**: Track MCP messages in real-time
- **Latency Analysis**: Identify protocol bottlenecks
- **Tool Invocation Tracking**: Debug tool calls and responses
- **Error Diagnosis**: Analyze protocol errors and recovery

### 4. Time-Travel Capabilities
- **State Recording**: Record complete system states
- **Reverse Debugging**: Step backward through execution
- **State Comparison**: Compare different execution paths
- **Divergence Detection**: Find where execution paths diverge

## Testing Procedures

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_session_creation() {
        let debugger = create_test_debugger();
        let session_id = debugger.start_session(true, true);
        
        let session = debugger.get_session_state(&session_id);
        assert!(session.is_some());
        assert_eq!(session.unwrap().id, session_id);
    }

    #[test]
    fn test_state_capture() {
        let debugger = create_test_debugger();
        let session_id = debugger.start_session(true, false);
        
        // Wait for some states to be captured
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let session = debugger.get_session_state(&session_id).unwrap();
        assert!(!session.states.is_empty());
    }

    #[test]
    fn test_breakpoint_evaluation() {
        let evaluator = BreakpointEvaluator::new();
        let mut state = create_test_state();
        
        let breakpoint = Breakpoint {
            id: "test_bp".to_string(),
            breakpoint_type: BreakpointType::Cognitive,
            condition: Some("subcortical.activation > 0.7".to_string()),
            location: None,
            enabled: true,
            hit_count: 0,
        };
        
        // Set high activation
        state.cognitive.layers.get_mut("subcortical").unwrap().activation_sum = 80.0;
        
        let triggered = evaluator.evaluate(&[breakpoint], &state);
        assert_eq!(triggered.len(), 1);
    }

    #[test]
    fn test_time_travel() {
        let debugger = create_test_debugger();
        let session_id = debugger.start_session(true, false);
        
        // Capture some states
        std::thread::sleep(std::time::Duration::from_millis(200));
        
        // Step forward
        let state1 = debugger.step_forward(&session_id);
        assert!(state1.is_some());
        
        // Step backward
        let state2 = debugger.step_backward(&session_id);
        assert!(state2.is_some());
        
        // Timestamps should be different
        assert_ne!(state1.unwrap().timestamp, state2.unwrap().timestamp);
    }
}
```

### Integration Tests
```typescript
// tests/advanced-debugger.test.tsx
describe('Advanced Debugger', () => {
  it('should start and stop debug sessions', async () => {
    const { container, getByText } = render(<AdvancedDebugger />);
    
    // Start debugging
    fireEvent.click(getByText('Start Debugging'));
    
    await waitFor(() => {
      expect(getByText('Stop Debugging')).toBeInTheDocument();
      expect(container.querySelector('.debug-state-indicator')).toBeInTheDocument();
    });
    
    // Stop debugging
    fireEvent.click(getByText('Stop Debugging'));
    
    await waitFor(() => {
      expect(getByText('Start Debugging')).toBeInTheDocument();
    });
  });

  it('should inspect SDR patterns', async () => {
    const { container } = render(<SDRInspector session={mockSession} />);
    
    // Select an SDR
    const sdrLink = container.querySelector('[data-sdr-id="sdr_001"]');
    fireEvent.click(sdrLink);
    
    await waitFor(() => {
      expect(container.querySelector('canvas')).toBeInTheDocument();
      expect(container.querySelector('.sdr-details')).toHaveTextContent('Sparsity: 2.00%');
    });
  });

  it('should handle breakpoints', async () => {
    const { container, getByText, getByPlaceholderText } = render(<BreakpointManager />);
    
    // Add a breakpoint
    fireEvent.change(getByPlaceholderText('Condition'), {
      target: { value: 'activation > 0.8' }
    });
    fireEvent.click(getByText('Add Breakpoint'));
    
    await waitFor(() => {
      expect(container.querySelector('.breakpoint-item')).toBeInTheDocument();
      expect(container.querySelector('.ant-switch-checked')).toBeInTheDocument();
    });
  });

  it('should support time-travel debugging', async () => {
    const { container, getByText } = render(<TimeTravelDebugger session={mockSession} />);
    
    // Check initial state
    expect(container.querySelector('.state-index')).toHaveTextContent('1 / 100');
    
    // Step forward
    fireEvent.click(getByText('Step Forward'));
    
    await waitFor(() => {
      expect(container.querySelector('.state-index')).toHaveTextContent('2 / 100');
    });
    
    // Step backward
    fireEvent.click(getByText('Step Back'));
    
    await waitFor(() => {
      expect(container.querySelector('.state-index')).toHaveTextContent('1 / 100');
    });
  });
});
```

## Performance Considerations

### 1. State Recording Optimization
- Use circular buffers for state history
- Implement state compression
- Record deltas instead of full states

### 2. Breakpoint Evaluation
- Compile conditions to bytecode
- Cache evaluation results
- Use efficient pattern matching

### 3. Visualization Performance
- Lazy load debug data
- Use virtual scrolling for large datasets
- Implement progressive rendering

## Deliverables Checklist

- [ ] Advanced debugging dashboard interface
- [ ] SDR bit pattern inspector with comparison tools
- [ ] Cognitive state step-through debugger
- [ ] MCP protocol analyzer with message tracking
- [ ] Time-travel debugging with state recording/replay
- [ ] Breakpoint system with conditional breaks
- [ ] State comparison and diff tools
- [ ] Debug session management
- [ ] Performance profiling integration
- [ ] Export functionality for debug sessions
- [ ] Integration with existing monitoring tools
- [ ] Comprehensive debugging documentation