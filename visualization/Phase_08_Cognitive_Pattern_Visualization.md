# Phase 8: Cognitive Pattern Visualization

## Overview
Phase 8 focuses on visualizing the cognitive patterns within LLMKG's brain-inspired architecture, including activation spreading, inhibitory circuits, attention mechanisms, and emergent behavior patterns.

## Objectives
1. **Activation Pattern Visualization**
   - Display neural activation spreading across layers
   - Show temporal dynamics of activation
   - Visualize pattern emergence and stability
   - Track activation pathways

2. **Inhibitory Circuit Visualization**
   - Map inhibitory connections and their effects
   - Show lateral inhibition patterns
   - Visualize competition between patterns
   - Display inhibitory strength dynamics

3. **Attention Mechanism Display**
   - Visualize thalamic attention focusing
   - Show attention switching patterns
   - Display attention-driven modulation
   - Track attention history

4. **Emergent Pattern Analysis**
   - Identify recurring cognitive patterns
   - Visualize pattern hierarchies
   - Show pattern evolution over time
   - Display pattern similarity networks

## Technical Implementation

### Cognitive Pattern Visualizer Component
```typescript
// src/components/CognitivePatternVisualizer.tsx
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Text } from '@react-three/drei';
import { Card, Row, Col, Slider, Switch, Select, Button } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StepForwardOutlined } from '@ant-design/icons';
import { useMCPConnection } from '../hooks/useMCPConnection';
import * as THREE from 'three';

interface CognitivePattern {
  id: string;
  layer: 'subcortical' | 'cortical' | 'thalamic';
  position: [number, number, number];
  activation: number;
  connections: Connection[];
  type: 'excitatory' | 'inhibitory' | 'modulatory';
  timestamp: number;
}

interface Connection {
  source: string;
  target: string;
  strength: number;
  type: 'excitatory' | 'inhibitory';
  active: boolean;
}

interface AttentionFocus {
  center: [number, number, number];
  radius: number;
  strength: number;
  targetPatterns: string[];
}

export const CognitivePatternVisualizer: React.FC = () => {
  const { client, connected } = useMCPConnection();
  const [patterns, setPatterns] = useState<CognitivePattern[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [attentionFocus, setAttentionFocus] = useState<AttentionFocus | null>(null);
  const [playing, setPlaying] = useState(false);
  const [timeStep, setTimeStep] = useState(0);
  const [showInhibitory, setShowInhibitory] = useState(true);
  const [showExcitatory, setShowExcitatory] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState<string>('all');
  const animationRef = useRef<number>();

  useEffect(() => {
    if (!connected || !client) return;

    const fetchCognitiveState = async () => {
      try {
        const response = await client.request('cognitive/getPatternState', {
          includeHistory: true,
          timeRange: 1000 // Last 1000ms
        });

        setPatterns(response.patterns);
        setConnections(response.connections);
        setAttentionFocus(response.attentionFocus);
      } catch (error) {
        console.error('Failed to fetch cognitive state:', error);
      }
    };

    // Initial fetch
    fetchCognitiveState();

    // Set up real-time updates
    const interval = setInterval(fetchCognitiveState, 100);

    return () => clearInterval(interval);
  }, [connected, client]);

  // Animation loop
  useEffect(() => {
    if (playing) {
      const animate = () => {
        setTimeStep(prev => prev + 1);
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [playing]);

  // Filter patterns based on selected layer
  const filteredPatterns = patterns.filter(p => 
    selectedLayer === 'all' || p.layer === selectedLayer
  );

  // Filter connections based on visibility settings
  const filteredConnections = connections.filter(c => {
    if (!showExcitatory && c.type === 'excitatory') return false;
    if (!showInhibitory && c.type === 'inhibitory') return false;
    
    const sourcePattern = patterns.find(p => p.id === c.source);
    const targetPattern = patterns.find(p => p.id === c.target);
    
    return sourcePattern && targetPattern && 
           (selectedLayer === 'all' || 
            sourcePattern.layer === selectedLayer || 
            targetPattern.layer === selectedLayer);
  });

  return (
    <div className="cognitive-pattern-visualizer">
      <h2>Cognitive Pattern Visualization</h2>

      {/* Controls */}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={4}>
            <Button
              icon={playing ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={() => setPlaying(!playing)}
              type="primary"
            >
              {playing ? 'Pause' : 'Play'}
            </Button>
            <Button
              icon={<StepForwardOutlined />}
              onClick={() => setTimeStep(prev => prev + 1)}
              style={{ marginLeft: 8 }}
            >
              Step
            </Button>
          </Col>
          <Col span={6}>
            <div>Layer Filter:</div>
            <Select
              value={selectedLayer}
              onChange={setSelectedLayer}
              style={{ width: '100%' }}
            >
              <Select.Option value="all">All Layers</Select.Option>
              <Select.Option value="subcortical">Subcortical</Select.Option>
              <Select.Option value="cortical">Cortical</Select.Option>
              <Select.Option value="thalamic">Thalamic</Select.Option>
            </Select>
          </Col>
          <Col span={4}>
            <Switch
              checked={showExcitatory}
              onChange={setShowExcitatory}
              checkedChildren="Excitatory"
              unCheckedChildren="Excitatory"
            />
          </Col>
          <Col span={4}>
            <Switch
              checked={showInhibitory}
              onChange={setShowInhibitory}
              checkedChildren="Inhibitory"
              unCheckedChildren="Inhibitory"
            />
          </Col>
          <Col span={6}>
            <div>Time: {timeStep}</div>
            <Slider
              value={timeStep}
              onChange={setTimeStep}
              max={1000}
              marks={{
                0: '0',
                500: '500',
                1000: '1000'
              }}
            />
          </Col>
        </Row>
      </Card>

      {/* 3D Visualization */}
      <Card title="3D Cognitive Pattern View" style={{ marginBottom: 16 }}>
        <div style={{ height: 600 }}>
          <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <OrbitControls enablePan enableZoom enableRotate />

            {/* Layer planes */}
            <LayerPlanes />

            {/* Patterns */}
            {filteredPatterns.map(pattern => (
              <PatternNode
                key={pattern.id}
                pattern={pattern}
                timeStep={timeStep}
              />
            ))}

            {/* Connections */}
            {filteredConnections.map((connection, idx) => (
              <ConnectionLine
                key={idx}
                connection={connection}
                patterns={patterns}
                timeStep={timeStep}
              />
            ))}

            {/* Attention focus */}
            {attentionFocus && (
              <AttentionSphere
                focus={attentionFocus}
                timeStep={timeStep}
              />
            )}
          </Canvas>
        </div>
      </Card>

      {/* Pattern Analysis */}
      <Row gutter={16}>
        <Col span={12}>
          <ActivationHeatmap patterns={patterns} />
        </Col>
        <Col span={12}>
          <InhibitionPatternMap patterns={patterns} connections={connections} />
        </Col>
      </Row>

      {/* Emergent Patterns */}
      <Card title="Emergent Pattern Detection" style={{ marginTop: 16 }}>
        <EmergentPatternAnalysis patterns={patterns} connections={connections} />
      </Card>
    </div>
  );
};

// 3D Components
const PatternNode: React.FC<{ pattern: CognitivePattern; timeStep: number }> = ({ pattern, timeStep }) => {
  const getColor = () => {
    switch (pattern.layer) {
      case 'subcortical': return '#4a90e2';
      case 'cortical': return '#50c878';
      case 'thalamic': return '#ff6b6b';
      default: return '#666666';
    }
  };

  const scale = 0.1 + pattern.activation * 0.9;
  const opacity = 0.3 + pattern.activation * 0.7;

  return (
    <group position={pattern.position}>
      <Sphere args={[scale, 32, 32]}>
        <meshStandardMaterial
          color={getColor()}
          opacity={opacity}
          transparent
          emissive={getColor()}
          emissiveIntensity={pattern.activation}
        />
      </Sphere>
      <Text
        position={[0, scale + 0.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {pattern.id}
      </Text>
    </group>
  );
};

const ConnectionLine: React.FC<{
  connection: Connection;
  patterns: CognitivePattern[];
  timeStep: number;
}> = ({ connection, patterns, timeStep }) => {
  const source = patterns.find(p => p.id === connection.source);
  const target = patterns.find(p => p.id === connection.target);

  if (!source || !target) return null;

  const points = [
    new THREE.Vector3(...source.position),
    new THREE.Vector3(...target.position)
  ];

  const color = connection.type === 'excitatory' ? '#00ff00' : '#ff0000';
  const lineWidth = connection.strength * 5;
  const opacity = connection.active ? 0.8 : 0.2;

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      opacity={opacity}
      transparent
      dashed={connection.type === 'inhibitory'}
    />
  );
};

const AttentionSphere: React.FC<{
  focus: AttentionFocus;
  timeStep: number;
}> = ({ focus, timeStep }) => {
  const scale = focus.radius * (1 + Math.sin(timeStep * 0.05) * 0.1);

  return (
    <Sphere
      position={focus.center}
      args={[scale, 32, 32]}
    >
      <meshStandardMaterial
        color="#ffff00"
        opacity={0.2 * focus.strength}
        transparent
        wireframe
      />
    </Sphere>
  );
};

const LayerPlanes: React.FC = () => {
  return (
    <>
      <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#1a1a2e" opacity={0.1} transparent />
      </mesh>
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#16213e" opacity={0.1} transparent />
      </mesh>
      <mesh position={[0, 2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#0f3460" opacity={0.1} transparent />
      </mesh>
    </>
  );
};

// Activation Heatmap Component
const ActivationHeatmap: React.FC<{ patterns: CognitivePattern[] }> = ({ patterns }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || patterns.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = 500;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    svg.selectAll('*').remove();

    // Group patterns by layer and time
    const layerData = ['subcortical', 'cortical', 'thalamic'].map(layer => ({
      layer,
      activations: patterns
        .filter(p => p.layer === layer)
        .map(p => p.activation)
    }));

    // Create heatmap
    const x = d3.scaleBand()
      .range([margin.left, width - margin.right])
      .domain(layerData.map(d => d.layer))
      .padding(0.1);

    const y = d3.scaleLinear()
      .range([height - margin.bottom, margin.top])
      .domain([0, 1]);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Draw heatmap cells
    layerData.forEach((layer, layerIdx) => {
      layer.activations.forEach((activation, idx) => {
        svg.append('rect')
          .attr('x', x(layer.layer)!)
          .attr('y', y(1) + (idx * (height - margin.top - margin.bottom) / layer.activations.length))
          .attr('width', x.bandwidth())
          .attr('height', (height - margin.top - margin.bottom) / layer.activations.length)
          .attr('fill', colorScale(activation))
          .attr('opacity', 0.8);
      });
    });

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));

    // Add labels
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0)
      .attr('x', -(height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Activation Level');

  }, [patterns]);

  return (
    <Card title="Activation Heatmap">
      <svg ref={svgRef} width={500} height={400} />
    </Card>
  );
};

// Inhibition Pattern Map
const InhibitionPatternMap: React.FC<{
  patterns: CognitivePattern[];
  connections: Connection[];
}> = ({ patterns, connections }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 500;
    const height = 400;

    svg.selectAll('*').remove();

    // Filter inhibitory connections
    const inhibitoryConnections = connections.filter(c => c.type === 'inhibitory');

    // Create force simulation
    const nodes = patterns.map(p => ({
      id: p.id,
      layer: p.layer,
      activation: p.activation,
      x: width / 2 + (Math.random() - 0.5) * 100,
      y: height / 2 + (Math.random() - 0.5) * 100
    }));

    const links = inhibitoryConnections.map(c => ({
      source: c.source,
      target: c.target,
      strength: c.strength
    }));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(20));

    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#ff0000')
      .attr('stroke-opacity', d => d.strength)
      .attr('stroke-width', d => d.strength * 3);

    // Draw nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', d => 5 + d.activation * 15)
      .attr('fill', d => {
        switch (d.layer) {
          case 'subcortical': return '#4a90e2';
          case 'cortical': return '#50c878';
          case 'thalamic': return '#ff6b6b';
          default: return '#666666';
        }
      })
      .attr('opacity', 0.8);

    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(nodes)
      .enter().append('text')
      .text(d => d.id)
      .attr('font-size', '10px')
      .attr('text-anchor', 'middle')
      .attr('dy', -10);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

  }, [patterns, connections]);

  return (
    <Card title="Inhibition Network">
      <svg ref={svgRef} width={500} height={400} />
    </Card>
  );
};

// Emergent Pattern Analysis
const EmergentPatternAnalysis: React.FC<{
  patterns: CognitivePattern[];
  connections: Connection[];
}> = ({ patterns, connections }) => {
  const [emergentPatterns, setEmergentPatterns] = useState<any[]>([]);
  const { client, connected } = useMCPConnection();

  useEffect(() => {
    if (!connected || !client) return;

    const analyzePatterns = async () => {
      try {
        const response = await client.request('cognitive/analyzeEmergentPatterns', {
          patterns: patterns.map(p => ({
            id: p.id,
            layer: p.layer,
            activation: p.activation
          })),
          connections: connections.map(c => ({
            source: c.source,
            target: c.target,
            type: c.type,
            strength: c.strength
          }))
        });

        setEmergentPatterns(response.emergentPatterns || []);
      } catch (error) {
        console.error('Failed to analyze patterns:', error);
      }
    };

    if (patterns.length > 0) {
      analyzePatterns();
    }
  }, [patterns, connections, connected, client]);

  return (
    <div>
      <Row gutter={16}>
        {emergentPatterns.map((pattern, idx) => (
          <Col span={8} key={idx}>
            <Card size="small" title={pattern.name}>
              <p>Type: {pattern.type}</p>
              <p>Stability: {(pattern.stability * 100).toFixed(1)}%</p>
              <p>Frequency: {pattern.frequency}</p>
              <p>Components: {pattern.components.join(', ')}</p>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};
```

### Cognitive Pattern Analysis Backend
```rust
// src/cognitive/pattern_analyzer.rs
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use crate::cognitive::{CognitiveLayer, LayerType};
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePattern {
    pub id: String,
    pub layer: LayerType,
    pub position: [f32; 3],
    pub activation: f32,
    pub connections: Vec<Connection>,
    pub pattern_type: PatternType,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub source: String,
    pub target: String,
    pub strength: f32,
    pub connection_type: ConnectionType,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Excitatory,
    Inhibitory,
    Modulatory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFocus {
    pub center: [f32; 3],
    pub radius: f32,
    pub strength: f32,
    pub target_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentPattern {
    pub name: String,
    pub pattern_type: String,
    pub stability: f32,
    pub frequency: u32,
    pub components: Vec<String>,
    pub signature: Vec<f32>,
}

pub struct PatternAnalyzer {
    cognitive_layers: Vec<CognitiveLayer>,
    pattern_history: Vec<Vec<CognitivePattern>>,
    emergent_patterns: HashMap<String, EmergentPattern>,
    attention_controller: AttentionController,
}

impl PatternAnalyzer {
    pub fn new(cognitive_layers: Vec<CognitiveLayer>) -> Self {
        Self {
            cognitive_layers,
            pattern_history: Vec::new(),
            emergent_patterns: HashMap::new(),
            attention_controller: AttentionController::new(),
        }
    }

    pub fn capture_current_state(&mut self) -> (Vec<CognitivePattern>, Vec<Connection>, Option<AttentionFocus>) {
        let mut patterns = Vec::new();
        let mut connections = Vec::new();

        // Capture patterns from each layer
        for (layer_idx, layer) in self.cognitive_layers.iter().enumerate() {
            let layer_patterns = self.extract_layer_patterns(layer, layer_idx);
            let layer_connections = self.extract_layer_connections(layer, &layer_patterns);
            
            patterns.extend(layer_patterns);
            connections.extend(layer_connections);
        }

        // Get attention focus
        let attention_focus = self.attention_controller.get_current_focus(&patterns);

        // Store in history
        self.pattern_history.push(patterns.clone());
        if self.pattern_history.len() > 1000 {
            self.pattern_history.remove(0);
        }

        (patterns, connections, attention_focus)
    }

    fn extract_layer_patterns(&self, layer: &CognitiveLayer, layer_idx: usize) -> Vec<CognitivePattern> {
        let neurons = layer.get_active_neurons();
        let layer_type = layer.layer_type();
        
        neurons.iter().enumerate().map(|(idx, neuron)| {
            let position = self.calculate_3d_position(layer_type, idx, neurons.len());
            
            CognitivePattern {
                id: format!("{:?}_{}", layer_type, idx),
                layer: layer_type,
                position,
                activation: neuron.activation,
                connections: Vec::new(), // Will be filled later
                pattern_type: if neuron.is_inhibitory {
                    PatternType::Inhibitory
                } else {
                    PatternType::Excitatory
                },
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            }
        }).collect()
    }

    fn calculate_3d_position(&self, layer_type: LayerType, index: usize, total: usize) -> [f32; 3] {
        let y = match layer_type {
            LayerType::Subcortical => -2.0,
            LayerType::Cortical => 0.0,
            LayerType::Thalamic => 2.0,
        };

        let angle = (index as f32 / total as f32) * 2.0 * std::f32::consts::PI;
        let radius = 3.0;
        
        [
            radius * angle.cos(),
            y,
            radius * angle.sin(),
        ]
    }

    fn extract_layer_connections(
        &self,
        layer: &CognitiveLayer,
        patterns: &[CognitivePattern]
    ) -> Vec<Connection> {
        let mut connections = Vec::new();
        
        for (i, source_pattern) in patterns.iter().enumerate() {
            let neuron_connections = layer.get_neuron_connections(i);
            
            for (j, &weight) in neuron_connections.iter().enumerate() {
                if weight.abs() > 0.01 && i != j {
                    if let Some(target_pattern) = patterns.get(j) {
                        connections.push(Connection {
                            source: source_pattern.id.clone(),
                            target: target_pattern.id.clone(),
                            strength: weight.abs(),
                            connection_type: if weight > 0.0 {
                                ConnectionType::Excitatory
                            } else {
                                ConnectionType::Inhibitory
                            },
                            active: source_pattern.activation > 0.5 && target_pattern.activation > 0.1,
                        });
                    }
                }
            }
        }
        
        connections
    }

    pub fn analyze_emergent_patterns(&mut self) -> Vec<EmergentPattern> {
        if self.pattern_history.len() < 10 {
            return Vec::new();
        }

        // Analyze recent history for recurring patterns
        let recent_history = &self.pattern_history[self.pattern_history.len() - 100..];
        
        // Extract activation sequences
        let sequences = self.extract_activation_sequences(recent_history);
        
        // Identify recurring motifs
        let motifs = self.find_recurring_motifs(&sequences);
        
        // Analyze pattern stability
        for motif in motifs {
            let stability = self.calculate_pattern_stability(&motif, recent_history);
            let frequency = self.count_pattern_occurrences(&motif, recent_history);
            
            let emergent = EmergentPattern {
                name: self.generate_pattern_name(&motif),
                pattern_type: self.classify_pattern_type(&motif),
                stability,
                frequency,
                components: motif.components.clone(),
                signature: motif.signature.clone(),
            };
            
            self.emergent_patterns.insert(emergent.name.clone(), emergent);
        }
        
        self.emergent_patterns.values().cloned().collect()
    }

    fn extract_activation_sequences(&self, history: &[Vec<CognitivePattern>]) -> Vec<Vec<f32>> {
        history.iter().map(|patterns| {
            patterns.iter().map(|p| p.activation).collect()
        }).collect()
    }

    fn find_recurring_motifs(&self, sequences: &[Vec<f32>]) -> Vec<Motif> {
        let mut motifs = Vec::new();
        let window_size = 5;
        let threshold = 0.8;
        
        // Use sliding window to find similar subsequences
        for i in 0..sequences.len() - window_size {
            let window = &sequences[i..i + window_size];
            let pattern_signature = self.compute_signature(window);
            
            let mut occurrences = 1;
            for j in i + 1..sequences.len() - window_size {
                let other_window = &sequences[j..j + window_size];
                let other_signature = self.compute_signature(other_window);
                
                let similarity = self.cosine_similarity(&pattern_signature, &other_signature);
                if similarity > threshold {
                    occurrences += 1;
                }
            }
            
            if occurrences > 3 {
                motifs.push(Motif {
                    signature: pattern_signature,
                    components: self.identify_active_components(window),
                    occurrences,
                });
            }
        }
        
        // Remove duplicate motifs
        self.deduplicate_motifs(motifs)
    }

    fn compute_signature(&self, window: &[Vec<f32>]) -> Vec<f32> {
        // Compute statistical features of the activation pattern
        let mut signature = Vec::new();
        
        for timestep in window {
            signature.push(timestep.iter().sum::<f32>() / timestep.len() as f32); // Mean
            signature.push(timestep.iter().fold(0.0, |a, b| a.max(*b))); // Max
            signature.push(self.calculate_variance(timestep)); // Variance
        }
        
        signature
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a * norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    fn calculate_variance(&self, values: &[f32]) -> f32 {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32
    }

    fn identify_active_components(&self, window: &[Vec<f32>]) -> Vec<String> {
        let mut active_components = HashSet::new();
        
        for (timestep_idx, timestep) in window.iter().enumerate() {
            for (neuron_idx, &activation) in timestep.iter().enumerate() {
                if activation > 0.5 {
                    active_components.insert(format!("N{}", neuron_idx));
                }
            }
        }
        
        active_components.into_iter().collect()
    }

    fn deduplicate_motifs(&self, motifs: Vec<Motif>) -> Vec<Motif> {
        let mut unique_motifs = Vec::new();
        
        for motif in motifs {
            let is_duplicate = unique_motifs.iter().any(|existing: &Motif| {
                self.cosine_similarity(&motif.signature, &existing.signature) > 0.95
            });
            
            if !is_duplicate {
                unique_motifs.push(motif);
            }
        }
        
        unique_motifs
    }

    fn calculate_pattern_stability(&self, motif: &Motif, history: &[Vec<CognitivePattern>]) -> f32 {
        // Calculate how consistent the pattern is when it appears
        let mut consistency_scores = Vec::new();
        
        for i in 0..history.len() - 5 {
            let window = &history[i..i + 5];
            let window_sequences: Vec<Vec<f32>> = window.iter()
                .map(|patterns| patterns.iter().map(|p| p.activation).collect())
                .collect();
            
            let signature = self.compute_signature(&window_sequences);
            let similarity = self.cosine_similarity(&motif.signature, &signature);
            
            if similarity > 0.7 {
                consistency_scores.push(similarity);
            }
        }
        
        if consistency_scores.is_empty() {
            0.0
        } else {
            consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32
        }
    }

    fn count_pattern_occurrences(&self, motif: &Motif, history: &[Vec<CognitivePattern>]) -> u32 {
        let mut count = 0;
        
        for i in 0..history.len() - 5 {
            let window = &history[i..i + 5];
            let window_sequences: Vec<Vec<f32>> = window.iter()
                .map(|patterns| patterns.iter().map(|p| p.activation).collect())
                .collect();
            
            let signature = self.compute_signature(&window_sequences);
            let similarity = self.cosine_similarity(&motif.signature, &signature);
            
            if similarity > 0.8 {
                count += 1;
            }
        }
        
        count
    }

    fn generate_pattern_name(&self, motif: &Motif) -> String {
        let pattern_hash = self.hash_signature(&motif.signature);
        format!("EP_{:08x}", pattern_hash)
    }

    fn hash_signature(&self, signature: &[f32]) -> u32 {
        let mut hash: u32 = 0;
        for (i, &val) in signature.iter().enumerate() {
            hash = hash.wrapping_add((val * 1000.0) as u32);
            hash = hash.wrapping_mul(31);
        }
        hash
    }

    fn classify_pattern_type(&self, motif: &Motif) -> String {
        // Analyze the signature to determine pattern type
        let mean_activation = motif.signature.iter().sum::<f32>() / motif.signature.len() as f32;
        let variance = self.calculate_variance(&motif.signature);
        
        if variance < 0.1 {
            "Stable".to_string()
        } else if mean_activation > 0.7 {
            "Burst".to_string()
        } else if motif.signature.windows(2).any(|w| (w[1] - w[0]).abs() > 0.5) {
            "Oscillatory".to_string()
        } else {
            "Complex".to_string()
        }
    }
}

struct Motif {
    signature: Vec<f32>,
    components: Vec<String>,
    occurrences: usize,
}

pub struct AttentionController {
    focus_history: Vec<AttentionFocus>,
    attention_weights: DMatrix<f32>,
}

impl AttentionController {
    pub fn new() -> Self {
        Self {
            focus_history: Vec::new(),
            attention_weights: DMatrix::zeros(100, 100),
        }
    }

    pub fn get_current_focus(&mut self, patterns: &[CognitivePattern]) -> Option<AttentionFocus> {
        if patterns.is_empty() {
            return None;
        }

        // Find the most active thalamic patterns (attention controllers)
        let thalamic_patterns: Vec<&CognitivePattern> = patterns.iter()
            .filter(|p| matches!(p.layer, LayerType::Thalamic))
            .collect();

        if thalamic_patterns.is_empty() {
            return None;
        }

        // Calculate attention center based on thalamic activity
        let mut center = [0.0, 0.0, 0.0];
        let mut total_activation = 0.0;
        let mut target_patterns = Vec::new();

        for pattern in &thalamic_patterns {
            if pattern.activation > 0.3 {
                for i in 0..3 {
                    center[i] += pattern.position[i] * pattern.activation;
                }
                total_activation += pattern.activation;
                target_patterns.push(pattern.id.clone());
            }
        }

        if total_activation > 0.0 {
            for i in 0..3 {
                center[i] /= total_activation;
            }

            let focus = AttentionFocus {
                center,
                radius: 2.0 + total_activation,
                strength: total_activation / thalamic_patterns.len() as f32,
                target_patterns,
            };

            self.focus_history.push(focus.clone());
            if self.focus_history.len() > 100 {
                self.focus_history.remove(0);
            }

            Some(focus)
        } else {
            None
        }
    }
}
```

## LLMKG-Specific Features

### 1. Brain-Inspired Visualization
- **Layer Hierarchy**: Visual separation of subcortical, cortical, and thalamic layers
- **Activation Spreading**: Show how activation propagates through the network
- **Inhibitory Competition**: Visualize lateral inhibition and winner-take-all dynamics

### 2. Attention Mechanisms
- **Thalamic Control**: Show how thalamic layer modulates other layers
- **Attention Spotlight**: Visualize focused attention on specific patterns
- **Context Switching**: Display attention shifts over time

### 3. Pattern Emergence
- **Motif Detection**: Identify recurring activation patterns
- **Stability Analysis**: Show how stable emergent patterns are
- **Pattern Evolution**: Track how patterns change over time

### 4. Cognitive Dynamics
- **Oscillatory Patterns**: Detect and visualize brain-like oscillations
- **Synchronization**: Show synchronized activity across layers
- **Phase Relationships**: Display phase coupling between regions

## Testing Procedures

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_extraction() {
        let analyzer = create_test_analyzer();
        let (patterns, connections, _) = analyzer.capture_current_state();
        
        assert!(!patterns.is_empty());
        assert!(!connections.is_empty());
        assert!(patterns.iter().any(|p| matches!(p.layer, LayerType::Subcortical)));
    }

    #[test]
    fn test_emergent_pattern_detection() {
        let mut analyzer = create_test_analyzer();
        
        // Simulate pattern history
        for _ in 0..100 {
            analyzer.capture_current_state();
        }
        
        let emergent = analyzer.analyze_emergent_patterns();
        assert!(!emergent.is_empty());
        assert!(emergent.iter().any(|p| p.stability > 0.5));
    }

    #[test]
    fn test_attention_focus() {
        let mut controller = AttentionController::new();
        let patterns = create_test_patterns();
        
        let focus = controller.get_current_focus(&patterns);
        assert!(focus.is_some());
        
        let f = focus.unwrap();
        assert!(f.strength > 0.0);
        assert!(!f.target_patterns.is_empty());
    }
}
```

### Integration Tests
```typescript
// tests/cognitive-pattern-visualization.test.tsx
describe('Cognitive Pattern Visualization', () => {
  it('should render 3D pattern view', async () => {
    const { container } = render(<CognitivePatternVisualizer />);
    
    await waitFor(() => {
      expect(container.querySelector('canvas')).toBeInTheDocument();
      expect(container.querySelector('.pattern-node')).toBeInTheDocument();
    });
  });

  it('should show layer filtering', async () => {
    const { container, getByText } = render(<CognitivePatternVisualizer />);
    
    // Select subcortical layer
    fireEvent.click(getByText('All Layers'));
    fireEvent.click(getByText('Subcortical'));
    
    await waitFor(() => {
      const patterns = container.querySelectorAll('.pattern-node');
      patterns.forEach(node => {
        expect(node).toHaveAttribute('data-layer', 'subcortical');
      });
    });
  });

  it('should animate pattern activation', async () => {
    const { container, getByText } = render(<CognitivePatternVisualizer />);
    
    // Start animation
    fireEvent.click(getByText('Play'));
    
    const initialState = container.querySelector('.pattern-node').getAttribute('data-activation');
    
    await waitFor(() => {
      const newState = container.querySelector('.pattern-node').getAttribute('data-activation');
      expect(newState).not.toBe(initialState);
    });
  });
});
```

## Performance Considerations

### 1. 3D Rendering Optimization
- Use instanced rendering for multiple similar objects
- Implement level-of-detail for distant patterns
- Batch geometry updates

### 2. Pattern Analysis
- Use GPU compute shaders for pattern matching
- Implement sliding window optimization
- Cache emergent pattern signatures

### 3. Real-time Updates
- Use WebSocket for streaming updates
- Implement delta compression for state changes
- Throttle visualization updates based on frame rate

## Deliverables Checklist

- [ ] 3D cognitive pattern visualization component
- [ ] Real-time pattern state streaming
- [ ] Activation heatmap visualization
- [ ] Inhibition network diagram
- [ ] Attention focus visualization
- [ ] Emergent pattern detection and analysis
- [ ] Pattern history playback controls
- [ ] Layer filtering and visibility controls
- [ ] Pattern similarity analysis
- [ ] Cognitive dynamics metrics
- [ ] Export functionality for pattern data
- [ ] Integration with performance monitoring