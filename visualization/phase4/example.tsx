/**
 * Example usage of LLMKG Phase 4 Data Flow Visualization
 * Demonstrates integration with React and real-time data
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { 
  DataFlowCanvas,
  useVisualizationEngine,
  createDemoNode,
  createDemoConnection,
  createDemoCognitivePattern,
  type DataFlowNode,
  type DataFlowConnection,
  type CognitivePattern
} from './src';

// Example 1: Simple Integration
export const SimpleExample: React.FC = () => {
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000' }}>
      <DataFlowCanvas
        websocketUrl="ws://localhost:8080/llmkg"
        showControls={true}
        showPerformanceMetrics={true}
        enableDemoMode={true}
        autoStart={true}
      />
    </div>
  );
};

// Example 2: Advanced Usage with Custom Controls
export const AdvancedExample: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedNodeType, setSelectedNodeType] = useState<DataFlowNode['type']>('processing');
  const [isConnected, setIsConnected] = useState(false);

  const { state, actions, visualizer } = useVisualizationEngine({
    canvasRef,
    width: 1200,
    height: 800,
    websocketUrl: isConnected ? 'ws://localhost:8080/llmkg' : undefined,
    autoStart: true,
    performanceMonitoring: true,
    config: {
      backgroundColor: 0x001122,
      cameraPosition: new THREE.Vector3(0, 8, 12),
      targetFPS: 60,
      maxNodes: 500,
      maxConnections: 1000
    }
  });

  // Add a node at random position
  const addRandomNode = () => {
    const node = createDemoNode(`node_${Date.now()}`, selectedNodeType);
    actions.addNode(node);
  };

  // Create a neural network pattern
  const createNeuralNetwork = () => {
    const nodeIds: string[] = [];
    
    // Create input layer (3 nodes)
    for (let i = 0; i < 3; i++) {
      const node: DataFlowNode = {
        id: `input_${i}`,
        position: new THREE.Vector3(-5, i * 2 - 2, 0),
        type: 'input',
        activation: Math.random(),
        connections: []
      };
      actions.addNode(node);
      nodeIds.push(node.id);
    }
    
    // Create hidden layer (4 nodes)
    for (let i = 0; i < 4; i++) {
      const node: DataFlowNode = {
        id: `hidden_${i}`,
        position: new THREE.Vector3(0, i * 1.5 - 2.25, 0),
        type: 'processing',
        activation: Math.random(),
        connections: []
      };
      actions.addNode(node);
      nodeIds.push(node.id);
    }
    
    // Create output layer (2 nodes)
    for (let i = 0; i < 2; i++) {
      const node: DataFlowNode = {
        id: `output_${i}`,
        position: new THREE.Vector3(5, i * 2 - 1, 0),
        type: 'output',
        activation: Math.random(),
        connections: []
      };
      actions.addNode(node);
      nodeIds.push(node.id);
    }
    
    // Connect input to hidden
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        const connection: DataFlowConnection = {
          id: `input_hidden_${i}_${j}`,
          source: `input_${i}`,
          target: `hidden_${j}`,
          strength: 0.3 + Math.random() * 0.7,
          dataType: 'neural_signal',
          isActive: Math.random() > 0.2
        };
        actions.addConnection(connection);
      }
    }
    
    // Connect hidden to output
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        const connection: DataFlowConnection = {
          id: `hidden_output_${i}_${j}`,
          source: `hidden_${i}`,
          target: `output_${j}`,
          strength: 0.3 + Math.random() * 0.7,
          dataType: 'neural_signal',
          isActive: Math.random() > 0.2
        };
        actions.addConnection(connection);
      }
    }
    
    // Add cognitive attention pattern
    const attentionPattern: CognitivePattern = {
      id: 'attention_network',
      center: new THREE.Vector3(0, 0, 0),
      complexity: 0.8,
      strength: 0.9,
      type: 'attention',
      nodes: nodeIds.slice(3, 7) // Focus on hidden layer
    };
    actions.addCognitivePattern(attentionPattern);
  };

  // Simulate data flow animation
  const animateDataFlow = () => {
    const updateNodes = () => {
      // Simulate neural activation propagation
      // This would normally come from WebSocket data
      
      // Update input nodes with new activation
      for (let i = 0; i < 3; i++) {
        actions.updateNode(`input_${i}`, {
          activation: Math.sin(Date.now() * 0.001 + i) * 0.5 + 0.5
        });
      }
      
      // Update hidden nodes based on input (simplified)
      setTimeout(() => {
        for (let i = 0; i < 4; i++) {
          actions.updateNode(`hidden_${i}`, {
            activation: Math.sin(Date.now() * 0.0015 + i) * 0.5 + 0.5
          });
        }
      }, 100);
      
      // Update output nodes (delayed)
      setTimeout(() => {
        for (let i = 0; i < 2; i++) {
          actions.updateNode(`output_${i}`, {
            activation: Math.sin(Date.now() * 0.002 + i) * 0.5 + 0.5
          });
        }
      }, 200);
    };
    
    // Start animation loop
    const interval = setInterval(updateNodes, 500);
    
    // Clean up after 10 seconds
    setTimeout(() => clearInterval(interval), 10000);
  };

  const handlePerformanceUpdate = (metrics: any) => {
    // Log performance metrics for monitoring
    if (metrics.fps < 45) {
      console.warn('Performance degradation detected:', metrics);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#000' }}>
      {/* Control Panel */}
      <div style={{
        width: '300px',
        background: '#1a1a1a',
        color: 'white',
        padding: '20px',
        overflowY: 'auto'
      }}>
        <h2>LLMKG Phase 4 Demo</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>Status</h3>
          <p>Initialized: {state.isInitialized ? 'Yes' : 'No'}</p>
          <p>Running: {state.isRunning ? 'Yes' : 'No'}</p>
          <p>Nodes: {state.nodeCount}</p>
          <p>Connections: {state.connectionCount}</p>
          <p>Patterns: {state.patternCount}</p>
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>Controls</h3>
          <button 
            onClick={state.isRunning ? actions.stop : actions.start}
            style={{ marginRight: '10px', marginBottom: '10px' }}
          >
            {state.isRunning ? 'Stop' : 'Start'}
          </button>
          
          <button 
            onClick={actions.generateDemoData}
            style={{ marginBottom: '10px' }}
          >
            Generate Demo Data
          </button>
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>Add Nodes</h3>
          <select 
            value={selectedNodeType}
            onChange={(e) => setSelectedNodeType(e.target.value as DataFlowNode['type'])}
            style={{ marginBottom: '10px', width: '100%' }}
          >
            <option value="input">Input</option>
            <option value="processing">Processing</option>
            <option value="output">Output</option>
            <option value="cognitive">Cognitive</option>
          </select>
          
          <button onClick={addRandomNode} style={{ width: '100%' }}>
            Add Random Node
          </button>
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>Presets</h3>
          <button 
            onClick={createNeuralNetwork}
            style={{ width: '100%', marginBottom: '10px' }}
          >
            Create Neural Network
          </button>
          
          <button 
            onClick={animateDataFlow}
            style={{ width: '100%' }}
          >
            Animate Data Flow
          </button>
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <h3>WebSocket</h3>
          <label>
            <input 
              type="checkbox"
              checked={isConnected}
              onChange={(e) => setIsConnected(e.target.checked)}
            />
            Connect to WebSocket
          </label>
        </div>
        
        {state.performanceMetrics && (
          <div>
            <h3>Performance</h3>
            <p>FPS: {state.performanceMetrics.fps}</p>
            <p>Draw Calls: {state.performanceMetrics.renderer?.calls}</p>
            <p>Active Particles: {state.performanceMetrics.particles?.activeParticles}</p>
          </div>
        )}
        
        {state.error && (
          <div style={{ color: 'red', marginTop: '20px' }}>
            Error: {state.error}
          </div>
        )}
      </div>
      
      {/* Visualization Canvas */}
      <div style={{ flex: 1 }}>
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />
      </div>
    </div>
  );
};

// Example 3: Integration with Dashboard
export const DashboardIntegration: React.FC = () => {
  return (
    <div style={{ 
      display: 'grid', 
      gridTemplate: '1fr 1fr / 1fr 1fr',
      height: '100vh',
      gap: '10px',
      padding: '10px',
      background: '#f0f0f0'
    }}>
      <div style={{ background: 'white', padding: '10px', borderRadius: '5px' }}>
        <h3>LLMKG Data Flow</h3>
        <DataFlowCanvas
          showControls={false}
          showPerformanceMetrics={false}
          enableDemoMode={true}
        />
      </div>
      
      <div style={{ background: 'white', padding: '10px', borderRadius: '5px' }}>
        <h3>Other Dashboard Panel</h3>
        <p>Phase 2 dashboard integration ready</p>
      </div>
      
      <div style={{ background: 'white', padding: '10px', borderRadius: '5px' }}>
        <h3>Phase 3 Component</h3>
        <p>Ready for specialized visualization components</p>
      </div>
      
      <div style={{ background: 'white', padding: '10px', borderRadius: '5px' }}>
        <h3>MCP Request Tracing</h3>
        <p>Integration point for request visualization</p>
      </div>
    </div>
  );
};

// Export all examples
export default {
  SimpleExample,
  AdvancedExample,
  DashboardIntegration
};