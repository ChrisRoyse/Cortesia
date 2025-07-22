import React, { useState, useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ThemeProvider } from '../components/ThemeProvider/ThemeProvider';
import { WebSocketProvider } from '../providers/WebSocketProvider';
import { MCPProvider } from '../providers/MCPProvider';
import { store } from '../stores';
import { AppRouter } from '../routing/AppRouter';
import DashboardLayout from '../components/Layout/DashboardLayout';
import {
  mockKnowledgeGraphData,
  mockCognitivePatterns,
  mockMemoryMetrics,
  mockNeuralActivity,
} from '../utils/testUtils';

// Demo WebSocket server that sends mock data
class DemoWebSocketServer {
  private clients: Set<WebSocket> = new Set();
  private intervalId: NodeJS.Timeout | null = null;

  constructor() {
    // In a real implementation, this would be a WebSocket server
    // For demo purposes, we'll simulate it client-side
    this.startDataGeneration();
  }

  connect(ws: WebSocket) {
    this.clients.add(ws);
    
    // Send initial data
    this.sendToClient(ws, {
      type: 'connection_established',
      timestamp: Date.now(),
    });
    
    // Send initial state
    this.sendAllData(ws);
  }

  disconnect(ws: WebSocket) {
    this.clients.delete(ws);
  }

  private startDataGeneration() {
    this.intervalId = setInterval(() => {
      this.broadcastUpdates();
    }, 2000); // Update every 2 seconds
  }

  private broadcastUpdates() {
    const updates = this.generateRandomUpdate();
    this.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        this.sendToClient(client, updates);
      }
    });
  }

  private generateRandomUpdate() {
    const updateTypes = [
      'knowledge_graph_update',
      'cognitive_patterns_update',
      'memory_metrics_update',
      'neural_activity_update',
    ];
    
    const type = updateTypes[Math.floor(Math.random() * updateTypes.length)];
    
    switch (type) {
      case 'knowledge_graph_update':
        return {
          type,
          data: this.generateKnowledgeGraphUpdate(),
          timestamp: Date.now(),
        };
      case 'cognitive_patterns_update':
        return {
          type,
          data: this.generateCognitiveUpdate(),
          timestamp: Date.now(),
        };
      case 'memory_metrics_update':
        return {
          type,
          data: this.generateMemoryUpdate(),
          timestamp: Date.now(),
        };
      case 'neural_activity_update':
        return {
          type,
          data: this.generateNeuralUpdate(),
          timestamp: Date.now(),
        };
      default:
        return null;
    }
  }

  private generateKnowledgeGraphUpdate() {
    const baseData = mockKnowledgeGraphData();
    // Add some variation
    baseData.nodes.forEach(node => {
      node.properties.confidence = Math.random() * 0.3 + 0.7; // 0.7-1.0
    });
    baseData.edges.forEach(edge => {
      edge.properties.weight = Math.random() * 0.5 + 0.5; // 0.5-1.0
    });
    return baseData;
  }

  private generateCognitiveUpdate() {
    const baseData = mockCognitivePatterns();
    baseData.patterns.forEach(pattern => {
      pattern.activation = Math.random() * 0.5 + 0.5; // 0.5-1.0
      pattern.connections = Math.floor(Math.random() * 50) + 20;
    });
    return baseData;
  }

  private generateMemoryUpdate() {
    const baseData = mockMemoryMetrics();
    baseData.workingMemory.used = Math.floor(Math.random() * 7) + 1;
    baseData.workingMemory.efficiency = Math.random() * 0.3 + 0.7;
    baseData.longTermMemory.recentAccess = Math.floor(Math.random() * 100);
    baseData.longTermMemory.consolidationRate = Math.random() * 0.2 + 0.8;
    return baseData;
  }

  private generateNeuralUpdate() {
    return mockNeuralActivity();
  }

  private sendToClient(ws: WebSocket, data: any) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }

  private sendAllData(ws: WebSocket) {
    this.sendToClient(ws, {
      type: 'knowledge_graph_update',
      data: this.generateKnowledgeGraphUpdate(),
      timestamp: Date.now(),
    });
    
    setTimeout(() => {
      this.sendToClient(ws, {
        type: 'cognitive_patterns_update',
        data: this.generateCognitiveUpdate(),
        timestamp: Date.now(),
      });
    }, 100);
    
    setTimeout(() => {
      this.sendToClient(ws, {
        type: 'memory_metrics_update',
        data: this.generateMemoryUpdate(),
        timestamp: Date.now(),
      });
    }, 200);
    
    setTimeout(() => {
      this.sendToClient(ws, {
        type: 'neural_activity_update',
        data: this.generateNeuralUpdate(),
        timestamp: Date.now(),
      });
    }, 300);
  }

  cleanup() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
}

// Mock WebSocket that connects to our demo server
class DemoWebSocket extends WebSocket {
  private static server = new DemoWebSocketServer();
  
  constructor(url: string) {
    super(url);
    
    // Simulate connection
    setTimeout(() => {
      (this as any).readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
      DemoWebSocket.server.connect(this);
    }, 100);
    
    // Handle messages from server
    (this as any).send = (data: string) => {
      console.log('Client sending:', data);
    };
  }
  
  close() {
    DemoWebSocket.server.disconnect(this);
    (this as any).readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

// Replace global WebSocket with our demo version
(window as any).WebSocket = DemoWebSocket;

interface DemoControlsProps {
  onToggleData: () => void;
  onSimulateError: () => void;
  onChangeUpdateSpeed: (speed: number) => void;
  dataEnabled: boolean;
  updateSpeed: number;
}

const DemoControls: React.FC<DemoControlsProps> = ({
  onToggleData,
  onSimulateError,
  onChangeUpdateSpeed,
  dataEnabled,
  updateSpeed,
}) => {
  return (
    <div className="fixed bottom-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 z-50">
      <h3 className="text-lg font-semibold mb-4">Demo Controls</h3>
      
      <div className="space-y-3">
        <div>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={dataEnabled}
              onChange={onToggleData}
              className="rounded"
            />
            <span>Enable Data Updates</span>
          </label>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">
            Update Speed: {updateSpeed}s
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={updateSpeed}
            onChange={(e) => onChangeUpdateSpeed(Number(e.target.value))}
            className="w-full"
          />
        </div>
        
        <button
          onClick={onSimulateError}
          className="w-full px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          Simulate Error
        </button>
      </div>
      
      <div className="mt-4 pt-4 border-t">
        <h4 className="text-sm font-medium mb-2">Demo Features:</h4>
        <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
          <li>• Real-time data updates</li>
          <li>• Responsive design</li>
          <li>• Theme switching</li>
          <li>• Drag & drop layouts</li>
          <li>• WebSocket integration</li>
        </ul>
      </div>
    </div>
  );
};

const FullDashboardDemo: React.FC = () => {
  const [dataEnabled, setDataEnabled] = useState(true);
  const [updateSpeed, setUpdateSpeed] = useState(2);
  const [showDemo, setShowDemo] = useState(true);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      (window as any).WebSocket = WebSocket; // Restore original
    };
  }, []);

  const handleToggleData = () => {
    setDataEnabled(!dataEnabled);
    // In real implementation, this would pause/resume the WebSocket connection
  };

  const handleSimulateError = () => {
    // Simulate a WebSocket error
    const event = new Event('error');
    window.dispatchEvent(event);
  };

  const handleChangeUpdateSpeed = (speed: number) => {
    setUpdateSpeed(speed);
    // In real implementation, this would adjust the server update interval
  };

  return (
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider>
          <WebSocketProvider url="ws://localhost:8080">
            <MCPProvider>
              <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
                <DashboardLayout>
                  <AppRouter />
                </DashboardLayout>
                
                {showDemo && (
                  <DemoControls
                    onToggleData={handleToggleData}
                    onSimulateError={handleSimulateError}
                    onChangeUpdateSpeed={handleChangeUpdateSpeed}
                    dataEnabled={dataEnabled}
                    updateSpeed={updateSpeed}
                  />
                )}
                
                <button
                  onClick={() => setShowDemo(!showDemo)}
                  className="fixed bottom-4 left-4 px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg shadow-lg z-50"
                >
                  {showDemo ? 'Hide' : 'Show'} Demo Controls
                </button>
              </div>
            </MCPProvider>
          </WebSocketProvider>
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  );
};

export default FullDashboardDemo;

// Export for standalone usage
export const runDemo = () => {
  const root = document.getElementById('root');
  if (root) {
    import('react-dom/client').then(({ createRoot }) => {
      createRoot(root).render(<FullDashboardDemo />);
    });
  }
};