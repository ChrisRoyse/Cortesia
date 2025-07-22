/**
 * Integration example showing how to use all LLMKG Phase 4 controls together
 * This demonstrates the complete control system in action
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import VisualizationControls from './VisualizationControls';
import PerformanceMonitor from './PerformanceMonitor';
import DebugConsole from './DebugConsole';
import { filteringSystem, exportTools } from './index';

interface MockData {
  id: string;
  timestamp: Date;
  type: 'mcp_request' | 'cognitive_pattern' | 'memory_operation' | 'performance_metric';
  pattern_type?: string;
  operation_type?: string;
  duration: number;
  status: 'success' | 'error' | 'pending';
  data: any;
}

interface ControlsIntegrationProps {
  // Props from main application
  onVisualizationUpdate?: (data: any) => void;
  onSystemStatusChange?: (status: any) => void;
  initialData?: MockData[];
}

const ControlsIntegration: React.FC<ControlsIntegrationProps> = ({
  onVisualizationUpdate,
  onSystemStatusChange,
  initialData = []
}) => {
  // UI state
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(false);
  const [showDebugConsole, setShowDebugConsole] = useState(false);
  
  // Data state
  const [rawData, setRawData] = useState<MockData[]>(initialData);
  const [filteredData, setFilteredData] = useState<MockData[]>([]);
  const [systemData, setSystemData] = useState<any>({});
  
  // Refs for export
  const visualizationRef = useRef<HTMLDivElement>(null);
  
  // WebSocket simulation
  const websocketRef = useRef<WebSocket | null>(null);

  // Initialize mock data and WebSocket
  useEffect(() => {
    generateMockData();
    setupMockWebSocket();
    
    // Set up filtering system listener
    const filterId = 'main-integration';
    filteringSystem.addListener(filterId, (data) => {
      // Apply filters to raw data
      const filtered = filteringSystem.applyFilters(rawData);
      setFilteredData(filtered);
      onVisualizationUpdate?.(filtered);
    });

    return () => {
      filteringSystem.removeListener(filterId);
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [rawData, onVisualizationUpdate]);

  // Generate mock data for demonstration
  const generateMockData = useCallback(() => {
    const mockData: MockData[] = [];
    const now = Date.now();
    
    for (let i = 0; i < 1000; i++) {
      const timestamp = new Date(now - (Math.random() * 3600000)); // Last hour
      
      const types: MockData['type'][] = ['mcp_request', 'cognitive_pattern', 'memory_operation', 'performance_metric'];
      const type = types[Math.floor(Math.random() * types.length)];
      
      let data: MockData = {
        id: crypto.randomUUID(),
        timestamp,
        type,
        duration: Math.random() * 1000 + 10, // 10-1010ms
        status: Math.random() > 0.1 ? 'success' : (Math.random() > 0.5 ? 'error' : 'pending'),
        data: {}
      };

      // Add type-specific data
      switch (type) {
        case 'cognitive_pattern':
          data.pattern_type = ['ConvergentThinking', 'DivergentThinking', 'AnalyticalReasoning', 'CreativeSynthesis'][
            Math.floor(Math.random() * 4)
          ];
          data.data = {
            activation_strength: Math.random(),
            inhibition_level: Math.random() * 0.5,
            processing_time: data.duration
          };
          break;
          
        case 'memory_operation':
          data.operation_type = ['store', 'retrieve', 'update', 'delete'][Math.floor(Math.random() * 4)];
          data.data = {
            key: `memory_${Math.floor(Math.random() * 1000)}`,
            size_bytes: Math.floor(Math.random() * 10000),
            cache_hit: Math.random() > 0.3
          };
          break;
          
        case 'mcp_request':
          data.data = {
            method: ['GET', 'POST', 'PUT', 'DELETE'][Math.floor(Math.random() * 4)],
            endpoint: `/api/v1/resource/${Math.floor(Math.random() * 100)}`,
            response_size: Math.floor(Math.random() * 5000)
          };
          break;
          
        case 'performance_metric':
          data.data = {
            metric_name: ['fps', 'memory_usage', 'cpu_usage', 'network_latency'][Math.floor(Math.random() * 4)],
            value: Math.random() * 100,
            threshold_exceeded: Math.random() > 0.8
          };
          break;
      }
      
      mockData.push(data);
    }
    
    setRawData(mockData.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime()));
  }, []);

  // Mock WebSocket setup
  const setupMockWebSocket = useCallback(() => {
    // Simulate WebSocket connection
    const mockWs = {
      readyState: WebSocket.OPEN,
      close: () => {},
      send: (data: string) => {
        console.log('Mock WebSocket send:', data);
      }
    };
    
    websocketRef.current = mockWs as any;
    
    // Simulate incoming messages
    const interval = setInterval(() => {
      // Generate new data periodically
      if (Math.random() > 0.7) {
        const newEntry: MockData = {
          id: crypto.randomUUID(),
          timestamp: new Date(),
          type: ['mcp_request', 'cognitive_pattern'][Math.floor(Math.random() * 2)] as any,
          duration: Math.random() * 500 + 50,
          status: 'success',
          pattern_type: 'ConvergentThinking',
          data: { real_time: true }
        };
        
        setRawData(prev => [...prev.slice(-999), newEntry]);
      }
      
      // Update system data
      setSystemData({
        websocket: {
          connected: true,
          messageCount: Math.floor(Math.random() * 1000),
          lastMessage: new Date()
        },
        cognitive: {
          activePatterns: ['ConvergentThinking', 'AnalyticalReasoning'],
          processingQueue: Math.floor(Math.random() * 10)
        },
        memory: {
          cacheHitRate: 75 + Math.random() * 20,
          operations: {
            store: Math.floor(Math.random() * 100),
            retrieve: Math.floor(Math.random() * 500)
          }
        },
        visualization: {
          rawDataCount: rawData.length,
          filteredDataCount: filteredData.length,
          activeFilters: filteringSystem.getState().groups.filter(g => g.enabled).length
        }
      });
      
    }, 2000);
    
    return () => clearInterval(interval);
  }, [rawData.length, filteredData.length]);

  // Event handlers
  const handleSettingsChange = useCallback((settings: any) => {
    console.log('Visualization settings changed:', settings);
    onSystemStatusChange?.(settings);
    
    // Apply settings to actual visualization components
    // This would integrate with Three.js, WebGL, or other rendering systems
  }, [onSystemStatusChange]);

  const handleExportRequest = useCallback(async (type: 'screenshot' | 'video' | 'data') => {
    if (!visualizationRef.current) return;
    
    try {
      switch (type) {
        case 'screenshot':
          await exportTools.downloadScreenshot(visualizationRef.current, {
            format: 'png',
            quality: 0.95,
            annotations: true,
            watermark: true,
            timestamp: true
          });
          break;
          
        case 'video':
          // Start video recording
          await exportTools.startVideoRecording(visualizationRef.current, {
            format: 'webm',
            quality: 'high',
            duration: 30,
            fps: 30
          });
          
          // Show recording indicator
          alert('Video recording started (30 seconds)');
          break;
          
        case 'data':
          await exportTools.downloadData(filteredData, {
            format: 'json',
            includeMetadata: true,
            includeTimestamps: true,
            includeFilters: true
          });
          break;
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [filteredData]);

  const handleDebugToggle = useCallback((enabled: boolean) => {
    setShowDebugConsole(enabled);
  }, []);

  const handlePerformanceMonitor = useCallback((enabled: boolean) => {
    setShowPerformanceMonitor(enabled);
  }, []);

  const handleOptimizationChange = useCallback((settings: any) => {
    console.log('Optimization settings changed:', settings);
    // Apply optimization settings
  }, []);

  const handleQualityAdjustment = useCallback((quality: string) => {
    console.log('Quality adjusted to:', quality);
    // Adjust rendering quality
  }, []);

  const handleDebugCommand = useCallback(async (command: string, args: string[]) => {
    console.log('Debug command:', command, args);
    
    // Handle custom debug commands
    switch (command) {
      case 'data':
        return {
          raw: rawData.length,
          filtered: filteredData.length,
          filters: filteringSystem.getState().groups.length
        };
        
      case 'generate':
        const count = parseInt(args[0]) || 10;
        generateMockData();
        return `Generated ${count} new data points`;
        
      case 'export':
        const format = args[0] || 'json';
        await exportTools.downloadData(filteredData, { format: format as any });
        return `Data exported as ${format}`;
        
      default:
        throw new Error(`Unknown command: ${command}`);
    }
  }, [rawData.length, filteredData.length, generateMockData, filteredData]);

  // Apply initial filters
  useEffect(() => {
    const filtered = filteringSystem.applyFilters(rawData);
    setFilteredData(filtered);
  }, [rawData]);

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      {/* Main visualization area */}
      <div
        ref={visualizationRef}
        className="w-full h-96 bg-white rounded-lg shadow-lg border-2 border-dashed border-gray-300 flex items-center justify-center mb-4"
      >
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-4">🎛️</div>
          <h2 className="text-xl font-semibold mb-2">LLMKG Visualization Area</h2>
          <p className="mb-4">This would be your actual 3D visualization</p>
          <div className="text-sm space-y-1">
            <div>Raw Data: {rawData.length} items</div>
            <div>Filtered Data: {filteredData.length} items</div>
            <div>Active Filters: {filteringSystem.getState().groups.filter(g => g.enabled).length}</div>
          </div>
        </div>
      </div>

      {/* Sample data display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold mb-2">Recent MCP Requests</h3>
          <div className="space-y-2 text-sm">
            {filteredData
              .filter(d => d.type === 'mcp_request')
              .slice(-3)
              .map(item => (
                <div key={item.id} className="flex justify-between">
                  <span>{item.data.method} {item.data.endpoint}</span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    item.status === 'success' ? 'bg-green-100 text-green-800' :
                    item.status === 'error' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {item.status}
                  </span>
                </div>
              ))
            }
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold mb-2">Active Cognitive Patterns</h3>
          <div className="space-y-2 text-sm">
            {filteredData
              .filter(d => d.type === 'cognitive_pattern')
              .slice(-3)
              .map(item => (
                <div key={item.id} className="flex justify-between">
                  <span>{item.pattern_type}</span>
                  <span>{item.duration.toFixed(0)}ms</span>
                </div>
              ))
            }
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold mb-2">Memory Operations</h3>
          <div className="space-y-2 text-sm">
            {filteredData
              .filter(d => d.type === 'memory_operation')
              .slice(-3)
              .map(item => (
                <div key={item.id} className="flex justify-between">
                  <span>{item.operation_type} {item.data.key}</span>
                  <span>{item.data.cache_hit ? '💰' : '🔍'}</span>
                </div>
              ))
            }
          </div>
        </div>
      </div>

      {/* Control instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <h3 className="font-semibold text-blue-800 mb-2">🎮 Controls Demo</h3>
        <div className="text-sm text-blue-700 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <strong>Visualization Controls (Top-Left):</strong>
            <ul className="mt-1 ml-4 list-disc">
              <li>Layer visibility toggles</li>
              <li>Advanced filtering system</li>
              <li>Quality and camera settings</li>
              <li>Playback controls</li>
              <li>Theme and accessibility options</li>
            </ul>
          </div>
          <div>
            <strong>Debug Tools:</strong>
            <ul className="mt-1 ml-4 list-disc">
              <li>Performance Monitor (Right-Bottom)</li>
              <li>Debug Console (Left-Bottom)</li>
              <li>Data inspector and system status</li>
              <li>Real-time logging and metrics</li>
              <li>Export tools for analysis</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Main visualization controls */}
      <VisualizationControls
        onSettingsChange={handleSettingsChange}
        onExportRequest={handleExportRequest}
        onDebugToggle={handleDebugToggle}
        onPerformanceMonitor={handlePerformanceMonitor}
      />

      {/* Performance monitor */}
      <PerformanceMonitor
        isVisible={showPerformanceMonitor}
        onClose={() => setShowPerformanceMonitor(false)}
        onOptimizationChange={handleOptimizationChange}
        onQualityAdjustment={handleQualityAdjustment}
      />

      {/* Debug console */}
      <DebugConsole
        isVisible={showDebugConsole}
        onClose={() => setShowDebugConsole(false)}
        onCommandExecute={handleDebugCommand}
        systemData={systemData}
      />
    </div>
  );
};

export default ControlsIntegration;