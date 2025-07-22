import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { SystemArchitectureDiagram } from '../../src/components/SystemArchitectureDiagram';
import { RealTimeMonitor } from '../../src/monitoring/RealTimeMonitor';
import { useRealTimeUpdates } from '../../src/hooks/useRealTimeUpdates';
import { 
  createMockArchitectureData,
  MockWebSocketServer,
  MockTelemetryStream,
  createMockMetricsData
} from '../utils/test-helpers';

// Mock WebSocket
class MockWebSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  readyState: number = WebSocket.CONNECTING;

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      this.onopen?.(new Event('open'));
    }, 100);
  }

  send(data: string) {
    // Simulate server response
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', { data }));
      }
    }, 50);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close'));
  }
}

(global as any).WebSocket = MockWebSocket;

describe('Real-Time Monitoring Integration Tests', () => {
  let store: any;
  let mockData: any;
  let mockWebSocketServer: MockWebSocketServer;
  let mockTelemetryStream: MockTelemetryStream;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        architecture: (state = {}, action: any) => state,
        realtime: (state = { isConnected: false, metrics: {} }, action: any) => {
          switch (action.type) {
            case 'realtime/updateMetrics':
              return { ...state, metrics: action.payload };
            case 'realtime/setConnection':
              return { ...state, isConnected: action.payload };
            default:
              return state;
          }
        }
      }
    });
    
    mockData = createMockArchitectureData();
    mockWebSocketServer = new MockWebSocketServer();
    mockTelemetryStream = new MockTelemetryStream();
  });

  afterEach(() => {
    mockWebSocketServer.close();
    mockTelemetryStream.stop();
  });

  describe('WebSocket Connection', () => {
    test('establishes WebSocket connection successfully', async () => {
      const onConnectionChange = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            websocketUrl="ws://localhost:8080"
            onConnectionChange={onConnectionChange}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(onConnectionChange).toHaveBeenCalledWith(true);
      }, { timeout: 5000 });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
    });

    test('handles WebSocket connection failures', async () => {
      const onConnectionError = jest.fn();

      // Mock failed connection
      (global as any).WebSocket = class {
        constructor() {
          setTimeout(() => {
            this.onerror?.(new Event('error'));
          }, 100);
        }
        onerror: ((event: Event) => void) | null = null;
      };

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            websocketUrl="ws://invalid-url"
            onConnectionError={onConnectionError}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(onConnectionError).toHaveBeenCalled();
      });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
    });

    test('reconnects automatically after connection loss', async () => {
      let mockSocket: any;

      (global as any).WebSocket = class {
        constructor() {
          mockSocket = this;
          this.readyState = WebSocket.CONNECTING;
          setTimeout(() => {
            this.readyState = WebSocket.OPEN;
            this.onopen?.(new Event('open'));
          }, 100);
        }
        onopen: ((event: Event) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        readyState: number;
        close() {
          this.readyState = WebSocket.CLOSED;
          this.onclose?.(new CloseEvent('close'));
        }
      };

      const onReconnect = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            websocketUrl="ws://localhost:8080"
            autoReconnect={true}
            onReconnect={onReconnect}
          />
        </Provider>
      );

      // Wait for initial connection
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
      });

      // Simulate connection loss
      act(() => {
        mockSocket.close();
      });

      // Should attempt to reconnect
      await waitFor(() => {
        expect(onReconnect).toHaveBeenCalled();
      }, { timeout: 5000 });
    });
  });

  describe('Real-Time Data Updates', () => {
    test('receives and processes telemetry data', async () => {
      const onMetricsUpdate = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            onMetricsUpdate={onMetricsUpdate}
          />
        </Provider>
      );

      const metricsData = createMockMetricsData();

      // Simulate receiving telemetry data
      act(() => {
        mockTelemetryStream.emit('metrics', metricsData);
      });

      await waitFor(() => {
        expect(onMetricsUpdate).toHaveBeenCalledWith(metricsData);
      });

      // Verify UI updates with new data
      expect(screen.getByTestId('metrics-display')).toBeInTheDocument();
      expect(screen.getByText(`CPU: ${metricsData.cpu.current}%`)).toBeInTheDocument();
    });

    test('updates node status in real-time', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
          />
        </Provider>
      );

      const nodeId = mockData.nodes[0].id;
      const initialNode = screen.getByTestId(`node-${nodeId}`);
      expect(initialNode).toHaveClass('status-healthy');

      // Simulate status change
      const statusUpdate = {
        nodeId,
        status: 'warning',
        metrics: {
          cpu: { current: 85, average: 75, peak: 90 },
          memory: { current: 78, average: 65, peak: 82 }
        }
      };

      act(() => {
        mockTelemetryStream.emit('nodeStatusUpdate', statusUpdate);
      });

      await waitFor(() => {
        const updatedNode = screen.getByTestId(`node-${nodeId}`);
        expect(updatedNode).toHaveClass('status-warning');
      });
    });

    test('animates connection data flow', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            showConnectionFlow={true}
            enableAnimations={true}
          />
        </Provider>
      );

      const connectionId = mockData.connections[0].id;

      // Simulate data flow update
      const flowUpdate = {
        connectionId,
        dataFlow: 0.8,
        throughput: 1200,
        latency: 15
      };

      act(() => {
        mockTelemetryStream.emit('connectionFlow', flowUpdate);
      });

      await waitFor(() => {
        const flowElement = screen.getByTestId(`flow-${connectionId}`);
        expect(flowElement).toHaveStyle({ opacity: '0.8' });
        expect(flowElement).toHaveClass('flow-active');
      });
    });

    test('handles high-frequency updates efficiently', async () => {
      const updateCallback = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            updateThrottleMs={100}
            onMetricsUpdate={updateCallback}
          />
        </Provider>
      );

      // Send rapid updates
      for (let i = 0; i < 50; i++) {
        act(() => {
          mockTelemetryStream.emit('metrics', createMockMetricsData());
        });
      }

      await waitFor(() => {
        // Updates should be throttled
        expect(updateCallback.mock.calls.length).toBeLessThan(50);
        expect(updateCallback.mock.calls.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Performance Metrics Monitoring', () => {
    test('displays real-time performance metrics', async () => {
      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            enableMetricsDisplay={true}
          />
        </Provider>
      );

      const performanceMetrics = {
        overall: {
          cpu: 45.2,
          memory: 67.8,
          throughput: 1250,
          latency: 23.5,
          errorRate: 0.1
        },
        byComponent: mockData.nodes.map((node: any) => ({
          id: node.id,
          metrics: createMockMetricsData()
        }))
      };

      act(() => {
        mockTelemetryStream.emit('performanceMetrics', performanceMetrics);
      });

      await waitFor(() => {
        expect(screen.getByTestId('performance-metrics')).toBeInTheDocument();
        expect(screen.getByText('CPU: 45.2%')).toBeInTheDocument();
        expect(screen.getByText('Memory: 67.8%')).toBeInTheDocument();
        expect(screen.getByText('Throughput: 1,250/s')).toBeInTheDocument();
        expect(screen.getByText('Latency: 23.5ms')).toBeInTheDocument();
      });
    });

    test('tracks performance trends over time', async () => {
      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            enableTrendAnalysis={true}
            trendWindowSize={100}
          />
        </Provider>
      );

      // Send metrics over time
      const timestamps = [];
      for (let i = 0; i < 10; i++) {
        const metrics = {
          ...createMockMetricsData(),
          timestamp: Date.now() + i * 1000
        };
        
        timestamps.push(metrics.timestamp);
        
        act(() => {
          mockTelemetryStream.emit('metrics', metrics);
        });
        
        await new Promise(resolve => setTimeout(resolve, 50));
      }

      await waitFor(() => {
        expect(screen.getByTestId('trend-chart')).toBeInTheDocument();
        expect(screen.getByTestId('performance-timeline')).toBeInTheDocument();
      });
    });

    test('calculates health scores accurately', async () => {
      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            enableHealthScoring={true}
          />
        </Provider>
      );

      const healthData = {
        nodes: mockData.nodes.map((node: any) => ({
          id: node.id,
          healthScore: Math.random() * 100,
          factors: {
            performance: Math.random(),
            reliability: Math.random(),
            resources: Math.random()
          }
        }))
      };

      act(() => {
        mockTelemetryStream.emit('healthUpdate', healthData);
      });

      await waitFor(() => {
        expect(screen.getByTestId('health-scores')).toBeInTheDocument();
        
        healthData.nodes.forEach(node => {
          expect(screen.getByTestId(`health-${node.id}`)).toBeInTheDocument();
        });
      });
    });
  });

  describe('Alert System', () => {
    test('triggers alerts for performance thresholds', async () => {
      const onAlert = jest.fn();

      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            alertThresholds={{
              cpu: 80,
              memory: 85,
              latency: 100,
              errorRate: 5
            }}
            onAlert={onAlert}
          />
        </Provider>
      );

      // Send metrics that exceed thresholds
      const criticalMetrics = {
        cpu: { current: 95, average: 85, peak: 98 },
        memory: { current: 92, average: 88, peak: 95 },
        latency: { current: 150, average: 120, peak: 180 },
        errorRate: { current: 8.5, average: 6.2, peak: 12.1 }
      };

      act(() => {
        mockTelemetryStream.emit('metrics', criticalMetrics);
      });

      await waitFor(() => {
        expect(onAlert).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'critical',
            message: expect.stringContaining('CPU usage'),
            threshold: 80,
            current: 95
          })
        );
      });

      expect(screen.getByTestId('alert-panel')).toBeInTheDocument();
      expect(screen.getByText(/Critical Alert/)).toBeInTheDocument();
    });

    test('manages alert lifecycle correctly', async () => {
      const onAlertDismiss = jest.fn();

      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            onAlertDismiss={onAlertDismiss}
            alertAutoExpireMs={5000}
          />
        </Provider>
      );

      // Trigger an alert
      act(() => {
        mockTelemetryStream.emit('metrics', {
          cpu: { current: 95, average: 85, peak: 98 }
        });
      });

      await waitFor(() => {
        expect(screen.getByTestId('alert-panel')).toBeInTheDocument();
      });

      // Wait for auto-expire
      await waitFor(() => {
        expect(screen.queryByTestId('alert-panel')).not.toBeInTheDocument();
      }, { timeout: 6000 });
    });

    test('prioritizes alerts by severity', async () => {
      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            maxVisibleAlerts={3}
          />
        </Provider>
      );

      // Trigger multiple alerts
      const alerts = [
        { severity: 'info', cpu: 60 },
        { severity: 'critical', cpu: 95 },
        { severity: 'warning', cpu: 75 },
        { severity: 'critical', memory: 90 },
        { severity: 'info', latency: 40 }
      ];

      alerts.forEach((alert, index) => {
        act(() => {
          mockTelemetryStream.emit('metrics', {
            cpu: { current: alert.cpu || 50 },
            memory: { current: alert.memory || 60 },
            latency: { current: alert.latency || 30 }
          });
        });
      });

      await waitFor(() => {
        const alertElements = screen.getAllByTestId(/alert-item/);
        expect(alertElements).toHaveLength(3); // Max visible alerts
        
        // Critical alerts should be visible
        expect(screen.getByText(/Critical.*CPU/)).toBeInTheDocument();
        expect(screen.getByText(/Critical.*Memory/)).toBeInTheDocument();
      });
    });
  });

  describe('Data Buffering and History', () => {
    test('maintains metrics history buffer', async () => {
      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            historyBufferSize={100}
            enableHistoryExport={true}
          />
        </Provider>
      );

      // Send multiple metrics updates
      for (let i = 0; i < 50; i++) {
        act(() => {
          mockTelemetryStream.emit('metrics', {
            ...createMockMetricsData(),
            timestamp: Date.now() + i * 1000
          });
        });
      }

      await waitFor(() => {
        expect(screen.getByTestId('history-export-btn')).toBeInTheDocument();
      });

      // Test history export
      const exportBtn = screen.getByTestId('history-export-btn');
      fireEvent.click(exportBtn);

      await waitFor(() => {
        expect(screen.getByTestId('export-dialog')).toBeInTheDocument();
      });
    });

    test('handles buffer overflow correctly', async () => {
      const bufferSize = 10;

      render(
        <Provider store={store}>
          <RealTimeMonitor
            architectureData={mockData}
            historyBufferSize={bufferSize}
          />
        </Provider>
      );

      // Send more data than buffer can hold
      for (let i = 0; i < bufferSize + 5; i++) {
        act(() => {
          mockTelemetryStream.emit('metrics', {
            id: i,
            timestamp: Date.now() + i * 1000,
            cpu: { current: i * 2 }
          });
        });
      }

      // Should maintain only the most recent entries
      await waitFor(() => {
        const historyItems = screen.getAllByTestId(/history-item/);
        expect(historyItems).toHaveLength(bufferSize);
      });
    });
  });

  describe('Cross-Phase Integration', () => {
    test('integrates with Phase 1 WebSocket infrastructure', async () => {
      const phase1WebSocket = new MockWebSocket('ws://localhost:8080');

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            phase1WebSocket={phase1WebSocket}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('phase1-integration')).toBeInTheDocument();
        expect(screen.getByText(/Phase 1 Connected/)).toBeInTheDocument();
      });
    });

    test('receives data from Phase 4 visualization engine', async () => {
      const phase4DataStream = {
        nodes: mockData.nodes,
        connections: mockData.connections,
        flows: [
          { connectionId: 'conn-1', intensity: 0.8, direction: 'forward' }
        ]
      };

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            realTimeEnabled={true}
            phase4Integration={true}
          />
        </Provider>
      );

      act(() => {
        mockTelemetryStream.emit('phase4Data', phase4DataStream);
      });

      await waitFor(() => {
        expect(screen.getByTestId('phase4-flows')).toBeInTheDocument();
        phase4DataStream.flows.forEach(flow => {
          expect(screen.getByTestId(`phase4-flow-${flow.connectionId}`)).toBeInTheDocument();
        });
      });
    });
  });
});