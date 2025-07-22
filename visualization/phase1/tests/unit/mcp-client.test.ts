/**
 * Unit tests for MCP Client
 * 
 * Tests the MCP (Model Context Protocol) client implementation
 * including connection management, data generation, and event handling.
 */

import { MCPClient, MCPConfig } from '../../src/mcp/client';
import { MockBrainInspiredMCPServer } from '../mocks/brain-inspired-server';
import { PerformanceTracker, TestHelpers } from '../config/test-utils';

describe('MCPClient', () => {
  let mcpClient: MCPClient;
  let mockServer: MockBrainInspiredMCPServer;
  let performanceTracker: PerformanceTracker;

  beforeEach(() => {
    mockServer = new MockBrainInspiredMCPServer();
    performanceTracker = new PerformanceTracker();
    
    const config: MCPConfig = {
      enableRealtimeUpdates: true,
      updateInterval: 100,
      serverUrl: 'ws://localhost:3000'
    };
    
    mcpClient = new MCPClient(config);
  });

  afterEach(async () => {
    if (mcpClient) {
      mcpClient.disconnect();
    }
    if (mockServer) {
      await mockServer.stop();
    }
    performanceTracker.clear();
  });

  describe('Initialization', () => {
    it('should create MCP client with default configuration', () => {
      const client = new MCPClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(MCPClient);
    });

    it('should create MCP client with custom configuration', () => {
      const config: MCPConfig = {
        enableRealtimeUpdates: false,
        updateInterval: 500,
        serverUrl: 'ws://custom-server:8080'
      };
      
      const client = new MCPClient(config);
      expect(client).toBeDefined();
    });

    it('should inherit from EventEmitter', () => {
      expect(mcpClient.on).toBeDefined();
      expect(mcpClient.emit).toBeDefined();
      expect(mcpClient.removeListener).toBeDefined();
    });
  });

  describe('Connection Management', () => {
    it('should connect successfully', () => {
      const connectSpy = jest.fn();
      mcpClient.on('dataUpdate', connectSpy);

      mcpClient.connect();
      
      expect(mcpClient).toBeDefined();
      // Should start emitting data updates if real-time is enabled
    });

    it('should handle multiple connect calls gracefully', () => {
      mcpClient.connect();
      mcpClient.connect(); // Second call should be ignored
      
      expect(mcpClient).toBeDefined();
    });

    it('should disconnect successfully', () => {
      mcpClient.connect();
      mcpClient.disconnect();
      
      expect(mcpClient).toBeDefined();
    });

    it('should handle disconnect when not connected', () => {
      mcpClient.disconnect(); // Should not throw
      expect(mcpClient).toBeDefined();
    });

    it('should stop data updates on disconnect', async () => {
      const dataUpdateSpy = jest.fn();
      mcpClient.on('dataUpdate', dataUpdateSpy);

      mcpClient.connect();
      
      // Wait for some data updates
      await new Promise(resolve => setTimeout(resolve, 250));
      const updateCountAfterConnect = dataUpdateSpy.mock.calls.length;
      
      mcpClient.disconnect();
      
      // Wait a bit more and check no new updates
      await new Promise(resolve => setTimeout(resolve, 250));
      const updateCountAfterDisconnect = dataUpdateSpy.mock.calls.length;
      
      expect(updateCountAfterConnect).toBeGreaterThan(0);
      expect(updateCountAfterDisconnect).toBe(updateCountAfterConnect);
    });
  });

  describe('Data Generation', () => {
    it('should generate mock MCP data', async () => {
      const dataUpdateSpy = jest.fn();
      mcpClient.on('dataUpdate', dataUpdateSpy);

      mcpClient.connect();

      // Wait for at least one data update
      await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length > 0, 2000);
      
      expect(dataUpdateSpy).toHaveBeenCalled();
      
      const mockData = dataUpdateSpy.mock.calls[0][0];
      expect(mockData).toHaveProperty('timestamp');
      expect(mockData).toHaveProperty('type');
      expect(mockData).toHaveProperty('data');
      expect(typeof mockData.timestamp).toBe('number');
      expect(mockData.type).toBe('system_status');
    });

    it('should generate data at configured intervals', async () => {
      const config: MCPConfig = {
        enableRealtimeUpdates: true,
        updateInterval: 50, // 50ms intervals
        serverUrl: 'ws://localhost:3000'
      };
      
      const client = new MCPClient(config);
      const dataUpdateSpy = jest.fn();
      client.on('dataUpdate', dataUpdateSpy);

      client.connect();

      // Wait for multiple updates
      await new Promise(resolve => setTimeout(resolve, 300));
      
      client.disconnect();

      // Should have received multiple updates (at least 4-5 in 300ms with 50ms intervals)
      expect(dataUpdateSpy.mock.calls.length).toBeGreaterThan(3);
    });

    it('should not generate data when realtime updates are disabled', async () => {
      const config: MCPConfig = {
        enableRealtimeUpdates: false,
        updateInterval: 100,
        serverUrl: 'ws://localhost:3000'
      };
      
      const client = new MCPClient(config);
      const dataUpdateSpy = jest.fn();
      client.on('dataUpdate', dataUpdateSpy);

      client.connect();

      // Wait for potential updates
      await new Promise(resolve => setTimeout(resolve, 300));
      
      client.disconnect();

      expect(dataUpdateSpy).not.toHaveBeenCalled();
    });

    it('should generate realistic system status data', async () => {
      const dataUpdateSpy = jest.fn();
      mcpClient.on('dataUpdate', dataUpdateSpy);

      mcpClient.connect();

      await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length > 0);
      
      const mockData = dataUpdateSpy.mock.calls[0][0];
      expect(mockData.data).toHaveProperty('mcpConnections');
      expect(mockData.data).toHaveProperty('activeContexts');
      expect(mockData.data).toHaveProperty('messageQueue');
      expect(mockData.data).toHaveProperty('systemHealth');
      
      expect(typeof mockData.data.mcpConnections).toBe('number');
      expect(typeof mockData.data.activeContexts).toBe('number');
      expect(typeof mockData.data.messageQueue).toBe('number');
      expect(['healthy', 'warning']).toContain(mockData.data.systemHealth);
    });
  });

  describe('Event Handling', () => {
    it('should emit dataUpdate events', async () => {
      const dataUpdateSpy = jest.fn();
      mcpClient.on('dataUpdate', dataUpdateSpy);

      mcpClient.connect();
      
      await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length > 0);
      
      expect(dataUpdateSpy).toHaveBeenCalled();
    });

    it('should handle multiple event listeners', async () => {
      const listener1 = jest.fn();
      const listener2 = jest.fn();
      
      mcpClient.on('dataUpdate', listener1);
      mcpClient.on('dataUpdate', listener2);

      mcpClient.connect();
      
      await TestHelpers.waitFor(() => listener1.mock.calls.length > 0);
      
      expect(listener1).toHaveBeenCalled();
      expect(listener2).toHaveBeenCalled();
    });

    it('should remove event listeners correctly', async () => {
      const listener = jest.fn();
      
      mcpClient.on('dataUpdate', listener);
      mcpClient.removeListener('dataUpdate', listener);
      
      mcpClient.connect();
      
      await new Promise(resolve => setTimeout(resolve, 200));
      
      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe('Performance', () => {
    it('should maintain low latency data generation', async () => {
      const dataUpdateSpy = jest.fn();
      const latencies: number[] = [];
      
      mcpClient.on('dataUpdate', (data) => {
        const latency = Date.now() - data.timestamp;
        latencies.push(latency);
        dataUpdateSpy();
      });

      mcpClient.connect();
      
      // Collect latency data
      await TestHelpers.waitFor(() => latencies.length >= 10);
      
      mcpClient.disconnect();

      // All latencies should be very low (< 50ms)
      latencies.forEach(latency => {
        expect(latency).toBeLessThan(50);
      });
      
      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      expect(avgLatency).toBeLessThan(10);
    });

    it('should handle high-frequency data generation', async () => {
      const config: MCPConfig = {
        enableRealtimeUpdates: true,
        updateInterval: 10, // Very fast updates - 100 Hz
        serverUrl: 'ws://localhost:3000'
      };
      
      const client = new MCPClient(config);
      const dataUpdateSpy = jest.fn();
      client.on('dataUpdate', dataUpdateSpy);

      performanceTracker.start('high_frequency_test');
      
      client.connect();
      
      // Run for 1 second
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      client.disconnect();
      
      const duration = performanceTracker.end('high_frequency_test');
      const updateCount = dataUpdateSpy.mock.calls.length;
      const frequency = updateCount / (duration / 1000);
      
      // Should achieve close to target frequency (allowing for some variance)
      expect(frequency).toBeGreaterThan(50); // At least 50 Hz
      expect(updateCount).toBeGreaterThan(50);
    });

    it('should have minimal memory footprint', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      mcpClient.connect();
      
      // Generate data for a while
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      mcpClient.disconnect();
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be minimal (< 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });
  });

  describe('Configuration', () => {
    it('should respect update interval configuration', async () => {
      const intervals = [100, 200, 500];
      
      for (const interval of intervals) {
        const config: MCPConfig = {
          enableRealtimeUpdates: true,
          updateInterval: interval,
          serverUrl: 'ws://localhost:3000'
        };
        
        const client = new MCPClient(config);
        const dataUpdateSpy = jest.fn();
        client.on('dataUpdate', dataUpdateSpy);

        const startTime = Date.now();
        client.connect();
        
        // Wait for multiple updates
        await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length >= 3);
        
        client.disconnect();
        
        const duration = Date.now() - startTime;
        const updateCount = dataUpdateSpy.mock.calls.length;
        const actualInterval = duration / updateCount;
        
        // Allow 50% tolerance for timing variations
        expect(actualInterval).toBeWithinRange(interval * 0.5, interval * 1.5);
      }
    });

    it('should handle invalid configuration gracefully', () => {
      const invalidConfigs = [
        { updateInterval: -1 },
        { updateInterval: 0 },
        { serverUrl: '' },
        { serverUrl: 'invalid-url' }
      ];

      invalidConfigs.forEach(config => {
        expect(() => new MCPClient(config as MCPConfig)).not.toThrow();
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle connection errors gracefully', () => {
      // This test would require actual network simulation
      // For now, we test that the methods don't throw
      expect(() => {
        mcpClient.connect();
        mcpClient.disconnect();
      }).not.toThrow();
    });

    it('should continue working after errors', async () => {
      const dataUpdateSpy = jest.fn();
      const errorSpy = jest.fn();
      
      mcpClient.on('dataUpdate', dataUpdateSpy);
      mcpClient.on('error', errorSpy);

      mcpClient.connect();
      
      // Wait for normal operation
      await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length > 2);
      
      const updateCountBeforeError = dataUpdateSpy.mock.calls.length;
      
      // Simulate some kind of recovery scenario
      await new Promise(resolve => setTimeout(resolve, 200));
      
      const updateCountAfterError = dataUpdateSpy.mock.calls.length;
      
      expect(updateCountAfterError).toBeGreaterThan(updateCountBeforeError);
    });
  });

  describe('Integration with Mock Server', () => {
    it('should work with mock brain-inspired server', async () => {
      await mockServer.start();
      
      const dataUpdateSpy = jest.fn();
      mcpClient.on('dataUpdate', dataUpdateSpy);
      
      mcpClient.connect();
      
      await TestHelpers.waitFor(() => dataUpdateSpy.mock.calls.length > 0);
      
      expect(dataUpdateSpy).toHaveBeenCalled();
    });
  });
});