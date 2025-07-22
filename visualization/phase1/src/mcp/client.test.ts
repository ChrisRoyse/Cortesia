/**
 * @fileoverview Unit Tests for MCP Client
 * 
 * Basic unit tests to validate the MCP client implementation.
 * These tests focus on core functionality and type safety.
 */

import { MCPClient, MCPClientConfig, ConnectionState, MCPEventType } from './index.js';

// Mock WebSocket for testing
class MockWebSocket {
  public readyState = 1; // OPEN
  public onopen: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;

  constructor(public url: string) {
    // Simulate successful connection after short delay
    setTimeout(() => {
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }

  send(data: string) {
    // Mock successful responses
    const message = JSON.parse(data);
    setTimeout(() => {
      if (this.onmessage) {
        let response;
        
        if (message.method === 'initialize') {
          response = {
            jsonrpc: "2.0",
            id: message.id,
            result: {
              protocolVersion: "2024-11-05",
              serverInfo: {
                name: "Mock LLMKG Server",
                version: "1.0.0"
              },
              capabilities: {
                tools: { listChanged: true }
              }
            }
          };
        } else if (message.method === 'tools/list') {
          response = {
            jsonrpc: "2.0",
            id: message.id,
            result: {
              tools: [
                {
                  name: "brain_visualization",
                  description: "Mock brain visualization tool",
                  inputSchema: { type: "object" }
                }
              ]
            }
          };
        } else {
          response = {
            jsonrpc: "2.0",
            id: message.id,
            result: { success: true, mockData: "test" }
          };
        }

        this.onmessage(new MessageEvent('message', {
          data: JSON.stringify(response)
        }));
      }
    }, 5);
  }

  close() {
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code: 1000 }));
    }
  }
}

// Replace global WebSocket with mock for testing
(global as any).WebSocket = MockWebSocket;

describe('MCPClient', () => {
  let client: MCPClient;

  beforeEach(() => {
    const config: MCPClientConfig = {
      enableTelemetry: false, // Disable for cleaner tests
      autoDiscoverTools: false,
      requestTimeout: 5000
    };
    client = new MCPClient(config);
  });

  afterEach(async () => {
    await client.disconnectAll();
  });

  test('should create client with default configuration', () => {
    const defaultClient = new MCPClient();
    expect(defaultClient).toBeInstanceOf(MCPClient);
    expect(defaultClient.isConnected).toBe(false);
  });

  test('should connect to mock server', async () => {
    const serverInfo = await client.connect('ws://localhost:8001/mcp');
    
    expect(serverInfo).toBeDefined();
    expect(serverInfo.name).toBe('Mock LLMKG Server');
    expect(serverInfo.version).toBe('1.0.0');
    expect(client.isConnected).toBe(true);
  });

  test('should handle connection state changes', (done) => {
    client.on(MCPEventType.CONNECTION_STATE_CHANGED, (event) => {
      if (event.data.newState === ConnectionState.CONNECTED) {
        expect(event.data.endpoint).toBe('ws://localhost:8001/mcp');
        done();
      }
    });

    client.connect('ws://localhost:8001/mcp');
  });

  test('should list tools after connection', async () => {
    await client.connect('ws://localhost:8001/mcp');
    
    // Manually trigger tool discovery since autoDiscoverTools is disabled
    const tools = await client.listTools();
    
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe('brain_visualization');
    expect(tools[0].description).toBe('Mock brain visualization tool');
  });

  test('should call tools successfully', async () => {
    await client.connect('ws://localhost:8001/mcp');
    
    const result = await client.callTool('brain_visualization', {
      region: 'test_region',
      type: 'activation'
    });

    expect(result).toBeDefined();
    expect(result.success).toBe(true);
    expect(result.mockData).toBe('test');
  });

  test('should provide client statistics', async () => {
    const stats = client.statistics;
    
    expect(stats).toBeDefined();
    expect(stats.activeConnections).toBe(0);
    expect(stats.totalToolCalls).toBe(0);
    expect(stats.successfulToolCalls).toBe(0);
    expect(stats.failedToolCalls).toBe(0);
  });

  test('should handle multiple connections', async () => {
    const servers = await client.connectMultiple([
      'ws://localhost:8001/mcp',
      'ws://localhost:8002/mcp'
    ]);

    expect(servers).toHaveLength(2);
    expect(client.connectedEndpoints).toHaveLength(2);
    expect(client.isConnected).toBe(true);
  });

  test('should disconnect gracefully', async () => {
    await client.connect('ws://localhost:8001/mcp');
    expect(client.isConnected).toBe(true);

    await client.disconnect('ws://localhost:8001/mcp');
    expect(client.isConnected).toBe(false);
  });

  test('should provide LLMKG-specific tool methods', async () => {
    await client.connect('ws://localhost:8001/mcp');
    
    // Test that LLMKG methods exist and are callable
    expect(client.llmkg.brainVisualization).toBeDefined();
    expect(client.llmkg.federatedMetrics).toBeDefined();
    expect(client.llmkg.knowledgeGraphQuery).toBeDefined();
    expect(client.llmkg.getActivationPatterns).toBeDefined();
    expect(client.llmkg.analyzeConnectivity).toBeDefined();
    expect(client.llmkg.analyzeSdr).toBeDefined();

    // Test actual call
    const result = await client.llmkg.getActivationPatterns('hippocampus', 5000);
    expect(result).toBeDefined();
  });
});

// Export for Jest
export {};