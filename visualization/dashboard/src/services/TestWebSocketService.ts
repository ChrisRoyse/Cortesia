import { EventEmitter } from 'events';
import { TestExecution, TestExecutionLog } from './TestExecutionTracker';

export interface TestWebSocketMessage {
  type: 'test_started' | 'test_progress' | 'test_completed' | 'test_failed' | 'test_log' | 'test_cancelled';
  executionId: string;
  timestamp: Date;
  data: any;
}

export interface TestStreamingEvent {
  executionId: string;
  event: 'started' | 'progress' | 'completed' | 'failed' | 'log' | 'cancelled';
  data: any;
  timestamp: Date;
}

export class TestWebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private activeSubscriptions = new Set<string>();
  private serverUrl: string;
  
  constructor(serverUrl: string = 'ws://localhost:8083') {
    super();
    this.serverUrl = serverUrl;
  }

  /**
   * Connect to the WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.serverUrl);

        this.ws.onopen = () => {
          console.log('Test WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connected');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: TestWebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('Test WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.ws = null;
          this.emit('disconnected', event);
          
          // Attempt to reconnect if not intentionally closed
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('Test WebSocket error:', error);
          this.isConnecting = false;
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.activeSubscriptions.clear();
  }

  /**
   * Subscribe to test execution updates
   */
  subscribeToExecution(executionId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot subscribe to execution');
      return;
    }

    this.activeSubscriptions.add(executionId);
    
    const subscribeMessage = {
      type: 'subscribe',
      executionId,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(subscribeMessage));
  }

  /**
   * Unsubscribe from test execution updates
   */
  unsubscribeFromExecution(executionId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.activeSubscriptions.delete(executionId);
    
    const unsubscribeMessage = {
      type: 'unsubscribe',
      executionId,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(unsubscribeMessage));
  }

  /**
   * Subscribe to all test executions
   */
  subscribeToAllExecutions(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot subscribe to all executions');
      return;
    }

    const subscribeMessage = {
      type: 'subscribe_all',
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(subscribeMessage));
  }

  /**
   * Send a command to start a test execution
   */
  startTestExecution(suiteId: string, options: any = {}): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot start test execution');
      return;
    }

    const commandMessage = {
      type: 'start_test',
      suiteId,
      options,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(commandMessage));
  }

  /**
   * Send a command to cancel a test execution
   */
  cancelTestExecution(executionId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, cannot cancel test execution');
      return;
    }

    const commandMessage = {
      type: 'cancel_test',
      executionId,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(commandMessage));
  }

  /**
   * Get connection status
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: TestWebSocketMessage): void {
    const streamingEvent: TestStreamingEvent = {
      executionId: message.executionId,
      event: this.mapMessageTypeToEvent(message.type),
      data: message.data,
      timestamp: new Date(message.timestamp)
    };

    // Emit specific event
    this.emit(message.type, streamingEvent);
    
    // Emit general streaming event
    this.emit('streaming_event', streamingEvent);

    // Emit execution-specific event
    this.emit(`execution:${message.executionId}`, streamingEvent);
  }

  /**
   * Map WebSocket message types to streaming events
   */
  private mapMessageTypeToEvent(messageType: string): TestStreamingEvent['event'] {
    switch (messageType) {
      case 'test_started':
        return 'started';
      case 'test_progress':
        return 'progress';
      case 'test_completed':
        return 'completed';
      case 'test_failed':
        return 'failed';
      case 'test_log':
        return 'log';
      case 'test_cancelled':
        return 'cancelled';
      default:
        return 'progress';
    }
  }

  /**
   * Schedule a reconnect attempt
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }
    }, delay);
  }

  /**
   * Create a mock WebSocket service for development
   */
  static createMockService(): TestWebSocketService {
    const mockService = new TestWebSocketService();
    
    // Override connect method for mock
    mockService.connect = async () => {
      console.log('Mock TestWebSocketService connected');
      mockService.emit('connected');
      
      // Simulate test events for demonstration
      setTimeout(() => {
        mockService.simulateMockTestExecution();
      }, 2000);
    };

    return mockService;
  }

  /**
   * Simulate a mock test execution for development
   */
  private simulateMockTestExecution(): void {
    const executionId = `mock-exec-${Date.now()}`;
    
    // Test started
    this.emit('test_started', {
      executionId,
      event: 'started',
      data: {
        suiteId: 'core-graph-tests',
        suiteName: 'Core Graph Tests',
        totalTests: 15
      },
      timestamp: new Date()
    });

    // Simulate progress updates
    let currentTest = 0;
    const totalTests = 15;
    
    const progressInterval = setInterval(() => {
      currentTest++;
      
      this.emit('test_progress', {
        executionId,
        event: 'progress',
        data: {
          current: currentTest,
          total: totalTests,
          currentTest: `test_graph_node_creation_${currentTest}`,
          status: Math.random() > 0.8 ? 'failed' : 'passed'
        },
        timestamp: new Date()
      });

      if (currentTest >= totalTests) {
        clearInterval(progressInterval);
        
        // Test completed
        setTimeout(() => {
          this.emit('test_completed', {
            executionId,
            event: 'completed',
            data: {
              passed: 13,
              failed: 2,
              ignored: 0,
              executionTime: 3420
            },
            timestamp: new Date()
          });
        }, 500);
      }
    }, 200);
  }
}

export const testWebSocketService = new TestWebSocketService();