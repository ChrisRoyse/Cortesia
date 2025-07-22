/**
 * Jest setup file for LLMKG Visualization Phase 1 testing
 * 
 * Global test configuration, mocks, and utilities setup
 */

import { performance } from 'perf_hooks';

// Global test timeout for async operations
jest.setTimeout(10000);

// Performance measurement utilities
// Fix performance compatibility issues
Object.assign(global, { performance });

// Mock WebSocket if not available in test environment
if (typeof global.WebSocket === 'undefined') {
  global.WebSocket = class MockWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    readyState = MockWebSocket.CONNECTING;
    url: string;
    protocol: string;
    onopen: ((event: Event) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;

    constructor(url: string, protocols?: string | string[]) {
      this.url = url;
      this.protocol = Array.isArray(protocols) ? protocols[0] : protocols || '';
      
      setTimeout(() => {
        this.readyState = MockWebSocket.OPEN;
        if (this.onopen) {
          this.onopen(new Event('open'));
        }
      }, 10);
    }

    send(data: string | ArrayBuffer | Blob | ArrayBufferView): void {
      if (this.readyState !== MockWebSocket.OPEN) {
        throw new Error('WebSocket is not open');
      }
      
      // Simulate message echo for testing
      setTimeout(() => {
        if (this.onmessage) {
          this.onmessage(new MessageEvent('message', { data }));
        }
      }, 10);
    }

    close(code?: number, reason?: string): void {
      this.readyState = MockWebSocket.CLOSING;
      setTimeout(() => {
        this.readyState = MockWebSocket.CLOSED;
        if (this.onclose) {
          this.onclose(new CloseEvent('close', { code, reason }));
        }
      }, 10);
    }

    addEventListener(type: string, listener: EventListener): void {
      switch (type) {
        case 'open':
          this.onopen = listener as any;
          break;
        case 'close':
          this.onclose = listener as any;
          break;
        case 'message':
          this.onmessage = listener as any;
          break;
        case 'error':
          this.onerror = listener as any;
          break;
      }
    }

    removeEventListener(): void {
      // Mock implementation
    }
  } as any;
}

// Mock high-resolution time if not available
if (!global.performance?.now) {
  Object.assign(global, {
    performance: {
      now: () => Date.now(),
      mark: () => {},
      measure: () => {},
      getEntriesByName: () => [],
      getEntriesByType: () => [],
      clearMarks: () => {},
      clearMeasures: () => {},
      eventCounts: new Map(),
      navigation: {},
      onresourcetimingbufferfull: null,
      timing: {},
      timeOrigin: Date.now()
    } as any
  });
}

// Mock process.hrtime for Node.js environments that don't have it
if (typeof process !== 'undefined' && !process.hrtime) {
  const hrtimeFn = function(time?: [number, number]): [number, number] {
    const now = Date.now();
    const seconds = Math.floor(now / 1000);
    const nanoseconds = (now % 1000) * 1000000;
    
    if (time) {
      return [seconds - time[0], nanoseconds - time[1]];
    }
    
    return [seconds, nanoseconds];
  };
  
  // Add bigint method
  (hrtimeFn as any).bigint = function(): bigint {
    const [seconds, nanoseconds] = hrtimeFn();
    return BigInt(seconds) * BigInt(1e9) + BigInt(nanoseconds);
  };
  
  process.hrtime = hrtimeFn as any;
}

// Global test utilities
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(floor: number, ceiling: number): R;
      toHaveBeenCalledWithinTime(timeout: number): R;
    }
  }
}

// Custom Jest matchers
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },

  toHaveBeenCalledWithinTime(received: jest.MockedFunction<any>, timeout: number) {
    const startTime = Date.now();
    
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        
        if (received.mock.calls.length > 0) {
          clearInterval(checkInterval);
          resolve({
            message: () => `expected function to not have been called within ${timeout}ms`,
            pass: true,
          });
        } else if (elapsed >= timeout) {
          clearInterval(checkInterval);
          resolve({
            message: () => `expected function to have been called within ${timeout}ms but it wasn't`,
            pass: false,
          });
        }
      }, 10);
    });
  },
});

// Suppress console logs in tests unless explicitly enabled
const originalConsole = global.console;
global.console = {
  ...originalConsole,
  log: process.env.JEST_VERBOSE === 'true' ? originalConsole.log : jest.fn(),
  info: process.env.JEST_VERBOSE === 'true' ? originalConsole.info : jest.fn(),
  warn: process.env.JEST_VERBOSE === 'true' ? originalConsole.warn : originalConsole.warn,
  error: process.env.JEST_VERBOSE === 'true' ? originalConsole.error : originalConsole.error,
};

// Clean up after each test
afterEach(() => {
  // Clear all timers
  jest.clearAllTimers();
  
  // Clear all mocks
  jest.clearAllMocks();
  
  // Reset modules
  jest.resetModules();
});

// Global test cleanup
afterAll(() => {
  // Restore console
  global.console = originalConsole;
});